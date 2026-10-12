# Benchmarks and evidence

Run from the repository root with the project's environment. Execution hardware, scoring scope and timing boundaries are recorded per artifact; there is no shared hardware assumption across runs.

## Running model jobs

Run one model job per host, confined so an overrun fails the job instead of the machine:

```
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python -m bench.<name> …
```

Every script that loads a model enters `bench.outcomes.runtime()`, whose `Runtime` owns one `semantic_switch.model_job()` from its first native load until close: an exclusive lock at `$SEMANTIC_MODEL_LOCK` (default `$XDG_RUNTIME_DIR/semantic-switch/model-job.lock`) plus admission of the estimated host bytes and a 2 GiB reserve against available memory and the cgroup limit. A second job fails immediately and names the holder; a job that does not fit fails before loading. The trainer CLI (including `--dry-run`) acquires one job and passes it explicitly (`job=`) to its native probes, conversion and fitting; slow tests take the lock through each `Runtime`. Independent acquisitions always contend; only an explicit `job=` handle joins, and its allocation must fit the declared envelope. Agents must not run model jobs in the background or start a second one while one is running; a process that starts model-loading children holds neither a model nor the lock.

Resumable scripts (`bench.judges`, `bench.select_quality`, `bench.policy_workflow`, `bench.external_primitives`, `bench.routing_specialist`, `bench.snapshots`) commit each completed row (a complete episode for episodes; an interrupted episode restarts from its initial state) to the shared execution journal `<output>.journal.jsonl` (`semantic_switch.journal`) under a run lease, keyed by a full digest of the work identity. A rerun with the same configuration digest (script, arguments, model digests, native settings, cohort, code provenance) resumes the missing rows; a different digest is another run, refused until `--fresh` moves the old journal aside. The published artifact's summaries are recomputed from the current rows. Recorded failures are kept; `--retry-failed` (where offered) appends new attempts and the published `execution` record lists the superseded ones.

Current executable contracts:

- `python -m bench.reanalyze_fresh`: corrected-gold reanalysis preserving historical predictions. The original refusal labels were reversed in two EN/PT pairs; original fresh accuracy and extreme-confidence error summaries are superseded by `results/fresh-corrected-gold-v2.json`. This is retrospective scoring, not new inference.
- `python -m bench.model_comparison --model MODEL --output result.json`: isolated native regression quality and complete-operation latency, including distinct-question and identical-question packing workloads.
- `python -m bench.policy_workflow --model MODEL --cli --output result.json`: frozen authored bounded-policy facts/actions and SDK/CLI parity. Synthetic evidence, not independent domain accuracy.
- `python -m bench.policy_workflow --cohort DIR --judge SPEC … --output result.json`: checkpointed procedure benchmark over the small-context cohort (see *Procedure benchmark*).
- `python -m bench.openrouter --key-file PATH --models MODELS [--provider NAME] --policy-workflow --output result.json`: hosted preflight and policy workflow through `judge_from('openrouter:MODEL')` (the SDK resolves the chat or structured protocol; default models include `typesafe/jev-1.13`), with per-attempt request receipts and reported token usage; spend is limited by the key's provider-side limit, not by the bench. Companion protocol failures and policy-question availability are separate.
- `python -m bench.select_quality --judge SPEC [--provider NAME] [--key-file PATH] --output result.json`: checkpointed public `Select` on the S1MB multi-label slice with parity against the equivalent named Checks (see *Select quality*).
- `python -m bench.policy_audit definition.yaml audit.jsonl --model MODEL --output result.json`: untouched domain-owner labels against a matching frozen lock.
- `python -m bench.policy_adaptation DEFINITIONS --base BASE --adapter ADAPTER [--adapter ADAPTER] --precise PRECISE [--selection REPORT] --output result.json`: base, one arm per adapter (named by its directory) and precise arms over the frozen target policies of a materialized adaptation cohort; per-window complete latency, native work and the resident-set peak within each window (reset at window start). Authored development data, no quality promotion.
- `python -m bench.native_segmentation FIXTURE --model BASE [--index N] --output result.json`: white-box single-sequence decode of exported presentations as whole prefill, segments and 37-token chunks; the full answer-position logits are published as the report's content-addressed `.npz` payload.

Tables below retain their original fixture/presentation/hardware conditions. They do not describe current runtime latency or the repaired fixture versions. Repetitive-filler context tests measure clause position/capacity, not safe policy quality limits.

| Script | Question it answers |
|---|---|
| `python -m bench.judges --judges a,b [--output PATH] [--fresh]` | Which judge and quantisation? Accuracy, AUC, ECE, coverage and precision at 0.9 (low answer mass never settles), cross-fitted coverage at 95%, picks, latency, memory, shift against Q8_0 |
| `python -m bench.fresh SRC …` | Consulted authored regression cases; corrected labels and neutral evidence rendering. |
| `python -m bench.identifiers SRC` | Near-identical ids (MX-120 vs MX-120A), with 4 and 20 candidates |
| `python -m bench.unstated SRC` | Does a three-way check separate "false" from "not stated"? |
| `python -m bench.evidence fetch MANIFEST SOURCE` / `pin MANIFEST SOURCE [--excerpt FILE]` / `verify MANIFEST` | Download a source's recorded document beside the manifest (the only network command; copies stay out of version control), pin its document hash, edition, page/section locators, extraction tool and excerpt hash, or verify the used-source contract. `python -m bench.{procedures,support} pin-authored` pins the authored rule and catalogue digests. Only sources cited by used rules or slices are verified; others declare `unused: <reason>`. A pack leaves `skeleton` only once that contract is recorded |
| review protocol | Two tiers per frame: mechanical validation of every frame (spans/paths, provenance, recomputed reference, scope, rule/source bindings, hidden-state invariance, role assignment) and human confirmation of `review_worksheet.jsonl` (every rule/composition branch, unresolved/ambiguous/contradictory frame and condition/language/role/safety stratum plus a seeded 20% sample). A confirmation carries the worksheet `binding` (frame evidence, effective gold/composition, source contract, protocol); `replace`/`skip` carry `content_sha256`. A replacement recomputes gold and needs a confirmation of the corrected frame; conflicting records are disputes. `reviewed` needs the complete source contract, all-frame mechanical success, every required confirmation current and no dispute; it discloses sampled human review |
| `python -m bench.procedures pack --output DIR [--variants N] [--reviews REVIEWS.jsonl]` | Zebra ZT411/ZT421 snapshot pack: planned scenarios (base plus a seeded sample of up to N−1 contrast variants each), structured/prose frames, independent reference, exact whole-family roles counted in situations (gold-complete structured frames: 100 fit across ≥30 families, 150 model_select across ≥40, ≥200 check across ≥60; other conditions stay in their family's role), canonical rule-origin labels for the structured definition; unreviewed until reviews cover it |
| `python -m bench.procedures transfer --output DIR` | Bounded four-document guidance transfer: one definition with `unverified_step_{a,b,c,d}` bound to document A (pinned manual rules) and authored convention variants B/C/D that change only the verification convention (ZT-COMMAND); fit labels A/B, model_select C, check D, whole families per role |
| `python -m bench.support pack --output DIR [--variants N] [--abcd DATA --abcd-annotations ANN] [--reviews REVIEWS.jsonl]` | Support classification pack; ABCD prefixes form a separately reported check realism slice |
| `python -m bench.evidence stage PACK --to DIR` | Writes a published pack's definition and role-assigned `DEFINITION.test.jsonl`, and publishes `DEFINITION.dataset.json` (canonical families, aliases, review disclosure) with its empty `DEFINITION.audit.jsonl` exposure ledger; move or back up the ledger with the manifest |
| `python -m bench.snapshots {zt411,support} --pack DIR --evidence STAGED.dataset.json --baseline NAME=SPEC [--specialist NAME=SPEC] [--reference NAME=SPEC ...] [--native KEY=VALUE ...] [--kind workflow\|direct] --margin REQUIRED_IMPROVEMENT --output PATH [--fresh]` | Matched snapshot evaluation over structured/prose/permuted-menu conditions and EN/PT counterparts (`zt411-episodes` runs complete episodes): `SPEC` is a GGUF, a deployment `receipt.json` or `openrouter:MODEL`; the specialist is a receipt evaluated through `Runtime.from_specialization(receipt).load()` (frozen task, policies and execution) on the pack's trained condition, where every other arm runs the same receipt task; local arms share one model job and the receipt's native settings overlaid by `--native`, and `device=gpu` refuses CPU realization; every arm is frozen before one plan is recorded as one consultation, before scoring; journaled rows, alternating arm order, family-bootstrap pairing against the baseline, specialist-only adoption, action/membership/exact/ordered metrics, latency with known-work populations and fingerprints |
| `python -m bench.recompute REPORT --to PATH` | Recomputes the set aggregates (metric revision 3: `actions` and `exact`) of a published `bench.snapshots` report (model rows and canonical baseline rows) or `zt411-episodes` report (steps) where outcomes and per-member reference and prediction states establish completeness; each other aggregate is marked `unrecoverable` and old `frontier` fields are dropped; writes a new artifact with the source hash, leaving the source unchanged |
| `python -m bench.equipment [--judge SPEC ...] [--direct]` | **Fictional composition regression:** 56 items, 46 EN/PT reports, every trap type, three switch shapes, a θ sweep, and the judge choosing directly from the whole catalogue. `SPEC` is a local GGUF (default: E2B and E4B), or `openrouter:MODEL` with `--provider PROVIDER` and `--key-file`; `openrouter:typesafe/jev-1.13` also needs `--jev-consent`, given only with TypeSafe's consent |
| `python -m bench.compare_jev`, `bench.compare_jev_quality` | Matched latency against Jev through OpenRouter (`openrouter:typesafe/jev-1.13`); quality takes repeatable `--judge SPEC` (local or `openrouter:MODEL`) with `--provider` and `--key-file` (see the contract note below) |

**Concluded experiment code was removed in v0.26** (probe, order averaging, staged/eager, prompt examples, model comparison, prompt framing, learning curve, topical, GPU, the old equipment stress test). Their raw results stay in `bench/results/`, their summaries below, and their code in git history before v0.26.

The selection suite lives in `bench/quality.py`. `ss doctor` checks native conformance. **It chose the default judge and the exception phrasing, so it is regression data, not independent evidence.** `bench/fresh.py` has also been consulted during development and is not an untouched audit.

## Snapshot specialization (explicitly invoked jobs)

The frozen-task experiment uses only public commands. Packs, staging and export are offline or load the base once; fitting, selection, check and the matched comparison are model jobs, one per host.

```
PACK_KIND=zt411 bench/zt411_gpu.sh            # or PACK_KIND=support; UNTIL=stage stops after the model-free pack/stage steps; MARGIN=0 is the predeclared required improvement
```

The script runs, in order and each step only when its identity changed (`$WORK/done/STEP` holds the identity it completed):

```
python -m bench.procedures pack --output packs/zt411                     # likewise: python -m bench.support pack --output packs/support
python -m bench.evidence stage packs/zt411 --to work/zt411/stages/PACK_ID # printer.yaml, printer.test.jsonl, printer.dataset.json + audit ledger
semantic-trainer assets --model google/gemma-4-E2B-it --quant Q8_0 --revision REVISION
ss --judge BASE_Q8_0.gguf --device gpu export STAGE/printer.yaml --native --task-family bench:zt411 --labels STAGE/printer.test.jsonl \
    --to work/zt411/exports/EXPORT_ID/printer.export.jsonl
.venv-xpu/bin/semantic-trainer prepare STAGE/printer.yaml EXPORT/printer.export.jsonl.manifest.json --assets DIGEST --quant Q8_0 --device xpu:0 \
    --native-device gpu --fit-situations 100 --recipe selection_unit=complete_situation --recipe seed=0 --recipe eval_every_examples=1000000 \
    --budget max_tokens=2000000 --budget max_monitor_tokens=20000000 --budget max_seconds=18000 --budget feasibility_updates=8 --to work/zt411/provisional/JOB_ID
systemd-run --user --scope -q -p MemoryMax=22G -p MemorySwapMax=0 .venv-xpu/bin/semantic-trainer profile work/zt411/provisional/JOB_ID/job.yaml \
    --to work/zt411/profiles/PROFILE_ID.json --steps 3 --warmup 1                # prints the cadence table, then stops unless EVAL_EVERY_EXAMPLES is set
.venv-xpu/bin/semantic-trainer prepare ... (same settings) --recipe eval_every_examples=$EVAL_EVERY_EXAMPLES --profile PROFILE --to work/zt411/jobs/JOB_ID
.venv-xpu/bin/semantic-trainer lora JOB/job.yaml --to JOB/runs/RUN_ID --dry-run
systemd-run --user --scope -q -p MemoryMax=22G -p MemorySwapMax=0 .venv-xpu/bin/semantic-trainer lora JOB/job.yaml --to JOB/runs/RUN_ID
.venv-xpu/bin/semantic-trainer select JOB/job.yaml --run JOB/runs/RUN_ID
semantic-trainer check RUN/deployment/receipt.json STAGE/printer.test.jsonl --evidence STAGE/printer.dataset.json --to work/zt411/checks/CHECK_ID
python -m bench.snapshots zt411 --pack packs/zt411 --evidence STAGE/printer.dataset.json --baseline base=BASE_Q8_0.gguf \
    --specialist specialist=RUN/deployment/receipt.json --reference jev=openrouter:typesafe/jev-1.13 --native device=gpu \
    --margin 0 --output bench/results/zt411-specialist-COMPARE_ID.json
```

Identities bind the pack sources/reviews, the published `benchmark.json`, the asset manifest and presentation identity, the trainer loss/job schemas plus prepare settings, the prepared `job.yaml`, the deployment receipt and evidence, and the snapshot report/decision/metric schemas. A changed contract therefore reruns the affected steps into new identified stage/export/job/run/check directories and a new result path; partial directories are moved aside, never deleted. `MARGIN` is recorded in `$WORK/margin` before fitting and a different value is refused for that work directory. The monitoring cadence has no default: the script prepares a provisional job whose placeholder cadence plans only point zero and the final point, profiles it on the GPU and prints the measured cadence table, then stops until `EVAL_EVERY_EXAMPLES=N` is given; the final job copies the profile and records the cadence, profile id and projection (`prepare` refuses a profile of another fitting configuration, a token workload changed by more than half or a cadence projected beyond the budgets). `UNTIL=profile` stops after profiling; `PROFILE=PATH` reuses an existing profile (for example after review changed only metadata or modestly changed text) instead of profiling a new provisional job. Profiles and fits are guarded by `MemoryMax`, `MemorySwapMax=0`, a minimum available-memory precondition and a watchdog that stops the job below `GUARD_GIB`.

`bench.snapshots` freezes one evaluation plan (arms with explicit baseline/specialist/reference roles, conditions, scope, metrics, decision schema and margin) and records it as one consultation in the staged evidence audit before inference. Each arm/kind/frame row commits once to `<output>.journal.jsonl`; a rerun resumes pending rows and retains failed rows unless `--fresh` archives the journal. Workflow rows take the decision, every scored target and cost from one public call (`switch(input, facts=...)`). Set rows carry `scope`, known positives `gold`, unknown `reference_unresolved`, `complete_gold`, `selected`, `unresolved`, `prediction_complete`, `outcome` and `prohibited`; model rows, always-empty/select-all/rule-only baselines and episode steps use the same adapter, and a missing rule-only membership is unresolved. `exact` (`exact_sets`) is the only exact match: complete reference, successful operation, complete prediction and equal sets, over all gold-complete situations with failed/unavailable/incomplete predictions as misses; incomplete-reference situations are reported, never scored. `actions` reports precision/recall over known-reference memberships, selected unknown-reference memberships as unassessed, prohibited selections, known required omissions and prediction-unresolved members. A raised operation is a failed submitted row with its elapsed time and known work; every scorer prints running submitted/completed/correct/accepted/failed counts from the same rows as its report, resumed rows included. Differences are arm minus baseline and row-weighted; their interval is a within-pack family-resampling interval (whole canonical families with all their rows per draw) over the authored scenario grid, not production precision or a representative sample of customer or equipment populations. Only the specialist receives an adoption verdict (decision schema 2): adopt only when the 95% family-resampling lower bound strictly exceeds `--margin`, with no more forbidden-action situations or failures than the baseline, on independent reviewed evidence. Reports from decision schema 1 (`tolerated_regression`) keep their meaning and are not improvement evidence. Reference arms such as Jev are descriptive.

`prepare` refuses unless whole fit families total exactly 100 situations and every declared answer has fit support; acceptance stays authored and uncalibrated without `--acceptance-fit`. Local arms share one explicit native configuration; differing realized executions or CPU realization under `device=gpu` are refused. The comparison is a quality claim only once the pack's review status is `reviewed`. The transfer pack follows the same path with `python -m bench.procedures transfer`, one `ss export … --fact unverified_step_{a,b,c} --task-family zt411:{a,b,c} --to …` per document and `semantic-trainer prepare … --objective instruction_transfer`; `semantic-trainer check` evaluates document D.

## Judge ladder: selection suite (240 checks, 24 picks)

| Judge | File | Acc. | Picks | Check ms | ms per question in an ask | RAM | AUC | ECE | Settled @0.9 (precision) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.6 GB | 0.69 | 0.79 | 111 | 65 | 1.6 GB | 0.85 | 0.15 | 36% (89.5%) |
| Qwen3-0.6B Q4_K_M | 0.4 GB | 0.74 | 0.83 | 99 | 54 | 1.5 GB | 0.85 | 0.07 | 14% (97.0%) |
| Qwen3-1.7B Q8_0 | 1.7 GB | 0.84 | 0.92 | 290 | 202 | 2.8 GB | 0.93 | 0.16 | 97% (84.5%) |
| Qwen3-1.7B Q4_K_M | 1.1 GB | 0.76 | 0.96 | 306 | 159 | 2.7 GB | 0.89 | 0.22 | 92% (79.5%) |
| Qwen3.5-0.8B Q8_0 | 0.8 GB | 0.71 | 0.67 | 215 | 101 | 1.7 GB | 0.80 | 0.12 | 35% (83.3%) |
| Qwen3.5-0.8B Q4_K_M | 0.5 GB | 0.77 | 0.62 | 194 | 80 | 1.4 GB | 0.83 | 0.08 | 29% (91.4%) |
| Qwen3.5-2B Q8_0 | 1.9 GB | 0.90 | 0.92 | 540 | 216 | 2.8 GB | 0.95 | 0.04 | 68% (94.5%) |
| Qwen3.5-2B Q4_K_M | 1.2 GB | 0.78 | 0.92 | 348 | 197 | 2.4 GB | 0.96 | 0.11 | 67% (93.1%) |
| Gemma 3 1B Q8_0 | 1.0 GB | 0.70 | 0.88 | 210 | 101 | 1.3 GB | 0.79 | 0.29 | 96% (70.9%) |
| Gemma 4 E2B Q8_0 | 4.7 GB | 0.96 | 0.92 | 440 | 289 | 5.0 GB | 0.99 | 0.04 | 99% (96.2%) |
| **Gemma 4 E2B Q4_0** | 2.7 GB | **0.95** (0.963 after "Except: X.") | 0.96 | **228** | **128** | 3.8 GB | 0.99 | 0.04 | **92% (97.7%)** |
| Gemma 4 E2B QAT Q4_0 | 3.2 GB | 0.94 | 0.96 | 264 | 149 | 4.2 GB | 0.99 | 0.04 | 92% (97.7%) |
| Gemma 4 E4B QAT Q4_0 | 4.8 GB | 0.975 | 0.96 | 572 | 318 | 7.5 GB | 0.99 | 0.02 | 95% (98.7%) |

**Readings:**
- **Size alone doesn't buy precision.** Gemma 4 E2B (2B compute) beat Qwen3-1.7B and Qwen3.5-2B; the cause (training, architecture, tokenizer, prompt fit) isn't separated.
- **Pair ordering stays high where accuracy is low:** Qwen3-0.6B orders 91% of pairs correctly at 0.69 accuracy.
- **Qwen3-1.7B is overconfident:** 85% of its answers sit at ≥ 0.999, so its errors are unfixable by thresholds.

**Quantisation shift against Q8_0 (240 checks):**

| Model | Mean probability shift | Flips at 0.5 |
|---|---|---|
| Gemma 4 E2B | 0.035 | 5–7 |
| Qwen3-0.6B Q4_K_M | 0.10 | 27 |
| Qwen3-1.7B Q4_K_M | 0.11 | 29 |
| Qwen3.5-0.8B Q4_K_M | 0.09 | 26 |
| Qwen3.5-0.8B Q4_0 | 0.27 | 93 |

## Original fresh regression results (superseded gold)

The original gold reverses four decline examples. The table and derived threshold/phenomenon summaries in this section must not guide model or calibration choices; use the corrected-gold artifact and versioned reruns.

8 new claims in new domains, 10 distinct minimal pairs each (5 EN, 5 PT), and HR topic picks.

| Judge | Accuracy | AUC | ECE | Settled @0.9 (precision) | Picks | Check ms |
|---|---|---|---|---|---|---|
| Gemma 4 E4B QAT | **0.931** | 0.969 | 0.054 | 95% (94.7%) | 1.00 | 578 |
| Gemma 4 E2B Q4_0 | 0.856 | 0.917 | 0.119 | 90% (**88.9%**) | 1.00 | 375 |
| Qwen3.5-2B Q8_0 | 0.844 | 0.938 | 0.063 | 66% (**95.3%**) | 1.00 | 419 |
| Qwen3-1.7B Q8_0 | 0.794 | 0.868 | 0.196 | 93% (83.1%) | 0.95 | 382 |
| Qwen3-0.6B Q4_K_M | 0.656 | 0.720 | 0.141 | 13% (90.0%) | 0.90 | 136 |

- Original threshold summaries used defective gold and are superseded; they do not establish a failure of confidence ranking.
- **Its weakest phenomena:** hypothetical 0.70; contrast and implicit 0.80; negation and temporal 0.82.

## Claim versus question wording (`bench/question_form.py`)

`python -m bench.question_form --judge SPEC [--key-file KEY] --output result.json` runs every check of the selection suite (240) and the fresh suite (160) twice with the same inputs: the authored claim (`The customer threatens legal action against us.`, answered true/false) and a hand-written question with the same meaning (`Is the customer threatening legal action against us?`, answered yes/no). Results: `question-form-e2b.json`, `question-form-e4b.json` (local GPU, 2026-10-09).

| Judge | Suite | Form | Accuracy | AUC | Settled at 0.9 | Precision at 0.9 |
|---|---|---|---|---|---|---|
| E2B | selection | claim | 0.929 | 0.962 | 90.0% | 94.9% |
| E2B | selection | question | 0.946 | 0.971 | 84.6% | 96.6% |
| E2B | fresh | claim | 0.900 | 0.948 | 88.1% | 92.9% |
| E2B | fresh | question | 0.900 | 0.965 | 79.4% | 96.1% |
| E4B | selection | claim | 0.975 | 0.997 | 92.1% | 99.1% |
| E4B | selection | question | 0.963 | 0.993 | 94.6% | 96.9% |
| E4B | fresh | claim | 0.963 | 0.992 | 92.5% | 98.6% |
| E4B | fresh | question | 0.963 | 0.996 | 93.8% | 99.3% |

Paired by item ID (a fix is wrong as a claim and right as a question; a regression is the reverse; `paired` in the output). The arms must share unique item IDs, gold and evidence. Fixes and regressions count only pairs scored in both forms; a failed request (no score, or low answer mass) is reported separately as scored → failed, failed → scored or failed → failed with its correct/incorrect count, so the end-to-end totals reconcile. Every local row below was scored in both forms:

| Judge | Suite | Fixes | Regressions | Claim → question correct |
|---|---|---|---|---|
| E2B | selection | 9 | 5 | 223 → 227 of 240 |
| E2B | fresh | 6 | 6 | 144 → 144 of 160 |
| E4B | selection | 0 | 3 | 234 → 231 of 240 |
| E4B | fresh | 3 | 3 | 154 → 154 of 160 |

**Reading:** no consistent difference. The largest net gap is 4 of 240 items (E2B selection), and the sign flips between judges; the fresh suites net to zero on both. The two arms change both the wording and the answer alphabet (true/false versus yes/no), so these counts do not separate the two effects. Both forms are supported and both are re-measured by `ss lock` on the application's own labels; write whichever states the criterion most precisely.

## Local, Jev and a same-price hosted model (2026-10-10)

Same cases, gold and exclusions through public Runtime calls. Local: Gemma 4 E2B Q4_0 (default) and E4B (precise), serial grouping, laptop GPU. Hosted: Jev 1.13 (TypeSafe System One) and Gemma 4 26B-A4B on OpenRouter pinned to Io Net (bf16, $0.048/M input, Jev's price bracket; chosen after a per-provider latency preflight). Results: `question-form-{e2b-serial,jev}.json` and, for the 26B checks, the single run `question-form-gemma26b-v2.json` (scoring `answer-token-lower-bound@2`); `latency-{e2b-gpu,e2b-cpu,jev,gemma26b}.json`; `equipment-{local,hosted}.json` (the hosted equipment arms predate the lower-bound scoring; in their `catalogue_facts` and `cases` arms every settled hosted fact's floor `p·M` is ≥ 0.947).

**Checks** (claim / question wording; selection 240, fresh 160):

| Judge | Selection correct | Fresh correct | Settled at 0.9 (sel., question) | Precision at 0.9 (sel., question) |
|---|---|---|---|---|
| E2B local | 223 / 227 | 144 / 144 | 85.0% (204/240) | 96.6% |
| Gemma 26B hosted (v2 run) | 236 / 236 | 146 / 153 | 100% (240/240) | 98.3% |
| Jev | 236 / 237 | 156 / 157 | 85.4% (205/240) | 100% |

Precision here is selective precision: correct among accepted. Settled counts are the recorded runtime statuses (`value`/`status` at the default θ 0.9), and threshold sweeps replay the same `semantic_switch.judgment` acceptance on the recorded scores. Jev accepted, all correctly, 183 / 205 of 240 selection checks (claim / question) and 111 / 121 of 160 fresh checks. Earlier summaries tested `p ≤ 1 − θ` in floating point, which excludes a reported p = 0.1 at θ 0.9, and undercounted Jev as 178 / 199 and 110 / 116; the local counts are unchanged. These are the historical rows recomputed, not a new run.

Gemma 26B fresh wording, paired, in `question-form-gemma26b-v2.json` only: 155 items were scored in both forms, with 8 fixes and 2 regressions. Separately, 3 claims were unscored and 2 of them were answered correctly as questions; 2 questions were unscored, one whose claim had been correct. End to end 146 → 153 = 146 + 8 − 2 + 2 − 1. That run settles 240/240, 240/240, 155/160 and 157/160 checks (selection claim/question, fresh claim/question) with selective precision at 0.9 of 98.3%, 98.3%, 92.9% and 97.5%. Its OpenRouter scoring `answer-token-lower-bound@2` returns `m[i]/(M+R)` (answer mass M, unreported vocabulary mass R), an acceptance lower bound rather than answer mass renormalized to one; locks fitted on `answer-token-logprobs@1` are refused. The earlier `question-form-gemma26b.json` run used the old scoring and is not combined with these counts.

**Equipment** (56 decisions over 46 fictional reports; correct / wrong / abstained at the default θ 0.9, and correct at the best θ with zero wrong actions). Arms: hand-written facts plus exact cases, a catalogue of plain claim strings, and the judge picking directly from the whole catalogue:

| Judge | Facts + cases | Catalogue (strings) | Judge picks from the whole catalogue |
|---|---|---|---|
| E2B local | 22 / 0 / 34 (41 at θ 0.6) | 42 / 0 / 14 | 5 / **3** / 48 |
| E4B local | 55 / 0 / 1 (56 at θ 0.6) | 55 / 0 / 1 | 18 / **10** / 26 |
| Gemma 26B hosted | 52 / 0 / 4 (53 at θ 0.6) | 51 / 0 / 5 | 38 / **11** / 3 |
| Jev | 47 / 0 / 9 (56 at θ 0.6) | 44 / 0 / 12 | 26 / **4** / 26 |

**Latency** (one client host, warm, n = 30, p50 / p95 ms; hosted includes the network):

| Judge | Same input, 1 question | New input, 1 question | New input, 4 questions |
|---|---|---|---|
| E2B local GPU | 58 / 59 | 110 / 112 | 292 / 295 |
| E2B local CPU | 131 / 136 | 240 / 245 | 1,061 / 1,230 |
| Jev | 268 / 310 | 259 / 346 | 264 / 296 |
| Gemma 26B hosted | 547 / 1,335 | 522 / 2,090 | 833 / 5,717 |

**Readings:**
- **Observed on this suite:** in the facts + cases and catalogue arms no judge took a wrong action at any swept θ, and the judges' errors appeared as abstentions (1–34 per arm at 0.9); choosing directly from the whole catalogue, every judge took wrong actions (3–11). This is a result on 56 authored items, not a guarantee that exact rules prevent every confidently wrong semantic action.
- **Jev is the most precise per question** (no wrong accepted check at 0.9) and settles less; **E4B is the best composed-decision judge** measured here (55–56/56); the hosted 26B answers almost everything with lower selective precision on the fresh claims.
- **Latency:** the local GPU wins single questions; Jev answers four questions in one request at single-question latency; the OpenRouter route is slowest with a long provider tail. The hosted 26B sends one request per question, issued concurrently.
- **Wording:** Jev and the 26B prefer questions slightly. The 26B's fresh suite rises 146 → 153 end to end in the v2 run: +6 from wording (8 fixes, 2 regressions) and +1 from request availability. E2B is mixed (above).
- n = 30 latency samples and these authored suites are development evidence, not production tails or domain accuracy.

## Targeted sets

**Identifiers** (14 reports with near-identical machines):

| Judge | 4 options | 20 options | Confidently wrong | ms per pick (4 / 20) |
|---|---|---|---|---|
| Gemma 4 E2B Q4_0 | 13/14 | 13/14 | 0 | 495 / 1139 |
| Qwen3-0.6B Q4_K_M | 11/14 | 8/14 | 0 | 198 / 446 |
| Qwen3-1.7B Q8_0 | 10/14 | 12/14 | 4 / 2 | 616 / 1488 |

Exact mentions narrowed 20 candidates to 1–3 in 12/14 reports and kept the right answer every time.

**Unstated** (10 reports): plain true/false gave "false" at p_true 0.00 on all 4 silent reports. Three-way: Gemma 4 E2B 9/10, with no confident errors; Qwen3-0.6B 6/10.

**Procedures** (10 situations on the bundled catalog): 8/10, and both misses were abstentions.
- The first design (one three-way question per card) scored 1/10.
- Splitting yes/no triggers from three-way exclusions fixed it.

Card cost with the default judge:

| Cards | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| Time | 300 ms | 1.27 s | 2.44 s | 4.60 s |

**Topical heads** (4 topics, disjoint templates; the judge alone gets 80/80 at 458 ms):

| Head @ labels per class | Accuracy | Settled (wrong) | Judge calls | Cascade ms |
|---|---|---|---|---|
| embedding @ 10 / 30 / 100 | 0.54 / 0.55 / 0.53 | 0 / 0 / 7 (0) | 100% / 100% / 91% | 462 / 462 / 422 |
| lexical @ 30 | 0.76 | 21 (0) | 74% | 338 |
| lexical @ 100 | 0.74 | 48 (3) | 40% | 183 |

With the minimal-pair results (embedding heads at chance), this is why v0.19 removes the embedding head.

**Prompt examples** (old 6-family set, Qwen): no consistent gain, at 1.7–2.5× latency.

**Retrieved precedents (E5, default judge, fresh suite, leave-one-pair-out, 2 pairs per item):**

| Arm | Accuracy | Settled at 0.9 | Precision at 0.9 | p50 ms |
|---|---|---|---|---|
| none | 0.856 | 90.0% | 88.9% | 334 |
| random pairs | 0.887 | 80.6% | 91.5% | 876 |
| retrieved (BM25 nearest pairs) | 0.869 | 86.3% | 89.9% | 870 |

Under the 5-point bar at 2.6× latency: prompt examples were deleted in v0.21.

**Staged vs eager with the input cache (E4, removed `schedule`, route_ticket):** two stages took 243 ms when the first case decided (eager 942 ms), and were within 0–8% of eager when a later case decided. One round per case was 8–40% slower than eager. The `schedule` option was deleted.

**Exception phrasing** (31 exception items): "Except: X." got Gemma 4 E2B to 30/31 (from 27/31), and Qwen3-0.6B went 28 → 27. It was chosen on the items it was scored on.

## Option position, order averaging, comparative choice

**Option position** (`bench/positions.py`, default judge, 24 picks, correct option moved through a 20-option list):

| Slot | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Top answer correct | 21 | 24 | 24 | 24 | 24 |
| Settled wrong | 3 | 0 | 0 | 0 | 0 |

All three misses are ambiguous items whose plausible rival came second in that layout. A direct two-option test flips confidently: "Não consigo entrar, então nem vejo minhas faturas" → billing 0.95 as [account, billing], account 0.85 as [billing, account]. In 3 of the 4 ambiguous cases the second-listed option won; clear items don't move.

**Averaging forward and reversed orders** (removed `permute`, default judge):

| Suite | Wrong, one order → two | Settled, one → two | Median ms, one → two |
|---|---|---|---|
| selection (5 options) | 1 → 1 | 22 → 21 | 456 → 625 |
| fresh (5 options) | 0 → 0 | 20 → 20 | 594 → 835 |
| selection, 20 options | 0 → 0 | 23 → 20 | 1,045 → 1,738 |

Rejected: no error removed, coverage lost, latency up.

**Comparative choice** (`bench/comparative.py`, 6 scenarios × 2 opposite goals × 2 orders = 24):

| Judge | Top-1 | Settled (wrong) | Order-stable pairs | Goal-sensitive scenarios |
|---|---|---|---|---|
| Gemma 4 E2B | 20 | 23 (3) | 12/12 | 4/6 |
| Gemma 4 E4B QAT | 24 | 22 (0) | 12/12 | 6/6 |

The default's misses answer the situation instead of a contradicting goal (sunny commute → bike for "keeps us dry in a storm"; rarely-read photos → cold storage for "lowest read latency").

**Cost split** (default judge, one 45-token check): 23-token input 405 ms cold vs 170 ms with the input cached; 213 tokens 1,147 vs 199 ms. See `EXPERIMENTS.md` E7.

## Probe on the judge's hidden state (removed `probe`)

Default judge, checks. The features are the final-layer hidden state at the question's last token, from the same decode as the answer logits. The probe is logistic regression (C = 0.05, standardised). Within each claim it is evaluated leave-one-pair-out, with the given number of training pairs.

| Readout | Fresh: accuracy | Fresh: AUC | Fresh: settled at 0.9 | Fresh: precision at 0.9 | Selection: accuracy / precision at 0.9 |
|---|---|---|---|---|---|
| answer tokens (the judge today) | 0.856 | 0.916 | 90.6% | 89.0% | 0.963 / 98.7% |
| generic probe (trained on selection, tested on fresh) | 0.881 | 0.894 | 95.0% | 88.2% | — |
| per-claim probe, 2 pairs | 0.819 | 0.864 | 73.8% | 89.0% | 0.954 / 97.7% |
| per-claim probe, 4 pairs | 0.887 | 0.951 | 82.5% | 94.7% | 0.967 / 98.2% |
| per-claim probe, 6 pairs | 0.912 | 0.953 | 79.4% | 95.3% | 0.963 / 98.2% |
| per-claim probe, 9 pairs | **0.925** | **0.986** | 81.2% | **97.7%** | 0.975 / 98.3% |

**Retracted (v0.23):** the per-claim rows above held out one pair at a time, but the Portuguese items translate the English ones, so a test item's translated twin stayed in training. Holding whole translation families out (16 labels per claim):

| Readout | Fresh: accuracy | AUC | Settled at 0.9 | Precision at 0.9 |
|---|---|---|---|---|
| judge | 0.856 | 0.916 | 90.6% | 89.0% |
| probe | 0.844 | 0.919 | 76.2% | 92.6% |
| Platt calibration | 0.875 | 0.909 | 66.2% | 90.6% |
| free cutpoint | 0.831 | — | — | — |
| probe, 16 labels with the translated twin kept (control) | 0.931 | 0.984 | 79.4% | 98.4% |

None beats the judge on new families beyond noise (one standard error is about 2.7 points). The probe was removed from the library (spec §5.4; E9).

**Lock validation** (`semantic_switch.tools.lock`, 300 family splits, target 0.95, true side):
- **Fresh suite:** θ chosen on one half held 83.4% precision on the other, and 0 of 300 splits met the target. Default θ = 0.9 held 81.8%.
- **Selection suite:** 97.0%, gate passed 255 of 300.

**Other findings:**
- **E2B → E4B fallback on uncertain answers:** 9.4% escalated, precision 89.9%, unchanged from E2B alone. E4B alone reaches 94.7%.
- **Answer mass on the default judge's 23 fresh-suite errors:** 0.9986, against 0.9987 for its correct answers.

## GPU (removed `gpu`, E3)

| | E2B CPU | E2B GPU (Arc B390, Vulkan) | E4B GPU |
|---|---|---|---|
| check, 20 / 261 tokens | 324 / 1,009 ms | 147 / 201 ms | 248 / 330 ms |
| extra question, input cached | 158 ms | 61 ms | 91 ms |
| fresh suite: accuracy, precision at 0.9 | 0.856, 88.9% | 0.863, 89.0% | 0.938, 94.7% |

CPU and GPU agree on the default judge (1 flip in 160 fresh items; identical to two decimals on the probe). Qwen3-0.6B Q4_K_M does not: +0.10–0.26 on Vulkan, 2 of 12 probe answers flipped. `device="cpu"` in a Vulkan build was 2× slower until it loaded with `no_host` and op offload off. Hidden-state output on every token cost 7–17% on the GPU, so it is enabled only for probe questions.

## Batching inputs and the question-first layout (v0.24)

**v0.26, as implemented** (route_ticket, 16 short inputs, full switch with staged rounds, default judge): GPU `sw.many` 4.1 inputs/s → `sw.run(list)` 6.3/s (1.5×, windows of 8); CPU 0.8/s either way (window of 1).

**One decode over N inputs vs N sequential calls** (default judge, one question per input, identical scores):

| Inputs | GPU, 20 tokens | GPU, 261 tokens | CPU, 20 tokens | CPU, 261 tokens |
|---|---|---|---|---|
| 4 | 2.54× | 1.30× | 1.03× | 1.10× |
| 8 | 3.48× | 1.39× | 1.13× | 0.92× |
| 16 | 3.91× | 1.39× | — | — |

**Question-first layout** (question cached, each input decoded after it; fresh suite, same claim streamed):

| Layout | GPU ms / input | CPU ms / input | Accuracy | Settled at 0.9 | Precision |
|---|---|---|---|---|---|
| input-first (current) | 104 | 289 | 0.863 / 0.856 | 0.906 | 0.890 |
| question-first | 110 | 325 | 0.863 | 0.800 | 0.930–0.938 |

**LoRA mechanics** (hand-built rank-32 Q/K/V adapter, Gemma 4 E2B):
- zero adapter: identical scores;
- question-only adapter: cached input stays byte-identical;
- cost: +2 ms per question set on the GPU (59 → 61 ms; 7 questions 137 → 139 ms); +4–7% on the CPU; whole-prompt +1 ms GPU / +7% CPU.

## Equipment benchmark, the permanent gate (`bench/equipment.py`, v0.26)

`python -m bench.equipment --device gpu` → `bench/results/equipment-v026.json`.

**The set:**
- 56 items over **46 distinct reports** (23 situations, each in English and Portuguese; translations share a group);
- 4 facts: overheating, three-way leak, abnormal vibration, failure to start;
- 12 fictional cards for 2 near-identical machines, plus an unsupported look-alike (`AX-410BX`).

**Traps:** firmware and pressure boundaries (`fw 2/3`, `0.99 / 1.0 bar`), other units, past and resolved faults, hypotheticals, operator-vs-controller contradictions (E17 claimed but not in `active_alarms`, and the reverse), leak precedence, silence about the leak (correct answer: `needs_input`), missing pressure (`needs_input`), and no applicable procedure.

**Expected answers** come from the declared gold facts and the exact fields in code, never from a judge.

**Three shapes:**
- `select` over named facts (`{fact: …}`);
- `select` over plain claim strings;
- hand-written facts plus exact cases.

**Results, GPU** (three runs: identical decisions, latency ranges; θ sets every named check):

| Judge | Shape | Default θ 0.9 | Best θ at equal error (0 wrong) | Wrong actions | `needs_input` correct | No model call | Median |
|---|---|---|---|---|---|---|---|
| E2B | catalogue, facts | 22/56 | 41/56 at 0.6 | 0 | 4 (6 at 0.6) | 2 | 126–136 ms |
| E2B | catalogue, strings | 42/56 | — | 0 | 8 | 2 | 217–227 ms |
| E2B | facts + cases | 22/56 | 41/56 at 0.6 | 0 | 4 (6 at 0.6) | 2 | 142–148 ms |
| E2B | judge picks from the whole catalogue (`--direct`) | 7/56 | — | **4** | 0 | 0 | 479–532 ms |
| E4B | catalogue, facts | **56/56** | 56/56 (0.6–0.9) | 0 | 8 | 2 | 327–404 ms |
| E4B | catalogue, strings | **56/56** | — | 0 | 8 | 2 | 318–809 ms |
| E4B | facts + cases | **56/56** | 56/56 (0.6–0.9) | 0 | 8 | 2 | 348–456 ms |
| E4B | judge picks from the whole catalogue (`--direct`) | 17/56 | — | **9** | 0 | 0 | 732 ms |

**Readings:**
- **Observed structure effect:** the same judges choosing directly from the whole catalogue (rules spelled out in the options) got 7 and 17 right with 4 and 9 wrong procedures, and never once answered `needs_input`. Inside the pipeline, E4B gets all 56.
- **No wrong action was observed in any pipeline shape** on these 56 items, for either judge or any θ from 0.6 to 0.99; E2B abstained on 34 (named facts) and 14 (strings) items at 0.9. Exact eligibility and three-way exclusions turned these judge errors into abstentions here; they do not prevent every confidently wrong semantic action.
- **The catalogue and the hand-written cases agree on every θ** (the "one engine" invariant, now on 56 items).
- **E2B's losses are mostly one judgment:** "Sem vazamento" scores p(states false) 0.86 on the three-way leak question, under θ 0.9, so the leak card stays unresolved and blocks first-match. Lowering θ to 0.6 recovers 19 decisions with no wrong action, which is what `ss lock` would find with labels. E4B settles them all at 0.9.
- **Strings vs named facts on E2B (42 vs 22):** a plain leak string in `applies_when` compiles to a two-way check, which E2B answers confidently; the named three-way `leak` must establish "states false". Same safety, different abstention. Prefer the named three-way fact when silence matters, and lock it.
- **Latency:** E4B costs about 2.5× E2B on the GPU, and the pipeline is faster than one whole-catalogue pick. Single-machine latencies vary run to run (E4B strings 809, 406, 318 ms).

## Equipment procedures (removed `equipment_jev`, v0.16–v0.21; superseded by the gate above)

34 cases over 16 fictional procedures for 4 near-identical machines, with firmware and pressure boundaries, misleading suggestions, other units, past faults and missing observations. **Only 10 distinct report texts and 2 facts**, so treat differences of a few cases as noise. Both switch designs reproduce every label given perfect semantic answers.

| Design (default judge, Gemma 4 E2B) | Correct | Wrong procedures | Median |
|---|---|---|---|
| Whole catalogue, one pick | 9/34 | 23 | 6,488 ms |
| `select`, trigger questions carrying card titles (v0.16) | 17/34 | 0–2 | 1,123–1,694 ms |
| `select`, same cards **without titles** | **28/34** | 0 | 1,025 ms |
| Shared facts + exact cases, staged | 26/34 | 0 | 857–1,110 ms |
| Shared facts + exact cases, eager | 26/34 | 0 | 726 ms |
| **v0.21 `select` over shared facts** (`{fact: hot}`, `{fact: leak}`) | 26/34 | 0 | 691 ms |
| **v0.21 shared facts + exact cases** | 26/34 | 0 | 806 ms |
| v0.21 whole catalogue, one pick (`min_score=0.5`) | 4/34 | 19 (18 ineligible) | 15.7 s |

**v0.21 rerun** (`bench/results/equipment.json`, `equipment-switch.json`): the catalogue and the hand-written facts + cases give **identical outcomes on all 34 cases** (26 correct, 4 correct `needs_input`, 4 to review, no wrong procedure), the "one engine" invariant on the real judge. The first rerun exposed a bug, fixed and tested: cards whose claims were all `{fact: …}` references were reported as unexamined, turning 4 correct `needs_input` answers into `uncertain`. The budget never applies here because every claim is a named fact. Latencies are single runs on a loaded machine.

**Other judges, facts + cases:**

| Judge | Correct | Wrong | Median |
|---|---|---|---|
| Qwen3-0.6B | 2/34 | **8** | 245 ms |
| Qwen3.5-0.8B | 2/34 | 0 (defers 32) | 486 ms |
| Qwen3.5-2B Q4_K_M | 26/34 | 0 | 1,375 ms (2 rounds on 26 cases) |
| Jev (hosted) | 30/34 | 0 | 549 ms |

Given the whole catalogue, Jev got 31/34 (at ≥ 0.9 confidence: 22 accepted, all correct) and the Qwen3.5 models 2/34.

**Readings:**
- **Asking about the procedure instead of the situation was the largest single loss** (17 → 28 by removing titles).
- **Exact rules prevent wrong procedures:** no design with exact eligibility committed a wrong procedure on the default judge.
- **BM25 had nothing to do:** eligibility left at most 2 cards per input.
- **Sub-1B judges fail this workload zero-shot.**

## Latency and backend

| | Gemma 4 E2B | Qwen3-0.6B |
|---|---|---|
| 1 question, 261-token input | 1.32 s | 0.72 s |
| the same input scored twice (two calls) | 2.73 s | 1.47 s |
| 2 questions in one call | 1.65 s | 0.84 s |

**Readings:**
- **Every repeated round re-reads the input**, which is the motivation for the input cache (exact reuse, v0.21).
- **Prefix reuse was measured and dropped:** trimming a cached input to a shared prefix and decoding the rest changed scores (Qwen3-0.6B probe: 0.11 vs 0.06 and 0.60 vs 0.48 on two items, 9/12 vs 10/12 correct), because quantised kernels depend on batch shape. Exact reuse is byte-identical to a fresh read.
- **Flash attention is mixed:** 1.32 → 1.63 s on Gemma, 0.72 → 0.65 s on Qwen.
- **4-bit is faster** (Q4_0 / Q4_K repack, no Q8_0 path): Gemma 4 E2B takes 228 ms at Q4_0 vs 440 ms at Q8_0.
- **Shared prefix:** a 6-question ask takes 0.83 s instead of 2.30 s (Gemma) and 1.28 s instead of 3.62 s (Qwen3.5-2B, hybrid; D1 exact).
- **Catalog index:** 12 ms per request at 1,000 cards.
- **Rendering:** 0.3 µs.

## Execution shape (`bench/execution_shape.py`)

Default judge (Gemma 4 E2B Q4_0), selection + fresh suites, every suite question asked of every input (4,524 questions per arm), public `Runtime.ask_many`, default policy. Arms: **batched** (multi-input windows on the GPU, all questions co-decoded per input), **fitting** (one question per input across input windows, the lock-fitting shape), **serial** (the ordinary `Model` judge: prefix cached, each input decoded alone, each question decoded alone). Batched and fitting now run through the explicit diagnostic primitive `LlamaScorer(grouping="batched", inputs=8)`; ordinary judges have no grouping option. One process per configuration under a 9 GiB cap; CPU timings ran beside other host load.

| Configuration | batched vs serial | fitting vs serial | batched vs fitting | ms/question batched → serial |
|---|---|---|---|---|
| GPU | 5 changes, max Δp 0.051 | 6, 0.065 | 3, 0.043 | 26 → 88 (3.3×) |
| CPU, repack (`q4_0_8x8`) | 16, 0.137 | 0, 0.0 | 16, 0.137 | 223 → 216 |
| CPU, `repack=False` | 24, 0.168 | 0, 0.0 | 24, 0.168 | 413 → 443 |

Changes are accepted value/status flips, all at the 0.9/0.1 boundary; gold-scored accuracy moves by at most 5 accepted rows (379/400 vs 374/395 on the CPU). Across devices the serial arm still differs (GPU vs CPU 19 changes, CPU repack on/off 18), so device and repacking stay in execution identity. Because grouping changed accepted decisions on the default judge, every ordinary check, switch, lock fit and evaluation uses the serial shape: the lock records the realized `shared` mode, `grouping` and the serial contract revision (`serial: 1`), and admission rejects or (with `allow_model_change`) resets policies when any differs, including locks fitted before the revision existed. Multiple resident guidance prefixes stay enabled: on Qwen3-0.6B (CPU, `tests/test_config_and_lifecycle.py`) asking a guided pick after an unguided check moves its scores by up to 0.028 against the same pick decoded cold, because the other prefix stays resident in the unified KV cache, but no accepted value/status changed across composed, warm, cold and `n_batch` 512/1024 runs, so this does not justify single-prefix execution. The batched primitive is 3.3× faster per question on the GPU and not faster on the CPU; it is a throughput diagnostic, not an application option.

## Fixed four-lane execution geometry (rejected, `results/execution-geometry.json`)

Candidate: a diagnostic scheduler below Judge that decoded prefix, state and suffix segments in separate stages, each submission holding exactly four lanes of one length class (16-token multiples through 128, then 128-token multiples; 4 output rows in the suffix stage, none otherwise). Unused rows were dummy tokens in four isolated sequences, removed after each submission; real text was never padded. It ran as `LlamaScorer(grouping="lanes", inputs=8, prefix_slots=8)`; serial is the ordinary `Model` judge (`prefix_slots` 4, `inputs` 1). Workload: the 25 S1MB go-emotions cases (28 guided checks each; every question has its own guidance prefix, 75 + 35 + 25 tokens), default judge, public `Runtime.ask`, plus `bench/latency.py` single/four-question timing. Repeatability compares 5 cases against a cold full call under warm, evicted, reversed and shuffled order, 1/4/17-question and tail calls, mixed guided/unguided with a 5-option long-suffix Pick.

| | GPU lanes | GPU serial | CPU lanes | CPU serial |
|---|---|---|---|---|
| 28-question case p50 / p95 | 2.61 / 3.47 s | 5.48 / 5.55 s | 17.6 / 24.5 s | 16.3 / 17.9 s |
| one question, new input p50 | 200 ms | 112 ms | 2,284 ms | 311 ms |
| four questions, new input p50 | 338 ms | 301 ms | 3,086 ms | 996 ms |
| submissions / tokens (25 cases) | 525 / 104,832 | 2,100 / 91,150 | same | same |
| accepted changes (max Δp) across call shapes | 0 (0.014) | 0 (0.011) | 2 (0.066) | 2 (0.067) |

Real work was identical (91,150 tokens: 28 distinct prefixes exceed either residency); dummy work was 15% of real. Order, neighbours, warm and evicted caches were bit-identical for lanes on both devices; call size and mixed guidance moved scores on both schedulers, through visible KV extent rather than lane placement. **Rejected:** the GPU 28-question p50 missed 2 s, single-question latency was 1.79× serial (on the CPU 7.3× at one question and 3.1× at four), and the CPU lanes changed accepted outcomes. Acceptance fitting on the cohort was therefore not run. Serial remains the only production scheduler; the lane code was deleted. The 2.1× GPU gain comes from fewer submissions, not less work; reaching 2 s on this workload needs less real prefix work (guidance residency), not a batch geometry.

## Native execution reproduction (`execution_reproduction.py`, `results/execution-reproduction.json`)

Fixed numerical configuration: default judge, `device="cpu"`, repack on (`q4_0_8x8`), 16 threads, `n_ctx` 8192, `n_batch` 512, `prefix_slots` 4, serial grouping, llama-cpp-python 0.3.36 (the build of the geometry run). Only question count, question order and prefix/state cache history vary. Workload: the first 5 S1MB go-emotions cases (28 guided checks; prefix 75 / state 35 / suffix 25 tokens), public `Runtime.ask`; "cold" resets the scorer cache. Per case: cold full call (reference), cold repeat, warm repeat, after another case, reversed, shuffled, each of the first four questions alone, 4 / 17 / last-17 questions, first 10 plus an unguided 5-option Pick. Then the two recorded flip targets alone after controlled histories, and alone after N filler tokens (two contents) decoded into a spare sequence.

| | case 0 `annoyance` (recorded serial CPU flip) | case 4 `surprise` (recorded lanes flip) |
|---|---|---|
| cold full call | uncertain, p .893991 (recorded serial reference, bit-identical) | uncertain, p .100985 |
| repeat, warm, after another case, reversed, shuffled, 4, 17 questions | unchanged, bit-identical | unchanged |
| alone (cold, warm), after 1, 2, 4, 5, 8 other prefixes | **ok/true**, p .901504–.910896 | uncertain, .100985–.107494 |
| after 3 other prefixes (separate or one call) | uncertain, .893991 | uncertain, .114639 |
| filler offset scan, 0–256 tokens step 4 | 16 / 65 offsets uncertain, p .8859–.9118 | 8 / 65 offsets **ok/false**, p .0980–.1129 |

Reproduced: the same accepted flips as the geometry run (`one` and mixed guidance on `annoyance`), with bit-identical scores; cases 1–3 had no accepted change in any arm. Order and repetition never change a score: the judge sorts requests by prefix, so a call's question set fixes its decode history. **Transition:** the cell offset at which the target's freshly decoded prefix (and the state and suffix after it) lands in the unified KV buffer. In the scan the score is a function of that offset alone, periodic in 64 cells (14–15 distinct values per target, identical for both filler contents). Call size, mixed guidance and resident-prefix eviction act through that offset: the histories repeat with period 4 (the prefix-slot count), consistent with eviction returning the same freed cells, and a prefix restored from a resident slot keeps its cells and its original score. No layout change is proposed; the effect is CPU-backend accumulation order (Δp ≤ .026 at these items), and repeatable for a fixed history.

## Native layout and reducer (`backend_diagnostics.py`, `results/native-layout.json`)

`python -m bench.backend_diagnostics {reducer,layout} --model GGUF --device gpu`. Pipeline-built Gemma 4 E2B Q8_0, Vulkan (Arc iGPU), `n_ctx` 8192, `n_batch` 512, `prefix_slots` 4, serial, public `Runtime.ask`. **Right-sized** is the canonical `backend.capacity` (`n_seq_max` 6, `n_outputs_max` 6, one output per decode, resident prefixes in sequences 3–5). **Reserved-37** is the previous layout, built only inside the diagnostic by substituting a `Capacity` subclass: `n_seq_max` = `n_outputs_max` = 37, one output per decode, residents 34–36. The unsafe one-reserved-row configuration is never built.

**Reducer** (`reduce_row`, 262,144-token vocabulary, real logits and slots; warm repeats, 10 calls each): reduction p50 0.9 ms of 88 ms for a Check (2 slots × 4 ids), 2.7 of 126 ms for a 5-option Pick, 1.8 of 480 ms for a 400-token suffix, 55 of 10,915 ms for 28 guided Checks: 0.4–2.1% of the call. In isolation 1.05–1.27 ms p50; peak traced allocation 65 KiB with the reusable fp64 scratch (2.1 MiB without). Decode, mostly the wait for logits, is the cost. The fp64 stable reducer and packed slots are kept; no replacement was measured.

**Layout** (5 S1MB go-emotions cases × cold, cold repeat, warm, evicted, reversed, shuffled, 4 × one, 4, 17, last 17, 10 guided + 5-option Pick, 5 unguided, 28 guided + 400-token Check, plus the two history targets: 1,311 answers per run; runs alternated right, reserved, right, reserved, each a fresh load):

| | right-sized | reserved-37 |
|---|---|---|
| within layout (run 1 vs 2) | 1,311 / 1,311 bit-identical | 1,311 / 1,311 bit-identical |
| across layouts (both pairs) | 1,311 / 1,311 bit-identical: 0 answer, 0 acceptance flips, max Δp 0 | |
| cold 28 questions p50 / p95 | 8.16 / 12.08 s | 7.08 / 7.84 s |
| one question p50 / p95 | 258 / 454 ms | 248 / 539 ms |
| 17 questions p50 | 5.29 s | 4.14 s |
| output buffer realized / estimated | 6.0 / 12.0 MiB | 37.0 / 74.0 MiB |
| KV, compute (Vulkan0) | 144, 120.4 MiB | 144, 120.9 MiB |

No borderline acceptance flip exists to report: every score is bit-identical. The latency gap in this table follows run order, not layout: a reversed-order control (reserved first, 2 cases without histories, `order_control`) totalled 113.7 / 109.6 / 113.9 / 115.6 s for reserved, right, reserved, right, also bit-identical across layouts. Latencies are reported per layout and never pooled. `n_seq_max` and `n_outputs_max` are the only identity fields that change, so a lock fitted under one layout is rejected under the other. Right-sized stays canonical: same scores, no measurable latency difference, 31 MiB less output buffer.

## Hosted Jev

Matched, warm, network included for Jev:

| Input | Jev | Qwen3-0.6B | Gemma 4 E2B |
|---|---|---|---|
| 20 tokens, 1 question | 270 ms | 98 ms | 247 ms |
| 20 tokens, 7 questions | 292 ms | 352 ms | 906 ms |
| 261 tokens, 1 question | 286 ms | 423 ms | 1,179 ms |
| 261 tokens, 7 questions | 281 ms | 706 ms | 2,034 ms |

**Selection suite:**
- Jev 235/240, with all 216 answers at ≥ 0.9 correct and ECE 0.053;
- Gemma 4 E2B 231/240;
- Qwen3-0.6B 176/240.

This earlier selection run kept no raw rows. The fresh suite and a recomputable selection run are in the 2026-10-10 section above.

**Contract:** TypeSafe's Master Customer Agreement §2.3(b) forbids using Jev outputs for distillation, to train a model to imitate it, or to develop a competing product. Further Jev runs wait for a reading of it or consent, and Jev outputs never enter training data.

## S1MB external subset: first 25 cases per benchmark (`bench/external_primitives.py`)

Pinned S1MB evaluator (commit `39f122b`, dataset revision `b8dc678`, `english-v1`) over public Check (Noul), Pick (Choice) and Rate (Score). The E2B arm recorded the first 25 cases of all 137 benchmarks; the E4B (`--precise`) run was stopped after 2,233 of 3,388 cases. `python -m bench.external_primitives --compare bench/results/s1mb-e2b.json bench/results/s1mb-e4b.json --output bench/results/s1mb-paired.json` reads both checkpoints without changing them and scores each arm's matched predictions with the pinned evaluator from the cached case definitions. It loads no model and uses no download or network. The report stores both artifact paths, configuration digests and the shared evaluator/dataset provenance and case digest; arms recorded with different inputs are refused.

Pairing matches cases by benchmark and case ID and questions by question ID. Duplicate question IDs, differing question sets, tasks or targets fail before any comparison, so question order cannot create transitions. Population: 2,233 matched cases in 91 benchmarks and 3,747 judgments. Availability uses all 3,747: E2B ok 2,039, uncertain 1,459, unsupported 226, needs_input 12, failed 11; E4B ok 2,501, uncertain 997, and the same 226/12/11. Those 249 judgments are unscored in both arms and are counted, not compared. Model-quality comparisons use the 3,498 jointly scored judgments. Because these are the first N cases, several questions share a case and no benchmark is complete, the counts are neither leaderboard results nor independent samples. The evaluator's own task scores are therefore null.

Evaluator benchmark means. Each value is the equal-weight mean over benchmarks of the evaluator's primary metric, and N/A when any benchmark's value is undefined:

| Task (metric) | Benchmarks (with failed cases) | E2B | E4B |
|---|---|---|---|
| Noul (Brier, lower is better) | 41 (2) | .248 | .223 |
| Choice (target mass at prediction) | 42 (7) | N/A | N/A |
| Score (normalized MAE, lower is better) | 8 (0) | .286 | .212 |

Choice is N/A because the large catalogues below have no supported prediction.

Pooled hard-label judgments (target mass 1 on one answer). Accepted means settled at the default 0.9; selective precision is correct / accepted. A fix is a wrong E2B accepted answer that E4B accepts correctly; a regression is the reverse:

| Group | Items | E2B accepted: correct / wrong (selective precision) | E4B accepted: correct / wrong (selective precision) | Fixes | Regressions |
|---|---|---|---|---|---|
| All | 3,257 | 1,914: 1,550 / 364 (81.0%) | 2,341: 1,923 / 418 (82.1%) | 56 | 27 |
| Noul | 1,695 | 1,204: 1,009 / 195 (83.8%) | 1,356: 1,155 / 201 (85.2%) | 31 | 18 |
| Choice | 1,364 | 687: 523 / 164 (76.1%) | 925: 725 / 200 (78.4%) | 25 | 9 |
| Score | 198 | 23: 18 / 5 (78.3%) | 60: 43 / 17 (71.7%) | 0 | 0 |
| In audit scope | 2,640 | 1,668: 1,418 / 250 (85.0%) | 2,037: 1,731 / 306 (85.0%) | 33 | 16 |
| Out of audit scope | 617 | 246: 132 / 114 (53.7%) | 304: 192 / 112 (63.2%) | 23 | 11 |

Accepted transitions (all hard labels): E4B settles 700 items that E2B left unsettled, 213 of them wrongly, and leaves unsettled 143 items that E2B got right. Raw top-answer transitions are separate and ignore acceptance: E2B's top answer is correct on 2,198 and E4B's on 2,359; 412 wrong → correct, 251 correct → wrong. Soft targets (241 items) report expected agreement, not precision: E2B .560 over 125 accepted answers and E4B .582 over 160. Pooling soft and hard targets over all jointly scored judgments, E2B accepts 2,039 with expected agreement .7945 and E4B accepts 2,501 with .8061; that pooled figure is not a hard-label precision. Selective precision at the default threshold is not a precision ceiling, and neither the extreme thresholds nor the 24-case audit below estimate a benchmark-wide label-noise rate.

Score transport: `distribution()` maps public `result.scores` by authored option ID. A sub-stochastic lower bound, such as a hosted true .08 / false .72, is kept as returned in the row's `scores` and marked unsupported for the original evaluator, which then fails that whole case. It is never reconstructed as `1 − p_true` or renormalized. The saved local arms have complete scores and are unaffected.

### Jev on the same 2,233 cases (`results/s1mb-jev.json`, `results/s1mb-matched-jev.json`)

S1MB is a community benchmark; whether any arm's training data overlaps it is unknown. Jev is a focused-model reference of undisclosed size. Its outputs are kept only in these benchmark artifacts and never enter training data.

`TYPESAFE_API_KEY=… python -m bench.external_primitives --judge jev --limit 25 --matched bench/results/s1mb-e2b.json bench/results/s1mb-e4b.json --output bench/results/s1mb-jev.json` sent exactly the cases recorded by both local arms (2,233; same pinned evaluator, dataset revision and case file digest; the key set digest is in the configuration). It made one `Runtime.ask` per case through `JevJudge` (Jev 1.13.0; every response reported `jev-1.13.0` except one failed request). Each row records end-to-end case milliseconds and each request's elapsed time and provider token usage. Rows recorded from now on also carry `input_characters` (the JSON-serialized case state).

`python -m bench.s1mb_matched --arm e2b … --arm e4b … --arm jev … --output bench/results/s1mb-matched-jev.json` is model-free and makes no network calls. It joins the three arms in one pass, by benchmark/case key and question ID, through the shared `aligned` validation. Before aggregating, every arm must have the same evaluator/dataset provenance and case-file digest, case identity, task, gold targets, question IDs and authored option IDs. A changed definition, duplicate question ID or conflicting gold fails, and reordering cases or questions does not change the report. Model/provider configurations are expected to differ and are stored as digests. Acceptance is replayed on the recorded scores (top answer, score ≥ θ, answer mass ≥ .5 where reported). At .9 the replay has 0 mismatches against the recorded decisions in all three arms.

**Populations.** These are distinct populations, kept separate:

- 2,233 common cases and 3,747 judgments, used for availability. Jev: ok 1,887, uncertain 1,610, unsupported 226 (the same large catalogues), needs_input 12, failed 12; E2B and E4B as above.
- 3,486 judgments scored by all three arms. Unscored judgments (E2B 249, E4B 249, Jev 250) count towards availability and are never given invented scores.
- 3,245 of the jointly scored judgments have hard labels.
- 241 soft-target judgments.

Evaluator benchmark means over the same cases: Noul Brier .107 (E2B .248, E4B .223); Score normalized MAE .142 (.286, .212; 4 of Jev's 8 Score benchmarks contain a failed case); Choice N/A for all.

Hard-label curve on the jointly scored 3,245. Each cell is accepted: correct / wrong / abstained (selective precision, coverage = accepted / 3,245). Raw top answer correct: E2B 2,194, E4B 2,354, Jev 2,612.

| θ | E2B | E4B | Jev |
|---|---|---|---|
| .5 | 3,119: 2,155 / 964 / 126 (69.1%, 96.1%) | 3,167: 2,326 / 841 / 78 (73.4%, 97.6%) | 3,192: 2,593 / 599 / 53 (81.2%, 98.4%) |
| .9 | 1,911: 1,549 / 362 / 1,334 (81.1%, 58.9%) | 2,338: 1,921 / 417 / 907 (82.2%, 72.0%) | 1,812: 1,695 / 117 / 1,433 (93.5%, 55.8%) |
| .95 | 1,446: 1,235 / 211 / 1,799 (85.4%, 44.6%) | 2,027: 1,720 / 307 / 1,218 (84.9%, 62.5%) | 1,313: 1,247 / 66 / 1,932 (95.0%, 40.5%) |
| .99 | 554: 520 / 34 / 2,691 (93.9%, 17.1%) | 1,377: 1,247 / 130 / 1,868 (90.6%, 42.4%) | 681: 659 / 22 / 2,564 (96.8%, 21.0%) |
| .999 | 77: 75 / 2 / 3,168 (97.4%, 2.4%) | 578: 545 / 33 / 2,667 (94.3%, 17.8%) | 496: 486 / 10 / 2,749 (98.0%, 15.3%) |

By primitive and scope at θ .9. Each cell is accepted: correct / wrong / abstained (selective precision), followed by the number of benchmarks that contribute accepted answers. The same declared scope tags apply to every arm. Out of scope includes the temporal and context-reading benchmarks (temporal NLI, context retention); their failures are reported, not excluded:

| Group | Items (benchmarks) | E2B | E4B | Jev |
|---|---|---|---|---|
| Noul | 1,695 (41) | 1,204: 1,009 / 195 / 491 (83.8%), 41 | 1,356: 1,155 / 201 / 339 (85.2%), 41 | 822: 792 / 30 / 873 (96.4%), 38 |
| Choice | 1,357 (38) | 684: 522 / 162 / 673 (76.3%), 33 | 922: 723 / 199 / 435 (78.4%), 37 | 894: 812 / 82 / 463 (90.8%), 36 |
| Score | 193 (8) | 23: 18 / 5 / 170 (78.3%), 3 | 60: 43 / 17 / 133 (71.7%), 7 | 96: 91 / 5 / 97 (94.8%), 8 |
| In scope | 2,630 (62) | 1,665: 1,417 / 248 / 965 (85.1%), 59 | 2,034: 1,729 / 305 / 596 (85.0%), 62 | 1,570: 1,471 / 99 / 1,060 (93.7%), 61 |
| Out of scope | 615 (25) | 246: 132 / 114 / 369 (53.7%), 18 | 304: 192 / 112 / 311 (63.2%), 23 | 242: 224 / 18 / 373 (92.6%), 21 |
| Noul, in scope | 1,445 (31) | 1,031: 914 / 117 / 414 (88.7%), 31 | 1,182: 1,044 / 138 / 263 (88.3%), 31 | 756: 728 / 28 / 689 (96.3%), 30 |
| Noul, out of scope | 250 (10) | 173: 95 / 78 / 77 (54.9%), 10 | 174: 111 / 63 / 76 (63.8%), 10 | 66: 64 / 2 / 184 (97.0%), 8 |
| Choice, in scope | 1,091 (27) | 619: 488 / 131 / 472 (78.8%), 26 | 815: 656 / 159 / 276 (80.5%), 27 | 769: 699 / 70 / 322 (90.9%), 27 |
| Choice, out of scope | 266 (11) | 65: 34 / 31 / 201 (52.3%), 7 | 107: 67 / 40 / 159 (62.6%), 10 | 125: 113 / 12 / 141 (90.4%), 9 |
| Score, in scope | 94 (4) | 15: 15 / 0 / 79 (100.0%), 2 | 37: 29 / 8 / 57 (78.4%), 4 | 45: 44 / 1 / 49 (97.8%), 4 |
| Score, out of scope | 99 (4) | 8: 3 / 5 / 91 (37.5%), 1 | 23: 14 / 9 / 76 (60.9%), 3 | 51: 47 / 4 / 48 (92.2%), 4 |

The report keeps every primitive × scope curve at every θ.

**The accepted population changes with θ.** Jev accepts 822 binary (Noul) judgments at .9, 431 at .95, only 27 at .99 and none at .999. Its 496 accepted answers at .999 are therefore 448 Choice and 48 Score from 38 benchmarks, and the pooled .999 precision (98.0%) describes choice/score judgments, not binary confidence that keeps improving. E2B's .999 population is the opposite: 39 Noul and 38 Choice from 18 benchmarks. Each high-θ cell rests on a few benchmarks, and several questions share a case, so these are not independent samples. Neither a precision ceiling nor a causal claim about training versus size follows from these curves.

Accepted-outcome transitions at .9 (hard labels), counted as wrong → correct / correct → wrong / abstain → correct / abstain → wrong / correct → abstain / wrong → abstain: E2B → Jev 73 / 4 / 530 / 61 / 453 / 237; E4B → Jev 56 / 5 / 261 / 34 / 538 / 283; E2B → E4B 56 / 27 / 487 / 213 / 143 / 130. Raw top-answer transitions (wrong → correct / correct → wrong): E2B → Jev 646 / 228; E4B → Jev 474 / 216; E2B → E4B 412 / 251. Soft targets (241 items, expected agreement of accepted answers at .9, not precision): E2B .560 over 125, E4B .582 over 160, Jev .553 over 69.

**Timing.** Case timing is end to end and is reported separately from hosted request timing. Each cell is p50 / p95 ms (number of cases), using the latency helper's convention: the nearest-rank p95, and null for an empty group. A failed case has a judgment with status `failed`. A case has model work when it recorded a provider request; for the local arms, which record no requests, it is a case with at least one judgment that was neither unsupported nor missing its input. Both definitions give 2,121 cases.

| | All cases | Non-failed | With model work |
|---|---|---|---|
| E2B (laptop GPU) | 243 / 1,279 (2,233) | 245 / 1,279 (2,222) | 261 / 1,296 (2,121) |
| E4B (laptop GPU) | 387 / 2,525 (2,233) | 389 / 2,525 (2,222) | 426 / 2,757 (2,121) |
| Jev (one client host, network included) | 276 / 350 (2,233) | 276 / 349 (2,221) | 278 / 352 (2,121) |

Jev made 2,121 requests (1 failed, a 60 s read timeout). Request p50 / p95 was 277 / 350 ms, and the provider reported 1,630,196 input and 113,806 output tokens.

The local arms decode multi-question cases serially. The table splits cases with model work by recorded question count:

| Questions per case (cases with model work) | E2B | E4B | Jev | Jev provider tokens in / out |
|---|---|---|---|---|
| 1 (1,883) | 234 / 1,025 | 372 / 2,074 | 276 / 340 | 1,340,053 / 66,772 |
| 2 (146) | 587 / 811 | 853 / 1,184 | 295 / 411 | 125,621 / 9,944 |
| 7 (27) | 1,663 / 2,182 | 2,471 / 3,258 | 289 / 314 | 19,658 / 4,075 |
| 17 (25) | 4,546 / 7,259 | 5,755 / 8,688 | 316 / 390 | 95,834 / 15,550 |
| 28 (25) | 6,606 / 24,068 | 9,399 / 9,697 | 296 / 325 | 34,127 / 12,450 |

The other strata (3–6, 8 and 12 questions; 34 cases, 15 of them with model work) are in the report. The 12-question case had no model work in any arm.

These historical rows did not record input length, so the input-length strata are null (unavailable, not zero); the provider token counts are usage, not input length. Jev's flat latency across question counts comes from one request per case. It says nothing about Jev's internal parallelism, batching or datacenter execution.

## S1MB accepted-error audit (`--audit`, `results/s1mb-audit.json`)

Scope: the judge reads one answer token and stays confident when it is wrong. Arithmetic, logic, games and multi-step reasoning are out of scope, and their errors are excluded from the review: GSM8K, AQuA-RAT, FOL-NLI, logical entailment, RuleTaker, RobustLR, NLSat, CLadder, Corr2Cause, bAbI-NLI, LoNLI, temporal NLI, StepGame, SpartQA, BoardgameQA, and the Jev context-retention, reasoning, painting-geometry, tic-tac-toe, snake, drone and ViZDoom controls. An accepted error is an accepted answer whose target mass is below .5, so these counts include soft targets and differ from the hard-label table. The counts cover the same 2,233 matched cases:

| Arm | In scope: errors / accepted | Out of scope: errors / accepted |
|---|---|---|
| E2B | 284 / 1,792 | 114 / 247 |
| E4B | 346 / 2,197 | 112 / 304 |

Workflow controls, errors / accepted (E2B; E4B): invoice 8/19; 0/5. Security 0/11; 8/20. Customer service 2/20; 0/23. Agent trace 3/18; 4/19.

The review covers 24 in-scope errors: 6 each from invoice processing, security incidents, customer routing, and temporal/agency. Each one, and 19 matched correct contrasts, was rebuilt through public `Runtime.ask` with a recording `OpenRouterJudge` transport that records the request and runs no inference. The other 5 errors have no correct accepted answer to the same question in that arm. All 43 requests carried the authored instructions, the rendered input and the option descriptions in authored order.

Findings: 0 SDK mapping, 0 presentation, 4 task/gold, 20 model interpretation. Because no mapping or presentation defect was found, the SDK is unchanged.

- **Task/gold:**
  - Three workflow rules depend on rules defined in sibling cases: SCHEDULE on PAY, HOLD on PAY/SCHEDULE/SHORT PAY, and ESCALATE URGENT on the playbook branches. Each workflow case asks one action, so no judge that sees one question at a time receives them.
  - One churn item has a uniform target, so any accepted answer scores 1/3.
- **Model interpretation (definite: every condition is visible in the transported request):**
  - E4B treats an approved responder grant as the trigger for a security action. It does this in 5 of the 6 reviewed security cases, even when reach, event kind or mail delivery rules the action out, and twice with a grant that expired before the alert.
  - E2B answers "no" at 0.94–0.97 to invoice, customer-service and trace actions whose conditions are all literal matching fields.
  - E4B applies the "question about charges" exclusion to the opening turn even though the latest turn explicitly asks for a refund, and it lets a later cancellation override a stated equal priority.
  - Both arms miss that a grant for another resource, or no grant at all, leaves the mutation unauthorized.
  - E2B attributes the subject to the sending relay or follows an occupational stereotype.
- **Hypothesis, untested:** E2B's negative bias on action rules may come from the authored preamble ("Missing approval is not approval"). Testing that needs a separately invoked paired inference run.

These patterns go to the shared cohort under the artifact's `cohort_patterns` key, by kind (temporal, actor, exception, policy reading) with source IDs. They are development evidence, not independent quality claims.

Unsupported large catalogues stay visible and are not shortlisted or decomposed: Banking77 (77 options), CLINC (151), Gretel PII (36), HWU64 (64), MASSIVE (60) and MTOP (113) exceed the 20-option direct presentation.

## Procedure benchmark (`bench/policy_workflow.py --cohort`)

```
python -m bench.policy_workflow --cohort DIR --judge SPEC [--judge SPEC …] [--provider NAME] [--key-file PATH] [--control] [--chunk 16] [--retry-failed] [--fresh] --output result.json
```

`SPEC` is a local GGUF (default: E2B and E4B) or `openrouter:MODEL` (Jev is `openrouter:typesafe/jev-1.13`). The source is the small-context cohort: 32 authored families, 128 situations × EN/PT = 256 correlated observations, so the independent sample is 32 families, not 256 cases. Arms run sequentially. Cases are grouped by authored judgment and guidance and sent in chunks of `--chunk` through public `Check/Pick/Rate.run`; hosted concurrency stays inside the judge. `--control` adds the guidance-omission condition (the same primitives and evidence without the procedure, not a deployable arm); it is off by default. Every chunk is committed to `<output>.journal.jsonl`, keyed by arm, condition and case ID and bound to the code fingerprint, judge specs, provider, conditions, chunk size and cohort source/assignment digests; a rerun resumes the missing rows only and a changed configuration is refused. Completed / failed / remaining counts go to stderr after every chunk. Reports give raw correctness and failures first, then wall-clock throughput (cases per second, amortized ms per case) separately from per-attempt HTTP latency for hosted arms (public results do not expose per-request latency), then acceptance. Arm comparisons count only case IDs scored by both arms.

The table is recomputed from the saved rows of `results/procedure-cohort-*.json` and `results/small-context-cohort-base.json` (E2B, E4B) with the shared correctness reduction (`raw_correct`; accepted = status `ok`). Rows marked *chunked* come from the chunked runner (2026-10-10; guided only, 8 requests in flight, a provider pinned where named): latency is per final request including its retries and backoff, and the HTTP column counts every transport attempt including superseded retry passes. The other files predate it: each case was one sequential call, latency is per request over all 256 requests, attempts per request were not recorded, and the HTTP column counts transport attempts over the guided and control passes together. `Jev 1.13 (direct API)` ran through the removed direct TypeSafe API. Accepted counts are correct / wrong / not accepted.

| Arm | Raw correct of 256 (predicted) | Failed | p50 / p95 ms | HTTP attempts (failed) | Accepted | Precision, coverage |
|---|---|---|---|---|---|---|
| Gemma-4-31B (Io Net), chunked | 249 (256) | 0 | 582 / 1106 | 267 (11) | 241 / 7 / 8 | 97.2%, 96.9% |
| Jev 1.13 (direct API) | 243 (256) | 0 | 262 / 331 | 512 (0) | 192 / 1 / 63 | 99.5%, 75.4% |
| Qwen3.5-397B-A17B (DigitalOcean), chunked, 4 passes | 242 (256) | 0 | 927 / 13050 | 1300 (1044) | 221 / 8 / 27 | 96.5%, 89.5% |
| Jev 1.13 (OpenRouter structured), chunked | 241 (255) | 1 | 430 / 526 | 256 (0) | 193 / 1 / 62 | 99.5%, 75.8% |
| Gemma-4-26B-A4B | 239 (256) | 0 | 1353 / 2396 | 512 (0) | 219 / 7 / 30 | 96.9%, 88.3% |
| Qwen3.6-35B-A3B | 227 (256) | 0 | 1214 / 2201 | 512 (0) | 206 / 20 / 30 | 91.2%, 88.3% |
| E4B QAT Q4_0 (local) | 220 (256) | 0 | 292 / 371 | n/a | 189 / 20 / 47 | 90.4%, 81.6% |
| Qwen3-14B | 214 (256) | 0 | 326 / 437 | 512 (0) | 198 / 32 / 26 | 86.1%, 89.8% |
| Qwen3-32B | 214 (256) | 0 | 568 / 684 | 512 (0) | 196 / 35 / 25 | 84.8%, 90.2% |
| Qwen3.5-397B-A17B (DigitalOcean), **incomplete, superseded** | 211 (223) | 33 | 824 / 13147 | 512 (72) | 192 / 7 / 57 | 96.5%, 77.7% |
| Qwen3.5-9B | 207 (256) | 0 | 728 / 1205 | 512 (0) | 160 / 20 / 76 | 88.9%, 70.3% |
| E2B Q4_0 (local) | 181 (256) | 0 | 208 / 254 | n/a | 133 / 32 / 91 | 80.6%, 64.5% |
| Gemma-4-31B (Parasail), **incomplete, superseded** | 150 (154) | 102 | 887 / 6821 | 512 (114) | 146 / 4 / 106 | 97.3%, 58.6% |
| Qwen3.6-27B, **no scored rows** | 0 (0) | 256 | 205 / 467 | 512 (512) | 0 / 0 / 256 | unavailable, 0% |

Every failure in the sequential files is HTTP 429 from the routed provider. The two completed reruns agree with every raw answer the superseded incomplete files scored (Gemma-4-31B 154 / 154, Qwen3.5-397B 223 / 223); the missing cases were the hard ones, so the incomplete precision and coverage do not carry over. The incomplete arms are not comparable on raw correctness or acceptance: their failures remove a nonrandom subset of cases, and their precision is over the cases that happened to succeed. Local latency is a different machine boundary from hosted round trips.

**Providers (2026-10-10, one key).** A tiny probe per provider (one-token claim, `logprobs`, `top_logprobs` 20, reasoning disabled, `require_parameters`, no fallbacks) before each run:

| Model | Answers with 20 log probabilities | Fails |
|---|---|---|
| Gemma-4-31B | Io Net (used; 11 of 267 attempts 429, all recovered by retry), CoreWeave (fp4), Parasail (fp8; throttled the earlier run) | Novita and Venice return no log probabilities (HTTP 200, empty `logprobs`); DeepInfra, Crusoe, Chutes, Friendli, SambaNova, ModelRun, SiliconFlow do not list `logprobs` |
| Qwen3.5-397B-A17B | DigitalOcean (used), StreamLake, Parasail | Alibaba rejects `top_logprobs` 20 (HTTP 400, range); Venice drops log probabilities; DeepInfra, Phala, AtlasCloud, Novita do not list `logprobs` |

Throttling on Qwen3.5-397B is not a provider property: every pinned provider (DigitalOcean, StreamLake, Parasail, GMICloud) returned the same OpenRouter message, `Rate limit exceeded: new-account-rpm/qwen/qwen3.5-397b-a17b-20260216 … new accounts are limited to 20 requests` per minute, an account-level limit on that model shared by everything using the key. At 8 requests in flight, StreamLake failed 33 of its first 64 cases and was abandoned (`…@StreamLake.json.checkpoint.jsonl`, no artifact); DigitalOcean completed in 4 `--retry-failed` passes 60 s apart (116 → 48 → 9 → 0 failures; 1044 of 1300 HTTP attempts were 429). Its successful attempts took 853 / 1253 ms (p50 / p95); the table's 927 / 13050 ms includes the backoff of the 61 requests that needed more than one attempt. The checkpoint's code fingerprint hashes the whole working tree, so a CLI `--retry-failed` rerun was refused whenever another edit landed between passes; the passes ran in one process (`run_cohort` called repeatedly) with the fingerprint recorded once at its start.

### Jev through the OpenRouter structured binding (`results/procedure-cohort-jev-openrouter.json`, 2026-10-10)

`openrouter:typesafe/jev-1.13` through the chunked runner (System One, one joint request per case, 8 in flight; returned `typesafe/jev-1.13-20260917@TypeSafe`) against the saved direct-API run on the same 256 case IDs, cohort source and assignment digests:

| | Direct API (`procedure-cohort-jev.json`) | OpenRouter binding |
|---|---|---|
| Raw correct (predicted) | 243 (256) | 241 (255) |
| Failed | 0 | 1 |
| Accepted correct / wrong / not accepted | 192 / 1 / 63 | 193 / 1 / 62 |
| Precision, coverage | 99.5%, 75.4% | 99.5%, 75.8% |
| Request p50 / p95 ms | 262 / 331 (sequential) | 430 / 526 (8 in flight) |

- **Raw answers:** 255 IDs scored by both; 254 agree. The one flip, `routing-05:actor:en`, is a near tie in both runs (billing .43 / security .38 direct, .40 / .41 OpenRouter); fixes 0, regressions 1. The single accepted error (`security-03:quoted:en`) is the same case in both.
- **Distributions:** |Δ| of the direct run's top-answer probability: median 0.00, mean 0.011, p95 0.05, max 0.11; 152 / 255 distributions are identical. Jev returns two-decimal probabilities and is not deterministic: three identical OpenRouter requests for `security-10:base:pt` returned monitor .93, .94 and .91, so these differences are within its own repeat variation. Seven statuses differ, all threshold crossings near 0.9 (four become accepted, two become uncertain) plus the failure.
- **Failure:** `security-10:base:pt` returned HTTP 200 with a choice distribution the binding rejected as `probabilities must sum to one` (tolerance 1e-3). Receipts keep no response bodies; two-decimal rounding of five options can sum to 0.99 or 1.01, which is the probable cause. The direct run had none in 256.
- **Latency:** about +170 ms per request through OpenRouter, measured under different concurrency (sequential direct versus 8 in flight), so the hop and contention are not separated. Wall clock 19.7 s for 256 cases (13 cases/s).

**GPT-6 Luna Decisions** (`results/luna-smoke.json`): `openrouter:openai/gpt-6-luna-decisions` through the same System One payload builder, 10 public calls (two-way Check, unstated Check, Pick, Rate): every exchange HTTP 200 and parsed, returned `openai/gpt-6-luna-decisions-20261006@OpenAI`, median 418 ms. Shapes: `noul` as 0 / 1, choice and score probabilities (score also carries `score` and `legend`, which the binding ignores). It is now in the verified structured table; this is a protocol smoke, not a quality measurement.

## Small-context cohort base (`results/small-context-cohort-base.json`)

This is the authored development cohort: 256 cases in 32 families, covering 3 domains (customer routing, invoice handling, security response) in English and Portuguese. It runs through public Check/Pick/Rate on rendered named evidence, with no eligibility callbacks, shortlist or exact rules, and the default .9 acceptance. Before any model result, the cohort declared a target of precision .95 at coverage ≥ .5. The fixed family assignment puts 8 families in fit, 4 in model select, 3 in acceptance fit and 17 in check. This is development evidence, not independent domain quality. The figures below are read from the saved artifact; no model was rerun.

| | E2B (Gemma 4 E2B Q4_0) | E4B (Gemma 4 E4B QAT Q4_0) |
|---|---|---|
| Raw top answer correct | 181 / 256 (70.7%) | 220 / 256 (85.9%) |
| Accepted: correct / wrong / not accepted | 165: 133 / 32 / 91 | 209: 189 / 20 / 47 |
| Accepted precision, coverage | 80.6%, 64.5% | 90.4%, 81.6% |
| Statuses ok / uncertain / needs_input / failed | 165 / 84 / 7 / 0 | 209 / 35 / 12 / 0 |
| Status agreement, operational agreement | 169, 137 | 217, 197 |
| Check role (136): raw; accepted correct / wrong / not accepted | 97; 73 / 14 / 49 | 116; 99 / 15 / 22 |
| Contrast pairs correct (192): change 138 / preserve 54 | 104: 70 / 34 | 152: 102 / 50 |
| Families with every case raw-correct (of 32) | 6 | 17 |
| Families with ≥ 1 accepted error | 14 | 8 |
| Families at precision ≥ .95 with coverage ≥ .5 | 12 | 21 |

Not accepted means uncertain plus needs_input. Some needs_input results are the expected outcome for unstated evidence, and operational agreement counts those as correct. Neither arm meets the declared target pooled. On the check role, E4B reaches 86.8% precision at 83.8% coverage.

Comparing E4B against E2B case by case: 55 fixes, 16 regressions and 20 unchanged mistakes. By family, 11 families become fully correct and none regress.

The procedure-omission control asks the same primitives over the same evidence without the procedure. It is not a deployable arm. Without the procedure, E2B gets 149 raw correct and 134 accepted (93 correct / 41 wrong), with 75 of 192 contrast pairs correct. E4B gets 176 raw correct and 140 accepted (108 / 32), with 101 contrast pairs correct.

By phenomenon: cases; raw correct for each arm; then accepted correct / wrong / not accepted for each arm:

| Phenomenon | Cases | E2B raw | E4B raw | E2B accepted | E4B accepted |
|---|---|---|---|---|---|
| base | 64 | 51 | 62 | 46 / 8 / 10 | 61 / 2 / 1 |
| quoted | 38 | 19 | 28 | 13 / 10 / 15 | 23 / 6 / 9 |
| actor | 32 | 23 | 22 | 12 / 4 / 16 | 16 / 4 / 12 |
| reference | 28 | 17 | 24 | 12 / 4 / 12 | 20 / 0 / 8 |
| exception | 24 | 18 | 19 | 12 / 3 / 9 | 18 / 4 / 2 |
| time | 24 | 18 | 23 | 15 / 4 / 5 | 22 / 0 / 2 |
| unstated | 18 | 14 | 16 | 4 / 0 / 14 | 8 / 0 / 10 |
| procedure | 16 | 11 | 10 | 7 / 2 / 7 | 7 / 6 / 3 |
| resolution | 16 | 12 | 16 | 11 / 0 / 5 | 15 / 0 / 1 |
| contradictory | 12 | 8 | 10 | 5 / 1 / 6 | 5 / 1 / 6 |
| indirect | 12 | 8 | 12 | 7 / 0 / 5 | 12 / 0 / 0 |
| grant | 6 | 6 | 6 | 6 / 0 / 0 | 6 / 0 / 0 |
| paraphrase | 6 | 4 | 6 | 4 / 2 / 0 | 6 / 0 / 0 |
| negation | 2 | 2 | 2 | 2 / 0 / 0 | 2 / 0 / 0 |

By domain (raw; accepted correct / wrong / not accepted), E2B → E4B:

- Routing (88 cases): 64; 56 / 10 / 22 → 75; 69 / 9 / 10.
- Invoice (80): 55; 34 / 13 / 33 → 68; 55 / 6 / 19.
- Security (88): 62; 43 / 9 / 36 → 77; 65 / 5 / 18.

By language, English and Portuguese are within 4 raw cases of each other in both arms.

E4B's errors concentrate in quoted, actor, exception and procedure contrasts. Its 6 accepted procedure errors are more than E2B's 2.

## Cohort adjudication (v2, `tests/fixtures/small-context-cohort-v2.json`; superseded by v3 below)

Audited against the authored procedure text; the decisions and rationales are in the fixture's `adjudication` key, and Portuguese variants inherit each decision. v2 has a new source identity and is materialized in `results/small-context-cohort-v2/` (same family assignment, `assignment_sha256` unchanged; new `dataset_id`). Every table above was run on v1 (`results/small-context-cohort/`, kept) and was not rescored; the v2 reruns are below. A materialized directory keeps resolving to the source frozen in its `assignment.json`; new directories use v3 (current, below).

| Item | Gold | Decision | Rationale |
|---|---|---|---|
| `routing-10:unstated` ("Renewal.") | clarify | upheld, unchanged | A topic, not a need: routing_v1 sends renewal dates to billing and stopping renewal to retention, so either queue requires inventing the request; "If the message does not state what the customer needs, ask for details." |
| `security-09:contradictory` | escalate | ambiguous; message and security_v1 rewritten | The owner's denial read as use by someone else (reset), suspected exposure (monitor) had no precedence against conflicting sources (escalate), and the message did not say the calls were ongoing (clarify). The calls are now "right now", and security_v1 says conflicting sources escalate "even when one source alone would call for a credential reset or monitoring". |
| `invoice-01:boundary` (3% over, v1) | approve | ambiguous; invoice_v1 wording | Not held by the >5% rule, but approval required an invoice that "matches its purchase order". invoice_v1 now says an amount up to 5% above the purchase order still matches it. |
| `invoice-01:procedure` (7%, v2), `invoice-09:procedure` (9%, v2) | approve | ambiguous; invoice_v2 wording | Within the 10% hold limit but not a literal match; invoice_v2 now says up to 10% above still matches. The other tolerance items (7% and 9% under v1, 13% under v2: hold) are unchanged. |

### Procedure arms on v2 (`results/procedure-cohort-v2-*.json`, 2026-10-10)

`bench.policy_workflow --cohort results/small-context-cohort-v2`, guided only, chunk 16, one invocation per hosted arm with its provider pinned (8 in flight), then E2B and E4B in one local invocation (`procedure-cohort-v2-local.json`, GPU, under the model-job lock, `MemoryMax=10G`). Providers were probed first: Qwen3.6-35B-A3B answered with log probabilities on Darkbloom, AkashML, DekaLLM, Parasail, Phala and CoreWeave (DeepInfra, Venice, AtlasCloud and SiliconFlow do not list `logprobs`); Darkbloom was pinned because the unpinned v1 run was routed there. Accepted counts are correct / wrong / not accepted. Hosted p50 / p95 is per final request including retries; local chunked runs expose only chunk wall clock, so local latency is amortized ms per case. The v1 column is the arm's row in the v1 table above; fixes / regressions are over the case IDs scored in both.

| Arm | Raw correct of 256 (predicted) | Failed | p50 / p95 ms | HTTP attempts (failed) | Accepted | Precision, coverage | v1 raw correct; fixes / regressions |
|---|---|---|---|---|---|---|---|
| Gemma-4-31B (Io Net) | 249 (256) | 0 | 600 / 1265 | 260 (4) | 241 / 7 / 8 | 97.2%, 96.9% | 249; 0 / 0 |
| Gemma-4-26B-A4B (Darkbloom) | 239 (256) | 0 | 995 / 1585 | 256 (0) | 222 / 5 / 29 | 97.8%, 88.7% | 239 (sequential, unpinned, routed to Darkbloom); 4 / 4 |
| Jev 1.13 (OpenRouter structured) | 235 (256) | 0 | 427 / 640 | 256 (0) | 191 / 2 / 63 | 99.0%, 75.4% | 241 of 255 scored; 1 / 8 |
| Qwen3.6-35B-A3B (Darkbloom) | 231 (256) | 0 | 1117 / 1823 | 256 (0) | 203 / 14 / 39 | 93.5%, 84.8% | 227; 7 / 3 |
| E4B QAT Q4_0 (local) | 217 (256) | 0 | 321 amortized | n/a | 185 / 19 / 52 | 90.7%, 79.7% | 220 (sequential); 5 / 8 |
| E2B Q4_0 (local) | 182 (256) | 0 | 192 amortized | n/a | 131 / 31 / 94 | 80.9%, 63.3% | 181 (sequential); 2 / 1 |

Pooled over all 256, Gemma-4-31B, Gemma-4-26B and Jev meet the declared target (precision ≥ .95 at coverage ≥ .5); Qwen3.6-35B, E4B and E2B do not. Every returned model is the pinned one (Jev `typesafe/jev-1.13-20260917@TypeSafe`).

**Adjudicated items, v1 → v2** (10 case IDs: the five items × EN/PT; gold is unchanged in every one, only the message or procedure wording changed). Raw-correct count v1 → v2, then what moved:

| Arm | Correct | `routing-10:unstated` (clarify, upheld) | `security-09:contradictory` (escalate) | `invoice-01:boundary`, `invoice-01:procedure` (approve) | `invoice-09:procedure` (approve) |
|---|---|---|---|---|---|
| Gemma-4-31B | 6 → 6 | clarify both | reset (1.00) both, unchanged | approve both, unchanged | hold (1.00) both, unchanged |
| Gemma-4-26B | 1 → 4 | retention (en) / billing (pt), unchanged | reset; now accepted (.99) | hold → approve on 3 of 4 (pt boundary already approve); all 4 accepted | hold (.99–1.00) both, unchanged |
| Jev 1.13 | 5 → 6 | billing both, unchanged | clarify → reset (.66–.74), still wrong | approve all 4; three uncertain → accepted (.93–.94) | en hold → approve; both approve, uncertain (.83/.89) |
| Qwen3.6-35B | 2 → 8 | clarify both, unchanged | reset (1.00) both, unchanged | hold → approve on all 4 (three accepted) | hold → approve both, uncertain (.56/.82) |
| E4B | 1 → 4 | retention both, unchanged | reset; now accepted (.98–.99) | hold → approve on 3 (en boundary already approve) | hold, now uncertain (.74/.80) |
| E2B | 7 → 7 | en retention → billing (wrong both), pt clarify | monitor both, unchanged | approve all 4, unchanged | approve both, unchanged |

- **The invoice tolerance wording works on its items** for four of six arms: every arm except Gemma-4-31B (hold on 9%) and E2B (already right) now approves `invoice-01:boundary`/`procedure`, and Qwen and Jev also approve `invoice-09:procedure`. The 31B and 26B Gemmas still hold the 9% `invoice_v2` case at ≥ .99.
- **`security-09:contradictory` is not repaired:** no arm answers escalate on v2. Five arms answer reset (Gemma-4-31B, Gemma-4-26B, Qwen and E4B at ≥ .98, Jev uncertain) and E2B monitor; the rewrite moved Jev from clarify to reset and made the 26B and E4B resets accepted errors. The added precedence clause is not read as overriding the owner's denial; the item needs another adjudication pass before it counts as evidence for any arm.
- **Side effect on unchanged gold:** the new `invoice_v1` sentence ("up to 5% above still matches") pulled neighbouring hold cases (7% and 9% over under `invoice_v1`) to approve: Jev regresses on all eight `invoice-01:base`/`paraphrase` and `invoice-09:base`/`boundary` IDs (uncertain, .51–.75), E4B on six (four accepted at .95–.98), Qwen and Gemma-4-26B on `invoice-01:paraphrase`. These are most of each arm's regressions; the remaining moves are single cases.

## Cohort adjudication (v3, `tests/fixtures/small-context-cohort-v3.json`; current)

v3 supersedes v2 (`source_sha256` c59f515c…; its decisions are kept in v2's own `adjudication` key and still hold unless revised here). The decisions, rationales and the v2 digest are in the fixture's `adjudication` key. Materialized in `results/small-context-cohort-v3/` (`source_sha256` 51e37205…, same `assignment_sha256` 4f996dcf…, new `dataset_id`); `bench.policy_adaptation` now defaults new directories to v3, and the v1/v2 directories keep their frozen sources. Gold is unchanged everywhere; only wording changed.

| Item | Gold | Decision | Change |
|---|---|---|---|
| invoice_v1 tolerance: `invoice-01:boundary` (3%) approve; `invoice-01:base`/`paraphrase` (7%), `invoice-09:base`/`quoted` (9%) hold | unchanged | upheld; wording revised | v2's "an invoice amount up to 5% above its purchase order still matches it" pulled the 7%/9% hold items to approve. Approval no longer says "matches" (purchase order and posted goods receipt, unless another rule applies) and the tolerance lives only in the hold rule: "The amount tolerance ends at 5%: an amount up to 5% above its purchase order is not held for its amount, and any larger overage is held." |
| invoice_v2 tolerance: `invoice-01:procedure` (7%), `invoice-09:procedure` (9%) approve; `invoice-09:boundary` (13%) hold | unchanged | upheld; wording revised | The saved v2 rows show the same pull: Jev answered approve on `invoice-09:boundary` (.70 / .75), which it held on v1. Rewritten the same way at 10%. The same sentence also governs `invoice-06:procedure`/`boundary` (no purchase order), whose evidence is unchanged. |
| `security-09:contradictory` | escalate | ambiguous; message rewritten (en, pt) | v2's sources disagreed about who used the token, and every arm read the owner's denial as use by someone else (reset). Now: "Calls with dev.lee's API token are coming right now from a foreign IP address. The API gateway reports that every call was rejected, but the database audit log shows the same calls reading customer records." The sources conflict about whether the activity succeeded (security_v1's escalate trigger, as in `security-04:contradictory`); neither says who is calling (no reset on its own), the asset and ongoing state are stated (not clarify), no host is named (not isolate). security_v1 and the other security-09 variants are unchanged. |

**Adjudicated items first** (`results/cohort-v3-adjudicated-items.json`: the 24 IDs of `invoice-01`, `invoice-09` and `security-09`, guided, before the full arms). Raw correct v2 → v3 by group:

| Arm | All 24 | invoice_v1 hold 7%/9% (8) | invoice_v1 approve 3% (2) | invoice_v2 approve 7%/9% (4) | invoice_v2 hold 13% (2) | `security-09:contradictory` (2) | other security-09 (6) |
|---|---|---|---|---|---|---|---|
| Gemma-4-31B (Io Net) | 20 → 22 | 8 → 8 | 2 → 2 | 2 → 2 | 2 → 2 | 0 → 2 | 6 → 6 |
| Jev 1.13 (OpenRouter) | 14 → 24 | 2 → 8 | 2 → 2 | 4 → 4 | 0 → 2 | 0 → 2 | 6 → 6 |
| E4B (local) | 13 → 17 | 1 → 7 | 2 → 2 | 2 → 0 | 2 → 2 | 0 → 0 | 6 → 6 |
| E2B (local) | 10 → 11 | 0 → 0 | 2 → 2 | 4 → 4 | 0 → 0 | 0 → 1 | 4 → 4 |

Gemma-4-31B's only disagreement is `invoice-09:procedure` (9% under invoice_v2: hold at 1.00, as on v1 and v2). The wording was re-examined once with direct probes on that item: it holds at 8% and 9% (also with the amounts scaled ×10) and under v2's wording, but approves at 7% and approves the same 9% invoice when the message states "9% above the order". The miss is the unstated percentage computation, not the procedure text, so neither wording nor gold changed further.

### Procedure arms on v3 (`results/procedure-cohort-v3-*.json`, 2026-10-11)

Same protocol as v2: `bench.policy_workflow --cohort results/small-context-cohort-v3`, guided only, chunk 16, one invocation per hosted arm with its provider pinned (8 in flight), then E2B and E4B in one local invocation (`procedure-cohort-v3-local.json`, under the model-job lease, `MemoryMax=10G`, `MemorySwapMax=0`, with a MemAvailable ≥ 2 GiB guard; minimum seen 7.9 GiB). Hosted p50 / p95 is per final request including retries; local latency is amortized ms per case. Accepted counts are correct / wrong / not accepted; fixes / regressions are against the arm's v2 row over all 256 case IDs (all scored in both).

| Arm | Raw correct of 256 (predicted) | Failed | p50 / p95 ms | HTTP attempts (failed) | Accepted | Precision, coverage | v2 raw correct; fixes / regressions |
|---|---|---|---|---|---|---|---|
| Gemma-4-31B (Io Net) | 251 (256) | 0 | 673 / 1538 | 274 (18) | 243 / 5 / 8 | 98.0%, 96.9% | 249; 2 / 0 |
| Jev 1.13 (OpenRouter structured) | 246 (256) | 0 | 434 / 572 | 256 (0) | 190 / 1 / 65 | 99.5%, 74.6% | 235; 11 / 0 |
| Gemma-4-26B-A4B (Darkbloom) | 240 (256) | 0 | 832 / 1483 | 256 (0) | 224 / 4 / 28 | 98.2%, 89.1% | 239; 4 / 3 |
| Qwen3.6-35B-A3B (Darkbloom) | 230 (256) | 0 | 879 / 1562 | 256 (0) | 207 / 14 / 35 | 93.7%, 86.3% | 231; 2 / 3 |
| E4B QAT Q4_0 (local) | 222 (256) | 0 | 314 amortized | n/a | 187 / 18 / 51 | 91.2%, 80.1% | 217; 7 / 2 |
| E2B Q4_0 (local) | 184 (256) | 0 | 206 amortized | n/a | 133 / 28 / 95 | 82.6%, 62.9% | 182; 4 / 2 |

Pooled, Gemma-4-31B, Jev and Gemma-4-26B meet the declared target (precision ≥ .95 at coverage ≥ .5); Qwen3.6-35B, E4B and E2B do not, as on v2. Every returned model is the pinned one. Gemma-4-31B's 18 failed HTTP attempts were all recovered by retry.

**Adjudicated items, v2 → v3** (paired, full runs):

- **`security-09:contradictory` is repaired for the strong arms:** Gemma-4-31B (1.00), Gemma-4-26B (.97, accepted) and Jev (.88 / .75, uncertain) now answer escalate in both languages; Qwen in English (.94; Portuguese still reset .98, accepted). E4B still answers reset (now uncertain, .86 / .81) and E2B escalate in English only (.59). Every arm was wrong on it in v2.
- **The tolerance pull on hold items is removed:** Jev recovers all eight 7%/9%/13% hold IDs it lost on v2 (all uncertain, .60–.80), E4B six (`invoice-01:base:en` still approve, uncertain), Gemma-4-26B and Qwen `invoice-01:paraphrase` (one language each; Qwen's English paraphrase is still approve, now uncertain). E2B approves every invoice-01/09 item in v1, v2 and v3.
- **Side effect on invoice_v2 approvals:** the stricter wording pulls some 7%/9% invoice_v2 approvals to hold: Gemma-4-26B `invoice-01:procedure` (both, .98 accepted / .90 uncertain), E4B `invoice-01:procedure` (both, accepted .99), Qwen `invoice-01:procedure:pt`, `invoice-01:boundary:pt` (3% under invoice_v1, uncertain) and `invoice-09:procedure:en` (accepted .94). `invoice-09:procedure` (9%) is held at ≥ .99 by both Gemmas and E4B; Jev and E2B approve all four invoice_v2 approve IDs.
- **Other moves:** E2B's `invoice-06` (no purchase order) moved toward hold under both rewritten invoice texts: `base:en`, `boundary:en`, `quoted:pt` fixed, `procedure` (both) regressed, all uncertain. The rest are single cases on unchanged text: Jev `routing-05:actor:en` (the near tie noted above), Gemma-4-26B `invoice-02:actor:pt` fixed and `routing-09:quoted:en` regressed, E4B `invoice-10:reference_instruction:pt` fixed.

## Small-context LoRA pilot (`results/cohort-adapter-selection.json`, `results/cohort-pilot-check.json`)

The existing cohort assignment (8 fit / 4 select / 3 accept / 17 check families) and target (precision .95, coverage ≥ .5) were used unchanged. The base is Gemma 4 E2B Q8_0 converted from the pinned HF revision `3e22461`; both candidates fit the existing hard-label slot loss on the 64 fit observations (128 presentations, 104,264 tokens), rank 8, alpha 16, lr 1e-4, 2 epochs, bf16 on CPU, converted with llama.cpp `a4cb4c6` `convert_lora_to_gguf.py`.

| Candidate | Projections (language model) | Trainable | Fit job | Peak RSS | Fit presentations correct, native (base → adapter) |
|---|---|---|---|---|---|
| all-linear | q/k/v/o, gate/up/down: 205 modules | 12.08 M | 5,087 s | 11.8 GB | 89 → 122 / 128 |
| query-value | q/v: 50 modules | 1.34 M | 4,147 s | 10.1 GB | 89 → 119 / 128 |

Conversion fidelity is not task quality: the zero adapter reproduces native base scores exactly (max logp difference 0); bf16 HF and Q8_0 native disagree by up to 7.3 (all-linear job) / 6.5 (query-value) in answer log-probability while agreeing on 126–128 of 128 top answers.

**Selection on select (32 observations, 4 families).** Raw errors / failed pairs / low-mass / regressions: base 9 / 6 / 0 / 0, all-linear 5 / 4 / 0 / 0, query-value 6 / 4 / 2 / 0. All-linear wins the lexicographic criterion with 4 fixes and 0 regressions, but an exact one-sided sign test gives p = .0625 (query-value: 3 / 0, p = .125), so the base was retained as inconclusive under the bench gate of that time. Selection now runs in `semantic-trainer select` on the deployed merged, quantized GGUFs: most raw-correct, then fewer answer classes regressing against base, then earlier epoch, then base; no significance gate.

**Check (136 observations, 17 families).** Every arm fits acceptance on the 3 accept families with `ss lock`, then is frozen. Read from the saved rows of `results/cohort-pilot-check.json` (recomputed with the shared correctness reduction); raw correctness and failures first, latency next, acceptance last. Every arm returned a prediction for all 136 observations and none failed. Accepted counts are correct / wrong / not accepted:

| Arm | Raw correct (of 136) | Failed | Raw fixes / regressions vs base | Families fixed / regressed (of 17) | p50 / p95 ms per cohort item | Accepted | Precision, coverage | Correct / wrong predictions not accepted |
|---|---|---|---|---|---|---|---|---|
| E2B Q8_0 base | 101 | 0 | — | — | 189.5 / 209.2 | 96 / 31 / 9 | 75.6%, 93.4% | 5 / 4 |
| all-linear | 98 | 0 | 13 / 16 | 5 / 2 | 259.7 / 312.3 | 94 / 28 / 14 | 77.0%, 89.7% | 4 / 10 |
| query-value | 101 | 0 | 14 / 14 | 5 / 1 | 201.8 / 221.6 | 95 / 27 / 14 | 77.9%, 89.7% | 6 / 8 |
| E4B QAT Q4_0 | 116 | 0 | 26 / 11 | 7 / 0 | not measured | 99 / 14 / 23 | 87.6%, 83.1% | 17 / 6 |

Accepted transitions from base (correct / wrong / abstained): all-linear wrong→correct 9, correct→wrong 9, wrong→abstained 5, correct→abstained 6, abstained→correct 4, abstained→wrong 2; query-value 9, 10, 7, 4, 4, 2; E4B 16, 9, 10, 8, 4, 0. Accepted errors stay concentrated in the same families: invoice-01 (base 4, all-linear 5, query-value 6, E4B 3), security-03 (1, 4, 4, 3), security-04 (4, 2, 2, 0), routing-04 (3, 2, 2, 2); both adapters add errors in invoice-08 (0 → 3) and security-03 while removing them in routing-05/08/09 and security-04. No arm meets the declared target; replays of each frozen window agree on all 136 accepted outcomes.

**Deployed GGUF latency** (`bench/latency.py --cohort`, GPU, 256 cohort items; `results/latency-cohort-{e2b-q8,all-linear,query-value}.json`): p50 / p95 per item 189.5 / 209.2 ms base, 259.7 / 312.3 ms all-linear (+37%), 201.8 / 221.6 ms query-value (+6%); one question on a new input 123.9 / 157.0 / 134.5 ms.

Outcome: no held-out improvement under this dataset, split and recipe (64 fit observations in 8 families, rank 8, 2 epochs); the base is retained and E4B remains the better predictor. The fit set was skewed against the check set: it held no `clarify` and no contradictory cases, and `isolate` was 31% of fit labels against 4% in check. Higher fit-presentation correctness (+30–33) is recorded, not interpreted as a cause. Model jobs: selection 270 s, check 487 s, latency 71–100 s per arm.

## Trainer validation on XPU (`results/trainer-xpu-validation.json`)

One end-to-end run of the public trainer path on the Intel Arc B390 (integrated; device memory is host memory): 40 agent-authored synthetic families × {escalate, continue, clarify} of the overheating Pick (`results/trainer-xpu-validation/`), assets manifest `1091cf49…` Q4_0, `ss export --native --labels` (39 tune rows) and `--split check` (81 rows), `prepare` (fit 9 / model_select 3 / acceptance_fit 1 / check 27 families), `lora` on `xpu:0` with the default recipe and 2 epochs, `select --check`, `Model.from_specialization` and one public Pick. Every model step ran under the model-job lock and a systemd cap.

| Step | Wall |
|---|---|
| native export / prepare / dry run | 3.2 / 1.1 / 5.0 s |
| fit (load, baseline, 2 epochs = 8 steps of 8 observations, native base scoring) | 144 s; 6.5 / 5.4 s per step; peak device 7.29 GB, process RSS 9.5 GB |
| select (zero + 2 epochs merged into F16 and quantized, 9 select + 3 accept + 81 check rows) | 215 s |
| `Model.from_specialization` + one call | 8.7 s |

The zero adapter merged into the F16 base and quantized to Q4_0 reproduced the base byte for byte (`4d0933b6…`). Fit observations all-presentations-correct (HF) went 17 → 27 of 27. Selection: base 6/9 raw correct (never predicted escalate), epoch-1 7/9 (fixes kiln, robot-arm; clarify regressed one item), epoch-2 4/9 (predicted clarify 8 of 9); epoch-1 was selected. Acceptance fitted on one family (threshold .9); on the 81 check rows the specialist was 77 raw correct, 68 accepted (67 correct). Twenty-seven fit observations, three select families and one acceptance family make none of this a quality claim. The public call returned `continue` (.993) from the specialist and `continue` (.999) from the base.

Two findings. With the whole model on the device, loading copied the 4.4 GiB per-layer embedding table to the XPU: one attempt ended in `UR_RESULT_ERROR_DEVICE_LOST`, another consumed more than 12.4 GiB of host memory and the kernel OOM killer took a browser outside the job, because integrated-GPU allocations are not charged to the systemd scope. The trainer now keeps that table (a lookup) on the host for accelerator fits and passes its rows as `per_layer_inputs` (identical logits on CPU); a guarded load then consumed 5.7 GiB with 5.15 GiB on the device. Second, the pipeline-built Q4_0 base agrees with the bf16 HF model on only 35 of 54 fit presentations' top answers (max log-probability difference 10.4) where the ggml-org Q4_0 default judge agrees on 47; Q8_0 agreed on 126–128 of 128 in the pilot above. The specialist base quant should be chosen with that measured before the single-task run.

## Select quality (`bench/select_quality.py`)

```
python -m bench.select_quality [--judge SPEC] [--provider NAME] [--key-file PATH] --output result.json
```

The S1MB `english-v1` benchmarks whose questions are independent yes/no criteria over one input: go-emotions (28 emotions) and civil-comments (7 toxicity attributes), the first 100 cases of each (the pinned data of `bench/external_primitives.py`). aegis2 (prompt/response safety of different fields, one question in some cases) and followir (alternative relevance criteria) are not memberships of one set and are excluded. Each case is asked as public `Select(question, {id: Label(true-option description)})` and then as named `Check(description)` per label (select first, fixed order; a local judge reuses the cached evidence in the second call). Gold is a membership only for a hard target (0 or 1); soft targets are unlabelled and count only toward parity, statuses and unresolved. Every case is one journal record (resumable, refused on a changed configuration until `--fresh`). `tests/test_select_quality.py` passes the fake end-to-end run (resume, failures, refusal) and the presentation parity: each Select member and its Check send the same rendered evidence, one request each, and identical scores give identical member results. The wording differs (`Question … Candidate: id — description … Does this candidate apply?`, yes/no, against the claim, true/false), so real-run parity measures that difference.

The slice is every case of the pinned data: positions 1–100 of each benchmark (`LIMIT = 100`; the pinned dataset revision holds exactly 100 cases per benchmark, so there is no further go-emotions data to scan). That is 3,500 memberships (go-emotions 2,800, civil-comments 700), 2,700 hard-labelled (2,490 / 210) and 29 unanimous positives: 27 in go-emotions (4 in cases 1–25, 8 by 50, 17 by 75) and 2 in civil-comments (both in cases 26–50; it is kept because it has positives). The ≥ 40 go-emotions positives a recall estimate was meant to rest on are not reachable from the pinned data. One case is fully hard-labelled (go-emotions `test:3766`, gold disgust + embarrassment). Exact-set agreement otherwise covers each case's hard-labelled memberships, counted only where every one of them was examined and predicted (raw) or accepted; 183 cases have hard memberships (100 go-emotions, 83 civil-comments).

Results (2026-10-10): `select-quality-100-{e2b,jev,gemma26b}.json`; E2B Q4_0 local GPU under the model-job lock (`MemoryMax=10G`); Jev 1.13 through OpenRouter (one request per arm per case, returned `typesafe/jev-1.13-20260917`); Gemma 4 26B-A4B through OpenRouter pinned to Darkbloom (one request per membership, 3,500 per arm). **No arm had a failed membership or a failed HTTP attempt; every Select was complete.** The earlier first-25 runs stay in `select-quality-{e2b,jev,gemma26b}.json` (4 positives) and are superseded by these.

**Raw per-member correctness** (hard-labelled memberships, all 2,700 predicted in every arm). Positive class: TP / FP / FN, precision, recall over the raw predictions:

| Judge | Arm | Raw correct | go-emotions (2,490) | civil-comments (210) | go-emotions positives (27) | civil-comments positives (2) | Exact set raw (183 eligible): go / civil |
|---|---|---|---|---|---|---|---|
| E2B | Select | 2,448 (90.7%) | 2,271 | 177 | 19 / 211 / 8: 8.3%, 70.4% | 1 / 32 / 1: 3.0%, 50% | 71: 13 / 58 |
| E2B | Checks | 2,404 (89.0%) | 2,231 | 173 | 18 / 250 / 9: 6.7%, 66.7% | 1 / 36 / 1: 2.7%, 50% | 69: 12 / 57 |
| Jev | Select | 2,397 (88.8%) | 2,204 | 193 | 23 / 282 / 4: 7.5%, 85.2% | 2 / 17 / 0: 10.5%, 100% | 72: 4 / 68 |
| Jev | Checks | 2,328 (86.2%) | 2,142 | 186 | 24 / 345 / 3: 6.5%, 88.9% | 2 / 24 / 0: 7.7%, 100% | 65: 2 / 63 |
| Gemma 26B | Select | 2,413 (89.4%) | 2,217 | 196 | 21 / 267 / 6: 7.3%, 77.8% | 2 / 14 / 0: 12.5%, 100% | 78: 8 / 70 |
| Gemma 26B | Checks | 2,274 (84.2%) | 2,103 | 171 | 23 / 383 / 4: 5.7%, 85.2% | 2 / 39 / 0: 4.9%, 100% | 58: 6 / 52 |

The fully hard-labelled case is wrong in every arm: each finds disgust and adds anger, annoyance and disapproval (Jev and Gemma also disappointment); only Jev's Checks also find embarrassment.

**Parity, Select against named Checks** (all 3,500 memberships predicted by both; hard-label transitions checks → select):

| Judge | Same raw answer | Mean / max |Δp_true| | Same status and value | Fixes / regressions |
|---|---|---|---|---|
| E2B | 3,249 (92.8%) | 0.080 / 0.922 | 2,943 | 91 / 47 |
| Jev | 3,349 (95.7%) | 0.040 / 0.680 | 3,201 | 87 / 18 |
| Gemma 26B | 3,210 (91.7%) | 0.082 / 0.992 | 2,992 | 157 / 18 |

**Acceptance** (default θ 0.9; accepted correct / wrong, precision and coverage of hard memberships; accepted positives TP / FP with the share of the 29 gold positives accepted; unresolved counts every membership not accepted, including soft ones):

| Judge | Arm | Accepted | Precision, coverage | Accepted positives (of 29) | Unresolved of 3,500 | Exact set (accepted: eligible, agree) | Case p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| E2B | Select | 2,150 / 115 | 94.9%, 83.9% | 15 / 112 (15) | 699 | 45, 43 | 1,752 / 1,873 |
| E2B | Checks | 2,212 / 196 | 91.9%, 89.2% | 19 / 190 (19) | 461 | 63, 50 | 1,235 / 1,868 |
| Jev | Select | 1,498 / 58 | 96.3%, 57.6% | 15 / 58 (15) | 1,676 | 32, 31 | 408 / 482 |
| Jev | Checks | 1,550 / 80 | 95.1%, 60.4% | 15 / 80 (15) | 1,581 | 31, 30 | 401 / 469 |
| Gemma 26B | Select | 2,303 / 192 | 92.3%, 92.4% | 21 / 189 (21) | 351 | 83, 64 | 2,073 / 4,374 |
| Gemma 26B | Checks | 2,150 / 331 | 86.7%, 91.9% | 22 / 328 (22) | 336 | 85, 47 | 2,185 / 4,486 |

**Readings:** with 29 positives the earlier pattern holds and now has a recall estimate. Select is raw-correct more often than the named claims for all three judges (+44, +69, +139 memberships), by fewer false positives on unanimous negatives (43, 70 and 141 fewer across both tasks) while true positives move by +1, −1 and −2. Raw positive precision stays 2.7–12.5% in every arm: nearly every predicted membership, and nearly every accepted error (112 of 115 for E2B Select), is a membership the raters unanimously rejected, so the precision of an accepted positive is 6–21%. Raw go-emotions recall is 67–89% (Jev highest, E2B lowest); civil-comments' two positives say nothing. Exact-set agreement on go-emotions is 2–13 of 100 cases; one extra emotion among 28 breaks a set. Recall still rests on 27 correlated go-emotions memberships in 100 cases; a ≥ 40-positive estimate needs another multi-label source or an authored set. Local latency compares a cold evidence read (Select) with a warm one (Checks).

## Deferred, with the reason

- **GLiClass multilang mini:** its card reports a multilingual zero-shot F1 of 0.54 and it needs torch.
- **FTS5:** unnecessary below about 10k candidates.
- **NER, stemmers, ontologies:** each waits for a failure it fixes.
