# Changelog

The spec (`semantic-switch.md`) states the current contract. This file records how it got there and why. The full v0.18 text is archived in `docs/semantic-switch-v0.18.md`.

## v0.23 (implemented) — measured honesty
A review reproduced against saved results and new runs:
- **The probe's +7 points was leakage.** The fresh suite's Portuguese items translate the English ones; leave-one-pair-out kept the twin in training. With families held out, the probe (0.844), Platt calibration (0.875) and a free cutpoint (0.831) all sit within noise of the judge (0.856). **`specialize: probe` and the hidden-state path are removed.** The idea moves to E9 (real traffic).
- **Lock "tuned" didn't mean "meets the target".** Tuned on one half of the fresh suite's families, θ held 83% precision on the other, 0 of 300 splits at 95%, though every lock reported "tuned". `ss lock` now **validates on the check half**: each side is met (with held-out precision and interval), not met (abstains) or untuned.
- **Lexical heads train on corrections plus tune-half labels,** tuned on grouped out-of-fold predictions; the check half validates. **`ss specialize` runs the executor** with timing separated from training, and reports gained / avoided / introduced / judge calls avoided, plus the judge's ranking AUC and a diagnosis. `ss test` reports AUC and validated side states.
- **Boundaries:**
  - NaN answer mass is rejected by the `Judge` contract;
  - export writes atomically, refuses the switch's own files, merges duplicate labels, keeps groups and rejects conflicts;
  - inline and saved focus share one escape decision under the active policy;
  - `ss bench` names its workloads and explains a switch without inputs.
- **Evidence corrected:**
  - with the system prompt cached, the GPU default answers in 104–224 ms (1–7 questions, 20–261 tokens), faster than Jev everywhere measured;
  - E2B→E4B fallback was rejected (confident errors don't escalate);
  - answer mass is a format check only;
  - the 8 equipment misses are one judgment on two texts.

## v0.22 (implemented) — one acceptance rule
- **Everything specified for v0.22 is implemented** (169 tests: 153 deterministic, 16 on real models):
  - truth-vector atoms with compile-time collapse of single-fact conditions, and one `settle`;
  - one policy per judgment, `min_score` in (0.5, 1], three-way silence needing θ;
  - caseless status from outputs, dependency-aware claim budget, bare-string saved input, full rendering identity;
  - tuned / untuned / disabled policy sides, per-fact projection membership;
  - `specialize: [lexical, probe]` with hidden states, `ss export` with runtime token ids, the probe sets in `ss doctor --quality`, `correct()` scope;
  - the `Judge` output contract, a closed-runtime error, an explicit-GPU requirement, exit code 5;
  - shortlisted picks never `none`, the catalogue ranking index at load, a per-call render cache, prompt `many()`, `--batch` removed.
- **Rate equality was rewritten, not patched:** `u == "today"`, `u >= "today" and u < "high"` and the like are now one atom over the distribution; pick facts carry an extra "no eligible option" value, so all-ineligible picks settle by rule correctly for `==` and `!=`.
- **Found while implementing:**
  - a staged-scoring stop rule that looked only at newly added facts (it added a third round);
  - tokenising after `close()` read freed memory (it segfaulted; it now raises);
  - a missing switch file raised a raw traceback (now a `ConfigError`, exit 5);
  - always-on hidden-state output cost 7–17%, so it is now per question.
- **E3 (GPU) measured:** 2–5× faster with matching quality; `device="cpu"` in a GPU build fixed (2× slower before); the Vulkan banner silenced; the default judge kept device-independent; per-backend quality noted.

## v0.22 (spec) — one acceptance rule
- **`specialize: probe` (measured):** a per-fact linear probe on the judge's hidden state at the question's end, from the same decode. With 18 labels per claim it took the fresh suite from 0.856 / 89.0% to 0.925 / 97.7% (accuracy / precision at 0.9) with no added latency. It breaks even at about 8 labels and matches the judge where the judge is already strong. A generic probe across claims did not transfer. Specialisation now has two in-library tiers, `lexical` (speed) and `probe` (accuracy), with the same binding, tuning and admission.
- **`ss export`** writes training rows with runtime token ids, the only contract with external training.
- **E8 (activated LoRA)** is specified as the next tier, used only where a probe falls short. It reuses the base input cache, unlike plain LoRA. The earlier "runtime LoRA fights the cache" objection holds only for plain LoRA. Per-fact named judges and a marketplace were rejected: a specialist belongs to a fact in a lock.
- **Measured follow-ups to an external review** (projects, issues and papers):
  - **Option position** (`bench/positions.py`): no late-slot failure at 20 options. Order bias only between two plausible options, confidently in both directions.
  - **Averaging two option orders, rejected** (`bench/permute.py`): no wrong answer removed on three suites, lower coverage, +37–66% latency.
  - **Comparative choice** (`bench/comparative.py`): the default judge anchors on the situation when the goal contradicts it (3 of 23 settled answers wrong); `--precise` 24/24. Now in §1 as a supported kind, with its basis rule and "a score is not a utility".
  - **Teaching:** a table of what each action changes (§5.5); `correct()` returns its scope.
  - **Probe:** option-position and comparative sets added to `ss doctor --quality`.
  - **New experiment E7:** the input-vs-question cost split shows a question-first cache could be about 2× faster for one question over short inputs.
  - **Rejected:** GLiClass's cache (it is our design), Jeff as a judge (own output head, English only; now the E1 reference), SetFit, RouteLLM, GEPA, kNN, CeQe, BM25F.
A review of the v0.21 code, with every claim reproduced through the real executor, found that the same judgment didn't always mean the same thing.
- **Rate equality settled below its threshold:** with 0.08 / 0.84 / 0.08, `u == "today"` fired (the two `>=` pieces were each 0.92) while `u.value` stayed None. Now every single-fact sub-expression compiles to one atom evaluated on the distribution (`pT`, `pF` from a truth vector); `atom_q`, `propagate` and the three-way FALSE special case go. One policy per judgment (θ above 0.5 makes acceptance consistent by arithmetic), and one pure `settle` used everywhere.
- **A caseless switch said `ok` with nothing settled;** its status now comes from the outputs' own values.
- **Three-way silence needed no evidence:** 0.33 / 0.33 / 0.34 gave `needs_input`. `not_stated` now needs `p(does not say) ≥ θ`.
- **`{fact: name}` claims bypassed `max_claims`** (3 questions vs 1 for the same catalogue as strings). The budget now depends on why a fact is needed, not how it's written.
- **Locks could keep untested defaults silently;** each side is now recorded and reported as tuned / untuned / disabled.
- **Saving changed the input shape** (bare string → failed), and **`meaning_rev` ignored `max_chars` / `truncate` / type.** Both fixed.
- **New findings:** a shortlisted pick could answer `none` with 11 candidates never shown; a malformed custom judge failed with a `KeyError`; relabelling moved examples between folds; scoring after `close()` would hit freed native memory; config errors shared exit 1 with FALSE.
- **Rejected review suggestions:** invalidating case labels when a referenced fact changes (case labels record intended decisions); an input-session object (exact reuse is already correctness-safe under concurrency); budgeting caseless outputs (they are requested by definition).
- **`--batch` removed** (nothing batches across inputs); `min_score` is now a majority threshold in (0.5, 1]; the Jev comparison was reworded to what was measured.

## v0.21 (implemented)
- **Everything specified in v0.17–v0.21 is implemented:** `judgment.py` (definition and pure acceptance), one acceptance policy inline and saved, callable `Check` / `Pick` / `Rate` with `.save()`, one `render`; `select` as cases over shared facts (anonymous claim facts deduplicated by `meaning_rev`, card applicability and precedence compiled to Kleene expressions, whole-card budget, `complete`, per-card states, `is empty` restored, cycles rejected); per-input rounds with the input cache; `specialize: lexical`; highest-coverage threshold scan; `tuned_keys` with taught / trained-on / unseen; projection- and group-based splits; `diff` by outcome; `label` through fact values; `r.truncated`; `correct()` feedback; immediate `--each` with `--batch`; `explain --json`; `lock --dry-run`; `specialize` on the real runtime.
- **Removed:** the embedding head, e5 and the `embed` role; prompt examples (after E5); the fast judge; `r.near`; the `strong` role and per-fact `model:`; bench cost frames; `ss tune`; the `schedule` option (after E4); `codes` and `top_k`.
- **E5 (prompt examples, default judge, fresh suite):** none 0.856 / 88.9% at 0.9; random 2 pairs 0.887 / 91.5%; retrieved 2 nearest pairs 0.869 / 89.9%; latency 334 → ~870 ms. Under the 5-point bar, so the feature is gone.
- **E4 (with the cache):** two-stage scoring is 4× faster than eager when the first case decides and within 0–8% otherwise; staged per-case rounds (one round per case) were 8–40% slower and were replaced by two stages.
- **Input cache, exact reuse only:** common-prefix trimming made scores depend on the previous input (batch-shape-dependent quantised kernels; Qwen3-0.6B probe 9/12 cached vs 10/12 fresh). Exact reuse is byte-identical to a fresh read.
- **Found while implementing:** with `exclusive`, a single applicable card was final before the other cards were examined, so a later conflict could be missed; exclusive selects now examine every card. Tuning keeps the default policy when a side has fewer than 20 accepted labels, and only an evidenced failure unsets it.
- **Default inline acceptance changed** from argmax to 0.90 / 0.10. Tests that measured prompt separation on Qwen3-0.6B now opt in with `min_score=0.5`.

## v0.21 (spec) — contracts that hold
- **One acceptance policy, inline and saved** (0.90 / 0.10 by default, or the judgment's `min_score`). Inline had defaulted to argmax, so saving could turn a 0.75 True into "uncertain", and a 0.55 came back as an answer, against principle 2. `.save()` keeps the policy, and the tie rule is defined.
- **Claim identity is the fact's `meaning_rev`,** not its text (a three-way and a plain "There is a leak." are different judgments). Cards may reference shared facts with `{fact: name}`.
- **`is empty` restored:** it needs a complete search *and* every card ineligible, rejected or superseded. The v0.20 wording let an unresolved card count as empty. `complete` now means only "examined".
- **Precedence is one order-free pass over base applicability.** An applicable card is final if no applicable card supersedes it; an unresolved or unexamined overrider makes it "may be superseded"; `exclusive` is checked on the final set.
- **`codes` dropped.** A code mentioned in text isn't authoritative ("a historical E17 on another machine"). Authoritative codes go through `requires: 'mentions(active_alarms, …)'`; text mentions are `aliases`.
- **The budget counts unsettled distinct claims after the cheap steps,** with whole-card packing.
- **Threshold scan:** the highest-coverage threshold meeting the target (precision isn't monotonic). **`ss test`** reports taught / trained-on / unseen. Splits use the read-field projection, plus an optional `group`. Head admission is stated against labels, reporting errors introduced and corrected.
- **The input cache's reach is stated honestly:** each input runs its rounds consecutively (the executor interleaved inputs, which would have evicted a single cached input); a failed trim falls back to a rebuild; answer mass costs a full-vocabulary reduction.
- **`Judgment` is a definition** (data) plus pure acceptance functions, not an executor.
- **`specialize: lexical`** replaces `tier: 1`; `r.truncated`; typed reasons; `correct()` feedback; the support and equipment examples are split.
- **"Beating Jev: decisions, not questions"**, with the E6 scoreboard: wrong actions, correct `needs_input`, decisions with no model call, and latency including them.

## v0.20 (spec) — ask about the situation
- **`select` redesigned as cases over shared facts:**
  - titles never enter prompts, and `codes` are exact requirements;
  - identical claims are judged once;
  - `max_claims` replaces `max_cards`, so cost scales with distinct claims.
  - Evidence: removing titles from the trigger questions took the default judge from 17/34 to 28/34 on the equipment set. The v0.16 title "fix" for E17/E71 had caused the "weak catalogue pipeline" result.
- **New prompt rule:** questions are about the situation only; identifiers are exact evidence.
- **The fast judge (Qwen3-0.6B) is removed:** it committed 8 wrong procedures on the equipment set. Qwen3.5-2B is noted as the calibrated alternative.
- **The prompt-example removal is now conditional on E5** (retrieved precedents on the default judge). The earlier negative result was Qwen-only.
- **E4 got its first data point** (eager equal in correctness, 15% faster), and **E6** expands the equipment set to at least 40 distinct reports, since the current one has only 10 texts.

## v0.19 (spec) — restructure and removals
- **Restructured** to nine sections, about a third of the previous length. Evidence moved to `bench/README.md`, experiments to `EXPERIMENTS.md`, history here.
- **Removed, because the evidence never supported them:**
  - the embedding classifier head, the e5 model and the `embed` role (chance on minimal pairs, 0.53–0.55 on topics);
  - prompt examples (no consistent gain at 1.7–2.5× latency), together with `count` in the `Judge` protocol;
  - the typo suggestions `r.near`;
  - the `strong` role and per-fact `model:`;
  - the cost-frame formulas;
  - the physics table (10× off for Qwen3-1.7B);
  - `ss tune`, now `ss lock --dry-run`;
  - the `schedule` option, once the input cache is re-measured.
- **The default judge no longer has to pass a 95% probe gate.** The probe chose it, so the gate measured fit, not safety. The precision promise is stated honestly (about 89% untuned on unseen domains), and the fresh suite gates changes to the default.
- **Fixed contradictions:** test counts, the `Judge` member count, adapter versions, cache vs pseudo-code, stale "still open" items, the old experiment plan, and stale reference machines.

## v0.18 (spec) — one definition, one pipeline
- **`Judgment`** becomes the single place for questions, value parsing and acceptance. These were implemented 6, 5 and 9 times, and inline vs switch rendering differed.
- **The input cache:** exact reuse, plus common-prefix reuse on attention models. Before it, a repeated round on a 261-token ticket re-read about 1.1 s of input.
- **Callable `Check` / `Pick` / `Rate`, `.save()`**, and the zero-to-hero ladder.
- **The fine-tuning plan**, now in `EXPERIMENTS.md`.
- **The Jev comparison and contract note.**

## v0.17 (spec) — select correctness and honest tools
- **Card states and `complete`:** a shortlist omission used to be treated as "doesn't apply".
- **Supersedes:** cycles and self-references rejected; precedence before `exclusive`; "may be superseded".
- **Tuning keys stored in the lock:** appending one test line used to move 26 tuning inputs into "unseen".
- **`diff` by outcome:** it compared 0 inputs after a meaning change.
- **`label` through fact values:** it couldn't record "unstated".
- **`specialize` runs the real runtime.**
- **Also:** payloads in JSON, immediate `--each`, and head-driven model roles.

## v0.16 (implemented) — procedures and silence
- **Three-way checks:** silence was a confident "false" (p_true 0.00).
- **`select` over procedure cards.** One three-way question per card scored 1/10; splitting triggers (yes/no, closed-world) from exclusions (three-way, open-world) scored 8/10.
- **"Except: X." exception phrasing.**
- **Fixes:** D6 tolerance (Q4 kernels drift at tiny batches); shared scoring decided by D1/D3 only; NumPy truths in threshold tuning.
- **Fresh held-out suite:** the default fell from 0.96 to 0.86, and from 98.2% to 88.9% untuned precision.
- **Topical learning curve:** the lexical head helps; the embedding head doesn't.

## v0.15 — exact evidence
- **`mentions()` atoms, ids and aliases, and `focus="mentions"`** with an escape option. Gemma 4 E2B kept near-identical ids apart; focus halves the cost of 20-option picks.
- **Switches without cases** as the teachable judgment.
- **The lexical head.**
- **`Judge` as a protocol.**

## v0.14 — review fixes
- **Six correctness fixes:**
  - a group atom no longer settles a pick's value;
  - `resolve: all` labels;
  - the answer-mass rule applies inline too;
  - `rate` slot limit;
  - embedding length deferral;
  - `reads` in `meaning_rev`.
- **Cheap work before model work.** Exit codes unified. Exact model sources.

## v0.13 — judge ladder
- **A 240-item suite replaced the old one,** which was 6 sentence pairs counted as 24 items.
- **Gemma 3 had been benchmarked with the Qwen chat template;** a generic chat-template adapter fixed that.
- **New default judge:** Gemma 4 E2B Q4_0, replacing Qwen3-1.7B (overconfident: 84.5% precision at default thresholds).
- **Quantisation policy per family.** Hybrid models share the prefix once D1 passes.

## v0.6–v0.12
- **The core:** three-valued switches with masking, missing-is-not-uncertain, exact atoms, corrections as exact overrides, revisions, `ss lock` / `test` / `diff`, model pinning, the llama.cpp shared-prefix backend, and user content tokenised without special tokens.
- **Prompt rules measured on Qwen:** true/false claims over yes/no, and lettered options.
