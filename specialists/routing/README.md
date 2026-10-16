# routing_v1 specialist

Data for the single-task Gemma 4 E2B specialist (TODO: "Single-task E2B specialist"). Customer message routing to billing, technical, security, retention or clarify under the `routing_v1` procedure in the prompt.

## Frozen task

`acquire.py` owns the one task definition (`definition()`): the `routing_v1` guidance and the five options copied verbatim from `tests/fixtures/small-context-cohort.json`. `acquire.py define` emits it as `routing.yaml`; every other step refuses a `routing.yaml` that differs from the generator.

A fact never reads a missing field, so the generator emits the one judgment as two concrete facts that differ only in what they read:

| Fact | Reads | Meaning revision |
| --- | --- | --- |
| `routing` | `message` | `757ab6d6` |
| `routing_reference` | `message`, `reference` (retrieved excerpt) | `b8277f2d` |

The application asks `routing_reference` when retrieval returned an excerpt and `routing` otherwise. Guidance digest: `aaac763aecd8…`. If the definition changes, these labels are void: every meaning revision changes, so the labels are refused at export.

Selection objective (frozen before acquisition):

- **Candidates.** `semantic-trainer select` picks among epoch candidates on the model_select families:
  - raw correctness first;
  - then fewer answer classes regressing against base;
  - then the earlier epoch;
  - then base on ties.
- **Acceptance.** Fitted on the acceptance_fit families. It is reported once on the human check.
- **Adoption.** The specialist is adopted only when two conditions hold:
  - its check raw correctness is at least Gemma-4-31B's on matched check IDs;
  - no answer class regresses against base E2B by more than one family.

  A flat learning curve over 100/300/1,000 ends the effort.
- **Reporting.** For base E2B, specialist, Gemma-4-31B and Jev (Jev is evaluation only), report:
  - raw correctness;
  - per-class and per-family regressions;
  - contrast-pair correctness;
  - Portuguese versus English;
  - failures;
  - latency.

## Coverage grid

`grid.yaml` lists the phenomena drawn from the guidance:

- plain request
- latest request replaces earlier
- priority versus later aside
- relayed request
- quoted text and phishing
- someone else accessed or changed the account, including the saved card
- conflicting accounts
- already resolved
- planted reference instructions
- insufficient information
- multi-topic
- negated or hypothetical intents

For each phenomenon it gives the rule, the allowed answers and the style mix. Every situation is tagged with its phenomenon and its style tags (`informal`, `short`, `long`, `conversation`, `pt`), plus `translation` for translations.

A family is one base situation plus 1–3 minimal-pair variants, where one changed detail flips the answer, plus its Portuguese translations. Families are joined by `group`, and each contrast has a `pairs` id. Translations are not counted as situations.

## Roles

Every situation and label record carries its family's planned `role`; the dataset turns it into the export's split and fold:

| Role | Assignment |
| --- | --- |
| fit | tune split, folds 0–2 |
| model_select | tune split, fold 3 |
| acceptance_fit | tune split, fold 4 |
| check | check split |

Families are named `routing-NNNN` (plan index) and `routing-check-NNNN`. The role travels from `plan.jsonl` through the collect files and teacher labels into the assembled sets, so `prepare` and `select --check` see the intended roles.

## Pipeline (`acquire.py`, explicit and checkpointed)

```
python specialists/routing/acquire.py define                                 # routing.yaml from the generator definition
python specialists/routing/acquire.py plan                                   # data/plan.jsonl: 540 families (400 fit, 80 model_select, 60 acceptance_fit)
python specialists/routing/acquire.py plan --base clarify --fit 60 --model-select 12 --acceptance-fit 8   # appended clarify-based families
python specialists/routing/acquire.py generate --provider Alibaba Parasail Novita Phala DeepInfra   # qwen/qwen3.5-397b-a17b (Apache-2.0) → data/generated.jsonl, appended per family; rerun resumes
python specialists/routing/acquire.py check                                  # check/authored.yaml → check/situations.jsonl, check/collect/<fact>.jsonl (role check)
python specialists/routing/acquire.py situations                             # near-duplicate and leakage filter → data/situations.jsonl, data/collect/<fact>.jsonl
python specialists/routing/acquire.py label --pool fit --teacher openrouter:google/gemma-4-31b-it --provider Parasail --training-use use-31b.yaml --name gemma-4-31b
python specialists/routing/acquire.py label --pool fit --teacher openrouter:google/gemma-4-26b-a4b-it --provider Parasail --training-use use-26b.yaml --name gemma-4-26b
python specialists/routing/acquire.py review --pool fit --name disagreements --size 400 --audit 100 --stratum conflict --stratum teacher_uncertain
python specialists/routing/acquire.py review --pool check --name check --teachers gemma-4-31b --size 1000 --audit 1000
python specialists/routing/acquire.py assemble [--reviewed FILE ...]         # data/sets/fit-{100,300,1000}.jsonl (+ model_select and acceptance_fit records)
python specialists/routing/acquire.py report                                 # assigned roles, families and round coverage per set → data/report.json
```

Details of the steps:

- **Collect envelopes.** `situations` and `check` write schema-3 situation envelopes (`data/collect/<fact>.jsonl`) carrying each family's group, pairs, role and tags.
- **Labelling.** `label` runs `semantic-learn collect` once per fact under the declared training use; `labels/<name>/<fact>.collection.json` names the published labels and `<fact>.journal.jsonl` resumes an interrupted run (`--retry-failed` repeats transport failures).
- **Review.** `review` is `semantic-learn review` over the named teachers' collections: a seeded random audit plus declared triage strata. Two teachers that disagree leave the target unresolved (`conflict` stratum); `claims.jsonl` lists both alternatives for an explicit resolution.
- **Filtering.** `situations` drops whole families when either holds:
  - a message's word-trigram Jaccard reaches 0.6 against an earlier family;
  - it reaches 0.5 against the check pool or the small-context cohort.
- **Nested sets.** `assemble` accumulates one teacher's collection with the reviewed files through the correction-history accumulator, excludes unresolved situations, and orders fit families once (`data/sets/order.json`, greedy cover of fact × answer × phenomenon), so the 100, 300 and 1,000 sets are nested.
- **Receipts.** `data/receipts.jsonl` records generation cost from the OpenRouter usage receipts, and the key's cumulative usage plus collection counts after each labelling run.
- **Fitting.** Each set is fitted by `semantic-learn round specialists/routing/routing.yaml --labels data/sets/fit-N.jsonl --request RECIPE.json --to WORK --cutoff ... --serving ... --min-reviewed ... --min-per-class ...`, where `RECIPE.json` is a specialization request (task family `customer:routing_v1`).

Label files collected before schema-3 envelopes (`*/labels/*.jsonl`, `*/collect/shards/`, `check/check.draft*.jsonl`) are an unsupported schema and are not read; rerun `check`, `situations` and `label` to regenerate them. `plan.jsonl`, `generated.jsonl`, `situations.jsonl` and `receipts.jsonl` remain readable.

## State (2026-10-10)

| Pool | Count |
| --- | --- |
| Families generated | 620 of 620 planned; 1 near-duplicate dropped, 0 leaked |
| Situations | 1,866, plus 355 translations |
| fit-100 / fit-300 / fit-1000 | 30 / 100 / 338 families (102 / 301 / 1,002 situations) |
| model_select | 92 families, 275 situations (15% of all 619 families) |
| acceptance_fit | 68 families, 218 situations |
| Unused | about 120 fit families beyond the 1,000 set |

**Teacher agreement.** The two teachers agree on 91.8% of the fit pool (182 disagreements or unsettled). On the check pool they agree on 94.4%.

**Prepare support.** The `prepare` support check passes for all three sizes. This was a dry check: non-native export plus `fit_support`. Per-cell counts are in `data/report.json`.

## Gate: human review (pending)

Nothing here is verified yet. Do not fit or select until the check pool and the fit-pool review queue are labelled with `ss label specialists/routing/routing.yaml QUEUE --to REVIEWED`, conflicts are resolved from `claims.jsonl`, and `assemble --reviewed REVIEWED ...` plus `report` show feasible coverage. When review changes a label, update the translations of that situation the same way.
