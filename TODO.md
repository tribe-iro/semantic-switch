# Open work: establish and improve the existing 2B specialist

Keep ordinary LoRA, the causal-LM answer-token scorer, full-vocabulary group NLL, merge/GGUF deployment and one Executor. Deliver a reusable recipe measured on language plus context, required behaviors and unfamiliar families. Current results establish neither a 2B capacity limit nor production reliability.

## Verified starting point

Evidence: [current source/corpus audit](work/review-2b-evidence-20261015/audit.json), [fit-only lexical control](work/review-2b-evidence-20261015/lexical.json), [T7 metric audit](work/training-audit-20261015/t7-implementation-review.json), [lost-relation witness](work/training-audit-20261015/review-followup.json).

| Same T7 model-selection population | Correct primitive answers |
| --- | --- |
| Per-target majority fitted on fit only | 1,116/1,500: 74.4% |
| Existing lexical features + logistic regression, fit only | 1,344/1,500: 89.6% |
| Base through native execution | 970/1,500: 64.7% |
| Selected LoRA through native execution | 1,414/1,500: 94.3% |

These are development measurements. The lexical control uses existing hashed words, bigrams and character n-grams, C=10, without selection tuning; it misses all four calibration_verified positives and all eight step_verified positives. Aggregate accuracy does not establish required behaviors. T7's 427/560 measures exact public action sets, not every primitive correct; another 176/736 frames have incomplete action reference. Repeatedly consulted check data remains regression evidence. The reference reads annotations, not public notes: it is not a demonstrated 100%-accurate text parser.

The [controlled coverage experiment](work/coverage-experiment-20261015/cohorts.json) now supplies two additive fit publications, each 96 situations in 16 size-matched new families, with the original evaluation fixed. [CPU controls](work/coverage-experiment-20261015/cpu-controls.json) score 89.6% original, 94.1% random enrichment and 94.5% targeted enrichment. [Raw action projection](work/coverage-experiment-20261015/lexical-actions.json) improves exposed-check exact sets from 295/560 to 481/560 and 463/560, but introduces two/one forbidden-action situations. These are uncalibrated lexical controls, not new LoRA or deployment qualification results. [Exposure](work/coverage-experiment-20261015/cpu-exposure.json) is explicitly consulted/non-independent. The [LoRA run status](work/coverage-experiment-20261015/status.json) records the queued matched experiment.

Present in code; extend these owners instead of rebuilding them: prose-view materialization, additive coverage publication, family closure, class_family allocation, authored slices, development recall requirements, shared counters, checkpoint recovery, fixed five-run qualification, bounded fit diagnosis and review reasons. Their existence does not demonstrate effectiveness. Qualification currently measures selected primitive answers; operational actions remain unassessed.

## Owners, transports and execution

| Stage | Existing owner | Boundary data |
| --- | --- | --- |
| Author/acquire | Task producer; semantic-learn collect/review | Public input, reference evidence, ancestry, derivation, explicit role |
| Publish/materialize | Evidence publication; label accumulator | Immutable parent plus additions/view mappings; no model loading |
| Prepare | Trainer jobs; export encoder | Canonical rows, closed families/roles, shared token segments, weights/support |
| Fit | Trainer objective/lora | Existing presentations and scalar weights; recovery/checkpoints |
| Measure/select | Core metrics; monitor/select; compare | Scoped saved semantic predictions, statuses, families and scores |
| Serve | Runtime/Executor; ordinary GGUF | Public input and trusted context; typed decisions/non-answers |

Annotations, slice membership and expected answers never enter serving prompts. Reuse current manifests/JSONL and packed tokens. Generation, tokenization, fitting and evaluation remain separately invocable; recipe changes do not regenerate source data. Reference immutable publications and existing row IDs instead of copying corpora into each run.

Execution dependencies: C1–C3 are CPU-first correctness/evidence work. C4 supplies approved additions; C5 compares data/allocation interventions. C6 follows a useful nominal recipe; C8 independently confirms it. C7 runs only on an identified fit failure. C9 consumes verified failures. C10 closes existing experiments independently. No new runtime, scheduler, database, experiment platform or compulsory qualification gate.

## C1 — Align policy, reference and learning targets

**Verified scope:** the [perfect-fact audit](work/coverage-experiment-20261015/contract.json) matches all action-applicability states on the 1,000 published structured frames. The counterexample below is outside that publication; it does not explain T7 errors. Independent summaries can still discard decisive relations.

- Add an offline perfect-fact arm to existing task diagnostics: supply authored semantic answers through the scorer test seam, run normal Executor/cards, compare action memberships and unknown states with the reference. This measures composition, not language accuracy; expose no oracle SDK provider.
- Enumerate small task-local valid domains with itertools.product plus explicit missing/conflicting evidence fixtures. Enumerate admissible states, not arbitrary products of marginal labels. Emit mismatched actions, premises and source rules; keep enumeration in task producers/tests, outside core execution.
- First fixture: calibration_1, TT method, TT media, ribbon absent. Reference gives method_inconsistent=false and cancel=true; the card requires that fact true. This is a source/card contract conflict, not a measured cause of T7 errors. None of the 1,000 published structured frames contains it. Adjudicate cancellation against the source/convention, then fix the card, reference or authored evidence predicate. Do not widen a fact's meaning merely to conceal a mismatch.
- Test sufficiency: equal extracted answers must determine equal reference memberships and unknown states, or have an explicitly justified conservative abstraction. The readiness witness has equal summaries but false versus unknown reference output. Retain a directly learned joint predicate or extract the missing relation; a conservative unknown is an explicitly measured coverage trade-off. No generic possible-world solver or new negation syntax.
- Preserve false, semantic unstated/none, conflicting evidence and confidence abstention through labels and composition. Use existing references/exclusions for complements. Publish corrections as additive criterion/reference revisions; retain old payloads, reviews and bindings. Recompute only invalidated labels/stages.

**Verify/output:** CPU contract report, complete/missing/conflicting cases, the cancellation witness, the readiness witness and a non-printer composition fixture. Separate perfect-fact errors from learned errors. Unresolved disputes remain visible without preventing diagnostic training.

## C2 — Measure the language users actually supply

**Defect:** all 986 original labeled rows are structured. Existing prose has a small phrase inventory; re-rendered annotations are not independent language evidence.

- Reuse prose materialization after C1. Preserve source families, roles, translations and permutation links. Recheck guards with complete/missing/conflicting/stale evidence: absent structured fields must be interpreted from visible text or remain unresolved, never read from hidden state.
- Add independently authored or consented real notes through existing publication/review. Authors see the operational brief, not generator phrases or model answers. Reviewers label from public input, declared trusted context and procedure; inaccessible physical state cannot justify a label. Preserve missing evidence, disagreements and source citations.
- Declare separate representative workload and targeted behavioral evaluations. Synthetic/teacher paraphrases supplement stress coverage but retain ancestry and derivation; they do not become independent domain confirmation. A 50–100-note pilot locates failures, not certifies reliability.
- Use existing roles and family closure. Development may guide changes; separate confirmation families stay out of fit, acceptance fitting and checkpoint selection. Partition incident/source ancestry before rendering. Report actual wording/template/time/site separation; do not claim unimplemented holdout axes.
- Exercise entity attribution, temporal supersession, negation, command versus completion, contradictory/absent observations, irrelevant context, paraphrases and option order. Authors specify invariant transformations and label-changing contrasts. Include the same surface phrase under contexts requiring different answers.

**Verify/output:** reusable serving-shaped development and confirmation publications; labels justified from visible fields; no annotation leakage; siblings never cross roles. Reuse existing views rather than regenerating the original corpus. Changed criterion text creates a new view binding, not inherited review validity.

## C3 — One fair baseline and evaluation path

**Defect:** summaries mix primitive accuracy, exact actions and accepted precision. The lexical entry point fits every supplied tune row without resolving fit/model_select roles: its cross-fitting protocol cannot silently serve as the fixed-role control. Qualification omits action execution.

- Keep train_lexical as the public SDK/CLI entry point. Add an explicit fit-cohort selection resolved by the shared role/cohort owner, without torch/device/base-model setup; reuse the existing features/solver. Bind eligible IDs to bundle provenance and restrict final/fold fitting to those IDs for this control. Existing tune-wide cross-fitting measures a different protocol and cannot substitute for fixed-role evaluation. For a one-class target use the constant baseline.
- Project majority, lexical, base and student primitive predictions through normal Executor/cards for action comparisons. Keep C1's perfect-fact arm separate. The majority baseline has an explicit always-answer policy, not fabricated calibrated confidence or vocabulary mass. Lexical and LM scores retain their own meanings and acceptance fitting. Until projected, a target-only baseline reports action quality unassessed.
- Reuse core reducers for raw class recall, macro recall, complete primitive sets, accepted precision/coverage, operational recall, exact action sets, omissions and forbidden recommendations. Report complete/incomplete-reference denominators. Withheld gold positives are operational misses; failed/missing predictions stay in scope.
- Extend saved prediction rows only where fields are missing: observation/input ID, task/criterion, family, role/view, semantic gold/prediction, status and existing score/mass. Join authored slices offline by ID; reducers consume iterators. Do not repeat full inputs or vocabulary arrays in every report.
- Match input fields, criteria, IDs and deployed execution across arms. Fit thresholds only on acceptance-fit data. Reuse family-aware paired intervals; disclose support/method. Degenerate zero-error bootstrap draws do not establish a zero upper error bound. Independent-family binomial bounds apply only when the event unit and sampling assumptions match.

**Verify/output:** fit-role leakage sentinel; identical baseline/student populations; abstention/incomplete-reference fixtures; named metric units; reports reproducible from saved predictions without model calls. Preserve the measured lexical 1,344/1,500 as target-only development evidence.

## C4 — Produce valid coverage independently of fitting

**Open evidence work:** class totals conceal contextual shortages. The producer now selects existing authored flows without automatic prompt rewriting; 24 focused tests pass, including unmet-context requests. Additions must still be checked for coherent operational sequences and faithful text; reference recomputation is not independent domain validation.

- Audit coverage transformations against task-local state/event semantics before using additions. Update linked premises together; historical steps, completion and commands remain coherent or become explicitly authored contradictory evidence. Preserve alternatives instead of overwriting scalar conflicts. Report rejected invalid cases.
- Use existing coverage requests for distinct fit families in measured weak `(target, class, slice)` cells, with matched negatives/unknowns and independent nuisance variation. Publish requested/achieved coverage. No automatic ontology, tag Cartesian product or student-produced gold.
- Preserve ancestry across contrasts/renderings. Exclude parent and previous-publication inputs/families with existing closure; a different seed is insufficient. Evaluation additions use declared independent families and roles, never reassigned to fit after inspecting predictions.
- Bind slices to published metadata/reference evidence, never predictions. Reuse behavior_support after allocation: observations, canonical families, objective share and empty declared cells. Overlapping slices do not duplicate loss. Review noisy/ambiguous labels before increasing their weight.
- Consume original fit plus approved additions through existing merge/cohort logic. Keep enriched fit frequencies separate from representative validation prevalence. Additions and materialized views are reusable publications; fitting has no producer side effects.

**Verify/output:** valid/invalid transition fixtures, a generic non-printer producer fixture, ancestry and role-exclusion round trips. Produce once and fit twice with zero generation calls and unchanged parent hashes.

## C5 — Compare data and effort before changing optimization

**Defect:** allocation/retention features exist without demonstrated benefit. One dead class or a declining checkpoint does not identify an LR, rank or capacity cause.

- Fix base/tokenizer, criterion/view, development population, ordinary LoRA targets, rank/alpha, batch, cadence and budget. Compare current versus class_family allocation on identical rows/seed through jobs.objective_weights. Compare approved additions next under the retained allocation. Keep catalog changes separate.
- When enough fit families exist, use three nested fit-family subsets to inspect the data learning curve. Keep contrasts/siblings intact and report support. Compare at common canonical-example exposure and wall time; different update counts are not automatically data gains. This is an explicitly requested diagnostic, not routine overhead.
- Retain one planner, deterministic shuffling and the current full-vocabulary objective. Existing class_family preserves task/target budgets and equalizes supported class/family cells; report multipliers. It cannot create missing contexts, correct labels or restore lost relations. Keep it optional until measured.
- Read existing checkpoint per-class previous/best deltas, NLL, answer mass and fit probes when requested. Poor audited fit goes to C7; good fit with independent-family failures goes to coverage/shortcut audit. Acceptance calibration does not repair semantic argmax errors.
- Keep score-based stopping and optional development recall requirements. Compare the supported cosine schedule only if repeated constant-schedule runs deteriorate late, using the same horizon and declared warmup. No optimizer collection, adaptive sampler or automatic sweep.

**Verify/output:** matched intervention report with class/slice/action outcomes, learning curves, token/update counts, wall time and peak host/device memory. Report regressions and the best available result even when requirements are unmet. Reuse compatible completed fits; no paid reference by default.

## C6 — Complete operational recipe qualification

**Defect:** qualification.completed reports operational_actions=unassessed; summarize aggregates primitive selection metrics. This cannot qualify composed operational behavior.

- Extend existing qualify to invoke existing acceptance fitting/compare for each selected receipt on the same declared development task/population. Reuse matching native prediction tables; no second evaluator/runtime. Bind corpus, criterion, calibration inputs and execution shape.
- Use the implemented finite design after choosing a useful nominal recipe: seeds 0,1,2 at nominal LR, seed 0 at half/twice LR. Fix other settings and reuse matching nominal runs. This measures local sensitivity, not universal robustness or all optimizer settings.
- Keep raw model-select requirements distinct from operational requirements/data roles. Persist met/unmet/unassessed with scope, counts and families. Raw retention requirements never imply downstream safety; target-only evidence cannot be operationally met.
- Report worst/median/range across all declared variants for class/context recall, exact actions, omissions, accepted precision/coverage and forbidden recommendations. Include failed, interrupted and budget-limited runs. Never replace poor seeds or present a convenient subset as the full design.

**Verify/output:** fake-transport full-operation fixture with abstentions/composed errors; nominal reuse compatibility; action-unassessed versus met scopes. One comparable five-run development report, not a new mandatory SDK gate. Slice reduction adds no inference.

## C7 — Diagnose adaptation limits on audited fit data

**Trigger:** persistent audited fit errors after C1/C4, not merely a disappointing held-out average.

- Use existing diagnose on a bounded fit-only cohort of failures, contrasts and unknowns. Audit slot/class mapping, conflicting duplicate labels and visible-evidence sufficiency first. Evaluate at declared points with the existing optional probe; routine fit probes stay off.
- Reuse gold NLL, semantic recall, answer mass, LR and pre-clip norms without per-example forward/synchronization overhead. Early clipping does not prove a bad LR. Memorizing fit cases checks optimization, not generalization.
- If rank 16 cannot fit audited cases, compare ordinary rank 64 with explicit alpha/initializer/LR/horizon through the same attention+MLP LoRA/merge path. Record effective scaling and memory: rank changes alter dynamics, so a failed comparison cannot establish a backbone limit.
- If higher rank fits, measure independent development families before retaining it. Otherwise report the unresolved label/objective/optimization/capacity diagnosis and bounded work performed. One failed LR/rank combination does not exhaust the 2B model.

**Verify/output:** corrupted-slot and contradictory-label fixtures; scoped fit-only report; unchanged holdout exposure; ordinary deployable GGUF. No custom head, special tokens, RL objective or full-finetune prerequisite.

## C8 — Confirm on new language and families

**Defect:** exposed synthetic check data cannot confirm generalization after repeated task/model changes.

- Freeze recipe, criterion and acceptance policy using existing identities before evaluating C2's independent confirmation. Exclude confirmation families from fitting, qualification and calibration. Reuse compare/exposure tracking; no new holdout registry or certification layer.
- Compare matched majority/lexical/base/student through C3. Report representative traffic separately from stress cases. Labels corrected after inspecting outputs remain documented corrections; the revised system has not thereby received fresh independent confirmation.
- Report per-behavior/family support and operational failure costs. Application owners declare precision/recall/coverage needs; absent support is unassessed. A pilot with a handful of positives cannot qualify a rare behavior.
- Reuse a second existing non-printer task/publication through the same interfaces before calling the method reusable; add only missing serving-language cases. Its semantics remain task-owned; no printer names or tags enter trainer/core abstractions.

**Verify/output:** reproducible deployed 2B comparison on independent evidence, uncertainty/failures, latency units and resource cost. State tested domain/shift axes and unsupported behaviors. No automatic adoption or larger-model substitution for unresolved benchmark defects.

## C9 — Improve from verified failures without forgetting

**Defect:** uncertain-only/latest-error mining misses confident mistakes and can erase earlier behaviors.

- Reuse capture, random audits and targeted review. Preserve random inclusion probabilities separately from targeted strata; disagreement/high loss requests review, never gold. Keep teacher derivation and training-use permissions.
- Use existing review reasons: meaning/label disputes to C1/review, input mismatch to C2, coverage deficits to C4, fit failures to C7, withheld-correct cases to existing acceptance fitting. No autonomous ontology editor or additional routing service.
- Train on original fit plus approved additions through the same allocation planner. Re-evaluate prior and newly targeted development requirements; never replace fit with only recent failures. Measure fresh random audits separately from deliberately selected errors.

**Verify/output:** one explicit reviewed cycle recovering a weakness while reporting regressions, reusable publications and zero generation during fitting. No background scheduler, replay database, automatic promotion or required paid teacher.

## C10 — Close existing presentation/native evidence

- Finish/report the existing catalog experiment from its frozen source/job and recovery state; do not restart under changed code or report intermediate selection as final quality. Compare matched inline/catalog criteria, raw/operational behavior, whole-case latency, evaluated tokens and peak memory. Existing check results remain regression evidence.
- The later [production-shaped grouped control](work/robust-implementation-20261015/grouped-production.json) already measures eight exposed frames, whole-operation timing and accepted outputs. It found a small timing benefit with changed probabilities. Retain production scheduling until representative isolated measurement earns a change; no second scheduler or universal cross-shape equality requirement.
- The doctor control now distinguishes repeated/history stability from chunk-shape differences. Verify current controls in their declared scope; same-build history stability protects threshold decisions, while cross-hardware/shape bit equality is not an accuracy requirement.

**Verify/output:** completed scoped receipts or explicit interrupted/unassessed states. Reuse saved predictions for reductions; new inference measurements follow actual execution-contract changes.

## Research constraints

- [ShieldGemma](https://ai.google.dev/gemma/docs/shieldgemma/model_card) demonstrates narrow 2B Yes/No scoring; its optimal F1 and policy sensitivity do not promise arbitrary operational reasoning or supply our acceptance thresholds.
- [LoRA Land §5.2](https://arxiv.org/html/2405.00732v1) supports specialization but reports fine-tuned 7B beating fine-tuned 2B on 29/31 tasks. Its mixed-model average is not a 2B promise. [Distilling Step-by-Step](https://arxiv.org/html/2305.02301v2) uses rationale supervision; ordinary hard-label LoRA does not inherit its results by citation.
- [CheckList](https://aclanthology.org/2020.acl-main.442/) and [WILDS](https://proceedings.mlr.press/v139/koh21a.html) motivate behavioral and distribution-shift evaluation through existing task producers/evaluators.
- [Group DRO](https://arxiv.org/abs/1911.08731) and [JTT](https://proceedings.mlr.press/v139/liu21f.html) motivate minority evaluation; hard-example reweighting is not a generalization guarantee. Measure the existing simple allocation before adding algorithms.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) motivates checking rank, batch and MLP coverage; this trainer already adapts MLPs. Another model's optimum LR is not a prescription for this workload.

Temperature/prior correction, soft targets, rationale training, consistency losses and alternative heads remain separate hypotheses until measurements identify residual failures they can address. Keep one deployed scorer and one training objective until a controlled experiment earns a change. Robustness is measured over specified evidence and perturbations; no loop is automatically antifragile.
