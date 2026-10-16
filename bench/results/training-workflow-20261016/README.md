# Training workflow verification

The standard specialization pipeline now measures controls before fitting, returns support and learning reports, preserves recovery state, and connects stored predictions to review. These runs verify that workflow; they do not establish production reliability.

| Real native run | Base correct | Selected LoRA candidates | Outcome |
| --- | ---: | --- | --- |
| Customer/verification development | 11/14; 2 wrong accepted | 8/14 with 6 wrong accepted; 11/14 with 3 | Base retained; class-regression ranking broke the accuracy tie |
| Operation-only view of the same corpus | 6/6; 0 wrong accepted | 6/6 with 0 wrong accepted; 3/6 with 3 | Base retained by its tie preference |

Both used the existing Gemma E2B, ordinary LoRA and merged Q4_0 through stock llama.cpp. Native selection reused the committed base scores in both runs. These are small, disclosed development populations; the operation view is not independent evidence. No deployment was adopted.

The public optional-control run measured majority 7/14, lexical 7/14 and matched native base 11/14. Majority/lexical fit only fitting observations; their acceptance is unassessed. Repeating the same `ss specialize --until baseline` command reused every stage without scoring or optimizing.

The operation fit was interrupted with SIGTERM, saved recovery and resumed the same physical fit directory. Its final update journal contains steps 1–8 exactly once. Failed-run output retained the completed baseline/support paths. Normal execution completion is separate from whether a student improves.

The diagnostic selected four existing fit families associated with native errors, including ten observations and contrasting labels. The full-precision fit probe was already 10/10 before training and remained solved; it reports `already_solved`, not a capacity limit. Rank-16/alpha-32 and rank-64/alpha-128 controls were prepared with equal alpha/rank. The rank-64 control was not run.

A separate tiny, synthetic assigned-registry fixture exercised actual 2B optimizer updates and fit probes. Its fit NLL fell from 10.427 to 0.686 over eight updates, while raw fit correctness remained 4/8. This checks loss optimization, not successful registry memorization or language generalization. It stopped at fitting; no native student quality is claimed for that fixture.

Stored-row assessment rebuilding reproduced the customer metrics and ranking explanation without inference. Per-target thresholds remain `uncalibrated_default` in that run. Review imported existing reference histories and roles; predictions did not become labels. A round retained all 54 existing observations and stopped at preflight because it added no new fit evidence.

The legacy benchmark producer had put its name in `created`. A separate, documented timestamped reimport was used for the round's date cutoff, preserving inputs, labels, histories, criterion revisions, families and roles. Existing publications were not rewritten. New benchmark materializations now receive a real timestamp once and reuse their label bytes.

The original printer publication remains SHA-256 `66d3ae00a9ac53d9feda9afdc7f6bf5f8ebc17db9faec51b5f23a2d970e704e1`. Original customer training labels also retain their starting digest.

Validation: 1,353 core/repository tests, 288 trainer tests and 50 learn tests passed. The suite counts exclude 33 slow tests; the GPU runs, interruption/resume, CLI replay, controls and report recomputation above ran separately.

[summary.json](summary.json) records the measured candidate outcomes, support, policy status, recipe/base identities, recovery, diagnostic and control evidence, and a digested prediction payload. Detailed immutable exports, weights, checkpoints and logs remain under `work/training-workflow-20261016/`.
