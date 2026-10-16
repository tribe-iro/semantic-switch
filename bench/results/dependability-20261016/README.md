# Dependability implementation and development evidence

The family-audited experiment did **not** demonstrate a useful LoRA improvement. Native selection retained the base. No deployment was promoted.

| Native arm | Selection answers correct | Wrong accepted |
| --- | ---: | ---: |
| Fit-only majority | 7/14 | Unassessed |
| Fit-only lexical | 7/14 | Unassessed |
| Matched pipeline base | 11/14 | 2 |
| Rank-16 LoRA, step 5 | 8/14 | 6 |
| Rank-16 LoRA, step 10 | 11/14 | 3 |

The corpus contains 54 authored observations across customer intent and operation verification: 36 fit, 14 model selection, 2 acceptance fitting and 2 check. These are development cases, including inspected failures. Two check observations are insufficient for a reliability claim; neither they nor the synthetic printer reference provide independent human confirmation.

Bilingual behavior origins were conservatively joined. Eight groups with incompatible historical roles were excluded; compatible old roles and ancestry were retained. Earlier studies with per-pair family IDs are preserved as superseded diagnostics in the report. Their small gains are not generalization evidence.

Situation and class/family allocation produced identical numerical weights on this paired corpus, so a duplicate fit was skipped. The separate rank-16 fit diagnostic was already 6/6 before training and stayed 6/6. It provides no reason to increase rank. Five-run qualification and temperature fitting were not triggered by an unsuccessful nominal adaptation and sparse calibration data.

The composed-decision evaluator was also exercised on the existing printer specialist's 150 model-selection situations. All 1,286 known action memberships matched. It reported 124 completely referenced situations and 26 partial references separately, with zero known required omissions or prohibited recommendations. This checks the development population; it does not upgrade its unreviewed reference or establish production safety.

The CLI returned identical results with and without matching device/thread options on the frozen printer deployment. The original printer publication remains unchanged at SHA-256 `66d3ae00a9ac53d9feda9afdc7f6bf5f8ebc17db9faec51b5f23a2d970e704e1`.

[summary.json](summary.json) pins model, quantization, recipe, execution, family audit, native candidate results, conditional-experiment decisions and compact prediction payloads. Reports derive from those rows; baseline acceptance and composed probabilities are not fabricated. Weights, immutable exports, checkpoints and detailed journals remain under the named local work directories.

Validation: 1,353 core/repository tests and 327 optional-package tests passed; slow checks were excluded from those counts. Real Gemma GPU fitting, merge/quantization, native selection, frozen CLI execution and composed evaluation ran separately from the scripted tests.
