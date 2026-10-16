# GA operation measurements — 2026-10-14

## Full printer specialization

The [bound run](../work/zt411-ga-20261014/full-fit-measurement.json) consumes the existing printer corpus with manifest SHA-256 `66d3ae00a9ac53d9feda9afdc7f6bf5f8ebc17db9faec51b5f23a2d970e704e1`. No corpus generation, relabeling or review publication ran. The recipe uses 100 fit situations (1,200 observations), 150 model-selection situations (1,800 observations), seed 0, evaluation batch 1, no fit probe, epoch cadence, eager attention, unfused AdamW and a finite 18,000-second allowance. It requests at most four epochs; the existing loss-plateau rule stops at two, after 300 updates. Convergence is recorded as `not_demonstrated`; no unevaluated weights are selected.

| Development measurement | Base/zero | Step 150 | Step 300 |
| --- | --- | --- | --- |
| HF canonical observations correct / 1,800 | 1,174 | 1,603 | 1,654 |
| HF complete target sets / 150 | 0 | 42 | 75 |
| HF full-vocabulary NLL | 2.5956 | 0.3664 | 0.6566 |
| Native canonical observations correct / 1,800 | 1,145 | 1,608 | 1,655 |
| Native complete target sets / 150 | 0 | 42 | 76 |
| Native accepted / 1,800 | 1,121 | 1,420 | 1,581 |
| Native correct among accepted | 808 | 1,315 | 1,438 |

These are model-selection measurements, not held-out quality. Native scores use the merged Q8_0 GGUF and its frozen execution; HF scores use the training readout. Their counts are kept separate. Authored thresholds remain unchanged because acceptance-fit is empty. The [public workflow development baseline](../work/zt411-ga-20261014/results/zt411-model_select-46213214c7abe842.json) has zero exact workflow successes among 124 gold-complete situations; it also retains primitive fit-majority support. Per-target accuracy does not establish that a complete operational decision is correct.

The [constant majority diagnostic](../work/zt411-ga-20261014/fit-majority.json), fitted only from all 1,200 canonical fit observations, gets 1,328/1,800 model-selection observations right (73.8%). This beats the unfitted model and is below both trained checkpoints. It predicts primitive labels only, with no workflow actions or acceptance policy. The public snapshot report's majority diagnostic has a different denominator because it uses rendered frames and deduplicates counterpart inputs; the two supports are reported separately.

The complete fit takes 3,483.26 seconds. The 300 measured updates total 1,463.04 seconds (median 4.81); the three full monitoring points take 409.18, 552.30 and 456.38 seconds. Fourteen checkpoints total 4.86 seconds. Work counters record 1,025,066 training tokens and 2,323,185 monitoring tokens. Process peak host memory is 6,524,010,496 bytes; peak PyTorch device allocation is 8,355,647,488 bytes, with 8,554,283,008 bytes reserved. These allocator values do not measure physical GPU traffic. Input loading, initialization, conversion and public native selection are outside optimizer-update timings.

The [batch measurement](../work/zt411-ga-20261014/monitor-batching.json) covers 24 fixed short-to-long observations, two reversed sweeps and caps 1/2/4. Median times are 9.707/8.677/11.266 seconds. Larger shapes preserve canonical argmax in this sample but shift NLL; [the recorded decision](../work/zt411-ga-20261014/monitor-batching-choice.json) keeps cap 1. This bounded sample is not a full-population throughput claim.

Native selection scores 1,800 public observations per arm and reuses their canonical scores in the 2,850-presentation diagnostic population, decoding the remaining 1,050 presentations. Public/diagnostic times are 468.52/221.05 seconds for base, 571.89/265.93 for step 150 and 414.83/314.08 for step 300. These are completed selection passes, not matched latency trials. A late publication failure exposed repeated family metadata; the [published receipt](../work/zt411-ga-20261014/deployment/receipt.json) now stores 70 distinct lineage descriptors for 3,000 observation memberships. It is 4,498,693 bytes versus 80,480,343 before normalization. Compact per-row score evidence lives in its digest-bound `selection.rows.jsonl`. Publication retries reused all committed scores. The finalized deployment is selected through the existing `semantic-trainer select` command against the verified completed fit; the earlier controller failure remains in its journal rather than being rewritten as success.

The successful publication-only retry takes approximately 102 seconds from shell-created log to final result write, including three GGUF conversions, native setup, file verification and publication. Every public and native observation is reused. An isolated conversion timer was not recorded, so this is the combined retry cost, not a conversion throughput claim. The fit measurement also retains the observed per-target permutation diagnostics from the same monitoring forwards.

Reproduce native selection without fitting or rebuilding data:

```sh
.venv-xpu/bin/semantic-trainer select \
  work/zt411-ga-20261014/specialization/finalize/28aa606c1916d2d1-164e1f1a/job.yaml \
  --run work/zt411-ga-20261014/specialization/fit/991c8a9193a0762d-ca739027 \
  --to NEW_DEPLOYMENT_DIRECTORY
```

### Held-out printer comparison

The [completed experiment](../work/zt411-ga-20261014/result.json), [full comparison](../work/zt411-ga-20261014/results/zt411-check-991aca151403c7a1.json) and [derived check](../work/zt411-ga-20261014/checks/991aca151403c7a1.json) cover 736 structured frames per arm, 164 canonical assignment families and 560 frames with complete references. All 1,472 public calls complete without failure. The remaining 176 frames have incomplete reference targets and are excluded from complete-set agreement. Both arms use the same native execution; weights, labels and authored thresholds stay frozen throughout the comparison. No paid reference is used.

| Held-out stored-reference measurement | Base | Quantized student |
| --- | --- | --- |
| Every scored target agrees, complete-reference frames | 0/560 | 438/560 (78.2%) |
| Fully accepted target sets on those 560 frames | 0 | 527 (94.1% coverage) |
| Correct among those fully accepted sets | Unavailable: none accepted | 406/527 (77.0%) |
| Action-selection positive precision | 242/551 (43.9%) | 748/872 (85.8%) |
| Required-action recall | 242/934 (25.9%) | 748/934 (80.1%) |
| Required actions omitted | 692 | 186 |
| Situations with a prohibited action, all 736 frames | 23 | 0 |
| Selected actions with unresolved reference | 50 | 16 |
| Unresolved action memberships, all 6,624 memberships | 1,402 | 353 |
| Failed public calls | 0/736 | 0/736 |
| Coarse act/wait case agrees, all frames | 488/736 | 644/736 |
| Public-call p50 / p95 | 1,434.6 / 3,019.4 ms | 1,393.3 / 2,933.0 ms |

The primary within-pack family bootstrap reports an agreement improvement of 78.2 percentage points, with a 95% interval of 73.4–83.0 points. It is descriptive agreement with the stored reference, not certified production precision. Calls alternate arms per frame; setup is outside row latency. The complete scoring loop takes 2,599.98 seconds. Each arm submits 840,558 native tokens in 9,567 decode submissions; logical public-call token counts are 945,382 per arm. The starting scorer snapshot matches the final sources exactly. The [host snapshot](../work/zt411-ga-20261014/comparison-resources.json) is explicitly not a lifetime memory peak; per-call measurement resets the host high-water counter, and native physical device-memory counters are unavailable.

Target-level accepted agreement is separate from complete-set agreement. For three-valued targets the counts preserve `true`, `false` and `unstated`; binary counts use the existing accepted-outcome reducer. Unassessed references are excluded from each target's support.

| Target | Reference support | Base correct / accepted | Student correct / accepted | Student unresolved |
| --- | --- | --- | --- | --- |
| tt_media | 736 | 520/695 | 736/736 | 0 |
| dt_media | 736 | 97/450 | 736/736 | 0 |
| ribbon_loaded | 712 | 263/571 | 648/710 | 2 |
| ribbon_absent | 712 | 648/697 | 696/712 | 0 |
| detection_symptom | 712 | 610/710 | 634/698 | 14 |
| verification_due | 712 | 345/483 | 634/712 | 0 |
| step_verified | 736 | 658/715 | 736/736 | 0 |
| unverified_step | 736 | 590/727 | 658/718 | 18 |
| calibration_verified | 736 | 496/630 | 736/736 | 0 |
| method_inconsistent | 736 | 194/442 | 672/733 | 3 |
| ready_to_print | 736 | 32/130 | 598/736 | 0 |
| needs_observation | 736 | 363/633 | 688/729 | 7 |

The student misses all 122 `ready_to_print=true` cases; `resume_printing` has F1 0 on 122 positive action cases. Its 598/736 accepted target agreement on readiness hides that failure. The fit population contains eight true readiness observations, 74 false and 18 unstated; this is support information, not a demonstrated causal explanation. Class support, confusion tables and all action F1 values remain in the raw comparison. The run does not establish production readiness or calibrated acceptance.

Adoption is **`unreviewed_reference`**. Exposure is fresh and receipt lineage is declared disjoint; both operational constraints pass. Existing reviews cover a different criterion binding for the structured task, and all 305 required evidence confirmations in this evaluation scope are absent. Those review limitations are retained rather than converting agreement into a reviewed quality claim. No promotion, corpus rebuild, label edit or review-history rewrite follows the result. All eight published corpus payloads and the original manifest verify after completion; the private corpus archive still hashes to `25e501581da37a67ce291b00277faa69a125c7bcb7f907352737312892f67b60`.

Exact comparison command; retrying the same plan reuses committed rows:

```sh
.venv/bin/python -m bench.snapshots zt411 \
  --pack packs/zt411 \
  --evidence work/zt411-q1/stages/66d3ae00a9ac53d9/printer.dataset.json \
  --baseline base=/home/tribeiro/.cache/semantic-switch/trainer/models/google--gemma-4-E2B-it/3e22461f65e89153144f8adb70e3b8c2cc9845a7/llama.cpp-a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd/Q8_0.gguf \
  --specialist specialist=work/zt411-ga-20261014/deployment/receipt.json \
  --kind workflow --condition structured \
  --corpus 66d3ae00a9ac53d9feda9afdc7f6bf5f8ebc17db9faec51b5f23a2d970e704e1 \
  --native device=gpu --native threads=4 --margin 0 \
  --results work/zt411-ga-20261014/results \
  --check work/zt411-ga-20261014/checks
```

Current implementation verification: 1,308 non-slow core tests and 280 trainer tests pass; the earlier focused integration checks pass separately. [Verification](../work/zt411-ga-20261014/verification.json) records commands, scope and source hashes. The CPU-backed mocked accelerator test fixes its thread count to one; it verifies placement and memory reporting, while the actual XPU fit above supplies hardware execution evidence. No battery checks, automatic sleep inhibitors, new predictor or hosted driver were introduced.

The current implementation uses separate native KV streams and packed token segments. Receipts identify their source snapshot, configuration and native build. Historical unified-KV measurements below describe a different execution; they are not an isolated speedup comparison.

| Control | Result |
| --- | --- |
| CPU and Vulkan base, 3 inputs × 4 questions | First-touch, repeat, prefix churn/eviction and whole-sequence reference log probabilities and mass match exactly. |
| Vulkan runtime adapters A/B | Repeat and residency controls match exactly. Whole-sequence reference differs: maximum answer-probability delta 0.017315 / 0.005642. Cross-path equivalence remains unsupported for these adapters/build. |
| Grouped diagnostic, five matched runs | Probability delta up to 0.0037513; zero changed default-threshold outcomes in the measured 12 rows. Grouping was slower. Production remains serial. |
| Default context | 3584 logical tokens per stream; six streams at four prefix slots; 21,504 physical tokens. Full-SWA KV allocation is 378 MiB on this Gemma4 model. Prefix slots multiply allocation, not the logical request window. |

Native source changes invalidate execution-bound locks and exports. There are no tracked native example locks to refit; the tracked hosted regression locks do not consume native execution. Loading never refits a lock. Native conformance is diagnostic and preserves the scorer's production mode; no score tolerance or automatic reference fallback hides a failed control.

Complete [GPU conformance](../bench/results/ga-conformance-gpu-20261014.json) and [CPU conformance](../bench/results/ga-conformance-cpu-20261014.json) pass first-touch/residency, question-order/call-size, answer-token and template controls. Changing internal decode chunks from 512 to 7 does **not** preserve exact scores: maximum differences are 0.0006983 for the GPU base, 0.0007473/0.0003232 for adapters A/B and 1.65e-14 on CPU. Doctor reports those cross-chunk failures. The controls no longer accept matching argmax or widened tolerances, and they leave the production mode unchanged. Exact equivalence is limited to the recorded fixed execution shape.

Current native receipts: [GPU](../bench/results/ga-native-gpu-20261014.json), [CPU](../bench/results/ga-native-cpu-20261014.json). These establish the listed scopes only. The grouped experiment uses an explicitly private backend boundary and includes setup and decode; it is not a public SDK throughput claim.

## Public operations

The [pre-change public baseline](../bench/results/ga-pre-c1-20261014.json) was captured before native edits, with 100 warm observations each for Check/Pick/Rate/Select and complete public serialization. Its p50/p95 values were 164.35/199.36, 167.56/173.77, 164.47/171.03 and 256.50/397.19 ms respectively. It used the old total-context interpretation and unified KV. The [intermediate post-change receipt](../bench/results/ga-post-c1-20261014.json) reports the new allocation; both configuration and source changed, so it cannot isolate the cost of the KV flag.

| Operation | Two slots p50 / p95 ms | Four slots p50 / p95 ms |
| --- | --- | --- |
| Check | 132.1 / 155.1 | 130.7 / 135.3 |
| Pick | 155.8 / 217.1 | 164.8 / 296.0 |
| Rate | 149.9 / 286.8 | 223.0 / 292.8 |
| Select | 231.4 / 250.2 | 235.0 / 243.5 |
| Switch | 156.9 / 198.8 | 147.4 / 192.5 |

Both runs include 100 warm observations per operation across two repetitions, complete result serialization, guided-prefix churn and three near-capacity calls. Every near-capacity call succeeds at 3,414 of 3,584 tokens (990 prefix, 1,000 input, 1,424 question). The 510 paired public results are identical; there are zero failed calls. Four slots allocate 396,361,728 KV bytes, versus 264,241,152 with two. In the four-guidance churn, four slots avoid 20 prefix decodes and 1,360 submitted tokens; observed mean churn latency is 182.5 versus 319.4 ms. Four slots remain the default for this working-set benefit, while two slots are available for tighter memory budgets. The busy-host timings do not isolate the effect of slot count; the source snapshots also differ by prefix-cache key byte accounting. No automatic tuner was added.

[Two-slot receipt](../bench/results/ga-public-slots2-20261014.json), [four-slot receipt](../bench/results/ga-public-slots4-20261014.json), [comparison with physical work and allocation](../bench/results/ga-public-comparison-20261014.json).


## Transport and algorithm scaling

| Complete operation / component | Measured result |
| --- | --- |
| Full all-partition printer export | 11,520 observations, 968 situations, 1,016 interned token segments; 7.42 s encode, 0.517 s read; 23,723,482 payload bytes. This measurement-only export is rejected for fitting. |
| Printer tune-only export | 3,000 observations, 250 situations, 298 interned token segments; 6.60 s encode, 0.360 s read. Six payloads total 6,073,451 bytes. |
| Export host high-water memory | 223,305,728 bytes after native load; 360,480,768 after export plus readback. This includes model, labels, tables and output rows. |
| Lock integration, deterministic judge | 128/512 labelled inputs submit exactly 128/512 physical questions, with overlapping embedded expectations. No extra model pass for those expectations. |
| Reuse, chunk 128 | Peak traced storage stays about 61 KiB at 1,024 and 8,192 requests; retained entries never exceed 128. Dataset/report memory is excluded. |
| Journal index, 256 inputs | About 120.5 KiB retained index at payload sizes 64, 4,096 and 65,536 bytes. A 16.8 MB journal does not retain its completion payloads. |
| Full teacher collection/resume | At 256 × 65,536-character inputs, input file 16.8 MB, traced peak 34.3 MB. Resume makes zero teacher calls and reproduces both output hashes. Input-map storage is included; native model memory is excluded. |
| Mention matcher | All-hit scan medians about 0.37/0.80/1.67/3.57 ms at 250/500/1,000/2,000 aliases. No-hit scans stay near 0.25 ms for the fixed text. Catalog construction is separate. |
| Family bootstrap | 10,000 families × 2,000 draws: about 0.16 s and 10.2 MB traced peak including family aggregation. Draw and gather scratch are bounded to 8 MiB; NumPy take uses its unbuffered valid-index path. Same-draw reductions match exactly. |
| Packed token payload | 100,000 token IDs occupy 400,000 bytes. A NumPy view adds no second payload. Python owner/view overhead is separate from encoded payload accounting. |

The all-partition export reached 361,009,152 process high-water bytes after export/readback versus 223,051,776 after native load. [Full-population receipt](../bench/results/ga-export-all-20261014.json) and [manifest](../bench/results/ga-printer-all-20261014.export.jsonl.manifest.json) retain the larger scale.

The lock/collection measurements use a deterministic test judge to count physical attempts and isolate host transport. They do not measure inference throughput. These are current scaling receipts, not reconstructed before/after timings: [export](../bench/results/ga-export-full-20261014.json), [algorithms, journal and lock](../bench/results/ga-scaling-20261014.json), [teacher publication](../bench/results/ga-teacher-scaling-20261014.json). Publication/resume tests separately cover failure and retry semantics.

## Complete training updates

Each configuration runs two warmup updates followed by 20 measured complete optimizer updates with the same seed, four fit observations, five monitoring observations and token-budget batching. Each also performs cold and steady monitor/checkpoint work. This is a bounded systems workload, not evidence of specialist quality.

| XPU configuration | Mean measured update | Peak device allocation |
| --- | --- | --- |
| Eager, unfused | 0.581 s | 6,744,324,096 bytes |
| SDPA, unfused | 0.682 s | 6,958,948,352 bytes |
| Eager, fused AdamW | 0.800 s | 6,744,375,296 bytes |

Across 20 updates, fused/unfused loss and gradient norms no longer match exactly: maximum absolute deltas are 1.2861e-5 and 8.0003e-4. SDPA deltas are 4.3760e-5 and 2.7394e-3. Defaults remain eager/unfused. This machine carried other work during these runs; the timing differences do not establish a general ranking. Numerical differences are retained in the [comparison](../bench/results/ga-training-steady-20261014.json) and complete [eager](../bench/results/ga-training-eager-20261014.json), [SDPA](../bench/results/ga-training-sdpa-20261014.json) and [fused](../bench/results/ga-training-fused-20261014.json) receipts.

A separate [actual fit](../bench/results/ga-fit-steady-20261014.json) uses a newly exported scoped population: 14 fit observations, 14 model-selection observations, 84 optimizer updates, 22 evaluation points and 13 checkpoints. It reaches the declared epoch cap with no unevaluated work and retains zero plus the ranked candidates. Native/HF base decisions agree on all 14 selection observations. Elapsed time is 191.86 s; this does not establish convergence or production quality. The receipt discloses source edits during the run (bootstrap buffer handling and diagnostic cleanup); it is operational/decision evidence, not a matched timing baseline. The source snapshot was captured before execution and was not replaced afterward.

Validation: **1,569 tests pass**, with 32 slow-marked tests excluded from the broad command. Real hardware and complete operations listed here were measured separately. A final focused run covering guidance/cache accounting, native transport and the added regressions passes 97 tests. [Validation commands, source snapshots, script bodies and receipt hashes](../bench/results/ga-validation-20261014.json) retain the reproduction details. Existing development virtual environments and pinned model assets are required for the hardware runs.

---

# Historical GA operation measurements

Measurements bind the code, data, model and execution identities recorded in their receipts. These are small baselines; no matched earlier public-call/export receipt exists, and no speedup or general quality claim is made.

| Operation | Measured scope | Result |
| --- | --- | --- |
| Native load | Gemma4 E2B Q8_0, Vulkan, context 8192 | 4236.7 ms |
| Warm check | Five public calls, complete result serialization | p50 146.2 ms; p95 149.4 ms |
| Warm pick | Five public calls, complete result serialization | p50 158.2 ms; p95 592.0 ms |
| Warm select | Five public calls, complete result serialization | p50 252.6 ms; p95 263.4 ms |
| Native serial/batch | Check, Pick and Select; actual submitted-token counters in raw rows | 24 measurements; zero failures |
| Normalized printer export | Nine tune observations; descriptor, input, family, answer and token tables | encode 3.231 s; read 0.0038 s; manifest 5730 bytes; payload 104864 bytes |

The export encode interval includes judge setup, label materialization and publication. Export RSS is the process high-water mark with the native model loaded, not an isolated table-memory claim. The table byte counts and row counts are retained in the receipt.

| XPU complete-update configuration | First update | Second update | Peak allocated device bytes |
| --- | --- | --- | --- |
| eager | 13.462 s | 0.944 s | 6741645312 |
| sdpa | 4.598 s | 0.740 s | 6957059072 |
| fused | 5.080 s | 0.838 s | 6741645312 |

The configurations share seed, data, precision, recipe and optimizer work. The first update includes compilation; only two updates were available. Fused AdamW matched eager/unfused loss and gradient norms exactly on both updates. SDPA changed their numerical values and increased peak allocation. Eager/unfused defaults remain unchanged. Transfer and host-embedding phase times are recorded; a physical device-traffic counter was unavailable.

The bounded fit trained four detection_symptom observations, monitored five selection observations, performed two optimizer updates and stopped at epoch_cap. It retained zero and step-2 adapters with no unevaluated work. These labels and this small fit do not establish specialist quality.

Native controls reproduced the original build and thresholds. One base-only urgency request changes probabilities by 0.03734 across cache namespaces; the same single request with adapter A changes by 0.01731 between shared and whole-sequence paths. Same-schedule switching, repeat determinism and warm-state reuse pass. Probability equivalence within 0.001 across those layouts/paths remains unsupported for the exact recorded Vulkan/Gemma4/Q8_0 scope. This isolates native execution geometry; it does not attribute a particular kernel instruction.

A real 262,144-entry float32 native logits row uses a reused 2 MiB float64 scratch, written twice per reduction. Fifty CPU-only repetitions measured a 0.780 ms reduction median and 0.00477 ms answer-conversion median. Decode and GPU synchronization are excluded. Precision and reduction behavior remain unchanged.

Claim materialization of independent equal-valued roots measured approximately 11, 21 and 46 ms at 500, 1,000 and 2,000 claims. This measured case uses the containment postings; dense actual containment retains its output-dependent cost.

Quiescent pruning removed 27 support and 36 printer generations: 265,236,592 bytes total. Current publications verify before and after; the reference inventory found no external generation references.

Raw receipts:

- [Public native calls](../bench/results/ga-native-public-final-20261013.json)
- [Normalized export](../bench/results/ga-export-baseline-20261013.json) and [export manifest](../bench/results/ga-printer-tune.export.jsonl.manifest.json)
- [XPU profiles and fit](../bench/results/ga-lora-comparison-20261013.json)
- [Native controls and unsupported scope](../bench/results/ga-native-scope-20261013.json)
- [Reduction attribution](../bench/results/ga-reduction-attribution-20261013.json)
- [Claim scaling](../bench/results/ga-claim-scaling-20261013.json)
- [Pack cleanup](../bench/results/ga-pack-cleanup-20261013.json)

Validation: 1,250 core/integration tests and 310 trainer/learning tests pass. The 32 slow-marked tests were excluded from the broad suite; the required native controls and XPU profile/fit runs above were executed separately. [Commands and raw logs](../bench/results/ga-validation-20261013.json).
