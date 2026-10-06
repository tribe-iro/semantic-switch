# Benchmarks and evidence

Run from the repository root with the project's environment. All numbers are on a Core Ultra X7 358H (16 cores, AVX2, 16 threads, CPU only), unless noted.

| Script | Question it answers |
|---|---|
| `python -m bench.judges --judges a,b` · `--analyze bench/results/judges` | Which judge and quantisation? Accuracy, AUC, ECE, coverage and precision at 0.9 (low answer mass never settles), cross-fitted coverage at 95%, picks, latency, memory, shift against Q8_0 |
| `python -m bench.fresh SRC …` | The fresh held-out suite: does the ranking hold on items never used for selection? |
| `python -m bench.identifiers SRC` | Near-identical ids (MX-120 vs MX-120A), with 4 and 20 candidates |
| `python -m bench.unstated SRC` | Does a three-way check separate "false" from "not stated"? |
| `python -m bench.procedures [SRC]` | Procedure selection over `switches/procedures.cards.yaml` |
| `python -m bench.topical [SRC]` | Do tier-1 heads earn their keep on topic routing? |
| `python -m bench.compare_jev`, `bench.compare_jev_quality` | Matched latency and quality against hosted Jev (see the contract note below) |
| `python -m bench.models`, `bench.learning_curve`, `bench.prompts`, `bench.probe` | Older: prompt examples, hidden-state probes (retracted, see below), framings |

The selection suite lives in `semantic_switch/quality.py` (run by `ss doctor --quality`); `bench/suite.py` re-exports it. **It chose the default judge and the exception phrasing, so it is regression data, not independent evidence.** `bench/fresh.py` is the independent check.

## Judge ladder: selection suite (240 checks, 24 picks)

| Judge | File | Acc. | AUC | ECE | Settled @0.9 (precision) | Picks | Check ms | ms per question in an ask | RAM |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.6 GB | 0.69 | 0.85 | 0.15 | 36% (89.5%) | 0.79 | 111 | 65 | 1.6 GB |
| Qwen3-0.6B Q4_K_M | 0.4 GB | 0.74 | 0.85 | 0.07 | 14% (97.0%) | 0.83 | 99 | 54 | 1.5 GB |
| Qwen3-1.7B Q8_0 | 1.7 GB | 0.84 | 0.93 | 0.16 | 97% (84.5%) | 0.92 | 290 | 202 | 2.8 GB |
| Qwen3-1.7B Q4_K_M | 1.1 GB | 0.76 | 0.89 | 0.22 | 92% (79.5%) | 0.96 | 306 | 159 | 2.7 GB |
| Qwen3.5-0.8B Q8_0 | 0.8 GB | 0.71 | 0.80 | 0.12 | 35% (83.3%) | 0.67 | 215 | 101 | 1.7 GB |
| Qwen3.5-0.8B Q4_K_M | 0.5 GB | 0.77 | 0.83 | 0.08 | 29% (91.4%) | 0.62 | 194 | 80 | 1.4 GB |
| Qwen3.5-2B Q8_0 | 1.9 GB | 0.90 | 0.95 | 0.04 | 68% (94.5%) | 0.92 | 540 | 216 | 2.8 GB |
| Qwen3.5-2B Q4_K_M | 1.2 GB | 0.78 | 0.96 | 0.11 | 67% (93.1%) | 0.92 | 348 | 197 | 2.4 GB |
| Gemma 3 1B Q8_0 | 1.0 GB | 0.70 | 0.79 | 0.29 | 96% (70.9%) | 0.88 | 210 | 101 | 1.3 GB |
| Gemma 4 E2B Q8_0 | 4.7 GB | 0.96 | 0.99 | 0.04 | 99% (96.2%) | 0.92 | 440 | 289 | 5.0 GB |
| **Gemma 4 E2B Q4_0** | 2.7 GB | **0.95** (0.963 after "Except: X.") | 0.99 | 0.04 | **92% (97.7%)** | 0.96 | **228** | **128** | 3.8 GB |
| Gemma 4 E2B QAT Q4_0 | 3.2 GB | 0.94 | 0.99 | 0.04 | 92% (97.7%) | 0.96 | 264 | 149 | 4.2 GB |
| Gemma 4 E4B QAT Q4_0 | 4.8 GB | 0.975 | 0.99 | 0.02 | 95% (98.7%) | 0.96 | 572 | 318 | 7.5 GB |

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

## Fresh held-out suite (160 checks, 20 picks)

8 new claims in new domains, 10 distinct minimal pairs each (5 EN, 5 PT), and HR topic picks.

| Judge | Accuracy | AUC | ECE | Settled @0.9 (precision) | Picks | Check ms |
|---|---|---|---|---|---|---|
| Gemma 4 E4B QAT | **0.931** | 0.969 | 0.054 | 95% (94.7%) | 1.00 | 578 |
| Gemma 4 E2B Q4_0 | 0.856 | 0.917 | 0.119 | 90% (**88.9%**) | 1.00 | 375 |
| Qwen3.5-2B Q8_0 | 0.844 | 0.938 | 0.063 | 66% (**95.3%**) | 1.00 | 419 |
| Qwen3-1.7B Q8_0 | 0.794 | 0.868 | 0.196 | 93% (83.1%) | 0.95 | 382 |
| Qwen3-0.6B Q4_K_M | 0.656 | 0.720 | 0.141 | 13% (90.0%) | 0.90 | 136 |

- **Stricter thresholds don't fix Gemma 4 E2B:** 89.7% at 0.95, 93.3% at 0.99 (66% coverage), 91.1% at 0.999.
- **Its weakest phenomena:** hypothetical 0.70; contrast and implicit 0.80; negation and temporal 0.82.

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

**Staged vs eager with the input cache (E4, `bench/schedule.py`, route_ticket):** two stages took 243 ms when the first case decided (eager 942 ms), and were within 0–8% of eager when a later case decided. One round per case was 8–40% slower than eager. The `schedule` option was deleted.

**Exception phrasing** (31 exception items): "Except: X." got Gemma 4 E2B to 30/31 (from 27/31), and Qwen3-0.6B went 28 → 27. It was chosen on the items it was scored on.

## Option position, order averaging, comparative choice

**Option position** (`bench/positions.py`, default judge, 24 picks, correct option moved through a 20-option list):

| Slot | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Top answer correct | 21 | 24 | 24 | 24 | 24 |
| Settled wrong | 3 | 0 | 0 | 0 | 0 |

All three misses are ambiguous items whose plausible rival came second in that layout. A direct two-option test flips confidently: "Não consigo entrar, então nem vejo minhas faturas" → billing 0.95 as [account, billing], account 0.85 as [billing, account]. In 3 of the 4 ambiguous cases the second-listed option won; clear items don't move.

**Averaging forward and reversed orders** (`bench/permute.py`, default judge):

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

## Probe on the judge's hidden state (`bench/probe.py`)

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

## GPU (`bench/gpu.py`, E3)

| | E2B CPU | E2B GPU (Arc B390, Vulkan) | E4B GPU |
|---|---|---|---|
| check, 20 / 261 tokens | 324 / 1,009 ms | 147 / 201 ms | 248 / 330 ms |
| extra question, input cached | 158 ms | 61 ms | 91 ms |
| fresh suite: accuracy, precision at 0.9 | 0.856, 88.9% | 0.863, 89.0% | 0.938, 94.7% |

CPU and GPU agree on the default judge (1 flip in 160 fresh items; identical to two decimals on the probe). Qwen3-0.6B Q4_K_M does not: +0.10–0.26 on Vulkan, 2 of 12 probe answers flipped. `device="cpu"` in a Vulkan build was 2× slower until it loaded with `no_host` and op offload off. Hidden-state output on every token cost 7–17% on the GPU, so it is enabled only for probe questions.

## Equipment procedures (`bench/equipment_jev.py`)

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

Jev hasn't been run on the fresh suite.

**Contract:** TypeSafe's Master Customer Agreement §2.3(b) forbids using Jev outputs for distillation, to train a model to imitate it, or to develop a competing product. Further Jev runs wait for a reading of it or consent, and Jev outputs never enter training data.

## Deferred, with the reason

- **GLiClass multilang mini:** its card reports a multilingual zero-shot F1 of 0.54 and it needs torch.
- **FTS5:** unnecessary below about 10k candidates.
- **NER, stemmers, ontologies:** each waits for a failure it fixes.
