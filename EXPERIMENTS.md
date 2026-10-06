# Experiments

Planned work that may change defaults. It isn't library behaviour: the spec describes what ships, and this file describes what we try next and how to decide.

## E1. A fine-tuned small judge

**Question:** can a small, fast judge (Qwen3-0.6B, about 100 ms on 20-token inputs, 420 ms on 261) reach the default judge's precision after LoRA fine-tuning? A fine-tuned judge is just another model file for the `judge` role; nothing else changes.

**Data (licences checked):**
- **Never Jev outputs.** TypeSafe's Master Customer Agreement §2.3(b) forbids distillation and training to imitate its output.
- **MASSIVE** (Amazon, CC-BY-4.0): real utterances in English and Portuguese with intents. Intents become claims, with related intents as hard negatives. This gives scale, but no negation or role phenomena.
- **A local teacher, Gemma 4 E4B** (Apache-2.0, outputs free to use): it labels generated minimal pairs across the suite's phenomenon tags. Answers below its tuned threshold are dropped, and a sample is checked by hand.
- **Real labels** from `ss correct` / `ss label`, once there is traffic.
- **Splits** are by sentence family, with translations kept together.

**Training:**
- LoRA on the base model. Prompts are produced by `Judgment.questions`, so training and runtime prompts are byte-identical.
- Loss only on the answer token.
- Every question kind (check, three-way, pick, rate) is included.

**Export:** merge → GGUF → the same 4-bit format as the baseline (`llama_model_quantize`) → `ss doctor --quality`.

**Evaluation:**
- the fresh suite (headline);
- the selection suite (regression);
- a **changed-definition set** (the same inputs with an edited claim or exclusion; the model must follow the edit, or it has become a fixed classifier);
- the identifier and unstated sets;
- latency at 20 and 261 tokens.

**Gate:** beat Gemma 4 E2B's fresh-suite accuracy (0.856) and untuned precision (88.9%) at Qwen3-0.6B speed, after quantisation. Variant: Qwen3-1.7B with LoRA, aimed at its confidence errors.

**Outside reference: Jeff** (firelex/jeff): a 0.8B model fine-tuned for choice / yes-no / score, Apache-2.0 weights in GGUF, trained on synthetic data written by open Qwen models (no contract problem). It is E1 built by someone else. Two limits keep it out of our judge list:
- it reads answers through its own trained output head, not letter tokens, so our adapter can't run it fairly; a fair test needs a `Judge` over its own server;
- it is English only.

Worth one fresh-suite run (English half) through such a `Judge` before we spend training time, to see what a tuned 0.8B buys.

**What a small trained model must clear on this CPU** (matched: one check, input not cached, median of 5, idle machine):

| Judge | 20 tokens | 261 tokens | Cost relative to the judge (c) |
|---|---|---|---|
| Gemma 4 E2B | 345 ms | 1,000 ms | 1 |
| Qwen3.5-0.8B (Jeff's base) | 209 ms | 487 ms | 0.61 / 0.49 |
| Qwen3-0.6B | 150 ms | 399 ms | 0.43 / 0.40 |

**Two ways to use it:**
- **As the judge** (`--judge`): every decision gets 1.6–2.5× faster, and it must match the default's fresh-suite accuracy and precision.
- **As a per-fact specialist in front of the judge:** it reads the input itself, so the expected cost is c + (1 − s), where s is the share it settles. It saves time only if **s > c**: it must settle about 40–60% of inputs, at least as accurately as the judge. Jeff's "28× faster with a 27B fallback" comes from a GPU and a much larger fallback; here the fallback is already small.

Jeff's own curve (triage: 33.8% with no adapter on plain Qwen3.5-0.8B; 76% at 250 labels; 91% at 72,000; "with plenty of data the base hardly matters") says this regime starts at hundreds to thousands of labels.

**Evidence against expecting a free lunch** (`bench/probe.py`): a single generic readout trained on the selection suite's claims did not transfer to the fresh suite's claims. Accuracy went 0.856 → 0.881, but AUC fell 0.916 → 0.894 and precision at 0.9 fell 88.9% → 88.2%. Per-judgment readouts (probe, calibration) also failed to beat the judge once translation families were held out (spec §5.4). Adaptation with a handful of labels doesn't buy semantics, so E1 is the path to a better judge; it must beat E2B and approach E4B on the fresh suite **with families held out**.

**Cost:** a few hours per epoch at about 2k examples on this CPU (an estimate), or about an hour on a rented GPU.

## E2. Real traffic

One real switch with at least 300 labelled items, split chronologically. Report precision and coverage after `ss lock`, silent misses per fact, lexical-head admission, and latency. This replaces synthetic suites as the basis for defaults.

## E3. GPU (Vulkan) — done

`bench/gpu.py`, Arc B390, all layers offloaded. All six backend checks pass on the GPU (D1 and D3 exact, D6 drift 0.02).

| | E2B CPU | E2B GPU | E4B GPU |
|---|---|---|---|
| load | 1.7 s | 1.0–1.4 s | 3.0 s |
| check, 20 / 261 tokens | 324 / 1,009 ms | 147 / 201 ms | 248 / 330 ms |
| extra question, input cached | 158 ms | 61 ms | 91 ms |
| 4 questions, 261 tokens | — | 288 ms | 395 ms |
| fresh suite: accuracy, precision at 0.9 | 0.856, 88.9% | 0.863, 89.0% | 0.938, 94.7% |

**Findings and decisions:**
- **Fixed:** `device="cpu"` in a GPU build ran 2× slower (GPU host buffers disable the repacked kernels; large batches were offloaded). CPU mode now loads with `no_host` and op offload off.
- **Fixed:** the Vulkan device banner leaked to the terminal; logging is silenced before backend initialisation.
- **The default judge stays device-independent**, for lock portability; `--precise` is recommended with a GPU.
- **Two-stage scoring stays** (±20–30% either way on the GPU).
- **Small models drift across backends:** Qwen3-0.6B on Vulkan shifts 0.10–0.26 and flips 2 of 12 probe answers. Run `ss doctor --quality` per device.
- **Hidden-state output is opt-in per question:** always-on cost 7–17%.

**Next:** batching independent inputs into one decode. With a question at about 60 ms, throughput on streams may gain; measure `many()` throughput against one decode over several inputs' questions.

## E4. Staged vs eager with the input cache — done

`bench/results/schedule.json`, route_ticket, default judge, median of 3. With the cache, two-stage scoring took 243 ms when the first case decided (eager 942 ms) and 972–1,334 ms when a later case decided (eager 941–1,238 ms, 0–8% faster). One round per case was 8–40% slower than eager and was dropped. **Decision:** two stages, no `schedule` option.

## E5. Retrieved precedents on the default judge — done

Gemma 4 E2B, fresh suite, leave-one-pair-out, 2 pairs (4 examples) per item:

| Arm | Accuracy | Settled at 0.9 | Precision at 0.9 | p50 ms |
|---|---|---|---|---|
| none | 0.856 | 90.0% | 88.9% | 334 |
| random | 0.887 | 80.6% | 91.5% | 876 |
| retrieved (BM25 nearest pairs) | 0.869 | 86.3% | 89.9% | 870 |

Neither arm reaches the ~5-point bar, and both cost 2.6× latency. **Decision:** prompt examples deleted. Labels keep working as exact overrides, head training data and tuning data.

## E6. The expanded equipment set

The current set (`bench/equipment_jev.py`) has 34 cases but only **10 distinct report texts** and 2 semantic facts, so it can't separate designs or judges by a few cases.

**Expand to:**
- **at least 40 distinct reports,** written fresh, EN/PT, not translations of each other only;
- **at least 4 facts** (temperature, leak, noise, display fault);
- **traps:** near-identical machines (AX-410 / AX-410B / AX-401B), firmware and pressure boundaries, contradictions between the operator and the sensors, other units, past and resolved faults, hypotheticals, missing observations, and no applicable procedure;
- **two shapes:** shared claims across cards, and cards with unique claims (over budget, so retrieval matters).

**Arms:** whole catalogue in one question; `select` with titles removed; facts + cases. Each at `max_claims` within and over budget.

**Score, per decision:** correct, wrong action (wrong or ineligible procedure), correct `needs_input`, abstained, settled with no model call, distinct claims judged, and p50 / p95 latency including the decisions that needed no model. This is the "beating Jev" scoreboard (spec §8). Jev arms run only with TypeSafe's consent.

This becomes the gate for changes to procedure selection.

## E7. Question-first cache layout for one question over many inputs

**Measured cost split** (default judge, one check with a 45-token question):

| Input | Prompt + input + question | Input cached (question only) |
|---|---|---|
| 23 tokens | 405 ms | 170 ms |
| 213 tokens | 1,147 ms | 199 ms |

For one fixed question over a stream of short inputs (`ss check --each`, an inline loop), the question is most of the work. Caching system + question and decoding only each new input (Jeff's layout) could be about 2× faster at 20 tokens, and about 15% at 260.

**Why it isn't default:**
- switches ask several questions about one input, so input-first stays the main layout;
- a second layout is a second prompt identity, and the question-before-input order may judge differently.

**Setup:** an adapter layout `question-first`, used only by single-question calls over a stream. Measure the fresh and selection suites, and latency at 20 and 261 tokens.

**Decision:** adopt only if accuracy and precision at 0.9 hold within noise on both suites and the speed-up at 20 tokens is at least 1.5×. Otherwise drop it.

## E8. Activated LoRA specialists on the judge

**Question:** when a real judgment has hundreds of labels and the plain judge misses its target on held-out families, does an activated LoRA close the gap at an acceptable cost?

**Why aLoRA and not LoRA:**
- an activated LoRA (Greenewald et al., NeurIPS 2025) adapts the attention Q/K/V weights only from an invocation sequence onwards. The cached base input stays valid, and a specialist costs only its question;
- a plain LoRA changes every token's computation, so each switch of adapter would re-read the input (about 1.1 s per 261 tokens), which defeats the input cache.

**What the sources establish:**
- PEFT trains it (`alora_invocation_tokens`; rank about 32 where LoRA uses 8; it can't be merged by definition);
- the paper reports parity with LoRA (0.0% median, 0.6% mean difference on 1B–8B models) on classification-like tasks;
- llama.cpp reads the invocation tokens from GGUF, and its server runs the uncached prefix with the adapter scale at 0.

**Setup:**
1. Pick one real judgment that has at least 200 labels and where the judge's validated policy is `not met`.
2. Train with `ss export` rows, which carry runtime token ids. The question suffix's leading tokens are the invocation sequence.
3. Train PEFT aLoRA at rank 32 on the default judge's weights. Labels come from real data and Gemma 4 E4B, never Jev.
4. Convert to GGUF.

**Verify, in order:**
1. Gemma 4 E2B converts and runs: per-layer embeddings and shared KV layers are the risk.
2. Reusing the cached base input gives byte-identical scores to a fresh read with the adapter applied only to the question.
3. Questions with different adapters cost one decode per adapter group; measure the overhead.
4. Cost per question with the adapter vs without.
5. Accuracy and precision at 0.9 vs the bare judge and `--precise`, on the same split, with families held out.

**Decision:** adopt as `specialize: {adapter: file}` (bound to `meaning_rev` and the judge's sha, tuned and admitted like the other tiers) only if it beats the probe by about 5 points or more, at under about 10% added latency per question. Otherwise the probe remains the top in-library tier.

## E9. Readout adaptation on real traffic

**Status:** measured on synthetic suites and not offered (spec §5.4).

**Question:** on real traffic, where near-duplicates and translations of labelled inputs recur, does a per-judgment readout (hidden-state probe or Platt calibration) beat the plain judge on held-out inputs? On the fresh suite it matched the judge with whole families held out (0.844 / 0.875 vs 0.856), and gained only when a translated twin was in training (0.931).

**Setup:** E2's data, split chronologically, with groups for known near-duplicates. Compare against the plain judge with the validated lock: correct answers at matched precision. `bench/probe.py` has the extraction.

**Decision:** offer it only if it beats the validated judge on held-out data at equal precision by about 5 points or more.

