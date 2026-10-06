# Semantic Switch

Language-based judgments inside ordinary code: a `switch/case` over meaning, running locally on small open models.

```python
import semantic_switch as ss

ss.check("I was charged twice, please refund one.", "The customer asks for money back.")
# <ok · True · p_true 0.99>
```

## Install

```bash
pip install -e .            # Python ≥ 3.12; builds llama-cpp-python
pip install -e ".[train]"   # optional: scikit-learn + scipy, for lexical heads and ss test --intervals
```

The first call downloads the judge: **Gemma 4 E2B Q4_0** (Apache-2.0, 2.7 GB file, about 3.8 GB of RAM). `--precise` / `ss.PRECISE_JUDGE` uses Gemma 4 E4B QAT: the most precise, 7.5 GB, 2.2× slower on CPU. There is no fast judge: small ones commit wrong answers.

**GPU:** build llama-cpp-python with Vulkan (`CMAKE_ARGS="-DGGML_VULKAN=on" pip install --force-reinstall --no-binary llama-cpp-python llama-cpp-python`). `device="auto"` then offloads everything: checks drop from 324 / 1,009 ms to 147 / 201 ms (20 / 261 tokens) on an Arc B390, with the same answers. With a GPU, `--precise` (248 / 330 ms, about +8 accuracy points) is the recommended judge. `--device gpu` fails if the build has no GPU.

Any GGUF chat model works if its embedded chat template renders and `ss doctor --quality` passes.

## Inline judgments

```python
rt = ss.Runtime()
rt.check(ticket, "The customer threatens legal action against us")       # True / False / uncertain
rt.check(report, "There is a visible hydraulic leak.", unstated=True)    # true / false / needs_input (not stated)
rt.pick(doc, "Which address gets the receipt?", addresses,              # returns your object, unchanged
        describe=lambda a: a.text)
rt.pick(report, "Which machine failed?", machines, ids=lambda m: (m.id, m.aliases), focus="mentions")
rt.rate(idea, "How feasible is this?", ["impossible", "hard", "doable", "easy"])
rt.ask(ticket, {"legal": ss.Check("..."), "urgency": ss.Rate("...", ("low", "high"))})  # one shared read
```

Results print as one line and have no truth value: use `.value` and `.status` (`ok`, `none`, `uncertain`, `needs_input`, `failed`).
- **One acceptance policy everywhere:** an answer settles when its own probability reaches 0.90 (a check is false when p(false) reaches 0.90); anything between is `uncertain`. `min_score=m` (above 0.5, at most 1) sets both sides to `m`. It is a majority threshold, not "take the most likely answer".
- **Judgments are values:** `refund = ss.Check("…")`, then `refund(text)`. `refund.save("switches/refund.yaml", field="message")` writes a teachable switch that answers exactly like the inline call.
- A plain yes/no check turns silence into a confident "false". `unstated=True` asks three ways, so "the input doesn't say" comes back as `needs_input`.
- **Comparative choice is a pick with a criterion:** `rt.pick(message, "Which reply best calms the customer without promising money back?", replies)`. State the basis of "better"; a score is the judge's probability, not a utility. When the goal contradicts the situation, the default judge tends to answer the situation, so use `--precise` (`bench/comparative.py`).
- `focus="mentions"` shows the judge only the candidates the text names exactly (`MX-120` never matches inside `MX-120A`), plus an escape option. If the escape wins, it falls back to the full list.

## A switch in five minutes

`switches/support_route.yaml`:

```yaml
name: support_route
fields:
  message: { type: text, required: true, author: customer }
reads: [message]
facts:
  refund:
    check: The customer asks to receive money back for a charge.
cases:
  billing: refund
otherwise: support
```

```python
route = rt.load("switches/support_route.yaml")
r = route({"message": "Please give me my money back for May."})
print(r)              # billing  (case billing)
print(r.explain())    # which case fired and why
```

**Rules can be exact:** `mentions(report, "MX-120A")`, `payment.status == "settled"`, `exists(field)`. Exact rules run before any model.

**No cases:** a switch without `cases` is a saved, teachable set of judgments, and calling it returns the facts (`switches/refund.yaml`).

**Procedures:** a `select` fact picks the applicable procedure cards from a catalog you write (`switches/procedures.yaml`, `procedures.cards.yaml`). It is cases over shared facts:
- exact `requires` decides eligibility (authoritative alarm codes: `mentions(active_alarms, "E17")`);
- `applies_when` and `excludes` are claims about the situation: plain strings or `{fact: name}`. Identical claims are judged once, and titles never reach the model;
- an exclusion the input doesn't mention returns `needs_input`;
- `supersedes` is one order-free pass; cycles are rejected at load;
- over `max_claims`, cards are ranked (aliases, then BM25) and the rest stay `unexamined`, so the result is never `none` by omission;
- the card's payload (instructions) is returned, never shown to the model.

## Teaching

```python
route.correct({"message": "Reverse that payment, please."}, fact="refund", value=True)
```

1. **`correct()`** fixes that fact for that exact input: immediately on a draft, after the next `ss lock` on a locked switch. It's kept in `support_route.examples.jsonl`.
2. **`ss label support_route new.jsonl --to test`** labels separate inputs for tuning and checking.
3. **`ss specialize support_route refund`** runs your labelled inputs through the real executor in group-held-out folds, with and without a lexical head (a microsecond classifier that skips the judge on recurring wording). It reports judge calls avoided, correct answers gained, wrong answers avoided and introduced, and warm latency. It also reports how well the judge ranks the claim (AUC) and what that means: a misplaced boundary, or a claim the judge can't separate (reword it or use `--precise`). It changes nothing.
4. **Add `specialize: lexical`** to the fact if admitted.
5. **`ss lock support_route`** trains heads on corrections plus the test file's tune half, chooses each judgment's policy there, and **validates it on the untouched check half**. Each side is reported as met (with held-out precision and interval), not met (that side abstains), or untuned (too few labels). It's the only step that trains; `--dry-run` shows the result without writing. Put translations and paraphrases in one `group`, or the check half leaks.
6. **The call is unchanged.** `r.facts["refund"].source` says `example`, `classifier` or `judge`.
7. **`ss test support_route`** reports taught, trained-on and unseen inputs separately; only unseen measures generalisation.

Inline calls don't learn; teaching is a saved-switch feature. Each step has one effect: a correction fixes one input, labels are data, `ss lock` changes when judgments answer, a lexical head adds a cheaper predictor, and only `--judge` changes the judge (spec §5.5).

## CLI

```bash
echo "I want a refund" | ss check "The customer asks for money back."   # exit 0 true · 1 false · 2 uncertain/needs input · 3 failed
ss check --unstated "There is a visible leak." < report.txt
ss run support_route --each [--batch 16] < messages.jsonl > routed.jsonl  # JSON when piped, text on a terminal
ss explain support_route < message.json                                 # --json for machine-readable
ss lock support_route --dry-run
ss export support_route                                                 # training rows with runtime token ids
ss test support_route --intervals                                       # accuracy on support_route.test.jsonl, in sentences
ss --json doctor --quality                                              # backend conformance + the 240-item quality probe
ss bench support_route
```

Global options (before the command): `--judge`, `--precise`, `--device auto|cpu|gpu`, `--threads`, `--json`, `--verbose`. Exit codes: 0 ok / none, 1 false (`check`), 2 uncertain / needs input, 3 failed, 4 a record failed (`--each`), 5 configuration error.

## Measured (16-core laptop CPU, 16 threads; the 240-item selection suite of minimal pairs in English and Portuguese, plus 24 picks; it chose the default judge and prompt, so the fresh suite below is the honest number)

| Judge | Accuracy | Settled at 0.9 (precision) | Picks | ms per check |
|---|---|---|---|---|
| Gemma 4 E2B Q4_0 (default) | 0.96 | 95% (98.2%) | 0.96 | ~230–290 |
| Gemma 4 E4B QAT (`--precise`) | 0.975 | 95% (98.7%) | 0.96 | ~570 |
| Qwen3-0.6B Q4_K_M (not offered) | 0.74 | 14% (97%) | 0.83 | ~100 |
| Qwen3-1.7B Q8_0 (old default) | 0.84 | 97% (**84.5%**) | 0.92 | ~290 |

**Other measurements:**
- **Near-identical ids.** Gemma 4 E2B tells MX-120 from MX-120A at 13/14, and was never confidently wrong.
- **Procedure selection.** The bundled catalog gets 8/10 situations right, and both misses were abstentions, not wrong procedures.
- **Quantisation.** Gemma 4 barely changes at 4-bit: 5–7 of 240 answers flip, against 26–34 for Qwen.

**On a fresh suite** (8 new claims in new domains, written after the default was chosen, `bench/fresh.py`) the numbers drop:

| Judge | Accuracy | Settled at 0.9 (precision) |
|---|---|---|
| Gemma 4 E2B (default) | 0.86 | 90% (88.9%) |
| Gemma 4 E4B | 0.93 | 95% (94.7%) |
| Qwen3.5-2B | 0.84 | 66% (95.3%) |
| Qwen3-1.7B | 0.79 | 93% (83.1%) |

No small judge is 95% precise untuned on unseen domains. For precision-critical switches, label a few dozen inputs and run `ss lock`, which tunes thresholds to your target on your data. Then run `ss test`.

**Lexical head on topic routing** (`bench/topical.py`): at 30 labels per class it cut judge calls by a quarter with no added errors; at 100 labels it cut latency 60% but added 3 wrong answers in 80, which is why `ss specialize` reports errors introduced. An embedding head and prompt examples were measured and removed (`CHANGELOG.md`).

`bench/` holds the judge ladder, identifier, procedure, unstated and learning-curve benchmarks; `semantic-switch.md` is the full specification.
