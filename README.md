# semantic-switch

Ask a language question about a case inside ordinary Python code and get a typed answer or an explicit non-answer.

```
input evidence + question (+ your rules) → model scores the allowed answers → accepted answer, or uncertain / needs_input / failed
```

The model never chooses outside the answers you allow, exact rules in code are authoritative, and a judgment that is not confident enough says so instead of guessing.

**What it is for:** classification, routing, extraction and policy reading over bounded evidence: a message, a report, a few reference excerpts. **What it is not for:** arithmetic, logic puzzles or multi-step reasoning. The judge reads one answer token without reasoning and stays confident while wrong there; on a public benchmark subset (S1MB, first 25 cases per task) the default judge's answers settled at 0.9 were right about 81% of the time overall and far less on math and logic tasks.

The 0.9 cutoff is an acceptance setting, not a promised precision. `ss lock` fits thresholds on your labelled development data and reports the precision and coverage it observed there; a different judge, task mix or procedure needs its own evaluation. The precise judge (E4B) settles more answers at similar precision; it is an option to measure, not a cure for confident errors.

## Install

```bash
pip install -e '.[local]'   # Python ≥ 3.12; local judges: llama-cpp-python and Hugging Face downloads
pip install -e '.[hosted]'  # hosted judges only (httpx); no native packages
```

The base package authors and runs judgments without native packages; a local judge without the `local` extra fails before any download and names the extra. The default local judge (Gemma 4 E2B, Q4_0 GGUF) downloads from Hugging Face on first use.

## One question

```python
import semantic_switch as ss

message = "Charged twice AGAIN. Fix it by Friday or my lawyer files suit."
r = ss.check(message, "Is the customer threatening legal action against us?",
             excludes=["merely mentioning a lawyer"])
r.status, r.value          # ("ok", True)
```

A check written as a question is answered yes/no; a claim (`"The customer threatens legal action."`) is answered true/false. Both give `True`/`False`. Open questions belong to `pick`: `Check("Which department should handle this?")` would be asked as yes/no.

Write a claim so it pins down who (the input's author or someone else), when (current, past or planned) and certainty (stated, negated, hypothetical or unstated), and put the readings you do not mean in `excludes`. Measured judges accepted wrong answers where the claim left them open: "My son wants a person on the line; I'm fine with the bot." was accepted as the customer asking for a human, "My order arrived; my sister's didn't" as the customer's order not arriving, and "My heating works; it's my neighbour's that doesn't." as the tenant's heating failing (`bench/results/question-form-*.json`). "The customer, not someone they mention, asks for a human agent now" leaves less to guess.

## Non-answers are values

```python
match r.status:
    case "ok":          act(r.value)
    case "uncertain":   review(message)      # the model was not confident enough
    case "needs_input": ask_for_more()       # empty evidence, or the evidence does not settle the question
    case "failed":      log(r.error)         # invalid input or backend failure
```

`if r:` raises `TypeError`, so an uncertain result never passes as true. `needs_input` means the evidence is insufficient; it does not promise that one particular field would resolve it.

## Predictions beside accepted answers

```python
r = ss.pick(message, "What should support do next?", actions, allow_none=True)
r.status, r.value      # ("uncertain", None): not accepted at the cutoff
r.prediction           # Prediction(kind="value", key="refund", value="refund", score=0.62)

if r.status == "ok":                                         # accepted dispatch
    HANDLERS[r.value](message)
elif r.prediction and r.prediction.kind == "value":          # direct dispatch: you own its error rate
    HANDLERS[r.prediction.key](message)
```

`.prediction` is the strongest scored answer: `kind` is `value`, `none` or `unstated`, `key` the authored key, `value` the original option object, `score` its reported lower bound; ties go to the earlier declared option. Every scored Check, Pick and Rate carries one, including uncertain and low-mass results; failed, missing-input and unexamined judgments carry none. Select has no set prediction, only `r.members[key].prediction`; saved switch facts carry `fact.prediction`, and a switch never takes a branch on a prediction. Acceptance (`min_score`, lock thresholds) decides only `.value` and `.status`: changing it sends the same requests and leaves scores and predictions unchanged. A hosted prediction is the strongest reported answer, not proof that unreported mass could not change the ranking. Measure raw correctness of predictions before dispatching on them directly.

Keep authoritative numbers out of the model. Compute a percentage in code and pass it as a `number` field so a rule predicate such as `refund_percent <= 50` is exact, and make an override its own Check in the first rule so it is evaluated before the rest. Exact arithmetic still depends on correct semantic premises: a rule that combines an exact percentage with a judged fact is only as right as that fact. `examples/predictions.py` shows both forms.

## Classification: one, any, a level, several dimensions

Define labels once and reuse them. The key is the stable identifier your code uses; the description defines the meaning; `excludes` names the readings that do not count.

```python
topics = {
    "billing":  ss.Label("charges, invoices, payments or refunds", excludes=["a question about future prices"]),
    "delivery": "an order that is late, missing or damaged",
    "account":  "login, profile or data access",
}                                   # or ss.labels("taxonomy/support.yaml")

ss.pick(message, "What is the message mainly about?", topics, allow_none=True)   # exactly one, or none
ss.select(message, "Which issues does the customer raise?", topics)               # any subset
ss.rate(message, "How urgent is this?", ["can wait", "today", "immediately"])    # one ordered level

r = ss.ask(message, {"topic": ss.Pick("What is the message mainly about?", topics, allow_none=True),
                     "issues": ss.Select("Which issues does the customer raise?", topics),
                     "urgency": ss.Rate("How urgent is this?", ["can wait", "today", "immediately"])})
```

- **Pick** asks one question listing the labels; the answer alphabet allows 19 labels plus `none`. Above that, narrow the candidates first.
- **Select** asks one yes/no membership question per label, each carrying the full instruction and that label's definition, over one read of the evidence. `r["issues"].value` is the tuple of accepted labels; `r["issues"].members` holds each label's own result. `complete` means every eligible label was examined; `settled` means every examined label resolved. Status is `ok` when everything resolved with members, `none` when everything resolved empty, and otherwise the outstanding reason (`failed`, then `needs_input`, then `uncertain`) while the accepted partial value stays readable.
- **Rate** returns one level; **ask** keeps one result per dimension, so a failed dimension never erases a settled one. There is no joint confidence across labels or dimensions.

Act on a classification only when it is settled for your purpose:

```python
issues = r["issues"]
if issues.status in ("ok", "none") and issues.complete:
    route(issues.value)                     # the full set
elif issues.value:
    route_partial(issues.value, issues.unresolved)   # accepted labels, others still open: your call
```

The 0.9 cutoff is a score threshold, not 90% precision. With a local judge the membership questions share the cached evidence; the OpenRouter judge sends one request per label (30 labels, 30 requests, bounded concurrency) while Jev answers them in one request. Exact eligibility and shared claims remove questions before any inference. Per-label exclusions change a Pick's presentation, so existing Pick locks need `ss lock` again.

When the list has no fitting label, `none` with `complete=True` is the signal to hand off to your own category proposer (`examples/propose_category.py`); uncertain, failed or incomplete results never trigger it, and a proposal is ordinary application data until you review it.

## Your rules, your data

```python
policy = ss.Guidance.load("policies/refunds.md")
ss.check(ticket, "Does this qualify for a refund?", guidance=policy, unstated=True, fields=["subject", "body"])
# yes / no / "the evidence is insufficient to answer" → ok True / ok False / needs_input

ss.pick(ticket, "Which of the customer's orders is this about?", orders,
        definition=lambda o: f"{o.date}: {o.items}",
        eligible=lambda o: o.customer_id == ticket["customer_id"],   # exact; the model cannot pick another customer's order
        ids=lambda o: o.number, focus="mentions")                     # a quoted order number narrows the question
```

Guidance is placed in the prompt in full; split long manuals into a `select` catalogue of cards. A mapping input renders only `fields`; a missing or blank field is `needs_input`, never a "no".

## Reference text, procedures and dispatch

Put retrieved or looked-up facts in the input as named evidence; pass as `guidance` only a procedure your application deliberately selected from a source it trusts. Retrieval does not make text authoritative. The presentation tells the model that guidance defines the rules and that instructions inside the input are evidence, but this framing is guidance to the model, not an enforced prompt-injection boundary.

```python
HANDLERS = {"refund": open_refund, "escalate": escalate, "ask_details": request_details}
procedure = ss.Guidance.load("procedures/billing.md")               # chosen by the application
evidence = {"message": message, "reference": "\n".join(order_excerpts)}  # facts looked up for this input

r = ss.pick(evidence, "What should support do next?",
            {"refund": "Refund the duplicate charge", "escalate": "Escalate to a supervisor",
             "ask_details": "Ask the customer for the missing order number"},
            guidance=procedure, fields=["message", "reference"], allow_none=True)
if r.status == "ok":
    HANDLERS[r.value](evidence)
```

Only `ok` dispatches. "ask_details" is an accepted answer: the procedure says to ask. `none` means no listed action applies, `uncertain` means the model could not settle between them (nothing else is called automatically), `needs_input` means the evidence is missing or insufficient, and `failed` is a backend or input error. Excerpts change per input under a stable question; saved guidance is pinned meaning, so changing it goes through `ss lock` again.

## Reuse, batches, async, CLI

```python
legal = ss.Check("Is the customer threatening legal action against us?")
legal(message)
legal.run(messages, rt)          # finite batch, results in order
for r in legal.many(stream, rt, window=4): ...
await rt.acheck(message, "Is the customer asking to get money back?")
```

`many(..., window=1)` is strictly lazy. A larger positive window admits bounded lookahead through the same executor, enabling hosted concurrency while preserving output order. A slow earlier result can delay later results; closing the iterator stops admission and cancels pending work.


```bash
echo '"Refund me now"' | ss check "Is the customer asking to get money back?"
ss rate "How urgent is this?" --levels "can wait,today,immediately" --each < tickets.jsonl
```

Exit codes follow the worst status: ok/none 0, uncertain/needs_input 2, failed 3, configuration error 5; a single `check` exits 1 for an accepted false.

## Choosing the judge

```python
local  = ss.Runtime()                                                         # Gemma 4 E2B, local
strong = ss.Runtime(ss.judge(ss.PRECISE_JUDGE, device="gpu"))                 # Gemma 4 E4B
hosted = ss.Runtime(ss.judge("openrouter:google/gemma-4-26b-a4b-it"),        # reads OPENROUTER_API_KEY
                    own_judges=True)

with hosted as rt:
    rt.check(message, "Is the customer asking to get money back?")
```

`ss.judge` is the one judge factory; it resolves its source once through the judge driver registry: a GGUF path or `repo:file` (or `gguf:PATH`; native settings such as `device`, `threads`, `repack`), `openrouter:MODEL` (`provider`, `key`, `codec`, `profile`, `timeout`, `batch_inputs`; reads `OPENROUTER_API_KEY`) or a scheme registered by a plugin through the `semantic_switch.judges` entry-point group or `semantic_switch.drivers.register`; an unknown scheme and settings that do not apply to the driver are refused before any request. A hosted driver combines one shared engine with a provider (endpoint, authentication, routing and response envelope), a codec (question encoding and score decoding) and a model profile (question kinds and widths, questions per request, request size, truncation, probability precision and normalization). Built-in profiles map known models to their codec: `google/gemma-4-31b-it` and other listed chat models answer one question per request through chat log probabilities (top 20, `require_parameters`, no fallbacks); `typesafe/jev-1.13` and `openai/gpt-6-luna-decisions` answer every question of an input in one System One request under empirical two-decimal profiles pinned to recorded response fixtures. Any other model needs `codec="chat"` or an explicit `profile`; System One needs a profile declaring its serialization precision, and a returned distribution whose mass differs from one by more than `width × 0.5 × 10^-decimals` fails instead of being renormalized. `semantic_switch.testing.drivers` is the conformance kit for new providers, codecs and profiles. A provider that rejects the request shape fails with an error naming it; pin another with `provider`. `timeout` is each request unit's end-to-end budget, not a fresh per-attempt timeout: it starts before admission and covers admission and active-slot queueing, request encoding, every transport attempt and response-body read, retry waits (408/429/5xx and connection errors, at most 5 attempts) and decoding. Admission bounds how many units hold work, not how long one may queue; each attempt receives only the remaining budget, and a response completing after the deadline is rejected, not retried. A unit's request body is encoded once after admission and reused byte-for-byte by every retry, so at most `2 × batch_inputs` bodies exist; at most `batch_inputs` attempts are on the wire, and a unit releases its active slot during backoff and decoding. A `Retry-After` from the provider holds every unit of that judge until it passes; other retries back off privately. Request receipts split elapsed time into queue, service and retry seconds that sum to the total. An encoding, transport or decoding failure fails only its unit; the other units and their receipts are kept. Each hosted judge runs one HTTPX `AsyncClient` on its own AnyIO portal; `close()` (or `await judge.aclose()`) cancels queued and active units, then releases an owned transport once, and repeating it is harmless. A supplied `transport` is an async `send(url, body_bytes, remaining_seconds) -> Reply`, borrowed unless `own_transport=True`; an owned transport's `close` hook, synchronous or async, is optional. A runtime owns and closes a judge it constructs from a source (`ss.Runtime("openrouter:MODEL")`); a judge you supply is borrowed unless `own_judges=True`. `Runtime(bindings={"judge": ..., "hosted": ...})` binds several model roles; a runtime given only roles other than `judge` binds no default judge. Leaving a `with Runtime(...)` block by an exception cancels its in-flight work before closing; `close()` waits for it. Cancelling `await rt.aask(...)` cancels its hosted requests and starts no further rounds; a native decode finishes its current batch first. Locking a hosted judge requires a pinned provider; the lock records the protocol, the requested provider and the returned model and provider. The CLI uses the same registry and validator: `ss --judge openrouter:MODEL --provider PROVIDER --setting codec=chat check ...`; `--setting KEY=VALUE` reaches any driver setting, and local flags (`--device`, `--repack/--no-repack`, ...) apply only when given. Local judges use serial decoding with a separate KV stream per sequence. Three caps bound one request: `max_input_tokens` (evidence, default 1024), `max_prefix_tokens` (chat template plus guidance, default 1024) and the fixed 1536-token question suffix (framing included). Their sum is the request window (3584 by default) and the only useful native window. `n_ctx` is an advanced smaller aggregate limit for memory-constrained workloads; a larger value is refused before any model load, naming the useful window and the caps that form it. To score longer requests, raise the caps themselves. Overlong requests fail without truncation or cache mutation. Physical context allocation rounds each stream to native padding and multiplies by `prefix_slots + 2`; `prefix_slots=4` therefore allocates six streams, including full SWA storage. Reducing slots saves memory while preserving the request window. Runtime info reports requested and realized capacities. Locks and native exports bind the realized execution, including per-sequence capacity, KV layout and scheduler revision; old unified-KV locks require explicit `ss lock` regeneration. On the measured Gemma4/Q8_0 CPU and Vulkan base scope, cold, warm, resident-prefix and whole-sequence reference scores match exactly. Runtime adapters are resident-history stable on that Vulkan build but still differ from whole-sequence reference scoring; see [measurements](docs/ga-measurements.md). Grouped decoding remains a backend experiment because the measured grouping changed probabilities and did not improve throughput. The application code is the same for every judge. Answer quality, confidence and latency are not: measure them on your inputs and re-run `ss lock` after changing judges. Hosted spending is limited on the provider's API key. The demos run from the repository root as packages (`python -m examples.quickstart`, `python -m examples.printer_snapshot`, `python -m examples.support_snapshot`); they share one judge/runtime helper, `examples/environment.py`, which selects OpenRouter when `OPENROUTER_MODEL` is set and the local judge otherwise, and the packaged example definitions in `examples/data/`.

## Saved switches: judgments + exact rules + decisions

When a decision combines several judgments with business data, write it down:

```yaml
name: support_route
guidance:
  refunds: { file: policies/refunds.md }
fields:
  subject:        { type: text, required: true, author: customer }
  body:           { type: text, required: true, author: customer }
  payment.status: { type: enum, values: [pending, settled, failed], source: payments }
reads: [subject, body]
facts:
  legal:   { check: "Is the customer threatening legal action against us?", excludes: [merely mentioning a lawyer] }
  refund:  { check: "Does this qualify for a refund?", guidance: [refunds], unstated: true }
  urgency: { rate: "How urgent is this?", levels: [can wait, today, immediately] }
rules:
  - { id: legal,  when: legal,                                     then: legal_team }
  - { id: refund, when: 'refund and payment.status == "settled"', then: refund_review }
  - { id: urgent, when: 'urgency >= "today"',                      then: priority_queue }
otherwise: normal_queue
```

```python
route = rt.load("switches/support_route.yaml")
r = route(ticket)
r.answer, r.rule, r.facts["refund"].value
print(r.explain())
```

Exact predicates settle first and only the facts a remaining rule depends on are judged. Exact rules prevent the specific invalid actions they encode, not every confidently wrong semantic answer. Any inline judgment saves to this form with `.save(path)` and behaves identically. For "which of these procedures apply?", a `select` fact over a reviewed card catalogue judges each shared situation claim once and applies exact eligibility and precedence in code; see `switches/procedures.yaml`. A card negates a fact with `excludes` (three-valued: an unknown fact leaves the card unresolved), so a logical complement never needs its own judged question. When a revision drops a fact, list it under `retired: [name]`: historical label files keep loading, their retired values are ignored, and an unknown or misspelled fact still fails.

## Optional: measure, correct, specialize

```python
route.correct(ticket, fact="refund", value=True, actor=ss.Actor("human"))   # exact override for this input
route.correct(ticket, rule="refund", actor=ss.Actor("human"))              # or the rule this input should reach (ss correct --rule)
```

```bash
ss label support_route new_tickets.jsonl --to test --actor human   # label real inputs; supplied values are reviewed: Enter confirms, s skips, an answer replaces
ss lock support_route                                # fit per-judgment thresholds on development labels, check them on held-out labels, pin the judge and execution
ss test support_route                                # raw correctness of predictions, then accepted precision and coverage on unseen labels (intervals where sampling supports them)
ss --verbose run support_route [--raw] < ticket.json # the decision with its explanation (an explain field with --json)
ss export support_route --native --labels reviewed.jsonl   # exact-token rows from a label file; never copied into corrections or tests
ss prune switches/                                   # remove generation files that no current manifest in the directory names
```

The same operations are Python functions in `semantic_switch.tools`; each CLI command above calls them:

```python
from semantic_switch import tools

with ss.Runtime() as rt:
    route = ss.Check("Is the customer asking to get money back?").save("switches/refund.yaml", rt=rt)
    tools.export(rt, "switches/refund.yaml", native=True)   # {"path", "export_id", "rows", "situations", "complete_situations", "dataset_id", "split", "skipped_rule"}
    report = tools.lock(rt, "switches/refund.yaml")          # {"lock", "id", "tests", "written", "heads", "policies", "targets", "evidence", "warnings", ...}
    tested = tools.run_test(rt, "switches/refund.yaml")      # {"fresh", "taught", "used", "facts", "fact_populations", "dataset_id", "candidate_id", "warnings", ...}
    print(tools.explain_test(tested))                        # the text `ss test` prints
```

- `tools.lock(rt, path, dry_run=False, *, head_bundles=None) -> dict` fits and checks thresholds and publishes `NAME.lock.json` unless `dry_run`.
- `tools.export(rt, path, to=None, split="tune", *, native=False, task_family=None, seed=0, scope=None, labels=None) -> dict` publishes an export manifest and its payloads.
- `tools.run_test(rt, path, data=None) -> dict` evaluates the switch on its labelled data, or on `data` (a label file) through a content-addressed sibling manifest.
- `tools.explain_test(report) -> str` renders a `run_test` report.

The optional packages expose their entry points the same way: `from semantic_trainer import train_lexical` (needs `semantic-trainer[lexical]`) and `from semantic_learn import collect`; importing either package loads no fitting or inference dependency. Specialization goes through `semantic_switch.training.specialization_request` and `specialize`.

An unlocked judgment still works; a lock pins configuration and fitted thresholds, not domain truth. Label records name a rule with `"rule"`. Held-out authored labels are development evidence, not an independent domain evaluation. `ss analyze-head SWITCH FACT --head BUNDLE` analyzes a supplied lexical head; `ss specialize` runs the trainer pipeline (`ss specialize --help`).

Task specialization (optional `packages/semantic-trainer`) fits LoRA on the default student and deploys one merged, quantized GGUF: `semantic-learn collect` (optional `packages/semantic-learn`) labels situations with a teacher whose terms permit training use, `ss label … --to reviewed.jsonl` reviews them, `semantic-trainer prepare` assigns family roles and creates the job, `semantic-trainer lora` fits under model_select monitoring and retains the zero adapter and the two best evaluated points, and `semantic-trainer select` merges each into the F16 base, quantizes it, selects on model_select families and publishes one self-contained fitted lock (GGUF, bundled task definition and guidance, frozen policies and execution); `semantic-trainer check` evaluates it independently. The specialist is adopted only by your explicit choice:

```python
decide = ss.Runtime().load("run/deployment")   # fitted lock: weights, task, guidance, corrections, policies, execution
judge = ss.judge("run/deployment")             # model only, with its verified provenance and no task policy
```

`ss.load(path)` and `Runtime.load(path)` are the one loader. An authored definition loads under its draft and lock rules; a frozen lock (`X.lock.json`, a fitted lock directory or its `receipt.json`, or a `current` pointer) loads exactly what it pinned, without the source files, the trainer or torch, so it can be moved. `draft` and `allow_model_change` are refused for a frozen lock, an altered payload fails verification, and a fitted lock refuses a runtime that binds a different model to its role. Publishing a fitted lock creates a new immutable entry; `semantic_switch.training.point(POINTER, LOCK, reason=..., expected=INCUMBENT_ID)` moves a `current` pointer by compare-and-update (`expected=None` for a first adoption; a different incumbent fails unchanged), `current(POINTER)` resolves it and `rollback(POINTER, reason=..., expected=INCUMBENT_ID)` points it back at a previously verified lock without touching files (`ss promote` / `ss rollback` from the shell). A name without a path or suffix loads `switches/NAME.yaml` when it exists, otherwise `NAME.yaml`; the CLI passes its argument to this same resolver.

```bash
ss compare --baseline current.json --candidate run/deployment --examples labels.jsonl --evidence labels.dataset.json --margin 0.02 --to compare/
ss promote current.json run/deployment --reason "round 2" --decision compare/decision.json
ss rollback current.json --reason "regression reported" --expected CANDIDATE_ENTRY_ID
```

`ss compare` scores each frozen lock over the check-role inputs in one bounded operation per arm and requested-target set, journaled per input and resumable, and reports the paired difference with a decision; the same comparison publishes `decision.json`, the promotion decision. `--claim non_inferiority --benefit latency|cost`, `--required-precision`, `--tolerate failed|unresolved|forbidden=N`, `--forbidden ADDRESS=KEY` and `--recall ADDRESS=CLASS:raw|operational=MIN` complete the predeclared claim. A recall floor protects one semantic class: operational recall counts withheld answers as misses, a class without references is unassessed rather than passed, and an unmet or unassessed floor blocks automatic adoption without changing fitting. `ss promote` moves the pointer only to the candidate of a supported decision and only while the pointer still serves the incumbent that decision compared against; a stale incumbent changes nothing. The decision is an adoption claim only on fresh independent evidence with disjoint lineage and every scored target inside the declared reference scope (`--reference`, default verified and computed labels; authored labels count only when declared); otherwise it is descriptive and says why. None and unstated are scored as answers: a confident none is accepted, a correct unstated stays needs_input. `--robustness MANIFEST` adds a descriptive per-arm diagnostic over authored option permutations/renames, meaning-preserving formatting and contrast siblings, and `recompute_comparison` rebuilds a published comparison from its stored rows without inference.

A fitted lock declares one representation. `merged` (the default) is one quantized GGUF with the adapter merged in; it serves a whole-task specialization and is not many-adapter serving. `runtime-adapter` (`semantic-trainer select --representation runtime-adapter`) publishes the pipeline-built quantized base and a converted LoRA GGUF, hashed independently, and applies the adapter at scale 1 on one resident base: `Model(base, lora="adapter.gguf")`. Roles naming the same base and native settings share one loaded model and context; their adapters share a bounded pool (`max_adapters`), and only unreferenced adapters are evicted. Representations are never interchanged: a merged binding loads no adapter and a runtime-adapter binding never falls back to merged weights. `bench/adapter_profile.py` (an explicitly invoked GPU job) measures the pinned llama.cpp build and Vulkan device: switching is exact against a matched no-switch control and adapters fit 128 MiB, but complete scoring calls over its probe workload exceed the declared 1,000 ms requirement with or without a switch, so runtime adapters remain unsupported for that workload. Cached scores also differ from freshly cleared references by up to 0.061 probability, above the 0.001 tolerance and of undetermined cause; the current profile assesses that cache equivalence separately from switching through one-variable controls (`bench/README.md`, *Adapter profile*); fake lifecycle tests do not establish capability.

Model admission charges one estimate per resident: physical host bytes plus resident bytes per offload device, derived from each GGUF tensor's type and the layer it is placed on (on unified memory, device bytes are part of physical memory and are not counted twice). An adapter is charged before native allocation and its reported buffers are verified afterwards; an allocation exceeding its charge is freed and reported, never absorbed by enlarging the job. A job you pass with `Runtime(..., job=...)` must already cover the same estimate (`semantic_switch.training.estimate`).

Facts answer under the definition's `binding:` (default `judge`; a fact or Select may name another model role, and staged questions and Select members inherit it). A job prepared with `semantic-trainer prepare --scope FACT[/MEMBER]` fits and selects only those targets: other bindings are neither loaded nor scored, and the result is scoped support, not a runnable application. Install it by role, `Runtime(bindings={"refund_model": ss.judge("run/deployment"), "judge": ...}).load("app.yaml")`, or publish a complete application with `semantic-trainer select ... --binding ROLE=JUDGE` for every remaining role; its lock pins those identities and policies, and loading it requires them: `Runtime(bindings={"hosted": ...}).load("run/deployment")`. Selection on the scoped targets says nothing about the whole application, which needs its own check.

A model's adaptation provenance comes only from its fitted lock (or a verified pipeline-built base); a plain `Model(path)` has unknown provenance, and a lock of a specialist excludes every family that adapted, selected or calibrated it.

## Repository

`semantic_switch/` is the library (`semantic_switch.training` is the trainer integration surface: `Engine` adds native scoring, cache invalidation and detokenization to the `Runtime` constructor; `model_job` admits explicit model jobs), `packages/semantic-trainer/` the optional trainer, `bench/` the measured evidence (`bench/README.md`), `experiments/` archived explorations, `docs/invariants.md` the invariants and stage boundaries with their owning modules, `TODO.md` the open work. Tests: `pytest -m "not slow"` (controlled judges) and `pytest -m slow` (real models; one model job per host).

### Scores and acceptance

Allowed-answer probabilities describe the model distribution, not the probability that its answer is correct. Answer mass is the probability assigned to the allowed answer tokens before conditioning; it remains separate from class probabilities. Reports distinguish argmax accuracy, precision among accepted answers, coverage and wrong acceptances per submission, with denominators and family counts. NLL, Brier and reliability bins describe stored complete distributions; missing and staged distributions are explicitly unassessed.

Authored thresholds are uncalibrated unless an acceptance-fit population was used. Checkpoint-selection data remains development evidence. Fixed answer sets prevent malformed labels; three-valued rules propagate unknown facts but cannot detect a confidently wrong fact. `uncertain` and `needs_input` are outcomes, not guarantees of detecting every unanswerable question. Determinism applies to the recorded model, native build and execution shape; changed quantization or batching requires a measured comparison.

Training recipes can opt into class/family effort allocation, authored behavioral slices and development recall requirements. See [trainer usage](packages/semantic-trainer/README.md) for bounded seed/LR qualification and fit-only diagnosis. Data generation and input-view materialization remain explicit separate operations; reusing a corpus does not regenerate it.

Reviewed failures can be recorded with `semantic-learn diagnose REVIEW.json UNIT --reason coverage --reviewer NAME --evidence ARTIFACT --note TEXT --to DIAGNOSES.jsonl`. Reasons route work to materialization, production, review, task interpretation, fit diagnosis or acceptance calibration. These records are review metadata, never training labels. Keep existing fit examples when consuming approved additions through `semantic-learn round`; promotion remains a separate decision.
