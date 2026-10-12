# semantic-switch

Ask a language question about a case inside ordinary Python code and get a typed answer or an explicit non-answer.

```
case evidence + question (+ your rules) → model scores the allowed answers → accepted answer, or uncertain / needs_input / failed
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

Write a claim so it pins down who (the case author or someone else), when (current, past or planned) and certainty (stated, negated, hypothetical or unstated), and put the readings you do not mean in `excludes`. Measured judges accepted wrong answers where the claim left them open: "My son wants a person on the line; I'm fine with the bot." was accepted as the customer asking for a human, "My order arrived; my sister's didn't" as the customer's order not arriving, and "My heating works; it's my neighbour's that doesn't." as the tenant's heating failing (`bench/results/question-form-*.json`). "The customer, not someone they mention, asks for a human agent now" leaves less to guess.

## Non-answers are values

```python
match r.status:
    case "ok":          act(r.value)
    case "uncertain":   review(message)      # the model was not confident enough
    case "needs_input": ask_for_more()       # empty evidence, or the case does not settle the question
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

Keep authoritative numbers out of the model. Compute a percentage in code and pass it as a `number` field so a case predicate such as `refund_percent <= 50` is exact, and make an override its own Check in the first case so it is evaluated before the rest. Exact arithmetic still depends on correct semantic premises: a case that combines an exact percentage with a judged fact is only as right as that fact. `examples/predictions.py` shows both forms.

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
case = {"message": message, "reference": "\n".join(order_excerpts)} # facts looked up for this case

r = ss.pick(case, "What should support do next?",
            {"refund": "Refund the duplicate charge", "escalate": "Escalate to a supervisor",
             "ask_details": "Ask the customer for the missing order number"},
            guidance=procedure, fields=["message", "reference"], allow_none=True)
if r.status == "ok":
    HANDLERS[r.value](case)
```

Only `ok` dispatches. "ask_details" is an accepted answer: the procedure says to ask. `none` means no listed action applies, `uncertain` means the model could not settle between them (nothing else is called automatically), `needs_input` means the evidence is missing or insufficient, and `failed` is a backend or input error. Excerpts change per case under a stable question; saved guidance is pinned meaning, so changing it goes through `ss lock` again.

## Reuse, batches, async, CLI

```python
legal = ss.Check("Is the customer threatening legal action against us?")
legal(message)
legal.run(messages, rt)          # finite batch, results in order
for r in legal.many(stream, rt): ...
await rt.acheck(message, "Is the customer asking to get money back?")
```

```bash
echo '"Refund me now"' | ss check "Is the customer asking to get money back?"
ss rate "How urgent is this?" --levels "can wait,today,immediately" --each < tickets.jsonl
```

Exit codes follow the worst status: ok/none 0, uncertain/needs_input 2, failed 3, configuration error 5; a single `check` exits 1 for an accepted false.

## Choosing the judge

```python
local  = ss.Runtime()                                                                    # Gemma 4 E2B, local
strong = ss.Runtime({"judge": ss.judge_from(ss.PRECISE_JUDGE, device="gpu")})           # Gemma 4 E4B
hosted = ss.Runtime({"judge": ss.judge_from("openrouter:google/gemma-4-26b-a4b-it")},    # reads OPENROUTER_API_KEY
                    own_judges=True)

with hosted as rt:
    rt.check(message, "Is the customer asking to get money back?")
```

`judge_from` accepts a GGUF path or `repo:file` (native settings such as `device`, `threads`, `repack`) or `openrouter:MODEL` (`provider`, `key`, `timeout`, `batch_inputs`; reads `OPENROUTER_API_KEY`); settings that do not apply to the source are refused. The wire protocol is fixed at construction: ordinary models answer one question per request through chat log probabilities (top 20, `require_parameters`, no fallbacks); verified structured models (`openrouter:typesafe/jev-1.13`, `openrouter:openai/gpt-6-luna-decisions`) answer every question of an input in one System One request. A provider that rejects the request shape fails with an error naming it; pin another with `provider`. `timeout` is one total deadline per request unit, starting when it asks for admission and covering queueing, every attempt, response-body reading, retries (408/429/5xx and connection errors, `Retry-After` honoured, at most 5 attempts) and waits; a response completing after it is rejected, not retried. Each hosted judge runs one HTTPX `AsyncClient` on its own AnyIO portal with at most `batch_inputs` requests on the wire and `2 × batch_inputs` admitted; closing the judge cancels queued and active requests. A supplied `transport` is an async `send(url, body, remaining_seconds) -> Reply`, borrowed unless `own_transport=True`. Cancelling `await rt.aask(...)` cancels its hosted requests and starts no further rounds; a native decode finishes its current batch first. Locking a hosted judge requires a pinned provider; the lock records the protocol, the requested provider and the returned model and provider. The CLI uses the same parser: `ss --judge openrouter:MODEL --provider PROVIDER check ...`, and local flags (`--device`, `--repack/--no-repack`, ...) only when given. Local judges decode each input and each question serially; checks, switches, lock fitting and evaluation use that one strategy. Serial decoding is the current strategy, not a guarantee of identical scores: the number of questions in a call and the resident cache history still move scores, and on the CPU two accepted outcomes changed across call sizes (`bench/results/execution-geometry.json`). Locks and native exports record the realized execution (device, tensor and KV types, `n_ctx`, `n_batch`, `n_seq_max`, output rows, SWA/KV layout, `prefix_slots` and the cache/scheduling revision), and a lock refuses a judge whose recorded execution differs; matching settings does not prove invariance across dynamic shapes. Serial decoding costs GPU throughput (about 3× per question on the measured GPU) and is not an accuracy improvement. The application code is the same for every judge. Answer quality, confidence and latency are not: measure them on your cases and re-run `ss lock` after changing judges. Hosted spending is limited on the provider's API key. `examples/quickstart.py` selects OpenRouter when `OPENROUTER_MODEL` is set and the local judge otherwise.

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
cases:
  - { id: legal,  when: legal,                                     then: legal_team }
  - { id: refund, when: 'refund and payment.status == "settled"', then: refund_review }
  - { id: urgent, when: 'urgency >= "today"',                      then: priority_queue }
otherwise: normal_queue
```

```python
route = rt.load("switches/support_route.yaml")
r = route(ticket)
r.answer, r.case, r.facts["refund"].value
print(r.explain())
```

Exact predicates settle first and only the facts a remaining case depends on are judged. Exact rules prevent the specific invalid actions they encode, not every confidently wrong semantic answer. Any inline judgment saves to this form with `.save(path)` and behaves identically. For "which of these procedures apply?", a `select` fact over a reviewed card catalogue judges each shared situation claim once and applies exact eligibility and precedence in code; see `switches/procedures.yaml`.

## Optional: measure, correct, specialize

```python
route.correct(ticket, fact="refund", value=True)     # exact override for this input
```

```bash
ss label support_route new_tickets.jsonl --to test   # label real cases; supplied values are reviewed: Enter confirms, s skips, an answer replaces
ss lock support_route                                # fit per-judgment thresholds on development labels, check them on held-out labels, pin the judge and execution
ss test support_route                                # raw correctness of predictions, then accepted precision and coverage on unseen labels
ss export support_route --native --labels reviewed.jsonl   # exact-token rows from a label file; never copied into corrections or tests
ss prune switches/                                   # remove generation files that no current manifest in the directory names
```

An unlocked judgment still works; a lock pins configuration and fitted thresholds, not domain truth. Held-out authored labels are development evidence, not an independent domain evaluation. `ss specialize SWITCH FACT --head BUNDLE` analyzes a supplied lexical head.

Task specialization (optional `packages/semantic-trainer`) fits LoRA on the default student and deploys one merged, quantized GGUF: `semantic-trainer collect` labels situations with a teacher whose terms permit training use, `ss label … --to reviewed.jsonl` reviews them, `semantic-trainer prepare` assigns family roles and creates the job, `semantic-trainer lora` fits under model_select monitoring and retains the zero adapter and the two best evaluated points, and `semantic-trainer select` merges each into the F16 base, quantizes it, selects on model_select families and publishes one self-contained deployment directory (GGUF, bundled task definition and guidance, lock with the frozen policies, receipt); `semantic-trainer check` evaluates it independently. The specialist is adopted only by your explicit choice:

```python
runtime = Runtime.from_specialization("run/deployment/receipt.json")   # evaluated binding: weights, task, policies, execution
decide = runtime.load()
judge = Model.from_specialization("run/deployment/receipt.json")       # model only, with its verified provenance and no task policy
```

There is no runtime adapter loading. A model's adaptation provenance comes only from its loaded receipt (or a verified pipeline-built base); a plain `Model(path)` has unknown provenance, and a lock of a specialist excludes every family that adapted, selected or calibrated it.

## Repository

`semantic_switch/` is the library, `packages/semantic-trainer/` the optional trainer, `bench/` the measured evidence (`bench/README.md`), `TODO.md` the open work. Tests: `pytest -m "not slow"` (controlled judges) and `pytest -m slow` (real models; one model job per host).
