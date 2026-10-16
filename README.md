# semantic-switch

Ask a language question about a case inside ordinary Python code and get a typed answer or an explicit non-answer.

```
input evidence + question (+ your rules) → model scores the allowed answers → accepted answer, or uncertain / needs_input / failed
```

Your application receives a typed member of the allowed answer set or an explicit non-answer. Exact rules execute in code. Confidence-based acceptance can withhold an answer, but a model can still be confidently wrong.

**What it is for:** classification, routing, bounded choice extraction and policy reading over bounded evidence: a message, a report, a few reference excerpts. **What it is not for:** arithmetic, logic puzzles or multi-step reasoning. The native judge reads answer-token scores without generating a reasoning trace. On the matched S1MB subset, the default judge was correct on 81.0% of accepted hard-label judgments at the 0.9 cutoff, with weaker results on reasoning tasks. See [benchmark scope and results](#benchmarks-and-how-to-read-them).

The 0.9 cutoff is an acceptance setting, not a promised precision. `ss lock` fits thresholds on your labelled development data and reports the precision and coverage it observed there; a different judge, task mix or procedure needs its own evaluation. The precise judge (E4B) settles more answers at similar precision; it is an option to measure, not a cure for confident errors.

## Features at a glance

| Need | Public path |
|---|---|
| Boolean judgment, one choice, ordered level, any matching labels | `check`, `pick`, `rate`, `select` |
| Several questions over the same evidence | `ask` with named `Check`, `Pick`, `Rate`, `Select` specifications |
| Business rules and procedure applicability | YAML switches, exact predicates, shared facts and card exclusions |
| Finite batches, bounded streams, async | `spec.run`, `spec.many`, `Runtime.ask_many`, `Runtime.aask` |
| Inspect requests and record decisions | `Runtime.preview`, `Runtime(on_decision=...)`, `result.as_dict()` |
| Label, evaluate and fit acceptance thresholds | `ss label`, `ss test`, `ss lock`; Python `semantic_switch.tools` |
| Optional task specialization | CPU lexical heads or LoRA, with support reports, matched controls and checkpoint selection |
| Understand a training result | `semantic-trainer assess`, fit-family `diagnose`, assessment-driven review |
| Optional data collection and repeated fitting | Explicit capture, teacher collection, review and rounds through `semantic-learn` |
| Deploy and recover a chosen version | Frozen locks, `ss compare`, `ss promote`, `ss rollback` |

The core package does not train models, collect traffic or run background learning automatically. Those operations are explicit and optional.

## Install

```bash
pip install -e '.[local]'   # Python ≥ 3.12; local judges: llama-cpp-python and Hugging Face downloads
pip install -e '.[hosted]'  # hosted judges only (httpx); no native packages
```

These editable-install commands run from this repository root. Use `pip install -e .` for the base package alone. For LoRA, install the optional [trainer and device-specific PyTorch environment](packages/semantic-trainer/README.md#install); for collection and review, install `pip install -e packages/semantic-learn` plus the local or hosted extra needed by its teacher.

The base package supports authoring and custom judges without native packages; a local judge without the `local` extra fails before any download and names the extra. The default local judge (Gemma 4 E2B, Q4_0 GGUF) downloads from Hugging Face on first use.

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
actions = {"refund": "Refund the duplicate charge", "review": "Request manual review"}
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

- **Pick** asks one question listing up to 20 labels, or 19 plus `none`. For larger catalogues, filter with `eligible=` or explicitly use `shortlist="lexical"` (up to 1,000 candidates). Shortlisting leaves candidates unexamined: inspect `complete` and `removed["not_examined"]`; a shortlist is not an exhaustive search.
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

Guidance is placed in the prompt in full; split long manuals into a `select` catalogue of cards. For inline judgments, `fields=[...]` selects mapping fields to render; without it, all mapping fields are rendered as named evidence, with non-text values encoded as JSON. Saved switches declare typed `fields` and `reads`. A missing required field produces `needs_input`; a field with the wrong type produces `failed`. Neither becomes a negative answer.

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
messages = ["I will sue over this charge.", "Please explain the charge."]
with ss.Runtime() as rt:
    results = legal.run(messages, rt)
    for result in legal.many(iter(messages), rt, window=4):
        print(result.status, result.value)
```

`await rt.aask(input, items)` and `await rt.acheck(...)` / `apick` / `arate` / `aselect` provide async calls on an open runtime. Saved switches offer `run(inputs)` and `many(inputs, window=...)`.

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

`ss.judge` resolves a GGUF path, Hugging Face `repo:file`, fitted deployment, or registered driver source. `ss.DEFAULT_JUDGE` is Gemma 4 E2B Q4_0; `ss.PRECISE_JUDGE` is E4B Q4_0. Native `device="gpu"` requires GPU offload; `device="auto"` may use CPU. A GPU-capable llama.cpp build is required for acceleration; installing the local extra alone does not guarantee GPU support.

OpenRouter sources use `openrouter:MODEL`. Known chat profiles include `google/gemma-4-26b-a4b-it` and `google/gemma-4-31b-it`; another chat model needs `codec="chat"` or a declared profile and must support the requested log probabilities. The built-in System One profiles for `typesafe/jev-1.13` and `openai/gpt-6-luna-decisions` return distributions for multiple questions per request. Profiles describe supported question kinds, widths and response precision; they do not certify model quality or continued provider availability. Pin `provider=...` when fitting a hosted lock. Enforce a hard spending ceiling on the provider API key; the SDK does not enforce a dollar budget.

Use a context-managed `Runtime` for application lifetime. It owns a judge it constructs from a source; a supplied judge is borrowed unless `own_judges=True`. `Runtime(bindings={"judge": local_judge, "reviewer": hosted_judge})` supplies named model roles used by saved definitions. Hosting, retries and concurrency use the same executor as local calls. Set `batch_inputs` to bound active hosted attempts and `timeout` for the end-to-end request-unit budget, including queueing and retries. Driver settings are also available through `ss --judge SOURCE --setting KEY=VALUE ...`.

Native evidence defaults to a 1024-token cap, guidance/template to 1024, and question suffix to 1536: a 3584-token logical window per sequence. Increase `max_input_tokens` or `max_prefix_tokens` for longer requests. `n_ctx` is an advanced **smaller per-stream window**, not a total allocation; four prefix slots allocate six streams. Overlong native requests fail without implicit truncation. Determinism applies to a fixed model, build and execution shape; changing quantization or batching can change probabilities and acceptance.

See [judge contracts and execution limits](docs/judges.md) for ownership, transports, retries, profiles and native memory details. See [benchmarks](#benchmarks-and-how-to-read-them) for quality and latency evidence.

Repository demos run as packages: `python -m examples.quickstart`, `python -m examples.printer_snapshot`, and `python -m examples.support_snapshot`. Their shared helper uses `OPENROUTER_MODEL` when set, with optional `OPENROUTER_PROVIDER` and `OPENROUTER_CODEC`; otherwise it uses the local judge. Demo inputs are authored examples, not quality guarantees.

## Saved switches: judgments + exact rules + decisions

When a decision combines several judgments with business data, save a definition such as `switches/support_route.yaml` and supply the referenced `policies/refunds.md`:

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
ticket = {"subject": "Duplicate charge", "body": "Please refund the second payment.",
          "payment": {"status": "settled"}}
with ss.Runtime() as rt:
    route = rt.load("switches/support_route.yaml")
    result = route(ticket)
    print(result.answer, result.rule)
    print(result.explain())
```

Rules run in declared order. Exact predicates settle first and only the facts a remaining rule depends on are judged; an unresolved earlier rule can withhold the decision rather than silently falling through. Exact rules prevent the specific invalid actions they encode, not every confidently wrong semantic answer. Any inline judgment saves to this form with `.save(path)` and behaves identically. For "which of these procedures apply?", a `select` fact over a reviewed card catalogue judges each shared situation claim once and applies exact eligibility and precedence in code; see `switches/procedures.yaml`. A card negates a fact with `excludes` (three-valued: an unknown fact leaves the card unresolved), so a logical complement never needs its own judged question. When a revision drops a fact, list it under `retired: [name]`: historical label files keep loading, their retired values are ignored, and an unknown or misspelled fact still fails.

## Optional: measure, correct, specialize

Start with labeled inputs and measured failures. Exact corrections, lexical heads, acceptance fitting and LoRA are separate options; none is required for ordinary judgments. The `route` below is the loaded switch from the previous example, used while its runtime is open.

```python
route.correct(ticket, fact="refund", value=True, actor=ss.Actor("human"))   # exact override for this input
route.correct(ticket, rule="refund", actor=ss.Actor("human"))              # or the rule this input should reach (ss correct --rule)
```

```bash
ss label support_route new_tickets.jsonl --to test --actor human   # label real inputs; supplied values are reviewed: Enter confirms, s skips, an answer replaces
ss lock support_route                                # fit per-judgment thresholds on development labels, check them on held-out labels, pin the judge and execution
ss test support_route                                # raw correctness of predictions, then accepted precision and coverage on unseen labels (intervals where sampling supports them)
ss --verbose run support_route < ticket.json # the decision with its explanation (an explain field with --json)
ss export support_route --native                     # export tune rows from the switch dataset
ss prune switches/                                   # remove generation files that no current manifest in the directory names
```

Situation JSONL for `ss label` and teacher collection uses schema-3 envelopes, one per line, for example:

```json
{"schema":3,"input":{"subject":"Duplicate charge","body":"Please refund the second payment.","payment":{"status":"settled"}},"groups":["ticket:123"],"pairs":[],"tags":[]}
```

`groups` keep related inputs together during assignment. Use distinct groups for unrelated situations; explicit `role` values can preserve an existing assignment. Label outputs add fact values, meaning revisions and source histories; let `ss label` write those records. `--to test` stores the switch's labeled dataset, which tools partition into development/check families; it does not declare every row an untouched check sample. `--to reviewed.jsonl` instead writes a separate label file for explicit export or specialization. `--to examples` installs exact corrections and is not independent evaluation evidence.

The same lock, export and test operations are Python functions in `semantic_switch.tools`, using the labeled switch from the CLI example:

```python
from semantic_switch import tools

with ss.Runtime() as rt:
    exported = tools.export(rt, "switches/support_route.yaml", native=True)
    report = tools.lock(rt, "switches/support_route.yaml")
    tested = tools.run_test(rt, "switches/support_route.yaml")
    print(tools.explain_test(tested))
```

- `tools.lock(rt, path, dry_run=False, *, head_bundles=None) -> dict` fits and checks thresholds and publishes `NAME.lock.json` unless `dry_run`.
- `tools.export(rt, path, to=None, split="tune", *, native=False, task_family=None, seed=0, scope=None, labels=None) -> dict` publishes an export manifest and its payloads.
- `tools.run_test(rt, path, data=None) -> dict` evaluates the switch on its labelled data, or on `data` (a label file) through a content-addressed sibling manifest.
- `tools.explain_test(report) -> str` renders a `run_test` report.

The optional packages expose their entry points the same way: `from semantic_trainer import train_lexical` (needs `semantic-trainer[lexical]`) and `from semantic_learn import collect`; importing either package loads no fitting or inference dependency. Specialization goes through `semantic_switch.training.specialization_request` and `specialize`.

An unlocked judgment still works; a lock pins configuration and fitted thresholds, not domain truth. Label records name a rule with `"rule"`. Held-out authored labels are development evidence, not an independent domain evaluation. `ss analyze-head SWITCH FACT --head BUNDLE` analyzes a supplied lexical head; `ss specialize` runs the trainer pipeline (`ss specialize --help`).

Task specialization (optional [`packages/semantic-trainer`](packages/semantic-trainer/README.md)) fits LoRA on the default student and deploys one merged, quantized GGUF: `semantic-learn collect` (optional `packages/semantic-learn`) labels situations with a teacher whose terms permit training use, `ss label … --to reviewed.jsonl` reviews them, `semantic-trainer prepare` assigns family roles and creates the job, `semantic-trainer lora` fits under model_select monitoring and retains the zero adapter and the two best evaluated points, and `semantic-trainer select` merges each into the F16 base, quantizes it, selects on model_select families and publishes one self-contained fitted lock (GGUF, bundled task definition and guidance, frozen policies and execution); `semantic-trainer check` evaluates the deployed model on the declared check population; independence depends on that population's exposure and lineage. The specialist is adopted only by your explicit choice:

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

A fitted lock declares one representation. `merged` is the default: one GGUF with the adapter merged into its weights, loaded without the trainer or PyTorch. `runtime-adapter` keeps a base GGUF and a converted LoRA GGUF separate and binds them at scale 1. Named roles with the same base and native settings share its resident model/context and a bounded adapter pool. These representations are explicit, never automatic fallbacks. Runtime-adapter repeat/residency controls passed on the measured Vulkan build, while whole-sequence reference scores differed by up to 0.017315. That evidence does not establish equivalence across execution paths or a universal switching-latency guarantee; see [native execution measurements](docs/ga-measurements.md) and [trainer deployment](packages/semantic-trainer/README.md).

Ordinary runtimes do not require model-job admission. An explicitly supplied `model_job` owns a host allocation envelope; model and adapter allocations are checked against it. This is an opt-in resource control for model workloads, not a prerequisite for using the SDK.

Facts answer under the definition's `binding:` (default `judge`; a fact or Select may name another model role, and staged questions and Select members inherit it). A job prepared with `semantic-trainer prepare --scope FACT[/MEMBER]` fits and selects only those targets: other bindings are neither loaded nor scored, and the result is scoped support, not a runnable application. Install it by role, `Runtime(bindings={"refund_model": ss.judge("run/deployment"), "judge": ...}).load("app.yaml")`, or publish a complete application with `semantic-trainer select ... --binding ROLE=JUDGE` for every remaining role; its lock pins those identities and policies, and loading it requires them: `Runtime(bindings={"hosted": ...}).load("run/deployment")`. Selection on the scoped targets says nothing about the whole application, which needs its own check.

A model's adaptation provenance comes only from its fitted lock (or a verified pipeline-built base); a plain `Model(path)` has unknown provenance, and a lock of a specialist excludes every family that adapted, selected or calibrated it.

## Specialization, reuse and interruption

```bash
ss specialize switches/support_route.yaml --examples reviewed.jsonl --to work/support --trainer /absolute/path/to/semantic-trainer --device xpu:0 --native-device gpu
ss specialize --to work/support --trainer /absolute/path/to/semantic-trainer
```

Choose `cpu`, `xpu:N` or `cuda:N` for fitting; the native scorer separately uses `auto`, `cpu` or `gpu`. The first command freezes inputs, resolves assets, exports model-specific token rows, prepares the population, measures fit-majority/native-base controls, fits, selects and publishes a deployment. The second resumes the same request. Corpus generation is separate: these commands consume existing labels and never regenerate situations. Token exports depend on the task, tokenizer and presentation; reuse requires the same content and settings. Completed pipeline stages are reused within the work root, not through a global cross-run stage cache.

Profiling is optional. The default monitoring cadence is one fit epoch; `--cadence` overrides it. Monitoring uses the full model-selection population, with separate `evaluation_batch_size` and no fit probe by default. Fits preserve evaluated candidates and rolling recovery checkpoints; repeating `semantic-trainer lora JOB --to RUN` resumes that identified attempt. A new `--attempt` starts a fresh fit. Selection compares retained candidates on development data, so training longer does not imply deploying the last weights. See [training and recovery](packages/semantic-trainer/README.md#monitoring-and-interruption). For CPU lexical fitting without LoRA, see [the lexical-head example](examples/lexical_head.py): export labeled rows, call `train_lexical`, then pass the resulting bundle to `tools.lock(..., head_bundles=...)`.

## Train and understand the result

Semantic-switch supplies the training and evaluation tools. You supply the task meaning, relevant inputs, trustworthy labels and acceptable error costs. Training can complete correctly while retaining the base model; that is a measured outcome, not a failed command or proof of a model-size limit.

The workflow is: inspect supplied evidence → measure controls → fit → compare checkpoints → report the selected model. Collection and label generation happen separately. Start with the same work directory for every command below; completed stages are reused.

```bash
ss specialize task.yaml --examples labels.jsonl --to work/task --trainer /absolute/path/to/semantic-trainer --device xpu:0 --native-device gpu --until freeze
ss specialize --to work/task --trainer /absolute/path/to/semantic-trainer --until baseline
ss specialize --to work/task --trainer /absolute/path/to/semantic-trainer
```

`--until freeze` checks the supplied inputs and support on CPU, without loading torch or the native model. `training_support.path` names class/family support, missing input dependencies and, after population preparation, objective shares. Sparse classes are reported without forcing another data-generation round. `--until baseline` measures the matched native base and fit-only majority on model-selection data, before any optimizer updates. Add `--baseline lexical` to the first command only if you want the optional lexical control; install the trainer's `lexical` extra in that environment. Resuming uses the frozen request.

The completed result's `learning.path` names `learning.json`: candidate/class/behavior outcomes, confident errors, coverage, checkpoint regressions, acceptance status and the actual reason a candidate won. Model-selection correctness remains the default ranking objective; accepted errors are reported separately. An accuracy win can still have more accepted errors. Uncalibrated thresholds remain explicitly uncalibrated. Raw model scores are not demonstrated probabilities of correctness.

```bash
semantic-trainer assess work/task/select/DEPLOYMENT --to work/task-assessment.json
semantic-learn review task.yaml --learning work/task/select/DEPLOYMENT/learning.json --size 20 --audit 10 --stratum disagreement --to work/failures.review.json
```

Use the actual deployment path returned by specialization. `assess` rebuilds the report from stored rows without inference; `--development CONFIG.json` explicitly requests composed evaluation using existing action references. Review preserves supplied reference histories and family roles; predictions select review and never become gold labels. Keep selection/check families out of fit. Acquire new fit families for missing behaviors and retain the original fit examples when running another [learning round](packages/semantic-trainer/README.md#reviewed-additions-through-the-same-pipeline). Adoption remains a separate compare/promote operation.

For a fitting problem, isolate existing fit families before changing the recipe:

```bash
semantic-trainer diagnose JOB.yaml --family FIT_COMPONENT_ID --to work/fit-diagnostic
semantic-trainer diagnose JOB.yaml --family FIT_COMPONENT_ID --to work/fit-diagnostic --run
```

Use the resolved job and family IDs from the run's support report. The first command prepares the diagnostic; `--run` executes it. Repeated `--family` options include whole related families. A probe already correct before training reports `already_solved`; it does not justify increasing rank or model size. This is an in-sample fitting check, not a generalization test. See [assessment and diagnosis](packages/semantic-trainer/README.md#baselines-assessments-and-controlled-diagnosis) for rank/alpha controls and report preservation.

## Collection and observability

`Runtime(on_decision=callback)` sends typed `DecisionEvent` records to your callback; `event.as_dict()` serializes them. Raw input is excluded unless `decision_input=True`. `Runtime.preview(input, items).as_dict()` shows rendered questions and available driver request previews without scoring; constructing a local preview can still initialize the native model. These hooks do not automatically save data or create training labels. See [decision events](examples/events.py).

The optional `semantic-learn` package provides `Capture` in `semantic_learn.capture` with explicit redaction, consent, sampling and bounds, then explicit `situations`, `collect`, `review`, `diagnose` and `round` commands. Teacher labels require a declared permitted training route; confident `unstated` and allowed `none` are valid semantic labels. Collection journals resume unfinished work. Reviewed additions can feed another round while preserving prior fit evidence and family lineage. There is no background scheduler, automatic specialist discovery or automatic promotion. The [teacher–student cycle](docs/teacher-student-cycle.md) and [specialist portfolio](docs/specialist-portfolio.md) are extension designs, not shipped automatic modes.

## Benchmarks and how to read them

| Evidence | What it measures | Limits |
|---|---|---|
| [S1MB matched subset](bench/README.md#s1mb-external-subset-first-25-cases-per-benchmark-benchexternal_primitivespy) | Fixed-answer Check, Pick and Rate on public tasks; E2B accepted hard-label precision 81.0%, E4B 82.1% at 0.9 | First-N subset; 2,233 matched cases, 3,257 hard-label judgments; not a leaderboard score or a precision guarantee |
| [Select quality](bench/README.md#select-quality-benchselect_qualitypy) | Public multi-label Select versus named Checks, local and hosted | Mostly negative hard labels: only 29 unanimous positives; aggregate correctness hides low positive precision |
| [Procedure and equipment studies](bench/README.md) | Exact-rule composition, procedure applicability and SDK/CLI behavior | Authored fixtures and historical runs; source, prompt and execution versions differ |
| [Native and training measurements](docs/ga-measurements.md) | Cache/residency controls, latency, allocation and completed training operations | Hardware/build/workload-specific; historical timings are not current performance promises |
| [Printer enrichment experiment](#measured-readiness-2b-printer-specialist-2026-10-16) | Completed LoRA deployment comparison with random and targeted fit additions | Synthetic structured text, repeatedly consulted check population, unreviewed reference |
| [Training workflow verification](bench/results/training-workflow-20261016/README.md) | Real 2B fitting, native selection, controls, interruption/resume and stored-row reports | Small development evidence; both task runs retained the base; no adoption |

Raw accuracy, accepted-answer precision, coverage, per-class recall and exact action-set agreement answer different questions. A 0.9 score cutoff is not 90% correctness. Latency must name its unit: one question and one whole situation with ten questions are different workloads. Provider and quantization changes also change the comparison.

The [benchmark catalog](bench/README.md) lists commands, artifacts, versions and historical limitations. Start quality comparisons with `python -m bench.model_comparison --model MODEL --output result.json`; maintained report writers require an explicit output path. Snapshot experiments consume an existing published corpus; [printer commands](bench/README.md#printer-fit-and-comparison) describe fitting, recovery and optional paid reference arms. Running examples demonstrates API behavior, not task accuracy.

The latest [training workflow verification](bench/results/training-workflow-20261016/README.md) exercised actual E2B LoRA fits and merged Q4_0 native selection. On 14 customer/verification development observations, the base scored 11/14 with two wrong accepted answers; the two student checkpoints scored 8/14 with six and 11/14 with three. Selection retained the base. On a six-observation operation view of the same corpus, the base and one student tied at 6/6, so the base was retained again. These small, related populations verify the controls and selection workflow; they do not establish unfamiliar-language quality. The same verification resumed an interrupted fit, reused committed base scores and rebuilt assessments without inference. No adoption occurred.

## Measured readiness: 2B printer specialist (2026-10-16)

The completed printer experiment supports a scoped pilot with reviewed actions and a fallback. It demonstrates strong task specialization on synthetic structured inputs; it does not establish general availability for autonomous operation on real operator notes.

Both arms used Gemma E2B with LoRA, deployed as merged Q8_0 GGUFs through the existing llama.cpp scorer. Each added 96 fit situations from 16 families to the original 100 fit situations. One addition was randomly selected; the other targeted four under-covered behaviors. The model-selection population stayed fixed at 150 situations. The original corpus and role assignment were preserved; additions were published separately. Both runs completed 980 updates, and deployment selected checkpoints on model-selection data rather than taking the last update.

| Completed deployed-model check | Random enrichment | Targeted enrichment |
|---|---:|---:|
| Exact action sets on fully labeled situations | 560/560 | 560/560 |
| Action precision across known reference memberships | 99.5% | 99.1% |
| Action recall across known reference memberships | 92.5% | 97.4% |
| Situations with a reference-forbidden recommendation | 4 | 8 |
| Median latency per whole situation (10 scored questions) | 1.26 s | 1.10 s |

The comparison evaluated 736 structured situations. Only 560 have complete action references and enter exact-set accuracy. The remaining 176 still contribute their known action labels to precision, recall and forbidden-recommendation counts; their unresolved labels are unassessed. This is why 100% exact-set accuracy coexists with forbidden recommendations. These are action-set results, not a claim that every fact in every situation was correct.

Training also showed why checkpoint selection matters: the random-enrichment arm reached 1,500/1,500 correct development answers across all 150 situations at update 600, then fell to 1,487/1,500 at update 750. The targeted arm reached 1,500/1,500 at update 600 and maintained it at every later evaluation. More updates did not consistently improve the model.

The evidence supports improving training coverage before increasing model size. It does not show that targeted sampling always beats random sampling: this was one cohort pair and one training seed, and historical runs used fewer fit examples and updates. Both additions came from the same narrow authored language templates. The check population has been consulted repeatedly, so these results are regression evidence rather than independent confirmation. The comparison reference remains unreviewed: all 305 required evidence confirmations are outstanding. No adoption was performed by this experiment.

The next quality evaluation is fresh, freely written operator notes with reviewed labels, including missing and contradictory evidence. Measure incorrect accepted actions, withholding and per-behavior recall there before claiming autonomous production readiness. A pilot should retain review or fallback for consequential actions.

Published evidence: [summary](bench/results/printer-coverage-20261016/summary.json), [cohort declaration](bench/results/printer-coverage-20261016/cohorts.json), [run declaration](bench/results/printer-coverage-20261016/declaration.json), and [complete comparison and preservation inventory](bench/results/printer-coverage-20261016/README.md). Reports and small evidence are preserved in the repository; trained weights and recovery state are archived separately.

## Repository

See [repository layout, preservation and developer setup](docs/repository.md) for directory ownership, ignored outputs, corpus preservation, archives and complete multi-package test installation. Public usage stays in this README; current contracts are in [invariants](docs/invariants.md), implementation status in [TODO](TODO.md), and the superseded specification is explicitly [historical](docs/history/semantic-switch-v0.26-20261006.md).

Tests cover numerical correctness, decision logic, ownership, concurrency, recovery and SDK behavior. Scripted SDK examples and controller integrations do not measure model quality. A separate real deployment test checks nonzero LoRA → merge → Q8_0 GGUF → native probabilities against PyTorch, using a tiny locally generated model. See [test scope and commands](docs/repository.md#what-the-tests-establish). Passing tests do not establish accuracy on unfamiliar language; the benchmark results above retain their stated limitations.


## Scores and acceptance

Allowed-answer probabilities describe the model distribution, not the probability that its answer is correct. Answer mass is the probability assigned to the allowed answer tokens before conditioning; it remains separate from class probabilities. Reports distinguish argmax accuracy, precision among accepted answers, coverage and wrong acceptances per submission, with denominators and family counts. NLL, Brier and reliability bins describe stored complete distributions; missing and staged distributions are explicitly unassessed.

Authored thresholds are uncalibrated unless an acceptance-fit population was used. Checkpoint-selection data remains development evidence. Fixed answer sets prevent malformed labels; three-valued rules propagate unknown facts but cannot detect a confidently wrong fact. `uncertain` and `needs_input` are outcomes, not guarantees of detecting every unanswerable question. Determinism applies to the recorded model, native build and execution shape; changed quantization or batching requires a measured comparison.

Training recipes can opt into class/family effort allocation, authored behavioral slices and development recall requirements. See [trainer usage](packages/semantic-trainer/README.md) for bounded seed/LR qualification and fit-only diagnosis. Data generation and input-view materialization remain explicit separate operations; reusing a corpus does not regenerate it.

Reviewed failures can be recorded with `semantic-learn diagnose REVIEW.json UNIT --reason coverage --reviewer NAME --evidence ARTIFACT --note TEXT --to DIAGNOSES.jsonl`. Reasons route work to materialization, production, review, task interpretation, fit diagnosis or acceptance calibration. These records are review metadata, never training labels. Keep existing fit examples when consuming approved additions through `semantic-learn round`; promotion remains a separate decision.

New unassigned label families now receive deterministic allocation using their answer classes and declared behavior tags. Existing published roles stay fixed, including check families. Allocation reports actual counts and missing support; it does not claim every behavior is covered or refuse an ordinary diagnostic fit.

For a fair lexical baseline, pass resolved fitting observation IDs with `train_lexical(..., fit_cohort=IDS)`. Its default tune-wide cross-fitting remains a different protocol. Optional recipe qualification can also evaluate composed actions through the existing development evaluator. See [the behavioral development command](bench/README.md#behavioral-decision-development) and [operational qualification](packages/semantic-trainer/README.md#fixed-role-lexical-controls-and-operational-qualification).

The [dependability implementation report](bench/results/dependability-20261016/README.md) and subsequent [training workflow verification](bench/results/training-workflow-20261016/README.md) include actual 2B fits and native selection on a family-audited language corpus. Neither established a student improvement over the base. Independent unfamiliar-language confirmation remains outstanding; the earlier synthetic printer gains do not establish it.
