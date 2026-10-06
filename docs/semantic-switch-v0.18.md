# Semantic Switch — MVP Specification

| | |
|---|---|
| **Status** | MVP spec v0.18 — v0.16 implemented (`semantic_switch/`, 142 tests); v0.17 and v0.18 are specified, not implemented (§15). v0.18 makes `Judgment` the single definition of questions, values and acceptance (it is implemented 6 times today), keeps the input cached in the scoring layer (a repeated round on a realistic ticket costs a full ~1.1 s re-read today), and lays out a zero-to-hero path and a fine-tuning subsystem. The fresh suite puts the default judge's untuned precision at 88.9%; precision comes from `ss lock` on your labels. |
| **Date** | 2026-10-05 |
| **Owner** | Tulio Ribeiro |
| **One line** | A small, fast library for language-based judgments inside ordinary code — local, cheap, multilingual, open source, with rules you can read, teach and diff. |

---

## 1. What it is

A Python library and CLI with **three inline primitives** and **one saved form**.

```python
import semantic_switch as ss

ss.check(ticket, "The customer threatens legal action against us")                # predicate
ss.pick(doc, "Which address should receive the receipt?", candidates)             # choice, dynamic options
ss.rate(idea, "How feasible is this?", ["impossible", "hard", "doable", "easy"])  # rubric score

ss.ask(ticket, {                                                                  # several judgments,
    "legal":   ss.Check("The customer threatens legal action against us"),        # one shared read
    "topic":   ss.Pick("What is this about?", {"billing": "...", "tech": "...", "other": "..."}),
    "urgency": ss.Rate("How urgent is it?", ["can wait", "today", "immediately"]),
})

route = ss.load("switches/route_ticket.yaml")                                     # saved form:
route(ticket)                                                                     # facts + cases → answer
```

> **Code builds a bounded decision → the model judges → code executes.**

### Zero to hero (one new concept per step, same semantics throughout)

| Step | API | New concept |
|---|---|---|
| 0 | `ss.check(text, "The customer requests a refund.")` | none |
| 1 | `refund = ss.Check("…")`, then `refund(text)` and `ss.ask(text, {"refund": refund, "topic": topic})` | A reusable judgment: `Check` / `Pick` / `Rate` are callable |
| 2 | `sw = refund.save("switches/refund.yaml")`, then `sw.correct(text, fact="refund", value=True)` | Saved and teachable. It is the same `Judgment`, so it gives identical answers before and after saving. |
| 3 | add `cases:`, `mentions(...)` and exact rules in the YAML | Rules |
| 4 | `ss label … --to test`, `ss lock`, `ss test` | Precision on your data. **This is where precision comes from**: untuned, the default judge is 88.9% precise on unseen domains (§14.6). |
| 5 | `ss specialize` → `tier: 1`, `head: lexical` | Cheaper repeats |
| 6 | `--judge my-finetuned.gguf`, `ss doctor --quality` | A better judge, with no API change (§7.2) |

### Where fast and cheap come from

1. **Less compute per decision.** Work goes to the cheapest step that can settle it:
   ```
   exact conditions → exact taught matches → optional tiny classifier → semantic judge
   ```
   - The judge's system prompt is computed once.
   - Each input is read once per model and reads group.
   - Several questions share that read.
2. **Hardware you already have:** a laptop CPU, a Mac, an existing GPU. No per-call fees, no rate limits, no network round trip.
3. **Small quantised models** through llama.cpp.

Speed and cost are **gates measured per named reference machine** (§13).

### Not for

Generating text, planning, multi-step reasoning, arithmetic, dates or counting (do those in code), open-ended extraction without candidates, fetching data, or **ingesting documents** (parsing manuals, extracting procedures, building corpus indexes). A catalog of cards is an input the user supplies (§4.2.2).

### Not in scope

Decision logs, review queues, audit trails, replaying production traffic, crash recovery, statistical certification, multi-tenancy, servers, UIs, generated summaries of procedures, and safety-critical use as a replacement for certified alerting or checklist systems. This is a decision-support library.

### Relative to Jev

Jev [R7] is a hosted API at $0.042 per million input tokens (≈ $3.6 per million decisions at the ~85 tokens a short decision really uses, §13.4), capped at 80 requests/s. Semantic Switch runs locally, is open and multilingual, and keeps rules and examples in git. Speed and cost comparisons are made only by matched benchmarks (§14).

---

## 2. Principles

1. **Judgments in the model; logic in code.**
2. **Options are data**, from a file or built per call.
3. **Three truth values.** UNKNOWN is never silently turned into an answer.
4. **Missing is not uncertain.** Missing input is reported by field name, and the model is never asked to judge data that isn't there.
5. **Scores are model probabilities, not correctness.** Correctness comes from `ss test` on labelled, **unseen** data.
6. **Less compute per decision beats faster hardware.**
7. **Rules are stable; models are replaceable** behind small protocols.
8. **Small.** Every feature must make a single function call faster, cheaper, more correct or easier to edit.

---

## 3. Inline primitives

### 3.1 Inputs

| Argument | Meaning |
|---|---|
| `state` | A string or a dict of fields. `fields=[...]` selects and orders them. |
| question / statement | One plain-language sentence. For `check` it is a statement to verify. |
| `options` (`pick`) | A dict `{id: description}`, a list of strings, or a list of objects plus `describe=` (a callable giving the text the model sees for each object) |
| `levels` (`rate`) | Ordered, from low to high |
| `eligible=` (`pick`) | Callable `candidate → bool`, applied before scoring |
| `allow_none=` (`pick`) | Adds a "none of these" option (uses one slot) |
| `shortlist=` (`pick`) | `None` (default) or `"lexical"`: opt-in narrowing when candidates exceed the slots (§3.4) |
| `focus=` (`pick`) | `None` (default) or `"mentions"`: show the judge only the candidates the input names exactly, plus an escape option (§3.5) |
| `ids=` (`pick`) | Callable `candidate → (id, aliases)` or a dict option's `{id: {description, aliases}}`; enables exact mention matching (§3.5) |
| `unstated=` (`check`) | `False` (default) or `True`: three-way question; "the input does not say" returns `needs_input` (§3.6) |
| `examples=` | `[(state, answer), …]` |
| `min_score=` | Gate, a number in **[0.5, 1]**. NaN or values outside the range raise `ValueError`. |
| `model=` | Named model role (§10.1). Default `"judge"`. |

- **Programming errors raise exceptions:** unknown model, invalid `min_score`, duplicate option keys, malformed options.
- **Execution problems return `status="failed"`** with `r.error`.

### 3.2 Results

Every result, inline or switch, uses **one status vocabulary**:

| Status | Meaning |
|---|---|
| `ok` | A value was produced (and passed `min_score`, if given) |
| `none` | `pick` with `allow_none`: "none of these" won, or no candidate was eligible |
| `uncertain` | Below `min_score` (inline), or a blocking fact is not settled (switch) |
| `needs_input` | A blocking condition reads a field the caller didn't supply (switch), or a three-way check found the input doesn't say (inline and switch, §3.6) |
| `failed` | Invalid context, backend error on a blocking fact, or a limit exceeded |

```python
r = ss.check(ticket, "The customer threatens legal action against us", min_score=0.9)
r              # <ok · True · p_true 0.97>
r.status       # "ok" | "uncertain" | "failed"
r.value        # True | False | None
r.p_true       # model probability that the statement holds (not a correctness probability)
r.mass         # share of model probability on valid answer tokens: a format signal

r = ss.pick(doc, "Which address should receive the receipt?", addresses, describe=lambda a: a.text)
r              # <ok · Address(...) · 0.94>
r.value        # the chosen candidate object, unchanged
r.index        # its position (or its key, for dict options)
r.score, r.scores, r.removed, r.truncated

r = ss.rate(idea, "How feasible is this?", ["impossible", "hard", "doable", "easy"])
r              # <ok · 'doable' · 0.60>
r.value, r.index, r.expected, r.scores
```

Gating:

- **Without `min_score`:** `check` returns `p_true ≥ 0.5` (an exact 0.5 is True); `pick` and `rate` return the most likely option.
- **With `min_score = m`:** `check` gives True if `p_true ≥ m`, False if `p_true ≤ 1 − m`, otherwise `uncertain`; `pick` and `rate` give `uncertain` when the top score is below `m`.
- **Answer format is checked before any gate, by the `Judge`, the same way for inline calls and switches:** when under half the model's probability lands on the valid answer tokens (answer mass < 0.5), the result is `uncertain` with `error="low answer mass"`, whatever `min_score` says. (Before v0.14, inline calls ignored mass and a switch didn't, so moving a call into a switch changed its behaviour.)

**Cheap work runs before model work.** Validation (`min_score`, levels, options), eligibility, the candidate limit and immediate answers (no eligible candidate, an empty `ask`) all run before the judge is acquired. An invalid or immediately-answered call never loads or downloads a model. `describe()` runs only for eligible candidates, and only after the candidate limit is checked. `rate` levels and `pick` candidates are validated against the adapter's slot limit (20) with a `ConfigError`, never an adapter crash.

**`bool(result)` raises `TypeError`** ("use `.value`"). Results print as one-line reprs and hold only compact score rows, indexes, and (for `pick`) a reference to the chosen object.

**Examples are typed.** `check` examples accept booleans or the strings `true/false/yes/no`. `pick` examples name their answer by key, by index, or by the candidate object itself: matching is by identity, then equality, never by hashing, so unhashable objects work.

### 3.2.1 Reusable and saveable judgments

`Check`, `Pick` and `Rate` are **callable**. `refund = ss.Check("The customer requests a refund.")` is a value. `refund(text)` runs it on the default runtime, `refund(text, rt=rt)` on a given runtime, and `ss.ask(text, {"refund": refund})` composes it.

`refund.save(path, field="message")` writes a switch without cases whose one fact is this judgment and returns the loaded `Switch`:
- If the file exists with the same definition, it loads it.
- If the definition differs, it raises `ConfigError` instead of rewriting.

`Pick` saves only with dict options; per-call candidate objects and callables can't be saved. This replaces `rt.judgment()` (v0.17): the method hangs off the object the user already holds.

### 3.3 `ask`

`ss.ask(state, {name: ss.Check|ss.Pick|ss.Rate, …})` reads the state once per model and scores all its questions in one batched call (§10.4). Each question is answered independently. Combine the answers in code.

### 3.4 `pick` edge cases

| Situation | Behaviour |
|---|---|
| No eligible candidate | `none` if `allow_none`, else `failed("no eligible candidates")` |
| One eligible candidate | Scored normally (never auto-selected) |
| More eligible candidates than the adapter's slots (20, or 19 with `allow_none`) | `failed("too many candidates")` unless `shortlist="lexical"`. With it, an in-memory BM25 over candidate texts against `question + state` keeps the top ones; the dropped indexes appear in `r.removed`. |
| Candidate text longer than 64 tokens | Truncated; listed in `r.truncated` |
| A dict option named `"none"` together with `allow_none` | `ConfigError`: `none` is reserved for the none-of-these slot |
| `shortlist` other than `None` / `"lexical"`; `focus` other than `None` / `"mentions"`; `rate` levels empty, single or duplicated | `ConfigError` |
| BM25 shortlisting | Candidates exactly mentioned in the input (§3.5) rank first; BM25 orders the rest. The tokenizer keeps `MX-120A` → `mx`, `120a` distinct from `MX-120` → `mx`, `120`. |

`describe()` is called only for eligible candidates, so an expensive description callback never runs for candidates that can't be chosen.

### 3.5 Exact text evidence

Language models blur small, decisive differences; exact matching can't tell who did what. Each gets the job it does exactly, and **evidence never becomes a weighted score**: no BM25 + cosine + probability fusion, since those numbers mean different things.

**Mention matching (exact, microseconds, no model):**
- A candidate **id** or **alias** is mentioned when it occurs in the input with no letter, digit, `-` or `_` directly on either side, and no `.` that continues into a word (`MX-120.5` is not `MX-120`, but `MX-120.` at the end of a sentence is). `MX-120` does not match inside `MX-120A`. Matching is case-insensitive and Unicode-NFC; nothing else is normalised (no stripping of `-`, no `O`→`0`).
- Ids and aliases come from `ids=` (inline) or from options declared as `{description, aliases}` (saved switches).

**`focus="mentions"` (opt-in, `pick`):**
1. Find the candidates mentioned exactly.
2. If there are none, score the full list as usual.
3. Otherwise score only those, plus one escape option, "another one, not listed above".
4. If the escape wins, or the focused answer doesn't settle, score the full list once more. The result reports `r.focus = "mentions" | "full" | "fallback"`.

So **absence of a mention never excludes a candidate.** "MX-120 is fine, the other one broke" falls through to the full list via the escape.

**Fuzzy matching only proposes, never decides.** When nothing matched exactly, candidates within edit distance 1 of a token in the input are listed in `r.near` for the caller. Fuzzy matching is disabled for any pair of catalog ids within distance 1 of each other, so `MX-120` and `MX-12O` can never be merged. Token-set or subset ratios are not used.

**Measured (identifier set, §14.6):**
- **Gemma 4 E2B** kept near-identical ids apart (13/14 at 4 and at 20 options, never confidently wrong). **Qwen3-1.7B** confidently swapped `MX-120A` → `MX-120` at 1.00.
- **The payoff of focus is speed and guarantees, not fixing the default judge:** a 20-option pick costs 1.1 s against 0.5 s for 4. Exact mentions narrowed 20 candidates to 1–3 in 12 of 14 cases and never dropped the right answer. The two misses (`MX120A`, `MX-12OA`) matched nothing, so the full list is used.

### 3.6 Three-way checks: false is not "not stated"

A yes/no check forces a guess when the input is silent, and the guess is confident. On reports that never mention a leak, Gemma 4 E2B scored "There is a visible hydraulic leak" at **p_true 0.00** on all four. A switch reading that as an established FALSE would select a procedure whose exclusion is "visible leak" without anyone having looked.

**`unstated=True` (inline) / `unstated: true` (fact)** asks a three-way question with one answer token, so it costs the same as a check:

```
Claim about the input above: There is a visible hydraulic leak.
Does the input state the claim is true, state it is false, or not say?
Answer with only the letter. A) states true  B) states false  C) does not say
```

| Answer | Atom | Status when blocking |
|---|---|---|
| states true | TRUE (thresholded as usual) | — |
| states false | FALSE (thresholded as usual) | — |
| does not say | UNKNOWN, reason `not_stated` | **`needs_input`**, with `missing=("fact:visible_leak",)` and `why: "not stated in the input"` |

- `Fact.scores` is `{"true": …, "false": …, "unstated": …}`. `p_true` stays the probability of "states true".
- **Measured** (`bench/unstated.py`, 10 reports, EN/PT): Gemma 4 E2B 9/10 correct. Its one miss ("floor is dry, nothing dripping" → "does not say" at 0.68) came back unsettled, not wrong. Qwen3-0.6B 6/10.
- **When to use it:** preconditions and exclusions about the world ("is there a leak", "was the customer charged"), where silence is common and a guess is costly. Plain checks stay right for claims about the text itself ("the customer asks for a refund").
- Examples and labels for a three-way fact accept `true`, `false` and `unstated`.

## 4. Switches: facts + cases

### 4.1 Files

```
switches/
  route_ticket.yaml            definition: fields, facts, cases, tests     — edited by people
  route_ticket.examples.jsonl  teaching examples, append-only              — written by `correct` / `ss label`
  route_ticket.test.jsonl      labelled inputs for tune / test / diff      — written by `ss label` / you
  route_ticket.lock.json       frozen, self-contained behaviour            — written only by `ss lock`
```

Plain files, all in git. **No database.** `ss lock` is the only command that writes the lock.

### 4.2 Definition

```yaml
# switches/route_ticket.yaml — runs on a default install
name: route_ticket
schedule: auto                       # auto | eager | staged (§4.6)

fields:
  subject:        { type: text,   required: true,  max_chars: 300,  author: customer }
  body:           { type: text,   required: true,  max_chars: 3000, truncate: middle, author: customer }
  product:        { type: text,   source: crm }
  payment.status: { type: enum,   values: [pending, settled, failed], source: payments }

reads: [subject, body]

facts:
  legal_threat:
    check: The customer threatens legal action against us.
    excludes: [quoting someone else's threat, merely mentioning a lawyer]
    target: 0.98
    examples:
      - { input: { subject: Final notice, body: My lawyer will file suit on Monday. }, value: true }
      - { input: { subject: Question, body: My lawyer asked me to request an invoice. }, value: false }
  refund_requested:
    check: The customer asks to receive money back for a charge.
    excludes: [asking about future prices]
  disputes_payment_status:
    reads: [body, payment.status]
    check: The customer disputes the recorded payment status.
  topic:
    pick: What is the message mainly about?
    options:
      billing:   charges, invoices, payments, plans
      technical: the product is broken, erroring, slow or unavailable
      account:   login, profile, access, data
      hardware:
        description: a physical device is damaged or not working
        requires: product in ["router", "modem"]
    allow_none: true
  urgency:
    rate: How urgent is the customer's situation?
    levels: [can wait, today, immediately]

cases:                               # first TRUE case wins
  - { id: legal,           when: legal_threat,                                       then: legal }
  - { id: refund_settled,  when: 'refund_requested and payment.status == "settled"', then: refund_review }
  - { id: payment_dispute, when: exists(payment.status) and disputes_payment_status, then: billing_escalation }
  - { id: refund_other,    when: refund_requested,                                   then: billing }
  - { id: tech_urgent,     when: 'topic == "technical" and urgency >= "today"',      then: tech_priority }
  - { id: tech,            when: 'topic == "technical" or topic == "hardware"',      then: tech }
otherwise: general

tests:
  - input: { subject: Complaint, body: "Broken again. Refund me or my lawyer will file a lawsuit." }
    expect: legal
```

**Cases shorthand.** When the answer is the case's name, use an ordered mapping:

```yaml
cases:
  legal: legal_threat
  refund_review: refund_requested and payment.status == "settled"
  billing: refund_requested
otherwise: general
```

In the long form, `then` defaults to `id`, and `id` defaults to `then`.

**Pick options with identity:**

```yaml
facts:
  failed_machine:
    pick: Which machine does the report say failed?
    focus: mentions
    options:
      MX-120:  { description: standard mixer }
      MX-120A: { description: advanced mixer, aliases: [Mixer 120 Advanced] }
      MX-210:  { description: industrial mixer }
```

An option key is its id; `aliases` are extra exact forms. `mentions(report, "MX-120A")` can also guard cases and `requires:` directly.

- **`author:` / `source:`** only change how a value is rendered (`[customer] body: …`). Measured: attribution matters. Without it, Qwen3-1.7B scored "my lawyer will file a lawsuit" at 0.00 for "the customer threatens legal action", because the raw text never says who is speaking. With `[customer]` it scored it correctly.
- **`reads:`** controls what each judgment sees (switch default, per-fact override).
- **`model: <role>`** names a runtime role. The role must be bound in the runtime, or loading fails with a `ConfigError`.
- **Guard facts that read optional fields** with `exists(field) and …`. Otherwise every input lacking that field correctly returns `needs_input`.

### 4.2.1 Switches without cases: the teachable judgment

`cases` and `otherwise` are optional. A switch without them is a **named, saved set of judgments**. Calling it returns the facts, like `ask`: `r.facts`, with `r.answer = None` and status `ok` when every fact settles, else `uncertain`, `needs_input` or `failed` as usual.

```yaml
# switches/refund.yaml
name: refund
fields: { message: { type: text, required: true, author: customer } }
facts:
  refund: { check: The customer requests money back for a charge already made. }
```

```python
refund = rt.load("switches/refund.yaml")
refund({"message": "Reverse that charge"}).facts["refund"].value
refund.correct({"message": "Reverse that charge"}, fact="refund", value=True)   # sticks now (draft)
```

This is the teachable form of an inline `Check`. Corrections, labels, `ss specialize`, `tier: 1`, `ss lock` and the cascade all work unchanged, so there is **one teaching system**. Inline `check` / `pick` / `rate` stay stateless; there is no separate `Check.teach()` / `fit()` store.

### 4.2.2 `select`: applicable procedures from a catalog

For "which procedures apply to this situation": candidates are **cards** in a catalog file; several may apply, one may supersede another, and a decisive detail may be unstated.

```yaml
facts:
  procedures:
    select: Which procedures apply to this situation?
    catalog: procedures.yaml
    max_cards: 16         # cards judged per call (~150 ms each on CPU); within budget, the search is complete
    exclusive: false      # true: two final cards → uncertain
```

```yaml
# procedures.yaml: written and reviewed by people (or drafted offline and reviewed); never generated at runtime
- id: P02
  title: MX-120A startup fault E17
  codes: [E17]                       # exact triggers; matched against fields and text
  aliases: [MX-120A, Mixer 120 Advanced]
  requires: 'machine == "MX-120A"'   # exact atoms over typed fields
  applies_when: Startup fails after restarting.
  excludes: [visible hydraulic leak]
  supersedes: [P01]
  source: { manual: "Maint. manual rev 7", section: "7.3", pages: "142-145" }
  payload: { instructions: procedures/P02.md }   # returned, never scored
```

**Principle: a shortlist ranks; only an exact rule or an examination rules a card out.** BM25 and embeddings are recall tools, not evidence. A card left out of the search is unknown, never "doesn't apply". (v0.16 treated it as FALSE: two cards, `top_k=1`, the examined one rejected → a settled "nothing applies" from an incomplete search.)

**Card states** (`SelectState`, one per card; separate from the probability distributions of check / pick / rate):

| State | Means | Atom `has` |
|---|---|---|
| `ineligible` | Ruled out by an exact `requires`. Sound. | FALSE |
| `unexamined` | Not judged in this call | UNKNOWN (`not_examined`) |
| `applicable` | Judged: it applies | TRUE |
| `rejected` | Judged: trigger absent, or an exclusion stated | FALSE |
| `unresolved` | Judged, not settled: `not_stated`, `missing_input`, `uncertain`, `low_answer_mass`, `model_error`, or `may be superseded by X` | UNKNOWN |
| `superseded` | Applies, but an applicable card overrides it | FALSE |

Each kept card also records how it was retrieved (`code`, `alias`, `bm25`).

**Pipeline per call:**
1. **Exact filter:** cards whose `requires` is FALSE become `ineligible`. UNKNOWN keeps the card and records the missing field.
2. **Budget:**
   - **Within budget** (eligible cards ≤ `max_cards`): every eligible card is examined, and the result is `complete`.
   - **Over budget:** exact code links (codes in a typed field such as `active_alarms`, or mentioned in the text, §3.5) are always kept. Alias mentions rank next, then BM25 over `title`, `applies_when` and aliases fills the budget. The rest stay `unexamined`, and `complete` is false.
   - BM25 statistics, alias and code tables, and card questions are built once at load.
3. **Judge:** two kinds of question, all sharing one read of the situation:
   - **Trigger: a yes/no check per examined card**, "The situation calls for: `title`: `applies_when`". The title carries the exact codes, so E17 and E71 stay apart. A trigger is **closed-world**: a procedure applies only if the situation states its trigger, so silence means it doesn't apply.
   - **Exclusion: a three-way check per distinct exclusion** (asked once per call, even when several cards share it). An exclusion is **open-world**: silence means it can't be ruled out. When a card's trigger holds and an exclusion "states true", the card is `rejected`. When it "does not say", the card is `unresolved` with `not_stated` ("P02: a visible hydraulic leak").

   The first design asked one three-way question per card ("does the input say this procedure applies?"). Every card whose trigger wasn't mentioned came back "not stated", and the bundled benchmark scored 1/10. Splitting triggers from exclusions took it to 8/10. Measured cost with the default judge: 1 card 300 ms, 8 cards 1.3 s, 16 cards 2.4 s, 32 cards 4.6 s (about 150 ms per extra card).
4. **Precedence** (explicit edges only; nothing inferred from similarity, no causal diagnosis):
   - An `applicable` card is `superseded` when an `applicable` card lists it in `supersedes`. A card that doesn't apply overrides nothing (edges are not transitive through it).
   - An `applicable` card becomes `unresolved` ("may be superseded by X") when a card that would override it is `unresolved` or `unexamined`. Reporting it as final would be a guess.
   - At compile time, self-references and cycles are `ConfigError`s that name the cycle ("supersedes cycle: P01 → P02 → P01"), and edges are stored as integer indices. With no cycles, a non-empty applicable set always leaves at least one final card. (v0.16 allowed a cycle, and two applicable cards erased each other into a settled empty result.)
5. **Exclusive** is checked **after** precedence: more than one final card → those cards `unresolved` (`uncertain`). (v0.16 checked it first, so "A supersedes B, both apply" returned uncertain instead of A.)

**Result** (`r.facts["procedures"]`):

| Field | Meaning |
|---|---|
| `value` | Tuple of final applicable card ids; one id when `exclusive` |
| `complete` | Every eligible card was examined |
| `states` | `{id: state}` for every card |
| `unresolved` | `{id: what it needs}`: missing fields, unstated details, uncertain, may be superseded, not examined |
| `superseded` | `{id: by_id}` |
| `cards` | The original card objects for `value`, payload and `source` included. JSON output carries `id`, `source` and `payload` for each. |

**Atoms and status:**
- `fact has "id"` follows the card state (table above).
- `fact is empty` is TRUE only when the search is `complete` and every eligible card is `ineligible`, `rejected` or `superseded`. It is FALSE when a card is final, and UNKNOWN otherwise.
- **Status:**
  - `ok` when the cases resolve;
  - `needs_input` when a blocking card needs a field or an unstated detail;
  - `uncertain` when it is unsettled, or an incomplete search blocks;
  - `none` only for a switch without cases whose `select` facts are complete and empty.

  **An incomplete search can never return `none`.**
- Cases can test it: `when: 'procedures has "P02"'`, `when: 'procedures is empty'`.

**Not included:** parsing manuals, extracting cards from documents, embedding retrieval (only after BM25 recall is measured and falls short), summarising instructions, and labels or tier 1 on select facts (a later step: compose card conditions from ordinary teachable facts).

### 4.3 Atoms

| Fact kind | Canonical atoms | Atom score `q` |
|---|---|---|
| `check` | `fact` | `p_true` (three-way: p(states true); "does not say" winning makes the atom UNKNOWN, `not_stated`) |
| `select` | `fact has "id"`, `fact is empty` | per card state (§4.2.2); `is empty`: TRUE only for a complete search with no final card; an unexamined card is UNKNOWN (`not_examined`) |
| `pick` | `fact == "x"`, `fact in {x, y}` | `p(x)`, `p(x) + p(y)` |
| `rate` | `fact >= "L"` per level | sum of the probabilities of levels ≥ L |
| exact | `field OP literal`, `field in [...]` | — (UNKNOWN if the field is missing) |
| exact | `exists(field)` | — (**never UNKNOWN**: a missing field gives FALSE) |
| exact | `mentions(field, "MX-120A")`, `mentions(field, ["MX-120A", "Mixer 120 Advanced"])` | — (TRUE if any term occurs with the §3.5 boundaries; a missing field gives UNKNOWN, `missing_input`). It answers "does the text name it", never "what does the text say about it": that remains a fact. |

The compiler canonicalises comparisons:

| Written | Compiled to |
|---|---|
| `!=`, `not in` | `NOT` of the canonical atom |
| `< L` | `NOT (>= L)` |
| `> L` | `>= next(L)` |
| `<= L` | `NOT (>= next(L))` |
| `== L` on a rate fact | `>= L AND NOT (>= next(L))` |

Each canonical semantic atom has thresholds `θ_true > θ_false`, defaulting to 0.90 / 0.10 until tuned at lock time:

```
TRUE if q ≥ θ_true · FALSE if q ≤ θ_false · UNKNOWN otherwise
```

- **Consistency propagation:** rate levels stay ordered; `== x` TRUE implies every `in` set containing x is TRUE; at most one `== x` per pick fact can be TRUE (two → both UNKNOWN).
- **Low answer mass** (< 0.5) makes the fact's atoms UNKNOWN (reason `low_answer_mass`).
- **Eligibility:** an option whose `requires:` is FALSE is removed before scoring, **including from exact-example overrides** (§6.3). If its `requires:` is UNKNOWN, the option is scored but its atoms cannot become TRUE.

### 4.4 Missing data and errors

| Situation | Result |
|---|---|
| A `required` field is missing | `failed`, with `missing=(field,)` |
| A value has the wrong type or enum value | `failed`, with `invalid=(field,)` |
| An exact comparison's field is missing | That atom is UNKNOWN (`missing_input`); `exists(field)` is FALSE |
| A three-way fact's answer is "does not say" | The fact is UNKNOWN (`not_stated`); a blocking case returns `needs_input` naming the fact |
| A semantic fact's `reads` include a missing field | The fact is UNKNOWN (`missing_input`) and the model is not called |
| An option's `requires` reads a missing field | That option can't be ruled out: its TRUE is capped to UNKNOWN (`missing_input`, naming the field), so a case needing it returns `needs_input` instead of guessing |
| Every option of a `pick` is ineligible | Settled without inference by source `rule` (`why: "no eligible option"`): `none`, or FALSE for every option |
| Scoring a fact fails (backend error) | That fact is UNKNOWN (`model_error`); other facts are unaffected |
| An unknown value is masked by a FALSE elsewhere | Irrelevant: nothing is reported |

### 4.5 Three-valued logic and resolution

Kleene logic: `T and U = U`, **`U and F = F`**, `T or U = T`, `U or F = U`, `not U = U`. A confident FALSE masks an uncertain, missing or failed value.

```
for each case, in order:
    TRUE    → ok (answer = case.then, case = case.id)
    FALSE   → continue
    UNKNOWN → blocked
all FALSE  → ok (answer = otherwise)
```

A blocked case is classified from the UNKNOWN atoms left in its **simplified** expression:

| Remaining UNKNOWN atoms include | Status | Reported |
|---|---|---|
| any `missing_input` or `not_stated` atom | `needs_input` | `missing` (blocking fields, and `fact:<name>` for unstated facts), plus `uncertain` facts |
| any `not_examined` atom (and no missing input) | `uncertain` | `uncertain`, with `why: "incomplete search: P07, P11 not examined"`; raise `max_cards` to examine them |
| any `model_error` fact | `failed` | `errors` |
| otherwise | `uncertain` | `uncertain` facts; each fact's `why` says "uncertain" or "low answer mass" |

Errors in facts that don't block the answer never change the result; they appear in `r.errors`.

**`resolve: all`** returns every TRUE case, **always as tuples**: `answer=("legal", "billing")`, and `answer=("general",)` with `case=("otherwise",)` when nothing matched.

### 4.6 Execution

**At `load()`:** parse and canonicalise; resolve names and type-check (errors are a `ConfigError` naming the file, YAML location and a did-you-mean suggestion); group facts by (model, reads); read tiers, thresholds, heads, examples and the measured schedule from the lock (or use defaults).

**Per call** (the plan reference is read once per call):

1. Validate and project the context.
2. Evaluate exact atoms and eligibility.
3. Score the facts the current cases still need, cheapest source first: rule (all options ineligible) → exact-example override → classifier head → judge (one batched call per (model, reads) group). **After each cheap source, the cases are re-simplified, and facts no longer needed are dropped from the judge batch** — even in `eager`, an override that settles the answer means no judge call.
4. Threshold, propagate, re-simplify, resolve.

**Scheduling** decides how much speculative scoring step 3 does:

| `schedule` | Behaviour |
|---|---|
| `eager` | One round: every fact any reachable case needs, in one batch |
| `staged` | Round 1: only the facts of the first unresolved case. If that settles the switch, stop. Round 2: every remaining reachable fact. At most two rounds; a case blocked by missing input stops scheduling. |
| `auto` (default) | **With a lock:** the schedule `ss lock` measured as faster on the labelled inputs. **Without:** eager when every judge runs on an accelerator, staged on CPU. |

Measured on a 16-core CPU with Qwen3-0.6B, a call costs about 150 ms plus about 126 ms per question (1 question 278 ms, 4 questions 657 ms). Staged is 2.4× faster when the first case decides and about 20% slower when a late case does. That's why the choice is measured per switch rather than fixed.

### 4.7 Result

```python
r = route(ticket)
r              # <ok · 'billing' · case refund_other · 2 judged · 181 ms>
r.status       # "ok" | "needs_input" | "uncertain" | "failed"
r.answer       # "billing" | None   (tuples under resolve: all)
r.case         # "refund_other" | "otherwise" | None
r.missing      # ("payment.status",)     needs_input, or a missing required field
r.invalid      # ("payment.status",)     invalid context
r.uncertain    # ("refund_requested",)   blocking unsettled facts
r.blocked_by   # case id
r.errors       # {fact: message}
r.facts        # {"refund_requested": Fact(True, 0.96, judge), ...}
r.cost         # {"rules": 3, "example": 0, "classifier": 0, "judge": 2, "rounds": 1, "tokens": 412, "ms": 181.0}
print(r.explain())
```

**`Fact(value, score, settled, source, why, scores)`** speaks the user's vocabulary:

- `value` is `None` unless **the fact's own answer** is settled: for `check`, `p_true` past a threshold; for `pick` and `rate`, the top option's own probability past θ_true. A case can still fire on a group atom, so `topic in [billing, technical]` can be TRUE at 0.96 while `topic.value` stays `None` (0.48 / 0.48). Cases settle on atoms; `Fact.value` settles on the category. (Before v0.14, the fact reported `'billing'`, settled, at 0.48.)
- `source` is `example`, `classifier` or `judge`.
- `why` is a short reason (`"uncertain"`, `"low answer mass"`, `"missing: payment.status"`, `"error: …"`).
- `scores` is a labelled mapping (`{"true": 0.96, "false": 0.04}`, `{"billing": 0.98, …}`).

Compiled atom keys and raw distributions never appear in the public result.

**`explain()`** prints the decision path:

```
route_ticket → <ok · 'billing' · case refund_other · 0 ms>
cases
  ✗ legal                    legal_threat
  ✗ refund_settled           refund_requested and payment.status == "settled"
  ✗ payment_dispute          exists(payment.status) and disputes_payment_status
  ✓ refund_other             refund_requested
  · tech_urgent              topic == "technical" and urgency >= "today"
  · otherwise                → 'general'
facts
  ✓ legal_threat             False  0.97  judge
  ✓ refund_requested         True  0.96  judge
  · topic                    not needed
```

`explain(raw=True)` appends each compiled atom's state, for debugging the compiler.

## 5. Context, grammar, revisions

### 5.1 Context

- **Typed values** are validated against `fields` and used by exact atoms and eligibility.
- **One rendered string per reads group:**
  - one line per field, in declared order, inside `<input>…</input>`
  - an `[author]` / `[source]` prefix when declared
  - each value truncated per field (`max_chars`, `truncate`)
  - tokenised once per call **per tokenizer**
- **Canonical key** — `canonical(fact, ctx)`:
  - the fact's **untruncated, typed projection** of its `reads`
  - serialised as JSON with sorted keys, strings in Unicode NFC, and nothing else changed (no lowercasing, punctuation or whitespace changes)

  This one function is used for example ids, exact-example overrides and test-file grouping. There is no fuzzy matching.
- The caller's data is never deep-copied.

### 5.2 Grammar (allowlisted)

```
expr := or ; or := and ("or" and)* ; and := not ("and" not)* ; not := "not" not | "(" expr ")" | atom
atom := FACT | FACT ("=="|"!=") STR | FACT ("in"|"not in") "[" STR ("," STR)* "]"
      | FACT ("<"|"<="|">"|">=") STR | FIELD ("=="|"!="|"<"|"<="|">"|">=") LIT
      | FIELD ("in"|"not in") "[" LIT ("," LIT)* "]" | "exists" "(" FIELD ")"
      | "mentions" "(" FIELD "," STR ")" | "mentions" "(" FIELD "," "[" STR ("," STR)* "]" ")"
      | FACT "has" STR | FACT "is" "empty"
LIT  := STR | NUMBER | "true" | "false"
```

There is no `eval`, no attribute access and no other functions. Unknown names and type mismatches are load-time errors.

### 5.3 Revisions

| Hash | Covers | When it changes |
|---|---|---|
| `meaning_rev` (per fact) | statement or question, `excludes`, options, levels, `requires`, **`reads` and the rendering of those fields (`author`, `source`)** | Older examples and test-file labels for the fact are **stale** (ignored, with a warning); its tier-1 head is dropped. A label describes the fact *as seen through its projection*: "the customer threatens legal action" read from the customer's message is a different labelled question from the same sentence read from the agent's reply. Changing the model keeps labels; changing what the fact reads does not. |
| `switch_rev` | cases, `otherwise`, `resolve` | Answer labels in the test file are stale (ignored, with a warning) |

Labels written by the tools carry the revisions they were made under: `correct(case=…)` records `switch_rev`, and `ss label --to test` records each fact's `meaning_rev` under `revs`. Hand-written test lines without them are always trusted. `ss tune`, `ss test` and `ss lock` report how many labels they ignored.

There is no separate execution revision: a lock embeds its spec, examples, heads and thresholds and pins its models (§9), so tuned thresholds can't drift from what they were tuned for.

---

## 6. Examples and corrections

### 6.1 Examples file

```json
{"id":"e:7f3a9c21b0d4e6f8","fact":"legal_threat","meaning_rev":"a41c09e2","input":{"subject":"…","body":"…"},"value":true,"created":"2026-10-05T14:20:03Z"}
```

- `id = sha256(fact, meaning_rev, canonical(fact, input), value)[:16]`.
- **Append-only.** A later line with the same `(fact, meaning_rev, canonical key)` replaces the earlier one.
- Examples from older `meaning_rev`s are ignored, with a warning. `ss label --stale` re-confirms them quickly.

### 6.2 Selection for the judge's prompt

| Setting | Default |
|---|---|
| `max_examples` | **0** (no examples in the judge's prompt) |
| `max_example_tokens` | 320 per fact |

**Why 0.** Measured on held-out minimal pairs (§14.6), prompt examples gave **no consistent improvement** (one random draw scored 0.83 against 0.79 without examples; the others scored lower) and cost 1.7–2.5× latency:

| Judge | No examples | 4 random examples (3 draws) | 4 nearest examples (balanced) |
|---|---|---|---|
| Qwen3-0.6B | **0.79** | 0.58 / 0.83 / 0.75 | 0.75 |
| Qwen3-1.7B | **0.92** | 0.88 / 0.75 / 0.88 | 0.83 |

These were measured on the old 6-family set (§14.6) with Qwen judges. The conclusion stands as a default, and `bench/models.py` re-checks it against the new default judge.

Examples remain valuable elsewhere: exact overrides, tuning labels, classifier training, `ss diff`. Setting `max_examples > 0` re-enables them in the prompt (deterministic balanced subset, newest first). Retrieval-based selection was tested and didn't help either.

### 6.3 Corrections and exact-example overrides

```python
route.correct(ctx, fact="refund_requested", value=False)    # appends to route_ticket.examples.jsonl
route.correct(ctx, fact="refund_requested", value="false")  # strings true/false/yes/no are accepted
route.correct(ctx, case="refund_other")                     # answer label → test file + derived fact labels
```

**Exact-example override (default on).** If `canonical(fact, ctx)` equals the key of an active example for the same `fact@meaning_rev`, and the labelled value is eligible, the fact takes that value directly (`source="example"`, no inference).

**What a correction promises:** it fixes that fact for that projected input, not necessarily the final answer.

| Switch state | Effect of `correct()` |
|---|---|
| Draft | Appended to the file **and** added to the loaded snapshot incrementally (only that fact's examples, overrides and prompt selection are recomputed) |
| Locked | Appended; the result says `effective="after next ss lock"` |

**Answer → fact labels** come only from direct `check` atoms (TRUE in the chosen case, and, **under `resolve: first` only**, FALSE in earlier single-atom cases) and positive `pick` equality. Under `resolve: all`, an earlier case not being named proves nothing, so only the named case's direct atoms are derived; anything else needs explicit fact labels. Similar inputs are never relabelled automatically.

## 7. Tier 1: specialised facts

A classifier head on a frozen embedding: one shared embedding pass per input, microseconds per fact. **It is a speed lever for topical facts only, and it settles nothing until `ss lock` has tuned it.**

**What the measurements show** (held-out minimal pairs, §14.6):

| Classifier | Held-out accuracy | When it settles |
|---|---|---|
| multilingual-e5-small + logistic regression, 3–100 examples/class | 0.33–0.50 (chance) | never: probabilities stay at 0.42–0.53 |
| Same, standardised features | 0.46–0.50 | settles 4–16 of 24, and is **wrong 67–79% of the time** |

Our multilingual-e5-small setup failed these minimal pairs: "refund me" and "don't refund me" are neighbours (mean pairwise cosine 0.969), so a head on them couldn't separate the pairs. This is a result for this embedder and these pairs, not a law about sentence embeddings. Whether the classifier earns its keep on **topical** facts, its intended use, is still unmeasured (§14.6, topical learning curve). Its 0.90 / 0.10 default thresholds hid this; standardising would have made it confidently wrong.

**Rules that follow:**

- **Untuned classifiers never settle.** Their default thresholds are `(None, None)`. `ss lock` tunes them on the tune split of `*.test.jsonl`, separately from the judge's. A side that can't reach the fact's precision target stays unset, and the lock report warns: "classifier trained but not tuned … it defers to the judge".
- **Use `tier: 1` for topical facts** (product area, language, intent families), not for facts decided by negation or who-did-what.
- **`ss specialize` runs the real runtime; it doesn't re-implement it.** For each fold:
  1. Train the candidate head (`embedding` and `lexical`) on the training examples.
  2. Tune its thresholds by cross-validation inside the training folds.
  3. Install head and thresholds in a temporary snapshot whose exact overrides exclude the held-out fold.
  4. Run the normal executor on the held-out inputs.

  It reports, measured:
  - the source that settled each input (example / classifier / judge);
  - correct and wrong settled answers;
  - uncertain and needs-input counts;
  - the actual judge-fallback rate;
  - the timed end-to-end latency, warm.

  Its plain-sentence verdict weighs wrong answers first, then coverage and latency. v0.16 used its own acceptance rules and computed latency from component timings.
- **The lexical head never loads the embedder.** Model roles in a lock follow the head kinds actually used, and `ss specialize --head lexical` skips embeddings.

### 7.1 Teaching path (one documented workflow)

```python
refund = ss.Check("The customer requests money back for a charge already made.")
sw = refund.save("switches/refund.yaml")                       # §3.2.1: same Judgment, now saved
sw({"message": "Reverse that charge"}).facts["refund"].value
sw.correct({"message": "Reverse that charge"}, fact="refund", value=True)
```

`save()` returns the `Switch`, so everything below applies unchanged. There is one teaching system, and no YAML editing is needed to start.

```bash
# 1. correct as you go: each correction fixes that exact input now (draft) or after the next lock (locked)
# 2. label separate inputs for tuning and checking
ss label support_route new_inputs.jsonl --to test     # y / n / u ("not stated") for three-way facts
# 3. compare: does a classifier help this fact? (reports only; changes nothing)
ss specialize support_route billing
# 4. enable it where it helps: add `tier: 1` to the fact in the YAML
# 5. train and tune the executable version (the only step that trains)
ss lock support_route
# 6. unchanged call; r.facts["billing"].source says example / classifier / judge
```

- **Inline calls don't learn.** `check` / `pick` / `rate` take `examples=` per call but never train. Teaching is a saved-switch feature.
- **No automatic promotion.** Enabling tier 1 stays a deliberate edit, after a report.
- **Fallback:** whenever a still-needed atom is UNKNOWN, the judge scores the fact and replaces the whole distribution.
- **No silent truncation.** If a reads group's rendered text is longer than the embedder's window (512 tokens for e5-small), its classifier facts defer to the judge for that call (`why: "too long for the classifier"`). The embedder reports truncation instead of slicing silently.
- Heads are multinomial (binary for `check`), float32, embedded in the lock; training uses scikit-learn, inference is NumPy.
- **Two head features:** `embedding` (default; the shared e5 vector) and `lexical` (TF-IDF word 1–2-grams plus character 3–5-grams of the rendered reads, hashed to 2^18 features, with a sparse linear model). The lexical head keeps codes and exact wording that pooled embeddings blur, and costs no model call. Neither feature fixes negation. `ss specialize` reports both, and the fact picks one with `head: lexical | embedding`.
- **Admission rule:** latency is not the obstacle. A classifier at about 9 ms against a 228 ms judge pays for itself once it settles more than about 4% of calls, and fallback calls cost about 4% more. **A head is enabled only if, on held-out inputs, it adds no more wrong accepted answers than the judge alone at the same thresholds.**
- A `meaning_rev` change drops the head; changing the embedding model or its text prefix invalidates the lock (§9).

### 7.2 Fine-tuning the judge (offline subsystem)

A fine-tuned judge is **just another model file for the `judge` role**. The runtime, prompts and API don't change. It answers whether a small, fast judge (Qwen3-0.6B, about 100 ms on short inputs) can reach the default judge's precision.

**Data (licences checked):**
- **Never from Jev.** TypeSafe's Master Customer Agreement §2.3(b) forbids using its outputs "to perform model distillation, train a model to imitate the output of the Services, or develop … a similar or competing product". Jev results never enter training data.
- **Gold labels:** MASSIVE (Amazon, CC-BY-4.0, real utterances in English and Portuguese with intents). Intents become claims, with related intents as hard negatives. This gives scale, but no negation or role phenomena.
- **A local teacher:** Gemma 4 E4B (Apache-2.0, outputs free to use; 0.93 on the fresh suite) labels generated minimal pairs across the §14.6 phenomena tags. Answers below its tuned threshold are dropped, and a sample is spot-checked by hand.
- **Your own labels:** `ss correct` / `ss label` data, once real traffic exists.
- **Splits are by sentence family,** and translations stay with their source.

**Training:**
- LoRA on the base model. Training prompts are produced by **`Judgment.questions`**, so training and runtime prompts are byte-identical (this needs the single definition layer, §12.1).
- Loss only on the answer token (`true` / `false`, three-way letters, pick and rate letters), never on prompt tokens.
- All question kinds are included, or the untrained ones degrade.

**Export:** merge the LoRA → convert to GGUF → quantise to the same 4-bit format as the baseline (`llama_model_quantize` is in our binding) → `ss doctor --quality`.

**Tests:**
- the fresh suite (headline);
- the selection suite (regression);
- a **changed-definition set** (the same inputs with an edited claim or exclusion: a fine-tuned judge must still follow the edited rule, or it has become a fixed classifier);
- the identifier and unstated benchmarks;
- latency on short (20-token) and long (261-token) inputs.

**Gate:** beat Gemma 4 E2B's fresh-suite accuracy (0.856) and untuned precision (88.9%) at Qwen3-0.6B speed, after quantisation. A second variant, Qwen3-1.7B with LoRA, targets its confidence errors.

**Hardware:** LoRA on a 0.6B model runs on this CPU in roughly a few hours per epoch at about 2k examples (an estimate), or about an hour on a rented GPU.

**Measured alternative, kept as an experiment:** a linear probe on the **judge's own answer-position hidden state** is question-conditioned. On Qwen3-1.7B it ranks the old 6-family held-out set almost perfectly (AUC 0.99–1.00); on Qwen3-0.6B it reaches 0.76–0.81. (The Gemma 3 numbers once quoted here were measured with the Qwen chat template and are void, §14.6.) It costs a judge forward pass, so it improves precision on the large judge but doesn't make anything cheaper (`bench/models.py --probe`).

## 8. Test, tune, diff

All three work on **`*.test.jsonl`**: labelled inputs, with fact labels and/or case labels.

```json
{"input":{"subject":"…","body":"…"},"facts":{"refund_requested":true,"topic":"billing"},"case":"refund_other"}
```

The file is split once, deterministically by canonical input hash, into **tune** (50%) and **check** (50%).

| Command | Writes | Does |
|---|---|---|
| `ss tune route_ticket` | **nothing** | Shows the thresholds `ss lock` would choose on tune, separately for the judge and the classifier heads, and their effect on check |
| `ss test route_ticket [--data file] [--intervals]` | nothing | Runs the switch (draft or lock) and reports in plain sentences, with **unseen** and **exact-taught** inputs separated. **Inputs that tuned the lock are always excluded:** the lock stores their canonical keys (`tuned_keys`), so adding lines to the test file never brings them back into the report ("excluding 30 inputs used for tuning"). v0.16 matched the whole file's digest, so appending one line put the tuning half back into "unseen" (35 → 61 items). `--intervals` adds exact 95% Clopper–Pearson intervals for accepted accuracy and per-fact precision (needs `scipy`) |
| `ss diff route_ticket` | nothing | Runs the **lock** and the **edited draft** and lists every input whose **observable outcome** changed (old → new, with the label where known). The outcome compared is: status and answer (with cases); status and settled fact values (without cases); applicable ids and `complete` (select facts). Inputs come from the test file plus **every** example record, stale ones included as inputs, not as truth. v0.16 compared case answers only, and after a meaning change it compared 0 inputs. |

Threshold scan for one atom (run inside `ss lock`; previewed by `ss tune`):

1. Start at the strictest threshold that accepts ≥ 20 tune items. This counts items only, without labels.
2. Relax toward 0.50 while precision on tune stays ≥ the fact's `target` (default 0.95).
3. Set `θ_false` the same way from 0.01 upward.
4. A side that can't meet the target stays unset.

`ss test` output:

```
route_ticket (lock 4e1a…), 212 labelled inputs, check split (the lock was tuned on the other half):
  Unseen (180): ok 124 (69%): 120 correct, 4 wrong [95% interval 92%–99%]. 44 uncertain, 12 needs_input, 0 failed.
  Exact-taught (32): 32 settled, all as labelled. (Checks that corrections stick, not understanding.)
  legal_threat (unseen): 16 actually true — 13 accepted as TRUE (all correct) [95% interval 75%–100%], 1 settled as FALSE (silent miss), 2 left uncertain.
  Note: with fewer than ~60 accepted inputs, 95% and 90% precision can't be told apart.
  Warning: 3 stale answer labels ignored (the cases changed since they were labelled)
```

Thresholds are chosen on tune and reported on check, and taught inputs never count as evidence of understanding.

---

## 9. Lock file

`ss lock route_ticket` is the **only writer** of frozen behaviour. In order:

1. Compile the YAML.
2. Select prompt examples.
3. Train heads for `tier: 1` facts.
4. Tune thresholds on the tune split (if a test file exists).
5. **Choose the schedule** (`auto` only): run eager and staged on up to 50 labelled inputs and keep the faster.
6. Assemble the frozen plan and **run `tests` against it**; any failure aborts.
7. Write one self-contained file atomically.

```json
{
  "schema": 2, "id": "a437467a6c0f", "switch": "route_ticket", "switch_rev": "…", "spec": { "…" },
  "models": {
    "judge": {"file": "gemma-4-E2B-it-Q4_0.gguf", "sha256": "…", "template": "chat@2:0a2c8073"},
    "embed": {"file": "multilingual-e5-small-Q8_0.gguf", "sha256": "…"}
  },
  "schedule": "staged",
  "facts": {"refund_requested": {"meaning_rev": "…", "tier": 1, "examples": [ … ], "prompt_examples": [ … ],
                                 "head": { … float32 … },
                                 "thresholds": {"judge": { … }, "classifier": { … }}}},
  "tuned_on": "…", "tests": "1/1 passed against this plan", "created": "…"
}
```

**A lock pins its execution context.** At load, every model role in the lock is compared with the runtime's binding: the file's sha256 (cached by path, size and mtime), plus the prompt template version for judges.

- **On mismatch, loading fails** with a `ConfigError` naming both models: "Re-run `ss lock`, or load with allow_model_change=True."
- **With `allow_model_change=True`**, the tuned thresholds of the affected facts fall back to the defaults (and, for `embed`, the classifier heads are dropped), with a warning.

Other rules:

- With a lock, `load()` executes it exactly: embedded examples, heads, thresholds, schedule. `load(path, draft=True)` runs the edited YAML. They are never mixed.
- Rollback is `git checkout` of the lock file.
- Decisions are reproducible on the same machine class, model file and runtime build.

## 10. Models and backend

### 10.1 Roles and defaults

| Role | Used for | Default |
|---|---|---|
| `judge` | Scoring questions | **Gemma 4 E2B instruct, GGUF `Q4_0`** (`ggml-org/gemma-4-E2B-it-GGUF`, Apache-2.0) — `ss.DEFAULT_JUDGE` |
| precise judge | Opt-in: `--precise`, or `Model(ss.PRECISE_JUDGE)` | Gemma 4 E4B instruct, Google's official QAT `Q4_0`: the most precise, 2.2× slower, 7.5 GB resident |
| fast judge | Opt-in: `--fast`, or `Model(ss.FAST_JUDGE)` | Qwen3-0.6B `Q4_K_M`: 2.3× faster, 1.5 GB; settles few hard cases and defers the rest |
| `embed` | Classifier facts | multilingual-e5-small Q8_0, with the `query: ` prefix (topical facts only, §7) |
| `strong` | Facts that declare `model: strong` | Unset (bind it to use it) |

**Why Gemma 4 E2B Q4_0 (measured on the 240-item suite, §14.6).**
- **It had the best result at default thresholds among the tested candidates on this suite:** accuracy 0.95, AUC 0.986, calibration error 0.038. (Qwen3-0.6B Q4_K_M also clears 95% precision at 0.90 / 0.10, but settles only 14% of items.)
- **At 0.90 / 0.10 it settles 92% of items, 97.7% of them correctly,** with no labels or tuning.
- **It is faster than the previous default:** 228 ms per check and about 128 ms per question inside an `ask` on a 16-core CPU.
- **The cost is memory:** 3.8 GB resident, because Gemma 4's "E" models carry per-layer embedding tables (2B compute, 5B weights).
- **Google's official QAT Q4_0** (`google/gemma-4-E2B-it-qat-q4_0-gguf`) performed alike **on this suite**: 0.942 accuracy, the same coverage and precision, 3.2 GB file, 264 ms. That is one 240-item comparison, not a general equivalence. Either is a valid pin; the ggml-org file is the default because it is smaller and faster here.
- **Caveat:** the suite chose this default, so it no longer confirms it independently. A fresh held-out set (§14.6) and real traffic (E0) must.

**Why not Qwen3-1.7B any more.** It is overconfident on indirect phrasing: 85% of its answers sit at ≥ 0.999 confidence. At default thresholds it settles 97% of items but only 84.5% correctly, and no threshold or per-fact offset gets it to 95% on more than 44%. Its errors are true claims scored at 0.000–0.001: "Me transfere para um humano" for "asks to talk to a human agent", or "When I click save, my changes disappear" for "reports that the product is not working". A switch running on it would silently accept about 1 wrong answer in 7 on such inputs.

**Measured alternatives:**
- **Qwen3.5-2B Q8_0:** the best Qwen at 0.90 and well calibrated (ECE 0.044). Per-fact calibration from labels lifts it to 0.91 and 88% coverage at 95%, but it is 2.4× slower than the default and still less precise.
- **Qwen3.5-0.8B:** no better than Qwen3-0.6B (0.71–0.77), and worse at picks.
- **Gemma 3 1B:** 0.70 once given its own template, but overconfident (71% precision at default thresholds).

**Other rules:**
- A GGUF chat model works through the generic adapter (§10.3) **if** its embedded template renders with a system and a user message, its answer tokens (`true` / `false`, `A`–`T`) are single tokens, and `ss doctor` passes. Anything else is refused at load with the reason. Hybrid models (Qwen3.5) use shared-prefix scoring once `ss doctor` passes D1 (§10.4).
- Models load on first use. Counting tokens for prompt selection uses a **vocab-only** load (≈150 ms), never the inference context, so `load()` doesn't initialise the judge.
- Quantisation, the chat template and the embedder's text prefix are part of model identity (§9).
- **A default judge must pass the quality probe** (§10.8): at least 95% precision at 0.90 / 0.10 on the suite, with at least 60% coverage.

### 10.2 Protocols (the replaceable boundary)

**`Judge`** is the only object the engine and the tools talk to. It owns an adapter and a scorer, and no caller sees tokens, `ScoreRequest`s or backend internals:

```python
class Judge(Protocol):        # the semantic boundary: text and questions in, scores out
    identity: dict            # pinned by locks
    accelerated: bool         # drives schedule: auto
    max_slots: int
    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]: ...  # Scored: scores + tokens
    def fit(self, text: str, limit: int) -> tuple[str, bool]: ...   # candidate / example text within this judge's budget
    def count(self, text: str) -> int: ...                           # example budgeting (prompt examples)

class LlamaJudge:             # the decoder implementation: adapter + scorer, tokens stay inside
    ...                       # + score(), count(), selftest()

@dataclass(frozen=True, slots=True)
class Scores:
    probs: np.ndarray | None  # normalised over the question's answer slots
    mass: float | None
    error: str | None
```

**`Judge` is a protocol, not a class hierarchy.** The engine, inline primitives and tools use only `score_many`, `fit`, `count`, `max_slots`, `identity` and `accelerated`. Tokens, adapters and `ScoreRequest`s are internal to `LlamaJudge`. A non-decoder judge (a cross-encoder, a remote API) implements these six members directly and never impersonates a token scorer. The answer-mass rule (§3.2) is applied by the judge that has answer tokens; a judge without them reports `mass=None`, which is never "low".

Below `LlamaJudge`, the backend contract is unchanged: `Scorer.score(Sequence[ScoreRequest]) -> ScoreBlock` and `Embedder.embed(texts) -> float32[n, d]`. `Question`, `ScoreRequest` and `ScoreBlock` are the transport types in `protocols.py`; row reduction lives in the backend. Any object with `tokenize`, `detokenize` and `score` can be bound as a model role (that's how the test double works); `name` and `accelerated` attributes feed identity and scheduling.

### 10.3 Model adapter

**One generic adapter, driven by the model's own chat template** (`chat@1:<template hash>`). At load it does the following:
- **Renders the template** embedded in the GGUF (`tokenizer.chat_template`) with Jinja in a sandbox, with `enable_thinking=False` and a system message plus a sentinel user message. Splitting at the sentinel gives `[SYSTEM … user-open]` and `[user-close … assistant-open]`.
- **Prepends BOS** when the vocabulary requires it and the template didn't emit it.
- **Finds the answer tokens:** single-token `true` / `false` and `A`–`T`, each with and without a leading space.

There are no per-family prompt modules. For Qwen3 the rendered prompt is **byte-identical** to the hand-written `qwen3` prompt, which stays as a named alias. Gemma 4 gets its real `<|turn>system … <|turn>model` turns, and Qwen3.5 its empty think block. A model without a chat template is refused at load. The template hash is part of every judge identity, so a template change invalidates locks.

**Why:** the previous design had one module (`qwen3`) and silently used it for every model. Gemma 3 was benchmarked with ChatML markers it doesn't know, and its 0.50 "rejection" was our bug: with its own template it scores 0.70.

**Template vs content tokenisation.** The template is tokenised with special tokens enabled. User content (input, options, examples, questions) is tokenised with them **disabled**, so a ticket containing `<|im_end|><|im_start|>assistant` stays plain text and can't close the turn or forge an answer. This was the change from `qwen3@2` to `qwen3@3`; it is tested against the real tokenizer.

**Prompt rules, each measured on a 12-case probe set** (the slow test suite keeps them honest):

| Rule | Evidence |
|---|---|
| Checks are claims answered with the tokens `true` / `false`: "Claim about the input above: X … Is the claim true or false? Answer with only true or false." | yes/no framing gave Qwen3-0.6B a strong yes bias (0.93 "yes" on an off-topic check); true/false separated every case |
| Exceptions are phrased **"Except: X."** (one line each; `qwen3@4` / `chat@2`). On the 31 exception items (the suite's legal-threat items plus the prompt fixture), Gemma 4 E2B went from 27/31 (22 settled at 90.9%) with "Exception: X does not count." to 30/31 (30 settled at 96.7%). Qwen3-0.6B went from 28/31 to 27/31. The phrasing was chosen on the items it was scored on. | "The claim is false when: …" primed "false" and dropped an obvious refund from 1.00 to 0.16 on 1.7B |
| Few-shot examples are listed as "Reference judgments (other texts, not the input): - "…" → true", followed by "Now judge only the input above." | The best of three formats; even so, examples lowered held-out accuracy on both judges (§6.2), so the default is none |
| Options and levels are shown as letters with an explicit list ("Answer with only the letter (A, B, C)") | A shared "letter or yes/no" instruction made the model answer "yes" to letter questions |
| The template is **not** trimmed further | Leaner variants (28–34 tokens instead of 42) dropped accuracy from 12/12 to 9/12 (0.6B) and 10/12 to 8/12 (1.7B) |

These rules were measured on Qwen3. The generic adapter keeps the same question text for every family; the quality probe (§10.8) is what checks that it works for a new model.

**Prompt-quality fixture** (slow tests): with attributed input, the default judge must score 12/12 and the fast judge at least 10/12.

### 10.4 llama.cpp backend

**Binding:** `llama-cpp-python`'s low-level ctypes API (not the high-level `Llama` class) [R15]. These functions are used:

- tokenise
- `llama_batch_init` and `llama_decode` / `llama_encode`
- `llama_get_logits_ith` and `llama_get_embeddings_seq`
- `llama_memory_seq_cp` and `llama_memory_seq_rm`
- `llama_model_has_encoder` / `has_decoder` / `is_hybrid`

Symbols are checked at startup; a missing one switches off that capability. A custom binding is written only if this one demonstrably limits us.

**Hybrid (recurrent + attention) models.** `llama_memory_seq_cp` copies a whole sequence's recurrent state copy-on-write, so the shared-prefix flow works unchanged when:
- `n_seq_max` is at least 2 + the number of concurrent questions;
- the KV cache is unified;
- sequences are only ever removed whole (partial `seq_rm` is unsupported for recurrent state).

Measured on Qwen3.5-2B: D1 shared = reference is exact (max difference 0.0), and a 6-question ask takes 1.28 s instead of 3.62 s. **Shared scoring is enabled for hybrid models after `ss doctor` passes D1** (and on first load, the backend runs D1 once and caches it). They no longer default to the reference path.

**Input caching (the scoring layer keeps the input).** Today every call removes sequence 1 (system prompt + input) and re-reads the input. On a 261-token ticket that costs:

| | Gemma 4 E2B | Qwen3-0.6B |
|---|---|---|
| 1 question | 1.32 s | 0.72 s |
| same input scored twice, two calls | 2.73 s | 1.47 s |
| 2 questions in one call | 1.65 s | 0.84 s |

So each repeated round re-reads about 1.1 s (Gemma) or 0.6 s (Qwen). That hits staged round 2 (the CPU default), pick and focus fallbacks, and `specialize`.

The fix:
- **Exact reuse:** keep sequence 1 after a call. A request with the same prompt and input skips straight to its questions.
- **Common-prefix reuse (attention models only):** when the next input shares a beginning with the cached one, remove the cache from the first differing position and decode only the rest. This is the llama.cpp server's prompt-cache approach. Hybrid models can't trim recurrent state, so they reuse on an exact match only.
- **One cached input per context.** It's freed with the context, and D1/D3 cover it (reused tokens are identical tokens).
- **Order `reads`** so that facts' projections of the same record share a beginning (fields in declared order, the most widely read first). Then the common-prefix reuse applies across facts.
- **Staged scheduling's advantage is re-measured with caching on:** a second round then costs only its questions, so staged should win on CPU whenever the first case decides.

**Not levers:**
- **Flash attention:** 1.32 → 1.63 s on Gemma, 0.72 → 0.65 s on Qwen. Keep llama.cpp's automatic setting.
- **A shorter question template:** measured to cost accuracy, §10.3.

**Separate limits:**

| Limit | Meaning |
|---|---|
| Request batch (Python) | Questions per `score()` call (§12.3) |
| `n_batch` (native) | Most tokens per `llama_decode` call. Larger work is **chunked**. |
| `n_ctx` (KV capacity) | Must hold `|system| + |state| + Σ|suffix_i|`. In the unified KV cache, copying a sequence for an attention model tags the existing prefix cells, so the prefix is not duplicated. |
| `n_seq_max` | ≥ 2 + concurrent questions (recurrent models need a state slot per sequence) |

If the capacity would be exceeded, the request is split, or returns `failed(limit)`.

**Judge flow (shared prefix):**

```
seq 0: SYSTEM decoded once per context, never modified

with context_lock:
  try:
    seq_cp(0 → 1); decode STATE on seq 1 in n_batch chunks (no logits)      # check every return code
    seq_cp(1 → 2+i) for each question i
    pack all suffix tokens (seq 2+i, positions continuing from the state) into decode CHUNKS ≤ n_batch;
      flag logits only on each suffix's last token;
      record for question i: (chunk k, local batch index j)
    for each chunk k:
        rc = decode(chunk k); if rc != 0 → mark the affected questions' rows as errors
        for each question i that ends in chunk k:
            row = view(llama_get_logits_ith(ctx, j), n_vocab)       # valid only until the next decode
            reduce(row) → write into the compact ScoreBlock row i
  finally:
    seq_rm(1), seq_rm(2+i) for all i
```

Reduction, per row:

```
lse_all   = logsumexp(row)
slot_lp_j = logsumexp(row[variants of slot j])
log_mass  = logsumexp(slot_lp) − lse_all
logp      = slot_lp − logsumexp(slot_lp)
```

- Rows are vocabulary-sized (~152k floats for Qwen3). They are reduced through views **before the next decode**, and only the compact result is copied.
- **No result keeps a view into native memory.**

**Embedding flow (tier 1):**

- A separate context per `embed` model, with the model's pooling.
- `llama_encode` if the model has an encoder and no decoder; otherwise `llama_decode`.
- `llama_get_embeddings_seq`, then L2-normalise.
- One embedding per reads group per call.

### 10.5 Concurrency and lifetime

- **Weights** are shared. **Each inference context** is used by one operation at a time under its lock.
- **`close()` acquires the context lock** before freeing, so it waits for an in-flight operation; `Runtime.close()` closes every judge, embedder and vocab-only tokenizer.
- **A failed context creation frees the weights** it loaded.
- **Detokenisation uses a call-local buffer**, never shared state.
- Temporary sequences are always removed in `finally`; switch calls read their plan reference once.
- **Native logs are silenced** unless `SEMANTIC_SWITCH_DEBUG=1`.

### 10.6 `ss doctor`

`ss doctor` calls `judge.selftest()`:
- **Backend checks** that need internals run inside `LlamaScorer.selftest()`: D1 shared = reference, D3 no leakage, D6 chunking.
- **Generic checks** run through the `Judge`: D2 batching, D4 answer tokens, D5 template mass.

Results are cached in `~/.cache/semantic_switch/doctor.json` under **model identity + backend (cpu / vulkan / …) + device + llama.cpp build**, and applied when that same combination loads again. A pass on CPU never authorises a GPU backend or a new build. Quality-probe results are cached separately, keyed by the same plus the suite version.
- **Shared-prefix scoring is decided by D1 and D3 alone** (shared = reference, no leakage). A D1 or D3 failure keeps the backend on the reference path across processes; a pass enables shared scoring, hybrid models included. D2, D4, D5 and D6 report problems but don't affect sharing: chunking applies to both paths.
- **D6 (chunking)** requires the same top answer for every question and at most 0.25 probability drift when the native batch is forced down to 7 tokens. At Q4, llama.cpp uses different matrix kernels for tiny batches, and Qwen3-0.6B Q4_K_M drifted 0.164. A real chunking bug (wrong positions) produces garbage, not drift.

Measured: D1–D6 pass for Gemma 4 E2B (sliding-window attention) and Qwen3.5-2B (hybrid).

`ss doctor --quality` also runs the quality probe (§10.8).

### 10.7 Devices and acceleration

The reference laptop (Core Ultra X7 358H) has an **Intel Arc B390 integrated GPU** and an Intel NPU. The current llama-cpp-python build is CPU-only (`llama_supports_gpu_offload() = False`), so every number in this spec is CPU.

- **GPU via Vulkan.** This is the path for this machine and most non-NVIDIA laptops. llama-cpp-python is rebuilt with `CMAKE_ARGS="-DGGML_VULKAN=on"`, which needs `cmake`, `vulkan-headers` and `glslc` (`shaderc`). Our workload is almost all prompt reading, which is arithmetic-bound, which is exactly what a GPU accelerates. The size of the gain on this iGPU is unmeasured.
- **`device="auto"` says what it did.** `Runtime.device(role)` and `ss doctor` report the requested device, the selected backend, and the **actual** layer offload. When `auto` falls back to CPU, it says so once ("GPU requested but this llama.cpp build has no GPU backend; running on CPU"). `accelerated` means layers actually offloaded, not "the build could".
- **Unknown names fail.** Adapter and device names are validated with a `ConfigError` and suggestions.
- **NPU: deferred.** It needs a different runtime (OpenVINO) and model format; it isn't reachable through llama.cpp. Revisit only after the GPU is measured.
- **Batching independent inputs into one decode** (`many()` today runs requests one after another inside a call) is deferred until GPU throughput is measured.

### 10.8 Quality probe and quantisation policy

**The quality probe** is the held-out suite of §14.6 (240 check items, 24 picks), shipped inside the package. `ss doctor --quality [--judge SRC]` scores it once per model identity, caches it next to the doctor results, and reports:

| Metric | Why |
|---|---|
| Accuracy, AUC, ECE | Understanding and calibration |
| Coverage and precision at the untuned 0.90 / 0.10 | What a draft switch will actually do |
| Coverage at 95% precision | Headroom for `ss lock` tuning |
| Per-phenomenon and per-language accuracy | Where it breaks (negation, role, reported speech, …) |
| ms per check, ms per question in an ask, resident memory | Cost |

**Untuned thresholds are only trusted for a measured model.** When a judge's cached probe shows < 95% precision at 0.90 / 0.10, `load()` and inline calls warn once: "this judge settles untuned answers at 85% precision on the quality probe; run `ss lock` with labelled data, or use the default judge". The probe itself never tunes anything.

**Quantisation is chosen per family, by measurement.** The policy for the shipped defaults, from §14.6:

| Family | Policy | Evidence (vs Q8_0 on the suite) |
|---|---|---|
| Gemma 4 | `Q4_0` (ggml-org or official QAT) | Mean probability shift 0.035, 5–7 of 240 answers flip, accuracy 0.963 → 0.950. QAT and plain Q4_0 performed alike here. |
| Qwen3 | `Q4_K_M` for the fast judge only | Shift 0.10–0.11, 27–29 flips; for 1.7B accuracy 0.84 → 0.76 |
| Qwen3.5 | `Q4_K_M` or better; **never `Q4_0`** | Q4_0: shift 0.27, 93 flips. Q4_K_M: shift 0.09, 26 flips. Stock Q4_0 quantises the delta-net tensors (`ssm_alpha`, `ssm_beta`, `ssm_out`) to 4 bits. |

**4-bit is faster on CPUs too:** llama.cpp repacks `Q4_0` / `Q4_K` into AVX2 8×8 layouts and has no such path for `Q8_0`. Gemma 4 E2B: 228 ms at Q4_0 vs 440 ms at Q8_0. A quantisation change changes the file hash, so locks force a re-lock (§9).

## 11. Interfaces

### 11.1 Python

```python
import semantic_switch as ss

with ss.Runtime(models={"judge": ss.Model("ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf")}) as rt:
    rt.check(state, statement, *, excludes=(), fields=None, examples=None, min_score=None, model="judge")
    rt.pick(state, question, options, *, describe=None, eligible=None, allow_none=False, shortlist=None,
            fields=None, examples=None, min_score=None, model="judge")
    rt.rate(state, question, levels, *, fields=None, examples=None, min_score=None, model="judge")
    rt.ask(state, {"legal": ss.Check(...), "topic": ss.Pick(...), "urgency": ss.Rate(...)})

    route = rt.load("switches/route_ticket.yaml", draft=False, allow_model_change=False)
    r = route(ctx)
    for r in route.many(contexts):   # iterator, bounded batches, input order preserved
        ...
    route.correct(ctx, fact=..., value=...)
    print(route.explain(ctx))
```

- `ss.Runtime()` binds defaults lazily: the judge (Gemma 4 E2B Q4_0; `ss.PRECISE_JUDGE` / `--precise` for E4B, `ss.FAST_JUDGE` / `--fast` for Qwen3-0.6B) and the embedder load on first use. Every result has `as_dict(verbose=False)` with stable fields and a one-line `str()`.
- Module-level `ss.check`, `ss.pick`, `ss.rate`, `ss.ask` and `ss.load` use a default runtime.
- Every configuration problem raises `ss.ConfigError` with the file, YAML location and a suggestion.
- The library takes explicit paths; resolving `route_ticket` to `switches/route_ticket.yaml` is a CLI convenience.

### 11.2 CLI

```bash
ss [--judge SRC | --fast | --precise] [--embed SRC] [--device auto|cpu|gpu] [--threads N] [--json] [--verbose] COMMAND

ss check "The customer threatens legal action" < ticket.txt     # exit 0 true · 1 false · 2 uncertain · 3 failed
ss pick  "Which address gets the receipt?" --options addresses.txt < invoice.txt
ss rate  "How urgent?" --levels "can wait,today,immediately" < ticket.txt
ss ask   questions.yaml [--each] < in.jsonl > out.jsonl         # worst status: 0 · 2 · 3; --each: 4 if any record failed

ss run support_route --each < messages.jsonl > routed.jsonl     # exit 0 ok · 2 uncertain/needs_input · 3 failed
ss explain support_route [--raw] < message.json
ss test route_ticket [--intervals]                              # plain sentences on a terminal, JSON when piped
ss doctor [--quality]                                          # backend conformance; --quality runs the judge quality probe
ss correct · label · specialize · tune · diff · lock · bench [--embed]
```

**Two output contracts:**

- **Human (a terminal):** one short line per result, e.g. `billing  (case billing)`, `needs input: payment.status  (blocked at refund_settled)`, `true (p_true 0.97)`. `--verbose` prints `explain()`.
- **Machine (piped, or `--json`):** one JSON object per line from `result.as_dict()`, with stable fields `status`, `answer`, `case`, `missing`, `invalid`, `uncertain`, `blocked_by`, `errors`, and per fact `value`, `settled`, `score`, `source`, `why`. `--verbose` adds `scores`, `cost` and `partial`.

**Other behaviour:**

- `ss label` shows each fact's question, parses answers through the fact's own values (`y` / `n` / `u` for three-way facts, option keys or numbers for picks), rejects invalid answers and asks again. **Prompts go to stderr**, so stdout stays clean.
- **`--each` answers each line as it arrives**, so an interactive producer gets a reply per line. `--batch N` groups N lines for bulk throughput. v0.16 always waited for 16 lines.
- **`ss explain`** prints the decision path on a terminal and `as_dict(verbose=True)` when piped or with `--json`.
- **Per-call `ms`** in `cost` is the batch time divided by its calls: exact for one call, an average for `many()`.
- **Exit codes are one convention for every command:** 0 ok / none, 1 false (`check` only), 2 uncertain or needs_input, 3 failed, and for `--each` streams the worst status seen (4 if any record failed). `ss ask` uses the worst status across its questions, so an uncertain answer exits 2, as `run` does.
- **Configuration vs execution.** Bad configuration, unknown models, unloadable files and ambiguous model sources are `ConfigError`s (`error: …`, exit 1). Scoring problems on valid inputs are results with `status="failed"`.
- **Model sources must be exact.** `repo:file.gguf`, or `repo:Q4_0` only when exactly one file matches; several matches is an error listing them, never a silent pick.
- Malformed JSONL fails with `error: stdin line N: invalid JSON (…)`, exit 1.
- Every command closes its runtime.
- `SEMANTIC_SWITCH_JUDGE` / `_EMBED` / `_STRONG` still work when no flag is given.

## 12. Architecture

### 12.1 Layers (one definition, one pipeline)

| Layer | Owns | Modules |
|---|---|---|
| **1. Definition** | **`Judgment`** (check, three-way check, pick, rate): `questions(view)` builds every prompt question; `parse(value)` reads every label, example, correction and CLI answer; `accept(scores, policy)` applies answer mass, thresholds, three-way mapping and eligibility capping. `Catalog` (cards prepared at load). `Plan` (fields, judgments, atoms, cases). One `render` for every input. | `judgment.py` (new), `compiler.py`, `context.py` |
| **2. Evidence** | Exact atoms, `mentions`, eligibility, exact overrides, catalog shortlist, classifier heads, as named stages | `kernel.py` (exact), `executor.py`, `examples.py`, `heads.py` |
| **3. Scoring** | `Judge` protocol (six members); `LlamaJudge` with the input cache; adapters; native backend | `judge.py`, `adapters/`, `backend.py` |
| **4. Resolution** | Three-valued logic, cases, relevance, card precedence and completeness | `kernel.py`, `executor.py` |
| **5. Interfaces and tools** | Inline API, `Switch`, results, CLI. `lock`, `test`, `diff`, `label`, `specialize`, `doctor`, `bench` **only call layers 1–4**; no tool keeps its own acceptance, parsing or question rules. | `inline.py`, `switch.py`, `results.py`, `api.py`, `cli.py`, `tools/` |

Adaptation crosses layers 2–3 and is produced offline: corrections → exact overrides; labels → tuned thresholds (`ss lock`); heads (`ss specialize`, `ss lock`); a fine-tuned judge model (§7.2).

**Why a definition layer.** Counted in v0.16: acceptance was implemented 6 times (`executor._set`, `_own`, `_card`, inline `gate`/`finish`, `tools/common.settle`, `specialize._settle`), value parsing 5 times (`truth`, `FactPlan.index`, `Check._slot`, `_slot_of`, `label.parse`), and question construction in 2 modules at 9 sites. Inputs were rendered in 2 formats: an inline dict became `key: value` while a switch wrote `[author] field: value`. **The same input got a different prompt inline than in a switch**, and attribution was measured to flip answers. Every bug in the last two reviews sat in one of these copies (`label` couldn't parse "unstated", `specialize` used different thresholds from `lock`, `diff` had its own idea of "changed", `select` had its own card rules).

**One pipeline per decision:**

```
project input → exact atoms → what's still needed → evidence (overrides, shortlist, heads)
  → score (one batched call per input; cached input) → Judgment.accept → resolve → result
```

Inline calls keep a short driver of their own, because callables and per-call candidate objects don't belong in a compiled plan. They build a `Judgment` and use its `questions`, `accept` and `parse`, so **inline and saved judgments answer identically by construction.** `select` card triggers (checks) and exclusions (three-way checks) are `Judgment`s too.

**Deleted or merged:**
- `tools/common.settle` and `specialize._settle` / `_tune` → `Judgment.accept` plus the real runtime;
- `truth`, `Check._slot`, `label.parse` → `Judgment.parse`;
- the question sites in `executor.py` and `inline.py` → `Judgment.questions`;
- `render_free` → `render` with implicit field specs.

### 12.2 Data structures

| Structure | Representation |
|---|---|
| Compiled plan | Frozen, slotted dataclasses; tuples; precomputed `atom_fact`, `exact_atoms`, `fact_atoms`, `roles` |
| Truth states | Small integers in a per-call list |
| Per-fact state per call | `FactState(dist, source, mass, error, missing, eligible)` for check / pick / rate; `SelectState(states, trigger scores, reasons, retrieved_by, superseded_by, complete)` for select. One meaning per structure. |
| Prepared at load (in the `Plan`) | Card questions, code and alias tables, BM25 term statistics, supersedes edges as indices, and the typo index (symmetric deletes) over declared ids. v0.16 rebuilt the catalog index per request (about 12 ms at 1,000 cards). Re-rendering the input costs 0.3 µs and is not cached. |
| Exact overrides | `dict[canonical key → value]` per fact |
| Model results | `Scores` per question; one owned `ScoreBlock` inside the backend |
| Candidates | Borrowed references; only the chosen object is retained |
| Public results | Slotted dataclasses with reprs; `Fact.scores` is a read-only labelled mapping; an internal `Trace` powers `explain()` |

Engine overhead is about **0.2 ms per switch call**: all latency is model time.

### 12.3 Limits

| Limit | Default | When exceeded |
|---|---|---|
| Rendered state per reads group | 1,024 tokens | `failed(limit)` (after per-field truncation) |
| Question with its options and examples | 1,536 tokens (room for 20 options × 64 tokens) | Drop examples from the end; then `failed(limit)` |
| Candidates per `pick` | Adapter slots (20) without `shortlist` · 1,000 with `shortlist` | `failed("too many candidates")` |
| Text per candidate | 64 tokens | Truncated, listed in `r.truncated` |
| Request batch | 64 questions or 16,384 tokens | Split into more `score()` calls |
| Facts per switch | 32 | Load-time error |

Limits are enforced in the `Judge`, so inline calls, switches and tools behave the same: an over-limit question fails alone (`error: "limit: …"`), other questions in the call are still scored, and a switch reports it like any model error. Set `max_chars` so a reads group stays under 1,024 tokens (about 3,000 characters of prose; less for languages that tokenise longer). Native limits (`n_batch`, `n_ctx`, `n_seq_max`) are handled by the backend (§10.4).

---

## 13. Performance and cost

### 13.1 Physics (a rough budget, not a prediction)

```
time ≈ (compute-relevant params × tokens processed × 2 FLOPs) / effective throughput + overhead
```

The estimates are per ~300 processed tokens and ignore attention, memory traffic and host transfers. Compute-relevant parameters exclude the embedding lookup. The tied output matrix only runs on answer rows (~0.16 GFLOP per row for Qwen3-0.6B, about 0.1% of reading the input).

| Path | Compute-relevant params | FLOPs | Consumer GPU | Laptop CPU |
|---|---|---|---|---|
| Judge Qwen3-1.7B | ~1.4B | ~0.84 T | ~40–80 ms | ~2–4 s |
| Judge Gemma 4 E2B (per-layer embeddings are lookups) | ~2B | ~1.2 T | ~50–100 ms | measured 0.23 s at Q4_0 |
| Judge Qwen3-0.6B | ~0.44B | ~0.26 T | ~15–30 ms | ~0.5–1.3 s |
| Embed (multilingual-e5-small class: 12 layers × 384) | ~21M | ~13 G | ~2–4 ms | ~25–60 ms |
| Tier-1 head per fact | — | ~0 | μs | μs |

Cross-check: Jeff reports ~460 ms on a 32-thread CPU and 22 ms on an RTX PRO 6000 for a 0.8B decision model [R8].

**What follows:** judges are fast on GPUs and Apple Silicon. On laptop CPUs, speed comes from exact conditions, exact overrides and tier 1. **Measured request latency decides; this table only explains.**

### 13.2 Reference machines and gates

| Class | Reference machine | Gate | Measured (16-core laptop CPU) |
|---|---|---|---|
| **H1** laptop CPU | Ryzen 7 7840U / Core Ultra 7 155H, 16 GB | Tier-1 switch, 5 facts: ≤ 30 ms p50 | Core Ultra X7 358H (16 cores, AVX2), 16 threads. Check p50: Gemma 4 E2B Q4_0 228 ms, Qwen3-0.6B Q4_K_M 99 ms, Gemma 4 E4B QAT 572 ms. Per question inside a 10-question ask: 128 / 54 / 318 ms. Embedding 8.6 ms. |
| **H1g** laptop iGPU (Vulkan) | Core Ultra X7 358H, Arc B390 | Default judge check ≤ 100 ms p50; record backend, device, offload, cold start, check, ask, full switch | not measured (needs the Vulkan build) |
| **H2** Apple Silicon | MacBook Air M2, 16 GB | Fast judge ≤ 60 ms; default judge ≤ 120 ms p50 | not measured |
| **H3** consumer GPU | RTX 4060 8 GB | Judge ≤ 30 ms p50; ≥ 50 decisions/s | not measured |
| **H4** cloud GPU | NVIDIA L4 | Cost per million, both frames | not measured |

**Where the time goes on short inputs:** a 43-token ticket against 47–73 tokens of question template per question. Speculative scoring is the lever (§4.6). Native batch filling (~0 ms) and row reduction (0.2 ms) are negligible.

### 13.3 Cost frames (`ss bench`)

`ss bench [switch] [--embed]` times **public calls** (`rt.check`, a full switch call), measures **throughput** with `switch.many()`, and derives cost from measured decisions per second and the mean tokens per decision:

- **Owned hardware:** `power_W / decisions_per_s × 1e6 / 3.6e6 × $/kWh` per million decisions.
- **Rented hardware:** `hourly_price / (3600 × decisions_per_s) × 1e6`.
- **Jev at the same token count:** `$0.042 × tokens_per_decision` per million decisions.

The embedder is timed only with `--embed`.

### 13.4 Hosted-price sanity check (first principles)

Hosted per-token prices are a **sanity check, not a bound** on self-hosted cost. The token counts below exclude our cached system prefix; Jev's real billing for an equivalent decision may count differently, so a fair comparison uses each system's own input accounting:

| Path | $ per million input tokens | Per million decisions at ~85 new tokens |
|---|---|---|
| Jev | 0.042 | ≈ 3.6 |
| Qwen3-0.6B hosted (Featherless) | 0.040 (cached input 0.008) | ≈ 3.4 |
| Qwen3-1.7B hosted (Featherless) | 0.320 | ≈ 27 (7.6× Jev) |
| Gemma 4 E2B Q4_0 on owned CPU (228 ms; **assumed** 65 W, $0.15/kWh) | — | ≈ 0.6 in energy (illustrative; excludes hardware), ≈ 0.23 s per decision |

Our input is only billed-equivalent for the input and question: the system prefix is cached. The tension this exposes is real:
- The judge that is **cheap enough** (0.6B) isn't precise enough on hard cases.
- The judge that is **precise enough** used to be 1.7B: 7.6× Jev's hosted price, or slow on a CPU. With Gemma 4 E2B Q4_0, a precise judge costs roughly $0.6 per million decisions in assumed energy on owned hardware, at about 0.23 s each. Latency is the caller's wait: a 10-question ask is 1.28–1.50 s end to end, even though each question costs about 128 ms of throughput.
- **Matched comparison with hosted Jev** (`bench/compare_jev.py`, `bench/compare_jev_quality.py`, warm, network included for Jev):

  | Input | Jev | Qwen3-0.6B (CPU) | Gemma 4 E2B (CPU) |
  |---|---|---|---|
  | 20 tokens, 1 question | 270 ms | **98 ms** | 247 ms |
  | 20 tokens, 7 questions | 292 ms | 352 ms | 906 ms |
  | 261 tokens, 1 question | 286 ms | 423 ms | 1,179 ms |
  | 261 tokens, 7 questions | 281 ms | 706 ms | 2,034 ms |

  - **On the selection suite:** Jev 235/240 (all 216 answers at ≥ 0.9 correct), Gemma 4 E2B 231/240, Qwen3-0.6B 176/240. That suite favours Gemma (§14.6), and Jev hasn't been run on the fresh suite.
  - **Reading:** on CPU, cost is mostly input length (about 3.6 ms per input token for Gemma, 1.3 for Qwen); Jev is flat at about 280 ms. Local wins on short inputs and on repeated questions about one input (with the input cache, §10.4), and loses on long single questions.
  - **Contract:** Jev's agreement §2.3(b) restricts using its outputs to train or develop competing products. Further Jev runs wait for a reading of the agreement or TypeSafe's consent, and Jev outputs never enter training data.

What breaks it is fewer judge calls (rules, overrides, staged scheduling, early pruning), the GPU (§10.7), and only if needed a fine-tuned small judge (§15).

## 14. Evaluation

### 14.1 First benchmark (small, on purpose)

| Task | Tests |
|---|---|
| One intent-classification task in **the language of your traffic** | `pick` quality |
| One five-question `ask` over real records | Shared-prefix value |
| One real saved switch (≥ 300 labelled items, chronological split) | End-to-end behaviour |
| One candidate-selection task (regex-found amounts, emails, dates) | `pick` returns the original object |

### 14.2 Systems

1. Judge Gemma 4 E2B / E4B and Qwen3-0.6B, 0 and ≤ 4 examples
2. Qwen3.5-2B (where `doctor` passes)
3. One challenger decoder family of similar size
4. Tier-1 heads per `embed` candidate (with judge fallback)
5. Embeddings + logistic regression alone
6. GLiClass
7. Frontier LLM (quality reference)
8. *(optional)* Jev

All quality numbers are reported on **unseen** inputs.

### 14.3 Deterministic suite (must pass 100%)

Everything from v0.11, plus:

- user content (input, options, examples) never becomes chat control tokens (fake and real tokenizers)
- missing eligibility data produces `needs_input` naming the field
- an all-ineligible pick is settled by rule with no judge call
- `none` is reserved (inline and saved)
- `ss label` rejects invalid answers and re-asks
- the embedder's prefix is part of its identity
- eager scoring prunes judge work once an override settles the answer
- `describe()` runs only for eligible candidates
- `load()` doesn't initialise the judge
- prompt examples are off by default
- an untuned classifier never settles; a tuned one does
- saved and inline validation: duplicate case ids, enum without values, target range, the 20-option limit, rate levels, shortlist mode
- stable `as_dict()` fields and human `str()`
- CLI line-numbered JSON errors
- the minimal example compiles
- `mentions(body, "MX-120")` is FALSE on "MX-120A failed"; aliases match; a missing field is UNKNOWN
- `focus="mentions"` shows only mentioned candidates plus the escape; escape or no mention falls back to the full list; a never-mentioned answer is still reachable
- fuzzy proposals never merge two catalog ids within distance 1
- a switch without cases returns its facts, and `correct` / `lock` work on it
- the engine runs against a minimal non-decoder `Judge` (five members) with no tokenizer
- a three-way check answering "does not say" gives UNKNOWN (`not_stated`) and `needs_input` naming the fact, inline and in a switch; it never becomes FALSE
- v0.17 `select`:
  - an over-budget search leaves unexamined cards UNKNOWN, never returns `none`, and `is empty` stays UNKNOWN;
  - a within-budget search is `complete`;
  - self-supersede and cycles are `ConfigError`s;
  - precedence runs before `exclusive` ("A supersedes B, both apply" → A);
  - an unresolved overrider makes the overridden card unresolved;
  - payloads appear in JSON
- tuning keys stored in the lock stay excluded from `ss test` after the test file grows
- `ss diff` reports a changed fact value in a switch without cases, using stale-example inputs
- `ss label` records `unstated`
- a protocol judge (six members, no tokenizer) works with prompt examples on
- the typo index finds the same risky ids as the pairwise check
- `--each` emits the first result before the second line arrives
- a lexical-only lock records no `embed` role
- `ss specialize` reports the source that settled each input, from real runtime calls
- `Check(...).save(path)` creates once, reloads after, and refuses a changed definition; the saved switch answers the same as the inline judgment on the same input
- `Check` / `Pick` / `Rate` are callable
- one `Judgment` drives questions, parsing and acceptance: the inline, switch, `label`, `specialize` and `diff` paths agree on the same scores
- an inline dict and a switch render the same field identically
- the input cache: a second scoring of the same input decodes no input tokens; a common-prefix input decodes only its suffix (attention models); a hybrid model reuses on an exact match only; D1 and D3 still pass
- `select`:
  - a card with a FALSE `requires` is never judged;
  - a code-linked card survives a shortlist that BM25 alone would cut;
  - a superseded card is reported as `superseded`, never applicable;
  - `exclusive: true` with two applicable cards is `uncertain`;
  - the payload is returned untouched and never appears in the judge's input
- a pick's `value` stays `None` when only a group atom is settled
- under `resolve: all`, a case correction derives no FALSE labels for other cases
- low answer mass gives `uncertain` inline and in a switch alike
- invalid inline input, empty asks and all-ineligible picks never acquire the judge; a draft with `max_examples: 0` builds no tokenizer
- `describe()` isn't called when the candidate limit is exceeded; `rate` with more levels than slots is a `ConfigError`
- an over-long classifier input defers to the judge
- changing a fact's `reads` makes its labels stale
- doctor results don't carry across backends or builds
- an ambiguous model source is an error; unknown adapter or device names suggest alternatives
- `ss ask` exits 2 on an uncertain answer; `ss label` writes prompts to stderr
- the generic chat adapter renders Qwen3's template byte-identical to the `qwen3` prompt, emits Gemma's turns and BOS, and refuses a model without a template
- a hybrid judge uses shared scoring once D1 is recorded as passing
- a judge whose cached quality probe is below 95% precision at the default thresholds warns once
- §12.3 limits: an over-long input or question fails with `limit`, large asks split into bounded batches, examples are dropped to fit, more than 32 facts or 1,000 shortlist candidates are rejected
- answer labels go stale when the cases change, fact labels when a fact's meaning changes, and `ss tune` / `ss test` / `ss lock` say so
- `ss test` reads as sentences, with exact intervals on request

### 14.4 Experiments

| Exp | What | Answers |
|---|---|---|
| **E0** | Judge Gemma 4 E2B / E4B and Qwen3-0.6B on the §14.1 benchmark, 0 and ≤ 4 examples, on H1–H3 | **Gate for increment 1:** ≥ 40% coverage at ≥ 95% accepted accuracy on unseen inputs of the classification task, plus the H2/H3 judge speed gates |
| E1 | Learning curve 0 / 3 / 10 / 30 / 100 examples; judge vs tier-1 heads per `embed` candidate | When does tier 1 match the judge? Which `embed` model? |
| E2 | Representation: flattened vs typed vs typed + provenance + per-fact `reads` | Does structure help? |
| E3 | Switch execution: one call per case vs compiled; masking on vs off | Fewer passes, fewer reviews? |
| E4 | Hardware matrix H1–H4: shared prefix on vs off; 1 / 5 / 20 questions; Q8_0 vs Q4_K_M | Latency, throughput, memory, cost frames |

### 14.5 Success and kill criteria

**Success:**
- On the real switch and the classification task: ≥ 60% accepted at ≥ 95% accuracy on unseen inputs, with 10 examples per fact.
- All §13.2 gates met.
- At 100 examples, tier 1 settles ≥ 50% of fact evaluations on the real switch, without more silent misses than the judge.
- The deterministic suite and D1–D6 pass for the defaults.

**Kill or rethink:**
- **E0 quality fails** with Gemma 4 E2B and E4B → change the judge class.
- **Tier 1 never matches the judge** at 100 examples → try another `embed` model, or fine-tune a small judge.
- **The H2/H3 judge gates fail** with shared prefixes on → rethink the architecture.
- **Embeddings + logistic regression or GLiClass match the judge zero-shot** → make them the default judge.
- **E2 shows no gain from structure** → drop provenance rendering.

---

### 14.6 Standing synthetic benchmarks (`bench/`)

| Script | Question it answers |
|---|---|
| `bench/judges.py` | **The judge ladder.** Per model and quantisation: accuracy, pair ordering, AUC, ECE, coverage and precision at 0.90 / 0.10, coverage at 95% (thresholds cross-fitted across facts), per-phenomenon accuracy, picks, latency, memory, and probability shift vs the same model at Q8_0 |
| `bench/suite.py` | **The held-out suite:** 10 claims × 12 distinct minimal pairs (6 English, 6 Portuguese), each tagged with its phenomenon (negation, contrast, role, reported speech, temporal, hypothetical, keyword trap, double negation, exception, implicit, plain), plus 24 topic picks. Nothing in it is used for training. |
| `bench/prompts.py` | Prompt framings and ensembles, chosen on 5 dev facts and reported on the other 5 |
| `bench/models.py`, `bench/learning_curve.py` | Prompt examples, hidden-state probes, classifier learning curve |

**The old benchmark was too small.** Its "24 held-out items" were 6 sentence pairs, each repeated with a different order number, and two of 1.7B's "errors" were one sentence. Its Gemma rows were measured with the Qwen template. v0.12's model choice rested on it, and v0.13 replaces it.

**Judge ladder (240 checks + 24 picks, 16-core CPU, 16 threads; accuracy columns from `bench/results/judges/`, latency from a separate clean run):**

| Judge | File | Acc. | AUC | ECE | Settled @0.9 (precision) | Pick acc. | Check ms | ms / question in ask | Resident |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 0.6 GB | 0.69 | 0.85 | 0.15 | 36% (89.5%) | 0.79 | 111 | 65 | 1.6 GB |
| Qwen3-0.6B Q4_K_M | 0.4 GB | 0.74 | 0.85 | 0.07 | 14% (97.0%) | 0.83 | 99 | 54 | 1.5 GB |
| Qwen3-1.7B Q8_0 | 1.7 GB | 0.84 | 0.93 | 0.16 | 97% (84.5%) | 0.92 | 290 | 202 | 2.8 GB |
| Qwen3-1.7B Q4_K_M | 1.1 GB | 0.76 | 0.89 | 0.22 | 92% (79.5%) | 0.96 | 306 | 159 | 2.7 GB |
| Qwen3.5-0.8B Q8_0 | 0.8 GB | 0.71 | 0.80 | 0.12 | 35% (83.3%) | 0.67 | 215 | 101 | 1.7 GB |
| Qwen3.5-0.8B Q4_K_M | 0.5 GB | 0.77 | 0.83 | 0.08 | 29% (91.4%) | 0.62 | 194 | 80 | 1.4 GB |
| Qwen3.5-0.8B Q4_0 | 0.5 GB | 0.68 | 0.78 | 0.10 | 30% (88.7%) | 0.62 | — | — | 1.4 GB |
| Qwen3.5-2B Q8_0 | 1.9 GB | 0.90 | 0.95 | 0.04 | 68% (94.5%) | 0.92 | 540 | 216 | 2.8 GB |
| Qwen3.5-2B Q4_K_M | 1.2 GB | 0.78 | 0.96 | 0.11 | 67% (93.1%) | 0.92 | 348 | 197 | 2.4 GB |
| Gemma 3 1B Q8_0 (own template) | 1.0 GB | 0.70 | 0.79 | 0.29 | 96% (70.9%) | 0.88 | 210 | 101 | 1.3 GB |
| Gemma 4 E2B Q8_0 | 4.7 GB | 0.96 | 0.99 | 0.04 | 99% (96.2%) | 0.92 | 440 | 289 | 5.0 GB |
| **Gemma 4 E2B Q4_0 (ggml-org)** | 2.7 GB | **0.95** | 0.99 | 0.04 | **92% (97.7%)** | 0.96 | **228** | **128** | 3.8 GB |
| Gemma 4 E2B QAT Q4_0 (Google) | 3.2 GB | 0.94 | 0.99 | 0.04 | 92% (97.7%) | 0.96 | 264 | 149 | 4.2 GB |
| Gemma 4 E4B QAT Q4_0 (Google) | 4.8 GB | 0.975 | 0.99 | 0.02 | 95% (98.7%) | 0.96 | 572 | 318 | 7.5 GB |

**Readings:**
- **Size alone doesn't buy precision.** Gemma 4 E2B (2B compute) beat Qwen3-1.7B and Qwen3.5-2B on every quality metric here. Why (training, architecture, tokenizer, prompt fit) isn't separated by this experiment. Qwen3.5-0.8B doesn't beat Qwen3-0.6B.
- **Pair ordering stays high where accuracy is low:** Qwen3-0.6B orders 91% of pairs correctly at 0.69 accuracy. Its knowledge is there; its decision boundary is biased per claim. Per-claim offsets learned from labels (what `ss lock` tuning does) recover 0.69 → 0.79 for it and 0.90 → 0.91 for Qwen3.5-2B, but they can't fix Qwen3-1.7B's saturated errors.
- **The hardest phenomena** for every Qwen are reported speech, role ("my manager says…, but I…") and temporal reversals. Gemma 4 E2B is ≥ 0.92 on all of them except exceptions (0.83).
- **On this suite, quantisation tolerance differed by family and format more than by size** (§10.8).

Confident errors still exist on every model. They are found by labelled tuning, `ss test`, audits and corrections, not by tighter defaults.

**Fresh held-out suite** (`bench/fresh.py`): 8 new claims in new domains (scheduling, HR, security, housing, health, logistics, job offers, deliveries), 10 distinct minimal pairs each (5 EN, 5 PT), and 20 HR topic picks. It was written after the default was chosen and never used for selection or prompt wording.

| Judge | Accuracy | AUC | ECE | Settled @0.9 (precision) | Picks | Check ms |
|---|---|---|---|---|---|---|
| Gemma 4 E4B QAT | **0.931** | 0.969 | 0.054 | 95% (94.7%) | 1.00 | 578 |
| Gemma 4 E2B Q4_0 (default) | 0.856 | 0.917 | 0.119 | 90% (**88.9%**) | 1.00 | 375 |
| Qwen3.5-2B Q8_0 | 0.844 | 0.938 | 0.063 | 66% (**95.3%**) | 1.00 | 419 |
| Qwen3-1.7B Q8_0 | 0.794 | 0.868 | 0.196 | 93% (83.1%) | 0.95 | 382 |
| Qwen3-0.6B Q4_K_M | 0.656 | 0.720 | 0.141 | 13% (90.0%) | 0.90 | 136 |

**The ranking reproduces; the absolute numbers don't:**
- Gemma 4 E2B falls from 0.963 to 0.856.
- Its untuned precision falls from 98.2% to 88.9%.
- Stricter thresholds don't recover it: 89.7% at 0.95, 93.3% at 0.99 (66% coverage), 91.1% at 0.999. Its errors on unseen phrasing are confident.
- Weakest phenomena: hypothetical 0.70, contrast and implicit 0.80, negation and temporal 0.82.
- Qwen3.5-2B is the best-calibrated small judge (95.3% at 66% coverage).
- Gemma 4 E4B comes closest to the target with high coverage.

**Conclusion:** the selection suite overstated the default, because it chose the model and the exception phrasing. **No small judge reaches 95% untuned precision at high coverage on unseen domains.** Precision comes from `ss lock` tuning on the switch's own labels, `ss test`, and corrections. The quality probe's 95% gate measures fit to its suite, not general safety.

**Topical learning curve** (`bench/topical.py`): 4 topics, training and test templates disjoint, thresholds tuned on a separate tune set as `ss lock` would. Over 80 test items the judge alone gets 80/80 at 458 ms.

| Head @ labels per class | Head accuracy | Settled by head (wrong) | Judge calls | Cascade wrong | Cascade ms |
|---|---|---|---|---|---|
| embedding @ 10 / 30 / 100 | 0.54 / 0.55 / 0.53 | 0 / 0 / 7 (0) | 100% / 100% / 91% | 0 | 462 / 462 / 422 |
| lexical @ 10 | 0.76 | 7 (0) | 91% | 0 | 418 |
| lexical @ 30 | 0.76 | 21 (0) | 74% | 0 | 338 |
| lexical @ 100 | 0.74 | 48 (**3**) | 40% | **3** | 183 |

**Readings:**
- **The embedding head doesn't help, even for topics:** e5-small vectors with logistic regression generalise poorly to new wording.
- **The lexical head is the useful one.** At 30 labels per class it cuts judge calls by a quarter with no new errors. At 100 it cuts latency by 60% but adds 3 wrong answers, because thresholds tuned on in-distribution labels are optimistic on new wording.
- **So the admission rule matters:** a head goes on only if, on held-out inputs, it adds no wrong answers.
- **The run exposed a bug:** threshold tuning treated NumPy booleans as never-true (`y is True`), so every threshold came back unset. `ss specialize` was affected; `ss lock` was not, because it passes Python booleans. It's fixed and covered by a test.

**v0.17 investigation** (fake judges unless noted):
- **A supersedes cycle** erased both applicable cards (settled `none`).
- **The exhaustive check** costs 300 ms for 1 card, 1.27 s for 8, 2.44 s for 16 and 4.60 s for 32 on the default judge (real model). That sets `max_cards: 16`.
- **The typo index** found 869 risky ids among 1,000 in 845 ms pairwise; symmetric deletes found the same 869 in 7 ms.
- **The catalog index** cost about 12 ms per request at 1,000 cards.
- **Rendering** costs 0.3 µs.
- **Tuning contamination:** appending one test line moved 26 tuning inputs into "unseen".

**Limits of this evidence:**
- **The suite chose the default and the quantisation**, so it is now regression data, not independent confirmation. A **fresh held-out suite** (new claims, new sentence families, written after v0.13, never used for selection) must reproduce the ranking before broader quality claims are made.
- **Acceptance in `bench/judges.py` must use execution's rules:** low-mass rejection and errors as unsettled. The v0.13 numbers didn't apply the mass rule; median mass was ≥ 0.999 for every model, so they are unlikely to move. The rerun uses `Judge` validity.
- **Synthetic, two languages, one domain.** It ranks judges; `ss test` on your labelled data is what certifies a switch.

**Identifier set** (`bench/identifiers.py`): 14 reports naming near-identical machines, run with 4 and with 20 candidates. It covers suffixes (`MX-120` / `MX-120A` / `MX-120B`), two machines with different outcomes, negation, "I meant…", an alias, a missing hyphen, a typo, and Portuguese.

| Judge | 4 options | 20 options | Confidently wrong | ms per pick (4 / 20) |
|---|---|---|---|---|
| Gemma 4 E2B Q4_0 | 13/14 | 13/14 | 0 | 495 / 1139 |
| Qwen3-0.6B Q4_K_M | 11/14 | 8/14 | 0 | 198 / 446 |
| Qwen3-1.7B Q8_0 | 10/14 | 12/14 | 4 / 2 | 616 / 1488 |

Exact mention matching (offline) narrowed 20 candidates to 1–3 in 12 of 14 reports and kept the right answer every time. It is to be extended with "the other one" cases and catalogs over 20 candidates.

**Measured or deferred, with the reason:**
- **GLiClass multilang mini** (Apache-2.0, ~288M): its card reports a multilingual zero-shot F1 of 0.54, no CPU latency, and needs torch plus transformers (~1 GB). Deferred; at most a topical router, and the tier-1 head holds that slot.
- **BTZSC** (encoder and reranker zero-shot classification benchmark): English-only, not checked by us. It doesn't change the plan.
- **FTS5 or an index layer:** unnecessary until catalogs reach about 10k candidates, since in-memory scanning is microseconds below that.
- **NER models, stemmers, ontologies:** each waits for a failure it fixes.

**Default judge after the v0.16 prompt change** (`ss doctor --quality`, same suite): accuracy 0.963, AUC 0.988, ECE 0.027, **94.6% settled at 98.2% precision**, picks 0.958, check p50 291 ms. Exceptions went from 0.83 to 1.00. The phrasing was chosen on overlapping items (§10.3).

**Procedure set** (`bench/procedures.py`, 10 situations on the bundled catalog, EN/PT, default judge): **8/10 correct; both misses were abstentions, never a wrong procedure.**
- "Sensor fault on the display, no leak" left the leak check uncertain: three-way probabilities are less peaked, and "no leak" put 0.86 on "states false", under the 0.9 default.
- Portuguese "Sem vazamento" read as "does not say" (0.56).
- About 1.1–1.7 s per situation.

Defaults favour abstaining, and `ss lock` tunes the thresholds from labels.

**Unstated set** (`bench/unstated.py`, 10 reports EN/PT, claim "there is a visible hydraulic leak"):

| Judge | Forced true/false on the 4 silent reports | Three-way accuracy | Confidently wrong (three-way) |
|---|---|---|---|
| Gemma 4 E2B Q4_0 | "false" at p_true 0.00 on all 4 | 9/10 | 0 |
| Qwen3-0.6B Q4_K_M | guesses, 0.12–0.49 | 6/10 | 0 at 0.9 |

**Still to measure:**
- **Procedure selection** (`bench/procedures.py`, to write). Cards:
  - near-identical cards (MX-120 vs MX-120A, E17 vs E71);
  - exclusions and unstated preconditions;
  - `supersedes`;
  - two cards that legitimately both apply;
  - codes in a typed field vs only in text;
  - catalogs of 20, 200 and 2,000 cards.

  Report BM25 + exact-link recall at `max_cards`, wrong applicable ids, correct `needs_input`, and end-to-end latency.
- **Topical learning curve:** billing / technical / account / shipping on unseen wording, at 0 / 10 / 30 / 100 labels per class, comparing the `embedding` head, the `lexical` head and the judge alone. Report classifier coverage, wrong accepted answers, judge fallback rate and warm cascade latency. This decides whether tier 1 is worth its workflow. Negation and attribution are measured separately. Embedding fine-tuning comes only after tier 1 helps somewhere.
- **GPU (H1g)** on this machine.

## 15. Delivery

**Increment 1: fast inline judgments**
- The protocols, the llama.cpp backend (low-level API) with chunked shared-prefix scoring, view-based reduction, concurrency and cleanup
- The Qwen3 adapter
- `ss doctor` (D1–D6), `ss bench`
- `check` / `pick` / `rate` / `ask` with the §3.4 edge cases
- CLI with exit codes and streaming
- **Gate:** E0

**Increment 2: saved switches**
- Compiler and kernel: canonical atoms, propagation, Kleene logic, missing data and per-fact errors, eligibility, provenance, (model, adapter, reads) groups
- Canonical keys, examples, overrides, corrections
- `ss label`, `ss test`, `ss tune`, `ss diff`, `ss lock`
- **Gates:** E2, E3, the deterministic suite

**Increment 3: tier 1**
- The `embed` role, `ss specialize`, heads trained at lock time, whole-fact fallback
- **Gates:** E1, E4, the H1 tier-1 gate

**Status:** everything through v0.16 is implemented. The suite has 141 tests: 126 fast and deterministic, and 15 against the real models (Gemma 4 E2B, Qwen3-0.6B, Qwen3.5-0.8B and multilingual-e5-small).

**Still open:**
- **The GPU path** (§10.7), deferred.
- **Batching independent inputs into one decode.**
- **A fresh held-out suite.**
- **The topical learning curve.**
- **Experiments E0–E4 on real traffic.**

**Work order (v0.13 + v0.14):**

1. **Correctness:**
   - A pick's value is reported only when its own top option settles; group atoms settle separately.
   - Earlier-case FALSE labels are derived only under `resolve: first`.
   - The answer-mass rule is enforced in `Judge` for every caller.
   - `rate` levels are checked against the slot limit.
   - Long classifier inputs defer to the judge.
   - `reads` and rendering are part of `meaning_rev`.
2. **v0.13 defaults** (below), plus adapter and device validation and exact model sources.
3. **GPU (Vulkan):**
   - Install `cmake` and `vulkan-headers` on the machine, then rebuild llama-cpp-python.
   - Key the doctor and quality caches by backend, device and build.
   - Report the actual offload.
   - Measure H1g.
4. **Cheap paths first:** validation and immediate answers before judge acquisition, a tokenizer only when prompt examples are on, and the candidate limit before `describe()`.
5. **Evidence:** a fresh held-out suite, `bench/judges.py` on `Judge` validity rules, and the topical learning curve.
6. **Teaching:** `ss specialize` on deployed acceptance rules with warm latency, the §7.1 workflow in the README, and stderr prompts in `ss label`.
7. **Batching independent inputs together**, only if the GPU numbers call for it.
8. **Exact evidence (v0.15):**
   - `mentions()` atoms.
   - Candidate ids and aliases, `focus="mentions"` with the escape option, and exact-first BM25.
   - `r.near` fuzzy proposals.
   - `bench/identifiers.py`.
9. **One teaching system:** switches without cases, and the `lexical` head next to `embedding`.
10. **`Judge` as a protocol:** `LlamaJudge` behind it, and `fit()` replacing `truncate` / `count` in the engine.
11. **Three-way checks (v0.16):** `unstated=` / `unstated: true`, the `not_stated` atom reason, `needs_input` for facts, and inline `needs_input`.
12. **`select` facts (v0.16):**
   - Catalog loading and validation: unique ids, `supersedes` pointing at real ids, a `requires` that compiles.
   - Exact filter, exact links, BM25 shortlist and three-way card checks in one ask.
   - Precedence, the `has` / `is empty` atoms, and the result fields.
   - `bench/procedures.py`.

13. **v0.17 correctness:**
   - Card states with `complete` and `max_cards`;
   - compile-time supersedes validation, precedence before `exclusive`, and the may-be-superseded rule;
   - `SelectState`;
   - `tuned_keys` in the lock;
   - `diff` per object outcome;
   - `label` through fact values.
14. **v0.17 tools and interfaces:**
   - `specialize` on the real runtime;
   - `count` in the `Judge` protocol;
   - head-driven model roles;
   - immediate `--each` with `--batch`;
   - `explain --json`;
   - payloads in JSON.
15. **v0.17 preparation:** card questions, code and alias tables, BM25 statistics and the symmetric-delete typo index, all built at load.
16. **v0.17 teaching:** a `focus="mentions"` run in `bench/identifiers.py` (`rt.judgment()` is replaced by `save()`, item 19).
17. **v0.18 definition layer:**
    - `judgment.py` with `questions` / `parse` / `accept`;
    - inline, executor, `select` cards and tools moved onto it;
    - one `render`;
    - delete the duplicated settle, parse and question code.
18. **v0.18 input cache:**
    - exact reuse, and common-prefix reuse for attention models;
    - the `reads`-ordering rule;
    - re-measure staged against eager on long inputs.
19. **v0.18 ergonomics:** callable `Check` / `Pick` / `Rate`, `.save()`, and the zero-to-hero ladder in the README.
20. **v0.18 fine-tuning experiment (§7.2):**
    - data from MASSIVE and the Gemma 4 E4B teacher, never Jev;
    - LoRA on Qwen3-0.6B (and 1.7B);
    - export to GGUF Q4;
    - fresh suite, changed-definition set, and latency; the gate.

**v0.13 code changes (from §14.6):**
1. **Default judges:**
   - `DEFAULT_JUDGE` becomes Gemma 4 E2B Q4_0, with `PRECISE_JUDGE` (E4B QAT) and `FAST_JUDGE` (Qwen3-0.6B Q4_K_M).
   - The CLI gets `--precise`.
   - The slow tests move to the new defaults.
2. **Generic chat adapter** (`adapters/chat.py`) as the default. `qwen3` stays as an alias.
3. **Shared scoring for hybrid models,** gated by a recorded D1.
4. **Quality probe:** the suite moves into the package, with `ss doctor --quality` and the untuned-precision warning.
5. **README and the measured tables** updated.

**Next:**
- **Real traffic (E0–E4)** with the new default.
- **Fine-tuning the embedding model** (SetFit-style contrastive training of e5-small), **only after** the topical learning curve shows tier 1 helps on its intended facts.

Fine-tuning a small judge drops from "main bet" to "only if needed": Gemma 4 E2B already gives 95% accuracy and 97.7% precision at default thresholds.

**Later, only if needed:** a distilled or LoRA-tuned fast judge (Qwen3-0.6B class) from Gemma 4 labels · the judge hidden-state probe · the fact-first layout · a Transformers reference backend · remote adapters (Jev, OpenAI-compatible) · a custom native binding · hierarchical `pick` for large option sets.

**Layout:**

```
semantic_switch/
  api.py            Runtime, Model, default runtime, model identity and acceleration
  inline.py         Check / Pick / Rate, gate(), ask()
  switch.py         Switch: call, many, correct, explain, reload, schedule
  executor.py       FactState, decide() (eager / staged), score(), public facts
  snapshot.py       Snapshot, draft building, lock loading with model pinning
  examples.py       Example, canonical keys, prompt selection, label derivation
  compiler.py       YAML → Plan, grammar, canonical atoms, cases shorthand, ConfigError hints
  kernel.py         stdlib only: Kleene logic, relevance, propagation, resolution
  judge.py          Judge facade, Scores, self-test fixture
  protocols.py      Question, ScoreRequest, ScoreBlock, Scorer, Embedder
  adapters/         chat.py: generic template from the GGUF · qwen3.py: qwen3@3 alias (content tokenised without special tokens)
  backend.py        llama.cpp: shared prefix, chunked batching, reduction, embeddings, selftest
  results.py        public results, Fact, reprs, explain()
  context.py · files.py · heads.py · errors.py
  tools/            lock · evaluate (tune/test/diff) · specialize · label · doctor · bench
  cli.py
tests/              kernel, inline, switch, tools, v012, limits and labels (fast) · backend + prompt quality (slow)
bench/              suite.py, judges.py, prompts.py, models.py, learning_curve.py
switches/           support_route.yaml (minimal), route_ticket.yaml (advanced)
README.md           install, first call, a switch in five minutes, measured numbers
```

Size: about **3,400 lines** of Python, plus about 1,100 lines of tests and 300 of benchmarks.

**Dependencies:** `numpy`, `pyyaml`, `pydantic`, `typer`, `llama-cpp-python`. `jinja2` (chat templates). Optional: `scikit-learn` (tier-1 training), `scipy` (exact intervals in `ss test --intervals`), `datasets` (benchmarks).

---

## 16. Known limitations

- **Judges are slow on laptop CPUs.** On CPU-only machines, speed depends on exact conditions, exact overrides, staged scheduling and (for topical facts) tier 1.
- **The adapter's slot limit (20 letters) caps options per question.** More candidates need explicit `shortlist`.
- **Tier 1 needs labelled examples** and is enabled per fact by hand.
- **`pick` with per-call candidates always uses the judge.**
- **A correction fixes a fact for an input, not necessarily the final answer.**
- **Scores are model probabilities.** `ss test` on labelled, unseen data is the only measure of correctness, and only for that data.
- **The default judge needs about 4 GB of RAM** (Gemma 4's per-layer embeddings). On 4 GB machines use `--fast`, which settles far fewer hard cases.
- **Judges other than the shipped defaults** have unmeasured untuned precision until `ss doctor --quality` runs. Qwen3-1.7B, for example, settles untuned answers at 84.5% precision.
- **The suite is synthetic** (240 items, 2 languages, support-ticket domain), and it chose the default. It ranks judges; it doesn't certify a switch. `ss test` on your labelled data does.
- **CPU only today.** This machine's Arc B390 GPU needs a Vulkan build (§10.7); the NPU isn't supported.
- **Inline calls don't learn**; teaching is a saved-switch feature (§7.1).
- **Classifier facts are for topical decisions only,** and settle only after tuning on labelled data.
- **Prompt examples hurt small judges.** They're off by default; corrections still work through exact overrides and tuning.
- **Staged scheduling costs a second round when a late case decides.** With labelled inputs, `ss lock` measures which schedule is faster for the switch.
- **Not a planner, not a fetcher, not a platform.**

---

## References

- [R1] Sun & Xu, *Type-safe is not error-free: a constrained decision head follows the option name, not the rubric bound to it*, arXiv:2609.26758 (2026)
- [R5] Kleene, *Introduction to Metamathematics* (1952), strong three-valued logic
- [R6] FitzGerald et al., *MASSIVE: A 1M-Example Multilingual NLU Dataset*, 2022
- [R7] TypeSafe AI, Jev documentation and pricing/limits: https://docs.typesafe.ai/introduction, https://docs.typesafe.ai/models
- [R8] Jeff — open one-pass decision models on Qwen3.5 (English): https://github.com/firelex/jeff
- [R9] `decide` CLI: https://github.com/vsekhar/decide
- [R11] Qwen3.5 model cards: https://huggingface.co/Qwen/Qwen3.5-0.8B
- [R15] llama.cpp C API (`llama.h`: `llama_get_logits_ith` batch-index semantics and validity until the next decode, `llama_encode`, `llama_memory_seq_*`, pooling): https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- [R16] llama.cpp issue #29092 — Gated DeltaNet recurrent state carried across requests (qwen35): https://github.com/ggml-org/llama.cpp/issues/29092
