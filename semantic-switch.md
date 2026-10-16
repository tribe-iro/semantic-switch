# Semantic Switch — Specification

| | |
|---|---|
| **Version** | v0.26 (2026-10-06) |
| **Implemented** | v0.26 (`semantic_switch/`, 194 tests: 176 deterministic, 18 on real models). §9.1 items 1–13 are implemented; the open experiment E8 remains. History is in `CHANGELOG.md`; evidence is in `bench/README.md`. |
| **Owner** | Tulio Ribeiro |
| **One line** | Language-based judgments inside ordinary code: local, cheap, multilingual, open source, with rules you can read, teach and diff. |

---

## 1. What it is

A Python library and CLI: **four inline judgments** and **one saved form**.

```python
import semantic_switch as ss

ss.check(ticket, "The customer threatens legal action against us")                # claim: true / false
ss.check(report, "There is a visible hydraulic leak.", unstated=True)              # true / false / not stated
ss.pick(doc, "Which address should receive the receipt?", candidates)             # one of the given options
ss.rate(idea, "How feasible is this?", ["impossible", "hard", "doable", "easy"])  # an ordered level

route = ss.load("switches/route_ticket.yaml")                                     # facts + rules → answer
route(ticket)
```

> **Code builds a bounded decision → the model judges → code executes.**

**The precision promise.** Untuned, the default judge is right about 89% of the time on the answers it settles in unseen domains (§8). Labels don't make the judge smarter. They let you:
- **fix exact inputs** (corrections);
- **measure generalisation** (`ss test` on unseen inputs);
- **choose how often the switch answers:** `ss lock` tunes thresholds to your precision target, and **where the judge can't meet it, the switch abstains** rather than guess.

Until a switch is locked and tested, treat its answers as drafts.

**The product, in one line: a decision compiler.** Exact evidence, cheap learned shortcuts and shared semantic judgments, combined so that model compute goes only to what remains unresolved.

**What is and isn't distinctive.**
- **Typed judgments over shared input are now a commodity:** the OpenAI Decisions API (October 2026) offers predicate / choice / score at $0.10 per million input tokens.
- **What this library adds is everything around the judgment:**
  - exact eligibility and identifiers enforced in code;
  - facts and rules resolved with three-valued logic;
  - claims shared across a catalogue;
  - corrections and lexical heads that settle recurring inputs without a model;
  - measured abstention (held-out observations with intervals, not guarantees);
  - local, private and offline execution;
  - any judge behind one contract, including a hosted one.

### Zero to hero (one new concept per step)

| Step | API | New concept |
|---|---|---|
| 0 | `ss.check(text, "The customer requests a refund.")` | none |
| 1 | `refund = ss.Check("…")`, then `refund(text)` and `ss.ask(text, {"refund": refund, "topic": topic})` | A reusable judgment |
| 2 | `sw = refund.save("switches/refund.yaml")`, then `sw.correct(text, fact="refund", value=True, actor=ss.Actor("human"))` | Saved and teachable: same meaning, same acceptance policy, same answers |
| 3 | `rules:`, `mentions(...)` and exact predicates in the YAML | Rules |
| 4 | `ss label … --to test`, `ss lock`, `ss test` | Precision on your data |
| 5 | `semantic-trainer lexical` → `ss analyze-head --head` → `ss lock --head FACT=BUNDLE` | Cheaper repeats |
| 6 | `--judge my-finetuned.gguf`, `ss doctor --quality` | A better judge, with no API change |

### Where fast and cheap come from

1. **The cheapest step that can settle a question settles it:** exact rules → exact taught matches → a lexical classifier → the judge.
2. **One read per input projection:** every question about the same rendered input shares one read, and the read stays cached while that input's rounds run (§6.4).
3. **Local small models** through llama.cpp: no per-call fee, no rate limit, no network.

On CPU, cost grows with input length: about 3.6 ms per input token for the default judge (§8). Questions cost too: with the input already read, a 45-token check still takes about 170 ms. Speed comes from asking fewer, shorter questions (exact rules first, shared claims, one read per input, few `Except:` lines), not from a smaller judge.

### What it judges

Three kinds of judgment, all over **supplied** information and **supplied** options:

| Kind | Example | API |
|---|---|---|
| Classification | "Which department does this ticket belong to?" | `pick`, `check` |
| Applicability | "Which procedures apply to this incident?" | `select`, or facts + rules |
| Comparative choice | "Which reply best calms the customer without promising money back?" | `pick` with a criterion in the question |

**Comparative choice** is a free-form criterion over bounded candidates with a typed result. Two rules make it usable:
- **State the basis:** "which proposal best fits a small team that prioritises simple maintenance?" can be checked, corrected and taught; "which proposal is better?" leaves the preference to the model.
- **A choice score is not a utility:** 0.9 is the judge's probability that this option is the answer to the question, not "90% good". Adding a candidate changes every score.

**Measured** (`bench/comparative.py`, 6 scenarios × 2 opposite goals × 2 option orders):
- the default judge is order-stable (12/12 pairs) but **anchors on the situation when the goal contradicts it**: "it is sunny" + "which keeps us dry if a storm starts?" → bike at 0.94. It was wrong on 3 of the 23 answers it settled;
- `--precise` got 24/24, with no wrong settled answer, and followed the goal in all 6 scenarios.

Use `--precise` for comparative choices whose goal runs against the stated situation.

### Not for, not in scope

- **Not for:** generating text or alternatives (candidates are always supplied), planning or sequencing, multi-step reasoning, arithmetic, dates, counting or numerical optimisation (do those in code), extraction without candidates, fetching missing evidence, ingesting documents.
- **Not in scope:** decision logs, review queues, audit trails, traffic replay, crash recovery, multi-tenancy, servers, UIs, summaries of procedures, and safety-critical use in place of certified systems.

---

## 2. Concepts

**Principles**
1. **Judgments in the model, logic in code.**
2. **Three truth values.** UNKNOWN is never silently turned into an answer, and a confident FALSE masks an unknown.
3. **Missing is not uncertain.** Missing input is reported by name; the model never judges data that isn't there.
4. **Scores are model probabilities, not correctness.** Correctness is measured by `ss test` on labelled, unseen data.
5. **Every feature must earn its place** with a measurement: faster, cheaper, more correct or easier to edit.

**Judgment kinds**

| Kind | Asks | Value |
|---|---|---|
| `check` | Is this claim true of the input? | `True` / `False` |
| three-way `check` (`unstated`) | Does the input state it's true, state it's false, or not say? | `True` / `False`; "not stated" → `needs_input` |
| `pick` | Which of these options? | the option (the original object for inline candidates) |
| `rate` | Which ordered level? | the level |
| `select` (saved only) | Which cards from a catalog apply? Several may | a tuple of card ids |

**Statuses, the same everywhere**

| Status | Meaning |
|---|---|
| `ok` | A value or answer was produced |
| `none` | `pick` with `allow_none` chose "none of these"; a switch without rules whose `select` facts are complete and empty |
| `uncertain` | Not confident enough, or a search was incomplete |
| `needs_input` | A field is missing, or a three-way check found the input doesn't say |
| `failed` | Invalid input, a model error on something that decides, or a limit exceeded |

Results have no truth value: `bool(result)` raises `TypeError` ("use `.value`").

---

## 3. Inline API

### 3.1 Calls

```python
rt.check(state, statement, *, excludes=(), unstated=False, fields=None, min_score=None)
rt.pick(state, question, options, *, definition=None, eligible=None, allow_none=False,
        shortlist=None, ids=None, focus=None, fields=None, min_score=None)
rt.rate(state, question, levels, *, fields=None, min_score=None)
rt.ask(state, {"legal": ss.Check(...), "topic": ss.Pick(...), "urgency": ss.Rate(...)})   # one shared read
```

- `state` is a string or a dict; `fields=[...]` selects and orders dict fields. **Inline and saved inputs render identically** (§4.1).
- `options` are a dict `{id: description | {description, aliases}}`, a list of strings, or a list of objects with `definition=`.
- **Acceptance is the same everywhere** (§4.5): one rule, `settle`, used by inline calls, switches and tools. A judgment's policy reads "this answer's probability is at least θ", 0.90 by default:
  - **a check has two sides,** one per answer: it answers True at `p(true) ≥ θ_true` and False at `p(false) ≥ θ_false`;
  - **a pick or rate has one θ:** a value is chosen when its own probability is `≥ θ`.

  Anything in between is `uncertain`, so a 0.55 never comes back as an answer.
  - `min_score=m`, with `0.5 < m ≤ 1`, sets every θ to `m`. It is a **majority** threshold, not "take the most likely answer": with 0.45 / 0.35 / 0.20 nothing settles at any allowed `m`.
  - θ above 0.5 is what makes acceptance consistent by arithmetic: two answers that exclude each other can never both be accepted.
- **Answer format is checked first, the same way inline and in switches:** under 0.5 of the probability on valid answer tokens → `uncertain`, `error="low answer mass"`.
- **Cheap work runs before model work.** Validation, eligibility, candidate limits and immediate answers (no eligible candidate, an empty `ask`) never load a model. `definition()` runs only for eligible candidates, after the limit check.
- Programming errors (invalid `min_score`, malformed options, too many levels) raise; execution problems return `status="failed"` with `r.error`.

```python
r = rt.check(ticket, "The customer threatens legal action against us", min_score=0.9)
r.status, r.value, r.p_true, r.mass          # "ok", True, 0.97, 1.0; three-way adds r.p_unstated
r = rt.pick(doc, "Which address?", addresses, definition=lambda a: a.text)
r.value, r.index, r.score, r.scores, r.removed, r.truncated, r.focus
r = rt.rate(idea, "How feasible?", levels)
r.value, r.index, r.scores
```

Handle the result, not the scores:

```python
r = ss.check(text, "The customer requests a refund.")
if r.status == "ok":
    handle_refund_request(r.value)
else:
    handle_unresolved(r)          # r.status says why: uncertain, needs_input, failed
```

### 3.2 Reusable and saveable judgments

`Check`, `Pick` and `Rate` are values. `refund(text)` runs on the default runtime, `refund(text, rt=rt)` on a given one, and `ss.ask` composes them.

`refund.save(path, field="message")` writes a switch without rules whose one fact is this judgment, **with its acceptance policy** (`min_score` becomes the fact's `min_score`), and returns the loaded `Switch`:
- An existing file with the same definition is loaded.
- A different definition raises `ConfigError`.

**Same input, same rendering, same answer:**
- **Input:** a switch with exactly one field accepts a bare string as that field, so `refund(text)` and `saved(text)` take the same argument. Dicts work as usual.
- **Rendering:** a single field without `author` / `source` renders as its bare text, which is what the inline call sent.

The saved judgment answers exactly as the inline one did until `ss lock` tunes its policy, which is an explicit change.

**What saves:** `Pick` saves only with dict options and no callbacks or shortlist. Inline dict options take `description`, `excludes` and `aliases`, validated by the same rules as `ss ask` question files and saved switches. `requires` is YAML-only (inline uses `eligible=`), and inline raises `ConfigError` if it is given.

**Inspection:** `item.describe()` and `switch.describe()` return immutable authored metadata (address, kind, definition, reads, guidance references, revision) without loading a model. Verbose results (`as_dict(True)`) carry `presentations`: each stage actually sent (`full`, `mentions`, `fallback`) with its answer → key/option mapping.

### 3.3 `pick` rules

| Situation | Behaviour |
|---|---|
| No eligible candidate | `none` with `allow_none`, else `failed("no eligible candidates")` |
| More candidates than the 20 slots (19 with `allow_none`) | `failed("too many candidates")`, unless `shortlist="lexical"`: exact mentions rank first, BM25 fills the rest (at most 1,000 candidates). The answer is **relative to the shortlist**: `r.complete` is false, `r.removed["not_examined"]` lists the dropped candidates (separate from `r.removed["ineligible"]`), and **"none of these" becomes `uncertain`**, because an omitted candidate may be the answer |
| Candidate text over 64 tokens | Truncated, listed in `r.truncated` |
| An option named `none` with `allow_none`; unknown `shortlist` or `focus` mode | `ConfigError` |

### 3.4 Exact evidence: mentions and focus

Models blur small decisive differences; exact matching can't tell who did what. Each does its own job, and **evidence is never blended into one weighted score.**

- **Mentioned:** an id or alias occurs in the input with no letter, digit, `-` or `_` directly beside it, and no `.` that continues into a word. Matching is case-insensitive and NFC; nothing else is normalised. `MX-120` never matches inside `MX-120A`.
- **`focus="mentions"`** (opt-in), given ids via `ids=` or dict aliases:
  1. The judge sees only the mentioned candidates, plus an escape option, "another one, not listed above".
  2. If nothing is mentioned, the escape wins, the answer doesn't reach the policy's θ, or the answer mass is low, the full list is scored. Inline and saved picks make this decision with **one shared function** under the active policy, so they can't diverge.
  3. `r.focus` says which happened: `"mentions"`, `"full"` or `"fallback"`.

  Absence of a mention never excludes a candidate. The payoff is speed (a 20-option pick costs about 2× a 4-option one), not correctness.

### 3.5 Three-way checks

A yes/no check forces a guess on silence, and the guess is confident: the default judge scored "there is a visible hydraulic leak" at p_true 0.00 on every report that never mentioned one. `unstated=True` asks one three-way question (`claim_question` with the options "states true / states false / does not say") at the cost of one check:

| Scores | Atom | Blocking status |
|---|---|---|
| `p(states true) ≥ θ_true` | TRUE | — |
| `p(states false) ≥ θ_false` | FALSE | — |
| `p(does not say) ≥ θ_true` | UNKNOWN, reason `not_stated` | `needs_input`, `missing=("fact:<name>",)`, `why="not stated in the input"` |
| none of these | UNKNOWN, reason `uncertain` | `uncertain` |

Silence must be established like any other answer. A slight lean towards "does not say" (0.33 / 0.33 / 0.34) is uncertainty, not missing information, and asking the user for input would be the wrong action.

Use it for conditions about the world, where silence is common and a guess is costly. Use plain checks for claims about the text itself. Labels accept `true`, `false`, `unstated` (`u` in `ss label`).

---

## 4. The switch language

### 4.1 Files and fields

```
switches/
  route_ticket.yaml            definition: fields, facts, rules, tests     — edited by people
  route_ticket.examples.jsonl  corrections, append-only                    — written by correct / ss label
  route_ticket.test.jsonl      labelled inputs for lock / test / diff      — written by ss label / you
  route_ticket.lock.json       frozen behaviour                            — written only by ss lock
```

Plain files in git; no database. `ss.load(path)` / `Runtime.load(path)` is the one loader: a definition loads as authored (draft or locked), and a frozen lock (`X.lock.json`, a fitted lock directory or its `receipt.json`, or a `current` pointer) loads exactly what it pinned, without the definition files beside it.

Fields are typed (`text`, `number`, `bool`, `enum` with `values`, `json`), optionally `required`, with `max_chars` and `truncate: end | middle`. **Rendering, one function for inline and saved inputs:**
- one line per field, in declared order, inside `<input>…</input>`;
- an `[author]` or `[source]` prefix when declared (attribution was measured to flip answers: "my lawyer will file a lawsuit" without `[customer]` scored 0.00 for "the customer threatens legal action");
- each value truncated per field.

`reads:` sets what judgments see (switch default, per-fact override). A single untagged field renders as its bare text, so a saved one-field judgment sees exactly what the inline call saw.

### 4.2 Definition

```yaml
name: route_ticket
fields:
  subject:        { type: text, required: true, max_chars: 300,  author: customer }
  body:           { type: text, required: true, max_chars: 3000, truncate: middle, author: customer }
  product:        { type: text, source: crm }
  payment.status: { type: enum, values: [pending, settled, failed], source: payments }
reads: [subject, body]

facts:
  legal_threat:
    check: The customer threatens legal action against us.
    excludes: [quoting someone else's threat, merely mentioning a lawyer]
    target: 0.98                                  # precision target for ss lock (default 0.95)
    examples:                                     # seed labels: exact overrides and tuning data
      - { input: { subject: Final notice, body: My lawyer will file suit on Monday. }, value: true }
  refund_requested:
    check: The customer asks to receive money back for a charge.
  topic:
    pick: What is the message mainly about?
    allow_none: true
    options:
      billing:   charges, invoices, payments, plans
      technical: the product is broken, erroring, slow or unavailable
      hardware:  { description: a physical device is damaged, requires: 'product in ["router", "modem"]' }
  urgency:
    rate: How urgent is the customer's situation?
    levels: [can wait, today, immediately]

rules:                                            # first TRUE rule wins (resolve: all returns every TRUE rule)
  - { id: legal,          when: legal_threat,                                       then: legal }
  - { id: refund_settled, when: 'refund_requested and payment.status == "settled"', then: refund_review }
  - { id: tech_urgent,    when: 'topic == "technical" and urgency >= "today"',      then: tech_priority }
otherwise: general

tests:
  - input: { subject: Complaint, body: "Broken again. Refund me or my lawyer will file a lawsuit." }
    expect: legal
```

- **Shorthand:** `rules: {legal: legal_threat, billing: refund_requested}`, where the answer is the rule name.
- **Long form:** `then` defaults to `id` and vice versa.
- **Guard facts that read optional fields** with `exists(field) and …`.
- **At most 32 named facts.** Anonymous claims compiled from catalogue strings don't count; they are bounded by `max_claims` at runtime. Equipment and catalogue examples are in §4.4; three-way facts are in §3.5.
- **Errors:** every configuration error is a `ConfigError` naming the file, the YAML location and a did-you-mean suggestion.

### 4.3 Switches without rules

`rules` and `otherwise` are optional. Without them, a switch is a saved set of judgments: **every named fact is a requested output**, and calling it returns `r.facts`. The status comes from the outputs' own values, not from intermediate atoms:
- `ok` when every fact's own value settles (a `select` fact settles when it has no unresolved card);
- `none` when its `select` facts are complete and empty (§4.4) and everything else settled;
- otherwise `uncertain`, `needs_input` or `failed`, from the unsettled facts' reasons.

A fact you don't want as an output doesn't belong in a switch without rules: write it as a catalogue claim string instead.

This is the teachable form of an inline judgment (§3.2), so there is one teaching system.

### 4.4 `select`: applicable cards from a catalog

`select` is **rules over shared facts, compiled from a catalogue**. Exact rules decide eligibility, each distinct claim about the situation is judged once, and precedence is applied in code. A catalogue and the equivalent hand-written rules and facts give identical results: it is one engine.

```yaml
# switches/equipment.yaml
name: equipment
fields:
  machine:        { type: text, required: true }
  firmware_major: { type: number }
  pressure_bar:   { type: number }
  active_alarms:  { type: json, source: controller }
  report:         { type: text, required: true, author: operator }
reads: [report]
facts:
  overheating: { check: The current unit currently overheats or has a persistent temperature alarm. }
  leak:        { check: There is a current physical coolant leak on the current unit., unstated: true }
  procedures:
    select: Which maintenance procedures apply to this situation?
    catalog: equipment.cards.yaml
    max_claims: 16         # unsettled distinct claims judged per call; bounds the questions, not exact milliseconds
    exclusive: false       # true: more than one final card → uncertain
```

```yaml
# equipment.cards.yaml: written and reviewed by people; never generated at runtime
- id: AX410B_coolant
  title: AX-410B coolant recovery                       # for people only; never part of a prompt
  requires: 'machine == "AX-410B" and firmware_major >= 3 and pressure_bar < 1.0'   # exact
  aliases: [AX-410B]                                    # exact text evidence; only ranks cards over budget
  applies_when: [{fact: overheating}]                   # shared facts, or plain claim strings
  excludes: [{fact: leak}]                              # stated true → rejected; not stated → unresolved
  supersedes: []
  source: { manual: Maintenance manual rev 7, section: "7.3", pages: "142-145" }
  payload: { instructions: procedures/AX410B_coolant.md }   # returned, never shown to the model
- id: AX410B_leak
  requires: 'machine == "AX-410B"'
  applies_when: [{fact: leak}]
  supersedes: [AX410B_coolant]
  payload: { instructions: procedures/AX410B_leak.md }
```

**Two rules carry the design:**
1. **Ask about the situation, never about the procedure.** Titles, ids and names never appear in a judged claim; identifiers are exact evidence. Measured on the equipment set: trigger questions carrying card titles got 17/34, the same cards without titles 28/34, with no wrong procedures.
2. **A shortlist ranks; only an exact rule or an examination rules a card out.** An unexamined card is unknown, never "doesn't apply".

**Claims and identity:**
- An `applies_when` / `excludes` entry is either **`{fact: name}`**, naming a **check** fact of the switch (use this for anything shared), or a **plain claim string**, which compiles to an anonymous check fact reading the switch's `reads`. Facts never depend on `select` results, so the dependency graph is rules → facts / selects → claims, and has no cycles except `supersedes`, which is validated.
- Identity is the fact's `meaning_rev` (kind, text, `excludes`, `unstated`, `reads`, the full definition of every read field). Two entries are the same judgment only if their identities match, never because their text does.
- `applies_when` claims are closed-world: silence means they don't hold. Exclusions are open-world, three-way facts: "states true" rejects the card, and "does not say" leaves it `unresolved` with `not_stated`.
  - An exclusion given as a plain string compiles to a three-way check.
  - An exclusion given as `{fact: name}` **must name a three-way check** (`unstated: true`); otherwise it is a `ConfigError`. A two-way check would read silence as a confident "false" and let the card apply.
- **Authoring rule:** what kind of claim is it?
  - **About the text itself:** "the report requests a refund" is an ordinary check, and silence means no.
  - **About the world:** "the equipment has a leak" is a three-way check or an exact observation (`requires: 'leak_sensor == true'`). Silence about the world is not evidence that it is false.
- **Authoritative codes are exact rules on structured fields:** `requires: 'mentions(active_alarms, "E17")'`. A code merely mentioned in the operator's text is an `alias`. It ranks, and never decides ("a historical E17 on another machine" mustn't qualify).

| Card state (base) | Means | `has "id"` |
|---|---|---|
| `ineligible` | Its `requires` is FALSE | FALSE |
| `unexamined` | Its claims weren't judged this call (over budget) | UNKNOWN (`not_examined`) |
| `applicable` | Every `applies_when` claim holds and every exclusion is stated false | TRUE, unless superseded |
| `rejected` | An `applies_when` claim is false, or an exclusion is stated true | FALSE |
| `unresolved` | Otherwise: a claim unsettled, an exclusion not stated, a missing field, low answer mass, or a model error | UNKNOWN |

**Per call:**
1. **Exact filter:** `requires` decides eligibility. An UNKNOWN `requires` keeps the card as unresolved-capable and records the missing field.
2. **Cheap settlement:** facts settled by rules, corrections or lexical heads are used as is. A card that a settled claim already rejects spends no budget.
3. **Budget** (`max_claims` counts **unsettled distinct claims** still needing the judge). How a claim is written never changes its budget: a claim needed **only** through this catalogue obeys the budget, whether it is a string or `{fact: name}`. A fact the rules need directly (or an output of a switch without rules) is scored for that reason, and the catalogue reuses it for free.
   - **Within budget:** every remaining claim is judged, and the result is `complete`.
   - **Over budget:** rank the remaining cards (alias mentions first, then BM25 over their titles, aliases and claims). Take cards **whole** in rank order: add a card's new claims if they fit, otherwise skip it and continue. Claims already scheduled are reused. A skipped card whose claims all got judged through other cards is settled anyway (card states are evaluated from claim states). The rest stay `unexamined`, and `complete` is false.
4. **Judge** the scheduled claims in one read of the situation.
5. **Precedence:** one pass over base states, order-free.
   - **Final:** a card is final if it is `applicable` and no `applicable` card lists it in `supersedes`.
   - **May be superseded:** an `applicable` card that some `unresolved` or `unexamined` card lists in `supersedes` becomes `unresolved` ("may be superseded by X").
   - **Base states, not final ones:** suppression reads base applicability, so with A > B > C all applicable, B still suppresses C and only A is final.
   - **Cycles:** self-references and cycles are compile-time `ConfigError`s that name the cycle.
6. **`exclusive`:** the answer settles only when exactly one card is final **and** every other card is ruled out (ineligible, rejected or superseded). Two final cards, or one final card with a rival still unresolved or unexamined → `uncertain`. Exclusive selects therefore examine every card the budget allows.

**Result** (`r.facts[name]`):
- `value`: the final card ids (one id when `exclusive`);
- `complete`: every eligible card was examined;
- `states`: base state and final flag per card;
- `unresolved`: `{id: need}`;
- `superseded`: `{id: by}`;
- `cards`: the final cards' `id`, `source` and `payload`. Payloads are borrowed from the loaded catalogue, not copied; treat them as read-only.

**Known subset vs settled answer.** A rule can rely on `procedures has "P01"` as soon as P01 is final, even while other cards are unresolved; that is the point of composing with rules. The fact's own `value` is settled only when no card is unresolved. `r.facts[name].settled` and `complete` make the difference explicit.

**Settled vs complete.** `complete` only says everything was examined. **`is empty`** is TRUE only when the search is complete **and** every card is `ineligible` or `rejected`, or applicable but superseded. It is FALSE when a card is final, and UNKNOWN otherwise. An incomplete search, or any unresolved card, never yields `none`.

At load, the compiler builds the claims, claim-to-card links, precedence expressions and the ranking index: each card's term frequencies and length, and the catalogue's document frequencies. A call only scores its query against that index, and only when it is over budget.

### 4.5 Grammar and atoms

```
expr := or ; or := and ("or" and)* ; and := not ("and" not)* ; not := "not" not | "(" expr ")" | atom
atom := FACT | FACT ("=="|"!=") STR | FACT ("in"|"not in") "[" STR ("," STR)* "]"
      | FACT ("<"|"<="|">"|">=") STR | FACT "has" STR | FACT "is" "empty"
      | FIELD ("=="|"!="|"<"|"<="|">"|">=") LIT | FIELD ("in"|"not in") "[" LIT ("," LIT)* "]"
      | "exists" "(" FIELD ")" | "mentions" "(" FIELD "," (STR | "[" STR ("," STR)* "]") ")"
LIT  := STR | NUMBER | "true" | "false"
```

There is no `eval`, no attribute access and no other functions. Unknown names and type mismatches are load-time errors.

| Atom | Evaluated on | Notes |
|---|---|---|
| `fact`, `fact == "x"`, `fact in [x, y]`, `fact >= "L"`, … | that judgment's distribution (below) | one **semantic atom** per maximal sub-expression over one fact |
| `fact has "id"`, `fact is empty` (select) | card states | §4.4 |
| `field OP literal`, `field in [...]` | exact | UNKNOWN when the field is missing |
| `exists(field)` | exact | never UNKNOWN |
| `mentions(field, …)` | exact | §3.4 boundaries; "does it name it", never "what does it say about it" |

**A condition over one judgment is evaluated on its distribution, never assembled from separately accepted pieces, and never changed by how it is written.**
- **Compile time:**
  1. **Flatten:** nested `and` / `or` of the same operator are flattened first, because both are associative and commutative. `(u >= "today" and g) and u < "high"` and `u >= "today" and u < "high" and g` compile to the same atoms. Nothing else is rewritten; there is no distribution and no probabilistic algebra.
  2. **Group:** within each flattened `and` / `or`, the operands that refer to the same fact (and `not` over them) become **one semantic atom**: `u >= "today" and u < "high"`, `u == "today"`, `topic in [billing, technical]`. Operands over different facts stay separate and are combined in Kleene logic.
  3. **Truth vector:** the compiler enumerates the fact's values (at most 20) and records the atom's truth vector: TRUE, FALSE or UNKNOWN for each value. Three-way "does not say" is UNKNOWN for every atom.
- **Before v0.26, both of the following failed** (reproduced):
  - **grouping only looked at immediate children:** at 0.08 / 0.84 / 0.08, `(u >= "today" and g) and u < "high"` returned `ok` while the flat form returned `uncertain`;
  - **`not` was folded into the truth vector before acceptance:** a disabled FALSE side was bypassed by writing `not leak`.
- **Runtime, by the fact's kind:**
  - **Check facts: answer first, then evaluate.**
    - The check settles its **answer**: *true* at `p(true) ≥ θ_true`, *false* at `p(false) ≥ θ_false`, *unstated* when the "does not say" mass reaches `θ_true`; otherwise it is unsettled.
    - Every atom over the check is its truth vector **evaluated at that answer** (UNKNOWN while unsettled).
    - `not leak` is therefore Kleene negation of a settled answer: a disabled side stays disabled whatever the expression, and asymmetric thresholds can't make `not x` differ from negating `x`.
  - **Pick and rate facts: set events under one θ.**
    - With `p` the distribution, `pT` is the mass of values where the atom is TRUE and `pF` the mass where it is FALSE.
    - The atom is **TRUE** at `pT ≥ θ`, **FALSE** at `pF ≥ θ`, otherwise **UNKNOWN**.
    - One symmetric θ makes negation and complements consistent by construction: `not S` is the complement set under the same θ. Since θ > 0.5, exclusive atoms can't both be TRUE, and a narrower atom is never more probable than a wider one.
- **Example:** with 0.08 / 0.84 / 0.08, `u == "today"` has `pT = 0.84` → UNKNOWN, written either way.
- **A set condition is a different event from an answer.** With A .46 / B .46 / C .08, no answer settles, but `topic in [A, B]` is TRUE at 0.92. Policies are tuned on answers, so `ss lock` reports each compiled condition's own held-out precision separately (§5.3); a judgment's target does not transfer to its set conditions automatically.

**Kleene logic combines accepted judgments; it does not compute the probability that a compound decision is right.** `legal_threat and topic == "billing"` is TRUE when both atoms are accepted, and that is weaker evidence than either alone. End-to-end correctness is measured with `ss test`.

**One pure function decides acceptance:** `settle(judgment, p, truth_vector, policy, mass, pending) → (state, reason)`. It applies the answer-first rule for checks and the set rule for picks and rates. Inline results, switch atoms, a fact's own value, lock tuning and validation, `specialize` and `ss test` all call it. Nothing else interprets scores.

**Reasons for UNKNOWN,** in order:
1. answer mass under 0.5 → `low_answer_mass`;
2. eligibility-UNKNOWN options would have tipped the atom → `missing_input`;
3. the "does not say" mass is `≥ θ_true` → `not_stated`;
4. otherwise `uncertain`.

**Policy.** One policy per judgment, until `ss lock` tunes it:
- **checks:** `(θ_true, θ_false)`, one per answer;
- **picks and rates:** a single `θ`.

The default is 0.90 (or the fact's `min_score`). It is the same policy as inline calls (§3.1). There are no per-atom thresholds, so all atoms over a fact agree.

**Eligibility:**
- An option whose `requires` is FALSE is removed before scoring (its mass is dropped and the rest renormalised), and from exact overrides too.
- An option whose `requires` is UNKNOWN maps to UNKNOWN in every truth vector, so it can never make an atom TRUE.
- If every option is ineligible, the fact is settled by rule, with no model call.

### 4.6 Resolution

Strong Kleene logic: `T and U = U`, `U and F = F`, `T or U = T`, `not U = U`. Rules are evaluated in order: TRUE → `ok`; FALSE → next; UNKNOWN → blocked; all FALSE → `otherwise`.

A blocked rule is classified from the UNKNOWN atoms left in its simplified expression:

| Remaining UNKNOWN includes | Status | Reported |
|---|---|---|
| `missing_input` or `not_stated` | `needs_input` | `missing` (fields, `fact:<name>`, `card: exclusion`) |
| `not_examined` | `uncertain` | "incomplete search: P07, P11 not examined" |
| `model_error` | `failed` | `errors` |
| otherwise | `uncertain` | the uncertain facts |

Errors in facts that don't decide never change the result. `resolve: all` returns every TRUE rule, always as tuples.

### 4.7 Execution

**At load:** compile, type-check and canonicalise; read thresholds, heads and examples from the lock (or use defaults).

**Per call:**
1. Project and validate the input.
2. Evaluate exact atoms and eligibility.
3. Find the facts the rules still need: **directly** (their atoms are in a rule) or **through a `select`** (only its cards need them; these obey that catalogue's budget, §4.4).
4. Settle what's cheap: rules, then exact overrides, then lexical heads. Re-simplify after each.
5. **Two-stage scoring:** score the first unresolved rule's facts in one round; if that settles the switch, stop; otherwise score everything still needed in a second round. Each round is one batched judge call.

   **Two execution surfaces,** because responsiveness and throughput are different needs:

   | Surface | Behaviour | For |
   |---|---|---|
   | `sw(input)`, `sw.many(iterable)`, `ss run --each` | **streaming:** each input runs all its rounds and its result is yielded before the next input is read | interactive use, pipes, iterators that wait on their consumer |
   | `sw.run(list)`, `ss run --each --batch N` | **finite batch:** inputs go in windows of the judge's batch capacity; each round runs for the whole window in **one decode** (every input in its own sequence, every question in its own branch); an input that settles in round 1 leaves the window's later rounds | throughput over a known set |

   - **The judge declares its batch capacity** (`Judge.batch_inputs`): `LlamaJudge` on a GPU 8, on the CPU 1, and hosted judges their own. The executor doesn't guess from the device.
   - Results always come back in input order.

   **Measured** (one question per input, default judge):
   - **GPU:** one decode over 4 / 8 / 16 inputs gave 2.5× / 3.5× / 3.9× the throughput of sequential calls at 20 tokens, and 1.3–1.4× at 261 tokens, with identical scores (max difference 0.0002).
   - **CPU:** 0.92–1.13×, so no gain; its compute is already saturated.
   - Measured with the cache (E4, route_ticket, default judge): when the first rule decides, 243 ms against 942 ms for scoring every fact at once; when a later rule decides, within 0–8% of it. There is no `schedule` option.
6. Accept (`settle`), resolve, build the result. A rendered projection is built once per call and reused by heads, questions and the result.

Engine overhead is about 0.2 ms per call.

### 4.8 Result

```python
r = route(ticket)
r                # <ok · 'billing' · rule refund_other · 2 judged · 181 ms>
r.status, r.answer, r.rule, r.missing, r.invalid, r.uncertain, r.blocked_by, r.errors
r.facts          # {"refund_requested": FactResult(True, 0.96, judge), ...}
r.cost           # {"rules": 3, "example": 0, "classifier": 0, "judge": 2, "rounds": 1, "tokens": 412,
                 #  "elapsed_ms": 181.0, "amortized_ms": 181.0}
print(r.explain())   # rules ✓/✗/·, facts with value, score, source, why; explain(raw=True) adds atom states
```

**`FactResult(value, score, settled, source, why, scores)`:**
- **`value`** is set only when **the fact's own answer** settles. A rule may fire on `topic in [billing, technical]` while `topic.value` stays `None`.
- **`source`** is `rule`, `example`, `classifier` or `judge`.
- **`why`** is short ("uncertain", "low answer mass", "missing: payment.status", "not stated in the input", "error: …"). Reasons are typed internally, and the strings are made at the boundary.
- **Truncation is visible:** `r.truncated` lists the fields cut by `max_chars` for this call. A settled answer on a truncated input stays settled, but the result says what the judge didn't see.
- **Internals:** compiled atom keys and raw distributions never appear in the result.
- **Timing, two quantities:**
  - **`elapsed_ms`** is the wall time of the operation that produced this result: what a caller actually waited;
  - **`amortized_ms`** is that operation's time divided by its inputs, a throughput figure.

  They are equal for a single call. Eight results from one 400 ms batch each show 400 elapsed and 50 amortized.

---

## 5. Teaching and evaluation

### 5.1 Corrections

```python
route.correct(ctx, fact="refund_requested", value=False, actor=ss.Actor("human"))    # also "false", "u" for three-way
route.correct(ctx, rule="refund_other", actor=ss.Actor("human"))                     # answer label → test file + derived fact labels
```

- **Who wrote it:** every correction is an `authored` label naming its actor: `ss.Actor("human")`, or `ss.Actor("agent", model=…, session=…)` when an agent writes it (`author=` defaults to the login name; `ss correct`/`ss label` take `--actor`, `--model`, `--session`). The actor is traceability, not validation: a correction counts as verified only once a verification record confirms it.

- **Examples file:** one JSON line per example, `{"id", "fact", "meaning_rev", "input", "value", "created"}`. It's append-only, and a later line with the same fact and input replaces the earlier one.
- **Exact override:** when an active example's canonical key equals the input's projection and its value is eligible, the fact takes that value (`source="example"`, no model).
  - **Canonical key:** the fact's untruncated, typed projection, as JSON with sorted keys and NFC strings.
  - **No fuzzy matching.**
- **Timing and scope:** on a draft, a correction takes effect immediately; on a locked switch, after the next `ss lock`. `correct()` says both: `{"recorded": 1, "scope": "this exact input", "effective": "now"}` (or `"after ss lock"`). A correction never changes how any other input is judged.
- **Derived labels:** a rule correction derives direct check atoms (TRUE in the chosen rule, and FALSE in earlier single-atom rules **under first-match only**) and positive pick equality. Similar inputs are never relabelled.

### 5.2 Revisions

| Hash | Covers | When it changes |
|---|---|---|
| `meaning_rev` (per fact) | statement or question, `excludes`, `unstated`, options, levels, `requires`, `reads`, and the **full definition of every read field** (type, `values`, `max_chars`, `truncate`, `author`, `source`) | that fact's examples and labels go stale (ignored, with a warning); its head is dropped |
| `switch_rev` | rules, `otherwise`, `resolve` | answer labels go stale |

Tool-written labels record the revisions they were made under; hand-written lines without them are trusted. Changing the model keeps labels; changing what a fact reads, or how it is rendered, doesn't.

**Answer labels don't depend on fact revisions.** A rule label records the decision you want for that input. Rewording a fact to fix it must not invalidate the answers you want; it only invalidates that fact's own labels. `ss diff` shows what the rewording changed.

### 5.3 Lock, test, diff

The test file holds labelled inputs in the versioned envelope: `{"schema": 3, "input": …, "facts": {…}, "label_sources": {…}, "rule": "…", "groups": […], "pairs": […], "role": …, "tags": […], "selections": […]}`. `label_sources` mirrors `facts`: every labelled fact or member keeps an ordered history of sources. Each source records three separate dimensions: its **derivation** (`computed` by a deterministic procedure with its revision, `judged` by a judge/model route, `authored` directly by a recorded actor — `{"kind": "human"}` or `{"kind": "agent", "model": …, "session": …}` — or `unknown`), its **verifications** (verifier, actor, method `procedure-recompute`/`source-check`/`review`, the criterion revision checked against, content digest, outcome) and, for judged labels only, its **training use** declaration. Who wrote a label is never a verification of it. The per-target `label_derivation` in exports and the `label_provenance` summary in reports are derived from that history (`mixed` when contributors differ), never inputs.

**One labelled-data collector.** Export, lock, `specialize` and membership all read labels through a single collector, so tools can't disagree about what an example is:
- **Sources:** corrections, test-file labels, and (if E8 is adopted) teacher labels.
- **Merging:** records are merged per fact meaning and input projection; one input is one example, whatever its source.
- **Groups are joined, never chosen.** Each example starts in its decision key's group plus every explicit `group` any of its records names. Groups that share an example are merged (union-find). Two explicit groups meeting on one input are joined, with a warning naming both. Joining can only make held-out sets more conservative; choosing one would leak the other.
- **Conflicts:** conflicting labels for one example are a `ConfigError` naming the input. Agreeing records merge their source histories; no derivation wins by rank.
- **Splits and folds** are assigned to the joined groups, never to raw records. **Corrections are examples too:** a correction whose group falls in the check half validates and never trains, while it still fixes its exact input at runtime.
- **Taught inputs are excluded from validation:** an exact correction settles its input by rule, so it is no evidence for any policy.

It is split once, deterministically, into **tune** and **check** halves:
- **by the decision key:** the canonical projection of every field any fact reads or any exact rule tests, so two lines differing only in unused fields land together;
- **with an optional `group` kept together,** for translations, paraphrases and near-duplicates.

**`ss lock`** is the only writer of frozen behaviour. `--dry-run` shows what it would choose without writing.
1. Compile.
2. **Admit supplied lexical heads** (`--head FACT=BUNDLE`), each fitted by `semantic-trainer lexical` on that fact's **tune-split examples** (corrections and labels alike, §5.4).
3. **Choose one policy per judgment and source** (the judge; the lexical head):
   - **checks:** the TRUE side scores "the answer is true"; the FALSE side scores "the answer is false";
   - **picks and rates:** a single θ scores "the top answer is the labelled one";
   - candidate θ run from 0.99 down to 0.51;
   - pick the θ with the **highest coverage** whose precision is at least the fact's `target`, among θ with at least 20 accepted items. Precision isn't monotonic in θ, so the scan doesn't stop at the first failure; among equal coverage, the highest θ wins;
   - **where the scores come from:** the judge's on the tune half. A head's are **out-of-fold** predictions over its own training rows, with folds that keep each `group` (or decision key) together, so a head is never tuned on inputs it was trained on;
   - **acceptance is the runtime's:** a candidate θ counts an answer as accepted only if `settle` (§4.5) would accept it, with answer mass, silence and pending options included. An answer the runtime would reject never counts toward coverage or precision. Without this, a judge returning p(true) 0.99 at answer mass 0.10 was reported `met` on 57 held-out answers that the runtime returns as `uncertain`.
4. **Validate each chosen θ once on the check half,** which no step above has seen (taught inputs excluded).
   - **When a head exists, the judge is validated on the inputs the head leaves to it.** That residue is the judge's real workload in the deployed cascade, and it is harder than the full population: a judge right on 980 of 1,000 can be 80% precise on the 100 a head doesn't settle.
   - **Every side ends in one of three recorded states,** always shown with its count and 95% interval. They are observations, not guarantees: 20/20 correct has an interval down to about 84%.
     - **observed target met:** at least 20 check-half items were accepted and their precision reaches `target`. The side settles at θ.
     - **observed target missed:** no θ met the target on the tune half, or the check half shows precision below it. **The side never settles**: the switch abstains rather than guess, and the other side may still settle.
     - **insufficient evidence:** too few labels to choose θ, or to validate it (fewer than 20 accepted on either half). The judge keeps its default policy (or `min_score`); a head never settles.
   - **The consequence is reported, not hidden.** Adding labels can move a side from "insufficient evidence" (default policy, settling) to "observed target missed" (abstaining). The lock shows each side's check-half coverage under the default and under the result, so a drop in automation is visible.
5. **Report two more levels of evidence on the check half.** They are reported, not gated: a target is set per judgment, and doesn't transfer automatically.
   - **Conditions:** each compiled atom the rules use (for example `topic in [A, B]`, or `urgency >= today`), with its own held-out precision and coverage.
   - **Decisions:** the deployed snapshot (corrections → heads → judge → resolution) runs on the check half's labelled inputs. It reports correct, wrong, abstained, coverage and judge calls avoided. This is the number a user acts on.

   **Why the check half, measured** (300 random splits by sentence family, target 0.95, true side):
   - **Fresh suite:** a θ chosen on one half reached only **83.4%** precision on the other, and **0 of 300** splits met the target. Tuning-only locks had reported "tuned" in all 300. The judge's errors are confident (p ≥ 0.99), so the half that tunes can't see the other half's.
   - **Selection suite:** the gate passed 255 of 300.

   A lock that skipped validation would promise a precision it doesn't deliver.
6. Run the YAML `tests` against the frozen plan; any failure aborts.
7. Record membership per merged example: **trained** (head training), **tuned** (policy choice), **validated** (check half), and **trained the judge** (from an adapter's training manifest, §6.6). If the judge was trained on check-half examples, the lock warns that its validation is not independent.
8. Write one self-contained file atomically: spec, examples, heads, policies with their states and held-out evidence, the condition and decision reports, membership and model identities.

**Groups are load-bearing.** Translations, paraphrases and near-duplicates must share a `group`, or the split leaks. On the fresh suite, a readout trained with the translated twin of a test item present scored 0.931 accuracy; with the whole translation family held out it scored 0.844, below the plain judge.

**Model pinning:**
- **At load,** each local model's sha256 and the prompt-template version are compared with the runtime's. A mismatch fails with a `ConfigError`.
- **Hosted judges can't be pinned to a file.** The lock records the requested model name and the model identity the provider returns, and warns when the returned identity changes. That is the reproducibility a hosted service allows.
- **`allow_model_change=True`** resets the affected thresholds, with a warning.
- **Rollback** is `git checkout` of the lock file.

**`ss test [--data f]`:**
- Reports in plain sentences (JSON when piped).
- Reports three groups separately:
  - **taught:** the input is an exact correction;
  - **used by the lock:** it trained a head, tuned a policy, validated one, or trained the judge (from an adapter manifest). Validation decides which policies run, so it is not an untouched evaluation; the report names the use;
  - **fresh:** everything else, the only group that measures generalisation. After a lock, fresh inputs are new lines in the test file or a `--data` file.
- **Never counts training, tuning or validation inputs as fresh**, whatever the file later gains. Membership is matched by fact projection, so an input sharing a projection with a used one counts as used.
- Gives, per check fact:
  - **the deployed result** (whatever settled it: correction, head or judge): silent misses, and each policy side's state (observed target met / missed / insufficient evidence) with its held-out precision and interval;
  - **a separate judge-only pass** for diagnosis: the judge's **ranking AUC** (true above false). Corrections and heads never contribute to it.
- **Contrast metrics:**
  - **both-correct pair rate:** both members of a `pair` answered right. Correct ordering alone isn't enough; 0.99 vs 0.98 passes ordering;
  - **group invariance:** among groups whose members carry the **same label** (translations, paraphrases), the share answered consistently. Scenario families with different labels inside are excluded; for those, differing answers are expected.
- Exact 95% intervals wherever sampling supports them, reported automatically, and a note when fewer than about 60 accepted inputs make 90% and 95% indistinguishable.
- **Diagnosis, worded for what it shows:**
  - **AUC ≥ 0.9 with errors:** "ranking is strong; a boundary change may help". Strong ranking doesn't prove every error is a boundary error;
  - **AUC below about 0.75:** "the judge barely separates this claim; reword it, add exact rules, or use `--precise`".

**`ss diff`** runs the lock and the edited draft and lists every input whose **observable outcome** changed:
- With rules: status and answer.
- Without rules: status and settled fact values.
- `select`: applicable ids and `complete`.

Its inputs are the test file plus every example record, stale ones included as inputs, not as truth.

### 5.4 Lexical heads: cheaper repeats

A per-fact linear model on hashed word 1–2-grams and character 3–5-grams of the rendered input. It costs microseconds and no model call. It keeps codes and exact wording, and like any classifier it doesn't fix negation. It buys **speed on recurring wording**, never better semantics.

- **Training data:** that fact's examples in the **tune split** of an `ss export`: corrections and test labels, merged by the collector (§5.3). A correction whose group falls in the check half doesn't train; it validates. `semantic-trainer lexical EXPORT --to BUNDLE` fits it after the same fitting admission as LoRA; it needs at least two distinct values, otherwise fitting fails with a `ConfigError` naming the fact and the value it has. `ss lock SWITCH --head FACT=BUNDLE` binds it.
- **Binding:** bound to the fact's `meaning_rev`, so editing the claim drops it.
- **It never settles unless validated:** its policy is chosen on out-of-fold predictions and must be `met` on the check half.
- **Fallback:** whenever an atom the decision needs is still UNKNOWN after the head, the judge scores the fact.
- **`ss analyze-head <switch> <fact> --head BUNDLE` measures a supplied head, through the executor users run:**
  1. Gather the fact's labelled inputs (examples and test labels) and assign 5 folds by `group` (or decision key).
  2. Per fold, train the head and choose its policy on out-of-fold predictions of the training part. This runs **outside** the timed section.
  3. Install the head in a temporary snapshot whose corrections exclude the held-out fold.
  4. Run the held-out inputs through the executor twice, judge alone and head-then-judge. **Measurement:**
     - the judge is loaded and warmed before any timing; load time is reported separately;
     - the two arms alternate order from fold to fold, so neither always inherits the other's cache;
     - latency is total elapsed time over total inputs, not an average of fold averages.

  **It reports:**
  - the judge alone: correct, wrong, abstained, ms per input, and its ranking AUC with the diagnosis above;
  - with the head: judge calls avoided, correct answers **gained**, wrong answers **avoided** (by abstaining), wrong answers **introduced**, and ms per input.

  **Admission is advisory:** it recommends the head only if it avoids judge calls without adding wrong answers. Binding it with `ss lock --head` is the developer's decision; `ss lock` does not re-check admission.

**Readout adaptation was measured and is not offered.**
- **The idea:** a probe on the judge's hidden state, Platt calibration, or a free cutpoint, each trained on about 16 labels per judgment.
- **The result:** with whole translation families held out, none reliably beat the plain judge on new expression families:

  | Readout | Fresh-suite accuracy | Settled at 0.9 | Precision at 0.9 |
  |---|---|---|---|
  | judge | 0.856 | 90.6% | 89.0% |
  | calibration | 0.875 | 66% | 90.6% |
  | probe | 0.844 | 76% | 92.6% |
  | free cutpoint | 0.831 | — | — |

  The differences sit inside one standard error (about 2.7 points), and higher precision came only from settling less. An earlier "+7 points" for the probe came from translated twins in training. A boundary-only claim ("leave": AUC 1.0, accuracy 0.80) was fixed by calibration and by the probe alike; a weakly separated one ("decline": AUC 0.67) by neither.
- **The remedy is diagnosis** (`ss test`, `ss analyze-head`): reword, add exact rules, abstain, or use `--precise`.
- **Verdict: rejected.** It is not offered, and its benchmark code is scheduled for removal.

**Specialisation keeps the base model frozen.** The intended path is LoRA specialists on the off-the-shelf judge, decided by E8 (`EXPERIMENTS.md`):
- trained externally from `ss export` (§5.7); the library never trains model weights;
- adopted only if they beat the plain judge on held-out families.

Any other model, including a fully fine-tuned one, remains just another `--judge`.

### 5.5 What each teaching action changes

Users of example-based routers report adding examples and seeing no change, because "examples", "tuning" and "training" blur together. Here each action has exactly one effect, and each says what it did:

| Action | Changes | Doesn't change | Reports |
|---|---|---|---|
| `correct()` | that exact input's fact (exact override) | any other input | `scope`, `effective` |
| `ss label --to test` | the labelled data only | any answer, until you lock | how many inputs were labelled |
| `ss lock` | **when** each judgment answers (its policy), validated on held-out labels, and trains configured heads | what the judge thinks | each policy side: observed target met / missed / insufficient evidence, with held-out precision and interval; condition and decision reports |
| `ss lock --head FACT=BUNDLE` | adds a cheaper predictor in front of the judge | the judge; inputs the head can't settle | `ss analyze-head`: errors introduced and corrected, coverage, latency |
| `--judge` / `--precise` | the judge itself, so every semantic answer | rules, corrections | `ss doctor --quality`, then `ss test` (re-lock required) |

No implicit learned changes:
- there are no prompt examples and no automatic threshold relaxation as labels or categories grow;
- policies and heads change only at `ss lock`;
- the one immediate change is an explicit correction on a draft, which fixes exactly its input.

### 5.6 The teaching path

```bash
# 1. correct as you go (Python: sw.correct(...))
ss label support_route new_inputs.jsonl --to test --actor human     # 2. labels for tuning and checking (y / n / u)
ss export support_route && semantic-trainer lexical support_route.export.jsonl --to head/ --fact refund
ss analyze-head support_route refund --head head/     # 3. does a lexical head help? (reports only)
ss lock support_route --head refund=head/             # 4-5. if it helps: bind it, tune and freeze
ss test support_route                                 # 6. measure on unseen labelled inputs
```

### 5.7 `ss export`: training data for an external judge or adapter

`ss export <switch> [--to rows.jsonl]` writes one row per labelled judgment, in the **exact runtime form**:
- **Readable context:** the rendered input text, the claim or question text with its options or levels, and the fact's `excludes`, so a row can be inspected and used by other tools.
- **Model input:** the judge identity and template, then the token ids of each segment (prompt prefix, rendered input, question suffix) **from the runtime's own tokenizer**. Tokens, not text: re-tokenising a concatenation can merge segment boundaries and silently change the training input (an activated LoRA's invocation boundary is exactly such a boundary).
- **Answer:** the answer-slot token ids and the labelled slot.
- **Bookkeeping:** fact, `meaning_rev`, decision key, joined `group`, split, and example id.
- **Manifest:** `ss export --split tune` (the default) writes only tune-split rows, so the check half stays independent of anything trained on the export. A manifest beside the rows lists the example ids, groups and splits exported. `--split all` exists, and its manifest says so.
- **Merging:** a correction and a test label for the same input and fact become one row. An explicit `group` is kept, and conflicting labels fail with a `ConfigError` naming the input.
- **Writing:** rows go to a temporary file that replaces the destination only after every row was built. A destination that is one of the switch's own files is refused.

It is the only contract with external training. The trainer owns how the model learns; `ss doctor --quality`, `ss test` and the lock own whether the resulting artifact is used. Never export or train on Jev outputs (TypeSafe's Master Customer Agreement §2.3(b)).

---

## 6. Runtime

### 6.1 Layers

| Layer | Owns | Modules |
|---|---|---|
| **1. Definition** | **`Judgment`** (check, three-way check, pick, rate) is **data**: meaning, identity (`meaning_rev`), its question text and its value parsing (every label, example, correction and CLI answer). **Acceptance** is one pure function, `settle(p, truth_vector, policy, mass) → (state, reason)` (§4.5). Also the catalogue index (claims, links, precedence expressions and ranking statistics prepared at load), `Plan`, and one `render`. | `judgment.py`, `compiler.py`, `context.py` |
| **2. Evidence** | Exact atoms, mentions, eligibility, exact overrides, catalog shortlist, lexical heads | `kernel.py`, `executor.py`, `examples.py`, `heads.py` |
| **3. Scoring** | `Judge` protocol; `LlamaJudge` with the input cache; adapter; native backend | `judge.py`, `adapters/`, `backend.py` |
| **4. Resolution** | Three-valued logic, rules, relevance, card precedence and completeness | `kernel.py`, `executor.py` |
| **5. Interfaces and tools** | Inline API, `Switch`, results, CLI; `lock`, `test`, `diff`, `label`, `specialize`, `doctor`, `bench` **only call layers 1–4** | `inline.py`, `switch.py`, `results.py`, `api.py`, `cli.py`, `tools/` |

Inline calls keep their own short driver, because callables and per-call objects don't belong in a compiled plan. They build `Judgment`s and call the same acceptance functions with the same policy, so **inline and saved judgments answer identically by construction.** Catalogue claims are `Judgment`s too. No tool keeps its own acceptance, parsing or question rules. `Judgment` is a definition, not an executor: relevance, cheap settlement and scheduling stay in the executor, tokens in the backend.

**Data structures:**
- **The compiled plan:** frozen, slotted, with precomputed indexes.
- **Per-call state:** truth states as small integers and `FactState` per fact. `select` needs no state of its own: each card's base applicability and final flag are Kleene expressions over its claim atoms and exact atoms, compiled at load, so relevance, short-circuiting and staged scoring apply to cards exactly as to cases. The call adds only which cards the budget examined.
- **Authoring vs runtime:** Pydantic validates YAML; compiled and per-call data are plain slotted records.
- **Candidates and payloads:** borrowed references.
- **Public results:** built at the boundary.

### 6.2 The `Judge` protocol

```python
class Judge(Protocol):          # text and questions in, scores out
    identity: dict              # pinned by locks
    accelerated: bool
    max_slots: int
    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]: ...
    def fit(self, text: str, limit: int) -> tuple[str, bool]: ...   # candidate text within this judge's budget; a judge without tokens returns it unchanged

@dataclass(frozen=True, slots=True)
class Scores:
    probs: np.ndarray | None    # normalised over the question's answer slots
    mass: float | None          # None for judges without answer tokens; never "low"
    error: str | None
```

`LlamaJudge` (adapter + scorer) implements it, and so can any non-decoder judge (a cross-encoder, a remote API) directly.

**Hosted judges** implement the same contract through one built-in binding, OpenRouter (`ss.judge('openrouter:MODEL')`): chat log probabilities for ordinary models, System One for an explicit table of verified structured models (`typesafe/jev-1.13`, used only with TypeSafe's consent; `openai/gpt-6-luna-decisions`). Applications may supply any other `Judge`. The structured mapping:
- **Probabilities:** per-option and per-level probabilities map to `probs`. A predicate's probability p maps to (p, 1 − p).
- **Answer mass:** `None`, since there are no answer tokens.
- **Refusals** become `Scores(None, None, error)`.
- **Provider "confidence":** carried in the optional `Scores.confidence` field as a diagnostic only. Acceptance reads probabilities through `settle`, and `ss lock` calibrates against your labels (OpenAI's own guidance).
- **A provider's scalar score** (OpenAI's is the probability-weighted mean of level indices) **is never used in rules.** A distribution split between levels 0 and 2 has mean 1 with no mass on level 1. Rate atoms sum probability over level sets (§4.5).
- **Training: hosted outputs never become training data. That is this library's policy, stated as such.**
  - **Jev:** TypeSafe's agreement §2.3(b) forbids training on its outputs.
  - **OpenAI:** the terms forbid using output to develop competing models, with a *Permitted Exception* for internal classifiers that are not distributed or sold.
  - **Why a blanket policy:** the library can't verify which case a user is in, so the tools never route hosted outputs into training (no `ss distill` from hosted judges, no hosted rows in `ss export`). Users with their own legal basis can still do it outside the library.

**Output contract,** checked at one boundary function that every caller (inline, executor, tools) goes through:
- one `Scored` per item, one `Scores` per question, in order;
- `probs` has exactly the question's width, finite, non-negative, summing to 1 within 1e-3;
- `mass` is `None` or a finite value in [0, 1] (NaN used to slip through and settle an answer);
- `confidence` is `None` or a finite value in [0, 1], and is never read by acceptance;
- a per-question failure is `Scores(None, None, error)`.

The judge also declares **`batch_inputs`**, how many independent inputs it scores per call (§4.7). The executor uses it to size batch windows.

A judge that breaks the contract raises a `TypeError` naming the judge, the item and the violation. It is a programming error, not a result. Below it: `Scorer.score(ScoreRequest…) -> ScoreBlock`. Rows are reduced through views before the next decode, and no result keeps native memory. The reduction is a logsumexp over the whole vocabulary per answer row (for answer mass), about 0.2 ms; only the compact result is kept.

### 6.3 Adapter

**One generic adapter, driven by the model's own chat template** (`chat@2:<template hash>`):
- It renders the GGUF's `tokenizer.chat_template` in a Jinja sandbox, with thinking off, a system message and a sentinel user message.
- It prepends BOS when needed, and finds single-token `true` / `false` and `A`–`T`.
- A model without a template or without single-token answers is refused at load.
- The hand-written Qwen3 prompt (`qwen3@4`) is byte-identical and stays as an alias.

**Template and content are tokenised differently:** the template with special tokens on, user content with them off. An input containing `<|im_end|>` can't close the turn.

**Prompt rules** (measured; the slow tests keep them honest):
- **Questions are about the situation only.** Procedure names, titles, ids and codes never appear in a judged claim; those are exact evidence. Mixing them in cost 11 of 34 equipment cases (§4.4).
- Checks are claims answered `true` / `false`; yes/no framing produced a yes bias.
- Exceptions read "Except: X." (one line each).
- Options and levels are lettered, with an explicit list.
- The template isn't trimmed further; shorter variants lost accuracy.

The template hash and version are part of the judge identity, so a prompt change invalidates locks.

### 6.4 Backend

llama-cpp-python's low-level API: tokenise, batch decode, `llama_get_logits_ith`, `llama_memory_seq_cp` / `seq_rm`.

```
seq 0: system prompt, decoded once per context
seq 1..W: system + input, one sequence per input in the window (W = 1 on CPU, up to 8 on GPU), kept between rounds
per round, under the context lock:
  for each input: reuse its sequence if (prompt, input) is unchanged, else rebuild it from seq 0
  decode every new input in one batch (n_batch chunks)
  seq_cp(input → branch) per question; decode all question suffixes in one batch, logits only at each suffix end
  reduce each row before the next decode; remove the branches
```

- **Why the cache:** without it, every round re-reads the input. On a 261-token ticket that's about 1.1 s with the default judge; staged round 2, pick and focus fallbacks, and `specialize` all repeat inputs.
- **One cached input per sequence.** A CPU context keeps one input (rounds run consecutively, §4.7); a GPU context keeps the window's inputs, one per sequence, for the window's rounds. The context is sized for the window (8 inputs at the 1,024-token input limit plus their questions), and a window that doesn't fit is split.
- **Exact reuse only, by measurement.** Common-prefix reuse (trim seq 1, decode the rest) was implemented and removed: quantised kernels give batch-shape-dependent results, so a trimmed cache made the same input score differently depending on the previous one (Qwen3-0.6B: 0.11 vs 0.06, 0.60 vs 0.48 on the probe; 9 vs 10 of 12 correct). Exact reuse returns byte-identical scores to a fresh read, and the slow suite checks it.
- **Capacity:** a request over capacity is split, or fails with `limit`.
- **Lifetime:**
  - a context serves one operation at a time, and `close()` waits for it;
  - scoring on a closed runtime raises `RuntimeError("runtime is closed")` instead of touching freed native memory;
  - a failed decode invalidates the cache, because llama.cpp may have kept part of the batch;
  - a failed context creation frees the weights.
- **Threads:** a `Runtime` may be shared across threads. Calls serialise on the context. Another thread's call can evict the cached input between two rounds, which costs a re-read but never changes a score, because reuse compares the full prompt and input tokens. No session object is needed.
- **Native logs** are off unless `SEMANTIC_SWITCH_DEBUG=1`.
- **Flash attention:** llama.cpp's automatic setting is kept (measured mixed).

### 6.5 Limits

| Limit | Default | When exceeded |
|---|---|---|
| Rendered input per reads group | 1,024 tokens | `failed(limit)` after per-field truncation (`max_chars` of about 3,000 prose characters keeps you under it) |
| Question with its options | 1,536 tokens | `failed(limit)` |
| Candidates | 20 slots; 1,000 with `shortlist` | `failed("too many candidates")` |
| Text per candidate | 64 tokens | truncated |
| Request batch | 64 questions or 16,384 tokens | split |

Limits are enforced in the `Judge`, so inline calls, switches and tools behave the same. An over-limit question fails alone.

### 6.6 `ss doctor` and the quality probe

**`ss doctor`** runs:
- **D1:** shared scoring equals the reference path;
- **D2:** batching;
- **D3:** no leakage between calls;
- **D4:** answer tokens;
- **D5:** template mass;
- **D6:** chunking (the same top answer, at most 0.25 drift at tiny batches).

**Results are cached** per model identity + backend + device + llama.cpp build. **Shared scoring is decided by D1 and D3 alone.** A hybrid model runs D1 once on first load.

**`ss doctor --quality`** scores the shipped 240-item probe and caches accuracy, AUC, ECE, coverage and precision at 0.90, per-phenomenon accuracy and latency. It is a **diagnostic, not a certificate**:
- the probe chose the default judge;
- a judge under 95% precision there triggers a one-time warning ("run `ss lock` with labelled data").

It also runs two small regression sets that probe things the classification items can't:
- **Option position:** each pick is repeated with the correct option at slots 1, 5, 10, 15 and 20 of 20. It reports accuracy per slot, and the answers that flip with order. A judge must hold its accuracy at every slot it claims (`max_slots`). The interface limit is not evidence of capability.
- **Comparative choice:** the `bench/comparative.py` items. It reports whether the choice follows the goal and stays the same when the options are reversed.

**Quantisation policy:**
- **Gemma 4:** `Q4_0`; it barely moves from Q8.
- **Qwen3:** `Q4_K_M`.
- **Qwen3.5:** `Q4_K_M` or better, never `Q4_0`: stock Q4_0 quantises its delta-net tensors to 4 bits and flips about 40% of answers.

4-bit is also faster on CPU (repacked kernels). A quantisation change changes the file hash, so it forces a re-lock.

**Devices** (measured, E3; Intel Arc B390 via a Vulkan build of llama.cpp). Adapter and device names are validated.

**Latency** (default judge; warm model; a new input each time, with the system prompt cached, which is the normal request in a running application):

| Questions on the input | GPU, 20 tokens | GPU, 261 tokens | CPU, 20 tokens | CPU, 261 tokens |
|---|---|---|---|---|
| 1 | **104 ms** | **158 ms** | 164 ms | ~1.0–1.3 s |
| 4 | 141 ms | 198 ms | — | — |
| 7 | 178 ms | 224 ms | — | — |

| Default judge | CPU | GPU |
|---|---|---|
| fresh suite: accuracy, precision at 0.9 | 0.856, 88.9% | 0.863, 89.0% (1 flip in 160) |

Earlier tables cleared the cached system prompt as well (324 / 1,009 ms CPU, 147 / 201 ms GPU), so they overstated a normal request. The CPU long-input numbers were measured on a loaded machine and vary.

`--precise` on the GPU: 248 / 330 ms per check, fresh suite 0.938 accuracy and 94.7% precision. That is faster than the default on the CPU. **The default judge stays the same on every device**, because a device-dependent default would break lock portability. With a GPU, `--precise` is the recommended judge.

- `device="auto"` offloads every layer to a GPU if the build offers one.
- **`device="cpu"` means the CPU only.** The model loads without the GPU's host buffer (`no_host`) and with op offload off. In a GPU build, llama.cpp otherwise put CPU weights in the GPU's host buffer, which disables the repacked 4-bit kernels (2× slower), and quietly offloaded large batches.
- **Quality belongs to model + quantisation + backend.** Gemma 4 E2B Q4_0 agrees across CPU and GPU to two decimals; Qwen3-0.6B Q4_K_M shifts by 0.10–0.26 on Vulkan and flips 2 of 12 probe answers. Run `ss doctor --quality` on each device you deploy on; its cache is keyed by backend and device.
- **Two-stage scoring on the GPU** is a symmetric trade: 27–29% faster than eager when the first rule decides, 16–21% slower otherwise. It stays the only behaviour.
- **`device="gpu"` is a requirement:** a build without GPU offload raises `ConfigError` instead of silently running on CPU.
- `ss doctor` and `ss bench` report the **requested device** (`gpu` = offload requested and supported by this build). llama.cpp doesn't expose where each layer finally runs, so reports never claim more than that.

**Specialists are ordinary models.** A task specialist built by `semantic-trainer select` is a fitted lock that declares its representation: merged weights (LoRA converted to an F16 adapter, merged into the F16 base with `llama-export-lora`, quantized with `llama-quantize`) or an independently pinned base plus runtime adapter. It is loaded by the one loader as `Runtime().load(BUNDLE)` or `ss.load(BUNDLE)` (weights, bundled task, frozen policies and realized execution, relocatable and without the trainer or torch) or model-only as `ss.judge(BUNDLE)`.
- Its identity is the sha256 of the merged file (or of the base and adapter files) and the template, pinned by locks like any model.
- **Provenance:** the fitted lock's `receipt.json` records every observation that adapted, selected or calibrated it, and `ss.judge(BUNDLE)` retains that verified record on the model. The lock marks those examples "trained the judge", so `ss test` never counts them as fresh, and warns if any fall in the check half. A plain `Model(path)` has unknown provenance and supports no independent-evidence interval; filenames prove nothing.

---

## 7. CLI

```bash
ss [--judge SRC | --precise] [--device auto|cpu|gpu] [--threads N] [--json] [--verbose] COMMAND

ss check "The customer threatens legal action" < ticket.txt      # exit 0 true · 1 false · 2 uncertain/needs input · 3 failed
ss check --unstated "There is a visible leak." < report.txt
ss pick  "Which address?" --options addresses.txt [--focus mentions] < invoice.txt
ss rate  "How urgent?" --levels "can wait,today,immediately" < ticket.txt
ss ask   questions.yaml [--each] < in.jsonl
ss run   support_route [--each] < messages.jsonl
ss --verbose run support_route [--raw] < message.json        # the decision with its explanation; an explain field with --json
ss correct [--fact F --value V | --rule R] --actor human|agent [--model M --session S] [--author ID] · label --actor … · specialize · lock [--dry-run] · test · diff · export · doctor [--quality] · bench
ss compare --baseline LOCK --candidate LOCK --examples LABELS --evidence MANIFEST --margin M --to DIR [--claim C --benefit B --required-precision P --tolerate K=N --forbidden ADDRESS=KEY]
ss promote POINTER LOCK --reason R [--decision DIR/decision.json]
ss rollback POINTER --reason R --expected ENTRY_ID [--to ENTRY_ID]
```

**Output and exit codes:**
- **Two outputs:**
  - **On a terminal:** one short line per result (`billing  (rule billing)`, `needs input: payment.status`), with `--verbose` giving `explain()`.
  - **Piped or `--json`:** one JSON object per line from `as_dict()`, with stable fields. `--verbose` adds scores, cost and partial results, and for `run` an `explain` field (`--raw` shows the raw scores in it).
- **Exit codes:**
  - 0 ok / none;
  - 1 false (`check` only);
  - 2 uncertain or needs input;
  - 3 failed;
  - `--each` exits with the worst status seen, 4 if any record failed;
  - 5 configuration error (bad YAML, unknown option, a model that can't load). It is never confused with a valid FALSE.
- **Errors vs results:** configuration problems are `error: …` on stderr with exit 5; scoring problems are results with `status="failed"`.

**Behaviour:**
- **`ss compare`** scores each frozen lock over the check-role inputs of the evidence manifest in one bounded scoring operation per arm, journaled per input and resumable, reports the paired difference with a decision and publishes `decision.json`, the promotion decision, beside `comparison.json`. `--claim improvement|non_inferiority`, `--benefit latency|cost`, `--required-precision`, `--tolerate failed|unresolved|forbidden=N` and `--forbidden ADDRESS=KEY` complete the predeclared claim. The report is an adoption claim only on fresh independent evidence with disjoint lineage and a typed per-target reference assessment that places every scored target inside the declared `--reference` scope, and otherwise descriptive with its reason. `--robustness MANIFEST` adds a descriptive flip/contrast/probability diagnostic per arm.
- **`ss promote`** moves a `current` pointer to a lock: a first adoption without `--decision`, otherwise only to the candidate of a supported `decision.json` and only while the pointer still serves the incumbent that decision compared against; a stale incumbent changes nothing. **`ss rollback`** points it back at a previously verified entry from its history (`--to`, default the previous one) only while `--expected` names the current entry.
- **`ss bench`** reports three workloads, each named for what it measures: the same input repeated (cached responsiveness); a new input each time with the system prompt cached (a normal request); and 4 questions on a new input. With a switch it adds per-decision latency and `many()` throughput. A switch with no labelled inputs or YAML tests is a `ConfigError`, not a crash.
- **`--each`** answers each line as it arrives (streaming). **`--each --batch N`** reads N lines, runs them as one finite batch, and emits their results in order. It pays only on judges with batch capacity above 1 (§4.7).
- **`ss label`** parses answers through the fact's own values and re-asks on invalid answers. Its prompts go to stderr.
- **Model sources must name exactly one file:** `repo:file.gguf`, or `repo:Q4_0` when it matches exactly one file.
- **Malformed JSONL** is reported by line.
- **Environment:** `SEMANTIC_SWITCH_JUDGE` works when no flag is given.

---

## 8. Defaults and evidence

| Role | Model | Notes |
|---|---|---|
| default judge | Gemma 4 E2B instruct, GGUF `Q4_0` (ggml-org; Apache-2.0) | 3.8 GB RAM |
| `--precise` | Gemma 4 E4B instruct, official QAT `Q4_0` | most precise; 2.2× slower; 7.5 GB |
| alternative (`--judge`) | Qwen3.5-2B `Q8_0` | best-calibrated small judge (95% precision at 66% coverage on fresh items), 2.8 GB, slower |

**There is no fast judge.** Qwen3-0.6B committed 8 wrong procedures on the equipment set and settles 13% of fresh items at 90% precision; a sub-1B judge isn't offered.

**Judge quality** (16-core laptop CPU). "Fresh" is a suite written after the default was chosen and never used for selection:

| Judge | Accuracy (selection suite → fresh) | Settled at 0.90, precision (fresh) | ms per check (20 / 261 tokens) |
|---|---|---|---|
| Gemma 4 E4B QAT | 0.975 → 0.93 | 95% at 94.7% | 572 / — |
| **Gemma 4 E2B Q4_0** | 0.96 → **0.86** | 90% at **88.9%** | 230–290 / 1,180 |
| Qwen3.5-2B Q8_0 | 0.90 → 0.84 | 66% at 95.3% | 540 / — |
| Qwen3-0.6B Q4_K_M (not offered) | 0.74 → 0.66 | 13% at 90% | 100 / 420 |

**Readings:**
- **None of the small judges we tested reached 95% untuned precision at high coverage on unseen domains.** Stricter thresholds don't fix it (the errors are confident). `ss lock` measures what a judge achieves on your labels and abstains where it falls short; a stronger judge (`--precise`) is the measured way to raise precision.
- **Measured contributions:**
  - three-way checks turn confident false answers on silence into `needs_input` (9/10 correct);
  - focus halves 20-option picks;
  - the lexical head cut judge calls by about a quarter with no new errors at 30 labels per topic;
  - **structure:** on the equipment set, the default judge got 9/34 given the whole catalogue in one question, 26/34 with shared facts plus exact rules, and 28/34 with catalogue cards whose claims carry no titles. Exact rules took impossible procedures off the table; with them, no wrong procedure was committed;
  - the embedding head never helped (removed in v0.19);
  - **prompt examples were removed after E5:** on the fresh suite with the default judge, retrieved precedents moved accuracy 0.856 → 0.869 and precision at 0.9 88.9% → 89.9%, random ones 0.887 / 91.5%, both under the 5-point bar and at 2.6× latency (334 → 870 ms);
  - two-stage scoring with the input cache: 4× faster than eager when the first rule decides, within 0–8% when a later rule decides (E4);
  - the input cache removes the input re-read on repeated rounds, with byte-identical scores;
  - **option position** (`bench/positions.py`, correct option at slots 1–20 of 20): no late-position failure (24/24 at slots 5–20). Order bias appears only between two plausible options: "can't log in to see my invoices" → billing 0.95 in one order, account 0.85 in the other; clear items don't move;
  - **averaging two option orders was measured and rejected** (removed `permute` script; results in `bench/results/permute.json`): no wrong answer removed on the selection, fresh or 20-option suites, 3 fewer settled at 20 options, +37–66% latency. Ambiguity is handled by labels, tuning and abstention, not by permutation;
  - **comparative choice** (§1): the default anchors on the situation when the goal contradicts it (3 of 23 settled were wrong); `--precise` was 24/24;
  - **readout adaptation (probe, calibration, cutpoint) with about 16 labels per judgment does not beat the plain judge on new expression families** (§5.4). An earlier +7 points for a hidden-state probe came from translated twins in training, and was retracted;
  - **answer mass is a format check, not a correctness signal:** the default judge's 23 fresh-suite errors have mean mass 0.9986, and its correct answers 0.9987;
  - **falling back from E2B to E4B on uncertain answers doesn't raise precision:** it escalates 9.4% of items and ends at 89.9%, unchanged, because E2B's 16 wrong answers were confident and never escalate (E4B would fix 9). With a GPU, use `--precise` outright (94.7%);
  - **lock tuning must be validated on held-out data** (§5.3): tuned on one half of the fresh suite's families, θ held 83% on the other, not the 95% target;
  - **batching inputs per decode** pays on the GPU only (2.5–3.9× throughput for short inputs, §4.7);
  - **a question-first prompt layout** (cache the question, decode only each new input) was measured and rejected: 110 vs 104 ms per input on the GPU, 325 vs 289 ms on the CPU, with the same accuracy (0.863). It settled fewer answers at higher precision, a calibration shift rather than a better judge.

**Against the OpenAI Decisions API** (released 2026-10-06, beta, `gpt-6-luna`):
- **What OpenAI claims:** "10× faster than the Responses API", and nothing against Jev or local engines. Its absolute latency and accuracy on our suites are **unmeasured**.
- **Cost:** $0.10 per million input tokens, so about $50 per million decisions at 500 tokens. A local GPU's marginal cost is energy, around $0.25 per million decisions at about 60 W and 10 decisions per second (an estimate), before hardware and upkeep.
- **Where each wins:** local wins on marginal cost at volume, privacy and offline use; hosted wins without hardware, at low volume, or where its quality is needed.
- **The comparison that matters:** the Decisions API **inside** our pipeline against the Decisions API choosing directly, which isolates what structure contributes whatever the judge. It needs an OpenAI key.

**Against hosted Jev** (warm, network included):
- **Speed:** Jev is flat at 270–292 ms. The default judge on the GPU is faster at every measured shape: 104–224 ms for 1–7 questions on 20–261-token inputs, with the system prompt cached. On the CPU it wins only a short single question (164 ms) and loses on long inputs.
- **Semantic quality, where both were measured:** Jev leads. Selection suite: Jev 235/240 (97.9%, no wrong answer at 0.9), Gemma E2B 231 (96.3%), E4B 0.975. Jev hasn't been run on the fresh suite.
- **Equipment set** (shared facts + exact rules): Jev 30/34, E2B 26/34, both with no wrong procedure. All 8 E2B misses are **one judgment (`leak`) on two report texts**, each repeated over 2 machines × 2 languages:
  - a near miss: "the other unit leaked yesterday; this unit has no leak" scores 0.86–0.89 for "states false";
  - a real misread: a hypothetical about another unit, read as "not stated".

  The set can't separate designs; the equipment benchmark replaces it (§9.1). The fair comparison is correct decisions at equal error rate: abstaining on everything also commits no wrong procedure.
- **Jev's agreement §2.3(b)** restricts using its outputs to train or develop competing products. Further runs wait for a reading of it or consent, and Jev outputs never enter training data.

**Beating Jev: decisions, not questions.** A 2B model on a CPU won't out-answer a hosted model question by question. We win decision by decision:
1. **Decisions that need no model:** exact rules, corrections and lexical heads settle in microseconds; our measured hosted medians were 270–292 ms, network included.
2. **Fewer wrong actions from structure:** exact eligibility gave zero *ineligible* selections in every design we ran (zero ineligible is not zero wrong: a wrong eligible procedure is still possible). Given the whole catalogue, Jev picked ineligible procedures twice and our judge 18 times. The structure is the advantage, not the judge: Jev could be wrapped in the same exact rules, and the comparison that matters is structure against no structure.
3. **Fewer, smaller questions:** shared claims, situation-only questions, cached input.
4. **A precision target you can measure:** labels → per-judgment policy tuned to a target → abstention where it isn't met, with `ss test` / `ss diff` and pinned locks. A target is not a guarantee; `ss test` reports what was achieved on unseen inputs. Plus local, private, offline, and no rate limit.

**The scoreboard,** run on the equipment benchmark (§9.1), per decision:
- correct;
- **wrong action**;
- correct `needs_input`;
- abstained;
- **settled with no model call**;
- p50 / p95 latency, reported twice: over **all decisions** under the stated workload, and over **the decisions that needed the judge** (with their input lengths and question counts), so cheap exact matches can't hide a slow language path. Cold start (process to first answer) is reported separately from warm latency.

Run it for our pipeline (default and `--precise`), and for Jev only with TypeSafe's consent. **Where we still lose:** raw per-question quality on unseen phrasing and long single questions on CPU. The levers are `--precise` (on a GPU, about Jev's latency), LoRA specialists (E8), and structure: exact rules, shared facts, abstention.

**Evaluation protocol:**
- Changing the default judge, prompt rules or quantisation must hold on **both** the fresh suite and the selection suite.
- Changing procedure selection must hold on the equipment benchmark (§9.1). The current set has 34 cases but only 10 distinct report texts and 2 facts, so differences of a few cases between designs or judges are noise.
- `ss test` on your own labelled data is the only measure for a switch.
- Detailed tables and every benchmark are in `bench/README.md`.

---

## 9. Status, work, limitations

### 9.1 Work

v0.17–v0.26 are implemented (see `CHANGELOG.md`). **v0.24–v0.26, as built:**
1. **Kernel correctness** (§4.5):
   - flatten nested `and` / `or` before grouping same-fact operands;
   - checks settle their answer first, then atoms are evaluated at it;
   - picks and rates use one symmetric θ for set events.

   These fix two reproduced bugs: parentheses changing a decision, and `not` bypassing a disabled side.
2. **One labelled-data collector** (§5.3):
   - merge per fact meaning and input projection;
   - join groups with union-find;
   - assign splits to joined groups, so check-half corrections never train;
   - exclude taught inputs from validation;
   - reject conflicting labels.
3. **Lock evidence** (§5.3):
   - acceptance through `settle` (answer mass, silence, pending options);
   - the judge validated on the residue a head leaves;
   - states named "observed target met / missed / insufficient evidence", with intervals and coverage before and after;
   - condition-level and decision-level reports on the check half;
   - membership recorded, including validation and judge training.
4. **Honest diagnostics:**
   - judge-only AUC, with wording that doesn't overclaim;
   - both-correct pair rate (optional `pair` field), and group invariance over same-label groups only;
   - `specialize` warm, alternating and total-time measurement;
   - a public way for tools to read a fact's settled value (§5.3, §5.4).
5. **Execution surfaces** (§4.7, §4.8, §6.4):
   - streaming `sw(x)` / `sw.many()` / `--each`, and finite batches `sw.run(list)` / `--each --batch N`;
   - windows sized by `Judge.batch_inputs` (GPU llama 8, CPU 1);
   - `elapsed_ms` and `amortized_ms`.
6. **Fitted locks** loaded like ordinary locks (`ss.load(BUNDLE)`, model-only `ss.judge(BUNDLE)`), with training provenance from the fitted lock (§6.6).
7. **Hosted judges** behind the `Judge` contract (§6.2):
   - OpenRouter chat log probabilities; System One for verified structured models (Jev with consent);
   - probabilities mapped, `mass=None`, refusals as errors, `Scores.confidence` as a diagnostic;
   - protocol, requested provider and returned model and provider recorded; locking requires a pinned provider;
   - never training data, by library policy.

   Then the matched comparison of the Decisions API inside the pipeline vs choosing directly, when a key is provided (§8).
8. **Catalogue authoring:** `{fact: …}` exclusions must be three-way checks, with the text-vs-world authoring rule (§4.4).
9. **Export:** readable context; a default tune-split export with a manifest (§5.7).
10. **Cost and immutability:** lexical features built once per input projection per call; catalogue payloads deep-frozen once at load.
11. **Housekeeping:**
    - the package version follows the implemented spec version;
    - `ss doctor` / `ss bench` say "requested device";
    - `.gitignore` adds `*.export.jsonl`.
12. **The equipment benchmark** (formerly E6) as a permanent gate for procedure selection:
    - **at least 40 distinct reports** in English and Portuguese, over at least 4 facts and both catalogue shapes;
    - **the trap types:** near-identical machines, firmware and pressure boundaries, operator vs sensor contradictions, other units, past and resolved faults, hypotheticals, missing observations, and no applicable procedure;
    - **it reports** correct decisions at equal error rate, wrong actions, correct `needs_input`, decisions with no model call, and latency;
    - **it runs** on E2B and E4B, and on Jev only with TypeSafe's consent.
13. **Remove concluded experiment code** from `bench/`: the probe, order-averaging, staged/eager, prompt-example, model-comparison, prompt-framing, learning-curve, topical and GPU scripts. Their raw JSON results stay in `bench/results/` as data for reproducibility, and their summaries in `bench/README.md` and `CHANGELOG.md`. `bench/` keeps only benchmarks that gate decisions.

**The one open experiment:** E8, a frozen base model plus LoRA specialists on question tokens (`EXPERIMENTS.md`). It is the intended path for specialisation and is decided by its own rule: adopted or rejected, not left open.

**How experiments are run:**
- An experiment is a question with a setup and a decision rule, decided in one campaign.
- Its code lives outside `semantic_switch/` and outside the permanent benchmarks.
- Its result is recorded and the verdict applied to the spec; then its code is deleted, while the raw results stay as data.

### 9.2 Invariants (the deterministic suite enforces these and more)

1. User content never becomes chat control tokens.
2. UNKNOWN never silently becomes a value; "not stated", "not examined" and "missing" are never FALSE.
3. A missing field is reported by name, and the model is never asked about it.
4. Cheap or invalid calls never load a model.
5. `FactResult.value` is set only when the fact's own answer settles, and a switch without rules is `ok` only when every output's value settled.
6. Inline and saved judgments answer identically on the same input: same meaning, same question, same acceptance policy. A catalogue and its equivalent hand-written rules and facts answer identically too.
7. Only an exact rule or an examination rules a candidate out. An incomplete search (a catalogue over budget, a shortlisted pick), or any unresolved card, never yields `none`. Precedence reads base applicability, and cycles are rejected. How a claim is written never changes its budget.
8. A correction fixes exactly its input; labels go stale when meaning changes; inputs used by a lock never count as fresh.
9. A lexical head settles only on a policy side whose target was observed met on held-out labels. `ss analyze-head` never recommends a head that adds wrong answers.
10. A lock runs exactly what was tested, on exactly the pinned models.
11. Limits fail one question, never a whole call.
12. Payloads and catalog instructions never reach the model.
13. A condition over one judgment is evaluated on that judgment's distribution, so its acceptance never exceeds what the distribution supports. **Equivalent expressions get equivalent answers:** regrouping with parentheses or writing `not` never changes acceptance, and a side that doesn't settle can't be reached through negation. One function interprets scores everywhere.
14. A malformed judge output fails loudly at the boundary; it never reaches resolution.
15. Batching inputs never changes an answer: windowed and sequential scoring agree within kernel noise (D2's tolerance), and results keep input order. Streaming never waits on later inputs.
16. Every tool sees the same examples: one input is one example per fact, groups are joined rather than chosen, check-half examples never train, conflicting labels fail, and lock tuning and validation accept exactly what the runtime accepts.
17. Outputs of hosted judges never become training data through the library's tools.

### 9.3 Known limitations

- **Untuned precision** is about 89% on unseen domains, and the default judge's errors are confident. Labels and `ss lock` give measured abstention (observations with intervals, not guarantees), not a smarter judge. A stronger judge (`--precise`) is the measured way up (94.7%).
- **Speed:** on CPU, cost grows with input length; long single questions are slower than a hosted GPU service.
- **Memory:** the default judge needs about 4 GB of RAM; Qwen3.5-2B needs about 2.8 GB. There is no supported judge for smaller machines yet.
- **Slots:** 20 per question; more candidates need `shortlist`.
- **Learning:** inline calls don't learn; teaching is a saved-switch feature. A correction fixes a fact for one input, not necessarily the final answer.
- **Evidence:** the synthetic suites are two languages and a few domains; they rank judges and certify nothing.
- **Not a planner, not a fetcher, not a platform.**

### References

- [R1] Sun & Xu, *Type-safe is not error-free*, arXiv:2609.26758 (2026)
- [R5] Kleene, *Introduction to Metamathematics* (1952)
- [R6] FitzGerald et al., *MASSIVE*, 2022 (CC-BY-4.0)
- [R7] TypeSafe AI, Jev documentation https://docs.typesafe.ai/introduction; Master Customer Agreement https://typesafe.ai/legal/mca
- [R8] Jeff, open one-pass decision models: https://github.com/firelex/jeff
- [R15] llama.cpp C API (`llama.h`): https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- [R16] llama.cpp issue #29092, recurrent state across requests (qwen35)
