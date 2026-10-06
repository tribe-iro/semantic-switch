from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .context import ContextError, canonical, project, render, truncated
from .judge import scored
from .judgment import accept, escaped, settle
from .kernel import F, T, U, Card, Sem, atoms_of, evaluate, exact_state, mentioned, relevant, resolve
from .protocols import Question
from .results import Fact, SwitchResult, Trace
from .snapshot import Snapshot

SOURCES = ("example", "classifier", "judge")
COUNTED = ("rule", "example", "classifier", "judge")
ESCAPE = "another one, not listed above"
WHY = {"low_answer_mass": "low answer mass", "not_stated": "not stated in the input", "uncertain": "uncertain",
       "not_examined": "not examined (over max_claims)", "superseded": "may be superseded"}


@dataclass(slots=True)
class FactState:
    dist: list | None = None
    source: str | None = None
    mass: float | None = None
    error: str | None = None
    missing: tuple = ()
    eligible: list | None = None
    unsure_fields: tuple = ()
    focus_done: bool = False
    examined: frozenset | None = None

    @property
    def pending(self) -> frozenset:
        return frozenset(k for k, e in enumerate(self.eligible or ()) if e == U)


@dataclass(slots=True)
class Call:
    data: object
    values: dict | None = None
    error: ContextError | None = None
    st: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    tokens: int = 0
    rounds: int = 0


def reason_text(reason: tuple) -> str:
    if reason[0] == "missing_input":
        return f"missing: {', '.join(reason[1])}"
    if reason[0] == "model_error":
        return f"error: {reason[1]}"
    if reason[0] == "not_stated" and len(reason) > 1 and reason[1]:
        return f"not stated: {'; '.join(reason[1])}"
    return WHY.get(reason[0], reason[0])


class Executor:
    def __init__(self, runtime):
        self.rt = runtime

    def text(self, snap: Snapshot, call: Call, reads: tuple) -> str:
        if reads not in call.texts:
            call.texts[reads] = render(snap.plan.fields, call.values, reads)
        return call.texts[reads]

    def start(self, snap: Snapshot, data) -> Call:
        plan = snap.plan
        call = Call(data)
        try:
            call.values = project(plan.fields, data)
        except ContextError as e:
            call.error = e
            return call
        call.st = [U] * len(plan.atoms)
        for i in plan.exact_atoms:
            a = plan.atoms[i]
            call.st[i] = exact_state(a, call.values)
            if call.st[i] == U:
                call.reasons[i] = ("missing_input", (a.field,))
        call.facts = [FactState() for _ in plan.facts]
        for f, fact in enumerate(plan.facts):
            fs = call.facts[f]
            if fact.kind == "pick":
                fs.eligible = [T if r is None else evaluate(r, call.st) for r in fact.requires]
                fs.unsure_fields = tuple(sorted({plan.atoms[i].field for r, e in zip(fact.requires, fs.eligible)
                                                 if e == U for i in atoms_of(r) if call.st[i] == U}))
            fs.missing = tuple(r for r in fact.reads if r not in call.values)
            for i, _ in plan.fact_atoms[f] if fs.missing else ():
                call.reasons[i] = ("missing_input", fs.missing)
        self._derive(snap, call)
        return call

    def decide(self, snap: Snapshot, calls: list[Call]):
        stages = (True, False) if snap.plan.first and snap.plan.cases else (False,)
        for c in calls:
            if c.error is not None:
                continue
            asked: set = set()
            for first_only in stages:
                while (want := [f for f in self._wanted(snap, c, first_only) if f not in asked]):
                    asked |= set(want)
                    c.rounds += 1
                    self._round(snap, c, want)
                    if first_only:
                        break

    def score(self, snap: Snapshot, calls: list[Call], facts: list[int], sources: tuple = SOURCES):
        for c in calls:
            if c.error is None:
                self._round(snap, c, [f for f in facts if not c.facts[f].missing], sources, recheck=False)

    def _claims(self, snap: Snapshot, call: Call, i: int) -> set[int]:
        plan = snap.plan
        sem = plan.atoms[i]
        fact = plan.facts[sem.fact]
        cards = [sem.card] if sem.card is not None and not fact.exclusive else range(len(fact.cards))
        return {plan.atom_fact[j] for k in cards for j in relevant(fact.cards[k].has, call.st)
                if plan.atom_fact[j] >= 0}

    def _needs(self, snap: Snapshot, call: Call, first_only: bool, fresh: bool) -> tuple[set, set]:
        plan = snap.plan
        keep = lambda f: not call.facts[f].missing and (not fresh or call.facts[f].source is None)

        def split(atoms) -> tuple[set, set]:
            direct, via = set(), set()
            for i in atoms:
                if plan.atom_fact[i] < 0:
                    continue
                if isinstance(plan.atoms[i], Card):
                    via.update(self._claims(snap, call, i))
                else:
                    direct.add(plan.atom_fact[i])
            return {f for f in direct if keep(f)}, {f for f in via if keep(f)}

        if not plan.cases:
            direct, via = split(i for i, s in enumerate(call.st) if s == U and isinstance(plan.atoms[i], Card))
            direct |= {f for f, fact in enumerate(plan.facts) if not fact.hidden and not fact.cards and keep(f)}
            return direct, via - direct
        direct, via = set(), set()
        for case in plan.cases:
            v = evaluate(case.expr, call.st)
            if v == F:
                continue
            if v == T and plan.first:
                break
            d, w = split(relevant(case.expr, call.st))
            direct |= d
            via |= w
            if first_only or (plan.first and not d | w):
                break
        return direct, via - direct

    def _wanted(self, snap: Snapshot, call: Call, first_only: bool) -> list[int]:
        direct, via = self._needs(snap, call, first_only, fresh=True)
        return sorted(direct | via)

    def _examined(self, snap: Snapshot, call: Call, f: int) -> frozenset:
        state = call.facts[f]
        if state.examined is None:
            plan = snap.plan
            fact = plan.facts[f]
            open_ = [k for k, c in enumerate(fact.cards) if evaluate(c.base, call.st) == U]
            text = self.text(snap, call, fact.reads)
            named = [k for k in open_ if any(mentioned(text, a) for a in fact.cards[k].aliases)]
            rest = [k for k in open_ if k not in named]
            ranked = named + fact.ranking.rank(f"{fact.text} {text}", rest)
            spent: set = set()
            examined = set(range(len(fact.cards))) - set(open_)
            for k in ranked:
                new = {x for x in fact.cards[k].claims if call.facts[x].source is None} - spent
                if len(spent) + len(new) <= fact.max_claims:
                    spent |= new
                    examined.add(k)
            state.examined = frozenset(examined)
        return state.examined

    def _allowed(self, snap: Snapshot, call: Call, f: int, direct: set) -> bool:
        plan = snap.plan
        if f in direct:
            return True
        return any(f in plan.facts[s].cards[k].claims for s in plan.claim_of.get(f, ())
                   for k in self._examined(snap, call, s))

    def _unsettled(self, snap: Snapshot, call: Call, f: int) -> bool:
        atoms = snap.plan.fact_atoms[f]
        if atoms:
            return any(call.st[i] == U for i, _ in atoms)
        return self._own(snap, call, f)[0] is None

    def _round(self, snap: Snapshot, call: Call, facts: list[int], sources: tuple = SOURCES, recheck: bool = True):
        plan = snap.plan
        facts = self._rules(snap, call, facts)
        if "example" in sources:
            facts = self._overrides(snap, call, facts)
        if "classifier" in sources:
            self._heads(snap, call, [f for f in facts if plan.facts[f].name in snap.heads])
        self._derive(snap, call)
        if "judge" not in sources:
            return
        direct, via = self._needs(snap, call, False, fresh=False) if recheck else (set(facts), set())
        judge = []
        for f in facts:
            if f not in direct | via or not self._allowed(snap, call, f, direct):
                continue
            if call.facts[f].source == "classifier" and self._unsettled(snap, call, f):
                self._clear(snap, call, f)
            if call.facts[f].source is None:
                judge.append(f)
        self._judge(snap, call, judge)
        self._derive(snap, call)

    def _set(self, snap: Snapshot, call: Call, f: int, dist, source: str, mass=None, error=None):
        plan = snap.plan
        fact = plan.facts[f]
        state = call.facts[f]
        state.source, state.mass, state.error = source, mass, error
        items = [(i, sem) for i, sem in plan.fact_atoms[f] if isinstance(sem, Sem)]
        if error is not None:
            state.dist = None
            for i, _ in items:
                call.st[i] = U
                call.reasons[i] = ("model_error", error)
            return
        state.dist = dist
        policy = snap.policy(source, fact)
        label = fact.text if fact.hidden else f"fact:{fact.name}"
        for i, sem in items:
            call.st[i], why = settle(dist, sem.truth, policy, mass, fact.judgment.silence, state.pending)
            if why is None:
                call.reasons.pop(i, None)
            else:
                call.reasons[i] = ("missing_input", state.unsure_fields) if why == "missing_input" else \
                    ("not_stated", (label,)) if why == "not_stated" else (why,)

    def _card_reason(self, snap: Snapshot, call: Call, f: int, k: int) -> tuple:
        card = snap.plan.facts[f].cards[k]
        if evaluate(card.base, call.st) == T:
            return ("superseded",)
        reasons = [call.reasons.get(i, ("uncertain",)) for i in relevant(card.base, call.st)]
        missing = tuple(sorted({x for r in reasons if r[0] == "missing_input" for x in r[1]}))
        if missing:
            return ("missing_input", missing)
        if self._unjudged(call, card):
            return ("not_examined",)
        stated = tuple(f"{card.id}: {x}" for r in reasons if r[0] == "not_stated" for x in r[1])
        if stated:
            return ("not_stated", stated)
        return next((r for r in reasons if r[0] in ("model_error", "low_answer_mass")), ("uncertain",))

    def _derive(self, snap: Snapshot, call: Call):
        plan = snap.plan
        for f, fact in enumerate(plan.facts):
            if not fact.cards:
                continue
            has = [evaluate(c.has, call.st) for c in fact.cards]
            why = {k: self._card_reason(snap, call, f, k) for k, s in enumerate(has) if s == U}
            if fact.exclusive and T in has and (has.count(T) > 1 or U in has):
                for k, s in enumerate(has):
                    if s == T:
                        has[k], why[k] = U, ("uncertain",)
            empty = F if T in has else U if U in has else T
            for i, sem in plan.fact_atoms[f]:
                call.st[i] = has[sem.card] if sem.card is not None else empty
                reason = why.get(sem.card) if sem.card is not None else next(iter(why.values()), None)
                if call.st[i] == U:
                    call.reasons[i] = reason or ("uncertain",)
                else:
                    call.reasons.pop(i, None)
            sources = {call.facts[x].source for c in fact.cards for x in c.claims}
            call.facts[f].source = next((s for s in ("judge", "classifier", "example", "rule")
                                         if s in sources), None if U in has else "rule")

    def _clear(self, snap: Snapshot, call: Call, f: int):
        for i, _ in snap.plan.fact_atoms[f]:
            call.st[i] = U
            call.reasons.pop(i, None)
        old = call.facts[f]
        call.facts[f] = FactState(missing=old.missing, eligible=old.eligible, unsure_fields=old.unsure_fields,
                                  focus_done=old.focus_done)

    def _rules(self, snap: Snapshot, call: Call, facts: list[int]) -> list[int]:
        rest = []
        for f in facts:
            elig = call.facts[f].eligible
            if elig is not None and all(e == F for e in elig):
                self._set(snap, call, f, [0.0] * len(elig), "rule")
                for i, sem in snap.plan.fact_atoms[f]:
                    call.st[i] = sem.truth[-1]
                    call.reasons.pop(i, None)
            else:
                rest.append(f)
        return rest

    def _overrides(self, snap: Snapshot, call: Call, facts: list[int]) -> list[int]:
        plan = snap.plan
        rest = []
        for f in facts:
            fact = plan.facts[f]
            table = snap.overrides.get(fact.name) if plan.exact_override else None
            hit = table.get(canonical(call.values, fact.reads)) if table else None
            idx = None if hit is None else fact.index(hit)
            elig = call.facts[f].eligible
            if idx is None or (elig is not None and elig[idx] == F):
                rest.append(f)
                continue
            dist = [0.0] * len(fact.keys)
            dist[idx] = 1.0
            self._set(snap, call, f, dist, "example")
        return rest

    def _eligible_dist(self, call: Call, f: int, dist) -> list:
        elig = call.facts[f].eligible
        if elig is None:
            return list(map(float, dist))
        d = np.array([p if e != F else 0.0 for p, e in zip(dist, elig)], dtype=np.float64)
        if d.sum() <= 0:
            d = np.array([1.0 if e != F else 0.0 for e in elig])
        return list(d / d.sum())

    def _heads(self, snap: Snapshot, call: Call, facts: list[int]):
        plan = snap.plan
        for f in facts:
            fact = plan.facts[f]
            dist = snap.heads[fact.name].probs(self.text(snap, call, fact.reads))
            self._set(snap, call, f, self._eligible_dist(call, f, dist), "classifier")

    def _questions(self, snap: Snapshot, call: Call, f: int) -> tuple[Question, list]:
        fact = snap.plan.facts[f]
        state = call.facts[f]
        if fact.kind == "check":
            return fact.judgment.question(), list(range(len(fact.keys)))
        elig = state.eligible
        slot_map = [i for i in range(len(fact.keys)) if elig is None or elig[i] != F]
        if fact.focus and not state.focus_done:
            text = self.text(snap, call, fact.reads)
            named = [i for i in slot_map if i < len(fact.aliases) and any(mentioned(text, a) for a in fact.aliases[i])]
            if named and len(named) < len([i for i in slot_map if i < len(fact.aliases)]):
                focused = named + [i for i in slot_map if i >= len(fact.aliases)]
                options = tuple(fact.descriptions[i] for i in focused) + (ESCAPE,)
                return fact.judgment.question(options), focused + [-1]
        return fact.judgment.question(tuple(fact.descriptions[i] for i in slot_map)), slot_map

    def _apply(self, snap: Snapshot, call: Call, f: int, slot_map: list, sc) -> bool:
        fact = snap.plan.facts[f]
        if sc.error:
            self._set(snap, call, f, None, "judge", error=sc.error)
            return False
        probs = sc.probs
        if slot_map and slot_map[-1] == -1:
            if escaped(probs, snap.policy("judge", fact), sc.mass):
                call.facts[f].focus_done = True
                return True
            slot_map, probs = slot_map[:-1], probs[:-1] / probs[:-1].sum()
        dist = [0.0] * len(fact.keys)
        for s, i in enumerate(slot_map):
            dist[i] = float(probs[s])
        self._set(snap, call, f, dist, "judge", sc.mass)
        return False

    def _judge(self, snap: Snapshot, call: Call, pending: list[int]):
        plan = snap.plan
        groups: dict = {}
        for f in pending:
            groups.setdefault(plan.facts[f].reads, []).append(f)
        if not groups:
            return
        asks = [(reads, [(f, *self._questions(snap, call, f)) for f in fs]) for reads, fs in groups.items()]
        results = scored(self.rt.judge(), [(self.text(snap, call, reads), [q for _, q, _ in qs]) for reads, qs in asks])
        retry = []
        for (_, qs), result in zip(asks, results):
            call.tokens += result.tokens
            for (f, _, slot_map), sc in zip(qs, result.scores):
                if self._apply(snap, call, f, slot_map, sc):
                    retry.append(f)
        self._judge(snap, call, retry)

    def _fact_reason(self, snap: Snapshot, call: Call, f: int) -> tuple:
        plan = snap.plan
        fact = plan.facts[f]
        state = call.facts[f]
        if state.missing:
            return ("missing_input", state.missing)
        if state.error:
            return ("model_error", state.error)
        if fact.cards:
            reasons = [call.reasons.get(i, ("uncertain",)) for i, _ in plan.fact_atoms[f] if call.st[i] == U]
            return _merge(reasons)
        if state.dist is None:
            return ("uncertain",)
        why = self._own(snap, call, f)[1] or "uncertain"
        return ("missing_input", state.unsure_fields) if why == "missing_input" else \
            ("not_stated", (f"fact:{fact.name}",)) if why == "not_stated" else (why,)

    def result(self, snap: Snapshot, call: Call, ms: float) -> SwitchResult:
        plan = snap.plan
        if call.error is not None:
            key = "invalid" if call.error.kind == "invalid_context" else "missing"
            return SwitchResult("failed", **{key: (call.error.field,)}, errors={call.error.field: str(call.error)})
        sources = [s.source for f, s in zip(plan.facts, call.facts) if not f.cards]
        cost = {"rules": len(plan.exact_atoms), **{s: sources.count(s) for s in COUNTED},
                "rounds": call.rounds, "tokens": call.tokens, "ms": round(ms, 2)}
        errors = {f.name if not f.hidden else f.text: s.error for f, s in zip(plan.facts, call.facts) if s.error}
        trace = Trace(plan, tuple(call.st), MappingProxyType(dict(call.reasons)))
        read = {r for f, s in zip(plan.facts, call.facts) if s.source not in (None, "rule", "example") for r in f.reads}
        cut = truncated(plan.fields, call.values, [n for n in plan.fields if n in read])
        common = dict(errors=errors, cost=cost, truncated=cut, _trace=trace)
        if not plan.cases:
            facts = self.public_facts(snap, call, {f for f, x in enumerate(plan.facts) if not x.hidden})
            open_ = {fact.name: self._fact_reason(snap, call, f) for f, fact in enumerate(plan.facts)
                     if not fact.hidden and not facts[fact.name].settled}
            if not open_:
                selects = [facts[f.name] for f in plan.visible if f.cards]
                empty = bool(selects) and all(x.value == () and x.complete for x in selects)
                return SwitchResult("none" if empty else "ok", facts=facts, **common)
            return _blocked(open_, None, (), facts, common)
        res = resolve(plan.cases, plan.otherwise, call.st, plan.first)
        facts = self.public_facts(snap, call, {plan.atom_fact[i] for i in res.blocking})
        if res.status == "accepted":
            answer, case = res.answer, res.case
            if not plan.first and case == "otherwise":
                answer, case = (answer,), ("otherwise",)
            return SwitchResult("ok", answer, case, facts=facts, **common)
        open_ = {}
        for i in res.blocking:
            f = plan.atom_fact[i]
            name = plan.atoms[i].field if f < 0 else plan.facts[f].name
            reason = call.reasons.get(i, ("uncertain",))
            open_[name] = _merge([open_[name], reason]) if name in open_ else reason
        return _blocked(open_, res.blocked_by, res.partial, facts, common)

    def _own(self, snap: Snapshot, call: Call, f: int) -> tuple[int | None, str | None]:
        fact = snap.plan.facts[f]
        state = call.facts[f]
        if state.dist is None:
            return None, None
        return accept(fact.judgment, state.dist, snap.policy(state.source, fact), state.mass, state.pending)

    def public_facts(self, snap: Snapshot, call: Call, blocking: set = frozenset()) -> dict:
        plan = snap.plan
        out = {}
        for f, state in enumerate(call.facts):
            fact = plan.facts[f]
            if fact.hidden:
                continue
            if state.missing and f in blocking:
                out[fact.name] = Fact(None, None, False, None, f"missing: {', '.join(state.missing)}")
            elif state.error:
                out[fact.name] = Fact(None, None, False, state.source, f"error: {state.error}")
            elif fact.cards:
                if state.source is not None or f in blocking:
                    out[fact.name] = self._select_fact(snap, call, f)
            elif state.dist is not None:
                top = int(np.argmax(state.dist))
                own, why = self._own(snap, call, f)
                if state.source == "rule":
                    own, why = None, "no eligible option"
                elif why == "missing_input":
                    why = f"missing: {', '.join(state.unsure_fields)}"
                elif why is not None:
                    why = WHY.get(why, why)
                scores = MappingProxyType({k: float(p) for k, p in zip(fact.keys, state.dist)})
                value = None if own is None else fact.value(own)
                out[fact.name] = Fact(value, float(state.dist[top]), own is not None, state.source, why, scores)
            elif f in blocking:
                out[fact.name] = Fact(None, None, False, None, "uncertain")
        return out

    @staticmethod
    def _unjudged(call: Call, card) -> bool:
        return any(call.facts[x].source is None and not call.facts[x].missing for x in card.claims)

    def card_states(self, snap: Snapshot, call: Call, f: int) -> dict:
        fact = snap.plan.facts[f]
        out = {}
        for c in fact.cards:
            base, has = evaluate(c.base, call.st), evaluate(c.has, call.st)
            req = T if c.requires is None else evaluate(c.requires, call.st)
            if req == F:
                out[c.id] = "ineligible"
            elif base == F:
                out[c.id] = "rejected"
            elif base == T:
                out[c.id] = "applicable" if has == T else "superseded" if has == F else "may_be_superseded"
            else:
                out[c.id] = "unexamined" if req == T and self._unjudged(call, c) else "unresolved"
        return out

    def _select_fact(self, snap: Snapshot, call: Call, f: int) -> Fact:
        plan = snap.plan
        fact = plan.facts[f]
        states = self.card_states(snap, call, f)
        has = {plan.atoms[i].card: call.st[i] for i, sem in plan.fact_atoms[f] if sem.card is not None}
        has = [has.get(k, evaluate(c.has, call.st)) for k, c in enumerate(fact.cards)]
        final = tuple(c.id for c, s in zip(fact.cards, has) if s == T)
        unresolved = {c.id: reason_text(self._card_reason(snap, call, f, k))
                      for k, (c, s) in enumerate(zip(fact.cards, has)) if s == U}
        settled = not unresolved
        superseded = {c.id: next(fact.cards[j].id for j, o in enumerate(fact.cards)
                                 if k in o.supersedes and evaluate(o.base, call.st) == T)
                      for k, c in enumerate(fact.cards) if states[c.id] == "superseded"}
        value = (final[0] if final else None) if fact.exclusive else final
        if not settled and fact.exclusive:
            value = None
        skipped = [k for k, v in unresolved.items() if v == WHY["not_examined"]]
        rest = {k: v for k, v in unresolved.items() if k not in skipped}
        why = "; ".join(([f"incomplete search: {', '.join(skipped)} not examined"] if skipped else []) +
                        (["unresolved: " + ", ".join(f"{k} ({v})" for k, v in rest.items())] if rest else [])) or None
        complete = all(s != "unexamined" for s in states.values())
        return Fact(value, None, settled, call.facts[f].source, why, MappingProxyType({}),
                    MappingProxyType(unresolved), MappingProxyType(superseded),
                    tuple({"id": c.id, "source": c.raw.get("source", {}), "payload": c.raw.get("payload")}
                          for c, s in zip(fact.cards, has) if s == T),
                    complete, MappingProxyType(states))


def _merge(reasons: list[tuple]) -> tuple:
    named = tuple(sorted({x for r in reasons if r[0] in ("missing_input", "not_stated") for x in r[1]}))
    if any(r[0] == "missing_input" for r in reasons):
        return ("missing_input", named)
    if any(r[0] == "not_stated" for r in reasons):
        return ("not_stated", named)
    return next((r for r in reasons if r[0] == "model_error"), None) or \
        next((r for r in reasons if r[0] != "uncertain"), ("uncertain",))


def _blocked(open_: dict, blocked_by, partial, facts: dict, common: dict) -> SwitchResult:
    missing = tuple(sorted({x for r in open_.values() if r[0] in ("missing_input", "not_stated") for x in r[1]}))
    unsure = tuple(sorted(n for n, r in open_.items() if r[0] not in ("missing_input", "not_stated")))
    kw = dict(uncertain=unsure, blocked_by=blocked_by, partial=partial, facts=facts, **common)
    if missing:
        return SwitchResult("needs_input", missing=missing, **kw)
    if any(r[0] == "model_error" for r in open_.values()):
        return SwitchResult("failed", **kw)
    return SwitchResult("uncertain", **kw)
