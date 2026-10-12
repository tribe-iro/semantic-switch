from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
import warnings
import threading

from .context import ContextError, canonical, error_status, project, render, truncated
from .heads import features
from .judge import checkpoint, distinct, scored
from .judgment import REASONS as WHY, accept, outcome, predict, reason_text, settle, unpresent
from .kernel import NEEDS_INPUT, F, T, U, Card, Sem, SelectionState, atoms_of, budgeted, compose, evaluate, exact_state, ranked, relevant, resolve
from .presentation import Presentation, focused, present
from .results import thaw, Fact, SwitchResult, Trace
from .snapshot import Snapshot

SOURCES = ("example", "classifier", "judge")
COUNTED = ("rule", "example", "classifier", "judge")


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
    observed_models: tuple = ()
    requests: tuple = ()
    scored: tuple | None = None
    selection: SelectionState | None = None
    presentations: tuple = ()

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
    features: dict = field(default_factory=dict)
    tokens: int = 0
    rounds: int = 0
    mentions: dict = field(default_factory=dict)
    work: dict = field(default_factory=lambda: {"classifier": 0, "judge": 0})
    batch_native_work: dict = field(default_factory=dict)
    batch_inputs: int = 1


class Executor:
    def __init__(self, runtime):
        self.rt = runtime
        self._drift_lock = threading.Lock()
        self._drift_warned = set()

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

    @staticmethod
    def _batch(calls: list[Call]):
        work = {}
        for c in calls:
            c.batch_native_work, c.batch_inputs = work, len(calls)

    def decide(self, snap: Snapshot, calls: list[Call], sources: tuple = SOURCES, outputs: frozenset | None = None, extra: frozenset = frozenset()):
        self._batch(calls)
        stages = (True, False) if snap.plan.first and snap.plan.cases else (False,)
        live = [c for c in calls if c.error is None]
        asked = {id(c): set() for c in live}
        for first_only in stages:
            todo = live
            while todo:
                checkpoint()
                work = []
                for c in todo:
                    want = [f for f in self._wanted(snap, c, first_only, outputs, extra) if f not in asked[id(c)]]
                    if want:
                        asked[id(c)] |= set(want)
                        c.rounds += 1
                        work.append((c, want))
                self._rounds(snap, work, sources, outputs, extra)
                todo = [] if first_only else [c for c, _ in work]

    def _claims(self, snap: Snapshot, call: Call, f: int, cards) -> set[int]:
        plan = snap.plan
        return {plan.atom_fact[j] for k in cards for j in relevant(plan.facts[f].cards[k].has, call.st)
                if plan.atom_fact[j] >= 0}

    def _needs(self, snap: Snapshot, call: Call, first_only: bool, fresh: bool, outputs: frozenset | None, extra: frozenset = frozenset()) -> tuple[set, set]:
        plan = snap.plan
        keep = lambda f: not call.facts[f].missing and (not fresh or call.facts[f].source is None)

        def split(atoms) -> tuple[set, set]:
            direct, via = set(), set()
            for i in atoms:
                if plan.atom_fact[i] < 0:
                    continue
                atom = plan.atoms[i]
                if isinstance(atom, Card):
                    fact = plan.facts[atom.fact]
                    via.update(self._claims(snap, call, atom.fact, [atom.card] if atom.card is not None and not fact.exclusive else range(len(fact.cards))))
                else:
                    direct.add(plan.atom_fact[i])
            return {f for f in direct if keep(f)}, {f for f in via if keep(f)}

        def targeted(targets) -> tuple[set, set]:
            direct = {f for f in targets if not plan.facts[f].cards and keep(f)}
            via = {x for f in targets for x in self._claims(snap, call, f, range(len(plan.facts[f].cards)))}
            return direct, {f for f in via - direct if keep(f)}

        if outputs is not None or not plan.cases:
            return targeted(outputs if outputs is not None else {f for f, fact in enumerate(plan.facts) if not fact.hidden})
        direct, via = targeted(extra) if extra and not first_only else (set(), set())
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

    def _wanted(self, snap: Snapshot, call: Call, first_only: bool, outputs: frozenset | None, extra: frozenset) -> list[int]:
        direct, via = self._needs(snap, call, first_only, True, outputs, extra)
        return sorted(direct | via)

    def _examined(self, snap: Snapshot, call: Call, f: int) -> frozenset:
        state = call.facts[f]
        if state.examined is None:
            plan = snap.plan
            fact = plan.facts[f]
            open_ = [k for k, c in enumerate(fact.cards) if evaluate(c.base, call.st) == U]
            order = open_ if fact.max_claims is None else ranked(self.text(snap, call, fact.reads), fact.text, open_,
                                                                 [c.aliases for c in fact.cards], fact.ranking, call.mentions)
            known = {x for c in fact.cards for x in c.claims if call.facts[x].source is not None}
            state.examined = frozenset(set(range(len(fact.cards))) - set(open_) |
                                       budgeted(order, [c.claims for c in fact.cards], known, fact.max_claims))
        return state.examined

    def _allowed(self, snap: Snapshot, call: Call, f: int, direct: set) -> bool:
        plan = snap.plan
        if f in direct:
            return True
        return any(bool(set(plan.facts[s].claim_cards.get(f, ())) & self._examined(snap, call, s))
                   for s in plan.claim_of.get(f, ()))

    def _unsettled(self, snap: Snapshot, call: Call, f: int) -> bool:
        atoms = snap.plan.fact_atoms[f]
        if atoms:
            return any(call.st[i] == U for i, _ in atoms)
        return self.own(snap, call, f)[0] is None

    def _rounds(self, snap: Snapshot, work: list, sources: tuple, outputs: frozenset | None, extra: frozenset):
        pending = [(c, self._prepare(snap, c, facts, sources, outputs, extra)) for c, facts in work]
        if "judge" in sources:
            self._judge(snap, pending)
            for c, _ in pending:
                self._derive(snap, c)

    def _prepare(self, snap: Snapshot, call: Call, facts: list[int], sources: tuple, outputs: frozenset | None, extra: frozenset) -> list[int]:
        plan = snap.plan
        facts = self._rules(snap, call, facts)
        facts = [f for f in facts if call.facts[f].source is None]
        if "example" in sources:
            facts = self._overrides(snap, call, facts)
        if "classifier" in sources:
            self._heads(snap, call, [f for f in facts if plan.facts[f].name in snap.heads])
        self._derive(snap, call)
        if "judge" not in sources:
            return []
        direct, via = self._needs(snap, call, False, False, outputs, extra)
        judge = []
        for f in facts:
            if f not in direct | via or not self._allowed(snap, call, f, direct):
                continue
            if call.facts[f].source == "classifier" and self._unsettled(snap, call, f):
                self._clear(snap, call, f)
            if call.facts[f].source is None:
                judge.append(f)
        return judge

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
        label = fact.shown if fact.hidden else f"fact:{fact.name}"
        for i, sem in items:
            call.st[i], why = settle(fact.judgment, dist, sem.truth, policy, mass, state.pending)
            if why is None:
                call.reasons.pop(i, None)
            else:
                call.reasons[i] = ("missing_input", state.unsure_fields) if why == "missing_input" else \
                    ("not_stated", (label,)) if why == "not_stated" else (why,)

    def _base_reason(self, call: Call, card) -> tuple:
        reasons = [call.reasons.get(i, ("uncertain",)) for i in relevant(card.base, call.st)]
        missing = tuple(sorted({x for r in reasons if r[0] == "missing_input" for x in r[1]}))
        if missing:
            return ("missing_input", missing)
        if self._unjudged(call, card):
            return ("not_examined",)
        stated = tuple(f"{card.id}: {x}" for r in reasons if r[0] == "not_stated" for x in r[1])
        if stated:
            return ("not_stated", stated)
        return next((r for r in reasons if r[0] in NEEDS_INPUT | {"model_error", "low_answer_mass"}), ("uncertain",))

    def _derive(self, snap: Snapshot, call: Call):
        plan = snap.plan
        for f, fact in enumerate(plan.facts):
            if not fact.cards:
                continue
            bases = [evaluate(card.base, call.st) for card in fact.cards]
            state = compose([T if c.requires is None else evaluate(c.requires, call.st) for c in fact.cards], bases,
                            fact.incoming, fact.exclusive, [b == U and self._unjudged(call, c) for c, b in zip(fact.cards, bases)],
                            [self._base_reason(call, c) if b == U else None for c, b in zip(fact.cards, bases)])
            first = next((r for r in state.reasons if r is not None), None)
            for i, sem in plan.fact_atoms[f]:
                call.st[i] = state.values[sem.card] if sem.card is not None else state.empty
                if call.st[i] == U:
                    call.reasons[i] = (state.reasons[sem.card] if sem.card is not None else first) or ("uncertain",)
                else:
                    call.reasons.pop(i, None)
            sources = {call.facts[x].source for c in fact.cards for x in c.claims}
            call.facts[f].selection = state
            call.facts[f].source = next((s for s in ("judge", "classifier", "example", "rule")
                                         if s in sources), None if not state.settled else "rule")

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
            fact, elig = snap.plan.facts[f], call.facts[f].eligible
            if elig is not None and all(e == F for e in elig[:len(elig) - fact.allow_none]):
                self._set(snap, call, f, [0.0] * (len(elig) - fact.allow_none) + [1.0] * fact.allow_none, "rule")
                for i, sem in () if fact.allow_none else snap.plan.fact_atoms[f]:
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
            hit = table.get(canonical(call.values, fact.context_reads or fact.reads)) if table else None
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
            if fact.reads not in call.features:
                call.features[fact.reads] = features(self.text(snap, call, fact.reads))
            dist = snap.heads[fact.name].probs_features(call.features[fact.reads])
            call.work["classifier"] += 1
            self._set(snap, call, f, self._eligible_dist(call, f, dist), "classifier")

    def presentation(self, snap: Snapshot, call: Call, f: int) -> Presentation:
        fact, state, judge = snap.plan.facts[f], call.facts[f], self.rt.judge()
        shown = [i for i in range(len(fact.keys)) if state.eligible is None or state.eligible[i] != F]
        if fact.focus and not state.focus_done:
            named = focused(self.text(snap, call, fact.reads), fact.aliases,
                            [i for i in shown if i < len(fact.aliases)], call.mentions)
            if named:
                return present(judge, fact.judgment, named, escape=True, stage="mentions", framing=snap.framing)
        return present(judge, fact.judgment, shown, stage="fallback" if state.focus_done else "full",
                       framing=snap.framing)

    def _judge(self, snap: Snapshot, pending: list):
        plan = snap.plan
        asks = []
        for call, fs in pending:
            groups: dict = {}
            for f in fs:
                groups.setdefault(plan.facts[f].reads, []).append(f)
            asks += [(call, reads, [(f, self.presentation(snap, call, f)) for f in gfs]) for reads, gfs in groups.items()]
        if not asks:
            return
        results = scored(self.rt.judge(), [(self.text(snap, call, reads), [p.question for _, p in qs])
                                           for call, reads, qs in asks])
        with self._drift_lock:
            models = sorted({model for result in results for model in
                             (result.model, *(score.model for score in result.scores)) if model is not None})
            for model in models:
                if snap.locked and snap.returned_models and model not in snap.returned_models:
                    pair = (snap.returned_models, model)
                    if pair not in self._drift_warned:
                        warnings.warn(f"hosted returned-model drift: lock {snap.returned_models}, response {model!r}", stacklevel=3)
                        self._drift_warned.add(pair)
        batch = asks[0][0].batch_native_work
        for name, value in results[0].native_work:
            batch[name] = batch.get(name, 0) + value
        retry: dict = {}
        for (call, _, qs), result in zip(asks, results):
            call.tokens += result.tokens
            for (f, p), sc in zip(qs, result.scores):
                fact, state = plan.facts[f], call.facts[f]
                call.work["judge"] += 1
                state.presentations += (p.record(),)
                if sc.model is not None:
                    state.observed_models += (sc.model,)
                state.requests = distinct((*state.requests, sc.request))
                if snap.locked and snap.returned_models and sc.model not in snap.returned_models:
                    self._set(snap, call, f, None, "judge", error="predictor_identity_unverified" if sc.model is None else "predictor_identity_changed")
                    continue
                if sc.error:
                    self._set(snap, call, f, None, "judge", error=sc.error)
                    continue
                dist = unpresent(p.slots, sc, len(fact.keys))
                if dist is None:
                    state.focus_done = True
                    retry.setdefault(id(call), (call, []))[1].append(f)
                else:
                    state.scored = tuple(sorted(s for s in p.slots if s >= 0))
                    self._set(snap, call, f, list(dist), "judge", sc.mass)
        self._judge(snap, list(retry.values()))

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
        why = self.own(snap, call, f)[1] or "uncertain"
        return ("missing_input", state.unsure_fields) if why == "missing_input" else \
            ("not_stated", (f"fact:{fact.name}",)) if why == "not_stated" else (why,)

    def result(self, snap: Snapshot, call: Call, requested: frozenset = frozenset()) -> SwitchResult:
        plan = snap.plan
        if call.error is not None:
            key = "invalid" if call.error.kind == "invalid_context" else "missing"
            return SwitchResult(error_status(call.error), **{key: (call.error.field,)}, errors={call.error.field: str(call.error)},
                                facts={plan.facts[f].name: self.public_fact(snap, call, f) for f in sorted(requested)})
        sources = [s.source for f, s in zip(plan.facts, call.facts) if not f.cards]
        cost = {"rules": len(plan.exact_atoms), **call.work, "settled_by": {s: sources.count(s) for s in COUNTED},
                "rounds": call.rounds, "logical_tokens": call.tokens, "batch_native_work": call.batch_native_work,
                "batch_inputs": call.batch_inputs,
                "requests": [request.as_dict() for request in distinct(r for s in call.facts for r in s.requests)]}
        errors = {f.shown: s.error for f, s in zip(plan.facts, call.facts) if s.error}
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
        facts = self.public_facts(snap, call, {plan.atom_fact[i] for i in res.blocking} | requested)
        if res.status == "accepted":
            answer, case = res.answer, res.case
            if not plan.first and case == "otherwise":
                answer, case = (answer,), ("otherwise",)
            return SwitchResult("ok", tuple(thaw(value) for value in answer) if not plan.first and isinstance(answer, tuple) else thaw(answer), case, facts=facts, **common)
        open_ = {}
        for i in res.blocking:
            f = plan.atom_fact[i]
            name = plan.atoms[i].field if f < 0 else plan.facts[f].name
            reason = call.reasons.get(i, ("uncertain",))
            open_[name] = _merge([open_[name], reason]) if name in open_ else reason
        return _blocked(open_, res.blocked_by, res.partial, facts, common)

    def own(self, snap: Snapshot, call: Call, f: int) -> tuple[int | None, str | None]:
        fact = snap.plan.facts[f]
        state = call.facts[f]
        if state.dist is None:
            return None, None
        return accept(fact.judgment, state.dist, snap.policy(state.source, fact), state.mass, state.pending)

    def public_facts(self, snap: Snapshot, call: Call, blocking: set = frozenset()) -> dict:
        plan = snap.plan
        return {fact.name: self.public_fact(snap, call, f) for f, (fact, state) in enumerate(zip(plan.facts, call.facts))
                if not fact.hidden and (f in blocking or state.error or
                                        (state.source is not None if fact.cards else state.dist is not None))}

    def public_fact(self, snap: Snapshot, call: Call, f: int) -> Fact:
        if call.error:
            return Fact(None, None, False, None, str(call.error), status=error_status(call.error))
        fact, state = snap.plan.facts[f], call.facts[f]
        if state.missing:
            out = Fact(None, None, False, None, f"missing: {', '.join(state.missing)}")
        elif state.error:
            out = Fact(None, None, False, state.source, f"error: {state.error}", observed_models=state.observed_models, requests=state.requests)
        elif fact.cards:
            out = self._select_fact(snap, call, f)
        elif state.dist is not None:
            top = int(np.argmax(state.dist))
            own, why = self.own(snap, call, f)
            if state.source == "rule":
                why = "no eligible option"
            elif why == "missing_input":
                why = f"missing: {', '.join(state.unsure_fields)}"
            elif why is not None:
                why = WHY.get(why, why)
            none = len(fact.keys) - 1 if fact.allow_none else None
            shown = state.scored if state.scored is not None else \
                [k for k in range(len(fact.keys)) if state.eligible is None or state.eligible[k] != F]
            scores = MappingProxyType({fact.keys[k]: float(state.dist[k]) for k in shown})
            value = None if own is None or own == none else fact.value(own)
            out = Fact(value, float(state.dist[top]), own is not None or state.source == "rule", state.source,
                       why, scores, mass=state.mass, observed_models=state.observed_models, requests=state.requests,
                       prediction=predict(fact.judgment, state.dist, none))
        else:
            out = Fact(None, None, False, None, "uncertain")
        return replace(out, guidance=fact.judgment.guidance.references, status=self.fact_status(snap, call, f),
                       presentations=state.presentations)

    def fact_status(self, snap, call, f):
        if call.error:
            return "failed"
        state, fact = call.facts[f], snap.plan.facts[f]
        if state.error:
            return "failed"
        if state.missing:
            return "needs_input"
        if fact.cards:
            return state.selection.status
        if state.source == "rule":
            return "none"
        if state.dist is None:
            return "uncertain"
        return outcome(fact.judgment, state.dist, snap.policy(state.source, fact), state.mass, state.pending,
                       none=len(fact.keys) - 1 if fact.allow_none else None)[0]

    @staticmethod
    def _unjudged(call: Call, card) -> bool:
        return any(call.facts[x].source is None and not call.facts[x].missing for x in card.claims)

    def _select_fact(self, snap: Snapshot, call: Call, f: int) -> Fact:
        fact, sel = snap.plan.facts[f], call.facts[f].selection
        ids = [c.id for c in fact.cards]
        chosen = tuple(ids[k] for k in sel.selected)
        unresolved = {ids[k]: reason_text(r) for k, r in enumerate(sel.reasons) if r is not None}
        superseded = {ids[k]: ids[j] for k, j in enumerate(sel.superseding) if j is not None}
        value = ((chosen[0] if chosen else None) if sel.settled else None) if fact.exclusive else chosen
        skipped = [k for k, v in unresolved.items() if v == WHY["not_examined"]]
        rest = {k: v for k, v in unresolved.items() if k not in skipped}
        why = "; ".join(([f"incomplete search: {', '.join(skipped)} not examined"] if skipped else []) +
                        (["unresolved: " + ", ".join(f"{k} ({v})" for k, v in rest.items())] if rest else [])) or None
        members = {c.id: self.public_fact(snap, call, c.claims[0]) for c in fact.cards
                   if fact.labels and (call.facts[c.claims[0]].source is not None or call.facts[c.claims[0]].error)}
        return Fact(value, None, sel.settled, call.facts[f].source, why, MappingProxyType({}),
                    MappingProxyType(unresolved), MappingProxyType(superseded),
                    () if fact.labels else tuple(MappingProxyType({"id": c.id, "source": c.raw.get("source", {}),
                                                                   "payload": c.raw.get("payload")})
                                                 for c in (fact.cards[k] for k in sel.selected)),
                    sel.complete, MappingProxyType(dict(zip(ids, sel.states))), members=MappingProxyType(members))


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
    if any(r[0] in NEEDS_INPUT for r in open_.values()):
        return SwitchResult("needs_input", missing=missing, **kw)
    if any(r[0] == "model_error" for r in open_.values()):
        return SwitchResult("failed", **kw)
    return SwitchResult("uncertain", **kw)
