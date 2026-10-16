from __future__ import annotations

import queue
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from .context import ContextError, canonical, error_status, project, render, truncated
from .heads import features
from .judge import OPERATION, Cancelled, Scored, Scores, checkpoint, distinct, guided, judge_name, scored, validated
from .judgment import REASONS as WHY, accept, outcome, predict, reason_text, settle, unpresent
from .kernel import NEEDS_INPUT, F, T, U, Card, Sem, atoms_of, budgeted, compose, evaluate, exact_state, ranked, relevant, resolve
from .presentation import Presentation, focused, present
from .results import FactResult, Preview, Row, Stage, SwitchResult, Target, Trace, thaw
from .snapshot import Snapshot

SOURCES = ("example", "classifier", "judge")
COUNTED = ("rule", "example", "classifier", "judge")
PRESENTED = 4096


@dataclass(slots=True)
class FactState:
    dist: tuple | None = None
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
    selection: object = None
    presentations: tuple = ()
    binding: str | None = None

    @property
    def pending(self) -> frozenset:
        return frozenset(k for k, e in enumerate(self.eligible or ()) if e == U)


def stage_of(text: str, aliases, shown, memo, focus: bool, done: bool) -> tuple[tuple, bool, str]:
    if focus and not done:
        named = focused(text, aliases, [i for i in shown if i < len(aliases)], memo)
        if named:
            return named, True, "mentions"
    return tuple(shown), False, "fallback" if done else "full"


def absorb(state: FactState, p: Presentation, sc, width: int, locked=()) -> bool:
    state.presentations += (p,)
    if sc.model is not None:
        state.observed_models += (sc.model,)
    state.requests = distinct((*state.requests, sc.request))
    if locked and sc.model not in locked:
        state.dist, state.error = None, "predictor_identity_unverified" if sc.model is None else "predictor_identity_changed"
        return False
    if sc.error:
        state.dist, state.error = None, sc.error
        return False
    dist = unpresent(p.slots, sc, width)
    if dist is None:
        state.focus_done = True
        return True
    state.scored = tuple(sorted(s for s in p.slots if s >= 0))
    state.dist, state.mass, state.error = tuple(map(float, dist)), sc.mass, None
    return False


@dataclass(slots=True, eq=False)
class Call:
    data: object
    snap: Snapshot | None = None
    sources: tuple = SOURCES
    outputs: frozenset | None = None
    extra: frozenset = frozenset()
    values: dict | None = None
    error: ContextError | None = None
    st: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    cuts: dict = field(default_factory=dict)
    keys: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    tokens: int = 0
    rounds: int = 0
    mentions: dict = field(default_factory=dict)
    work: dict = field(default_factory=lambda: {"classifier": 0, "judge": 0})
    stage: int = 0
    asked: set = field(default_factory=set)
    retry: list = field(default_factory=list)
    judged: bool = False

    def advance(self, ex):
        return ex.advance(self)

    def absorb(self, ex, binding, f, p, sc):
        ex.absorb(self, binding, f, p, sc)


@dataclass(slots=True, eq=False)
class Live:
    index: int
    work: object
    started: float
    waiting: int = 0
    done: bool = False
    value: object = None


@dataclass(slots=True, eq=False)
class Ask:
    live: Live
    binding: str
    text: str
    entries: tuple
    scores: list
    left: int
    key: object = None
    tokens: int = 0
    model: str | None = None

    @property
    def questions(self) -> tuple:
        return tuple(p.question for _, p in self.entries)


class Reuse:
    def __init__(self):
        self._done, self.reused, self.scored = {}, 0, 0

    def get(self, key):
        found = self._done.get(key)
        self.reused += found is not None
        return found

    def put(self, key, result):
        owned = tuple(Scores(value.probs if value.probs is None or type(value.probs) is tuple and all(type(p) is float for p in value.probs) else tuple(map(float, value.probs)), value.mass, value.error,
                             value.confidence, value.model, value.request) for value in result.scores)
        self._done[key] = Scored(owned, result.tokens, result.model, result.native_work)
        self.scored += 1
        return self._done[key]

    def clear(self):
        self._done.clear()


class Operation:
    def __init__(self, ex, bound: int, finish, reuse: Reuse | None = None, receipt: bool = True):
        self.ex, self.bound, self.finish, self.reuse = ex, bound, finish, reuse
        self.queue, self.inflight, self.futures, self.returned = queue.SimpleQueue(), 0, set(), {}
        self.pending = {}
        self.receipt = {"inputs": 0, "window": bound, "native_work": {}, "elapsed_ms": None, "amortized_ms": None} if receipt else None
        self.native_work = self.receipt["native_work"] if receipt else {}

    def elapsed(self, live: Live) -> float:
        return round((time.perf_counter() - live.started) * 1000, 2)

    def drive(self, items, start, ordered: bool):
        began, source, exhausted = time.perf_counter(), iter(items), False
        order, ready, live, n = deque(), deque(), 0, 0
        try:
            while True:
                checkpoint()
                while not exhausted and (len(order) if ordered else live) < self.bound:
                    try:
                        data = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    entry = Live(n, start(data), time.perf_counter())
                    n, live = n + 1, live + 1
                    if ordered:
                        order.append(entry)
                    ready.append(entry)
                submissions, finished = {}, []
                while ready:
                    entry = ready.popleft()
                    asks = entry.work.advance(self.ex)
                    if asks:
                        entry.waiting += len(asks)
                        for binding, text, entries in asks:
                            submissions.setdefault(binding, []).append(Ask(entry, binding, text, tuple(entries), [None] * len(entries), len(entries)))
                    else:
                        entry.value, entry.done, live = self.finish(entry), True, live - 1
                        finished.append(entry)
                for binding, asks in submissions.items():
                    self.submit(binding, asks, ready)
                if ordered:
                    while order and order[0].done:
                        entry = order.popleft()
                        yield entry.index, entry.value
                else:
                    for entry in finished:
                        yield entry.index, entry.value
                if exhausted and not live and not (ordered and order):
                    break
                if ready or not exhausted and (len(order) if ordered else live) < self.bound:
                    continue
                if not self.inflight:
                    raise RuntimeError("execution stalled without pending judge work")
                handle = OPERATION.get()
                release = handle.register(lambda: self.queue.put(None)) if handle is not None else None
                try:
                    item = self.queue.get()
                finally:
                    if release is not None:
                        release()
                self.apply(item, ready)
                while True:
                    try:
                        item = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    self.apply(item, ready)
            for name, models in self.returned.items():
                if len(models) > 1:
                    warnings.warn(f"hosted operation returned mixed model versions: {sorted(models)} from {name}", stacklevel=3)
        finally:
            self.pending.clear()
            for future in self.futures:
                future.cancel()
            if self.receipt is not None:
                elapsed = (time.perf_counter() - began) * 1000
                self.receipt.update(inputs=n, elapsed_ms=round(elapsed, 2), amortized_ms=round(elapsed / n, 2) if n else None)

    def submit(self, binding: str, asks: list, ready) -> None:
        judge, fresh = self.ex.rt.judge(binding), []
        for ask in asks:
            if self.reuse is not None:
                from .compiler import digest
                ask.key = (id(judge), digest([ask.text, [(question.kind, question.text, question.options, question.excludes,
                    question.guidance.digest, None if question.framing is None else question.framing.as_dict(), question.candidate,
                    None if question.catalog is None else [question.catalog.revision, question.catalog.entries, question.entry])
                    for question in ask.questions]]))
                hit = self.reuse.get(ask.key)
                if hit is not None:
                    ask.key = None
                    self.complete(ask, tuple(range(len(ask.entries))), hit, ready, False)
                    continue
                if ask.key in self.pending:
                    self.pending[ask.key].append(ask)
                    continue
                self.pending[ask.key] = []
            fresh.append(ask)
        if not fresh:
            return
        items = [(ask.text, ask.questions) for ask in fresh]
        start = getattr(judge, "start", None)
        if start is None:
            results = scored(judge, items)
            for name, value in results[0].native_work if results else ():
                self.native_work[name] = self.native_work.get(name, 0) + value
            for ask, result in zip(fresh, results):
                self.complete(ask, tuple(range(len(ask.entries))), result, ready, False)
            return
        guided(judge, items)
        token = (fresh, judge_name(judge))
        future = start(items, lambda done: self.queue.put((token, done)))
        self.inflight += len(fresh)
        self.futures.add(future)
        future.add_done_callback(lambda done: self.queue.put((None, done)))

    def apply(self, item, ready) -> None:
        if item is None:
            checkpoint()
            return
        token, done = item
        if token is None:
            self.futures.discard(done)
            if done.cancelled():
                handle = OPERATION.get()
                if handle is not None and handle.cancelled:
                    raise Cancelled("operation cancelled")
                raise RuntimeError("judge is closed")
            if done.exception() is not None:
                raise done.exception()
            return
        fresh, name = token
        ask = fresh[done.owner]
        self.returned.setdefault(name, set()).update(score.model for score in done.scored.scores if score.model is not None)
        questions = [ask.entries[k][1].question for k in done.index]
        self.complete(ask, done.index, validated(name, done.owner, questions, done.scored), ready, True)

    def complete(self, ask: Ask, index, result, ready, hosted: bool) -> None:
        live = ask.live
        work = live.work
        work.tokens += result.tokens
        for k, sc in zip(index, result.scores):
            key, p = ask.entries[k]
            ask.scores[k] = sc
            work.absorb(self.ex, ask.binding, key, p, sc)
        ask.left -= len(index)
        ask.tokens += result.tokens
        ask.model = ask.model or result.model
        if ask.left:
            return
        self.inflight -= hosted
        if ask.key is not None:
            saved = self.reuse.put(ask.key, Scored(tuple(ask.scores), ask.tokens, ask.model))
            for dependent in self.pending.pop(ask.key, ()):
                dependent.key = None
                self.reuse.reused += 1
                self.complete(dependent, tuple(range(len(dependent.entries))), saved, ready, False)
        live.waiting -= 1
        if not live.waiting:
            ready.append(live)


class Executor:
    def __init__(self, runtime):
        self.rt = runtime
        self._drift_lock = threading.Lock()
        self._drift_warned = set()
        self._presented = {}

    def window(self, bindings) -> int:
        sizes = [self.rt.batch_inputs(name) for name in sorted(bindings)]
        return 2 * max(sizes) if sizes else 1

    def bindings(self, snap: Snapshot, sources: tuple, facts=None) -> set:
        if "judge" not in sources:
            return set()
        plan = snap.plan
        chosen = range(len(plan.facts)) if facts is None else \
            {x for f in facts for x in ((f,) + tuple(k for card in plan.facts[f].cards for k in card.claims))}
        return {plan.binding_of(f) for f in chosen if not plan.facts[f].cards}

    def operation(self, bound: int, finish, reuse: Reuse | None = None, receipt: bool = True) -> Operation:
        return Operation(self, bound, finish, reuse, receipt)

    def text(self, snap: Snapshot, call: Call, reads: tuple) -> str:
        if reads not in call.texts:
            call.texts[reads] = render(snap.plan.fields, call.values, reads)
        return call.texts[reads]

    def _cut(self, snap: Snapshot, call: Call, reads: tuple) -> tuple:
        if reads not in call.cuts:
            call.cuts[reads] = truncated(snap.plan.fields, call.values, reads)
        return call.cuts[reads]

    def _key(self, call: Call, reads: tuple) -> str:
        if reads not in call.keys:
            call.keys[reads] = canonical(call.values, reads)
        return call.keys[reads]

    def start(self, snap: Snapshot, data, sources: tuple = SOURCES, outputs: frozenset | None = None, extra: frozenset = frozenset()) -> Call:
        plan = snap.plan
        call = Call(data, snap, sources, outputs, extra)
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

    def advance(self, call: Call) -> list:
        if call.error is not None:
            return []
        snap = call.snap
        if call.retry:
            facts, call.retry = call.retry, []
            return self._asks(snap, call, facts)
        if call.judged:
            call.judged = False
            self._derive(snap, call)
        stages = (True, False) if snap.plan.first and snap.plan.rules else (False,)
        while call.stage < len(stages):
            first_only = stages[call.stage]
            want = [f for f in self._wanted(snap, call, first_only, call.outputs, call.extra) if f not in call.asked]
            if first_only or not want:
                call.stage += 1
            if not want:
                continue
            call.asked.update(want)
            call.rounds += 1
            judge = self._prepare(snap, call, want, call.sources, call.outputs, call.extra)
            if judge:
                call.judged = True
                return self._asks(snap, call, judge)
        return []

    def _asks(self, snap: Snapshot, call: Call, facts: list[int]) -> list:
        plan, groups = snap.plan, {}
        for f in facts:
            groups.setdefault((plan.binding_of(f), plan.facts[f].reads), []).append(f)
        return [(binding, self.text(snap, call, reads), tuple((f, self.presentation(snap, call, f)) for f in gfs))
                for (binding, reads), gfs in groups.items()]

    def absorb(self, call: Call, binding: str, f: int, p: Presentation, sc) -> None:
        snap = call.snap
        fact, state = snap.plan.facts[f], call.facts[f]
        call.work["judge"] += 1
        state.binding = binding
        locked = snap.returned_models.get(binding, ()) if snap.locked else ()
        if locked and sc.model is not None and sc.model not in locked:
            with self._drift_lock:
                if (locked, sc.model) not in self._drift_warned:
                    self._drift_warned.add((locked, sc.model))
                    warnings.warn(f"hosted returned-model drift: lock {locked}, response {sc.model!r}", stacklevel=2)
        if absorb(state, p, sc, len(fact.keys), locked):
            call.retry.append(f)
        elif state.error is not None:
            self._set(snap, call, f, None, "judge", error=state.error)
        else:
            self._set(snap, call, f, state.dist, "judge", state.mass)

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

        if outputs is not None or not plan.rules:
            return targeted(outputs if outputs is not None else {f for f, fact in enumerate(plan.facts) if not fact.hidden})
        direct, via = targeted(extra) if extra and not first_only else (set(), set())
        for rule in plan.rules:
            v = evaluate(rule.expr, call.st)
            if v == F:
                continue
            if v == T and plan.first:
                break
            d, w = split(relevant(rule.expr, call.st))
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
                self._set(snap, call, f, (0.0,) * (len(elig) - fact.allow_none) + (1.0,) * fact.allow_none, "rule")
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
            hit = table.get(self._key(call, fact.context_reads or fact.reads)) if table else None
            idx = None if hit is None else fact.index(hit)
            elig = call.facts[f].eligible
            if idx is None or (elig is not None and elig[idx] == F):
                rest.append(f)
                continue
            self._set(snap, call, f, tuple(1.0 if k == idx else 0.0 for k in range(len(fact.keys))), "example")
        return rest

    @staticmethod
    def eligible_dist(state: FactState, dist) -> tuple:
        elig = state.eligible
        if elig is None:
            return tuple(map(float, dist))
        d = np.array([p if e != F else 0.0 for p, e in zip(dist, elig)], dtype=np.float64)
        if d.sum() <= 0:
            d = np.array([1.0 if e != F else 0.0 for e in elig])
        return tuple(map(float, d / d.sum()))

    def _heads(self, snap: Snapshot, call: Call, facts: list[int]):
        plan = snap.plan
        for f in facts:
            fact = plan.facts[f]
            if fact.reads not in call.features:
                call.features[fact.reads] = features(self.text(snap, call, fact.reads))
            dist = snap.heads[fact.name].probs_features(call.features[fact.reads])
            call.work["classifier"] += 1
            self._set(snap, call, f, self.eligible_dist(call.facts[f], dist), "classifier")

    def presentation(self, snap: Snapshot, call: Call, f: int) -> Presentation:
        plan = snap.plan
        fact, state, binding = plan.facts[f], call.facts[f], plan.binding_of(f)
        shown = tuple(i for i in range(len(fact.keys)) if state.eligible is None or state.eligible[i] != F)
        slots, escape, stage = stage_of(self.text(snap, call, fact.reads) if fact.focus and not state.focus_done else "",
                                        fact.aliases, shown, call.mentions, bool(fact.focus), state.focus_done)
        key = (id(plan), id(snap.framing), binding, f, slots, escape, stage)
        found = self._presented.get(key)
        if found is None:
            judge = self.rt.judge(binding)
            if len(self._presented) >= PRESENTED:
                self._presented.clear()
            found = self._presented[key] = (present(judge, fact.judgment, slots, escape=escape, stage=stage,
                                                    framing=snap.framing if hasattr(judge, "adapter") else None,
                                                    catalog=plan.catalog, entry=plan.catalog and plan.catalog_ids.get(f)), plan, snap.framing)
        return found[0]

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
        from .results import Cost
        cost = Cost({"rules": len(plan.exact_atoms), **call.work, "settled_by": {s: sources.count(s) for s in COUNTED},
                "bindings": {f.shown: s.binding for f, s in zip(plan.facts, call.facts) if s.binding is not None},
                "rounds": call.rounds, "logical_tokens": call.tokens},
                    distinct(r for s in call.facts for r in s.requests))
        errors = {f.shown: s.error for f, s in zip(plan.facts, call.facts) if s.error}
        trace = Trace(plan, tuple(call.st), MappingProxyType(dict(call.reasons)))
        read = {r for f, s in zip(plan.facts, call.facts) if s.source not in (None, "rule", "example") for r in f.reads}
        cut = self._cut(snap, call, tuple(n for n in plan.fields if n in read))
        common = dict(errors=errors, cost=cost, truncated=cut, _trace=trace)
        if not plan.rules:
            facts = self.public_facts(snap, call, {f for f, x in enumerate(plan.facts) if not x.hidden})
            open_ = {fact.name: self._fact_reason(snap, call, f) for f, fact in enumerate(plan.facts)
                     if not fact.hidden and not facts[fact.name].settled}
            if not open_:
                selects = [facts[f.name] for f in plan.visible if f.cards]
                empty = bool(selects) and all(x.value == () and x.complete for x in selects)
                return SwitchResult("none" if empty else "ok", facts=facts, **common)
            return _blocked(open_, None, (), facts, common)
        res = resolve(plan.rules, plan.otherwise, call.st, plan.first)
        facts = self.public_facts(snap, call, {plan.atom_fact[i] for i in res.blocking} | requested)
        if res.status == "accepted":
            answer, rule = res.answer, res.rule
            if not plan.first and rule == "otherwise":
                answer, rule = (answer,), ("otherwise",)
            return SwitchResult("ok", tuple(thaw(value) for value in answer) if not plan.first and isinstance(answer, tuple) else thaw(answer), rule, facts=facts, **common)
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

    def public_fact(self, snap: Snapshot, call: Call, f: int) -> FactResult:
        if call.error:
            return FactResult(None, None, False, None, str(call.error), status=error_status(call.error))
        fact, state = snap.plan.facts[f], call.facts[f]
        if state.missing:
            out = FactResult(None, None, False, None, f"missing: {', '.join(state.missing)}")
        elif state.error:
            out = FactResult(None, None, False, state.source, f"error: {state.error}", observed_models=state.observed_models, requests=state.requests)
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
            scores = MappingProxyType({fact.keys[k]: state.dist[k] for k in shown})
            value = None if own is None or own == none else fact.value(own)
            out = FactResult(value, state.dist[top], own is not None or state.source == "rule", state.source,
                             why, scores, mass=state.mass, observed_models=state.observed_models, requests=state.requests,
                             prediction=predict(fact.judgment, state.dist, none))
        else:
            out = FactResult(None, None, False, None, "uncertain")
        return replace(out, guidance=fact.judgment.guidance.references, status=self.fact_status(snap, call, f),
                       _shown=state.presentations)

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

    def _select_fact(self, snap: Snapshot, call: Call, f: int) -> FactResult:
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
        return FactResult(value, None, sel.settled, call.facts[f].source, why, MappingProxyType({}),
                          MappingProxyType(unresolved), MappingProxyType(superseded),
                          () if fact.labels else tuple(MappingProxyType({"id": c.id, "source": c.raw.get("source", {}),
                                                                         "payload": c.raw.get("payload")})
                                                       for c in (fact.cards[k] for k in sel.selected)),
                          sel.complete, MappingProxyType(dict(zip(ids, sel.states))), members=MappingProxyType(members))

    def target(self, snap: Snapshot, call: Call, f: int) -> Target:
        plan = snap.plan
        fact, result = plan.facts[f], self.public_fact(snap, call, f)
        if call.error is not None:
            return Target(fact.path, fact.meaning_rev, plan.binding_of(f), None, None, None, None, frozenset(), (), str(call.error), result.status,
                          None, False, None, None, (), (), result)
        state = call.facts[f]
        selection = state.selection if fact.cards else None
        return Target(fact.path, fact.meaning_rev, plan.binding_of(f), state.source, state.dist, state.mass,
                      None if state.eligible is None else tuple(state.eligible), state.pending, state.missing, state.error, result.status,
                      None if fact.cards else self.own(snap, call, f)[0], not self._unsettled(snap, call, f),
                      None if selection is None else tuple(fact.cards[k].id for k in selection.selected),
                      None if selection is None else selection.complete, state.observed_models, state.requests, result)

    @staticmethod
    def _scope(plan, call: Call) -> set:
        if call.outputs is not None or not plan.rules:
            roots = call.outputs if call.outputs is not None else {f for f, fact in enumerate(plan.facts) if not fact.hidden}
        else:
            roots = {plan.atom_fact[i] for rule in plan.rules for i in atoms_of(rule.expr) if plan.atom_fact[i] >= 0} | set(call.extra)
        return {x for f in roots for x in ((f,) + tuple(k for card in plan.facts[f].cards for k in card.claims))}

    def row(self, call: Call, requested: list[int], decide: bool) -> Row:
        snap = call.snap
        plan = snap.plan
        conditions = {} if call.error is not None else \
            {plan.atom_keys[i]: call.st[i] for f in requested for i, sem in plan.fact_atoms[f] if isinstance(sem, Sem)}
        return Row(None if call.error is None else str(call.error), MappingProxyType({plan.facts[f].name: self.target(snap, call, f) for f in requested}),
                   MappingProxyType(conditions), self.result(snap, call, frozenset(requested)) if decide else None)

    def preview(self, call: Call, requests: bool = True) -> Preview:
        if call.error is not None:
            return Preview(str(call.error), (), ())
        snap = call.snap
        plan = snap.plan
        asks = self.advance(call)
        initial = {f for _, _, entries in asks for f, _ in entries}
        stages = []
        for binding, _, entries in asks:
            for f, p in entries:
                state = call.facts[f]
                stages.append(Stage(plan.facts[f].path, binding, p.stage, p, None if state.eligible is None else tuple(state.eligible), False))
                if p.stage == "mentions":
                    state.focus_done = True
                    later = self.presentation(snap, call, f)
                    state.focus_done = False
                    stages.append(Stage(plan.facts[f].path, binding, later.stage, later, None if state.eligible is None else tuple(state.eligible), True))
        if "judge" in call.sources:
            for f in sorted(self._scope(plan, call) - initial):
                fact, state = plan.facts[f], call.facts[f]
                if fact.cards or state.source is not None or state.missing:
                    continue
                p = self.presentation(snap, call, f)
                stages.append(Stage(fact.path, plan.binding_of(f), p.stage, p, None if state.eligible is None else tuple(state.eligible), True))
        prepared_requests = []
        for binding, text, entries in asks if requests else ():
            judge = self.rt.judge(binding)
            prepared = getattr(judge, "preview", None)
            for request in prepared([(text, tuple(p.question for _, p in entries))]) if prepared else ():
                prepared_requests.append(MappingProxyType({**request, "binding": binding, "rendered_input": text,
                                                           "targets": [list(plan.facts[entries[k][0]].path) for k in request["questions"]]}))
        return Preview(None, tuple(stages), tuple(prepared_requests))


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
