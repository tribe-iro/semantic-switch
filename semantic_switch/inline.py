from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as np

from . import files
from .guidance import normalize
from .compiler import FieldSpec, compile_spec, digest, document, membership_cards, memberships
from .context import Bm25, ContextError, absent, error_status, project, render
from .errors import ConfigError
from .executor import FactState, absorb, stage_of
from .judge import distinct
from .judgment import MAX_OPTIONS, WHY, Judgment, Policy, authored_bool, authored_budget, authored_levels, authored_strings, authored_text, complete, \
    definitions, outcome, predict, reason_text
from .presentation import Presentation, definition, present
from .kernel import F, T, U, atom, budgeted, compose, matched_mentions, ranked
from .results import CheckResult, PickResult, Preview, RateResult, SelectResult, Stage, freeze

MAX_CANDIDATES = 1000


def rendered(state, fields=None) -> str:
    if isinstance(state, str):
        if absent(state):
            raise ContextError("missing_context", "input")
        return state
    if not isinstance(state, Mapping):
        raise ContextError("invalid_context", "input", "expected text or mapping")
    if fields is not None and (isinstance(fields, (str, bytes)) or not isinstance(fields, (list, tuple)) or
                               not all(isinstance(name, str) for name in fields)):
        raise ConfigError("fields must be a sequence of field names")
    if any(not isinstance(key, str) for key in state):
        raise ContextError("invalid_context", "input", "field names must be strings")
    specs = {k: FieldSpec(type="json", required=True) for k in (tuple(state) if fields is None else fields)}
    if not specs:
        raise ContextError("missing_context", "input")
    return render(specs, project(specs, state), tuple(specs))


class Saveable:
    def record(self) -> dict:
        raise NotImplementedError

    def fact(self) -> dict:
        record = self.record()
        dynamic = [key for key in ("callbacks", "shortlist", "candidates") if record.pop(key, None)]
        if dynamic:
            raise ConfigError(f"a {type(self).__name__.lower()} with {', '.join(dynamic)} does not save; "
                              "a saved switch uses named definitions and requires")
        return record

    def describe(self) -> Mapping:
        definition, guidance = self.record(), normalize(self.guidance)
        return freeze({"address": None, "kind": type(self).__name__.lower(), "definition": definition, "reads": None,
                       "guidance": guidance.references, "revision": digest([definition, guidance.digest])[:8]})

    def __call__(self, input, rt=None, fields=None, *, guidance=None):
        from .api import default

        return (rt or default()).ask(input, {"_": self}, fields=fields, guidance=guidance)["_"]

    def run(self, inputs, rt=None, fields=None, *, guidance=None):
        from .api import default
        return [row["_"] for row in (rt or default()).ask_many(inputs, {"_": self}, fields=fields, guidance=guidance)]

    def many(self, inputs, rt=None, fields=None, *, guidance=None, window=1):
        from .api import default
        for row in (rt or default())._asked_stream(inputs, {'_': self}, fields, guidance, 'judge', window):
            yield row['_']

    def save(self, path, field: str = "message", rt=None, *, guidance=None):
        from .api import default

        path = Path(path)
        name = path.name.removesuffix(".yaml")
        spec = {"name": name, "fields": {field: {"type": "text", "required": True}}, "reads": [field],
                "facts": {name: {k: v for k, v in self.fact().items() if v not in (None, [], {}, False)}}}
        binding = normalize(self.guidance if self.guidance is not None else guidance)
        if binding:
            spec["guidance"] = binding.as_dict()
        compile_spec(spec, path.name, path.parent)
        if path.exists():
            if files.read_yaml(path) != spec:
                raise ConfigError(f"{path} exists with a different definition")
        else:
            files.publish(path, spec)
        return (rt or default()).load(path)


def _common(item):
    object.__setattr__(item, "excludes", authored_strings(item.excludes, "excludes"))
    Policy.of(item.min_score)
    if item.guidance is not None:
        object.__setattr__(item, "guidance", normalize(item.guidance))


@dataclass(frozen=True)
class Check(Saveable):
    statement: str
    excludes: tuple = ()
    min_score: float | None = None
    unstated: bool = False
    guidance: object = None

    def __post_init__(self):
        authored_text(self.statement)
        authored_bool(self.unstated, "unstated")
        _common(self)

    def record(self) -> dict:
        return {"check": self.statement, "excludes": list(self.excludes), "unstated": self.unstated, "min_score": self.min_score}

    def lower(self, guidance) -> Lowered:
        return _Check(self, guidance)


@dataclass(frozen=True)
class Rate(Saveable):
    question: str
    levels: tuple
    min_score: float | None = None
    guidance: object = None
    excludes: tuple = ()

    def __post_init__(self):
        authored_text(self.question)
        object.__setattr__(self, "levels", authored_levels(self.levels))
        _common(self)

    def record(self) -> dict:
        return {"rate": self.question, "levels": list(self.levels), "excludes": list(self.excludes), "min_score": self.min_score}

    def lower(self, guidance) -> Lowered:
        return _Rate(self, guidance)


def identities(objs: list, ids: Callable) -> list[tuple[str, ...]]:
    out = []
    for obj in objs:
        value = ids(obj)
        if isinstance(value, str):
            out.append((value,))
        elif isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str) and \
                isinstance(value[1], (list, tuple)) and all(isinstance(alias, str) for alias in value[1]):
            out.append((value[0], *value[1]))
        else:
            raise ConfigError("ids must return a string or (string, sequence of string aliases)")
    return out


@dataclass(frozen=True)
class Pick(Saveable):
    question: str
    options: object
    definition: Callable | None = None
    eligible: Callable | None = None
    allow_none: bool = False
    shortlist: str | None = None
    min_score: float | None = None
    ids: Callable | None = None
    focus: str | None = None
    guidance: object = None
    excludes: tuple = ()
    compiled: tuple | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        authored_text(self.question)
        authored_bool(self.allow_none, "allow_none")
        named = isinstance(self.options, Mapping)
        if not named and (isinstance(self.options, (str, bytes)) or not isinstance(self.options, Iterable)):
            raise ConfigError("options must be named definitions or a sequence of candidates")
        object.__setattr__(self, "options", definitions(self.options) if named else tuple(self.options))
        _common(self)
        if self.shortlist not in (None, "lexical"):
            raise ConfigError(f"unknown shortlist mode {self.shortlist!r}; use None or 'lexical'")
        if self.focus not in (None, "mentions"):
            raise ConfigError(f"unknown focus mode {self.focus!r}; use None or 'mentions'")
        if self.focus is not None and not named and self.ids is None:
            raise ConfigError("focus needs option identities: dict options or ids=")
        if named and self.allow_none and "none" in self.options:
            raise ConfigError("'none' is reserved for allow_none; rename that option")
        if not named and self.definition is None:
            for option in self.options:
                authored_text(option, "option")
        descs = tuple(definition(label.description, label.excludes) for label in self.options.values()) if named else None
        object.__setattr__(self, "compiled", (tuple(self.options), descs, tuple((key, *label.aliases) for key, label in self.options.items()),
                                              Bm25(list(descs)) if self.shortlist else None) if named else None)

    def record(self) -> dict:
        named = self.compiled is not None
        return {"pick": self.question, "options": {key: label.record() for key, label in self.options.items()} if named else None,
                "candidates": None if named else len(self.options), "excludes": list(self.excludes), "allow_none": self.allow_none,
                "focus": self.focus, "shortlist": self.shortlist, "min_score": self.min_score,
                "callbacks": [name for name in ("definition", "eligible", "ids") if getattr(self, name) is not None]}

    def lower(self, guidance) -> Lowered:
        return _Pick(self, guidance)


@dataclass(frozen=True)
class Select(Saveable):
    question: str
    labels: object
    _: KW_ONLY
    excludes: tuple = ()
    unstated: bool = False
    min_score: float | None = None
    guidance: object = None
    eligible: Callable | None = None
    max_claims: int | None = None
    compiled: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        authored_text(self.question)
        authored_bool(self.unstated, "unstated")
        object.__setattr__(self, "labels", definitions(self.labels))
        _common(self)
        authored_budget(self.max_claims)
        n = len(self.labels)
        cards = membership_cards(self.labels, [atom(k) for k in range(n)], range(n), (None,) * n)
        object.__setattr__(self, "compiled", (memberships(self.question, self.labels, self.excludes, self.unstated), cards,
                                              Bm25([document(card, (self.question,)) for card in cards])))

    def record(self) -> dict:
        return {"select": self.question, "labels": {key: label.record() for key, label in self.labels.items()},
                "excludes": list(self.excludes), "unstated": self.unstated, "min_score": self.min_score, "max_claims": self.max_claims,
                "callbacks": ["eligible"] if self.eligible is not None else []}

    def lower(self, guidance) -> Lowered:
        return _Select(self, guidance)


def check_result(j: Judgment, policy: Policy, state: FactState) -> tuple[CheckResult, tuple | None]:
    models, sent, shown = state.observed_models, state.requests, state.presentations
    if state.error is not None:
        return CheckResult("failed", error=state.error, observed_models=models, requests=sent, _shown=shown), ("model_error", state.error)
    dist = np.asarray(state.dist)
    status, i, why = outcome(j, dist, policy, state.mass)
    return CheckResult(status, None if i is None else i == 0, float(dist[0]), state.mass, WHY.get(why),
                       float(dist[2]) if j.unstated else None, observed_models=models, requests=sent, prediction=predict(j, dist),
                       _shown=shown, _probs=dist), None if status == "ok" else (why or "uncertain",)


class Lowered:
    __slots__ = ("spec", "guidance", "policy", "presented")

    def __init__(self, spec, guidance):
        self.spec, self.guidance, self.policy, self.presented = spec, guidance, Policy.of(spec.min_score), {}

    def shown(self, ask: Asking, key, j: Judgment) -> Presentation:
        found = self.presented.get((ask.binding, key))
        if found is None:
            found = self.presented[(ask.binding, key)] = present(ask.judge(), j, range(len(j.keys)))
        return found


class Single:
    __slots__ = ("low", "state")

    def __init__(self, low):
        self.low, self.state = low, FactState()

    def entries(self, ask: Asking) -> list:
        return [(None, self.low.shown(ask, None, self.low.j))]

    def absorb(self, ask: Asking, key, p: Presentation, sc):
        absorb(self.state, p, sc, len(self.low.j.keys))

    def result(self):
        return self.low.finish(self.state)


class _Check(Lowered):
    __slots__ = ("j",)

    def __init__(self, spec, guidance):
        super().__init__(spec, guidance)
        self.j = Judgment.check(spec.statement, spec.excludes, spec.unstated, guidance)

    def item(self, ask: Asking):
        return Single(self)

    def finish(self, state: FactState) -> CheckResult:
        return check_result(self.j, self.policy, state)[0]

    def failed(self, status: str, error: str) -> CheckResult:
        return CheckResult(status, error=error)


class _Rate(Lowered):
    __slots__ = ("j",)

    def __init__(self, spec, guidance):
        super().__init__(spec, guidance)
        self.j = Judgment("rate", spec.question, spec.levels, spec.levels, spec.excludes, guidance=guidance)

    def item(self, ask: Asking):
        return Single(self)

    def finish(self, state: FactState) -> RateResult:
        levels, common = self.spec.levels, dict(observed_models=state.observed_models, requests=state.requests, _shown=state.presentations)
        if state.error is not None:
            return RateResult("failed", error=state.error, _levels=levels, **common)
        dist = np.asarray(state.dist)
        status, i, why = outcome(self.j, dist, self.policy, state.mass)
        return RateResult(status, None if i is None else levels[i], i, float((dist * np.arange(len(levels))).sum()) if complete(dist) else None,
                          float(dist.max()), state.mass, WHY.get(why), prediction=predict(self.j, dist), _levels=levels, _probs=dist, **common)

    def failed(self, status: str, error: str) -> RateResult:
        return RateResult(status, error=error)


class Choice:
    __slots__ = ("low", "text", "memo", "keys", "objs", "keep", "removed", "examined", "j", "none", "aliases", "state", "immediate")

    def __init__(self, low: _Pick, ask: Asking):
        spec = low.spec
        self.low, self.text, self.memo, self.state, self.immediate = low, ask.text, ask.mentions, FactState(), None
        if spec.compiled is not None:
            keys, given, idents, corpus = spec.compiled
            objs, describe = keys, given.__getitem__
        else:
            objs, corpus = spec.options, None
            keys = range(len(objs))
            describe = (lambda i: authored_text(spec.definition(objs[i]), "description")) if spec.definition else objs.__getitem__
            idents = identities(objs, spec.ids) if spec.ids is not None and (spec.focus or spec.shortlist) else None
        keep = [i for i, o in enumerate(objs) if spec.eligible is None or spec.eligible(o)]
        kept = set(keep)
        removed = {"ineligible": [keys[i] for i in range(len(objs)) if i not in kept], "not_examined": []}
        self.keys, self.objs, self.removed = keys, objs, removed
        if not keep:
            self.immediate = PickResult("none", removed=removed)
            return
        slots = MAX_OPTIONS - int(spec.allow_none)
        if len(keep) > slots and (spec.shortlist != "lexical" or len(keep) > MAX_CANDIDATES):
            limit = MAX_CANDIDATES if spec.shortlist else slots
            self.immediate = PickResult("failed", removed=removed, error=f"too many candidates ({len(keep)} > {limit})")
            return
        descs = {i: describe(i) for i in keep}
        if len(keep) > slots:
            named = [keep[index] for index in matched_mentions(self.text, [idents[i] for i in keep], self.memo)] if idents else []
            chosen = set(named)
            rest = [i for i in keep if i not in chosen]
            ranking = corpus.subset(rest) if spec.compiled is not None else Bm25([descs[i] for i in rest])
            order = named + [rest[k] for k in ranking.rank(f"{spec.question} {self.text}", range(len(rest)), max(0, slots - len(named)))]
            chosen = set(order[:slots])
            removed["not_examined"] = [keys[i] for i in keep if i not in chosen]
            keep = [i for i in keep if i in chosen]
        self.keep, self.examined = keep, not removed["not_examined"]
        self.none = len(keep) if spec.allow_none else None
        self.j = Judgment("pick", spec.question, tuple(keys[i] for i in keep) + (("none",) if spec.allow_none else ()),
                          tuple(descs[i] for i in keep) + (("none of the above",) if spec.allow_none else ()), spec.excludes, guidance=low.guidance)
        self.aliases = tuple(idents[i] for i in keep) if spec.focus == "mentions" else ()

    def present(self, ask: Asking) -> Presentation:
        slots, escape, stage = stage_of(self.text, self.aliases, range(len(self.j.keys)), self.memo, bool(self.aliases), self.state.focus_done)
        return present(ask.judge(), self.j, slots, escape=escape, stage=stage)

    def entries(self, ask: Asking) -> list:
        return [] if self.immediate is not None else [(None, self.present(ask))]

    def absorb(self, ask: Asking, key, p: Presentation, sc):
        if absorb(self.state, p, sc, len(self.j.keys)):
            return self.present(ask)

    def result(self) -> PickResult:
        if self.immediate is not None:
            return self.immediate
        state, j, keys, objs, keep = self.state, self.j, self.keys, self.objs, self.keep
        scored = sorted(s for s in state.presentations[-1].slots if s >= 0)
        common = dict(removed=self.removed, _keys=tuple(j.keys[s] for s in scored), focus=state.presentations[-1].stage, complete=self.examined,
                      _shown=state.presentations, observed_models=state.observed_models, requests=state.requests)
        if state.error is not None:
            return PickResult("failed", error=state.error, **common)
        dist = np.asarray(state.dist)
        guess = predict(j, dist, self.none, lambda s: (keys[keep[s]], objs[keep[s]]))
        status, s, why = outcome(j, dist, self.low.policy, state.mass, none=self.none, complete=self.examined)
        if status == "ok":
            return PickResult("ok", objs[keep[s]], keys[keep[s]], float(dist[s]), state.mass, prediction=guess, _probs=dist[scored], **common)
        return PickResult(status, score=float(dist.max()), mass=state.mass, error=WHY.get(why), prediction=guess, _probs=dist[scored], **common)


class _Pick(Lowered):
    __slots__ = ()

    def item(self, ask: Asking):
        return Choice(self, ask)

    def failed(self, status: str, error: str) -> PickResult:
        return PickResult(status, error=error)


class Members:
    __slots__ = ("low", "allowed", "states")

    def __init__(self, low: _Select, ask: Asking):
        spec = low.spec
        _, cards, ranking = spec.compiled
        self.low = low
        self.allowed = [T if spec.eligible is None or spec.eligible(key) else F for key in low.keys]
        open_ = [k for k, a in enumerate(self.allowed) if a == T]
        order = open_ if spec.max_claims is None else ranked(ask.text, spec.question, open_, [c.aliases for c in cards], ranking, ask.mentions)
        self.states = {k: FactState() for k in sorted(budgeted(order, [c.claims for c in cards], set(), spec.max_claims))}

    def entries(self, ask: Asking) -> list:
        return [(k, self.low.shown(ask, k, self.low.members[k])) for k in self.states]

    def absorb(self, ask: Asking, key, p: Presentation, sc):
        absorb(self.states[key], p, sc, len(self.low.members[key].keys))

    def result(self) -> SelectResult:
        low, n = self.low, len(self.allowed)
        verdicts = {k: check_result(low.members[k], low.policy, state) for k, state in self.states.items()}
        bases = [F if a == F else U if k not in verdicts or verdicts[k][1] is not None else T if verdicts[k][0].value else F
                 for k, a in enumerate(self.allowed)]
        state = compose(self.allowed, bases, ((),) * n, False, [k not in verdicts for k in range(n)],
                        [verdicts[k][1] if k in verdicts else ("not_examined",) for k in range(n)])
        keys = low.keys
        return SelectResult(state.status, tuple(keys[k] for k in state.selected), state.complete, state.settled,
                            MappingProxyType(dict(zip(keys, state.states))),
                            MappingProxyType({keys[k]: reason_text(r) for k, r in enumerate(state.reasons) if r is not None}),
                            MappingProxyType({keys[k]: r for k, (r, _) in verdicts.items()}),
                            observed_models=tuple(m for s in self.states.values() for m in s.observed_models),
                            requests=distinct(r for s in self.states.values() for r in s.requests))


class _Select(Lowered):
    __slots__ = ("keys", "members")

    def __init__(self, spec, guidance):
        super().__init__(spec, guidance)
        self.keys = tuple(spec.labels)
        self.members = tuple(replace(member, guidance=guidance) if guidance else member for member in spec.compiled[0])

    def item(self, ask: Asking):
        return Members(self, ask)

    def failed(self, status: str, error: str) -> SelectResult:
        return SelectResult(status, error=error)


class Asking:
    __slots__ = ("ex", "lowered", "binding", "text", "error", "items", "retry", "first", "tokens", "mentions")

    def __init__(self, ex, lowered: Mapping, binding: str, state, fields=None):
        self.ex, self.lowered, self.binding, self.retry, self.first, self.tokens, self.mentions = ex, lowered, binding, [], True, 0, {}
        try:
            self.text, self.error = rendered(state, fields), None
        except ContextError as error:
            self.text, self.error = "", error
        self.items = {} if self.error is not None else {key: low.item(self) for key, low in lowered.items()}

    def judge(self):
        return self.ex.rt.judge(self.binding)

    def advance(self, ex) -> list:
        if self.first:
            self.first = False
            entries = [((key, sub), p) for key, item in self.items.items() for sub, p in item.entries(self)]
        else:
            entries, self.retry = self.retry, []
        return [(self.binding, self.text, tuple(entries))] if entries else []

    def absorb(self, ex, binding, key, p, sc):
        name, sub = key
        again = self.items[name].absorb(self, sub, p, sc)
        if again is not None:
            self.retry.append((key, again))

    def result(self) -> dict:
        if self.error is not None:
            out = {key: low.failed(error_status(self.error), str(self.error)) for key, low in self.lowered.items()}
        else:
            out = {key: item.result() for key, item in self.items.items()}
        for key, answer in out.items():
            for result in (answer, *getattr(answer, "members", {}).values()):
                result.guidance = self.lowered[key].guidance.references
        return out


def lower(items: Mapping, guidance=None) -> dict:
    if not isinstance(items, Mapping):
        raise ConfigError("ask needs a mapping of names to Check, Pick, Rate or Select")
    default = normalize(guidance)
    out = {}
    for key, spec in items.items():
        if not isinstance(spec, Saveable):
            raise ConfigError(f"{key!r}: expected Check, Pick, Rate or Select")
        out[key] = spec.lower(normalize(spec.guidance) if spec.guidance is not None else default)
    return out


def preview(ask: Asking) -> Preview:
    if ask.error is not None:
        return Preview(str(ask.error), (), ())
    stages, requests = [], []
    for binding, text, entries in ask.advance(ask.ex):
        for (key, sub), p in entries:
            item = ask.items[key]
            target = (key,) if sub is None else (key, ask.lowered[key].keys[sub])
            stages.append(Stage(target, binding, p.stage, p, None, False))
            if p.stage == "mentions":
                item.state.focus_done = True
                later = item.present(ask)
                item.state.focus_done = False
                stages.append(Stage(target, binding, later.stage, later, None, True))
        prepared = getattr(ask.judge(), "preview", None)
        for request in prepared([(text, tuple(p.question for _, p in entries))]) if prepared else ():
            requests.append(MappingProxyType({**request, "binding": binding, "targets": [list(entries[k][0]) for k in request["questions"]]}))
    return Preview(None, tuple(stages), tuple(requests))
