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
from .judge import Scores, checkpoint, distinct, scored
from .judgment import MAX_OPTIONS, WHY, Judgment, Policy, authored_bool, authored_budget, authored_levels, authored_strings, authored_text, complete, \
    definitions, outcome, predict, reason_text, unpresent
from .presentation import Presentation, definition, focused, present
from .kernel import F, T, U, atom, budgeted, compose, matched_mentions, ranked
from .protocols import Judge
from .results import CheckResult, PickResult, RateResult, SelectResult, freeze

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


@dataclass(frozen=True)
class Prepared:
    make: Callable[[Judge], tuple[Presentation, ...]] | None
    finish: Callable[[tuple[Presentation, ...], tuple[Scores, ...]], object] | None
    immediate: object = None


def single(make: Callable[[Judge], Presentation], finish: Callable[[Presentation, Scores], object]) -> Prepared:
    return Prepared(lambda judge: (make(judge),), lambda ps, scs: finish(ps[0], scs[0]))


def checked(j: Judgment, policy: Policy, p: Presentation, sc: Scores) -> tuple[CheckResult, tuple | None]:
    models, shown, sent = ((sc.model,) if sc.model else ()), (p.record(),), distinct((sc.request,))
    if sc.error:
        return CheckResult("failed", error=sc.error, observed_models=models, requests=sent, presentations=shown), ("model_error", sc.error)
    dist = unpresent(p.slots, sc, len(j.keys))
    status, i, why = outcome(j, dist, policy, sc.mass)
    return CheckResult(status, None if i is None else i == 0, float(dist[0]), sc.mass, WHY.get(why),
                       float(dist[2]) if j.unstated else None, observed_models=models, requests=sent, prediction=predict(j, dist),
                       presentations=shown, _probs=dist), None if status == "ok" else (why or "uncertain",)


def done(result) -> Prepared:
    return Prepared(None, None, result)


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

    def __call__(self, state, rt=None, fields=None, *, guidance=None):
        from .api import default

        return (rt or default()).ask(state, {"_": self}, fields=fields, guidance=guidance)["_"]

    def run(self, states, rt=None, fields=None, *, guidance=None):
        from .api import default
        return [row["_"] for row in (rt or default()).ask_many(states, {"_": self}, fields=fields, guidance=guidance)]

    def many(self, states, rt=None, fields=None, *, guidance=None):
        for state in states:
            yield self(state, rt, fields, guidance=guidance)

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

    def prepare(self, text: str, guidance) -> Prepared:
        j = Judgment.check(self.statement, self.excludes, self.unstated, guidance)
        policy = Policy.of(self.min_score)
        return single(lambda judge: present(judge, j, range(len(j.keys))), lambda p, sc: checked(j, policy, p, sc)[0])


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

    def prepare(self, text: str, guidance) -> Prepared:
        levels = self.levels
        j = Judgment("rate", self.question, levels, levels, self.excludes, guidance=guidance)
        policy = Policy.of(self.min_score)

        def finish(p: Presentation, sc: Scores) -> RateResult:
            if sc.error:
                return RateResult("failed", error=sc.error, presentations=(p.record(),), _levels=levels)
            dist = unpresent(p.slots, sc, len(levels))
            status, i, why = outcome(j, dist, policy, sc.mass)
            return RateResult(status, None if i is None else levels[i], i,
                              float((dist * np.arange(len(levels))).sum()) if complete(dist) else None,
                              float(dist.max()), sc.mass, WHY.get(why), prediction=predict(j, dist),
                              presentations=(p.record(),), _levels=levels, _probs=dist)

        return single(lambda judge: present(judge, j, range(len(levels))), finish)


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

    def prepare(self, text: str, guidance) -> Prepared:
        policy = Policy.of(self.min_score)
        if self.compiled is not None:
            keys, given, idents, corpus = self.compiled
            objs, describe = keys, given.__getitem__
        else:
            objs, corpus = self.options, None
            keys = range(len(objs))
            describe = (lambda i: authored_text(self.definition(objs[i]), "description")) if self.definition else objs.__getitem__
            idents = identities(objs, self.ids) if self.ids is not None and (self.focus or self.shortlist) else None
        keep = [i for i, o in enumerate(objs) if self.eligible is None or self.eligible(o)]
        kept = set(keep)
        removed = {"ineligible": [keys[i] for i in range(len(objs)) if i not in kept], "not_examined": []}
        if not keep:
            return done(PickResult("none", removed=removed))
        slots = MAX_OPTIONS - int(self.allow_none)
        if len(keep) > slots and (self.shortlist != "lexical" or len(keep) > MAX_CANDIDATES):
            limit = MAX_CANDIDATES if self.shortlist else slots
            return done(PickResult("failed", removed=removed, error=f"too many candidates ({len(keep)} > {limit})"))
        memo: dict = {}
        descs = {i: describe(i) for i in keep}
        if len(keep) > slots:
            named = [keep[index] for index in matched_mentions(text, [idents[i] for i in keep], memo)] if idents else []
            chosen = set(named)
            rest = [i for i in keep if i not in chosen]
            ranking = corpus.subset(rest) if self.compiled is not None else Bm25([descs[i] for i in rest])
            order = named + [rest[k] for k in ranking.rank(f"{self.question} {text}", range(len(rest)), max(0, slots - len(named)))]
            chosen = set(order[:slots])
            removed["not_examined"] = [keys[i] for i in keep if i not in chosen]
            keep = [i for i in keep if i in chosen]
        examined = not removed["not_examined"]
        none = len(keep) if self.allow_none else None
        j = Judgment("pick", self.question, tuple(keys[i] for i in keep) + (("none",) if self.allow_none else ()),
                     tuple(descs[i] for i in keep) + (("none of the above",) if self.allow_none else ()), self.excludes,
                     guidance=guidance)

        def stage(shown, mode: str, before: tuple = ()) -> Prepared:
            def finish(p: Presentation, sc: Scores):
                scored = sorted(s for s in p.slots if s >= 0)
                record = before + (p.record(),)
                common = dict(removed=removed, _keys=tuple(j.keys[s] for s in scored), focus=mode, complete=examined, presentations=record)
                if sc.error:
                    return PickResult("failed", error=sc.error, **common)
                dist = unpresent(p.slots, sc, len(j.keys))
                if dist is None:
                    return stage(range(len(j.keys)), "fallback", record)
                guess = predict(j, dist, none, lambda s: (keys[keep[s]], objs[keep[s]]))
                status, s, why = outcome(j, dist, policy, sc.mass, none=none, complete=examined)
                if status == "ok":
                    return PickResult("ok", objs[keep[s]], keys[keep[s]], float(dist[s]), sc.mass, prediction=guess, _probs=dist[scored], **common)
                return PickResult(status, score=float(dist.max()), mass=sc.mass, error=WHY.get(why), prediction=guess, _probs=dist[scored], **common)

            return single(lambda judge: present(judge, j, shown, escape=mode == "mentions", stage=mode), finish)

        named = focused(text, [idents[i] for i in keep], range(len(keep)), memo) if self.focus == "mentions" else None
        return stage(named, "mentions") if named else stage(range(len(j.keys)), "full")


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

    def prepare(self, text: str, guidance) -> Prepared:
        members, cards, ranking = self.compiled
        n = len(cards)
        policy, keys = Policy.of(self.min_score), tuple(self.labels)
        allowed = [T if self.eligible is None or self.eligible(key) else F for key in keys]
        open_ = [k for k, a in enumerate(allowed) if a == T]
        order = open_ if self.max_claims is None else ranked(text, self.question, open_, [c.aliases for c in cards], ranking)
        examined = sorted(budgeted(order, [c.claims for c in cards], set(), self.max_claims))
        judgments = [replace(members[k], guidance=guidance) if guidance else members[k] for k in examined]

        def finish(ps, scs) -> SelectResult:
            verdicts = dict(zip(examined, (checked(j, policy, p, sc) for j, p, sc in zip(judgments, ps, scs))))
            bases = [F if a == F else U if k not in verdicts or verdicts[k][1] is not None else T if verdicts[k][0].value else F
                     for k, a in enumerate(allowed)]
            state = compose(allowed, bases, ((),) * n, False, [k not in verdicts for k in range(n)],
                            [verdicts[k][1] if k in verdicts else ("not_examined",) for k in range(n)])
            return SelectResult(state.status, tuple(keys[k] for k in state.selected), state.complete, state.settled,
                                MappingProxyType(dict(zip(keys, state.states))),
                                MappingProxyType({keys[k]: reason_text(r) for k, r in enumerate(state.reasons) if r is not None}),
                                MappingProxyType({keys[k]: r for k, (r, _) in verdicts.items()}))

        if not examined:
            return done(finish((), ()))
        return Prepared(lambda judge: tuple(present(judge, j, range(len(j.keys))) for j in judgments), finish)


RESULTS = {Check: CheckResult, Pick: PickResult, Rate: RateResult, Select: SelectResult}


def ask_many(get_judge: Callable[[], Judge], states, items: Mapping, fields=None, guidance=None) -> list[dict]:
    default_guidance = normalize(guidance)
    bound = {key: normalize(spec.guidance) if spec.guidance is not None else default_guidance for key, spec in items.items()}
    texts, pending = [], []
    for state in states:
        try:
            text = rendered(state, fields)
            row = {key: spec.prepare(text, bound[key]) for key, spec in items.items()}
        except ContextError as error:
            text = ""
            row = {key: done(RESULTS[type(spec)](error_status(error), error=str(error))) for key, spec in items.items()}
        texts.append(text)
        pending.append(row)
    outputs = [{} for _ in texts]
    receipts = [{} for _ in texts]
    requests = [{} for _ in texts]
    while any(pending):
        checkpoint()
        live = []
        for n, row in enumerate(pending):
            questions = {}
            for key, prepared in row.items():
                if prepared.make is None:
                    outputs[n][key] = prepared.immediate
                else:
                    questions[key] = prepared
            pending[n] = {}
            if questions:
                live.append((n, questions))
        if not live:
            break
        judge = get_judge()
        presented = [{key: prepared.make(judge) for key, prepared in row.items()} for _, row in live]
        results = scored(judge, [(texts[n], [p.question for ps in shown.values() for p in ps]) for (n, _), shown in zip(live, presented)])
        for (n, row), shown, result in zip(live, presented, results):
            at = 0
            for key, prepared in row.items():
                ps = shown[key]
                scs, at = result.scores[at:at + len(ps)], at + len(ps)
                models = receipts[n].setdefault(key, [])
                models.extend(score.model for score in scs if score.model is not None)
                requests[n][key] = distinct((*requests[n].get(key, ()), *(score.request for score in scs)))
                answer = prepared.finish(ps, scs)
                if isinstance(answer, Prepared):
                    pending[n][key] = answer
                else:
                    answer.observed_models = tuple(models)
                    answer.requests = requests[n][key]
                    outputs[n][key] = answer
    for out in outputs:
        for key, answer in out.items():
            for result in (answer, *getattr(answer, "members", {}).values()):
                result.guidance = bound[key].references
    return [{key: out[key] for key in items} for out in outputs]


def ask(get_judge: Callable[[], Judge], state, items: Mapping, fields=None, guidance=None) -> dict:
    return ask_many(get_judge, [state], items, fields, guidance)[0]
