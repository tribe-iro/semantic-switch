from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from . import files
from .compiler import MAX_OPTIONS, FieldSpec
from .context import bm25_rank, project, render
from .errors import ConfigError
from .judge import Scores, scored
from .judgment import Judgment, Policy, accept, escaped
from .kernel import mentioned
from .protocols import Judge, Question
from .results import CheckResult, PickResult, RateResult

CANDIDATE_TOKENS = 64
MAX_CANDIDATES = 1000
ESCAPE = "another one, not listed above"
WHY = {"low_answer_mass": "low answer mass", "not_stated": "not stated in the input"}


def rendered(state, fields=None) -> str:
    if isinstance(state, str):
        return state
    specs = {k: FieldSpec(type="json") for k in fields or tuple(state)}
    return render(specs, project(specs, state), tuple(specs))


@dataclass(frozen=True)
class Prepared:
    make: Callable[[Judge], Question] | None
    finish: Callable[[Scores], object] | None
    immediate: object = None


def done(result) -> Prepared:
    return Prepared(None, None, result)


class Saveable:
    def fact(self) -> dict:
        raise NotImplementedError

    def __call__(self, state, rt=None, fields=None):
        from .api import default

        return (rt or default()).ask(state, {"_": self}, fields=fields)["_"]

    def save(self, path, field: str = "message", rt=None):
        from .api import default

        path = Path(path)
        name = path.name.removesuffix(".yaml")
        spec = {"name": name, "fields": {field: {"type": "text", "required": True}}, "reads": [field],
                "facts": {name: {k: v for k, v in self.fact().items() if v not in (None, [], {}, False)}}}
        if path.exists():
            if files.read_yaml(path) != spec:
                raise ConfigError(f"{path} exists with a different definition")
        else:
            files.write_yaml(path, spec)
        return (rt or default()).load(path)


@dataclass(frozen=True)
class Check(Saveable):
    statement: str
    excludes: tuple = ()
    min_score: float | None = None
    unstated: bool = False

    def fact(self) -> dict:
        return {"check": self.statement, "excludes": list(self.excludes), "unstated": self.unstated,
                "min_score": self.min_score}

    def prepare(self, text: str) -> Prepared:
        j = Judgment.check(self.statement, self.excludes, self.unstated)
        policy = Policy.of(self.min_score)

        def finish(sc: Scores) -> CheckResult:
            if sc.error:
                return CheckResult("failed", error=sc.error)
            p, pu = float(sc.probs[0]), float(sc.probs[2]) if self.unstated else None
            i, why = accept(j, sc.probs, policy, sc.mass)
            if i is not None:
                return CheckResult("ok", i == 0, p, sc.mass, p_unstated=pu)
            status = "needs_input" if why == "not_stated" else "uncertain"
            return CheckResult(status, None, p, sc.mass, WHY.get(why), pu)

        return Prepared(lambda judge: j.question(), finish)


@dataclass(frozen=True)
class Rate(Saveable):
    question: str
    levels: tuple
    min_score: float | None = None

    def fact(self) -> dict:
        return {"rate": self.question, "levels": list(self.levels), "min_score": self.min_score}

    def prepare(self, text: str) -> Prepared:
        levels = tuple(self.levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ConfigError(f"rate needs at least two distinct levels, got {list(levels)}")
        if len(levels) > MAX_OPTIONS:
            raise ConfigError(f"rate takes at most {MAX_OPTIONS} levels, got {len(levels)}")
        j = Judgment("rate", self.question, levels, levels)
        policy = Policy.of(self.min_score)

        def finish(sc: Scores) -> RateResult:
            if sc.error:
                return RateResult("failed", error=sc.error, _levels=levels)
            i, why = accept(j, sc.probs, policy, sc.mass)
            top = int(np.argmax(sc.probs))
            return RateResult("ok" if i is not None else "uncertain", None if i is None else levels[i], i,
                              float((sc.probs * np.arange(len(levels))).sum()), float(sc.probs[top]), sc.mass,
                              WHY.get(why), _levels=levels, _probs=sc.probs)

        return Prepared(lambda judge: j.question(), finish)


def identities(options, objs: list, ids: Callable | None) -> list[tuple[str, ...]] | None:
    if isinstance(options, Mapping):
        return [(k, *((v.get("aliases") or ()) if isinstance(v, Mapping) else ())) for k, v in options.items()]
    if ids is None:
        return None
    out = []
    for o in objs:
        got = ids(o)
        out.append((got,) if isinstance(got, str) else (got[0], *got[1]))
    return out


@dataclass(frozen=True)
class Pick(Saveable):
    question: str
    options: object
    describe: Callable | None = None
    eligible: Callable | None = None
    allow_none: bool = False
    shortlist: str | None = None
    min_score: float | None = None
    ids: Callable | None = None
    focus: str | None = None

    def fact(self) -> dict:
        if not isinstance(self.options, Mapping) or self.describe or self.eligible or self.ids or self.shortlist:
            raise ConfigError("only a pick with dict options (and no describe, eligible, ids or shortlist) saves")
        return {"pick": self.question, "options": dict(self.options), "allow_none": self.allow_none,
                "focus": self.focus, "min_score": self.min_score}

    def prepare(self, text: str) -> Prepared:
        policy = Policy.of(self.min_score)
        if self.shortlist not in (None, "lexical"):
            raise ConfigError(f"unknown shortlist mode {self.shortlist!r}; use None or 'lexical'")
        if self.focus not in (None, "mentions"):
            raise ConfigError(f"unknown focus mode {self.focus!r}; use None or 'mentions'")
        if isinstance(self.options, Mapping):
            keys = list(self.options)
            if self.allow_none and "none" in keys:
                raise ConfigError("'none' is reserved for allow_none; rename that option")
            if any(isinstance(v, Mapping) and "requires" in v for v in self.options.values()):
                raise ConfigError("option requires is YAML-only; inline, use eligible=")
            objs = keys
            given = [str(v.get("description", k)) if isinstance(v, Mapping) else str(v)
                     for k, v in self.options.items()]
            describe = given.__getitem__
        else:
            objs = list(self.options)
            keys = list(range(len(objs)))
            describe = (lambda i: str(self.describe(objs[i]))) if self.describe else (lambda i: str(objs[i]))
        idents = identities(self.options, objs, self.ids)
        keep = [i for i, o in enumerate(objs) if self.eligible is None or self.eligible(o)]
        kept = set(keep)
        removed = {"ineligible": [keys[i] for i in range(len(objs)) if i not in kept], "not_examined": []}
        if not keep:
            return done(PickResult("none", removed=removed) if self.allow_none else
                        PickResult("failed", removed=removed, error="no eligible candidates"))
        slots = MAX_OPTIONS - int(self.allow_none)
        if len(keep) > slots and (self.shortlist != "lexical" or len(keep) > MAX_CANDIDATES):
            limit = MAX_CANDIDATES if self.shortlist else slots
            return done(PickResult("failed", removed=removed, error=f"too many candidates ({len(keep)} > {limit})"))
        named = [i for i in keep if any(mentioned(text, t) for t in idents[i])] if idents else []
        descs = {i: describe(i) for i in keep}
        if len(keep) > slots:
            rest = [i for i in keep if i not in set(named)]
            ranked = named + [rest[k] for k in bm25_rank(f"{self.question} {text}", [descs[i] for i in rest])]
            chosen = set(ranked[:slots])
            removed["not_examined"] = [keys[i] for i in keep if i not in chosen]
            keep = [i for i in keep if i in chosen]

        def stage(shown: list[int], mode: str, escape: bool) -> Prepared:
            out_keys = tuple(keys[i] for i in shown) + (("none",) if self.allow_none else ())
            truncated: list = []

            def make(judge: Judge) -> Question:
                if len(shown) + self.allow_none + escape > judge.max_slots:
                    raise ConfigError(f"this judge takes at most {judge.max_slots} options per question")
                texts, cut = [], []
                for i in shown:
                    t, was_cut = judge.fit(descs[i], CANDIDATE_TOKENS)
                    texts.append(t)
                    if was_cut:
                        cut.append(keys[i])
                truncated[:] = cut
                tail = (["none of the above"] if self.allow_none else []) + ([ESCAPE] if escape else [])
                return Judgment("pick", self.question, descriptions=tuple(texts + tail)).question()

            def finish(sc: Scores):
                common = dict(removed=removed, truncated=list(truncated), _keys=out_keys, focus=mode,
                              complete=not removed["not_examined"])
                if sc.error:
                    return PickResult("failed", error=sc.error, **common)
                probs = sc.probs
                if escape:
                    if escaped(probs, policy, sc.mass):
                        return stage(keep, "fallback", False)
                    probs = probs[:-1] / probs[:-1].sum()
                s, why = accept(Judgment("pick", self.question, out_keys), probs, policy, sc.mass)
                top = int(probs.argmax())
                if s is None:
                    return PickResult("uncertain", score=float(probs[top]), mass=sc.mass, error=WHY.get(why),
                                      _probs=probs, **common)
                if self.allow_none and s == len(shown):
                    status = "none" if common["complete"] else "uncertain"
                    return PickResult(status, score=float(probs[s]), mass=sc.mass, _probs=probs, **common)
                i = shown[s]
                return PickResult("ok", objs[i], keys[i], float(probs[s]), sc.mass, _probs=probs, **common)

            return Prepared(make, finish)

        if self.focus == "mentions" and named and len(named) < len(keep):
            return stage(named, "mentions", True)
        return stage(keep, "full", False)


def ask(get_judge: Callable[[], Judge], state, items: Mapping, fields=None) -> dict:
    text = rendered(state, fields)
    pending = {k: spec.prepare(text) for k, spec in items.items()}
    out = {}
    while pending:
        live = {}
        for k, p in pending.items():
            if p.make is None:
                out[k] = p.immediate
            else:
                live[k] = p
        if not live:
            break
        judge = get_judge()
        questions = [p.make(judge) for p in live.values()]
        results = scored(judge, [(text, questions)])[0].scores
        pending = {}
        for (k, p), sc in zip(live.items(), results):
            r = p.finish(sc)
            if isinstance(r, Prepared):
                pending[k] = r
            else:
                out[k] = r
    return {k: out[k] for k in items}
