from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .kernel import F, STATE_NAMES, T, U, evaluate

EMPTY = MappingProxyType({})
MARK = {T: "✓", F: "✗", U: "?"}


def _num(x) -> str:
    return "–" if x is None else f"{x:.2f}"


class Result:
    __slots__ = ()

    def __bool__(self):
        raise TypeError(f"{type(self).__name__} has no truth value; use .value or .status")


@dataclass(slots=True, eq=False, repr=False)
class CheckResult(Result):
    status: str
    value: bool | None = None
    p_true: float | None = None
    mass: float | None = None
    error: str | None = None
    p_unstated: float | None = None

    def __repr__(self):
        return f"<{self.status} · {self.error}>" if self.error else \
            f"<{self.status} · {self.value} · p_true {_num(self.p_true)}>"

    def __str__(self):
        return _human(self.status, "true" if self.value else "false", f"p_true {_num(self.p_true)}", self.error)

    def as_dict(self, verbose: bool = False) -> dict:
        out = {"status": self.status, "value": self.value, "p_true": self.p_true, "mass": self.mass,
               "error": self.error}
        return out | {"p_unstated": self.p_unstated} if self.p_unstated is not None else out


@dataclass(slots=True, eq=False, repr=False)
class PickResult(Result):
    status: str
    value: object = None
    index: object = None
    score: float | None = None
    mass: float | None = None
    removed: dict = field(default_factory=dict)
    truncated: list = field(default_factory=list)
    error: str | None = None
    focus: str | None = None
    complete: bool = True
    _keys: tuple = ()
    _probs: np.ndarray | None = None

    @property
    def scores(self) -> dict:
        return {} if self._probs is None else {k: float(p) for k, p in zip(self._keys, self._probs)}

    def __repr__(self):
        return f"<{self.status} · {self.error}>" if self.error else \
            f"<{self.status} · {self.value!r} · {_num(self.score)}>"

    def __str__(self):
        return _human(self.status, str(self.value), _num(self.score), self.error)

    def as_dict(self, verbose: bool = False) -> dict:
        out = {"status": self.status, "value": self.value, "index": self.index, "score": self.score,
               "error": self.error, "complete": self.complete}
        return out | {"scores": self.scores, "removed": self.removed, "truncated": self.truncated,
                      "mass": self.mass, "focus": self.focus} if verbose else out


@dataclass(slots=True, eq=False, repr=False)
class RateResult(Result):
    status: str
    value: str | None = None
    index: int | None = None
    expected: float | None = None
    score: float | None = None
    mass: float | None = None
    error: str | None = None
    _levels: tuple = ()
    _probs: np.ndarray | None = None

    @property
    def scores(self) -> dict:
        return {} if self._probs is None else {k: float(p) for k, p in zip(self._levels, self._probs)}

    def __repr__(self):
        return f"<{self.status} · {self.error}>" if self.error else \
            f"<{self.status} · {self.value!r} · {_num(self.score)}>"

    def __str__(self):
        return _human(self.status, str(self.value), _num(self.score), self.error)

    def as_dict(self, verbose: bool = False) -> dict:
        out = {"status": self.status, "value": self.value, "index": self.index, "score": self.score,
               "error": self.error}
        return out | {"expected": self.expected, "scores": self.scores, "mass": self.mass} if verbose else out


def _human(status: str, value: str, detail: str, error: str | None) -> str:
    if status == "failed":
        return f"failed: {error}"
    if status == "needs_input":
        return f"needs input: {error}"
    if status in ("uncertain", "none"):
        return f"{status} ({detail})"
    return f"{value} ({detail})"


@dataclass(frozen=True, slots=True, repr=False)
class Fact:
    value: object
    score: float | None
    settled: bool
    source: str | None
    why: str | None = None
    scores: Mapping = EMPTY
    unresolved: Mapping = EMPTY
    superseded: Mapping = EMPTY
    cards: tuple = ()
    complete: bool | None = None
    states: Mapping = EMPTY

    def __repr__(self):
        shown = repr(self.value) if self.settled else "?"
        extra = f", {self.why}" if self.why else ""
        return f"Fact({shown}, {_num(self.score)}, {self.source or 'not scored'}{extra})"

    def as_dict(self, verbose: bool = False) -> dict:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        out = {"value": value, "settled": self.settled, "score": self.score, "source": self.source,
               "why": self.why}
        if self.complete is not None:
            out |= {"complete": self.complete, "cards": list(self.cards), "states": dict(self.states),
                    "unresolved": dict(self.unresolved), "superseded": dict(self.superseded)}
        return out | {"scores": dict(self.scores)} if verbose else out


@dataclass(frozen=True, slots=True)
class Trace:
    plan: object
    states: tuple
    reasons: Mapping


@dataclass(slots=True, eq=False, repr=False)
class SwitchResult(Result):
    status: str
    answer: object = None
    case: object = None
    missing: tuple = ()
    invalid: tuple = ()
    uncertain: tuple = ()
    blocked_by: str | None = None
    partial: tuple = ()
    errors: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    truncated: tuple = ()
    _trace: Trace | None = None

    def __repr__(self):
        ms = f" · {self.cost['ms']:.0f} ms" if self.cost.get("ms") is not None else ""
        if self.status in ("ok", "none") and not self._cased:
            return f"<{self.status} · {self._settled_facts()}{ms}>"
        if self.status == "ok":
            judged = self.cost.get("judge", 0)
            return f"<ok · {self.answer!r} · case {self.case}{f' · {judged} judged' if judged else ''}{ms}>"
        if self.status == "needs_input":
            return f"<needs_input · missing {', '.join(self.missing)} · blocked at {self.blocked_by}{ms}>"
        if self.status == "uncertain":
            return f"<uncertain · {', '.join(self.uncertain)} · blocked at {self.blocked_by}{ms}>"
        detail = ", ".join(self.invalid or self.missing or tuple(self.errors))
        return f"<failed · {detail}>"

    @property
    def _cased(self) -> bool:
        return self._trace is None or bool(self._trace.plan.cases)

    def _settled_facts(self) -> str:
        return ", ".join(f"{k}={v.value!r}" for k, v in self.facts.items() if v.settled)

    def __str__(self):
        if self.status in ("ok", "none") and not self._cased:
            return self._settled_facts() or self.status
        if self.status == "ok":
            return f"{self.answer}  (case {self.case})"
        if self.status == "needs_input":
            return f"needs input: {', '.join(self.missing)}  (blocked at {self.blocked_by})"
        if self.status == "uncertain":
            return f"uncertain: {', '.join(self.uncertain)}  (blocked at {self.blocked_by})"
        if self.invalid:
            return f"failed: invalid {', '.join(self.invalid)}"
        if self.missing:
            return f"failed: missing required {', '.join(self.missing)}"
        return "failed: " + "; ".join(f"{k}: {v}" for k, v in self.errors.items())

    def as_dict(self, verbose: bool = False) -> dict:
        out = {"status": self.status, "answer": self.answer, "case": self.case, "missing": list(self.missing),
               "invalid": list(self.invalid), "uncertain": list(self.uncertain), "blocked_by": self.blocked_by,
               "errors": self.errors, "facts": {k: f.as_dict(verbose) for k, f in self.facts.items()},
               "truncated": list(self.truncated)}
        return out | {"partial": list(self.partial), "cost": self.cost} if verbose else out

    def explain(self, raw: bool = False) -> str:
        if self._trace is None:
            return repr(self)
        plan, st = self._trace.plan, list(self._trace.states)
        width = max([len(c.id) for c in plan.cases] + [len(f.name) for f in plan.visible] + [9])
        lines = [f"{plan.name} → {self!r}"]
        if plan.cases:
            lines.append("cases")
            reached = True
            for case in plan.cases:
                v = evaluate(case.expr, st)
                mark = MARK[v] if reached else "·"
                lines.append(f"  {mark} {case.id:<{width}}  {case.when}")
                if plan.first and v != F:
                    reached = False
            lines.append(f"  {'✓' if self.case == 'otherwise' else '·'} {'otherwise':<{width}}  → {plan.otherwise!r}")
        lines.append("facts")
        for f in plan.visible:
            fact = self.facts.get(f.name)
            if fact is None:
                lines.append(f"  · {f.name:<{width}}  not needed")
                continue
            mark = "✓" if fact.settled else "?"
            shown = repr(fact.value) if fact.settled else "unsettled"
            lines.append(f"  {mark} {f.name:<{width}}  {shown}  {_num(fact.score)}  {fact.source or ''}"
                         f"{'  — ' + fact.why if fact.why else ''}")
        if raw:
            lines.append("atoms")
            for i, key in enumerate(plan.atom_keys):
                reason = self._trace.reasons.get(i)
                lines.append(f"  {STATE_NAMES[st[i]]} {key}{'  (' + reason[0] + ')' if reason else ''}")
        return "\n".join(lines)
