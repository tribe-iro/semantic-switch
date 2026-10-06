from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigError, suggest
from .kernel import F, T, U
from .protocols import Question

TRUE_WORDS = {"true", "yes", "y", "1"}
FALSE_WORDS = {"false", "no", "n", "0"}
UNSTATED_WORDS = {"unstated", "not stated", "not_stated", "unknown", "u"}
THREE_WAY = ("the input states it is true", "the input states it is false", "the input does not say")
MIN_MASS = 0.5


def truth(value) -> bool:
    if isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if word in TRUE_WORDS | FALSE_WORDS:
        return word in TRUE_WORDS
    raise ConfigError(f"{value!r} is not a true/false value")


def claim_question(claim: str) -> str:
    return f"Claim: {claim} What does the input say about this claim?"


@dataclass(frozen=True, slots=True)
class Policy:
    t_true: float | None = 0.90
    t_false: float | None = 0.90

    @classmethod
    def of(cls, min_score: float | None) -> Policy:
        if min_score is None:
            return DEFAULT
        if not 0.5 < min_score <= 1:
            raise ValueError(f"min_score must be above 0.5 and at most 1, got {min_score!r}")
        return cls(min_score, min_score)


DEFAULT = Policy()
CERTAIN = Policy(1.0, 1.0)
OFF = Policy(None, None)


@dataclass(frozen=True, slots=True)
class Judgment:
    kind: str
    text: str
    keys: tuple[str, ...] = ("true", "false")
    descriptions: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    unstated: bool = False

    @classmethod
    def check(cls, text: str, excludes=(), unstated: bool = False) -> Judgment:
        keys = ("true", "false", "unstated") if unstated else ("true", "false")
        return cls("check", text, keys, keys, tuple(excludes), unstated)

    @property
    def silence(self) -> frozenset:
        return frozenset({2}) if self.unstated else frozenset()

    def question(self, options: tuple[str, ...] | None = None) -> Question:
        if self.kind == "check":
            if self.unstated:
                return Question("pick", claim_question(self.text), THREE_WAY, self.excludes)
            return Question("check", self.text, (), self.excludes)
        return Question(self.kind, self.text, self.descriptions if options is None else options, self.excludes)

    def index(self, value, name: str = "") -> int:
        if self.kind == "check":
            if self.unstated and str(value).strip().lower() in UNSTATED_WORDS:
                return 2
            return 0 if truth(value) else 1
        if str(value) not in self.keys:
            raise ConfigError(f"{value!r} is not a value of {name or self.text}{suggest(str(value), self.keys)}")
        return self.keys.index(str(value))

    def value(self, i: int):
        if self.kind == "check":
            return "unstated" if i == 2 else i == 0
        return self.keys[i]

    def truth_of(self, i: int) -> tuple[int, ...]:
        return tuple(U if k in self.silence else T if k == i else F for k in range(len(self.keys)))

    @property
    def main(self) -> tuple[int, ...]:
        return self.truth_of(0) if self.kind == "check" else ()


def settle(dist, truth_vector, policy: Policy, mass: float | None = None, silence=frozenset(),
           pending=frozenset()) -> tuple[int, str | None]:
    if mass is not None and mass < MIN_MASS:
        return U, "low_answer_mass"
    p_true = sum(p for i, (p, t) in enumerate(zip(dist, truth_vector)) if t == T and i not in pending)
    p_false = sum(p for i, (p, t) in enumerate(zip(dist, truth_vector)) if t == F and i not in pending)
    if policy.t_true is not None and p_true >= policy.t_true:
        return T, None
    if policy.t_false is not None and p_false >= policy.t_false:
        return F, None
    if pending:
        held_t = sum(dist[i] for i in pending if truth_vector[i] == T)
        held_f = sum(dist[i] for i in pending if truth_vector[i] == F)
        if (policy.t_true is not None and p_true + held_t >= policy.t_true) or \
                (policy.t_false is not None and p_false + held_f >= policy.t_false):
            return U, "missing_input"
    if silence and policy.t_true is not None and sum(dist[i] for i in silence) >= policy.t_true:
        return U, "not_stated"
    return U, "uncertain"


def escaped(probs, policy: Policy, mass: float | None) -> bool:
    top = int(np.argmax(probs))
    return top == len(probs) - 1 or policy.t_true is None or probs[top] < policy.t_true or \
        (mass is not None and mass < MIN_MASS)


def accept(j: Judgment, dist, policy: Policy, mass: float | None = None,
           pending=frozenset()) -> tuple[int | None, str | None]:
    if j.kind == "check":
        state, why = settle(dist, j.main, policy, mass, j.silence, pending)
        return (0 if state == T else 1 if state == F else None), why
    top = int(np.argmax(dist))
    state, why = settle(dist, j.truth_of(top), policy, mass, j.silence, pending)
    return (top if state == T else None), why or "uncertain"
