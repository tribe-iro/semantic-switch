from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import math

import numpy as np

from .errors import ConfigError, suggest
from .kernel import F, NEEDS_INPUT, T, U
from .protocols import Question
from .guidance import Guidance, normalize
from .results import Prediction

TRUE_WORDS = {"true", "yes", "y", "1"}
FALSE_WORDS = {"false", "no", "n", "0"}
UNSTATED_WORDS = {"unstated", "not stated", "not_stated", "unknown", "u"}
THREE_WAY = {
    (False, False): ("Claim: {} What does the input say about this claim?",
                     ("the input states it is true", "the input states it is false", "the input does not say")),
    (False, True): ("Claim: {} What follows from the supplied guidance and case evidence?",
                    ("the claim is supported", "the claim is contradicted", "the evidence is insufficient to determine the claim")),
    (True, False): ("{} What does the input say?", ("yes", "no", "the input does not say")),
    (True, True): ("{} What follows from the supplied guidance and case evidence?",
                   ("yes", "no", "the evidence is insufficient to answer")),
}
MIN_MASS = 0.5
MAX_OPTIONS = 20
TOLERANCE = 1e-3
ESCAPE = "another one, not listed above"
WHY = {"low_answer_mass": "low answer mass", "not_stated": "not stated in the input", "insufficient_evidence": "insufficient_evidence",
       "not_examined": "not examined (over max_claims)"}
REASONS = {**WHY, "uncertain": "uncertain", "superseded": "may be superseded", "exclusive_conflict": "exclusive conflict"}


def reason_text(reason: tuple) -> str:
    if reason[0] == "missing_input":
        return f"missing: {', '.join(reason[1])}"
    if reason[0] == "model_error":
        return f"error: {reason[1]}"
    if reason[0] == "not_stated" and len(reason) > 1 and reason[1]:
        return f"not stated: {'; '.join(reason[1])}"
    return REASONS.get(reason[0], reason[0])


def authored_text(value, name="criterion"):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string")
    return value


def authored_bool(value, name):
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be boolean")
    return value


def authored_strings(value, name):
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{name} must be a sequence of strings")
    return tuple(authored_text(item, name) for item in value)


def authored_levels(value):
    levels = authored_strings(value, "levels")
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ConfigError(f"rate needs at least two distinct levels, got {list(levels)}")
    if len(levels) > MAX_OPTIONS:
        raise ConfigError(f"rate takes at most {MAX_OPTIONS} levels, got {len(levels)}")
    return levels


def authored_budget(value):
    if value is not None and (type(value) is not int or not 1 <= value <= 256):
        raise ConfigError(f"max_claims must be an integer from 1 to 256, got {value!r}")
    return value


def truth(value) -> bool:
    if isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if word in TRUE_WORDS | FALSE_WORDS:
        return word in TRUE_WORDS
    raise ConfigError(f"{value!r} is not a true/false value")


@dataclass(frozen=True, slots=True)
class Label:
    description: str
    excludes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        authored_text(self.description, "description")
        object.__setattr__(self, "excludes", authored_strings(self.excludes, "excludes"))
        object.__setattr__(self, "aliases", authored_strings(self.aliases, "aliases"))

    @classmethod
    def of(cls, value) -> Label:
        if isinstance(value, (cls, str)):
            return value if isinstance(value, cls) else cls(value)
        if not isinstance(value, Mapping):
            raise ConfigError("a label is a description or a mapping of description, excludes and aliases")
        unknown = sorted(map(str, set(value) - set(cls.__dataclass_fields__)))
        if unknown:
            raise ConfigError(f"unknown label field {unknown[0]!r}" + ("; requires is exact eligibility: use eligible= inline or "
                                                                     "requires in a saved switch" if "requires" in unknown else ""))
        if "description" not in value:
            raise ConfigError("a label needs a description")
        return cls(value["description"], value.get("excludes", ()), value.get("aliases", ()))

    def record(self):
        if not self.excludes and not self.aliases:
            return self.description
        return {"description": self.description, **({"excludes": list(self.excludes)} if self.excludes else {}),
                **({"aliases": list(self.aliases)} if self.aliases else {})}


def definitions(value) -> Mapping[str, Label]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigError("labels must be a nonempty mapping of keys to definitions")
    return MappingProxyType({authored_text(key, "label key"): Label.of(label) for key, label in value.items()})


def labels(path) -> Mapping[str, Label]:
    from .files import read_yaml
    try:
        return definitions(read_yaml(path))
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None


@dataclass(frozen=True, slots=True)
class Policy:
    t_true: float | None = 0.90
    t_false: float | None = 0.90

    def thresholds(self, kind):
        return {"true": self.t_true, "false": self.t_false} if kind == "check" else {"answer": self.t_true}

    @classmethod
    def of(cls, min_score: float | None) -> Policy:
        if min_score is None:
            return DEFAULT
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0.5 < min_score <= 1:
            raise ConfigError(f"min_score must be above 0.5 and at most 1, got {min_score!r}")
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
    guidance: Guidance = Guidance()
    candidate: str | None = None

    def __post_init__(self):
        authored_text(self.text)
        authored_bool(self.unstated, "unstated")
        object.__setattr__(self, "excludes", authored_strings(self.excludes, "excludes"))
        object.__setattr__(self, "descriptions", authored_strings(self.descriptions, "descriptions"))

    @classmethod
    def check(cls, text: str, excludes=(), unstated: bool = False, guidance=None, candidate: str | None = None) -> Judgment:
        keys = ("true", "false", "unstated") if unstated else ("true", "false")
        return cls("check", text, keys, keys, authored_strings(excludes, "excludes"), unstated, normalize(guidance), candidate)

    @property
    def silence(self) -> frozenset:
        return frozenset({2}) if self.unstated else frozenset()

    def question(self, options: tuple[str, ...] | None = None) -> Question:
        if self.kind == "check":
            asked = Question("check", self.text, (), self.excludes, self.guidance, candidate=self.candidate)
            if not self.unstated:
                return asked
            wording, options = THREE_WAY[asked.interrogative, bool(self.guidance)]
            return Question("pick", self.text if self.candidate else wording.format(self.text), options, self.excludes,
                            self.guidance, candidate=self.candidate)
        return Question(self.kind, self.text, self.descriptions if options is None else options, self.excludes, self.guidance)

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


def answer(j: Judgment, dist, policy: Policy, mass: float | None = None,
           pending=frozenset()) -> tuple[int | None, str | None]:
    if mass is not None and mass < MIN_MASS:
        return None, "low_answer_mass"
    if j.kind == "check":
        if policy.t_true is not None and dist[0] >= policy.t_true:
            return 0, None
        if policy.t_false is not None and dist[1] >= policy.t_false:
            return 1, None
        if j.unstated and policy.t_true is not None and dist[2] >= policy.t_true:
            return 2, "insufficient_evidence" if j.guidance else "not_stated"
        return None, "uncertain"
    top = int(np.argmax(dist))
    state, why = _sets(dist, j.truth_of(top), policy.t_true, pending)
    return (top, None) if state == T else (None, why or "uncertain")


def lower(dist, indices) -> float:
    return 1. if len(indices) == len(dist) else sum(dist[i] for i in indices)


def _sets(dist, truth_vector, theta: float | None, pending) -> tuple[int, str | None]:
    true = {i for i, (_, t) in enumerate(zip(dist, truth_vector)) if t == T and i not in pending}
    false = {i for i, (_, t) in enumerate(zip(dist, truth_vector)) if t == F and i not in pending}
    if theta is None:
        return U, "uncertain"
    if lower(dist, true) >= theta:
        return T, None
    if lower(dist, false) >= theta:
        return F, None
    held = {i for i in pending if truth_vector[i] in (T, F)}
    if pending and max(lower(dist, true | held), lower(dist, false | held)) >= theta:
        return U, "missing_input"
    return U, "uncertain"


def settle(j: Judgment, dist, truth_vector, policy: Policy, mass: float | None = None,
           pending=frozenset()) -> tuple[int, str | None]:
    if mass is not None and mass < MIN_MASS:
        return U, "low_answer_mass"
    if j.kind == "check":
        i, why = answer(j, dist, policy, mass)
        if i is None:
            return U, why
        state = truth_vector[i]
        return state, (why if state == U else None)
    return _sets(dist, truth_vector, policy.t_true, pending)


def accept(j: Judgment, dist, policy: Policy, mass: float | None = None,
           pending=frozenset()) -> tuple[int | None, str | None]:
    i, why = answer(j, dist, policy, mass, pending)
    return (None, why) if j.kind == "check" and i == 2 else (i, why)


def outcome(j: Judgment, dist, policy: Policy, mass: float | None = None, pending=frozenset(), *,
            none: int | None = None, complete: bool = True) -> tuple[str, int | None, str | None]:
    i, why = accept(j, dist, policy, mass, pending)
    if i is None:
        return "needs_input" if why in NEEDS_INPUT else "uncertain", None, why
    return "ok" if i != none else "none" if complete else "uncertain", i, why


def unpresent(slots, sc, width: int) -> np.ndarray | None:
    probs, slots = np.asarray(sc.probs, dtype=np.float64), list(slots)
    if -1 in slots:
        e = slots.index(-1)
        if escaped(probs, e, sc.mass):
            return None
        probs, slots = np.delete(probs, e) / (1 - probs[e]), slots[:e] + slots[e + 1:]
    dist = np.zeros(width)
    dist[slots] = probs
    return dist


def escaped(probs, e: int, mass: float | None) -> bool:
    return int(np.argmax(probs)) == e or (mass is not None and mass < MIN_MASS)


def complete(dist) -> bool:
    return abs(float(np.sum(dist)) - 1) <= TOLERANCE


def winner(dist) -> int | None:
    return None if dist is None or max(dist) <= 0 else int(np.argmax(dist))


def predict(j: Judgment, dist, none: int | None = None, named=None) -> Prediction | None:
    i = winner(dist)
    if i is None:
        return None
    kind = "unstated" if j.kind == "check" and i == 2 else "none" if i == none else "value"
    key, value = ((named or (lambda k: (j.keys[k], j.value(k))))(i)) if kind == "value" else (None, None)
    return Prediction(kind, key, value, float(dist[i]))
