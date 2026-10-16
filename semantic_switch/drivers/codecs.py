from __future__ import annotations

import math
from functools import lru_cache
from types import MappingProxyType
from typing import Protocol, Sequence

import numpy as np

from ..errors import ConfigError
from ..judge import Request, Scores
from ..protocols import Question
from .profiles import ModelProfile, Precision

TOP_LOGPROBS = 20
SPELLINGS = "declared-exact-spellings@1"


class Codec(Protocol):
    name: str
    revision: int
    joint: bool
    parameters: bool
    response_bytes: int
    identity: dict

    def accepts(self, profile: ModelProfile) -> None: ...

    def encode(self, state: str, questions: Sequence[Question], profile: ModelProfile) -> dict: ...

    def decode(self, payload: dict, questions: Sequence[Question], profile: ModelProfile, model: str, request: Request) -> tuple[Scores, ...]: ...


def probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and within [0, 1]")
    return value


def distribution(probs: dict, keys: Sequence[str], precision: Precision) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(probs, dict):
        raise ValueError("probabilities must be an object")
    if set(probs) != set(keys) or len(probs) != len(keys):
        raise ValueError(f"probabilities must cover the option ids {', '.join(keys)} exactly once")
    raw = np.array([probability(probs[key]) for key in keys])
    total = math.fsum(raw)
    if total <= 0:
        raise ValueError("probabilities have zero total mass")
    bound = precision.bound(len(keys))
    if abs(total - 1) > bound:
        raise ValueError(f"probability mass {total!r} differs from one by more than the declared {precision.decimals}-decimal bound {bound:.6g}")
    return raw, raw / total


def confidence(answer):
    try:
        return probability(answer.get("confidence"))
    except ValueError:
        return None


def instructions(q: Question) -> str:
    from ..presentation import criterion
    return "\n".join([*([f"Guidance:\n{q.guidance.text}\n\nCriterion:"] if q.guidance else []), criterion(q), *(f"Except: {x}." for x in q.excludes)])


class Chat:
    name, revision, joint, parameters, response_bytes = "chat", 2, False, True, 1 << 18
    identity = MappingProxyType({"protocol": "chat-logprobs", "reasoning": "disabled", "top_logprobs": TOP_LOGPROBS, "answers": SPELLINGS})

    @staticmethod
    def accepts(profile: ModelProfile) -> None:
        if profile.codec != "chat" or profile.max_questions != 1 or profile.max_slots > TOP_LOGPROBS:
            raise ConfigError(f"profile {profile.name} is incompatible with one-token chat log probabilities (one question, at most {TOP_LOGPROBS} answers)")

    @staticmethod
    def encode(state: str, questions: Sequence[Question], profile: ModelProfile) -> dict:
        from ..presentation import messages
        system, user = messages(state, *questions)
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 1, "logprobs": True, "top_logprobs": TOP_LOGPROBS, "reasoning": {"enabled": False}}

    @staticmethod
    def decode(payload: dict, questions: Sequence[Question], profile: ModelProfile, model: str, request: Request) -> tuple[Scores, ...]:
        if ((payload.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"):
            raise ValueError("reasoning tokens were billed despite disabled reasoning")
        return (token_scores(*questions, payload, model, request),)


@lru_cache(maxsize=64)
def spellings(words: tuple[str, ...]) -> MappingProxyType:
    return MappingProxyType({spelled: k for k, word in enumerate(words) for form in dict.fromkeys((word, word.capitalize()))
                             for spelled in (form, " " + form)})


def token_scores(question: Question, response: dict, model: str | None = None, request: Request | None = None) -> Scores:
    from ..presentation import answers
    choices = response["choices"]
    if len(choices) != 1 or choices[0].get("finish_reason") not in ("stop", "length"):
        raise ValueError("expected one completed answer token")
    content = (choices[0].get("logprobs") or {}).get("content") or ()
    if len(content) != 1:
        raise ValueError("answer-position log probabilities unavailable")
    found = {}
    for row in (content[0], *content[0]["top_logprobs"]):
        token, logprob = row["token"], row["logprob"]
        if not isinstance(token, str) or isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob) or logprob > 0:
            raise ValueError("invalid answer log probability")
        if found.setdefault(token, logprob) != logprob:
            raise ValueError("conflicting answer token log probabilities")
    words = answers(question)
    legal, mass = spellings(words), np.zeros(len(words))
    for token, logprob in found.items():
        if token in legal:
            mass[legal[token]] += math.exp(logprob)
    reported, answered = math.fsum(map(math.exp, found.values())), float(mass.sum())
    if reported > 1 + 1e-3:
        raise ValueError("reported token mass exceeds one")
    if answered <= 0:
        raise ValueError("no answer token among the returned alternatives")
    return Scores(mass / (answered + max(0., 1 - reported)), min(answered, 1.), model=model, request=request)


class SystemOne:
    name, revision, joint, parameters, response_bytes = "systemone", 4, True, False, 1 << 22
    identity = MappingProxyType({"protocol": "systemone", "levels": "index@1", "optional": "discard-invalid@1"})
    TYPES = MappingProxyType({"check": "noul", "rate": "score", "pick": "choice"})

    @staticmethod
    def accepts(profile: ModelProfile) -> None:
        if profile.codec != "systemone" or profile.precision is None:
            raise ConfigError(f"profile {profile.name} must declare the System One serialization precision; unknown precision is not decoded")
        if set(profile.kinds) - set(SystemOne.TYPES):
            raise ConfigError(f"profile {profile.name}: System One answers check, pick and rate questions only")

    @staticmethod
    def encode(state: str, questions: Sequence[Question], profile: ModelProfile) -> dict:
        out = {}
        for k, question in enumerate(questions):
            row = {"instructions": instructions(question), "type": SystemOne.TYPES[question.kind]}
            if question.kind == "rate":
                row["criteria"] = list(question.options)
            elif question.kind == "pick":
                row["criteria"] = dict(zip(profile.ids(question.width), question.options))
            out[f"q{k}"] = row
        return {"state": state, "questions": out}

    @staticmethod
    def decode(payload: dict, questions: Sequence[Question], profile: ModelProfile, model: str, request: Request) -> tuple[Scores, ...]:
        if payload.get("refusal"):
            return tuple(Scores(None, None, f"hosted refusal: {payload['refusal']}", model=model, request=request) for _ in questions)
        answers = payload["answers"]
        if not isinstance(answers, dict) or set(answers) - {f"q{k}" for k in range(len(questions))}:
            raise ValueError("answers must be an object keyed by the requested question names")
        return tuple(answer(question, answers.get(f"q{k}"), profile, model, request) for k, question in enumerate(questions))


def answer(question: Question, reply, profile: ModelProfile, model: str, request: Request) -> Scores:
    try:
        if reply is None:
            raise ValueError("no answer for the requested question")
        if not isinstance(reply, dict):
            raise ValueError("expected an object")
        if reply.get("refusal") or reply.get("type") == "refusal":
            return Scores(None, None, f"hosted refusal: {reply.get('refusal', 'refused')}", model=model, request=request)
        expected = SystemOne.TYPES[question.kind]
        if reply.get("type") != expected:
            raise ValueError(f"answer type {reply.get('type')!r} differs from the requested {expected!r}")
        if question.kind == "check":
            p = probability(reply["noul"])
            return Scores(np.array([p, 1 - p]), None, None, confidence(reply), model, request, np.array([p]))
        keys = tuple(map(str, range(question.width))) if question.kind == "rate" else profile.ids(question.width)
        if "probabilities" not in reply:
            raise ValueError("no per-level distribution; a provider's expected score or selected option is not a substitute")
        raw, probs = distribution(reply["probabilities"], keys, profile.precision)
        return Scores(probs, None, None, confidence(reply), model, request, raw, auxiliary(question, reply, keys))
    except (KeyError, TypeError, ValueError) as error:
        return Scores(None, None, f"hosted: malformed answer ({error})", model=model, request=request)


def auxiliary(question: Question, reply: dict, keys) -> dict | None:
    if question.kind == "rate":
        mean = reply.get("score")
        if not isinstance(mean, bool) and isinstance(mean, (int, float)) and math.isfinite(mean) and 0 <= mean <= len(keys) - 1:
            return {"score": float(mean), "basis": "level-index-mean"}
    if question.kind == "pick" and isinstance(reply.get("choice"), str) and reply["choice"] in keys:
        return {"choice": reply["choice"]}
    return None


CODECS = MappingProxyType({codec.name: codec for codec in (Chat, SystemOne)})
