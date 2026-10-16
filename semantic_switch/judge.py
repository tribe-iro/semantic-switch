from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, replace
from contextlib import nullcontext
from contextvars import ContextVar
from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np

from .protocols import MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS, MAX_QUESTION_TOKENS, Question, ScoreBlock, ScoreRequest

from .judgment import MIN_MASS
from .metrics import OPTION_SOFTMAX
from .errors import ConfigError
BATCH_QUESTIONS = 64
BATCH_TOKENS = 16384
ADVISED = 1024
SCORE_SEMANTICS = "acceptance-lower-bound@1"

CONFORMANCE_PROBES = (
    "subject: Double charge\nbody: I was charged twice in May, please refund one of them.",
    "subject: Router dead\nbody: My router stopped working yesterday and nothing blinks. I need it today.",
    "subject: Final notice\nbody: If this is not fixed by Monday my lawyer will file a lawsuit against you.",
)
CONFORMANCE_QUESTIONS = (
    Question("check", "The customer asks to receive money back for a charge."),
    Question("pick", "What is the message mainly about?",
             ("charges, invoices or payments", "the product is broken, erroring or unavailable",
              "login, profile or account access")),
    Question("rate", "How urgent is the customer's situation?", ("can wait", "today", "immediately")),
    Question("check", "The customer threatens legal action against us."),
)


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    seconds: float
    attempts: int
    status: int | None
    usage: dict | None
    queue_seconds: float = 0.
    service_seconds: float = 0.
    retry_seconds: float = 0.
    admission: dict | None = None

    def __post_init__(self):
        from .results import freeze
        object.__setattr__(self, 'usage', freeze(self.usage))
        object.__setattr__(self, 'admission', freeze(self.admission))

    def as_dict(self) -> dict:
        from .results import thaw
        return {"id": self.id, "seconds": self.seconds, "attempts": self.attempts, "status": self.status, "usage": thaw(self.usage),
                "queue_seconds": self.queue_seconds, "service_seconds": self.service_seconds, "retry_seconds": self.retry_seconds,
                "admission": thaw(self.admission)}


def distinct(requests) -> tuple:
    return tuple({request.id: request for request in requests if request is not None}.values())


class Cancelled(RuntimeError):
    pass


class Cancellation:
    def __init__(self):
        self._lock, self._hooks, self.cancelled = threading.Lock(), {}, False

    def cancel(self):
        with self._lock:
            if self.cancelled:
                return
            self.cancelled, hooks = True, list(self._hooks.values())
        for hook in hooks:
            hook()

    def register(self, hook) -> Callable[[], None]:
        with self._lock:
            if not self.cancelled:
                key = object()
                self._hooks[key] = hook
                return lambda: self._hooks.pop(key, None)
        hook()
        return lambda: None


OPERATION: ContextVar[Cancellation | None] = ContextVar("semantic_switch_operation", default=None)


def checkpoint():
    handle = OPERATION.get()
    if handle is not None and handle.cancelled:
        raise Cancelled("operation cancelled")


@dataclass(frozen=True, slots=True)
class Scores:
    probs: np.ndarray | tuple[float, ...] | None
    mass: float | None
    error: str | None = None
    confidence: float | None = None
    model: str | None = None
    request: Request | None = None
    raw: np.ndarray | None = None
    auxiliary: dict | None = None

    @property
    def low_mass(self) -> bool:
        return self.mass is not None and self.mass < MIN_MASS


@dataclass(frozen=True, slots=True)
class Scored:
    scores: tuple[Scores, ...]
    tokens: int
    model: str | None = None
    native_work: tuple = ()


def _limit(req: ScoreRequest, k: int, max_input_tokens=MAX_INPUT_TOKENS) -> Scores | None:
    if len(req.state) > max_input_tokens:
        return Scores(None, None, f"limit: input is {len(req.state)} tokens (max {max_input_tokens})")
    if len(req.suffixes[k]) > MAX_QUESTION_TOKENS:
        return Scores(None, None, f"limit: question is {len(req.suffixes[k])} tokens (max {MAX_QUESTION_TOKENS})")
    return None


def _batches(requests: Sequence[ScoreRequest], live: list[tuple[int, int]]):
    batch, questions, tokens = [], 0, 0
    for n, k in live:
        cost = len(requests[n].suffixes[k]) + (0 if batch and batch[-1][0] == n else len(requests[n].state))
        if batch and (questions >= BATCH_QUESTIONS or tokens + cost > BATCH_TOKENS):
            yield batch
            batch, questions, tokens = [], 0, 0
            cost = len(requests[n].suffixes[k]) + len(requests[n].state)
        if batch and batch[-1][0] == n:
            batch[-1][1].append(k)
        else:
            batch.append((n, [k]))
        questions, tokens = questions + 1, tokens + cost
    if batch:
        yield batch


def _scores(block: ScoreBlock, row: int, width: int) -> Scores:
    if block.error[row]:
        return Scores(None, None, block.error[row])
    p = np.exp(block.logp[row, :width].astype(np.float64))
    lm = block.log_mass[row]
    if not np.isfinite(p).all() or p.sum() <= 0:
        return Scores(None, None, "native: invalid answer logits")
    return Scores(p / p.sum(), None if np.isnan(lm) else float(np.exp(lm)))


def _stamped(result: Scored) -> Scored:
    if result.model is None or all(score.model is not None for score in result.scores):
        return result
    return replace(result, scores=tuple(score if score.model is not None else replace(score, model=result.model) for score in result.scores))


def judge_name(judge) -> str:
    return judge.identity.get("file") or judge.identity.get("source") or type(judge).__name__


def validated(name: str, n: int, questions: Sequence[Question], result: Scored) -> Scored:
    result = _stamped(result)
    if len(result.scores) != len(questions):
        raise TypeError(f"judge {name}, input {n}: {len(result.scores)} scores for {len(questions)} questions")
    for k, (q, sc) in enumerate(zip(questions, result.scores)):
        if sc.error is not None:
            continue
        p = np.asarray(sc.probs, dtype=np.float64)
        if p.shape != (q.width,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() > 1 + 1e-3:
            raise TypeError(f"judge {name}, input {n}, question {k}: scores must be {q.width} finite "
                            f"non-negative values summing to at most 1, got {p.tolist()}")
        if sc.mass is not None and not 0 <= sc.mass <= 1:
            raise TypeError(f"judge {name}, input {n}, question {k}: answer mass must be None or within [0, 1], "
                            f"got {sc.mass}")
        if sc.confidence is not None and not 0 <= sc.confidence <= 1:
            raise TypeError(f"judge {name}, input {n}, question {k}: confidence must be None or within [0, 1], "
                            f"got {sc.confidence}")
    return result


def guided(judge, items) -> None:
    if any(q.guidance for _, qs in items for q in qs) and not getattr(judge, "supports_guidance", False):
        raise ConfigError("this judge does not support guidance")


def scored(judge, items: Sequence[tuple[str, Sequence[Question]]]) -> list:
    guided(judge, items)
    results = judge.score_many(items)
    name = judge_name(judge)
    if len(results) != len(items):
        raise TypeError(f"judge {name} returned {len(results)} results for {len(items)} inputs")
    return [validated(name, n, questions, result) for n, ((_, questions), result) in enumerate(zip(items, results))]


def validate_requests(requests: Sequence[ScoreRequest]) -> tuple[ScoreRequest, ...]:
    requests = tuple(requests)
    for n, r in enumerate(requests):
        if not isinstance(r, ScoreRequest) or not r.suffixes or len(r.suffixes) != len(r.slots):
            raise ConfigError(f"native request {n}: needs one answer-slot set per question")
        for slots in r.slots:
            ids = [t for group in slots for t in group]
            if not slots or not all(slots) or len(set(ids)) != len(ids):
                raise ConfigError(f"native request {n}: answer slots must be non-empty and disjoint")
        if any(np.any(segment.array < 0) for segment in (r.prefix, r.state, *r.suffixes)) or \
                any(type(token) is not int or token < 0 for slots in r.slots for group in slots for token in group):
            raise ConfigError(f"native request {n}: token IDs must be non-negative integers")
    return requests


class LlamaJudge:
    supports_guidance = True
    normalization = OPTION_SOFTMAX

    def __init__(self, scorer, adapter, identity: dict | None = None, accelerated: bool = False,
                 batch_inputs: int | None = None, *, max_input_tokens=MAX_INPUT_TOKENS, max_prefix_tokens=MAX_PREFIX_TOKENS, quality_profile=None):
        if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise ConfigError("max_input_tokens must be a positive integer")
        self.max_input_tokens = max_input_tokens
        self.max_prefix_tokens = max_prefix_tokens
        self.quality_profile = quality_profile
        self._advised = OrderedDict()
        self.scorer = scorer
        self.adapter = adapter
        from .presentation import identity as presentation_identity
        self.identity = {**(identity or {}), "template": adapter.name, "presentation": presentation_identity()}
        self.accelerated = accelerated
        self.batch_inputs = batch_inputs or getattr(scorer, "inputs", 8 if accelerated else 1)

    def tokenizer_identity(self):
        from .errors import ConfigError
        if not hasattr(self.scorer, 'tokenizer_identity'):
            raise ConfigError('native training export requires a scorer tokenizer identity')
        return self.scorer.tokenizer_identity()

    @property
    def max_slots(self) -> int:
        from .presentation import LETTERS
        return min(len(LETTERS), self.adapter.max_slots)

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]:
        with self.scorer.operation() if hasattr(self.scorer, "operation") else nullcontext():
            return self._score_many(items)

    def _requests(self, items):
        requests, positions, tokens = [], [], [0] * len(items)
        for n, (state, questions) in enumerate(items):
            state, groups = self.adapter.state(state), OrderedDict()
            for k, question in enumerate(questions):
                groups.setdefault((question.guidance.digest, question.framing, question.catalog), []).append((k, question))
            for group in groups.values():
                request = self.adapter.request(state, [q for _, q in group])
                context = getattr(self.scorer, "ctx", None)
                if len(request.prefix) > self.max_prefix_tokens:
                    raise ConfigError(f"prefix requires {len(request.prefix)} tokens; prefix limit {self.max_prefix_tokens}")
                if context is not None and any(q.guidance or q.catalog for _, q in group):
                    static = len(request.prefix) + max(map(len, request.suffixes), default=0)
                    if static + 1 > context.sequence_limit:
                        ids = [section.id for section in group[0][1].guidance.sections] if group[0][1].guidance else ["criteria catalog"]
                        raise ConfigError(f"guidance {ids}: prefix/question requires {static} tokens; per-sequence context capacity {context.sequence_limit}")
                requests.append(request)
                positions.append([(n, k) for k, _ in group])
                tokens[n] += len(request.state) + sum(map(len, request.suffixes))
                if group[0][1].guidance or group[0][1].catalog:
                    tokens[n] += len(request.prefix)
        return requests, positions, tokens

    def preview(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[dict]:
        from .presentation import messages
        requests, positions, _ = self._requests(items)
        return [{"representation": "local", "input": positions[r][0][0], "questions": [k for _, k in positions[r]],
                 "prompts": [messages(items[n][0], items[n][1][k]) for n, k in positions[r]],
                 "prefix": list(request.prefix), "state": list(request.state), "suffixes": [list(s) for s in request.suffixes],
                 "slots": [[list(group) for group in slots] for slots in request.slots]}
                for r, request in enumerate(requests)]

    def _score_many(self, items):
        context = getattr(self.scorer, "ctx", None)
        before = {key: getattr(context, key, 0) for key in ("submissions", "submitted_tokens")}
        scores = [[None for _ in qs] for _, qs in items]
        requests, positions, tokens = self._requests(items)
        for request, at in zip(requests, positions):
            self._advise(items[at[0][0]][1][at[0][1]].guidance)
        limits = [[_limit(req, k, self.max_input_tokens) for k in range(len(req.suffixes))] for req in requests]
        for r, row in enumerate(limits):
            for k, score in enumerate(row):
                n, original = positions[r][k]
                scores[n][original] = score
        live = [(r, k) for r, row in enumerate(limits) for k, score in enumerate(row) if score is None]
        live.sort(key=lambda pair: requests[pair[0]].prefix)
        for batch in _batches(requests, live):
            checkpoint()
            parts = [ScoreRequest(requests[r].prefix, requests[r].state,
                                  tuple(requests[r].suffixes[k] for k in ks), tuple(requests[r].slots[k] for k in ks))
                     for r, ks in batch]
            block, row = self.scorer.score(parts), 0
            for r, ks in batch:
                for k in ks:
                    n, original = positions[r][k]
                    scores[n][original] = _scores(block, row, items[n][1][original].width)
                    row += 1
        work = {key: getattr(context, key, 0) - before[key] for key in ("submissions", "submitted_tokens")}
        return [Scored(tuple(row), count, native_work=tuple(work.items()) if context is not None else ())
                for row, count in zip(scores, tokens)]

    def _advise(self, guidance):
        profile = self.quality_profile
        if not profile or not guidance or guidance.digest in self._advised:
            return
        from .api import behavior_identity
        context = getattr(self.scorer, "ctx", None)
        configuration = {"max_input_tokens": self.max_input_tokens, "n_ctx": getattr(context, "n_ctx", None),
                         "n_batch": getattr(context, "n_batch", None), "prefix_slots": getattr(self.scorer, "prefix_slots", None)}
        if profile.get("judge") != behavior_identity(self.identity) or profile.get("configuration") != configuration:
            return
        self._advised[guidance.digest] = None
        if len(self._advised) > ADVISED:
            self._advised.popitem(last=False)
        received = len(self.adapter.content(guidance.text))
        maximum = profile["max_guidance_tokens"]
        if received > maximum:
            import warnings
            warnings.warn(f"guidance is {received} tokens; tested through {maximum} tokens on {profile['cohort']}; "
                          f"quality outside that range is unmeasured (evidence: {profile['evidence']})", stacklevel=3)

    def score(self, state: str, questions: Sequence[Question]) -> list[Scores]:
        return list(self.score_many([(state, questions)])[0].scores)

    def selftest(self) -> dict:
        with self.scorer.operation() if hasattr(self.scorer, "operation") else nullcontext():
            return self._selftest()

    def _selftest(self):
        checks = dict(self.scorer.selftest(self.adapter, CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS)) \
            if hasattr(self.scorer, "selftest") else {}
        short, long_ = CONFORMANCE_QUESTIONS[0], CONFORMANCE_QUESTIONS[1]
        initial = self.score(CONFORMANCE_PROBES[1], [short])[0]
        if initial.error:
            return {**checks, "D2 batching": {"ok": False, "error": initial.error}}
        alone = initial.probs
        self.score(CONFORMANCE_PROBES[0], [long_])
        after = self.score(CONFORMANCE_PROBES[1], [short])[0]
        batched = self.score(CONFORMANCE_PROBES[1], [long_, short])[1]
        if after.error or batched.error:
            checks['D2 batching'] = {'ok': False, 'error': after.error or batched.error}
        else:
            checks["D2 batching"] = {"ok": bool(np.array_equal(alone, after.probs) and np.array_equal(alone, batched.probs)
                                                and initial.mass == after.mass == batched.mass),
                                     "after_diff": float(np.abs(alone - after.probs).max()),
                                     "batched_diff": float(np.abs(alone - batched.probs).max()), 'includes_mass': True}
        ids = [t for group in self.adapter.letters for t in group]
        checks["D4 answer tokens"] = {"ok": all(len(slots) == len(set(slots)) for slots in
                                                (ids, *([t for group in pair for t in group] for pair in (self.adapter.binary, self.adapter.polar))))}
        masses = [s.mass for text in CONFORMANCE_PROBES for s in self.score(text, CONFORMANCE_QUESTIONS) if s.mass is not None]
        median = statistics.median(masses) if masses else None
        checks["D5 template"] = {"ok": median is None or median >= 0.8, "median_mass": median}
        return dict(sorted(checks.items()))
