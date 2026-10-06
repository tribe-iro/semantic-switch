from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .protocols import Question, ScoreBlock, ScoreRequest

MIN_MASS = 0.5
MAX_STATE_TOKENS = 1024
MAX_QUESTION_TOKENS = 1536
BATCH_QUESTIONS = 64
BATCH_TOKENS = 16384

FIXTURE = (
    "subject: Double charge\nbody: I was charged twice in May, please refund one of them.",
    "subject: Router dead\nbody: My router stopped working yesterday and nothing blinks. I need it today.",
    "subject: Final notice\nbody: If this is not fixed by Monday my lawyer will file a lawsuit against you.",
)
FIXTURE_QUESTIONS = (
    Question("check", "The customer asks to receive money back for a charge."),
    Question("pick", "What is the message mainly about?",
             ("charges, invoices or payments", "the product is broken, erroring or unavailable",
              "login, profile or account access")),
    Question("rate", "How urgent is the customer's situation?", ("can wait", "today", "immediately")),
    Question("check", "The customer threatens legal action against us."),
)


@dataclass(frozen=True, slots=True)
class Scores:
    probs: np.ndarray | None
    mass: float | None
    error: str | None = None

    @property
    def low_mass(self) -> bool:
        return self.mass is not None and self.mass < MIN_MASS


@dataclass(frozen=True, slots=True)
class Scored:
    scores: tuple[Scores, ...]
    tokens: int


def _limit(req: ScoreRequest, k: int) -> Scores | None:
    if len(req.state) > MAX_STATE_TOKENS:
        return Scores(None, None, f"limit: input is {len(req.state)} tokens (max {MAX_STATE_TOKENS})")
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
    return Scores(p / p.sum(), None if np.isnan(lm) else float(np.exp(lm)))


def scored(judge, items: Sequence[tuple[str, Sequence[Question]]]) -> list:
    results = judge.score_many(items)
    name = judge.identity.get("file") or judge.identity.get("source") or type(judge).__name__
    if len(results) != len(items):
        raise TypeError(f"judge {name} returned {len(results)} results for {len(items)} inputs")
    for n, ((_, questions), result) in enumerate(zip(items, results)):
        if len(result.scores) != len(questions):
            raise TypeError(f"judge {name}, input {n}: {len(result.scores)} scores for {len(questions)} questions")
        for k, (q, sc) in enumerate(zip(questions, result.scores)):
            if sc.error is not None:
                continue
            p = np.asarray(sc.probs, dtype=np.float64)
            if p.shape != (q.width,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 1e-3:
                raise TypeError(f"judge {name}, input {n}, question {k}: probabilities must be {q.width} finite "
                                f"non-negative values summing to 1, got {p.tolist()}")
            if sc.mass is not None and not 0 <= sc.mass <= 1:
                raise TypeError(f"judge {name}, input {n}, question {k}: answer mass must be None or within [0, 1], "
                                f"got {sc.mass}")
    return results


class LlamaJudge:
    def __init__(self, scorer, adapter, identity: dict | None = None, accelerated: bool = False):
        self.scorer = scorer
        self.adapter = adapter
        self.identity = {**(identity or {}), "template": adapter.name}
        self.accelerated = accelerated

    @property
    def max_slots(self) -> int:
        return self.adapter.max_slots

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]:
        requests = [self.adapter.request(self.adapter.state(state), qs) for state, qs in items]
        scores = [[_limit(req, k) for k in range(len(req.suffixes))] for req in requests]
        live = [(n, k) for n, row in enumerate(scores) for k, s in enumerate(row) if s is None]
        for batch in _batches(requests, live):
            parts = [ScoreRequest(requests[n].prefix, requests[n].state,
                                  tuple(requests[n].suffixes[k] for k in ks), tuple(requests[n].slots[k] for k in ks))
                     for n, ks in batch]
            block, row = self.scorer.score(parts), 0
            for n, ks in batch:
                for k in ks:
                    scores[n][k] = _scores(block, row, items[n][1][k].width)
                    row += 1
        return [Scored(tuple(row), len(req.state) + sum(map(len, req.suffixes))) for req, row in zip(requests, scores)]

    def score(self, state: str, questions: Sequence[Question]) -> list[Scores]:
        return list(self.score_many([(state, questions)])[0].scores)

    def fit(self, text: str, limit: int) -> tuple[str, bool]:
        toks = self.scorer.tokenize(text)
        return (text, False) if len(toks) <= limit else (self.scorer.detokenize(toks[:limit]), True)

    def selftest(self) -> dict:
        checks = dict(self.scorer.selftest(self.adapter, FIXTURE, FIXTURE_QUESTIONS)) \
            if hasattr(self.scorer, "selftest") else {}
        short, long_ = FIXTURE_QUESTIONS[0], FIXTURE_QUESTIONS[1]
        alone = self.score(FIXTURE[1], [short])[0].probs
        self.score(FIXTURE[0], [long_])
        after = self.score(FIXTURE[1], [short])[0].probs
        batched = self.score(FIXTURE[1], [long_, short])[1].probs
        d_after, d_batch = float(np.abs(alone - after).max()), float(np.abs(alone - batched).max())
        checks["D2 batching"] = {"ok": d_after < 1e-3 and d_batch < 0.15, "after_diff": d_after,
                                 "batched_diff": d_batch}
        ids = [t for group in self.adapter.letters for t in group]
        binary = [t for group in self.adapter.binary for t in group]
        checks["D4 answer tokens"] = {"ok": len(ids) == len(set(ids)) and len(binary) == len(set(binary))}
        masses = [s.mass for text in FIXTURE for s in self.score(text, FIXTURE_QUESTIONS) if s.mass is not None]
        median = statistics.median(masses) if masses else None
        checks["D5 template"] = {"ok": median is None or median >= 0.8, "median_mass": median}
        return dict(sorted(checks.items()))
