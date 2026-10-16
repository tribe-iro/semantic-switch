from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
import math


@dataclass(slots=True)
class _Request:
    items: tuple
    future: Future
    questions: int
    characters: int
    submitted: float
    id: int


class BatchingJudge:
    def __init__(self, judge, *, wait_ms=1.0, max_pending=256, max_inputs=8,
                 max_questions=64, max_chars=16384, own=False, enqueue_timeout=0.):
        for name, value in (("max_pending", max_pending), ("max_inputs", max_inputs),
                            ("max_questions", max_questions), ("max_chars", max_chars)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(enqueue_timeout, bool) or not math.isfinite(enqueue_timeout) or enqueue_timeout < 0:
            raise ValueError("enqueue_timeout must be a finite nonnegative duration")
        if not math.isfinite(wait_ms) or not 0 <= wait_ms <= 1000:
            raise ValueError("wait_ms must be within [0, 1000]")
        self.judge, self.own = judge, own
        self.wait = wait_ms / 1000
        self.limits = max_inputs, max_questions, max_chars
        self.max_pending = max_pending
        self.enqueue_timeout = enqueue_timeout
        self._next_id = 0
        self._metrics = deque(maxlen=4096)
        self._condition = threading.Condition()
        self._queue = deque()
        self._closed = False
        self._worker = None

    @property
    def identity(self):
        return self.judge.identity

    @property
    def accelerated(self):
        return self.judge.accelerated

    @property
    def max_slots(self):
        return self.judge.max_slots

    @property
    def batch_inputs(self):
        return self.limits[0]

    @property
    def supports_guidance(self):
        return getattr(self.judge, "supports_guidance", False)

    @staticmethod
    def _characters(question):
        return len(question.text) + sum(map(len, question.options)) + sum(map(len, question.excludes)) + len(question.guidance.text)

    def submit(self, items):
        items = tuple((state, tuple(questions)) for state, questions in items)
        questions = sum(len(qs) for _, qs in items)
        characters = sum(len(state) + sum(self._characters(q) for q in qs) for state, qs in items)
        if any(value > limit for value, limit in zip((len(items), questions, characters), self.limits)):
            raise ValueError("request exceeds batching admission limits")
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("batching judge is closed")
            deadline = time.monotonic() + self.enqueue_timeout
            while len(self._queue) >= self.max_pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("batching queue is full")
                self._condition.wait(remaining)
                if self._closed:
                    raise RuntimeError("batching judge is closed")
            self._next_id += 1
            self._queue.append(_Request(items, future, questions, characters, time.monotonic(), self._next_id))
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="semantic-batching", daemon=True)
                self._worker.start()
            self._condition.notify()
        return future

    def score_many(self, items):
        if threading.current_thread() is self._worker:
            raise RuntimeError("recursive submission from the serving worker")
        if not items:
            return []
        from semantic_switch.judge import Scored
        jobs, mappings = [], []
        for n, (state, questions) in enumerate(items):
            group, characters = [], len(state)
            for question in questions:
                cost = self._characters(question)
                if len(state) + cost > self.limits[2]:
                    raise ValueError("single question exceeds character admission bound")
                if group and (len(group) >= self.limits[1] or characters + cost > self.limits[2]):
                    jobs.append((state, tuple(group)))
                    mappings.append(n)
                    group, characters = [], len(state)
                group.append(question)
                characters += cost
            if group:
                jobs.append((state, tuple(group)))
                mappings.append(n)
        futures = [self.submit([job]) for job in jobs]
        scores, tokens, models = [[] for _ in items], [0 for _ in items], [None for _ in items]
        for n, future in zip(mappings, futures):
            result = future.result()[0]
            scores[n].extend(result.scores)
            tokens[n] += result.tokens
            models[n] = result.model or models[n]
        return [Scored(tuple(row), count, model) for row, count, model in zip(scores, tokens, models)]

    async def ascore_many(self, items):
        import asyncio
        return await asyncio.to_thread(self.score_many, items)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue:
                    return
                deadline = self._queue[0].submitted + self.wait
                while not self._closed and time.monotonic() < deadline:
                    totals = (sum(len(request.items) for request in self._queue),
                              sum(request.questions for request in self._queue), sum(request.characters for request in self._queue))
                    if any(value >= limit for value, limit in zip(totals, self.limits)):
                        break
                    self._condition.wait(max(0., deadline - time.monotonic()))
                batch, sizes = [], [0, 0, 0]
                while self._queue:
                    request = self._queue[0]
                    if request.future.cancelled():
                        self._queue.popleft()
                        continue
                    cost = len(request.items), request.questions, request.characters
                    if batch and any(a + b > limit for a, b, limit in zip(sizes, cost, self.limits)):
                        break
                    self._queue.popleft()
                    if request.future.set_running_or_notify_cancel():
                        batch.append(request)
                        sizes = [a + b for a, b in zip(sizes, cost)]
                self._condition.notify_all()
            if not batch:
                continue
            started = time.monotonic()
            try:
                results = self.judge.score_many([item for request in batch for item in request.items])
                if len(results) != sizes[0]:
                    raise TypeError("batched judge returned an incorrect number of results")
                offset = 0
                for request in batch:
                    count = len(request.items)
                    with self._condition:
                        self._metrics.append({"id": request.id, "inputs": count,
                                              "queue_ms": (started - request.submitted) * 1000,
                                              "service_ms": (time.monotonic() - started) * 1000})
                    request.future.set_result(results[offset:offset + count])
                    offset += count
            except BaseException as error:
                for request in batch:
                    request.future.set_exception(error)

    def metrics(self):
        with self._condition:
            return tuple(dict(row) for row in self._metrics)

    def close(self, *, cancel_pending=False):
        with self._condition:
            self._closed = True
            if cancel_pending:
                for request in self._queue:
                    request.future.cancel()
                self._queue.clear()
            self._condition.notify_all()
            worker = self._worker
        if worker:
            worker.join()
        if self.own:
            self.judge.close()
            self.own = False
