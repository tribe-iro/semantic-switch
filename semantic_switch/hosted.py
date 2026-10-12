from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from email.utils import parsedate_to_datetime
import json
import math
import os
import random
import threading
import time
import uuid
import warnings
from typing import Awaitable, Callable, Sequence

import numpy as np

from .errors import ConfigError
from .judge import OPERATION, Cancelled, Request, Scored, Scores
from .protocols import Question

try:
    import anyio
except ImportError:
    anyio = None

OPENROUTER = "https://openrouter.ai/api/v1"
STRUCTURED = frozenset({"typesafe/jev-1.13", "openai/gpt-6-luna-decisions"})
TOP_LOGPROBS = 20
IN_FLIGHT = 2
ATTEMPTS = 5


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Reply:
    status: int
    body: object = None
    retry_after: str | None = None


def unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate response field {name!r}")
        result[name] = value
    return result


def hosted_dependencies():
    if anyio is None:
        raise ConfigError("hosted judges require semantic-switch[hosted]")
    try:
        import httpx
    except ImportError:
        raise ConfigError("hosted judges require semantic-switch[hosted]") from None
    return httpx


class Wire:
    def __init__(self, httpx, key: str, connections: int, max_bytes: int):
        self.httpx, self.headers, self.connections, self.max_bytes = httpx, {"Authorization": f"Bearer {key}"}, connections, max_bytes
        self.client = None

    async def __call__(self, url: str, body: dict, remaining: float) -> Reply:
        httpx = self.httpx
        if self.client is None:
            self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=self.connections, max_keepalive_connections=self.connections))
        timeout = httpx.Timeout(remaining, connect=min(10.0, remaining))
        try:
            async with self.client.stream("POST", url, json=body, headers=self.headers, timeout=timeout) as response:
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise TransportError(f"response body exceeds {self.max_bytes} bytes")
                    chunks.append(chunk)
        except httpx.TimeoutException as error:
            raise TimeoutError(type(error).__name__) from error
        except httpx.TransportError as error:
            raise ConnectionError(type(error).__name__) from error
        content, hint = b"".join(chunks), response.headers.get("retry-after")
        if response.is_success:
            try:
                return Reply(response.status_code, json.loads(content, object_pairs_hook=unique), hint)
            except (ValueError, UnicodeError) as error:
                raise TransportError(f"invalid response JSON: {error}") from error
        try:
            return Reply(response.status_code, json.loads(content), hint)
        except (ValueError, UnicodeError):
            return Reply(response.status_code, None, hint)

    async def close(self):
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()


def http(key: str | None = None, connections: int = 8, max_bytes: int = 1 << 20) -> Wire:
    httpx = hosted_dependencies()
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ConfigError("OpenRouter judges need an API key: set OPENROUTER_API_KEY or pass key=...")
    return Wire(httpx, key, connections, max_bytes)


def retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, IndexError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def retryable(status: int) -> bool:
    return status in (408, 429) or status >= 500


def instructions(q: Question) -> str:
    from .presentation import criterion
    return "\n".join([*([f"Guidance:\n{q.guidance.text}\n\nCriterion:"] if q.guidance else []), criterion(q), *(f"Except: {x}." for x in q.excludes)])


ROUNDING = 0.005


def probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be a number")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and within [0, 1]")
    return value


def distribution(q: Question, probs: dict) -> np.ndarray:
    if set(probs) != set(range(q.width)):
        raise ValueError("probabilities must cover each option exactly once")
    p = np.array([probability(probs[k]) for k in range(q.width)])
    if abs(float(p.sum()) - 1) > max(1e-3, ROUNDING * q.width):
        raise ValueError("probabilities must sum to one")
    return p / p.sum()


def level(value) -> int:
    if not isinstance(value, str) or not value.isdigit() or str(int(value)) != value:
        raise ValueError("level keys must be canonical integer indices")
    return int(value)


def confidence(answer):
    value = answer.get("confidence")
    return None if value is None else probability(value)


class Chat:
    name, path, joint, response_bytes = "chat-logprobs", "chat/completions", False, 1 << 18
    routing = {"require_parameters": True}
    identity = {"scoring": f"answer-token-lower-bound@2:top{TOP_LOGPROBS}", "reasoning": "disabled"}

    @staticmethod
    def validate(questions: Sequence[Question]):
        return None

    @staticmethod
    def body(state: str, questions: Sequence[Question]) -> dict:
        from .presentation import messages
        system, user = messages(state, *questions)
        return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 1, "logprobs": True, "top_logprobs": TOP_LOGPROBS, "reasoning": {"enabled": False}}

    @staticmethod
    def scores(response: dict, questions: Sequence[Question]) -> tuple[Scores, ...]:
        if ((response.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"):
            raise ValueError("reasoning tokens were billed despite disabled reasoning")
        return (token_scores(*questions, response),)


class SystemOne:
    name, path, joint, response_bytes = "systemone", "systemone", True, 1 << 22
    routing = {}
    identity = {}

    @staticmethod
    def validate(questions: Sequence[Question]):
        for question in questions:
            if question.kind not in ("check", "rate", "pick"):
                raise ConfigError(f"unsupported System One question kind {question.kind!r}")
            if question.kind == "rate" and not 2 <= len(question.options) <= 10:
                raise ConfigError(f"System One scores take 2 to 10 levels, got {len(question.options)}; use a local judge or an openrouter chat model for this Rate")

    @staticmethod
    def body(state: str, questions: Sequence[Question]) -> dict:
        out = {}
        for k, question in enumerate(questions):
            row = {"instructions": instructions(question)}
            if question.kind == "check":
                row["type"] = "noul"
            elif question.kind == "rate":
                row.update(type="score", criteria=list(question.options))
            else:
                row.update(type="choice", criteria={chr(65 + i): option for i, option in enumerate(question.options)})
            out[f"q{k}"] = row
        return {"state": state, "questions": out}

    @staticmethod
    def scores(response: dict, questions: Sequence[Question]) -> tuple[Scores, ...]:
        if response.get("refusal"):
            return tuple(Scores(None, None, f"hosted refusal: {response['refusal']}") for _ in questions)
        answers = response["answers"]
        if not isinstance(answers, dict) or set(answers) - {f"q{k}" for k in range(len(questions))}:
            raise ValueError("answers must be an object keyed by the requested question names")
        return tuple(systemone_answer(question, answers.get(f"q{k}")) for k, question in enumerate(questions))


def systemone_answer(question: Question, answer) -> Scores:
    try:
        if not isinstance(answer, dict):
            raise ValueError("expected an object")
        if answer.get("refusal") or answer.get("type") == "refusal":
            return Scores(None, None, f"hosted refusal: {answer.get('refusal', 'refused')}")
        expected = {"check": "noul", "rate": "score", "pick": "choice"}[question.kind]
        if "type" in answer and answer["type"] != expected:
            raise ValueError("answer type differs from requested question")
        if question.kind == "check":
            p = probability(answer["noul"])
            return Scores(np.array([p, 1 - p]), None, None, confidence(answer))
        probabilities = answer["probabilities"]
        if not isinstance(probabilities, dict):
            raise ValueError("probabilities must be an object")
        if expected == "score":
            probabilities = {level(key): value for key, value in probabilities.items()}
        elif any(not isinstance(letter, str) or len(letter) != 1 or not "A" <= letter <= "T" for letter in probabilities):
            raise ValueError("probabilities must be a mapping of answer letters")
        else:
            probabilities = {ord(letter) - 65: value for letter, value in probabilities.items()}
        return Scores(distribution(question, probabilities), None, None, confidence(answer))
    except (KeyError, TypeError, ValueError) as error:
        return Scores(None, None, f"hosted: malformed answer ({error})")


def token_scores(question: Question, response: dict) -> Scores:
    from .presentation import answers
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
    words = tuple(answer.lower() for answer in answers(question))
    mass = np.zeros(len(words))
    for token, logprob in found.items():
        if token.strip().lower() in words:
            mass[words.index(token.strip().lower())] += math.exp(logprob)
    reported, answered = math.fsum(map(math.exp, found.values())), float(mass.sum())
    if reported > 1 + 1e-3:
        raise ValueError("reported token mass exceeds one")
    if answered <= 0:
        raise ValueError("no answer token among the returned alternatives")
    return Scores(mass / (answered + max(0., 1 - reported)), min(answered, 1.))


def merged(results: Sequence[Scored]) -> Scored:
    models = {result.model for result in results} - {None}
    return Scored(tuple(replace(score, model=score.model or result.model) for result in results for score in result.scores),
                  sum(result.tokens for result in results), next(iter(models)) if len(models) == 1 else None)


def leaf(failure: BaseException) -> BaseException:
    while isinstance(failure, BaseExceptionGroup):
        failure = next((error for error in failure.exceptions if not isinstance(error, anyio.get_cancelled_exc_class())), failure.exceptions[0])
    return failure


@dataclass(slots=True)
class Progress:
    started: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    status: int | None = None
    usage: dict | None = None
    service: float | None = None
    retry: float = 0.

    def request(self, now: float) -> Request:
        queued = (now if self.service is None else self.service) - self.started
        return Request(self.id, now - self.started, self.attempts, self.status, self.usage, queued,
                       0. if self.service is None else now - self.service, self.retry)


class OpenRouterJudge:
    source = "openrouter"
    supports_guidance = True
    accelerated = True
    max_slots = 20

    def __init__(self, model: str, *, provider: str | None = None, key: str | None = None,
                 transport: Callable[[str, dict, float], Awaitable[Reply]] | None = None, own_transport: bool = False,
                 timeout: float = 60.0, batch_inputs: int = 8,
                 max_response_bytes: int | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] | None = None):
        if isinstance(batch_inputs, bool) or not isinstance(batch_inputs, int) or batch_inputs < 1:
            raise ValueError("batch_inputs must be a positive integer")
        if anyio is None:
            raise ConfigError("hosted judges require semantic-switch[hosted]")
        from .presentation import identity as presentation_identity
        self.model, self.provider, self.timeout, self.batch_inputs = model, provider, timeout, batch_inputs
        self.protocol = SystemOne if model in STRUCTURED else Chat
        self.max_response_bytes = self.protocol.response_bytes if max_response_bytes is None else max_response_bytes
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        self.url = f"{OPENROUTER}/{self.protocol.path}"
        self.send = transport if transport is not None else http(key, batch_inputs, self.max_response_bytes)
        self.own_transport = transport is None or own_transport
        self.clock, self.sleep = clock, sleep or anyio.sleep
        self.identity = {"source": self.source, "model": model, "presentation": presentation_identity(), "endpoint": self.url,
                         "protocol": self.protocol.name, "provider": provider,
                         "question_scope": "joint" if self.protocol.joint else "individual", **self.protocol.identity}
        self.returned: str | None = None
        self.observed_models = set()
        self._lock = threading.Lock()
        self._closed = False
        self._portal = None
        self._scopes = set()
        self._drained = None

    def payload(self, state: str, questions: Sequence[Question]) -> dict:
        return {"model": self.model, **self.protocol.body(state, questions),
                "provider": {**({"only": [self.provider]} if self.provider else {}), "allow_fallbacks": False, **self.protocol.routing}}

    def _units(self, items):
        for owner, (_, questions) in enumerate(items):
            if self.protocol.joint:
                if questions:
                    yield owner, tuple(questions)
            else:
                yield from ((owner, (question,)) for question in questions)

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]:
        from .presentation import validate_question
        items = [(state, tuple(questions)) for state, questions in items]
        for _, questions in items:
            for question in questions:
                validate_question(self, question)
            self.protocol.validate(questions)
        portal = self._bound()
        try:
            future = portal.start_task_soon(self._score_many, items)
        except RuntimeError:
            raise RuntimeError("judge is closed") from None
        handle = OPERATION.get()
        release = handle.register(future.cancel) if handle is not None else (lambda: None)
        try:
            results = future.result()
        except CancelledError:
            if self._closed:
                raise RuntimeError("judge is closed") from None
            raise Cancelled("operation cancelled") from None
        finally:
            release()
        models = sorted({model for result in results for model in (result.model, *(score.model for score in result.scores)) if model is not None})
        if len(models) > 1:
            warnings.warn(f"hosted operation returned mixed model versions: {models}", stacklevel=2)
        with self._lock:
            self.observed_models.update(models)
            if len(models) == 1:
                self.returned = models[0]
        return results

    def _bound(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("judge is closed")
            if self._portal is None:
                manager = anyio.from_thread.start_blocking_portal()
                portal = manager.__enter__()
                try:
                    self._admission, self._active, self._drained = portal.call(self._primitives)
                except BaseException:
                    manager.__exit__(None, None, None)
                    raise
                self._portal = manager, portal
            return self._portal[1]

    async def _primitives(self):
        return anyio.Semaphore(IN_FLIGHT * self.batch_inputs), anyio.Semaphore(self.batch_inputs), anyio.Event()

    async def _score_many(self, items) -> list[Scored]:
        if self._closed:
            raise RuntimeError("judge is closed")
        units, results = [], []
        with anyio.CancelScope() as operation:
            self._scopes.add(operation)
            try:
                async with anyio.create_task_group() as group:
                    for owner, questions in self._units(items):
                        progress, limit = Progress(self.clock()), anyio.current_time() + self.timeout
                        units.append(owner)
                        results.append(None)
                        with anyio.CancelScope(deadline=limit) as admission:
                            await self._admission.acquire()
                        if admission.cancelled_caught:
                            results[-1] = self._failed(questions, "deadline exceeded awaiting admission", progress)
                            continue
                        group.start_soon(self._unit, results, len(results) - 1, items[owner][0], questions, progress, limit)
            except BaseExceptionGroup as failure:
                raise leaf(failure) from None
            finally:
                self._scopes.discard(operation)
                if self._closed and not self._scopes:
                    self._drained.set()
        if operation.cancelled_caught:
            raise RuntimeError("judge is closed")
        groups = [[] for _ in items]
        for owner, result in zip(units, results):
            groups[owner].append(result)
        return [merged(group) for group in groups]

    async def _unit(self, results, index, state, questions, progress: Progress, limit: float):
        try:
            body = self.payload(state, questions)
            with anyio.CancelScope(deadline=limit) as unit:
                async with self._active:
                    progress.service = self.clock()
                    results[index] = await self._score(body, questions, progress)
            if unit.cancelled_caught:
                results[index] = self._failed(questions, f"deadline of {self.timeout:g}s exceeded after {progress.attempts} attempts", progress)
        finally:
            self._admission.release()

    def _failed(self, questions, message, progress: Progress, model=None, tokens=0) -> Scored:
        request = progress.request(self.clock())
        return Scored(tuple(Scores(None, None, f"hosted: {message}", model=model, request=request) for _ in questions), tokens, model)

    async def _score(self, body: dict, questions: Sequence[Question], progress: Progress) -> Scored:
        deadline = progress.started + self.timeout
        response, error = await self._exchange(body, deadline, progress)
        if error:
            return self._failed(questions, error, progress)
        model, provider, usage = response.get("model"), response.get("provider"), response.get("usage")
        if not isinstance(model, str) or provider is not None and not isinstance(provider, str) or usage is not None and not isinstance(usage, dict):
            return self._failed(questions, "malformed response (returned model, provider or usage)", progress)
        returned = f"{model}@{provider}" if provider else model
        tokens = (usage or {}).get("prompt_tokens", (usage or {}).get("input_tokens", 0))
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            return self._failed(questions, "malformed response (prompt tokens must be a nonnegative integer)", progress, returned)
        if self.provider and provider != self.provider:
            return self._failed(questions, f"provider route changed to {provider!r}", progress, returned, tokens)
        try:
            scores = self.protocol.scores(response, questions)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return self._failed(questions, f"malformed response ({error})", progress, returned, tokens)
        if self.clock() >= deadline:
            return self._failed(questions, f"response completed after the {self.timeout:g}s deadline", progress, returned, tokens)
        request = progress.request(self.clock())
        return Scored(tuple(replace(score, model=returned, request=request) for score in scores), tokens, returned)

    async def _exchange(self, body: dict, deadline: float, progress: Progress) -> tuple[dict | None, str | None]:
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                return None, f"deadline of {self.timeout:g}s exceeded after {progress.attempts} attempts"
            progress.attempts += 1
            try:
                reply = await self.send(self.url, body, remaining)
            except TransportError as error:
                return None, str(error)
            except (ConnectionError, TimeoutError) as error:
                progress.status, hint, failure = None, None, f"connection failed ({type(error).__name__})"
            else:
                progress.status = reply.status
                if 200 <= reply.status < 300:
                    if not isinstance(reply.body, dict):
                        return None, "response must be a JSON object"
                    progress.usage = reply.body.get("usage") if isinstance(reply.body.get("usage"), dict) else None
                    if self.clock() >= deadline:
                        return None, f"response completed after the {self.timeout:g}s deadline"
                    return reply.body, None
                if not retryable(reply.status):
                    return None, self.rejected(reply)
                hint, failure = retry_after(reply.retry_after), f"HTTP {reply.status}"
            if progress.attempts == ATTEMPTS:
                return None, f"{failure} after {ATTEMPTS} attempts"
            wait = hint if hint is not None else 2 ** (progress.attempts - 1) * random.uniform(.5, 1)
            if self.clock() + wait >= deadline:
                return None, f"{failure}; retry after {wait:.1f}s would exceed the {self.timeout:g}s deadline"
            before = self.clock()
            try:
                await self.sleep(wait)
            finally:
                progress.retry += self.clock() - before

    def rejected(self, reply: Reply) -> str:
        error = reply.body.get("error") if isinstance(reply.body, dict) else None
        error = error if isinstance(error, dict) else {}
        message = error.get("message")[:300] if isinstance(error.get("message"), str) else None
        if reply.status != 400:
            return f"HTTP {reply.status}" + (f": {message}" if message else "")
        metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
        named = metadata.get("provider_name") if isinstance(metadata.get("provider_name"), str) else self.provider
        return ((f"provider {named}" if named else "the routed provider") + " rejected the request shape (HTTP 400" + (f": {message}" if message else "") +
                f"); pin a provider that accepts it with provider=PROVIDER or --provider PROVIDER")

    async def _shutdown(self):
        for scope in list(self._scopes):
            scope.cancel()
        if self._scopes:
            await self._drained.wait()
        if self.own_transport:
            await self.send.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            bound, self._portal = self._portal, None
        if bound is None:
            if self.own_transport:
                anyio.run(self.send.close)
            return
        manager, portal = bound
        try:
            portal.call(self._shutdown)
        finally:
            manager.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
