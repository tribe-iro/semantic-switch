from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
import inspect
import json
import math
import random
import threading
import time
import uuid
import warnings
from typing import TYPE_CHECKING, Awaitable, Callable, Mapping, Protocol, Sequence

from .errors import ConfigError
from .judge import OPERATION, Cancelled, Request, Scored, Scores
from .protocols import Question

if TYPE_CHECKING:
    from .drivers.profiles import ModelProfile

try:
    import anyio
except ImportError:
    anyio = None

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


def serialized(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def hosted_dependencies():
    if anyio is None:
        raise ConfigError("hosted judges require semantic-switch[hosted]")
    try:
        import httpx
    except ImportError:
        raise ConfigError("hosted judges require semantic-switch[hosted]") from None
    return httpx


class Wire:
    def __init__(self, httpx, headers: dict, connections: int, max_bytes: int):
        self.httpx, self.headers, self.connections, self.max_bytes = httpx, {**headers, "Content-Type": "application/json"}, connections, max_bytes
        self.client = None

    async def __call__(self, url: str, body: bytes, remaining: float) -> Reply:
        httpx = self.httpx
        if self.client is None:
            self.client = httpx.AsyncClient(limits=httpx.Limits(max_connections=self.connections, max_keepalive_connections=self.connections))
        timeout = httpx.Timeout(remaining, connect=min(10.0, remaining))
        try:
            async with self.client.stream("POST", url, content=body, headers=self.headers, timeout=timeout) as response:
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


@dataclass(frozen=True, slots=True)
class Envelope:
    returned: str
    route: str | None
    tokens: int
    payload: dict


class Provider(Protocol):
    name: str
    revision: int
    route: str | None
    identity: dict

    def endpoint(self, codec) -> str: ...

    def wire(self, connections: int, max_bytes: int) -> Wire: ...

    def envelope(self, model: str, codec, body: dict) -> dict: ...

    def usage(self, body: dict) -> dict | None: ...

    def unwrap(self, body: dict) -> Envelope: ...

    def rejected(self, reply: Reply) -> str: ...


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


def leaf(failure: BaseException) -> BaseException:
    while isinstance(failure, BaseExceptionGroup):
        failure = next((error for error in failure.exceptions if not isinstance(error, anyio.get_cancelled_exc_class())), failure.exceptions[0])
    return failure


@dataclass(frozen=True, slots=True)
class Unit:
    owner: int
    index: tuple[int, ...]
    state: str
    questions: tuple[Question, ...]


@dataclass(frozen=True, slots=True)
class Completion:
    owner: int
    index: tuple[int, ...]
    scored: Scored


@dataclass(slots=True)
class Progress:
    started: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    attempts: int = 0
    status: int | None = None
    usage: dict | None = None
    queued: float = 0.
    retry: float = 0.
    admission: dict | None = None

    def request(self, now: float) -> Request:
        seconds = now - self.started
        return Request(self.id, seconds, self.attempts, self.status, self.usage, self.queued, seconds - self.queued - self.retry, self.retry,
                       self.admission)


class HostedJudge:
    supports_guidance = True
    accelerated = True

    def __init__(self, provider: Provider, model: str, profile: ModelProfile, *,
                 transport: Callable[[str, bytes, float], Awaitable[Reply]] | None = None, own_transport: bool = False,
                 timeout: float = 60.0, batch_inputs: int = 8,
                 max_response_bytes: int | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] | None = None):
        if anyio is None:
            raise ConfigError("hosted judges require semantic-switch[hosted]")
        from .drivers.codecs import CODECS
        from .presentation import identity as presentation_identity
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
            raise ConfigError("timeout must be a positive number of seconds")
        if isinstance(batch_inputs, bool) or not isinstance(batch_inputs, int) or batch_inputs < 1:
            raise ConfigError("batch_inputs must be a positive integer")
        if not isinstance(own_transport, bool):
            raise ConfigError("own_transport must be boolean")
        if transport is not None and not callable(transport):
            raise ConfigError("transport must be an async send(url, body_bytes, remaining_seconds) callable")
        if own_transport and transport is None:
            raise ConfigError("own_transport transfers a supplied transport; the built-in transport is always owned")
        if profile.codec not in CODECS:
            raise ConfigError(f"profile {profile.name} names unknown codec {profile.codec!r}")
        self.codec = CODECS[profile.codec]
        self.codec.accepts(profile)
        self.max_response_bytes = self.codec.response_bytes if max_response_bytes is None else max_response_bytes
        if isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or self.max_response_bytes < 1:
            raise ConfigError("max_response_bytes must be a positive integer")
        self.provider, self.model, self.profile, self.timeout, self.batch_inputs = provider, model, profile, timeout, batch_inputs
        self.url = provider.endpoint(self.codec)
        self.send = transport if transport is not None else provider.wire(batch_inputs, self.max_response_bytes)
        self.own_transport = transport is None or own_transport
        self.clock, self.sleep = clock, sleep or anyio.sleep
        record = profile.record()
        self._scope, self._limits = "joint" if self.codec.joint else "individual", profile.admission()
        self.identity = {"source": provider.name, "driver": f"{provider.name}@{provider.revision}", "model": model,
                         "presentation": presentation_identity(), "endpoint": self.url,
                         "remote": {"route": provider.route, "observed": "per-response", "weights": profile.weights,
                                    "guarantee": profile.pinning},
                         **provider.identity, "codec": f"{self.codec.name}@{self.codec.revision}", **self.codec.identity,
                         "question_scope": self._scope,
                         "profile": {"name": profile.name, "revision": profile.revision, "record": record},
                         "score_semantics": profile.semantics, "normalization": profile.normalization, "precision": record["precision"]}
        self.normalization = profile.normalization
        self.returned: str | None = None
        self.observed_models = set()
        self._lock = threading.Lock()
        self._closed = False
        self._released = threading.Event()
        self._portal = None
        self._scopes = set()
        self._drained = None
        self._not_before = -math.inf

    @property
    def max_slots(self) -> int:
        return self.profile.max_slots

    def static(self, questions: Mapping[str, Question]) -> None:
        from .presentation import validate_question
        for name, question in questions.items():
            try:
                validate_question(self, question)
                self.profile.validate((question,))
            except ConfigError as error:
                raise ConfigError(f"{name}: {error}") from None

    def _prepared(self, items) -> tuple[list, list[Unit]]:
        from .presentation import validate_question
        items = [(state, tuple(questions)) for state, questions in items]
        for _, questions in items:
            for question in questions:
                validate_question(self, question)
        units = []
        for owner, (state, questions) in enumerate(items):
            if not isinstance(state, str):
                raise ConfigError(f"profile {self.profile.name} accepts text evidence only; images and video need a separate evidence contract")
            if not questions:
                continue
            for index in (tuple(range(len(questions))),) if self.codec.joint else ((k,) for k in range(len(questions))):
                unit = tuple(questions[k] for k in index)
                self.profile.validate(unit)
                units.append(Unit(owner, index, state, unit))
        return items, units

    def preview(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[dict]:
        return [{"representation": "hosted", "input": unit.owner, "questions": list(unit.index), "joint": self.codec.joint, "endpoint": self.url,
                 "body": self.provider.envelope(self.model, self.codec, self.codec.encode(unit.state, unit.questions, self.profile))}
                for unit in self._prepared(items)[1]]

    def start(self, items: Sequence[tuple[str, Sequence[Question]]], complete: Callable[[Completion], None]) -> Future:
        seen, previous = set(), self.returned

        def observed(done: Completion):
            models = {score.model for score in done.scored.scores if score.model is not None}
            if models:
                with self._lock:
                    seen.update(models)
                    self.observed_models.update(models)
                    self.returned = next(iter(seen)) if len(seen) == 1 else previous
            complete(done)
        return self._submit(self._prepared(items)[1], observed)

    def _submit(self, units: list[Unit], complete: Callable[[Completion], None]) -> Future:
        portal = self._bound()
        try:
            return portal.start_task_soon(self._operation, units, complete)
        except RuntimeError:
            raise RuntimeError("judge is closed") from None

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]:
        slots = [[None] * len(questions) for _, questions in items]
        tokens, returned = [0] * len(items), [set() for _ in items]

        def complete(done: Completion):
            for k, score in zip(done.index, done.scored.scores):
                slots[done.owner][k] = score
            tokens[done.owner] += done.scored.tokens
            if done.scored.model is not None:
                returned[done.owner].add(done.scored.model)
        future = self.start(items, complete)
        handle = OPERATION.get()
        release = handle.register(future.cancel) if handle is not None else (lambda: None)
        try:
            future.result()
        except CancelledError:
            if self._closed:
                raise RuntimeError("judge is closed") from None
            raise Cancelled("operation cancelled") from None
        finally:
            release()
        results = [Scored(tuple(scores), tokens[n], next(iter(returned[n])) if len(returned[n]) == 1 else None) for n, scores in enumerate(slots)]
        models = sorted({score.model for result in results for score in result.scores if score.model is not None})
        if len(models) > 1:
            warnings.warn(f"hosted operation returned mixed model versions: {models}", stacklevel=2)
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

    async def _operation(self, units: list[Unit], complete: Callable[[Completion], None]) -> None:
        if self._closed:
            raise RuntimeError("judge is closed")
        with anyio.CancelScope() as operation:
            self._scopes.add(operation)
            try:
                async with anyio.create_task_group() as group:
                    for unit in units:
                        await self._admit(group, unit, complete)
            except BaseExceptionGroup as failure:
                raise leaf(failure) from None
            finally:
                self._scopes.discard(operation)
                if self._closed and not self._scopes:
                    self._drained.set()
        if operation.cancelled_caught:
            raise RuntimeError("judge is closed")

    async def _admit(self, group, unit: Unit, complete) -> None:
        progress = Progress(self.clock())
        deadline = progress.started + self.timeout
        with anyio.move_on_after(max(0., deadline - self.clock())) as waited:
            await self._admission.acquire()
        progress.queued = self.clock() - progress.started
        if waited.cancelled_caught:
            complete(Completion(unit.owner, unit.index, self._failed(unit.questions, "deadline exceeded awaiting admission", progress)))
            return
        try:
            body = serialized(self.provider.envelope(self.model, self.codec, self.codec.encode(unit.state, unit.questions, self.profile)))
            reason = self.profile.unsupported(len(body))
            progress.admission = {**self._limits, "bytes": len(body), "grouping": {"scope": self._scope, "questions": len(unit.questions)},
                                  "unsupported": reason}
            failure = reason and f"unsupported request ({reason})"
        except MemoryError:
            self._admission.release()
            raise
        except Exception as error:
            body, failure = None, f"request encoding failed ({type(error).__name__}: {error})"
        if failure:
            self._admission.release()
            complete(Completion(unit.owner, unit.index, self._failed(unit.questions, failure, progress)))
            return
        group.start_soon(self._unit, unit, body, progress, deadline, complete, self.clock())

    async def _unit(self, unit: Unit, body: bytes, progress: Progress, deadline: float, complete, ready: float) -> None:
        progress.queued += self.clock() - ready
        try:
            scored = await self._score(unit, body, progress, deadline)
        except MemoryError:
            raise
        except Exception as error:
            scored = self._failed(unit.questions, f"unit failed ({type(error).__name__}: {error})", progress)
        finally:
            self._admission.release()
        complete(Completion(unit.owner, unit.index, scored))

    def _failed(self, questions, message, progress: Progress, model=None, tokens=0) -> Scored:
        request = progress.request(self.clock())
        return Scored(tuple(Scores(None, None, f"hosted: {message}", model=model, request=request) for _ in questions), tokens, model)

    def _expired(self, progress: Progress) -> str:
        return f"deadline of {self.timeout:g}s exceeded after {progress.attempts} attempts"

    async def _score(self, unit: Unit, body: bytes, progress: Progress, deadline: float) -> Scored:
        questions = unit.questions
        response, error = await self._exchange(body, deadline, progress)
        if error:
            return self._failed(questions, error, progress)
        try:
            envelope = self.provider.unwrap(response)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return self._failed(questions, f"malformed response ({error})", progress)
        returned, tokens = envelope.returned, envelope.tokens
        if self.provider.route and envelope.route != self.provider.route:
            return self._failed(questions, f"provider route changed to {envelope.route!r}", progress, returned, tokens)
        try:
            scores = tuple(self.codec.decode(envelope.payload, questions, self.profile, returned, progress.request(self.clock())))
            if len(scores) != len(questions):
                raise ValueError(f"codec returned {len(scores)} scores for {len(questions)} questions")
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            return self._failed(questions, f"malformed response ({error})", progress, returned, tokens)
        if self.clock() >= deadline:
            return self._failed(questions, f"response completed after the {self.timeout:g}s deadline", progress, returned, tokens)
        return Scored(scores, tokens, returned)

    async def _wait(self, seconds: float, progress: Progress) -> None:
        before = self.clock()
        try:
            await self.sleep(seconds)
        finally:
            spent = self.clock() - before
            if progress.attempts:
                progress.retry += spent
            else:
                progress.queued += spent

    async def _exchange(self, body: bytes, deadline: float, progress: Progress) -> tuple[dict | None, str | None]:
        failure = "provider Retry-After"
        while True:
            now = self.clock()
            if now >= deadline:
                return None, self._expired(progress)
            hold = self._not_before - now
            if hold > 0:
                if now + hold >= deadline:
                    return None, f"{failure}; retry after {hold:.1f}s would exceed the {self.timeout:g}s deadline"
                await self._wait(hold, progress)
                continue
            reply = None
            with anyio.move_on_after(deadline - now) as bounded:
                async with self._active:
                    progress.queued += self.clock() - now
                    remaining = deadline - self.clock()
                    if self._not_before > self.clock() or remaining <= 0:
                        continue
                    progress.attempts += 1
                    try:
                        reply = await self.send(self.url, body, remaining)
                    except TransportError as error:
                        return None, str(error)
                    except (ConnectionError, TimeoutError) as error:
                        failure = f"connection failed ({type(error).__name__})"
            if bounded.cancelled_caught:
                return None, self._expired(progress)
            hint = None
            progress.status = None if reply is None else reply.status
            if reply is not None:
                if 200 <= reply.status < 300:
                    if not isinstance(reply.body, dict):
                        return None, "response must be a JSON object"
                    progress.usage = self.provider.usage(reply.body)
                    if self.clock() >= deadline:
                        return None, f"response completed after the {self.timeout:g}s deadline"
                    return reply.body, None
                if not retryable(reply.status):
                    return None, self.provider.rejected(reply)
                hint, failure = retry_after(reply.retry_after), f"HTTP {reply.status}"
            if hint is not None:
                self._not_before = max(self._not_before, self.clock() + hint)
            if progress.attempts == ATTEMPTS:
                return None, f"{failure} after {ATTEMPTS} attempts"
            if hint is not None:
                continue
            wait = 2 ** (progress.attempts - 1) * random.uniform(.5, 1)
            if self.clock() + wait >= deadline:
                return None, f"{failure}; retry after {wait:.1f}s would exceed the {self.timeout:g}s deadline"
            await self._wait(wait, progress)

    async def _release_transport(self):
        close = getattr(self.send, "close", None) if self.own_transport else None
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _shutdown(self):
        for scope in list(self._scopes):
            scope.cancel()
        if self._scopes:
            await self._drained.wait()
        await self._release_transport()

    def close(self):
        with self._lock:
            closing, self._closed = not self._closed, True
            bound, self._portal = (self._portal, None) if closing else (None, self._portal)
        if not closing:
            self._released.wait()
            return
        try:
            if bound is not None:
                manager, portal = bound
                try:
                    portal.call(self._shutdown)
                finally:
                    manager.__exit__(None, None, None)
            elif self.own_transport and getattr(self.send, "close", None) is not None:
                with anyio.from_thread.start_blocking_portal() as portal:
                    portal.call(self._release_transport)
        finally:
            self._released.set()

    async def aclose(self):
        await anyio.to_thread.run_sync(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
