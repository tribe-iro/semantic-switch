from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import threading
from typing import Callable
import uuid
import warnings
import weakref

from semantic_switch import ConfigError, Runtime
from semantic_switch.training import Task, artifact_digest, evidence_snapshot, files, thaw

SCHEMA = 3
LINKS = ('thread', 'entity')
DROPS = ('records', 'bytes', 'encoding', 'storage')
FLUSH = 64 << 10
INDEPENDENCE = 'thread and entity links are missing; differing raw inputs cannot certify family-disjoint independence'
_capturing: ContextVar[list | None] = ContextVar('semantic_learn_capturing', default=None)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{name} must be nonempty text')
    return value


@dataclass(frozen=True)
class Consent:
    policy: str
    revision: str
    scope: str = 'capture'

    def __post_init__(self):
        for name in ('policy', 'revision', 'scope'):
            _text(getattr(self, name), f'consent {name}')

    def record(self) -> dict:
        return {'policy': self.policy, 'revision': self.revision, 'scope': self.scope}


@dataclass(frozen=True)
class Redactor:
    revision: str
    apply: Callable

    def __post_init__(self):
        _text(self.revision, 'redactor revision')
        if not callable(self.apply):
            raise ConfigError('a redactor applies a deterministic callable to every input before inference and capture')

    def __call__(self, data):
        try:
            owned = evidence_snapshot(data)
        except ConfigError as error:
            raise ConfigError(f'redactor {self.revision} input is {error}; nothing was sent') from None
        try:
            returned = self.apply(owned)
        except Exception as error:
            raise ConfigError(f'redactor {self.revision} failed before inference ({type(error).__name__}: {error}); nothing was sent') from None
        if not isinstance(returned, (str, dict)):
            raise ConfigError(f'redactor {self.revision} returned {type(returned).__name__}, not text or a mapping; nothing was sent')
        try:
            return evidence_snapshot(returned, mutable=False)
        except ConfigError as error:
            raise ConfigError(f'redactor {self.revision} returned {error}; nothing was sent') from None


@dataclass(frozen=True)
class Sampling:
    probability: float
    seed: str

    def __post_init__(self):
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)) or not math.isfinite(self.probability) \
                or not 0 < self.probability <= 1:
            raise ConfigError('capture sampling probability must be in (0, 1]')
        _text(self.seed, 'sampling seed')

    def chosen(self, key: str) -> bool:
        return int(hashlib.sha256(f'{self.seed}\0{key}'.encode()).hexdigest(), 16) < self.probability * (1 << 256)

    def record(self) -> dict:
        return {'probability': self.probability, 'seed': artifact_digest(self.seed), 'scope': 'situation', 'method': 'sha256(seed, situation identity) threshold'}


@dataclass(frozen=True)
class Bounds:
    max_records: int
    max_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.max_records, self.max_bytes)):
            raise ConfigError('capture bounds are positive record and byte counts')


def period(moment=None) -> str:
    return (moment or datetime.now(timezone.utc)).strftime('%Y-%m')


def _torn(fragment: bytes) -> bool:
    try:
        json.loads(fragment)
        return False
    except ValueError:
        return True


class Log:
    def __init__(self, path, header: dict, bounds: Bounds):
        self.path, self.bounds, self.pending, self.buffered, self.error = Path(path), bounds, [], 0, None
        self.records = self.bytes = self.persisted = self.lost = 0
        self.reopened = {'records': 0, 'bytes': 0, 'torn': False}
        self.lease = files.Lease(self.path.with_name(self.path.name + '.writer'))
        try:
            encoded = files.jsonl([header])
            if not self._scan(encoded):
                files.append(self.path, encoded)
                self.bytes = len(encoded)
        except BaseException:
            self.lease.release()
            raise

    def _scan(self, header: bytes) -> bool:
        if not self.path.exists():
            return False
        with self.path.open('rb') as fh:
            first = fh.readline()
            if not first.endswith(b'\n'):
                if not first or _torn(first):
                    self.reopened['torn'] = bool(first)
                    return False
                if json.loads(first) != json.loads(header):
                    raise ConfigError(f'{self.path}: holds another capture configuration or schema; capture into a new NAME.capture.jsonl')
                with files.locked(self.path):
                    with self.path.open('ab') as tail:
                        tail.write(b'\n')
                        tail.flush()
                        os.fsync(tail.fileno())
                first += b'\n'
                fh.seek(len(first))
            if json.loads(first) != json.loads(header):
                raise ConfigError(f'{self.path}: holds another capture configuration or schema; capture into a new NAME.capture.jsonl')
            size = len(first)
            for line in fh:
                if line.endswith(b'\n'):
                    self.records, size = self.records + 1, size + len(line)
                elif _torn(line):
                    self.reopened['torn'] = True
                else:
                    self.records, size = self.records + 1, size + len(line) + 1
        with files.locked(self.path):
            with self.path.open('r+b') as tail:
                if self.reopened['torn']:
                    tail.truncate(size)
                elif tail.seek(0, 2) < size:
                    tail.write(b'\n')
                tail.flush()
                os.fsync(tail.fileno())
        self.bytes = size
        self.reopened.update(records=self.records, bytes=size)
        return True

    def admit(self, encoded: bytes) -> str | None:
        if self.records + len(self.pending) >= self.bounds.max_records:
            return 'records'
        if self.bytes + self.buffered + len(encoded) > self.bounds.max_bytes:
            return 'bytes'
        if len(encoded) > FLUSH:
            self.flush()
            self.write([encoded])
            return None
        self.pending.append(encoded)
        self.buffered += len(encoded)
        if self.buffered >= FLUSH:
            self.flush()
        return None

    def flush(self):
        if self.pending:
            chunks, self.pending, self.buffered = self.pending, [], 0
            self.write(chunks)

    def write(self, chunks):
        data = b''.join(chunks)
        try:
            files.append(self.path, data)
        except Exception as error:
            self.lost, self.error = self.lost + len(chunks), error
            self.bytes = self.path.stat().st_size if self.path.exists() else 0
            return
        self.records, self.persisted, self.bytes = self.records + len(chunks), self.persisted + len(chunks), self.bytes + len(data)


class Capture:
    def __init__(self, log, *, redactor: Redactor, sampling: Sampling, bounds: Bounds, link_key: bytes, judge=None, bindings=None, job=None):
        if not isinstance(link_key, bytes) or len(link_key) < 16:
            raise ConfigError('link_key is a secret of at least 16 bytes that keys privacy-preserving correlation identifiers')
        self.path = Path(log)
        if not self.path.name.endswith('.capture.jsonl'):
            raise ConfigError(f'{self.path}: capture logs are named NAME.capture.jsonl')
        self.redactor, self.sampling, self.bounds, self._key, self._lock = redactor, sampling, bounds, link_key, threading.Lock()
        self._tasks, self._closed = weakref.WeakKeyDictionary(), False
        self._lifetime = threading.Condition()
        self._active = 0
        self._closing = False
        self._owners = {}
        self.counts = {'captured': 0, 'feedback': 0, 'opted_out': 0, 'not_sampled': 0, 'unwrapped': 0, 'dropped': dict.fromkeys(DROPS, 0)}
        self.log = Log(self.path, {'schema': SCHEMA, 'kind': 'capture', 'redactor': redactor.revision, 'sampling': sampling.record(),
                                   'bounds': {'max_records': bounds.max_records, 'max_bytes': bounds.max_bytes},
                                   'links': artifact_digest(hmac.new(link_key, b'links', 'sha256').hexdigest())}, bounds)
        try:
            self.runtime = Runtime(judge, bindings=bindings, on_decision=self._observe, decision_input=True, job=job)
        except BaseException:
            self.log.lease.release()
            raise

    def _count(self, name):
        with self._lock:
            self.counts[name] += 1

    def _observe(self, event):
        found = _capturing.get()
        if found is None:
            self._count('unwrapped')
        else:
            found.append(event)

    @contextmanager
    def _observed(self):
        events = []
        token = _capturing.set(events)
        try:
            yield events
        finally:
            _capturing.reset(token)

    def load(self, path, **options):
        return self.runtime.load(path, **options)

    def links(self, correlation) -> dict:
        if correlation is None:
            correlation = {}
        if not isinstance(correlation, dict) or any(not isinstance(kind, str) or not kind for kind in correlation):
            raise ConfigError('correlation maps a link kind (thread, entity, ...) to NAMESPACE:ID')
        out = {}
        for kind, value in sorted(correlation.items()):
            namespace, separator, identifier = _text(value, f'correlation {kind}').partition(':')
            if not separator or not namespace or not identifier:
                raise ConfigError(f'correlation {kind} {value!r} needs its source namespace: NAMESPACE:ID')
            out[kind] = {'namespace': namespace, 'id': hmac.new(self._key, f'{kind}\0{namespace}\0{identifier}'.encode(), 'sha256').hexdigest()}
        return out

    def task(self, switch, snapshot=None) -> Task:
        snapshot = switch.snapshot if snapshot is None else snapshot
        with self._lock:
            cached = self._tasks.get(switch)
            if cached is None or cached[0] is not snapshot:
                cached = self._tasks[switch] = (snapshot, Task(snapshot))
        return cached[1]

    @contextmanager
    def _admitted(self):
        owner = threading.get_ident()
        with self._lifetime:
            if self._closing or self._closed:
                raise ConfigError(f'{self.path}: capture is closed')
            self._active += 1
            self._owners[owner] = self._owners.get(owner, 0) + 1
        try:
            yield
        finally:
            with self._lifetime:
                self._active -= 1
                remaining = self._owners[owner] - 1
                if remaining:
                    self._owners[owner] = remaining
                else:
                    del self._owners[owner]
                self._lifetime.notify_all()

    def ask(self, state, items, *, consent: Consent | None, correlation=None, fields=None, guidance=None) -> dict:
        with self._admitted():
            redacted, links = self.redactor(state), self.links(correlation)
            key = artifact_digest(['inline', redacted, fields])
            return self._captured(consent, key, links, None, lambda: self.runtime.ask(redacted, items, fields=fields, guidance=guidance))

    def decide(self, switch, data, *, consent: Consent | None, correlation=None):
        with self._admitted():
            redacted, links = self.redactor(data), self.links(correlation)
            snapshot = switch.snapshot
            task = self.task(switch, snapshot)
            return self._captured(consent, task.identity('situation')(redacted), links, task, lambda: switch.run([redacted], snapshot)[0])

    def _captured(self, consent, key, links, task, call):
        if consent is not None and not isinstance(consent, Consent):
            raise ConfigError('consent is a Consent declaration, or None for uncaptured inference')
        chosen = consent is not None and self.sampling.chosen(key)
        with self._observed() as events:
            result = call()
        if not chosen:
            self._count('opted_out' if consent is None else 'not_sampled')
            return result
        for event in events:
            self._store(lambda event=event: self.record(event, consent, links, task))
        return result

    def record(self, event, consent, links, task) -> dict:
        missing = [kind for kind in LINKS if kind not in links]
        selection = {'selected_by': 'random', 'inclusion_probability': self.sampling.probability, 'window': period(), 'selection_id': artifact_digest(self.sampling.seed)}
        groups = [f"{kind}:{link['namespace']}:{link['id']}" for kind, link in links.items()]
        observed = thaw(event.input)
        return {'kind': 'captured-decision', 'operation': event.operation, 'event': event.event, 'call': event.call, 'item': event.item,
                'attempt': uuid.uuid4().hex, 'period': period(), 'consent': consent.record(), 'redactor': self.redactor.revision,
                'sampling': {'reason': 'random', 'probability': self.sampling.probability},
                'situation': task.situation(observed, groups=groups, selections=[selection]) if task is not None else None,
                'input': None if task is not None else observed, 'task': thaw(event.task), 'predictor': thaw(event.predictor),
                'decision': {'status': event.status, 'rule': thaw(event.rule), 'complete': event.complete, 'lock_id': event.lock_id,
                             'candidate_id': event.candidate_id, 'facts': thaw(event.facts), 'cost': thaw(event.cost), 'score_semantics': event.score_semantics},
                'links': links, 'independence': INDEPENDENCE if missing else 'linked', 'missing_links': missing}

    def feedback(self, event, address, value, *, source):
        with self._admitted():
            for name, text in (('event', event), ('address', address), ('source', source)):
                _text(text, f'feedback {name}')
            self._store(lambda: {'kind': 'feedback', 'event': event, 'address': address, 'value': value, 'source': source, 'weak': True,
                                 'period': period(), 'attempt': uuid.uuid4().hex}, 'feedback')
    

    def _failed(self):
        error, self.log.error = self.log.error, None
        if error is not None:
            warnings.warn(f'capture storage failed ({type(error).__name__}); the decision result is preserved and the drop is disclosed', RuntimeWarning,
                          stacklevel=4)

    def _store(self, build, counted='captured'):
        try:
            encoded = files.jsonl([build()])
        except Exception as error:
            self._drop('encoding')
            warnings.warn(f'capture record encoding failed ({type(error).__name__}); the decision result is preserved and the drop is disclosed',
                          RuntimeWarning, stacklevel=3)
            return
        with self._lock:
            if self._closed:
                raise ConfigError(f'{self.path}: capture is closed')
            refused = self.log.admit(encoded)
            if refused:
                self.counts['dropped'][refused] += 1
            else:
                self.counts[counted] += 1
            self._failed()

    def _drop(self, reason):
        with self._lock:
            self.counts['dropped'][reason] += 1

    def flush(self):
        with self._lock:
            self.log.flush()
            self._failed()

    def disclosure(self) -> dict:
        with self._lock:
            return self._disclosed()

    def _disclosed(self) -> dict:
        return {**self.counts, 'dropped': {**self.counts['dropped'], 'storage': self.log.lost}, 'persisted': self.log.persisted,
                'buffered': len(self.log.pending), 'records': self.log.records,
                'bytes': self.log.bytes, 'buffered_bytes': self.log.buffered, 'reopened': dict(self.log.reopened),
                'bounds': {'max_records': self.bounds.max_records, 'max_bytes': self.bounds.max_bytes, 'flush_bytes': FLUSH},
                'scope': 'only calls through this wrapper with a consent declaration are captured; unwrapped runtime calls are counted, never stored, '
                         'and their inputs are not protected by this redactor; captured and feedback records are persisted, buffered or dropped for '
                         'storage, and bound or encoding drops were never accepted'}

    def close(self) -> dict:
        with self._lifetime:
            if threading.get_ident() in self._owners:
                raise ConfigError('capture cannot close inside an admitted call')
            if self._closing:
                self._lifetime.wait_for(lambda: self._closed)
                return self._summary
            self._closing = True
            self._lifetime.wait_for(lambda: self._active == 0)
        try:
            with self._lock:
                self.log.flush()
                self._failed()
                self._summary = self._disclosed()
            try:
                files.publish(self.path.with_name(self.path.name.removesuffix('.capture.jsonl') + '.capture.json'),
                              {'schema': SCHEMA, 'kind': 'capture', 'redactor': self.redactor.revision, 'disclosure': self._summary})
            except Exception as error:
                warnings.warn(f'capture disclosure publication failed ({type(error).__name__}); the log is flushed', RuntimeWarning, stacklevel=2)
            return self._summary
        finally:
            try:
                self.runtime.close()
            finally:
                self.log.lease.release()
                with self._lifetime:
                    self._closed = True
                    self._lifetime.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def captured(path):
    path = Path(path)
    header = None
    for line, record in files.iter_records(path, with_lines=True):
        if header is None:
            header = record
            if not isinstance(record, dict) or record.get('kind') != 'capture' or record.get('schema') != SCHEMA:
                raise ConfigError(f'{path}: not a schema-{SCHEMA} capture log')
            continue
        if not isinstance(record, dict) or record.get('kind') not in ('captured-decision', 'feedback'):
            raise ConfigError(f'{path}:{line}: not a capture record')
        yield record
    if header is None:
        raise ConfigError(f'{path}: not a schema-{SCHEMA} capture log')
