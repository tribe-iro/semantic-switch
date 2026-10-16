from __future__ import annotations

import contextvars
import functools
import json
import os
import threading
import warnings
from dataclasses import dataclass
from functools import cache
from contextlib import ExitStack, contextmanager
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from types import MappingProxyType
import hashlib
import asyncio
from pathlib import Path

import numpy as np

from . import files
from .errors import ConfigError
from .inline import Check, Pick, Rate, Select
from .judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS, LlamaJudge
from .protocols import MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS
from .resident import Resources
from .models import DEFAULT_JUDGE, PRECISE_JUDGE, Model, NATIVE_SETTINGS, native_settings
from .model_jobs import ModelJob, estimate, model_job
from .switch import Switch

if TYPE_CHECKING:
    from .artifacts import Provenance
    from .protocols import Judge


def judge(source, **settings):
    from .drivers import build
    return build(source, settings)


def located(name) -> Path:
    path = Path(name)
    if path.suffix in (".yaml", ".yml", ".json") or path.exists():
        return path
    local = Path("switches") / f"{path}.yaml"
    return local if local.exists() else path.with_name(f"{path.name}.yaml")


def _gpu() -> bool:
    from .backend import gpu_available

    return gpu_available()


@cache
def _build() -> str:
    try:
        import llama_cpp

        native = llama_cpp.llama_cpp._lib
        path = str(native._name)
        system = llama_cpp.llama_cpp.llama_print_system_info()
        return json.dumps({'version': llama_cpp.__version__, 'native': files.sha256_file(path),
                           'system': system.decode() if isinstance(system, bytes) else str(system)}, sort_keys=True)
    except ImportError:
        return "none"


@dataclass(frozen=True, slots=True)
class _Role:
    value: object
    kind: str
    acceleration: object
    batching: object


def _role(value) -> _Role:
    if isinstance(value, Model):
        return _Role(value, "model", lambda: value.device != "cpu" and _gpu(), lambda: 1)
    kind = "judge" if callable(getattr(value, "score_many", None)) else "scorer"
    return _Role(value, kind, lambda: bool(getattr(value, "accelerated", False)), lambda: int(getattr(value, "batch_inputs", 1)))


def behavior_identity(identity):
    return {key: value for key, value in identity.items() if key != "file"}


MODE = ("shared",)
EXECUTION = ("device", "tensor_types", "flash_attn", "type_k", "type_v", "use_extra_bufts", "repacked", "n_ctx", "n_batch", "n_ubatch",
             "n_ctx_seq", "sequence_limit", "n_seq_max", "n_outputs_max", "swa_full", "kv_unified", "prefix_slots", "scheduling") + MODE


def execution_identity(realized: dict) -> dict | None:
    return {**{key: realized[key] for key in EXECUTION}, "build": _build()} if realized else None


class Runtime:
    def __init__(self, judge=None, *, bindings: Mapping | None = None, on_decision=None, decision_input=False, own_judges=False,
                 job: ModelJob | None = None):
        if on_decision is not None and not callable(on_decision):
            raise ConfigError("on_decision must be callable")
        if not isinstance(decision_input, bool) or decision_input and on_decision is None:
            raise ConfigError("decision_input is a boolean request for the evaluated input in on_decision events")
        if not isinstance(own_judges, bool):
            raise ConfigError("own_judges must be boolean")
        if job is not None and not isinstance(job, ModelJob):
            raise ConfigError("job must be a ModelJob from model_job(...)")
        if bindings is not None and (not isinstance(bindings, Mapping) or not all(isinstance(name, str) and name for name in bindings)):
            raise ConfigError("bindings maps model role names to a judge source, Model or judge")
        if judge is not None and "judge" in (bindings or {}):
            raise ConfigError("bind the judge role once: pass judge=... or bindings={'judge': ...}, not both")
        supplied = {**({"judge": judge} if judge is not None else {}), **(bindings or {})}
        self._implicit = frozenset() if supplied else frozenset({"judge"})
        supplied = supplied or {"judge": DEFAULT_JUDGE}
        self._own_judges = own_judges
        self._job = job
        self._jobs = ExitStack()
        self._on_decision = on_decision
        self._decision_input = decision_input
        self._event_warning = False
        self._constructed = frozenset(name for name, value in supplied.items() if isinstance(value, str))
        from .drivers import build
        self._roles = {name: _role(build(value, {}) if isinstance(value, str) else value) for name, value in supplied.items()}
        self._bindings = MappingProxyType({name: role.value for name, role in self._roles.items()})
        from .adapters.chat import RenderContext
        self._render = RenderContext.now()
        self._loaded: dict = {}
        self._executor = None
        self._residents: dict = {}
        self._identities = {}
        self._identity_fingerprints = {}
        self._lock = threading.RLock()
        self._closed = False
        self._active = 0
        self._handles: set = set()
        self._operations = threading.local()
        self._idle = threading.Condition(self._lock)

    def _install(self, role: str, model: Model, source) -> None:
        with self._lock:
            present = self._roles.get(role)
            if present is not None and role not in self._implicit:
                if present.value == model:
                    return
                raise ConfigError(f"{source}: binding {role!r} is this fitted lock's model, but the runtime binds {getattr(present.value, 'source', present.value)!r} "
                                  "to it; load the lock in a runtime that leaves that role to the lock")
            if role in self._loaded:
                raise ConfigError(f"{source}: this runtime already loaded its default judge for {role!r}; load the fitted lock in a new Runtime()")
            self._roles[role] = _role(model)
            self._implicit -= {role}
            self._constructed -= {role}
            self._bindings = MappingProxyType({name: descriptor.value for name, descriptor in self._roles.items()})

    @contextmanager
    def operation(self):
        from .judge import OPERATION, Cancellation
        with self._idle:
            depth = getattr(self._operations, "depth", 0)
            if self._closed and not depth:
                raise RuntimeError("runtime is closed")
            self._operations.depth = depth + 1
            self._active += 1
            handle, token = OPERATION.get(), None
            if not depth:
                if handle is None:
                    handle = Cancellation()
                    token = OPERATION.set(handle)
                self._handles.add(handle)
        try:
            yield
        finally:
            with self._idle:
                if token is not None:
                    OPERATION.reset(token)
                if not depth:
                    self._handles.discard(handle)
                self._operations.depth -= 1
                self._active -= 1
                self._idle.notify_all()

    @property
    def models(self):
        return self._bindings

    def _role(self, role):
        if role not in self._roles:
            raise ConfigError(f"model role {role!r} is not bound in this runtime")
        return self._roles[role]

    def _spec(self, role: str):
        return self._role(role).value

    def cache_key(self, role: str = "judge", capacity: dict | None = None) -> str:
        spec = self._spec(role)
        settings = {name: getattr(spec, name, None) for name in ("n_ctx", "n_batch", "threads", "max_input_tokens", "max_prefix_tokens", "prefix_slots")}
        execution = execution_identity(self._capacity(role) if capacity is None else capacity)
        return json.dumps({**behavior_identity(self.identity(role)), "execution": execution and {key: value for key, value in execution.items() if key not in MODE},
                           "settings": settings, "inputs": self.batch_inputs(role), "conformance": 8}, sort_keys=True)

    def execution(self, role: str = "judge") -> dict | None:
        if self._role(role).kind == "judge":
            return None
        self.judge(role)
        return execution_identity(self._capacity(role))

    def _capacity(self, role):
        capacity = getattr(getattr(self._loaded.get(role), "scorer", None), "capacity", None)
        return capacity() if capacity else {}

    def judge(self, role: str = "judge") -> Judge:
        with self._lock:
            if self._closed and not getattr(self._operations, "depth", 0):
                raise RuntimeError("runtime is closed")
            if role not in self._loaded:
                spec = self._spec(role)
                if self._role(role).kind == "model":
                    self._loaded[role] = self._load_judge(role, spec)
                elif self._role(role).kind == "judge":
                    self._loaded[role] = spec
                else:
                    from .adapters import adapter_for

                    adapter = adapter_for(getattr(spec, "adapter", "qwen3"), spec, self._render)
                    self._loaded[role] = LlamaJudge(self._view(role, spec), adapter, self.identity(role), self.accelerated(role),
                                                    max_input_tokens=getattr(spec, "max_input_tokens", MAX_INPUT_TOKENS))
            return self._loaded[role]

    def _load_judge(self, role: str, spec: Model) -> LlamaJudge:
        from .adapters import adapter_for
        from .artifacts import load_quality_profile
        from .backend import LlamaScorer
        from .resident import ResidentBase, binding_identity

        if spec.device == "gpu" and not _gpu():
            raise ConfigError("device='gpu' was requested, but this llama.cpp build has no GPU offload; "
                              "rebuild with GPU support or use device='auto' or 'cpu'")
        try:
            place = spec.resident_key()
        except (RuntimeError, ValueError) as e:
            raise ConfigError(f"cannot load model {spec.source}: {e}") from None
        resident = self._residents.get(place)
        if resident is not None and resident.pool.max_adapters != spec.max_adapters:
            raise ConfigError(f"model role {role!r} shares a resident base with max_adapters {resident.pool.max_adapters}; declare one adapter bound per base")
        charge = ExitStack()
        if resident is None:
            try:
                if self._job is not None:
                    charge.enter_context(model_job(extra={f"{spec.source}:{use}": value for use, value in spec.resources().items() if use != "adapter"},
                                                   job=self._job))
                scorer = LlamaScorer(spec.path(), n_ctx=spec.n_ctx, n_batch=spec.n_batch, threads=spec.threads, n_gpu_layers=spec.gpu_layers,
                                     prefix_slots=spec.prefix_slots, repack=spec.repack, max_input_tokens=spec.max_input_tokens,
                                     max_prefix_tokens=spec.max_prefix_tokens)
            except (RuntimeError, ValueError) as e:
                charge.close()
                raise ConfigError(f"cannot load model {spec.source}: {e}") from None
            except BaseException:
                charge.close()
                raise
            fresh = ResidentBase(scorer, max_adapters=spec.max_adapters, job=self._job, placement=spec.placement)
        else:
            fresh, scorer = None, resident.scorer
        try:
            try:
                adapter = adapter_for(spec.adapter, scorer, self._render)
            except ValueError as e:
                raise ConfigError(f"model {spec.source}: {e}") from None
            lora = spec.adapter_content()
            identity = _local_identity(spec, self._render, lora, scorer.chat)
            if spec.provenance is not None and identity["sha256"] != spec.provenance.gguf_sha256:
                raise ConfigError(f"{spec.source}: weights changed since their provenance was verified ({spec.provenance.gguf_sha256}); "
                                  "reload the deployment or re-run semantic-trainer select")
            self._identities[role] = identity
            self._identity_fingerprints[role] = _file_fingerprint(spec)
            capacity = getattr(scorer, "capacity", dict)
            key = self.cache_key(role, capacity())
            known = files.read_cached("doctor", key)
            if not known and scorer.weights.hybrid:
                try:
                    checks = (fresh or resident).view(lora).selftest(adapter, CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS)
                except RuntimeError as error:
                    scorer.invalidate()
                    checks = {"D1 shared = reference": {"ok": False, "error": str(error)}}
                files.write_cached("doctor", key, {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "at": files.now()})
            view = (fresh or resident).view(lora, binding_identity(identity, execution_identity(capacity())))
            bound = LlamaJudge(self._view(role, view), adapter, self.identity(role), self.accelerated(role), self.batch_inputs(role),
                               max_input_tokens=spec.max_input_tokens, max_prefix_tokens=spec.max_prefix_tokens,
                               quality_profile=load_quality_profile(spec.quality_profile) if spec.quality_profile else None)
        except BaseException:
            if fresh is not None:
                fresh.close()
                charge.close()
            self._identities.pop(role, None)
            self._identity_fingerprints.pop(role, None)
            raise
        if fresh is not None:
            self._residents[place] = fresh
            self._jobs.enter_context(charge)
        self._loaded[role] = bound
        return bound

    @property
    def job(self) -> ModelJob | None:
        return self._job

    def identity(self, role: str = "judge") -> dict:
        with self._lock:
            descriptor = self._role(role)
            spec = descriptor.value
            if descriptor.kind == "judge":
                from .presentation import identity as presentation_identity
                return {**spec.identity, "presentation": presentation_identity()}
            if role in self._loaded and role in self._identities:
                return dict(self._identities[role])
            fingerprint = None
            if descriptor.kind == "model":
                fingerprint = _file_fingerprint(spec)
            if role in self._identities and self._identity_fingerprints.get(role) == fingerprint:
                return dict(self._identities[role])
            if descriptor.kind == "scorer":
                from .adapters import adapter_name
                from .presentation import identity as presentation_identity
                return {"source": getattr(spec, "name", type(spec).__name__),
                        "template": adapter_name(getattr(spec, "adapter", "qwen3"), getattr(spec, "chat", None)), "presentation": presentation_identity()}
            identity = _local_identity(spec, self._render)
            self._identities[role] = identity
            self._identity_fingerprints[role] = fingerprint
            return dict(identity)

    def representation(self, role: str = "judge") -> str:
        descriptor = self._role(role)
        if descriptor.kind == "model":
            return descriptor.value.representation
        return "base" if descriptor.kind == "scorer" or hasattr(descriptor.value, "adapter") else "hosted"

    def binding(self, role: str = "judge", scope=()):
        from .resident import Binding
        return Binding(role, self.representation(role), self.identity(role), self.execution(role), tuple(scope))

    def accelerated(self, role: str) -> bool:
        return self._role(role).acceleration()

    def batch_inputs(self, role: str = "judge") -> int:
        return self._role(role).batching()

    def _emit_decision(self, event):
        try:
            self._on_decision(event)
        except Exception:
            if not self._event_warning:
                self._event_warning = True
                warnings.warn('decision callback failed; completed result preserved', RuntimeWarning, stacklevel=3)

    @property
    def executor(self):
        with self._lock:
            if self._executor is None:
                from .executor import Executor
                self._executor = Executor(self)
            return self._executor

    def _asked(self, inputs, items: dict, fields, guidance, binding: str) -> list[dict]:
        from .inline import Asking, lower
        lowered, ex = lower(items, guidance), self.executor
        operation = ex.operation(ex.window((binding,)), lambda live: (live.work.result(), operation.elapsed(live)))
        rows = [None] * len(inputs)
        for index, row in operation.drive(inputs, lambda state: Asking(ex, lowered, binding, state, fields), ordered=False):
            rows[index] = row
        if self._on_decision is not None:
            from .events import InlineContext, call_id
            context, call = InlineContext(self, items, guidance, binding), call_id()
            for item, ((row, elapsed), data) in enumerate(zip(rows, inputs)):
                self._emit_decision(context.event(row, {'elapsed_ms': elapsed, 'operation': operation.receipt}, call, item,
                                                  data if self._decision_input else None))
        return [row for row, _ in rows]

    def _asked_stream(self, inputs, items, fields, guidance, binding, window):
        from collections import deque
        from .inline import Asking, lower
        from .events import InlineContext, call_id
        if type(window) is not int or window < 1:
            raise ConfigError('window must be a positive integer')
        ex, lowered, admitted = self.executor, lower(items, guidance), deque()
        context = InlineContext(self, items, guidance, binding) if self._on_decision is not None else None
        call = call_id()
        operation = ex.operation(window, lambda live: (live.work.result(), operation.elapsed(live)), receipt=False)
        def start(data):
            admitted.append(data)
            return Asking(ex, lowered, binding, data, fields)
        stream = operation.drive(inputs, start, ordered=True)
        try:
            while True:
                with self.operation():
                    try:
                        index, (row, elapsed) = next(stream)
                    except StopIteration:
                        return
                data = admitted.popleft()
                if context is not None:
                    self._emit_decision(context.event(row, {'elapsed_ms': elapsed}, call, index, data if self._decision_input else None))
                yield row
        finally:
            stream.close()

    def preview(self, input, items: dict, *, fields=None, guidance=None, binding: str = "judge"):
        from .inline import Asking, lower, preview
        with self.operation():
            return preview(Asking(self.executor, lower(items, guidance), binding, input, fields))

    def ask(self, input, items: dict, *, fields=None, guidance=None, binding: str = "judge") -> dict:
        with self.operation():
            return self._asked([input], items, fields, guidance, binding)[0]

    def ask_many(self, inputs, items: dict, *, fields=None, guidance=None, binding: str = "judge") -> list[dict]:
        with self.operation():
            if isinstance(inputs, (str, bytes, dict)) or not isinstance(inputs, Sequence):
                raise ConfigError("ask_many requires a finite list or tuple; use many for streaming")
            return self._asked(inputs, items, fields, guidance, binding)

    async def _offload(self, call, *args, **kwargs):
        from .judge import OPERATION, Cancellation
        handle, context = Cancellation(), contextvars.copy_context()
        context.run(OPERATION.set, handle)
        worker = asyncio.get_running_loop().run_in_executor(None, functools.partial(context.run, call, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            handle.cancel()
            await asyncio.wait([worker])
            if not worker.cancelled():
                worker.exception()
            raise

    async def aask(self, input, items: dict, *, fields=None, guidance=None):
        return await self._offload(self.ask, input, items, fields=fields, guidance=guidance)

    async def acheck(self, input, statement: str, **kwargs):
        return await self._offload(self.check, input, statement, **kwargs)

    async def apick(self, input, question: str, options, **kwargs):
        return await self._offload(self.pick, input, question, options, **kwargs)

    async def arate(self, input, question: str, levels, **kwargs):
        return await self._offload(self.rate, input, question, levels, **kwargs)

    async def aselect(self, input, question: str, labels, **kwargs):
        return await self._offload(self.select, input, question, labels, **kwargs)

    def _view(self, role, scorer):
        return scorer

    def info(self, role="judge") -> dict:
        with self._lock:
            descriptor = self._role(role)
            spec, judge = descriptor.value, self._loaded.get(role)
            scorer = getattr(judge, "scorer", None)
            context, resident = getattr(scorer, "ctx", None), getattr(scorer, "resident", None)
            return {"identity": self.identity(role) if judge is not None or descriptor.kind != "model" else
                                {"source": spec.source, "template": spec.adapter},
                    "requested_device": spec.device if descriptor.kind == "model" else "provider",
                    "threads": spec.threads if descriptor.kind == "model" else None,
                    "build": _build() if descriptor.kind == "model" or context is not None else None, "loaded": judge is not None,
                    "observed_models": sorted(getattr(judge, "observed_models", ())) or ([judge.returned] if getattr(judge, "returned", None) else []),
                    "accelerated": self.accelerated(role), "batch_inputs": self.batch_inputs(role),
                    "max_slots": getattr(judge, "max_slots", None),
                    "capabilities": {"shared": getattr(scorer, "shared", None),
                                     "conformance": hasattr(judge, "selftest"),
                                     "counters": {**getattr(scorer, "stats", {}), "submissions": getattr(context, "submissions", 0), "submitted_tokens": getattr(context, "submitted_tokens", 0)}},
                    "requested_capacity": getattr(context, "requested", {}),
                    "configuration": {"max_input_tokens": getattr(judge, "max_input_tokens", None),
                                      "max_prefix_tokens": getattr(judge, "max_prefix_tokens", None),
                                      "prefix_slots": getattr(scorer, "prefix_slots", None)},
                    "capacity": self._capacity(role),
                    "binding": {"representation": self.representation(role),
                                **({"adapter": getattr(scorer, "namespace", None), "pool": {**resident.pool.stats, "max_adapters": resident.pool.max_adapters,
                                                                                         "resident": resident.pool.placement()}}
                                   if resident is not None else {})}}

    def check(self, input, statement: str, *, excludes=(), fields=None, min_score=None, unstated: bool = False, guidance=None):
        return Check(statement, excludes, min_score, unstated, guidance)(input, self, fields)

    def pick(self, input, question: str, options, *, definition=None, eligible=None, allow_none=False, shortlist=None,
             fields=None, min_score=None, ids=None, focus=None, guidance=None, excludes=()):
        return Pick(question, options, definition, eligible, allow_none, shortlist, min_score, ids, focus, guidance,
                    excludes)(input, self, fields)

    def rate(self, input, question: str, levels, *, fields=None, min_score=None, guidance=None, excludes=()):
        return Rate(question, levels, min_score, guidance, excludes)(input, self, fields)

    def select(self, input, question: str, labels, *, excludes=(), unstated=False, min_score=None, guidance=None, eligible=None,
               max_claims=None, fields=None):
        return Select(question, labels, excludes=excludes, unstated=unstated, min_score=min_score, guidance=guidance,
                      eligible=eligible, max_claims=max_claims)(input, self, fields)

    def load(self, path, draft: bool = False, allow_model_change: bool = False) -> Switch:
        from .tools.deployment import frozen
        path = located(path)
        entry = frozen(path)
        if entry is None:
            return Switch(self, path, draft, allow_model_change)
        if draft or allow_model_change:
            raise ConfigError(f"{path}: a frozen lock loads its task, policies and execution exactly; draft and allow_model_change apply to authored definitions")
        if entry.receipt is not None:
            role = entry.receipt.data["scope"]["binding"]
            if entry.receipt.data["publication"] != "application":
                raise ConfigError(f"{path}: scoped support for targets {list(entry.receipt.data['scope']['targets'])} under binding {role!r}, not a runnable "
                                  f"application; install it by explicit target binding, Runtime(bindings={{{role!r}: ss.judge({str(path)!r}), ...}}).load(APPLICATION), "
                                  "or publish a complete application with bindings for the remaining targets")
            self._install(role, entry.receipt.model(), path)
        return Switch(self, entry.lock, frozen=True)

    def cancel(self) -> None:
        with self._idle:
            handles = list(self._handles)
        for handle in handles:
            handle.cancel()

    def close(self):
        with self._lock:
            if self._closed:
                return
            if getattr(self._operations, "depth", 0):
                raise RuntimeError("cannot close runtime from inside an active operation")
            self._closed = True
            while self._active:
                self._idle.wait()
            self._loaded = {}
            owned = [*self._residents.values(), *(descriptor.value for role, descriptor in self._roles.items()
                                                   if descriptor.kind != "model" and (self._own_judges or role in self._constructed))]
            resources = {id(resource): resource for resource in owned}
            self._residents = {}
        errors = []
        for resource in resources.values():
            close = getattr(resource, "close", None)
            if close:
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        self._jobs.close()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime resource cleanup failed", errors)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.cancel()
        self.close()


def _file_fingerprint(spec):
    stat = os.stat(spec.path())
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _local_identity(spec, context, lora=None, chat=None):
    from .presentation import LETTERS, SYSTEM, identity as presentation_identity, text
    path = spec.path()
    if spec.adapter == "qwen3":
        from .adapters.qwen3 import Qwen3Adapter
        name = Qwen3Adapter.name
        prompt = (Qwen3Adapter.prefix_text, Qwen3Adapter.tail_text, SYSTEM, LETTERS, "true/True/false/False")
    else:
        from .adapters.chat import Template
        template = Template.of((chat or spec.descriptor().chat)())
        name = template.name
        prompt = (template.render(context), template.add_bos, LETTERS, "true/True/false/False")
    prompt = (prompt, "<input>\n{rendered}\n</input>\n\n", tuple(text(question) for question in CONFORMANCE_QUESTIONS))
    lora = lora or spec.adapter_content()
    return {"presentation": presentation_identity(), "file": os.path.basename(path), "sha256": files.sha256_file(path), "template": name,
            "prompt": hashlib.sha256(repr(prompt).encode()).hexdigest(), **({"lora": lora.record()} if lora else {})}


_default: Runtime | None = None
_default_lock = threading.Lock()


def default() -> Runtime:
    global _default
    with _default_lock:
        if _default is None or _default._closed:
            _default = Runtime()
    return _default


def check(input, statement, **kw):
    return default().check(input, statement, **kw)


def pick(input, question, options, **kw):
    return default().pick(input, question, options, **kw)


def rate(input, question, levels, **kw):
    return default().rate(input, question, levels, **kw)


def select(input, question, labels, **kw):
    return default().select(input, question, labels, **kw)


def ask_(input, items, **kw):
    return default().ask(input, items, **kw)


def load(path, draft: bool = False, allow_model_change: bool = False) -> Switch:
    return default().load(path, draft, allow_model_change)
