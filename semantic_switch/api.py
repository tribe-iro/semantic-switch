from __future__ import annotations

import contextvars
import functools
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from functools import cache
from contextlib import ExitStack, contextmanager, nullcontext
from collections.abc import Sequence
from typing import TYPE_CHECKING
from types import MappingProxyType
import math
import hashlib
import asyncio
import shlex
import sys
import tempfile
from pathlib import Path

from . import files
from .errors import ConfigError, suggest
from .inline import Check, Pick, Rate, Select, ask, ask_many
from .judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS, LlamaJudge, validate_requests
from .protocols import ScoreBlock, ScoreLedger, ScoreRequest
from .switch import Switch

if TYPE_CHECKING:
    from .artifacts import Provenance
    from .protocols import Judge

DEFAULT_JUDGE = "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf"
PRECISE_JUDGE = "google/gemma-4-E4B-it-qat-q4_0-gguf:gemma-4-E4B_q4_0-it.gguf"
ADAPTERS = ("chat", "qwen3")
DEVICES = ("auto", "cpu", "gpu")
HOST_RESERVE = 2 << 30
MEMINFO = Path("/proc/meminfo")
CGROUP_ROOT = Path("/sys/fs/cgroup")
SELF_CGROUP = Path("/proc/self/cgroup")


@dataclass(frozen=True)
class Model:
    source: str
    adapter: str = "chat"
    device: str = "auto"
    n_ctx: int | None = None
    n_batch: int = 512
    threads: int | None = None
    max_input_tokens: int = 1024
    prefix_slots: int = 4
    quality_profile: str | None = None
    repack: bool = True
    provenance: Provenance | None = field(default=None, repr=False)

    def __post_init__(self):
        from .backend import validate_native_integer
        for name, value, allowed in (("adapter", self.adapter, ADAPTERS), ("device", self.device, DEVICES)):
            if value not in allowed:
                raise ConfigError(f"unknown {name} {value!r}{suggest(value, allowed)}")
        if not isinstance(self.repack, bool):
            raise ConfigError("repack must be boolean")
        for name in ("n_ctx", "n_batch", "threads", "max_input_tokens", "prefix_slots"):
            try:
                validate_native_integer(name, getattr(self, name), optional=name in ("n_ctx", "threads"))
            except ValueError as error:
                raise ConfigError(str(error)) from None
        try:
            self.capacity()
        except ValueError as error:
            raise ConfigError(str(error)) from None
        if self.quality_profile:
            from .artifacts import load_quality_profile
            load_quality_profile(self.quality_profile)
        from .artifacts import Provenance
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ConfigError("provenance must come from Model.from_specialization or a verified pipeline-built base")

    @classmethod
    def from_specialization(cls, receipt, **settings) -> Model:
        from .artifacts import load_receipt
        loaded = load_receipt(receipt)
        return native_spec(loaded.gguf, {**loaded.native, **settings}, loaded.provenance)

    def path(self) -> str:
        from .backend import require_local
        require_local("llama_cpp", "huggingface_hub")
        return _resolve(self.source)

    @property
    def gpu_layers(self) -> int:
        return 0 if self.device == "cpu" else 999

    def capacity(self):
        from .backend import capacity
        return capacity(self.n_ctx, self.n_batch, 1, self.prefix_slots)

    def _host_bytes(self) -> dict:
        from .backend import gguf_metadata
        path = self.path()
        meta = gguf_metadata(path)
        arch = meta["general.architecture"]
        layout = self.capacity()
        n_ctx = layout.n_ctx
        layers = meta[f"{arch}.block_count"] - meta.get(f"{arch}.attention.shared_kv_layers", 0)
        heads = meta[f"{arch}.attention.head_count_kv"]
        heads = heads if isinstance(heads, list) else [heads] * layers
        dim = meta.get(f"{arch}.attention.key_length", meta[f"{arch}.embedding_length"] // meta[f"{arch}.attention.head_count"])
        full = dim + meta.get(f"{arch}.attention.value_length", dim)
        swa = meta.get(f"{arch}.attention.key_length_swa", dim) + meta.get(f"{arch}.attention.value_length_swa", dim)
        pattern = meta.get(f"{arch}.attention.sliding_window_pattern", [False] * layers)
        return {"weights": os.path.getsize(path),
                "kv": n_ctx * sum(heads[i] * (swa if pattern[i] else full) * 2 for i in range(layers)),
                "output": len(meta["tokenizer.ggml.tokens"]) * layout.n_outputs_max * 4 * 2}

    def estimated_host_bytes(self) -> int:
        return sum(self._host_bytes().values())


NATIVE_SETTINGS = ("adapter", "device", "n_ctx", "n_batch", "threads", "max_input_tokens", "prefix_slots", "quality_profile", "repack")


def native_settings(model: Model) -> dict:
    return {name: getattr(model, name) for name in NATIVE_SETTINGS}


def native_spec(source, settings, provenance: Provenance | None = None) -> Model:
    if not isinstance(settings, dict) or set(settings) - set(NATIVE_SETTINGS):
        unknown = sorted(set(settings) - set(NATIVE_SETTINGS)) if isinstance(settings, dict) else settings
        raise ConfigError(f"unknown native settings {unknown}; supported: {', '.join(NATIVE_SETTINGS)}")
    return Model(str(source), **settings, provenance=provenance)


def judge_from(source, *, provider=None, key=None, transport=None, own_transport=False, timeout=60, batch_inputs=8, **native_settings):
    if not isinstance(source, str) or not source.strip():
        raise ConfigError("judge source must be nonempty text: a GGUF path, repo:file or openrouter:MODEL")
    for name, value in (("provider", provider), ("key", key)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"{name} must be nonempty text")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < math.inf:
        raise ConfigError("timeout must be a positive number of seconds")
    if isinstance(batch_inputs, bool) or not isinstance(batch_inputs, int) or batch_inputs < 1:
        raise ConfigError("batch_inputs must be a positive integer")
    if not source.startswith("openrouter:"):
        hosted = [name for name, value in (("provider", provider), ("key", key), ("transport", transport)) if value is not None]
        hosted += ["own_transport"] if own_transport else []
        hosted += [name for name, value, default in (("timeout", timeout, 60), ("batch_inputs", batch_inputs, 8)) if value != default]
        if hosted:
            raise ConfigError(f"{', '.join(hosted)} apply only to hosted judges (openrouter:MODEL); {source} is a local GGUF judge")
        for name in native_settings:
            if name not in NATIVE_SETTINGS:
                raise ConfigError(f"unknown native setting {name!r}{suggest(name, NATIVE_SETTINGS)}")
        return Model(source, **native_settings)
    if native_settings:
        raise ConfigError(f"{', '.join(sorted(native_settings))} apply only to local GGUF judges; {source} is hosted")
    if key is not None and transport is not None:
        raise ConfigError("key applies only to the built-in transport; a supplied transport carries its own credentials")
    if own_transport and transport is None:
        raise ConfigError("own_transport transfers a supplied transport; the built-in transport is always owned")
    model = source.removeprefix("openrouter:")
    if not model.strip() or "@" in model:
        raise ConfigError("use openrouter:MODEL and pass the provider separately (--provider PROVIDER or provider=...)")
    from .hosted import OpenRouterJudge
    return OpenRouterJudge(model, provider=provider, key=key, transport=transport, own_transport=own_transport, timeout=timeout,
                           batch_inputs=batch_inputs)


def _model_lock() -> Path:
    explicit, runtime = os.environ.get("SEMANTIC_MODEL_LOCK"), os.environ.get("XDG_RUNTIME_DIR")
    if explicit:
        return Path(explicit)
    if runtime:
        return Path(runtime) / "semantic-switch" / "model-job.lock"
    directory = Path(tempfile.gettempdir()) / f"semantic-switch-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory / "model-job.lock"


def available_host_bytes() -> int:
    try:
        fields = dict(line.split(":", 1) for line in MEMINFO.read_text().splitlines())
    except FileNotFoundError:
        raise ConfigError(f"{MEMINFO} is unavailable; host memory admission is unsupported on this platform") from None
    available = sum(int(fields[name].split()[0]) * 1024 for name in ("MemAvailable", "GPUReclaim") if name in fields)
    try:
        unified = [line[3:].strip().lstrip("/") for line in SELF_CGROUP.read_text().splitlines() if line.startswith("0::")]
    except FileNotFoundError:
        unified = []
    for group in unified:
        group = CGROUP_ROOT / group
        for directory in (group, *group.parents):
            limit = directory / "memory.max"
            if limit.exists() and limit.read_text().strip() != "max":
                stat = dict(line.split() for line in (directory / "memory.stat").read_text().splitlines())
                working_set = int((directory / "memory.current").read_text()) - int(stat["inactive_file"])
                available = min(available, int(limit.read_text()) - working_set)
            if directory == CGROUP_ROOT:
                break
    return available


class ModelJob:
    def __init__(self, lease: files.Lease, estimates: dict, available: int, device_bytes: int):
        self._lease, self.owner, self.pid = lease, lease.owner, lease.pid
        self.estimates = MappingProxyType(dict(estimates))
        self.host_bytes = sum(self.estimates.values())
        self.device_bytes = device_bytes
        self.available_bytes = available
        self.state = "open"
        self._resident: dict = {}
        self._peak = (0, 0)
        self._condition = threading.Condition()

    def _open(self):
        if self.pid != os.getpid():
            raise ConfigError(f"model job {self.owner} belongs to process {self.pid}; acquire a new job in process {os.getpid()}")
        if self.state != "open":
            raise ConfigError(f"model job {self.owner} is {self.state}; acquire a new job")

    @contextmanager
    def join(self, host_bytes: int, device_bytes: int = 0, *, use: str = "participant"):
        token = object()
        with self._condition:
            self._open()
            host, device = (sum(values) for values in zip((0, 0), *self._resident.values()))
            if host + host_bytes > self.host_bytes or device + device_bytes > self.device_bytes:
                raise ConfigError(f"model job {self.owner}: {use} needs {host_bytes} host and {device_bytes} device bytes beside {host}/{device} resident; "
                                  f"the declared envelope is {self.host_bytes}/{self.device_bytes}. Declare the peak before acquiring the job")
            self._resident[token] = (host_bytes, device_bytes)
            self._peak = (max(self._peak[0], host + host_bytes), max(self._peak[1], device + device_bytes))
        try:
            yield self
        finally:
            with self._condition:
                del self._resident[token]
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self.state == "closed" or self.pid != os.getpid():
                return
            self.state = "closing"
            while self._resident:
                self._condition.wait()
            self.state = "closed"
        self._lease.release()

    def diagnostics(self) -> dict:
        import resource
        with self._condition:
            return {"owner": self.owner, "pid": self.pid, "state": self.state, "lock": str(self._lease.path), "participants": len(self._resident),
                    "estimated_bytes": dict(self.estimates), "estimated_device_bytes": self.device_bytes,
                    "admitted_bytes": self.host_bytes + HOST_RESERVE, "available_bytes": self.available_bytes,
                    "resident_peak_bytes": {"host": self._peak[0], "device": self._peak[1]},
                    "observed_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "uncertainty": "estimates cover weights, buffers and declared working sets; activations are not bounded, so admission is no OOM guarantee",
                    "scope": "cooperating processes of this user sharing this lock path; not a machine-wide scheduler"}


def _estimates(models, extra) -> dict:
    out = {}
    for model in models:
        for use, value in model._host_bytes().items():
            out[f"{model.source}:{use}"] = out.get(f"{model.source}:{use}", 0) + value
    for use, value in (extra or {}).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"model job estimate {use!r} must be a nonnegative byte count")
        out[use] = value
    return out


@contextmanager
def model_job(*models: Model, extra: dict | None = None, device_bytes: int = 0, job: ModelJob | None = None):
    if job is not None:
        with job.join(sum(_estimates(models, extra).values()), device_bytes, use=", ".join(m.source for m in models) or ", ".join(extra or ())):
            yield job
        return
    estimates = _estimates(models, extra)
    lease = files.Lease(_model_lock())
    try:
        required = sum(estimates.values()) + HOST_RESERVE
        available = available_host_bytes()
        if required > available:
            raise ConfigError(f"model job requires {required} bytes of host memory including the {HOST_RESERVE}-byte reserve; "
                              f"{available} bytes are available. Run one model job per host under a memory limit: "
                              f"systemd-run --user --scope -q -p MemoryMax={-(-required // (1 << 30))}G -p MemorySwapMax=0 {shlex.join(sys.orig_argv)}")
        handle = ModelJob(lease, estimates, available, device_bytes)
    except BaseException:
        lease.release()
        raise
    try:
        yield handle
    finally:
        handle.close()


@cache
def _resolve(source: str) -> str:
    if os.path.exists(source):
        return source
    from huggingface_hub import hf_hub_download, list_repo_files, try_to_load_from_cache

    repo, _, spec = source.partition(":")
    if spec.endswith(".gguf"):
        cached = try_to_load_from_cache(repo, spec)
        return cached if isinstance(cached, str) else hf_hub_download(repo, spec)
    names = sorted(f for f in list_repo_files(repo) if f.endswith(".gguf") and spec.lower() in f.lower()
                   and not f.startswith(("mmproj", "mtp")))
    if len(names) != 1:
        found = ", ".join(names) or "nothing"
        raise ConfigError(f"model source {source!r} must name exactly one GGUF file; it matches {found}")
    return hf_hub_download(repo, names[0])


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


def behavior_identity(identity):
    return {key: value for key, value in identity.items() if key != "file"}


MODE = ("shared", "grouping", "serial")
EXECUTION = ("device", "tensor_types", "flash_attn", "type_k", "type_v", "use_extra_bufts", "repacked", "n_ctx", "n_batch", "n_ubatch",
             "n_seq_max", "n_outputs_max", "swa_full", "kv_unified", "prefix_slots", "scheduling") + MODE


def execution_identity(realized: dict) -> dict | None:
    return {**{key: realized[key] for key in EXECUTION}, "build": _build()} if realized else None


class Runtime:
    @classmethod
    def from_specialization(cls, receipt, *, on_decision=None, job: ModelJob | None = None) -> Runtime:
        from .artifacts import load_receipt
        loaded = load_receipt(receipt)
        runtime = cls({"judge": native_spec(loaded.gguf, dict(loaded.native), loaded.provenance)}, on_decision=on_decision, job=job)
        runtime._deployment = loaded
        return runtime

    def __init__(self, models: dict | None = None, *, on_decision=None, own_judges=False, job: ModelJob | None = None):
        if on_decision is not None and not callable(on_decision):
            raise ConfigError("on_decision must be callable")
        if not isinstance(own_judges, bool):
            raise ConfigError("own_judges must be boolean")
        if job is not None and not isinstance(job, ModelJob):
            raise ConfigError("job must be a ModelJob from model_job(...)")
        self._own_judges = own_judges
        self._job = job
        self._jobs = ExitStack()
        self._deployment = None
        self._on_decision = on_decision
        self._event_warning = False
        bindings = dict(models or {})
        bindings.setdefault("judge", Model(DEFAULT_JUDGE))
        self._bindings = MappingProxyType(bindings)
        self._roles = {}
        for name, value in bindings.items():
            if isinstance(value, Model):
                acceleration = lambda value=value: value.device != "cpu" and _gpu()
                batching = lambda: 1
                kind = "model"
            else:
                acceleration = lambda value=value: bool(getattr(value, "accelerated", False))
                batching = lambda value=value: int(getattr(value, "batch_inputs", 1))
                kind = "judge" if callable(getattr(value, "score_many", None)) else "scorer"
            self._roles[name] = _Role(value, kind, acceleration, batching)
        self._loaded: dict = {}
        self._identities = {}
        self._identity_fingerprints = {}
        self._lock = threading.RLock()
        self._closed = False
        self._active = 0
        self._operations = threading.local()
        self._idle = threading.Condition(self._lock)

    @contextmanager
    def operation(self):
        with self._idle:
            depth = getattr(self._operations, "depth", 0)
            if self._closed and not depth:
                raise RuntimeError("runtime is closed")
            self._operations.depth = depth + 1
            self._active += 1
        try:
            yield
        finally:
            with self._idle:
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

    def cache_key(self, role: str = "judge") -> str:
        spec = self._spec(role)
        settings = {name: getattr(spec, name, None) for name in ("n_ctx", "n_batch", "threads", "max_input_tokens", "prefix_slots")}
        execution = execution_identity(self._capacity(role))
        return json.dumps({**behavior_identity(self.identity(role)), "execution": execution and {key: value for key, value in execution.items() if key not in MODE},
                           "settings": settings, "inputs": self.batch_inputs(role), "conformance": 4}, sort_keys=True)

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

                    adapter = adapter_for(getattr(spec, "adapter", "qwen3"), spec)
                    self._loaded[role] = LlamaJudge(spec, adapter, self.identity(role), self.accelerated(role),
                                                    max_input_tokens=getattr(spec, "max_input_tokens", 1024))
            return self._loaded[role]

    def _load_judge(self, role: str, spec: Model) -> LlamaJudge:
        from .adapters import adapter_for
        from .artifacts import load_quality_profile
        from .backend import LlamaScorer, shared_ok

        if spec.device == "gpu" and not _gpu():
            raise ConfigError("device='gpu' was requested, but this llama.cpp build has no GPU offload; "
                              "rebuild with GPU support or use device='auto' or 'cpu'")
        job, owned = ExitStack(), self._job is None

        def release():
            job.close()
            if owned:
                self._job = None
        try:
            if owned:
                self._job = job.enter_context(model_job(*(d.value for d in self._roles.values() if d.kind == "model")))
            job.enter_context(model_job(spec, job=self._job))
            scorer = LlamaScorer(spec.path(), n_ctx=spec.n_ctx, n_batch=spec.n_batch, threads=spec.threads,
                                 n_gpu_layers=spec.gpu_layers, prefix_slots=spec.prefix_slots, repack=spec.repack)
        except (RuntimeError, ValueError) as e:
            release()
            raise ConfigError(f"cannot load model {spec.source}: {e}") from None
        except BaseException:
            release()
            raise
        try:
            try:
                adapter = adapter_for(spec.adapter, scorer)
            except ValueError as e:
                raise ConfigError(f"model {spec.source}: {e}") from None
            identity = _local_identity(spec, scorer, adapter)
            if spec.provenance is not None and identity["sha256"] != spec.provenance.gguf_sha256:
                raise ConfigError(f"{spec.source}: weights changed since their provenance was verified ({spec.provenance.gguf_sha256}); "
                                  "reload the deployment or re-run semantic-trainer select")
            self._identities[role] = identity
            self._identity_fingerprints[role] = _file_fingerprint(spec)
            bound = LlamaJudge(scorer, adapter, self.identity(role), self.accelerated(role), self.batch_inputs(role), max_input_tokens=spec.max_input_tokens,
                               quality_profile=load_quality_profile(spec.quality_profile) if spec.quality_profile else None)
        except BaseException:
            scorer.close()
            release()
            self._identities.pop(role, None)
            self._identity_fingerprints.pop(role, None)
            raise
        self._jobs.enter_context(job)
        self._loaded[role] = bound
        if scorer.weights.hybrid:
            scorer.shared = True
        key = self.cache_key(role)
        known = files.read_cached("doctor", key)
        if known:
            scorer.shared = shared_ok(known.get("checks", {}))
        elif scorer.weights.hybrid:
            scorer.shared = True
            try:
                checks = scorer.selftest(adapter, CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS)
            except RuntimeError as error:
                scorer.shared = False
                scorer.invalidate()
                checks = {"D1 shared = reference": {"ok": False, "error": str(error)}}
            files.write_cached("doctor", key, {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "at": files.now()})
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
                from .presentation import identity as presentation_identity
                return {"source": getattr(spec, "name", type(spec).__name__),
                        "template": _template(getattr(spec, "adapter", "qwen3")), "presentation": presentation_identity()}
            identity = _local_identity(spec)
            self._identities[role] = identity
            self._identity_fingerprints[role] = fingerprint
            return dict(identity)

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

    def ask(self, state, items: dict, *, fields=None, guidance=None) -> dict:
        with self.operation():
            started = time.perf_counter() if self._on_decision is not None else None
            result = ask(self.judge, state, items, fields, guidance)
            if self._on_decision is not None:
                from .events import inline_event
                cost = {'elapsed_ms': (time.perf_counter() - started) * 1000, 'inputs': 1}
                self._emit_decision(inline_event(self, result, items, cost))
            return result

    def ask_many(self, states, items: dict, *, fields=None, guidance=None) -> list[dict]:
        with self.operation():
            started = time.perf_counter() if self._on_decision is not None else None
            if isinstance(states, (str, bytes, dict)) or not isinstance(states, Sequence):
                raise ConfigError("ask_many requires a finite list or tuple; use many for streaming")
            size = self.batch_inputs()
            results = [row for start in range(0, len(states), size)
                       for row in ask_many(self.judge, states[start:start + size], items, fields, guidance)]
            if self._on_decision is not None:
                from .events import inline_event
                cost = {'batch_elapsed_ms': (time.perf_counter() - started) * 1000, 'inputs': len(results)}
                for row in results:
                    self._emit_decision(inline_event(self, row, items, cost))
            return results

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

    async def aask(self, state, items: dict, *, fields=None, guidance=None):
        return await self._offload(self.ask, state, items, fields=fields, guidance=guidance)

    async def acheck(self, state, statement: str, **kwargs):
        return await self._offload(self.check, state, statement, **kwargs)

    async def apick(self, state, question: str, options, **kwargs):
        return await self._offload(self.pick, state, question, options, **kwargs)

    async def arate(self, state, question: str, levels, **kwargs):
        return await self._offload(self.rate, state, question, levels, **kwargs)

    async def aselect(self, state, question: str, labels, **kwargs):
        return await self._offload(self.select, state, question, labels, **kwargs)

    def _scorer(self, role):
        scorer = getattr(self.judge(role), "scorer", None)
        if scorer is None:
            raise ConfigError(f"model role {role!r} has no native scorer")
        return scorer

    def score_native(self, requests: Sequence[ScoreRequest], *, role: str = "judge", reference: bool = False) -> ScoreBlock:
        with self.operation():
            requests = validate_requests(requests)
            scorer = self._scorer(role)
            with scorer.operation() if hasattr(scorer, "operation") else nullcontext():
                block = scorer.score(requests, reference=True) if reference else scorer.score(requests)
            rows = sum(len(r.suffixes) for r in requests)
            if block.logp.shape[0] != rows or len(block.log_mass) != rows or len(block.error) != rows:
                raise ConfigError("native scorer returned incomplete ordered scores")
            return block

    @contextmanager
    def recording(self, role: str = "judge"):
        judge, scorer = self.judge(role), self._scorer(role)
        ledger = ScoreLedger(scorer)
        judge.scorer = ledger
        try:
            yield ledger
        finally:
            judge.scorer = scorer

    def invalidate(self, role: str = "judge") -> None:
        with self.operation():
            self._scorer(role).invalidate()

    def detokenize(self, tokens: Sequence[int], *, role: str = "judge") -> str:
        with self.operation():
            return self._scorer(role).detokenize(tokens)

    def info(self, role="judge") -> dict:
        with self._lock:
            descriptor = self._role(role)
            spec, judge = descriptor.value, self._loaded.get(role)
            scorer = getattr(judge, "scorer", None)
            context = getattr(scorer, "ctx", None)
            return {"identity": self.identity(role) if judge is not None or descriptor.kind != "model" else
                                {"source": spec.source, "template": spec.adapter},
                    "requested_device": spec.device if descriptor.kind == "model" else "provider",
                    "threads": spec.threads if descriptor.kind == "model" else None,
                    "build": _build() if descriptor.kind == "model" or context is not None else None, "loaded": judge is not None,
                    "observed_models": sorted(getattr(judge, "observed_models", ())) or ([judge.returned] if getattr(judge, "returned", None) else []),
                    "accelerated": self.accelerated(role), "batch_inputs": self.batch_inputs(role),
                    "max_slots": getattr(judge, "max_slots", None),
                    "capabilities": {"shared": getattr(scorer, "shared", None),
                                     "fused": getattr(scorer, "fused", None),
                                     "conformance": hasattr(judge, "selftest"),
                                     "counters": {**getattr(scorer, "stats", {}), "submissions": getattr(context, "submissions", 0), "submitted_tokens": getattr(context, "submitted_tokens", 0)}},
                    "requested_capacity": getattr(context, "requested", {}),
                    "configuration": {"max_input_tokens": getattr(judge, "max_input_tokens", None),
                                      "prefix_slots": getattr(scorer, "prefix_slots", None)},
                    "capacity": self._capacity(role)}

    def check(self, state, statement: str, *, excludes=(), fields=None, min_score=None, unstated: bool = False, guidance=None):
        return Check(statement, excludes, min_score, unstated, guidance)(state, self, fields)

    def pick(self, state, question: str, options, *, definition=None, eligible=None, allow_none=False, shortlist=None,
             fields=None, min_score=None, ids=None, focus=None, guidance=None, excludes=()):
        return Pick(question, options, definition, eligible, allow_none, shortlist, min_score, ids, focus, guidance,
                    excludes)(state, self, fields)

    def rate(self, state, question: str, levels, *, fields=None, min_score=None, guidance=None, excludes=()):
        return Rate(question, levels, min_score, guidance, excludes)(state, self, fields)

    def select(self, state, question: str, labels, *, excludes=(), unstated=False, min_score=None, guidance=None, eligible=None,
               max_claims=None, fields=None):
        return Select(question, labels, excludes=excludes, unstated=unstated, min_score=min_score, guidance=guidance,
                      eligible=eligible, max_claims=max_claims)(state, self, fields)

    def load(self, path=None, draft: bool = False, allow_model_change: bool = False) -> Switch:
        if path is not None:
            return Switch(self, path, draft, allow_model_change)
        if self._deployment is None:
            raise ConfigError("load() without a path needs Runtime.from_specialization(receipt); pass the switch path otherwise")
        if draft or allow_model_change:
            raise ConfigError("a specialization deployment loads its frozen task, policies and execution exactly; draft and allow_model_change do not apply")
        return Switch(self, self._deployment.verified_task(), False, False)

    def close(self):
        with self._lock:
            if self._closed:
                return
            if getattr(self._operations, "depth", 0):
                raise RuntimeError("cannot close runtime from inside an active operation")
            self._closed = True
            while self._active:
                self._idle.wait()
            loaded, self._loaded = self._loaded, {}
            owned = [loaded.get(role) if descriptor.kind == "model" else descriptor.value if self._own_judges else None
                     for role, descriptor in self._roles.items()]
            resources = {id(resource): resource for resource in (getattr(item, "scorer", item) for item in owned if item is not None)}
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

    def __exit__(self, *exc):
        self.close()


def _file_fingerprint(spec):
    stat = os.stat(spec.path())
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _local_identity(spec, scorer=None, adapter=None):
    path = scorer.path if scorer is not None else spec.path()
    if spec.adapter == "qwen3":
        from .adapters.qwen3 import Qwen3Adapter
        from .presentation import LETTERS, SYSTEM
        name = Qwen3Adapter.name
        prompt = (Qwen3Adapter.prefix_text, Qwen3Adapter.tail_text, SYSTEM, LETTERS, "true/True/false/False")
    else:
        from .adapters.chat import name_of, render
        if scorer is not None:
            metadata = scorer.chat()
        else:
            from .backend import Weights
            weights = Weights(path, vocab_only=True)
            try:
                metadata = weights.chat()
            finally:
                weights.close()
        name = name_of(metadata)
        prompt = ((adapter.prefix_text, adapter.tail_text) if adapter is not None else render(metadata), metadata["add_bos"], "ABCDEFGHIJKLMNOPQRST", "true/True/false/False")
    from .presentation import identity as presentation_identity, text
    prompt = (prompt, "<input>\n{rendered}\n</input>\n\n", tuple(text(question) for question in CONFORMANCE_QUESTIONS))
    return {"presentation": presentation_identity(), "file": os.path.basename(path), "sha256": files.sha256_file(path), "template": name,
            "prompt": hashlib.sha256(repr(prompt).encode()).hexdigest()}


@cache
def _template(adapter: str, path: str | None = None) -> str:
    from .adapters.chat import name_of
    from .adapters.qwen3 import Qwen3Adapter

    if adapter == "chat" and path:
        from .backend import Weights

        vocab = Weights(path, vocab_only=True)
        try:
            return name_of(vocab.chat())
        finally:
            vocab.close()
    return {"qwen3": Qwen3Adapter.name, "chat": "chat@2"}[adapter]


_default: Runtime | None = None
_default_lock = threading.Lock()


def default() -> Runtime:
    global _default
    with _default_lock:
        if _default is None or _default._closed:
            _default = Runtime()
    return _default


def check(state, statement, **kw):
    return default().check(state, statement, **kw)


def pick(state, question, options, **kw):
    return default().pick(state, question, options, **kw)


def rate(state, question, levels, **kw):
    return default().rate(state, question, levels, **kw)


def select(state, question, labels, **kw):
    return default().select(state, question, labels, **kw)


def ask_(state, items, **kw):
    return default().ask(state, items, **kw)


def load(path, draft: bool = False, allow_model_change: bool = False) -> Switch:
    return default().load(path, draft, allow_model_change)
