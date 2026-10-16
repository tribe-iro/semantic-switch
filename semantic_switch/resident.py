from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .errors import ConfigError
from .results import EMPTY, freeze, thaw

REPRESENTATIONS = ("base", "runtime-adapter", "merged", "hosted")
SCALE = 1.0


@dataclass(frozen=True, slots=True)
class Resources:
    host: int = 0
    devices: Mapping[str, int] = EMPTY

    def __post_init__(self):
        devices = {name: value for name, value in dict(self.devices).items() if value}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.host, *devices.values())):
            raise ConfigError("resource estimates are nonnegative byte counts")
        object.__setattr__(self, "devices", MappingProxyType(devices))

    def __add__(self, other):
        if not isinstance(other, Resources):
            return NotImplemented
        devices = dict(self.devices)
        for name, value in other.devices.items():
            devices[name] = devices.get(name, 0) + value
        return Resources(self.host + other.host, devices)

    def __radd__(self, other):
        return self if other == 0 else NotImplemented

    def covers(self, other: Resources) -> bool:
        return other.host <= self.host and all(value <= self.devices.get(name, 0) for name, value in other.devices.items())

    def record(self) -> dict:
        return {"host": self.host, "devices": dict(self.devices)}


@dataclass(frozen=True, slots=True)
class Adapter:
    path: str
    sha256: str
    rank: int
    alpha: float
    targets: tuple[str, ...]
    file_bytes: int
    tensors: tuple[tuple[str, int], ...]
    header: int

    @classmethod
    def verified(cls, path: str, base: str, sha256: str | None = None) -> Adapter:
        from . import files
        from .backend import lora_layout
        try:
            layout = lora_layout(path, base)
        except (OSError, ValueError, KeyError, UnicodeDecodeError) as error:
            raise ConfigError(f"adapter {path}: {error}") from None
        actual = files.sha256_file(path)
        if sha256 is not None and actual != sha256:
            raise ConfigError(f"adapter {path}: content {actual} differs from its verified identity {sha256}")
        return cls(path, actual, layout["rank"], layout["alpha"], tuple(layout["targets"]), layout["file_bytes"], tuple(sorted(layout["tensors"].items())),
                   layout["header"])

    @property
    def tensor_bytes(self) -> int:
        return sum(size for _, size in self.tensors)

    def record(self) -> dict:
        from .compiler import digest
        return {"sha256": self.sha256, "rank": self.rank, "alpha": self.alpha, "targets": digest(list(self.targets)), "scale": SCALE}


@dataclass(frozen=True, slots=True)
class Binding:
    name: str
    representation: str
    identity: Mapping
    execution: Mapping | None = None
    scope: tuple[str, ...] = ()

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ConfigError(f"binding {self.name}: unknown representation {self.representation!r}; supported: {', '.join(REPRESENTATIONS)}")
        if (self.representation == "runtime-adapter") != isinstance(self.identity.get("lora"), Mapping):
            raise ConfigError(f"binding {self.name}: a runtime-adapter binding names exactly one adapter; other representations activate none")
        for name in ("identity", "execution"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        object.__setattr__(self, "scope", tuple(sorted(self.scope)))

    def record(self) -> dict:
        from .api import behavior_identity
        return {"representation": self.representation, "identity": behavior_identity(thaw(self.identity)), "execution": thaw(self.execution),
                "scope": list(self.scope)}

    @property
    def computation(self) -> str:
        return binding_identity(self.identity, self.execution)


def binding_identity(identity, execution) -> str:
    from .api import behavior_identity
    from .compiler import digest
    return digest({"identity": behavior_identity(thaw(identity)), "execution": thaw(execution)})


class AdapterPool:
    def __init__(self, scorer, *, max_adapters: int, job, placement):
        self.scorer, self.max_adapters, self.job = scorer, max_adapters, job
        self._placement, self._placed = placement, None
        self._entries: OrderedDict = OrderedDict()
        self.stats = {"loads": 0, "hits": 0, "evictions": 0, "underestimates": 0, "unestimated": 0}

    def allocation(self, adapter: Adapter):
        if self.job is not None:
            if self._placed is None:
                self._placed = self._placement()
            return self._placed.lora(adapter)
        from .backend import advise
        try:
            if self._placed is None:
                self._placed = self._placement()
            return self._placed.lora(adapter)
        except (ConfigError, OSError, ValueError, KeyError) as error:
            self.stats["unestimated"] += 1
            advise("adapter-estimate", f"adapter {adapter.sha256[:12]}: allocation estimate unavailable: {error}")
            return None

    @property
    def resident(self) -> Resources:
        return sum((entry["allocation"].resources for entry in self._entries.values() if entry["allocation"] is not None), Resources())

    def placement(self) -> dict:
        return {key: {"file_bytes": entry["adapter"].file_bytes, "tensor_bytes": entry["adapter"].tensor_bytes,
                      "charged": None if entry["allocation"] is None else entry["allocation"].resources.record(),
                      "buffers": dict(entry["lora"].buffers), "users": entry["users"]}
                for key, entry in self._entries.items()}

    @contextmanager
    def use(self, adapter: Adapter | None):
        if adapter is None:
            yield None
            return
        entry = self._acquire(adapter)
        try:
            yield entry["lora"]
        finally:
            entry["users"] -= 1

    def _acquire(self, adapter: Adapter) -> dict:
        from .backend import Lora, advise
        entry = self._entries.get(adapter.sha256)
        if entry is not None:
            self._entries.move_to_end(adapter.sha256)
            self.stats["hits"] += 1
            entry["users"] += 1
            return entry
        while len(self._entries) >= self.max_adapters:
            if not self._evict():
                raise ConfigError(f"adapter pool holds {len(self._entries)} in-use adapters (max_adapters {self.max_adapters}); "
                                  f"cannot load {adapter.sha256[:12]} without evicting an adapter in use")
        allocation = self.allocation(adapter)
        charge = ExitStack()
        while self.job is not None:
            need = allocation.resources
            try:
                charge.enter_context(self.job.join(need.host, need.devices, use=f"adapter {adapter.sha256[:12]}"))
                break
            except ConfigError:
                if not self._evict():
                    raise
        try:
            lora = Lora(self.scorer.weights, adapter.path, adapter.sha256)
        except BaseException:
            charge.close()
            raise
        escaped = allocation and allocation.escapes(lora.buffers)
        if escaped:
            self.stats["underestimates"] += 1
            if self.job is not None:
                self._release({"lora": lora, "charge": charge})
                raise ConfigError(f"{escaped}; the adapter was detached and freed: re-measure the profile before serving runtime adapters "
                                  f"inside model job {self.job.owner}")
            advise("adapter-allocation", escaped)
        entry = self._entries[adapter.sha256] = {"adapter": adapter, "allocation": allocation, "lora": lora, "charge": charge, "users": 1}
        self.stats["loads"] += 1
        return entry

    def _evict(self) -> bool:
        victim = next((key for key, entry in self._entries.items() if not entry["users"]), None)
        if victim is None:
            return False
        self._release(self._entries.pop(victim))
        self.stats["evictions"] += 1
        return True

    def _release(self, entry):
        try:
            self.scorer.forget(entry["lora"])
        finally:
            try:
                entry["lora"].close()
            finally:
                entry["charge"].close()

    def close(self):
        errors = []
        while self._entries:
            try:
                self._release(self._entries.popitem(last=False)[1])
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]


class ResidentBase:
    def __init__(self, scorer, *, max_adapters: int, job, placement):
        self.scorer = scorer
        self.pool = AdapterPool(scorer, max_adapters=max_adapters, job=job, placement=placement)
        self.closed = False

    def view(self, adapter: Adapter | None, identity: str | None = None) -> BoundScorer:
        return BoundScorer(self, adapter, identity)

    def close(self):
        with getattr(getattr(self.scorer, "ctx", None), "lock", None) or nullcontext():
            if self.closed:
                return
            self.closed = True
            try:
                self.pool.close()
            finally:
                self.scorer.close()


class BoundScorer:
    __slots__ = ("resident", "adapter", "binding_identity")

    def __init__(self, resident: ResidentBase, adapter: Adapter | None, identity: str | None):
        object.__setattr__(self, "resident", resident)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "binding_identity", identity)

    def __getattr__(self, name):
        return getattr(self.resident.scorer, name)

    def __setattr__(self, name, value):
        setattr(self.resident.scorer, name, value)

    @contextmanager
    def _bound(self):
        scorer = self.resident.scorer
        with getattr(scorer, "operation", nullcontext)(), self.resident.pool.use(self.adapter) as lora:
            yield scorer, {} if lora is None and self.binding_identity is None else {"lora": lora, "binding": self.binding_identity}

    def score(self, requests, *, reference: bool = False):
        with self._bound() as (scorer, bound):
            return scorer.score(requests, **({"reference": True} if reference else {}), **bound)

    def selftest(self, adapter, texts, questions) -> dict:
        with self._bound() as (scorer, bound):
            return scorer.selftest(adapter, texts, questions, **bound)

    def close(self):
        self.resident.close()
