from __future__ import annotations

import os
import shlex
import sys
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from . import files
from .errors import ConfigError
from .models import Model
from .resident import Resources

HOST_RESERVE = 2 << 30
MEMINFO = Path("/proc/meminfo")
CGROUP_ROOT = Path("/sys/fs/cgroup")
SELF_CGROUP = Path("/proc/self/cgroup")


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
    def __init__(self, lease: files.Lease, estimates: dict[str, Resources], available: int):
        self._lease, self.owner, self.pid = lease, lease.owner, lease.pid
        self.estimates = MappingProxyType(dict(estimates))
        self.envelope = sum(self.estimates.values(), Resources())
        self.host_bytes = self.envelope.host
        self.available_bytes = available
        self.state = "open"
        self._resident: dict = {}
        self._peak = Resources()
        self._condition = threading.Condition()

    def _open(self):
        if self.pid != os.getpid():
            raise ConfigError(f"model job {self.owner} belongs to process {self.pid}; acquire a new job in process {os.getpid()}")
        if self.state != "open":
            raise ConfigError(f"model job {self.owner} is {self.state}; acquire a new job")

    @contextmanager
    def join(self, host_bytes: int, devices: Mapping[str, int] | None = None, *, use: str = "participant"):
        token, need = object(), Resources(host_bytes, devices or {})
        with self._condition:
            self._open()
            resident = sum(self._resident.values(), Resources())
            if not self.envelope.covers(resident + need):
                raise ConfigError(f"model job {self.owner}: {use} needs {need.record()} beside {resident.record()} resident; "
                                  f"the declared envelope is {self.envelope.record()} (host bytes are physical; unified-memory device bytes are "
                                  "included in them). Declare the peak before acquiring the job")
            self._resident[token] = need
            total = resident + need
            self._peak = Resources(max(self._peak.host, total.host), {name: max(self._peak.devices.get(name, 0), total.devices.get(name, 0))
                                                                     for name in {*self._peak.devices, *total.devices}})
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
                    "estimated_bytes": {use: value.record() for use, value in self.estimates.items()}, "envelope": self.envelope.record(),
                    "admitted_bytes": self.host_bytes + HOST_RESERVE, "available_bytes": self.available_bytes,
                    "resident_peak_bytes": self._peak.record(),
                    "observed_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "uncertainty": "estimates cover weights, adapters, buffers and declared working sets under the pinned placement profile; "
                                   "compute scratch and activations are not bounded, so admission is no OOM guarantee",
                    "scope": "cooperating processes of this user sharing this lock path; not a machine-wide scheduler"}


def estimate(*models: Model, extra: dict | None = None) -> dict[str, Resources]:
    out, resident = {}, set()
    for model in models:
        for use, value in model.resources().items():
            key = (os.path.realpath(model.lora_path()) if use == "adapter" else model.resident_key(), use)
            if key not in resident:
                resident.add(key)
                name = f"{model.lora}:adapter" if use == "adapter" else f"{model.source}:{use}"
                out[name] = out.get(name, Resources()) + value
    for use, value in (extra or {}).items():
        if not isinstance(value, Resources) and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ConfigError(f"model job estimate {use!r} must be a nonnegative host byte count or Resources")
        out[use] = value if isinstance(value, Resources) else Resources(value)
    return out


def peak(*stages: dict) -> dict[str, Resources]:
    totals = [sum(estimate(extra=stage).values(), Resources()) for stage in stages]
    chosen = max(range(len(stages)), key=lambda i: totals[i].host)
    devices = {}
    for total in totals:
        for name, value in total.devices.items():
            devices[name] = max(devices.get(name, 0), value)
    shortfall = Resources(0, {name: value - totals[chosen].devices.get(name, 0) for name, value in devices.items()})
    return {**stages[chosen], **({"stages:devices": shortfall} if shortfall.devices else {})}


@contextmanager
def model_job(*models: Model, extra: dict | None = None, job: ModelJob | None = None):
    estimates = estimate(*models, extra=extra)
    if job is not None:
        need = sum(estimates.values(), Resources())
        with job.join(need.host, need.devices, use=", ".join(m.source for m in models) or ", ".join(extra or ())):
            yield job
        return
    lease = files.Lease(_model_lock())
    try:
        required = sum(estimates.values(), Resources()).host + HOST_RESERVE
        available = available_host_bytes()
        if required > available:
            raise ConfigError(f"model job requires {required} bytes of host memory including the {HOST_RESERVE}-byte reserve; "
                              f"{available} bytes are available. Run one model job per host under a memory limit: "
                              f"systemd-run --user --scope -q -p MemoryMax={-(-required // (1 << 30))}G -p MemorySwapMax=0 {shlex.join(sys.orig_argv)}")
        handle = ModelJob(lease, estimates, available)
    except BaseException:
        lease.release()
        raise
    try:
        yield handle
    finally:
        handle.close()

