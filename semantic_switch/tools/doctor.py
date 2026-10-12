from __future__ import annotations

from .. import files
from ..errors import ConfigError


def doctor(rt, role: str = "judge", probe: bool = False) -> dict:
    if probe:
        raise ConfigError("quality measurements are owned by bench.quality; doctor checks native conformance")
    judge = rt.judge(role)
    checks = judge.selftest() if hasattr(judge, "selftest") else {}
    ok = bool(checks) and all(c["ok"] for c in checks.values())
    report = {"role": role, "model": judge.identity, "ok": ok, "requested_device": "gpu" if judge.accelerated else "cpu",
              "shared_prefix": rt.info(role)["capabilities"]["shared"], "checks": checks, "boundary": "renderer/readout/execution conformance; not domain accuracy"}
    files.write_cached("doctor", rt.cache_key(role), {"ok": ok, "checks": checks, "at": files.now()})
    return report
