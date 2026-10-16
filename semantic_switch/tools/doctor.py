from __future__ import annotations

from .. import files

BOUNDARY = "renderer/readout/execution conformance; not domain accuracy"


def doctor(rt, role: str = "judge") -> dict:
    judge = rt.judge(role)
    base = {"role": role, "model": judge.identity, "requested_device": "gpu" if rt.accelerated(role) else "cpu", "boundary": BOUNDARY}
    if not hasattr(judge, "selftest"):
        return {**base, "status": "unsupported", "ok": True, "checks": {}, "cached": False,
                "reason": "this binding exposes no native conformance self-test; no probe request is sent"}
    key = rt.cache_key(role)
    known = files.read_cached("doctor", key)
    cached = bool(known) and known.get("scope") == "complete"
    checks = known["checks"] if cached else judge.selftest()
    ok = bool(checks) and all(c["ok"] for c in checks.values())
    if not cached:
        files.write_cached("doctor", key, {"ok": ok, "scope": "complete", "checks": checks, "at": files.now()})
    return {**base, "status": "passed" if ok else "failed", "ok": ok, "checks": checks, "cached": cached,
            "shared_prefix": rt.info(role)["capabilities"]["shared"]}
