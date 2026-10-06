from __future__ import annotations

from .. import files, quality


def doctor(rt, role: str = "judge", probe: bool = False) -> dict:
    judge = rt.judge(role)
    checks = judge.selftest()
    ok = all(c["ok"] for c in checks.values())
    report = {"role": role, "model": judge.identity, "ok": ok, "device": "gpu" if judge.accelerated else "cpu",
              "shared_prefix": getattr(judge.scorer, "shared", None), "checks": checks}
    store_path = files.cache_dir() / "doctor.json"
    store = files.read_json(store_path) or {}
    store[rt.cache_key(role)] = {"ok": ok, "checks": checks, "at": files.now()}
    files.write_json_atomic(store_path, store)
    if probe:
        report["quality"] = quality.probe(rt, role)
    return report
