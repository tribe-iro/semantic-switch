from __future__ import annotations

import statistics
import time

from ..errors import ConfigError
from ..judge import FIXTURE, FIXTURE_QUESTIONS
from ..switch import Switch
from .common import test_items


def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {"p50": round(statistics.median(s), 1), "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1)}


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def bench(rt, path=None, n: int = 20) -> dict:
    judge = rt.judge()
    claim = FIXTURE_QUESTIONS[0].text
    rt.check(FIXTURE[0], claim)
    report = {"model": judge.identity, "device": "gpu" if judge.accelerated else "cpu",
              "same_input_ms": _stats([_timed(lambda: rt.check(FIXTURE[0], claim)) for _ in range(n)]),
              "new_input_ms": _stats([_timed(lambda k=k: rt.check(f"{FIXTURE[k % 3]} ({k})", claim))
                                      for k in range(n)]),
              "new_input_4_questions_ms": _stats([_timed(lambda k=k: judge.score_many(
                  [(f"{FIXTURE[k % 3]} ({k})", FIXTURE_QUESTIONS)])) for k in range(n)])}
    if path:
        sw = Switch(rt, path)
        inputs = [it["input"] for it in test_items(sw.paths, sw.plan)[0]] or [t.input for t in sw.plan.tests]
        if not inputs:
            raise ConfigError(f"{sw.paths.yaml.name} has no test inputs or YAML tests to benchmark; "
                              f"add some with `ss label --to test`")
        inputs = (inputs * (n // len(inputs) + 1))[:n]
        sw(inputs[0])
        report["switch_ms"] = _stats([_timed(lambda d=d: sw(d)) for d in inputs])
        t0 = time.perf_counter()
        results = list(sw.many(inputs))
        report["switch_throughput_per_s"] = round(len(results) / (time.perf_counter() - t0), 2)
        report["tokens_per_decision"] = round(statistics.mean(r.cost.get("tokens", 0) for r in results), 1)
    return report
