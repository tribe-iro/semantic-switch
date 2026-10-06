"""E3: CPU vs GPU (Vulkan) for the default judge: load, check latency, multi-question ask, a switch, fresh-suite quality.

python -m bench.gpu
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from semantic_switch import Check, Model, Runtime, quality
from semantic_switch.api import DEFAULT_JUDGE

from .fresh import FRESH

FILL = " I have been a customer for years and the service was always fine until this month, honestly."
SHORT = "[customer] message: I was charged twice, please refund one."
LONG = SHORT + FILL * 10
CLAIM = "The customer asks to receive money back for a charge."
ASK = {f"q{i}": Check(c) for i, c in enumerate((CLAIM, "The customer threatens legal action against us.",
                                                 "The customer is angry.", "The customer mentions a date."))}
ROUTE = [{"subject": "Help", "body": b} for b in ("My lawyer will file a lawsuit on Monday.",
                                                   "The app crashes every time and I need it today.",
                                                   "Thanks for yesterday, just saying hello.")]


def median_ms(fn, n=5, fresh=None):
    out = []
    for _ in range(n):
        if fresh:
            fresh()
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(out), 1)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JUDGE
    devices = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cpu", "gpu"]
    report = {"source": source}
    items = list(quality.check_items(FRESH))
    for device in devices:
        t0 = time.perf_counter()
        rt = Runtime({"judge": Model(source, device=device)})
        judge = rt.judge()
        load = (time.perf_counter() - t0) * 1000
        rt.check("warm", "x")
        clear = judge.scorer.invalidate
        row = {"load_ms": round(load), "accelerated": judge.accelerated,
               "check_20": median_ms(lambda: rt.check(SHORT, CLAIM), fresh=clear),
               "check_261": median_ms(lambda: rt.check(LONG, CLAIM), fresh=clear),
               "check_261_cached_input": median_ms(lambda: rt.check(LONG, CLAIM)),
               "ask4_261": median_ms(lambda: rt.ask(LONG, ASK), fresh=clear)}
        sw = rt.load(Path("switches/route_ticket.yaml"), draft=True)
        sw(ROUTE[0])
        row["switch_ms"] = [median_ms(lambda d=d: sw(d), n=3, fresh=clear) for d in ROUTE]
        t0 = time.perf_counter()
        ps = np.array([rt.check(quality.render(it["text"]), it["claim"], excludes=it["excludes"]).p_true for it in items])
        row["fresh_ms_per_check"] = round((time.perf_counter() - t0) * 1000 / len(items), 1)
        y = np.array([it["label"] for it in items])
        settled = (ps >= .9) | (ps <= .1)
        row["fresh_accuracy"] = round(float(((ps >= .5) == y).mean()), 3)
        row["fresh_precision_0.9"] = round(float(((ps >= .5) == y)[settled].mean()), 3)
        row["fresh_coverage_0.9"] = round(float(settled.mean()), 3)
        report[device] = row
        report.setdefault("p", {})[device] = ps.tolist()
        print(device, json.dumps(row), flush=True)
        rt.close()
    if len(devices) == 2:
        a, b = np.array(report["p"][devices[0]]), np.array(report["p"][devices[1]])
        report["max_p_diff"] = round(float(np.abs(a - b).max()), 4)
        report["answer_flips"] = int(((a >= .5) != (b >= .5)).sum())
        print("max p diff", report["max_p_diff"], "flips", report["answer_flips"])
    name = source.split(":")[-1].removesuffix(".gguf")
    Path(f"bench/results/gpu-{name}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
