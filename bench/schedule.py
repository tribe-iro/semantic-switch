"""E4 follow-up: two-stage scoring latency with the input cache, on route_ticket (eager was removed after E4).

python -m bench.schedule
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from semantic_switch import Runtime

FILLER = " I have been a customer for years and the service was always fine until this month, honestly."
INPUTS = {
    "first_case_decides": "My lawyer will file a lawsuit on Monday if this is not fixed.",
    "late_case_decides": "The app crashes every time I open it and I need it working today.",
    "otherwise": "Thanks for the help yesterday, just saying hello.",
}


def main():
    report = {}
    with Runtime() as rt:
        sw = rt.load(Path("switches/route_ticket.yaml"), draft=True)
        sw({"subject": "warm", "body": "warm up"})
        for length, pad in (("20 tokens", ""), ("261 tokens", FILLER * 10)):
            for name, body in INPUTS.items():
                data = {"subject": "Help", "body": body + pad}
                row = {}
                for mode in ("staged",):
                    ms, results = [], []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        results.append(sw.run([data])[0])
                        ms.append((time.perf_counter() - t0) * 1000)
                    r = results[-1]
                    row[mode] = {"ms": round(statistics.median(ms), 1), "answer": r.answer,
                                 "judged": r.cost["judge"], "rounds": r.cost["rounds"]}
                report[f"{length}/{name}"] = row
                print(f"{length}/{name}", json.dumps(row), flush=True)
    Path("bench/results/schedule.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
