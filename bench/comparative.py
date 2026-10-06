"""Comparative choice: does the pick follow the stated goal, and survive option reordering?

python -m bench.comparative
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from semantic_switch import Model, Runtime
from semantic_switch.quality import SCENARIOS




def main():
    rows = []
    with Runtime({"judge": Model(sys.argv[1])} if len(sys.argv) > 1 else None) as rt:
        for state, options, *goals in SCENARIOS:
            for question, want in goals:
                for order in (list(options), list(reversed(options))):
                    r = rt.pick(state, question, {k: options[k] for k in order})
                    top = max(r.scores, key=r.scores.get)
                    rows.append({"q": question, "want": want, "top": top, "p": round(r.scores[top], 3),
                                 "status": r.status, "value": r.value})
    correct = sum(r["top"] == r["want"] for r in rows)
    settled = [r for r in rows if r["status"] == "ok"]
    stable = sum(rows[i]["top"] == rows[i + 1]["top"] for i in range(0, len(rows), 2))
    report = {"items": len(rows), "top1_correct": correct, "settled": len(settled),
              "settled_wrong": sum(r["value"] != r["want"] for r in settled), "order_stable_pairs": stable,
              "pairs": len(rows) // 2,
              "goal_sensitive_scenarios": sum(rows[i]["top"] != rows[i + 2]["top"] for i in range(0, len(rows), 4)),
              "scenarios": len(SCENARIOS)}
    print(json.dumps(report, indent=1))
    for r in rows:
        if r["top"] != r["want"]:
            print("miss", r)
    Path(f"bench/results/comparative{'-' + sys.argv[1].split(':')[-1] if len(sys.argv) > 1 else ''}.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
