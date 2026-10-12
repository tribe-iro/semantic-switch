"""Option-position robustness: does the answer survive moving the correct option to slot 1, 5, 10, 15 or 20?

python -m bench.positions [judge source]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, Model, files
from bench.outcomes import runtime
from bench import quality
from bench.quality import DISTRACTORS

SLOTS = (0, 4, 9, 14, 19)


def main():
    topics = quality.TOPICS
    with runtime(Model(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JUDGE)) as rt:
        rows = []
        for text, answer in quality.PICKS:
            others = [d for k, d in topics.items() if k != answer] + DISTRACTORS
            for slot in SLOTS:
                options = others[:slot] + [topics[answer]] + others[slot:]
                r = rt.pick(quality.render(text), quality.TOPIC_QUESTION, options, min_score=0.9)
                top = max(range(len(options)), key=lambda i: r.scores[i])
                rows.append({"text": text, "slot": slot + 1, "top": top == slot, "p": r.scores[slot],
                             "settled": r.status == "ok", "wrong_settled": r.status == "ok" and r.index != slot})
    by = defaultdict(list)
    for row in rows:
        by[row["slot"]].append(row)
    report = {slot: {"top1": sum(r["top"] for r in rs), "settled": sum(r["settled"] for r in rs),
                     "wrong_settled": sum(r["wrong_settled"] for r in rs),
                     "mean_p": round(sum(r["p"] for r in rs) / len(rs), 3), "n": len(rs)} for slot, rs in by.items()}
    stable = defaultdict(set)
    for row in rows:
        stable[row["text"]].add(row["top"])
    report["all_slots_correct"] = sum(s == {True} for s in stable.values())
    report["texts"] = len(stable)
    print(json.dumps(report, indent=1))
    files.publish(Path("bench/results/positions.json"), {"report": report, "rows": rows})


if __name__ == "__main__":
    main()
