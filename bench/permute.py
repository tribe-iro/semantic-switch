"""Order debiasing for picks: one order vs the average of forward and reversed option orders.

python -m bench.permute
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from semantic_switch import Pick, Runtime, quality

from .fresh import FRESH_PICKS, FRESH_QUESTION, FRESH_TOPICS
from .positions import DISTRACTORS


def suites():
    yield "selection", quality.PICKS, quality.TOPICS, quality.TOPIC_QUESTION
    yield "fresh", FRESH_PICKS, FRESH_TOPICS, FRESH_QUESTION
    wide = {**quality.TOPICS, **{f"d{i}": d for i, d in enumerate(DISTRACTORS)}}
    yield "selection-20", quality.PICKS, wide, quality.TOPIC_QUESTION


def main():
    report = {}
    with Runtime() as rt:
        rt.check("warm", "x")
        for name, items, topics, question in suites():
            rows = []
            for text, answer in items:
                state = quality.render(text)
                fwd = Pick(question, topics, min_score=0.9)
                rev = Pick(question, dict(reversed(list(topics.items()))), min_score=0.9)
                t0 = time.perf_counter()
                one = rt.ask(state, {"f": fwd})["f"]
                t1 = time.perf_counter()
                both = rt.ask(state, {"f": fwd, "r": rev})
                t2 = time.perf_counter()
                avg = {k: (both["f"].scores[k] + both["r"].scores[k]) / 2 for k in topics}
                top = max(avg, key=avg.get)
                rev_top = max(both["r"].scores, key=both["r"].scores.get)
                rows.append({"answer": answer, "one": one.value if one.status == "ok" else None,
                             "avg": top if avg[top] >= 0.9 else None, "flip": rev_top != max(one.scores, key=one.scores.get),
                             "ms_one": (t1 - t0) * 1000, "ms_two": (t2 - t1) * 1000})
            def score(key):
                settled = [r for r in rows if r[key] is not None]
                return {"settled": len(settled), "wrong": sum(r[key] != r["answer"] for r in settled)}
            report[name] = {"n": len(rows), "one_order": score("one"), "two_orders": score("avg"),
                            "top_answer_flips_with_order": sum(r["flip"] for r in rows),
                            "ms_one": round(statistics.median(r["ms_one"] for r in rows), 1),
                            "ms_two": round(statistics.median(r["ms_two"] for r in rows), 1)}
            print(name, json.dumps(report[name]), flush=True)
    Path("bench/results/permute.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
