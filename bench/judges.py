"""Judge ladder: accuracy, coverage at a precision target, calibration, quantisation shift, latency, memory.

Run from the repository root:
    python -m bench.judges --judges "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf,unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
Each completed judge is appended to <output>.journal.jsonl with per-item probabilities; an interrupted run resumes
the missing judges, and the summary in <output> is recomputed from all recorded judges.
Acceptance follows execution: low answer mass is never settled.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from semantic_switch import Model
from semantic_switch.judgment import Policy
from semantic_switch.artifacts import artifact_digest
from bench import quality
from bench import diagnostics
from semantic_switch.metrics import Progress
from bench.outcomes import configuration, failure, finish, measure, resumable, resumed, runtime, work_key

OUT = Path("bench/results/judges.json")
GRID = np.round(np.arange(0.50, 1.0, 0.01), 2)


def run(rt, model: Model) -> dict:
    load = measure(lambda: bool(rt.judge()), runtime=rt)
    if load.error:
        return failure(load, source=model.source, stage="load")
    scored = measure(lambda: quality.run(rt), runtime=rt)
    if scored.error:
        return failure(scored, source=model.source, stage="score")
    return {"outcome": "predicted", "provenance": diagnostics.provenance(rt, list(quality.check_items()), boundary="public SDK"), "source": model.source,
            "adapter": rt.info()["identity"]["template"], "shared": rt.info()["capabilities"]["shared"],
            "file_mb": round(os.path.getsize(model.path()) / 2**20), "load_peak_rss_mb": round(load.peak_rss_bytes / 2**20),
            "peak_rss_mb": round(scored.peak_rss_bytes / 2**20), "load_s": round(load.ms / 1000, 1),
            "threads": model.threads or os.cpu_count(), **scored.output}


def _best_threshold(rows, target: float, min_settled: int = 10):
    return next((float(t) for t in GRID if (s := quality.selective(rows, float(t)))["accepted"] >= min_settled and s["precision"] >= target), None)


def coverage_at(rows, target: float = 0.95) -> dict:
    names = sorted({r["fact"] for r in rows})
    halves = (set(names[0::2]), set(names[1::2]))
    settled = correct = 0
    for tune, test in (halves, halves[::-1]):
        t = _best_threshold([r for r in rows if r["fact"] in tune], target)
        if t is None:
            continue
        s = quality.selective([r for r in rows if r["fact"] in test], t)
        settled, correct = settled + s["accepted"], correct + s["correct"]
    return {"coverage": round(settled / len(rows), 3), "precision": round(correct / settled, 3) if settled else None}


def summarize(res: dict) -> dict:
    return {"model": Path(res["source"].split(":")[-1]).stem, "adapter": res["adapter"], "file_mb": res["file_mb"],
            "peak_rss_mb": res["peak_rss_mb"], **quality.summarize(res), "at_95": coverage_at(res["checks"])}


def family(name: str) -> str:
    return re.sub(r"[-_.]?(q\d.*|bf16|f16|ud-.*)$", "", name.lower().replace("_q4_0-it", "-it-q4_0"))


def shift(a: dict, b: dict) -> dict:
    pa = np.array([r["p"] if r["p"] is not None else np.nan for r in a["checks"]])
    pb = np.array([r["p"] if r["p"] is not None else np.nan for r in b["checks"]])
    return {"mean_abs_dp": round(float(np.nanmean(np.abs(pa - pb))), 3),
            "p95_abs_dp": round(float(np.nanpercentile(np.abs(pa - pb), 95)), 3),
            "flips_0.5": int(((pa >= 0.5) != (pb >= 0.5)).sum()),
            "settle_changes_0.9": sum(quality.decision(x, Policy(.9, .9)) != quality.decision(y, Policy(.9, .9)) for x, y in zip(a["checks"], b["checks"]))}


def analyze(rows: list) -> dict:
    results = {row["name"]: row for row in rows if row["outcome"] != "failed"}
    shifts = {}
    for k, v in results.items():
        ref = [r for r in results if r != k and family(r) == family(k) and "q8" in r.lower()]
        if ref and "q8" not in k.lower():
            shifts[f"{k} vs {ref[0]}"] = shift(results[ref[0]], v)
    return {"summary": {k: summarize(v) for k, v in results.items()}, "quantisation_shift": shifts,
            "failed": sorted(row["name"] for row in rows if row["outcome"] == "failed"), "judges": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--adapter", default="chat")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    models = {Path(source.split(":")[-1]).stem + ("" if args.adapter == "chat" else f"@{args.adapter}"): Model(source, adapter=args.adapter, threads=args.threads)
              for source in args.judges.split(",")}
    cohort = artifact_digest([quality.SUITE_VERSION, quality.CHECKS, quality.PICKS, quality.TOPICS, quality.TOPIC_QUESTION])
    keys = {work_key(name): name for name in models}
    with resumable(args.output, configuration("bench.judges", args, *models.values(), cohort=cohort), fresh=args.fresh) as journal:
        todo = journal.pending(keys)
        progress = Progress("bench.judges", len(keys), unit="judges", resumed=resumed(journal, keys, todo))
        for key in todo:
            with runtime(models[keys[key]]) as rt:
                res = {"name": keys[key], **run(rt, models[keys[key]])}
            journal.commit([(key, res)])
            progress.record([res])
        progress.finish()
        print(json.dumps({k: v for k, v in finish(journal, args.output, analyze).items() if k != "judges"}, indent=2))


if __name__ == "__main__":
    main()
