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

from semantic_switch import Model
from semantic_switch.judgment import Policy
from semantic_switch.artifacts import artifact_digest
from semantic_switch.metrics import percentile
from bench.outcomes import Progress
from bench import quality
from bench.model_comparison import suite
from bench.outcomes import configuration, failure, finish, measure, output_argument, resumable, resumed, runtime, work_key


def run(rt, model: Model) -> dict:
    load = measure(lambda: bool(rt.judge()), runtime=rt)
    if load.error:
        return failure(load, source=model.source, stage="load")
    scored = measure(lambda: suite(rt), runtime=rt)
    if scored.error:
        return failure(scored, source=model.source, stage="score")
    return {"outcome": "predicted", "source": model.source,
            "adapter": rt.info()["identity"]["template"], "shared": rt.info()["capabilities"]["shared"],
            "file_mb": round(os.path.getsize(model.path()) / 2**20), "load_peak_rss_mb": round(load.peak_rss_bytes / 2**20),
            "peak_rss_mb": round(scored.peak_rss_bytes / 2**20), "load_s": round(load.ms / 1000, 1),
            "threads": model.threads or os.cpu_count(), **scored.output}


def summarize(res: dict) -> dict:
    return {"model": Path(res["source"].split(":")[-1]).stem, "adapter": res["adapter"], "file_mb": res["file_mb"],
            "peak_rss_mb": res["peak_rss_mb"], **res["summary"]}


def family(name: str) -> str:
    return re.sub(r"[-_.]?(q\d.*|bf16|f16|ud-.*)$", "", name.lower().replace("_q4_0-it", "-it-q4_0"))


def shift(a: dict, b: dict) -> dict:
    rows = list(zip(a["raw"]["checks"], b["raw"]["checks"], strict=True))
    shifts = sorted(abs(x["p"] - y["p"]) for x, y in rows if x["p"] is not None and y["p"] is not None)
    return {"pairs": len(shifts), "mean_abs_dp": round(sum(shifts) / len(shifts), 3) if shifts else None,
            "p95_abs_dp": round(percentile(shifts, .95), 3) if shifts else None,
            "flips_0.5": sum((x["p"] >= .5) != (y["p"] >= .5) for x, y in rows if x["p"] is not None and y["p"] is not None),
            "settle_changes_0.9": sum(quality.decision(x, Policy(.9, .9)) != quality.decision(y, Policy(.9, .9)) for x, y in rows)}


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
    output_argument(ap, help="published summary; the resumable journal is OUTPUT.journal.jsonl")
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
