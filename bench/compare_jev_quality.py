"""Compare hosted (Jev, OpenRouter) and local judges on identical labelled English/Portuguese cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, files
from bench import quality
from bench import diagnostics
from bench.outcomes import JEV, judged

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"


def summarize(result):
    rows = result["checks"]
    summary = quality.summarize(result)
    def tuning(row):
        fact, index = row["pair"].rsplit(":", 1)
        pairs = len(quality.CHECKS[fact][2])
        return (int(index) % (pairs // 2)) % 2 == 0
    tune, test = [r for r in rows if tuning(r)], [r for r in rows if not tuning(r)]
    threshold = next((t / 100 for t in range(50, 100)
                      if (s := quality.selective(tune, t / 100))["accepted"] >= 20
                      and s["precision"] >= .95), None)
    summary["heldout_after_tuning_to_95"] = quality.selective(test, threshold) if threshold is not None else {
        "threshold": None, "accepted": 0, "total": len(test), "coverage": 0, "precision": None}
    summary["fixed_thresholds"] = [quality.selective(rows, t) for t in (.5, .8, .9, .95, .99)]
    summary["total_checks"] = len(rows)
    summary["total_picks"] = len(result["picks"])
    summary["failures"] = [dict(row, **item) for row, item in zip(rows, quality.check_items())
                           if row["p"] is None or (row["p"] >= .5) != row["label"]]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", help=f"repeatable: local GGUF (path or repo:file) or openrouter:MODEL (default: {JEV}, Qwen3-0.6B, E2B)")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL judges")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-switch-jev-quality.json"))
    args = parser.parse_args()
    report = {"suite": quality.SUITE_VERSION, "threads": args.threads, "device": "cpu", "engines": {},
              "scope": "Synthetic support cases, English and Portuguese; zero-shot, no examples or retrieval",
              "boundary": "Public call entry to parsed typed result; excludes model loading and warmup",
              "caveat": "A small existing development suite, not proof of general parity. 95% tuning target is not a guarantee."}
    for spec in args.judge or [JEV, QWEN3, DEFAULT_JUDGE]:
        print(f"Evaluating {spec} on {len(list(quality.check_items()))} checks and {len(quality.PICKS)} picks...", flush=True)
        started = time.perf_counter()
        try:
            with judged(spec, provider=args.provider, key_file=args.key_file, device="cpu", threads=args.threads) as (rt, recording):
                result = quality.run(rt)
                summary = summarize(result)
                report["engines"][spec] = {"source": spec, "raw": result, "summary": summary, "execution": rt.info(),
                                           "provenance": diagnostics.provenance(rt, list(quality.check_items()), boundary="public SDK"),
                                           **({"models": sorted(rt.judge().observed_models), "requests": recording.summary()} if recording else {})}
            print(json.dumps({"engine": spec, **{k: v for k, v in summary.items() if k != "failures"}}, indent=2), flush=True)
            print(f"Finished in {time.perf_counter() - started:.1f}s", flush=True)
        finally:
            files.publish(args.output, report)
    print(f"Saved results and mistakes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
