"""Compare hosted (Jev, OpenRouter) and local judges on identical labelled English/Portuguese cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, files
from bench import quality
from bench.model_comparison import suite
from bench.outcomes import JEV, destination, judged, output_argument

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"


def failures(raw):
    return [dict(row, **item) for row, item in zip(raw["checks"], quality.check_items()) if row["p"] is None or (row["p"] >= .5) != row["label"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="append", help=f"repeatable: local GGUF (path or repo:file) or openrouter:MODEL (default: {JEV}, Qwen3-0.6B, E2B)")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL judges")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--threads", type=int, default=16)
    output_argument(parser)
    args = parser.parse_args()
    destination(args.output)
    report = {"suite": quality.SUITE_VERSION, "threads": args.threads, "device": "cpu", "engines": {},
              "scope": "Synthetic support cases, English and Portuguese; zero-shot, no examples or retrieval",
              "boundary": "Public call entry to parsed typed result; excludes model loading and warmup",
              "caveat": "A small existing development suite, not proof of general parity. 95% tuning target is not a guarantee."}
    for spec in args.judge or [JEV, QWEN3, DEFAULT_JUDGE]:
        print(f"Evaluating {spec} on {len(list(quality.check_items()))} checks and {len(quality.PICKS)} picks...", flush=True)
        started = time.perf_counter()
        try:
            with judged(spec, provider=args.provider, key_file=args.key_file, device="cpu", threads=args.threads) as (rt, receipts):
                result = suite(rt, minimum=20)
                summary = result["summary"]
                report["engines"][spec] = {"source": spec, **result, "failures": failures(result["raw"]), "execution": rt.info(), "requests": receipts.summary()}
            print(json.dumps({"engine": spec, **summary}, indent=2), flush=True)
            print(f"Finished in {time.perf_counter() - started:.1f}s", flush=True)
        finally:
            files.publish(args.output, report)
    print(f"Saved results and mistakes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
