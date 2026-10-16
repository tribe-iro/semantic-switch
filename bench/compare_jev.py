from __future__ import annotations

import argparse
import random
from pathlib import Path

from semantic_switch import Check, DEFAULT_JUDGE, Model, files
from bench.outcomes import JEV, Receipts, destination, failure, judge_spec, latency, measure, output_argument, provenance, runtime

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"

QUESTIONS = (
    "The customer requests a refund.",
    "The customer threatens legal action.",
    "The customer reports a broken device.",
    "The customer reports a duplicate charge.",
    "The customer needs help today.",
    "The customer asks about an invoice.",
    "The customer wants to cancel a subscription.",
)
SHORT = "subject: Double charge\nbody: I was charged twice in May, please refund one of them."
LONG = SHORT + "\n" + (
    "The account shows an active subscription. The customer contacted support about a payment "
    "and provided their order details. " * 12
)
CELLS = [(name, state, n) for name, state in (("short", SHORT), ("long", LONG)) for n in (1, 7)]


def run(label, call, repeats, records):
    jobs = CELLS * repeats
    random.Random(42).shuffle(jobs)
    for warmup, (name, state, count) in [(True, cell) for cell in CELLS] + [(False, cell) for cell in jobs]:
        m = measure(lambda: call(state, count))
        known = dict(engine=label, cell=name, questions=count, warmup=warmup)
        records.append(failure(m, **known) if m.error else dict(known, ms=round(m.ms, 3), result=m.output, outcome="predicted"))
    for name, _, count in CELLS:
        measured = [r for r in records if r["engine"] == label and r["cell"] == name and r["questions"] == count and not r["warmup"]]
        timing = latency(r["ms"] for r in measured if r["outcome"] != "failed")
        failed = len(measured) - timing["samples"]
        if not timing["samples"]:
            print(f"{label:12} {name:5} {count}q  all {failed} calls failed", flush=True)
            continue
        print(f"{label:12} {name:5} {count}q  p50 {timing['p50']:8.1f} ms  p95 {timing['p95']:8.1f} ms  n={timing['samples']} failed={failed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Matched hosted Jev (OpenRouter System One) / local CPU timing")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--provider", help="OpenRouter provider route for Jev")
    parser.add_argument("--repeats", type=int, default=15, help="Measured calls per cell and engine")
    output_argument(parser)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    destination(args.output)
    records, identities, receipts = [], {}, Receipts()

    def asked(rt):
        return lambda state, count: {name: result.as_dict() for name, result in rt.ask(
            state, {f"q{i}": Check(question) for i, question in enumerate(QUESTIONS[:count])}).items()}
    try:
        with runtime(judge_spec(JEV, provider=args.provider, key_file=args.key_file), receipts=receipts) as hosted:
            run("jev", asked(hosted), args.repeats, records)
            identities["jev"] = provenance(hosted, CELLS)
        for label, source in (("qwen-0.6b", QWEN3), ("gemma-e2b", DEFAULT_JUDGE)):
            with runtime(Model(source, device="cpu", threads=args.threads)) as rt:
                run(label, asked(rt), args.repeats, records)
                identities[label] = provenance(rt, CELLS)
                files.publish(args.output, {"partial": True, "provenance": identities, "records": records})
    finally:
        requests = receipts.summary()
        usage = requests['usage'].get('prompt_tokens', requests['usage'].get('input_tokens')) if requests['usage']['reported'] == requests['requests'] else None
        files.publish(args.output, {
            "provenance": identities, "schema": 2,
            "boundary": "call entry to parsed result and JSON serialization; warm models/connections",
            "threads": args.threads, "device": "cpu", "questions": QUESTIONS,
            "inputs": {"short": SHORT, "long": LONG}, "local_models": [QWEN3, DEFAULT_JUDGE],
            "jev_input_tokens": usage, "requests": requests,
            "records": records,
        })
        print(f"Saved {len(records)} calls to {args.output}; Jev input tokens: {usage}", flush=True)


if __name__ == "__main__":
    main()
