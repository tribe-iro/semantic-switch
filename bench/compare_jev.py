"""Matched native Jev / local CPU timing. Run: python -m bench.compare_jev --help."""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import time
from pathlib import Path

from semantic_switch import Check, DEFAULT_JUDGE, Model, Runtime

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


def read_key(path: Path) -> str:
    raw = path.read_text().strip()
    # Accept a bare key, quoted key, or a single KEY=value line; never execute the file.
    if "=" in raw and "\n" not in raw and raw.split("=", 1)[0].replace("export ", "").strip().isidentifier():
        raw = raw.split("=", 1)[1].strip()
    key = raw.strip("\"'")
    if not key or any(c.isspace() for c in key):
        raise ValueError("Key file must contain one API key, optionally as KEY=value")
    return key


def measure(call, state, count):
    start = time.perf_counter()
    result = call(state, count)
    # Include serialization at both boundaries, not just the underlying inference call.
    json.dumps(result)
    return round((time.perf_counter() - start) * 1000, 3), result


def run(label, call, repeats, records):
    for name, state, count in CELLS:
        ms, result = measure(call, state, count)
        records.append(dict(engine=label, cell=name, questions=count, warmup=True, ms=ms, result=result))
    jobs = CELLS * repeats
    random.Random(42).shuffle(jobs)
    for name, state, count in jobs:
        ms, result = measure(call, state, count)
        records.append(dict(engine=label, cell=name, questions=count, warmup=False, ms=ms, result=result))
    for name, _, count in CELLS:
        values = sorted(r["ms"] for r in records if r["engine"] == label and r["cell"] == name
                        and r["questions"] == count and not r["warmup"])
        p95 = values[max(0, int(len(values) * .95 + .999999) - 1)]
        print(f"{label:12} {name:5} {count}q  median {statistics.median(values):8.1f} ms"
              f"  p95 {p95:8.1f} ms  n={len(values)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path,
                        default=Path("/home/tribeiro/Projects/keys/jev_first_test.txt"))
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--repeats", type=int, default=15, help="Measured calls per cell and engine")
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-switch-jev-paired.json"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    key = read_key(args.key_file)
    records = []
    conn = http.client.HTTPSConnection("api.typesafe.ai", timeout=30)

    def jev(state, count):
        payload = json.dumps({"model": args.jev_model, "state": state, "questions": {
            f"q{i}": {"type": "noul", "instructions": q} for i, q in enumerate(QUESTIONS[:count])}})
        conn.request("POST", "/v1/systemone", payload,
                     {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"Jev returned HTTP {response.status}; response body omitted")
        result = json.loads(body)
        answers = result.get("answers", {})
        if set(answers) != {f"q{i}" for i in range(count)} or any(
                a.get("type") != "noul" or not isinstance(a.get("noul"), (int, float))
                or not 0 <= a["noul"] <= 1 for a in answers.values()):
            raise RuntimeError("Jev returned an unexpected answer shape")
        return result

    try:
        run("jev", jev, args.repeats, records)
        for label, source in (("qwen-0.6b", QWEN3), ("gemma-e2b", DEFAULT_JUDGE)):
            with Runtime(models={"judge": Model(source, device="cpu", threads=args.threads)}) as rt:
                def local(state, count):
                    results = rt.ask(state, {f"q{i}": Check(q) for i, q in enumerate(QUESTIONS[:count])})
                    return {k: v.as_dict() for k, v in results.items()}
                run(label, local, args.repeats, records)
    finally:
        conn.close()
        usage = sum(r["result"].get("usage", {}).get("input_tokens", 0)
                    for r in records if r["engine"] == "jev")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "boundary": "call entry to parsed result and JSON serialization; warm models/connections",
            "threads": args.threads, "device": "cpu", "questions": QUESTIONS,
            "inputs": {"short": SHORT, "long": LONG}, "local_models": [QWEN3, DEFAULT_JUDGE],
            "jev_input_tokens": usage, "jev_estimated_usd_at_list_price": usage * .042 / 1_000_000,
            "records": records,
        }, indent=2))
        print(f"Saved {len(records)} calls to {args.output}; Jev input tokens: {usage}", flush=True)


if __name__ == "__main__":
    main()
