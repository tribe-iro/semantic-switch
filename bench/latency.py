from __future__ import annotations

import argparse
import statistics
from collections import Counter

from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, Check, ConfigError, Pick, Rate, files
from bench import outcomes
from bench.outcomes import Progress, failure, judged, measure, output_argument, provenance, report

FIXTURE = Path(__file__).resolve().parent / "data" / "latency.json"
KINDS = {"check": lambda item: Check(item["question"]), "pick": lambda item: Pick(item["question"], item["options"]),
         "rate": lambda item: Rate(item["question"], item["options"])}


def workload(path=FIXTURE) -> dict:
    fixture = files.read_json(Path(path))
    if not isinstance(fixture, dict) or fixture.get("schema") != 1:
        raise ConfigError(f"{path}: unsupported latency workload fixture")
    return {"fixture": fixture, "digest": outcomes.artifact_digest(fixture), "inputs": tuple(fixture["inputs"]),
            "items": {item["name"]: KINDS[item["kind"]](item) for item in fixture["items"]},
            "claim": next(item["question"] for item in fixture["items"] if item["name"] == fixture["claim"])}


def _summary(samples: list[dict]) -> dict | None:
    return outcomes.latency([sample["ms"] for sample in samples]) if samples else None


def _stats(samples: list[dict], progress) -> dict:
    return {"all": _summary(samples), "non_failed": _summary([s for s in samples if "failed" not in s["statuses"]]),
            "failed": sum(s["outcome"] == "failed" for s in samples), "statuses": dict(Counter(status for s in samples for status in s["statuses"])),
            "progress": progress.finish(), "samples": samples}


def _series(name, operations, rt) -> dict:
    progress = Progress(f"bench.latency {name}", len(operations), unit="calls")
    return _stats([_sample(operation, rt, progress) for operation in operations], progress)


def _sample(operation, rt, progress) -> dict:
    sample = _measured(operation, rt)
    progress.record([{"outcome": sample["outcome"], "accepted": set(sample["statuses"]) == {"ok"}}])
    return sample


def _measured(operation, rt) -> dict:
    measured = measure(operation, runtime=rt)
    if measured.error is not None:
        return failure(measured, statuses=["failed"])
    results = measured.result.values() if isinstance(measured.result, dict) else [measured.result]
    return {"outcome": "predicted", "ms": measured.ms, "statuses": [result.status for result in results], "native_work": measured.work}


def _throughput(count, measured) -> float | None:
    return None if measured.error is not None else round(count / (measured.ms / 1000), 2)


def logical_tokens(results) -> float | None:
    counts = [result.cost.get("logical_tokens") for result in results]
    return None if not counts or None in counts else round(statistics.mean(counts), 1)


PROTOCOL = {"boundary": "public call: bind, render, score, compose and JSON serialization of the result",
            "backend_timings": "separate (native_work counters); not part of the public milliseconds",
            "cache": "warm: each series follows one untimed warmup call and keeps the runtime caches",
            "order": ["same_input", "new_input", "new_input_4_questions", "switch", "switch_throughput", "switch_batch"],
            "concurrency": "serial calls; switch_batch uses the runtime batch_inputs"}


def cohort_latency(rt, directory) -> dict:
    from bench.policy_workflow import cohort_cases, chunks, CHUNK
    _, cases = cohort_cases(rt, directory)
    indexed = {case["case_id"]: case for case in cases}
    items = {case["case_id"]: item for _, item, members in chunks(cases, "guided", CHUNK) for case in members}
    report = {"cohort_ms": _series("cohort", [lambda c=c: items[c["case_id"]](c["rendered_input"], rt) for c in cases], rt)}
    pairs = [(case, other) for case in cases if "procedure" in case["tags"]
             and (other := indexed[f"{case['situation']}:{case['contrast']}:{case['language']}"])["evidence"] == case["evidence"]]
    changed, repeated = [], []
    progress = {name: Progress(f"bench.latency {name}", len(pairs), unit="calls") for name in ("changed_guidance", "repeated_guidance")}
    for case, other in pairs:
        items[other["case_id"]](other["rendered_input"], rt)
        changed.append(_sample(lambda: items[case["case_id"]](case["rendered_input"], rt), rt, progress["changed_guidance"]))
        repeated.append(_sample(lambda: items[case["case_id"]](case["rendered_input"], rt), rt, progress["repeated_guidance"]))
    report["changed_guidance_ms"] = {"boundary": "same input, first call after answering it under the contrasting procedure", **_stats(changed, progress["changed_guidance"])}
    report["repeated_guidance_ms"] = {"boundary": "the same call repeated immediately", **_stats(repeated, progress["repeated_guidance"])}
    return report


def latency(rt, path=None, n: int = 20, cohort=None, receipts=None, fixture=FIXTURE) -> dict:
    if n < 1:
        raise ConfigError("benchmark n must be positive")
    loaded = workload(fixture)
    texts, items, claim = loaded["inputs"], loaded["items"], loaded["claim"]
    rt.check(texts[0], claim)
    info = rt.info()
    report = {"provenance": provenance(rt, loaded["fixture"]), "workload": {"path": str(fixture), "digest": loaded["digest"]}, "model": rt.identity(),
              "execution": info, "samples": n, "protocol": PROTOCOL,
              "same_input_ms": _series("same_input", [lambda: rt.check(texts[0], claim)] * n, rt),
              "new_input_ms": _series("new_input", [lambda k=k: rt.check(f"{texts[k % len(texts)]} ({k})", claim) for k in range(n)], rt),
              "new_input_4_questions_ms": _series("new_input_4_questions", [lambda k=k: rt.ask(f"{texts[k % len(texts)]} ({k})", items) for k in range(n)], rt)}
    if path:
        sw = rt.load(path)
        inputs = [it["input"] for it in files.read_jsonl(sw.paths.test)] or [t.input for t in sw.plan.tests]
        if not inputs:
            raise ConfigError(f"{sw.paths.yaml.name} has no test inputs or YAML tests to benchmark; "
                              f"add some with `ss label --to test`")
        inputs = (inputs * (n // len(inputs) + 1))[:n]
        sw(inputs[0])
        report["switch_ms"] = _series("switch", [lambda d=d: sw(d) for d in inputs], rt)
        sequential = measure(lambda: list(sw.many(inputs)), runtime=rt)
        report["switch_throughput_per_s"] = _throughput(len(inputs), sequential)
        batched = measure(lambda: sw.run(inputs), runtime=rt)
        report["batch_inputs"] = rt.batch_inputs()
        report["switch_batch_throughput_per_s"] = _throughput(len(inputs), batched)
        report["throughput_errors"] = {"sequential": sequential.error, "batched": batched.error}
        report["logical_tokens_per_decision"] = None if sequential.error else logical_tokens(sequential.result)
        report["switch_native_work"] = sequential.work or None
    if cohort:
        report["cohort"] = cohort_latency(rt, cohort)
    report["provider_usage"] = receipts.summary()["usage"] if receipts is not None else None
    return report


def main():
    parser = argparse.ArgumentParser(description="public-SDK latency of inline judgments and an optional saved switch")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="local GGUF (path or repo:file) or openrouter:MODEL")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--switch")
    parser.add_argument("--cohort", help="materialized small-context cohort directory: per-item and changed-guidance timing")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--device")
    output_argument(parser)
    args = parser.parse_args()
    with judged(args.judge, provider=args.provider, key_file=args.key_file, device=args.device) as (rt, receipts):
        report(args.output, {"judge": args.judge, **latency(rt, args.switch, args.n, args.cohort, receipts)})


if __name__ == "__main__":
    main()
