"""Compare native Jev and local judges on identical labelled English/Portuguese cases."""
from __future__ import annotations

import argparse
import http.client
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from semantic_switch import DEFAULT_JUDGE, Model, Runtime, quality

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
from .compare_jev import read_key


class Jev:
    def __init__(self, key, model):
        self.key, self.model = key, model
        self.connection = http.client.HTTPSConnection("api.typesafe.ai", timeout=30)
        self.tokens = 0
        self.models = set()

    def request(self, state, question):
        body = json.dumps({"model": self.model, "state": state, "questions": {"answer": question}})
        self.connection.request("POST", "/v1/systemone", body,
                                {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        response = self.connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"Jev HTTP {response.status}; response body omitted")
        result = json.loads(body)
        self.tokens += result.get("usage", {}).get("input_tokens", 0)
        self.models.add(result.get("model", "unknown"))
        answer = result["answers"]["answer"]
        if answer.get("type") != question["type"]:
            raise ValueError("Jev returned an unexpected answer type")
        return answer

    def check(self, state, statement, *, excludes=(), model=None):
        instructions = statement + ("\nExclude: " + "; ".join(excludes) if excludes else "")
        answer = self.request(state, {"type": "noul", "instructions": instructions})
        p = answer["noul"]
        if not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid Jev probability")
        return SimpleNamespace(p_true=p, mass=None, error=None)

    def pick(self, state, question, options, *, model=None):
        answer = self.request(state, {"type": "choice", "instructions": question, "criteria": options})
        scores = answer["probabilities"]
        if set(scores) != set(options) or any(not isinstance(p, (float, int)) or not math.isfinite(p)
                                             or not 0 <= p <= 1 for p in scores.values()) \
                or not math.isclose(sum(scores.values()), 1, abs_tol=.01):
            raise ValueError("Invalid Jev choice distribution")
        return SimpleNamespace(scores=scores, mass=None, error=None)

    def close(self):
        self.connection.close()


def accepted(row, threshold):
    p = row["p"]
    return (row.get("error") is None and p is not None
            and (row["mass"] is None or row["mass"] >= .5)
            and max(p, 1 - p) >= threshold)


def selective(rows, threshold):
    settled = [r for r in rows if accepted(r, threshold)]
    correct = sum((r["p"] >= .5) == r["label"] for r in settled)
    return {"threshold": threshold, "accepted": len(settled), "total": len(rows),
            "coverage": len(settled) / len(rows),
            "precision": correct / len(settled) if settled else None}


def summarize(result):
    rows = result["checks"]
    summary = quality.summarize(result)
    summary["accuracy"] = sum(r["p"] is not None and r.get("error") is None
                              and (r["p"] >= .5) == r["label"] for r in rows) / len(rows)
    summary["pick_accuracy"] = sum(bool(r["scores"]) and r.get("error") is None
                                   and max(r["scores"], key=r["scores"].get) == r["answer"]
                                   for r in result["picks"]) / len(result["picks"])
    fixed = selective(rows, .9)
    summary["coverage_0.9"], summary["precision_0.9"] = fixed["coverage"], fixed["precision"]
    # Keep translated and positive/negative versions of a pair together when tuning.
    def tuning(row):
        fact, index = row["pair"].rsplit(":", 1)
        pairs = len(quality.CHECKS[fact][2])
        return (int(index) % (pairs // 2)) % 2 == 0
    tune, test = [r for r in rows if tuning(r)], [r for r in rows if not tuning(r)]
    threshold = next((t / 100 for t in range(50, 100)
                      if (s := selective(tune, t / 100))["accepted"] >= 20
                      and s["precision"] >= .95), None)
    summary["heldout_after_tuning_to_95"] = selective(test, threshold) if threshold is not None else {
        "threshold": None, "accepted": 0, "total": len(test), "coverage": 0, "precision": None}
    summary["fixed_thresholds"] = [selective(rows, t) for t in (.5, .8, .9, .95, .99)]
    summary["total_checks"] = len(rows)
    summary["total_picks"] = len(result["picks"])
    summary["failures"] = [dict(row, **item) for row, item in zip(rows, quality.check_items())
                           if row["p"] is None or (row["p"] >= .5) != row["label"]]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path,
                        default=Path("/home/tribeiro/Projects/keys/jev_first_test.txt"))
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--engines", default="jev,qwen,gemma")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-switch-jev-quality.json"))
    args = parser.parse_args()
    sources = {"qwen": QWEN3, "gemma": DEFAULT_JUDGE}
    engines = args.engines.split(",")
    if any(e not in {"jev", *sources} for e in engines):
        parser.error("--engines accepts jev,qwen,gemma")
    report = {"suite": quality.SUITE_VERSION, "threads": args.threads, "device": "cpu", "engines": {},
              "scope": "Synthetic support cases, English and Portuguese; zero-shot, no examples or retrieval",
              "boundary": "Public call entry to parsed typed result; excludes model loading and warmup",
              "caveat": "A small existing development suite, not proof of general parity. 95% tuning target is not a guarantee."}
    for engine in engines:
        judge = Jev(read_key(args.key_file), args.jev_model) if engine == "jev" else \
            Runtime(models={"judge": Model(sources[engine], device="cpu", threads=args.threads)})
        print(f"Evaluating {engine} on {len(list(quality.check_items()))} checks and {len(quality.PICKS)} picks...", flush=True)
        started = time.perf_counter()
        try:
            result = quality.run(judge)
            summary = summarize(result)
            report["engines"][engine] = {"source": args.jev_model if engine == "jev" else sources[engine],
                                         "raw": result, "summary": summary}
            if engine == "jev":
                report["engines"][engine].update(models=sorted(judge.models), input_tokens=judge.tokens,
                                                 estimated_usd=judge.tokens * .042 / 1_000_000)
            print(json.dumps({"engine": engine, **{k: v for k, v in summary.items() if k != "failures"}}, indent=2), flush=True)
            print(f"Finished in {time.perf_counter() - started:.1f}s", flush=True)
        finally:
            judge.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
    print(f"Saved results and mistakes to {args.output}", flush=True)


if __name__ == "__main__":
    main()
