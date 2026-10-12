from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

from semantic_switch import Check, Model, Pick, Runtime, files, model_job
from semantic_switch.adapters import adapter_for
from semantic_switch.api import DEFAULT_JUDGE, execution_identity
from semantic_switch.backend import LlamaScorer
from semantic_switch.judge import LlamaJudge
from bench import diagnostics, fresh, quality
from bench.outcomes import failure, measure, runtime
from semantic_switch.metrics import Progress

OUTPUT = Path("bench/results/execution-shape.json")
CONFIGS = {"cpu": ("cpu", True), "cpu-norepack": ("cpu", False), "gpu": ("gpu", True)}
SUITES = {"selection": (quality.CHECKS, quality.PICKS, quality.TOPICS, quality.TOPIC_QUESTION),
          "fresh": (fresh.FRESH, fresh.FRESH_PICKS, fresh.FRESH_TOPICS, fresh.FRESH_QUESTION)}
ARMS = {"batched": ("batched", False), "fitting": ("batched", True), "serial": ("serial", False)}
INPUTS = 8


def suite(name):
    checks, picks, topics, question = SUITES[name]
    items = list(quality.check_items(checks))
    specs = {fact: Check(claim, excludes) for fact, (claim, excludes, _) in checks.items()} | {"pick": Pick(question, topics)}
    states = [quality.render(item["text"]) for item in items] + [quality.render(text) for text, _ in picks]
    gold = [{item["fact"]: item["label"]} for item in items] + [{"pick": answer} for _, answer in picks]
    return specs, states, gold


def row(result):
    probs = [] if result.status == "failed" else [result.p_true] if hasattr(result, "p_true") else list(result.scores.values())
    return {"value": result.value, "status": result.status, "p": probs, "mass": result.mass}


FAILED = {"value": None, "status": "failed", "p": [], "mass": None}


def scored(rows, gold):
    return [{"outcome": "failed" if results[key]["status"] == "failed" else "predicted", "correct": results[key]["value"] == value,
             "accepted": results[key]["status"] == "ok"} for results, answers in zip(rows, gold) for key, value in answers.items()]


def asked(rt, states, specs, fitting):
    if not fitting:
        return rt.ask_many(states, specs)
    columns = {key: rt.ask_many(states, {key: spec}) for key, spec in specs.items()}
    return [{key: columns[key][n][key] for key in specs} for n in range(len(states))]


@contextmanager
def bound(model, grouping):
    if grouping == "serial":
        with runtime(model) as rt:
            yield rt
        return
    with model_job(model), LlamaScorer(model.path(), inputs=INPUTS, n_gpu_layers=model.gpu_layers, repack=model.repack, grouping=grouping) as scorer, \
            Runtime({"judge": LlamaJudge(scorer, adapter_for(model.adapter, scorer), {"source": model.source}, scorer.weights.offload, INPUTS)}) as rt:
        yield rt


def run(setting):
    report = {"provenance": {}, "execution": {}, "arms": {}}
    device, repack = CONFIGS[setting]
    progress = Progress(f"bench.execution_shape {setting}", sum(len(suite(name)[2]) for name in SUITES) * len(ARMS), unit="gold answers")
    for grouping in ("batched", "serial"):
        with bound(Model(DEFAULT_JUDGE, device=device, repack=repack), grouping) as rt:
            rt.check("warm up", "The text is about food.")
            report["provenance"][grouping] = diagnostics.provenance(rt, list(SUITES), boundary="public SDK" if grouping == "serial" else "explicit batched ScoreRequest primitive")
            report["execution"][grouping] = execution_identity(rt.judge().scorer.capacity())
            for arm, (shape, fitting) in ARMS.items():
                if shape != grouping:
                    continue
                report["arms"][arm] = {}
                for name in SUITES:
                    specs, states, gold = suite(name)
                    measured = measure(lambda: asked(rt, states, specs, fitting), runtime=rt)
                    known = {"questions": len(states) * len(specs), "work": measured.work}
                    report["arms"][arm][name] = failure(measured, **known, rows=[dict.fromkeys(specs, FAILED) for _ in states]) if measured.error else \
                        {"ms": measured.ms, **known, "rows": [{key: row(result) for key, result in results.items()} for results in measured.result]}
                    progress.record(scored(report["arms"][arm][name]["rows"], gold))
    report["progress"] = progress.finish()
    return report


def compare(a, b):
    changes, total, difference, mass = [], 0, 0., 0.
    for name in SUITES:
        for n, (x, y) in enumerate(zip(a[name]["rows"], b[name]["rows"])):
            for key in x:
                total += 1
                if (x[key]["value"], x[key]["status"]) != (y[key]["value"], y[key]["status"]):
                    changes.append({"suite": name, "input": n, "question": key, "a": x[key], "b": y[key]})
                difference = max([difference, *(abs(p - q) for p, q in zip(x[key]["p"], y[key]["p"]))])
                if x[key]["mass"] is not None and y[key]["mass"] is not None:
                    mass = max(mass, abs(x[key]["mass"] - y[key]["mass"]))
    return {"questions": total, "accepted_changes": len(changes), "max_probability_difference": difference,
            "max_mass_difference": mass, "changes": changes}


def accuracy(arm):
    rows = [(results[key], value) for name in SUITES for results, gold in zip(arm[name]["rows"], suite(name)[2]) for key, value in gold.items()]
    accepted = [(result, value) for result, value in rows if result["status"] == "ok"]
    return {"gold": len(rows), "accepted": len(accepted), "accepted_correct": sum(result["value"] == value for result, value in accepted),
            "ms": sum(arm[name]["ms"] for name in SUITES), "questions": sum(arm[name]["questions"] for name in SUITES)}


def summarize(configs):
    within = {setting: {f"{a} vs {b}": compare(data["arms"][a], data["arms"][b]) for a, b in combinations(ARMS, 2)}
              for setting, data in configs.items()}
    across = {f"{arm}: {a} vs {b}": compare(configs[a]["arms"][arm], configs[b]["arms"][arm])
              for a, b in combinations(configs, 2) for arm in ARMS}
    return {"schema": 1, "model": DEFAULT_JUDGE, "policy": "default (fact policy, min_score unset)",
            "boundary": "Runtime.ask_many; serial: ordinary Model judge; batched: diagnostic LlamaScorer(grouping='batched', inputs=8) ScoreRequest primitive with multi-input windows and every suite question per input; fitting: that primitive with one question per input",
            "arms": {setting: {arm: accuracy(data["arms"][arm]) for arm in ARMS} for setting, data in configs.items()},
            "within": within, "across": across, "configs": configs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("setting", choices=tuple(CONFIGS))
    args = parser.parse_args()
    configs = {**(files.read_json(OUTPUT) or {}).get("configs", {}), args.setting: run(args.setting)}
    report = summarize(configs)
    files.publish(OUTPUT, report)
    for setting, pairs in report["within"].items():
        print(setting, {pair: (value["accepted_changes"], round(value["max_probability_difference"], 4)) for pair, value in pairs.items()})
    print({pair: (value["accepted_changes"], round(value["max_probability_difference"], 4)) for pair, value in report["across"].items()})


if __name__ == "__main__":
    main()
