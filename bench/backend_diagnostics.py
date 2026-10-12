from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from semantic_switch import Check, Model, Pick, Rate, backend, files
from semantic_switch.api import execution_identity
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
from bench import diagnostics
from bench.execution_reproduction import EMOTION, TARGETS, Session, arms, cases, history
from bench.model_comparison import timings
from bench.outcomes import peak_rss, reset_peak, runtime

OUTPUT = Path("bench/results/native-layout.json")
IMPLEMENTATION = Path("bench/results/implementation-v026.json")
ORDER = ("right-sized", "reserved-37", "right-sized", "reserved-37")
LONG = Check("The text expresses one dominant emotion, judged by these descriptions: " +
             " ".join(f"{name} means {text}." for _ in range(4) for name, text in EMOTION.options.items()))
POPULATIONS = ("cold", "cold_repeat", "warm", "evicted", "one", "four", "seventeen", "mixed_guidance", "long_suffix")


class Reserved(backend.Capacity):
    @property
    def n_seq_max(self) -> int:
        return self.prefix_slots + self.states + backend.BRANCHES


LAYOUTS = {"right-sized": backend.Capacity, "reserved-37": Reserved}


@contextmanager
def layout(name):
    canonical = backend.capacity
    with patch.object(backend, "capacity", lambda *args, **kwargs: LAYOUTS[name](**asdict(canonical(*args, **kwargs)))):
        yield


@contextmanager
def phases(scorer):
    events = {"submissions": [], "logits": [], "reductions": [], "rows": []}
    run, logits, reduce = scorer.ctx.run, scorer.ctx.logits, backend.reduce_row

    def timed(target, bucket):
        def call(*args):
            begin = time.perf_counter()
            try:
                return target(*args)
            finally:
                events[bucket].append((time.perf_counter() - begin) * 1000)
        return call

    def reduction(row, slots, scratch=None):
        events["rows"].append((np.array(row), slots))
        return timed(reduce, "reductions")(row, slots, scratch)

    with patch.object(scorer.ctx, "run", timed(run, "submissions")), patch.object(scorer.ctx, "logits", timed(logits, "logits")), \
            patch.object(backend, "reduce_row", reduction):
        yield events


def split(events):
    run, wait, reduce = (sum(events[key]) for key in ("submissions", "logits", "reductions"))
    return {"submissions": len(events["submissions"]), "outputs": len(events["reductions"]), "decode_ms": run - reduce,
            "logits_wait_ms": wait, "reduce_ms": reduce}


def percentiles(values):
    return {"n": len(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None


def allocations(row, slots, scratch):
    tracemalloc.start()
    backend.reduce_row(row, slots, scratch)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def reducer_cost(row, slots, repeats):
    scratch = np.empty(len(row))
    backend.reduce_row(row, slots, scratch)
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter()
        backend.reduce_row(row, slots, scratch)
        samples.append((time.perf_counter() - begin) * 1000)
    return {"vocabulary": len(row), "slots": [len(ids) for ids in slots], "ms": percentiles(samples),
            "peak_bytes": allocations(row, slots, scratch), "peak_bytes_without_scratch": allocations(row, slots, None)}


def shapes(data):
    text, specs = data[0]
    key = next(iter(specs))
    return {"check": (text, {key: specs[key]}), "pick": (text, {"emotion": EMOTION}), "long_suffix": (text, {"long": LONG}),
            "guided_28": (text, specs)}


def profile(rt, data, repeats):
    scorer, out = rt.judge().scorer, {}
    for name, (text, specs) in shapes(data).items():
        rt.ask(text, specs)
        calls, rows = [], []
        for _ in range(repeats):
            with phases(scorer) as events:
                begin = time.perf_counter()
                rt.ask(text, specs)
                calls.append({"ask_ms": (time.perf_counter() - begin) * 1000, **split(events)})
            rows = events["rows"]
        totals = {key: percentiles([call[key] for call in calls]) for key in ("ask_ms", "decode_ms", "logits_wait_ms", "reduce_ms")}
        out[name] = {"questions": len(specs), "warm": True, "submissions": calls[0]["submissions"], "outputs": calls[0]["outputs"],
                     **totals, "reduce_share": float(np.median([call["reduce_ms"] / call["ask_ms"] for call in calls])),
                     "reducer": reducer_cost(*rows[0], repeats=20 * repeats)}
    return out


def answers(run):
    found = {}
    for group, items in (("case", run["cases"]), ("history", run["histories"])):
        for item, measured in items.items():
            for arm, call in measured.items():
                for key, result in call["results"].items():
                    found[f"{group} {item}/{arm}/{key}"] = result
    return found


def top(result):
    return None if not result["p"] else result["p"][0] >= .5 if len(result["p"]) == 1 else int(np.argmax(result["p"]))


def agreement(a, b):
    x, y = answers(a), answers(b)
    keys = sorted(x.keys() & y.keys())
    dp = [max((abs(p - q) for p, q in zip(x[k]["p"], y[k]["p"])), default=0.) for k in keys]
    flips = [{"answer": k, "a": x[k], "b": y[k]} for k in keys if (x[k]["value"], x[k]["status"]) != (y[k]["value"], y[k]["status"])]
    return {"answers": len(keys), "unmatched": len(x.keys() ^ y.keys()), "identical": sum(x[k] == y[k] for k in keys),
            "answer_changes": sum(top(x[k]) != top(y[k]) for k in keys),
            "acceptance_changes": sum(x[k]["status"] != y[k]["status"] for k in keys), "accepted_changes": len(flips),
            "max_dp": max(dp, default=0.), "mean_dp": float(np.mean(dp)) if dp else 0., "flips": flips}


def population(arm):
    return arm.split(":")[0]


def latency(runs):
    found = {}
    for run in runs:
        for measured in run["cases"].values():
            for arm, call in measured.items():
                if population(arm) in POPULATIONS:
                    found.setdefault(population(arm), []).append(call)
    return {name: {"ms": percentiles([call["ms"] for call in calls]),
                   "ms_per_question": percentiles([call["ms"] / len(call["questions"]) for call in calls]),
                   "submissions": sum(call["work"]["submissions"] for call in calls) / len(calls)}
            for name, calls in found.items()}


def measured_run(name, model, data):
    with layout(name):
        estimate = model._host_bytes()
        reset_peak()
        with runtime(model) as rt:
            session = Session(rt)
            session.ask("warm up", {"warm": Check("The text is about food.")})
            out = {"layout": name, "execution": execution_identity(rt.judge().scorer.capacity()),
                   "buffers": rt.judge().scorer.capacity()["buffers"], "estimated_host_bytes": estimate,
                   "provenance": diagnostics.provenance(rt, [text for text, _ in data], boundary="public Runtime.ask; cache reset through LlamaScorer.invalidate"),
                   "cases": {}, "histories": {}}
            for index in range(len(data)):
                found = arms(session, data, index)
                session.cold()
                found["long_suffix"] = session.ask(data[index][0], {**data[index][1], "long": LONG})
                out["cases"][str(index)] = found
                print(name, index, round(found["cold"]["ms"]), flush=True)
            for index, target in (item for item in TARGETS if item[0] < len(data)):
                out["histories"][f"{index}:{target}"] = history(session, data, index, target)["arms"]
    out["peak_rss_bytes"] = peak_rss()[0]
    return out


def summarize(runs):
    by = {name: [run for run in runs if run["layout"] == name] for name in LAYOUTS}
    first, second = by.values()
    return {"within": {name: agreement(*found[:2]) for name, found in by.items() if len(found) > 1},
            "across": {f"run {n}": agreement(a, b) for n, (a, b) in enumerate(zip(first, second))},
            "latency": {name: latency(found) for name, found in by.items()},
            "memory": {name: {"estimated_host_bytes": found[0]["estimated_host_bytes"], "buffers": found[0]["buffers"],
                              "peak_rss_bytes": [run["peak_rss_bytes"] for run in found],
                              **{key: found[0]["execution"][key] for key in ("n_seq_max", "n_outputs_max")}}
                       for name, found in by.items()},
            "identity_changed": sorted(k for k in first[0]["execution"] if first[0]["execution"][k] != second[0]["execution"].get(k))}


def model_of(args):
    return Model(args.model, device=args.device, prefix_slots=4)


def reducer(args):
    data = cases(1)
    with runtime(model_of(args)) as rt:
        rt.ask("warm up", {"warm": Check("The text is about food.")})
        return {"execution": execution_identity(rt.judge().scorer.capacity()), "n_vocab": rt.judge().scorer.weights.n_vocab,
                "shapes": profile(rt, data, args.repeats)}


def layouts(args):
    model, data = model_of(args), cases(args.cases)
    runs = [measured_run(name, model, data) for name in ORDER]
    return {**summarize(runs), "order": ORDER, "cases": args.cases, "runs": runs}


def implementation(args):
    output = {}
    for name, model in (("gpu", Model("ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf", device="gpu")),
                        ("cpu", Model("unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf", adapter="qwen3", device="cpu"))):
        begin = time.perf_counter()
        with runtime(model) as rt:
            judge, load = rt.judge(), (time.perf_counter() - begin) * 1000
            claim = CONFORMANCE_QUESTIONS[0].text
            rt.check(CONFORMANCE_PROBES[0], claim)
            row = {"provenance": diagnostics.provenance(rt, CONFORMANCE_PROBES), "load_ms": load,
                   "cached_check": timings(lambda i: rt.check(CONFORMANCE_PROBES[0], claim), 10),
                   "new_check": timings(lambda i: rt.check(f"{CONFORMANCE_PROBES[0]} new-{i}", claim), 10)}
            for count in (1, 4, 7):
                qs = CONFORMANCE_QUESTIONS[:count] if count < 7 else CONFORMANCE_QUESTIONS + (CONFORMANCE_QUESTIONS[0],) * 3
                specs = {str(k): Check(q.text) if q.kind == "check" else Pick(q.text, q.options) if q.kind == "pick" else Rate(q.text, q.options)
                         for k, q in enumerate(qs)}
                row[f"{count}_questions"] = timings(lambda i: rt.ask(f"{CONFORMANCE_PROBES[i % 3]} {i}", specs), 10)
            for count in (1, 2, 4, 8):
                row[f"batch_{count}"] = timings(lambda i: rt.ask_many([f"{CONFORMANCE_PROBES[k % 3]} batch-{i}-{k}" for k in range(count)],
                                                                      {"check": Check(claim)}), 5)
            if name == "gpu":
                row["fusion"] = diagnostics.fusion_comparison(rt, lambda i: [(f"{CONFORMANCE_PROBES[0]} arm-{i}", CONFORMANCE_QUESTIONS)], 10)
                if args.cost_curves:
                    row["cost_curve"] = diagnostics.cost_curve(rt)
                row["native_attribution"] = diagnostics.timing(rt, [(CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[:1])], 5)
                row["order_control"] = diagnostics.counterbalanced(judge, CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[1])
                row["embedding_control"] = {k: v for k, v in diagnostics.vectors(rt, [(CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[:1])]).items()
                                            if k not in ("vectors", "results")}
            output[name] = row
            files.publish(IMPLEMENTATION, output)
            print(json.dumps({"finished": name, "load_ms": load, "cached": row["cached_check"], "new": row["new_check"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Explicit development diagnostics; production settings remain unchanged.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("implementation").add_argument("--cost-curves", action="store_true")
    for name in ("reducer", "layout"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--device", default="gpu")
        command.add_argument("--repeats", type=int, default=10)
        command.add_argument("--cases", type=int, default=5)
    args = parser.parse_args()
    if args.command == "implementation":
        return implementation(args)
    found = {"reducer": reducer, "layout": layouts}[args.command](args)
    files.publish(OUTPUT, {**(files.read_json(OUTPUT) or {}), "schema": 1, "model": args.model, "device": args.device, args.command: found})
    print(json.dumps({k: v for k, v in found.items() if k != "runs"}, default=str)[:4000])


if __name__ == "__main__":
    main()
