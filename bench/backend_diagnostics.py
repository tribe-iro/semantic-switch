from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from semantic_switch import Check, Model, Pick, Rate, backend, files
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
from bench import diagnostics
from bench.execution_reproduction import EMOTION, cases
from bench.model_comparison import timings
from bench.outcomes import destination, latency, output_argument, provenance, report, runtime, work_argument, work_directory

LONG = Check("The text expresses one dominant emotion, judged by these descriptions: " +
             " ".join(f"{name} means {text}." for _ in range(4) for name, text in EMOTION.options.items()))


@contextmanager
def phases(scorer):
    events = {"submissions": [], "logits": [], "reductions": [], "rows": []}
    run, logits, reduce = scorer.ctx.decode, scorer.ctx.logits, backend.reduce_row

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

    with patch.object(scorer.ctx, "decode", timed(run, "submissions")), patch.object(scorer.ctx, "logits", timed(logits, "logits")), \
            patch.object(backend, "reduce_row", reduction):
        yield events


def split(events):
    run, wait, reduce = (sum(events[key]) for key in ("submissions", "logits", "reductions"))
    return {"submissions": len(events["submissions"]), "outputs": len(events["reductions"]), "decode_ms": run - reduce,
            "logits_wait_ms": wait, "reduce_ms": reduce}


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
    return {"vocabulary": len(row), "slots": [len(ids) for ids in slots], "latency": latency(samples),
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
        totals = {key: latency(call[key] for call in calls) for key in ("ask_ms", "decode_ms", "logits_wait_ms", "reduce_ms")}
        out[name] = {"questions": len(specs), "warm": True, "submissions": calls[0]["submissions"], "outputs": calls[0]["outputs"],
                     **totals, "reduce_share": latency([call["reduce_ms"] / call["ask_ms"] for call in calls], unit="fraction")["p50"],
                     "reducer": reducer_cost(*rows[0], repeats=20 * repeats)}
    return out


class Counted:
    def __init__(self, L):
        self.L, self.calls, self.tokens = L, 0, []

    def llama_decode(self, ctx, batch):
        self.calls += 1
        self.tokens.append(batch.n_tokens)
        return self.L.llama_decode(ctx, batch)

    def __getattr__(self, name):
        return getattr(self.L, name)


def native(rt):
    scorer = rt.judge().scorer
    return getattr(getattr(scorer, "resident", None), "scorer", scorer)


def captured(rt, switch, inputs, facts):
    scorer, requests = native(rt), []
    score = scorer.score

    def record(batch, **kwargs):
        requests.extend((request, kwargs) for request in batch)
        return score(batch, **kwargs)
    with patch.object(scorer, "score", record):
        for data in inputs:
            switch(data, facts=facts)
    return requests


def replay(scorer, request, kwargs, grouped):
    from semantic_switch.protocols import ScoreBlock
    width = max(len(slots) for slots in request.slots)
    out, done = ScoreBlock.empty(len(request.suffixes), width), set()
    with scorer.operation():
        scorer._bind(kwargs.get("lora"), kwargs.get("binding"))
        if not grouped:
            scorer._shared(request, 0, out)
            return out
        c = scorer.ctx
        scorer._ensure_prefix(request.prefix)
        sequence = scorer._ensure_state(request.prefix, request.state)
        branches = [c_ for c_ in range(scorer.layout.branch, c.n_seq_max) if c_ not in scorer._resident.values()]
        base, write = len(request.prefix) + len(request.state), scorer._emit(out, done)
        try:
            for start in range(0, len(request.suffixes), len(branches)):
                stop = min(start + len(branches), len(request.suffixes))
                used = branches[:stop - start]
                for branch in used:
                    c.cp(sequence, branch)
                c.outputs_per_decode = len(used)
                c.decode(tuple((request.suffixes[row], base, branch, (row, request.slots[row])) for row, branch in zip(range(start, stop), used)), write)
                for branch in used:
                    c.rm(branch)
        finally:
            c.outputs_per_decode = scorer.layout.outputs_per_decode
            for branch in branches:
                c.rm(branch)
        if done != set(range(len(request.suffixes))):
            raise RuntimeError(f"grouped decode materialized outputs {sorted(done)}, not every suffix row")
        return out


def grouped_control(args):
    from semantic_switch.artifacts import load_receipt
    from semantic_switch.tools.deployment import fitted
    from bench import evidence
    from bench.snapshots import DOMAINS, requested
    domain = DOMAINS["zt411"]
    data = evidence.read(args.pack, args.corpus)
    _, frames, meta = evidence.frames_of(data, args.pack)
    inputs = [frame.input for frame in frames if meta[frame.id]["role"] == "check" and frame.condition == domain.module.TRAINED][:args.frames]
    receipt = load_receipt(args.receipt)
    with runtime(fitted(args.receipt, n_ctx=args.n_ctx, prefix_slots=args.prefix_slots)) as rt:
        switch = rt.load(receipt.verified_task(), allow_model_change=True)
        facts = requested(domain.scoped(files.read_yaml(receipt.verified_task())["facts"]))
        requests = captured(rt, switch, inputs, facts)
        scorer = native(rt)
        counted = Counted(scorer.ctx.L)
        scorer.ctx.L = counted
        rows, stable = {"serial": [], "grouped": []}, True
        for request, kwargs in requests:
            replay(scorer, request, kwargs, False)
            replay(scorer, request, kwargs, True)
        for request, kwargs in requests:
            blocks = {}
            for name in (("serial", "grouped") if len(rows["serial"]) % 2 == 0 else ("grouped", "serial")):
                samples = []
                for _ in range(args.repeats):
                    calls, tokens = counted.calls, len(counted.tokens)
                    begin = time.perf_counter()
                    block = replay(scorer, request, kwargs, name == "grouped")
                    samples.append({"ms": (time.perf_counter() - begin) * 1000, "decode_calls": counted.calls - calls,
                                    "submitted_tokens": sum(counted.tokens[tokens:])})
                    stable = stable and (name not in blocks or np.array_equal(blocks[name].logp, block.logp))
                    blocks[name] = block
                rows[name].append({"suffixes": len(request.suffixes), "suffix_tokens": sum(map(len, request.suffixes)), "state_tokens": len(request.state),
                                   "decode_calls": samples[-1]["decode_calls"], "submitted_tokens": samples[-1]["submitted_tokens"],
                                   "latency": latency(sample["ms"] for sample in samples)})
            serial, grouped = blocks["serial"], blocks["grouped"]
            finite = np.isfinite(serial.logp) & np.isfinite(grouped.logp)
            rows["grouped"][-1]["versus_serial"] = {"max_abs_logp": float(np.max(np.abs(serial.logp[finite] - grouped.logp[finite]))) if finite.any() else None,
                                                    "argmax_changes": int(np.sum(np.argmax(serial.logp, 1) != np.argmax(grouped.logp, 1)))}
        workflow = workflow_control(rt, switch, inputs, facts, scorer, counted, args.repeats)
        return {"execution": rt.execution(), "requests": len(requests), "frames": len(inputs), "same_shape_bit_stable": stable,
                'workflow': workflow,
                "kv": {key: scorer.ctx.realized[key] for key in ("kv_unified", "n_ubatch")}, "n_batch": scorer.ctx.n_batch,
                "totals": {name: {"decode_calls": sum(r["decode_calls"] for r in found), "submitted_tokens": sum(r["submitted_tokens"] for r in found),
                                  "p50_ms_sum": sum(r["latency"]["p50"] for r in found)} for name, found in rows.items()},
                "argmax_changes": sum(r["versus_serial"]["argmax_changes"] for r in rows["grouped"]),
                "max_abs_logp": max((r["versus_serial"]["max_abs_logp"] or 0.) for r in rows["grouped"]),
                "scope": "white-box development control on one context: same KV layout, same requests and tokens; serial is the production path, grouped "
                         "uses currently free streams on this instance only; group width can change with prefix residency. Timings include production reduction",
                "rows": rows}


def workflow_control(rt, switch, inputs, facts, scorer, counted, repeats):
    from semantic_switch.protocols import ScoreBlock
    original = scorer.score
    def grouped(requests, **kwargs):
        blocks = [replay(scorer, request, kwargs, True) for request in requests]
        result = ScoreBlock.empty(sum(len(block.logp) for block in blocks), max(block.logp.shape[1] for block in blocks))
        start = 0
        for block in blocks:
            stop = start + len(block.logp)
            result.logp[start:stop, :block.logp.shape[1]], result.log_mass[start:stop], result.error[start:stop] = block.logp, block.log_mass, block.error
            start = stop
        return result
    def semantic(result):
        out = {'status': result.status, 'answer': result.answer, 'rule': result.rule,
                'facts': {name: {key: value for key, value in fact.as_dict().items() if key in ('status', 'value', 'settled', 'prediction', 'complete', 'states', 'unresolved')}
                          for name, fact in result.facts.items()}}
        for fact in out['facts'].values():
            if fact.get('prediction') is not None:
                fact['prediction'] = {key: value for key, value in fact['prediction'].items() if key != 'score'}
        return out
    rows, decisions = {'serial': [], 'grouped': []}, {}
    for repeat in range(repeats):
        for name in (('serial', 'grouped') if repeat % 2 == 0 else ('grouped', 'serial')):
            calls, began = counted.calls, time.perf_counter()
            with patch.object(scorer, 'score', grouped if name == 'grouped' else original):
                found = [semantic(switch(data, facts=facts)) for data in inputs]
            rows[name].append({'ms': (time.perf_counter() - began) * 1000, 'decode_calls': counted.calls - calls})
            decisions.setdefault(name, []).append(found)
    return {'unit': 'whole operation over fixed situations', 'situations': len(inputs),
            'latency': {name: latency(row['ms'] for row in measured) for name, measured in rows.items()},
            'decode_calls': {name: [row['decode_calls'] for row in measured] for name, measured in rows.items()},
            'decision_changes': sum(left != right for a, b in zip(decisions['serial'], decisions['grouped']) for left, right in zip(a, b)),
            'same_shape_decision_stable': all(found == runs[0] for runs in decisions.values() for found in runs),
            'decisions': {name: runs[0] for name, runs in decisions.items()}}


def model_of(args):
    return Model(args.model, device=args.device, prefix_slots=4)


def reducer(args):
    data = cases(work_directory(args.output, args.work), 1)
    with runtime(model_of(args)) as rt:
        rt.ask("warm up", {"warm": Check("The text is about food.")})
        return {"execution": rt.execution(), "n_vocab": rt.judge().scorer.weights.n_vocab,
                "shapes": profile(rt, data, args.repeats)}


def implementation(args):
    output = {}
    destination(args.output)
    for name, model in (("gpu", Model("ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf", device="gpu")),
                        ("cpu", Model("unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf", adapter="qwen3", device="cpu"))):
        begin = time.perf_counter()
        with runtime(model) as rt:
            judge, load = rt.judge(), (time.perf_counter() - begin) * 1000
            claim = CONFORMANCE_QUESTIONS[0].text
            rt.check(CONFORMANCE_PROBES[0], claim)
            row = {"provenance": provenance(rt, CONFORMANCE_PROBES, boundary="explicit backend diagnostic"), "load_ms": load,
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
                if args.cost_curves:
                    row["cost_curve"] = diagnostics.cost_curve(rt)
                row["native_attribution"] = diagnostics.timing(rt, [(CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[:1])], 5)
                row["order_control"] = diagnostics.counterbalanced(judge, CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[1])
                row["embedding_control"] = {k: v for k, v in diagnostics.vectors(rt, [(CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[:1])]).items()
                                            if k not in ("vectors", "results")}
            output[name] = row
            files.publish(args.output, output)
            print(json.dumps({"finished": name, "load_ms": load, "cached": row["cached_check"], "new": row["new_check"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Explicit development diagnostics; production settings remain unchanged.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("implementation")
    command.add_argument("--cost-curves", action="store_true")
    output_argument(command)
    command = commands.add_parser("grouped")
    command.add_argument("--receipt", required=True)
    command.add_argument("--pack", required=True)
    command.add_argument("--corpus")
    command.add_argument("--frames", type=int, default=8)
    command.add_argument("--repeats", type=int, default=5)
    command.add_argument("--n-ctx", type=int)
    command.add_argument("--prefix-slots", type=int, default=4)
    output_argument(command)
    command = commands.add_parser("reducer")
    command.add_argument("--model", required=True)
    command.add_argument("--device", default="gpu")
    command.add_argument("--repeats", type=int, default=10)
    output_argument(command)
    work_argument(command)
    args = parser.parse_args()
    if args.command == "implementation":
        return implementation(args)
    if args.command == "grouped":
        found = grouped_control(args)
        report(args.output, {"schema": 1, "receipt": args.receipt, "grouped": found})
        print(json.dumps({k: v for k, v in found.items() if k != "rows"}, default=str))
        return
    found = reducer(args)
    report(args.output, {"schema": 1, "model": args.model, "device": args.device, args.command: found})
    print(json.dumps({k: v for k, v in found.items() if k != "runs"}, default=str)[:4000])


if __name__ == "__main__":
    main()
