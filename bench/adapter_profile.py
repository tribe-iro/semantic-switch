from __future__ import annotations

import argparse
import hashlib
import json
import resource
import struct
import time
from pathlib import Path

import numpy as np

from semantic_switch import Model, Runtime, files
from semantic_switch.backend import gguf
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS, LlamaJudge
from semantic_switch.judgment import Judgment, Policy, outcome, predict
from semantic_switch.metrics import answer_key, observed, percentile, scored_answer
from semantic_switch.resident import Adapter

from bench.diagnostics import lora_support, native_build, row_scores, switch_adapter
from bench.outcomes import destination, fingerprint, report

SCHEMA = 3
TARGETS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
GGUF_STRING, GGUF_FLOAT32, GGUF_F16, ALIGNMENT = 8, 6, 1, 32
SEQUENCE = ("A", "B", "base", "A", "A", "B", "B", "base", "base", "A")
SCOPES = {
    "setter": "llama_set_adapters_lora host call alone under the context lock; the scheduler re-reservation it marks is paid by the next scoring call",
    "first_scoring": "one complete Scorer.score call over the workload directly after the binding step: prefix/state preparation, every suffix decode, "
                     "logits read for every row",
    "steady_scoring": "the identical Scorer.score call repeated at once on the same binding (repeated identical request)",
    "public_call": "one public Judge.score(text, questions) call through the Runtime role binding on the first probe input",
    "public_reload": "the same public call on a pool limited to one adapter, so every switch evicts and reloads an adapter",
}
SYNCHRONIZATION = "scoring timers stop after llama_get_logits_ith has read every scored row, which synchronizes the backend; the setter timer covers the host call"
CONTROL = "the same binding replayed without switching through the identical schedule of prefix-cache namespaces, in the same repeat"
DIAGNOSTIC = ("per scored row: native finite/error shape, log probabilities and log answer mass; the canonical judge scores of that ScoreBlock row "
              "(semantic_switch.judge) as a probability observation (semantic_switch.metrics); and the public status, answer and acceptance of the "
              "policy reducer (semantic_switch.judgment.outcome) at the declared acceptance threshold")
EQUIVALENCE = ("equal native shape, errors, public status, answer and acceptance, and canonical probabilities and answer mass within the declared "
               "tolerance; log probabilities are reported, not gated")
CONTROLS = {
    "fresh_repeat": "the same binding scored again after clearing every cache, against its fresh reference: determinism",
    "state_reuse": "the same binding scored again at once on the warm prefix/state left by its fresh reference: cache reuse only",
    "batch_geometry": "the same binding scoring each input as its own call from a cleared cache, against the batched fresh reference: native "
                      "batch/sequence geometry only",
    "reference_path": "the same binding through the uncached whole-sequence reference path, against the shared prefix/state/suffix path: "
                      "native path only",
    "layout_numerics": "every first scoring call of the A -> B -> base -> A cycles, against its binding's fresh reference: resident cache "
                       "contents of other namespaces",
}
CHANGES = ("shape_changes", "answer_changes", "status_changes", "acceptance_changes")


def _string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def probe_adapter(base: str, destination: Path, *, rank: int, alpha: float, seed: int, label: str) -> dict:
    descriptor = gguf(base)
    meta, layout = descriptor.meta, descriptor.tensors
    stems = sorted(name for name in layout if name.endswith(".weight") and name.split(".")[0] == "blk" and name.split(".")[2] in TARGETS)
    if not stems:
        raise SystemExit(f"{base}: no {', '.join(TARGETS)} projection tensors to adapt")
    tensors = []
    for stem in stems:
        n_in, n_out = layout[stem].shape[:2]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(json.dumps([seed, label, stem]).encode()).digest()[:8], "little"))
        tensors.append((stem + ".lora_a", (n_in, rank), (rng.standard_normal((rank, n_in)) / np.sqrt(n_in)).astype(np.float16)))
        tensors.append((stem + ".lora_b", (rank, n_out), (rng.standard_normal((n_out, rank)) * 0.02 / np.sqrt(rank)).astype(np.float16)))
    kv = [("general.architecture", GGUF_STRING, meta["general.architecture"]), ("general.type", GGUF_STRING, "adapter"),
          ("adapter.type", GGUF_STRING, "lora"), ("adapter.lora.alpha", GGUF_FLOAT32, float(alpha)),
          ("general.name", GGUF_STRING, f"semantic-switch deterministic probe {label} (not trained)")]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, kind, value in kv:
        header += _string(key) + struct.pack("<I", kind) + (_string(value) if kind == GGUF_STRING else struct.pack("<f", value))
    offset, infos, blobs = 0, b"", []
    for name, dims, data in tensors:
        infos += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", GGUF_F16, offset)
        blob = data.tobytes()
        blobs.append(blob + b"\0" * (-len(blob) % ALIGNMENT))
        offset += len(blobs[-1])
    preamble = header + infos
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(preamble + b"\0" * (-len(preamble) % ALIGNMENT) + b"".join(blobs))
    return {"path": str(destination), "label": label, "kind": "deterministic-probe", "trained": False, "rank": rank, "alpha": alpha, "seed": seed,
            "targets": len(stems), "generator": "bench.adapter_profile.probe_adapter@1"}


def requests(judge, texts):
    adapter = judge.adapter
    return [adapter.request(adapter.state(text), list(CONFORMANCE_QUESTIONS)) for text in texts]


def workload(batch, texts) -> dict:
    return {"inputs": len(texts), "questions": len(CONFORMANCE_QUESTIONS), "rows": sum(len(r.suffixes) for r in batch),
            "prefix_tokens": sum(map(len, dict.fromkeys(r.prefix for r in batch))), "state_tokens": sum(len(r.state) for r in batch),
            "suffix_tokens": sum(len(suffix) for r in batch for suffix in r.suffixes), "digest": hashlib.sha256(json.dumps(texts).encode()).hexdigest()}


def judgment(question) -> Judgment:
    return Judgment.check(question.text) if question.kind == "check" else Judgment(question.kind, question.text, question.options, question.options)


def finite(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def native(block, judgments, policy) -> list[dict]:
    rows = []
    for row, j in enumerate(judgments):
        logp, scores = block.logp[row, :len(j.keys)], row_scores(block, row, len(j.keys))
        status, index, why = ("failed", None, scores.error) if scores.error else outcome(j, scores.probs, policy, scores.mass)
        answer = scored_answer(None if scores.error else predict(j, scores.probs), None, status, j.keys)
        rows.append({"native": {"columns": int(block.logp.shape[1]), "finite": np.isfinite(logp).tolist(), "logp": [finite(v) for v in logp],
                                "log_mass": finite(block.log_mass[row]), "error": block.error[row]},
                     "observation": observed(j.keys, None if scores.probs is None else scores.probs.tolist(), kind=j.kind, stage="final",
                                             normalization=LlamaJudge.normalization, status=status, source="judge", mass=scores.mass,
                                             accepted=answer_key(j.value(index)) if status == "ok" else None),
                     "public": {"status": status, "reason": why, "answer": answer["answer"], "accepted": answer["accepted"]}})
    return rows


def gap(pairs) -> float:
    return max((abs(x - y) for x, y in pairs if x is not None and y is not None), default=0.)


def shape(row) -> tuple:
    return row["native"]["finite"], row["native"]["error"] is None, row["native"]["log_mass"] is None, row["observation"]["mass"] is None


def compare(a, b) -> dict:
    if len(a) != len(b):
        raise ValueError(f"cannot compare {len(a)} scored rows with {len(b)}")
    pairs = list(zip(a, b))
    probabilities = [pair for x, y in pairs if x["observation"]["probabilities"] and y["observation"]["probabilities"]
                     for pair in zip(x["observation"]["probabilities"], y["observation"]["probabilities"])]
    return {"rows": len(pairs), "max_abs_prob": gap(probabilities), "max_abs_mass": gap((x["observation"]["mass"], y["observation"]["mass"]) for x, y in pairs),
            "max_abs_logp": gap(pair for x, y in pairs for pair in zip(x["native"]["logp"], y["native"]["logp"])),
            "max_abs_log_mass": gap((x["native"]["log_mass"], y["native"]["log_mass"]) for x, y in pairs),
            "shape_changes": sum(shape(x) != shape(y) for x, y in pairs),
            **{f"{name}_changes": sum(x["public"][field] != y["public"][field] for x, y in pairs)
               for name, field in (("answer", "answer"), ("status", "status"), ("acceptance", "accepted"))},
            "errors": [row["public"]["reason"] for row in (*a, *b) if row["public"]["status"] == "failed"]}


def within(difference: dict, tolerance: float) -> bool:
    return not difference["errors"] and not any(difference[key] for key in CHANGES) and \
        max(difference["max_abs_prob"], difference["max_abs_mass"]) <= tolerance


def timed(call):
    started = time.perf_counter()
    out = call()
    return out, (time.perf_counter() - started) * 1000


def distribution(values) -> dict | None:
    v = sorted(float(value) for value in values)
    return {"n": len(v), "min": v[0], "median": percentile(v, .5), "p90": percentile(v, .9), "max": v[-1]} if v else None


def scored(scorer, call, read):
    before = dict(scorer.stats)
    block, ms = timed(call)
    return read(block), ms, {key: value - before.get(key, 0) for key, value in scorer.stats.items() if value != before.get(key, 0)}


def cycle(scorer, views, batch, steps, read):
    scorer.invalidate()
    for role in steps:
        view = views[role]
        with scorer.operation(), view.resident.pool.use(view.adapter) as lora:
            switched, setter = switch_adapter(scorer, lora)
            first = scored(scorer, lambda: view.score(batch), read)
            steady = scored(scorer, lambda: view.score(batch), read)
        yield {"binding": role, "switched": switched, "setter_ms": setter, "first": first, "steady": steady}


def replay(scorer, views, batch, steps, role, read):
    view, out = views[role], []
    with scorer.operation(), view.resident.pool.use(view.adapter) as lora:
        scorer.score(batch, lora=lora, binding=f"layout:{role}:")
        scorer.invalidate()
        for step in steps:
            bound = {"lora": lora, "binding": f"layout:{role}:{step}"}
            out.append({kind: scored(scorer, lambda: scorer.score(batch, **bound), read) for kind in ("first", "steady")})
    return out


def cache(work: dict) -> dict:
    return {"prefix": "hit" if work.get("prefix_hits") else "miss", "state_hits": work.get("state_hits", 0), "state_misses": work.get("state_misses", 0)}


def strip(work: dict) -> dict:
    return {key: value for key, value in work.items() if key != "adapter_switches"}


def paired(step: dict, control: dict, kind: str) -> dict:
    (got, ms, work), (reference, control_ms, control_work) = step[kind], control[kind]
    return {"scope": f"{kind}_scoring", "ms": ms, "cache": cache(work), "work": work,
            "control": {"ms": control_ms, "cache": cache(control_work), "matched_work": strip(work) == strip(control_work)},
            "overhead_ms": ms - control_ms, "vs_control": compare(got, reference)}


def timing(cycles: list) -> dict:
    out = {"setter": {group: distribution(c["setter"]["ms"] for c in cycles if c["switched"] == flag) for group, flag in (("switched", True), ("unchanged", False))}}
    for kind in ("first", "steady"):
        out[f"{kind}_scoring"] = {group: {"ms": distribution(c[kind]["ms"] for c in rows), "control_ms": distribution(c[kind]["control"]["ms"] for c in rows),
                                          "paired_overhead_ms": distribution(c[kind]["overhead_ms"] for c in rows)}
                                  for group, flag in (("switched", True), ("unchanged", False)) if (rows := [c for c in cycles if c["switched"] == flag])}
    return out


def public(rt, scorer, roles, text, repeats) -> dict:
    orders = {"switched": SEQUENCE, "grouped": tuple(sorted(SEQUENCE, key=list(roles).index))}
    calls = []
    for repeat in range(repeats):
        for order, steps in orders.items():
            scorer.invalidate()
            previous = None
            for role in steps:
                _, ms = timed(lambda role=role: rt.judge(role).score(text, list(CONFORMANCE_QUESTIONS)))
                calls.append({"scope": "public_call", "repeat": repeat, "order": order, "binding": role, "switched": role != previous, "ms": ms})
                previous = role
    return {"workload": {"inputs": 1, "questions": len(CONFORMANCE_QUESTIONS)}, "orders": {name: list(steps) for name, steps in orders.items()},
            "pairing": "unpaired: each order starts from a cleared cache, so cache states differ between orders", "calls": calls,
            "summary": {order: {group: distribution(c["ms"] for c in calls if c["order"] == order and c["switched"] == flag)
                                for group, flag in (("switched", True), ("unchanged", False))} for order in orders}}


def worst(differences) -> dict:
    differences = list(differences)
    return {**{key: (sum if key == "rows" or key.endswith("_changes") else max)(d[key] for d in differences) for key in differences[0] if key != "errors"},
            "errors": [error for d in differences for error in d["errors"]]}


def bounded(measured, requirement) -> dict:
    return {"status": "undeclared" if requirement is None else "fail" if measured > requirement else "pass", "measured": measured, "requirement": requirement}


def assess(report: dict, args, cycles: list) -> dict:
    effect = report["primitives"]["adapter_effect"]
    switched = [c for c in cycles if c["switched"]]
    switching = [c[kind]["vs_control"] for c in cycles for kind in ("first", "steady")]
    roles = {"fresh_repeat": report["primitives"]["fresh_repeat"], "state_reuse": report["primitives"]["state_reuse"],
             "batch_geometry": report["primitives"]["batch_geometry"], "reference_path": report["primitives"]["uncached_vs_cached_reference"],
             "layout_numerics": report["layout_numerics"]}
    public = [c["ms"] for c in report["public"]["calls"] if c["switched"]]
    return {
        "application": {"status": "fail" if any(not d["max_abs_prob"] and not d["max_abs_mass"] for d in effect.values()) else "pass",
                        "measured": {name: {"max_abs_prob": d["max_abs_prob"], "max_abs_mass": d["max_abs_mass"]} for name, d in effect.items()},
                        "scope": "each adapter changes canonical scores against base and against the other adapter"},
        "switch_correctness": {"status": "pass" if all(within(d, args.tolerance) for d in switching) else "fail", "measured": worst(switching),
                               "requirement": EQUIVALENCE, "scope": CONTROL},
        **{name: {"status": "pass" if all(within(d, args.tolerance) for d in per_role.values()) else "fail", "measured": per_role,
                  "requirement": EQUIVALENCE, "scope": CONTROLS[name]} for name, per_role in roles.items()},
        "setter_latency": bounded(max(c["setter"]["ms"] for c in switched), args.max_setter_ms),
        "first_scoring_latency": bounded(max(c["first"]["ms"] for c in switched), args.max_first_scoring_ms),
        "public_latency": bounded(max(public), args.max_public_call_ms),
        "adapter_memory": bounded(max(entry["memory"]["allocated_bytes"] for entry in report["adapters"].values()),
                                  None if args.max_adapter_mib is None else args.max_adapter_mib * 2 ** 20),
    }


def profile(args) -> dict:
    base = str(Path(args.base).resolve())
    abi = lora_support()
    requirements = {"max_first_scoring_ms": args.max_first_scoring_ms, "max_setter_ms": args.max_setter_ms, "max_public_call_ms": args.max_public_call_ms,
                    "max_adapter_mib": args.max_adapter_mib, "max_prob_diff": args.tolerance, "acceptance_threshold": args.accept}
    report = {"schema": SCHEMA, "kind": "adapter-profile", "code": dict(fingerprint()), "build": native_build(), "abi": abi,
              "base": {"path": base, "sha256": files.sha256_file(base)}, "device": args.device, "requirements": requirements,
              "scopes": SCOPES, "synchronization": SYNCHRONIZATION, "control": CONTROL, "controls": CONTROLS, "diagnostic": DIAGNOSTIC,
              "equivalence": EQUIVALENCE, "sequence": list(SEQUENCE), "repeats": args.repeats,
              "scope": "fake-free native primitive measurement on standalone probe inputs; no check evidence, no specialist quality claim; "
                       "correctness under deterministic probe adapters does not establish fitted-adapter accuracy",
              "primitives": {}}
    if abi["status"] != "supported":
        report["assessment"] = {"key": None, "capability": "unsupported", "failing_primitive": "abi", "failures": ["abi"]}
        return report
    policy = Policy.of(args.accept)
    judged = [judgment(question) for _ in args.texts for question in CONFORMANCE_QUESTIONS]

    def read(block):
        return native(block, judged, policy)

    work = Path(args.work)
    adapters = {}
    for label, given in zip("AB", args.adapter or (None, None)):
        record = {"path": str(Path(given).resolve()), "kind": "supplied", "trained": None} if given else \
            probe_adapter(base, work / f"probe-{label}.gguf", rank=args.rank, alpha=args.alpha, seed=args.seed, label=label)
        verified = Adapter.verified(record["path"], base)
        adapters[label] = {**record, **verified.record(), "memory": {"file_bytes": verified.file_bytes, "tensor_bytes": verified.tensor_bytes}}
    report["adapters"] = adapters
    models = {"base": Model(base, device=args.device, max_adapters=2), **{label: Model(base, device=args.device, lora=entry["path"], max_adapters=2)
                                                                          for label, entry in adapters.items()}}
    with Runtime(bindings={**models, "judge": models["base"]}) as rt:
        judge, load_ms = timed(lambda: rt.judge("base"))
        report["cold_load_ms"], report["execution"] = load_ms, rt.execution("base")
        views = {role: rt.judge(role).scorer for role in models}
        scorer = views["base"].resident.scorer
        batch = requests(judge, args.texts)
        report["workload"] = workload(batch, args.texts)
        fresh = {}

        def reference(role):
            scorer.invalidate()
            return read(views[role].score(batch))

        def reused(role):
            reference(role)
            rows, _, spent = scored(scorer, lambda: views[role].score(batch), read)
            return {**compare(rows, fresh[role]), "cache": cache(spent)}

        def separate(role):
            rows, start = [], 0
            for request in batch:
                scorer.invalidate()
                rows += native(views[role].score([request]), judged[start:start + len(request.suffixes)], policy)
                start += len(request.suffixes)
            return rows

        for role in models:
            fresh[role], ms = timed(lambda role=role: reference(role))
            report["primitives"].setdefault("cold_first_scoring_ms", {})[role] = ms
        report["primitives"]["adapter_effect"] = {"A_vs_base": compare(fresh["A"], fresh["base"]), "B_vs_A": compare(fresh["B"], fresh["A"])}
        uncached = {role: read(views[role].score(batch, reference=True)) for role in models}
        report["references"] = {role: {"fresh": fresh[role], "reference_path": uncached[role]} for role in models}
        report["primitives"]["uncached_vs_cached_reference"] = {role: compare(uncached[role], fresh[role]) for role in models}
        report["primitives"]["fresh_repeat"] = {role: compare(reference(role), fresh[role]) for role in models}
        report["primitives"]["state_reuse"] = {role: reused(role) for role in models}
        report["primitives"]["batch_geometry"] = {role: compare(separate(role), fresh[role]) for role in models}
        cycles = []
        for repeat in range(args.repeats):
            controls = {role: replay(scorer, views, batch, SEQUENCE, role, read) for role in models}
            for index, step in enumerate(cycle(scorer, views, batch, SEQUENCE, read)):
                role = step["binding"]
                entry = {"repeat": repeat, "step": index, "binding": role, "switched": step["switched"], "setter": {"scope": "setter", "ms": step["setter_ms"]},
                         **{kind: paired(step, controls[role][index], kind) for kind in ("first", "steady")}}
                entry["first"]["vs_fresh"] = compare(step["first"][0], fresh[role])
                cycles.append(entry)
        report["cycles"] = cycles
        report["timing"] = timing(cycles)
        report["layout_numerics"] = {role: worst(c["first"]["vs_fresh"] for c in cycles if c["binding"] == role) for role in models}
        report["public"] = public(rt, scorer, models, args.texts[0], args.repeats)
        pool = views["A"].resident.pool
        report["pool"] = {**pool.stats, "placement": pool.placement()}
        for label, entry in adapters.items():
            placed = report["pool"]["placement"][entry["sha256"]]
            entry["memory"].update(estimate=models[label].resources()["adapter"].record(), charged=placed["charged"], allocated_buffers=placed["buffers"],
                                   allocated_bytes=sum(placed["buffers"].values()))
        report["memory"] = {"buffers": rt.info("base")["capacity"].get("buffers"), "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            "job": rt.job.diagnostics() if rt.job else None}
        report["counters"] = rt.info("base")["capabilities"]["counters"]
    with Runtime(bindings={"judge": Model(base, device=args.device, max_adapters=1),
                  **{role: Model(base, device=args.device, lora=entry["path"], max_adapters=1) for role, entry in adapters.items()}}) as rt:
        rt.judge("A")
        evictions = []
        for role in ("A", "B", "A", "B"):
            _, ms = timed(lambda role=role: rt.judge(role).score(args.texts[0], list(CONFORMANCE_QUESTIONS)))
            evictions.append({"scope": "public_reload", "binding": role, "ms": ms})
        report["eviction"] = {"calls": evictions, **rt.judge("A").scorer.resident.pool.stats}
    assessments = assess(report, args, cycles)
    failures = [name for name, entry in assessments.items() if entry["status"] == "fail"]
    placement = {"device": args.device, "execution": report["execution"],
                 "adapter_buffers": sorted({name for entry in adapters.values() for name in entry["memory"]["allocated_buffers"]})}
    report["assessment"] = {"key": {"build": report["build"], "base": report["base"]["sha256"], "placement": placement,
                                    "workload": {**report["workload"], "sequence": list(SEQUENCE), "repeats": args.repeats}, "requirements": requirements},
                            "capability": "unsupported" if failures else "unproven" if any(e["status"] == "undeclared" for e in assessments.values())
                            else "supported", "failing_primitive": failures[0] if failures else None, "failures": failures,
                            "max_switch_diff": assessments["switch_correctness"]["measured"]["max_abs_prob"], "assessments": assessments}
    return report


def main():
    parser = argparse.ArgumentParser(description="runtime LoRA adapter profile on one resident base: ABI, conversion validity, setter and scoring cost against "
                                                 "matched no-switch controls, A -> B -> base -> A cycles, eviction/reload and memory. Explicit GPU model job.")
    parser.add_argument("--base", required=True, help="pipeline-built quantized base GGUF (e.g. Gemma E2B Q8_0)")
    parser.add_argument("--adapter", action="append", help="two converted adapter GGUFs; omitted: deterministic non-trained probe adapters are generated")
    parser.add_argument("--work", required=True, help="directory for generated probe adapters")
    parser.add_argument("--output", required=True, help="new receipt JSON path; an existing receipt is never overwritten")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="declared max canonical probability and answer-mass difference for every correctness and cache-equivalence control")
    parser.add_argument("--accept", type=float, default=.9, help="policy threshold for the public status/acceptance of every scored row")
    parser.add_argument("--max-setter-ms", type=float, help="declared adapter setter latency requirement")
    parser.add_argument("--max-first-scoring-ms", type=float, help="declared latency of the first complete scoring call over the workload after a switch")
    parser.add_argument("--max-public-call-ms", type=float, help="declared latency of one complete public Judge.score call directly after a binding change")
    parser.add_argument("--max-adapter-mib", type=float, help="declared allocated bytes per adapter")
    parser.add_argument("--texts", type=Path, help="one standalone probe input per line (default: conformance probes); never check evidence")
    args = parser.parse_args()
    if args.adapter is not None and len(args.adapter) != 2:
        parser.error("give exactly two --adapter files, or none for generated probes")
    destination(args.output)
    args.texts = [line for line in args.texts.read_text().splitlines() if line.strip()] if args.texts else list(CONFORMANCE_PROBES)
    print(json.dumps(report(args.output, profile(args))["assessment"]))


if __name__ == "__main__":
    main()
