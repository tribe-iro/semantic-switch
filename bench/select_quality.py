from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from semantic_switch import Check, DEFAULT_JUDGE, Label, Model, Select, files
from semantic_switch.artifacts import artifact_digest
from semantic_switch.metrics import Progress, correctness, membership, ratio
from semantic_switch.presentation import identity as presentation_identity
from bench.external_primitives import CACHE, CATEGORY, evaluator, install, state
from bench.outcomes import configuration, finish, judge_spec, materialize, resumable, resumed, runtime, work_key

LIMIT = 100
TASKS = {"go-emotions-noul-test-v1": "Which emotions does the text express?",
         "civil-comments-noul-test-v1": "Which of these attributes does the comment have?"}
RECEIPT = ("elapsed_ms", "usage", "returned_model", "returned_provider", "completion", "error")
BOUNDARY = (f"first {LIMIT} cases (every case of the pinned data) of the {CATEGORY} S1MB benchmarks whose questions are independent yes/no criteria over one input "
            f"({', '.join(TASKS)}); each case is asked once as public Select over Label definitions taken from the benchmark's true-option "
            "descriptions and once as the equivalent named Checks of the same descriptions, select first. Gold is a membership only where the "
            "benchmark target is hard (0 or 1); soft targets are unlabelled. A community benchmark whose overlap with any judge's training data is unknown; "
            "pooled memberships are correlated within a case and are not independent samples")


def labels(benchmark, case):
    found = {}
    for q in case["questions"]:
        options = {o["id"]: o["description"] for o in q["options"]}
        if q["task"] != "noul" or q["system_prompt"] or set(options) != {"false", "true"}:
            raise ValueError(f"{benchmark}/{case['case_id']}#{q['id']}: not an unprompted true/false criterion; Select needs one membership definition per label")
        found[q["id"]] = Label(options["true"])
    return found


def definitions(question, members):
    return Select(question, members), {key: Check(label.description) for key, label in members.items()}


def timed(recording, operation):
    first = len(recording.records) if recording else 0
    started = time.perf_counter()
    result = materialize(operation())
    return {"ms": (time.perf_counter() - started) * 1000, "result": result,
            "transport": [{name: receipt[name] for name in RECEIPT} for receipt in recording.records[first:]] if recording else None}


def predict(rt, recording, case, question, members):
    select, checks = definitions(question, members)
    return {"select": timed(recording, lambda: rt.ask(state(case), {"select": select})["select"]),
            "checks": timed(recording, lambda: rt.ask(state(case), checks))}


def tasks(rows):
    found = {}
    for row in rows:
        if row["benchmark"] in TASKS:
            members = labels(row["benchmark"], row["case"])
            first = found.setdefault(row["benchmark"], (members, []))
            if first[0] != members:
                raise ValueError(f"{row['benchmark']}/{row['case']['case_id']}: label definitions differ from the benchmark's first case")
            first[1].append(row)
    missing = sorted(set(TASKS) - set(found))
    if missing:
        raise ValueError(f"S1MB cases lack {', '.join(missing)}; reinstall the pinned evaluator data")
    return found


def gold(target):
    return target["true"] == 1 if max(target.values()) == 1 else None


def raw(member):
    prediction = member and member["prediction"]
    return prediction["value"] if prediction and prediction["kind"] == "value" else None


def memberships(row, arm):
    result = row["arms"][arm]["result"]
    found = result["members"] if arm == "select" else result
    return {key: (gold(target), found.get(key)) for key, target in row["targets"].items()}


def members(pairs):
    hard = [(label, member) for label, member in pairs if label is not None]
    return {**membership({"gold": label, "raw": raw(member), "value": member["value"] if member else None, "accepted": member is not None and member["status"] == "ok",
                          "outcome": "unexamined" if member is None else member["status"]} for label, member in pairs),
            **correctness((None if raw(member) is None else raw(member) == label, member is not None and member["status"] == "ok") for label, member in hard),
            "statuses": dict(sorted(Counter("unexamined" if member is None else member["status"] for _, member in pairs).items()))}


def exact(rows, arm):
    cases = [list(memberships(row, arm).values()) for row in rows]
    hard = [[(label, member) for label, member in pairs if label is not None] for pairs in cases]
    predicted = [pairs for pairs in hard if pairs and all(raw(member) is not None for _, member in pairs)]
    accepted = [pairs for pairs in hard if pairs and all(member is not None and member["status"] == "ok" for _, member in pairs)]
    return {"scope": "per case, the hard-labelled memberships only; a case counts when every one of them was examined and predicted (accepted)",
            "cases": len(rows), "fully_labelled_cases": sum(all(label is not None for label, _ in pairs) for pairs in cases),
            "cases_with_hard_memberships": sum(bool(pairs) for pairs in hard),
            "raw": {"eligible": len(predicted), "agree": sum(all(raw(member) == label for label, member in pairs) for pairs in predicted)},
            "accepted": {"eligible": len(accepted), "agree": sum(all(member["value"] == label for label, member in pairs) for pairs in accepted)}}


def parity(rows):
    pairs = [(label, select, check) for row in rows for (label, select), (_, check) in zip(memberships(row, "select").values(), memberships(row, "checks").values())]
    both = [(label, s, c) for label, s, c in pairs if raw(s) is not None and raw(c) is not None]
    gaps = [abs(s["p_true"] - c["p_true"]) for _, s, c in both]
    moved = Counter(f"{'correct' if raw(c) == label else 'wrong'}->{'correct' if raw(s) == label else 'wrong'}" for label, s, c in both if label is not None)
    return {"scope": "same case and label; checks -> select", "memberships": len(pairs), "both_predicted": len(both),
            "raw_agreement": sum(raw(s) == raw(c) for _, s, c in both), "raw_agreement_rate": ratio(sum(raw(s) == raw(c) for _, s, c in both), len(both)),
            "status_value_agreement": sum((s or {}).get("status") == c["status"] and (s or {}).get("value") == c["value"] for _, s, c in pairs),
            "p_true_abs_difference": {"mean": statistics.fmean(gaps) if gaps else None, "max": max(gaps, default=None)},
            "hard_label_transitions": dict(sorted(moved.items())), "fixes": moved["wrong->correct"], "regressions": moved["correct->wrong"]}


def timing(rows, arm):
    ms = sorted(row["arms"][arm]["ms"] for row in rows)
    receipts = [receipt for row in rows for receipt in row["arms"][arm]["transport"] or ()]
    return {"case_ms_p50": statistics.median(ms) if ms else None, "case_ms_p95": ms[min(len(ms) - 1, int(.95 * len(ms)))] if ms else None,
            "http_attempts": len(receipts) if receipts else None, "failed_attempts": sum(r["completion"] != "response" for r in receipts) if receipts else None,
            "returned_models": sorted({r["returned_model"] for r in receipts if r["returned_model"]})}


def report(rows):
    def arm(group, name):
        return {"members": members([pair for row in group for pair in memberships(row, name).values()]), "exact_set": exact(group, name), "timing": timing(group, name),
                **({"incomplete_cases": sum(not row["arms"][name]["result"]["complete"] for row in group),
                    "statuses": dict(Counter(row["arms"][name]["result"]["status"] for row in group))} if name == "select" else {})}
    groups = {benchmark: [row for row in rows if row["benchmark"] == benchmark] for benchmark in TASKS}
    return {**{name: arm(rows, name) for name in ("select", "checks")}, "parity": parity(rows),
            "tasks": {benchmark: {"cases": len(group), "select": arm(group, "select"), "checks": arm(group, "checks"), "parity": parity(group)}
                      for benchmark, group in groups.items()}}


def summarize(rows, provenance, judge, identity, found):
    return {"schema": 1, "boundary": BOUNDARY, "provenance": provenance, "judge": judge, "identity": identity, "presentation": presentation_identity(),
            "definitions": {benchmark: {"question": TASKS[benchmark], "labels": {key: label.description for key, label in members.items()}}
                            for benchmark, (members, _) in found.items()},
            **report(sorted(rows, key=lambda row: (row["benchmark"], row["position"]))), "rows": rows}


def run(args):
    provenance = install()
    cases_path = CACHE / f"cases-{LIMIT}.jsonl"
    evaluator("cases", LIMIT, cases_path)
    found = tasks(files.read_jsonl(cases_path))
    keyed = {work_key(benchmark, row['case']['case_id']): (benchmark, position, row) for benchmark, (_, rows) in found.items() for position, row in enumerate(rows)}
    judge = args.judge or DEFAULT_JUDGE
    model, recording = judge_spec(judge, provider=args.provider, key_file=args.key_file, device=args.device, max_input_tokens=args.max_input_tokens)
    config = configuration("bench.select_quality", args, *[model] if isinstance(model, Model) else [], provenance=provenance,
                           cases=files.sha256_file(cases_path), keys=artifact_digest(sorted(keyed)), presentation=presentation_identity())
    with resumable(args.output, config, fresh=args.fresh) as journal, runtime(model) as rt:
        identity = rt.identity()
        todo = journal.pending(keyed)
        progress = Progress(f"bench.select_quality {judge}", len(keyed), unit="cases", resumed=resumed(journal, keyed, todo))
        for key in todo:
            benchmark, position, row = keyed[key]
            committed = {"benchmark": benchmark, "case_id": row["case"]["case_id"], "position": position, "targets": row["targets"],
                         "arms": predict(rt, recording, row["case"], TASKS[benchmark], found[benchmark][0])}
            journal.commit([(key, committed)])
            progress.record([committed])
        progress.finish()
        return finish(journal, args.output, lambda rows: summarize(rows, provenance, judge, identity, found))


def main():
    parser = argparse.ArgumentParser(description="Public Select on S1MB multi-label tasks, with parity against the equivalent named Checks.")
    parser.add_argument("--judge", help="judge source for semantic_switch.judge_from (default: DEFAULT_JUDGE)")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fresh", action="store_true")
    artifact = run(parser.parse_args())
    for name in ("select", "checks"):
        print(name, json.dumps({key: artifact[name]["members"][key] for key in ("hard", "predicted", "raw_correct", "accepted", "wrong_accepted", "failed", "unresolved")}))
    print("parity", json.dumps({key: artifact["parity"][key] for key in ("both_predicted", "raw_agreement", "fixes", "regressions")}))


if __name__ == "__main__":
    main()
