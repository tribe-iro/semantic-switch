from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import venv

COMMIT = "39f122bd51d42d21e4ee1ae95974e462fd09a58d"
ARCHIVE_SHA256 = "34ab8631bbfda943b961c2f77a10c282126d388a4f3f6b121eaf787fa89b6f87"
DATASET = "hotchpotch/s1mb-dataset"
REVISION = "b8dc678ecefb471c6d37311d8c93f65249622842"
CHECKSUMS_SHA256 = "1db098a52dc4174d4224587fa446b2bb2476d646fb5942d6f65b4460b0cd7983"
CATEGORY = "english-v1"
CACHE = Path(__file__).resolve().parents[1] / ".cache" / "s1mb"
SOURCE = CACHE / "src" / f"S1MB-{COMMIT}"
DATA = SOURCE / "evaluator" / "data"
PYTHON = CACHE / "venv" / "bin" / "python"
MAX_OPTIONS = 20
TOLERANCE = 1e-5
OUT_OF_SCOPE = ("gsm8k", "aqua-rat", "fol-nli", "logical-entailment", "ruletaker", "robust-lr", "nlsat", "cladder", "corr2cause", "babi-nli",
                "lonli", "temporal-nli", "stepgame", "spartqa", "boardgameqa", "context-retention", "reasoning-control", "painting-geometry",
                "tic-tac-toe", "snake", "drone-control", "vizdoom")
INVOICE = "open-jev--workflow-controls-v1--invoice-processing-noul-test-v1/open_jev__workflow-controls-v1__invoice_processing:workflow-controls-v1:invoice_processing:42:"
SECURITY = "open-jev--workflow-controls-v1--security-incidents-noul-test-v1/open_jev__workflow-controls-v1__security_incidents:workflow-controls-v1:security_incidents:42:"
TRACE = "open-jev--workflow-controls-v1--agent-trace-observability-noul-test-v1/open_jev__workflow-controls-v1__agent_trace_observability:workflow-controls-v1:agent_trace_observability:42:"
CUSTOMER = "open-jev--customer-control-v1-{}-test-v1/open_jev__customer-control-v1:customer-control-v1:42:"
REVIEW = (
    ("invoice", f"{INVOICE}1:v0:SCHEDULE", "SCHEDULE", "e2b", "task_gold",
     "The rule requires 'the same full-payment conditions as PAY'; PAY is a separate single-question case, so this criterion is incomplete for every judge."),
    ("invoice", f"{INVOICE}12:v1:ROUTE FOR APPROVAL", "ROUTE FOR APPROVAL", "e2b", "model",
     "No approval exists; vendor, account, purchase order and delivery match, so the invoice is otherwise payable and lacks a grant."),
    ("invoice", f"{INVOICE}12:v2:HOLD", "HOLD", "e2b", "task_gold",
     "HOLD is defined by whether PAY, SCHEDULE or SHORT PAY is permitted, rules that appear only in sibling cases; the paid duplicate makes the gold plausible but not stated."),
    ("invoice", f"{INVOICE}27:v0:ROUTE FOR APPROVAL", "ROUTE FOR APPROVAL", "e2b", "model",
     "15 of 16 items were delivered and the current grant forbids partial payment: a partially payable invoice without a sufficient grant."),
    ("invoice", f"{INVOICE}37:v0:PAY", "PAY", "e2b", "model",
     "Every PAY field matches and the current grant's maximum equals quantity times unit price (one multiplication)."),
    ("invoice", f"{INVOICE}45:v0:DISPUTE LINES", "DISPUTE LINES", "e2b", "model",
     "Delivered 8 of 9 invoiced items and the grant has allow_partial false; both conditions are literal fields."),
    ("security", f"{SECURITY}3:v0:BLOCK EGRESS ASSET", "BLOCK EGRESS ASSET", "e4b", "model",
     "Reach is organization-wide, not below it; the covering responder grant was accepted as if it were the trigger."),
    ("security", f"{SECURITY}3:v0:REVOKE SESSIONS", "REVOKE SESSIONS", "e4b", "model",
     "The only event is an outbound connection; no compromised session is described."),
    ("security", f"{SECURITY}3:v2:PURGE MAILBOXES", "PURGE MAILBOXES", "e4b", "model",
     "No mail event exists and the grant ended on day 420, before the day-421 alert."),
    ("security", f"{SECURITY}6:v0:ESCALATE URGENT", "ESCALATE URGENT", "e4b", "task_gold",
     "Escalation applies without a matching playbook branch; the leaked-key branch is defined only in a sibling case."),
    ("security", f"{SECURITY}21:v2:QUARANTINE MESSAGE", "QUARANTINE MESSAGE", "e4b", "model",
     "A connection event with no delivered mail, and the grant ended before the alert day."),
    ("security", f"{SECURITY}33:v0:BLOCK EGRESS ASSET", "BLOCK EGRESS ASSET", "e4b", "model",
     "The event is a leaked cloud access key, not outbound traffic; the action was selected because the grant lists it."),
    ("customer", f"{CUSTOMER.format('noul')}259:refund_requested", "refund_requested", "e4b", "model",
     "The latest turn says 'Please refund the duplicate charge to my card'; the exclusion for questions about charges was applied to the opening turn."),
    ("customer", f"{CUSTOMER.format('noul')}574:refund_requested", "refund_requested", "e4b", "model",
     "Same explicit refund request in the latest turn with a neutral tone."),
    ("customer", f"{CUSTOMER.format('choice')}519:category", "category", "e4b", "model",
     "Gold splits feature_request/account at stated equal priority; the later cancellation was chosen although secondary context does not override an explicit priority."),
    ("customer", f"{CUSTOMER.format('score')}46:bug_severity", "bug_severity", "e4b", "model",
     "No malfunction is described (a duplicate charge), which the criterion maps to no functional impact."),
    ("customer", f"{CUSTOMER.format('score')}192:churn_likelihood_level", "churn_likelihood_level", "e4b", "task_gold",
     "Intent is explicitly withheld and the target is uniform over three levels: every accepted answer has target 1/3 and counts as an error by construction."),
    ("customer", "open-jev--workflow-controls-v1--customer-service-noul-test-v1/open_jev__workflow-controls-v1__customer_service:workflow-controls-v1:customer_service:42:71:v0:REFUND",
     "REFUND", "e2b", "model", "Verified identity, approved consent matching proposal, charge, amount and latest message, a settled unrefunded charge: every condition holds."),
    ("temporal_agency", f"{TRACE}11:v0:HUMAN REVIEW", "HUMAN REVIEW", "e2b", "model",
     "Feedback is unknown, an explicit trigger; the grant (day 372) precedes the call (day 373), so paging does not take precedence."),
    ("temporal_agency", f"{TRACE}26:v2:PAGE ON-CALL", "PAGE ON-CALL", "e4b", "model",
     "The only grant names another resource; the mutation lacked a matching grant. Both arms answered no."),
    ("temporal_agency", f"{TRACE}74:v1:PAGE ON-CALL", "PAGE ON-CALL", "e2b", "model",
     "No grant exists for a mutation call that started; its failure does not authorize it. Both arms answered no."),
    ("temporal_agency", f"{TRACE}74:v2:FILE ISSUE · ROUTE", "FILE ISSUE · ROUTE", "e4b", "model",
     "The rule excludes cases where paging applies; the grant names another resource, so paging applies."),
    ("temporal_agency", "open-jev--mailroom-control-v1-noul-test-v1/open_jev__mailroom-control-v1:mailroom:e5c4d6484923:en:receipt:base:about_airline",
     "about_airline", "e2b", "model", "The relay states it only delivers documents; the body concerns the airline's own baggage service."),
    ("temporal_agency", "winobias-choice-test-v1/winobias:anti_stereotyped_type1.txt.test:test:1", "referent", "e2b", "model",
     "'she got less allowance' explains the reprimand, so she is the janitor; the answer follows the occupational stereotype."),
)
PATTERNS = (
    ("policy_reading", "An approval or responder grant that lists the action is read as the action's trigger; the stated trigger "
     "(reach, event kind, delivered mail) is not checked.", (6, 7, 8, 10, 11)),
    ("policy_reading", "E2B answers no to multi-condition action rules when every condition is a visible matching field.", (1, 3, 4, 5, 17, 18)),
    ("actor", "A grant for another resource, or no grant, is not recognized as a missing authorization for the mutation actually performed.", (19, 20)),
    ("actor", "The subject or referent is attributed to the sending relay or an occupational stereotype instead of the stated role.", (22, 23)),
    ("policy_reading", "An explicit default mapping in the criterion (no malfunction means no functional impact) is replaced by the "
     "options' everyday meaning.", (15,)),
    ("exception", "An exclusion is applied to the wrong turn, or an 'unless' exception that applies is ignored.", (12, 13, 21)),
    ("temporal", "An earlier turn or a later secondary request overrides the latest explicit request or the stated priority.", (12, 13, 14)),
    ("temporal", "A grant whose validity ended before the alert day is still treated as current (confounded with a failed trigger).", (8, 10)),
)


class UnsupportedMapping(Exception):
    pass


class NeedsInput(Exception):
    pass


class InferenceFailure(Exception):
    pass


FAILURES = {"unsupported": UnsupportedMapping, "needs_input": NeedsInput, "failed": InferenceFailure}


def evaluator(*args):
    subprocess.run([str(PYTHON), "-I", str(Path(__file__).resolve()), *map(str, args)], check=True, cwd=CACHE,
                   env={**os.environ, "HF_HUB_CACHE": str(CACHE / "hub")}, stdout=sys.stderr)


def install():
    from semantic_switch.files import sha256_file
    archive = CACHE / f"s1mb-{COMMIT}.tar.gz"
    CACHE.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(f"https://codeload.github.com/hotchpotch/S1MB/tar.gz/{COMMIT}", archive)
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise RuntimeError(f"{archive}: digest differs from the pinned S1MB evaluator {ARCHIVE_SHA256}")
    if not SOURCE.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(CACHE / "src", filter="data")
    if not PYTHON.exists():
        venv.create(CACHE / "venv", with_pip=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", str(SOURCE / "evaluator")], check=True)
    (CACHE / "source.json").write_text(json.dumps({"repo_id": DATASET, "revision": REVISION, "repo_type": "dataset"}))
    evaluator("materialize")
    checksums = sha256_file(DATA / "datasets" / "checksums.json")
    if checksums != CHECKSUMS_SHA256:
        raise RuntimeError(f"S1MB checksums.json digest {checksums} differs from the pinned {CHECKSUMS_SHA256}")
    receipt = json.loads((DATA / "datasets" / "hub-source.json").read_text())
    if receipt["revision"] != REVISION:
        raise RuntimeError(f"installed S1MB dataset {receipt['revision']} is not the pinned {REVISION}")
    packages = subprocess.run([str(PYTHON), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout.split()
    return {"evaluator": {"repository": "https://github.com/hotchpotch/S1MB", "commit": COMMIT, "archive_sha256": ARCHIVE_SHA256,
                          "packages": packages},
            "dataset": {"repo_id": DATASET, "revision": REVISION, "checksums_sha256": checksums, "category": CATEGORY,
                        "subsets": len(receipt["subsets"])}}


def unsupported(question):
    options = question["options"]
    if question["system_prompt"]:
        return "system prompt has no Check/Pick/Rate field"
    if len(options) > MAX_OPTIONS:
        return f"{question['task']}: {len(options)} options exceed the {MAX_OPTIONS}-option presentation"
    if question["task"] == "score":
        values = [option["value"] for option in options]
        if any(low >= high for low, high in zip(values, values[1:])):
            return "score: stored levels are not strictly ascending; Rate presents levels from lowest to highest"
        if len({option["description"] for option in options}) < len(options):
            return "score: duplicate level descriptions"
    return None


def primitive(question):
    from semantic_switch import Check, Guidance, Pick, Rate
    options = question["options"]
    if question["task"] == "noul":
        return Check(question["instructions"], guidance=Guidance(sections=[(o["id"], o["description"], None) for o in options]))
    if question["task"] == "choice":
        return Pick(question["instructions"], {o["id"]: o["description"] for o in options})
    return Rate(question["instructions"], [o["description"] for o in options])


def distribution(question, result):
    keys = [o["description"] if question["task"] == "score" else o["id"] for o in question["options"]]
    found = {o["id"]: result.scores[key] for o, key in zip(question["options"], keys)
             if key in result.scores and math.isfinite(result.scores[key])}
    if not result.scores:
        return found, None
    if set(result.scores) != set(keys) or len(found) < len(keys) or any(not 0 <= p <= 1 for p in found.values()):
        return found, f"unsupported: public scores {sorted(result.scores)} are not finite values for exactly the authored options"
    total = math.fsum(found.values())
    if abs(total - 1) > TOLERANCE:
        return found, f"unsupported: public scores sum to {total:.6f}; a lower bound has no original-evaluator distribution"
    return found, None


def accepted(question, result):
    if result.status != "ok":
        return None
    if question["task"] == "noul":
        return "true" if result.value else "false"
    return result.value if question["task"] == "choice" else question["options"][result.index]["id"]


def predict(rt, case):
    questions = {q["id"]: q for q in case["questions"]}
    reasons = {key: reason for key, q in questions.items() if (reason := unsupported(q))}
    items = {key: primitive(q) for key, q in questions.items() if key not in reasons}
    results = rt.ask(state(case), items) if items else {}
    rows = []
    for key, question in questions.items():
        if key in reasons:
            rows.append({"question_id": key, "task": question["task"], "status": "unsupported", "scores": {}, "probabilities": None,
                         "error": f"unsupported: {reasons[key]}", "accepted": None, "mass": None})
            continue
        result = results[key]
        scores, invalid = distribution(question, result)
        kind = "needs_input" if result.status == "needs_input" else "failed"
        error = invalid or (None if result.scores else f"{kind}: {result.error}")
        rows.append({"question_id": key, "task": question["task"], "status": result.status, "scores": scores,
                     "probabilities": None if error else scores, "error": error, "accepted": accepted(question, result), "mass": result.mass})
    return rows


def state(case):
    return case["state"] if isinstance(case["state"], (dict, str)) else case["state_json"]


def failure(rows):
    return next((row["error"] for row in rows if row["error"]), None)


def model_info(rt, judge):
    identity = rt.identity()
    return {"id": judge, "adapter": "semantic-switch-check-pick-rate", "revision": identity.get("sha256") or judge,
            "settings": {"identity": identity, "execution": rt.execution(), "noul": "Check(instructions, guidance=[false], [true] definitions)",
                         "choice": "Pick(instructions, {id: description})", "score": "Rate(instructions, descriptions in stored ascending order)"}}


def wrong(target, answer):
    return answer is not None and target[answer] < .5


def agreement(items):
    accepted = [(target, answer) for target, answer in items if answer is not None]
    return {"items": len(items), "accepted": len(accepted),
            "expected_agreement": math.fsum(target[answer] for target, answer in accepted) / len(accepted) if accepted else None}


def outcome(target, answer):
    return "abstain" if answer is None else "correct" if target[answer] == 1 else "wrong"


def top(probabilities):
    return max(probabilities, key=probabilities.get)


def arm(rows, scored):
    tasks = {}
    for task in ("noul", "choice", "score"):
        questions = [(case["targets"][q["question_id"]], q) for case in rows if case["task"] == task for q in case["predictions"]]
        benchmarks = [result for result in scored["benchmarks"].values() if result["task"] == task]
        primary = [result["metrics"][result["primary_metric"]] for result in benchmarks]
        tasks[task] = {"benchmarks": len(benchmarks), "benchmarks_with_failures": sum(bool(r["counts"]["failed"]) for r in benchmarks),
                       "evaluator_task_score": scored["task_scores"][task],
                       "primary_metric_benchmark_mean": statistics.mean(primary) if primary and None not in primary else None,
                       "questions": len(questions), "statuses": dict(Counter(q["status"] for _, q in questions)),
                       "accepted": agreement([(target, q["accepted"]) for target, q in questions])}
    return tasks


def summarize(rows, scored, provenance, info, limit):
    return {"schema": 2, "boundary": f"first {limit} cases of every {CATEGORY} benchmark (the evaluator's --limit); partial, not leaderboard-comparable"
            if limit else f"complete {CATEGORY}", "provenance": provenance, "model": info, "tasks": arm(rows, scored),
            "evaluator": scored, "case_ms_p50": statistics.median(case["ms"] for case in rows)}


def questions(key, case):
    rows = {q["question_id"]: q for q in case["predictions"]}
    if len(rows) != len(case["predictions"]):
        raise ValueError(f"{key}: duplicate question IDs")
    return rows


def aligned(base, *others):
    from bench.outcomes import work_key
    keys = [key for key in base if all(key in other for other in others)]
    items = []
    for key in keys:
        a, *rest = rows = [base[key], *(other[key] for other in others)]
        if any(b[field] != a[field] for b in rest for field in ("benchmark", "case_id", "task", "targets")) or key != work_key(a['benchmark'], a['case_id']):
            raise ValueError(f"{key}: arms differ in case identity, task or targets")
        found = [questions(key, row) for row in rows]
        if any(set(x) != set(a["targets"]) for x in found):
            raise ValueError(f"{key}: arms differ in question IDs or targets")
        if any(q["probabilities"] is not None and set(q["probabilities"]) != set(a["targets"][qid]) for x in found for qid, q in x.items()):
            raise ValueError(f"{key}: arms differ in authored option IDs")
        items += [(a["benchmark"], a["task"], a["targets"][qid], *(x[qid] for x in found)) for qid in a["targets"]]
    return keys, items


def member(item, group):
    return all(part in ("all", item[1], "in_scope" if scope(item[0]) == "in" else "out_of_scope") for part in group.split("/"))


def shared_inputs(headers):
    inputs = [{field: header["configuration"]["inputs"][field] for field in ("provenance", "cases")} for header in headers]
    if any(found != inputs[0] for found in inputs):
        raise ValueError("arms were recorded with different evaluator provenance or case definitions")
    return inputs[0]


def counts(hard, column, answer):
    outcomes = Counter(outcome(i[2], answer(i[column])) for i in hard)
    accepted = outcomes["correct"] + outcomes["wrong"]
    return {"accepted": accepted, "correct": outcomes["correct"], "wrong": outcomes["wrong"], "abstain": outcomes["abstain"],
            "selective_precision": outcomes["correct"] / accepted if accepted else None}


def transitions(hard, answer):
    return dict(sorted(Counter(f"{outcome(t, answer(a))}->{outcome(t, answer(b))}" for _, _, t, a, b in hard).items()))


def paired(base, other):
    keys, items = aligned(base, other)
    valid = [item for item in items if item[3]["probabilities"] is not None and item[4]["probabilities"] is not None]
    arms = (("base", 3), ("other", 4))
    accepted, raw = (lambda q: q["accepted"]), (lambda q: top(q["probabilities"]))
    table = {}
    for group in ("all", "noul", "choice", "score", "in_scope", "out_of_scope"):
        rows = [i for i in valid if member(i, group)]
        hard = [i for i in rows if max(i[2].values()) == 1]
        moved = transitions(hard, accepted)
        table[group] = {"hard_label": {"items": len(hard), "fixes": moved.get("wrong->correct", 0), "regressions": moved.get("correct->wrong", 0),
                                       "accepted_transitions": moved, "raw_transitions": transitions(hard, raw),
                                       **{name: {**counts(hard, column, accepted), "raw_correct": counts(hard, column, raw)["correct"]} for name, column in arms}},
                        "soft_target": {name: agreement([(i[2], i[column]["accepted"]) for i in rows if max(i[2].values()) < 1]) for name, column in arms}}
    missing = Counter((i[3]["probabilities"] is None, i[4]["probabilities"] is None) for i in items)
    return {"matched_cases": len(keys), "judgments": len(items), "jointly_scored": len(valid),
            "unscored": {"base_only": missing[True, False], "other_only": missing[False, True], "both": missing[True, True]},
            "availability": {name: dict(sorted(Counter(i[column]["status"] for i in items).items())) for name, column in arms},
            "pooled_soft_and_hard": {name: agreement([(i[2], i[column]["accepted"]) for i in valid]) for name, column in arms}, "table": table}


def recorded(path):
    from bench.outcomes import recorded as journal
    return journal(path)


def matched(header, rows, keys, stem):
    model = header["configuration"]["models"][0]
    predictions, info, scored = (CACHE / f"{stem}-matched.{suffix}" for suffix in ("predictions.jsonl", "model.json", "scored.json"))
    predictions.write_text("".join(json.dumps({"benchmark": rows[key]["benchmark"], "case_id": rows[key]["case_id"],
                                               "predictions": rows[key]["predictions"]}) + "\n" for key in keys))
    info.write_text(json.dumps({"id": model["source"], "adapter": "semantic-switch-check-pick-rate", "revision": model["sha256"],
                                "settings": {"journal_configuration_digest": header["digest"], "args": header["configuration"]["args"]}}))
    evaluator("score", predictions, info, f"{stem}-matched", scored)
    return json.loads(scored.read_text())


def compare(base, other, output):
    from semantic_switch import files
    arms = {name: (path, *recorded(path)) for name, path in (("base", base), ("other", other))}
    inputs = shared_inputs([arms[name][1] for name in arms])
    pairs = paired(arms["base"][2], arms["other"][2])
    keys = [key for key in arms["base"][2] if key in arms["other"][2]]
    summaries = {}
    for name, (path, header, rows) in arms.items():
        scored = matched(header, rows, keys, Path(path).stem)
        summaries[name] = {"artifact": str(path), "configuration_digest": header["digest"], "model": header["configuration"]["models"][0]["source"],
                           "tasks": arm([rows[key] for key in keys], scored), "evaluator": scored}
    report = {"schema": 3, "boundary": f"cases recorded by both arms within the first 25 cases of every {CATEGORY} benchmark; several questions "
              "per case and first-N selection make pooled counts neither leaderboard scores nor independent samples",
              "inputs": inputs, "arms": summaries, "paired": pairs}
    files.publish(output, report)
    return report


def scope(benchmark):
    return "out" if any(name in benchmark for name in OUT_OF_SCOPE) else "in"


def recording():
    from semantic_switch import Runtime, judge_from
    from semantic_switch.hosted import TransportError
    bodies = []

    async def record(url, body, timeout):
        bodies.append(body)
        raise TransportError("recording only")
    return Runtime({"judge": judge_from("openrouter:recording", transport=record)}, own_judges=True), bodies


def reconstruct(rt, bodies, case, question):
    from semantic_switch.inline import rendered
    rt.ask(state(case), {question["id"]: primitive(question)})
    system, user = (message["content"] for message in bodies[-1]["messages"])
    position = 1
    for option in question["options"]:
        position = position and user.find(option["description"], position - 1) + 1
    return {"system": system, "user": user, "transport": {"instructions": question["instructions"] in user, "options_in_authored_order": position > 0,
                                                          "input": rendered(state(case)) in user}}


def audit(base, other, output):
    from semantic_switch import files
    arms = {"e2b": recorded(base)[1], "e4b": recorded(other)[1]}
    cases = {f"{row['benchmark']}/{row['case']['case_id']}": row["case"] for row in files.read_jsonl(CACHE / "cases-25.jsonl")}
    counts, keys = {name: {} for name in arms}, aligned(*arms.values())[0]
    for name, rows in arms.items():
        for row in map(rows.get, keys):
            for q in row["predictions"]:
                if q["accepted"] is not None:
                    entry = counts[name].setdefault(row["benchmark"], {"scope": scope(row["benchmark"]), "accepted": 0, "errors": 0})
                    entry["accepted"] += 1
                    entry["errors"] += wrong(row["targets"][q["question_id"]], q["accepted"])
    reviewed = []
    rt, bodies = recording()
    with rt:
        for group, key, qid, name, finding, note in REVIEW:
            row = arms[name][key]
            question = next(q for q in cases[key]["questions"] if q["id"] == qid)
            saved = next(q for q in row["predictions"] if q["question_id"] == qid)
            target = row["targets"][qid]
            correct = [(k, q, r["targets"][qid]) for k, r in arms[name].items() if r["benchmark"] == row["benchmark"] and k != key
                       for q in r["predictions"] if q["question_id"] == qid and q["accepted"] is not None and not wrong(r["targets"][qid], q["accepted"])]
            contrast = min(correct, key=lambda c: c[1]["accepted"] != max(target, key=target.get), default=None)
            reviewed.append({"group": group, "source": key, "question_id": qid, "arm": name, "finding": finding, "note": note, "target": target,
                             "accepted": saved["accepted"], "scores": saved["probabilities"], "question": question,
                             **reconstruct(rt, bodies, cases[key], question),
                             "contrast": contrast and {"source": contrast[0], "accepted": contrast[1]["accepted"], "target": contrast[2],
                                                       "transport": reconstruct(rt, bodies, cases[contrast[0]], next(
                                                           q for q in cases[contrast[0]]["questions"] if q["id"] == qid))["transport"]}})
    report = {"schema": 2, "boundary": "cases recorded by both saved E2B/E4B first-25-case S1MB checkpoints; an accepted error is an accepted answer "
              "with target mass below .5", "matched_cases": len(keys),
              "out_of_scope": list(OUT_OF_SCOPE),
              "accepted_errors": {name: {"in_scope": sum(e["errors"] for e in benchmarks.values() if e["scope"] == "in"),
                                         "out_of_scope": sum(e["errors"] for e in benchmarks.values() if e["scope"] == "out"),
                                         "benchmarks": dict(sorted(benchmarks.items()))} for name, benchmarks in counts.items()},
              "unsupported_catalogues": sorted({row["benchmark"] for row in arms["e2b"].values() for q in row["predictions"]
                                                if q["status"] == "unsupported" and "options exceed" in q["error"]}),
              "findings": dict(Counter(case["finding"] for case in reviewed)), "cases": reviewed,
              "cohort_patterns": [{"kind": kind, "pattern": pattern, "sources": [f"{REVIEW[i][1]}#{REVIEW[i][2]}@{REVIEW[i][3]}" for i in sources]}
                                  for kind, pattern, sources in PATTERNS]}
    files.publish(output, report)
    return report


def run(args):
    from semantic_switch import DEFAULT_JUDGE, PRECISE_JUDGE, Model, files
    from semantic_switch.artifacts import artifact_digest
    from bench.outcomes import configuration, finish, judge_spec, resumable, runtime, work_key
    provenance = install()
    judge = PRECISE_JUDGE if args.precise else args.judge or DEFAULT_JUDGE
    cases_path = CACHE / f"cases-{args.limit or 'all'}.jsonl"
    evaluator("cases", args.limit or "all", cases_path)
    cases = files.read_jsonl(cases_path)
    model, recording = judge_spec(judge, provider=args.provider, key_file=args.key_file, device=args.device, max_input_tokens=args.max_input_tokens)
    keyed = {work_key(row['benchmark'], row['case']['case_id']): row for row in cases}
    if args.matched:
        common = set(recorded(args.matched[0])[1]) & set(recorded(args.matched[1])[1])
        keyed = {key: row for key, row in keyed.items() if key in common}
    config = configuration("bench.external_primitives", args, *[model] if isinstance(model, Model) else [],
                           provenance=provenance, cases=files.sha256_file(cases_path), keys=artifact_digest(sorted(keyed)))
    with resumable(args.output, config, fresh=args.fresh) as journal:
        with runtime(model) as rt:
            info = model_info(rt, judge)
            for key in journal.pending(keyed):
                row = keyed[key]
                first = len(recording.records) if recording else 0
                started = time.perf_counter()
                predictions = predict(rt, row["case"])
                receipts = recording.records[first:] if recording else []
                journal.commit([(key, {"benchmark": row["benchmark"], "task": row["task"], "case_id": row["case"]["case_id"],
                                        "ms": (time.perf_counter() - started) * 1000, "predictions": predictions,
                                        "input_characters": len(json.dumps(state(row["case"]), ensure_ascii=False)), "targets": row["targets"],
                                        **({"requests": [{name: receipt[name] for name in ("elapsed_ms", "usage", "returned_model", "response_id", "completion", "error")}
                                                         for receipt in receipts]} if recording else {})})])
        predictions_path, model_path, scored_path = (CACHE / f"{Path(args.output).stem}.{suffix}" for suffix in ("predictions.jsonl", "model.json", "scored.json"))
        predictions_path.write_text("".join(json.dumps({"benchmark": row["benchmark"], "case_id": row["case_id"],
                                                         "predictions": row["predictions"]}) + "\n" for row in journal.payloads()))
        model_path.write_text(json.dumps(info))
        evaluator("score", predictions_path, model_path, Path(args.output).stem, scored_path)
        scored = json.loads(scored_path.read_text())
        artifact = finish(journal, args.output, lambda rows: summarize(rows, scored, provenance, info, args.limit))
    for task, summary in artifact["tasks"].items():
        print(task, json.dumps(summary), flush=True)


def materialize():
    from s1mb.dataset_source import ensure_dataset
    ensure_dataset(DATA, CACHE / "source.json")


def export(limit, output):
    from s1mb.data import load_benchmark, load_cases, load_category
    with open(output, "w") as out:
        for name in load_category(DATA, CATEGORY).benchmarks:
            benchmark = load_benchmark(DATA, name)
            for case in load_cases(DATA, benchmark)[:None if limit == "all" else int(limit)]:
                out.write(json.dumps({"benchmark": name, "task": benchmark.task, "case": case.inference().model_dump(),
                                      "targets": {key: target.probabilities for key, target in case.targets.items()}}) + "\n")


def score(predictions, model, run_id, output):
    from s1mb.data import ModelInfo, Prediction, load_benchmark, load_category
    from s1mb.runner import evaluate
    from s1mb.submission_report import TASKS, Measurement, task_score
    stored = {}
    for line in Path(predictions).read_text().splitlines():
        row = json.loads(line)
        stored.setdefault(row["benchmark"], {})[row["case_id"]] = row["predictions"]
    info = ModelInfo.model_validate_json(Path(model).read_text())

    class Replay:
        def __init__(self, name):
            self.name = name

        def predict(self, case):
            rows = stored[self.name][case.case_id]
            if error := failure(rows):
                raise FAILURES[error.split(":")[0]](error)
            return [Prediction(case_id=case.case_id, question_id=row["question_id"], probabilities=row["probabilities"]) for row in rows]

        def metadata(self):
            return info

        def close(self):
            pass

    shutil.rmtree(CACHE / "runs" / run_id, ignore_errors=True)
    benchmarks = [load_benchmark(DATA, name) for name in load_category(DATA, CATEGORY).benchmarks]
    results = {b.id: evaluate(Replay(b.id), DATA, b, CACHE / "runs", run_id, len(stored[b.id])) for b in benchmarks if b.id in stored}
    Path(output).write_text(json.dumps({
        "benchmarks": {b.id: {"task": b.task, "primary_metric": b.primary_metric, **results[b.id].model_dump(include={"status", "counts", "metrics"}),
                              "failures": dict(Counter(p.error for p in results[b.id].predictions if p.error))} for b in benchmarks if b.id in results},
        "task_scores": {task: task_score([b for b in benchmarks if b.task == task], {name: Measurement.from_result(result) for name, result in results.items()})
                        for task in TASKS}}))


def main():
    commands = {"materialize": materialize, "cases": export, "score": score}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        return commands[sys.argv[1]](*sys.argv[2:])
    parser = argparse.ArgumentParser(description="Pinned S1MB external baseline through public Check/Pick/Rate and the original S1MB evaluator.")
    parser.add_argument("--judge", help="judge source for semantic_switch.judge_from (default: DEFAULT_JUDGE)")
    parser.add_argument("--precise", action="store_true", help="use PRECISE_JUDGE")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="first N cases of every benchmark, as the evaluator's --limit")
    parser.add_argument("--matched", nargs=2, metavar=("BASE", "OTHER"), help="run only the cases recorded in both saved arms' checkpoints")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "OTHER"), help="model-free matched analysis of two saved arms' checkpoints")
    parser.add_argument("--audit", nargs=2, metavar=("E2B", "E4B"), help="model-free accepted-error audit of two saved arms' checkpoints")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare, args.output)
    if args.audit:
        return audit(*args.audit, args.output)
    run(args)


if __name__ == "__main__":
    main()
