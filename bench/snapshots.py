from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
import statistics
from pathlib import Path

import yaml

from semantic_switch import Check, ConfigError, Guidance, Label, Model, Rate, Runtime, Select, files
from semantic_switch.artifacts import load_receipt
from semantic_switch.judgment import Judgment
from semantic_switch.metrics import (FAILED, METRICS, action_sets, answer_key, comparison_identity, confusion, exact_sets, membership, observation,
                                     observed, ordered, paired, probabilities, ratio, reconcile, semantic_answer, success, PROBABILITY)
from bench.outcomes import Progress
from semantic_switch.compiler import digest
from semantic_switch.presentation import identity as presentation_identity
from semantic_switch.training import EvidenceContext, assess_lineage, entry, evaluation_plan, fitted, model_job, unproven
from bench import evidence, outcomes, procedures, support
from bench.outcomes import Receipts, failure, finish, judge_spec, materialize, measure, resumable, resumed, runtime, work_directory, work_key

SETTLED = {"applicable": True, "rejected": False, "ineligible": False, "superseded": False}
BOOTSTRAP = {"seed": 0, "count": 2000}
SCHEMA = 4
DECISION = 3
CHECK = 1
PROTOCOL = {"boundary": "public call: bind, render, request, compose and JSON serialization; failed attempts included",
            "order": "arms alternate per frame (even frames in declared order, odd frames reversed)",
            "cache": "each arm owns one runtime and its warm caches; one warmup call per arm and condition, measured as setup outside the rows",
            "scope": "workflow: one public call demands the decision plus every scored target on the same Call; direct: one typed ask per frame",
            "concurrency": "serial calls, one frame at a time"}


@dataclass(frozen=True)
class Domain:
    name: str
    module: object
    sets: dict
    scalars: tuple
    conditions: tuple

    @property
    def pack(self):
        return evidence.load_pack(self.module.PACK)

    def scoped(self, names) -> Domain:
        names = set(names)
        return replace(self, scalars=tuple(name for name in self.scalars if name in names))

    def case(self, frame, gold):
        if self.name == "support":
            return support.response(frame, gold).value
        return "act" if procedures.justified(gold) else "wait"


DOMAINS = {"zt411": Domain("zt411", procedures, {"actions": procedures.MENU}, procedures.PRIMITIVES, procedures.CONDITIONS),
           "support": Domain("support", support, {"issues": support.ISSUES, "requests": support.REQUESTS}, ("hazard", "impact"), support.CONDITIONS)}


def definition_path(domain: Domain, condition: str, directory: Path) -> Path:
    path = Path(directory) / f"{domain.name}-{condition}.yaml"
    files.publish(path, yaml.safe_dump(domain.module.definition(condition), sort_keys=False, allow_unicode=True))
    return path


def direct_items(domain: Domain, spec: dict) -> dict:
    items = {}
    for name in domain.sets:
        fact = spec["facts"][name]
        labels = {card["id"]: Label(card["title"]) for card in fact["cards"]} if "cards" in fact else {k: Label(v) for k, v in fact["labels"].items()}
        items[name] = Select(fact["select"], labels, unstated=bool(fact.get("unstated")))
    for name in domain.scalars:
        fact = spec["facts"][name]
        items[name] = Check(fact["check"]) if "check" in fact else Rate(fact["rate"], tuple(fact["levels"]))
    return items


def judgments(domain: Domain, plan=None, items=None) -> dict:
    out = {}
    for name, keys in domain.sets.items():
        if plan is None:
            members = dict(zip(items[name].labels, items[name].compiled[0]))
        else:
            fact = plan.fact(name)
            members = {card.id: plan.facts[card.claims[0]].judgment for card in fact.cards} if fact.labels else {}
        out.update({f"{name}/{key}": members.get(key) for key in keys})
    for name in domain.scalars:
        if plan is not None:
            out[name] = plan.fact(name).judgment
        elif isinstance(items[name], Check):
            out[name] = Judgment.check(items[name].statement, items[name].excludes, items[name].unstated)
        else:
            out[name] = Judgment("rate", items[name].question, items[name].levels, items[name].levels, items[name].excludes)
    return out


def probability(capture, target, result):
    judgment = capture["judgments"][target]
    if judgment is None:
        return observed((), None, kind="composition", stage=None, normalization=None, status="composed", source="rule", execution=capture["execution"])
    return observation(result, judgment, normalization=capture["normalization"], execution=capture["execution"])


def raw_member(fact, key):
    member = (fact.get("members") or {}).get(key) or {}
    prediction = member.get("prediction")
    return prediction["value"] if prediction and prediction.get("kind") == "value" else None


def targets(domain: Domain, facts: dict, capture=None) -> dict:
    out = {}
    for name, keys in domain.sets.items():
        fact = facts.get(name) or {}
        states = fact.get("states") or {}
        for key in keys:
            out[f"{name}/{key}"] = {"raw": raw_member(fact, key), "value": SETTLED.get(states.get(key)), "state": states.get(key, fact.get("status", "missing")),
                                    **({"probability": probability(capture, f"{name}/{key}", (fact.get("members") or {}).get(key))} if capture else {})}
    for name in domain.scalars:
        fact = facts.get(name) or {}
        raw = scalar(fact.get("prediction"))
        out[name] = {"raw": raw, "value": fact.get("value") if fact.get("status") == "ok" else "unstated" if raw == "unstated" and fact.get("status") == "needs_input" else None,
                     "state": fact.get("status", "missing"), **({"probability": probability(capture, name, facts.get(name))} if capture else {})}
    return out


def scalar(prediction):
    answer = semantic_answer(prediction)
    return prediction["value"] if answer is not None and prediction["kind"] == "value" else answer if answer == "unstated" else None


def requested(domain: Domain) -> list:
    return [*domain.sets, *domain.scalars]


def predict(rt, domain: Domain, kind: str, frame, switch=None, items=None, document=None, capture=None) -> dict:
    if kind == "workflow":
        measured = measure(lambda: switch(frame.input, facts=requested(domain)), runtime=rt)
    else:
        measured = measure(lambda: rt.ask(frame.input, items, guidance=Guidance(document)), runtime=rt)
    return prediction(domain, kind, measured, capture)


def prediction(domain: Domain, kind: str, measured, capture=None) -> dict:
    unavailable = dict(measured.unavailable)
    if measured.error is not None:
        reason = f"operation failed at {measured.error['stage']} before a public result"
        return failure(measured, targets=targets(domain, {}, capture), case=None, requests=None, logical_tokens=None,
                       unavailable={**unavailable, "requests": reason, "logical_tokens": reason})
    if kind == "workflow":
        decision = measured.output
        facts, cost = decision["facts"], decision["cost"]
        case = decision["answer"] if decision["status"] == "ok" else decision["status"]
        requests, tokens = len(cost["requests"]), cost["logical_tokens"]
        failed = decision["status"] == "failed"
    else:
        facts, case = measured.output, None
        requests, tokens = sum(len(result.get("requests", ())) for result in facts.values()), None
        unavailable["logical_tokens"] = "typed asks report no logical token count"
        failed = False
    failed = failed or any(isinstance(fact, dict) and fact.get("status") == "failed" for fact in facts.values())
    return {"targets": targets(domain, facts, capture), "case": case, "ms": measured.ms, "native_work": measured.work, "requests": requests,
            "logical_tokens": tokens, "unavailable": unavailable, "outcome": "failed" if failed else "predicted"}


def gold_of(gold) -> dict:
    return {"values": {**gold.values(), **{name: d.value for name, d in gold.primitives.items()}}, "prohibited": list(gold.prohibited)}


def scored(domain: Domain, frame, gold, prediction, component, comparison) -> dict:
    values = gold.values()
    complete = gold.complete
    accepted = {t: p["value"] for t, p in prediction["targets"].items()}
    correct = None if prediction["outcome"] == "failed" else all(accepted.get(t) == values[t] for t in values) if complete else None
    return {"key": frame.id, "family": component, "comparison": comparison, "gold": gold_of(gold),
            "outcome": prediction["outcome"], "correct": correct, "accepted": prediction["outcome"] != "failed" and all(v is not None for v in accepted.values()),
            "complete_gold": complete, "tags": list(frame.tags), **prediction,
            "expected_case": domain.case(frame, gold)}


def condition_of(domain: Domain, kind: str, condition: str) -> dict:
    return {"kind": kind, "condition": condition, "requested_facts": requested(domain) if kind == "workflow" else None}


def expected(domain: Domain, frame, gold, component, kind, task) -> dict:
    return {"key": frame.id, "family": component, "gold": gold_of(gold),
            "comparison": comparison_identity(task=task, projection=frame.public, gold=gold_of(gold),
                                              answers=requested(domain), scope=list(gold.scope), condition=condition_of(domain, kind, frame.condition))}


def set_row(keys, gold, prohibited, got, outcome) -> dict:
    unknown = [k for k in keys if gold[k] is None]
    unresolved = [k for k in keys if got[k] is None]
    return {"scope": list(keys), "gold": [k for k in keys if gold[k] is True], "reference_unresolved": unknown, "complete_gold": not unknown,
            "selected": [k for k in keys if got[k] is True], "unresolved": unresolved,
            "prediction_complete": outcome not in FAILED and not unresolved, "outcome": outcome, "prohibited": [k for k in keys if k in prohibited]}


def set_rows(domain, rows, name):
    keys = domain.sets[name]
    return [set_row(keys, {k: row["gold"]["values"][f"{name}/{k}"] for k in keys}, row["gold"]["prohibited"],
                    {k: row["targets"][f"{name}/{k}"]["value"] for k in keys}, row["outcome"]) for row in rows]


def set_metrics(rows) -> dict:
    return {"actions": action_sets(rows), "exact": exact_sets(rows)}


def f1(rows, domain, name):
    scores = {}
    for key in domain.sets[name]:
        target = f"{name}/{key}"
        pairs = [(row["gold"]["values"][target], row["targets"][target]["value"]) for row in rows if row["gold"]["values"][target] is not None]
        tp = sum(g and v is True for g, v in pairs)
        fp = sum(not g and v is True for g, v in pairs)
        fn = sum(g and v is not True for g, v in pairs)
        scores[key] = {"support": sum(g for g, _ in pairs), "f1": ratio(2 * tp, 2 * tp + fp + fn)}
    present = [s["f1"] for s in scores.values() if s["support"] and s["f1"] is not None]
    return {"per_class": scores, "macro_f1": statistics.fmean(present) if present else None, "classes_with_support": len(present)}


def known(rows, field) -> dict:
    values = [row[field] for row in rows if row[field] is not None]
    reasons = sorted({row["unavailable"].get(field, "unreported") for row in rows if row[field] is None})
    if values and isinstance(values[0], dict):
        total = {key: sum(value.get(key, 0) for value in values) for key in sorted({key for value in values for key in value})}
    else:
        total = sum(values)
    return {"sum": total, "rows": len(values), "of": len(rows), "bound": "total" if len(values) == len(rows) else "lower_bound", "unavailable": reasons}


def latency(rows, setup):
    ms = [row["ms"] for row in rows]
    work = known(rows, "native_work")
    total = work["sum"] if isinstance(work, dict) and isinstance(work.get("sum"), dict) else {}
    questions = total.get("suffix_rows")
    return {"frames": len(rows), "unit": "situation: one public workflow call; per-question figures are totals divided by scored question rows, never a measured "
                                         "per-question percentile", **outcomes.latency(ms), "throughput_per_s": ratio(len(ms), sum(ms) / 1000),
            "requests": known(rows, "requests"), "logical_tokens": known(rows, "logical_tokens"), "native_work": work,
            "questions": {"rows": questions, "per_situation": ratio(questions, len(rows)) if questions is not None else None,
                          "ms_per_question": ratio(sum(ms), questions) if questions else None,
                          "suffix_tokens_per_question": ratio(total.get("suffix_tokens"), questions) if questions else None},
            "failed": sum(row["outcome"] == "failed" for row in rows), "setup": setup, **PROTOCOL}


def summary(domain: Domain, rows, scope, setup) -> dict:
    reconciled, support_ = reconcile(scope, [{k: row[k] for k in ("key", "comparison", "gold", "outcome", "correct", "accepted")} for row in rows if row["complete_gold"]])
    out = {"scored": {**success(reconciled), "support": support_, "unit": "gold-complete frames of this condition; missing or failed work counts in the denominator"},
           "unresolved_reference": {"frames": sum(not row["complete_gold"] for row in rows),
                                    "abstained_on_unresolved": sum(all(row["targets"][t]["value"] is None for t, v in row["gold"]["values"].items() if v is None and t in row["targets"])
                                                                   for row in rows if not row["complete_gold"])},
           "workflow_case": {"scored": sum(row["case"] is not None for row in rows),
                             "correct": sum(row["case"] == row["expected_case"] for row in rows if row["case"] is not None)},
           "latency": latency(rows, setup), "sets": {}, "scalars": {}}
    for name in domain.sets:
        flat = [{"gold": row["gold"]["values"][f"{name}/{k}"], "raw": row["targets"][f"{name}/{k}"]["raw"], "value": row["targets"][f"{name}/{k}"]["value"],
                 "accepted": row["targets"][f"{name}/{k}"]["value"] is not None, "outcome": row["outcome"]} for row in rows for k in domain.sets[name]]
        out["sets"][name] = {"membership": membership(flat), "classes": f1(rows, domain, name), **set_metrics(set_rows(domain, rows, name))}
    for name in domain.scalars:
        golds = [row["gold"]["values"][name] for row in rows]
        if name == "impact":
            index = {level: n for n, level in enumerate(support.LEVELS)}
            out["scalars"][name] = ordered({"gold": index.get(row["gold"]["values"][name]), "predicted": index.get(row["targets"][name]["value"])}
                                           for row in rows if row["gold"]["values"][name] is not None)
            continue
        out["scalars"][name] = {"positive": membership({"gold": None if gold is None else gold is True, "raw": None if row["targets"][name]["raw"] is None else row["targets"][name]["raw"] is True,
                                                        "value": row["targets"][name]["value"] is True, "accepted": row["targets"][name]["value"] is not None, "outcome": row["outcome"]}
                                                       for gold, row in zip(golds, rows)),
                                **({"confusion": confusion({"gold": gold, "predicted": row["targets"][name]["value"], "outcome": row["outcome"]}
                                                           for gold, row in zip(golds, rows) if gold is not None)} if "unstated" in golds else {})}
    return out


PROBABILITIES = "stored rows lack probability observations; their correctness metrics remain and no distribution is reconstructed"


def probability_entries(domain: Domain, rows) -> list:
    return [{"task": row["condition"], "target": target, "kind": "membership" if target.split("/")[0] in domain.sets else value["probability"]["kind"],
             "family": row["family"], "gold": answer_key(row["gold"]["values"][target]), "outcome": row["outcome"], "observation": value["probability"]}
            for row in rows for target, value in row["targets"].items()]


def probability_report(domain: Domain, results, roles, review, plan, exposure) -> dict:
    arms = {}
    for key, result in results.items():
        arm, kind, _ = key.split("|")
        arms.setdefault(f"{arm}|{kind}", []).extend(result["rows"])
    if not all("probability" in value for rows in arms.values() for row in rows for value in row["targets"].values()):
        return {"state": "unrecoverable", "reason": PROBABILITIES}
    return {"state": "reported", "roles": list(roles), "review": review, "plan": plan, "exposure": exposure,
            "scope": "diagnostics of stored judge distributions per arm and kind, tasks are conditions; model_select rows are selection diagnostics, "
                     "check rows inherit the exposure of this frozen plan and a later report cannot make them independent; judge-only distributions, "
                     "separate from rule and override application; agreement with the declared reference until it is reviewed",
            "arms": {key: probabilities(probability_entries(domain, rows)) for key, rows in sorted(arms.items())}}


def units_of(domain: Domain, rows, condition) -> dict:
    return {"frames": len(rows), "gold_complete_frames": sum(row["complete_gold"] for row in rows),
            "situations": sum(row["complete_gold"] for row in rows) if condition == domain.module.TRAINED else 0,
            "families": len({row["family"] for row in rows}), "targets": sum(len(row["gold"]["values"]) for row in rows), "definitions": evidence.UNITS}


def slices(rows) -> dict:
    out = {}
    for tag in sorted({tag for row in rows for tag in row["tags"]}):
        group = [row for row in rows if tag in row["tags"]]
        complete = [row for row in group if row["complete_gold"]]
        out[tag] = {"frames": len(group), "families": len({row["family"] for row in group}), "gold_complete": len(complete),
                    "correct": sum(row["correct"] is True for row in complete), "correct_per_submission": ratio(sum(row["correct"] is True for row in complete), len(complete))}
    return out


def counterparts(rows) -> dict:
    groups, out = {}, {}
    for row in rows:
        groups.setdefault(evidence.counterpart(row["key"]), []).append(row)
    for group in groups.values():
        for n, a in enumerate(group):
            for b in group[n + 1:]:
                if a["gold"] != b["gold"]:
                    continue
                (ca, la), (cb, lb) = a["key"].split(":")[2:], b["key"].split(":")[2:]
                dimension = "language" if ca == cb else "condition" if la == lb else "condition_and_language"
                same = {t: p["value"] for t, p in a["targets"].items()} == {t: p["value"] for t, p in b["targets"].items()}
                entry = out.setdefault(dimension, {"pairs": 0, "same_prediction": 0, "both_correct": 0, "one_correct": 0})
                entry["pairs"] += 1
                entry["same_prediction"] += same
                entry["both_correct"] += a["correct"] is True and b["correct"] is True
                entry["one_correct"] += (a["correct"] is True) != (b["correct"] is True)
    return {"scope": "counterpart frames of one scenario variant whose reference outputs agree; correlated, not independent evidence", **out}


def baselines(domain: Domain, frames, golds, ruled) -> dict:
    out = {}
    for name, keys in domain.sets.items():
        def rows(choose):
            return [set_row(keys, {k: golds[frame.id].values()[f"{name}/{k}"] for k in keys}, golds[frame.id].prohibited, choose(frame), "predicted")
                    for frame in frames]
        arms = {"always_empty": rows(lambda frame: dict.fromkeys(keys, False)), "select_all": rows(lambda frame: dict.fromkeys(keys, True)),
                "rule_only": rows(lambda frame: ruled[frame.id][name])}
        out[name] = {arm: {**set_metrics(value), "rows": value} for arm, value in arms.items()}
    return out



def majority_baseline(domain, all_frames, meta, evaluated):
    counts, seen = {}, set()
    for frame in all_frames:
        if meta[frame.id]['role'] != 'fit' or frame.condition != domain.module.TRAINED:
            continue
        unit = evidence.counterpart(frame.id)
        if unit in seen:
            continue
        seen.add(unit)
        for target, decision in domain.module.reference(frame).supervision.items():
            if decision.value is None:
                continue
            classes = counts.setdefault(target, {True: 0, False: 0, 'unstated': 0})
            classes[decision.value] = classes.get(decision.value, 0) + 1
    choices = {target: max(values, key=values.get) for target, values in counts.items()}
    support, complete, submitted = {}, 0, 0
    families = set()
    for frame in evaluated:
        correct, targets = 0, 0
        families.add(meta[frame.id]['family'])
        for target, decision in domain.module.reference(frame).supervision.items():
            if decision.value is None or target not in choices:
                continue
            group = support.setdefault(target, {'correct': 0, 'submitted': 0, 'classes': {}})
            group['correct'] += choices[target] == decision.value
            group['submitted'] += 1
            label = str(decision.value)
            group['classes'][label] = group['classes'].get(label, 0) + 1
            correct += choices[target] == decision.value
            targets += 1
        submitted += bool(targets)
        complete += targets > 0 and correct == targets
    return {'source': 'fit_label_majority', 'fitted_role': 'fit', 'fit_situations': len(seen), 'choices': choices, 'targets': support,
            'correct': sum(row['correct'] for row in support.values()), 'submitted': sum(row['submitted'] for row in support.values()),
            'families': len(families), 'complete_target_sets': complete, 'situations_with_gold': submitted,
            'action_accuracy': None, 'scope': 'constant primitive choices; no workflow action prediction'}

def rule_only(domain: Domain, frames, paths) -> dict:
    with Runtime() as rt:
        out = {}
        for frame in frames:
            switch = rt.load(paths[frame.condition], draft=True)
            (facts,) = switch.score_facts([frame.input], facts=list(domain.sets), sources=())
            out[frame.id] = {}
            for name, keys in domain.sets.items():
                states = materialize(facts[name]).get("states") or {}
                out[frame.id][name] = {k: states[k] != "ineligible" if k in states else None for k in keys}
        return out


def fingerprint(domain: Domain, manifest, paths, arms, corpus) -> dict:
    return {"corpus": corpus, "task": {condition: files.sha256_file(path) for condition, path in paths.items()},
            "documents": digest({"sources": manifest["pack"]["sources"], "rules": manifest["pack"]["rules"]}),
            "assignment": manifest["assignment_sha256"], "arms": arms, "scorer_code": outcomes.fingerprint(), "presentation": presentation_identity()}


def reference_identity(manifest) -> dict:
    return {"producer": manifest["reference"]["producer"], "code": manifest["reference"]["code"], "revision": manifest["reference"]["revision"],
            "reference_sha256": manifest["payloads"]["reference.jsonl"]["sha256"], "reviews_sha256": manifest["payloads"]["reviews.jsonl"]["sha256"]}


def scope_targets(domain: Domain) -> list:
    return [*(f"{name}/{key}" for name, keys in domain.sets.items() for key in keys), *domain.scalars]


def stored(payload) -> dict:
    return {key: value for key, value in payload.items() if key not in ("arm", "kind", "condition", "frame")}


def burdens(domain: Domain, rows) -> dict:
    return {"forbidden_action_situations": sum(action_sets(set_rows(domain, rows, name))["frames_with_prohibited"] for name in domain.sets),
            "failures": sum(row["outcome"] == "failed" for row in rows)}


def adoption(difference, margin, baseline, specialist, *, exposure, lineage, review) -> dict:
    interval = difference["interval"]
    constraints = {name: {"baseline": baseline[name], "specialist": specialist[name], "met": specialist[name] <= baseline[name]} for name in baseline}
    if exposure["state"] == "empty":
        verdict = "no_check_evidence"
    elif not exposure["independent"]:
        verdict = "reused_evidence"
    elif lineage["state"] != "declared_disjoint":
        verdict = "unproven_lineage" if lineage["state"] == "unproven" else "overlapping_lineage"
    elif review != "reviewed":
        verdict = "unreviewed_reference"
    elif interval is None or difference["support"] != "reported":
        verdict = "inconclusive"
    elif not all(c["met"] for c in constraints.values()):
        verdict = "constraint_violated"
    else:
        verdict = "adopt" if interval[0] > margin else "not_adopted"
    return {"schema": DECISION, "required_improvement": margin, "direction": "specialist - baseline", "unit": "situation-weighted correct_per_submission",
            "rule": "adopt only when the 95% family-resampling lower bound strictly exceeds the required improvement, with no more forbidden-action "
                    "situations or failures than the baseline, on unexposed reviewed evidence whose receipt lineage is established disjoint",
            "lower_bound": None if interval is None else interval[0], "constraints": constraints, "exposure": exposure["state"],
            "lineage": {key: lineage[key] for key in ("state", "reason") if key in lineage}, "review": review, "verdict": verdict}


def lineages(domain: Domain, receipt, context, frames, components, kinds, conditions) -> dict:
    trained = [frame for frame in frames if frame.condition == domain.module.TRAINED]
    assessed = assess_lineage(entry(receipt.path), context, [(frame.input, components[frame.id]) for frame in trained]) if trained else None
    return {f"{kind}|{condition}": assessed if kind == "workflow" and condition == domain.module.TRAINED else
            unproven("typed direct asks are outside the receipt's frozen task scope" if kind == "direct" else
                     f"condition {condition} evaluates an authored task the receipt never froze; cross-condition lineage is not assessed")
            for kind in kinds for condition in conditions}


def local_executions(executions, native) -> None:
    local = {name: execution for name, execution in executions.items() if execution is not None}
    if native.get("device") == "gpu" and any(execution.get("device") in (None, "cpu") for execution in local.values()):
        raise ConfigError(f"GPU evaluation was requested but realized executions are {local}; no CPU fallback")
    settings = {name: {key: value for key, value in execution.items() if key != "build"} for name, execution in local.items()}
    if len({json.dumps(value, sort_keys=True) for value in settings.values()}) > 1:
        raise ConfigError(f"local arms realized different native executions {settings}; evaluate under one explicit native configuration")


def frozen(spec):
    if not isinstance(spec, str) or not spec.endswith(".json"):
        raise ConfigError("the specialist arm is a deployment receipt.json from semantic-trainer select; its frozen task, policies and execution are evaluated")
    return load_receipt(spec)


def keyed(directory, domain_name, roles, plan_id) -> Path:
    return Path(directory) / f"{domain_name}-{'-'.join(roles)}-{plan_id[:16]}.json"


def evaluate(domain_name, pack_dir, evidence_manifest, arms, output=None, *, results=None, baseline, specialist=None, margin: float, kinds=("workflow", "direct"), roles=("check",), conditions=None,
             native=None, provider=None, key_file=None, fresh=False, corpus=None) -> dict:
    if not 0 <= margin < 1:
        raise ConfigError("predeclare the required improvement margin in [0, 1) before scoring")
    if (output is None) == (results is None):
        raise ConfigError("name either the comparison output or a results directory keyed by the experiment plan")
    if baseline not in arms or specialist is not None and (specialist not in arms or specialist == baseline):
        raise ConfigError("declare one baseline arm and at most one distinct specialist arm among the comparison arms")
    domain = DOMAINS[domain_name]
    pack = domain.pack
    receipt = frozen(arms[specialist]) if specialist else None
    data = evidence.read(pack_dir, corpus)
    manifest, frames, meta = evidence.frames_of(data, pack_dir)
    all_frames = frames
    frames = [frame for frame in frames if meta[frame.id]["role"] in roles and frame.condition in (conditions or domain.conditions)]
    if not frames:
        raise ConfigError(f"{pack_dir}: no frames for roles {list(roles)}")
    golds = {frame.id: evidence.validate_gold(frame, domain.module.reference(frame), domain.module.SCOPE, domain.module.REVISION) for frame in frames}
    work = work_directory(output) if output is not None else Path(results) / ".work"
    paths = {condition: receipt.verified_task() if receipt and condition == domain.module.TRAINED else definition_path(domain, condition, work)
             for condition in sorted({frame.condition for frame in frames})}
    specs = {condition: files.read_yaml(path) for condition, path in paths.items()}
    domain = domain.scoped(set.intersection(*(set(spec["facts"]) for spec in specs.values())))
    task = {condition: digest(spec) for condition, spec in specs.items()}
    document = evidence.document(pack)
    context = EvidenceContext.from_manifest(evidence_manifest)
    if context.review != manifest["review"]:
        raise ConfigError(f"{evidence_manifest}: evidence assignment was not staged from {pack_dir}; stage the pack again")
    review = evidence.scoped_review(data, pack, frames, golds, scope_targets(domain), {c: evidence.criterion(spec) for c, spec in specs.items()},
                                    compose=domain.module.standard().compose, authored=domain.module.authored())
    names = [baseline, *([specialist] if specialist else []), *(name for name in arms if name not in (baseline, specialist))]
    role = {name: "baseline" if name == baseline else "specialist" if name == specialist else "reference" for name in names}
    components = {frame.id: meta[frame.id]["family"] for frame in frames}
    rules = baselines(domain, frames, golds, rule_only(domain, frames, paths))
    majority = majority_baseline(domain, all_frames, meta, frames)
    settings = {**(receipt.native if receipt else {}), **(native or {})}
    with ExitStack() as stack:
        judges = {name: fitted(arms[name]) if name == specialist else
                  judge_spec(arms[name], provider=provider, key_file=key_file, **settings) if isinstance(arms[name], str) else arms[name] for name in names}
        local = [judge for judge in judges.values() if isinstance(judge, Model)]
        job = stack.enter_context(model_job(*local)) if local else None
        opened = {name: (stack.enter_context(runtime(judge, job=job, receipts=(receipts := Receipts()))), receipts) for name, judge in judges.items()}
        local_executions({name: rt.execution() for name, (rt, _) in opened.items()}, settings)
        switches = {(name, condition): rt.load(arms[name]) if name == specialist and condition == domain.module.TRAINED else rt.load(paths[condition], draft=True)
                    for name, (rt, _) in opened.items() for condition in paths}
        items = {condition: direct_items(domain, spec) for condition, spec in specs.items()}
        identities = {name: {"spec": arms[name] if isinstance(arms[name], str) else "in-process judge", "role": role[name], "model": rt.identity(),
                             **({"receipt": receipt.id} if name == specialist else {})} for name, (rt, _) in opened.items()}
        captures = {(name, kind, condition): {"judgments": judgments(domain, switches[(name, condition)].snapshot.plan) if kind == "workflow" else judgments(domain, items=items[condition]),
                                              "normalization": rt.judge().normalization, "execution": digest(identities[name])}
                    for name, (rt, _) in opened.items() for kind in kinds for condition in paths}
        executions = {name: rt.info() for name, (rt, _) in opened.items()}
        lineage = lineages(domain, receipt, context, frames, components, kinds, paths) if receipt else {}
        plan = evaluation_plan({f"{role[name]}:{name}": digest(identities[name]) for name in names}, scope=[[frame.id, components[frame.id]] for frame in frames],
                               conditions=[digest(condition_of(domain, kind, condition) | {"task": task[condition]}) for kind in kinds for condition in paths],
                               metrics={"schema": SCHEMA, "metrics": METRICS, "probabilities": PROBABILITY, "decision": DECISION, "difference": "arm - baseline", "bootstrap": BOOTSTRAP,
                                        "protocol": PROTOCOL, "roles": list(roles), "fingerprint": fingerprint(domain, manifest, paths, identities, data["manifest_sha256"]),
                                        "primary": {"comparison": f"workflow|{domain.module.TRAINED}", "targets": scope_targets(domain), "margin": margin,
                                                    "languages": sorted({tag for frame in frames if frame.condition == domain.module.TRAINED for tag in frame.tags}
                                                                       & set(getattr(domain.module, "LANGUAGES", ()))),
                                                    "diagnostics": "other presentations, direct primitives and language slices"}}, margin=margin)
        output = output if output is not None else keyed(results, domain.name, roles, plan["id"])
        exposure = context.consult(plan, {components[f.id] for f in frames if meta[f.id]["role"] == "check"})
        journal = stack.enter_context(resumable(output, {"script": "bench.snapshots", "schema": SCHEMA, "arms": {name: digest(identities[name]) for name in names},
                                                         "tasks": task, "documents": digest(document), "presentation": presentation_identity(),
                                                         "kinds": list(kinds), "requested": requested(domain), "protocol": PROTOCOL}, fresh=fresh))
        warmups = {work_key("setup", digest(identities[name]), task[condition], kind, condition): (name, kind, condition)
                   for name in names for kind in kinds for condition in paths}
        for key in journal.pending(warmups):
            name, kind, condition = warmups[key]
            warm = predict(opened[name][0], domain, kind, next(f for f in frames if f.condition == condition), switches[(name, condition)], items[condition], document)
            journal.commit([(key, {"setup": "|".join(warmups[key]), **{field: warm[field] for field in ("ms", "native_work", "requests", "outcome")}})])
        setup = {"|".join(unit): {k: v for k, v in journal.payload(key).items() if k != "setup"} for key, unit in warmups.items()}
        units = {work_key(digest(identities[name]), task[frame.condition], frame.id, frame.public, kind): (name, kind, frame)
                 for index, frame in enumerate(frames) for name in (names if index % 2 == 0 else names[::-1]) for kind in kinds}

        def row_of(name, kind, frame, payload):
            comparison = expected(domain, frame, golds[frame.id], components[frame.id], kind, task[frame.condition])["comparison"]
            return {"arm": name, "kind": kind, "condition": frame.condition, **scored(domain, frame, golds[frame.id], stored(payload), components[frame.id], comparison)}
        todo = set(journal.pending(units))
        progress = {key: Progress(f"bench.snapshots {domain.name} {key}", sum(f.condition == key.split("|")[2] for f in frames), unit="frames",
                                  resumed=[row_of(*units[k], journal.payload(k)) for k in units if k not in todo and "|".join((units[k][0], units[k][1], units[k][2].condition)) == key])
                    for key in setup}
        for key, (name, kind, frame) in units.items():
            if key not in todo:
                continue
            prediction_ = predict(opened[name][0], domain, kind, frame, switches[(name, frame.condition)], items[frame.condition], document,
                                  captures[(name, kind, frame.condition)])
            journal.commit([(key, {"arm": name, "kind": kind, "condition": frame.condition, "frame": frame.id, **prediction_})])
            progress[f"{name}|{kind}|{frame.condition}"].record([row_of(name, kind, frame, prediction_)])
        finals = {key: p.finish() for key, p in progress.items()}
        rows = {key: [] for key in setup}
        for key, (name, kind, frame) in units.items():
            rows[f"{name}|{kind}|{frame.condition}"].append(row_of(name, kind, frame, journal.payload(key)))
        receipts = {name: recorded.summary() for name, (_, recorded) in opened.items()}
        scopes = {(kind, condition): [expected(domain, f, golds[f.id], components[f.id], kind, task[condition])
                                      for f in frames if f.condition == condition and golds[f.id].complete] for kind in kinds for condition in paths}
        results = {key: {"progress": finals[key], "units": units_of(domain, value, key.split("|")[2]),
                         **summary(domain, value, scopes[tuple(key.split("|")[1:])], setup[key]), "slices": slices(value), "rows": value}
                   for key, value in rows.items()}
        controls = {f"{name}|{kind}": counterparts([row for key, value in rows.items() if key.startswith(f"{name}|{kind}|") for row in value])
                    for name in names for kind in kinds}
        comparisons = []
        for kind in kinds:
            for condition in paths:
                left = rows[f"{baseline}|{kind}|{condition}"]
                for other in names[1:]:
                    right = rows[f"{other}|{kind}|{condition}"]
                    comparison = {"condition": f"{kind}:{condition}", "baseline": baseline, "arm": other, "role": role[other],
                                  **paired(scopes[(kind, condition)], [r for r in left if r["complete_gold"]], [r for r in right if r["complete_gold"]],
                                           **BOOTSTRAP, names=(baseline, other))}
                    if role[other] == "specialist":
                        comparison["adoption"] = adoption(comparison["difference"], margin, burdens(domain, left), burdens(domain, right),
                                                          exposure=exposure, lineage=lineage[f"{kind}|{condition}"], review=review["status"])
                    comparisons.append(comparison)
        artifact = {"schema": SCHEMA, "domain": domain.name, "roles": list(roles), "pack": manifest["pack"]["name"], "review": review,
                    "reference": reference_identity(manifest),
                    "claims": "quality claims require current review of the declared comparison's frames, targets, criteria and sources" if review["status"] != "reviewed"
                              else f"reviewed reference for the declared comparison ({review['disclosure']})",
                    "plan": plan, "fingerprint": plan["metrics"]["fingerprint"], "fresh": fresh, "executions": executions, "exposure": exposure, "lineage": lineage,
                    "estimands": {
                        "workflow": "one public Switch call per frame over the authored definition; decision, every scored target and cost from that Call",
                        "direct": "typed independent membership questions over the same evidence, document and menu; no exact guards",
                        "references": "reference arms are descriptive; adoption applies only to the predeclared specialist against the baseline"},
                    "baselines": rules, "fit_label_majority": majority,
                    "results": results, "probabilities": probability_report(domain, results, roles, review["status"], plan["id"], exposure["state"]), "paired": comparisons, "counterparts": controls, "provider_receipts": receipts}
        return finish(journal, output, lambda _: artifact)


UNAVAILABLE = {"judge_auc": "stored rows keep public target states, not per-target judge-only scores of a separate diagnostic pass",
               "score_diagnostics": "per-source deployed score populations are not stored rows; recovering them needs inference",
               "label_provenance": "stored rows keep reference values, not label-source histories",
               "policy": "acceptance policies belong to the receipt's frozen lock, not to the comparison rows"}


def derived_check(comparison, output) -> dict:
    artifact = files.read_json(comparison)
    plan = artifact.get("plan") if isinstance(artifact, dict) else None
    if artifact.get("schema") != SCHEMA or artifact.get("domain") not in DOMAINS or not isinstance(plan, dict) \
            or plan.get("id") != digest({k: v for k, v in plan.items() if k != "id"}) or not isinstance(artifact.get("lineage"), dict):
        raise ConfigError(f"{comparison}: not a schema-{SCHEMA} bench.snapshots comparison; run the predeclared comparison again")
    specialist = next((arm.split(":", 1)[1] for arm in plan["arms"] if arm.startswith("specialist:")), None)
    domain = DOMAINS[artifact["domain"]].scoped(plan["metrics"]["primary"]["targets"])
    condition = domain.module.TRAINED
    key = f"{specialist}|workflow|{condition}"
    if specialist is None or key not in artifact["results"]:
        raise ConfigError(f"{comparison}: its plan scored no specialist workflow rows of the trained condition {condition}; no check derives from it")
    result = artifact["results"][key]
    rows = result["rows"]
    scope = [{key_: row[key_] for key_ in ("key", "family", "gold", "comparison")} for row in rows if row["complete_gold"]]
    (decision,) = [p["adoption"] for p in artifact["paired"] if p["arm"] == specialist and p["condition"] == f"workflow:{condition}"]
    report = {"schema": CHECK, "kind": "derived-check", "domain": domain.name, "arm": specialist, "receipt": artifact["fingerprint"]["arms"][specialist]["receipt"],
              "parent": {"plan": plan["id"], "schema": artifact["schema"], "comparison": str(comparison), "sha256": files.sha256_file(comparison),
                         "run_id": artifact["execution"]["run_id"]},
              "condition": condition, "evaluation": "workflow", "roles": artifact["roles"], "review": artifact["review"], "exposure": artifact["exposure"],
              "lineage": artifact["lineage"][f"workflow|{condition}"], "adoption": decision,
              "population": {**units_of(domain, rows, condition),
                             "scope": "the specialist workflow rows of the trained condition stored by the parent experiment; the same public Calls, "
                                      "no further inference or exposure consultation"},
              **summary(domain, rows, scope, result["latency"]["setup"]), "slices": slices(rows),
              "probabilities": probability_report(domain, {key: result}, artifact["roles"], artifact["review"]["status"], plan["id"], artifact["exposure"]["state"]), "unavailable": UNAVAILABLE}
    report["id"] = digest(report)
    return outcomes.report(output, report)


def episodes(pack_dir, spec, output, *, seed=0, cap=5, roles=("check",), condition="structured", fresh=False, corpus=None) -> dict:
    domain = DOMAINS["zt411"]
    data = evidence.read(pack_dir, corpus)
    roles_of = {evidence.counterpart(row["frame"]): row["role"] for row in data["reference"]}
    planned = {work_key("episode", row["scenario"], row["variant"]): row for row in data["scenarios"]
               if roles_of.get(f"{row['scenario']}:{row['variant']}") in roles}
    definition = definition_path(domain, condition, work_directory(output))
    domain = domain.scoped(files.read_yaml(definition)["facts"])
    with runtime(judge_spec(spec) if isinstance(spec, str) else spec) as rt:
        model = rt.identity()
        config = {"script": "bench.snapshots episodes", "code": outcomes.fingerprint(), "presentation": presentation_identity(),
                  "corpus": data["manifest_sha256"], "pack": {name: entry["sha256"] for name, entry in data["manifest"]["payloads"].items()}, "definition": files.sha256_file(definition),
                  "model": model, "execution": rt.execution(), "seed": seed, "cap": cap, "roles": list(roles), "condition": condition}
        with resumable(output, config, fresh=fresh) as journal:
            switch = rt.load(definition, draft=True)

            def decide(frame):
                prediction = predict(rt, domain, "workflow", frame, switch)
                values = {t.split("/")[1]: p["value"] for t, p in prediction["targets"].items() if t.startswith("actions/")}
                return {"selected": {k for k, v in values.items() if v}, "unresolved": {k for k, v in values.items() if v is None},
                        "outcome": prediction["outcome"]}
            todo = journal.pending(planned)
            progress = Progress("bench.snapshots episodes", len(planned), unit="episodes", resumed=map(episode_row, resumed(journal, planned, todo)))
            for key in todo:
                row = planned[key]
                trace = procedures.episode(row, decide, seed=seed, condition=condition, cap=cap)
                if not procedures.replay(row, trace, condition=condition):
                    raise ConfigError(f"{row['scenario']}:{row['variant']}: model-free replay differs from the recorded trace")
                journal.commit([(key, trace)])
                progress.record([episode_row(trace)])
            final = progress.finish()
            return finish(journal, output, lambda traces: episode_summary(traces, planned.values(), spec, model, seed, cap, condition, final))


def episode_row(trace) -> dict:
    return {"outcome": "failed" if trace["termination"] == "failed" else "predicted", "correct": trace["goal_reached"]}


def step_rows(traces) -> list[dict]:
    keys = procedures.MENU
    return [set_row(keys, {k: True if k in step["justified"] else None if k in step["reference_unresolved"] else False for k in keys}, step["prohibited"],
                    {k: True if k in step["selected"] else None if k in step["unresolved"] else False for k in keys}, step["outcome"])
            for trace in traces for step in trace["steps"]]


def episode_summary(traces, planned, spec, model, seed, cap, condition, progress) -> dict:
    controls = [procedures.controls(row, seed=seed, condition=condition) for row in planned]
    terminations = {}
    for trace in traces:
        terminations[trace["termination"]] = terminations.get(trace["termination"], 0) + 1
    return {"schema": 1, "spec": str(spec), "model": model, "seed": seed, "cap": cap, "condition": condition, "progress": progress,
            "scope": "complete episodes committed through the execution journal; an interrupted episode restarts from its initial state",
            "goal_reached": sum(t["goal_reached"] for t in traces), "terminations": terminations,
            "snapshot_steps": set_metrics(step_rows(traces)),
            "controls": {name: sum(c[name]["goal_reached"] for c in controls) for name in ("always_abstain", "always_cancel")},
            "traces": traces}


def main():
    parser = argparse.ArgumentParser(prog="bench.snapshots", description="explicit snapshot evaluation of published packs through the public Runtime")
    parser.add_argument("domain", choices=[*sorted(DOMAINS), "zt411-episodes", "check"],
                        help="check: derive the specialist check summary from a published --comparison without inference or exposure consultation")
    parser.add_argument("--pack", type=Path, help="published benchmark directory (benchmark.json)")
    parser.add_argument("--corpus", metavar="SHA256", help="the run's bound benchmark.json digest; a different publication is refused before scoring")
    parser.add_argument("--comparison", type=Path, help="check: the published predeclared comparison")
    parser.add_argument("--evidence", type=Path, help="staged evidence assignment (DEFINITION.dataset.json from bench.evidence stage); its audit records the consultation")
    spec = "NAME=SPEC: local GGUF path, deployment receipt.json or openrouter:MODEL"
    parser.add_argument("--baseline", metavar="NAME=SPEC", help=f"{spec}; every difference is arm minus baseline")
    parser.add_argument("--specialist", metavar="NAME=SPEC", help=f"{spec}; the only arm with an adoption verdict")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=SPEC", help=f"{spec}; descriptive comparison only")
    parser.add_argument("--native", action="append", default=[], metavar="KEY=VALUE",
                        help="native setting applied identically to every local arm, e.g. device=gpu adapter=chat; device=gpu refuses CPU fallback")
    parser.add_argument("--provider", help="pinned OpenRouter provider route for openrouter:MODEL arms")
    parser.add_argument("--key-file", type=Path, help="OpenRouter key file for openrouter:MODEL arms (default: OPENROUTER_API_KEY)")
    parser.add_argument("--condition", action="append", help="presentation condition, repeatable; defaults to the domain conditions")
    parser.add_argument("--kind", action="append", choices=("workflow", "direct"))
    parser.add_argument("--role", action="append", choices=("fit", "model_select", "acceptance_fit", "check"))
    parser.add_argument("--margin", type=float, help="predeclared required improvement in [0, 1) of situation-weighted correct_per_submission, specialist minus "
                                                     "baseline: adopt only when the 95%% family-resampling lower bound strictly exceeds it (required for packs)")
    parser.add_argument("--output", type=Path, help="published result path; a comparison may instead name --results")
    parser.add_argument("--results", type=Path, help="comparison: directory of results named DOMAIN-ROLES-PLAN.json by the frozen experiment plan")
    parser.add_argument("--check", type=Path, metavar="DIR", help="comparison: after scoring, derive the specialist check summary into DIR/PLAN.json")
    parser.add_argument("--fresh", action="store_true", help="archive the existing journal (failed rows included) and start a new recorded run")
    args = parser.parse_args()
    if args.domain == "check":
        if args.comparison is None or args.output is None:
            parser.error("check derives from --comparison RESULT into --output")
        try:
            report = derived_check(args.comparison, args.output)
        except ConfigError as error:
            parser.exit(2, f"{error}\n")
        print(json.dumps({"scored": report["scored"], "exposure": report["exposure"]["state"], "lineage": report["lineage"]["state"],
                          "adoption": report["adoption"]["verdict"]}, indent=1, default=str))
        return
    if args.pack is None or args.baseline is None:
        parser.error("--pack and --baseline are declared for every evaluation")
    if args.domain != "zt411-episodes" and (args.margin is None or args.evidence is None):
        parser.error("--margin (required improvement) and --evidence are declared before scoring")
    declared = [args.baseline, *([args.specialist] if args.specialist else []), *args.reference]
    arms = dict(arm.split("=", 1) for arm in declared)
    if len(arms) != len(declared):
        parser.error("arm names must be distinct")
    if args.domain == "zt411-episodes":
        if args.output is None:
            parser.error("episodes are published to --output")
        for name, spec_ in arms.items():
            artifact = episodes(args.pack, spec_, args.output.with_name(f"{args.output.stem}-{name}{args.output.suffix}"), fresh=args.fresh, corpus=args.corpus)
            print(json.dumps({name: {"goal_reached": artifact["goal_reached"], "terminations": artifact["terminations"]}}))
        return
    if (args.output is None) == (args.results is None) or args.check is not None and args.specialist is None:
        parser.error("a comparison names one of --output or --results; --check needs a --specialist")
    roles = tuple(args.role or ("check",))
    try:
        artifact = evaluate(args.domain, args.pack, args.evidence, arms, args.output, results=args.results, baseline=args.baseline.split("=", 1)[0],
                            specialist=args.specialist.split("=", 1)[0] if args.specialist else None, margin=args.margin,
                            kinds=tuple(args.kind or ("workflow", "direct")), roles=roles, conditions=tuple(args.condition) if args.condition else None,
                            native={key: yaml.safe_load(value) for key, value in (item.split("=", 1) for item in args.native)},
                            provider=args.provider, key_file=args.key_file, fresh=args.fresh, corpus=args.corpus)
        comparison = args.output or keyed(args.results, args.domain, roles, artifact["plan"]["id"])
        derived = derived_check(comparison, args.check / f"{artifact['plan']['id'][:16]}.json") if args.check is not None else None
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({"comparison": str(comparison), "scored": {key: {**value["units"], **value["scored"]} for key, value in artifact["results"].items()},
                      "adoption": {p["condition"]: p["adoption"]["verdict"] for p in artifact["paired"] if "adoption" in p},
                      **({"check": str(args.check / f"{artifact['plan']['id'][:16]}.json"), "derived": derived["adoption"]["verdict"]} if derived else {})},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
