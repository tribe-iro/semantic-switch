from __future__ import annotations

import json
from collections.abc import Mapping
from collections import Counter

from .. import files
from ..snapshot import Snapshot
from ..results import thaw
from ..switch import Switch
from ..evidence import flatten, label_provenance, path_record, record_path
from ..errors import ConfigError
from ..judgment import truth
from ..metrics import COUNTS, auc, confusion, correctness, ordered, rates, scored_answer
from .common import taught, test_items, uses, evidence_interval, input_key
from .lock import STATES, LockError


def labelled_targets(plan, item: dict) -> list[str]:
    gold = flatten(plan, item.get("facts", {}))
    requested = [f for f in plan.observed if plan.facts[f].name in gold]
    parents = {plan.fact_index[plan.facts[f].member[0]] for f in requested if plan.facts[f].member}
    return [plan.facts[f].name for f in sorted({*requested, *parents})]


def scored_population(sw: Switch, snap: Snapshot, items: list[dict], reuse=None) -> list:
    return sw.score([it["input"] for it in items], demand=[(labelled_targets(snap.plan, it), "rule" in it) for it in items], snap=snap,
                    reuse=reuse) if items else []


def evaluate_rules(sw: Switch, snap: Snapshot, items: list[dict], families=None, review=None, rows=None) -> dict:
    from ..selection import occurrences
    families = families or [None] * len(items)
    rows = scored_population(sw, snap, items) if rows is None else rows
    labelled = [(it, family, row.decision) for it, family, row in zip(items, families, rows, strict=True) if "rule" in it]
    results = [result for _, _, result in labelled]
    statuses = Counter(r.status for r in results)
    ok = [(r, it) for it, _, r in labelled if r.status == "ok"]
    interval = evidence_interval((family, occurrences(item.get('selections', ())), result.status == 'ok', result.rule == item['rule'])
                                 for item, family, result in labelled)
    if snap.locked and any(snap.lineage.get(f'{fact.source}:{name}', {}).get('state') != 'declared_disjoint'
                           for result in results for name, fact in result.facts.items() if fact.source in ('judge', 'classifier')):
        interval['interval'] = None
        interval['reason'] = 'decision depends on a source with unknown/overlapping lineage'
    return {'interval': interval, 'precision_95': interval['interval'] if interval.get('metric') == 'accepted_precision' else None,
            "items": len(labelled), "label_provenance": label_provenance(snap.plan, [it for it, _, _ in labelled], review=review), "ok": len(ok), "correct": sum(r.rule == it["rule"] for r, it in ok),
            "uncertain": statuses["uncertain"], "needs_input": statuses["needs_input"],
            "failed": statuses["failed"]}


def evaluate_facts(sw: Switch, snap: Snapshot, items: list[dict], families=None, review=None, rows=None) -> dict:
    from ..selection import occurrences
    plan = snap.plan
    families = families or [None] * len(items)
    inputs = [it["input"] for it in items]
    gold = [flatten(plan, it.get("facts", {})) for it in items]
    requested = [f for f in plan.observed if any(plan.facts[f].name in g for g in gold)]
    deployed = (scored_population(sw, snap, [{k: v for k, v in it.items() if k != "rule"} for it in items]) if rows is None else rows) if items else []
    judge_only = list(deployed)
    diagnostic_facts = [plan.facts[f].name for f in requested if plan.facts[f].kind == "check"]
    diagnostic_inputs = [n for n, row in enumerate(deployed) if row.error is None and
                         any(name in gold[n] and row.targets[name].source in ("example", "classifier") for name in diagnostic_facts)]
    if diagnostic_inputs:
        diagnostic = sw.score([inputs[n] for n in diagnostic_inputs], diagnostic_facts, sources=("judge",), snap=snap)
        for n, row in zip(diagnostic_inputs, diagnostic):
            judge_only[n] = row
    out = {}
    for f in requested:
        fact = plan.facts[f]
        rows_ = [(c.targets[fact.name], j.targets.get(fact.name), it) for c, j, it, g in zip(deployed, judge_only, items, gold) if fact.name in g]
        row_families = [family for family, g in zip(families, gold, strict=True) if fact.name in g]
        own = [(t.accepted if t.dist is not None else None, fact.index(g[fact.name])) for (t, _, _), g in
               zip(rows_, (g for g in gold if fact.name in g))]
        statuses = Counter(t.status for t, _, _ in rows_)
        answers = [scored_answer(t.result.prediction, fact.keys[label], t.status, fact.keys) for (t, _, _), (_, label) in zip(rows_, own)]
        row = {"labelled": len(rows_), **correctness((answer["correct"], answer["accepted"]) for answer in answers),
               "label_provenance": label_provenance(plan, [it for _, _, it in rows_], review=review), "statuses": dict(statuses), **contrasts(own, [it for _, _, it in rows_]),
               "policy": {src: thaw(sides[fact.name]) for src, sides in snap.sides.items() if fact.name in sides}}
        table = confusion({"gold": y, "predicted": o} for o, y in own)
        if fact.kind == "check":
            scored = [(j.dist[0], y == 0) for (_, j, _), (_, y) in zip(rows_, own) if y != 2 and j is not None and j.dist is not None]
            row |= check_counts(table) | {"judge_auc": auc([s for s, _ in scored], [y for _, y in scored])}
        else:
            row |= {"settled": table["predicted"], "settled_correct": table["correct"]}
            row |= pick_counts(fact, table) if fact.kind == "pick" else ordered_counts(own)
        row['interval'] = evidence_interval((family, occurrences(item.get('selections', ())), index is not None, index == label)
                                            for family, (_, _, item), (index, label) in zip(row_families, rows_, own))
        row['precision_95'] = row['interval']['interval'] if row['interval'].get('metric') == 'accepted_precision' else None
        actual_sources = {t.source for t, _, _ in rows_ if t.source in ('judge', 'classifier')}
        source_lineage = [snap.lineage.get(f'{source}:{fact.shown}', {}) for source in actual_sources]
        if snap.locked and any(value.get('state') != 'declared_disjoint' for value in source_lineage):
            row['interval']['interval'] = None
            row['interval']['reason'] = 'artifact lineage unknown or overlapping'
            row['precision_95'] = None
        from ..observations import Observation
        from .policy import score_diagnostics
        source_rows = {}
        for n, ((target, _, item), (_, label)) in enumerate(zip(rows_, own)):
            if target.source in ("judge", "classifier"):
                source_rows.setdefault(target.source, []).append(Observation(str(n), item["input"], label, target.dist, target.mass, target.pending))
        row["score_diagnostics"] = {source: score_diagnostics(fact, values, snap.policy(source, fact))
                                    for source, values in source_rows.items()}
        row['candidate_id'] = snap.candidate_id
        if fact.member:
            out.setdefault(fact.member[0], {"members": {}})["members"][fact.member[1]] = row
        else:
            out[fact.name] = row
    for name, report in out.items():
        if "members" in report:
            report |= selection_counts(snap, plan.fact_index[name], deployed, items, report["members"])
    return out


def cells(table: dict) -> tuple[Counter, Counter]:
    return Counter({(c["gold"], c["predicted"]): c["count"] for c in table["confusion"]}), Counter({c["gold"]: c["count"] for c in table["abstained"]})


def check_counts(table: dict) -> dict:
    counts, abstained = cells(table)
    positives = sum(n for (g, _), n in counts.items() if g == 0) + abstained[0]
    accepted_true, correct_true = sum(n for (_, p), n in counts.items() if p == 0), counts[0, 0]
    return {"actual_positives": positives, "accepted_true": accepted_true, "accepted_true_correct": correct_true,
            "accepted_false": sum(n for (_, p), n in counts.items() if p == 1), "accepted_false_correct": counts[1, 1],
            "precision_true": correct_true / accepted_true if accepted_true else None,
            "recall_true": correct_true / positives if positives else None,
            "silent_misses": counts[0, 1], "to_review": abstained[0]}


def pick_counts(fact, table: dict) -> dict:
    counts, abstained = cells(table)
    out = {key: {} for key in fact.keys}
    for (g, p), n in counts.items():
        out[fact.keys[g]][fact.keys[p]] = n
    for g, n in abstained.items():
        out[fact.keys[g]]["abstain"] = n
    return {"confusion": out, "per_label": {key: {"gold": sum(out[key].values()), "accepted": sum(n for (_, p), n in counts.items() if p == k),
                                                  "correct": counts[k, k]} for k, key in enumerate(fact.keys)}}


def ordered_counts(own: list) -> dict:
    found = ordered({"gold": y, "predicted": o} for o, y in own)
    return {"index_distance": {str(d): n for d, n in found["distances"].items()}, "mean_index_distance": found["mean_absolute_distance"]}


MICRO = ("labelled", "actual_positives", "accepted_true", "accepted_true_correct", "accepted_false", "accepted_false_correct",
         "silent_misses", "to_review")


def selection_counts(snap: Snapshot, f: int, rows: list, items: list[dict], members: dict) -> dict:
    fact = snap.plan.facts[f]
    keys = {card.id for card in fact.cards}
    statuses = Counter()
    for row in members.values():
        statuses.update(row["statuses"])
    micro = {name: sum(row[name] for row in members.values()) for name in MICRO + COUNTS}
    full = [(row.targets[fact.name], item["facts"][fact.name]) for row, item in zip(rows, items)
            if row.error is None and set(item.get("facts", {}).get(fact.name, {})) == keys]
    settled = [(set(target.selected), given) for target, given in full if target.result.settled]
    return {"labelled": micro["labelled"], **rates(micro), **micro, "statuses": dict(statuses),
            "precision_true": micro["accepted_true_correct"] / micro["accepted_true"] if micro["accepted_true"] else None,
            "recall_true": micro["accepted_true_correct"] / micro["actual_positives"] if micro["actual_positives"] else None,
            "full_gold": len(full), "full_gold_complete": sum(target.complete for target, _ in full), "full_gold_settled": len(settled),
            "exact_set_correct": sum(value == {key for key, v in given.items() if v != "unstated" and truth(v)} for value, given in settled)}


def fact_rows(facts: dict):
    for row in facts.values():
        yield from (row["members"].values() if "members" in row else (row,))


def contrasts(own: list, items: list[dict]) -> dict:
    from ..evidence import metadata_values
    pairs: dict = {}
    groups: dict = {}
    for (got, want), it in zip(own, items):
        for pair in metadata_values(it.get("pairs", ()), "pair"):
            pairs.setdefault(pair, []).append(got == want)
        for group in metadata_values(it.get("groups", ()), "group"):
            groups.setdefault(group, []).append((got, want))
    pairs = [v for v in pairs.values() if len(v) >= 2]
    same = [v for v in groups.values() if len(v) >= 2 and len({w for _, w in v}) == 1]
    return {"pairs": len(pairs), "pairs_both_correct": sum(all(v) for v in pairs),
            "invariant_groups": len(same), "invariant_consistent": sum(all(g is not None for g, _ in v) and len({g for g, _ in v}) == 1 for v in same)}


def run_test(rt, path, data: str | None = None) -> dict:
    from .data import Labels
    from .dataset import EvidenceContext, evaluation_plan
    sw = rt.load(path)
    snap = sw.snapshot
    plan = snap.plan
    frozen = evaluation_plan({'candidate': snap.candidate_id}, scope=[list(plan.facts[f].path) for f in plan.observed]) if snap.candidate_id else None
    collector = Labels(sw, data)
    collector.historical_stages(snap)
    groups = {"taught": [], "used": [], "fresh": []}
    used_by: dict = {}
    for it in collector.items:
        found = uses(snap, it["input"], collector)
        key = "taught" if taught(snap, it["input"]) else "used" if found else "fresh"
        groups[key].append(it)
        for u in found if key == "used" else ():
            used_by[u] = used_by.get(u, 0) + 1
    families = {name: [collector.group(('key', input_key(plan, it['input']))) for it in population] for name, population in groups.items()}
    context = EvidenceContext.of(collector)
    exposure = context.consult(frozen, families['fresh']) if frozen else None
    scored = {name: scored_population(sw, snap, population) for name, population in groups.items()}
    fact_populations = {name: evaluate_facts(sw, snap, population, families[name], review=context.review, rows=scored[name])
                        for name, population in groups.items()}
    report = {"switch": plan.name, "mode": f"lock {snap.lock_id}" if snap.locked else "draft",
              **{k: evaluate_rules(sw, snap, v, families[k], review=context.review, rows=scored[k]) for k, v in groups.items()}, "used_by": used_by,
              "facts": fact_populations["fresh"],
              "fact_populations": fact_populations,
              "warnings": list(collector.warnings), "dataset_id": collector.dataset_id, "dataset": str(context.path),
              "candidate_id": snap.candidate_id, "population": "identified sample"}
    if exposure is not None:
        report['audit_use'] = exposure
        withhold([report['fresh'], *fact_rows(report['facts'])], exposure)
    return report


def withhold(rows, exposure):
    if not exposure['consulted_families']:
        return
    for row in rows:
        row['precision_95'] = None
        row['interval']['interval'] = None
        row['interval']['reason'] = f"check families {exposure['state']}: reused evidence, not an independent adoption claim"


def _pct(k: int, n: int) -> str:
    return f"{k} ({k / n:.0%})" if n else str(k)


def _ci(row: dict, key: str) -> str:
    return f" [95% interval {row[key][0]:.0%}–{row[key][1]:.0%}]" if row.get(key) else ""


def _raw(f: dict) -> str:
    return (f"{f['raw_correct']} of {f['predicted']} predictions correct ({f['submitted']} submitted, "
            f"{f['submitted'] - f['predicted']} without a prediction); accepted {f['accepted']}: {f['correct_accepted']} correct, "
            f"{f['wrong_accepted']} wrong; {f['correct_rejected']} correct predictions rejected")


def explain_test(report: dict) -> str:
    u, t, s = report["fresh"], report["taught"], report["used"]
    total = u["items"] + t["items"] + s["items"]
    fact_total = sum(f["labelled"] for population in report.get("fact_populations", {}).values() for f in population.values())
    lines = [f"{report['switch']} ({report['mode']}), {total} rule observations, {fact_total} fact observations:"]
    if u["items"]:
        lines.append(f"  Fresh ({u['items']}): ok {_pct(u['ok'], u['items'])}: {u['correct']} correct, "
                     f"{u['ok'] - u['correct']} wrong{_ci(u, 'precision_95')}. {u['uncertain']} uncertain, "
                     f"{u['needs_input']} needs_input, {u['failed']} failed.")
    if s["items"]:
        how = ", ".join(f"{k} {v}" for k, v in sorted(report["used_by"].items()))
        lines.append(f"  Used by the lock ({s['items']}; {how}): {s['correct']} correct of {s['ok']} settled. "
                     "(They trained, tuned or validated the lock; not a measure of generalisation.)")
    if t["items"]:
        verdict = "all as labelled" if t["correct"] == t["items"] else f"{t['correct']} as labelled"
        lines.append(f"  Exact-taught ({t['items']}): {t['ok']} settled, {verdict}. "
                     "(Checks that corrections stick, not understanding.)")
    for population in ("taught", "used"):
        for name, fact in report.get("fact_populations", {}).get(population, {}).items():
            statuses = fact.get("statuses", {})
            lines.append(f"  {name} ({population}): {_raw(fact)}; {statuses.get('uncertain', 0)} uncertain, "
                         f"{statuses.get('needs_input', 0)} needs_input, {statuses.get('failed', 0)} failed.")
    accepted = []
    for name, f in report["facts"].items():
        lines.append(f"  {name} (fresh): {_raw(f)}.")
        if "accepted_true" in f:
            wrong = f["accepted_true"] - f["accepted_true_correct"]
            lines.append(f"  {name} (fresh): {f['actual_positives']} actually true — {f['accepted_true']} accepted as TRUE "
                         f"({'all correct' if not wrong else f'{wrong} wrong'}){_ci(f, 'precision_95')}, "
                         f"{f['silent_misses']} settled as FALSE (silent miss); "
                         f"{f.get('statuses', {}).get('uncertain', 0)} uncertain, "
                         f"{f.get('statuses', {}).get('needs_input', 0)} needs_input, "
                         f"{f.get('statuses', {}).get('failed', 0)} failed.")
            accepted.append(f["accepted_true"])
            if f.get("judge_auc") is not None:
                errors = f["silent_misses"] + wrong
                lines.append(f"    {name}: judge alone ranks true above false with AUC {f['judge_auc']:.2f}"
                             + ("; ranking is strong, so a boundary change may help" if f["judge_auc"] >= 0.9 and errors
                                else "; the judge barely separates this claim: reword it, add exact rules, or use "
                                     "--precise" if f["judge_auc"] < 0.75 else "") + ".")
        else:
            lines.append(f"  {name} (fresh): settled {_pct(f['settled'], f['labelled'])} of {f['labelled']}, "
                         f"{f['settled_correct']} correct{_ci(f, 'precision_95')}.")
            accepted.append(f["settled"])
        if "members" in f:
            lines.append(f"    {name}: {f['full_gold']} fully labelled, {f['full_gold_complete']} fully examined, "
                         f"{f['full_gold_settled']} settled, {f['exact_set_correct']} exact sets correct.")
        if f.get("pairs"):
            lines.append(f"    {name}: {f['pairs_both_correct']} of {f['pairs']} contrast pairs answered right on both sides.")
        if f.get("invariant_groups"):
            lines.append(f"    {name}: {f['invariant_consistent']} of {f['invariant_groups']} same-label groups "
                         f"answered consistently.")
        for src, side in f.get("policy", {}).items():
            lines.append(f"    {name} {src} policy: " + "; ".join(
                f"{k.upper()} side {STATES[sv['state']]}"
                + (f" (held-out precision {sv['held_out_precision']:.0%} on {sv['held_out_accepted']}, "
                   f"95% interval {sv['interval_95'][0]:.0%}–{sv['interval_95'][1]:.0%})"
                   if sv.get("held_out_precision") is not None and sv.get("interval_95") else "") for k, sv in sorted(side.items(), reverse=True)) + ".")
    lines += [f"  Warning: {w}" for w in report["warnings"]]
    return "\n".join(lines)


def score_observations(sw: Switch, rows, snap: Snapshot | None = None, sources: tuple = ("judge",)) -> list[dict]:
    snap = snap or sw.snapshot
    plan, rows = snap.plan, list(rows)
    ids = Counter(row.get("id") for row in rows)
    if None in ids or any(count > 1 for count in ids.values()):
        raise ConfigError("observation rows need unique ids; repeated or missing: "
                          + ", ".join(sorted(str(k) for k, n in ids.items() if k is None or n > 1)[:5]))
    groups = {}
    for row in rows:
        if "fact" not in row or "bound_input" not in row:
            raise ConfigError(f"observation {row['id']}: needs fact and bound_input; re-export")
        if plan.fact(row["fact"]).hidden:
            raise ConfigError(f"observation {row['id']}: {row['fact']!r} is not a visible fact; address members as {{fact, member}}")
        fact = plan.at(*record_path(row))
        if row.get("meaning_rev") != fact.meaning_rev:
            raise ConfigError(f"observation {row['id']}: {' / '.join(record_path(row))} changed meaning since export "
                              f"(exported {row.get('meaning_rev')}, current {fact.meaning_rev}); re-export")
        if type(row.get("label")) is not int or not 0 <= row["label"] < len(fact.keys):
            raise ConfigError(f"observation {row['id']}: label must index {list(fact.keys)}")
        groups.setdefault(plan.fact_index[fact.name], []).append(row)
    scored = {}
    for f, members in groups.items():
        fact = plan.facts[f]
        for row, scored_row in zip(members, sw.score([row["bound_input"] for row in members], [fact.name], sources=sources, snap=snap)):
            result = scored_row.targets[fact.name].result
            scored[row["id"]] = {"id": row["id"], **path_record(fact.path), "gold": fact.value(row["label"]),
                                 **scored_answer(result.prediction, fact.keys[row["label"]], result.status, fact.keys), "result": result.as_dict(True)}
    return [scored[row["id"]] for row in rows]


def contrast_pairs(rows) -> dict:
    grouped = {}
    for row in rows:
        for pair in row.get("pairs", ()):
            grouped.setdefault((pair, *record_path(row)), []).append(row["id"])
    return grouped


def outcome(plan, r) -> tuple:
    selects = tuple((n, f.value, f.complete) for n, f in sorted(r.facts.items()) if f.complete is not None)
    if plan.rules:
        return r.status, r.answer, selects
    return r.status, tuple((n, f.value) for n, f in sorted(r.facts.items()) if f.settled), selects


def diff(rt, path) -> dict:
    locked = rt.load(path)
    if not locked.snapshot.locked:
        raise LockError("no lock file to compare against; run `ss lock` first")
    draft = rt.load(path, draft=True)
    items, _ = test_items(draft.paths, draft.plan)
    labels = {json.dumps(it["input"], sort_keys=True): it.get("rule") for it in items}
    seen, inputs = set(), []
    records = [r["input"] for r in files.read_jsonl(draft.paths.examples)]
    for d in [it["input"] for it in items] + records + [e.input for exs in draft.snapshot.examples.values() for e in exs]:
        k = json.dumps(thaw(d), sort_keys=True)
        if k not in seen:
            seen.add(k)
            inputs.append(d)
    old_plan, new_plan = locked.plan, draft.plan
    revised = [name for name in new_plan.fact_index if name in old_plan.fact_index and
               old_plan.fact(name).meaning_rev != new_plan.fact(name).meaning_rev]
    changes = [{"input": d, "old": outcome(old_plan, a), "new": outcome(new_plan, b),
                "label": None if revised else labels.get(json.dumps(thaw(d), sort_keys=True)),
                "stale_evidence": revised}
               for d, a, b in zip(inputs, locked.run(inputs), draft.run(inputs))
               if outcome(old_plan, a) != outcome(new_plan, b)]
    summary = Counter(f"{c['old'][1] or c['old'][0]} → {c['new'][1] or c['new'][0]}" for c in changes)
    return {"compared": len(inputs), "changed": len(changes), "summary": dict(summary), "changes": changes, "revised_facts": revised}


def embedded_failures(test, result):
    from ..results import thaw
    supplied = test.model_fields_set
    failures = []
    expected_status = test.expect_status if "expect_status" in supplied else "ok"
    if result.status != expected_status:
        failures.append(f"expected status {expected_status}, got {result.status} for {test.input}")
    if "expect" in supplied and thaw(result.answer) != thaw(test.expect):
        failures.append(f"expected answer {test.expect!r}, got {result.answer!r} for {test.input}")
    if "expect_rule" in supplied and result.rule != test.expect_rule:
        failures.append(f"expected rule {test.expect_rule!r}, got {result.rule!r}")
    for name, value in (test.expect_facts or {}).items():
        fact = result.facts.get(name)
        structured = isinstance(value, Mapping) and "settled" in value
        if value == "unstated":
            value = {"settled": False, "value": None}
            structured = True
        matches = fact is not None and (fact.settled if not structured else fact.settled == value["settled"])
        if matches and (not structured or "value" in value):
            matches = thaw(fact.value) == thaw(value["value"] if structured else value)
        if not matches:
            failures.append(f"expected fact {name}={value!r}, got {None if fact is None else fact.value!r}")
    for name, complete in (test.expect_complete or {}).items():
        fact = result.facts.get(name)
        if fact is None or fact.complete != complete:
            failures.append(f"expected {name}.complete={complete}")
    return failures
