from __future__ import annotations

import json
from collections.abc import Mapping
from collections import Counter

from .. import files
from ..executor import SOURCES
from ..snapshot import Snapshot
from ..results import thaw
from ..switch import Switch
from ..examples import flatten, path_record, record_path
from ..errors import ConfigError
from ..judgment import truth, winner
from ..metrics import COUNTS, correctness, rates
from ..metrics import auc
from .common import taught, test_items, uses, evidence_interval, input_key
from .lock import STATES, LockError


def evaluate_cases(sw: Switch, snap: Snapshot, items: list[dict], families=None) -> dict:
    from ..selection import history
    families = families or [None] * len(items)
    labelled = [(it, family) for it, family in zip(items, families, strict=True) if "case" in it]
    results = sw.run([it["input"] for it, _ in labelled], snap) if labelled else []
    statuses = Counter(r.status for r in results)
    ok = [(r, it) for r, (it, _) in zip(results, labelled) if r.status == "ok"]
    interval = evidence_interval((family, history(item.get('selections', ())), result.status == 'ok', result.case == item['case'])
                                 for result, (item, family) in zip(results, labelled))
    if snap.locked and any(snap.lineage.get(f'{fact.source}:{name}', {}).get('state') != 'declared_disjoint'
                           for result in results for name, fact in result.facts.items() if fact.source in ('judge', 'classifier')):
        interval['interval'] = None
        interval['reason'] = 'decision depends on a source with unknown/overlapping lineage'
    return {'interval': interval, 'precision_95': interval['interval'] if interval.get('metric') == 'accepted_precision' else None,
            "items": len(labelled), "label_origins": origins([it for it, _ in labelled]), "ok": len(ok), "correct": sum(r.case == it["case"] for r, it in ok),
            "uncertain": statuses["uncertain"], "needs_input": statuses["needs_input"],
            "failed": statuses["failed"]}


def origins(items: list[dict]) -> dict:
    from ..examples import target_origins
    return dict(Counter(origin for it in items for origin in target_origins(it)))


def evaluate_facts(sw: Switch, snap: Snapshot, items: list[dict], families=None) -> dict:
    from ..selection import history
    plan = snap.plan
    families = families or [None] * len(items)
    inputs = [it["input"] for it in items]
    gold = [flatten(plan, it.get("facts", {})) for it in items]
    requested = [f for f in plan.observed if any(plan.facts[f].name in g for g in gold)]
    deployed = sw._score(inputs, snap, SOURCES, requested) if items else []
    judge_only = list(deployed)
    diagnostic_facts = [f for f in requested if plan.facts[f].kind == "check"]
    diagnostic_inputs = [n for n, call in enumerate(deployed) if call.facts and
                         any(plan.facts[f].name in gold[n] and
                             call.facts[f].source in ("example", "classifier") for f in diagnostic_facts)]
    if diagnostic_inputs:
        diagnostic = sw._score([inputs[n] for n in diagnostic_inputs], snap, ("judge",), diagnostic_facts)
        for n, call in zip(diagnostic_inputs, diagnostic):
            judge_only[n] = call
    out = {}
    for f in requested:
        fact = plan.facts[f]
        rows = [(c, j, it) for c, j, it, g in zip(deployed, judge_only, items, gold) if fact.name in g]
        row_families = [family for family, g in zip(families, gold, strict=True) if fact.name in g]
        own = [(sw.exec.own(snap, c, f)[0] if c.facts and c.facts[f].dist is not None else None,
                fact.index(g[fact.name])) for c, g in zip(deployed, gold) if fact.name in g]
        statuses = Counter(sw.exec.fact_status(snap, c, f) for c, _, _ in rows)
        guesses = [winner(c.facts[f].dist) if c.facts else None for c, _, _ in rows]
        row = {"labelled": len(rows), **correctness((None if guess is None else guess == label, index is not None)
                                                    for guess, (index, label) in zip(guesses, own)),
               "label_origins": origins([it for _, _, it in rows]), "statuses": dict(statuses), **contrasts(own, [it for _, _, it in rows]),
               "policy": {src: thaw(sides[fact.name]) for src, sides in snap.sides.items() if fact.name in sides}}
        if fact.kind == "check":
            scored = [(j.facts[f].dist[0], y == 0) for (_, j, _), (_, y) in zip(rows, own)
                      if y != 2 and j.facts and j.facts[f].dist is not None]
            row |= check_counts(own) | {"judge_auc": auc([s for s, _ in scored], [y for _, y in scored])}
        else:
            row |= {"settled": sum(o is not None for o, _ in own), "settled_correct": sum(o == y for o, y in own)}
            row |= confusion(fact, own) if fact.kind == "pick" else distances(own)
        row['interval'] = evidence_interval((family, history(item.get('selections', ())), index is not None, index == label)
                                            for family, (_, _, item), (index, label) in zip(row_families, rows, own))
        row['precision_95'] = row['interval']['interval'] if row['interval'].get('metric') == 'accepted_precision' else None
        actual_sources = {c.facts[f].source for c, _, _ in rows if c.facts and c.facts[f].source in ('judge', 'classifier')}
        source_lineage = [snap.lineage.get(f'{source}:{fact.shown}', {}) for source in actual_sources]
        if snap.locked and any(value.get('state') != 'declared_disjoint' for value in source_lineage):
            row['interval']['interval'] = None
            row['interval']['reason'] = 'artifact lineage unknown or overlapping'
            row['precision_95'] = None
        from ..observations import Observation
        from .policy import score_diagnostics
        source_rows = {}
        for n, ((call, _, item), (_, label)) in enumerate(zip(rows, own)):
            if not call.facts:
                continue
            state = call.facts[f]
            if state.source not in ("judge", "classifier"):
                continue
            source_rows.setdefault(state.source, []).append(Observation(str(n), item["input"], label,
                state.dist, state.mass, state.pending))
        row["score_diagnostics"] = {source: score_diagnostics(fact, values, snap.policy(source, fact))
                                    for source, values in source_rows.items()}
        row['candidate_id'] = snap.candidate_id
        if fact.member:
            out.setdefault(fact.member[0], {"members": {}})["members"][fact.member[1]] = row
        else:
            out[fact.name] = row
    for name, report in out.items():
        if "members" in report:
            report |= selection_counts(sw, snap, plan.fact_index[name], deployed, items, report["members"])
    return out


def check_counts(own: list) -> dict:
    positives = [o for o, y in own if y == 0]
    accepted_true = sum(o == 0 for o, _ in own)
    correct_true = sum(o == 0 and y == 0 for o, y in own)
    accepted_false = sum(o == 1 for o, _ in own)
    return {"actual_positives": len(positives), "accepted_true": accepted_true, "accepted_true_correct": correct_true,
            "accepted_false": accepted_false, "accepted_false_correct": sum(o == 1 and y == 1 for o, y in own),
            "precision_true": correct_true / accepted_true if accepted_true else None,
            "recall_true": correct_true / len(positives) if positives else None,
            "silent_misses": sum(o == 1 for o in positives), "to_review": sum(o is None for o in positives)}


def confusion(fact, own: list) -> dict:
    table = {key: {} for key in fact.keys}
    for got, want in own:
        predicted = "abstain" if got is None else fact.keys[got]
        table[fact.keys[want]][predicted] = table[fact.keys[want]].get(predicted, 0) + 1
    return {"confusion": table, "per_label": {key: {"gold": sum(y == k for _, y in own), "accepted": sum(o == k for o, _ in own),
                                                    "correct": sum(o == k and y == k for o, y in own)}
                                              for k, key in enumerate(fact.keys)}}


def distances(own: list) -> dict:
    accepted = [abs(o - y) for o, y in own if o is not None]
    return {"index_distance": {str(d): accepted.count(d) for d in sorted(set(accepted))},
            "mean_index_distance": sum(accepted) / len(accepted) if accepted else None}


MICRO = ("labelled", "actual_positives", "accepted_true", "accepted_true_correct", "accepted_false", "accepted_false_correct",
         "silent_misses", "to_review")


def selection_counts(sw: Switch, snap: Snapshot, f: int, calls: list, items: list[dict], members: dict) -> dict:
    fact = snap.plan.facts[f]
    keys = {card.id for card in fact.cards}
    statuses = Counter()
    for row in members.values():
        statuses.update(row["statuses"])
    micro = {name: sum(row[name] for row in members.values()) for name in MICRO + COUNTS}
    full = [(call, item["facts"][fact.name]) for call, item in zip(calls, items)
            if call.facts and set(item.get("facts", {}).get(fact.name, {})) == keys]
    ids = [card.id for card in fact.cards]
    states = [(call.facts[f].selection, given) for call, given in full]
    settled = [({ids[k] for k in state.selected}, given) for state, given in states if state.settled]
    return {"labelled": micro["labelled"], **rates(micro), **micro, "statuses": dict(statuses),
            "precision_true": micro["accepted_true_correct"] / micro["accepted_true"] if micro["accepted_true"] else None,
            "recall_true": micro["accepted_true_correct"] / micro["actual_positives"] if micro["actual_positives"] else None,
            "full_gold": len(full), "full_gold_complete": sum(state.complete for state, _ in states), "full_gold_settled": len(settled),
            "exact_set_correct": sum(value == {key for key, v in given.items() if v != "unstated" and truth(v)} for value, given in settled)}


def fact_rows(facts: dict):
    for row in facts.values():
        yield from (row["members"].values() if "members" in row else (row,))


def contrasts(own: list, items: list[dict]) -> dict:
    from ..examples import metadata_history
    pairs: dict = {}
    groups: dict = {}
    for (got, want), it in zip(own, items):
        for pair in metadata_history(it.get("pairs", ()), it.get("pair"), "pair"):
            pairs.setdefault(pair, []).append(got == want)
        for group in metadata_history(it.get("groups", ()), it.get("group"), "group"):
            groups.setdefault(group, []).append((got, want))
    pairs = [v for v in pairs.values() if len(v) >= 2]
    same = [v for v in groups.values() if len(v) >= 2 and len({w for _, w in v}) == 1]
    return {"pairs": len(pairs), "pairs_both_correct": sum(all(v) for v in pairs),
            "invariant_groups": len(same), "invariant_consistent": sum(all(g is not None for g, _ in v) and len({g for g, _ in v}) == 1 for v in same)}


def run_test(rt, path, data: str | None = None, intervals: bool = False) -> dict:
    from .data import Labels
    from .dataset import EvidenceContext, evaluation_plan
    sw = Switch(rt, path)
    snap = sw.snapshot
    plan = snap.plan
    frozen = evaluation_plan({'candidate': snap.candidate_id}, scope=[list(plan.facts[f].path) for f in plan.observed]) if snap.candidate_id else None
    items, stale = test_items(sw.paths, plan, data)
    collector = Labels(sw, data)
    collector.historical_stages(snap)
    groups = {"taught": [], "used": [], "fresh": []}
    used_by: dict = {}
    for it in items:
        found = uses(snap, it["input"], collector)
        key = "taught" if taught(snap, it["input"]) else "used" if found else "fresh"
        groups[key].append(it)
        for u in found if key == "used" else ():
            used_by[u] = used_by.get(u, 0) + 1
    families = {name: [collector.group(('key', input_key(plan, it['input']))) for it in population] for name, population in groups.items()}
    exposure = EvidenceContext.of(collector).consult(frozen, families['fresh']) if frozen else None
    fact_populations = {name: evaluate_facts(sw, snap, population, families[name]) for name, population in groups.items()}
    report = {"switch": plan.name, "mode": f"lock {snap.lock_id}" if snap.locked else "draft",
              **{k: evaluate_cases(sw, snap, v, families[k]) for k, v in groups.items()}, "used_by": used_by,
              "facts": fact_populations["fresh"],
              "fact_populations": fact_populations,
              "warnings": list(stale) + list(collector.warnings), "dataset_id": collector.dataset_id,
              "candidate_id": snap.candidate_id, "population": "identified sample"}
    if exposure is not None:
        report['audit_use'] = exposure
        withhold([report['fresh'], *fact_rows(report['facts'])], exposure)
    if intervals:
        _add_intervals(report)
    return report


def withhold(rows, exposure):
    if not exposure['consulted_families']:
        return
    for row in rows:
        row['precision_95'] = None
        row['interval']['interval'] = None
        row['interval']['reason'] = f"check families {exposure['state']}: reused evidence, not an independent adoption claim"


def _add_intervals(report):
    fresh = report['fresh']
    fresh.setdefault('precision_95', None)
    for row in fact_rows(report['facts']):
        row.setdefault('precision_95', None)


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
    lines = [f"{report['switch']} ({report['mode']}), {total} case observations, {fact_total} fact observations:"]
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
        for row, call in zip(members, sw._score([row["bound_input"] for row in members], snap, sources, [f])):
            result = sw.public(snap, call, f)
            guess = result.prediction
            scored[row["id"]] = {"id": row["id"], **path_record(fact.path), "gold": fact.value(row["label"]),
                                 "correct": None if guess is None else (guess.key if guess.kind == "value" else guess.kind) == fact.keys[row["label"]],
                                 "accepted": guess is not None and result.status in ("ok", "none"), "result": result.as_dict(True)}
    return [scored[row["id"]] for row in rows]


def contrast_pairs(rows) -> dict:
    grouped = {}
    for row in rows:
        for pair in row.get("pairs", ()):
            grouped.setdefault((pair, *record_path(row)), []).append(row["id"])
    return grouped


def outcome(plan, r) -> tuple:
    selects = tuple((n, f.value, f.complete) for n, f in sorted(r.facts.items()) if f.complete is not None)
    if plan.cases:
        return r.status, r.answer, selects
    return r.status, tuple((n, f.value) for n, f in sorted(r.facts.items()) if f.settled), selects


def diff(rt, path) -> dict:
    locked = Switch(rt, path)
    if not locked.snapshot.locked:
        raise LockError("no lock file to compare against; run `ss lock` first")
    draft = Switch(rt, path, draft=True)
    items, _ = test_items(draft.paths, draft.plan)
    labels = {json.dumps(it["input"], sort_keys=True): it.get("case") for it in items}
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
    if "expect_case" in supplied and result.case != test.expect_case:
        failures.append(f"expected case {test.expect_case!r}, got {result.case!r}")
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
