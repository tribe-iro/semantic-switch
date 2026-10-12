from __future__ import annotations

from collections import Counter
from pathlib import Path

from .. import files
from ..compiler import Plan, digest
from ..errors import ConfigError
from ..context import bind_input, canonical, project
from ..examples import LABEL_SCHEMA, partial, removed_warnings
from ..kernel import Exact
from ..snapshot import Snapshot
from ..metrics import wilson


def read_fields(plan: Plan) -> tuple[str, ...]:
    used = {r for f in plan.facts for r in f.reads} | {a.field for a in plan.atoms if isinstance(a, Exact)}
    return tuple(n for n in plan.fields if n in used)


def input_key(plan: Plan, data) -> str:
    data = project(plan.fields, bind_input(plan.fields, data))
    reads = read_fields(plan)
    return digest(canonical(partial(data, reads), reads))


def current(plan: Plan, name: str, value, old: dict):
    fact = plan.fact(name)
    if not fact.labels:
        return value if old.get(name, fact.meaning_rev) == fact.meaning_rev else None
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: label memberships as a mapping of label keys to values")
    keys, given = {card.id for card in fact.cards}, old.get(name, {})
    kept = {key: v for key, v in value.items() if key in keys and given.get(key, plan.at(name, key).meaning_rev) == plan.at(name, key).meaning_rev}
    return kept or None


def test_items(paths: files.Paths, plan: Plan, data: str | None = None) -> tuple[list[dict], tuple[str, ...]]:
    visible = {f.name for f in plan.visible}
    items, stale, removed = [], Counter(), Counter()
    source = Path(data) if data else paths.test
    for line, it in files.iter_records(source, with_lines=True):
        if not isinstance(it, dict) or it.get("schema") != LABEL_SCHEMA:
            raise ConfigError(f"{source}:{line}: not a schema-{LABEL_SCHEMA} label envelope; re-encode it with per-target label_sources")
        if "case" in it and it.get("switch_rev", plan.switch_rev) != plan.switch_rev:
            it = {k: v for k, v in it.items() if k != "case"}
            stale["answer"] += 1
        old = it.get("revs", {})
        removed.update(n for n in it.get("facts", {}) if n not in visible)
        facts = {n: kept for n, v in it.get("facts", {}).items() if n in visible and (kept := current(plan, n, v, old)) is not None}
        stale.update(n for n, v in it.get("facts", {}).items() if n in visible and facts.get(n) != v)
        given = it.get("label_sources", {})
        try:
            kept_sources = {n: {k: given[n][k] for k in v} if isinstance(v, dict) else given[n] for n, v in facts.items()}
        except (KeyError, TypeError):
            raise ConfigError(f"{source}:{line}: label_sources must give a source history for every labelled fact and member") from None
        items.append({**it, "facts": facts, "label_sources": kept_sources})
    return items, tuple(f"{n}: {c} stale labels ignored (its meaning changed; relabel with `ss label --to test`)"
                        if n != "answer" else f"{c} stale answer labels ignored (the cases changed since they were labelled)"
                        for n, c in stale.items()) + removed_warnings(removed)


def taught(snap: Snapshot, data: dict) -> bool:
    data = project(snap.plan.fields, bind_input(snap.plan.fields, data))
    return any(snap.overrides.get(f.name) and canonical(partial(data, f.context_reads or f.reads), f.context_reads or f.reads) in snap.overrides[f.name]
               for f in map(snap.plan.facts.__getitem__, snap.plan.observed))


def uses(snap: Snapshot, data: dict, collector=None) -> tuple[str, ...]:
    data = project(snap.plan.fields, bind_input(snap.plan.fields, data))
    found = {use for f in map(snap.plan.facts.__getitem__, snap.plan.observed) for use, keys in snap.membership.get(f.name, {}).items()
             if canonical(partial(data, f.context_reads or f.reads), f.context_reads or f.reads) in keys}
    if collector is not None:
        found.update(collector.family_uses(data))
    if input_key(snap.plan, data) in snap.tuned_keys:
        found.add("tuned")
    return tuple(sorted(found))


def evidence_interval(rows):
    rows = list(rows)
    if not rows:
        return {'method': 'wilson', 'level': .95, 'interval': None, 'reason': 'no observations'}
    if any(family is None for family, _, _, _ in rows):
        return {'method': 'wilson', 'level': .95, 'interval': None, 'reason': 'no canonical family assignment'}
    if any(len(selections) != 1 for _, selections, _, _ in rows):
        return {'method': 'wilson', 'level': .95, 'interval': None, 'reason': 'unknown or ambiguous selection history'}
    acquisitions = [selections[0] for _, selections, _, _ in rows]
    frames = {s.frame_id for s in acquisitions}
    probabilities = {s.inclusion_probability for s in acquisitions}
    if len(frames) != 1 or None in frames or len(probabilities) != 1 or None in probabilities or any(s.selected_by != 'random' or not s.independent for s in acquisitions):
        return {'method': 'wilson', 'level': .95, 'interval': None, 'reason': 'independent uniform sampling frame not declared'}
    grouped = {}
    for family, _, accepted, correct in rows:
        grouped.setdefault(family, []).append((accepted, correct))
    if all(s.unit == 'observation' for s in acquisitions) and all(len(v) == 1 for v in grouped.values()):
        trials = [correct for _, _, accepted, correct in rows if accepted]
        metric, unit = 'accepted_precision', 'observation'
    elif all(s.unit == 'family' for s in acquisitions):
        trials = [all(correct for accepted, correct in values if accepted) for values in grouped.values() if any(accepted for accepted, _ in values)]
        metric, unit = 'all_accepted_correct', 'family'
    else:
        return {'method': 'wilson', 'level': .95, 'interval': None, 'reason': 'dependent observations; no binomial row interval'}
    k, n = sum(trials), len(trials)
    return {'method': 'wilson', 'level': .95, 'metric': metric, 'unit': unit, 'families': len(grouped), 'rows': len(rows),
            'selected_by': 'random', 'selection_ids': sorted({s.selection_id for s in acquisitions if s.selection_id is not None}),
            'windows': sorted({s.window for s in acquisitions if s.window is not None}),
            'inclusion_probability': next(iter(probabilities)), 'weighting': 'uniform',
            'frame_id': next(iter(frames)), 'successes': k, 'trials': n, 'interval': wilson(k, n)}
