from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .. import files
from ..compiler import FactPlan, digest
from ..context import canonical, project, render
from ..errors import ConfigError
from ..examples import partial
from ..heads import train_lexical
from ..judgment import OFF, Policy
from ..snapshot import Snapshot
from ..switch import Switch
from .common import input_key, split_of, test_items, wilson

GRID = [round(0.99 - 0.01 * k, 2) for k in range(49)]
N_MIN = 20
FOLDS = 5


class LockError(ConfigError):
    pass


@dataclass(frozen=True, slots=True)
class Row:
    data: dict
    label: int
    group: str
    text: str


def rows_for(sw: Switch, fact: FactPlan, examples, items: list[dict]) -> list[Row]:
    plan = sw.plan
    text = lambda d: render(plan.fields, project(plan.fields, sw.bind(d, sw.snapshot)), fact.reads)
    out = [Row(e.input, fact.index(e.value), e.key, e.rendered) for e in examples]
    out += [Row(it["input"], fact.index(it["facts"][fact.name]), str(it.get("group") or input_key(plan, it["input"])),
                text(it["input"])) for it in items if fact.name in it.get("facts", {})]
    return out


def folds_of(rows: list[Row], folds: int = FOLDS) -> list[int]:
    return [int(digest(["fold", r.group])[:8], 16) % folds for r in rows]


def side_rows(fact: FactPlan, dists: list, labels: list[int]) -> tuple[list, list]:
    if fact.kind == "check":
        return ([(d[0], y == 0) for d, y in zip(dists, labels)], [(d[1], y == 1) for d, y in zip(dists, labels)])
    return ([(max(d), int(np.argmax(d)) == y) for d, y in zip(dists, labels)],
            [(1 - d[v], y != v) for d, y in zip(dists, labels) for v in range(len(d))])


def choose(rows: list, target: float) -> tuple[bool, float | None]:
    if sum(q >= GRID[-1] for q, _ in rows) < N_MIN:
        return False, None
    best, covered = None, 0
    for t in GRID:
        picked = [c for q, c in rows if q >= t]
        if len(picked) > covered and len(picked) >= N_MIN and sum(picked) / len(picked) >= target:
            best, covered = t, len(picked)
    return True, best


def validate(choice: tuple[bool, float | None], held: list, target: float, default: float | None) -> dict:
    enough, theta = choice
    if not enough:
        return {"theta": default, "state": "untuned"}
    if theta is None:
        return {"theta": None, "state": "not_met", "reason": "no threshold met the target on the tune half"}
    picked = [c for q, c in held if q >= theta]
    if len(picked) < N_MIN:
        return {"theta": default, "state": "untuned", "candidate": theta, "held_out_accepted": len(picked)}
    precision = sum(picked) / len(picked)
    out = {"candidate": theta, "held_out_precision": round(precision, 3), "held_out_accepted": len(picked),
           "interval_95": wilson(sum(picked), len(picked))}
    return out | ({"theta": theta, "state": "met"} if precision >= target else {"theta": None, "state": "not_met"})


def fit_policy(fact: FactPlan, tune: tuple[list, list], default: Policy) -> Policy:
    picks = [choose(rows, fact.target) for rows in side_rows(fact, *tune)]
    return Policy(*(theta if enough else fallback
                    for (enough, theta), fallback in zip(picks, (default.t_true, default.t_false))))


def tune_policy(fact: FactPlan, tune: tuple[list, list], held: tuple[list, list], default: Policy) -> tuple[Policy, dict]:
    sides = {}
    for name, a, b, d in (("true", side_rows(fact, *tune)[0], side_rows(fact, *held)[0], default.t_true),
                          ("false", side_rows(fact, *tune)[1], side_rows(fact, *held)[1], default.t_false)):
        sides[name] = validate(choose(a, fact.target), b, fact.target, d)
    return Policy(sides["true"]["theta"], sides["false"]["theta"]), sides


def train_head(fact: FactPlan, rows: list[Row]):
    y = [r.label for r in rows]
    if len(set(y)) < 2:
        have = sorted({fact.keys[i] for i in y}) or ["none"]
        raise LockError(f"{fact.name}: specialize needs labels of at least two values; it has only {have}")
    return train_lexical([r.text for r in rows], y, fact.keys)


def out_of_fold(fact: FactPlan, rows: list[Row]) -> tuple[list, list]:
    fold = folds_of(rows)
    dists, labels = [], []
    for k in range(FOLDS):
        fit = [r for r, f in zip(rows, fold) if f != k]
        test = [r for r, f in zip(rows, fold) if f == k]
        if test and len({r.label for r in fit}) >= 2:
            head = train_lexical([r.text for r in fit], [r.label for r in fit], fact.keys)
            dists += [list(head.probs(r.text)) for r in test]
            labels += [r.label for r in test]
    return dists, labels


def judged(sw: Switch, snap: Snapshot, f: int, items: list[dict], fact: FactPlan) -> tuple[list, list]:
    labelled = [it for it in items if fact.name in it.get("facts", {})]
    calls = sw.score_facts([it["input"] for it in labelled], snap, sources=("judge",), facts=[f]) if labelled else []
    pairs = [(c.facts[f].dist, fact.index(it["facts"][fact.name])) for c, it in zip(calls, labelled)
             if c.facts and c.facts[f].dist is not None]
    return [d for d, _ in pairs], [y for _, y in pairs]


def build_lock(rt, path) -> tuple[dict, Snapshot, Switch]:
    sw = Switch(rt, path, draft=True)
    items, stale = test_items(sw.paths, sw.plan)
    snap = replace(sw.snapshot, warnings=stale)
    plan = snap.plan
    tune = [it for it in items if split_of(plan, it) == "tune"]
    check = [it for it in items if split_of(plan, it) == "check"]
    heads, policies, sides, trained = {}, {}, {}, {}
    for f, fact in enumerate(plan.facts):
        if fact.hidden or fact.cards:
            continue
        policy, side = tune_policy(fact, judged(sw, snap, f, tune, fact), judged(sw, snap, f, check, fact), fact.policy)
        policies.setdefault("judge", {})[fact.name] = policy
        sides.setdefault("judge", {})[fact.name] = side
        trained[fact.name] = {canonical(partial(it["input"], fact.reads), fact.reads)
                              for it in tune if fact.name in it.get("facts", {})}
        if fact.specialize:
            rows = rows_for(sw, fact, snap.examples[fact.name], tune)
            heads[fact.name] = head = train_head(fact, rows)
            held = rows_for(sw, fact, (), check)
            policy, side = tune_policy(fact, out_of_fold(fact, rows),
                                       ([list(head.probs(r.text)) for r in held], [r.label for r in held]), OFF)
            policies.setdefault("classifier", {})[fact.name] = policy
            sides.setdefault("classifier", {})[fact.name] = side
            trained[fact.name] |= {canonical(partial(r.data, fact.reads), fact.reads) for r in rows}
    snap = replace(snap, heads=heads, policies=policies, sides=sides, locked=True)
    failures = [f"expected {t.expect!r}, got {r.answer!r} ({r.status}) for {t.input}"
                for t in plan.tests for r in sw.run([t.input], snap) if r.answer != t.expect]
    if failures:
        raise LockError("tests failed against the frozen plan:\n" + "\n".join(failures))
    tuned_keys = sorted({input_key(plan, it["input"]) for it in tune})
    data = {
        "schema": 5, "switch": plan.name, "switch_rev": plan.switch_rev, "spec": plan.spec,
        "models": {"judge": rt.identity("judge")},
        "facts": {f.name: {
            "meaning_rev": f.meaning_rev,
            "examples": [e.record() for e in snap.examples[f.name]],
            "head": heads[f.name].to_json() if f.name in heads else None,
            "policies": {src: sides[src][f.name] for src in sides if f.name in sides[src]},
            "trained": sorted(trained.get(f.name, ())),
        } for f in plan.visible},
        "tuned_keys": tuned_keys,
        "tests": f"{len(plan.tests)}/{len(plan.tests)} passed against this plan",
    }
    data["id"] = digest(data)[:12]
    data["created"] = files.now()
    snap = replace(snap, lock_id=data["id"], tuned_keys=frozenset(tuned_keys),
                   trained={k: frozenset(v) for k, v in trained.items()})
    return data, snap, sw


def lock(rt, path, dry_run: bool = False) -> dict:
    data, snap, sw = build_lock(rt, path)
    if not dry_run:
        files.write_json_atomic(sw.paths.lock, data)
    policies = {n: f["policies"] for n, f in data["facts"].items() if f["policies"]}
    notes = [f"{n}: {src} {side} side {s['state'].replace('_', ' ')}"
             + (f" (held-out precision {s['held_out_precision']:.0%} on {s['held_out_accepted']})"
                if "held_out_precision" in s else "")
             for n, table in policies.items() for src, sd in table.items() for side, s in sd.items()
             if s["state"] != "met" and (src != "judge" or s["state"] == "not_met")]
    return {"lock": None if dry_run else str(sw.paths.lock), "id": data["id"], "tests": data["tests"],
            "heads": [n for n, f in data["facts"].items() if f["head"]],
            "tuned_inputs": len(data["tuned_keys"]), "policies": policies,
            "warnings": notes + list(snap.warnings)}
