from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..compiler import FactPlan
from ..judge import SCORE_SEMANTICS
from ..judgment import MIN_MASS, Policy, accept
from ..observations import Observation
from .data import Example

N_MIN = 20
STATES = {"met": "observed target met", "missed": "observed target missed", "insufficient": "insufficient evidence", "absent": "no independent audit in this artifact"}



def sides_of(fact: FactPlan) -> tuple[str, ...]:
    return ("true", "false") if fact.kind == "check" else ("answer",)


def side_rows(fact: FactPlan, scored: list) -> dict[str, list]:
    rows = {s: [] for s in sides_of(fact)}
    for dist, mass, pending, label in scored:
        if dist is None:
            for rows_ in rows.values():
                rows_.append((-1.0, False))
            continue
        ok = mass is None or mass >= MIN_MASS
        if fact.kind == "check":
            rows["true"].append((dist[0] if ok else -1.0, label == 0))
            rows["false"].append((dist[1] if ok else -1.0, label == 1))
        else:
            top = int(np.argmax(dist))
            rows["answer"].append((dist[top] if ok and top not in pending else -1.0, top == label))
    return rows


def candidates(rows):
    return sorted({float(np.nextafter(.5, 1.)), 1., *(float(score) for score, _ in rows if .5 < score <= 1)}, reverse=True)


def choose(rows, target):
    valid = sorted(((q, c) for q, c in rows if q > .5), reverse=True)
    best, covered, enough, position, correct = None, 0, False, 0, 0
    for theta in candidates(rows):
        while position < len(valid) and valid[position][0] >= theta:
            correct += valid[position][1]
            position += 1
        enough |= position >= N_MIN
        if position >= N_MIN and position > covered and correct / position >= target:
            best, covered = theta, position
    return enough, best


def coverage(rows: list, theta: float | None) -> float | None:
    return None if not rows else round(sum(theta is not None and q >= theta for q, _ in rows) / len(rows), 3)


def validate(held, target, theta, default):
    picked = [c for q, c in held if theta is not None and q >= theta]
    precision = sum(picked) / len(picked) if picked else None
    state = 'insufficient' if len(picked) < N_MIN else 'met' if precision >= target else 'missed'
    return {'theta': theta, 'state': state,
            'held_out_precision': precision, 'held_out_accepted': len(picked),
            'held_out_submitted': len(held), 'wrong_accepted': len(picked) - sum(picked),
            'wrong_acceptance_rate': (len(picked) - sum(picked)) / len(held) if held else None,
            'correct_accepted': sum(picked), 'coverage_default': coverage(held, default),
            'coverage_result': coverage(held, theta), 'population': 'identified sample',
            'interval_95': None, 'interval_reason': 'sampling independence not declared'}


def as_policy(fact: FactPlan, sides: dict) -> Policy:
    if fact.kind == "check":
        return Policy(sides["true"]["theta"], sides["false"]["theta"])
    return Policy(sides["answer"]["theta"], sides["answer"]["theta"])


def defaults(fact: FactPlan, policy: Policy) -> dict:
    return policy.thresholds(fact.kind)


def fit_report(fact: FactPlan, tune: list, default: Policy) -> dict:
    rows, base = side_rows(fact, tune), defaults(fact, default)
    groups = [getattr(row, "group", None) for row in tune]
    out = {}
    for side, values in rows.items():
        enough, theta = choose(values, fact.target)
        deployed = theta if enough else base[side]
        accepted = [i for i, (q, _) in enumerate(values) if deployed is not None and q >= deployed]
        correct = sum(values[i][1] for i in accepted)
        out[side] = {"theta": deployed, "status": "uncalibrated_default" if not enough else "fitted" if theta is not None else "disabled",
                     "submitted": len(values), "accepted": len(accepted), "correct": correct, "wrong": len(accepted) - correct,
                     "accepted_families": len({groups[i] for i in accepted}) if None not in groups else None,
                     "searched_thresholds": len(candidates(values)), "target": fact.target, "in_sample_coverage": coverage(values, deployed),
                     "n_min": N_MIN, "n_min_meaning": "heuristic fitting support floor, not a precision guarantee",
                     "population": "in-sample threshold fit; independent check is separate"}
    return out


def fit_policy(fact: FactPlan, tune: list, default: Policy) -> Policy:
    return as_policy(fact, fit_report(fact, tune, default))


def observed(example: Example, row, name: str) -> Observation:
    target = row.targets[name]
    return Observation(example.id, example.data, example.label, target.dist, target.mass, target.pending,
                       error="invalid/missing input" if row.error is not None or target.missing else target.error, group=example.group,
                       observed_models=target.observed_models)


def head_scores(fact: FactPlan, head, examples: list[Example], sw, snap) -> list:
    installed = snap if snap.heads.get(fact.name) is head else replace(snap, heads={**snap.heads, fact.name: head})
    rows = sw.score([e.data for e in examples], [fact.name], sources=("classifier",), snap=installed) if examples else []
    return [observed(e, row, fact.name) for e, row in zip(examples, rows)]


def development_curve(fact, observations):
    rows = side_rows(fact, observations)
    return {side: [{"threshold": theta, "submitted": len(values), "accepted": len(selected),
                   "correct": sum(selected), "wrong": len(selected) - sum(selected),
                   "coverage": len(selected) / len(values) if values else None,
                   "precision": sum(selected) / len(selected) if selected else None,
                   "population": "development; not independent audit"}
                  for theta in candidates(values)
                  for selected in [[correct for score, correct in values if score >= theta]]]
            for side, values in rows.items()}


def score_diagnostics(fact, observations, policy):
    valid = [o for o in observations if o.dist is not None]
    values = [float(max(o.dist)) for o in valid]
    accepted = [(accept(fact.judgment, o.dist, policy, o.mass, o.pending)[0], o.label) for o in valid]
    return {"score_kind": "acceptance_score" if any(o.mass is not None for o in valid) else "categorical_score",
            "score_semantics": SCORE_SEMANTICS,
            "submitted": len(observations), "valid": len(valid), "distinct": len(set(values)),
            "ties": len(values) - len(set(values)), "zero": sum(v == 0 for v in values),
            "one": sum(v == 1 for v in values), "at_least_0_999": sum(v >= .999 for v in values),
            "accepted": sum(index is not None for index, _ in accepted),
            "wrong_accepted": sum(index is not None and index != label for index, label in accepted),
            "thresholds": policy.thresholds(fact.kind),
            "per_class": [{"class": key, "distinct": len({float(o.dist[index]) for o in valid}),
                           "zero": int(sum(o.dist[index] == 0 for o in valid)), "one": int(sum(o.dist[index] == 1 for o in valid))}
                          for index, key in enumerate(fact.keys)]}
