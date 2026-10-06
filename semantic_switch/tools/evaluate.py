from __future__ import annotations

import json
from collections import Counter

from .. import files
from ..errors import ConfigError
from ..snapshot import Snapshot
from ..switch import Switch
from .common import auc, taught, test_items, trained_on
from .lock import LockError


def evaluate_cases(sw: Switch, snap: Snapshot, items: list[dict]) -> dict:
    labelled = [it for it in items if "case" in it]
    results = sw.run([it["input"] for it in labelled], snap) if labelled else []
    statuses = Counter(r.status for r in results)
    ok = [(r, it) for r, it in zip(results, labelled) if r.status == "ok"]
    return {"items": len(labelled), "ok": len(ok), "correct": sum(r.case == it["case"] for r, it in ok),
            "uncertain": statuses["uncertain"], "needs_input": statuses["needs_input"],
            "failed": statuses["failed"]}


def evaluate_facts(sw: Switch, snap: Snapshot, items: list[dict]) -> dict:
    plan = snap.plan
    calls = sw.score_facts([it["input"] for it in items], snap) if items else []
    out = {}
    for f, fact in enumerate(plan.facts):
        if fact.hidden or fact.cards:
            continue
        rows = [(c, it["facts"][fact.name]) for c, it in zip(calls, items)
                if fact.name in it.get("facts", {}) and c.facts and c.facts[f].dist is not None]
        if not rows:
            continue
        if fact.kind == "check":
            own = [(sw.exec._own(snap, c, f)[0], fact.index(y)) for c, y in rows]
            positives = [o for o, y in own if y == 0]
            out[fact.name] = {
                "labelled": len(rows), "actual_positives": len(positives),
                "accepted_true": sum(o == 0 for o, _ in own),
                "accepted_true_correct": sum(o == 0 and y == 0 for o, y in own),
                "silent_misses": sum(o == 1 for o in positives),
                "to_review": sum(o is None for o in positives),
                "auc": auc([c.facts[f].dist[0] for c, y in rows if fact.index(y) != 2],
                           [fact.index(y) == 0 for c, y in rows if fact.index(y) != 2]),
                "policy": {src: sides[fact.name] for src, sides in snap.sides.items() if fact.name in sides}}
        else:
            own = [(sw.exec._own(snap, c, f)[0], fact.index(y)) for c, y in rows]
            out[fact.name] = {"labelled": len(rows), "settled": sum(o is not None for o, _ in own),
                              "settled_correct": sum(o == y for o, y in own)}
    return out


def run_test(rt, path, data: str | None = None, intervals: bool = False) -> dict:
    sw = Switch(rt, path)
    snap = sw.snapshot
    plan = snap.plan
    items, stale = test_items(sw.paths, plan, data)
    groups = {"taught": [], "trained_on": [], "unseen": []}
    for it in items:
        key = "taught" if taught(snap, it["input"]) else "trained_on" if trained_on(snap, it["input"]) else "unseen"
        groups[key].append(it)
    report = {"switch": plan.name, "mode": f"lock {snap.lock_id}" if snap.locked else "draft",
              **{k: evaluate_cases(sw, snap, v) for k, v in groups.items()},
              "facts": evaluate_facts(sw, snap, groups["unseen"]), "warnings": list(stale)}
    if intervals:
        _add_intervals(report)
    return report


def _add_intervals(report: dict):
    try:
        from scipy.stats import beta
    except ImportError:
        raise ConfigError("--intervals needs scipy: pip install scipy") from None

    def ci(k: int, n: int) -> list[float] | None:
        if not n:
            return None
        return [round(float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, 3),
                round(float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0, 3)]

    unseen = report["unseen"]
    unseen["accuracy_95"] = ci(unseen["correct"], unseen["ok"])
    for f in report["facts"].values():
        if "accepted_true" in f:
            f["precision_95"] = ci(f["accepted_true_correct"], f["accepted_true"])
        else:
            f["precision_95"] = ci(f["settled_correct"], f["settled"])


def _pct(k: int, n: int) -> str:
    return f"{k} ({k / n:.0%})" if n else str(k)


def _ci(row: dict, key: str) -> str:
    return f" [95% interval {row[key][0]:.0%}–{row[key][1]:.0%}]" if row.get(key) else ""


def explain_test(report: dict) -> str:
    u, t, s = report["unseen"], report["taught"], report["trained_on"]
    total = u["items"] + t["items"] + s["items"]
    lines = [f"{report['switch']} ({report['mode']}), {total} labelled answers:"]
    if u["items"]:
        lines.append(f"  Unseen ({u['items']}): ok {_pct(u['ok'], u['items'])}: {u['correct']} correct, "
                     f"{u['ok'] - u['correct']} wrong{_ci(u, 'accuracy_95')}. {u['uncertain']} uncertain, "
                     f"{u['needs_input']} needs_input, {u['failed']} failed.")
    if s["items"]:
        lines.append(f"  Trained-on ({s['items']}): {s['correct']} correct of {s['ok']} settled. "
                     "(Used to tune or train; not a measure of generalisation.)")
    if t["items"]:
        verdict = "all as labelled" if t["correct"] == t["items"] else f"{t['correct']} as labelled"
        lines.append(f"  Exact-taught ({t['items']}): {t['ok']} settled, {verdict}. "
                     "(Checks that corrections stick, not understanding.)")
    accepted = []
    for name, f in report["facts"].items():
        if "accepted_true" in f:
            wrong = f["accepted_true"] - f["accepted_true_correct"]
            lines.append(f"  {name} (unseen): {f['actual_positives']} actually true — {f['accepted_true']} accepted as TRUE "
                         f"({'all correct' if not wrong else f'{wrong} wrong'}){_ci(f, 'precision_95')}, "
                         f"{f['silent_misses']} settled as FALSE (silent miss), {f['to_review']} left uncertain.")
            accepted.append(f["accepted_true"])
            if f.get("auc") is not None:
                lines.append(f"    {name}: the judge ranks true above false with AUC {f['auc']:.2f}"
                             + (" — its boundary, not its understanding, is off" if f["auc"] >= 0.9 and
                                f["silent_misses"] + f["accepted_true"] - f["accepted_true_correct"] > 0 else
                                " — it barely separates this claim; reword it or use --precise" if f["auc"] < 0.75
                                else "") + ".")
            for src, side in f.get("policy", {}).items():
                lines.append(f"    {name} {src} policy: " + "; ".join(
                    f"{k.upper()} side {s['state'].replace('_', ' ')}"
                    + (f" (held-out precision {s['held_out_precision']:.0%} on {s['held_out_accepted']}, "
                       f"95% interval {s['interval_95'][0]:.0%}–{s['interval_95'][1]:.0%})"
                       if "held_out_precision" in s else "") for k, s in sorted(side.items(), reverse=True)) + ".")
        else:
            lines.append(f"  {name} (unseen): settled {_pct(f['settled'], f['labelled'])} of {f['labelled']}, "
                         f"{f['settled_correct']} correct{_ci(f, 'precision_95')}.")
            accepted.append(f["settled"])
    if any(n < 60 for n in accepted) or 0 < u["ok"] < 60:
        lines.append("  Note: with fewer than ~60 accepted inputs, 95% and 90% precision can't be told apart.")
    lines += [f"  Warning: {w}" for w in report["warnings"]]
    return "\n".join(lines)


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
        k = json.dumps(d, sort_keys=True)
        if k not in seen:
            seen.add(k)
            inputs.append(d)
    old_plan, new_plan = locked.plan, draft.plan
    changes = [{"input": d, "old": outcome(old_plan, a), "new": outcome(new_plan, b),
                "label": labels.get(json.dumps(d, sort_keys=True))}
               for d, a, b in zip(inputs, locked.run(inputs), draft.run(inputs))
               if outcome(old_plan, a) != outcome(new_plan, b)]
    summary = Counter(f"{c['old'][1] or c['old'][0]} → {c['new'][1] or c['new'][0]}" for c in changes)
    return {"compared": len(inputs), "changed": len(changes), "summary": dict(summary), "changes": changes}
