from __future__ import annotations

from collections import Counter
from pathlib import Path

from .. import files
from ..compiler import Plan, digest
from ..context import canonical
from ..examples import partial
from ..kernel import Exact
from ..snapshot import Snapshot


def read_fields(plan: Plan) -> tuple[str, ...]:
    used = {r for f in plan.facts for r in f.reads} | {a.field for a in plan.atoms if isinstance(a, Exact)}
    return tuple(n for n in plan.fields if n in used)


def input_key(plan: Plan, data) -> str:
    reads = read_fields(plan)
    return digest(canonical(partial(data, reads), reads))[:16]


def split_of(plan: Plan, item: dict) -> str:
    key = digest(["group", item["group"]]) if item.get("group") is not None else input_key(plan, item["input"])
    return "tune" if int(key[:8], 16) % 2 == 0 else "check"


def test_items(paths: files.Paths, plan: Plan, data: str | None = None) -> tuple[list[dict], tuple[str, ...]]:
    revs = {f.name: f.meaning_rev for f in plan.visible}
    items, stale = [], Counter()
    for it in files.read_jsonl(Path(data) if data else paths.test):
        if "case" in it and it.get("switch_rev", plan.switch_rev) != plan.switch_rev:
            it = {k: v for k, v in it.items() if k != "case"}
            stale["answer"] += 1
        old = it.get("revs", {})
        facts = {n: v for n, v in it.get("facts", {}).items() if old.get(n, revs.get(n)) == revs.get(n)}
        stale.update(n for n in it.get("facts", {}) if n not in facts)
        items.append({**it, "facts": facts})
    return items, tuple(f"{n}: {c} stale labels ignored (its meaning changed; relabel with `ss label --to test`)"
                        if n != "answer" else f"{c} stale answer labels ignored (the cases changed since they were labelled)"
                        for n, c in stale.items())


def taught(snap: Snapshot, data: dict) -> bool:
    return any(snap.overrides.get(f.name) and canonical(partial(data, f.reads), f.reads) in snap.overrides[f.name]
               for f in snap.plan.visible)


def trained_on(snap: Snapshot, data: dict) -> bool:
    return input_key(snap.plan, data) in snap.tuned_keys or any(
        canonical(partial(data, f.reads), f.reads) in snap.trained.get(f.name, ()) for f in snap.plan.visible)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)

