from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from . import files
from .compiler import FactPlan, Plan, digest
from .context import canonical, lookup, render
from .kernel import MISSING, Sem, T


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    fact: str
    meaning_rev: str
    input: dict
    value: object
    created: str
    key: str
    rendered: str

    def record(self) -> dict:
        return {"id": self.id, "fact": self.fact, "meaning_rev": self.meaning_rev, "input": self.input,
                "value": self.value, "created": self.created}


def partial(data, names: tuple) -> dict:
    return {n: v for n in names if (v := lookup(data, n)) is not MISSING}


def make_example(plan: Plan, fact: FactPlan, data: dict, value, created: str = "") -> Example:
    value = fact.value(fact.index(value))
    values = partial(data, fact.reads)
    key = canonical(values, fact.reads)
    return Example("e:" + digest([fact.name, fact.meaning_rev, key, value])[:16], fact.name, fact.meaning_rev, data,
                   value, created, key, render(plan.fields, values, fact.reads))


def collect(plan: Plan, records: list[dict]) -> tuple[dict, tuple]:
    by_fact = {f.name: {} for f in plan.facts}
    stale = defaultdict(int)
    for fact in plan.facts:
        for data, value in fact.examples:
            e = make_example(plan, fact, data, value)
            by_fact[fact.name][e.key] = e
    for rec in records:
        if rec["fact"] not in plan.fact_index:
            continue
        fact = plan.fact(rec["fact"])
        if rec["meaning_rev"] != fact.meaning_rev:
            stale[fact.name] += 1
            continue
        e = make_example(plan, fact, rec["input"], rec["value"], rec.get("created", ""))
        by_fact[fact.name][e.key] = e
    warnings = tuple(f"{n}: {c} stale examples ignored (its meaning changed)" for n, c in stale.items())
    return {f: tuple(v.values()) for f, v in by_fact.items()}, warnings


def overrides_of(examples: dict) -> dict:
    return {f: {e.key: e.value for e in exs} for f, exs in examples.items()}


def derive_fact_labels(plan: Plan, case_id: str) -> dict:
    labels = {}
    for case in plan.cases:
        hit = case.id == case_id
        if not hit and not plan.first:
            continue
        node = case.expr
        children = [node] if node[0] == "atom" else list(node[1]) if node[0] == "and" and hit else []
        for child in children:
            if child[0] != "atom" or not isinstance(plan.atoms[child[1]], Sem):
                continue
            a = plan.atoms[child[1]]
            fact = plan.facts[a.fact]
            only = [k for k, t in enumerate(a.truth[:len(fact.keys)]) if t == T]
            if len(only) != 1:
                continue
            if hit:
                labels[fact.name] = fact.value(only[0])
            elif node[0] == "atom" and fact.kind == "check" and not fact.unstated and only == [0]:
                labels[fact.name] = False
        if hit:
            break
    return labels


def label_record(plan: Plan, data: dict, facts: dict, case: str | None = None) -> dict:
    revs = {n: plan.fact(n).meaning_rev for n in facts}
    return {"input": data, "facts": facts, **({"case": case, "switch_rev": plan.switch_rev} if case else {}),
            **({"revs": revs} if revs else {}), "created": files.now()}
