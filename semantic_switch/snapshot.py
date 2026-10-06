from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from .compiler import FactPlan, Plan, compile_spec
from .errors import ConfigError
from .examples import Example, collect, make_example, overrides_of
from .heads import LexicalHead
from .judgment import CERTAIN, OFF, Policy


@dataclass(frozen=True, slots=True)
class Snapshot:
    plan: Plan
    examples: dict
    overrides: dict
    heads: dict = field(default_factory=dict)
    policies: dict = field(default_factory=dict)
    sides: dict = field(default_factory=dict)
    locked: bool = False
    lock_id: str | None = None
    warnings: tuple = ()
    models: Mapping = MappingProxyType({})
    tuned_keys: frozenset = frozenset()
    trained: Mapping = MappingProxyType({})

    def policy(self, source: str, fact: FactPlan) -> Policy:
        if source in ("example", "rule"):
            return CERTAIN
        table = self.policies.get(source, {})
        return table.get(fact.name, fact.policy if source == "judge" else OFF)

    def with_example(self, e: Example) -> Snapshot:
        kept = tuple(x for x in self.examples[e.fact] if x.key != e.key) + (e,)
        return replace(self, examples={**self.examples, e.fact: kept},
                       overrides={**self.overrides, e.fact: {**self.overrides[e.fact], e.key: e.value}})

    def with_examples(self, fact: str, kept: tuple, overrides: bool = True) -> Snapshot:
        table = {e.key: e.value for e in kept} if overrides else {}
        return replace(self, examples={**self.examples, fact: kept}, overrides={**self.overrides, fact: table})


def draft(plan: Plan, records: list[dict]) -> Snapshot:
    examples, warnings = collect(plan, records)
    warnings += tuple(f"{f.name}: specialize: lexical runs on the judge until `ss lock` trains its head"
                      for f in plan.facts if f.specialize)
    return Snapshot(plan, examples, overrides_of(examples), warnings=warnings)


def _same(locked: dict, current: dict) -> bool:
    return all(locked.get(k) == current.get(k) for k in ("sha256", "source", "template"))


def from_lock(data: dict, identity, allow_model_change: bool, source: str) -> Snapshot:
    plan = compile_spec(data["spec"], source)
    examples, heads, policies, sides, trained = {}, {}, {}, {}, {}
    for f in plan.visible:
        fd = data["facts"][f.name]
        examples[f.name] = tuple(make_example(plan, f, r["input"], r["value"], r.get("created", ""))
                                 for r in fd["examples"])
        if fd.get("head"):
            heads[f.name] = LexicalHead.from_json(fd["head"])
        for src, side in fd.get("policies", {}).items():
            policies.setdefault(src, {})[f.name] = Policy(side["true"]["theta"], side["false"]["theta"])
            sides.setdefault(src, {})[f.name] = side
        trained[f.name] = frozenset(fd.get("trained", ()))
    for f in plan.facts:
        examples.setdefault(f.name, ())
    warnings = []
    locked, current = data["models"]["judge"], identity("judge")
    if not _same(locked, current):
        what = f"judge: lock was made with {locked.get('file') or locked.get('source')} " \
               f"({locked.get('template', '-')}), runtime has {current.get('file') or current.get('source')} " \
               f"({current.get('template', '-')})"
        if not allow_model_change:
            raise ConfigError(f"{source}: {what}. Re-run `ss lock`, or load with allow_model_change=True.")
        policies.pop("judge", None)
        sides.pop("judge", None)
        warnings.append(f"{what}; its tuned judge policies were reset")
    return Snapshot(plan, examples, overrides_of(examples), heads, policies, sides, True, data["id"],
                    tuple(warnings), MappingProxyType(dict(data["models"])), frozenset(data.get("tuned_keys", ())),
                    MappingProxyType(trained))
