from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping
import math

from .compiler import FactPlan, Plan, compile_spec, digest
from .results import freeze, thaw
from .errors import ConfigError
from .examples import DATASET_SCHEMA, Example, collect, make_example, overrides_of
from .families import valid_seeds
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
    membership: Mapping = MappingProxyType({})

    family_seeds: Mapping = MappingProxyType({})
    returned_models: tuple = ()
    candidate_id: str | None = None
    dataset_id: str | None = None
    lineage: Mapping = MappingProxyType({})
    framing: object = None

    def __post_init__(self):
        for name in ("examples", "overrides", "heads", "policies", "sides", "models", "membership", "family_seeds", "lineage"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def policy(self, source: str, fact: FactPlan) -> Policy:
        if source in ("example", "rule"):
            return CERTAIN
        table = self.policies.get(source, {})
        return table.get(fact.name, fact.policy if source == "judge" else OFF)

    def with_example(self, e: Example) -> Snapshot:
        previous = next((example for example in self.examples[e.fact] if example.key == e.key), None)
        groups = tuple(dict.fromkeys((*getattr(previous, "groups", ()),
                                      *((previous.group,) if previous and previous.group is not None else ()),
                                      *e.groups, *((e.group,) if e.group is not None else ()))))
        e = replace(e, groups=groups, pairs=tuple(dict.fromkeys((*getattr(previous, "pairs", ()), *e.pairs))), selections=tuple(dict.fromkeys((*getattr(previous, "selections", ()), *e.selections))))
        kept = tuple(x for x in self.examples[e.fact] if x.key != e.key) + (e,)
        return replace(self, examples={**self.examples, e.fact: kept},
                       overrides={**self.overrides, e.fact: {**self.overrides[e.fact], e.key: e.value}})

    def with_examples(self, fact: str, kept: tuple, overrides: bool = True) -> Snapshot:
        table = {e.key: e.value for e in kept} if overrides else {}
        return replace(self, examples={**self.examples, fact: kept}, overrides={**self.overrides, fact: table})


def draft(plan: Plan, records: list[dict]) -> Snapshot:
    examples, warnings = collect(plan, records)
    if any(f.specialize for f in plan.facts):
        raise ConfigError("specialize: lexical no longer fits predictors; remove it, run ss export, semantic-trainer lexical EXPORT --to BUNDLE, then ss lock --head FACT=BUNDLE")
    return Snapshot(plan, examples, overrides_of(examples), warnings=warnings)


def _same(locked: dict, current: dict) -> bool:
    from .api import behavior_identity
    return behavior_identity(locked) == behavior_identity(current)


def _execution(execution) -> str:
    return "unknown" if execution is None else ", ".join(f"{key}={execution[key]}" for key in sorted(execution) if key != "build")


def from_lock(data: dict, identity, allow_model_change: bool, source: str) -> Snapshot:
    try:
        return _from_lock(data, identity, allow_model_change, source)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        raise ConfigError(f'{source}: malformed lock artifact: {error}') from None


LOCK_SCHEMA = 17


def _fact(source: str, plan: Plan, f: FactPlan, fd, tables: tuple):
    examples, heads, policies, sides, membership = tables
    if not isinstance(fd, dict):
        raise ConfigError(f"{source}: {f.shown}: malformed fact artifact")
    if fd.get("meaning_rev") != f.meaning_rev:
        raise ConfigError(f"{source}: {f.shown}: compiled meaning differs from the artifact")
    if not isinstance(fd.get("examples"), list) or any(not isinstance(record, dict) or
        not {"input", "value"} <= record.keys() for record in fd["examples"]):
        raise ConfigError(f"{source}: {f.shown}: invalid examples")
    examples[f.name] = tuple(make_example(plan, f, r["input"], r["value"], r.get("created", ""), r.get("group"), r.get("pair"), r.get("groups", ()), selections=r.get("selections", ()), sources=r["label_sources"], pairs=r.get("pairs", ()))
                             for r in fd["examples"])
    if fd.get('head') is not None and (not isinstance(fd['head'], dict) or not fd['head']):
        raise ConfigError(f'{source}: {f.shown}: invalid head payload')
    if not isinstance(fd.get('policies'), dict):
        raise ConfigError(f'{source}: {f.shown}: invalid policy table')
    if fd.get("head"):
        head = LexicalHead.from_json(fd["head"])
        if tuple(head.classes) != tuple(f.keys):
            raise ConfigError(f"{source}: {f.shown}: head classes differ from the fact")
        heads[f.name] = head
    for src, side in fd["policies"].items():
        if src not in ("judge", "classifier") or not isinstance(side, dict):
            raise ConfigError(f"{source}: {f.shown}: invalid policy source")
        names = ("true", "false") if f.kind == "check" else ("answer",)
        if set(side) != set(names) or src == 'classifier' and f.name not in heads:
            raise ConfigError(f'{source}: {f.shown}: incompatible policy sides/source')
        for name in names:
            record = side.get(name)
            if not isinstance(record, dict):
                raise ConfigError(f"{source}: {f.shown}: missing policy side {name}")
            for count in ('held_out_accepted', 'correct_accepted'):
                if count in record and (type(record[count]) is not int or record[count] < 0):
                    raise ConfigError(f'{source}: {f.shown}: invalid evidence count')
            for metric in ('held_out_precision', 'coverage_default', 'coverage_result'):
                value = record.get(metric)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ConfigError(f'{source}: {f.shown}: invalid evidence metric')
            interval = record.get('interval_95')
            if interval is not None and (not isinstance(interval, list) or len(interval) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in interval) or interval[0] > interval[1]):
                raise ConfigError(f'{source}: {f.shown}: invalid evidence interval')
            if record.get('correct_accepted', 0) > record.get('held_out_accepted', 0):
                raise ConfigError(f'{source}: {f.shown}: correct evidence exceeds accepted count')
            if record.get('state') == 'met' and (record.get('held_out_accepted', 0) < 20 or record.get('held_out_precision') is None):
                raise ConfigError(f'{source}: {f.shown}: unsupported met evidence')
            if 'theta' not in record:
                raise ConfigError(f'{source}: {f.shown}: missing policy threshold')
            theta = record.get("theta")
            if theta is not None and (isinstance(theta, bool) or not isinstance(theta, (int, float)) or not .5 < theta <= 1):
                raise ConfigError(f"{source}: {f.shown}: invalid policy threshold")
            if record.get("state") not in ("met", "missed", "insufficient", "absent"):
                raise ConfigError(f"{source}: {f.shown}: invalid policy evidence state")
            if (record.get("status"), record.get("calibration")) not in (("uncalibrated_default", "absent"), ("fitted", "in_sample"), ("disabled", "in_sample")) or \
                    record.get("fit") is not None and not isinstance(record["fit"], dict):
                raise ConfigError(f"{source}: {f.shown}: policy needs uncalibrated_default/absent or fitted|disabled/in_sample status and its fit diagnostics")
        policies.setdefault(src, {})[f.name] = Policy(*(side[s]["theta"] for s in (
            ("true", "false") if f.kind == "check" else ("answer", "answer"))))
        sides.setdefault(src, {})[f.name] = side
    member = fd.get("membership")
    if not isinstance(member, dict) or any(stage not in ("trained", "tuned", "validated", "judge") or
        not isinstance(keys, list) or not all(isinstance(key, str) for key in keys) for stage, keys in member.items()):
        raise ConfigError(f"{source}: {f.shown}: invalid membership")
    membership[f.name] = {use: frozenset(keys) for use, keys in member.items()}


def _from_lock(data: dict, identity, allow_model_change: bool, source: str) -> Snapshot:
    if not isinstance(data, dict) or data.get("schema") != LOCK_SCHEMA:
        raise ConfigError(f"{source}: unsupported lock schema {data.get('schema') if isinstance(data, dict) else None} "
                          f"(current {LOCK_SCHEMA}); regenerate this lock with ss lock")
    if not isinstance(data.get("spec"), dict) or not isinstance(data.get("id"), str):
        raise ConfigError(f"{source}: lock needs a spec object and artifact ID")
    if data.get("id") != digest({key: value for key, value in data.items() if key not in ("id", "created")})[:12]:
        raise ConfigError(f"{source}: lock payload does not match its artifact ID")
    plan = compile_spec(thaw(data["spec"]), source)
    if not isinstance(data.get("facts"), dict) or set(data["facts"]) != {fact.name for fact in plan.visible}:
        raise ConfigError(f"{source}: lock fact set differs from its compiled definition")
    if not isinstance(data.get("models"), dict) or not isinstance(data["models"].get("judge"), dict):
        raise ConfigError(f"{source}: lock is missing judge identity")
    if not isinstance(data.get("execution"), dict) or "judge" not in data["execution"]:
        raise ConfigError(f"{source}: lock is missing judge execution identity; regenerate this lock with ss lock")
    seeds = data.get("family_seeds")
    if not isinstance(seeds, dict) or any(not isinstance(stages, dict) or any(
        stage not in ("trained", "tuned", "validated", "judge") or not valid_seeds(nodes)
        for stage, nodes in stages.items())
        for stages in seeds.values()):
        raise ConfigError(f"{source}: invalid family seeds")
    if not isinstance(data.get('tuned_keys'), list) or not all(isinstance(key, str) for key in data['tuned_keys']):
        raise ConfigError(f'{source}: invalid tuned input membership')
    returned = data.get('returned_models')
    if not isinstance(returned, dict) or not isinstance(returned.get('judge'), list) or len(returned['judge']) > 1 or \
            not all(isinstance(model, str) for model in returned['judge']):
        raise ConfigError(f'{source}: a lock must freeze at most one returned judge identity')
    tables = examples, heads, policies, sides, membership = {}, {}, {}, {}, {}
    for f in plan.visible:
        fd = data["facts"][f.name]
        _fact(source, plan, f, fd, tables)
        if f.labels:
            members = fd.get("members")
            if not isinstance(members, dict) or set(members) != {card.id for card in f.cards}:
                raise ConfigError(f"{source}: {f.name}: lock label set differs from its compiled definition")
            for card in f.cards:
                _fact(source, plan, plan.facts[card.claims[0]], members[card.id], tables)
    for f in plan.facts:
        examples.setdefault(f.name, ())
    from .artifacts import candidate_digest
    if data.get('candidate_id') != candidate_digest(data):
        raise ConfigError(f'{source}: frozen candidate digest mismatch')
    dataset = data.get('dataset')
    if not isinstance(dataset, dict) or dataset.get('schema') != DATASET_SCHEMA or dataset.get('id') != digest({k: v for k, v in dataset.items() if k != 'id'}):
        raise ConfigError(f'{source}: embedded dataset is not a current schema-{DATASET_SCHEMA} materialization; regenerate this lock with ss lock')
    if not isinstance(data.get('lineage'), dict) or any(not isinstance(v, dict) or v.get('state') not in ('unknown', 'known_overlap', 'declared_disjoint') for v in data['lineage'].values()):
        raise ConfigError(f'{source}: missing lineage assessment')
    warnings = []
    framing = None
    if data.get("framing") is not None:
        from .protocols import Framing
        framing = Framing.parse(data["framing"])
        runtime = getattr(identity, "__self__", None)
        judge = runtime.judge() if runtime is not None else None
        if judge is None or not hasattr(judge, "adapter"):
            raise ConfigError("pinned native framing requires a native judge; re-run `ss lock` with this judge")
        judge.adapter.validate_framing(framing)
    locked, current = data["models"]["judge"], identity("judge")
    runtime_identity = current
    if locked.get("presentation") != current.get("presentation"):
        raise ConfigError(f"{source}: evidence/choice presentation changed; re-run ss lock")
    changed = not _same({k: v for k, v in locked.items() if k != "prompt" or framing is None},
                        {k: v for k, v in current.items() if k != "prompt" or framing is None})
    if changed:
        differences = sorted(key for key in set(locked) | set(current)
                             if key != "file" and locked.get(key) != current.get(key))
        what = f"judge identity changed ({', '.join(differences)}): lock: lock was made with {locked.get('file') or locked.get('source')} " \
               f"({locked.get('template', '-')}), runtime has {current.get('file') or current.get('source')} " \
               f"({current.get('template', '-')})"
        if not allow_model_change:
            raise ConfigError(f"{source}: {what}. Re-run `ss lock`, or load with allow_model_change=True.")
        policies.pop("judge", None)
        sides.pop("judge", None)
        warnings.append(f"{what}; its tuned judge policies were reset; predictor receipts are unverified until re-lock")
    runtime = getattr(identity, "__self__", None)
    execution = runtime.execution("judge") if runtime is not None and policies.get("judge") else None
    if execution is not None:
        before = data["execution"]["judge"]
        material = sorted(key for key in set(before or {}) | set(execution) if key != "build" and (before or {}).get(key) != execution.get(key))
        if material:
            what = f"judge execution changed ({', '.join(material)}): lock: {_execution(before)}, runtime has {_execution(execution)}"
            if not allow_model_change:
                raise ConfigError(f"{source}: {what}. Re-run `ss lock`, or load with allow_model_change=True.")
            changed = True
            policies.pop("judge", None)
            sides.pop("judge", None)
            warnings.append(f"{what}; its tuned judge policies were reset; predictor receipts are unverified until re-lock")
        elif before != execution:
            warnings.append("judge native build changed since lock; tuned judge policies kept, but their calibration has not been reverified on this build; re-run ss lock")
    result = Snapshot(plan, examples, overrides_of(examples), heads, policies, sides, True, data["id"],
                      tuple(warnings), MappingProxyType(dict(data["models"])), frozenset(data["tuned_keys"]),
                      MappingProxyType(membership), freeze(seeds), tuple(returned["judge"]), data.get("candidate_id"),
                      dataset["id"], data["lineage"], framing)
    if not result.returned_models and runtime_identity.get('model'):
        result = replace(result, warnings=result.warnings + ('hosted returned-model baseline is unverified',))
    if changed:
        from .artifacts import snapshot_digest
        result = replace(result, returned_models=(), models={**result.models, 'judge': runtime_identity},
                         lineage={key: value for key, value in result.lineage.items() if not key.startswith('judge:')})
        result = replace(result, candidate_id=snapshot_digest(result, runtime))
    return result
