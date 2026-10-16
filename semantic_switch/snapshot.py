from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping
import math

from .compiler import FactPlan, Plan, compile_spec, digest
from .results import freeze, thaw
from .errors import ConfigError
from .evidence import DATASET_SCHEMA, Example, collect, make_example, overrides_of, reduced
from .families import valid_seeds
from .heads import LexicalHead
from .judgment import CERTAIN, OFF, Policy
from .resident import REPRESENTATIONS


@dataclass(frozen=True, slots=True)
class Pending:
    models: Mapping
    execution: Mapping
    pinned: Mapping
    scopes: Mapping


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
    returned_models: Mapping = MappingProxyType({})
    candidate_id: str | None = None
    dataset_id: str | None = None
    lineage: Mapping = MappingProxyType({})
    framing: object = None
    pending: Pending | None = None

    def __post_init__(self):
        for name in ("examples", "overrides", "heads", "policies", "sides", "models", "membership", "family_seeds", "lineage", "returned_models"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def reconciled(self) -> bool:
        return self.pending is None

    def policy(self, source: str, fact: FactPlan) -> Policy:
        if source in ("example", "rule"):
            return CERTAIN
        table = self.policies.get(source, {})
        return table.get(fact.name, fact.policy if source == "judge" else OFF)

    def with_example(self, e: Example) -> Snapshot:
        fact = self.plan.fact_index[e.fact]
        previous = [example for example in self.examples[e.fact] if example.key == e.key]
        merged, _ = reduced(self.plan.facts[fact], (*previous, e))
        kept = tuple(x for x in self.examples[e.fact] if x.key != e.key) + ((merged,) if merged is not None else ())
        overrides = {key: value for key, value in self.overrides[e.fact].items() if key != e.key}
        if merged is not None:
            overrides[e.key] = merged.value
        warnings = self.warnings if merged is not None else self.warnings + (f"{'/'.join(e.path)}: competing corrections for one input have no recorded lineage; "
                                                                            "the override is withheld until ss correct records a resolution",)
        return replace(self, examples={**self.examples, e.fact: kept}, overrides={**self.overrides, e.fact: overrides}, warnings=warnings)


def draft(plan: Plan, records: list[dict]) -> Snapshot:
    examples, warnings = collect(plan, records)
    return Snapshot(plan, examples, overrides_of(examples), warnings=warnings)


def _same(locked: dict, current: dict) -> bool:
    from .api import behavior_identity
    return behavior_identity(locked) == behavior_identity(current)


def _execution(execution) -> str:
    return "unknown" if execution is None else ", ".join(f"{key}={execution[key]}" for key in sorted(execution) if key != "build")


def from_lock(data: dict, source: str) -> Snapshot:
    try:
        return _from_lock(data, source)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        raise ConfigError(f'{source}: malformed lock artifact: {error}') from None


LOCK_SCHEMA = 20


def lock_id(data: dict) -> str:
    return digest({key: value for key, value in data.items() if key not in ("id", "created")})[:12]


def _fact(source: str, plan: Plan, f: FactPlan, fd, tables: tuple):
    examples, heads, policies, sides, membership = tables
    if not isinstance(fd, dict):
        raise ConfigError(f"{source}: {f.shown}: malformed fact artifact")
    if fd.get("meaning_rev") != f.meaning_rev:
        raise ConfigError(f"{source}: {f.shown}: compiled meaning differs from the artifact")
    if not isinstance(fd.get("examples"), list) or any(not isinstance(record, dict) or
        not {"input", "value"} <= record.keys() for record in fd["examples"]):
        raise ConfigError(f"{source}: {f.shown}: invalid examples")
    examples[f.name] = tuple(make_example(plan, f, r["input"], r["value"], r.get("created", ""), groups=r.get("groups", ()), pairs=r.get("pairs", ()),
                                          selections=r.get("selections", ()), sources=r["label_sources"]) for r in fd["examples"])
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


def _envelope(data: dict, source: str):
    if not isinstance(data, dict) or data.get("schema") != LOCK_SCHEMA:
        raise ConfigError(f"{source}: unsupported lock schema {data.get('schema') if isinstance(data, dict) else None} "
                          f"(current {LOCK_SCHEMA}); regenerate this lock with ss lock")
    if not isinstance(data.get("spec"), dict) or not isinstance(data.get("id"), str):
        raise ConfigError(f"{source}: lock needs a spec object and artifact ID")
    if data.get("id") != lock_id(data):
        raise ConfigError(f"{source}: lock payload does not match its artifact ID")
    plan = compile_spec(thaw(data["spec"]), source)
    if not isinstance(data.get("facts"), dict) or set(data["facts"]) != {fact.name for fact in plan.visible}:
        raise ConfigError(f"{source}: lock fact set differs from its compiled definition")
    names = set(plan.bindings)
    if not isinstance(data.get("models"), dict) or set(data["models"]) != names or not all(isinstance(value, dict) for value in data["models"].values()):
        raise ConfigError(f"{source}: lock must pin one model identity per binding {sorted(names)}; regenerate this lock with ss lock")
    if not isinstance(data.get("execution"), dict) or set(data["execution"]) != names:
        raise ConfigError(f"{source}: lock is missing judge execution identity for its bindings {sorted(names)}; regenerate this lock with ss lock")
    scopes = {}
    for target, name in plan.targets.items():
        scopes.setdefault(name, []).append(target)
    pinned = data.get("bindings")
    if not isinstance(pinned, dict) or set(pinned) != names or any(not isinstance(record, dict) or record.get("scope") != sorted(scopes[name]) or
                                                                   record.get("representation") not in REPRESENTATIONS for name, record in pinned.items()):
        raise ConfigError(f"{source}: lock target-binding map differs from its compiled definition; regenerate this lock with ss lock")
    seeds = data.get("family_seeds")
    if not isinstance(seeds, dict) or any(not isinstance(stages, dict) or any(
        stage not in ("trained", "tuned", "validated", "judge") or not valid_seeds(nodes)
        for stage, nodes in stages.items())
        for stages in seeds.values()):
        raise ConfigError(f"{source}: invalid family seeds")
    if not isinstance(data.get('tuned_keys'), list) or not all(isinstance(key, str) for key in data['tuned_keys']):
        raise ConfigError(f'{source}: invalid tuned input membership')
    returned = data.get('returned_models')
    if not isinstance(returned, dict) or set(returned) != names or any(not isinstance(models, list) or len(models) > 1 or
                                                                       not all(isinstance(model, str) for model in models) for models in returned.values()):
        raise ConfigError(f'{source}: a lock must freeze at most one returned predictor identity per binding')
    return plan, names, scopes, pinned, seeds, returned


def _populated(data: dict, plan: Plan, source: str):
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
    return tables, dataset


def reconcile(snap: Snapshot, context, allow_model_change: bool, source: str) -> Snapshot:
    pending = snap.pending
    if pending is None:
        return snap
    names, scopes, pinned = sorted(pending.models), pending.scopes, pending.pinned
    policies = {src: dict(table) for src, table in snap.policies.items()}
    sides = {src: dict(table) for src, table in snap.sides.items()}
    warnings = []
    if snap.framing is not None:
        native = [judge for judge in (context.judge(name) for name in names) if hasattr(judge, "adapter")]
        if not native:
            raise ConfigError("pinned native framing requires a native judge; re-run `ss lock` with this judge")
        for judge in native:
            judge.adapter.validate_framing(snap.framing)
    models, returned, changed = dict(pending.models), dict(snap.returned_models), set()

    def reset(name, what):
        if not allow_model_change:
            raise ConfigError(f"{source}: {what}. Re-run `ss lock`, or load with allow_model_change=True.")
        for table in (policies, sides):
            for fact in scopes[name]:
                table.get("judge", {}).pop(fact, None)
            if not table.get("judge", True):
                del table["judge"]
        changed.add(name)
        warnings.append(f"{what}; its tuned judge policies were reset; predictor receipts are unverified until re-lock")

    for name in names:
        label = "judge" if name == "judge" else f"binding {name}"
        locked, current = pending.models[name], context.identity(name)
        if locked.get("presentation") != current.get("presentation"):
            raise ConfigError(f"{source}: evidence/choice presentation changed; re-run ss lock")
        representation = context.representation(name)
        if representation != pinned[name]["representation"]:
            reset(name, f"{label} representation changed: lock: {pinned[name]['representation']}, runtime has {representation}")
        elif not _same({k: v for k, v in locked.items() if k != "prompt" or snap.framing is None},
                       {k: v for k, v in current.items() if k != "prompt" or snap.framing is None}):
            differences = sorted(key for key in set(locked) | set(current)
                                 if key != "file" and locked.get(key) != current.get(key))
            reset(name, f"{label} identity changed ({', '.join(differences)}): lock: lock was made with {locked.get('file') or locked.get('source')} "
                        f"({locked.get('template', '-')}), runtime has {current.get('file') or current.get('source')} ({current.get('template', '-')})")
        models[name] = current if name in changed else locked
        tuned = any(fact in policies.get("judge", {}) for fact in scopes[name])
        execution = context.execution(name) if tuned else None
        if execution is not None:
            before = pending.execution[name]
            material = sorted(key for key in set(before or {}) | set(execution) if key != "build" and (before or {}).get(key) != execution.get(key))
            if material:
                reset(name, f"{label} execution changed ({', '.join(material)}): lock: {_execution(before)}, runtime has {_execution(execution)}")
            elif before != execution:
                warnings.append(f"{label} native build changed since lock; tuned judge policies kept, but their calibration has not been reverified on this build; re-run ss lock")
    if any(not returned[name] and context.identity(name).get('model') for name in names):
        warnings.append('hosted returned-model baseline is unverified')
    result = replace(snap, policies=policies, sides=sides, warnings=snap.warnings + tuple(warnings), models=MappingProxyType(models),
                     returned_models={name: () if name in changed else models_ for name, models_ in returned.items()}, pending=None)
    if changed:
        from .artifacts import snapshot_digest
        stale = {f"judge:{snap.plan.fact(fact).shown}" for name in changed for fact in scopes[name]}
        result = replace(result, lineage={key: value for key, value in result.lineage.items() if key not in stale})
        result = replace(result, candidate_id=snapshot_digest(result, context))
    return result


def _from_lock(data: dict, source: str) -> Snapshot:
    plan, names, scopes, pinned, seeds, returned = _envelope(data, source)
    tables, dataset = _populated(data, plan, source)
    examples, heads, policies, sides, membership = tables
    framing = None
    if data.get("framing") is not None:
        from .protocols import Framing
        framing = Framing.parse(data["framing"])
    return Snapshot(plan, examples, overrides_of(examples), heads, policies, sides, True, data["id"], (), MappingProxyType(dict(data["models"])),
                    frozenset(data["tuned_keys"]), MappingProxyType(membership), freeze(seeds), {name: tuple(models) for name, models in returned.items()},
                    data.get("candidate_id"), dataset["id"], data["lineage"], framing,
                    Pending(data["models"], data["execution"], pinned, {name: tuple(targets) for name, targets in scopes.items()}))
