from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
import json
from . import files
from .compiler import FactPlan, Plan, digest, is_digest
from .context import bind_input, canonical, lookup, render, project, json_value, json_default
from .results import thaw
from .errors import ConfigError
from .permissions import TrainingUse
from .selection import occurrences
from .kernel import MISSING, Exact, Sem, T

DATASET_SCHEMA = 7
LABEL_SCHEMA = 3
DERIVATIONS = ("computed", "judged", "authored", "unknown")
ACTORS = ("human", "agent", "unknown")
METHODS = ("procedure-recompute", "source-check", "review")
ROLES = {"fit": ("tune", (0, 1, 2)), "model_select": ("tune", (3,)), "acceptance_fit": ("tune", (4,)), "check": ("check", (0, 1, 2, 3, 4))}


def role_of(split, fold):
    return next((role for role, (where, folds) in ROLES.items() if where == split and fold in folds), None)


def validate_role(role):
    if role is not None and role not in ROLES:
        raise ConfigError(f"role must be one of {', '.join(ROLES)}, not {role!r}")
    return role


def read_fields(plan: Plan) -> tuple[str, ...]:
    used = {r for f in plan.facts for r in f.reads} | {a.field for a in plan.atoms if isinstance(a, Exact)}
    return tuple(n for n in plan.fields if n in used)


class Frame:
    __slots__ = ("plan", "values", "_full", "_projections", "_family", "_content")

    def __init__(self, plan: Plan, data):
        self.plan, self.values = plan, MappingProxyType(project(plan.fields, bind_input(plan.fields, data)))
        self._full = self._family = self._content = None
        self._projections = {}

    def __setattr__(self, name, value):
        if name in ("plan", "values") and hasattr(self, name):
            raise AttributeError("frame descriptors are immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def projected(cls, plan: Plan, values: Mapping) -> Frame:
        frame = object.__new__(cls)
        frame.plan, frame.values, frame._full, frame._family, frame._content, frame._projections = plan, json_value(values), None, None, None, {}
        return frame

    @property
    def full(self) -> str:
        if self._full is None:
            self._full = canonical(self.values, tuple(self.plan.fields))
        return self._full

    def projection(self, reads) -> str:
        reads = tuple(reads)
        found = self._projections.get(reads)
        if found is None:
            found = self._projections[reads] = canonical(partial(self.values, reads), reads)
        return found

    def read(self, fact: FactPlan) -> str:
        return self.projection(fact.context_reads or fact.reads)

    def observation(self, fact: FactPlan) -> str:
        return digest([list(fact.path), fact.meaning_rev, self.read(fact)])

    @property
    def family(self) -> str:
        if self._family is None:
            self._family = digest(self.projection(read_fields(self.plan)))
        return self._family

    @property
    def content(self) -> str:
        if self._content is None:
            self._content = canonical(self.values, tuple(sorted(self.values)))
        return self._content

    def record(self) -> dict:
        return thaw(self.values)


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
    groups: tuple = ()
    pairs: tuple = ()
    selections: tuple = ()
    sources: tuple = ()
    path: tuple = ()

    def record(self) -> dict:
        return {"id": self.id, **path_record(self.path), "meaning_rev": self.meaning_rev, "input": thaw(self.input),
                "value": self.value, "created": self.created,
                **({"groups": list(self.groups)} if self.groups else {}),
                **({"pairs": list(self.pairs)} if self.pairs else {}),
                "selections": [s.record() for s in self.selections], "label_sources": [source.record() for source in self.sources]}

    @property
    def claim(self) -> Claim:
        return Claim(self.value, self.sources)


@dataclass(frozen=True, slots=True)
class Actor:
    kind: str
    model: str | None = None
    session: str | None = None

    def __post_init__(self):
        if self.kind not in ACTORS:
            raise ConfigError(f"actor kind must be one of {', '.join(ACTORS)}")
        identified = all(isinstance(value, str) and value.strip() for value in (self.model, self.session))
        if (self.kind == "agent") != identified or self.kind != "agent" and (self.model is not None or self.session is not None):
            raise ConfigError("an agent actor records its model and session; a human or unknown actor records neither")

    def __str__(self) -> str:
        return self.kind if self.kind != "agent" else f"agent {self.model} session {self.session}"

    def record(self) -> dict:
        return {"kind": self.kind, **({"model": self.model, "session": self.session} if self.kind == "agent" else {})}

    @classmethod
    def parse(cls, value) -> Actor:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or "kind" not in value or set(value) - {"kind", "model", "session"}:
            raise ConfigError("an actor record is {kind[, model, session]}")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Verification:
    kind: str
    verifier: str
    actor: Actor
    method: str
    criterion: str
    content: str
    outcome: str

    def __post_init__(self):
        if self.kind not in ("label", "evidence") or self.outcome not in ("confirm", "replace", "skip"):
            raise ConfigError("verification kind must be label or evidence and outcome confirm, replace or skip")
        if self.kind == "evidence" and self.outcome == "replace":
            raise ConfigError("an evidence verification confirms or skips; replacing a value is a label decision")
        if not isinstance(self.verifier, str) or not self.verifier.strip() or not is_digest(self.content):
            raise ConfigError("verification needs a verifier identity and the verified content digest")
        if not isinstance(self.actor, Actor) or self.actor.kind not in ("human", "agent"):
            raise ConfigError("verification records its actor: human, or agent with its model and session")
        if self.method not in METHODS:
            raise ConfigError(f"verification method must be one of {', '.join(METHODS)}")
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise ConfigError("verification names the criterion or source revision it checked against")

    def record(self) -> dict:
        return {"kind": self.kind, "verifier": self.verifier, "actor": self.actor.record(), "method": self.method, "criterion": self.criterion,
                "content": self.content, "outcome": self.outcome}

    @classmethod
    def parse(cls, value) -> Verification:
        if isinstance(value, cls):
            return value
        fields = {"kind", "verifier", "actor", "method", "criterion", "content", "outcome"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ConfigError(f"a verification record needs exactly {sorted(fields)}")
        return cls(**{**value, "actor": Actor.parse(value["actor"])})


def _owned(value):
    return value if value is None or value is MISSING or isinstance(value, (str, bool, int, float)) else json_value(value)


SOURCE_FIELDS = frozenset({"derivation", "producer", "revision", "actor", "artifact", "address", "verifications", "replaces", "training_use", "resolves"})


@dataclass(frozen=True, slots=True, eq=False)
class LabelSource:
    derivation: str
    producer: str
    revision: str | None = None
    actor: Actor | None = None
    artifact: str | None = None
    address: tuple = ()
    verifications: tuple = ()
    replaces: object = MISSING
    training_use: TrainingUse | None = None
    resolves: tuple = ()
    identity: str = field(default="", init=False, repr=False)
    digest: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if self.derivation not in DERIVATIONS:
            raise ConfigError(f"label source derivation must be one of {', '.join(DERIVATIONS)}")
        if not isinstance(self.producer, str) or not self.producer.strip():
            raise ConfigError("label source needs a producer identity: procedure, judge route or author")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ConfigError("label source revision must be nonempty text")
        if self.derivation == "computed" and self.revision is None:
            raise ConfigError("a computed label names its procedure revision")
        if (self.derivation == "authored") != isinstance(self.actor, Actor) or self.actor is not None and not isinstance(self.actor, Actor):
            raise ConfigError("an authored label records its actor, and only an authored label carries one")
        if self.artifact is not None and not is_digest(self.artifact):
            raise ConfigError("label source artifact digest is a 64-hex sha256 value")
        if not isinstance(self.address, tuple) or any(not isinstance(part, str) or not part for part in self.address):
            raise ConfigError("label source address is a sequence of nonempty record/target names")
        if not isinstance(self.verifications, tuple) or any(not isinstance(entry, Verification) for entry in self.verifications):
            raise ConfigError("label source verifications must be verification entries")
        if self.replaces is not MISSING and self.derivation != "authored":
            raise ConfigError("only an authored label decision replaces a value")
        if self.training_use is not None and (not isinstance(self.training_use, TrainingUse) or self.derivation != "judged"):
            raise ConfigError("only a judged label source carries a training use declaration")
        if not isinstance(self.resolves, tuple) or any(not is_digest(claim) for claim in self.resolves) or len(set(self.resolves)) != len(self.resolves):
            raise ConfigError("label source resolves names distinct claim identities (64-hex sha256 values)")
        if self.resolves and (self.derivation != "authored" or self.replaces is not MISSING or len(self.resolves) < 2):
            raise ConfigError("only an authored decision resolves recorded alternatives: at least two of them, and it replaces no single value")
        object.__setattr__(self, "replaces", _owned(self.replaces))
        causal = self._causal()
        object.__setattr__(self, "identity", digest(causal))
        object.__setattr__(self, "digest", digest({**causal, **({"verifications": [v.record() for v in self.verifications]} if self.verifications else {})}))

    def _causal(self) -> dict:
        return {"derivation": self.derivation, "producer": self.producer, **({"revision": self.revision} if self.revision is not None else {}),
                **({"actor": self.actor.record()} if self.actor is not None else {}),
                **({"artifact": self.artifact} if self.artifact else {}), **({"address": list(self.address)} if self.address else {}),
                **({"replaces": self.replaces} if self.replaces is not MISSING else {}),
                **({"training_use": self.training_use.record()} if self.training_use is not None else {}),
                **({"resolves": list(self.resolves)} if self.resolves else {})}

    def __eq__(self, other) -> bool:
        return isinstance(other, LabelSource) and self.digest == other.digest

    def __hash__(self) -> int:
        return hash(self.digest)

    @classmethod
    def parse(cls, value) -> LabelSource:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) - SOURCE_FIELDS or not {"derivation", "producer"} <= set(value):
            raise ConfigError(f"label source needs derivation and producer; supported fields: {sorted(SOURCE_FIELDS)}")
        verifications = value.get("verifications", ())
        if not isinstance(verifications, (list, tuple)):
            raise ConfigError("label source verifications must be a list")
        address, resolves = value.get("address", ()), value.get("resolves", ())
        return cls(value["derivation"], value["producer"], value.get("revision"), None if value.get("actor") is None else Actor.parse(value["actor"]),
                   value.get("artifact"), tuple(address) if isinstance(address, (list, tuple)) else address, tuple(map(Verification.parse, verifications)),
                   value["replaces"] if "replaces" in value else MISSING, None if value.get("training_use") is None else TrainingUse.parse(value["training_use"]),
                   tuple(resolves) if isinstance(resolves, (list, tuple)) else resolves)

    def record(self) -> dict:
        record = self._causal()
        if self.verifications:
            record["verifications"] = [v.record() for v in self.verifications]
        return thaw(record)

    @property
    def boundary(self) -> bool:
        return self.replaces is not MISSING or bool(self.resolves)

    def verified(self, *entries: Verification) -> LabelSource:
        merged = tuple(dict.fromkeys((*self.verifications, *entries)))
        return self if merged == self.verifications else replace(self, verifications=merged)


UNKNOWN_SOURCE = LabelSource("unknown", "unknown")


def history_of(value) -> tuple:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError("each labelled target needs a nonempty ordered label source history")
    return tuple(LabelSource.parse(source) for source in value)


def merge_sources(*histories) -> tuple:
    merged = {}
    for history in histories:
        for source in history:
            found = merged.get(source.identity)
            merged[source.identity] = source if found is None else found.verified(*source.verifications)
    return tuple(merged.values())


def current(history) -> tuple:
    return tuple(history[max((i for i, source in enumerate(history) if source.boundary), default=0):])


def steps(history, value) -> tuple:
    resolved = max((i for i, source in enumerate(history) if source.resolves), default=0)
    out, after = [], value
    for at in reversed(range(len(history))):
        out.append(after if at >= resolved else MISSING)
        if history[at].replaces is not MISSING:
            after = history[at].replaces
    return tuple(reversed(out))


def derivation_summary(history) -> str:
    derivations = {source.derivation for source in current(history)}
    return derivations.pop() if len(derivations) == 1 else "mixed"


def _text(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, default=json_default)


def _same(left, right) -> bool:
    return left is right or _text(left) == _text(right)


class Claim:
    __slots__ = ("value", "history", "trajectory", "prefixes", "id", "key", "_size")

    def __init__(self, value, history):
        self.value, self.history = _owned(value), tuple(history)
        if not self.history:
            raise ConfigError("a label claim needs a nonempty source history")
        self.trajectory = tuple(zip((source.identity for source in self.history), steps(self.history, self.value)))
        chained, prefixes = "", []
        for identity, step in self.trajectory:
            chained = digest([chained, identity, None if step is MISSING else ["value", step]])
            prefixes.append(chained)
        self.prefixes, self.id = tuple(prefixes), prefixes[-1]
        self.key, self._size = digest([self.id, [source.digest for source in self.history]]), None

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("claim descriptors are immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def parse(cls, value, history) -> Claim:
        return cls(value, history_of(history))

    def __eq__(self, other) -> bool:
        return isinstance(other, Claim) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"Claim({self.value!r}, {self.id[:12]})"

    @property
    def size(self) -> int:
        if self._size is None:
            object.__setattr__(self, "_size", len(files._json(self.record()).encode("utf-8")))
        return self._size

    def record(self) -> dict:
        return {"claim": self.id, "value": thaw(self.value), "label_sources": [source.record() for source in self.history]}


@dataclass(frozen=True, slots=True)
class Resolution:
    value: object = MISSING
    history: tuple = ()
    alternatives: tuple = ()

    @property
    def resolved(self) -> bool:
        return self.value is not MISSING


def _ordered_chains(claims) -> list:
    return sorted(claims, key=lambda claim: (not any(s.boundary for s in claim.history), claim.id))


def _superseded(canon: dict) -> set:
    later = set()
    for claim in canon.values():
        later.update(claim.prefixes[:-1])
    edges = {}
    for claim in canon.values():
        identities = None
        for source in claim.history:
            for named in source.resolves:
                target = canon.get(named)
                if target is None or named == claim.id:
                    continue
                identities = identities or {s.identity for s in claim.history}
                if all(s.identity in identities for s in target.history):
                    edges.setdefault(claim.id, []).append(named)
    state = {}
    for start in edges:
        if start in state:
            continue
        stack = [(start, iter(edges.get(start, ())))]
        state[start] = 1
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                state[node] = 2
                stack.pop()
            elif state.get(child) == 1:
                raise ConfigError("recorded label resolutions form a cycle; the claim history is damaged")
            elif child not in state:
                state[child] = 1
                stack.append((child, iter(edges.get(child, ()))))
    return later | {named for found in edges.values() for named in found}


def resolve_claims(claims) -> Resolution:
    canon, extra = {}, {}
    for claim in claims:
        for source in claim.history:
            if source.verifications:
                extra.setdefault(source.identity, {}).update(dict.fromkeys(source.verifications))
        canon.setdefault(claim.id, claim)
    if not canon:
        return Resolution()
    superseded = _superseded(canon)
    terminals = _ordered_chains(claim for claim in canon.values() if claim.id not in superseded)
    unresolved = Resolution(alternatives=tuple(terminals))
    if len({_text(claim.value) for claim in terminals}) != 1:
        return unresolved
    entries = [frozenset(claim.trajectory) for claim in terminals]
    postings = defaultdict(set)
    for at, trajectory in enumerate(entries):
        for entry in trajectory:
            postings[entry].add(at)
    chains = []
    for at, trajectory in enumerate(entries):
        candidates = min((postings[entry] for entry in trajectory), key=len)
        if not any(other != at and trajectory <= entries[other] and (trajectory != entries[other] or other < at)
                   for other in candidates):
            chains.append(terminals[at])
    if len(chains) == 1:
        history = chains[0].history
    else:
        seen, history = set(), []
        for claim in chains:
            for source in claim.history:
                if source.identity not in seen:
                    seen.add(source.identity)
                    history.append(source)
        merged = set(Claim(chains[0].value, history).trajectory)
        if not all(entry in merged or entry[1] is MISSING for claim in chains for entry in claim.trajectory):
            return unresolved
    history = tuple(source.verified(*extra[source.identity]) if source.identity in extra else source for source in history)
    return Resolution(chains[0].value, history, tuple(terminals))


def _resolving(alternatives, decision: LabelSource) -> tuple:
    return (*merge_sources(*(claim.history for claim in alternatives)), replace(decision, resolves=tuple(sorted(claim.id for claim in alternatives))))


def successor(predecessors, value, decision: LabelSource) -> Claim:
    if decision.derivation != "authored" or decision.boundary:
        raise ConfigError("a correction is one new authored decision; its predecessors determine what it replaces or resolves")
    found = resolve_claims(predecessors)
    if found.resolved:
        return Claim(value, (*found.history, decision if _same(found.value, value) else replace(decision, replaces=found.value)))
    if found.alternatives:
        return Claim(value, _resolving(found.alternatives, decision))
    return Claim(value, (decision,))


ASSESSMENTS = ("verified", "computed", "authored", "judged", "unknown", "mixed", "disputed", "conflicting", "unassessed")
ASSESSMENT = 2


@dataclass(frozen=True, slots=True)
class Criteria:
    label: frozenset
    evidence: frozenset = frozenset()

    def accepts(self, entry: Verification) -> bool:
        return entry.criterion in (self.label if entry.kind == "label" else self.evidence)

    def record(self) -> dict:
        return {"revision": ASSESSMENT, "label": sorted(self.label), "evidence": sorted(self.evidence)}


@dataclass(frozen=True, slots=True)
class Assessment:
    state: str
    derivation: str
    procedures: tuple = ()
    current: tuple = ()
    stale: tuple = ()
    contrary: tuple = ()
    fidelity: str = "undeclared"
    criteria: Criteria | None = None

    @property
    def established(self) -> bool:
        return self.state in ("verified", "computed")

    def record(self) -> dict:
        return {"state": self.state, "derivation": self.derivation, "procedures": [list(p) for p in self.procedures], "fidelity": self.fidelity,
                "binding": None if self.criteria is None else self.criteria.record(),
                "verifications": {"current": [v.record() for v in self.current], "stale": len(self.stale), "contrary": [v.record() for v in self.contrary]}}


def _decided(fresh, contrary, otherwise) -> str:
    return "conflicting" if fresh and contrary else "disputed" if contrary else "verified" if fresh else otherwise


def assess(history, *, data, path, value, criteria: Criteria | None) -> Assessment:
    history = tuple(history)
    live = current(history)
    start, derivation = len(history) - len(live), derivation_summary(history)
    procedures = tuple(dict.fromkeys((s.producer, s.revision) for s in live if s.derivation == "computed"))
    every = tuple(entry for source in history for entry in source.verifications)
    if criteria is None:
        return Assessment("unassessed", derivation, procedures, stale=every)
    fresh, stale, contrary, frame, contents = [], [], [], None, {}
    for at, (source, answer) in enumerate(zip(history, steps(history, value))):
        for entry in source.verifications:
            if at < start or answer is MISSING or not criteria.accepts(entry):
                stale.append(entry)
                continue
            frame = canonical(data, tuple(sorted(data))) if frame is None else frame
            key = (entry.kind, _text(answer))
            if key not in contents:
                contents[key] = _content(entry.kind, frame, path, answer)
            if entry.content != contents[key]:
                stale.append(entry)
            else:
                (fresh if entry.outcome == "confirm" else contrary).append(entry)
    fidelity = _decided([e for e in fresh if e.kind == "evidence"], [e for e in contrary if e.kind == "evidence"], "unverified") if criteria.evidence else "undeclared"
    return Assessment(_decided(fresh, contrary, derivation), derivation, procedures, tuple(fresh), tuple(stale), tuple(contrary), fidelity, criteria)


def _content(kind: str, frame: str, path, value) -> str:
    return digest([frame, *([list(path), value] if kind == "label" else [])])


def reviewed_content(kind: str, data, path, value=None) -> str:
    return _content(kind, canonical(data, tuple(sorted(data))), path, value)


def path_record(path) -> dict:
    return {"fact": path[0], **({"member": path[1]} if len(path) > 1 else {})}


def record_path(record) -> tuple:
    return (record["fact"], *((record["member"],) if "member" in record else ()))


def labelled(plan: Plan, name: str, value) -> list[tuple[FactPlan, object]]:
    fact = plan.fact(name)
    if not fact.labels:
        return [(fact, value)]
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name}: label memberships as a mapping of label keys to values")
    return [(plan.at(name, key), member) for key, member in value.items()]


def flatten(plan: Plan, facts: Mapping) -> dict:
    return {fact.name: value for name, given in facts.items() for fact, value in labelled(plan, name, given)}


def revisions(plan: Plan, facts: Mapping) -> dict:
    return {name: {key: plan.at(name, key).meaning_rev for key in value} if plan.fact(name).labels else plan.fact(name).meaning_rev
            for name, value in facts.items()}


def partial(data, names: tuple) -> dict:
    return {n: v for n in names if (v := lookup(data, n)) is not MISSING}


def metadata(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or isinstance(value, str) and not value.strip():
        raise ConfigError(f"{name} must be a nonempty string or integer")
    return value


def metadata_values(values, name) -> tuple:
    if not isinstance(values, (list, tuple)):
        raise ConfigError(f"{name}s must be a sequence")
    return tuple(dict.fromkeys(metadata(value, name) for value in values))


EXAMPLE_FIELDS = frozenset({"id", "fact", "member", "meaning_rev", "input", "value", "created", "groups", "pairs", "selections", "label_sources"})


def make_example(plan: Plan, fact: FactPlan, data, value, created: str = "", *, groups=(), pairs=(), selections=(), sources=(UNKNOWN_SOURCE,)) -> Example:
    frame = data if isinstance(data, Frame) else Frame(plan, data)
    return _example(fact, frame, fact.value(fact.index(value)), created, metadata_values(groups, "group"), metadata_values(pairs, "pair"),
                    occurrences(selections), history_of(sources))


def _example(fact: FactPlan, frame: Frame, value, created, groups, pairs, selections, sources) -> Example:
    key = frame.read(fact)
    return Example("e:" + digest([fact.name, fact.meaning_rev, key, value])[:16], fact.name, fact.meaning_rev, frame.values, value, created, key,
                   render(frame.plan.fields, partial(frame.values, fact.context_reads or fact.reads), fact.reads), groups, pairs, selections, sources, fact.path)


def reduced(fact: FactPlan, examples) -> tuple[Example | None, Resolution]:
    examples = list(examples)
    found = resolve_claims(example.claim for example in examples)
    if not found.resolved:
        return None, found
    latest = examples[-1]
    return replace(latest, id="e:" + digest([fact.name, fact.meaning_rev, latest.key, found.value])[:16], value=found.value, sources=found.history,
                   groups=tuple(dict.fromkeys(g for e in examples for g in e.groups)), pairs=tuple(dict.fromkeys(p for e in examples for p in e.pairs)),
                   selections=tuple(dict.fromkeys(s for e in examples for s in e.selections))), found


def recorded(plan: Plan, records) -> tuple[dict, tuple]:
    by_fact = {f.name: {} for f in plan.facts}
    stale, removed = defaultdict(int), defaultdict(int)
    for fact in plan.facts:
        for data, value in fact.examples:
            e = make_example(plan, fact, data, value)
            by_fact[fact.name].setdefault(e.key, []).append(e)
    for rec in records:
        try:
            fact = plan.at(*record_path(rec))
        except ConfigError:
            removed[".".join(record_path(rec))] += 1
            continue
        if rec["meaning_rev"] != fact.meaning_rev:
            stale[fact.name] += 1
            continue
        if set(rec) - EXAMPLE_FIELDS or "label_sources" not in rec:
            raise ConfigError(f"{'.'.join(record_path(rec))}: unsupported correction record fields {sorted(set(rec) - EXAMPLE_FIELDS)} or no label_sources; "
                              f"supported: {sorted(EXAMPLE_FIELDS)}; record it again with ss correct")
        e = make_example(plan, fact, rec["input"], rec["value"], rec.get("created", ""), groups=rec.get("groups", ()), pairs=rec.get("pairs", ()),
                         selections=rec.get("selections", ()), sources=rec["label_sources"])
        by_fact[fact.name].setdefault(e.key, []).append(e)
    warnings = tuple(f"{n}: {c} stale examples ignored (its meaning changed)" for n, c in stale.items()) + removed_warnings(removed)
    return by_fact, warnings


def collect(plan: Plan, records) -> tuple[dict, tuple]:
    by_fact, warnings = recorded(plan, records)
    out, unresolved = {}, []
    for fact in plan.facts:
        kept = []
        for key, examples in by_fact[fact.name].items():
            example, _ = reduced(fact, examples)
            if example is None:
                unresolved.append(f"{'/'.join(fact.path)}: competing corrections for one input ({key[:60]}) have no recorded lineage; the override is withheld "
                                  "until ss correct records a resolution")
            else:
                kept.append(example)
        out[fact.name] = tuple(kept)
    return out, warnings + tuple(unresolved)


def removed_warnings(removed) -> tuple[str, ...]:
    return tuple(f"{n}: {c} labels for a removed fact ignored" for n, c in removed.items())


def overrides_of(examples: dict) -> dict:
    return {f: {e.key: e.value for e in exs} for f, exs in examples.items()}


def derive_fact_labels(plan: Plan, rule_id: str) -> dict:
    labels = {}
    for rule in plan.rules:
        hit = rule.id == rule_id
        if not hit and not plan.first:
            continue
        node = rule.expr
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


SITUATION_FIELDS = frozenset({"schema", "input", "groups", "pairs", "role", "tags", "selections"})
LABEL_FIELDS = SITUATION_FIELDS | {"facts", "revs", "label_sources", "rule", "switch_rev", "created"}


def validate_tags(tags) -> list:
    if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ConfigError("tags must be a list of nonempty strings")
    return sorted(set(tags))


def _metadata(groups=(), pairs=(), role=None, tags=(), selections=()) -> tuple:
    return (metadata_values(groups, "group"), metadata_values(pairs, "pair"), validate_role(role), tuple(validate_tags(tags)), occurrences(selections))


def _situation(frame: Frame, meta: tuple) -> dict:
    groups, pairs, role, tags, selections = meta
    return {"schema": LABEL_SCHEMA, "input": frame.record(), "groups": list(groups), "pairs": list(pairs), **({"role": role} if role is not None else {}),
            **({"tags": list(tags)} if tags else {}), "selections": [s.record() for s in selections]}


def situation_record(plan: Plan, data, *, groups=(), pairs=(), role=None, tags=(), selections=()) -> dict:
    return _situation(data if isinstance(data, Frame) else Frame(plan, data), _metadata(groups, pairs, role, tags, selections))


def targets(plan: Plan, facts: Mapping) -> list[tuple[FactPlan, object]]:
    return [pair for name, value in facts.items() for pair in labelled(plan, name, value)]


def _target_sources(plan: Plan, facts: Mapping, sources) -> dict:
    if not isinstance(sources, Mapping) or set(sources) != set(facts):
        raise ConfigError("label_sources must mirror facts: one source history per labelled fact or member")
    out = {}
    for name, value in facts.items():
        given = sources[name]
        if plan.fact(name).labels:
            if not isinstance(given, Mapping) or set(given) != set(value):
                raise ConfigError(f"{name}: label_sources must give one history per labelled member")
            out[name] = {key: history_of(history) for key, history in given.items()}
        else:
            out[name] = history_of(given)
    return out


def target_histories(record: Mapping) -> list[tuple]:
    return [history_of(history) for value in record.get("label_sources", {}).values()
            for history in (value.values() if isinstance(value, Mapping) else (value,))]


def target_assessments(record: Mapping, criteria: Mapping) -> dict:
    out = {}
    for name, value in record.get("label_sources", {}).items():
        for key, history in (value.items() if isinstance(value, Mapping) else ((None, value),)):
            path = (name,) if key is None else (name, key)
            answer = record["facts"][name] if key is None else record["facts"][name][key]
            out["/".join(path)] = assess(history_of(history), data=record["input"], path=path, value=answer, criteria=criteria.get("/".join(path)))
    return out


def criteria_of(plan: Plan, record: Mapping, review=None) -> dict:
    bindings = (review or {}).get("bindings") or {}
    declared = frozenset(bindings.get(reviewed_content("evidence", record["input"], ()), ())) if bindings else frozenset()
    return {"/".join(fact.path): Criteria(frozenset({fact.meaning_rev}) | declared, declared) for fact, _ in targets(plan, record.get("facts", {}))}


def assessments(plan: Plan, record: Mapping, review=None) -> dict:
    return target_assessments(record, criteria_of(plan, record, review))


def label_provenance(plan: Plan, records, *, review=None) -> dict:
    records = list(records)
    histories = [history for record in records for history in target_histories(record)]
    assessed = [a for record in records for a in assessments(plan, record, review).values()]
    verifications = [entry for history in histories for source in history for entry in source.verifications]
    return {"targets": len(assessed), "binding": {"revision": ASSESSMENT, "review": "declared" if (review or {}).get("bindings") else "undeclared"},
            "derivations": dict(sorted(Counter(map(derivation_summary, histories)).items())),
            "assessments": dict(sorted(Counter(a.state for a in assessed).items())), "established": sum(a.established for a in assessed),
            "verified": sum(a.state == "verified" for a in assessed), "stale": sum(bool(a.stale) for a in assessed),
            "conflicting": sum(a.state == "conflicting" for a in assessed), "fidelity": dict(sorted(Counter(a.fidelity for a in assessed).items())),
            "verifiers": sorted({f"{entry.verifier} ({entry.actor})" for entry in verifications}),
            "methods": dict(sorted(Counter(entry.method for entry in verifications).items()))}


def sources_at(sources: Mapping, path) -> tuple:
    value = sources[path[0]]
    return value[path[1]] if len(path) > 1 else value


def each_target(facts: Mapping, history) -> dict:
    return {name: {key: list(history) for key in value} if isinstance(value, Mapping) else list(history) for name, value in facts.items()}


def _claims(plan: Plan, frame: Frame, facts: Mapping, sources) -> dict:
    histories = _target_sources(plan, facts, sources if sources is not None else {})
    claims = {}
    for fact, member in targets(plan, facts):
        fact.index(member)
        history = sources_at(histories, fact.path)
        for source, value in zip(history, steps(history, member)):
            for entry in source.verifications:
                if value is not MISSING and entry.content != _content(entry.kind, frame.content, fact.path, value):
                    raise ConfigError(f"{'/'.join(fact.path)}: {entry.kind} verification by {entry.verifier} covers other content; verify the changed evidence or label again")
        claims[fact.path] = Claim(member, history)
    return claims


def _nested(plan: Plan, values: Mapping) -> dict:
    out = {}
    for path, value in values.items():
        if len(path) > 1:
            out.setdefault(path[0], {})[path[1]] = value
        else:
            out[path[0]] = value
    return out


def _label(plan: Plan, frame: Frame, meta: tuple, claims: Mapping, rule, created) -> dict:
    facts = _nested(plan, {path: claim.value for path, claim in claims.items()})
    revs = revisions(plan, facts)
    return {**_situation(frame, meta), "facts": facts, **({"rule": rule, "switch_rev": plan.switch_rev} if rule else {}), **({"revs": revs} if revs else {}),
            "label_sources": _nested(plan, {path: [s.record() for s in claim.history] for path, claim in claims.items()}), "created": created or files.now()}


def label_record(plan: Plan, data, facts: dict, rule: str | None = None, *, sources=None, groups=(), pairs=(), role=None, tags=(), selections=(),
                 created=None) -> dict:
    frame = data if isinstance(data, Frame) else Frame(plan, data)
    return _label(plan, frame, _metadata(groups, pairs, role, tags, selections), _claims(plan, frame, facts, sources), rule, created)


@dataclass(frozen=True, slots=True)
class Unit:
    frame: Frame
    role: str | None
    groups: frozenset
    pairs: frozenset
    tags: frozenset
    selections: frozenset
    created: str | None
    claims: Mapping

    @property
    def key(self) -> str:
        return self.frame.full

    @property
    def input(self) -> Mapping:
        return self.frame.values


def _unit(frame: Frame, meta: tuple, claims: Mapping, created) -> Unit:
    groups, pairs, role, tags, selections = meta
    return Unit(frame, role, frozenset(groups), frozenset(pairs), frozenset(tags), frozenset(selections), created,
                MappingProxyType({path: MappingProxyType({claim.key: claim}) for path, claim in claims.items()}))


def combine(left: Unit, right: Unit) -> Unit:
    if left.key != right.key:
        raise ConfigError("only envelopes for one decision input merge")
    if left.role != right.role:
        raise ConfigError(f"duplicate decision input with roles {left.role} and {right.role}; give it one role")
    created = [_available(c) for c in (left.created, right.created) if c]
    return Unit(left.frame, left.role, left.groups | right.groups, left.pairs | right.pairs, left.tags | right.tags, left.selections | right.selections,
                max(created).isoformat() if created else None,
                MappingProxyType({path: MappingProxyType({**left.claims.get(path, {}), **right.claims.get(path, {})}) for path in sorted({*left.claims, *right.claims})}))


def _available(value) -> datetime:
    try:
        moment = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise ConfigError(f"availability time {value!r} is not an ISO time with its UTC offset")
    return moment.astimezone(timezone.utc)


def _sorted_metadata(values) -> list:
    return sorted(values, key=_text)


def settle(plan: Plan, unit: Unit, *, strict: bool) -> tuple[dict, dict]:
    resolved, unresolved = {}, {}
    for path in sorted(unit.claims):
        found = resolve_claims(unit.claims[path].values())
        if found.resolved:
            resolved[path] = Claim(found.value, found.history)
        else:
            unresolved["/".join(path)] = found
    if unresolved and strict:
        raise ConfigError(f"{', '.join(unresolved)}: competing labels for one decision input and no recorded correction lineage orders them; review the "
                          "retained alternatives and record an explicit resolution instead of keeping the last one")
    meta = (tuple(_sorted_metadata(unit.groups)), tuple(_sorted_metadata(unit.pairs)), unit.role, tuple(sorted(unit.tags)),
            tuple(sorted(unit.selections, key=lambda s: _text(s.record()))))
    envelope = _label(plan, unit.frame, meta, resolved, None, unit.created) if resolved else _situation(unit.frame, meta)
    return envelope, unresolved


def merge_envelopes(plan: Plan, previous: Mapping, record: Mapping) -> dict:
    return settle(plan, combine(unit_of(plan, previous), unit_of(plan, record)), strict=True)[0]


class Accumulator:
    def __init__(self, plan: Plan, *, max_situations: int, max_claims: int, max_bytes: int):
        self._plan, self._units, self._limits, self._claims, self._bytes = plan, {}, (max_situations, max_claims, max_bytes), set(), 0

    @property
    def retained(self) -> tuple[int, int, int]:
        return len(self._units), len(self._claims), self._bytes

    def add(self, unit: Unit) -> str:
        situations, claims, size = self._limits
        key = unit.key
        previous = self._units.get(key)
        if previous is None and len(self._units) >= situations:
            raise ConfigError(f"situation {key[:60]}: more than {situations} situations exceed the acquisition budget; split the source or raise max_situations explicitly")
        fresh = {(key, path, claim.key): claim.size for path, found in unit.claims.items() for claim in found.values()}
        fresh = {name: n for name, n in fresh.items() if name not in self._claims}
        charge = sum(fresh.values()) + (len(key.encode()) if previous is None else 0)
        for name in ("groups", "pairs", "tags"):
            charge += sum(len(files._json(value).encode("utf-8")) for value in getattr(unit, name) - (getattr(previous, name) if previous else frozenset()))
        charge += sum(len(files._json(s.record()).encode("utf-8")) for s in unit.selections - (previous.selections if previous else frozenset()))
        if len(self._claims) + len(fresh) > claims or self._bytes + charge > size:
            raise ConfigError(f"situation {key[:60]}: retained inputs, label claims and metadata exceed the acquisition budget of {claims} claims and {size} bytes; "
                              "nothing was dropped: split the source or raise the budget explicitly")
        self._units[key] = unit if previous is None else combine(previous, unit)
        self._claims.update(fresh)
        self._bytes += charge
        return key

    def __len__(self) -> int:
        return len(self._units)

    def keys(self) -> list:
        return list(self._units)

    def settle(self, key: str, *, strict: bool) -> tuple[dict, dict]:
        return settle(self._plan, self._units[key], strict=strict)


def resolution(plan: Plan, data, path, alternatives, value, decision: LabelSource) -> dict:
    claims = sorted((alternative if isinstance(alternative, Claim) else Claim.parse(alternative["value"], alternative["label_sources"])
                     for alternative in alternatives), key=lambda claim: claim.id)
    if decision.derivation != "authored" or decision.boundary:
        raise ConfigError("an explicit resolution is one new authored decision")
    found = resolve_claims(claims)
    if len({claim.id for claim in claims}) < 2 or found.resolved:
        raise ConfigError(f"{'/'.join(path)}: an explicit resolution decides between at least two retained alternatives that no recorded lineage orders")
    fact = plan.at(*path)
    history = _resolving(found.alternatives, decision)
    facts, sources = ({fact.path[0]: {fact.path[1]: value}}, {fact.path[0]: {fact.path[1]: history}}) if fact.member else ({fact.name: value}, {fact.name: history})
    return label_record(plan, data, facts, sources=sources)


def current_scope(plan: Plan, item: Mapping) -> Mapping:
    if not plan.retired or not plan.retired & set(item["facts"]):
        return item
    keep = lambda mapping: {name: value for name, value in mapping.items() if name not in plan.retired} if isinstance(mapping, Mapping) else mapping
    return {**item, "facts": keep(item["facts"]), **{name: keep(item[name]) for name in ("revs", "label_sources") if name in item}}


def _decoded(plan: Plan, item, labelled: bool) -> tuple:
    if not isinstance(item, Mapping) or item.get("schema") != LABEL_SCHEMA:
        raise ConfigError(f"not a schema-{LABEL_SCHEMA} situation envelope; re-encode each line as "
                          f'{{"schema": {LABEL_SCHEMA}, "input": ..., "groups": [...], "pairs": [...], "role": ..., "tags": [...]}}'
                          + (' with facts and per-target label_sources' if labelled else ''))
    allowed = LABEL_FIELDS if labelled else SITUATION_FIELDS
    if set(item) - allowed:
        raise ConfigError(f"unsupported {'label' if labelled else 'situation'} fields {sorted(set(item) - allowed)}; supported: {sorted(allowed)}")
    if "input" not in item:
        raise ConfigError(f"{'label' if labelled else 'situation'} envelope needs input")
    for name in ("groups", "pairs", "selections"):
        if not isinstance(item.get(name, []), list):
            raise ConfigError(f"{name} must be a list")
    frame = Frame(plan, item["input"])
    meta = _metadata(item.get("groups", ()), item.get("pairs", ()), item.get("role"), item.get("tags", ()), item.get("selections", ()))
    if not labelled:
        return frame, meta, {}, None
    if "rule" in item:
        raise ConfigError("label files carry fact labels; record rule answers with ss correct --rule")
    if not isinstance(item.get("facts"), Mapping) or not item["facts"]:
        raise ConfigError("label records need a nonempty facts mapping")
    item = current_scope(plan, item)
    if not item["facts"]:
        raise ConfigError(f"label record labels only retired facts {sorted(plan.retired)}")
    hidden = [name for name in item["facts"] if plan.fact(name).hidden]
    if hidden:
        raise ConfigError(f"{hidden}: not visible facts; address members as {{fact: {{member: value}}}}")
    current = revisions(plan, item["facts"])
    given = item.get("revs", current)
    changed = [name for name, rev in current.items() if not isinstance(given, Mapping) or
               (any(given.get(name, {}).get(key) != value for key, value in rev.items()) if isinstance(rev, dict) and isinstance(given.get(name), Mapping) else given.get(name) != rev)]
    if changed:
        raise ConfigError(f"{', '.join(changed)}: meaning changed since labelling; review the values again with ss label")
    return frame, meta, _claims(plan, frame, item["facts"], item.get("label_sources")), item.get("created")


def decode(plan: Plan, item, *, labelled: bool) -> dict:
    frame, meta, claims, created = _decoded(plan, item, labelled)
    return _label(plan, frame, meta, claims, None, created) if labelled else _situation(frame, meta)


def decode_unit(plan: Plan, item, *, labelled: bool) -> Unit:
    frame, meta, claims, created = _decoded(plan, item, labelled)
    return _unit(frame, meta, claims, created)


def unit_of(plan: Plan, envelope: Mapping) -> Unit:
    return decode_unit(plan, envelope, labelled="facts" in envelope)


def addressed(plan: Plan, record: Mapping) -> list[tuple[str, tuple]]:
    return [("/".join(fact.path), history_of(sources_at(record["label_sources"], fact.path))) for fact, _ in targets(plan, record.get("facts", {}))]


def iter_units(plan: Plan, path, *, labelled: bool):
    for line, item in files.iter_records(path, with_lines=True):
        try:
            yield line, decode_unit(plan, item, labelled=labelled)
        except ConfigError as error:
            raise ConfigError(f"{path}:{line}: {error}") from None


def read_envelopes(plan: Plan, path, *, labelled: bool) -> list[tuple[int, dict]]:
    out = []
    for line, item in files.iter_records(path, with_lines=True):
        try:
            out.append((line, decode(plan, item, labelled=labelled)))
        except ConfigError as error:
            raise ConfigError(f"{path}:{line}: {error}") from None
    return out
