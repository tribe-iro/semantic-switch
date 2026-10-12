from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from . import files
from .compiler import FactPlan, Plan, digest
from .context import bind_input, canonical, lookup, render, project
from .results import freeze, thaw
from .errors import ConfigError
from .selection import history, occurrences
from .kernel import MISSING, Sem, T

DATASET_SCHEMA = 6
LABEL_SCHEMA = 2
ORIGINS = ("human", "rule", "synthetic", "unknown")
ROLES = {"fit": ("tune", (0, 1, 2)), "model_select": ("tune", (3,)), "acceptance_fit": ("tune", (4,)), "check": ("check", (0, 1, 2, 3, 4))}


def role_of(split, fold):
    return next((role for role, (where, folds) in ROLES.items() if where == split and fold in folds), None)


def validate_role(role):
    if role is not None and role not in ROLES:
        raise ConfigError(f"role must be one of {', '.join(ROLES)}, not {role!r}")
    return role


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
    group: str | None = None
    pair: str | None = None
    groups: tuple = ()
    pairs: tuple = ()
    selections: tuple = ()
    sources: tuple = ()
    path: tuple = ()

    def record(self) -> dict:
        return {"id": self.id, **path_record(self.path), "meaning_rev": self.meaning_rev, "input": thaw(self.input),
                "value": self.value, "created": self.created,
                **({"group": self.group} if self.group is not None else {}),
                **({"pair": self.pair} if self.pair is not None else {}),
                **({"groups": list(self.groups)} if self.groups else {}),
                **({"pairs": list(self.pairs)} if self.pairs else {}),
                "selections": [s.record() for s in self.selections], "label_sources": [source.record() for source in self.sources]}


@dataclass(frozen=True, slots=True)
class Review:
    kind: str
    reviewer: str
    content: str
    outcome: str

    def __post_init__(self):
        if self.kind not in ("label", "evidence") or self.outcome not in ("confirm", "replace", "skip"):
            raise ConfigError("review kind must be label or evidence and outcome confirm, replace or skip")
        if self.kind == "evidence" and self.outcome == "replace":
            raise ConfigError("an evidence review confirms or skips; replacing a value is a label decision")
        if not isinstance(self.reviewer, str) or not self.reviewer.strip() or not _digest(self.content):
            raise ConfigError("review needs a reviewer identity and the reviewed content digest")

    def record(self) -> dict:
        return {"kind": self.kind, "reviewer": self.reviewer, "content": self.content, "outcome": self.outcome}


@dataclass(frozen=True, slots=True)
class LabelSource:
    origin: str
    producer: str
    producer_digest: str | None = None
    artifact: str | None = None
    address: tuple = ()
    reviews: tuple = ()
    replaces: object = None

    def __post_init__(self):
        if self.origin not in ORIGINS:
            raise ConfigError(f"label source origin must be one of {', '.join(ORIGINS)}")
        if not isinstance(self.producer, str) or not self.producer.strip():
            raise ConfigError("label source needs a producer identity")
        if any(value is not None and not _digest(value) for value in (self.producer_digest, self.artifact)):
            raise ConfigError("label source producer and artifact digests are 64-hex sha256 values")
        if not isinstance(self.address, tuple) or any(not isinstance(part, str) or not part for part in self.address):
            raise ConfigError("label source address is a sequence of nonempty record/target names")
        if not isinstance(self.reviews, tuple) or any(not isinstance(review, Review) for review in self.reviews):
            raise ConfigError("label source reviews must be review entries")
        if self.replaces is not None and self.origin != "human":
            raise ConfigError("only a human label decision replaces a value")

    @classmethod
    def parse(cls, value) -> LabelSource:
        if isinstance(value, cls):
            return value
        fields = {"origin", "producer", "producer_digest", "artifact", "address", "reviews", "replaces"}
        if not isinstance(value, Mapping) or set(value) - fields or not {"origin", "producer"} <= set(value):
            raise ConfigError(f"label source needs origin and producer; supported fields: {sorted(fields)}")
        reviews = value.get("reviews", ())
        if not isinstance(reviews, (list, tuple)) or any(not isinstance(r, Mapping) or set(r) != {"kind", "reviewer", "content", "outcome"} for r in reviews):
            raise ConfigError("label source reviews need kind, reviewer, content and outcome")
        address = value.get("address", ())
        return cls(value["origin"], value["producer"], value.get("producer_digest"), value.get("artifact"),
                   tuple(address) if isinstance(address, (list, tuple)) else address, tuple(Review(**r) for r in reviews), value.get("replaces"))

    def record(self) -> dict:
        return {"origin": self.origin, "producer": self.producer, **({"producer_digest": self.producer_digest} if self.producer_digest else {}),
                **({"artifact": self.artifact} if self.artifact else {}), **({"address": list(self.address)} if self.address else {}),
                **({"reviews": [r.record() for r in self.reviews]} if self.reviews else {}),
                **({"replaces": self.replaces} if self.replaces is not None else {})}

    @property
    def digest(self) -> str:
        return digest(self.record())

    def reviewed(self, review: Review) -> LabelSource:
        return replace(self, reviews=(*self.reviews, review))


UNKNOWN_SOURCE = LabelSource("unknown", "unknown")


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def history_of(value) -> tuple:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError("each labelled target needs a nonempty ordered label source history")
    return tuple(LabelSource.parse(source) for source in value)


def merge_sources(*histories) -> tuple:
    return tuple({source.digest: source for history in histories for source in history}.values())


def origin_summary(history) -> str:
    current = history[max((i for i, source in enumerate(history) if source.replaces is not None), default=0):]
    origins = {source.origin for source in current}
    return origins.pop() if len(origins) == 1 else "mixed"


def reviewed_content(kind: str, data, path, value=None) -> str:
    return digest([canonical(data, tuple(sorted(data))), *([list(path), value] if kind == "label" else [])])


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
    if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int)) or
                              isinstance(value, str) and not value.strip()):
        raise ConfigError(f"{name} must be a nonempty string or integer")
    return value


def metadata_history(values, primary, name):
    if not isinstance(values, (list, tuple)):
        raise ConfigError(f"{name}s must be a sequence")
    return tuple(dict.fromkeys((*[metadata(value, name) for value in values], *((primary,) if primary is not None else ()))))


def make_example(plan: Plan, fact: FactPlan, data: dict, value, created: str = "", group=None, pair=None, groups=(), selection=None, selections=(), sources=(UNKNOWN_SOURCE,), pairs=()) -> Example:
    group, pair = metadata(group, "group"), metadata(pair, "pair")
    selected = history(selections, selection)
    sources = history_of(sources)
    value = fact.value(fact.index(value))
    data = project(plan.fields, bind_input(plan.fields, data))
    values = partial(data, fact.context_reads or fact.reads)
    key = canonical(values, fact.context_reads or fact.reads)
    return Example("e:" + digest([fact.name, fact.meaning_rev, key, value])[:16], fact.name, fact.meaning_rev, freeze(data),
                   value, created, key, render(plan.fields, values, fact.reads), group, pair, metadata_history(groups, group, "group"),
                   metadata_history(pairs, pair, "pair"), selected, sources, fact.path)


def collect(plan: Plan, records: list[dict]) -> tuple[dict, tuple]:
    by_fact = {f.name: {} for f in plan.facts}
    stale, removed = defaultdict(int), defaultdict(int)
    for fact in plan.facts:
        for data, value in fact.examples:
            e = make_example(plan, fact, data, value)
            by_fact[fact.name][e.key] = e
    for rec in records:
        try:
            fact = plan.at(*record_path(rec))
        except ConfigError:
            removed[".".join(record_path(rec))] += 1
            continue
        if rec["meaning_rev"] != fact.meaning_rev:
            stale[fact.name] += 1
            continue
        if "label_sources" not in rec:
            raise ConfigError(f"{'.'.join(record_path(rec))}: correction records without label_sources are obsolete; record them again with ss correct")
        e = make_example(plan, fact, rec["input"], rec["value"], rec.get("created", ""), rec.get("group"), rec.get("pair"), groups=rec.get("groups", ()), selection=rec.get("selection"),
                         selections=rec.get("selections", ()), sources=rec["label_sources"], pairs=rec.get("pairs", ()))
        previous = by_fact[fact.name].get(e.key)
        groups = tuple(dict.fromkeys((*getattr(previous, "groups", ()),
                                      *((previous.group,) if previous and previous.group is not None else ()),
                                      *rec.get("groups", ()), *((e.group,) if e.group is not None else ()))))
        by_fact[fact.name][e.key] = replace(e, groups=groups, pairs=tuple(dict.fromkeys((*getattr(previous, "pairs", ()), *e.pairs))), selections=tuple(dict.fromkeys(
            (*getattr(previous, "selections", ()), *e.selections))),
            sources=merge_sources(previous.sources, e.sources) if previous and previous.value == e.value else e.sources)
    warnings = tuple(f"{n}: {c} stale examples ignored (its meaning changed)" for n, c in stale.items()) + removed_warnings(removed)
    return {f: tuple(v.values()) for f, v in by_fact.items()}, warnings


def removed_warnings(removed) -> tuple[str, ...]:
    return tuple(f"{n}: {c} labels for a removed fact ignored" for n, c in removed.items())


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


SITUATION_FIELDS = frozenset({"schema", "input", "groups", "pairs", "role", "tags", "selections"})
LABEL_FIELDS = SITUATION_FIELDS | {"facts", "revs", "label_sources", "case", "switch_rev", "created"}


def validate_tags(tags) -> list:
    if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        raise ConfigError("tags must be a list of nonempty strings")
    return sorted(set(tags))


def situation_record(plan: Plan, data: dict, *, groups=(), pairs=(), role=None, tags=(), selections=()) -> dict:
    data = project(plan.fields, bind_input(plan.fields, data))
    return {"schema": LABEL_SCHEMA, "input": data, "groups": list(metadata_history(groups, None, "group")), "pairs": list(metadata_history(pairs, None, "pair")),
            **({"role": validate_role(role)} if role is not None else {}), **({"tags": validate_tags(tags)} if tags else {}),
            "selections": [s.record() for s in occurrences(selections)]}


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


def target_origins(record: Mapping) -> list[str]:
    return [origin_summary(history_of(history)) for value in record.get("label_sources", {}).values()
            for history in (value.values() if isinstance(value, Mapping) else (value,))]


def sources_at(sources: Mapping, path) -> tuple:
    value = sources[path[0]]
    return value[path[1]] if len(path) > 1 else value


def each_target(facts: Mapping, history) -> dict:
    return {name: {key: list(history) for key in value} if isinstance(value, Mapping) else list(history) for name, value in facts.items()}


def label_record(plan: Plan, data: dict, facts: dict, case: str | None = None, *, sources=None, groups=(), pairs=(), role=None, tags=(), selections=(),
                 created=None) -> dict:
    record = situation_record(plan, data, groups=groups, pairs=pairs, role=role, tags=tags, selections=selections)
    histories = _target_sources(plan, facts, sources if sources is not None else {})
    for fact, member in targets(plan, facts):
        fact.index(member)
        history = sources_at(histories, fact.path)
        for at, source in enumerate(history):
            value = next((later.replaces for later in history[at + 1:] if later.replaces is not None), member)
            for review in source.reviews:
                if review.content != reviewed_content(review.kind, record["input"], fact.path, value):
                    raise ConfigError(f"{'/'.join(fact.path)}: {review.kind} review by {review.reviewer} covers other content; review the changed evidence or label again")
    revs = revisions(plan, facts)
    return {**record, "facts": facts, **({"case": case, "switch_rev": plan.switch_rev} if case else {}), **({"revs": revs} if revs else {}),
            "label_sources": {name: {key: [s.record() for s in h] for key, h in value.items()} if isinstance(value, dict) else [s.record() for s in value]
                              for name, value in histories.items()},
            "created": created or files.now()}


def decode(plan: Plan, item, *, labelled: bool) -> dict:
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
    common = dict(groups=item.get("groups", ()), pairs=item.get("pairs", ()), role=item.get("role"), tags=item.get("tags", ()), selections=item.get("selections", ()))
    if not labelled:
        return situation_record(plan, item["input"], **common)
    if "case" in item:
        raise ConfigError("label files carry fact labels; record case answers with ss correct --case")
    if not isinstance(item.get("facts"), Mapping) or not item["facts"]:
        raise ConfigError("label records need a nonempty facts mapping")
    hidden = [name for name in item["facts"] if plan.fact(name).hidden]
    if hidden:
        raise ConfigError(f"{hidden}: not visible facts; address members as {{fact: {{member: value}}}}")
    current = revisions(plan, item["facts"])
    given = item.get("revs", current)
    changed = [name for name, rev in current.items() if not isinstance(given, Mapping) or
               (any(given.get(name, {}).get(key) != value for key, value in rev.items()) if isinstance(rev, dict) and isinstance(given.get(name), Mapping) else given.get(name) != rev)]
    if changed:
        raise ConfigError(f"{', '.join(changed)}: meaning changed since labelling; review the values again with ss label")
    return label_record(plan, item["input"], dict(item["facts"]), sources=item.get("label_sources"), created=item.get("created"), **common)


def read_envelopes(plan: Plan, path, *, labelled: bool) -> list[tuple[int, dict]]:
    out = []
    for line, item in files.iter_records(path, with_lines=True):
        try:
            out.append((line, decode(plan, item, labelled=labelled)))
        except ConfigError as error:
            raise ConfigError(f"{path}:{line}: {error}") from None
    return out
