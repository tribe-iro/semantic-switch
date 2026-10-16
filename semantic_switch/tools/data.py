from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .. import files
from ..families import FamilyIndex
from ..selection import occurrences

from ..compiler import digest
from ..context import canonical, render
from ..errors import ConfigError
from ..examples import Frame, derivation_summary, history_of, merge_sources, sources_at, targets, validate_role, validate_tags
from ..switch import Switch
from .common import input_key, test_items

FOLDS = 5


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    fact: str
    data: dict
    label: int
    projection: str
    group: str
    split: str
    taught: bool
    text: str
    fold: int = -1
    selections: tuple = ()
    sources: tuple = ()
    pairs: tuple = ()
    path: tuple = ()
    tags: tuple = ()
    situations: tuple = ()

    @property
    def label_derivation(self) -> str:
        return derivation_summary(self.sources)


def merged_sources(entry) -> tuple:
    return merge_sources(*sorted(entry["histories"], key=lambda history: digest([source.record() for source in history])))


def observation_id(fact, projection: str) -> str:
    return digest([list(fact.path), fact.meaning_rev, projection])


def task_identity(plan) -> str:
    return digest({fact.name: fact.meaning_rev for fact in plan.facts})


def situation_id(plan, values) -> str:
    return digest([canonical(values, tuple(plan.fields)), task_identity(plan)])


class Labels:
    def __init__(self, sw: Switch, data: str | None = None):
        from .dataset import source_digests
        data = None if data is not None and Path(data).resolve() == sw.paths.test.resolve() else data
        self.alternate = data is not None
        with files.locked(sw.paths.dataset):
            previous = files.read_published(sw.paths.dataset)
            for attempt in range(3):
                sources = source_digests(sw, data)
                self._collect(sw, data, previous, sources)
                if sources == source_digests(sw, data):
                    if data is None and self.manifest != previous:
                        files.publish(sw.paths.dataset, self.manifest)
                        if previous is None and not sw.paths.audit.exists():
                            files.write_new(sw.paths.audit, '')
                    break
            else:
                raise ConfigError("dataset: source files changed repeatedly during materialization")

    def _collect(self, sw, data, previous, sources):
        self.sw = sw
        plan = sw.plan
        items, self.warnings = test_items(sw.paths, plan, data)
        self.items = items
        self.families = FamilyIndex()
        self.warnings = tuple(self.warnings)
        from .dataset import restore, materialize
        restore(self.families, previous)
        from ..examples import collect
        current_examples, _ = collect(plan, files.read_jsonl(sw.paths.examples))
        records = [(Frame.projected(plan, e.input), e.fact, e.value, group, True, e.selections, e.sources, (), None) for exs in current_examples.values() for e in exs
                   for group in (e.groups or (None,))
                   if plan.fact_index[e.fact] in plan.observed]
        self.situations = {}
        pair_memberships = {}
        for examples in current_examples.values():
            for example in examples:
                key = ("key", Frame.projected(plan, example.input).family)
                pair_memberships.setdefault(key, set()).update(map(str, example.pairs))
                for pair in example.pairs:
                    self._join(key, ("pair", str(pair)))
        for it in items:
            frame = Frame(plan, it["input"])
            key = ("key", frame.family)
            self.families.add(key)
            pair_memberships.setdefault(key, set()).update(map(str, it.get("pairs", ())))
            for kind, names in (("pair", it.get("pairs", ())), ("group", it.get("groups", ()))):
                for name in names:
                    self._join(key, (kind, str(name)))
            if not it.get("facts"):
                continue
            situation = situation_id(plan, frame.values)
            self._join(key, ("situation", situation))
            entry = self.situations.setdefault(situation, {"records": set(), "input": it["input"], "labels": {}})
            entry["records"].add(digest({k: v for k, v in it.items() if k != "created"}))
            selected = occurrences(it.get("selections", ()))
            for fact, value in targets(plan, it["facts"]):
                records += [(frame, fact.name, value, group, False, selected, history_of(sources_at(it["label_sources"], fact.path)),
                             validate_tags(it.get("tags", ())), situation) for group in (it.get("groups") or [None])]
        self._roles = [(("key", input_key(plan, it["input"])), role) for it in items if (role := validate_role(it.get("role"))) is not None]
        merged: dict = {}
        for frame, name, value, group, taught, selected, provenance, tags, situation in records:
            fact = plan.fact(name)
            if fact.cards:
                continue
            projection = frame.read(fact)
            label = fact.index(value)
            entry = merged.get((name, projection))
            if entry is None:
                entry = merged[(name, projection)] = {"data": frame.record(), "label": label, "taught": False, "text": render(plan.fields, frame.values, fact.reads),
                                                      "nodes": [("observation", observation_id(fact, projection))], "selections": set(), "histories": set(),
                                                      "tags": set(), "situations": set()}
            if entry["label"] != label:
                raise ConfigError(f"{name}: conflicting labels {' and '.join(sorted(map(repr, (fact.value(entry['label']), fact.value(label)))))} "
                                  f"for the same input {projection}")
            entry["histories"].add(provenance)
            if situation is not None and plan.fact_index[name] in plan.observed:
                entry["situations"].add(situation)
                self.situations[situation]["labels"][fact.path] = observation_id(fact, projection)
            entry["selections"].update(selected)
            entry["taught"] |= taught
            entry["tags"].update(tags)
            entry["nodes"].append(("key", frame.family))
            if group is not None:
                entry["nodes"].append(("group", str(group)))
        for (name, _), entry in merged.items():
            entry["selections"] = sorted(entry["selections"], key=lambda s: json.dumps(s.record(), sort_keys=True))
            for node in (*entry["nodes"][1:], ("rendered", digest([name, entry["text"]]))):
                self._join(entry["nodes"][0], node)
        self._merged = merged
        materialize(self, previous, sources)
        self.by_fact: dict[str, list[Example]] = {}
        for (name, projection), entry in merged.items():
            group = self.group(entry["nodes"][0])
            self.by_fact.setdefault(name, []).append(Example(
                observation_id(plan.fact(name), projection), name, entry["data"], entry["label"], projection, group,
                self.manifest["families"][group]["split"], entry["taught"], entry["text"],
                self.manifest["families"][group]["fold"], tuple(entry["selections"]), merged_sources(entry),
                tuple(sorted({pair for node in entry["nodes"] for pair in pair_memberships.get(node, ())})), plan.fact(name).path, tuple(sorted(entry["tags"])),
                tuple(sorted(entry["situations"]))))

    @property
    def path(self) -> Path:
        return self.sw.paths.yaml.with_suffix(f".dataset.{self.dataset_id}.json") if self.alternate else self.sw.paths.dataset

    def committed(self) -> Path:
        if self.alternate:
            with files.locked(self.path):
                if files.read_published(self.path) != self.manifest:
                    files.publish(self.path, self.manifest)
        return self.path

    def family(self, node) -> str | None:
        node = tuple(node)
        return self.families.name(node) if node in self.families else None

    def resolve(self, name) -> str:
        return name

    def projection(self, fact, data) -> str:
        return Frame(self.sw.plan, data).read(fact)

    def _join(self, a, b):
        groups = [self.families.groups(node) if node in self.families else {node[1]} if node[0] == "group" else set() for node in (a, b)]
        if all(groups) and not groups[0] & groups[1] and (a not in self.families or b not in self.families or self.families.find(a) != self.families.find(b)):
            self.warnings += (f"groups {sorted(groups[0])} and {sorted(groups[1])} share an input; they are joined",)
        self.families.join(a, b)

    def group(self, node) -> str:
        return self.families.name(node)

    @staticmethod
    def split_of_group(group: str) -> str:
        return "tune" if int(digest(["split", group])[:8], 16) % 2 == 0 else "check"

    def component(self, group) -> str:
        return digest([self.manifest["namespace"], group])

    def seeds(self, group):
        return [tuple(node) for node in self.manifest["families"][group]["seeds"]]

    def historical_stages(self, snap):
        stages_by_root = {}
        for fact, stages in snap.family_seeds.items():
            for stage, nodes in stages.items():
                for node in nodes:
                    if tuple(node) in self.families:
                        stages_by_root.setdefault(self.families.find(node), set()).add(stage)
        self._historical = stages_by_root

    def family_uses(self, data):
        node = ("key", input_key(self.sw.plan, data))
        return tuple(sorted(getattr(self, "_historical", {}).get(self.families.find(node), ()))) if node in self.families else ()

    def split(self, data) -> str:
        node = ("key", input_key(self.sw.plan, data))
        return self.manifest["families"].get(self.group(node), {}).get("split", "conflict") if node in self.families else "conflict"

    def examples(self, fact: str, split: str | None = None, taught: bool | None = None) -> list[Example]:
        return [e for e in self.by_fact.get(fact, []) if (split is None or e.split == split)
                and (taught is None or e.taught == taught)]

    def rules(self, split: str | None = None) -> list[dict]:
        return [it for it in self.items if "rule" in it and (split is None or self.split(it["input"]) == split)]


def fold_of(example: Example, folds: int = FOLDS) -> int:
    return example.fold if example.fold >= 0 else int(digest(["fold", example.group])[:8], 16) % folds
