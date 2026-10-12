from __future__ import annotations

import time
import threading
from .context import bind_input, project
from typing import Iterable

from . import files
from .compiler import Plan, compile_spec
from .errors import ConfigError
from .examples import derive_fact_labels, label_record, labelled, make_example
from .executor import SOURCES, Call, Executor
from .results import Fact, SwitchResult, freeze
from .snapshot import Snapshot, draft, from_lock


_SELECTION_UNSET = object()


def visible(plan: Plan, names) -> list[str]:
    if isinstance(names, str):
        raise ConfigError("facts must be a sequence of fact names")
    names = list(names)
    for name in names:
        if not isinstance(name, str) or plan.fact(name).hidden:
            raise ConfigError(f"{name!r} is not a visible fact; score one of {', '.join(f.name for f in plan.visible)}")
    return names

class Switch:
    def __init__(self, runtime, path, draft: bool = False, allow_model_change: bool = False):
        self._update_lock = threading.RLock()
        self.rt = runtime
        self.paths = files.Paths.of(path)
        self.draft = draft
        self.allow_model_change = allow_model_change
        self.exec = Executor(runtime)
        self.reload()

    @property
    def snapshot(self) -> Snapshot:
        return self._snap

    @property
    def plan(self) -> Plan:
        return self._snap.plan

    @property
    def warnings(self) -> tuple:
        return self._snap.warnings

    def describe(self):
        plan = self._snap.plan
        spec = plan.spec

        def fact(f):
            out = {"address": (plan.name, f.name), "kind": f.kind, "definition": spec["facts"][f.name], "reads": f.reads,
                   "guidance": f.judgment.guidance.references, "revision": f.meaning_rev}
            if f.labels:
                out["members"] = {c.id: plan.facts[c.claims[0]].meaning_rev for c in f.cards}
            elif f.cards:
                out["claims"] = {c.id: tuple(plan.facts[k].meaning_rev for k in c.claims) for c in f.cards}
            return out

        facts = {f.name: fact(f) for f in plan.visible}
        guidance = tuple(dict.fromkeys(ref for f in facts.values() for ref in f["guidance"]))
        sections = {(s.id, s.content_digest): s for f in plan.facts for s in f.judgment.guidance.sections}
        documents = tuple({"reference": s.id, "digest": s.content_digest, "text": s.text} for s in sections.values())
        return freeze({"address": (plan.name,), "kind": "switch", "fields": spec["fields"], "reads": spec["reads"], "guidance": guidance,
                       "documents": documents, "cases": spec["cases"], "otherwise": spec["otherwise"], "resolve": spec["resolve"],
                       "revision": plan.switch_rev, "facts": facts})

    def draft_snapshot(self) -> Snapshot:
        plan = compile_spec(files.read_yaml(self.paths.yaml), self.paths.yaml.name, self.paths.yaml.parent)
        return draft(plan, files.read_jsonl(self.paths.examples))

    def reload(self):
        data = None if self.draft else files.read_published(self.paths.lock)
        snap = from_lock(data, self.rt.identity, self.allow_model_change, self.paths.lock.name) if data \
            else self.draft_snapshot()
        with self._update_lock:
            self._snap = snap

    def bind(self, data, snap: Snapshot):
        return bind_input(snap.plan.fields, data)

    def run(self, datas: list, snap: Snapshot | None = None, *, facts: Iterable[str] | None = None) -> list[SwitchResult]:
        with self.rt.operation():
            snap = snap or self._snap
            extra = self._requested(snap, facts)
            size = self.rt.batch_inputs()
            return [r for k in range(0, len(datas), size) for r in self._window(datas[k:k + size], snap, extra)]

    @staticmethod
    def _requested(snap: Snapshot, facts) -> frozenset:
        return frozenset() if facts is None else frozenset(snap.plan.fact_index[name] for name in visible(snap.plan, facts))

    def _window(self, datas: list, snap: Snapshot, extra: frozenset = frozenset()) -> list[SwitchResult]:
        with self.rt.operation():
            t0 = time.perf_counter()
            calls = [self.exec.start(snap, self.bind(d, snap)) for d in datas]
            self.exec.decide(snap, calls, extra=extra)
            results = [self.exec.result(snap, c, extra) for c in calls]
            elapsed = (time.perf_counter() - t0) * 1000
            for result in results:
                result.cost |= {"elapsed_ms": round(elapsed, 2), "amortized_ms": round(elapsed / len(calls), 2)}
            if getattr(self.rt, '_on_decision', None) is not None:
                from .events import switch_event
                for result in results:
                    self.rt._emit_decision(switch_event(self.rt, snap, result))
            return results

    def __call__(self, data, *, facts: Iterable[str] | None = None) -> SwitchResult:
        snap = self._snap
        return self._window([data], snap, self._requested(snap, facts))[0]

    def many(self, datas: Iterable):
        snap = self._snap
        for d in datas:
            yield self._window([d], snap)[0]

    def score_facts(self, datas: list, snap: Snapshot | None = None, sources: tuple = SOURCES,
                    facts: Iterable[str] | None = None) -> list[dict[str, Fact]]:
        snap = snap or self._snap
        names = [f.name for f in snap.plan.visible] if facts is None else visible(snap.plan, facts)
        chosen = {snap.plan.fact_index[name] for name in names}
        return [{name: self.public(snap, call, snap.plan.fact_index[name]) for name in names}
                for call in self._score(datas, snap, sources, sorted(chosen))]

    def public(self, snap: Snapshot, call: Call, f: int) -> Fact:
        return self.exec.public_fact(snap, call, f)

    def _score(self, datas: list, snap: Snapshot, sources: tuple, facts: list[int]) -> list[Call]:
        with self.rt.operation():
            calls = [self.exec.start(snap, self.bind(d, snap)) for d in datas]
            self.exec.decide(snap, calls, sources, frozenset(facts))
            return calls

    def explain(self, data, raw: bool = False) -> str:
        return self(data).explain(raw)

    def correct(self, data, fact: str | None = None, value=None, case: str | None = None, *, group=None, pair=None, selection=_SELECTION_UNSET, selections=(), groups=(), pairs=(), sources=None) -> dict:
        with self._update_lock, files.locked(str(self.paths.examples) + ".transaction"):
            return self._correct(data, fact, value, case, group, pair, selection, selections, groups, pairs, sources)

    def _correct(self, data, fact, value, case, group, pair, selection, selections, groups, pairs, sources):
        from .examples import LabelSource, history_of, metadata_history
        from .selection import Selection, history
        sources = (LabelSource("human", "ss correct"),) if sources is None else history_of(sources)
        selection = Selection("manual") if selection is _SELECTION_UNSET else selection
        selections = history(selections, selection)
        selection = None
        plan = self._snap.plan
        data = project(plan.fields, self.bind(data, self._snap))
        created = files.now()
        if case is not None:
            if case not in {c.id for c in plan.cases} | {"otherwise"}:
                raise ConfigError(f"unknown case {case!r}")
            examples = [make_example(plan, plan.fact(n), data, v, created, group, pair, selection=selection, selections=selections, sources=sources, groups=groups, pairs=pairs)
                        for n, v in derive_fact_labels(plan, case).items()]
        else:
            examples = [make_example(plan, member, data, given, created, group, pair, selection=selection, selections=selections, sources=sources, groups=groups, pairs=pairs)
                        for member, given in labelled(plan, fact, value)]
        if case is not None:
            files.append(self.paths.test, [label_record(plan, data, {}, case, groups=metadata_history(groups, group, "group"), pairs=metadata_history(pairs, pair, "pair"),
                                                        selections=selections)])
        files.append(self.paths.examples, [e.record() for e in examples])
        receipt = {"recorded": len(examples), "case_label_recorded": case is not None,
                   "facts": [list(example.path) for example in examples], "scope": "this exact input",
                   "effective": ("after ss lock" if self._snap.locked else "now") if examples else None}
        if self._snap.locked:
            return receipt
        snap = self._snap
        for e in examples:
            snap = snap.with_example(e)
        self._snap = snap
        return receipt
