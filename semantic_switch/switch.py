from __future__ import annotations

import time
from typing import Iterable

from . import files
from .compiler import Plan, compile_spec
from .errors import ConfigError
from .examples import derive_fact_labels, label_record, make_example
from .executor import SOURCES, Call, Executor
from .results import SwitchResult
from .snapshot import Snapshot, draft, from_lock


class Switch:
    def __init__(self, runtime, path, draft: bool = False, allow_model_change: bool = False):
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

    def draft_snapshot(self) -> Snapshot:
        plan = compile_spec(files.read_yaml(self.paths.yaml), self.paths.yaml.name, self.paths.yaml.parent)
        return draft(plan, files.read_jsonl(self.paths.examples))

    def reload(self):
        data = None if self.draft else files.read_json(self.paths.lock)
        snap = from_lock(data, self.rt.identity, self.allow_model_change, self.paths.lock.name) if data \
            else self.draft_snapshot()
        self._snap = snap

    def bind(self, data, snap: Snapshot):
        fields = snap.plan.fields
        return {next(iter(fields)): data} if isinstance(data, str) and len(fields) == 1 else data

    def run(self, datas: list, snap: Snapshot | None = None) -> list[SwitchResult]:
        snap = snap or self._snap
        t0 = time.perf_counter()
        calls = [self.exec.start(snap, self.bind(d, snap)) for d in datas]
        self.exec.decide(snap, calls)
        ms = (time.perf_counter() - t0) * 1000 / max(len(calls), 1)
        return [self.exec.result(snap, c, ms) for c in calls]

    def __call__(self, data) -> SwitchResult:
        return self.run([data])[0]

    def many(self, datas: Iterable):
        snap = self._snap
        for d in datas:
            yield self.run([d], snap)[0]

    def score_facts(self, datas: list, snap: Snapshot | None = None, sources: tuple = SOURCES,
                    facts: Iterable[int] | None = None) -> list[Call]:
        snap = snap or self._snap
        calls = [self.exec.start(snap, self.bind(d, snap)) for d in datas]
        self.exec.score(snap, calls, list(facts if facts is not None else range(len(snap.plan.facts))), sources)
        return calls

    def explain(self, data, raw: bool = False) -> str:
        return self(data).explain(raw)

    def correct(self, data, fact: str | None = None, value=None, case: str | None = None) -> dict:
        plan = self._snap.plan
        created = files.now()
        if case is not None:
            if case not in {c.id for c in plan.cases} | {"otherwise"}:
                raise ConfigError(f"unknown case {case!r}")
            files.append_jsonl(self.paths.test, [label_record(plan, data, {}, case)])
            examples = [make_example(plan, plan.fact(n), data, v, created)
                        for n, v in derive_fact_labels(plan, case).items()]
        else:
            examples = [make_example(plan, plan.fact(fact), data, value, created)]
        files.append_jsonl(self.paths.examples, [e.record() for e in examples])
        if self._snap.locked:
            return {"recorded": len(examples), "scope": "this exact input", "effective": "after ss lock"}
        snap = self._snap
        for e in examples:
            snap = snap.with_example(e)
        self._snap = snap
        return {"recorded": len(examples), "scope": "this exact input", "effective": "now"}
