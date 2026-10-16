from __future__ import annotations

import getpass
from collections import deque
import threading
from dataclasses import replace
from .context import bind_input
from typing import Iterable

from . import files
from .compiler import Plan, compile_spec
from .errors import ConfigError
from .evidence import Actor, Frame, LabelSource, derive_fact_labels, history_of, label_record, labelled, make_example, recorded, successor
from .executor import SOURCES, Executor, Reuse
from .results import FactResult, Preview, Row, SwitchResult, freeze
from .snapshot import Snapshot, draft, from_lock, reconcile


def visible(plan: Plan, names) -> list[str]:
    if isinstance(names, str):
        raise ConfigError("facts must be a sequence of fact names")
    names = list(names)
    for name in names:
        if not isinstance(name, str) or plan.fact(name).hidden:
            raise ConfigError(f"{name!r} is not a visible fact; score one of {', '.join(f.name for f in plan.visible)}")
    return names


def authored(sources, author, actor):
    if sources is not None:
        if author is not None or actor is not None:
            raise ConfigError("give either the label source history or the author and actor of a new label")
        return history_of(sources)
    if actor is None:
        raise ConfigError("record who wrote this label: actor=Actor('human') or Actor('agent', model=..., session=...), optionally with author=IDENTITY")
    actor = Actor.parse(actor)
    if actor.kind == "unknown":
        raise ConfigError("a new label records who wrote it: a human, or an agent with its model and session")
    return LabelSource("authored", author or getpass.getuser(), actor=actor)


class Switch:
    def __init__(self, runtime, path, draft: bool = False, allow_model_change: bool = False, scope: tuple | None = None, *, frozen: bool = False):
        if scope is not None and not draft:
            raise ConfigError("a scoped trial view scores the authored definition; load it as a draft")
        self.frozen = frozen
        self._update_lock = threading.RLock()
        self.rt = runtime
        self.paths = files.Paths.of(path)
        self.draft = draft
        self.allow_model_change = allow_model_change
        self.scope = scope
        self.exec = Executor(runtime)
        self._events = None
        self.reload()

    @property
    def snapshot(self) -> Snapshot:
        snap = self._snap
        if snap.reconciled:
            return snap
        with self._update_lock:
            if not self._snap.reconciled:
                self._snap = reconcile(self._snap, self.rt, self.allow_model_change, self.paths.lock.name)
            return self._snap

    @property
    def plan(self) -> Plan:
        return self._snap.plan

    @property
    def warnings(self) -> tuple:
        return self.snapshot.warnings

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
                       "documents": documents, "rules": spec["rules"], "otherwise": spec["otherwise"], "resolve": spec["resolve"],
                       "revision": plan.switch_rev, "facts": facts})

    def draft_snapshot(self) -> Snapshot:
        plan = compile_spec(files.read_yaml(self.paths.yaml), self.paths.yaml.name, self.paths.yaml.parent)
        return draft(plan, files.read_jsonl(self.paths.examples))

    def reload(self):
        data = None if self.draft else files.read_published(self.paths.lock)
        if self.frozen and not data:
            raise ConfigError(f"{self.paths.lock}: frozen lock not found")
        snap = from_lock(data, self.paths.lock.name) if data else self.draft_snapshot()
        required = set(snap.plan.bindings)
        if self.scope is not None:
            targets, binding = self.scope
            snap = replace(snap, plan=snap.plan.scoped(targets, binding))
            required = {binding}
        unbound = sorted(required - set(self.rt.models))
        if unbound:
            raise ConfigError(f"{self.paths.yaml.name}: targets are bound to model roles {unbound} that this runtime does not bind; "
                              f"construct the runtime with models={{{', '.join(repr(name) + ': ...' for name in unbound)}}}")
        for name in sorted(required):
            static = getattr(self.rt.models[name], "static", None)
            if static is not None:
                try:
                    static(snap.plan.questions(name))
                except ConfigError as error:
                    raise ConfigError(f"{self.paths.yaml.name}: binding {name!r}: {error}") from None
        with self._update_lock:
            self._snap = snap

    def _context(self, snap: Snapshot):
        from .events import SwitchContext
        cached = self._events
        if cached is None or cached.snap is not snap:
            cached = self._events = SwitchContext(self.rt, snap)
        return cached

    def _targets(self, snap: Snapshot, facts) -> list[int]:
        plan = snap.plan
        if facts is None:
            return [plan.fact_index[f.name] for f in plan.visible]
        if isinstance(facts, str):
            raise ConfigError("facts must be a sequence of fact names")
        return [plan.fact_index[name] if name in plan.fact_index else plan.fact_index[plan.fact(name).name] for name in facts]

    def _decided(self, snap: Snapshot, extra: frozenset, operation):
        def finish(live):
            result = self.exec.result(snap, live.work, extra)
            result.cost["elapsed_ms"] = operation.elapsed(live)
            if operation.receipt is not None:
                result.cost["operation"] = operation.receipt
            return result
        return finish

    def _emit(self, snap: Snapshot, pairs, call: str):
        if self.rt._on_decision is None:
            return
        context = self._context(snap)
        for item, (result, data) in pairs:
            self.rt._emit_decision(context.event(result, call, item, data if self.rt._decision_input else None))

    def run(self, inputs: list, snap: Snapshot | None = None, *, facts: Iterable[str] | None = None) -> list[SwitchResult]:
        with self.rt.operation():
            snap = snap or self.snapshot
            extra = self._requested(snap, facts)
            ex, results = self.exec, [None] * len(inputs)
            operation = ex.operation(ex.window(ex.bindings(snap, SOURCES)), None)
            operation.finish = self._decided(snap, extra, operation)
            fields = snap.plan.fields
            for index, result in operation.drive(inputs, lambda data: ex.start(snap, bind_input(fields, data), SOURCES, None, extra), ordered=False):
                results[index] = result
            from .events import call_id
            self._emit(snap, enumerate(zip(results, inputs)), call_id())
            return results

    @staticmethod
    def _requested(snap: Snapshot, facts) -> frozenset:
        return frozenset() if facts is None else frozenset(snap.plan.fact_index[name] for name in visible(snap.plan, facts))

    def __call__(self, input, *, facts: Iterable[str] | None = None) -> SwitchResult:
        return self.run([input], facts=facts)[0]

    def many(self, inputs: Iterable, *, window=1):
        if type(window) is not int or window < 1:
            raise ConfigError("window must be a positive integer")
        snap = self.snapshot
        ex, fields, admitted = self.exec, snap.plan.fields, deque()
        operation = ex.operation(window, None, receipt=False)
        operation.finish = self._decided(snap, frozenset(), operation)

        def start(data):
            admitted.append(data)
            return ex.start(snap, bind_input(fields, data))
        stream = operation.drive(inputs, start, ordered=True)
        from .events import call_id
        try:
            while True:
                with self.rt.operation():
                    try:
                        index, result = next(stream)
                    except StopIteration:
                        return
                self._emit(snap, [(index, (result, admitted.popleft()))], call_id())
                yield result
        finally:
            stream.close()

    def score(self, inputs: list, facts: Iterable[str] | None = None, *, sources: tuple = SOURCES, decide=False, demand=None,
              snap: Snapshot | None = None, reuse: Reuse | None = None) -> list[Row]:
        with self.rt.operation():
            snap = snap or self.snapshot
            if demand is None:
                shared = self._targets(snap, facts)
                demand = [(shared, wanted) for wanted in (list(decide) if not isinstance(decide, bool) else [decide] * len(inputs))]
            else:
                demand = [(self._targets(snap, names), bool(wanted)) for names, wanted in demand]
            if len(demand) != len(inputs):
                raise ConfigError("score needs one demand per input")
            ex, fields, rows = self.exec, snap.plan.fields, [None] * len(inputs)
            union = sorted({f for requested, _ in demand for f in requested})
            bound = ex.window(ex.bindings(snap, sources, None if any(wanted for _, wanted in demand) else union))
            requests, pending = {}, iter(demand)

            def start(data):
                requested, wanted = next(pending)
                call = ex.start(snap, bind_input(fields, data), sources, None if wanted else frozenset(requested), frozenset(requested) if wanted else frozenset())
                requests[id(call)] = (requested, wanted)
                return call

            def finish(live):
                requested, wanted = requests.pop(id(live.work))
                row = ex.row(live.work, requested, wanted)
                if row.decision is not None:
                    row.decision.cost["elapsed_ms"] = operation.elapsed(live)
                    if operation.receipt is not None:
                        row.decision.cost["operation"] = operation.receipt
                return row
            operation = ex.operation(bound, finish, reuse)
            for index, row in operation.drive(inputs, start, ordered=False):
                rows[index] = row
            return rows

    def score_facts(self, datas: list, snap: Snapshot | None = None, sources: tuple = SOURCES,
                    facts: Iterable[str] | None = None) -> list[dict[str, FactResult]]:
        snap = snap or self.snapshot
        names = [f.name for f in snap.plan.visible] if facts is None else visible(snap.plan, facts)
        return [{name: row.targets[name].result for name in names} for row in self.score(datas, names, sources=sources, snap=snap)]

    def preview(self, inputs: list, facts: Iterable[str] | None = None, *, sources: tuple = SOURCES, snap: Snapshot | None = None,
                requests: bool = True) -> list[Preview]:
        with self.rt.operation():
            snap = snap or self.snapshot
            outputs = None if facts is None else frozenset(self._targets(snap, facts))
            return [self.exec.preview(self.exec.start(snap, bind_input(snap.plan.fields, data), sources, outputs), requests) for data in inputs]

    def explain(self, input, raw: bool = False) -> str:
        return self(input).explain(raw)

    def correct(self, input, fact: str | None = None, value=None, rule: str | None = None, *, groups=(), pairs=(), selections=(), sources=None,
                author: str | None = None, actor=None, predecessors=()) -> dict:
        with self._update_lock, files.locked(str(self.paths.examples) + ".transaction"):
            return self._correct(input, fact, value, rule, groups, pairs, selections, authored(sources, author, actor), tuple(predecessors))

    def _recorded(self, plan: Plan) -> dict:
        try:
            stat = self.paths.examples.stat()
            mark = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            mark = None
        cached = getattr(self, "_claims_cache", None)
        if cached is None or cached[0] != mark or cached[1] is not plan:
            cached = self._claims_cache = (mark, plan, recorded(plan, files.read_jsonl(self.paths.examples))[0])
        return cached[2]

    def _correct(self, data, fact, value, rule, groups, pairs, selections, decision, predecessors):
        plan = self._snap.plan
        frame = Frame(plan, bind_input(plan.fields, data))
        created = files.now()
        if rule is not None:
            if rule not in {c.id for c in plan.rules} | {"otherwise"}:
                raise ConfigError(f"unknown rule {rule!r}")
            wanted = [(plan.fact(n), v) for n, v in derive_fact_labels(plan, rule).items()]
        else:
            wanted = labelled(plan, fact, value)
        known = self._recorded(plan)
        examples = []
        for member, given in wanted:
            given = member.value(member.index(given))
            if isinstance(decision, tuple):
                history = decision
            else:
                prior = [e.claim for e in known[member.name].get(frame.read(member), ())]
                history = successor((*prior, *predecessors), given, decision).history
            examples.append(make_example(plan, member, frame, given, created, groups=groups, pairs=pairs, selections=selections, sources=history))
        if rule is not None:
            files.append(self.paths.test, files.jsonl([label_record(plan, frame, {}, rule, groups=groups, pairs=pairs, selections=selections)]))
        files.append(self.paths.examples, files.jsonl(e.record() for e in examples))
        for e in examples:
            known[e.fact].setdefault(e.key, []).append(e)
        if examples and getattr(self, "_claims_cache", None) is not None and self._claims_cache[2] is known:
            stat = self.paths.examples.stat()
            self._claims_cache = ((stat.st_ino, stat.st_size, stat.st_mtime_ns), plan, known)
        receipt = {"recorded": len(examples), "rule_label_recorded": rule is not None,
                   "facts": [list(example.path) for example in examples], "scope": "this exact input",
                   "effective": ("after ss lock" if self._snap.locked else "now") if examples else None}
        if self._snap.locked:
            return receipt
        snap = self._snap
        for e in examples:
            snap = snap.with_example(e)
        self._snap = snap
        return receipt
