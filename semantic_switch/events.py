from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
from types import MappingProxyType
import uuid

from .judge import SCORE_SEMANTICS
from .results import freeze, predicted, requested, thaw

EVENT_SCHEMA = 9


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    schema: int
    event: str
    call: str
    item: int
    operation: str
    status: str
    lock_id: str | None
    candidate_id: str | None
    task: Mapping
    predictor: Mapping
    rule: object
    facts: Mapping
    cost: Mapping
    complete: bool
    input: object = None
    score_semantics: str = SCORE_SEMANTICS

    def __post_init__(self):
        for name in ('task', 'predictor', 'facts', 'cost', 'rule', 'input'):
            value = getattr(self, name)
            if name == 'input' or not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, freeze(value))

    def as_dict(self):
        return {field.name: thaw(getattr(self, field.name)) for field in fields(self)}


def call_id() -> str:
    return uuid.uuid4().hex


def _policy(policy) -> dict:
    return {'true': policy.t_true, 'false': policy.t_false}


def _members(members, policies) -> dict:
    return {key: {'status': member.status, 'prediction': predicted(member.prediction),
                  'score': member.p_true if hasattr(member, 'p_true') else member.score,
                  'scores': dict(member.scores), 'observed_models': member.observed_models,
                  'why': member.error if hasattr(member, 'error') else member.why, 'policy': _policy(policies[key])}
            for key, member in members.items()}


class SwitchContext:
    __slots__ = ('runtime', 'snap', 'roles', 'empty', '_judged')

    def __init__(self, runtime, snap):
        plan = snap.plan
        self.runtime, self.snap, self.roles = runtime, snap, {plan.binding_of(f) for f in plan.observed}
        self.empty = self._described(snap.models) if snap.models else self._described({})
        self._judged = self.empty if snap.models else None

    def _described(self, predictor):
        from .compiler import digest
        plan = self.snap.plan
        identities = {role: digest(predictor[role]) if role in predictor else None for role in self.roles}
        return freeze(predictor), freeze({'name': plan.name, 'switch_rev': plan.switch_rev,
                                          'targets': {'/'.join(plan.facts[f].path): {'meaning_rev': plan.facts[f].meaning_rev, 'binding': plan.binding_of(f),
                                                                                     'identity': identities[plan.binding_of(f)]} for f in plan.observed}})

    @property
    def judged(self):
        if self._judged is None:
            self._judged = self._described({name: self.runtime.identity(name) for name in self.snap.plan.bindings})
        return self._judged

    def event(self, result, call, item, data=None):
        snap, plan = self.snap, self.snap.plan
        facts = {}
        for name, fact in result.facts.items():
            spec = plan.fact(name)
            members = {card.id: snap.policy(fact.members[card.id].source, plan.facts[card.claims[0]]) for card in spec.cards if card.id in fact.members}
            facts[name] = {'binding': plan.binding_of(plan.fact_index[name]), 'meaning_rev': spec.meaning_rev, 'source': fact.source, 'status': fact.status,
                           'prediction': predicted(fact.prediction), 'score': fact.score, 'settled': fact.settled, 'complete': fact.complete,
                           'why': fact.why, 'mass': fact.mass, 'guidance': fact.guidance, 'observed_models': fact.observed_models,
                           'scores': fact.scores, 'policy': _policy(snap.policy(fact.source, spec)), 'requests': requested(fact.requests),
                           'members': _members(fact.members, members)}
        predictor, task = self.judged if any(f.source == 'judge' for f in result.facts.values()) else self.empty
        return DecisionEvent(EVENT_SCHEMA, call_id(), call, item, 'switch', result.status, snap.lock_id, snap.candidate_id, task,
                             predictor, result.rule, facts, result.cost, all(fact.complete is not False for fact in result.facts.values()), data)


def inline_task(items, guidance) -> dict:
    from .compiler import digest
    from .guidance import normalize
    out = {}
    for key, item in items.items():
        definition, bound = item.record(), normalize(item.guidance if item.guidance is not None else guidance)
        out[key] = {'kind': type(item).__name__.lower(), 'definition': digest(definition), 'guidance': bound.digest,
                    'meaning_rev': digest([definition, bound.digest])[:8]}
    return {'items': out}


class InlineContext:
    __slots__ = ('runtime', 'items', 'binding', 'task', 'policies', '_predictor')

    def __init__(self, runtime, items, guidance=None, binding='judge'):
        from .judgment import Policy
        self.runtime, self.items, self.binding, self.task = runtime, items, binding, freeze(inline_task(items, guidance))
        self.policies, self._predictor = {key: Policy.of(item.min_score) for key, item in items.items()}, None

    def event(self, results, cost, call, item, data=None):
        statuses = {r.status for r in results.values()}
        status = next((s for s in ('failed', 'needs_input', 'uncertain') if s in statuses), 'ok')
        facts = {}
        for name, result in results.items():
            policy = self.policies[name]
            members = getattr(result, 'members', {})
            scores = getattr(result, 'scores', {})
            facts[name] = {'status': result.status, 'prediction': predicted(getattr(result, 'prediction', None)), 'score': None if members else max(scores.values()) if scores else getattr(result, 'score', None),
                           'mass': getattr(result, 'mass', None), 'source': 'judge' if scores or members or result.observed_models else None,
                           'guidance': result.guidance, 'observed_models': result.observed_models, 'complete': getattr(result, 'complete', None),
                           'why': result.error, 'scores': scores, 'policy': _policy(policy), 'requests': requested(result.requests),
                           'members': _members(members, {key: policy for key in members})}
        judged = any(value['source'] == 'judge' for value in facts.values())
        if judged and self._predictor is None:
            self._predictor = freeze({self.binding: self.runtime.identity(self.binding)})
        return DecisionEvent(EVENT_SCHEMA, call_id(), call, item, 'inline', status, None, None, self.task, self._predictor if judged else {}, None, facts, cost,
                             all(getattr(result, 'complete', True) for result in results.values()), data)


def inline_event(runtime, results, items, cost, call, item, guidance=None, data=None, binding='judge'):
    return InlineContext(runtime, items, guidance, binding).event(results, cost, call, item, data)
