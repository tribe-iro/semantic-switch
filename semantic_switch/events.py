from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .judge import SCORE_SEMANTICS
from .results import freeze, predicted


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    schema: int
    operation: str
    status: str
    lock_id: str | None
    candidate_id: str | None
    predictor: Mapping
    case: object
    facts: Mapping
    cost: Mapping
    complete: bool
    score_semantics: str = SCORE_SEMANTICS

    def __post_init__(self):
        for name in ('predictor', 'facts', 'cost', 'case'):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def as_dict(self):
        from dataclasses import fields
        from .results import thaw
        return {field.name: thaw(getattr(self, field.name)) for field in fields(self)}


def _policy(policy) -> dict:
    return {'true': policy.t_true, 'false': policy.t_false}


def _members(members, policies) -> dict:
    return {key: {'status': member.status, 'prediction': predicted(member.prediction),
                  'score': member.p_true if hasattr(member, 'p_true') else member.score,
                  'scores': dict(member.scores), 'observed_models': list(member.observed_models),
                  'why': member.error if hasattr(member, 'error') else member.why, 'policy': _policy(policies[key])}
            for key, member in members.items()}


def switch_event(runtime, snap, result):
    plan = snap.plan
    facts = {}
    for name, fact in result.facts.items():
        spec = plan.fact(name)
        members = {card.id: snap.policy(fact.members[card.id].source, plan.facts[card.claims[0]]) for card in spec.cards if card.id in fact.members}
        facts[name] = {'source': fact.source, 'status': fact.status, 'prediction': predicted(fact.prediction), 'score': fact.score, 'settled': fact.settled, 'complete': fact.complete,
                       'why': fact.why, 'mass': fact.mass, 'guidance': fact.guidance, 'observed_models': fact.observed_models,
                       'scores': dict(fact.scores), 'policy': _policy(snap.policy(fact.source, spec)),
                       'members': _members(fact.members, members)}
    return DecisionEvent(5, 'switch', result.status, snap.lock_id, snap.candidate_id,
                         snap.models or ({"judge": runtime.identity()} if any(f.source == "judge" for f in result.facts.values()) else {}), result.case, facts, result.cost,
                         all(fact.complete is not False for fact in result.facts.values()))


def inline_event(runtime, results, items, cost):
    from .judgment import Policy
    statuses = {r.status for r in results.values()}
    status = next((s for s in ('failed', 'needs_input', 'uncertain') if s in statuses), 'ok')
    facts = {}
    for name, result in results.items():
        policy = Policy.of(items[name].min_score)
        members = getattr(result, 'members', {})
        scores = getattr(result, 'scores', {})
        facts[name] = {'status': result.status, 'prediction': predicted(getattr(result, 'prediction', None)), 'score': None if members else max(scores.values()) if scores else getattr(result, 'score', None),
                       'mass': getattr(result, 'mass', None), 'source': 'judge' if scores or members or result.observed_models else None,
                       'guidance': result.guidance, 'observed_models': result.observed_models, 'complete': getattr(result, 'complete', None),
                       'why': result.error, 'scores': scores, 'policy': _policy(policy),
                       'members': _members(members, {key: policy for key in members})}
    predictor = runtime.identity() if any(value['source'] == 'judge' for value in facts.values()) else {}
    return DecisionEvent(5, 'inline', status, None, None, predictor, None, facts, cost,
                         all(getattr(result, 'complete', True) for result in results.values()))
