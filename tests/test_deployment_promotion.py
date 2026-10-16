from pathlib import Path
import threading

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.artifacts import RECEIPT_SCHEMA, artifact_digest, load_receipt
from semantic_switch.tools import deployment
from semantic_switch.tools.deployment import DECISION_SCHEMA, current, decide, declaration, promote, promotion_decision, rollback

SPEC = {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
        'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}}
INDEPENDENT = {'state': 'fresh', 'independent': True}
DISJOINT = {'state': 'declared_disjoint'}
IMPROVE = declaration(claim='improvement', margin=.05, required_precision=.9)


def receipt(directory, name):
    root = Path(directory) / name
    root.mkdir(parents=True)
    (root / 'model.gguf').write_bytes(name.encode())
    files.publish(root / 'task.yaml', SPEC)
    files.publish(root / 'task.lock.json', {'id': f'lock-{name}'})
    data = {'schema': RECEIPT_SCHEMA, 'kind': 'specialization-receipt', 'gguf': 'model.gguf', 'task': 'task.yaml',
            'lock': {'path': 'task.lock.json', 'id': f'lock-{name}', 'candidate_id': f'candidate-{name}'},
            'files': {item: files.sha256_file(root / item) for item in ('model.gguf', 'task.yaml', 'task.lock.json')}, 'native': {}, 'execution': {},
            'scope': {'targets': ['refund'], 'whole_task': True, 'binding': 'judge'}, 'publication': 'application',
            'provenance': {'kind': 'base', 'rows': []}}
    data['id'] = artifact_digest(data)
    files.publish(root / 'receipt.json', data)
    return root / 'receipt.json'


def arm(*, failed=0, unresolved=0, forbidden=0, precision=.95, p50=10., cost=None):
    return {'failed': failed, 'unresolved': unresolved, 'forbidden_accepted': forbidden, 'targets': {'precision': precision},
            'latency_ms': {'p50': p50, 'p95': p50 * 2}, 'provider_cost': None if cost is None else {'total': cost, 'calls_with_usage': 1}}


def report(declared, *, lower=.08, support='reported', exposure=INDEPENDENT, lineage=DISJOINT, reference='established', baseline=None, candidate=None,
           ids=('incumbent', 'candidate')):
    found = {'id': 'r' * 64, 'declaration': declared, 'exposure': exposure, 'lineage': {'baseline': DISJOINT, 'candidate': lineage},
             'reference': {'state': reference},
             'arms': {'baseline': {'entry': ids[0], **(baseline or arm())}, 'candidate': {'entry': ids[1], **(candidate or arm())}},
             'paired': {'difference': {'interval': None if lower is None else [lower, lower + .1], 'support': support}}}
    return {**found, 'decision': decide(found)}


def published(path, found):
    files.publish(path, promotion_decision(found))
    return path


@pytest.mark.parametrize('changes, verdict', [
    ({}, 'supported'), ({'lower': .03}, 'not_supported'),
    ({'exposure': {'state': 'empty', 'independent': False}}, 'no_check_evidence'),
    ({'exposure': {'state': 'consulted', 'independent': False}}, 'reused_evidence'),
    ({'lineage': {'state': 'known_overlap'}}, 'overlapping_lineage'), ({'lineage': {'state': 'unproven'}}, 'unproven_lineage'),
    ({'reference': 'agreement'}, 'reference_agreement'), ({'reference': 'partial'}, 'partial_reference'),
    ({'support': 'sparse'}, 'inconclusive'), ({'lower': None, 'support': 'insufficient'}, 'inconclusive'),
    ({'candidate': arm(failed=1)}, 'constraint_violated'), ({'candidate': arm(forbidden=1)}, 'constraint_violated'),
    ({'candidate': arm(precision=.85)}, 'precision_below_requirement')])
def test_improvement_decisions_follow_the_predeclared_rule(changes, verdict):
    assert report(IMPROVE, **changes)['decision']['verdict'] == verdict


def test_non_inferiority_needs_its_declared_operational_benefit_and_tolerances():
    declared = declaration(claim='non_inferiority', margin=.02, benefit='cost', tolerated={'unresolved': 2})
    cheaper = {'baseline': arm(cost=4., unresolved=1), 'candidate': arm(cost=1., unresolved=3)}
    assert report(declared, lower=-.01, **cheaper)['decision']['verdict'] == 'supported'
    assert report(declared, lower=-.03, **cheaper)['decision']['verdict'] == 'not_supported'
    assert report(declared, lower=-.01, baseline=arm(cost=1.), candidate=arm(cost=4.))['decision']['verdict'] == 'not_supported'
    assert report(declared, lower=-.01, baseline=arm(cost=4.), candidate=arm(cost=1., unresolved=3))['decision']['verdict'] == 'constraint_violated'
    for wrong in (dict(claim='non_inferiority', margin=.02), dict(claim='improvement', margin=.02, benefit='latency'), dict(claim='better', margin=.1),
                  dict(claim='improvement', margin=1.), dict(claim='improvement', margin=.1, tolerated={'crashes': 1})):
        with pytest.raises(ConfigError):
            declaration(**wrong)


def test_supported_decision_updates_the_expected_pointer_and_rollback_reuses_the_previous_lock(tmp_path):
    first, second = receipt(tmp_path, 'first'), receipt(tmp_path, 'second')
    ids = (load_receipt(first).id, load_receipt(second).id)
    before = {path: path.read_bytes() for path in (first.parent / 'model.gguf', second.parent / 'model.gguf')}
    pointer = tmp_path / 'current.json'
    promote(pointer, first, reason='initial adoption')
    with pytest.raises(ConfigError, match='not the expected absent pointer; nothing changed'):
        promote(pointer, second, reason='newer')
    weak = published(tmp_path / 'weak.json', report(IMPROVE, ids=ids, lower=.01))
    assert files.read_json(weak)['verdict'] == 'not_supported' and files.read_json(weak)['schema'] == DECISION_SCHEMA
    with pytest.raises(ConfigError, match='not a supported claim'):
        promote(pointer, second, reason='newer', decision=weak)
    assert current(pointer) == first
    strong = published(tmp_path / 'strong.json', report(IMPROVE, ids=ids))
    tampered = tmp_path / 'tampered.json'
    files.publish(tampered, {**files.read_json(strong), 'baseline': 'x' * 64})
    with pytest.raises(ConfigError, match=f'not an intact schema-{DECISION_SCHEMA} promotion decision'):
        promote(pointer, second, reason='newer', decision=tampered)
    promote(pointer, second, reason='round 2', decision=strong)
    assert current(pointer) == second
    rolled = rollback(pointer, reason='regression reported', expected=ids[1])
    assert current(pointer) == first and rolled['id'] == ids[0]
    assert [entry['id'] for entry in rolled['history']] == [ids[0], ids[1], ids[0]] and rolled['history'][-1]['reason'] == 'rollback: regression reported'
    assert {path: path.read_bytes() for path in before} == before
    with pytest.raises(ConfigError, match='no previously verified lock'):
        rollback(pointer, reason='again', expected=ids[0], to='0' * 64)


def decision(directory, baseline, candidate, verdict='supported'):
    record = {'schema': DECISION_SCHEMA, 'kind': 'promotion-decision', 'comparison': 'c' * 64, 'baseline': baseline, 'candidate': candidate, 'verdict': verdict}
    path = directory / f'{candidate[:8]}-over-{(baseline or "none")[:8]}.json'
    files.publish(path, {**record, 'id': artifact_digest(record)})
    return path


def raced(monkeypatch, *actions):
    barrier, verified, outcomes = threading.Barrier(len(actions)), deployment.entry, [None] * len(actions)
    def entry(path):
        found = verified(path)
        barrier.wait(5)
        return found
    monkeypatch.setattr(deployment, 'entry', entry)
    def run(at, action):
        try:
            outcomes[at] = action()
        except ConfigError as error:
            outcomes[at] = error
    threads = [threading.Thread(target=run, args=item) for item in enumerate(actions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    monkeypatch.setattr(deployment, 'entry', verified)
    return outcomes


def history(pointer):
    return [item['id'] for item in files.read_json(pointer)['history']]


def test_promotion_is_one_compare_and_update_against_the_decided_incumbent(tmp_path, monkeypatch):
    a, b, c = (receipt(tmp_path, name) for name in ('a', 'b', 'c'))
    A, B, C = (load_receipt(path).id for path in (a, b, c))
    pointer = tmp_path / 'current.json'
    first = raced(monkeypatch, lambda: promote(pointer, a, reason='adopt a'), lambda: promote(pointer, b, reason='adopt b'))
    won = [isinstance(outcome, dict) for outcome in first]
    assert sorted(won) == [False, True] and history(pointer) == [[A, B][won.index(True)]]
    assert 'not the expected absent pointer' in str(first[won.index(False)])
    pointer = tmp_path / 'raced.json'
    promote(pointer, a, reason='adopt a')
    both = raced(monkeypatch, lambda: promote(pointer, b, reason='b', decision=decision(tmp_path, A, B)),
                 lambda: promote(pointer, c, reason='c', decision=decision(tmp_path, A, C)))
    won = [isinstance(outcome, dict) for outcome in both]
    winner, loser = ([B, C][won.index(True)], [B, C][won.index(False)])
    assert sorted(won) == [False, True] and history(pointer) == [A, winner] and current(pointer) == {B: b, C: c}[winner]
    assert f'serves {winner}, not the expected {A}' in str(both[won.index(False)]) and loser not in history(pointer)
    stale = tmp_path / 'stale.json'
    promote(stale, a, reason='adopt a')
    promote(stale, c, reason='c over a', decision=decision(tmp_path, A, C))
    before = files.read_json(stale)
    with pytest.raises(ConfigError, match=f'serves {C}, not the expected {A}; nothing changed'):
        promote(stale, b, reason='b over a', decision=decision(tmp_path, A, B))
    assert files.read_json(stale) == before and current(stale) == c
    rolled = raced(monkeypatch, lambda: rollback(stale, reason='one', expected=C), lambda: rollback(stale, reason='two', expected=C))
    won = [isinstance(outcome, dict) for outcome in rolled]
    assert sorted(won) == [False, True] and history(stale) == [A, C, A] and f'serves {A}, not the expected {C}' in str(rolled[won.index(False)])


def test_altered_decisions_and_failed_publication_leave_the_pointer_valid(tmp_path, monkeypatch):
    a, b = receipt(tmp_path, 'a'), receipt(tmp_path, 'b')
    A, B = load_receipt(a).id, load_receipt(b).id
    pointer = tmp_path / 'current.json'
    promote(pointer, a, reason='adopt a')
    before = files.read_json(pointer)
    tampered = decision(tmp_path, A, B)
    files.publish(tampered, {**files.read_json(tampered), 'baseline': 'f' * 64})
    with pytest.raises(ConfigError, match='not an intact'):
        promote(pointer, b, reason='b', decision=tampered)
    with pytest.raises(ConfigError, match='not a supported claim'):
        promote(pointer, b, reason='b', decision=decision(tmp_path, A, B, verdict='inconclusive'))
    honest = decision(tmp_path, A, B)
    (b.parent / 'model.gguf').write_bytes(b'altered')
    with pytest.raises(ConfigError):
        promote(pointer, b, reason='b', decision=honest)
    (b.parent / 'model.gguf').write_bytes(b'b')
    publish = deployment.files.publish
    def interrupted(path, *args, **kwargs):
        if Path(path) == pointer:
            raise OSError('disk full before the atomic rename')
        return publish(path, *args, **kwargs)
    monkeypatch.setattr(deployment.files, 'publish', interrupted)
    with pytest.raises(OSError, match='disk full'):
        promote(pointer, b, reason='b', decision=honest)
    monkeypatch.undo()
    assert files.read_json(pointer) == before and current(pointer) == a
    promote(pointer, b, reason='b', decision=honest)
    assert history(pointer) == [A, B]
