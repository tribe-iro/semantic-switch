import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.tools.data import Labels
from semantic_switch.training import Task, declaration
from experiments.stored_cascade import cascade, experiment
from semantic_testkit.scoring import FakeScorer

SPEC = {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
        'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}}
REVIEWED = [{'derivation': 'authored', 'producer': 'reviewer', 'actor': {'kind': 'human'}}]
CHECK = [('refund please', True), ('maybe a refund', True), ('hello 3', False), ('refund now', True), ('hello 4', False), ('maybe later', False)]


def teacher_oracle(state, question, width):
    return [.97, .03] if 'refund' in state or 'later' in state else [.03, .97]


def locked(directory, scorer):
    directory.mkdir(parents=True)
    path = directory / 'support.yaml'
    files.publish(path, SPEC)
    with Runtime(scorer) as runtime:
        tools.lock(runtime, path)
    return directory / 'support.lock.json'


def evidence(directory):
    directory.mkdir(parents=True)
    path = directory / 'support.yaml'
    files.publish(path, SPEC)
    paths = files.Paths.of(path)
    with Runtime(FakeScorer()) as runtime:
        task = Task(runtime.load(path, draft=True))
        files.publish(paths.test, ''.join(json.dumps(task.label({'message': message}, {'refund': value}, sources={'refund': REVIEWED}, role='check',
                                                                groups=[f'family-{n}'])) + '\n' for n, (message, value) in enumerate(CHECK)))
        Labels(runtime.load(path, draft=True))
    return paths.test, paths.dataset


def test_matched_cascade_runs_both_frozen_switches_once_and_defers_only_uncertain_calls(tmp_path):
    student, teacher = locked(tmp_path / 'student', FakeScorer(name='student')), locked(tmp_path / 'teacher', FakeScorer(name='teacher', oracle=teacher_oracle))
    labels, manifest = evidence(tmp_path / 'evidence')
    scorers = {'student': FakeScorer(name='student'), 'teacher': FakeScorer(name='teacher', oracle=teacher_oracle)}
    declared = declaration(claim='non_inferiority', margin=.2, benefit='latency')
    found = experiment(student, teacher, labels, manifest, tmp_path / 'run', declared=declared, failed='fail',
                       bindings={name: {'judge': scorer} for name, scorer in scorers.items()})
    assert len(scorers['student'].requests) == len(scorers['teacher'].requests) == len(CHECK)
    assert found['deferred'] == 2 and found['stages'] == {'student': 4, 'teacher': 2} and found['deferral_rate'] == pytest.approx(2 / 6)
    cascade, student_only, teacher_only = (found['arms'][name]['targets'] for name in ('cascade', 'student', 'teacher'))
    assert (student_only['accepted'], student_only['correct_accepted']) == (4, 4)
    assert (teacher_only['accepted'], teacher_only['correct_accepted'], teacher_only['wrong_accepted']) == (6, 5, 1)
    assert (cascade['accepted'], cascade['correct_accepted'], cascade['wrong_accepted']) == (6, 5, 1)
    assert found['arms']['student']['unresolved'] == 2 and found['arms']['cascade']['unresolved'] == 0
    assert found['exposure']['independent'] and found['lineage']['student']['state'] == 'declared_disjoint' and found['reference']['state'] == 'agreement' and found['reference']['strata'] == {'authored': len(CHECK)}
    assert found['student'] == files.read_json(student)['id'] and found['teacher'] == files.read_json(teacher)['id']
    assert files.read_json(tmp_path / 'run' / 'cascade.json')['id'] == found['id'] and (tmp_path / 'run' / 'arms' / 'comparison.json').exists()
    with pytest.raises(ConfigError, match='already published'):
        experiment(student, teacher, labels, manifest, tmp_path / 'run', declared=declared, failed='fail', bindings={name: {'judge': s} for name, s in scorers.items()})


def test_cascade_experiment_declares_its_failure_policy_and_arms_before_scoring(tmp_path):
    declared = declaration(margin=.1)
    with pytest.raises(ConfigError, match='failed-student policy'):
        experiment('student.lock.json', 'teacher.lock.json', 'labels', 'manifest', tmp_path / 'run', declared=declared, failed='retry')
    with pytest.raises(ConfigError, match='bindings name the arms'):
        experiment('student.lock.json', 'teacher.lock.json', 'labels', 'manifest', tmp_path / 'run', declared=declared, bindings={'oracle': {}})


def row(key, status, correct, *, ms=10., cost=None, accepted=True):
    return {'input': key, 'family': key, 'status': status, 'elapsed_ms': ms, 'requests': 1, 'provider_cost': cost, 'usage_known': int(cost is not None),
            'targets': {'refund': {'key': key, 'family': key, 'gold': 'true', 'correct': correct, 'accepted': accepted and correct is not None,
                                   'outcome': status, 'answer': None if correct is None else 'true' if correct else 'false',
                                   'prediction': None if correct is None else {'kind': 'value', 'key': 'true' if correct else 'false'}, 'source': 'judge'}}}


def test_cascade_defers_only_uncertain_whole_calls_and_keeps_both_stage_costs():
    student = [row('a', 'ok', True), row('b', 'uncertain', False, accepted=False), row('c', 'needs_input', None), row('d', 'none', True),
               row('e', 'failed', None)]
    teacher = [row(key, 'ok', True, ms=100., cost=.5) for key in 'abcde']
    found = cascade(student, teacher)
    assert found['deferred'] == 1 and found['stages'] == {'student': 4, 'teacher': 1}
    deferred = found['arms']['cascade']
    assert deferred['targets']['raw_correct'] == 3 and deferred['statuses'] == {'failed': 1, 'needs_input': 1, 'none': 1, 'ok': 2}
    assert deferred['latency_ms']['p50'] == 10. and deferred['provider_cost'] is None
    assert found['arms']['student']['targets']['raw_correct'] == 2 and found['arms']['teacher']['targets']['raw_correct'] == 5
    retried = cascade(student, teacher, failed='defer')
    assert retried['deferred'] == 2 and retried['arms']['cascade']['statuses'] == {'needs_input': 1, 'none': 1, 'ok': 3}
    costed = cascade([row('a', 'uncertain', None, cost=.1)], [row('a', 'ok', True, ms=100., cost=.5)])
    assert costed['arms']['cascade']['latency_ms']['p50'] == 110. and costed['arms']['cascade']['provider_cost']['total'] == pytest.approx(.6)
    with pytest.raises(ConfigError, match='same inputs'):
        cascade(student, teacher[:-1])
    with pytest.raises(ConfigError, match='failed-student policy'):
        cascade(student, teacher, failed='retry')
