import ast
import json
from importlib.util import find_spec
from pathlib import Path

import pytest

from semantic_switch import Actor, ConfigError, files
import semantic_learn
from semantic_learn import teacher as module
from semantic_learn.capture import Bounds, Capture, Consent, Redactor, Sampling, captured
from semantic_learn.review import review_export
from semantic_testkit.labels import TEACHER, declared
from semantic_testkit.scoring import FakeScorer

from test_teacher import collection_task, envelope, oracle

CONSENT = Consent('support-terms', '2026-09')


def student(state, question, width):
    if 'flip' in state:
        return [.04, .96] if 'money back' in question.lower() else [.5, .5]
    return oracle(state, question, width)


def test_review_export_samples_a_random_audit_then_declared_strata_with_disclosed_probabilities(tmp_path, monkeypatch):
    messages = ['please refund me', 'refund now flip', 'wrong invoice total', 'maybe something', 'refund this charge', 'refund again']
    path, situations = collection_task(tmp_path, [envelope(message, groups=[f'g{i}']) for i, message in enumerate(messages)])
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: FakeScorer(oracle))
    collected = module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), fact='refund')
    with Capture(tmp_path / 'support.capture.jsonl', redactor=Redactor('none', lambda data: data), sampling=Sampling(1., 's'), bounds=Bounds(100, 1 << 20),
                 link_key=b'k' * 32, judge=FakeScorer(student)) as learn:
        switch = learn.load(path)
        for message in messages:
            learn.decide(switch, {'message': message}, consent=CONSENT)
    exported = review_export(path, tmp_path / 'queue.review.json', size=4, audit=1, captures=[tmp_path / 'support.capture.jsonl'],
                             collected=[collected['manifest']], strata=('teacher_uncertain', 'disagreement'), seed=3)
    rows = files.read_jsonl(exported['queue'])
    by_message = {row['input']['message']: row for row in rows}
    assert len(rows) == 3 and exported['population']['rows'] == 6 and exported['population']['families'] == 6
    assert exported['audit']['rows'] == 1 and exported['audit']['inclusion_probability'] == pytest.approx(1 / 6)
    assert exported['strata']['teacher_uncertain']['rows'] + exported['strata']['disagreement']['rows'] == 2
    assert {'maybe something', 'refund now flip'} <= set(by_message)
    assert by_message['refund now flip']['facts'] == {'refund': True} and by_message['refund now flip']['label_sources']['refund'][0]['derivation'] == 'judged'
    assert 'facts' not in by_message['maybe something']
    audit = [row for row in rows if row['selections'][-1]['selected_by'] == 'random']
    assert len(audit) == 1 and audit[0]['selections'][-1]['inclusion_probability'] == pytest.approx(1 / 6)
    assert exported['review'][:3] == ['ss', 'label', str(path)] and 'biased review queue' in exported['estimates']['queue']
    again = review_export(path, tmp_path / 'again.review.json', size=4, audit=1, captures=[tmp_path / 'support.capture.jsonl'],
                          collected=[collected['manifest']], strata=('teacher_uncertain', 'disagreement'), seed=3)
    assert files.read_jsonl(again['queue']) == rows
    plain = review_export(path, tmp_path / 'random.review.json', size=2, audit=2, captures=[tmp_path / 'support.capture.jsonl'])
    assert plain['acquisition'] == 'random audit only' and plain['strata'] == {} and len(files.read_jsonl(plain['queue'])) == 2
    with pytest.raises(ConfigError, match='strata are distinct names'):
        review_export(path, tmp_path / 'x.review.json', size=2, audit=1, captures=[tmp_path / 'support.capture.jsonl'], strata=('active',))
    assert len(list(captured(tmp_path / 'support.capture.jsonl'))) == 6


def test_teacher_student_and_review_share_the_redacted_input_and_inconsistency_is_triage(tmp_path, monkeypatch):
    from semantic_learn.review import situations
    path, _ = collection_task(tmp_path, [])
    answers = iter([[.96, .04], [.04, .96], [.96, .04]])
    def drifting(state, question, width):
        return next(answers) if 'twice' in state and 'receive money back' in question else oracle(state, question, width)
    redactor = Redactor('email-v1', lambda data: {key: value.replace('ana@example.org', '[email]') for key, value in data.items()})
    with Capture(tmp_path / 'support.capture.jsonl', redactor=redactor, sampling=Sampling(1., 's'), bounds=Bounds(100, 1 << 20), link_key=b'k' * 32,
                 judge=FakeScorer(drifting)) as learn:
        switch = learn.load(path)
        for message in ('refund ana@example.org twice', 'refund ana@example.org twice', 'hello'):
            learn.decide(switch, {'message': message}, consent=CONSENT)
    written = situations(path, [tmp_path / 'support.capture.jsonl'], tmp_path / 'captured.jsonl')
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: teacher)
    collected = module.collect(path, tmp_path / 'captured.jsonl', tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), fact='refund')
    assert written['rows'] == 2 and all('ana@' not in state for state, _ in teacher.requests)
    assert {row['input']['message'] for row in files.read_jsonl(collected['labels'])} == {'refund [email] twice', 'hello'}
    exported = review_export(path, tmp_path / 'queue.review.json', size=1, audit=0, captures=[tmp_path / 'support.capture.jsonl'],
                             collected=[collected['manifest']], strata=('inconsistency',))
    (row,) = files.read_jsonl(exported['queue'])
    assert row['input'] == {'message': 'refund [email] twice'} and exported['strata']['inconsistency']['rows'] == 1



def lineage(envelope):
    return {'groups': sorted(envelope['groups']), 'pairs': sorted(envelope['pairs']), 'facts': envelope.get('facts'),
            'selections': sorted(json.dumps(item, sort_keys=True) for item in envelope['selections']),
            'sources': json.dumps(envelope.get('label_sources'), sort_keys=True)}


def test_duplicate_inputs_keep_every_thread_link_and_sample_only_exports_keep_full_family_bridges(tmp_path, monkeypatch):
    from semantic_learn.common import BUDGET, Families, task_of
    from semantic_learn.review import Population, collection, situations
    path, _ = collection_task(tmp_path, [])
    arrivals = [('refund a', 'crm:T-1'), ('refund a', 'crm:T-2'), ('refund b', 'crm:T-1'), ('refund c', 'crm:T-2'), ('refund d', 'crm:T-3')]
    with Capture(tmp_path / 'support.capture.jsonl', redactor=Redactor('none', lambda data: data), sampling=Sampling(1., 's'), bounds=Bounds(100, 1 << 20),
                 link_key=b'k' * 32, judge=FakeScorer(oracle)) as learn:
        switch = learn.load(path)
        for message, thread in arrivals:
            learn.decide(switch, {'message': message}, consent=CONSENT, correlation={'thread': thread, 'entity': f'crm:{message}'})
    records = list(captured(tmp_path / 'support.capture.jsonl'))
    written = situations(path, [tmp_path / 'support.capture.jsonl'], tmp_path / 'captured.jsonl')
    rows = {row['input']['message']: row for row in files.read_jsonl(written['situations'])}
    assert written['rows'] == 4 and len(rows['refund a']['groups']) == 3 and len(rows['refund a']['selections']) == 1
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: FakeScorer(oracle))
    collected = module.collect(path, tmp_path / 'captured.jsonl', tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), fact='refund')
    labels = files.read_jsonl(collection(collected['manifest'])['labels.jsonl'])
    task = task_of(path)
    forward, backward = Population(task, BUDGET), Population(task, BUDGET)
    for record in records:
        forward.capture(record)
    for record in labels:
        forward.label(record)
        backward.label(record)
    for record in records:
        backward.capture(record)
    forward.settle()
    backward.settle()
    assert set(forward.units) == set(backward.units) and len(forward.units) == 4
    for key, unit in forward.units.items():
        assert lineage(unit['envelope']) == lineage(backward.units[key]['envelope']) and unit['envelope']['facts'] == {'refund': True}
        assert not unit['unresolved'] and len(unit['student']) == len(backward.units[key]['student'])
    assert len(forward.units[task.key({'message': 'refund a'})]['student']) == 2
    full = review_export(path, tmp_path / 'full.review.json', size=4, audit=4, captures=[tmp_path / 'support.capture.jsonl'], collected=[collected['manifest']])
    assert full['population']['families'] == 2 and full['population']['rows'] == 4 and full['conflicts'] == []
    bridges = {row['input']['message']: {group for group in row['groups'] if group.startswith('review-bridge:')} for row in files.read_jsonl(full['queue'])}
    assert bridges['refund a'] == bridges['refund b'] == bridges['refund c'] and len(bridges['refund a']) == 1 and bridges['refund d'] == set()
    for seed in range(200):
        sample = files.read_jsonl(review_export(path, tmp_path / f'sample-{seed}.review.json', size=2, audit=2, captures=[tmp_path / 'support.capture.jsonl'],
                                                seed=seed)['queue'])
        if {row['input']['message'] for row in sample} == {'refund b', 'refund c'}:
            break
    b, c = sorted(sample, key=lambda row: row['input']['message'])
    own = [set(row['groups']) - bridges[row['input']['message']] for row in (b, c)]
    assert not own[0] & own[1] and bridges['refund b'] <= set(b['groups']) & set(c['groups'])
    index = Families(task, **BUDGET)
    for row in (b, c):
        index.add(row)
    assert index.component(b['input']) == index.component(c['input'])
    from semantic_switch.training import LabelSource
    contrary = task.label({'message': 'refund b'}, {'refund': False}, sources={'refund': [LabelSource('judged', 'other-teacher')]})
    forward.label(contrary)
    forward.settle()
    unit = forward.units[task.key({'message': 'refund b'})]
    assert set(unit['unresolved']) == {'refund'} and 'facts' not in unit['envelope'] and 'conflict' in forward.strata(unit)
    assert sorted((claim.value, [source.producer for source in claim.history]) for claim in unit['unresolved']['refund'].alternatives) == \
        [(False, ['other-teacher']), (True, [TEACHER])]

def sources(package):
    root = Path(find_spec(package).origin).parent
    return sorted(root.rglob('*.py'))


def imported(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.module


def test_learn_imports_only_the_public_and_integration_surfaces_and_no_fitting_library():
    found = {name for path in sources('semantic_learn') for name in imported(path)}
    switch = {name for name in found if name.split('.')[0] == 'semantic_switch'}
    assert switch <= {'semantic_switch', 'semantic_switch.training'}
    assert not {name.split('.')[0] for name in found} & {'torch', 'transformers', 'peft', 'safetensors', 'semantic_trainer', 'bench'}
    assert Path(semantic_learn.__file__).parent.name == 'semantic_learn'


@pytest.mark.parametrize('package', ['semantic_switch', 'semantic_trainer'])
def test_core_and_trainer_never_import_learn(package):
    if find_spec(package) is None:
        pytest.skip(f'{package} is not installed')
    assert not any((name or '').split('.')[0] == 'semantic_learn' for path in sources(package) for name in imported(path))


def test_review_export_queues_a_partial_envelope_and_publishes_every_unresolved_alternative(tmp_path):
    from semantic_learn.common import task_of
    from semantic_switch.training import LabelSource
    path, _ = collection_task(tmp_path, [])
    task = task_of(path)
    data = {'message': 'refund the double charge'}
    teacher = task.label(data, {'refund': True, 'issues': {'refund': True}}, sources={'refund': [LabelSource('judged', 'teacher-a')],
                                                                                       'issues': {'refund': [LabelSource('judged', 'teacher-a')]}})
    other = task.label(data, {'refund': False}, sources={'refund': [LabelSource('judged', 'teacher-b')]})
    manifest = tmp_path / 'labels.collection.json'
    files.publish(manifest, {'kind': 'collection'}, {'labels.jsonl': [json.dumps(row) + '\n' for row in (teacher, other)], 'unresolved.jsonl': []})
    exported = review_export(path, tmp_path / 'conflict.review.json', size=1, audit=1, collected=[manifest])
    (row,) = files.read_jsonl(exported['queue'])
    (claims,) = files.read_jsonl(exported['claims'])
    assert row['facts'] == {'issues': {'refund': True}} and exported['conflicts'] == ['refund'] and exported['population']['conflicted'] == 1
    assert claims['address'] == 'refund' and sorted((alt['value'], [s['producer'] for s in alt['label_sources']]) for alt in claims['alternatives']) == \
        [(False, ['teacher-b']), (True, ['teacher-a'])]
    resolved = task.resolve(data, 'refund', claims['alternatives'], False, reviewer='lead', actor=Actor('human'))
    assert task.merge(task.merge(teacher, resolved), other)['facts'] == {'refund': False, 'issues': {'refund': True}}
