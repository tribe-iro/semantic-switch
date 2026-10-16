
from semantic_switch import model_jobs
import json
from pathlib import Path

import pytest

from semantic_switch import Actor, ConfigError, Runtime, api, files, tools
from semantic_switch.training import TrainingUse
from semantic_switch.artifacts import read_export
from semantic_testkit.labels import TEACHER, declared
from semantic_testkit.scoring import FakeScorer


def oracle(state, question, width):
    text = state.lower()
    if 'crash' in text:
        raise KeyboardInterrupt('process killed')
    if 'boom' in text:
        raise RuntimeError('backend exploded')
    if 'maybe' in text:
        return [1 / width] * width
    wanted = 'refund' if 'money back' in question.lower() else 'invoice' if 'billing' in question.lower() else None
    return [.96, .04] if wanted and wanted in text else [.04, .96]


def collection_task(tmp_path, rows):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                                   'issues': {'select': 'Which issues does the customer raise?', 'max_claims': 1,
                                              'labels': {'refund': 'Money back for a charge', 'billing': 'A billing or invoice error'}}}})
    situations = tmp_path / 'situations.jsonl'
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in rows))
    return path, situations


def envelope(message, **meta):
    return {'schema': 3, 'input': {'message': message}, **meta}


def test_collection_accounts_every_target_and_publishes_reviewable_labels(tmp_path, monkeypatch):
    from semantic_switch.evidence import ROLES
    from semantic_learn import teacher as module
    path, situations = collection_task(tmp_path, [envelope('please refund me', groups=['a'], tags=['direct'], role='check', pairs=['p']),
                                                  envelope('wrong invoice total', groups=['b']), envelope('maybe something', groups=['c']),
                                                  envelope('boom', groups=['d']), envelope('please refund me', groups=['e'], role='check')])
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: teacher)
    joined, runtime = [], module.Runtime
    monkeypatch.setattr(module, 'Runtime', lambda models, **kwargs: joined.append(kwargs['job']) or runtime(models, **kwargs))
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    to = tmp_path / 'labels.collection.json'
    with model_jobs.model_job() as parent:
        report = module.collect(path, situations, to, teacher=TEACHER, training_use=declared('Example'), provider='Example', job=parent, window=2)
    counts = report['counts']
    assert joined == [parent] and counts['situations'] == 4 and counts['targets'] == 12 and counts['settled'] == 4
    assert counts['unresolved'] == {'unexamined': 4, 'missing': 0, 'failed': 2, 'rule': 0, 'uncertain': 2, 'drift': 0}
    assert counts['settled'] + sum(counts['unresolved'].values()) == counts['targets'] and counts['complete_situations'] == 0 and report['review'][:2] == ['ss', 'label']
    labels, unresolved = files.read_jsonl(report['labels']), files.read_jsonl(report['unresolved'])
    first = next(row for row in labels if row['input'] == {'message': 'please refund me'})
    assert first['facts']['refund'] is True and first['groups'] == ['a', 'e'] and first['tags'] == ['direct'] and first['role'] == 'check' and first['pairs'] == ['p']
    assert all(source['derivation'] == 'judged' and source['producer'] == TrainingUse.parse(declared('Example')).route and source['training_use'] == declared('Example')
               for row in labels for value in row['label_sources'].values() for history in (value.values() if isinstance(value, dict) else [value]) for source in history)
    assert files.read_published(to)['teacher']['training_use'] == declared('Example')
    assert {row['outcome'] for row in unresolved} <= {'unexamined', 'failed', 'uncertain'} and all(row['address'][0] in ('refund', 'issues') for row in unresolved)
    reviewed = tmp_path / 'reviewed.jsonl'
    with Runtime(FakeScorer(oracle)) as rt:
        assert tools.label(rt, path, report['labels'], to=str(reviewed), ask=lambda prompt: '', show=lambda text: None, reviewer='ana', actor=Actor('human')) == len(labels)
        exported = tools.export(rt, path, split='all', labels=reviewed)
    rows = read_export(exported['path'], fitting=False)[1]
    assert {row['label_derivation'] for row in rows} == {'judged'} and any(row['tags'] == ['direct'] for row in rows)
    assert all(source['verifications'][0]['verifier'] == 'ana' and source['verifications'][0]['actor'] == {'kind': 'human'}
               for row in rows for source in row['label_sources'])
    assert {(row['split'], row['fold']) for row in rows if row['group'] == 'group:a'} <= {(ROLES['check'][0], fold) for fold in ROLES['check'][1]}
    before = report['labels']
    again = module.collect(path, situations, to, teacher=TEACHER, training_use=declared('Example'), provider='Example')
    assert again['labels'] == before and Path(before).exists()


def test_collection_resumes_after_a_crash_and_failed_materialization_without_repeating_committed_work(tmp_path, monkeypatch):
    from semantic_learn import teacher as module
    rows = [envelope(f'please refund me {i}', groups=[f'g{i}']) for i in range(5)]
    path, situations = collection_task(tmp_path, [*rows[:3], envelope('crash now', groups=['x']), *rows[3:]])
    crashing = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: crashing)
    to = tmp_path / 'labels.collection.json'
    with pytest.raises(KeyboardInterrupt):
        module.collect(path, situations, to, teacher=TEACHER, training_use=declared(), window=1)
    committed = files.read_jsonl(tmp_path / 'labels.journal.jsonl')[1:]
    assert len(committed) == 3 and not to.exists()
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in [*rows[:3], envelope('calm now', groups=['x']), *rows[3:]]))
    with pytest.raises(ConfigError, match='different run'):
        module.collect(path, situations, to, teacher=TEACHER, training_use=declared(), window=1)
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in [*rows[:3], envelope('crash now', groups=['x']), *rows[3:]]))
    def resuming(state, question, width):
        return oracle(state.replace('crash', 'calm'), question, width)
    resumed = FakeScorer(resuming)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: resumed)
    original = module.materialize
    def failing(*args, **kwargs):
        raise ConfigError('materialization failed')
    monkeypatch.setattr(module, 'materialize', failing)
    with pytest.raises(ConfigError, match='materialization failed'):
        module.collect(path, situations, to, teacher=TEACHER, training_use=declared(), window=1)
    scored = len(resumed.requests)
    assert scored and len(files.read_jsonl(tmp_path / 'labels.journal.jsonl')) == 7 and not to.exists()
    monkeypatch.setattr(module, 'materialize', original)
    report = module.collect(path, situations, to, teacher=TEACHER, training_use=declared(), window=1)
    assert len(resumed.requests) == scored and report['counts']['situations'] == 6 and files.read_published(to)['execution']['completed'] == 6


def test_invalid_or_unversioned_sources_fail_before_any_teacher_request(tmp_path, monkeypatch):
    from semantic_learn import teacher as module
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: teacher)
    for rows, match in (([envelope('fine'), {'message': 'raw domain input'}], r'situations.jsonl:2: not a schema-3 situation envelope'),
                        ([envelope('fine'), {'input': {'message': 'x'}, 'facts': {'refund': True}}], r'situations.jsonl:2: not a schema-3'),
                        ([envelope('x', role='train')], r'situations.jsonl:1: role must be one of'),
                        ([envelope('x', role='fit'), envelope('x', role='check')], 'duplicate decision input with roles fit and check')):
        path, situations = collection_task(tmp_path, rows)
        with pytest.raises(ConfigError, match=match):
            module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared())
        assert not teacher.requests
    path, situations = collection_task(tmp_path, [envelope(f'case {i}') for i in range(3)])
    with pytest.raises(ConfigError, match='more than 2 situations exceed the acquisition budget'):
        module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), max_situations=2)
    assert not teacher.requests


def acquired(tmp_path, rows, **limits):
    from semantic_switch.training import Task
    path, situations = collection_task(tmp_path, rows)
    with Runtime(FakeScorer()) as runtime:
        return Task(runtime.load(path, draft=True)).situations(situations, **{'max_situations': 2, 'max_bytes': 1 << 20, **limits})


def test_unique_situation_limit_counts_keys_and_merges_duplicate_lineage(tmp_path):
    merged = acquired(tmp_path, [envelope('a', groups=['g1'], tags=['first']), envelope('b', groups=['g2']), envelope('a', groups=['g3'], pairs=['p'], tags=['again'])])
    assert [row['input'] for row in merged] == [{'message': 'a'}, {'message': 'b'}]
    assert merged[0]['groups'] == ['g1', 'g3'] and merged[0]['pairs'] == ['p'] and merged[0]['tags'] == ['again', 'first'] and merged[1]['groups'] == ['g2']
    assert [row['input'] for row in acquired(tmp_path, [envelope('a', groups=['g1']), envelope('a', groups=['g2']), envelope('b')])] == [{'message': 'a'}, {'message': 'b'}]
    with pytest.raises(ConfigError, match=r'situations.jsonl:3: .*more than 2 situations exceed'):
        acquired(tmp_path, [envelope('a'), envelope('b'), envelope('c')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:5: .*more than 2 situations exceed'):
        acquired(tmp_path, [envelope('a'), envelope('b'), envelope('a'), envelope('b'), envelope('c')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:3: duplicate decision input with roles fit and check'):
        acquired(tmp_path, [envelope('a', role='fit'), envelope('b'), envelope('a', role='check')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:2: duplicate decision input with roles fit and check'):
        acquired(tmp_path, [envelope('a', role='fit'), envelope('a', role='check')], max_situations=1)
    with pytest.raises(ConfigError, match='exceed the acquisition budget of .*10 bytes'):
        acquired(tmp_path, [envelope('a'), envelope('a')], max_bytes=10)
    with pytest.raises(ConfigError, match='no situations to label'):
        acquired(tmp_path, [])


def test_collection_materializes_duplicates_at_the_unique_limit(tmp_path, monkeypatch):
    from semantic_learn import teacher as module
    path, situations = collection_task(tmp_path, [envelope('please refund me', groups=['a']), envelope('wrong invoice total', groups=['b']),
                                                  envelope('please refund me', groups=['c'], tags=['repeat'])])
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: teacher)
    report = module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), max_situations=2)
    labels = files.read_jsonl(report['labels'])
    assert report['counts']['situations'] == 2 and len(labels) == 2
    refund = next(row for row in labels if row['input'] == {'message': 'please refund me'})
    assert refund['groups'] == ['a', 'c'] and refund['tags'] == ['repeat']


def test_shared_requests_count_once_and_returned_model_drift_is_an_outcome(tmp_path):
    from semantic_switch.judge import Request
    from semantic_switch.results import FactResult, Prediction
    from semantic_switch.training import describe_task
    from semantic_learn.teacher import unit
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text'}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'Money back.'}, 'legal': {'check': 'Legal action.'}}})
    targets = describe_task(path)['targets']
    shared = Request('r1', 1., 1, 200, {'prompt_tokens': 3})
    def fact(model):
        return FactResult(True, .9, True, 'judge', prediction=Prediction('value', 'true', True, .9), observed_models=(model,), requests=(shared,), status='ok')
    record = {'input': {'message': 'x'}}
    payload = unit(targets, record, {'refund': fact('m1'), 'legal': fact('m2')}, 'm1')
    assert len(payload['requests']) == 1 and payload['targets']['refund'] == {'value': True, 'models': ['m1']}
    assert payload['targets']['legal'] == {'outcome': 'drift', 'models': ['m2'], 'pinned': 'm1'} and payload['returned_models'] == ['m1', 'm2']


@pytest.mark.parametrize('teacher', ['openrouter:typesafe/jev-1.13', 'openrouter:TypeSafe/Jev-2'])
def test_collect_refuses_a_teacher_whose_terms_forbid_training_use(tmp_path, monkeypatch, teacher):
    from semantic_learn import teacher as module
    monkeypatch.setattr(module, 'judge', lambda *args, **kwargs: pytest.fail('a restricted teacher was contacted'))
    with pytest.raises(ConfigError, match=r'TypeSafe agreement §2.3\(b\).*gemma-4-31b-it'):
        module.collect(tmp_path / 'x.yaml', tmp_path / 'x.jsonl', tmp_path / 'labels.collection.json', teacher=teacher, training_use=declared())


@pytest.mark.parametrize('declaration, match', [({**declared(), 'permission': 'unknown'}, 'declared unknown for training use'),
                                                  ({**declared(), 'permission': 'forbidden'}, 'declared forbidden for training use'),
                                                  ({**declared(), 'model': 'typesafe/jev-1.13'}, r'cannot be declared allowed.*§2.3\(b\)'),
                                                  ({k: v for k, v in declared().items() if k != 'use'}, 'needs exactly')])
def test_collect_requires_an_allowed_training_use_declaration_before_any_request(tmp_path, monkeypatch, declaration, match):
    from semantic_learn import teacher as module
    monkeypatch.setattr(module, 'judge', lambda *args, **kwargs: pytest.fail('an undeclared teacher was contacted'))
    path, situations = collection_task(tmp_path, [envelope('please refund me', groups=['a'])])
    with pytest.raises(ConfigError, match=match):
        module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declaration)


def test_collected_labels_recheck_every_observed_producer_route_against_the_declaration(tmp_path):
    from semantic_switch.training import Task, TrainingUse
    from semantic_learn.teacher import materialize
    path, _ = collection_task(tmp_path, [])
    with Runtime(FakeScorer(oracle)) as runtime:
        task = Task(runtime.load(path, draft=True))
    targets = task.targets('refund')
    use = TrainingUse.parse(declared('Example', model='google/gemma-4-31b-it'))
    def payload(models):
        return {'record': envelope('please refund me', groups=['a'], pairs=[], selections=[]), 'requests': [],
                'targets': {'refund': {'value': True, 'models': models}}, 'collected_at': '2026-10-02T03:00:00+02:00'}
    labels, _, counts = materialize(task, targets, [payload(['google/gemma-4-31b-it@Example'])], {}, use, 'unused')
    assert counts['settled'] == 1 and labels[0]['label_sources']['refund'][0]['training_use'] == use.record()
    assert labels[0]['created'] == '2026-10-02T01:00:00+00:00'
    labels, _, _ = materialize(task, targets, [payload([])], {}, use, 'google/gemma-4-31b-it@Example')
    assert labels[0]['label_sources']['refund'][0]['training_use'] == use.record()
    for models, route, match in ((['google/gemma-4-31b-it@Other'], 'unused', 'differs from the declared google/gemma-4-31b-it@Example'),
                                 (['typesafe/jev-1.13-20260917@TypeSafe'], 'unused', r'cannot label training data: TypeSafe agreement'),
                                 ([], 'openrouter:google/gemma-4-31b-it', 'differs from the declared')):
        with pytest.raises(ConfigError, match=match):
            materialize(task, targets, [payload(models)], {}, use, route)


def special_oracle(state, question, width):
    text = state.lower()
    if 'maybe' in text:
        return [1 / width] * width
    if width == 3 and 'ribbon' in question.lower():
        return [.98, .01, .01] if 'loaded' in text else [.01, .98, .01] if 'empty' in text else [.005, .005, .99]
    return [.005, .005, .99] if 'weather' in text else [.98, .01, .01]


def test_declared_semantic_unknowns_become_labels_while_uncertain_and_missing_stay_unresolved(tmp_path, monkeypatch):
    from semantic_learn import teacher as module
    path = tmp_path / 'special.yaml'
    files.publish(path, {'name': 'special', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'ribbon': {'check': 'A ribbon is loaded.', 'unstated': True},
                                   'topic': {'pick': 'Which topic does the message raise?', 'options': {'refund': 'Money back', 'billing': 'Invoice error'},
                                             'allow_none': True}}})
    situations = tmp_path / 'situations.jsonl'
    files.publish(situations, ''.join(json.dumps(envelope(message, groups=[str(i)])) + '\n'
                                      for i, message in enumerate(['ribbon loaded, refund', 'ribbon empty', 'weather report', 'maybe'])))
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: FakeScorer(special_oracle))
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    report = module.collect(path, situations, tmp_path / 'special.collection.json', teacher=TEACHER, training_use=declared('Example'), provider='Example')
    labels = {row['input']['message']: row['facts'] for row in files.read_jsonl(report['labels'])}
    assert labels['ribbon loaded, refund']['ribbon'] is True and labels['ribbon empty']['ribbon'] is False
    assert labels['weather report'] == {'ribbon': 'unstated', 'topic': 'none'}
    assert 'maybe' not in labels and report['counts']['unresolved']['uncertain'] == 2
    assert None not in {value for facts in labels.values() for value in facts.values()}
    with Runtime(FakeScorer(special_oracle)) as rt:
        exported = tools.export(rt, path, split='all', labels=report['labels'])
    rows = read_export(exported['path'], fitting=False)[1]
    classes = {(row['fact'], row['classes'][row['label']]) for row in rows}
    assert {('ribbon', 'unstated'), ('topic', 'none')} <= classes
    with pytest.raises(ConfigError, match='differs'):
        module.PROJECTION = 'declared-answers@1'
        try:
            module.collect(path, situations, tmp_path / 'special.collection.json', teacher=TEACHER, training_use=declared('Example'), provider='Example')
        finally:
            module.PROJECTION = 'declared-answers@2'
