from __future__ import annotations

import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import digest
from semantic_switch.examples import ROLES, LabelSource, Review, each_target, label_record, reviewed_content, role_of

from .conftest import FakeScorer
from .test_classification import LABELS, saved

TEACHER = LabelSource('synthetic', 'openrouter:google/gemma-4-31b-it', digest({'model': 'gemma-4-31b-it'}), address=('teacher',)).record()


def switch(tmp_path):
    return saved(tmp_path, {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                            'issues': {'select': 'Which issues does the customer raise?', 'labels': LABELS}})


def records(sw, n=24, source=TEACHER, **extra):
    return [label_record(sw.plan, {'message': f'case {i}'}, facts, sources=each_target(facts, [source]), groups=[f'situation-{i}'], pairs=[f'pair-{i // 2}'],
                         tags=['relayed'], **extra)
            for i in range(n) for facts in ({'refund': bool(i % 2), 'issues': {'refund': bool(i % 2), 'billing': False}},)]


def publish(path, rows):
    files.publish(path, ''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_label_file_exports_without_touching_corrections_or_tests(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        sw = runtime.load(path, draft=True)
        labels = publish(tmp_path / 'synthetic.jsonl', records(sw))
        exported = tools.export(runtime, path, split='all', native=True, labels=labels)
    _, rows = read_export(exported['path'], fitting=False, native=True)
    assert not path.with_suffix('.test.jsonl').exists() and not path.with_suffix('.examples.jsonl').exists()
    assert {row['label_origin'] for row in rows} == {'synthetic'} and all(row['tags'] == ['relayed'] for row in rows)
    members = [row for row in rows if row.get('member')]
    assert {row['member'] for row in members} == {'refund', 'billing'} and len(members) == 48
    assert all(row['pairs'] for row in rows) and {row['fact'] for row in rows} == {'refund', 'issues'}


@pytest.mark.parametrize('change, error', [
    (lambda row: row.update(revs={**row['revs'], 'refund': 'stale'}), 'refund: meaning changed'),
    (lambda row: row['revs']['issues'].update(refund='stale'), 'issues: meaning changed'),
    (lambda row: row['label_sources'].update(refund=[{'origin': 'teacher', 'producer': 'x'}]), 'label source origin must be one of'),
    (lambda row: row['label_sources'].pop('refund'), 'label_sources must mirror facts'),
    (lambda row: row.pop('schema'), 'not a schema-2 situation envelope'),
    (lambda row: row.update(case='refund'), 'record case answers'),
    (lambda row: row.update(extra=True), 'unsupported label fields'),
    (lambda row: row.update(role='train'), 'role must be one of fit, model_select, acceptance_fit, check'),
])
def test_label_file_import_rejects_stale_or_ambiguous_records(tmp_path, change, error):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        rows = records(runtime.load(path, draft=True))
        change(rows[3])
        with pytest.raises(ConfigError, match=f'synthetic.jsonl:4: .*{error}'):
            tools.export(runtime, path, labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_conflicting_duplicate_labels_fail(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        rows = records(runtime.load(path, draft=True))
        rows.append({**rows[0], 'facts': {'refund': not rows[0]['facts']['refund']}, 'revs': {'refund': rows[0]['revs']['refund']},
                     'label_sources': {'refund': [TEACHER]}})
        with pytest.raises(ConfigError, match='conflicting labels'):
            tools.export(runtime, path, labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_review_confirms_replaces_and_skips_with_provenance(tmp_path):
    path = switch(tmp_path)
    answers = iter(['', 'y', 's', '', '', 's'])
    with Runtime({'judge': FakeScorer()}) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', records(runtime.load(path, draft=True), n=2))
        written = tools.label(runtime, path, str(supplied), to=str(tmp_path / 'reviewed.jsonl'), ask=lambda prompt: next(answers), show=lambda text: None,
                              reviewer='ana')
    reviewed = files.read_jsonl(tmp_path / 'reviewed.jsonl')
    assert written == 2 and not path.with_suffix('.test.jsonl').exists()
    assert reviewed[0]['facts'] == {'refund': False, 'issues': {'refund': True}}
    assert reviewed[1]['facts'] == {'refund': True, 'issues': {'refund': True}}
    confirmed, replaced = reviewed[0]['label_sources']['refund'], reviewed[0]['label_sources']['issues']['refund']
    assert [s['origin'] for s in confirmed] == ['synthetic'] and confirmed[0]['producer'] == TEACHER['producer']
    assert confirmed[0]['reviews'] == [{'kind': 'label', 'reviewer': 'ana', 'content': reviewed_content('label', {'message': 'case 0'}, ('refund',), False), 'outcome': 'confirm'}]
    assert [s['origin'] for s in replaced] == ['synthetic', 'human'] and replaced[1] == {'origin': 'human', 'producer': 'ana', 'replaces': False}
    assert replaced[0]['reviews'][0]['outcome'] == 'replace'
    skips = files.read_jsonl(tmp_path / 'reviewed.reviews.jsonl')
    assert [(row['address'], row['review']['outcome']) for row in skips] == [(['issues', 'billing'], 'skip'), (['issues', 'billing'], 'skip')]
    assert set(reviewed[0]['revs']) == {'refund', 'issues'} and set(reviewed[0]['revs']['issues']) == {'refund'}
    with Runtime({'judge': FakeScorer()}) as runtime:
        exported = tools.export(runtime, path, split='all', labels=tmp_path / 'reviewed.jsonl')
    _, rows = read_export(exported['path'], fitting=False)
    origins = {(row['bound_input']['message'], row['fact'], row.get('member')): row['label_origin'] for row in rows}
    assert origins == {('case 0', 'refund', None): 'synthetic', ('case 0', 'issues', 'refund'): 'human',
                       ('case 1', 'refund', None): 'synthetic', ('case 1', 'issues', 'refund'): 'synthetic'}


def test_heterogeneous_origins_merge_sources_and_keep_annotations_out_of_model_input(tmp_path):
    path = switch(tmp_path)
    rule = LabelSource('rule', 'bench.reference@1', digest('reference'), digest('frame'), ('frame-1', 'refund')).record()
    with Runtime({'judge': FakeScorer()}) as runtime:
        plan = runtime.load(path, draft=True).plan
        facts = {'refund': True, 'issues': {'refund': True}}
        mixed = label_record(plan, {'message': 'refund please'}, facts, sources={'refund': [rule], 'issues': {'refund': [TEACHER]}}, groups=['a'])
        again = label_record(plan, {'message': 'refund please'}, {'refund': True}, sources={'refund': [LabelSource('unknown', 'unknown').record()]}, groups=['a'])
        evidence = LabelSource('rule', 'bench.reference@1').reviewed(Review('evidence', 'ana', reviewed_content('evidence', {'message': 'refund please'}, ()), 'confirm')).record()
        reviewed = label_record(plan, {'message': 'refund please'}, {'refund': True}, sources={'refund': [evidence]}, groups=['a'])
        exported = tools.export(runtime, path, split='all', native=True, labels=publish(tmp_path / 'labels.jsonl', [mixed, again, reviewed]))
        with pytest.raises(ConfigError, match='evidence review by ana covers other content'):
            label_record(plan, {'message': 'refund please now'}, {'refund': True}, sources={'refund': [evidence]})
    _, rows = read_export(exported['path'], fitting=False, native=True)
    refund = next(row for row in rows if row['fact'] == 'refund')
    member = next(row for row in rows if row.get('member') == 'refund')
    assert refund['label_origin'] == 'mixed' and sorted(s['origin'] for s in refund['label_sources']) == ['rule', 'rule', 'unknown']
    assert len({json.dumps(s, sort_keys=True) for s in refund['label_sources']}) == 3
    assert member['label_origin'] == 'synthetic' and member['label_sources'] == [TEACHER]
    assert len(refund['situations']) == 1 == len(exported_situations(exported)) and refund['situations'] == member['situations']
    for row in rows:
        visible = json.dumps([row['bound_input'], row['text'], row['native']['presentations']])
        assert all(hidden not in visible for hidden in ('bench.reference', 'gemma-4-31b', 'teacher', 'frame-1'))


def exported_situations(exported):
    return files.read_published(exported['path'])['situations']


def test_review_refuses_its_own_inputs_as_destination(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', records(runtime.load(path, draft=True), n=1))
        with pytest.raises(ConfigError, match='must differ'):
            tools.label(runtime, path, str(supplied), to=str(supplied))
        with pytest.raises(ConfigError, match='examples, test or a label file'):
            tools.label(runtime, path, str(supplied), to='elsewhere')


def roled(sw, n=24, roled=8):
    return [{**row, 'role': list(ROLES)[i // 2 % len(ROLES)]} if i // 2 < roled else row for i, row in enumerate(records(sw, n))]


def assignment(row):
    return role_of(row['split'], row['fold'])


def test_explicit_roles_assign_families_and_unroled_families_keep_the_hash(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        labels = publish(tmp_path / 'synthetic.jsonl', roled(runtime.load(path, draft=True)))
        rows = read_export(tools.export(runtime, path, split='all', labels=labels)['path'], fitting=False)[1]
        tuned = read_export(tools.export(runtime, path, to=tmp_path / 'tune.jsonl', labels=labels)['path'])[1]
    given = {f"group:situation-{i}": role for i, role in enumerate(r.get('role') for r in files.read_jsonl(labels)) if i % 2 == 0}
    for row in rows:
        if given[row['group']] is not None:
            assert assignment(row) == given[row['group']]
        else:
            assert row['split'] == ('tune' if int(digest(['split', row['group']])[:8], 16) % 2 == 0 else 'check')
            assert row['fold'] == int(digest(['fold', row['group']])[:8], 16) % 5
    assert {assignment(row) for row in rows if given[row['group']]} == set(ROLES)
    assert tuned and all(given[row['group']] != 'check' and row['split'] == 'tune' for row in tuned)


def test_conflicting_roles_within_a_family_fail_naming_it(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        rows = records(runtime.load(path, draft=True))
        rows[0], rows[1] = {**rows[0], 'role': 'fit'}, {**rows[1], 'role': 'check'}
        with pytest.raises(ConfigError, match='family group:situation-0 has label records with roles check, fit; give every record'):
            tools.export(runtime, path, split='all', labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_manifest_keeps_explicit_roles_and_refuses_a_changed_role(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        rows = roled(runtime.load(path, draft=True))
        test = publish(path.with_suffix('.test.jsonl'), rows)
        first, again = (tools.export(runtime, path, split='all', to=tmp_path / f'{n}.jsonl') for n in ('first', 'again'))
        manifest = files.read_published(path.with_suffix('.dataset.json'))
        assert first['dataset_id'] == again['dataset_id'] == manifest['id']
        assert [assignment(manifest['families'][f'group:situation-{i}']) for i in range(0, 16, 2)] == [list(ROLES)[i % len(ROLES)] for i in range(8)]
        publish(test, [{**row, 'role': 'check'} if row['groups'][0] in ('situation-0', 'situation-1') else row for row in rows])
        with pytest.raises(ConfigError, match='family group:situation-0 was materialized as fit in .*dataset.json but its label records now assign role check; connected evidence keeps its first role'):
            tools.export(runtime, path, split='all', to=tmp_path / 'moved.jsonl')
        path.with_suffix('.dataset.json').unlink()
        rows = read_export(tools.export(runtime, path, split='all', to=tmp_path / 'rebuilt.jsonl')['path'], fitting=False)[1]
    assert {assignment(row) for row in rows if row['group'] == 'group:situation-0'} == {'check'}


def test_review_keeps_the_role_in_label_files_and_refuses_to_drop_it(tmp_path):
    path = switch(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', [{**row, 'role': 'check'} for row in records(runtime.load(path, draft=True), n=1)])
        tools.label(runtime, path, str(supplied), to=str(tmp_path / 'reviewed.jsonl'), ask=lambda prompt: '', show=lambda text: None)
        assert files.read_jsonl(tmp_path / 'reviewed.jsonl')[0]['role'] == 'check'
        with pytest.raises(ConfigError, match="role 'check' belongs to label files"):
            tools.label(runtime, path, str(supplied), ask=lambda prompt: '', show=lambda text: None)
