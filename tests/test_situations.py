from __future__ import annotations

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import export_identity, read_export

from semantic_testkit.labels import envelopes, republish
from semantic_testkit.scoring import FakeScorer


def definition(tmp_path):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}, 'channel': {'type': 'text'}},
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.', 'reads': ['message']},
                                   'urgent': {'check': 'The customer needs an answer today.', 'reads': ['message', 'channel']}}})
    return path


def exported(tmp_path, records, **options):
    path = definition(tmp_path)
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes(records)))
    with Runtime(FakeScorer()) as rt:
        report = tools.export(rt, path, split='all', **options)
    return report, *read_export(report['path'], fitting=False)


def test_frames_sharing_a_target_projection_keep_both_situation_memberships(tmp_path):
    report, manifest, rows = exported(tmp_path, [
        {'input': {'message': 'refund my order', 'channel': 'email'}, 'facts': {'refund': True, 'urgent': False}, 'group': 'a'},
        {'input': {'message': 'refund my order', 'channel': 'phone'}, 'facts': {'refund': True, 'urgent': True}, 'group': 'b'}])
    refund = [row for row in rows if row['fact'] == 'refund']
    assert len(refund) == 1 and len(refund[0]['situations']) == 2 and len(manifest['situations']) == 2
    assert {s['id'] for s in manifest['situations']} == set(refund[0]['situations'])
    for situation in manifest['situations']:
        linked = {tuple(t['address']): t['observation'] for t in situation['targets']}
        assert linked[('refund',)] == refund[0]['id'] and situation['complete']
    assert report['situations'] == report['complete_situations'] == 2 and manifest['scope'] == [['refund'], ['urgent']]


def test_partial_gold_is_not_complete_and_identical_public_inputs_share_one_situation(tmp_path):
    _, manifest, rows = exported(tmp_path, [
        {'input': {'message': 'refund please', 'channel': 'chat'}, 'facts': {'refund': True}, 'group': 'a'},
        {'input': {'message': 'refund please', 'channel': 'chat'}, 'facts': {'urgent': False}, 'group': 'a2'},
        {'input': {'message': 'hello', 'channel': 'chat'}, 'facts': {'refund': False}, 'group': 'c'}])
    situations = {tuple(sorted('/'.join(t['address']) for t in s['targets'] if 'observation' in t)): s for s in manifest['situations']}
    merged = situations[('refund', 'urgent')]
    assert merged['complete'] and len(merged['records']) == 2
    partial = situations[('refund',)]
    assert not partial['complete'] and [t.get('outcome') for t in partial['targets']] == [None, 'unlabelled']


def test_export_identity_binds_header_and_rows_and_ignores_locations(tmp_path):
    report, manifest, rows = exported(tmp_path, [
        {'input': {'message': 'refund my order', 'channel': 'email'}, 'facts': {'refund': True, 'urgent': False}, 'group': 'a'}])
    assert manifest['export_id'] == export_identity(manifest, [row['digest'] for row in rows], manifest['situations']) == report['export_id']
    commit = tmp_path / 'support.export.jsonl.manifest.json'
    moved = tmp_path / 'elsewhere'
    moved.mkdir()
    for entry in manifest['payloads'].values():
        (tmp_path / entry['file']).rename(moved / entry['file'])
    commit.rename(moved / 'any-name.manifest.json')
    assert read_export(moved / 'any-name.manifest.json', fitting=False)[0]['export_id'] == manifest['export_id']
    assert republish(tmp_path / 'copy.jsonl.manifest.json', manifest, rows)['export_id'] == manifest['export_id']
    for change, match in (({'scope': [['refund']]}, 'targets differ from the declared scope'),
                          ({'situations': [{**manifest['situations'][0], 'complete': False}]}, 'gold-completeness')):
        republish(tmp_path / 'bad.jsonl.manifest.json', manifest, rows, **change)
        with pytest.raises(ConfigError, match=match):
            read_export(tmp_path / 'bad.jsonl.manifest.json', fitting=False)
    taught = republish(tmp_path / 'taught.jsonl.manifest.json', manifest, [{**rows[0], 'taught': True}, *rows[1:]])
    assert taught['export_id'] != manifest['export_id']
    forged = {**files.read_published(tmp_path / 'taught.jsonl.manifest.json'), 'export_id': manifest['export_id']}
    files.publish(tmp_path / 'taught.jsonl.manifest.json', forged)
    with pytest.raises(ConfigError, match='export identity mismatch'):
        read_export(tmp_path / 'taught.jsonl.manifest.json', fitting=False)
    copied = tmp_path / files.read_published(tmp_path / 'copy.jsonl.manifest.json')['payloads']['copy.observations.jsonl']['file']
    copied.write_text(copied.read_text().replace('"taught": false', '"taught": true', 1))
    with pytest.raises(ConfigError, match='incomplete publication'):
        read_export(tmp_path / 'copy.jsonl.manifest.json', fitting=False)
    stale = {**files.read_published(moved / 'any-name.manifest.json'), 'schema': 13}
    files.publish(moved / 'any-name.manifest.json', stale)
    with pytest.raises(ConfigError, match='unsupported export manifest'):
        read_export(moved / 'any-name.manifest.json', fitting=False)


def test_envelope_merge_unions_lineage_and_partial_targets_and_never_keeps_the_last_conflicting_label(tmp_path):
    from semantic_switch.training import Actor, LabelSource, Task
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                                   'issues': {'select': 'Which issues does the customer raise?', 'max_claims': 1,
                                              'labels': {'refund': 'Money back for a charge', 'billing': 'A billing or invoice error'}}}})
    with Runtime() as runtime:
        task = Task(runtime.load(path, draft=True))
    teacher, other = LabelSource('judged', 'teacher-a'), LabelSource('judged', 'teacher-b')
    first = task.label({'message': 'refund it'}, {'refund': True}, sources={'refund': [teacher]}, groups=['thread:t1'],
                       selections=[{'selected_by': 'random', 'inclusion_probability': .5, 'selection_id': 's1'}], created='2026-01-02T00:00:00+00:00')
    second = task.label({'message': 'refund it'}, {'issues': {'billing': False}, 'refund': True}, sources={'issues': {'billing': [other]}, 'refund': [other]},
                        groups=['entity:e1'], pairs=['p'], tags=['seen'], created='2026-01-01T00:00:00+00:00')
    captured = task.situation({'message': 'refund it'}, groups=['thread:t2'], selections=[{'selected_by': 'random', 'inclusion_probability': 1., 'selection_id': 's2'}])
    forward = task.merge(task.merge(first, second), captured)
    backward = task.merge(captured, task.merge(second, first))
    for merged in (forward, backward):
        assert merged['facts'] == {'refund': True, 'issues': {'billing': False}} and set(merged['groups']) == {'thread:t1', 'thread:t2', 'entity:e1'}
        assert merged['pairs'] == ['p'] and merged['tags'] == ['seen'] and {s['selection_id'] for s in merged['selections']} == {'s1', 's2'}
        assert {s['producer'] for s in merged['label_sources']['refund']} == {'teacher-a', 'teacher-b'} and merged['created'] == '2026-01-02T00:00:00+00:00'
        assert [s['producer'] for s in merged['label_sources']['issues']['billing']] == ['teacher-b']
    contrary = task.label({'message': 'refund it'}, {'refund': False, 'issues': {'refund': True}}, sources={'refund': [other], 'issues': {'refund': [other]}})
    with pytest.raises(ConfigError, match='refund: competing labels'):
        task.merge(forward, contrary)
    held = task.accumulator(max_situations=1, max_claims=10, max_bytes=1 << 20)
    for record in (forward, contrary, first):
        held.add(task.unit(record))
    kept, unresolved = held.settle(task.key({'message': 'refund it'}), strict=False)
    assert set(unresolved) == {'refund'} and kept['facts'] == {'issues': {'billing': False, 'refund': True}}
    assert sorted(claim.value for claim in unresolved['refund'].alternatives) == [False, True]
    corrected = task.label({'message': 'refund it'}, {'refund': False},
                           sources={'refund': [teacher, LabelSource('authored', 'reviewer', actor=Actor('human'), replaces=True)]})
    for merged in (task.merge(first, corrected), task.merge(corrected, first)):
        assert merged['facts'] == {'refund': False} and merged['label_sources']['refund'][-1]['derivation'] == 'authored'
    with pytest.raises(ConfigError, match='roles check and None'):
        task.merge(task.situation({'message': 'refund it'}, role='check'), captured)
    with pytest.raises(ConfigError, match='one decision input'):
        task.merge(captured, task.situation({'message': 'other'}))
