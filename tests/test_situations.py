from __future__ import annotations

import hashlib
import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import export_identity, read_export

from .conftest import FakeScorer, envelopes


def definition(tmp_path):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}, 'channel': {'type': 'text'}},
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.', 'reads': ['message']},
                                   'urgent': {'check': 'The customer needs an answer today.', 'reads': ['message', 'channel']}}})
    return path


def exported(tmp_path, records, **options):
    path = definition(tmp_path)
    files.append(path.with_suffix('.test.jsonl'), envelopes(records))
    with Runtime({'judge': FakeScorer()}) as rt:
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
    assert manifest['export_id'] == export_identity(manifest) == report['export_id']
    commit = tmp_path / 'support.export.jsonl.manifest.json'
    payload = commit.with_name(manifest['payloads']['support.export.jsonl']['file'])
    moved = tmp_path / 'elsewhere'
    moved.mkdir()
    payload.rename(moved / payload.name)
    commit.rename(moved / 'any-name.manifest.json')
    assert read_export(moved / 'any-name.manifest.json', fitting=False)[0]['export_id'] == manifest['export_id']
    header = {k: v for k, v in manifest.items() if k not in ('payloads', 'export_id')}
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for row in rows]
    for change, match in (({'scope': [['refund']]}, 'targets differ from the declared scope'),
                          ({'situations': [{**manifest['situations'][0], 'complete': False}]}, 'gold-completeness'),
                          ({'export_id': 'f' * 64}, 'export identity mismatch')):
        tampered = {**header, **change}
        tampered['export_id'] = change.get('export_id', export_identity(tampered))
        files.publish(tmp_path / 'bad.jsonl.manifest.json', tampered, {'bad.jsonl': lines})
        with pytest.raises(ConfigError, match=match):
            read_export(tmp_path / 'bad.jsonl.manifest.json', fitting=False)
    edited = [lines[0].replace('"taught": false', '"taught": true')]
    files.publish(tmp_path / 'edited.jsonl.manifest.json', {**header, 'export_id': manifest['export_id']}, {'edited.jsonl': edited})
    with pytest.raises(ConfigError, match='row bytes differ'):
        read_export(tmp_path / 'edited.jsonl.manifest.json', fitting=False)
    assert hashlib.sha256(lines[0].encode()).hexdigest() == manifest['rows'][0]['sha256']
