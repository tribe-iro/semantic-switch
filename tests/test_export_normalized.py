from __future__ import annotations

from pathlib import Path

import pytest

from semantic_switch import Actor, ConfigError, Pick, Runtime, files, tools
from semantic_switch.artifacts import native_score_requests, read_export

from semantic_testkit.labels import envelopes, republish
from semantic_testkit.scoring import FakeScorer


def gated(tmp_path, records):
    path = tmp_path / 'route.yaml'
    files.publish(path, {'name': 'route', 'fields': {'message': {'type': 'text', 'required': True}, 'tier': {'type': 'text'}}, 'reads': ['message'],
                         'facts': {'action': {'pick': 'Which response does the policy require?',
                                              'options': {'escalate': {'description': 'Request a technician', 'requires': "tier == 'gold'"},
                                                          'continue': 'Continue operation', 'clarify': 'Ask for details'}}}})
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes(records)))
    return path


def test_gold_excluded_by_full_stage_eligibility_refuses_before_publishing(tmp_path):
    path = gated(tmp_path, [{'input': {'message': f'case {i}', 'tier': 'gold'}, 'facts': {'action': 'escalate'}, 'group': f'g{i}'} for i in range(3)]
                 + [{'input': {'message': 'case x', 'tier': 'silver'}, 'facts': {'action': 'escalate'}, 'group': 'gx'}])
    for native in (True, False):
        with Runtime(FakeScorer()) as runtime, pytest.raises(ConfigError, match=r"observation [0-9a-f]{64} target action stage full: gold 'escalate'.*requires \"tier == 'gold'\""):
            tools.export(runtime, path, split='all', native=native)
        assert not list(tmp_path.glob('route.export*'))


def test_focus_escape_gold_and_permutations_survive(tmp_path):
    with Runtime(FakeScorer()) as runtime:
        item = Pick('Choose the referenced part.', {'ax': 'AX', 'by': 'BY', 'cz': 'CZ'}, focus='mentions')
        path = tmp_path / 'part.yaml'
        switch = item.save(path, rt=runtime)
        switch.correct('Only AX is mentioned here.', fact='part', value='cz', actor=Actor('human'))
        report = tools.export(runtime, path, split='all', native=True)
    _, (row,) = read_export(report['path'], fitting=False, native=True)
    stages = {(entry.stage, entry.variant): entry for entry in row['presentations']}
    assert {'mentions', 'fallback'} <= {stage for stage, _ in stages}
    for entry in row['presentations']:
        assert entry.slot_map[entry.label] == (row['label'] if row['label'] in entry.slot_map else -1)
    assert stages[('mentions', 'canonical')].slot_map[stages[('mentions', 'canonical')].label] == -1
    assert sorted(stages[('fallback', 'permuted')].slot_map) == sorted(stages[('fallback', 'canonical')].slot_map)


def normalized(tmp_path, count=8):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                                   'topic': {'pick': 'What is the message mainly about?', 'options': {'charges': 'charges', 'login': 'login'}}}})
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes(
        [{'input': {'message': f'refund the charge {i}'}, 'facts': {'refund': bool(i % 2), 'topic': ('charges', 'login')[i % 2]}, 'group': f'g{i}'}
         for i in range(count)])))
    with Runtime(FakeScorer()) as runtime:
        report = tools.export(runtime, path, split='all', native=True, task_family='example:support')
    return report, *read_export(report['path'], fitting=False, native=True)


def test_export_interns_descriptors_segments_and_shares_views(tmp_path):
    report, manifest, rows = normalized(tmp_path)
    counts = manifest['counts']
    presentations = [entry for row in rows for entry in row['presentations']]
    assert counts['observations'] == len(rows) == 16 and len(manifest['targets']) == 2 and counts['inputs'] == 8
    assert counts['segments'] < 3 * len(presentations) and counts['answers'] < len(presentations)
    assert len({id(entry.prefix) for entry in presentations}) == 1 and len({id(row['native']) for row in rows}) == 1
    by_input = {}
    for row in rows:
        by_input.setdefault(row['text'], set()).update(id(entry.input) for entry in row['presentations'])
    assert all(len(found) == 1 for found in by_input.values())
    assert all(row['task_family'] == 'example:support' and row['value'] == row['classes'][row['label']] or row['kind'] == 'check' for row in rows)
    requests = native_score_requests(rows)
    assert len(requests) == len(presentations) and requests[0].prefix == presentations[0].prefix
    assert len({entry.id for entry in presentations}) == len(presentations)


def test_logical_identity_ignores_layout_and_tracks_content(tmp_path):
    report, manifest, rows = normalized(tmp_path)
    copied = republish(tmp_path / 'copy' / 'other.manifest.json', manifest, list(reversed(rows)))
    assert copied['export_id'] != manifest['export_id']
    same = republish(tmp_path / 'same' / 'renamed.manifest.json', manifest, rows)
    assert same['export_id'] == manifest['export_id'] == report['export_id']
    _, again = read_export(tmp_path / 'same' / 'renamed.manifest.json', fitting=False, native=True)
    assert [[(e.id, e.prefix + e.input + e.question, e.answers, e.slot_map, e.label) for e in row['presentations']] for row in again] == \
           [[(e.id, e.prefix + e.input + e.question, e.answers, e.slot_map, e.label) for e in row['presentations']] for row in rows]
    from dataclasses import replace
    changed = [{**rows[0], 'presentations': (replace(rows[0]['presentations'][0], question=rows[0]['presentations'][0].question + (1,)),
                                             *rows[0]['presentations'][1:])}, *rows[1:]]
    edited = republish(tmp_path / 'edited' / 'x.manifest.json', manifest, changed)
    _, found = read_export(tmp_path / 'edited' / 'x.manifest.json', fitting=False, native=True)
    assert edited['export_id'] != manifest['export_id'] and found[0]['presentations'][0].id != rows[0]['presentations'][0].id


def tables(path):
    manifest = files.read_published(path)
    return manifest, {name: Path(path).with_name(entry['file']) for name, entry in manifest['payloads'].items()}


@pytest.mark.parametrize('kind, before, after, match', [
    ('observations', '"target": 0', '"target": 9', 'target reference 9 outside its table'),
    ('segments', ']', ', -1]', 'invalid native token range'),
    ('observations', '"split": "tune"', '"split": "held"', 'invalid partition')])
def test_reader_refuses_range_and_schema_corruption(tmp_path, kind, before, after, match):
    _, manifest, rows = normalized(tmp_path)
    target = tmp_path / 'forged' / 'forged.manifest.json'
    republish(target, manifest, rows)
    found, payloads = tables(target)
    name = next(name for name in payloads if f'.{kind}.' in name)
    text = payloads[name].read_text()
    assert before in text
    forged = {key: value for key, value in found.items() if key != 'payloads'}
    files.publish(target, forged, {key: [path.read_text().replace(before, after, 1) if key == name else path.read_text()] for key, path in payloads.items()})
    with pytest.raises(ConfigError, match=match):
        read_export(target, fitting=False, native=True)
