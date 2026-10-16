
from semantic_switch import model_jobs
import json

from semantic_switch import api, files
from semantic_testkit.labels import TEACHER, declared
from semantic_testkit.scoring import FakeScorer


def oracle(state, question, width):
    return [.96, .04] if 'refund' in state.lower() else [.04, .96]


def collected(tmp_path, monkeypatch, count):
    from semantic_learn import teacher as module
    definition = tmp_path / 'support.yaml'
    files.publish(definition, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                               'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}})
    situations = tmp_path / 'situations.jsonl'
    files.publish(situations, ''.join(json.dumps({'schema': 3, 'input': {'message': f'case {i} refund'}, 'groups': [f'g{i}']}) + '\n' for i in range(count)))
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: FakeScorer(oracle))
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    report = module.collect(definition, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared('Example'), provider='Example')
    return definition, report['manifest'] if 'manifest' in report else tmp_path / 'labels.collection.json'


def export(tmp_path, definition, manifest, name, **settings):
    from semantic_learn.review import review_export
    out = review_export(definition, tmp_path / f'{name}.review.json', collected=[manifest], **settings)
    return out, files.read_jsonl(out['queue'])


def test_round_robin_allocation_cannot_be_starved_and_records_every_reason(tmp_path, monkeypatch):
    from semantic_learn import review
    definition, manifest = collected(tmp_path, monkeypatch, 8)
    hits = {f'case {i} refund': ({'feedback'} if i < 6 else set()) | ({'conflict'} if i >= 5 else set()) for i in range(8)}
    monkeypatch.setattr(review.Population, 'strata', lambda self, unit: hits[unit['envelope']['input']['message']])
    out, rows = export(tmp_path, definition, manifest, 'a', size=4, audit=0, strata=('feedback', 'conflict'))
    assert out['allocator'] == review.ALLOCATOR and len(rows) == 4
    stratum = [entry['stratum'] for entry in out['allocation']]
    assert stratum.count('feedback') == 2 and stratum.count('conflict') == 2 and len({entry['unit'] for entry in out['allocation']}) == 4
    assert any(entry['reasons'] == ['conflict', 'feedback'] for entry in out['allocation']) or all(len(entry['reasons']) == 1 for entry in out['allocation'])
    assert {row['selections'][-1]['selected_by'] for row in rows} == {'targeted'}
    again, replayed = export(tmp_path, definition, manifest, 'b', size=4, audit=0, strata=('feedback', 'conflict'))
    assert again['id'] == out['id'] and [row['input'] for row in replayed] == [row['input'] for row in rows]
    hits['case 0 refund'] = {'conflict'}
    changed, _ = export(tmp_path, definition, manifest, 'c', size=4, audit=0, strata=('feedback', 'conflict'))
    assert changed['id'] != out['id']


def test_exhausted_queues_redistribute_and_audit_stays_random(tmp_path, monkeypatch):
    from semantic_learn import review
    definition, manifest = collected(tmp_path, monkeypatch, 8)
    monkeypatch.setattr(review.Population, 'strata', lambda self, unit: {'feedback'} if unit['envelope']['input']['message'] == 'case 1 refund' else {'conflict'})
    out, rows = export(tmp_path, definition, manifest, 'd', size=5, audit=2, strata=('feedback', 'conflict'))
    selections = [row['selections'][-1] for row in rows]
    assert len(rows) == 5 and sum(s['selected_by'] == 'random' for s in selections) == 2
    assert all('inclusion_probability' not in s for s in selections if s['selected_by'] == 'targeted')
    assert len(out['allocation']) == 3
