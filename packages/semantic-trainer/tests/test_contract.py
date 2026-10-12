from pathlib import Path
import json

import pytest

from semantic_switch import Runtime, files, tools, ConfigError
from semantic_switch.artifacts import load_bundle, read_export
from semantic_switch.tools.data import Labels
from semantic_trainer.lexical import train
from tests.conftest import FakeScorer
from tests.conftest import envelopes


def test_trainer_consumes_the_same_feature_fixture():
    from semantic_trainer.lexical import matrix
    fixture = json.loads((Path(__file__).resolve().parents[3] / 'tests/fixtures/lexical-feature-v1.json').read_text())
    rows = matrix([fixture['text'], fixture['equivalent']])
    assert (rows[0] != rows[1]).nnz == 0
    assert set(rows[0].indices) == {int(k) for k in fixture['features']}
    for k, v in fixture['features'].items():
        assert rows[0, int(k)] == pytest.approx(v, rel=1e-6)


def test_grouped_probe_preprocessing_does_not_consult_check_vectors():
    import numpy as np
    from semantic_trainer.probe import grouped_probe
    x = np.array([[i % 2, i / 100] for i in range(100)], dtype=np.float32)
    y = [i % 2 for i in range(100)]
    groups = [str(i) for i in range(100)]
    held = set(groups[80:])
    folds = {group: int(group) % 5 for group in groups[:80]}
    before = grouped_probe(x, y, groups, held, folds=folds)
    x[80:] += 100000
    after = grouped_probe(x, y, groups, held, folds=folds)
    assert before['oof'] == after['oof']
    assert np.array_equal(before['model'][0].mean_, after['model'][0].mean_)
    assert len(before['oof']) == 80 and before['skipped'] == []
    assert before['sample_curves'] and before['check'] == 20


def test_export_train_lock_reload(tmp_path):
    path = tmp_path / 'refund.yaml'
    files.publish(path, {'name': 'refund', 'fields': {'message': {'type': 'text', 'required': True}},
        'reads': ['message'], 'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}})
    files.append(path.with_suffix('.test.jsonl'), envelopes([
        {'input': {'message': f'{"please refund my payment" if i % 2 else "hello thanks friend"} number {i}'},
         'facts': {'refund': bool(i % 2)}, 'group': str(i)} for i in range(240)]))
    with Runtime({'judge': FakeScorer()}) as rt:
        exported = tools.export(rt, path)
        report = train(exported['path'], tmp_path / 'bundle')
        labels = Labels(rt.load(path))
        bundle = load_bundle(report['path'], labels.sw.plan.fact('refund'), labels.training_id)
        assert bundle.head.probs('please refund my payment')[0] > .5
        assert bundle.head.probs('hello thanks friend')[1] > .5
        result = tools.lock(rt, path, head_bundles={'refund': report['path']})
        assert result['heads'] == ['refund']
        locked = rt.load(path)
        assert locked.snapshot.candidate_id and locked.snapshot.heads['refund'].classes == ('true', 'false')
        assert locked('please refund my payment').facts['refund'].value is True
        data = files.read_json(Path(report['path']) / 'manifest.json')
        data['feature_identity'] = 'wrong'
        files.publish(Path(report['path']) / 'manifest.json', data)
        with pytest.raises(ConfigError, match='feature'):
            load_bundle(report['path'])
        tools.export(rt, path, split='all')
        with pytest.raises(ConfigError, match='tune-only'):
            train(exported['path'], tmp_path / 'bad')


def republish(path, manifest, rows, **header):
    import hashlib
    from semantic_switch.artifacts import export_identity
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for row in rows]
    manifest = {**{k: v for k, v in manifest.items() if k not in ('payloads', 'export_id')}, **header}
    manifest['rows'] = [{**entry, 'sha256': hashlib.sha256(line.encode()).hexdigest()} for entry, line in zip(manifest['rows'], lines)]
    manifest['export_id'] = export_identity(manifest)
    files.publish(path, manifest, {path.name.removesuffix('.manifest.json'): lines})
    return manifest


def test_relocated_exports_keep_identity_and_duplicate_observations_merge_only_when_they_agree(tmp_path, monkeypatch):
    from semantic_switch.artifacts import digest, read_export
    from semantic_trainer.jobs import resolve
    from tests.test_adaptation_contracts import training_job
    job_path, job, rows = training_job(tmp_path, monkeypatch)
    resolved, observations = resolve(job_path)
    source = Path(job['exports'][0]['path'])
    original = files.read_published(source)
    copy = tmp_path / 'copy' / 'renamed.manifest.json'
    copied = republish(copy, original, rows)
    assert copied['export_id'] == original['export_id'] and read_export(copy, native=True)[0]['export_id'] == original['export_id']
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': copied['export_id'], 'roles': {}}]})
    with pytest.raises(ConfigError, match='one entry per immutable export'):
        resolve(job_path)
    files.publish(job_path, {**job, 'exports': [{**job['exports'][0], 'path': str(copy)}]})
    moved, _ = resolve(job_path)
    assert moved['id'] != resolved['id'] and moved['fit'] == resolved['fit'] and moved['monitor'] == resolved['monitor'] and moved['cohort'] == resolved['cohort']
    situated = republish(copy, original, rows, situations=[{**original['situations'][0], 'records': ['f' * 64]}, *original['situations'][1:]])
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': situated['export_id'], 'roles': {}}]})
    joined, merged = resolve(job_path)
    assert situated['export_id'] != original['export_id'] and joined['id'] != resolved['id'] and joined['fit']['id'] == resolved['fit']['id']
    assert len(merged) == len(observations) and joined['schema'] == 8 and joined['monitor'] == resolved['monitor'] and joined['fitting']['precision'] == 'float32'
    assert all({s['export_id'] for s in row['sources']} == {original['export_id'], situated['export_id']} for row in merged)
    assert [row['component'] for row in merged] == [row['component'] for row in observations]
    changed = [dict(row) for row in rows]
    entry = {**changed[0]['native']['presentations'][0], 'question': changed[0]['native']['presentations'][0]['question'] + [1]}
    entry['id'] = digest([changed[0]['id'], {k: v for k, v in entry.items() if k != 'id'}])
    changed[0] = {**changed[0], 'native': {**changed[0]['native'], 'presentations': [entry, *changed[0]['native']['presentations'][1:]]}}
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': republish(copy, original, changed)['export_id'], 'roles': {}}]})
    with pytest.raises(ConfigError, match='different meaning, labels, evidence or native presentations'):
        resolve(job_path)
    lines = copy.with_name(next(iter(files.read_published(copy)['payloads'].values()))['file'])
    lines.write_text(lines.read_text().replace('"label": 0', '"label": 1', 1))
    with pytest.raises(ConfigError, match='incomplete publication|row bytes differ'):
        resolve(job_path)


def test_recipe_target_modules_default_to_all_projections_and_reject_unknown(tmp_path, monkeypatch):
    from semantic_trainer.jobs import MODULES, resolve
    from tests.test_adaptation_contracts import training_job
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    assert resolve(job_path)[0]['recipe']['target_modules'] == list(MODULES)
    for modules in (['q_proj', 'v_proj'], ['v_proj']):
        files.publish(job_path, {**job, 'recipe': {**job.get('recipe', {}), 'target_modules': modules}})
        assert resolve(job_path)[0]['recipe']['target_modules'] == modules
    for modules in ([], ['q_proj', 'q_proj'], ['embed_tokens'], 'q_proj'):
        files.publish(job_path, {**job, 'recipe': {**job.get('recipe', {}), 'target_modules': modules}})
        with pytest.raises(ConfigError, match='target_modules'):
            resolve(job_path)
