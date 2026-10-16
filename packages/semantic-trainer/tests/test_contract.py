from pathlib import Path
import json

import pytest

from semantic_switch import Runtime, files, tools, ConfigError
from semantic_switch.artifacts import load_bundle, read_export
from semantic_switch.tools.data import Labels
from semantic_trainer.lexical import train
from semantic_testkit.labels import envelopes
from semantic_testkit.scoring import FakeScorer, fixture


def test_trainer_consumes_the_same_feature_fixture():
    from semantic_trainer.lexical import matrix
    feature = fixture('lexical-feature-v1.json')
    rows = matrix([feature['text'], feature['equivalent']])
    assert (rows[0] != rows[1]).nnz == 0
    assert set(rows[0].indices) == {int(k) for k in feature['features']}
    for k, v in feature['features'].items():
        assert rows[0, int(k)] == pytest.approx(v, rel=1e-6)


def test_lexical_fitting_compacts_active_columns_without_changing_the_head():
    import numpy as np
    from semantic_trainer import lexical
    texts = ['please refund my payment', 'hello thanks friend', 'refund now please', 'thanks for the help', 'Ünïcode refund café', 'café hello']
    y = np.array([0, 1, 0, 1, 0, 1])
    rows = lexical.matrix(texts)
    x, active = lexical.compact(rows)
    assert x.shape == (6, len(active)) and set(active.tolist()) == set(rows.indices.tolist()) and (x.toarray() == rows[:, active].toarray()).all()
    head = lexical.fit(x, active, y, ['refund', 'other'])
    full = lexical.fit(rows, np.arange(lexical.N_FEATURES, dtype=np.int32), y, ['refund', 'other'])
    assert set(head.idx.tolist()) == set(full.idx.tolist()) and np.allclose(head.probs('refund please café'), full.probs('refund please café'), atol=1e-5)
    fold = lexical.fit(x[[0, 1, 2]], active, y[[0, 1, 2]], ['refund', 'other'])
    assert set(fold.idx.tolist()) <= set(rows[[0, 1, 2]].indices.tolist())


def test_export_train_lock_reload(tmp_path, monkeypatch):
    from semantic_trainer import lexical
    extracted, original = [], lexical.features
    monkeypatch.setattr(lexical, 'features', lambda text: extracted.append(text) or original(text))
    path = tmp_path / 'refund.yaml'
    files.publish(path, {'name': 'refund', 'fields': {'message': {'type': 'text', 'required': True}},
        'reads': ['message'], 'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}})
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([
        {'input': {'message': f'{"please refund my payment" if i % 2 else "hello thanks friend"} number {i}'},
         'facts': {'refund': bool(i % 2)}, 'group': str(i)} for i in range(240)])))
    with Runtime(FakeScorer()) as rt:
        exported = tools.export(rt, path)
        report = train(exported['path'], tmp_path / 'bundle')
        assert len(extracted) == report['rows'] and report['folds'] == 5
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


from semantic_testkit.labels import republish


def test_relocated_exports_keep_identity_and_duplicate_observations_merge_only_when_they_agree(tmp_path, monkeypatch):
    from semantic_switch.artifacts import read_export
    from semantic_trainer.jobs import resolve
    from semantic_testkit.assets import training_job
    job_path, job, rows = training_job(tmp_path, monkeypatch)
    resolved, observations = resolve(job_path)
    source = Path(job['exports'][0]['path'])
    original = read_export(source, native=True)[0]
    copy = tmp_path / 'copy' / 'renamed.manifest.json'
    copied = republish(copy, original, rows)
    assert copied['export_id'] == original['export_id'] and read_export(copy, native=True)[0]['export_id'] == original['export_id']
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': copied['export_id'], 'task': job['exports'][0]['task'], 'roles': {}}]})
    with pytest.raises(ConfigError, match='one entry per immutable export'):
        resolve(job_path)
    files.publish(job_path, {**job, 'exports': [{**job['exports'][0], 'path': str(copy)}]})
    moved, _ = resolve(job_path)
    assert moved['id'] != resolved['id'] and moved['fit'] == resolved['fit'] and moved['monitor'] == resolved['monitor'] and moved['cohort'] == resolved['cohort']
    situated = republish(copy, original, rows, situations=[{**original['situations'][0], 'records': ['f' * 64]}, *original['situations'][1:]])
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': situated['export_id'], 'task': job['exports'][0]['task'], 'roles': {}}]})
    joined, merged = resolve(job_path)
    assert situated['export_id'] != original['export_id'] and joined['id'] != resolved['id'] and joined['fit']['id'] == resolved['fit']['id']
    assert len(merged) == len(observations) and joined['schema'] == 11 and joined['monitor'] == resolved['monitor'] and joined['fitting']['precision'] == 'float32'
    assert all({s['export_id'] for s in row['sources']} == {original['export_id'], situated['export_id']} for row in merged)
    assert [row['component'] for row in merged] == [row['component'] for row in observations]
    from dataclasses import replace
    first = rows[0]['presentations']
    changed = [{**rows[0], 'presentations': (replace(first[0], question=first[0].question + (1,)), *first[1:])}, *rows[1:]]
    files.publish(job_path, {**job, 'exports': [*job['exports'], {'path': str(copy), 'export_id': republish(copy, original, changed)['export_id'], 'task': job['exports'][0]['task'], 'roles': {}}]})
    with pytest.raises(ConfigError, match='different meaning, labels, evidence or native presentations'):
        resolve(job_path)
    lines = copy.with_name(files.read_published(copy)['payloads']['renamed.observations.jsonl']['file'])
    lines.write_text(lines.read_text().replace('"label": 0', '"label": 1', 1))
    with pytest.raises(ConfigError, match='incomplete publication'):
        resolve(job_path)


def test_recipe_target_modules_default_to_all_projections_and_reject_unknown(tmp_path, monkeypatch):
    from semantic_trainer.jobs import MODULES, resolve
    from semantic_testkit.assets import training_job
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    assert resolve(job_path)[0]['recipe']['target_modules'] == list(MODULES)
    for modules in (['q_proj', 'v_proj'], ['v_proj']):
        files.publish(job_path, {**job, 'recipe': {**job.get('recipe', {}), 'target_modules': modules}})
        assert resolve(job_path)[0]['recipe']['target_modules'] == modules
    for modules in ([], ['q_proj', 'q_proj'], ['embed_tokens'], 'q_proj'):
        files.publish(job_path, {**job, 'recipe': {**job.get('recipe', {}), 'target_modules': modules}})
        with pytest.raises(ConfigError, match='target_modules'):
            resolve(job_path)


def test_lexical_and_lora_reject_the_same_restricted_history_before_publication(tmp_path, monkeypatch):
    lexical = pytest.importorskip('semantic_trainer.lexical')
    from semantic_switch import Runtime
    from semantic_switch.evidence import LabelSource
    from semantic_switch.permissions import TrainingUse
    from semantic_testkit.assets import pipeline_base
    from semantic_testkit.labels import TEACHER, action_definition, action_export, declared, label_file
    from semantic_testkit.scoring import FakeScorer
    from semantic_trainer.jobs import prepare
    manifest = pipeline_base(tmp_path, monkeypatch)
    history = [LabelSource('judged', TEACHER, training_use=TrainingUse.parse(declared(permission='forbidden'))).record()]
    records = label_file(tmp_path, history=history)
    with Runtime(FakeScorer()) as runtime:
        exported = action_export(runtime, tmp_path / 'task', records, 'maintenance:test')
        definition = action_definition(runtime, tmp_path / 'task')
    with pytest.raises(ConfigError, match='forbidden'):
        lexical.train(exported['path'], tmp_path / 'head')
    with pytest.raises(ConfigError, match='forbidden'):
        prepare(definition, [exported['path']], tmp_path / 'job', assets_manifest=manifest, recipe={'eval_every_examples': 8})
    assert not (tmp_path / 'head').exists() and not (tmp_path / 'job' / 'job.yaml').exists()
