from __future__ import annotations

import pytest

from semantic_switch import Pick, api, files

from .labels import authored_export


def fake_assets(base, quant='Q4_0'):
    from semantic_trainer import assets
    repo, commit = 'example/tiny', 'a' * 40
    paths = assets.layout(repo, commit)
    for relative in assets.TOOLS:
        (base / paths['toolchain'] / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / paths['toolchain'] / relative).write_text(relative)
    (base / paths['hf']).mkdir(parents=True)
    files.publish(base / paths['hf'] / 'config.json', {'model_type': 'qwen3'})
    (base / paths['gguf']).mkdir(parents=True)
    for kind in ('F16', quant):
        (base / paths['gguf'] / f'{kind}.gguf').write_bytes(kind.encode())
    return assets.record(base, repo, commit, [quant]), base / paths['gguf'] / f'{quant}.gguf'


def pipeline_base(tmp_path, monkeypatch):
    pytest.importorskip('semantic_trainer.jobs')
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    digest, gguf = fake_assets(tmp_path / 'assets')
    original = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**original(self, role), 'sha256': files.sha256_file(str(gguf))})
    return digest


def training_job(tmp_path, monkeypatch):
    from semantic_switch.training import freeze_task
    from semantic_trainer.jobs import LOSS, SCHEMA
    digest = pipeline_base(tmp_path, monkeypatch)
    _, path, manifest, rows = authored_export(tmp_path, Pick('Choose.', {'x': 'One', 'y': 'Two'}))
    export = str(path.with_suffix('.export.jsonl')) + '.manifest.json'
    frozen = freeze_task(path, tmp_path / 'task')
    job = {'schema': SCHEMA, 'loss': LOSS, 'objective': 'specialization', 'tasks': {frozen['behavior']: {'source': str(path), 'bundle': 'task', 'weight': 1}},
           'initialization': 'fresh', 'assets': {'manifest': digest, 'quant': 'Q4_0'},
           'fitting': {'device': 'cpu'}, 'exports': [{'path': export, 'export_id': manifest['export_id'], 'task': frozen['behavior'],
                                                      'roles': {'fit': [row['group'] for row in rows[:-4]], 'model_select': [row['group'] for row in rows[-4:]]}}],
           'recipe': {'eval_every_examples': 5}, 'budget': {'max_tokens': 100000, 'max_sequence_tokens': 1000, 'max_seconds': 100}}
    job_path = tmp_path / 'job.yaml'
    files.publish(job_path, job)
    return job_path, job, rows
