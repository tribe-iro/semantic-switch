from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

from semantic_switch import files

from .candidates import Native, fake_deployment, fake_training
from .scoring import FakeScorer

WORKER = 'SEMANTIC_TESTKIT_WORKER'


def measured(context, directory, settings):
    from semantic_switch.training import artifact_digest
    from semantic_trainer import jobs
    from semantic_trainer.pipeline import output
    resolved = context.population()[0]
    costs = {'loading_seconds': 1., 'update_seconds_per_token': 1e-6, 'point_seconds_per_token': 1e-6, 'checkpoint_seconds': 1.}
    record = {'schema': jobs.PROFILE_SCHEMA, 'kind': jobs.PROFILE_KIND, 'population_id': resolved['id'], 'fit_id': resolved['fit']['id'],
              'configuration': jobs.profile_configuration(resolved),
              'workload': jobs.profile_workload(resolved), 'costs': costs, 'cadences': [jobs.projection(resolved, costs, cadence) for cadence in (8, 16)]}
    record['id'] = artifact_digest(record)
    files.write_new(directory / 'profile.json', record)
    return output('fitting-profile', record['id'], directory / 'profile.json', directory, measured=True)


def install(patch, *, gguf, vocab) -> dict:
    from semantic_switch import api
    from semantic_trainer import pipeline
    original, base = api.Runtime.identity, files.sha256_file(gguf)

    def identity(self, role='judge'):
        found = original(self, role)
        return found if 'sha256' in found else {**found, 'sha256': base}
    patch.setattr(api.Runtime, 'identity', identity)
    Native.tokens = FakeScorer()
    Native.tokens.adapter = 'chat'
    patch.setattr(pipeline, 'native_judge', lambda path, device: Native.tokens)
    fake_training(patch, vocab)
    fake_deployment(patch)
    return {**pipeline.OPERATIONS, 'profile': pipeline.Operation(measured, pipeline.verify_profile)}


def main() -> int:
    from semantic_trainer import pipeline
    command, *rest = sys.argv[1:]
    if command == 'protocol':
        print(json.dumps(pipeline.protocol()))
        return 0
    pipeline.OPERATIONS.update(install(SimpleNamespace(setattr=setattr), **json.loads(os.environ[WORKER])))
    request, flags = rest[0], dict(zip(rest[1::2], rest[2::2]))
    number = lambda name: int(flags[name]) if name in flags else None
    return pipeline.command(request, until=flags.get('--until'), attempt=number('--attempt'), cadence=number('--cadence'))


def worker(directory, monkeypatch, *, gguf, vocab) -> str:
    path = directory / 'semantic-trainer'
    path.write_text(f'#!{sys.executable}\nimport sys\nfrom semantic_testkit.pipeline import main\nsys.exit(main())\n')
    path.chmod(0o755)
    monkeypatch.setenv(WORKER, json.dumps({'gguf': str(gguf), 'vocab': vocab}))
    return str(path)
