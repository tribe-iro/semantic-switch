
from semantic_switch import model_jobs
from argparse import Namespace
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')
from semantic_switch import ConfigError, api, files
from semantic_switch.training import model_job
from semantic_switch.journal import Journal
from semantic_switch.tools.specialize import request
from semantic_switch.training import artifact_digest
from semantic_testkit.assets import fake_assets
from semantic_testkit.scoring import FakeScorer
from semantic_testkit.pipeline import native_scoring
from semantic_trainer import jobs, pipeline
from semantic_trainer.pipeline import Controller, Operation, output
from bench import evidence, procedures, task_experiment

FIT = 32
CADENCE = 552


@pytest.fixture
def staged(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    manifest, gguf = fake_assets(tmp_path / 'assets', 'Q8_0')
    identity = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**identity(self, role), 'sha256': files.sha256_file(str(gguf))})
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    native_scoring(monkeypatch)
    monkeypatch.setattr(task_experiment, 'DEVICES', {'fit': 'cpu', 'native': 'cpu'})
    monkeypatch.setattr(task_experiment, 'FIT_SITUATIONS', None)
    procedures.pack_command(Namespace(seed=0, per_template=2, variants=3, reviews=None, output=tmp_path / 'pack', fit=FIT, model_select=2, check=16))
    return manifest, evidence.stage_under(tmp_path / 'pack', tmp_path / 'stages')


def measured(context, directory, settings):
    resolved = context.population()[0]
    costs = {'loading_seconds': 1., 'update_seconds_per_token': 1e-6, 'point_seconds_per_token': 1e-6, 'checkpoint_seconds': 1.}
    record = {'schema': jobs.PROFILE_SCHEMA, 'kind': jobs.PROFILE_KIND, 'configuration': jobs.profile_configuration(resolved),
              'workload': jobs.profile_workload(resolved), 'costs': costs, 'cadences': [jobs.projection(resolved, costs, CADENCE)]}
    record['id'] = artifact_digest(record)
    files.write_new(directory / 'profile.json', record)
    return output('fitting-profile', record['id'], directory / 'profile.json', directory, measured=True)


def profiled(tmp_path, manifest, staged, name='profiling', recipe=task_experiment.RECIPE):
    work = tmp_path / name
    files.write_new(work / 'request.json', request(staged['definition'], staged['labels'], work, devices=task_experiment.DEVICES, assets=manifest, quant='Q8_0',
                                                  recipe=recipe, budget=task_experiment.BUDGET, task_family=task_experiment.TASK_FAMILY, measure={}))
    operations = {**pipeline.OPERATIONS, 'profile': Operation(measured, pipeline.verify_profile)}
    return Path(Controller(work / 'request.json', operations=operations, until='profile', emit=lambda **event: None).run()['profile'])


def described(tmp_path, staged, profile, name='early'):
    work = tmp_path / name
    files.write_new(work / 'request.json', task_experiment.descriptor(staged, work, profile, CADENCE))
    return work / 'request.json'


class Fitting:
    def __init__(self, stop):
        self.stop, self.attempts = stop, []

    def run(self, context, directory, settings):
        with model_job() as held:
            with pytest.raises(ConfigError, match='locked by'):
                with model_job():
                    pass
            directory.mkdir(parents=True)
            files.publish(directory / 'run.json', {'state': 'ready', 'job_id': context.outputs['finalize']['id'], 'stop': self.stop, 'owner': held.owner})
        self.attempts.append(settings['attempt'])
        return output('fitted-run', artifact_digest(files.read_json(directory / 'run.json')), directory, stop=self.stop)

    def verify(self, context, record):
        if artifact_digest(files.read_json(Path(record['path']) / 'run.json')) != record['id']:
            raise ConfigError('run changed')


def completions(path):
    with Journal(path.parent / pipeline.JOURNAL, files.read_json(path)) as journal:
        return [payload for payload in journal.payloads() if payload.get('state') == 'completed']


def test_descriptor_binds_the_staged_pack_profile_and_fixed_recipe_and_prepares_without_check_labels(tmp_path, staged):
    manifest, stage = staged
    profile = profiled(tmp_path, manifest, stage)
    path = described(tmp_path, stage, profile)
    declared = files.read_json(path)
    assert declared['model'] == {'assets': manifest, 'quant': 'Q8_0'} and declared['cadence'] == 552 and declared['representation'] == 'merged'
    assert declared['profile'] == {'path': str(profile.resolve()), 'sha256': files.sha256_file(profile)}
    assert declared['selection'] == {'fit_situations': None, 'acceptance_fit': False} and declared['task']['family'] == 'bench:zt411'
    assert declared['sources']['definition']['path'] == str(Path(stage['definition']).resolve())
    assert Path(stage['definition']).parent.name == files.sha256_file(tmp_path / 'pack' / 'benchmark.json')[:16]
    audit = Path(stage['audit'])
    before = audit.read_bytes() if audit.exists() else None
    fitted = []
    operations = {**pipeline.OPERATIONS, **{name: Operation(lambda *args: fitted.append(args), None) for name in ('fit', 'select', 'lock')}}
    result = Controller(path, operations=operations, until='finalize', emit=lambda **event: None).run()
    assert result['state'] == 'stopped' and result['stage'] == 'finalize' and not fitted and result['run'] is None and result['lock'] is None
    record = files.read_json(profile)
    assert result['stages']['profile']['id'] == record['id'] and result['profile'] == str(profile.resolve()) and result['cadence'] == 552
    resolved, observations = jobs.resolve(result['job'])
    assert resolved['cadence']['eval_every_examples'] == 552 and resolved['cadence']['profile'] == record['id']
    assert resolved['cadence']['workload_ratio'] == {'point_tokens': 1., 'fit_tokens_per_observation': 1.}
    assert files.read_json(Path(result['job']).parent / 'profile.json') == record
    assert {key: resolved['recipe'][key] for key in ('rank', 'alpha', 'dropout', 'batch_size', 'microbatch', 'learning_rate', 'schedule', 'epochs',
                                                     'min_epochs', 'seed', 'selection_unit')} == \
        {'rank': 16, 'alpha': 32, 'dropout': 0., 'batch_size': 8, 'microbatch': 1, 'learning_rate': 2e-4, 'schedule': 'constant', 'epochs': 4,
         'min_epochs': 2, 'seed': 0, 'selection_unit': 'complete_situation'}
    assert {key: resolved['budget'][key] for key in ('max_sequence_tokens', 'max_seconds', 'max_tokens', 'max_monitor_tokens')} == \
        {'max_sequence_tokens': 1024, 'max_seconds': 18000, 'max_tokens': None, 'max_monitor_tokens': None}
    assert resolved['assets'] == {'manifest': manifest, 'quant': 'Q8_0'} and resolved['populations']['fit']['situations'] > 0
    assignment = files.read_json(Path(result['stages']['freeze']['path']) / pipeline.MANIFEST)['assignment']
    held = set(assignment['check_components'])
    assert len(held) == assignment['roles']['check']['families'] > 0
    _, rows = pipeline.read_export(result['stages']['export']['path'], native=True)
    assert rows and not {row['component'] for row in rows} & held
    assert not {row['component'] for row in observations} & held and {row['role'] for row in observations} <= {'fit', 'model_select', 'unused'}
    assert (audit.read_bytes() if audit.exists() else None) == before


def test_profiles_of_another_workload_altered_bytes_or_another_base_are_rejected_before_fitting(tmp_path, staged):
    manifest, stage = staged
    other = profiled(tmp_path, manifest, stage, 'rank8', tuple(value if not value.startswith('rank=') else 'rank=8' for value in task_experiment.RECIPE))
    fitted = []
    operations = {**pipeline.OPERATIONS, 'fit': Operation(lambda *args: fitted.append(args), None)}
    with pytest.raises(ConfigError, match=r'measured another fitting configuration \(adapter\)'):
        Controller(described(tmp_path, stage, other, 'stale'), operations=operations, emit=lambda **event: None).run()
    profile = profiled(tmp_path, manifest, stage)
    path = described(tmp_path, stage, profile, 'edited')
    record = files.read_json(profile)
    profile.write_text(json.dumps({**record, 'costs': {**record['costs'], 'update_seconds_per_token': 1e-9}}))
    with pytest.raises(ConfigError, match='fitting profile changed since this request'):
        Controller(path, operations=operations, emit=lambda **event: None).run()
    with pytest.raises(ConfigError, match='not an intact fitting profile'):
        Controller(described(tmp_path, stage, profile, 'forged'), operations=operations, emit=lambda **event: None).run()
    files.publish(tmp_path / 'q4.json', {**record, 'configuration': {**record['configuration'], 'assets': {'manifest': manifest, 'quant': 'Q4_0'}}})
    with pytest.raises(ConfigError, match='measured a Q4_0 base'):
        task_experiment.descriptor(stage, tmp_path / 'q4', tmp_path / 'q4.json')
    assert not fitted


def test_attempts_fit_into_distinct_directories_keep_stop_records_and_own_the_model_job_one_stage_at_a_time(tmp_path, staged):
    manifest, stage = staged
    path = described(tmp_path, stage, profiled(tmp_path, manifest, stage))
    stop = {'reason': 'time_budget', 'convergence': 'not_demonstrated', 'unevaluated': {'steps': 3, 'examples': 24}, 'deadline': {'crossed': 'update'}}
    fitting = Fitting(stop)
    operations = {**pipeline.OPERATIONS, 'fit': Operation(fitting.run, fitting.verify)}
    first = Controller(path, operations=operations, until='fit', emit=lambda **event: None).run()
    second = Controller(path, operations=operations, until='fit', attempt=2, emit=lambda **event: None).run()
    resumed = Controller(path, operations=operations, until='fit', emit=lambda **event: None).run()
    assert fitting.attempts == [1, 2] and first['state'] == second['state'] == resumed['state'] == 'stopped'
    assert first['run'] != second['run'] == resumed['run'] and Path(first['run']).is_dir() and Path(second['run']).is_dir()
    assert first['job'] == second['job'] and second['stages']['finalize']['state'] == 'reused' and resumed['stages']['fit']['state'] == 'reused'
    runs = [entry for entry in completions(path) if entry['stage'] == 'fit']
    assert sorted(entry['settings']['attempt'] for entry in runs) == [1, 2]
    assert all(entry['output']['summary']['stop'] == stop for entry in runs)
    assert all(files.read_json(Path(entry['output']['path']) / 'run.json')['stop']['unevaluated'] == {'steps': 3, 'examples': 24} for entry in runs)
    with model_job():
        pass
