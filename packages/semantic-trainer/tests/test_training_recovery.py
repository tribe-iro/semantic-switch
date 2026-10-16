import math
from pathlib import Path

import pytest

from semantic_switch import files
from semantic_trainer import lora, monitor, objective, recovery
from semantic_trainer.jobs import resolve
from test_lora import fake_native, monitored_job


def test_phase_failure_keeps_original_exception_and_never_syncs_again():
    calls = []
    phases = objective.Phases(lambda: calls.append(1))
    with pytest.raises(ValueError, match='original'):
        with phases('forward'):
            raise ValueError('original')
    assert calls == [1]
    with pytest.raises(RuntimeError, match='closing'):
        with objective.Phases(lambda: calls.append(1) if len(calls) == 1 else (_ for _ in ()).throw(RuntimeError('closing')))('forward'):
            pass


def test_deadline_retains_consumed_work_without_counting_between_commands():
    now = [100.]
    deadline = monitor.Deadline(20, lambda: now[0], consumed=12)
    now[0] += 3
    assert deadline.elapsed() == 15
    resumed = monitor.Deadline(20, lambda: now[0] + 1000, consumed=deadline.elapsed())
    assert resumed.elapsed() == 15


def test_real_monitor_uses_evaluation_cap_and_disabled_probe(tmp_path, monkeypatch):
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 1, 'min_epochs': 1, 'microbatch': 1, 'evaluation_batch_size': 4, 'fit_probe_observations': 0})
    fake_native(monkeypatch)
    result = lora.train(job, tmp_path / 'run')
    points = [p for p in files.read_jsonl(Path(result['path']) / 'progress.jsonl') if p['kind'] == 'evaluation']
    assert all(p['probe']['submitted'] == 0 and not p['probe']['enabled'] for p in points)
    assert all('4' in p['execution_work']['batch_sizes'] for p in points)
    assert all(p['execution_work']['forwards'] == 6 for p in points)
    assert all(p['permutations']['action']['full']['comparable'] == 12 for p in points)


def test_resume_after_update_restores_dropout_and_partial_epoch_exactly(tmp_path, monkeypatch):
    import torch
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 2, 'min_epochs': 1, 'batch_size': 7, 'dropout': .15, 'eval_every_examples': 7, 'patience_examples': 10000})
    fake_native(monkeypatch)
    reference = tmp_path / 'reference'
    lora.train(job, reference)
    resolved, _ = resolve(job)
    expected = recovery.load(reference, resolved)
    original = monitor.Monitor.point
    def interrupted(self, step, examples):
        if step == 3:
            raise RuntimeError('interrupted')
        return original(self, step, examples)
    monkeypatch.setattr(monitor.Monitor, 'point', interrupted)
    output = tmp_path / 'resumed'
    with pytest.raises(RuntimeError, match='interrupted'):
        lora.train(job, output)
    assert (output / 'candidates' / 'zero').is_dir() and recovery.latest(output).is_dir()
    monkeypatch.setattr(monitor.Monitor, 'point', original)
    result = lora.train(job, output)
    actual = recovery.load(output, resolved)
    assert actual['state']['step'] == expected['state']['step'] == 8
    assert actual['state']['epoch'] == expected['state']['epoch'] == 2
    assert actual['state']['offset'] == 0
    for key, value in expected['adapter'].items():
        torch.testing.assert_close(value, actual['adapter'][key], rtol=0, atol=0)
    assert actual['scheduler'] == expected['scheduler']
    for parameter, state in expected['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, actual['optimizer']['state'][parameter][key], rtol=0, atol=0)
    updates = list(files.read_jsonl(output / 'updates.jsonl'))
    assert [r['step'] for r in updates] == list(range(1,9))
    assert result['stop']['steps_trained'] == 8
    monkeypatch.setattr(lora, 'adapted', lambda *a: pytest.fail('finalization loaded a model'))
    assert lora.train(job, output)['names'] == result['names']


def test_permutations_compare_semantic_classes_and_preserve_escape():
    from semantic_testkit.labels import presentation
    row = {'classes': ['x', 'y'], 'presentations': [presentation('a', 'full', 'canonical', [1], [2], [3], [[4], [5]], 0, [0, -1]),
                                                  presentation('b', 'full', 'permuted', [1], [2], [3], [[4], [5]], 1, [-1, 0])]}
    result = objective.permutation_diagnostics(row, [[math.log(.2), math.log(.8)], [math.log(.8), math.log(.2)]], [math.log(.9)] * 2)
    assert result[0]['comparable'] and not result[0]['disagreement'] and result[0]['max_probability_delta'] == 0


@pytest.mark.parametrize('name', ['candidates.json', 'metrics.json', 'run.json'])
def test_interrupted_finalization_needs_no_model_on_retry(tmp_path, monkeypatch, name):
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 1, 'min_epochs': 1, 'fit_probe_observations': 0})
    fake_native(monkeypatch)
    output = tmp_path / 'run'
    publish, failed = files.publish, []
    def interrupted(path, data, *args, **kwargs):
        if Path(path).name == name and (output / 'finalization.json').exists() and not failed:
            failed.append(1)
            raise RuntimeError('publication interrupted')
        return publish(path, data, *args, **kwargs)
    monkeypatch.setattr(files, 'publish', interrupted)
    with pytest.raises(RuntimeError, match='publication interrupted'):
        lora.train(job, output)
    assert files.read_json(output / 'run.json')['state'] == 'failed'
    assert (output / 'candidates' / 'zero').is_dir()
    monkeypatch.setattr(files, 'publish', publish)
    monkeypatch.setattr(lora, 'adapted', lambda *args: pytest.fail('publication retry loaded the model'))
    result = lora.train(job, output)
    assert files.read_json(output / 'run.json')['state'] == 'ready'
    assert result['stop']['steps_trained'] == 3


def test_optional_profile_keeps_completed_updates_and_resumes(tmp_path, monkeypatch):
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'fit_probe_observations': 0})
    original, calls = lora.update, []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('profile interrupted')
        return original(*args, **kwargs)
    monkeypatch.setattr(lora, 'update', interrupted)
    target = tmp_path / 'profile.json'
    with pytest.raises(RuntimeError, match='profile interrupted'):
        lora.profile(job, target, steps=2, warmup=1)
    assert not target.exists()
    progress = files.read_json(tmp_path / 'profile.work' / 'profile-progress.json')
    assert progress['status'] == 'failed'
    assert recovery.latest(tmp_path / 'profile.work').exists()
    monkeypatch.setattr(lora, 'update', original)
    result = lora.profile(job, target, steps=2, warmup=1)
    assert result['status'] == 'completed' and result['work']['updates'] == 3
    assert len(result['updates']) == 2 and len(result['points']) == 1
