import shutil

import pytest

import semantic_switch as ss
from semantic_switch import ConfigError, Runtime, files
from semantic_switch.artifacts import load_receipt
from semantic_switch.training import current, entry, fitted, point, rollback
from semantic_testkit.candidates import REALIZED, VALUES, Native
from semantic_testkit.scoring import FakeScorer

from test_specialize import selection


def deployed(tmp_path, monkeypatch):
    module, job, run, *_ = selection(tmp_path, monkeypatch)
    module.select(job, run)
    moved = tmp_path / 'moved'
    shutil.move(run / 'deployment', moved)
    shutil.rmtree(tmp_path / 'task')
    return moved


def test_every_spelling_of_a_relocated_fitted_lock_loads_the_same_frozen_switch(tmp_path, monkeypatch):
    moved = deployed(tmp_path, monkeypatch)
    receipt = load_receipt(moved / 'receipt.json')
    lock = files.read_json(moved / 'task.lock.json')
    pointer = tmp_path / 'current.json'
    point(pointer, moved, reason='first fitted lock', expected=None)
    for spelling in (moved, moved / 'receipt.json', moved / 'task.lock.json', pointer):
        with Runtime() as runtime:
            switch = runtime.load(spelling)
            assert switch.frozen and switch.snapshot.lock_id == lock['id'] and runtime.models['judge'].provenance == receipt.provenance
            result = switch({'message': 'report 4'})
            assert result.facts['action'].value == VALUES[1] and result.facts['action'].source == 'judge'
    assert entry(pointer).id == receipt.id and current(pointer) == (moved / 'receipt.json').resolve()
    model = ss.judge(moved, device='cpu')
    assert model == fitted(moved / 'task.lock.json', device='cpu') and model.provenance == receipt.provenance and model.representation == 'merged'
    with Runtime(bindings={'judge': ss.judge(moved)}) as runtime:
        assert runtime.load(moved).snapshot.lock_id == lock['id']


def test_a_fitted_lock_refuses_a_different_runtime_model_altered_weights_and_changed_execution(tmp_path, monkeypatch):
    moved = deployed(tmp_path, monkeypatch)
    other = deployed(tmp_path / 'other', monkeypatch)
    with pytest.raises(ConfigError, match="binding 'judge' is this fitted lock's model"):
        Runtime(FakeScorer()).load(moved)
    with Runtime() as runtime:
        runtime.load(moved)
        with pytest.raises(ConfigError, match="binding 'judge' is this fitted lock's model"):
            runtime.load(other)
    Native.realized = {**REALIZED, 'n_batch': 64}
    with Runtime() as runtime:
        with pytest.raises(ConfigError, match='judge execution changed'):
            runtime.load(moved).snapshot
    Native.realized = REALIZED
    with pytest.raises(ConfigError, match='draft and allow_model_change apply to authored definitions'):
        Runtime().load(moved, allow_model_change=True)
    weights = moved / load_receipt(moved / 'receipt.json').data['gguf']
    weights.write_bytes(weights.read_bytes() + b'changed')
    with pytest.raises(ConfigError, match='changed since selection'):
        Runtime().load(moved)


def test_rollback_restores_the_previous_fitted_lock_without_touching_either_entry(tmp_path, monkeypatch):
    first = deployed(tmp_path / 'one', monkeypatch)
    second = deployed(tmp_path / 'two', monkeypatch)
    before = {path: files.sha256_file(path) for root in (first, second) for path in root.rglob('*') if path.is_file()}
    pointer = tmp_path / 'current.json'
    point(pointer, first, reason='first', expected=None)
    point(pointer, second, reason='second', expected=load_receipt(first / 'receipt.json').id)
    assert current(pointer) == (second / 'receipt.json').resolve()
    record = rollback(pointer, reason='regression', expected=load_receipt(second / 'receipt.json').id)
    assert current(pointer) == (first / 'receipt.json').resolve() and record['id'] == load_receipt(first / 'receipt.json').id
    assert [item['reason'] for item in record['history']] == ['first', 'second', 'rollback: regression']
    with Runtime() as runtime:
        assert runtime.load(pointer).snapshot.lock_id == files.read_json(first / 'task.lock.json')['id']
    assert {path: files.sha256_file(path) for path in before} == before
