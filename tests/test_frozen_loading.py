from __future__ import annotations

from semantic_switch import models

import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import anyio
import pytest
from typer.testing import CliRunner

import semantic_switch as ss
from semantic_switch import ConfigError, Runtime, api, cli, files, tools
from semantic_switch.drivers import openrouter
from semantic_switch.hosted import Reply
from semantic_switch.judge import Cancelled
from semantic_switch.tools.data import Labels
from semantic_switch.training import Task, compare_deployments, current, declaration, entry, point, rollback
from semantic_testkit.scoring import FakeScorer

REFUND = 'The customer asks to receive money back for a charge.'
SPEC = {'name': 'route', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'], 'facts': {'refund': {'check': REFUND}},
        'rules': [{'id': 'refund', 'when': 'refund', 'then': 'billing'}], 'otherwise': 'general'}
REVIEWED = [{'derivation': 'authored', 'producer': 'reviewer', 'actor': {'kind': 'human'}}]
CHECK = [('refund please', True), ('hello 3', True), ('hello 4', False), ('refund now', True), ('hello 5', False), ('refund later', True)]


def undecided(state, question, width):
    return [.5, .5]


def locked(directory, scorer, corrections=()):
    directory.mkdir(parents=True)
    path = directory / 'route.yaml'
    files.publish(path, SPEC)
    with Runtime(scorer) as runtime:
        switch = runtime.load(path, draft=True)
        for message, value in corrections:
            switch.correct({'message': message}, 'refund', value, sources=REVIEWED)
        tools.lock(runtime, path)
    return directory / 'route.lock.json'


def evidence(directory, rows=CHECK):
    directory.mkdir(parents=True)
    path = directory / 'route.yaml'
    files.publish(path, SPEC)
    paths = files.Paths.of(path)
    with Runtime(FakeScorer()) as runtime:
        task = Task(runtime.load(path, draft=True))
        files.publish(paths.test, ''.join(json.dumps(task.label({'message': message}, {'refund': value}, sources={'refund': REVIEWED}, role='check',
                                                                groups=[f'family-{n}'])) + '\n' for n, (message, value) in enumerate(rows)))
        Labels(runtime.load(path, draft=True))
    return paths.test, paths.dataset


def test_an_ordinary_frozen_lock_loads_through_the_public_entry_after_its_sources_are_removed(tmp_path):
    lock = locked(tmp_path / 'source', FakeScorer(), [('hello 9', True)])
    with Runtime(FakeScorer()) as runtime:
        authored = runtime.load(tmp_path / 'source' / 'route.yaml')
        before = [authored({'message': message}).as_dict() for message in ('refund please', 'hello 9', 'hello 4')]
    moved = tmp_path / 'moved' / 'route.lock.json'
    moved.parent.mkdir()
    shutil.copy(lock, moved)
    shutil.rmtree(tmp_path / 'source')
    scorer = FakeScorer()
    with Runtime(scorer) as runtime:
        switch = runtime.load(moved)
        assert switch.frozen and switch.snapshot.locked and switch.snapshot.lock_id == files.read_json(moved)['id']
        after = [switch({'message': message}) for message in ('refund please', 'hello 9', 'hello 4')]
    assert [result.as_dict() for result in after] == before and [row['rule'] for row in before] == ['refund', 'refund', 'otherwise']
    assert after[1].facts['refund'].source == 'example' and after[1].facts['refund'].settled and not scorer.asked('hello 9')
    assert (after[0].status, after[0].answer, after[0].facts['refund'].value) == ('ok', 'billing', True)
    assert (after[2].status, after[2].rule, after[2].answer) == ('ok', 'otherwise', 'general')
    for result in after:
        with pytest.raises(TypeError, match='no truth value'):
            bool(result)


def test_frozen_lock_rejects_altered_payloads_runtime_mismatch_and_authored_options(tmp_path):
    lock = locked(tmp_path / 'source', FakeScorer())
    altered = tmp_path / 'altered.lock.json'
    record = files.read_json(lock)
    record['facts']['refund']['policies']['judge']['true']['theta'] = .51
    files.publish(altered, record)
    with Runtime(FakeScorer()) as runtime:
        with pytest.raises(ConfigError, match='not an intact frozen lock'):
            runtime.load(altered)
        with pytest.raises(ConfigError, match='draft and allow_model_change apply to authored definitions'):
            runtime.load(lock, draft=True)
    with Runtime(FakeScorer(name='another-judge')) as runtime:
        with pytest.raises(ConfigError, match='judge identity changed'):
            runtime.load(lock).snapshot
    with pytest.raises(ConfigError, match='no frozen lock'):
        Runtime(FakeScorer()).load(tmp_path / 'missing.lock.json')


def test_renamed_rule_vocabulary_has_concrete_regeneration_errors(tmp_path):
    old = tmp_path / 'old.yaml'
    files.publish(old, {**{k: v for k, v in SPEC.items() if k != 'rules'}, 'cases': SPEC['rules']})
    with pytest.raises(ConfigError, match='cases'):
        Runtime(FakeScorer()).load(old)
    path = tmp_path / 'route.yaml'
    files.publish(path, SPEC)
    with Runtime(FakeScorer()) as runtime:
        switch = runtime.load(path)
        receipt = switch.correct({'message': 'hello 1'}, rule='refund', sources=REVIEWED)
        assert receipt['rule_label_recorded'] and files.read_jsonl(files.Paths.of(path).test)[0]['rule'] == 'refund'
        stale = files.read_jsonl(files.Paths.of(path).test)[0]
        stale['case'] = stale.pop('rule')
        files.publish(files.Paths.of(path).test, json.dumps(stale) + '\n')
        with pytest.raises(ConfigError, match=r"unsupported label fields \['case'\]"):
            Labels(runtime.load(path, draft=True))


def test_a_runtime_with_only_other_roles_never_resolves_the_default_judge(monkeypatch):
    monkeypatch.setattr(models, '_resolve', lambda source: pytest.fail(f'resolved {source}'))
    runtime = Runtime(bindings={'refund': FakeScorer()})
    assert set(runtime.models) == {'refund'}
    with pytest.raises(ConfigError, match="model role 'judge' is not bound"):
        runtime.check('refund please', REFUND)
    with pytest.raises(ConfigError, match='bind the judge role once'):
        Runtime(FakeScorer(), bindings={'judge': FakeScorer()})
    assert set(Runtime().models) == {'judge'} and Runtime().models['judge'] == api.Model(api.DEFAULT_JUDGE)


class Transport:
    def __init__(self, hold=None):
        self.bodies, self.closed, self.hold = [], False, hold

    async def __call__(self, url, body, remaining):
        self.bodies.append(body)
        while self.hold is not None and not self.hold.is_set():
            await anyio.sleep(.005)
        return Reply(200, {'model': 'm', 'provider': 'p', 'choices': [{'logprobs': {'content': [{'token': 'A', 'logprob': 0.0,
                                                                                                    'top_logprobs': [{'token': 'A', 'logprob': 0.0}]}]}}]})

    async def close(self):
        self.closed = True


def test_supplied_transports_are_borrowed_and_specification_judges_are_owned(monkeypatch):
    supplied = Transport()
    with Runtime(ss.judge('openrouter:m', codec='chat', transport=supplied)):
        pass
    transferred = Transport()
    with Runtime(ss.judge('openrouter:m', codec='chat', transport=transferred, own_transport=True), own_judges=True):
        pass
    built = Transport()
    monkeypatch.setattr(openrouter.OpenRouter, 'wire', lambda self, *args: built)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    with Runtime('openrouter:typesafe/jev-1.13'):
        pass
    assert not supplied.closed and transferred.closed and built.closed


def test_an_exception_leaving_the_runtime_cancels_in_flight_hosted_work_promptly():
    hold, results = threading.Event(), []
    transport = Transport(hold)

    def call(runtime):
        try:
            results.append(runtime.check('refund please', REFUND))
        except BaseException as error:
            results.append(error)
    started = time.monotonic()
    with pytest.raises(KeyError, match='raised by the caller'):
        with Runtime(ss.judge('openrouter:m', codec='chat', transport=transport, timeout=30), own_judges=True) as runtime:
            worker = threading.Thread(target=call, args=(runtime,))
            worker.start()
            while not transport.bodies:
                time.sleep(.005)
            raise KeyError('raised by the caller')
    worker.join(5)
    assert time.monotonic() - started < 5 and not worker.is_alive() and not hold.is_set()
    assert len(transport.bodies) == 1 and isinstance(results[0], (Cancelled, RuntimeError))


def test_an_exception_in_an_async_block_cancels_pending_hosted_calls():
    hold = threading.Event()
    transport = Transport(hold)

    async def scenario():
        with Runtime(ss.judge('openrouter:m', codec='chat', transport=transport, timeout=30), own_judges=True) as runtime:
            task = asyncio.create_task(runtime.acheck('refund please', REFUND))
            while not transport.bodies:
                await asyncio.sleep(.005)
            try:
                raise TypeError('caller bug')
            finally:
                await asyncio.sleep(0)
                assert not task.done()
    started = time.monotonic()
    with pytest.raises(TypeError, match='caller bug'):
        asyncio.run(scenario())
    assert time.monotonic() - started < 5 and len(transport.bodies) == 1 and not hold.is_set()


def test_ordinary_close_still_waits_for_an_active_call():
    hold = threading.Event()
    transport = Transport(hold)
    runtime = Runtime(ss.judge('openrouter:m', codec='chat', transport=transport), own_judges=True)
    results = []
    worker = threading.Thread(target=lambda: results.append(runtime.check('refund please', REFUND)))
    worker.start()
    while not transport.bodies:
        time.sleep(.005)
    closer = threading.Thread(target=runtime.close)
    closer.start()
    closer.join(.2)
    assert closer.is_alive()
    hold.set()
    worker.join(5)
    closer.join(5)
    assert len(results) == 1 and hasattr(results[0], 'status') and transport.closed is False


def test_current_pointer_promotes_and_rolls_back_without_touching_locks(tmp_path):
    first, second = locked(tmp_path / 'first', FakeScorer()), locked(tmp_path / 'second', FakeScorer(), [('hello 7', True)])
    digests = {path: files.sha256_file(path) for path in (first, second)}
    pointer = tmp_path / 'current.json'
    point(pointer, first, reason='initial', expected=None)
    point(pointer, second, reason='promoted', expected=entry(first).id)
    assert current(pointer) == second.resolve() and entry(pointer).id == files.read_json(second)['id']
    with Runtime(FakeScorer()) as runtime:
        assert runtime.load(pointer)({'message': 'hello 7'}).facts['refund'].source == 'example'
    record = rollback(pointer, reason='regression found', expected=entry(second).id)
    assert record['entry'] == str(first.resolve()) and [item['reason'] for item in record['history']] == ['initial', 'promoted', 'rollback: regression found']
    with Runtime(FakeScorer()) as runtime:
        assert runtime.load(pointer)({'message': 'hello 7'}).facts['refund'].source == 'judge'
    assert {path: files.sha256_file(path) for path in (first, second)} == digests
    replaced = files.read_json(first)
    replaced['created'] = 'later'
    replaced['tests'] = 'edited'
    files.publish(first, {**replaced, 'id': entry(second).id})
    with pytest.raises(ConfigError, match='not an intact frozen lock'):
        current(pointer)


def compare(tmp_path, to, baseline, candidate, scorers, **kwargs):
    declared = declaration(margin=kwargs.pop('margin', 0.), reference=kwargs.pop('reference', ('verified', 'computed', 'authored')))
    return compare_deployments(baseline, candidate, *evidence_paths(tmp_path), to, declaration=declared, count=200,
                               bindings={'baseline': {'judge': scorers[0]}, 'candidate': {'judge': scorers[1]}}, **kwargs)


def evidence_paths(tmp_path):
    root = tmp_path / 'evidence'
    if not root.exists():
        return evidence(root)
    return root / 'route.test.jsonl', root / 'route.dataset.json'


def test_compare_runs_each_input_once_per_arm_resumes_and_reports_descriptive_reuse(tmp_path, monkeypatch):
    baseline, candidate = locked(tmp_path / 'baseline', FakeScorer(name='base', oracle=undecided)), locked(tmp_path / 'candidate', FakeScorer(name='cand'))
    from semantic_switch.tools import deployment
    original, calls = deployment.compared_row, []

    def interrupted(plan, item, family, result, review):
        calls.append(item['input']['message'])
        if len(calls) == len(CHECK) + 2:
            raise RuntimeError('interrupted')
        return original(plan, item, family, result, review)
    monkeypatch.setattr(deployment, 'compared_row', interrupted)
    first = (FakeScorer(name='base', oracle=undecided), FakeScorer(name='cand'))
    with pytest.raises(RuntimeError, match='interrupted'):
        compare(tmp_path, tmp_path / 'compared', baseline, candidate, first)
    resumed = (FakeScorer(name='base', oracle=undecided), FakeScorer(name='cand'))
    report = compare(tmp_path, tmp_path / 'compared', baseline, candidate, resumed)
    assert len(calls) == 2 * len(CHECK) + 1 and resumed[0].requests == [] and len(resumed[1].requests) == len(CHECK) - 1
    assert report['exposure']['independent'] and report['lineage']['candidate']['state'] == 'declared_disjoint'
    assert report['arms']['baseline']['targets']['accepted'] == 0
    assert (report['arms']['candidate']['targets']['correct_accepted'], report['arms']['candidate']['targets']['wrong_accepted']) == (5, 1)
    paired = report['paired']
    assert (paired['baseline']['correct'], paired['candidate']['correct'], paired['fixes'], paired['regressions'], paired['both_correct']) == (4, 5, 2, 1, 3)
    assert paired['difference']['estimate'] == pytest.approx(1 / 6) and paired['difference']['support'] == 'sparse'
    assert report['decision']['verdict'] == 'inconclusive' and report['decision']['constraints']['unresolved'] == {'baseline': 6, 'candidate': 0, 'tolerated': 0, 'met': True}
    rows = [(arm, row['input']) for arm, arm_rows in report['rows'].items() for row in arm_rows]
    assert len(rows) == len(set(rows)) == 2 * len(CHECK)
    with pytest.raises(ConfigError, match='already published'):
        compare(tmp_path, tmp_path / 'compared', baseline, candidate, resumed)
    again = compare(tmp_path, tmp_path / 'changed', baseline, candidate, (FakeScorer(name='base', oracle=undecided), FakeScorer(name='cand')), margin=.1)
    assert not again['exposure']['independent'] and again['decision']['verdict'] == 'reused_evidence' and again['population'].startswith('reused check evidence')


def test_compare_refuses_different_task_semantics_and_reports_overlapping_corrections(tmp_path):
    baseline = locked(tmp_path / 'baseline', FakeScorer(name='base'))
    overlapping = locked(tmp_path / 'overlap', FakeScorer(name='cand'), [('hello 3', True)])
    report = compare(tmp_path, tmp_path / 'overlap-run', baseline, overlapping, (FakeScorer(name='base'), FakeScorer(name='cand')))
    assert report['lineage']['candidate']['state'] == 'known_overlap' and report['decision']['verdict'] == 'overlapping_lineage'
    other = tmp_path / 'other'
    other.mkdir()
    files.publish(other / 'route.yaml', {**SPEC, 'facts': {'refund': {'check': 'The customer mentions a delivery.'}}})
    with Runtime(FakeScorer(name='cand')) as runtime:
        tools.lock(runtime, other / 'route.yaml')
    with pytest.raises(ConfigError, match='different targets or meanings'):
        compare(tmp_path, tmp_path / 'different', baseline, other / 'route.lock.json', (FakeScorer(name='base'), FakeScorer(name='cand')))


def test_ss_compare_cli_binds_the_judge_to_ordinary_locks(tmp_path, monkeypatch):
    baseline, candidate = locked(tmp_path / 'baseline', FakeScorer()), locked(tmp_path / 'candidate', FakeScorer())
    labels, manifest = evidence_paths(tmp_path)
    monkeypatch.setattr(cli, 'judge', lambda source, **settings: FakeScorer())
    result = CliRunner().invoke(cli.app, ['--json', '--judge', 'fake', 'compare', '--baseline', str(baseline), '--candidate', str(candidate),
                                          '--examples', str(labels), '--evidence', str(manifest), '--margin', '0.05', '--to', str(tmp_path / 'cli')])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out['paired']['difference']['estimate'] == 0 and out['paired']['fixes'] == out['paired']['regressions'] == 0
    assert out['promotion']['verdict'] == 'reference_agreement' and (tmp_path / 'cli' / 'comparison.json').exists()
    assert files.read_json(out['decision']) == out['promotion'] and out['promotion']['comparison'] == out['id']
    pointer = tmp_path / 'current.json'
    adopted = CliRunner().invoke(cli.app, ['--json', 'promote', str(pointer), str(baseline), '--reason', 'first'])
    assert adopted.exit_code == 0, adopted.output
    refused = CliRunner().invoke(cli.app, ['--json', 'promote', str(pointer), str(candidate), '--reason', 'r', '--decision', out['decision']])
    assert isinstance(refused.exception, ConfigError) and 'not a supported claim' in str(refused.exception) and current(pointer) == baseline.resolve()
    assert (out['reference']['state'], out['reference']['strata'], out['reference']['within_scope'], out['reference']['targets']) == ('agreement', {'authored': 6}, 0, 6)


def test_frozen_lock_loading_needs_neither_sources_nor_trainer_nor_torch(tmp_path):
    lock = locked(tmp_path / 'source', FakeScorer())
    moved = tmp_path / 'elsewhere.lock.json'
    shutil.copy(lock, moved)
    shutil.rmtree(tmp_path / 'source')
    program = ('import sys, semantic_switch as ss\n'
               'from semantic_testkit.scoring import FakeScorer\n'
               'result = ss.Runtime(FakeScorer()).load(sys.argv[1])({"message": "refund please"})\n'
               'assert result.rule == "refund", result\n'
               'assert not {"torch", "semantic_trainer", "semantic_learn"} & set(sys.modules), sorted(sys.modules)\n')
    done = subprocess.run([sys.executable, '-c', program, str(moved)], cwd=tmp_path, env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
