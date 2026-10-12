from copy import deepcopy
from dataclasses import replace
import json
from itertools import cycle

import anyio
import pytest

from semantic_switch import Check, ConfigError, Rate, Runtime, files, tools
from semantic_switch.cli import question_specs
from semantic_switch.compiler import compile_spec
from semantic_switch.examples import label_record
from tests.conftest import FakeScorer, transport
from tests.test_remaining_contracts import ReceiptJudge, definition
from .conftest import envelopes


@pytest.mark.parametrize('changes', [{'unstated': 'false'}, {'excludes': 'except'}, {'check': ''}])
def test_authored_validation_matches_sdk_runtime_compiler_cli(tmp_path, changes):
    spec = {'check': 'Criterion.', **changes}
    with pytest.raises(ConfigError):
        Check(spec['check'], **{key: value for key, value in spec.items() if key != 'check'})
    with Runtime({'judge': FakeScorer()}) as rt:
        with pytest.raises(ConfigError):
            rt.check('case', spec['check'], **{key: value for key, value in spec.items() if key != 'check'})
        assert not rt._loaded
    with pytest.raises(ConfigError):
        compile_spec({'name': 'invalid', 'facts': {'x': spec}})
    path = tmp_path / 'questions.yaml'
    files.publish(path, {'x': spec})
    with pytest.raises(ConfigError):
        question_specs(path)


def test_numeric_rate_levels_and_descriptions_rejected_before_judge():
    with pytest.raises(ConfigError):
        Rate('Rate.', ['low', 7])
    with pytest.raises(ConfigError):
        compile_spec({'name': 'invalid', 'facts': {'r': {'rate': 'Rate.', 'levels': ['low', 7]}}})
    with Runtime({'judge': FakeScorer()}) as rt:
        with pytest.raises(ConfigError):
            rt.pick('case', 'Pick.', {'x': 7, 'y': 'valid'})
        assert not rt._loaded


@pytest.mark.parametrize('body', ['facts:\n  x: {check: First}\n  x: {check: Second}\n',
                                 'guidance:\n  policy: {text: First, text: Second}\n',
                                 '- {id: x, id: y, applies_when: claim}\n',
                                 'x: {check: First, check: Second}\n'])
def test_duplicate_authored_yaml_rejected_with_location(tmp_path, body):
    path = tmp_path / 'duplicate.yaml'
    path.write_text(body)
    with pytest.raises(ConfigError, match=r'duplicate.yaml:\d+: duplicate key'):
        files.read_yaml(path)


def test_yaml_merge_semantics_are_explicit(tmp_path):
    path = tmp_path / 'merge.yaml'
    path.write_text('base: &base {check: First}\nx: {<<: *base, check: Second}\n')
    with pytest.raises(ConfigError, match='merge keys are unsupported'):
        files.read_yaml(path)
    path.write_text('base: &base {check: First}\nx: *base\n')
    assert files.read_yaml(path)['x'] == {'check': 'First'}


def labelled_definition(tmp_path):
    path = definition(tmp_path)
    files.append(path.with_suffix('.test.jsonl'), envelopes([
        {'input': {'message': f'case {i}'}, 'facts': {'refund': True}, 'group': f'family-{i}',
         'selection': {'selected_by': 'random'}} for i in range(24)]))
    return path


def test_lock_mixed_development_models_refuses_publication(tmp_path):
    path = labelled_definition(tmp_path)
    judge = ReceiptJudge(cycle(['old-v1', 'new-v2']))
    with Runtime({'judge': judge}) as rt:
        with pytest.raises(ConfigError, match='mixed predictor'):
            tools.lock(rt, path)
    assert not path.with_suffix('.lock.json').exists()


def test_lock_audit_drift_cannot_expand_baseline(tmp_path, monkeypatch):
    path = labelled_definition(tmp_path)
    import importlib
    module = importlib.import_module('semantic_switch.tools.lock')
    original = module.judged
    calls = []
    def observed(*args):
        rows = original(*args)
        calls.append(True)
        return [replace(row, observed_models=('old-v1' if len(calls) == 1 else 'new-v2',)) for row in rows]
    monkeypatch.setattr(module, 'judged', observed)
    with Runtime({'judge': ReceiptJudge(cycle(['old-v1']))}) as rt:
        with pytest.raises(ConfigError, match='baseline cannot expand'):
            tools.lock(rt, path)
    assert not path.with_suffix('.lock.json').exists()


def test_candidate_digest_binds_returned_model_and_matching_receipts(tmp_path):
    from semantic_switch.artifacts import candidate_digest
    path = labelled_definition(tmp_path)
    with Runtime({'judge': ReceiptJudge(cycle(['old-v1']))}) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        assert sw.snapshot.returned_models == ('old-v1',)
        assert sw({'message': 'new case'}).status == 'ok'
    lock = files.read_json(path.with_suffix('.lock.json'))
    changed = deepcopy(lock)
    changed['returned_models']['judge'] = ['new-v2']
    assert candidate_digest(lock) != candidate_digest(changed)


def test_per_fact_status_partition_is_independent_of_switch(tmp_path):
    from bench.policy_audit import summarize
    path = definition(tmp_path, facts={'a': {'check': 'Uncertain.'}, 'b': {'check': 'Missing.', 'unstated': True}},
                      cases=[{'id': 'act', 'when': 'a or b', 'then': 'act'}])
    def oracle(state, question, width):
        return [.6, .4] if width == 2 else [.01, .01, .98]
    with Runtime({'judge': FakeScorer(oracle)}) as rt:
        result = rt.load(path)({'message': 'case'})
    assert result.status == 'needs_input'
    assert result.facts['a'].status == 'uncertain'
    assert result.facts['b'].status == 'needs_input'
    metrics = summarize([{'labels': {'a': True, 'b': True}, 'result': result.as_dict(True)}])
    assert metrics['fact:a']['uncertain'] == metrics['fact:b']['needs_input'] == 1
    assert all(row['submitted'] == row['accepted'] + sum(row[k] for k in
               ('uncertain', 'needs_input', 'failed', 'none', 'unexamined')) for row in metrics.values())


def test_policy_audit_rejects_corrected_input_before_scoring(tmp_path):
    from bench.policy_audit import run
    path = definition(tmp_path)
    scorer = FakeScorer()
    with Runtime({'judge': scorer}) as rt:
        sw = rt.load(path)
        sw.correct({'message': 'corrected'}, fact='refund', value=True)
        tools.lock(rt, path)
        sw = rt.load(path)
        audit = tmp_path / 'audit.jsonl'
        files.publish(audit, json.dumps(label_record(sw.plan, {'message': 'corrected'}, {'refund': True}, sources={'refund': [{'origin': 'human', 'producer': 'fixture'}]}, groups=['audit'])))
        before = len(scorer.requests)
        with pytest.raises(ConfigError, match='known overlap.*corrected'):
            run(rt, path, audit)
        assert len(scorer.requests) == before


def test_policy_audit_rejects_known_family_alias_read_only(tmp_path):
    from bench.policy_audit import reject_known_overlap
    path = labelled_definition(tmp_path)
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        lock = files.read_json(sw.paths.lock)
        before = deepcopy(lock)
        items = [{'input': {'message': 'new wording'}, 'groups': ['family-0'], 'pairs': []}]
        with pytest.raises(ConfigError, match='known overlap'):
            reject_known_overlap(sw, items, lock)
        assert lock == before == files.read_json(sw.paths.lock)


def audit_artifact():
    from bench.policy_audit import summarize
    rows = [{'id': 'a', 'input': 'case', 'labels': {'x': True}, 'elapsed_ms': 10,
             'case_labelled': False, 'expected_case': None,
             'result': {'status': 'ok', 'facts': {'x': {'status': 'ok', 'settled': True, 'value': True}}}}]
    return {'schema': 1, 'audit_sha256': 'same', 'guidance': {}, 'candidate_id': 'candidate',
            'switch_revision': 'same', 'meanings': {'x': 'same'}, 'runtime': {'threads': 4},
            'metrics': summarize(rows), 'observations': rows}


def test_policy_audit_comparison_validates_metrics_meaning_configuration():
    from bench.policy_audit import compare
    left, right = audit_artifact(), audit_artifact()
    right['metrics']['fact:x']['wrong'] = 99
    with pytest.raises(ConfigError, match='metrics'):
        compare(left, right)
    right = audit_artifact()
    right['meanings']['x'] = 'other'
    with pytest.raises(ConfigError, match='authored meaning'):
        compare(left, right)
    right = audit_artifact()
    right['runtime']['threads'] = 8
    result = compare(left, right)
    assert result['configuration_differences']['threads'] == {'left': 4, 'right': 8}
    assert result['matched_error_budget'] == 'not established'


def test_runtime_close_releases_remaining_owned_resources_after_error():
    closed = []
    class Judge(ReceiptJudge):
        def close(self):
            closed.append(self.model_name)
            if self.model_name == 'first':
                raise RuntimeError('cleanup failed')
    first, second = Judge([]), Judge([])
    first.model_name, second.model_name = 'first', 'second'
    rt = Runtime({'judge': first, 'precise': second, 'alias': second}, own_judges=True)
    for role in ('judge', 'precise', 'alias'):
        rt.judge(role)
    with pytest.raises(RuntimeError, match='cleanup failed'):
        rt.close()
    assert closed == ['first', 'second'] and not rt._loaded
    rt.close()
    with pytest.raises(RuntimeError, match='closed'):
        rt.judge()


def test_case_label_without_derived_facts_has_no_effectiveness_claim(tmp_path):
    path = definition(tmp_path, facts={'a': {'check': 'A.'}, 'b': {'check': 'B.'}},
                      cases=[{'id': 'route', 'when': 'a or b', 'then': 'route'}])
    with Runtime({'judge': FakeScorer()}) as rt:
        sw = rt.load(path)
        receipt = sw.correct({'message': 'case'}, case='route')
        assert receipt['case_label_recorded'] and not receipt['facts'] and receipt['effective'] is None
        assert sw({'message': 'case'}).status == 'uncertain'
        receipt = sw.correct({'message': 'case'}, fact='a', value=True)
        assert receipt['facts'] == [['a']] and receipt['effective'] == 'now'


def test_benchmark_receipts_record_usage_and_provider_route():
    from bench.receipts import RecordingTransport
    from semantic_switch.hosted import Reply
    def response(url, body, timeout):
        return Reply(200, {'model': body['model'], 'provider': 'returned', 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}})
    recorder = RecordingTransport(send=transport(response))
    with recorder.recording('warmup', warmup=True):
        anyio.run(recorder, 'local', {'model': 'model', 'provider': {'only': ['expected']}}, 5)
    anyio.run(recorder, 'local', {'model': 'model'}, 5)
    summary = recorder.summary()
    assert summary['requests'] == 2 and summary['usage'] == {'reported': 2, 'prompt_tokens': 20, 'completion_tokens': 4}
    assert summary['receipts'][0]['operation']['warmup'] and summary['by_operation']['warmup']['requests'] == 1
    assert [(row['requested_provider'], row['returned_provider'], row['status']) for row in summary['receipts']] == [
        ('expected', 'returned', 200), (None, 'returned', 200)]
    rejected = RecordingTransport(send=transport(lambda url, body, timeout: Reply(429, None, '1')))
    anyio.run(rejected, 'local', {'model': 'model'}, 5)
    assert rejected.summary()['failed'] == 1 and rejected.summary()['receipts'][0]['completion'] == 'rejected'


def test_model_comparison_memory_arm_is_isolated(tmp_path, monkeypatch):
    from bench.model_comparison import isolated
    from types import SimpleNamespace
    calls = []
    output = tmp_path / 'arm.json'
    def subprocess_run(command, **kwargs):
        calls.append(command)
        files.publish(output, {'runtime': {}, 'process_peak_rss_kib': 123})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('bench.model_comparison.subprocess.run', subprocess_run)
    result = isolated('model', output, suites=False)
    assert '-m' in calls[0] and 'bench.model_comparison' in calls[0]
    assert result['memory_boundary'].startswith('fresh subprocess')


def test_native_replay_uses_frozen_policy_and_reports_disagreements(tmp_path):
    from contextlib import nullcontext
    from bench.diagnostics import frozen_replay
    class ModeScorer(FakeScorer):
        fused = True
        def operation(self):
            return nullcontext()
        def invalidate(self):
            pass
    scorer = ModeScorer()
    scorer.oracle = lambda *args: [.99, .005, .005] if scorer.fused else [.6, .2, .2]
    path = definition(tmp_path, cases=[{'id': 'act', 'when': 'refund', 'then': {'action': 'act'}}])
    with Runtime({'judge': scorer}) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        snapshot = sw.snapshot
        result = frozen_replay(rt, sw, [{'message': 'case'}], samples=1)
        assert sw.snapshot is snapshot and scorer.fused is True
        assert result['candidate_id'] == snapshot.candidate_id
        assert result['policies']['judge']['refund']['true'] == snapshot.policy('judge', sw.plan.fact('refund')).t_true
        assert any(row['status_changed'] and row['action_changed'] and row['fact_outcomes_changed'] for row in result['differences'])


def test_failed_picks_remain_submitted():
    from bench.quality import summarize
    result = summarize({'checks': [], 'picks': [
        {'answer': 'a', 'scores': {'a': .99, 'b': .01}, 'status': 'ok', 'error': None},
        {'answer': 'a', 'scores': {}, 'status': 'failed', 'error': 'transport'}]})
    assert result['pick_submitted'] == 2 and result['pick_accuracy'] == .5
    assert result['pick_valid_score_accuracy'] == 1


@pytest.fixture(scope='module')
def live_workflow():
    from bench.workflows import run
    return run()


@pytest.mark.slow
def test_guided_cli_subprocess_matches_sdk(live_workflow):
    for row in [*live_workflow['cli'][:2], live_workflow['direct_cli']]:
        assert row['same_public_outcome']
        assert row['exit_code'] in (0, 1, 2)
        assert not row['result'].get('error')


@pytest.mark.slow
def test_saved_switch_cli_reload_matches_sdk(live_workflow):
    assert live_workflow['cli'][2]['same_public_outcome']
    assert live_workflow['native_replay']['candidate_id']
    assert live_workflow['quickstart']['restored']['facts']['maintenance']['source'] == 'example'


@pytest.mark.slow
def test_public_workflow_refusal_and_invalid_input(live_workflow):
    missing, invalid = live_workflow['cli'][3:]
    assert missing['same_public_outcome'] and missing['result']['status'] == 'needs_input' and missing['exit_code'] == 2
    assert invalid['same_public_outcome'] and invalid['result']['status'] == 'failed' and invalid['exit_code'] != 0
    replay = live_workflow['native_replay']['arms'][0]['observations']
    assert replay[-3]['case'] == 'otherwise' and replay[-3]['cost'].get('judge', 0) == 0
    unknown = replay[4]
    if unknown['status'] == 'ok':
        assert unknown['case'] == 'otherwise'
        assert any(fact['settled'] and fact['value'] in (False, 'other') for fact in unknown['facts'].values())
    else:
        assert unknown['status'] in ('needs_input', 'uncertain') and unknown['answer'] is None


def test_unknown_semantic_evidence_cannot_silently_produce_otherwise(tmp_path):
    path = definition(tmp_path, cases=[{'id': 'refund', 'when': 'refund', 'then': 'refund'}], otherwise='support')
    with Runtime({'judge': FakeScorer(lambda *args: [.01, .01, .98])}) as rt:
        result = rt.load(path)({'message': 'case'})
    assert result.status == 'needs_input' and result.answer is None


def test_regression_comparison_freezes_grouped_policy_without_retuning(tmp_path):
    from bench.openrouter import regression_binding, run_audit
    from bench.receipts import RecordingTransport
    from semantic_switch.hosted import Reply
    audit = tmp_path / 'audit.jsonl'
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as rt:
        path = regression_binding(rt, tmp_path / 'model', audit)
        before = files.read_json(path.with_suffix('.lock.json'))
        result = run_audit(rt, RecordingTransport(send=transport(lambda url, body, timeout: Reply(200, {}))), path, audit)
        assert files.read_json(path.with_suffix('.lock.json')) == before
        assert result['candidate_id'] == before['candidate_id']
        assert result['metrics']['fact:refund']['submitted'] == 4
        assert all(row['label_origins'] == ['synthetic'] for row in result['observations'])


def test_receipts_from_worker_threads_keep_operation_and_count_once():
    from concurrent.futures import ThreadPoolExecutor
    from bench.receipts import RecordingTransport
    from semantic_switch.hosted import Reply
    recording = RecordingTransport(send=transport(lambda url, body, timeout: Reply(200, {'answers': {str(n): {} for n in range(8)},
                                                                               'usage': {'input_tokens': 80}})))
    with recording.recording('eight-questions'):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(anyio.run, recording, 'local', {'model': 'model'}, 5).result()
    summary = recording.summary()
    assert summary['requests'] == 1 and summary['usage']['input_tokens'] == 80
    assert summary['by_operation']['eight-questions']['requests'] == 1


def test_independent_audit_rejects_adapter_training_pair_from_other_fact(tmp_path):
    from bench.policy_audit import reject_known_overlap
    path = definition(tmp_path)
    with Runtime({'judge': FakeScorer()}) as rt:
        sw = rt.load(path)
        lock = {'predictor_artifacts': {'judge': {'rows': [
            {'fact': 'another_criterion', 'family_seeds': [['pair', 'trained-pair']]}]}}}
        with pytest.raises(ConfigError, match='overlap.*trained'):
            reject_known_overlap(sw, [{'input': {'message': 'new'}, 'groups': ['new-family'], 'pairs': ['trained-pair']}], lock)


def test_hosted_runtime_info_does_not_invent_a_native_build(monkeypatch):
    monkeypatch.setattr('semantic_switch.api._build', lambda: pytest.fail('hosted metadata loaded native library'))
    with Runtime({'judge': ReceiptJudge([])}) as rt:
        assert rt.info()['build'] is None
        rt.judge()
        assert rt.info()['build'] is None
