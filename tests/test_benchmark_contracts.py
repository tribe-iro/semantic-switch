from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from semantic_switch import Check, ConfigError, Runtime, files, tools
from semantic_switch.examples import label_record
from semantic_switch.results import thaw
from bench.fresh import FRESH
from bench.quality import check_items, render
from bench.outcomes import classification_metrics, finish, measure, resumable, task, work_key
from tests.conftest import FakeScorer, transport
from tests.test_remaining_contracts import definition


def test_fresh_decline_gold_orientation():
    rows = {row['text']: row['label'] for row in check_items(FRESH) if row['fact'] == 'decline'}
    assert rows["I'm afraid I'll have to pass on this one."] is True
    assert rows["Could we discuss the start date? I'm very keen."] is False
    assert rows['Infelizmente vou ter que passar essa.'] is True
    assert rows['Podemos conversar sobre a data de início? Estou animado.'] is False


def test_fixture_actor_matches_evidence():
    assert render('The tenant reports cold radiators.') == 'The tenant reports cold radiators.'
    assert render('case', 'operator') == '[operator] message: case'


def test_fixture_identity_changes_with_gold():
    from bench.reanalyze_fresh import reanalyze
    source = Path('bench/results/fresh_rows.json')
    before = source.read_bytes()
    result = reanalyze(source)
    arm = result['arms']['gemma-4-E2B-it-Q4_0.gguf']
    assert arm['old_gold_sha256'] != arm['corrected_gold_sha256']
    assert arm['corrected_metrics']['raw_correct'] == 141
    assert arm['corrected_metrics']['extreme_errors'] == 1
    assert source.read_bytes() == before
    original = files.read_json(source)
    assert [row['p'] for row in arm['observations']] == [row['p'] for row in original['gemma-4-E2B-it-Q4_0.gguf']]


def test_request_negative_and_missing_evidence_are_distinct():
    from bench.openrouter import scenarios
    rows = {identifier: (case, guidance, value, status) for identifier, case, guidance, value, status in scenarios()}
    assert 'only, no money returned' in rows['credit_excluded'][0]
    assert rows['credit_excluded'][2:] == (False, 'ok')
    assert rows['missing'][2:] == (None, 'needs_input')
    assert 'insufficient' in rows['missing'][1]


def test_rating_rubric_defines_expected_levels():
    from bench.openrouter import URGENCY_QUESTION, URGENCY_LEVELS
    assert URGENCY_LEVELS == ('can wait', 'today', 'immediately')
    assert all(word in URGENCY_QUESTION for word in ('after today', 'end of today', 'now'))


def test_preflight_class_acceptance_and_status_metrics():
    guess = lambda kind, value: {'kind': kind, 'key': None, 'value': value, 'score': .6}
    rows = [
        {'case_id': 'cash', 'family': 'cash', 'expected': {'value': True, 'status': 'ok'},
         'result': {'value': None, 'status': 'uncertain', 'p_true': .6, 'prediction': guess('value', True)}},
        {'case_id': 'cash', 'family': 'cash', 'expected': {'value': True, 'status': 'ok'},
         'result': {'value': False, 'status': 'ok', 'p_true': .01, 'prediction': guess('value', False)}},
        {'case_id': 'missing', 'family': 'missing', 'expected': {'value': None, 'status': 'needs_input'},
         'result': {'value': None, 'status': 'needs_input', 'p_true': .01, 'p_unstated': .98, 'prediction': guess('unstated', None)}},
        {'case_id': 'unavailable', 'family': 'unavailable', 'expected': {'value': True, 'status': 'ok'},
         'result': {'value': None, 'status': 'failed', 'p_true': None, 'prediction': None}},
    ]
    metric = classification_metrics(rows)
    assert metric['submitted'] == 4 and metric['unique_cases'] == 3
    assert metric['predicted'] == 3 and metric['raw_correct'] == 2 and metric['correct_per_submission'] == .5
    assert metric['accepted'] == metric['wrong_accepted'] == metric['protocol_failures'] == 1
    assert metric['precision'] == 0 and metric['correct_rejected'] == 2 and metric['operational_agreement'] == 1


def test_repeated_shapes_preserve_case_denominator():
    row = {'case_id': 'case', 'family': 'family', 'expected': {'value': True, 'status': 'ok'},
           'result': {'value': True, 'status': 'ok', 'p_true': .99}}
    metric = classification_metrics([deepcopy(row) for _ in range(3)])
    assert metric['submitted'] == 3 and metric['unique_cases'] == metric['families'] == 1


def test_audit_rejects_inconsistent_raw_correctness():
    from bench.policy_audit import summarize
    rows = [{'labels': {'refund': True}, 'result': {'facts': {}},
             'raw_answers': {'refund': {'value': False, 'correct': True}}}]
    with pytest.raises(ConfigError, match='correctness disagrees'):
        summarize(rows)
    rows[0]['raw_answers']['refund']['correct'] = False
    assert summarize(rows)['fact:refund']['raw_correct'] == 0


def test_audit_duplicate_ids_fail_before_predictor(tmp_path):
    from bench.policy_audit import run
    scorer = FakeScorer()
    with Runtime({'judge': scorer}) as runtime:
        path = definition(tmp_path)
        tools.lock(runtime, path)
        sw = runtime.load(path)
        rows = [label_record(sw.plan, {'message': f'case {n}'}, {'refund': True}, sources={'refund': [{'origin': 'human', 'producer': 'fixture'}]}, groups=[f'family{n}']) for n in range(2)]
        for row in rows:
            row['id'] = 'duplicate'
        audit = tmp_path / 'audit.jsonl'
        files.publish(audit, ''.join([json.dumps(row) + '\n' for row in rows]))
        before = len(scorer.requests)
        with pytest.raises(ConfigError, match='IDs'):
            run(runtime, path, audit)
        assert len(scorer.requests) == before


def test_benchmark_latency_includes_json_materialization(monkeypatch):
    from bench import outcomes
    original = outcomes.json.dumps
    def delayed(*args, **kwargs):
        time.sleep(.025)
        return original(*args, **kwargs)
    monkeypatch.setattr(outcomes.json, 'dumps', delayed)
    measured = measure(lambda: {'value': True})
    assert measured.output == {'value': True} and measured.ms >= 25 and measured.peak_rss_bytes > 0
    assert measured.error is None and measured.work is None and 'native_work' in measured.unavailable


class Counted:
    def __init__(self):
        self.calls = 0

    def info(self):
        self.calls += 1
        return {'capabilities': {'counters': {'decodes': 3 * self.calls}}}


class Clock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now


def test_failed_operation_keeps_elapsed_time_and_known_work():
    runtime, clock = Counted(), Clock()

    def failing():
        clock.now += .25
        raise ValueError('secret payload text')
    measured = measure(failing, runtime=runtime, clock=clock)
    assert measured.error == {'type': 'ValueError', 'stage': 'execute'} and measured.ms == 250. and measured.work == {'decodes': 3}
    assert measured.result is None and measured.output is None and 'secret' not in json.dumps(measured.error)
    clock.now = 0.
    unserializable = measure(lambda: {'value': float('nan')}, runtime=runtime, clock=clock)
    assert unserializable.error == {'type': 'ValueError', 'stage': 'serialize'} and unserializable.output is None and unserializable.work == {'decodes': 3}


def test_counter_failure_is_unknown_not_zero_and_cancellation_propagates():
    class Broken:
        def info(self):
            raise OSError('gone')
    measured = measure(lambda: {'ok': True}, runtime=Broken())
    assert measured.work is None and measured.unavailable['native_work'].startswith('counter collection failed') and measured.error is None
    with pytest.raises(KeyboardInterrupt):
        measure(lambda: (_ for _ in ()).throw(KeyboardInterrupt()), runtime=Counted())


def test_failed_snapshot_rows_retain_elapsed_work_and_unknown_counters():
    from bench import snapshots
    from bench.outcomes import Measurement
    domain = snapshots.DOMAINS['zt411']
    row = snapshots.prediction(domain, 'workflow', Measurement(None, None, 120., {'decodes': 4}, None, {'type': 'RuntimeError', 'stage': 'execute'}))
    assert row['outcome'] == 'failed' and row['ms'] == 120. and row['native_work'] == {'decodes': 4} and row['requests'] is None
    assert row['unavailable']['requests'].startswith('operation failed') and all(t['value'] is None for t in row['targets'].values())
    ok = {**row, 'requests': 2, 'native_work': {'decodes': 1}, 'unavailable': {}}
    report = snapshots.known([row, ok], 'requests')
    assert report == {'sum': 2, 'rows': 1, 'of': 2, 'bound': 'lower_bound', 'unavailable': [row['unavailable']['requests']]}
    assert snapshots.known([row, ok], 'native_work')['sum'] == {'decodes': 5} and snapshots.known([ok], 'requests')['bound'] == 'total'
    empty = {'status': 'failed', 'answer': None, 'facts': {}, 'cost': {'requests': [], 'logical_tokens': 0}}
    assert snapshots.prediction(domain, 'workflow', Measurement(None, empty, 5., {}, None))['outcome'] == 'failed'


def test_raised_scorer_operations_become_failed_submitted_rows(capsys):
    from contextlib import nullcontext
    from bench import equipment, openrouter

    class Failing(Counted):
        def pick(self, *args):
            time.sleep(.002)
            raise RuntimeError('backend down')
        ask = check = ask_many = pick

        def identity(self):
            return {}

    class Recording:
        records = []

        def recording(self, name):
            return nullcontext()

        def summary(self):
            return {}
    rows = [{'input': {'report': 'r'}, 'expected': equipment.NONE, 'trap': 't'}]
    got, progress = equipment.run_direct(Failing(), rows)
    assert got[0]['outcome'] == 'failed' and got[0]['error'] == {'type': 'RuntimeError', 'stage': 'execute'} and got[0]['ms'] > 0
    assert got[0]['native_work'] == {'decodes': 3} and equipment.summarize(got)['failed'] == 1 and progress['failed'] == progress['submitted'] == 1
    report = openrouter.run_preflight(Failing(), Recording())
    assert report['progress']['submitted'] == report['progress']['failed'] == report['progress']['total'] and not report['progress']['accepted']
    assert all(row['outcome'] == 'failed' and row['ms'] > 0 and row['result']['status'] == 'failed' for row in report['observations'])
    assert report['metrics']['protocol_failures'] == report['submitted'] and report['binary']['error']['stage'] == 'execute'
    assert 'bench.openrouter preflight [questions] final' in capsys.readouterr().err


def test_regression_template_copies_share_family(tmp_path):
    from bench.openrouter import regression_binding
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as runtime:
        path = regression_binding(runtime, tmp_path / 'regression', tmp_path / 'audit.jsonl')
    rows = files.read_jsonl(path.with_suffix('.test.jsonl'))
    assert len({row['groups'][0] for row in rows}) == 2
    assert all('two templates' in row['provenance']['purpose'] for row in rows)


def test_development_family_does_not_cross_partitions(tmp_path):
    from bench.openrouter import regression_binding
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as runtime:
        path = regression_binding(runtime, tmp_path / 'regression', tmp_path / 'audit.jsonl')
        sw = runtime.load(path)
        from semantic_switch.tools.data import Labels
        labels = Labels(sw)
        for fact in sw.plan.visible:
            tune, check = labels.examples(fact.name, 'tune'), labels.examples(fact.name, 'check')
            assert not {row.group for row in tune} & {row.group for row in check}


def test_generic_and_joint_request_counts():
    from bench.receipts import RecordingTransport
    from semantic_switch import judge_from
    from semantic_switch.hosted import Reply
    token = {'model': 'test', 'choices': [{'finish_reason': 'length', 'logprobs': {'content': [{'token': 'true', 'logprob': -.01,
              'top_logprobs': [{'token': 'false', 'logprob': -5}]}]}}], 'usage': {'prompt_tokens': 4}}
    recording = RecordingTransport(send=transport(lambda url, body, timeout: Reply(200, token)))
    with Runtime({'judge': judge_from('openrouter:test', transport=recording)}, own_judges=True) as runtime:
        runtime.ask('case', {'a': Check('A claim.'), 'b': Check('B claim.')})
    assert len(recording.records) == 2
    def joint(url, body, timeout):
        return Reply(200, {'model': 'test', 'answers': {key: {'noul': .99} for key in body['questions']}})
    recording = RecordingTransport(send=transport(joint))
    with Runtime({'judge': judge_from('openrouter:typesafe/jev-1.13', transport=recording)}, own_judges=True) as runtime:
        runtime.ask('case', {'a': Check('A claim.'), 'b': Check('B claim.')})
    assert len(recording.records) == 1


def test_heterogeneous_workload_questions_have_gold():
    from bench.model_comparison import heterogeneous_workload
    case, guidance, items, gold = heterogeneous_workload()
    assert len(items) == len(gold) == 8 and set(items) == set(gold)
    assert len({type(item) for item in items.values()}) == 3
    with Runtime({'judge': FakeScorer()}) as runtime:
        for item in items.values():
            description = task(runtime, item, case, guidance)
            assert description['definition'] == thaw(item.describe()) and description['guidance_digest']


def test_bounded_policy_expected_actions_are_literal():
    from bench.policy_workflow import CASES, CONTRASTS
    indexed = {row['id']: row for row in CASES}
    assert len(indexed) == len(CASES)
    assert indexed['cash_30']['action'] == 'approve_refund'
    assert indexed['cash_31']['action'] == 'deny_refund'
    for left, right, kind in CONTRASTS:
        assert (indexed[left]['action'] == indexed[right]['action']) == (kind == 'preserve')


def test_policy_fixture_exact_and_semantic_boundaries(tmp_path):
    from bench.policy_workflow import definition, POLICY
    path = tmp_path / 'returns.yaml'
    files.publish(path, definition(POLICY))
    scorer = FakeScorer(lambda *args: [.99, .005, .005])
    with Runtime({'judge': scorer}) as runtime:
        result = runtime.load(path)({'message': 'strong semantic approval', 'purchase_days': 45, 'account_hold': True})
    assert result.answer == {'action': 'deny_refund'} and not scorer.requests


@pytest.mark.slow
def test_policy_fixture_sdk_cli_task_parity(tmp_path):
    from bench.policy_workflow import definition, POLICY, CASES
    from bench.workflows import cli
    path = tmp_path / 'returns.yaml'
    files.publish(path, definition(POLICY))
    with Runtime() as runtime:
        tools.lock(runtime, path)
        case = next(row['input'] for row in CASES if row['id'] == 'held_leak')
        sdk = runtime.load(path)(case).as_dict(True)
        process = cli(['--json', 'run', str(path)], case)
    for key in ('case', 'answer', 'status', 'facts'):
        assert process['result'][key] == sdk[key]
    assert sdk['answer'] == {'action': 'deny_refund'}


def test_cli_model_and_capacity_options_are_invocation_local(monkeypatch):
    from contextlib import contextmanager
    from typer.testing import CliRunner
    from semantic_switch import cli
    from semantic_switch.results import CheckResult
    observed = []
    @contextmanager
    def session():
        observed.append(cli.invocation_options().copy())
        yield type('Runtime', (), {'ask': lambda *args, **kwargs: {'_': CheckResult('ok', True, .99, 1)}})()
    monkeypatch.setattr(cli, 'session', session)
    result = CliRunner().invoke(cli.app, ['--judge', 'other.gguf', '--max-input-tokens', '4096', '--json', '--verbose', 'check', 'Claim.'], input='case')
    assert result.exit_code == 0
    assert observed[0]['judge'] == 'other.gguf' and observed[0]['max_input_tokens'] == 4096 and observed[0]['verbose']
    assert cli.invocation_options()['judge'] != 'other.gguf'
    result = CliRunner().invoke(cli.app, ['--json', 'check', 'Claim.'], input='case')
    assert result.exit_code == 0 and observed[1]['max_input_tokens'] is None


def test_action_and_case_errors_remain_distinct():
    from bench.policy_workflow import summarize_actions
    row = {'correct': False, 'expected': {'case': 'expired', 'answer': {'action': 'deny'}, 'status': 'ok'},
           'result': {'case': 'not_cash', 'answer': {'action': 'deny'}, 'status': 'ok'}}
    metrics = summarize_actions([row])
    assert metrics['correct'] == 1 and metrics['operational_agreement'] == 0
    assert metrics['wrong_accepted'] == 0 and metrics['wrong_accepted_cases'] == 1


ROOT = Path(__file__).resolve().parents[1]
WHITE_BOX = {'diagnostics', 'continuations', 'native_segmentation', 'execution_shape', 'execution_reproduction', 'backend_diagnostics'}


def test_bench_modules_import_without_side_effects():
    results = ROOT / 'bench' / 'results'
    before = {path: path.stat().st_mtime_ns for path in results.rglob('*')}
    modules = sorted(path.stem for path in (ROOT / 'bench').glob('*.py') if path.stem != '__init__')
    code = ("import importlib, sys\nfrom semantic_switch import api\n"
            "def trap(*args, **kwargs):\n    raise SystemExit('Runtime constructed')\n"
            "api.Runtime.__init__ = trap\nsys.argv = ['x', 'model.gguf']\n"
            f"for name in {modules!r}:\n    importlib.import_module('bench.' + name)\n")
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    assert {path: path.stat().st_mtime_ns for path in results.rglob('*')} == before


def test_benchmark_resume_uses_the_shared_journal(tmp_path):
    output = tmp_path / 'result.json'
    keys = [work_key('case', name) for name in ('one', 'two', 'three')]
    with resumable(output, {'model': 'a'}) as journal:
        for key in journal.pending(keys)[:2]:
            journal.commit([(key, {'value': key})])
    with open(f'{output}.journal.jsonl', 'a') as log:
        log.write('{"schema": 1, "kind": "completion", "sess')
    with resumable(output, {'model': 'a'}) as resumed:
        assert resumed.pending(keys) == keys[2:]
        resumed.commit([(keys[2], {'value': keys[2]})])
        artifact = finish(resumed, output, lambda rows: {'rows': rows, 'total': len(rows)})
    assert artifact['total'] == 3 and artifact['execution']['completed'] == 3 and files.read_json(output) == artifact
    assert [row['key'] for row in files.read_jsonl(f'{output}.journal.jsonl')[1:]] == keys
    with pytest.raises(ConfigError, match='different run'):
        resumable(output, {'model': 'b'})
    with resumable(output, {'model': 'b'}, fresh=True) as fresh:
        assert fresh.pending(keys) == keys
    assert len(list(tmp_path.glob('result.json.journal.jsonl.2*'))) == 1


def test_native_model_and_requests_shared(tmp_path):
    from semantic_switch import Model, artifacts
    from semantic_switch.api import native_settings
    from semantic_switch.protocols import ScoreRequest
    lora = pytest.importorskip('semantic_trainer.lora')
    jobs = pytest.importorskip('semantic_trainer.jobs')
    select = pytest.importorskip('semantic_trainer.select')
    assert all(module.native_model is artifacts.native_model for module in (lora, jobs, select))
    assert lora.native_score_requests is artifacts.native_score_requests
    gguf, merged = tmp_path / 'base.gguf', tmp_path / 'merged.gguf'
    gguf.write_bytes(b'base')
    merged.write_bytes(b'merged')
    resolved = {'native': native_settings(Model(str(gguf), device='cpu', threads=4)), 'model': {'gguf': str(gguf)}}
    assert 'source' not in resolved['native'] and 'provenance' not in resolved['native']
    assert artifacts.native_model(resolved) == Model(str(gguf), device='cpu', threads=4)
    assert artifacts.native_model(resolved, merged) == Model(str(merged), device='cpu', threads=4)
    rows = [{'native': {'presentations': [{'prefix': [1], 'input': [2], 'question': [3], 'answer_slots': [[4], [5, 6]]}]}}]
    assert artifacts.native_score_requests(rows) == [ScoreRequest((1,), (2,), ((3,),), (((4,), (5, 6)),))]


def test_trainer_uses_public_scoring_only():
    sources = [*(ROOT / 'packages/semantic-trainer/src/semantic_trainer').glob('*.py'),
               *(path for path in (ROOT / 'bench').glob('*.py') if path.stem not in WHITE_BOX)]
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            assert not (isinstance(node, ast.Attribute) and node.attr == 'scorer'), f'{path}:{node.lineno}'
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.startswith('semantic_switch.backend') for name in names), f'{path}:{node.lineno}'
            if isinstance(node, ast.ImportFrom) and node.module == 'semantic_switch':
                assert 'backend' not in {alias.name for alias in node.names}, path
