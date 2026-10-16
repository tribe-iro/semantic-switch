import json

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.metrics import prediction_report, zero_failure_bound
from semantic_switch.tools.dataset import allocate


def test_sparse_allocation_is_deterministic_sticky_and_reports_impossible_support():
    families = {str(i): {'size': i % 3 + 1, 'cells': ['positive' if i % 2 else 'negative', f'behavior-{i % 3}']} for i in range(30)}
    families['rare'] = {'size': 17, 'cells': ['once']}
    targets = {'fit': 25, 'model_select': 10, 'check': 25}
    first = allocate(families, targets, pinned={'0': 'check'}, seed=17)
    assert first == allocate(dict(reversed(list(families.items()))), targets, pinned={'0': 'check'}, seed=17)
    assert first['families']['0'] == 'check' and first['families']['rare'] == 'fit'
    assert 'once' in first['deficits']['check'] and 'once' in first['deficits']['model_select']
    extended = allocate({**families, 'new': {'size': 2, 'cells': ['negative']}}, targets, pinned=first['families'], seed=17)
    assert all(extended['families'][family] == role for family, role in first['families'].items())
    assert allocate(families, {**targets, 'fit': 26}, seed=17)['id'] != first['id']
    with pytest.raises(ConfigError, match='pinned'):
        allocate(families, targets, pinned={'missing': 'fit'})


def test_prediction_summary_counts_confident_errors_withholding_and_failure():
    base = {'target': 'refund', 'options': ['true', 'false'], 'gold': 'false', 'tags': ['attribution'], 'pairs': ['incident']}
    rows = [{**base, 'answer': 'true', 'correct': False, 'accepted': True, 'outcome': 'ok'},
            {**base, 'answer': 'false', 'correct': True, 'accepted': False, 'outcome': 'uncertain'},
            {**base, 'answer': None, 'correct': None, 'accepted': False, 'outcome': 'failed'}]
    report = prediction_report(rows)
    assert report['submitted'] == 3 and report['raw_correct'] == 1 and report['wrong_accepted'] == 1
    assert report['correct_rejected'] == 1 and report['wrong_accepted_per_submission'] == pytest.approx(1 / 3)
    assert report['per_class']['refund']['false']['recall'] == pytest.approx(1 / 3)
    assert report['pairs'] == {'assessed': 1, 'all_correct': 0}
    assert zero_failure_bound(1)['upper'] == pytest.approx(.95)
    assert zero_failure_bound(0)['upper'] is None


def test_composed_memberships_keep_incomplete_reference_and_failures():
    from semantic_switch.compiler import compile_spec
    from semantic_switch.tools.deployment import development_summary
    plan = compile_spec({'name': 'operations', 'facts': {'actions': {'select': 'What can proceed?', 'cards': [
        {'id': 'restart', 'applies_when': ['The restart was authorized.']},
        {'id': 'cancel', 'applies_when': ['The cancellation was authorized.']}]}}})
    target = lambda gold, answer, accepted=True: {'gold': gold, 'answer': answer, 'correct': gold == answer,
        'accepted': accepted, 'options': ['true', 'false'], 'outcome': 'ok'}
    rows = [{'status': 'ok', 'prohibited': ['cancel'], 'targets': {'actions/restart': target('true', 'false'), 'actions/cancel': target('false', 'true')}},
            {'status': 'failed', 'targets': {'actions/restart': {**target('true', None, False), 'correct': None, 'outcome': 'failed'}}}]
    result = development_summary(plan, rows)['operational_actions']['sets']['actions']
    assert result['actions']['frames_with_prohibited'] == 1
    assert result['actions']['required_omissions'] == 2
    assert result['exact']['gold_complete'] == 1 and result['exact']['exact'] == 0


def test_cli_frozen_options_use_pinned_source_and_role_without_loading(monkeypatch):
    from types import SimpleNamespace
    from semantic_switch import cli, Model
    from semantic_switch.tools import deployment
    pinned = Model('/not-loaded/specialist.gguf', device='gpu', threads=4)
    receipt = SimpleNamespace(data={'scope': {'binding': 'specialist'}}, model=lambda **settings: Model(pinned.source, **{'device': 'gpu', 'threads': 4, **settings}))
    monkeypatch.setattr(deployment, 'frozen', lambda path: SimpleNamespace(receipt=receipt))
    with cli.session({'judge': None, 'device': 'gpu', 'threads': 4}, source='frozen.json') as runtime:
        assert runtime.models['specialist'] == pinned and not runtime._loaded
    monkeypatch.setattr(cli, 'judge', lambda source, **settings: Model(source, **settings))
    with pytest.raises(ConfigError, match='conflict'):
        with cli.session({'judge': '/different.gguf', 'device': 'gpu', 'threads': 4}, source='frozen.json'):
            pass


def test_cancellation_uses_supply_setup_without_changing_method_meaning(tmp_path):
    from itertools import product
    from bench import procedures
    from tests.test_printer_retirement import decisions
    frames = []
    for index, (prompt, method, media, ribbon) in enumerate(product(procedures.CALIBRATION, (procedures.TT, procedures.DT), (procedures.TT, procedures.DT), (True, False))):
        state = procedures.State('printer', 'ZT411', media, ribbon, method, prompt)
        events = [procedures.ev(0, 'configuration', 'printer', 'print_method', method),
                  procedures.ev(1, 'display', 'printer', 'prompt', prompt),
                  procedures.ev(2, 'observation', 'printer', 'media_type', media),
                  procedures.ev(3, 'observation', 'printer', 'ribbon_loaded', ribbon)]
        frames.append(procedures.render('cancellation', state, events, condition='structured', frame_id=str(index), family=str(index)))
    golds = {frame.id: procedures.reference(frame) for frame in frames}
    states, _ = decisions(procedures.DEFINITION, frames, golds)
    assert all((states[frame.id]['cancel_calibration'] == 'applicable') == golds[frame.id].decisions['actions/cancel_calibration'].value for frame in frames)
    witness = next(gold for gold in golds.values() if gold.primitives['method_inconsistent'].value is False and gold.primitives['calibration_setup_inconsistent'].value is True)
    assert witness.decisions['actions/cancel_calibration'].value is True


def test_input_contract_inspection_keeps_missing_dependencies_separate():
    from semantic_switch.training import inspect_inputs
    path = __import__('pathlib').Path('examples/data/printer.yaml')
    report = inspect_inputs(path, [{'asset': 'printer', 'model': 'ZT411', 'notes': 'The ribbon is fitted.'}])
    assert report['inputs'][0]['state'] == 'missing_dependencies'
    assert 'prompt' in report['inputs'][0]['missing']['ribbon_loaded']


def test_composed_row_reports_real_card_state_without_inventing_confidence(tmp_path):
    from semantic_switch import Runtime
    from semantic_switch.tools.deployment import compared_row
    from semantic_testkit.scoring import FakeScorer
    path = tmp_path / 'actions.yaml'
    files.publish(path, {'name': 'actions', 'fields': {'message': {'type': 'text'}}, 'reads': ['message'],
        'facts': {'allowed': {'check': 'Authorized'}, 'actions': {'select': 'Actions', 'cards': [{'id': 'restart', 'applies_when': [{'fact': 'allowed'}]}]}}})
    with Runtime(FakeScorer(lambda state, question, width: [.99, .01])) as runtime:
        switch = runtime.load(path)
        item = {'input': {'message': 'Authorized'}, 'facts': {'allowed': True}, 'decisions': {'actions/restart': {'value': True}}}
        result = switch(item['input'], facts=['actions'])
        row = compared_row(switch.plan, item, 'incident', result, {})
    target = row['targets']['actions/restart']
    assert target['correct'] is True and target['source'] == 'rule'
    assert target['mass'] is None and target['scores'] == {} and target['prediction']['score'] is None


def test_unscored_baseline_has_no_acceptance_or_withholding_claim():
    row = {'target': 'task', 'options': ['true', 'false'], 'gold': 'true', 'answer': 'true', 'correct': True,
           'accepted': False, 'acceptance': 'unassessed'}
    report = prediction_report([row])
    assert report['raw_correct'] == 1 and report['coverage'] is None and report['precision'] is None
    assert report['correct_rejected'] is None and report['wrong_accepted_per_submission'] is None


def test_budget_limited_qualification_cannot_hide_its_worse_measured_result():
    from semantic_trainer.qualification import summarize
    winner = lambda precision: {'correct': {'macro': {'observation': precision}}, 'precision': precision, 'coverage': 1., 'per_class': {}}
    report = summarize([{'state': 'completed', 'winner': winner(.95)},
                        {'state': 'budget_limited', 'winner': winner(.5)}, {'state': 'failed'}])
    assert report['state'] == 'incomplete' and report['metrics']['precision']['worst'] == .5
    assert report['completed'] == 1 and report['evaluated'] == 2 and report['run_states']['failed'] == 1
