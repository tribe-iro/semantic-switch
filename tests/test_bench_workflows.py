from __future__ import annotations

from pathlib import Path

import pytest

from semantic_switch import DEFAULT_JUDGE, Model, files, tools
from semantic_switch.cli import question_specs
from bench.diagnostics import frozen_replay
from bench.openrouter import scenarios
from bench.outcomes import cli, runtime as model_runtime


def outcomes(value):
    if 'status' in value:
        return {key: value.get(key) for key in ('status', 'value', 'rule', 'answer', 'guidance')}
    return {key: outcomes(result) for key, result in value.items()}


def workflow(root: Path, source=DEFAULT_JUDGE):
    from examples.quickstart import ACTION, POLICY, REPORTS, run as quickstart
    policy, questions, definition = root / 'policy.txt', root / 'questions.yaml', root / 'support.yaml'
    action_options, action_policy = root / 'actions.yaml', root / 'maintenance.txt'
    files.publish(policy, 'Refund means money returned. Store credit and replacements are not refunds.')
    files.publish(questions, {'refund': {'check': 'The customer requests a refund.', 'unstated': True}})
    files.publish(definition, {'name': 'support', 'guidance': {'policy': {'text': policy.read_text()}},
        'fields': {'message': {'type': 'text', 'required': True}, 'allowed': {'type': 'bool', 'required': True}}, 'reads': ['message'],
        'facts': {'refund': {'check': 'The customer requests a refund.', 'unstated': True},
                  'topic': {'pick': 'What is the topic?', 'options': {'billing': {'description': 'charges or payments', 'aliases': ['refund']},
                                                                   'other': 'other subjects'}, 'focus': 'mentions'}},
        'rules': [{'id': 'refund', 'when': 'allowed == true and refund and topic == "billing"', 'then': {'action': 'return_money'}}],
        'otherwise': {'action': 'support'}})
    files.publish(action_options, ACTION.fact()['options'])
    files.publish(action_policy, POLICY.sections[0].text)
    value = 'Please refund the duplicate charge.'
    with model_runtime(Model(source)) as rt:
        contrasts = []
        for identifier, case, guidance, expected, status in scenarios():
            result = rt.check(case, 'The customer requests a refund under the supplied definition.', unstated=True, guidance=guidance)
            contrasts.append({'id': identifier, 'expected': {'value': expected, 'status': status}, 'actual': result.as_dict(True),
                              'correct': (result.value, result.status) == (expected, status)})
        tools.lock(rt, definition)
        switch = rt.load(definition)
        inputs = [{'message': case, 'allowed': True} for _, case, *_ in scenarios()] + [
            {'message': 'Please refund the charge.', 'allowed': False}, {'message': 'Please refund the charge.'}, {'message': 7, 'allowed': True}]
        replay = frozen_replay(rt, switch, inputs)
        sdk = [rt.check(value, 'The customer requests a refund.', unstated=True, guidance=policy.read_text()).as_dict(True),
               {key: result.as_dict(True) for key, result in rt.ask(value, question_specs(questions), guidance=policy.read_text()).items()},
               switch({'message': value, 'allowed': True}).as_dict(True), switch({'message': value}).as_dict(True),
               switch({'message': 7, 'allowed': True}).as_dict(True), ACTION(REPORTS[0], rt).as_dict(True)]
        demonstration = quickstart(rt)
    prefix = ['--judge', source, '--json']
    commands = [([*prefix, 'check', 'The customer requests a refund.', '--unstated', '--guidance', str(policy)], value),
                ([*prefix, 'ask', str(questions), '--guidance', str(policy)], value),
                ([*prefix, 'run', str(definition)], {'message': value, 'allowed': True}),
                ([*prefix, 'run', str(definition)], {'message': value}),
                ([*prefix, 'run', str(definition)], {'message': 7, 'allowed': True}),
                ([*prefix, 'pick', ACTION.question, '--options', str(action_options), '--guidance', str(action_policy)], REPORTS[0])]
    rows = [{**cli(arguments, payload), 'sdk': expected} for (arguments, payload), expected in zip(commands, sdk)]
    for row in rows:
        row['same_public_outcome'] = outcomes(row['result']) == outcomes(row['sdk'])
    return {'contrasts': contrasts, 'cli': rows[:5], 'direct_cli': rows[5], 'native_replay': replay, 'quickstart': demonstration}


@pytest.fixture(scope='module')
def live_workflow(tmp_path_factory):
    return workflow(tmp_path_factory.mktemp('workflow'))


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
    assert replay[-3]['rule'] == 'otherwise' and replay[-3]['cost'].get('judge', 0) == 0
    unknown = replay[4]
    if unknown['status'] == 'ok':
        assert unknown['rule'] == 'otherwise'
        assert any(fact['settled'] and fact['value'] in (False, 'other') for fact in unknown['facts'].values())
    else:
        assert unknown['status'] in ('needs_input', 'uncertain') and unknown['answer'] is None
