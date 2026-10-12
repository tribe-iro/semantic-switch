from contextlib import contextmanager
from dataclasses import dataclass
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from semantic_switch import Check, ConfigError, Guidance, Pick, Runtime, files, judge_from, tools
from semantic_switch.cli import app
from semantic_switch.compiler import digest
from semantic_switch.artifacts import candidate_digest
from semantic_switch.hosted import Reply
from semantic_switch.inline import rendered
from semantic_switch.judge import Scored, Scores
from semantic_switch.presentation import identity
from tests.conftest import FakeScorer


class CapturingJudge:
    identity = {'source': 'captured-contract'}
    supports_guidance = True
    max_slots = 20
    batch_inputs = 8
    accelerated = False

    def __init__(self, distribution=(.98, .02)):
        self.distribution = distribution
        self.calls = []

    def score_many(self, items):
        self.calls.extend(items)
        return [Scored(tuple(Scores(np.array(self.distribution), None) for question in questions), 0)
                for state, questions in items]


def test_single_structured_field_preserves_name():
    assert rendered({'message': 'report'}) == 'message: report'


def test_single_field_units_remain_distinct():
    judge = CapturingJudge()
    with Runtime({'judge': judge}) as rt:
        rt.check({'temperature_c': 110}, 'criterion')
        rt.check({'pressure_bar': 110}, 'criterion')
    assert [state for state, _ in judge.calls] == ['temperature_c: 110', 'pressure_bar: 110']


def test_raw_text_rendering_unchanged():
    assert rendered('raw report') == 'raw report'


def test_inline_compiled_bound_evidence_parity(tmp_path):
    judge = CapturingJudge()
    with Runtime({'judge': judge}) as rt:
        check = Check('criterion', guidance='policy')
        check({'message': 'report'}, rt)
        saved = check.save(tmp_path / 'inspection.yaml', rt=rt)
        saved({'message': 'report'})
    assert judge.calls[0] == judge.calls[1]


@pytest.mark.parametrize('revision', ['evidence', 'choices'])
def test_rendering_revision_requires_relock(tmp_path, revision):
    with Runtime({'judge': FakeScorer(lambda *args: [.98, .02])}) as rt:
        sw = Check('criterion').save(tmp_path / 'inspection.yaml', rt=rt)
        sw.correct('report', fact='inspection', value=True)
        tools.lock(rt, sw.paths.yaml)
        data = files.read_json(sw.paths.lock)
        data['models']['judge']['presentation'][revision] = 'previous'
        data['candidate_id'] = candidate_digest(data)
        data['id'] = digest({k: v for k, v in data.items() if k not in ('id', 'created')})[:12]
        files.publish(sw.paths.lock, data)
        with pytest.raises(ConfigError, match='presentation changed.*ss lock'):
            rt.load(sw.paths.yaml, allow_model_change=True)
        tools.lock(rt, sw.paths.yaml)
        assert rt.load(sw.paths.yaml)('report').facts['inspection'].source == 'example'


def test_choice_presentation_revision_requires_relock(tmp_path):
    with Runtime({'judge': CapturingJudge()}) as rt:
        sw = Pick('choose', {'yes': 'full criterion', 'no': 'other'}).save(tmp_path / 'choice.yaml', rt=rt)
        tools.lock(rt, sw.paths.yaml)
        data = files.read_json(sw.paths.lock)
        data['models']['judge']['presentation'].pop('choices')
        data['candidate_id'] = candidate_digest(data)
        data['id'] = digest({k: v for k, v in data.items() if k not in ('id', 'created')})[:12]
        files.publish(sw.paths.lock, data)
        with pytest.raises(ConfigError, match='re-run ss lock'):
            rt.load(sw.paths.yaml)


LONG = 'Inspection information and maintenance observations. ' * 40 + 'Except repaired historical incidents: continue.'


def test_pick_retains_late_exception():
    judge = CapturingJudge()
    with Runtime({'judge': judge}) as rt:
        result = rt.pick('report', 'choose', {'escalate': LONG, 'continue': 'normal'})
    assert judge.calls[0][1][0].options[0] == LONG


def test_full_criteria_inline_saved_parity(tmp_path):
    judge = CapturingJudge()
    with Runtime({'judge': judge}) as rt:
        pick = Pick('choose', {'escalate': LONG, 'continue': 'normal'})
        pick({'message': 'report'}, rt)
        saved = pick.save(tmp_path / 'inspection.yaml', rt=rt)
        saved({'message': 'report'})
    assert judge.calls[0] == judge.calls[1]


def test_full_criteria_focus_fallback(tmp_path):
    scorer = FakeScorer(lambda *args: [.01, .99])
    options = {'a': {'description': LONG, 'aliases': ['alpha']},
               'b': {'description': LONG + ' beta', 'aliases': ['beta']}}
    with Runtime({'judge': scorer}) as rt:
        pick = Pick('choose', options, focus='mentions')
        assert pick('alpha', rt).value == 'b'
        sw = pick.save(tmp_path / 'choice.yaml', rt=rt)
        assert sw('alpha').facts['choice'].value == 'b'
    assert len(scorer.requests) == 4
    assert all('Except repaired historical incidents: continue.' in question for _, question in scorer.requests)
    assert all(' beta' in scorer.requests[i][1] for i in (1, 3))


def test_hosted_transmits_complete_criteria():
    requests = []
    async def send(url, body, timeout):
        requests.append(body)
        return Reply(200, {'model': 'typesafe/jev-1.13', 'answers': {'q0': {'probabilities': {'A': .98, 'B': .02}}}})
    with Runtime({'judge': judge_from('openrouter:typesafe/jev-1.13', transport=send)}, own_judges=True) as rt:
        assert rt.pick('report', 'choose', {'escalate': LONG, 'continue': 'normal'}).value == 'escalate'
    assert requests[0]['questions']['q0']['criteria']['A'] == LONG


def test_judge_without_fit():
    with Runtime({'judge': CapturingJudge()}) as rt:
        assert rt.pick('report', 'choose', {'yes': 'criterion', 'no': 'other'}).value == 'yes'
        assert rt.identity()['presentation'] == identity()


def test_full_question_capacity_failure():
    scorer = FakeScorer()
    with Runtime({'judge': scorer}) as rt:
        result = rt.pick('report', 'choose', {'yes': 'long ' * 2000, 'no': 'other'})
    assert result.status == 'failed' and 'limit: question' in result.error
    assert not scorer.requests


def test_pick_mapping_is_owned():
    options = {'yes': 'criterion', 'no': 'other'}
    pick = Pick('choose', options)
    options['yes'] = 'changed'
    options['third'] = 'added'
    assert pick.fact()['options'] == {'yes': 'criterion', 'no': 'other'}


def test_pick_nested_metadata_is_owned():
    options = {'a': {'description': 'criterion', 'aliases': ['alpha']}, 'b': 'other'}
    pick = Pick('choose', options, focus='mentions')
    options['a']['aliases'].append('beta')
    options['a']['description'] = 'changed'
    assert pick.fact()['options']['a'] == {'description': 'criterion', 'aliases': ['alpha']}


@dataclass
class Candidate:
    name: str
    eligible: bool = True


def test_pick_sequence_membership_is_owned():
    a, b = Candidate('a'), Candidate('b')
    options = [a, b]
    pick = Pick('choose', options, definition=lambda x: x.name)
    options.reverse()
    options.clear()
    assert pick.options == (a, b)


def test_pick_returns_original_candidate_object():
    a, b = Candidate('a'), Candidate('b')
    with Runtime({'judge': CapturingJudge()}) as rt:
        assert rt.pick('report', 'choose', [a, b], definition=lambda x: x.name).value is a


def test_pick_callbacks_remain_dynamic():
    a, b = Candidate('a'), Candidate('b')
    judge = CapturingJudge()
    pick = Pick('choose', [a, b], definition=lambda x: x.name, eligible=lambda x: x.eligible)
    with Runtime({'judge': judge}) as rt:
        assert pick('report', rt).value is a
        a.eligible = False
        b.name = 'updated'
        judge.distribution = (1.,)
        assert pick('report', rt).value is b
    assert judge.calls[-1][1][0].options == ('updated',)


def test_pick_save_returns_owned_serializable_definition(tmp_path):
    pick = Pick('choose', {'a': {'description': 'criterion', 'aliases': ['alpha']}, 'b': 'other'})
    record = pick.fact()
    record['options']['a']['aliases'].append('changed')
    assert pick.fact()['options']['a']['aliases'] == ['alpha']
    json.dumps(pick.fact())
    with Runtime({'judge': CapturingJudge()}) as rt:
        saved = pick.save(tmp_path / 'choice.yaml', rt=rt)
        assert files.read_yaml(saved.paths.yaml)['facts']['choice']['options'] == pick.fact()['options']


def test_direct_guided_choice_public_workflow():
    from examples.quickstart import run
    def response(state, question, width):
        slot = 2 if 'sensor is faulty' in state else 1 if 'now runs' in state else 0
        if width == 2:
            return [.98, .02]
        return [.98 if i == slot else .02 / (width - 1) for i in range(width)]
    with Runtime({'judge': FakeScorer(response)}) as rt:
        output = run(rt)
    assert [row['value'] for row in output['answers']] == ['escalate', 'continue', 'clarify']
    assert output['restored']['facts']['maintenance']['source'] == 'example'


def test_domain_clarification_is_accepted_choice():
    with Runtime({'judge': CapturingJudge((.01, .99))}) as rt:
        result = rt.pick('missing evidence', 'choose', {'continue': 'continue', 'clarify': 'ask for evidence'})
    assert (result.status, result.value) == ('ok', 'clarify')


def test_low_confidence_is_not_clarification():
    with Runtime({'judge': CapturingJudge((.5, .5))}) as rt:
        result = rt.pick('missing evidence', 'choose', {'continue': 'continue', 'clarify': 'ask for evidence'})
    assert (result.status, result.value) == ('uncertain', None)


def test_saved_judgment_workflow_matches_documented_fact_access(tmp_path, monkeypatch):
    from semantic_switch import cli
    with Runtime({'judge': CapturingJudge()}) as rt:
        pick = Pick('choose', {'escalate': 'attention', 'continue': 'normal'}, guidance='policy')
        sw = pick.save(tmp_path / 'maintenance.yaml', rt=rt)
        tools.lock(rt, sw.paths.yaml)
        saved = rt.load(sw.paths.yaml)({'message': 'report'})
        assert saved.answer is None and saved.facts['maintenance'].value == 'escalate'
        @contextmanager
        def session(*args, **kwargs):
            yield rt
        monkeypatch.setattr(cli, 'session', session)
        result = CliRunner().invoke(app, ['--json', 'run', str(sw.paths.yaml)], input='{"message":"report"}\n')
    assert result.exit_code == 0
    assert json.loads(result.stdout)['facts']['maintenance']['value'] == 'escalate'


def test_cli_guided_choice_matches_sdk(tmp_path, monkeypatch):
    from semantic_switch import cli
    path, policy = tmp_path / 'choices.yaml', tmp_path / 'policy.txt'
    files.publish(path, {'escalate': LONG, 'clarify': 'ask for evidence'})
    policy.write_text('Use the authored definitions.')
    judge = CapturingJudge()
    with Runtime({'judge': judge}) as rt:
        expected = rt.pick({'report': 'case'}, 'choose', files.read_yaml(path), guidance=Guidance.load(policy)).as_dict()
        @contextmanager
        def session(*args, **kwargs):
            yield rt
        monkeypatch.setattr(cli, 'session', session)
        result = CliRunner().invoke(app, ['--json', 'pick', 'choose', '--options', str(path), '--guidance', str(policy)], input='{"report":"case"}\n')
    assert result.exit_code == 0 and json.loads(result.stdout) == expected
    assert judge.calls[0] == judge.calls[1]


def test_direct_composed_cohort_policy_identity():
    from bench.policy_workflow import CASES, POLICIES, definition, DOMAIN_ACTIONS
    assert {row['id'] for row in CASES} == set(DOMAIN_ACTIONS)
    for identifier, text in POLICIES.items():
        assert definition(text)['guidance']['returns']['text'] == text


def test_expected_actions_are_not_execution_derived():
    from bench.policy_workflow import CASES, DOMAIN_ACTIONS
    indexed = {row['id']: row for row in CASES}
    assert DOMAIN_ACTIONS['missing_defect'] == 'request_details' and indexed['missing_defect']['action'] is None
    assert DOMAIN_ACTIONS['repaired'] == 'deny_refund' and DOMAIN_ACTIONS['repair_failed'] == 'approve_refund'


def test_domain_action_and_execution_status_remain_distinct():
    from bench.policy_workflow import decision_metrics
    direct = [{'expected_action': 'request_details', 'result': {'status': 'ok', 'value': 'request_details', 'scores': {'request_details': 1.}}}]
    composed = [{'expected_action': 'request_details', 'result': {'status': 'needs_input', 'answer': None}}]
    a, b = decision_metrics(direct, direct=True), decision_metrics(composed)
    assert a['normalized_action_correct'] == b['normalized_action_correct'] == 1
    assert a['accepted_clarifications'] == 1 and b['accepted'] == 0
    assert 'raw_action_correct' not in b


def test_contrast_requires_both_expected_outcomes():
    from bench.policy_workflow import contrast_result
    a = {'expected_action': 'approve_refund', 'result': {'status': 'ok', 'value': 'approve_refund'}}
    b = {'expected_action': 'deny_refund', 'result': {'status': 'ok', 'value': 'request_details'}}
    assert not contrast_result(a, b, 'change', direct=True)['correct']


def test_complete_presented_task_recorded():
    from bench.outcomes import task
    pick = Pick('choose', {'a': LONG, 'b': 'other'}, guidance='policy')
    with Runtime({'judge': CapturingJudge()}) as rt:
        record = task(rt, pick, {'report': 'case'}, result=pick({'report': 'case'}, rt).as_dict(True))
    assert record['rendered_evidence'] == 'report: case' and record['definition']['definition']['options']['a'] == LONG
    assert record['presentations'][0]['answers'][0]['option'] == LONG and record['presentation'] == identity()
