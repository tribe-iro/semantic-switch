import dataclasses

import pytest

from semantic_switch import Check, ConfigError, Pick, Runtime, files
from semantic_switch.compiler import digest
from semantic_switch.guidance import Guidance
from semantic_testkit.scoring import FakeScorer

REFUND = 'The customer asks to receive money back for a charge.'


def definition(tmp_path):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': REFUND},
                                   'issues': {'select': 'Which issues?', 'labels': {'billing': 'An invoice error', 'outage': 'A service outage'}}}})
    return path


def test_events_are_versioned_immutable_and_copy_no_input_unless_requested(tmp_path):
    seen = []
    with Runtime(FakeScorer(), on_decision=seen.append) as rt:
        rt.load(definition(tmp_path)).run([{'message': 'refund secret one'}, {'message': 'refund secret two'}])
    assert [event.schema for event in seen] == [9, 9] and all(event.input is None for event in seen)
    assert seen[0].call == seen[1].call and [event.item for event in seen] == [0, 1] and seen[0].event != seen[1].event
    assert 'secret' not in repr(seen) and 'secret' not in repr([event.as_dict() for event in seen])
    with pytest.raises(dataclasses.FrozenInstanceError):
        seen[0].status = 'ok'
    with pytest.raises(TypeError):
        seen[0].task['targets']['refund'] = {}
    with pytest.raises(ConfigError, match='decision_input'):
        Runtime(FakeScorer(), decision_input=True)


def test_requested_input_and_switch_target_meanings_bindings_and_correlation(tmp_path):
    seen = []
    with Runtime(FakeScorer(), on_decision=seen.append, decision_input=True) as rt:
        switch = rt.load(definition(tmp_path))
        switch({'message': 'a refund please'})
        switch({'message': 'outage now'})
        plan = switch.plan
    first, second = seen
    assert first.input == {'message': 'a refund please'} and second.input == {'message': 'outage now'} and first.call != second.call
    with pytest.raises(TypeError):
        first.input['message'] = 'changed'
    targets = first.task['targets']
    assert set(targets) == {'refund', 'issues/billing', 'issues/outage'} and first.task['switch_rev'] == plan.switch_rev
    assert targets['refund']['meaning_rev'] == plan.fact('refund').meaning_rev == first.facts['refund']['meaning_rev']
    assert targets['issues/outage']['meaning_rev'] == plan.at('issues', 'outage').meaning_rev
    assert {entry['binding'] for entry in targets.values()} == {'judge'}
    assert {entry['identity'] for entry in targets.values()} == {digest(dict(first.predictor['judge']))}


def test_inline_task_identity_includes_questions_options_and_guidance_without_a_lock():
    seen = []
    items = {'c': Check(REFUND), 'p': Pick('Which part failed?', {'ax': 'The axle', 'hub': 'The hub'})}
    with Runtime(FakeScorer(), on_decision=seen.append, decision_input=True) as rt:
        rt.ask_many(['axle broke', 'hub broke'], items)
        rt.ask('axle broke', items, guidance='Only count explicit requests.')
        rt.ask('axle broke', {'c': Check(REFUND), 'p': Pick('Which part failed?', {'ax': 'The axle', 'hub': 'The wheel hub'})})
    plain, other, guided, changed = seen
    assert plain.lock_id is None and plain.call == other.call and (plain.item, other.item) == (0, 1) and other.input == 'hub broke'
    assert plain.task == other.task and plain.task['items']['c']['kind'] == 'check' and plain.task['items']['p']['kind'] == 'pick'
    assert plain.task['items']['p']['definition'] == digest(items['p'].record())
    assert guided.task['items']['c']['guidance'] == Guidance('Only count explicit requests.').digest != plain.task['items']['c']['guidance']
    assert guided.task['items']['c']['meaning_rev'] != plain.task['items']['c']['meaning_rev']
    assert changed.task['items']['p']['meaning_rev'] != plain.task['items']['p']['meaning_rev'] and changed.task['items']['c'] == plain.task['items']['c']
    assert set(plain.predictor) == {'judge'}


def test_a_failing_observer_never_changes_the_decision():
    def failing(event):
        raise RuntimeError('observer failure')
    with Runtime(FakeScorer(), on_decision=failing, decision_input=True) as rt, pytest.warns(RuntimeWarning):
        assert rt.check('please refund me', REFUND).status == 'ok'
