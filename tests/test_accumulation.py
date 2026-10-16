from __future__ import annotations

from itertools import permutations

import pytest

from semantic_switch import ConfigError, Runtime, files
from semantic_switch.examples import Claim, Criteria, current, resolve_claims, reviewed_content
from semantic_switch.permissions import admit
from semantic_switch.training import Actor, LabelSource, Task, Verification

HUMAN = Actor('human')
DATA = {'message': 'printer overheating again'}
JEV = LabelSource('judged', 'openrouter:typesafe/jev-1.13')
A = LabelSource('judged', 'teacher-a')
B = LabelSource('authored', 'reviewer-1', actor=HUMAN, replaces='escalate')
C = LabelSource('authored', 'reviewer-2', actor=HUMAN, replaces='continue')
D = LabelSource('judged', 'teacher-b')


@pytest.fixture
def task(tmp_path):
    path = tmp_path / 'ops.yaml'
    files.publish(path, {'name': 'ops', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'action': {'pick': 'Which response does the policy require?',
                                              'options': {'escalate': 'Request a technician', 'continue': 'Continue operation', 'clarify': 'Ask first'}},
                                   'refund': {'check': 'The customer asks for money back.'},
                                   'issues': {'select': 'Which issues are raised?', 'max_claims': 2,
                                              'labels': {'heat': 'Overheating', 'jam': 'A paper jam'}}}})
    with Runtime() as runtime:
        return Task(runtime.load(path, draft=True))


def action(task, value, *history, **metadata):
    return task.label(DATA, {'action': value}, sources={'action': list(history)}, **metadata)


def settled(task, records, *, strict=False):
    held = task.accumulator(max_situations=4, max_claims=64, max_bytes=1 << 20)
    for record in records:
        held.add(task.unit(record))
    return held.settle(task.key(DATA), strict=strict)


def chain(envelope):
    return [(source['producer'], source.get('replaces')) for source in envelope['label_sources']['action']]


def test_every_order_and_grouping_of_a_correction_history_resolves_to_its_full_descendant(task):
    first, second, third = action(task, 'escalate', A), action(task, 'continue', A, B), action(task, 'escalate', A, B, C)
    expected = [('teacher-a', None), ('reviewer-1', 'escalate'), ('reviewer-2', 'continue')]
    for x, y, z in permutations((first, second, third)):
        for merged in (task.merge(task.merge(x, y), z), task.merge(x, task.merge(y, z))):
            assert merged['facts'] == {'action': 'escalate'} and chain(merged) == expected
        envelope, unresolved = settled(task, [x, y, z, z, x])
        assert unresolved == {} and envelope['facts'] == {'action': 'escalate'} and chain(envelope) == expected
    assert [source.producer for source in current(Claim.parse('escalate', third['label_sources']['action']).history)] == ['reviewer-2']


def test_a_value_text_match_or_file_order_never_establishes_ancestry(task):
    late = action(task, 'continue', D, created='2026-09-30T00:00:00+00:00')
    corrected = action(task, 'escalate', A, B, C, created='2026-01-01T00:00:00+00:00')
    envelope, unresolved = settled(task, [corrected, late])
    assert set(unresolved) == {'action'} and 'facts' not in envelope
    assert sorted((claim.value, [s.producer for s in claim.history]) for claim in unresolved['action'].alternatives) == \
        [('continue', ['teacher-b']), ('escalate', ['teacher-a', 'reviewer-1', 'reviewer-2'])]
    with pytest.raises(ConfigError, match='action: competing labels'):
        task.merge(late, corrected)


def test_independent_corroboration_keeps_its_provenance_without_moving_the_correction_boundary(task):
    corrected, confirmed = action(task, 'continue', A, B), action(task, 'continue', D)
    for records in ([corrected, confirmed], [confirmed, corrected]):
        envelope, unresolved = settled(task, records)
        assert unresolved == {} and envelope['facts'] == {'action': 'continue'}
        assert chain(envelope) == [('teacher-a', None), ('reviewer-1', 'escalate'), ('teacher-b', None)]
        history = Claim.parse('continue', envelope['label_sources']['action']).history
        assert [source.producer for source in current(history)] == ['reviewer-1', 'teacher-b']
    assert settled(task, [action(task, 'escalate', A), action(task, 'continue', D)])[1].keys() == {'action'}


def test_branched_corrections_stay_unresolved_until_an_explicit_resolution_names_the_alternatives(task):
    jev = action(task, 'escalate', JEV)
    left, right = action(task, 'continue', JEV, LabelSource('authored', 'reviewer-1', actor=HUMAN, replaces='escalate')), \
        action(task, 'clarify', JEV, LabelSource('authored', 'reviewer-3', actor=HUMAN, replaces='escalate'))
    envelope, unresolved = settled(task, [jev, left, right])
    alternatives = unresolved['action'].alternatives
    assert 'facts' not in envelope and sorted(claim.value for claim in alternatives) == ['clarify', 'continue']
    assert all(claim.history[0].producer == JEV.producer for claim in alternatives)
    with pytest.raises(ConfigError, match='at least two retained alternatives'):
        task.resolve(DATA, 'action', alternatives[:1], 'clarify', reviewer='lead', actor=HUMAN)
    decided = task.resolve(DATA, 'action', [claim.record() for claim in alternatives], 'clarify', reviewer='lead', actor=HUMAN)
    for records in permutations([jev, left, right, decided]):
        resolved, unresolved = settled(task, records)
        assert unresolved == {} and resolved['facts'] == {'action': 'clarify'}
        history = Claim.parse('clarify', resolved['label_sources']['action']).history
        assert (history[0].producer, {s.producer for s in history[1:3]}, history[-1].producer) == (JEV.producer, {'reviewer-1', 'reviewer-3'}, 'lead')
        assert set(history[-1].resolves) == {claim.id for claim in alternatives} and [s.producer for s in current(history)] == ['lead']
        with pytest.raises(ConfigError, match='TypeSafe agreement'):
            admit([('action', history)])
    assert task.merge(decided, left)['facts'] == {'action': 'clarify'}


def test_same_value_reached_by_two_correction_branches_needs_an_explicit_resolution(task):
    left, right = action(task, 'continue', A, B), action(task, 'continue', A, LabelSource('authored', 'reviewer-4', actor=HUMAN, replaces='escalate'))
    found = resolve_claims([Claim.parse('continue', left['label_sources']['action']), Claim.parse('continue', right['label_sources']['action'])])
    assert not found.resolved and len(found.alternatives) == 2


def test_partial_select_members_and_targets_merge_without_losing_unrelated_conflicts(task):
    teacher = task.label(DATA, {'issues': {'heat': True}, 'refund': True}, sources={'issues': {'heat': [A]}, 'refund': [A]}, groups=['thread:1'])
    other = task.label(DATA, {'issues': {'jam': False}, 'refund': False}, sources={'issues': {'jam': [D]}, 'refund': [D]}, groups=['thread:2'], tags=['seen'])
    envelope, unresolved = settled(task, [teacher, other])
    assert envelope['facts'] == {'issues': {'heat': True, 'jam': False}} and set(unresolved) == {'refund'}
    assert set(envelope['groups']) == {'thread:1', 'thread:2'} and envelope['tags'] == ['seen']
    assert settled(task, [other, teacher])[0] == envelope
    with pytest.raises(ConfigError, match='refund: competing labels'):
        settled(task, [teacher, other], strict=True)


def test_verifications_merge_by_source_identity_and_old_ones_stay_audit_history(task):
    meaning = task.targets()['action']['meaning_rev']
    confirm = Verification('label', 'ana', HUMAN, 'review', meaning, reviewed_content('label', DATA, ('action',), 'escalate'), 'confirm')
    reviewed, corrected = action(task, 'escalate', A.verified(confirm)), action(task, 'continue', A, B)
    envelope, _ = settled(task, [reviewed, corrected])
    history = envelope['label_sources']['action']
    assert envelope['facts'] == {'action': 'continue'} and [len(source.get('verifications', ())) for source in history] == [1, 0]
    assessed = task.assess(envelope)['action']
    assert assessed.state == 'authored' and len(assessed.stale) == 1
    assert task.assess(reviewed)['action'].state == 'verified'


def test_retained_claims_are_bounded_and_overflow_fails_before_dropping_provenance(task):
    held = task.accumulator(max_situations=1, max_claims=2, max_bytes=1 << 20)
    held.add(task.unit(action(task, 'escalate', A)))
    held.add(task.unit(action(task, 'escalate', A)))
    held.add(task.unit(action(task, 'continue', A, B)))
    with pytest.raises(ConfigError, match='exceed the acquisition budget of 2 claims'):
        held.add(task.unit(action(task, 'escalate', A, B, C)))
    assert held.settle(task.key(DATA), strict=True)[0]['facts'] == {'action': 'continue'}
    with pytest.raises(ConfigError, match='more than 1 situations'):
        held.add(task.unit(task.label({'message': 'other'}, {'action': 'escalate'}, sources={'action': [A]})))


def test_assessment_needs_the_effective_criterion_and_separates_label_from_source_fidelity(task):
    meaning = task.targets()['action']['meaning_rev']
    label = lambda criterion, value='escalate', path=('action',): Verification('label', 'ana', HUMAN, 'review', criterion,
                                                                               reviewed_content('label', DATA, path, value), 'confirm')
    frame = lambda criterion, outcome='confirm': Verification('evidence', 'ana', HUMAN, 'source-check', criterion, reviewed_content('evidence', DATA, ()), outcome)
    declared = {'bindings': {reviewed_content('evidence', DATA, ()): ['binding-2']}}
    cases = [(label(meaning), None, 'verified', 'undeclared'), (label('meaning-0'), None, 'authored', 'undeclared'),
             (label(meaning, 'continue'), None, 'authored', 'undeclared'), (label(meaning, path=('refund',)), None, 'authored', 'undeclared'),
             (frame('binding-2'), None, 'authored', 'undeclared'), (frame('binding-2'), declared, 'verified', 'verified'),
             (frame('binding-1'), declared, 'authored', 'unverified'), (label('binding-2'), declared, 'verified', 'unverified'),
             (frame('binding-2', 'skip'), declared, 'disputed', 'disputed')]
    for entry, review, state, fidelity in cases:
        record = {'input': DATA, 'facts': {'action': 'escalate'}, 'label_sources': {'action': [LabelSource('authored', 'ana', actor=HUMAN, verifications=(entry,)).record()]}}
        found = task.assess(record, review)['action']
        assert (found.state, found.fidelity) == (state, fidelity), (entry, review)
        bound = ['binding-2'] if review else []
        assert found.record()['binding'] == {'revision': 2, 'label': sorted({meaning, *bound}), 'evidence': bound}
    both = action(task, 'escalate', LabelSource('authored', 'ana', actor=HUMAN, verifications=(frame('binding-2'), frame('binding-2', 'skip'))))
    assert (task.assess(both, declared)['action'].state, task.assess(both, declared)['action'].fidelity) == ('conflicting', 'conflicting')
    from semantic_switch.examples import assess
    assert assess((A,), data=DATA, path=('action',), value='escalate', criteria=None).state == 'unassessed'
    assert assess((A.verified(label(meaning)),), data=DATA, path=('action',), value='escalate', criteria=Criteria(frozenset({meaning}))).state == 'verified'


def test_utf8_budget_is_atomic_and_deduplicates_metadata(task):
    from semantic_switch import files
    unit = task.unit(action(task, 'escalate', LabelSource('unknown', '汉' * 100), groups=['汉' * 100], pairs=['é' * 100], tags=['中文']))
    encoded = len(unit.key.encode('utf-8'))
    encoded += sum(claim.size for claims in unit.claims.values() for claim in claims.values())
    encoded += sum(len(files._json(value).encode('utf-8')) for name in ('groups', 'pairs', 'tags') for value in getattr(unit, name))
    encoded += sum(len(files._json(selection.record()).encode('utf-8')) for selection in unit.selections)
    held = task.accumulator(max_situations=2, max_claims=4, max_bytes=encoded)
    held.add(unit)
    assert held.retained[2] == encoded
    held.add(unit)
    assert held.retained[2] == encoded
    before = held.retained
    extra = task.unit(action(task, 'escalate', LabelSource('unknown', '汉' * 100), groups=['汉' * 100, '中文']))
    with pytest.raises(ConfigError, match='acquisition budget'):
        held.add(extra)
    assert held.retained == before
