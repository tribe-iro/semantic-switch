import json

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.examples import reviewed_content
from semantic_switch.training import LabelSource, TrainingUse, current, load_receipt
from semantic_learn.common import task_of
from semantic_switch.training import promote
from semantic_learn.rounds import affected, coverage, eligible, run_round, tombstone
from semantic_switch.training import specialization_request as direct
from semantic_testkit.labels import TEACHER, declared

from conftest import calls, definition, receipt

HUMAN = [{'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}}]
REFERENCE = [{'derivation': 'computed', 'producer': 'refund-reference', 'revision': '1'}]
ALLOWED = [LabelSource('judged', TEACHER, training_use=TrainingUse.parse(declared())).record()]
UNDECLARED = [{'derivation': 'judged', 'producer': TEACHER}]
JEV_CONFIRMED = [{'derivation': 'judged', 'producer': 'openrouter:typesafe/jev-1.13'}]
EARLY, CUTOFF, LATE = '2026-09-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00', '2026-10-05T00:00:00+00:00'
SETTINGS = dict(devices={'fit': 'cpu', 'native': 'cpu'}, model={'repo': 'example/tiny', 'revision': 'a' * 40})


def labelled(task, message, value, sources, *, groups=(), created=EARLY):
    return task.label({'message': message}, {'refund': value}, sources={'refund': sources}, groups=list(groups), created=created)


def corpus(tmp_path):
    path = definition(tmp_path)
    task = task_of(path)
    (tmp_path / 'old').mkdir()
    stale_task = task_of(definition(tmp_path / 'old', 'The customer wants money back.'))
    jev = labelled(task, 'refund via jev', True, JEV_CONFIRMED, groups=['j'])
    jev['label_sources']['refund'][0]['verifications'] = [{'kind': 'label', 'verifier': 'ana', 'actor': {'kind': 'human'}, 'method': 'review', 'criterion': 'meaning',
                                                           'content': reviewed_content('label', {'message': 'refund via jev'}, ('refund',), True),
                                                           'outcome': 'confirm'}]
    rows = [labelled(task, 'please refund me', True, REFERENCE, groups=['a']), labelled(task, 'hello there', False, HUMAN, groups=['b']),
            labelled(task, 'refund the charge', True, ALLOWED, groups=['c']), labelled(task, 'refund another charge', True, ALLOWED, groups=['d']),
            labelled(task, 'bridge refund', True, HUMAN, groups=['a', 'b']), labelled(stale_task, 'old meaning', True, HUMAN, groups=['e']),
            labelled(task, 'late refund', True, HUMAN, groups=['f'], created=LATE), jev, labelled(task, 'undeclared refund', True, UNDECLARED, groups=['u'])]
    labels = tmp_path / 'accumulated.jsonl'
    files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
    store = tmp_path / 'deleted.jsonl'
    tombstone(store, path, {'message': 'bridge refund'}, reason='erasure request 7')
    return path, labels, store


def recipe(path, labels):
    target = labels.parent / 'recipe' / 'request.json'
    files.publish(target, direct(path, labels, target.parent, **SETTINGS))
    return target


def round_(path, labels, work, trainer, store, **overrides):
    options = dict(trainer=str(trainer), request=recipe(path, labels), cutoff=CUTOFF, serving='separate_host', min_reviewed=1, min_per_class=1, deleted=store,
                   caps={f'judged:{TEACHER}': 1})
    return run_round(path, [labels], work, **{**options, **overrides})


def selected(task, rows, tmp_path, *, caps=None, cutoff=CUTOFF):
    from datetime import datetime
    labels = tmp_path / 'selected.jsonl'
    files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
    return eligible(task, [labels], cutoff=datetime.fromisoformat(cutoff), deleted={'situation': set(), 'projection': set()}, caps=caps or {})


def test_round_freezes_eligible_examples_and_invokes_the_trainer_from_the_pinned_base(tmp_path, trainer):
    path, labels, store = corpus(tmp_path)
    result = round_(path, labels, tmp_path / 'round-1', trainer, store)
    record = files.read_json(tmp_path / 'round-1' / 'round.json')
    assert result['state'] == 'completed' and result['eligible'] == 3 and record['eligible']['rows'] == 3
    assert record['excluded'] == {'envelopes': {'stale': 1, 'after_cutoff': 1}, 'situations': {'deleted': 1, 'unauthorized': 2, 'capped': 1}}
    assert record['identity'] == {'situations': 'situation', 'families': 'projection'}
    kept = files.read_jsonl(tmp_path / 'round-1' / 'round.labels.jsonl')
    assert sorted(row['input']['message'] for row in kept) in (['hello there', 'please refund me', 'refund another charge'],
                                                                 ['hello there', 'please refund me', 'refund the charge'])
    bridged = {row['input']['message']: [group for group in row['groups'] if group.startswith('round-bridge:')] for row in kept}
    assert bridged['please refund me'] == bridged['hello there'] and len(bridged['hello there']) == 1
    assert all(not groups for message, groups in bridged.items() if message not in ('please refund me', 'hello there'))
    (invocation,) = calls(trainer)
    request = files.read_json(tmp_path / 'round-1' / 'pipeline' / 'request.json')
    assert invocation[:2] == ['pipeline', str(tmp_path / 'round-1' / 'pipeline' / 'request.json')]
    assert request['sources']['labels']['path'] == str(tmp_path / 'round-1' / 'round.labels.jsonl') and request['model']['revision'] == 'a' * 40
    assert request == direct(path, tmp_path / 'round-1' / 'round.labels.jsonl', tmp_path / 'round-1' / 'pipeline', **SETTINGS)
    assert record['specialization'] == {key: value for key, value in request.items() if key not in ('sources', 'work')}
    assert record['initialization'].startswith('pinned base') and record['serving'] == 'separate_host' and record['cutoff'] == CUTOFF


def test_interrupted_round_resumes_the_same_request_and_changed_inputs_need_a_new_round(tmp_path, trainer):
    path, labels, store = corpus(tmp_path)
    (trainer.parent / 'fail-once').write_text('')
    assert round_(path, labels, tmp_path / 'round', trainer, store)['state'] == 'failed'
    frozen = (tmp_path / 'round' / 'round.json').read_bytes()
    assert round_(path, labels, tmp_path / 'round', trainer, store)['state'] == 'completed'
    first, second = calls(trainer)
    assert first == second and (tmp_path / 'round' / 'round.json').read_bytes() == frozen
    task = task_of(path)
    files.append(labels, files.jsonl([labelled(task, 'one more refund', True, HUMAN, groups=['n'])]))
    with pytest.raises(ConfigError, match='with other inputs'):
        round_(path, labels, tmp_path / 'round', trainer, store)
    assert len(calls(trainer)) == 2


def test_infeasible_or_undeclared_rounds_never_start_the_trainer(tmp_path, trainer):
    path, labels, store = corpus(tmp_path)
    with pytest.raises(ConfigError, match='infeasible.*missing_classes'):
        round_(path, labels, tmp_path / 'round', trainer, store, min_per_class=5)
    with pytest.raises(ConfigError, match='serving policy'):
        round_(path, labels, tmp_path / 'round', trainer, store, serving='background')
    with pytest.raises(ConfigError, match='UTC offset'):
        round_(path, labels, tmp_path / 'round', trainer, store, cutoff='2026-10-01T00:00:00')
    assert calls(trainer) == [] and not (tmp_path / 'round' / 'round.json').exists()


def test_deletion_identifies_affected_locks_and_publication_leaves_active_serving_unchanged(tmp_path, trainer):
    path, labels, store = corpus(tmp_path)
    key = task_of(path).input_key({'message': 'bridge refund'})
    trained, clean = receipt(tmp_path, 'trained', keys=[key, 'f' * 64]), receipt(tmp_path, 'clean', keys=['e' * 64])
    pointer = tmp_path / 'current.json'
    promote(pointer, trained, reason='initial adoption')
    report = affected(store, [trained, clean], pointer)
    (hit,) = report['affected']
    assert hit['receipt'] == load_receipt(trained).id and hit['active'] and hit['uses'] == ['adaptation'] and len(hit['observations']) == 1
    assert 'does not unlearn' in report['remedy']
    assert round_(path, labels, tmp_path / 'round', trainer, store, incumbent=trained)['state'] == 'completed'
    assert current(pointer) == trained and files.read_json(pointer)['id'] == load_receipt(trained).id


def test_human_authorship_is_not_verification_and_any_confirmed_verifier_counts(tmp_path):
    task = task_of(definition(tmp_path))
    agent = {'kind': 'agent', 'model': 'claude-opus-5-5', 'session': 's-1'}

    meaning = task.targets()['refund']['meaning_rev']

    def verified(actor, outcome='confirm', criterion=meaning):
        return [{**HUMAN[0], 'verifications': [{'kind': 'label', 'verifier': 'checker', 'actor': actor, 'method': 'procedure-recompute', 'criterion': criterion,
                                                'content': reviewed_content('label', {'message': 'refund'}, ('refund',), True), 'outcome': outcome}]}]
    counted = [coverage(task, [labelled(task, 'refund', True, history)], min_reviewed=1, min_per_class=0)['targets']['refund']['reviewed']
               for history in (HUMAN, verified({'kind': 'human'}), verified(agent), verified(agent, 'skip'), REFERENCE, UNDECLARED,
                               verified({'kind': 'human'}, criterion='refund@0'), verified(agent, criterion='refund@0'))]
    assert counted == [0, 1, 1, 0, 1, 0, 0, 0]


def test_round_preparation_and_review_settle_correction_histories_exactly_like_core(tmp_path):
    from datetime import datetime
    from itertools import permutations
    from semantic_learn.common import BUDGET
    from semantic_learn.review import Population
    task = task_of(definition(tmp_path))
    corrected = [*ALLOWED, {**HUMAN[0], 'replaces': True}]
    first, second, third = (labelled(task, 'refund me', value, history) for value, history in
                            ((True, ALLOWED), (False, corrected), (True, [*corrected, {'derivation': 'authored', 'producer': 'bo', 'actor': {'kind': 'human'},
                                                                                      'replaces': False}])))
    key = task.key({'message': 'refund me'})
    core = task.accumulator(**BUDGET)
    for record in (first, second, third):
        core.add(task.unit(record))
    expected = core.settle(key, strict=True)[0]
    assert [source['producer'] for source in expected['label_sources']['refund']] == [TEACHER, 'ana', 'bo'] and expected['facts'] == {'refund': True}
    for n, order in enumerate(permutations((first, second, third))):
        labels = tmp_path / f'order-{n}.jsonl'
        files.publish(labels, ''.join(json.dumps(record) + '\n' for record in order))
        records, _ = eligible(task, [labels], cutoff=datetime.fromisoformat(CUTOFF), deleted={'situation': set(), 'projection': set()}, caps={})
        population = Population(task, BUDGET)
        for record in order:
            population.label(record)
        population.settle()
        assert [record['label_sources'] for record in records] == [expected['label_sources']]
        assert population.units[key]['envelope']['label_sources'] == expected['label_sources'] and population.units[key]['unresolved'] == {}
    competing = labelled(task, 'refund me', False, HUMAN)
    labels = tmp_path / 'competing.jsonl'
    files.publish(labels, ''.join(json.dumps(record) + '\n' for record in (third, competing)))
    with pytest.raises(ConfigError, match='round preparation: refund: competing labels'):
        eligible(task, [labels], cutoff=datetime.fromisoformat(CUTOFF), deleted={'situation': set(), 'projection': set()}, caps={})
    core.add(task.unit(competing))
    population = Population(task, BUDGET)
    for record in (competing, first, third, second):
        population.label(record)
    population.settle()
    _, unresolved = core.settle(key, strict=False)
    assert [claim.record() for claim in population.units[key]['unresolved']['refund'].alternatives] == \
        [claim.record() for claim in unresolved['refund'].alternatives]
    assert sorted(claim.value for claim in unresolved['refund'].alternatives) == [False, True]


def test_caps_count_each_settled_situation_once_per_origin_and_corrections_survive(tmp_path):
    task = task_of(definition(tmp_path))
    corrected = [*ALLOWED, {**HUMAN[0], 'replaces': True}]
    original, correction = labelled(task, 'refund me', True, ALLOWED), labelled(task, 'refund me', False, corrected)
    rows, excluded = selected(task, [correction, original, original, correction], tmp_path, caps={f'judged:{TEACHER}': 1})
    assert [row['facts'] for row in rows] == [{'refund': False}] and excluded['situations']['capped'] == 0
    both = labelled(task, 'refund twice over', True, [*REFERENCE, *ALLOWED])
    rows, excluded = selected(task, [both, labelled(task, 'another refund', True, ALLOWED)], tmp_path, caps={f'judged:{TEACHER}': 1})
    assert len(rows) == 1 and excluded['situations'] == {'deleted': 0, 'unauthorized': 0, 'capped': 1}
    rows, _ = selected(task, [both, labelled(task, 'another refund', True, ALLOWED)], tmp_path, caps={'computed:refund-reference': 1})
    assert len(rows) == 2


def test_cutoff_applies_before_accumulation_and_later_corrections_never_backdate(tmp_path):
    task = task_of(definition(tmp_path))
    corrected = [*ALLOWED, {**HUMAN[0], 'replaces': True}]
    original = labelled(task, 'refund me', True, ALLOWED, created='2026-09-30T12:00:00+00:00')
    late = labelled(task, 'refund me', False, corrected, created='2026-10-02T00:00:00+00:00')
    rows, excluded = selected(task, [late, original], tmp_path, cutoff='2026-10-01T00:00:00+00:00')
    assert [row['facts'] for row in rows] == [{'refund': True}] and excluded['envelopes'] == {'stale': 0, 'after_cutoff': 1}
    assert [source['producer'] for source in rows[0]['label_sources']['refund']] == [TEACHER]
    offset = labelled(task, 'refund me', False, corrected, created='2026-10-01T02:00:00+02:00')
    rows, excluded = selected(task, [offset, original], tmp_path, cutoff='2026-10-01T00:00:00+00:00')
    assert [row['facts'] for row in rows] == [{'refund': False}] and excluded['envelopes']['after_cutoff'] == 0
    with pytest.raises(ConfigError, match='UTC offset'):
        selected(task, [{**original, 'created': '2026-09-30T12:00:00'}], tmp_path)
    alone = labelled(task, 'refund later', True, HUMAN, groups=['shared'], created='2026-10-02T00:00:00+00:00')
    rows, _ = selected(task, [alone, labelled(task, 'refund earlier', True, HUMAN, groups=['shared'])], tmp_path, cutoff='2026-10-01T00:00:00+00:00')
    assert len(rows) == 1 and any(group.startswith('round-bridge:') for group in rows[0]['groups'])


def test_collected_labels_carry_their_completion_time_into_rounds(tmp_path, monkeypatch):
    from semantic_learn import teacher as module
    from semantic_testkit.scoring import FakeScorer
    path = definition(tmp_path)
    situations = tmp_path / 'situations.jsonl'
    files.publish(situations, ''.join(json.dumps({'schema': 3, 'input': {'message': message}, 'groups': [message]}) + '\n'
                                      for message in ('refund please', 'hello')))
    monkeypatch.setattr(module, 'judge', lambda source, provider=None: FakeScorer(lambda state, question, width: [.96, .04] if 'refund' in state else [.04, .96]))
    report = module.collect(path, situations, tmp_path / 'labels.collection.json', teacher=TEACHER, training_use=declared(), provider=None)
    collected = files.read_jsonl(report['labels'])
    journal = {entry['payload']['record']['input']['message']: entry['payload']['collected_at'] for entry in files.read_jsonl(report['journal'])[1:]}
    assert all(row['created'] == journal[row['input']['message']] and not row['created'].startswith('collected') for row in collected)
    rows, excluded = selected(task_of(path), collected, tmp_path, cutoff='2999-01-01T00:00:00+00:00')
    assert len(rows) == 2 and excluded['envelopes'] == {'stale': 0, 'after_cutoff': 0}
    rows, excluded = selected(task_of(path), collected, tmp_path, cutoff='2000-01-01T00:00:00+00:00')
    assert rows == [] and excluded['envelopes']['after_cutoff'] == 2
