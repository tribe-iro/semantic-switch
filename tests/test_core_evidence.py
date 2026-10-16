from __future__ import annotations

import itertools
import random
import threading

import numpy as np
import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.compiler import compile_spec, digest, is_digest
from semantic_switch.examples import Claim, LabelSource, collect, resolve_claims
from semantic_switch.families import FamilyIndex, exact_families
from semantic_switch.selection import Selection
from semantic_switch.training import Actor, Task

HUMAN = Actor('human')
SPEC = {'name': 'ops', 'fields': {'ticket.body': {'type': 'text', 'required': True}, 'ticket.tier': {'type': 'text'}}, 'reads': ['ticket.body'],
        'facts': {'action': {'pick': 'Which response?', 'options': {'escalate': 'Request', 'continue': 'Continue', 'clarify': 'Ask'}},
                  'refund': {'check': 'The customer asks for money back.'}}}
DATA = {'ticket': {'body': 'printer overheating', 'tier': 'gold'}}
RANDOM = {'selected_by': 'random', 'inclusion_probability': .5, 'selection_id': 's1'}
SELECTED = Selection.parse(RANDOM).record()


@pytest.fixture
def path(tmp_path):
    path = tmp_path / 'ops.yaml'
    files.publish(path, SPEC)
    return path


def chain(example):
    return [(source.producer, source.replaces if source.boundary else '-') for source in example.sources]


def test_live_reload_and_shuffled_replay_apply_one_correction_reduction(path):
    with Runtime() as runtime:
        switch = runtime.load(path, draft=True)
        for value in ('escalate', 'continue', 'clarify', 'continue', 'continue'):
            switch.correct(DATA, fact='action', value=value, actor=HUMAN, author='ana', selections=[RANDOM])
        (live,) = switch.snapshot.examples['action']
        assert chain(live) == [('ana', '-'), ('ana', 'escalate'), ('ana', 'continue'), ('ana', 'clarify'), ('ana', '-')] and live.value == 'continue'
        assert [s.record() for s in live.selections] == [SELECTED]
        (reloaded,) = runtime.load(path, draft=True).snapshot.examples['action']
        records = files.read_jsonl(switch.paths.examples)
        for seed in range(4):
            random.Random(seed).shuffle(records)
            (replayed,) = collect(switch.plan, records + records[:2])[0]['action']
            assert [s.digest for s in replayed.sources] == [s.digest for s in reloaded.sources] == [s.digest for s in live.sources]


def test_independent_corrections_stay_alternatives_until_a_decision_names_every_predecessor(path):
    with Runtime() as runtime:
        switch = runtime.load(path, draft=True)
        switch.correct(DATA, fact='action', value='escalate', sources=[LabelSource('judged', 'teacher-a')])
        switch.correct(DATA, fact='action', value='continue', sources=[LabelSource('judged', 'teacher-b')])
        assert switch.snapshot.examples['action'] == () and any('competing corrections' in w for w in switch.warnings)
        switch.correct(DATA, fact='action', value='clarify', actor=HUMAN, author='lead')
        (decided,) = runtime.load(path, draft=True).snapshot.examples['action']
        assert decided.value == 'clarify' and [s.producer for s in decided.sources] == ['teacher-a', 'teacher-b', 'lead'] and len(decided.sources[-1].resolves) == 2


def test_missing_and_explicit_null_replaces_differ():
    explicit = LabelSource('authored', 'ana', actor=HUMAN, replaces=None)
    plain = LabelSource('authored', 'ana', actor=HUMAN)
    assert explicit.record()['replaces'] is None and 'replaces' not in plain.record() and explicit.identity != plain.identity
    assert LabelSource.parse(explicit.record()).boundary and not LabelSource.parse(plain.record()).boundary
    unknown = LabelSource('unknown', 'unknown')
    assert resolve_claims([Claim(None, (unknown,)), Claim('escalate', (unknown, explicit))]).value == 'escalate'


def test_stale_relabel_keeps_old_history_and_sampling_metadata(path):
    with Runtime() as runtime:
        runtime.load(path, draft=True).correct(DATA, fact='refund', value=True, actor=HUMAN, author='ana', selections=[RANDOM], groups=['thread'])
        files.publish(path, {**SPEC, 'facts': {**SPEC['facts'], 'refund': {'check': 'The customer wants money returned.'}}})
        assert tools.label(runtime, path, stale=True, ask=lambda prompt: 'n', show=lambda text: None, reviewer='lead', actor=HUMAN) == 1
        (current,) = runtime.load(path, draft=True).snapshot.examples['refund']
    assert current.value is False and chain(current) == [('ana', '-'), ('lead', True)]
    assert [s.record() for s in current.selections] == [SELECTED] and current.groups == ('thread',)


def test_label_arguments_are_validated_before_opening_the_switch(tmp_path):
    with pytest.raises(ConfigError, match='needs INPUTS'):
        tools.label(None, tmp_path / 'missing.yaml', None, actor=HUMAN)
    with pytest.raises(ConfigError, match='takes no INPUTS'):
        tools.label(None, tmp_path / 'missing.yaml', 'inputs.jsonl', stale=True, actor=HUMAN)


def test_empty_selection_history_is_not_fabricated(path):
    with Runtime() as runtime:
        switch = runtime.load(path, draft=True)
        switch.correct(DATA, fact='refund', value=True, actor=HUMAN)
        assert files.read_jsonl(switch.paths.examples)[0]['selections'] == []
        assert Task(switch).situation(DATA)['selections'] == []


def test_full_frame_identity_follows_nested_and_flat_inputs(path):
    with Runtime() as runtime:
        task = Task(runtime.load(path, draft=True))
    flat = {'ticket.body': 'printer overheating', 'ticket.tier': 'gold'}
    assert task.key(DATA) == task.key(flat) != task.key({'ticket': {'body': 'printer overheating', 'tier': 'silver'}})
    assert task.input_key(DATA) == task.input_key({'ticket': {'body': 'printer overheating', 'tier': 'silver'}})


def test_strict_digest_hashes_complete_typed_arrays():
    left, right = np.arange(2000.), np.arange(2000.)
    right[1000] = -1
    assert repr(left) == repr(right) and digest(left) != digest(right) and digest(np.int64(3)) == digest(3)
    for bad in (np.array([np.nan]), {1, 2}, object()):
        with pytest.raises((TypeError, ValueError)):
            digest(bad)
    assert is_digest('a' * 64) and not any(map(is_digest, ('A' * 64, 'a' * 63, b'a' * 64, None)))


def test_exclusive_focus_and_budget_changes_keep_label_meaning():
    select = {'select': 'Which issues?', 'cards': [{'id': 'heat', 'title': 'Heat', 'applies_when': ['it overheats']},
                                                   {'id': 'jam', 'title': 'Jam', 'applies_when': ['paper jams'], 'supersedes': ['heat']}],
              'max_claims': 4}
    base = {**SPEC, 'facts': {**SPEC['facts'], 'issues': select}, 'rules': [{'id': 'go', 'when': 'issues has "heat"', 'then': 'go'}]}
    plan = compile_spec(base)
    for change, composition in (({'exclusive': True}, True), ({'max_claims': 6}, False)):
        other = compile_spec({**base, 'facts': {**base['facts'], 'issues': {**select, **change}}})
        assert other.fact('issues').meaning_rev == plan.fact('issues').meaning_rev and other.execution_rev != plan.execution_rev
        assert (other.switch_rev != plan.switch_rev) == composition
    focused = compile_spec({**base, 'facts': {**base['facts'], 'action': {**SPEC['facts']['action'], 'focus': 'mentions'}}})
    assert focused.fact('action').meaning_rev == plan.fact('action').meaning_rev and focused.execution_rev != plan.execution_rev
    with pytest.raises(ConfigError, match='specialize'):
        compile_spec({**SPEC, 'facts': {**SPEC['facts'], 'refund': {**SPEC['facts']['refund'], 'specialize': 'lexical'}}})
    with pytest.raises(ConfigError, match='cases'):
        compile_spec({**SPEC, 'cases': {}})


def test_family_lookups_never_create_nodes_and_identities_ignore_join_order():
    edges = [(('key', str(i)), ('key', str(i + 1))) for i in range(40)] + [(('group', 'g'), ('key', '7'))]
    seen = set()
    for seed in range(5):
        index, order = FamilyIndex(), edges[:]
        random.Random(seed).shuffle(order)
        for a, b in order:
            index.join(a, b)
        seen.add((index.find(('key', '39')), index.name(('key', '0')), index.seeds(('key', '3'))))
        assert index.groups(('key', '20')) == {'g'}
        size, roots = len(index), index.roots()
        for lookup in (index.find, index.name, index.seeds, index.history, index.groups):
            with pytest.raises(ConfigError):
                lookup(('key', 'absent'))
        assert (len(index), index.roots()) == (size, roots)
    assert len(seen) == 1


def test_exact_family_quota_matches_subset_enumeration():
    for seed in range(60):
        rng = random.Random(seed)
        sizes = {f'f{k}': rng.randint(1, 6) for k in range(rng.randint(1, 7))}
        count = rng.randint(1, 20)
        feasible = {sum(sizes[f] for f in subset) for n in range(len(sizes) + 1) for subset in itertools.combinations(sizes, n)}
        if count in feasible:
            chosen = exact_families(sizes, count, seed)
            assert sum(sizes[f] for f in chosen) == count
        else:
            with pytest.raises(ConfigError, match='exactly'):
                exact_families(sizes, count, seed)


def test_input_only_situations_count_against_the_byte_budget(path):
    with Runtime() as runtime:
        task = Task(runtime.load(path, draft=True))
    held = task.accumulator(max_situations=10, max_claims=0, max_bytes=120)
    held.add(task.unit(task.situation({'ticket': {'body': 'x' * 40}})))
    held.add(task.unit(task.situation({'ticket': {'body': 'x' * 40}}, groups=['g'])))
    with pytest.raises(ConfigError, match='acquisition budget'):
        held.add(task.unit(task.situation({'ticket': {'body': 'y' * 80}})))
    assert held.retained[0] == 1 and held.retained[2] <= 120


def test_locks_are_reentrant_shared_and_kept_out_of_artifact_directories(tmp_path):
    target, refused = tmp_path / 'artifact.json', []

    def contend():
        try:
            with files.locked(target, blocking=False):
                pass
        except ConfigError as error:
            refused.append(str(error))
    with files.locked(target):
        with files.locked(target):
            worker = threading.Thread(target=contend)
            worker.start()
            worker.join(5)
            assert files.lock_file(target).exists()
    assert refused and 'another thread' in refused[0] and not list(tmp_path.iterdir())
    with files.locked(target, blocking=False):
        pass
    with pytest.raises(TypeError):
        files.append(tmp_path / 'log.jsonl', [{'x': 1}])


def test_prune_keeps_every_committed_generation_and_removes_obsolete_sidecars(tmp_path):
    manifest = tmp_path / 'm.json'
    first = files.publish(manifest, {'v': 1}, {'rows.jsonl': [b'{"a": 1}\n']})['payloads']['rows.jsonl']['file']
    second = files.publish(manifest, {'v': 2}, {'rows.jsonl': [b'{"a": 2}\n']})['payloads']['rows.jsonl']['file']
    (tmp_path / 'm.json.write-lock').write_text('')
    (tmp_path / 'rows.notageneration.jsonl').write_text('')
    assert sorted(files.prune(tmp_path)) == sorted([first, 'm.json.write-lock'])
    with files.open_published(manifest) as (found, handles):
        assert found['v'] == 2 and handles['rows.jsonl'].read() == b'{"a": 2}\n'
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(['m.json', second, 'rows.notageneration.jsonl'])
