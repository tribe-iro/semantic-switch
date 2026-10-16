from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_switch import Actor, Runtime, ConfigError, files, tools
from semantic_switch.artifacts import load_bundle, read_export
from semantic_switch.compiler import digest
from semantic_switch.selection import Selection
from semantic_switch.tools.common import evidence_interval, wilson
from semantic_switch.tools.data import Labels
from semantic_switch.tools.policy import choose, validate

from semantic_testkit.scoring import FakeScorer
from semantic_testkit.labels import envelopes


def test_feature_identity_unicode_and_hash_collision():
    import zlib
    from semantic_switch.heads import FEATURE_IDENTITY, N_FEATURES, features
    from semantic_testkit.scoring import fixture as load
    fixture = load('lexical-feature-v1.json')
    expected = {int(k): v for k, v in fixture['features'].items()}
    assert fixture['feature_identity'] == FEATURE_IDENTITY
    assert features(fixture['text']) == expected == features(fixture['equivalent'])
    a, b, index = fixture['collision']
    assert a != b and zlib.crc32(a.encode()) % N_FEATURES == zlib.crc32(b.encode()) % N_FEATURES == index


def definition(tmp_path):
    path = tmp_path / 'decision.yaml'
    files.publish(path, {'name': 'decision', 'fields': {'message': {'type': 'text', 'required': True}},
                           'reads': ['message'], 'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}})
    return path


def dataset(path, n=160):
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([
        {'input': {'message': f'{"refund" if i % 2 else "hello"} item{i}'}, 'facts': {'refund': bool(i % 2)},
         'group': f'family{i}', 'selection': {'selected_by': 'random', 'frame_id': 'audit', 'independent': True}}
        for i in range(n)])))


def test_family_assignment_survives_new_smaller_alias(tmp_path):
    path = definition(tmp_path)
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([{'input': {'message': 'hello'}, 'facts': {'refund': False}, 'group': 'zz'}])))
    with Runtime(FakeScorer()) as rt:
        before = Labels(rt.load(path))
        original = before.examples('refund')[0]
        files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([{'input': {'message': 'hello'}, 'facts': {'refund': False}, 'group': 'aa'}])))
        after = Labels(rt.load(path))
        current = after.examples('refund')[0]
        assert (current.group, current.split, current.fold) == (original.group, original.split, original.fold)
        assert after.dataset_id != before.dataset_id


def test_family_merge_quarantines_old_fold_or_partition_conflict(tmp_path):
    path = definition(tmp_path)
    dataset(path, 20)
    with Runtime(FakeScorer()) as rt:
        labels = Labels(rt.load(path))
        a = labels.examples('refund')[0]
        b = next(e for e in labels.examples('refund') if (e.split, e.fold) != (a.split, a.fold))
        records = files.read_jsonl(path.with_suffix('.test.jsonl'))
        group = next(r['groups'][0] for r in records if r['input'] == b.data)
        files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([{'input': a.data, 'facts': {'refund': a.label == 0}, 'group': group}])))
        changed = Labels(rt.load(path))
        affected = next(e for e in changed.examples('refund') if e.id == a.id)
        assert affected.split == 'conflict'
        assert any('conflict' in warning for warning in changed.warnings)


def test_export_lexical_does_not_load_a_judge_and_is_verified(tmp_path):
    path = definition(tmp_path)
    dataset(path)
    class NoJudge(Runtime):
        def judge(self, role='judge'):
            raise AssertionError('lexical export loaded model')
    with NoJudge() as rt:
        exported = tools.export(rt, path)
    manifest, rows = read_export(exported['path'])
    assert rows and all('native' not in row and 'presentations' not in row for row in rows) and manifest['native'] is None
    assert manifest['payloads']['decision.export.observations.jsonl']['count'] == len(rows) and manifest['counts']['segments'] == 0
    with open(Path(exported['path']).with_name(manifest['payloads']['decision.export.observations.jsonl']['file']), 'a') as stream:
        stream.write('{}\n')
    with pytest.raises(ConfigError, match='incomplete publication'):
        read_export(exported['path'])


def test_audit_labels_do_not_change_frozen_candidate(tmp_path):
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        original, _, _ = tools.build_lock(rt, path)
        labels = Labels(rt.load(path))
        check_inputs = {e.projection for e in labels.examples('refund', 'check')}
        records = files.read_jsonl(path.with_suffix('.test.jsonl'))
        for row in records:
            if labels.projection(labels.sw.plan.fact('refund'), row['input']) in check_inputs:
                row['facts']['refund'] = not row['facts']['refund']
        files.publish(path.with_suffix('.test.jsonl'), ''.join([json.dumps(r) + '\n' for r in records]))
        changed, _, _ = tools.build_lock(rt, path)
    assert original['candidate_id'] == changed['candidate_id']
    assert original['facts']['refund']['policies']['judge']['true']['theta'] == changed['facts']['refund']['policies']['judge']['true']['theta']
    assert original['facts']['refund']['policies']['judge']['true']['held_out_precision'] != changed['facts']['refund']['policies']['judge']['true']['held_out_precision']


def test_supplied_head_survives_audit_only_changes(tmp_path, head_bundle):
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        bundle = head_bundle(rt, path, 'refund')
        before, _, _ = tools.build_lock(rt, path, head_bundles={'refund': bundle})
        labels = Labels(rt.load(path))
        check = {e.projection for e in labels.examples('refund', 'check')}
        rows = files.read_jsonl(path.with_suffix('.test.jsonl'))
        for row in rows:
            if labels.projection(labels.sw.plan.fact('refund'), row['input']) in check:
                row['facts']['refund'] = not row['facts']['refund']
        files.publish(path.with_suffix('.test.jsonl'), ''.join([json.dumps(r) + '\n' for r in rows]))
        after, _, _ = tools.build_lock(rt, path, head_bundles={'refund': bundle})
        assert before['candidate_id'] == after['candidate_id']
        assert before['dataset']['id'] != after['dataset']['id']


def test_selection_hygiene_and_blind_stale_label(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(path)
        sw.correct('refund', fact='refund', value=True, groups=['g'], selections=[{'selected_by': 'uncertain', 'selection_id': 's'}], actor=Actor('human'))
        assert sw.snapshot.examples['refund'][0].selections[0].selected_by == 'uncertain'
        spec = files.read_yaml(path)
        spec['facts']['refund']['check'] = 'The customer explicitly wants a refund.'
        files.publish(path, spec)
        prompts, shown = [], []
        tools.label(rt, path, None, stale=True, ask=lambda prompt: prompts.append(prompt) or 'n', show=shown.append, actor=Actor('human'))
        assert 'was' not in prompts[0] and 'True' not in prompts[0]
        assert 'previous label' in shown[-1]
        record = files.read_jsonl(sw.paths.examples)[-1]
        assert record['groups'] == ['g'] and record['value'] is False and record['selections'][0]['selected_by'] == 'uncertain'
        assert [source.get('replaces') for source in record['label_sources']] == [None, True]
    with pytest.raises(ConfigError):
        Selection.parse({'inclusion_probability': float('nan')})


def test_one_interval_api_respects_sampling_unit():
    assert wilson(0, 0) is None
    assert wilson(20, 20)[0] < 1
    selected = Selection('random', frame_id='audit', independent=True, inclusion_probability=.1)
    independent = [(str(i), (selected,), True, True) for i in range(20)]
    assert evidence_interval(independent)['interval'] == wilson(20, 20)
    assert evidence_interval([('unknown', (replace(selected, inclusion_probability=None),), True, True)])['interval'] is None
    assert evidence_interval([(None, (selected,), True, True)])['reason'] == 'no canonical family assignment'
    dependent = [('family', (selected,), True, True)] * 20
    assert evidence_interval(dependent)['interval'] is None
    family = replace(selected, unit='family')
    grouped = [('family', (family,), True, True)] * 20
    assert evidence_interval(grouped)['trials'] == 1
    mixed = evidence_interval([('family', (family,), True, True), ('family', (family,), True, False)])
    assert (mixed['metric'], mixed['successes'], mixed['trials'], mixed['interval']) == ('all_accepted_correct', 0, 1, wilson(0, 1))
    manual = replace(family, selected_by='manual')
    assert evidence_interval([(str(i), (manual,), True, True) for i in range(20)])['interval'] is None
    with pytest.raises(ConfigError):
        wilson(True, 2)


def test_breakpoint_sweep_finds_between_grid_threshold():
    enough, threshold = choose([(.505, True)] * 20 + [(.501, False)] * 20, .95)
    assert enough and threshold == .505
    assert validate([(.97, False)] * 30, .95, .9, .7)['state'] == 'missed'


def test_events_order_failure_and_no_raw_input(tmp_path):
    path = definition(tmp_path)
    seen = []
    with Runtime(FakeScorer(), on_decision=seen.append) as rt:
        rt.ask_many(['refund secret', 'hello private'], {'a': __import__('semantic_switch').Check('The customer asks to receive money back for a charge.')})
        rt.load(path)('refund secret')
    assert [event.operation for event in seen] == ['inline', 'inline', 'switch']
    assert seen[1].facts['a']['score'] >= seen[1].facts['a']['policy']['false']
    assert seen[1].facts['a']['scores']['false'] > seen[1].facts['a']['scores']['true']
    assert all('secret' not in repr(event) and 'private' not in repr(event) for event in seen)
    with pytest.raises(TypeError):
        seen[0].facts['a'] = {}
    def failing(event):
        raise RuntimeError('failure')
    with Runtime(FakeScorer(), on_decision=failing) as rt, pytest.warns(RuntimeWarning):
        assert rt.check('refund', 'The customer asks to receive money back for a charge.').status == 'ok'


def test_runtime_workflow_without_training_imports(tmp_path):
    path = definition(tmp_path)
    script = '''import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('sklearn', 'scipy', 'semantic_trainer'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from semantic_switch import Runtime, tools
from semantic_testkit.scoring import FakeScorer
with Runtime(FakeScorer()) as rt:
    tools.export(rt, PATH)
    tools.lock(rt, PATH)
    tools.run_test(rt, PATH)
'''.replace('PATH', repr(str(path)))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bundle_rejects_disclosed_fold_assignment_changes(tmp_path, head_bundle):
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        bundle = head_bundle(rt, path, 'refund')
        manifest_path = bundle / 'manifest.json'
        manifest = files.read_json(manifest_path)
        manifest['rows'][0]['fold'] = (manifest['rows'][0]['fold'] + 1) % 5
        for number, entry in manifest['folds'].items():
            entry['fit_ids'] = sorted(r['id'] for r in manifest['rows'] if r['fold'] != int(number))
            entry['held_ids'] = sorted(r['id'] for r in manifest['rows'] if r['fold'] == int(number))
        with files.open_published(manifest_path) as (_, heads):
            payload = heads['head.json'].read()
        files.publish(manifest_path, {k: v for k, v in manifest.items() if k != 'payloads'}, {'head.json': [payload]})
        with pytest.raises(ConfigError, match='fit observations'):
            tools.lock(rt, path, head_bundles={'refund': bundle})


def test_changed_candidate_reports_prior_audit_exposure(tmp_path):
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        first, _, _ = tools.build_lock(rt, path)
        spec = files.read_yaml(path)
        spec['otherwise'] = 'changed action'
        files.publish(path, spec)
        second, _, _ = tools.build_lock(rt, path)
    assert first['candidate_id'] != second['candidate_id']
    assert second['evidence']['audit_use']['state'] == 'consulted'
    assert second['facts']['refund']['policies']['judge']['true']['interval_95'] is None


@pytest.mark.parametrize('mutation', [lambda d: d.update(feature_identity='wrong'),
                                     lambda d: d.update(n=True), lambda d: d.update(w='!'),
                                     lambda d: d.update(classes=['true', 'true'])])
def test_head_artifact_shape_is_checked_before_installation(tmp_path, head_bundle, mutation):
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        bundle = head_bundle(rt, path, 'refund')
        with files.open_published(bundle / 'manifest.json') as (manifest, heads):
            payload = json.load(heads['head.json'])
        mutation(payload)
        files.publish(bundle / 'manifest.json', {k: v for k, v in manifest.items() if k != 'payloads'}, {'head.json': [json.dumps(payload)]})
        with pytest.raises(ConfigError):
            load_bundle(bundle)


def test_auc_ties_and_large_input_share_one_rank_implementation():
    from semantic_switch.metrics import auc
    assert auc([.5, .5], [True, False]) == .5
    assert auc(list(range(10000)), [False] * 5000 + [True] * 5000) == 1.
    assert auc([.1], [True]) is None


@pytest.mark.parametrize('mutation', [lambda d: d.update(tuned_keys='bad'),
                                     lambda d: d.update(returned_models=[]),
                                     lambda d: d['facts']['refund'].update(policies=[]),
                                     lambda d: d['facts']['refund'].update(head={}),
                                     lambda d: d['facts']['refund']['policies']['judge']['true'].update(theta=True),
                                     lambda d: d['facts']['refund']['policies']['judge']['true'].update(held_out_accepted=-1),
                                     lambda d: d['facts']['refund']['policies']['judge']['true'].update(interval_95=[.9, .1])])
def test_lock_shape_failures_are_configuration_errors(tmp_path, mutation):
    from semantic_switch.snapshot import from_lock, reconcile
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        data, _, _ = tools.build_lock(rt, path)
        mutation(data)
        data['id'] = digest({k: v for k, v in data.items() if k not in ('id', 'created')})[:12]
        with pytest.raises(ConfigError):
            reconcile(from_lock(data, 'corrupted lock'), rt, False, 'corrupted lock')
