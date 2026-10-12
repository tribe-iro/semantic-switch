import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from tests.conftest import FakeScorer
from tests.test_adaptation_contracts import label_file, pipeline_base

RECIPE = {'eval_every_examples': 8}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    from examples import adapt_judgment
    manifest = pipeline_base(tmp_path, monkeypatch)
    rows = files.read_jsonl(label_file(tmp_path, n=12))
    scorer = FakeScorer()
    with Runtime({'judge': scorer}) as runtime:
        definition = adapt_judgment.definition(runtime, tmp_path / 'task')
    def export(name, *members, task='task'):
        labels = tmp_path / f'{name}.jsonl'
        files.publish(labels, ''.join(json.dumps({**rows[index], 'groups': [group], 'role': role}) + '\n' for index, group, role in members))
        with Runtime({'judge': scorer}) as runtime:
            return tools.export(runtime, adapt_judgment.definition(runtime, tmp_path / task), to=tmp_path / f'{name}.export.jsonl', native=True,
                                task_family='maintenance:test', labels=labels)['path']
    selecting = [(index, f'select-{index}', 'model_select') for index in range(6, 12)]
    return manifest, definition, export, selecting


def prepared(tmp_path, cohort, exports, name, **options):
    from semantic_trainer.jobs import prepare, resolve
    manifest, definition, _, _ = cohort
    result = prepare(definition, exports, tmp_path / name, assets_manifest=manifest, recipe=RECIPE, **options)
    return result, *resolve(result['job'])


def test_shared_family_across_exports_is_one_budget_unit_and_gains_later_seeds(tmp_path, cohort):
    from semantic_switch.artifacts import read_export
    _, _, export, selecting = cohort
    first = export('first', (0, 'a', 'fit'), *selecting)
    second = export('second', (0, 'b', 'fit'), (1, 'b', 'fit'), (2, 'b', 'fit'))
    with pytest.raises(ConfigError, match='exactly 1 situations'):
        prepared(tmp_path, cohort, [first, second], 'one', fit_situations=1)
    result, resolved, observations = prepared(tmp_path, cohort, [first, second], 'three', fit_situations=3)
    fit = [row for row in observations if row['role'] == 'fit']
    assert len(fit) == 3 and len({row['component'] for row in fit}) == 1 == resolved['populations']['fit']['families'] == len(result['selection']['families'])
    assert resolved['populations']['fit']['source_components'] == 2 and resolved['populations']['fit']['situations'] == 3
    first_rows, second_rows = read_export(first, native=True)[1], read_export(second, native=True)[1]
    shared = next(row for row in fit if len(row['sources']) == 2)
    alone = next(row for row in first_rows if row['id'] == shared['id'])
    later = {tuple(node) for row in second_rows for node in row['family_seeds'] if node[0] == 'key'}
    assert later <= {tuple(node) for node in shared['family_seeds']} and not later <= {tuple(node) for node in alone['family_seeds']}
    assert {source['component'] for source in shared['sources']} == {alone['component'], second_rows[0]['component']} != {shared['component']}
    assert all(node[0] != 'group' for node in shared['family_seeds'])
    job = files.read_yaml(result['job'])
    assert [set(entry['roles']) for entry in job['exports']] == [{'fit', 'model_select'} if entry['path'] == first else {'fit'} for entry in job['exports']]


def test_cohort_is_invariant_to_export_order_and_repeated_copies(tmp_path, cohort):
    from test_contract import republish
    _, _, export, selecting = cohort
    first = export('first', (0, 'a', 'fit'), (1, 'b', 'fit'), (2, 'c', 'fit'), *selecting)
    second = export('second', (1, 'd', 'fit'), (2, 'd', 'fit'), (3, 'e', 'fit'))
    copy = tmp_path / 'copy' / 'second.export.jsonl.manifest.json'
    from semantic_switch.artifacts import read_export
    republish(copy, files.read_published(second), read_export(second, native=True)[1])
    forward = prepared(tmp_path, cohort, [first, second], 'forward', fit_situations=3)[1]
    backward = prepared(tmp_path, cohort, [second, first, second], 'backward', fit_situations=3)[1]
    copied = prepared(tmp_path, cohort, [str(copy), first, second], 'copied', fit_situations=3)[1]
    assert forward['id'] == backward['id'] and forward['cohort'] == backward['cohort'] == copied['cohort'] and forward['fit'] == copied['fit']
    assert len(copied['sources']) == 2 and forward['populations']['fit']['situations'] == 3 and forward['populations']['fit']['families'] == 2


def test_transitive_bridges_join_and_equal_aliases_in_other_exports_stay_separate(tmp_path, cohort):
    _, _, export, selecting = cohort
    first = export('first', (0, 'g', 'fit'), (1, 'h', 'fit'), (2, 'k', 'fit'), *selecting)
    second = export('second', (0, 'x', 'fit'), (1, 'x', 'fit'))
    third = export('third', (1, 'y', 'fit'), (2, 'y', 'fit'))
    _, resolved, observations = prepared(tmp_path, cohort, [first, second, third], 'bridged')
    assert resolved['populations']['fit']['families'] == 1 and resolved['populations']['fit']['observations'] == 3
    alias = export('alias', (3, 'g', 'fit'), (4, 'h', 'fit'), (5, 'k', 'fit'), task='unrelated')
    _, resolved, observations = prepared(tmp_path, cohort, [first, alias], 'aliases')
    assert resolved['populations']['fit']['families'] == 6 and resolved['cohort']['families'] == 6 + 6


def test_an_unselected_bridge_export_cannot_split_a_family_across_roles(tmp_path, cohort):
    from semantic_trainer.jobs import resolve
    _, _, export, selecting = cohort
    first = export('first', (0, 'a', 'fit'), (1, 'b', 'fit'), (2, 'c', 'fit'), *selecting)
    bridge = export('bridge', (0, 'z', 'acceptance_fit'), (6, 'z', 'acceptance_fit'))
    with pytest.raises(ConfigError, match=r"crosses roles .*'fit': .*'model_select'.*connecting seeds"):
        prepared(tmp_path, cohort, [first, bridge], 'bridged')
    result, _, _ = prepared(tmp_path, cohort, [first], 'clean')
    job = files.read_yaml(result['job'])
    from semantic_switch.artifacts import read_export
    job['exports'].append({'path': bridge, 'export_id': read_export(bridge, native=True)[0]['export_id'], 'roles': {}})
    files.publish(result['job'], job)
    with pytest.raises(ConfigError, match='crosses roles'):
        resolve(result['job'])
    job['exports'][0]['export_id'] = 'f' * 64
    job['exports'].pop()
    files.publish(result['job'], job)
    with pytest.raises(ConfigError, match='changed .*prepare a new job'):
        resolve(result['job'])


def test_unused_bridges_are_kept_as_connectivity_only_sources(tmp_path, cohort):
    _, _, export, selecting = cohort
    first = export('first', (0, 'a', 'fit'), (1, 'b', 'fit'), (2, 'c', 'fit'), *selecting)
    bridge = export('bridge', (0, 'z', 'acceptance_fit'), (3, 'z', 'acceptance_fit'))
    result, resolved, observations = prepared(tmp_path, cohort, [first, bridge], 'kept')
    entries = {entry['path']: entry['roles'] for entry in files.read_yaml(result['job'])['exports']}
    assert entries[bridge] == {} and len(resolved['sources']) == 2
    fit = {row['component'] for row in observations if row['role'] == 'fit'}
    unused = [row for row in resolved['memberships'] if row['role'] == 'unused']
    assert len(unused) == 1 and unused[0]['component'] in fit and resolved['populations']['fit']['families'] == 3
