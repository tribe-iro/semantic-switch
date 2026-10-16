from __future__ import annotations

from argparse import Namespace
import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import digest
from semantic_switch.families import FamilyIndex
from semantic_switch.tools.data import Labels
from semantic_switch.tools.dataset import EvidenceContext, evaluation_plan
from bench import evidence, procedures

from semantic_testkit.labels import envelopes
from semantic_testkit.scoring import FakeScorer

CHAIN = [{'input': {'message': 'refund one'}, 'facts': {'refund': True}, 'groups': ['a'], 'role': 'check'},
         {'input': {'message': 'refund two'}, 'facts': {'refund': True}, 'groups': ['a'], 'pairs': ['p'], 'role': 'check'},
         {'input': {'message': 'hello three'}, 'facts': {'refund': False}, 'pairs': ['p'], 'role': 'check'},
         {'input': {'message': 'refund one', 'note': 'unread'}, 'facts': {'refund': True}, 'groups': ['b'], 'role': 'check'},
         {'input': {'message': 'hello other'}, 'facts': {'refund': False}, 'groups': ['c'], 'role': 'fit'}]


def switch(tmp_path, rows):
    path = tmp_path / 'chain.yaml'
    files.publish(path, {'name': 'chain', 'fields': {'message': {'type': 'text', 'required': True}, 'note': {'type': 'text'}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}}})
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes(rows)))
    return path


def materialized(path):
    with Runtime(FakeScorer()) as rt:
        return Labels(rt.load(path, draft=True))


def plan(**arms):
    return evaluation_plan(arms, scope=[['refund']], acceptance={'theta': .5}, metrics={'schema': 1}, margin=0)


def test_family_index_is_deterministic_with_aliases_and_typed_seeds():
    index = FamilyIndex()
    index.add(('group', 'b'), 'group:b')
    index.join(('key', 'k'), ('group', 'b'))
    index.join(('pair', 'p'), ('key', 'k'))
    assert index.name(('pair', 'p')) == 'group:b' and index.seeds(('key', 'k')) == (('group', 'b'), ('key', 'k'), ('pair', 'p'))
    index.join(('group', 'a'), ('pair', 'p'))
    assert index.find(('key', 'k')) == ('group', 'a') and index.name(('key', 'k')) == 'group:b'
    with pytest.raises(ConfigError, match='family seed'):
        index.add(('frame', 'x'))


def test_unread_fields_rendered_duplicates_and_link_chains_share_one_family(tmp_path):
    path = switch(tmp_path, CHAIN)
    labels = materialized(path)
    families = {labels.group(('key', tools.common.input_key(labels.sw.plan, row['input']))) for row in CHAIN[:4]}
    assert len(families) == 1 and labels.split(CHAIN[4]['input']) == 'tune'
    (family,) = families
    assert {('group', 'a'), ('group', 'b'), ('pair', 'p')} <= set(labels.seeds(family))
    assert {kind for kind, _ in labels.seeds(family)} == {'group', 'key', 'observation', 'pair', 'rendered', 'situation'}
    with Runtime(FakeScorer()) as rt:
        exported = tools.export(rt, path, split='all')
    _, rows = read_export(exported['path'], fitting=False)
    check = [row for row in rows if row['split'] == 'check']
    assert len(check) == 3 and {row['group'] for row in check} == {family} and len({row['component'] for row in check}) == 1
    assert all(row['aliases'] == [family] and row['component'] == labels.component(family) for row in check)
    context = EvidenceContext.from_manifest(path.with_suffix('.dataset.json'))
    assert {context.member(labels.sw.plan, row['input']) for row in CHAIN[:4]} == {family} and context.role(family) == 'check'
    with pytest.raises(ConfigError, match='has no family in evidence'):
        context.member(labels.sw.plan, {'message': 'never labelled'})


def test_bridged_historical_aliases_preserve_the_role_conflict(tmp_path):
    rows = [{'input': {'message': 'refund x'}, 'facts': {'refund': True}, 'groups': ['x'], 'role': 'fit'},
            {'input': {'message': 'refund y'}, 'facts': {'refund': True}, 'groups': ['y'], 'role': 'check'}]
    path = switch(tmp_path, rows)
    materialized(path)
    context = EvidenceContext.from_manifest(path.with_suffix('.dataset.json'))
    assert context.consult(plan(candidate='first'), {'group:y'})['state'] == 'fresh'
    unroled = [{k: v for k, v in row.items() if k != 'role'} for row in rows]
    files.publish(path.with_suffix('.test.jsonl'), ''.join(json.dumps(row) + '\n' for row in envelopes(
        [*unroled, {'input': {'message': 'refund xy'}, 'facts': {'refund': True}, 'groups': ['x', 'y']}])))
    bridged = materialized(path)
    record = bridged.manifest['families']['group:x']
    assert record['split'] == 'conflict' and record['aliases'] == ['group:x', 'group:y'] and record['conflicts'] == {'group:x': 'fit', 'group:y': 'check'}
    assert any('conflict' in warning for warning in bridged.warnings) and not bridged.examples('refund', 'check')
    files.publish(path.with_suffix('.test.jsonl'), ''.join(json.dumps(row) + '\n' for row in envelopes(rows)))
    assert materialized(path).manifest['families']['group:x']['conflicts'] == {'group:x': 'fit', 'group:y': 'check'}
    after = EvidenceContext.from_manifest(path.with_suffix('.dataset.json')).consult(plan(candidate='second'), {'group:y'})
    assert after['state'] == 'consulted' and after['consulted_families'] == ['group:x']


def test_one_frozen_plan_is_one_consultation_and_changes_are_overlap(tmp_path):
    path = switch(tmp_path, CHAIN)
    labels = materialized(path)
    context = EvidenceContext.from_manifest(path.with_suffix('.dataset.json'))
    families = {labels.group(('key', tools.common.input_key(labels.sw.plan, CHAIN[0]['input'])))}
    frozen = plan(base='b', specialist='s')
    assert context.consult(frozen, families)['state'] == 'fresh'
    resumed = context.consult(frozen, families)
    assert resumed['state'] == 'repeat' and resumed['independent'] and len(files.read_jsonl(context.audit)) == 1
    changed = [evaluation_plan({'base': 'b', 'specialist': 's'}, scope=[['refund']], acceptance={'theta': .6}, metrics={'schema': 1}, margin=0),
               plan(base='b', specialist='s', extra='undeclared'), plan(base='b', specialist='s2')]
    assert [context.consult(other, families)['state'] for other in changed] == ['consulted'] * 3
    assert context.consult(plan(candidate='by-group-name'), {'b'})['consulted_families'] == sorted(families)
    with pytest.raises(ConfigError, match='declares every comparison arm'):
        evaluation_plan({}, scope=[])
    with pytest.raises(ConfigError, match='frozen evaluation_plan'):
        context.consult({**frozen, 'margin': .1}, families)


def test_converted_exposure_withholds_independence_and_schema_1_is_refused(tmp_path):
    path = switch(tmp_path, CHAIN)
    labels = materialized(path)
    context = EvidenceContext.from_manifest(path.with_suffix('.dataset.json'))
    family = labels.group(('key', tools.common.input_key(labels.sw.plan, CHAIN[0]['input'])))
    old = {'arms': {'candidate': 'old'}, 'conversion': 'audit-schema-1'}
    converted = {'schema': 2, 'plan': {**old, 'id': digest(old)}, 'dataset_id': 'd', 'families': [family], 'aliases': [family], 'seeds': [],
                 'converted': {'schema': 1, 'candidate_id': 'old'}}
    files.append(context.audit, files.jsonl([converted]))
    found = context.consult(plan(candidate='new'), {family})
    assert found['state'] == 'consulted' and not found['independent'] and found['history'] == 'recorded'
    files.append(context.audit, files.jsonl([{'schema': 1, 'candidate_id': 'old', 'dataset_id': 'd', 'families': [family]}]))
    with pytest.raises(ConfigError, match='malformed audit exposure record'):
        context.consult(plan(candidate='newer'), {family})
    context.audit.unlink()
    for _ in range(2):
        missing = context.consult(plan(candidate='newer'), {family})
        assert missing['state'] == 'unknown_history' and not missing['independent'] and missing['consulted_families'] == [family]


def test_ordinary_test_records_exposure_before_evaluation(tmp_path, monkeypatch):
    from .test_artifact_pipeline import dataset, definition
    from semantic_switch.tools import evaluate
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        before = len(files.read_jsonl(path.with_suffix('.audit.jsonl')))
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt
        monkeypatch.setattr(evaluate, 'evaluate_facts', interrupted)
        rows = envelopes([{'input': {'message': f'refund new{i}'}, 'facts': {'refund': True}, 'group': f'new{i}'} for i in range(3)])
        files.publish(tmp_path / 'new.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
        manifest = path.with_suffix('.dataset.json').read_bytes()
        with pytest.raises(KeyboardInterrupt):
            tools.run_test(rt, path, tmp_path / 'new.jsonl')
    audit = files.read_jsonl(path.with_suffix('.audit.jsonl'))
    assert len(audit) == before + 1 and path.with_suffix('.dataset.json').read_bytes() == manifest
    sibling = path.with_suffix(f".dataset.{audit[-1]['dataset_id']}.json")
    assert audit[-1]['manifest'] == sibling.name and files.read_published(sibling)['id'] == audit[-1]['dataset_id']
    monkeypatch.undo()
    with Runtime(FakeScorer()) as rt:
        again = tools.run_test(rt, path, tmp_path / 'new.jsonl')
    assert again['dataset'] == str(sibling) and again['dataset_id'] == audit[-1]['dataset_id'] and again['audit_use']['state'] == 'repeat'


def test_stage_publishes_the_assignment_and_review_disclosure(tmp_path):
    published = procedures.pack_command(Namespace(seed=0, per_template=2, variants=3, reviews=None, output=tmp_path / 'pack', fit=32, model_select=2, check=16))
    staged = evidence.stage(tmp_path / 'pack', tmp_path / 'work')
    context = EvidenceContext.from_manifest(staged['evidence'])
    assert context.audit.exists() and not context.audit.read_text() and context.dataset_id == staged['dataset_id']
    assert context.review == published['review'] and context.review['status'].startswith('unreviewed') and context.review['frames']['remaining_frames']
    context.bind(staged['labels'])
    roles = {row['role'] for row in files.read_jsonl(staged['labels'])}
    assert roles == {'fit', 'model_select', 'check'} and set(staged['families']) == roles
    with Runtime(FakeScorer()) as rt:
        assert Labels(rt.load(staged['definition'], draft=True)).manifest['review'] == published['review']
    assert EvidenceContext.from_manifest(staged['evidence']).dataset_id == staged['dataset_id']
    reviews = evidence.read(tmp_path / 'pack')['reviews']
    assert reviews == []
