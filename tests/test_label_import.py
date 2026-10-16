from __future__ import annotations

import json

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import digest
from semantic_switch.evidence import ROLES, Actor, LabelSource, Verification, each_target, label_record, reviewed_content, role_of

from semantic_testkit.scoring import FakeScorer
from .test_classification import LABELS, saved

TEACHER = LabelSource('judged', 'openrouter:google/gemma-4-31b-it', digest({'model': 'gemma-4-31b-it'}), address=('teacher',)).record()
HUMAN = Actor('human')


def switch(tmp_path):
    return saved(tmp_path, {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                            'issues': {'select': 'Which issues does the customer raise?', 'labels': LABELS}})


def records(sw, n=24, source=TEACHER, **extra):
    return [label_record(sw.plan, {'message': f'case {i}'}, facts, sources=each_target(facts, [source]), groups=[f'situation-{i}'], pairs=[f'pair-{i // 2}'],
                         tags=['relayed'], **extra)
            for i in range(n) for facts in ({'refund': bool(i % 2), 'issues': {'refund': bool(i % 2), 'billing': False}},)]


def publish(path, rows):
    files.publish(path, ''.join(json.dumps(row) + '\n' for row in rows))
    return path


def test_label_file_exports_without_touching_corrections_or_tests(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        sw = runtime.load(path, draft=True)
        labels = publish(tmp_path / 'synthetic.jsonl', records(sw))
        exported = tools.export(runtime, path, split='all', native=True, labels=labels)
    _, rows = read_export(exported['path'], fitting=False, native=True)
    assert not path.with_suffix('.test.jsonl').exists() and not path.with_suffix('.examples.jsonl').exists()
    assert {row['label_derivation'] for row in rows} == {'judged'} and all(row['tags'] == ['relayed'] for row in rows)
    members = [row for row in rows if row.get('member')]
    assert {row['member'] for row in members} == {'refund', 'billing'} and len(members) == 48
    assert all(row['pairs'] for row in rows) and {row['fact'] for row in rows} == {'refund', 'issues'}


@pytest.mark.parametrize('change, error', [
    (lambda row: row.update(revs={**row['revs'], 'refund': 'stale'}), 'refund: meaning changed'),
    (lambda row: row['revs']['issues'].update(refund='stale'), 'issues: meaning changed'),
    (lambda row: row['label_sources'].update(refund=[{'derivation': 'teacher', 'producer': 'x'}]), 'label source derivation must be one of'),
    (lambda row: row['label_sources'].update(refund=[{'origin': 'synthetic', 'producer': 'x'}]), 'label source needs derivation and producer'),
    (lambda row: row['label_sources'].update(refund=[{'derivation': 'authored', 'producer': 'x'}]), 'an authored label records its actor'),
    (lambda row: row['label_sources'].update(refund=[{'derivation': 'computed', 'producer': 'x'}]), 'a computed label names its procedure revision'),
    (lambda row: row['label_sources'].pop('refund'), 'label_sources must mirror facts'),
    (lambda row: row.pop('schema'), 'not a schema-3 situation envelope'),
    (lambda row: row.update(rule='refund'), 'record rule answers'),
    (lambda row: row.update(extra=True), 'unsupported label fields'),
    (lambda row: row.update(role='train'), 'role must be one of fit, model_select, acceptance_fit, check'),
])
def test_label_file_import_rejects_stale_or_ambiguous_records(tmp_path, change, error):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        rows = records(runtime.load(path, draft=True))
        change(rows[3])
        with pytest.raises(ConfigError, match=f'synthetic.jsonl:4: .*{error}'):
            tools.export(runtime, path, labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_conflicting_duplicate_labels_fail(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        rows = records(runtime.load(path, draft=True))
        rows.append({**rows[0], 'facts': {'refund': not rows[0]['facts']['refund']}, 'revs': {'refund': rows[0]['revs']['refund']},
                     'label_sources': {'refund': [TEACHER]}})
        with pytest.raises(ConfigError, match='conflicting labels'):
            tools.export(runtime, path, labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_review_confirms_replaces_and_skips_with_provenance(tmp_path):
    path = switch(tmp_path)
    answers = iter(['', 'y', 's', '', '', 's'])
    with Runtime(FakeScorer()) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', records(runtime.load(path, draft=True), n=2))
        written = tools.label(runtime, path, str(supplied), to=str(tmp_path / 'reviewed.jsonl'), ask=lambda prompt: next(answers), show=lambda text: None,
                              reviewer='ana', actor=HUMAN)
    reviewed = files.read_jsonl(tmp_path / 'reviewed.jsonl')
    assert written == 2 and not path.with_suffix('.test.jsonl').exists()
    assert reviewed[0]['facts'] == {'refund': False, 'issues': {'refund': True}}
    assert reviewed[1]['facts'] == {'refund': True, 'issues': {'refund': True}}
    confirmed, replaced = reviewed[0]['label_sources']['refund'], reviewed[0]['label_sources']['issues']['refund']
    meaning = runtime.load(path, draft=True).plan.fact('refund').meaning_rev
    assert [s['derivation'] for s in confirmed] == ['judged'] and confirmed[0]['producer'] == TEACHER['producer']
    assert confirmed[0]['verifications'] == [{'kind': 'label', 'verifier': 'ana', 'actor': {'kind': 'human'}, 'method': 'review', 'criterion': meaning,
                                              'content': reviewed_content('label', {'message': 'case 0'}, ('refund',), False), 'outcome': 'confirm'}]
    assert [s['derivation'] for s in replaced] == ['judged', 'authored']
    assert replaced[1] == {'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}, 'replaces': False}
    assert replaced[0]['verifications'][0]['outcome'] == 'replace'
    skips = files.read_jsonl(tmp_path / 'reviewed.reviews.jsonl')
    assert [(row['address'], row['verification']['outcome']) for row in skips] == [(['issues', 'billing'], 'skip'), (['issues', 'billing'], 'skip')]
    assert set(reviewed[0]['revs']) == {'refund', 'issues'} and set(reviewed[0]['revs']['issues']) == {'refund'}
    with Runtime(FakeScorer()) as runtime:
        exported = tools.export(runtime, path, split='all', labels=tmp_path / 'reviewed.jsonl')
    _, rows = read_export(exported['path'], fitting=False)
    derivations = {(row['bound_input']['message'], row['fact'], row.get('member')): row['label_derivation'] for row in rows}
    assert derivations == {('case 0', 'refund', None): 'judged', ('case 0', 'issues', 'refund'): 'authored',
                           ('case 1', 'refund', None): 'judged', ('case 1', 'issues', 'refund'): 'judged'}


def test_heterogeneous_derivations_merge_sources_and_keep_annotations_out_of_model_input(tmp_path):
    path = switch(tmp_path)
    rule = LabelSource('computed', 'bench.reference', digest('reference'), artifact=digest('frame'), address=('frame-1', 'refund')).record()
    with Runtime(FakeScorer()) as runtime:
        plan = runtime.load(path, draft=True).plan
        facts = {'refund': True, 'issues': {'refund': True}}
        mixed = label_record(plan, {'message': 'refund please'}, facts, sources={'refund': [rule], 'issues': {'refund': [TEACHER]}}, groups=['a'])
        again = label_record(plan, {'message': 'refund please'}, {'refund': True}, sources={'refund': [LabelSource('unknown', 'unknown').record()]}, groups=['a'])
        evidence = LabelSource('computed', 'bench.reference', '1').verified(Verification('evidence', 'ana', HUMAN, 'source-check', 'reference@1',
                                                                                          reviewed_content('evidence', {'message': 'refund please'}, ()), 'confirm')).record()
        reviewed = label_record(plan, {'message': 'refund please'}, {'refund': True}, sources={'refund': [evidence]}, groups=['a'])
        exported = tools.export(runtime, path, split='all', native=True, labels=publish(tmp_path / 'labels.jsonl', [mixed, again, reviewed]))
        with pytest.raises(ConfigError, match='evidence verification by ana covers other content'):
            label_record(plan, {'message': 'refund please now'}, {'refund': True}, sources={'refund': [evidence]})
    _, rows = read_export(exported['path'], fitting=False, native=True)
    refund = next(row for row in rows if row['fact'] == 'refund')
    member = next(row for row in rows if row.get('member') == 'refund')
    assert refund['label_derivation'] == 'mixed' and sorted(s['derivation'] for s in refund['label_sources']) == ['computed', 'computed', 'unknown']
    assert len({json.dumps(s, sort_keys=True) for s in refund['label_sources']}) == 3
    assert member['label_derivation'] == 'judged' and member['label_sources'] == [TEACHER]
    assert len(refund['situations']) == 1 == len(exported_situations(exported)) and refund['situations'] == member['situations']
    for row in rows:
        visible = json.dumps([row['bound_input'], row['text'], [[entry.answers.claim, entry.answers.options, entry.answers.text] for entry in row['presentations']]])
        assert all(hidden not in visible for hidden in ('bench.reference', 'gemma-4-31b', 'teacher', 'frame-1'))


def exported_situations(exported):
    return read_export(exported['path'], fitting=False)[0]['situations']


def test_review_refuses_its_own_inputs_as_destination(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', records(runtime.load(path, draft=True), n=1))
        with pytest.raises(ConfigError, match='must differ'):
            tools.label(runtime, path, str(supplied), to=str(supplied), actor=HUMAN)
        with pytest.raises(ConfigError, match='examples, test or a label file'):
            tools.label(runtime, path, str(supplied), to='elsewhere', actor=HUMAN)
        with pytest.raises(ConfigError, match='records who decides'):
            tools.label(runtime, path, str(supplied), to=str(tmp_path / 'reviewed.jsonl'), actor=Actor('unknown'))


def roled(sw, n=24, roled=8):
    return [{**row, 'role': list(ROLES)[i // 2 % len(ROLES)]} if i // 2 < roled else row for i, row in enumerate(records(sw, n))]


def assignment(row):
    return role_of(row['split'], row['fold'])


def test_explicit_roles_assign_families_and_unroled_families_keep_the_hash(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        labels = publish(tmp_path / 'synthetic.jsonl', roled(runtime.load(path, draft=True)))
        rows = read_export(tools.export(runtime, path, split='all', labels=labels)['path'], fitting=False)[1]
        tuned = read_export(tools.export(runtime, path, to=tmp_path / 'tune.jsonl', labels=labels)['path'])[1]
    given = {f"group:situation-{i}": role for i, role in enumerate(r.get('role') for r in files.read_jsonl(labels)) if i % 2 == 0}
    for row in rows:
        if given[row['group']] is not None:
            assert assignment(row) == given[row['group']]
        else:
            assert row['split'] == ('tune' if int(digest(['split', row['group']])[:8], 16) % 2 == 0 else 'check')
            assert row['fold'] == int(digest(['fold', row['group']])[:8], 16) % 5
    assert {assignment(row) for row in rows if given[row['group']]} == set(ROLES)
    assert tuned and all(given[row['group']] != 'check' and row['split'] == 'tune' for row in tuned)


def test_conflicting_roles_within_a_family_fail_naming_it(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        rows = records(runtime.load(path, draft=True))
        rows[0], rows[1] = {**rows[0], 'role': 'fit'}, {**rows[1], 'role': 'check'}
        with pytest.raises(ConfigError, match='family group:situation-0 has label records with roles check, fit; give every record'):
            tools.export(runtime, path, split='all', labels=publish(tmp_path / 'synthetic.jsonl', rows))


def test_manifest_keeps_explicit_roles_and_refuses_a_changed_role(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        rows = roled(runtime.load(path, draft=True))
        test = publish(path.with_suffix('.test.jsonl'), rows)
        first, again = (tools.export(runtime, path, split='all', to=tmp_path / f'{n}.jsonl') for n in ('first', 'again'))
        manifest = files.read_published(path.with_suffix('.dataset.json'))
        assert first['dataset_id'] == again['dataset_id'] == manifest['id']
        assert [assignment(manifest['families'][f'group:situation-{i}']) for i in range(0, 16, 2)] == [list(ROLES)[i % len(ROLES)] for i in range(8)]
        publish(test, [{**row, 'role': 'check'} if row['groups'][0] in ('situation-0', 'situation-1') else row for row in rows])
        with pytest.raises(ConfigError, match='family group:situation-0 was materialized as fit in .*dataset.json but its label records now assign role check; connected evidence keeps its first role'):
            tools.export(runtime, path, split='all', to=tmp_path / 'moved.jsonl')
        path.with_suffix('.dataset.json').unlink()
        rows = read_export(tools.export(runtime, path, split='all', to=tmp_path / 'rebuilt.jsonl')['path'], fitting=False)[1]
    assert {assignment(row) for row in rows if row['group'] == 'group:situation-0'} == {'check'}


def test_review_keeps_the_role_in_label_files_and_refuses_to_drop_it(tmp_path):
    path = switch(tmp_path)
    with Runtime(FakeScorer()) as runtime:
        supplied = publish(tmp_path / 'synthetic.jsonl', [{**row, 'role': 'check'} for row in records(runtime.load(path, draft=True), n=1)])
        tools.label(runtime, path, str(supplied), to=str(tmp_path / 'reviewed.jsonl'), ask=lambda prompt: '', show=lambda text: None, actor=HUMAN)
        assert files.read_jsonl(tmp_path / 'reviewed.jsonl')[0]['role'] == 'check'
        with pytest.raises(ConfigError, match="role 'check' belongs to label files"):
            tools.label(runtime, path, str(supplied), ask=lambda prompt: '', show=lambda text: None, actor=HUMAN)


def test_judged_label_source_carries_its_training_use_declaration_identity():
    from semantic_switch.permissions import TrainingUse
    use = TrainingUse('allowed', 'google/gemma-4-31b-it', 'Example', 'https://example.invalid/terms', '2026-10-01', 'teacher-labels')
    source = LabelSource('judged', 'google/gemma-4-31b-it@Example', training_use=use)
    record = source.record()
    assert record == {'derivation': 'judged', 'producer': 'google/gemma-4-31b-it@Example', 'training_use': use.record()}
    assert LabelSource.parse(record) == source and LabelSource.parse(json.loads(json.dumps(record))).training_use == use
    assert source.digest != LabelSource('judged', 'google/gemma-4-31b-it@Example').digest
    assert source.digest != LabelSource('judged', 'google/gemma-4-31b-it@Example', training_use=TrainingUse.parse({**use.record(), 'revision': '2026-11-01'})).digest
    verification = Verification('label', 'ana', HUMAN, 'review', 'meaning-1', 'a' * 64, 'confirm')
    assert LabelSource.parse(source.verified(verification).record()).training_use == use
    with pytest.raises(ConfigError, match='only a judged label source carries a training use declaration'):
        LabelSource('authored', 'ana', actor=HUMAN, training_use=use)
    with pytest.raises(ConfigError, match='only a judged label source carries a training use declaration'):
        LabelSource('judged', 'teacher', training_use=use.record())
    with pytest.raises(ConfigError, match='needs exactly'):
        LabelSource.parse({**record, 'training_use': {'permission': 'allowed'}})
    with pytest.raises(ConfigError, match='cannot be declared allowed'):
        LabelSource.parse({**record, 'training_use': {**use.record(), 'model': 'typesafe/jev-1.13'}})


AGENT = Actor('agent', 'claude-opus-5-5', 'session-1')


@pytest.mark.parametrize('source, error', [
    (lambda: LabelSource('authored', 'ana'), 'an authored label records its actor'),
    (lambda: LabelSource('computed', 'bench.reference', actor=AGENT), 'a computed label names its procedure revision'),
    (lambda: LabelSource('computed', 'bench.reference', '1', actor=AGENT), 'only an authored label carries one'),
    (lambda: LabelSource('judged', 'teacher', replaces=True), 'only an authored label decision replaces a value'),
    (lambda: Actor('agent', 'claude-opus-5-5'), 'an agent actor records its model and session'),
    (lambda: Actor('human', 'claude-opus-5-5', 'session-1'), 'a human or unknown actor records neither'),
    (lambda: Actor('robot'), 'actor kind must be one of'),
    (lambda: Verification('label', 'ana', Actor('unknown'), 'review', 'meaning-1', 'a' * 64, 'confirm'), 'verification records its actor'),
    (lambda: Verification('label', 'ana', HUMAN, 'vibes', 'meaning-1', 'a' * 64, 'confirm'), 'verification method must be one of'),
    (lambda: Verification('label', 'ana', HUMAN, 'review', ' ', 'a' * 64, 'confirm'), 'names the criterion'),
    (lambda: LabelSource.parse({'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}, 'reviews': []}), 'supported fields'),
])
def test_label_provenance_dimensions_reject_incomplete_records(source, error):
    with pytest.raises(ConfigError, match=error):
        source()


def test_label_provenance_records_round_trip_every_dimension():
    from semantic_switch.permissions import TrainingUse
    use = TrainingUse('allowed', 'google/gemma-4-31b-it', None, 'https://example.invalid/terms', '2026-10-01', 'teacher-labels')
    recompute = Verification('label', 'claude-code', AGENT, 'procedure-recompute', 'policy@1', 'b' * 64, 'confirm')
    history = (LabelSource('computed', 'bench.reference', 'c' * 64, artifact='d' * 64, address=('frame', 'refund'), verifications=(recompute,)),
               LabelSource('judged', 'google/gemma-4-31b-it', 'e' * 64, training_use=use),
               LabelSource('authored', 'ana', actor=HUMAN, replaces=False),
               LabelSource('authored', 'claude-code', actor=AGENT))
    records = json.loads(json.dumps([source.record() for source in history]))
    assert tuple(map(LabelSource.parse, records)) == history
    assert records[0]['verifications'][0] == {'kind': 'label', 'verifier': 'claude-code', 'actor': {'kind': 'agent', 'model': 'claude-opus-5-5', 'session': 'session-1'},
                                              'method': 'procedure-recompute', 'criterion': 'policy@1', 'content': 'b' * 64, 'outcome': 'confirm'}
    assert records[2] == {'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}, 'replaces': False}


def test_training_eligibility_is_expressed_over_derivation_permission_and_restriction_not_verification():
    from semantic_switch.permissions import TrainingUse, admit
    allowed = TrainingUse('allowed', 'google/gemma-4-31b-it', 'Example', 'https://example.invalid/terms', '2026-10-01', 'teacher-labels')
    confirm = Verification('label', 'ana', HUMAN, 'review', 'meaning-1', 'a' * 64, 'confirm')
    recompute = Verification('label', 'claude-code', AGENT, 'procedure-recompute', 'policy@1', 'a' * 64, 'confirm')
    admitted = [(LabelSource('computed', 'reference', '1'),), (LabelSource('authored', 'ana', actor=HUMAN),), (LabelSource('authored', 'agent', actor=AGENT),),
                (LabelSource('unknown', 'S1MB train'),), (LabelSource('judged', allowed.route, training_use=allowed),),
                (LabelSource('judged', allowed.route, training_use=allowed), LabelSource('authored', 'ana', actor=HUMAN, replaces=False))]
    for history in admitted:
        admit([('target', history)])
    refused = [((LabelSource('judged', 'google/gemma-4-31b-it@Example').verified(confirm).verified(recompute),), 'declares no training use'),
               ((LabelSource('judged', 'google/gemma-4-31b-it@Other', training_use=allowed),), 'covers only the producer route it names'),
               ((LabelSource('judged', 'openrouter:typesafe/jev-1.13').verified(confirm),), 'TypeSafe agreement'),
               ((LabelSource('judged', 'openrouter:typesafe/jev-1.13'), LabelSource('authored', 'ana', actor=HUMAN, replaces=True).verified(recompute)), 'TypeSafe agreement'),
               ((LabelSource('computed', 'typesafe/jev-1.13', '1'),), 'TypeSafe agreement')]
    for history, error in refused:
        with pytest.raises(ConfigError, match=error):
            admit([('target', history)])


def test_per_target_assessment_accepts_only_current_confirmations_for_the_addressed_content_and_criterion():
    from semantic_switch.evidence import Criteria, assess, target_assessments
    data, path = {'message': 'refund please'}, ('refund',)
    content = lambda value: reviewed_content('label', data, path, value)
    confirm = lambda value, criterion='meaning-1', outcome='confirm': Verification('label', 'claude-code', AGENT, 'procedure-recompute', criterion, content(value), outcome)
    authored = LabelSource('authored', 'ana', actor=HUMAN)
    judged = LabelSource('judged', 'teacher')
    computed = LabelSource('computed', 'reference', 'r1')
    bind = dict(data=data, path=path, value=True, criteria=Criteria(frozenset({'meaning-1'})))
    cases = [((authored,), 'authored', 0, 0), ((judged,), 'judged', 0, 0), ((computed,), 'computed', 0, 0),
             ((authored.verified(confirm(True)),), 'verified', 0, 1),
             ((authored.verified(confirm(True, 'meaning-0')),), 'authored', 1, 0),
             ((authored.verified(confirm(False)),), 'authored', 1, 0),
             ((judged.verified(confirm(True, outcome='skip')),), 'disputed', 0, 0),
             ((judged.verified(confirm(True)).verified(confirm(True, outcome='skip')),), 'conflicting', 0, 1),
             ((judged.verified(confirm(False)), LabelSource('authored', 'ana', actor=HUMAN, replaces=False)), 'authored', 1, 0),
             ((computed, LabelSource('judged', 'teacher')), 'mixed', 0, 0)]
    for history, state, stale, current in cases:
        found = assess(history, **bind)
        assert (found.state, len(found.stale), len(found.current)) == (state, stale, current), history
        assert found.established == (state in ('verified', 'computed'))
    assert assess((computed,), **bind).procedures == (('reference', 'r1'),)
    record = {'input': data, 'facts': {'refund': True, 'issues': {'billing': False}},
              'label_sources': {'refund': [authored.verified(confirm(True)).record()], 'issues': {'billing': [judged.record()]}}}
    current = {'refund': Criteria(frozenset({'meaning-1'})), 'issues/billing': Criteria(frozenset({'meaning-1'}))}
    assert {address: a.state for address, a in target_assessments(record, current).items()} == {'refund': 'verified', 'issues/billing': 'judged'}
    assert target_assessments(record, {**current, 'refund': Criteria(frozenset({'meaning-2'}))})['refund'].state == 'authored'
    assert {address: a.state for address, a in target_assessments(record, {'issues/billing': current['issues/billing']}).items()} == \
        {'refund': 'unassessed', 'issues/billing': 'judged'}
