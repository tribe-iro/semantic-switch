from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.metrics import recall_counts, recall_floor
from semantic_trainer.jobs import (DEFAULTS, behavior_contract, behavior_rows, behavior_support, development_recall,
                                   fit_support, objective_weights, ranking, ranking_rule)
from semantic_trainer.qualification import spread, summarize, variants


def rows():
    return [{'id': str(n), 'task': task, 'fact': target, 'classes': ['true', 'false', 'unstated'], 'label': label,
             'component': family, 'situations': [f'{task}:{n}'], 'role': role, 'tags': tags,
             'presentations': [SimpleNamespace(length=2)]}
            for n, (task, target, label, family, role, tags) in enumerate([
                ('a', 'flag', 0, 'f1', 'fit', ['phase:calibration']),
                ('a', 'flag', 0, 'f1', 'fit', ['phase:calibration']),
                ('a', 'flag', 0, 'f2', 'fit', ['phase:paused']),
                ('a', 'flag', 1, 'f3', 'fit', ['phase:paused']),
                ('b', 'flag', 1, 'f4', 'fit', ['phase:paused']),
                ('a', 'flag', 0, 's1', 'model_select', ['phase:calibration']),
                ('a', 'flag', 1, 's2', 'model_select', ['phase:paused'])])]


def test_class_family_allocation_preserves_task_budgets_and_does_not_touch_selection():
    empirical, balanced = rows(), rows()
    tasks = {'a': {'weight': 3}, 'b': {'weight': 1}}
    objective_weights(empirical, tasks)
    report = objective_weights(balanced, tasks, 'class_family')
    assert sum(row['objective'] for row in balanced if row['role'] == 'fit') == pytest.approx(5)
    for task in tasks:
        before = sum(row['objective'] for row in empirical if row['role'] == 'fit' and row['task'] == task)
        after = sum(row['objective'] for row in balanced if row['role'] == 'fit' and row['task'] == task)
        assert before == pytest.approx(after)
    a = [row for row in balanced if row['task'] == 'a' and row['role'] == 'fit']
    assert sum(row['objective'] for row in a if row['label'] == 0) == pytest.approx(sum(row['objective'] for row in a if row['label'] == 1))
    assert sum(row['objective'] for row in a if row['component'] == 'f1') == pytest.approx(a[2]['objective'])
    assert [row['objective'] for row in empirical if row['role'] == 'model_select'] == [row['objective'] for row in balanced if row['role'] == 'model_select']
    assert report['fit_allocation']['mode'] == 'class_family'
    assert report['fit_allocation']['max_row_multiplier'] > 1


def test_absent_fit_answers_are_diagnostics_and_declared_slices_include_empty_cells():
    found = rows()
    recipe = {**DEFAULTS, 'behavior_slices': [{'name': 'calibration', 'tags': ['phase:calibration']}, {'name': 'absent', 'tags': ['never']}]}
    behavior_contract(recipe)
    behavior_rows(found, recipe)
    objective_weights(found, {'a': {'weight': 1}, 'b': {'weight': 1}})
    report = fit_support(found, 'specialization')
    assert 'unstated' in report['unsupported_classes']['a']['flag']
    support = behavior_support(found, recipe)
    assert support['fit']['a']['flag']['calibration']['true']['families'] == 1
    assert support['fit']['a']['flag']['calibration']['true']['observations'] == 2
    assert support['model_select']['a']['flag']['absent']['true']['observations'] == 0
    assert support['fit']['a']['flag']['absent']['true']['objective'] == 0


def test_development_floor_uses_scoped_gold_metadata_and_counts_absent_support():
    found = rows()
    required = [{'task': 'a', 'target': 'flag', 'class': 'true', 'slice': 'calibration', 'minimum': .8}]
    recipe = {**DEFAULTS, 'behavior_slices': [{'name': 'calibration', 'tags': ['phase:calibration']}], 'development_recall': required}
    behavior_contract(recipe)
    behavior_rows(found, recipe)
    selected = [row for row in found if row['role'] == 'model_select']
    results = {row['id']: {'correct': False, 'accepted': True} for row in selected}
    report = development_recall(selected, results, required)
    assert report['state'] == 'unmet' and report['requirements'][0]['references'] == 1
    results[selected[0]['id']]['correct'] = True
    assert development_recall(selected, results, required)['state'] == 'met'
    missing = [{**required[0], 'class': 'unstated'}]
    assert development_recall(selected, results, missing)['state'] == 'unassessed'
    ambiguous = {**recipe, 'development_recall': [{key: value for key, value in required[0].items() if key != 'task'}]}
    with pytest.raises(ConfigError, match='task scope'):
        behavior_rows(found, ambiguous)


def test_required_behavior_precedes_aggregate_retention_but_unmet_candidates_remain_ranked():
    def point(step, score, state):
        return {'step': step, 'correct': {'macro': {'observation': score}}, 'nll': 1., 'qualification': {'state': state}}
    high, good, unsupported = point(1, .99, 'unmet'), point(2, .91, 'met'), point(3, .95, 'unassessed')
    assert ranking([high, good, unsupported], 'observation') == [good, high, unsupported]
    assert ranking([high, unsupported], 'observation') == [high, unsupported]
    assert ranking_rule([high, good], 'observation')['order'][0] == 'qualified'


def test_shared_recall_reducer_preserves_operational_abstentions_and_families():
    counts = recall_counts([(('task', 'flag'), 'true', 'f1', True, False), (('task', 'flag'), 'true', 'f1', False, True)])
    cell = counts[('task', 'flag', 'true')]
    assert cell == {'references': 2, 'raw': 1, 'operational': 0, 'families': 1}
    assert recall_floor(cell['operational'], cell['references'], .1)['state'] == 'unmet'
    assert recall_floor(0, 0, .5)['state'] == 'unassessed'


def test_qualification_is_a_fixed_perturbation_design_and_keeps_incomplete_runs():
    declared = variants({'learning_rate': .0002})
    assert [(row['seed'], row['learning_rate']) for row in declared] == [(0, .0002), (1, .0002), (2, .0002), (0, .0001), (0, .0004)]
    assert summarize([{'state': 'completed'}, {'state': 'failed'}])['state'] == 'incomplete'
    report = spread([.8, .9, None])
    assert report['runs'] == 2 and report['worst'] == .8 and report['range'] == [.8, .9]
    assert report['median'] == pytest.approx(.85)


def test_prose_guards_preserve_structured_action_states(tmp_path, monkeypatch):
    from bench import procedures
    from tests.test_printer_retirement import decisions
    from semantic_switch import api
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    planned = procedures.scenarios(92, 1, 2)
    structured = procedures.frames(planned, conditions=('structured',))
    prose = procedures.frames(planned, conditions=('prose',))
    path = tmp_path / 'prose.yaml'
    files.publish(path, procedures.definition('prose'))
    before, _ = decisions(procedures.DEFINITION, structured, {frame.id: procedures.reference(frame) for frame in structured})
    after, asked = decisions(path, prose, {frame.id: procedures.view_gold(frame, procedures.reference(frame)) for frame in prose})
    def truth(states):
        return {name: 'false' if state in ('ineligible', 'rejected') else state for name, state in states.items()}
    assert list(map(truth, before.values())) == list(map(truth, after.values()))
    assert {'idle_prompt', 'paused_prompt', 'tt_method'} <= set(asked)
    assert set(procedures.definition('prose')['fields']) == {'asset', 'model', 'notes'}
    assert all('requires' not in card for card in procedures.definition('prose')['facts']['actions']['cards'])


def test_input_view_materializes_parent_evidence_without_generation_or_role_changes(tmp_path):
    from bench import evidence
    from semantic_switch.compiler import compile_spec
    definition = {'name': 'flag', 'fields': {'notes': {'type': 'text', 'required': True}}, 'reads': ['notes'],
                  'facts': {'flag': {'check': 'The report establishes the flag.', 'unstated': True}}}
    structured = evidence.Frame('s', 'family', 'scenario', 'structured', {'notes': 'enabled', 'trusted': True}, ())
    prose = evidence.Frame('p', 'family', 'scenario', 'prose', {'notes': 'enabled'}, (), pairs=('s',))
    gold = evidence.Gold('r1', (), primitives={'flag': evidence.Decision(True, ())})
    plan = compile_spec(definition)
    label = evidence.label(plan, structured, gold, 'fit', 'fixture', 'c' * 64, '2026-10-01T00:00:00Z', 'test')
    parent = tmp_path / 'parent'
    manifest = files.publish(parent / 'benchmark.json', {'schema': evidence.SCHEMA}, {
        'definition.yaml': ['name: flag\n'], 'labels.jsonl': evidence.lines([label]),
        'situations.jsonl': evidence.lines(evidence.situation(frame, 'fit') for frame in (structured, prose)),
        'annotations.jsonl': evidence.lines({'frame': frame.id, 'public_sha256': frame.public, 'assertions': []} for frame in (structured, prose)),
        'reference.jsonl': evidence.lines({'frame': frame.id, 'family': 'canonical-family', 'role': 'fit', 'condition': frame.condition,
                                           'binding': 'b' * 64, **gold.record()} for frame in (structured, prose))})
    before = (parent / 'benchmark.json').read_bytes()
    view = evidence.materialize(parent, tmp_path / 'view', condition='prose', definition=definition, producer='fixture', producer_digest='d' * 64)
    assert view['roles'] == {'fit': 1}
    assert (parent / 'benchmark.json').read_bytes() == before
    (row,) = files.read_jsonl(view['labels'])
    assert row['input'] == {'notes': 'enabled'} and row['groups'] == ['family'] and row['role'] == 'fit'
    assert row['label_sources']['flag'][0]['derivation'] == 'computed'
    assert not row['label_sources']['flag'][0].get('verifications')
    with pytest.raises(ConfigError, match='already published'):
        evidence.materialize(parent, tmp_path / 'view', condition='prose', definition=definition, producer='fixture', producer_digest='d' * 64)


def test_qualification_executes_existing_paths_and_resumes_only_incomplete_jobs(tmp_path, monkeypatch):
    from semantic_testkit.assets import training_job
    from semantic_trainer import lora, select
    from semantic_trainer.qualification import qualify
    source, _, _ = training_job(tmp_path / 'source', monkeypatch)
    calls, failing = [], {1}
    def train(job, run):
        recipe = files.read_yaml(job)['recipe']
        calls.append(recipe['seed'])
        if recipe['seed'] in failing:
            raise ConfigError('fixture interruption')
        files.publish(run / 'run.json', {'state': 'ready'})
    def selected(job, run, representation):
        point = {'name': 'step-1', 'correct': {'macro': {'observation': .9, 'complete_situation': .8}},
                 'per_class': {'flag': {'true': {'recall': .8}}}, 'precision': .9, 'coverage': .8, 'correct_accepted_rate': .72}
        files.publish(run / 'deployment' / 'receipt.json', {'selection': {'selected': 'step-1', 'candidates': [point]}})
    monkeypatch.setattr(lora, 'train', train)
    monkeypatch.setattr(select, 'select', selected)
    result = qualify(source, tmp_path / 'qualification', execute=True)
    assert len(result['runs']) == 5 and result['summary']['completed'] == 4 and result['summary']['state'] == 'incomplete'
    assert calls == [0, 1, 2, 0, 0]
    failing.clear()
    result = qualify(source, tmp_path / 'qualification', execute=True)
    assert result['summary']['completed'] == 5 and calls == [0, 1, 2, 0, 0, 1]
    assert result['summary']['metrics']['precision']['worst'] == .9
    assert result['runs'][0]['operational_actions']['state'] == 'unassessed'


def test_fit_diagnostic_rejects_slot_corruption_before_fitting(tmp_path, monkeypatch):
    from semantic_testkit.assets import training_job
    from semantic_trainer import qualification
    from semantic_trainer.jobs import resolve
    source, _, _ = training_job(tmp_path / 'source', monkeypatch)
    _, found = resolve(source)
    row = next(row for row in found if row['role'] == 'fit')
    monkeypatch.setattr(qualification, 'canonical_answer', lambda *args: -99)
    with pytest.raises(ConfigError, match='slot mismatches'):
        qualification.diagnostic(source, tmp_path / 'diagnostic', target=row['fact'], answer=row['classes'][row['label']], families=1)
    assert not (tmp_path / 'diagnostic' / 'job.yaml').exists()


def test_prose_guards_keep_unknown_and_conflicting_context_unresolved(tmp_path, monkeypatch):
    from bench import procedures as p
    from semantic_switch import api
    from tests.test_printer_retirement import decisions
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    structured, prose = [], []
    for methods in ((p.TT,), (p.DT,), (), (p.TT, p.DT)):
        for prompts in (('ready',), ('paused',), ('calibration_1',), ('calibration_2',), (), ('ready', 'calibration_1')):
            index = len(structured)
            events = [p.ev(0, 'configuration', 'P', 'print_method', value) for value in methods]
            events += [p.ev(0, 'display', 'P', 'prompt', value) for value in prompts]
            events += [p.ev(1, 'operator_report', 'P', 'media_type', p.TT), p.ev(1, 'verification', 'P', 'ribbon_loaded', True)]
            for condition, output in (('structured', structured), ('prose', prose)):
                output.append(p.render(str(index), p.State('P', 'ZT411', p.TT, True, p.TT, 'ready'), events,
                                       condition=condition, frame_id=f'{condition}-{index}', family=str(index)))
    path = tmp_path / 'prose.yaml'
    files.publish(path, p.definition('prose'))
    before, _ = decisions(p.DEFINITION, structured, {frame.id: p.reference(frame) for frame in structured})
    after, _ = decisions(path, prose, {frame.id: p.reference(frame) for frame in prose})
    def truth(states):
        return {name: 'false' if state in ('ineligible', 'rejected') else state for name, state in states.items()}
    for left, right, frame in zip(before.values(), after.values(), structured):
        if frame.input.get('prompt') and frame.input.get('print_method'):
            assert truth(left) == truth(right)
        else:
            assert not left
    for frame in prose:
        gold = p.reference(frame)
        if not any(assertion.property == 'prompt' for assertion in frame.assertions):
            assert gold.primitives['idle_prompt'].value == gold.primitives['paused_prompt'].value == 'unstated'
        if not any(assertion.property == 'print_method' for assertion in frame.assertions):
            assert gold.primitives['tt_method'].value == 'unstated'


def test_review_diagnosis_is_idempotent_metadata_and_never_a_label(tmp_path):
    from semantic_learn.review import diagnose
    review, evidence, output = tmp_path / 'test.review.json', tmp_path / 'analysis.json', tmp_path / 'diagnoses.jsonl'
    files.publish(review, {'kind': 'review-export', 'id': 'r', 'units': [{'id': 'u', 'stratum': 'audit'}]})
    files.publish(evidence, {'finding': 'verified context mismatch'})
    arguments = dict(reason='input_view', reviewer='reviewer', evidence=evidence, note='wrong input view', to=output)
    result = diagnose(review, 'u', **arguments)
    assert diagnose(review, 'u', **arguments) == result and len(files.read_jsonl(output)) == 1
    assert result['next'] == 'materialize' and 'facts' not in result and 'label_sources' not in result
    with pytest.raises(ConfigError, match='published review queue'):
        diagnose(review, 'other', **arguments)


def test_fit_diagnostic_uses_export_group_selectors_and_closed_roles(tmp_path, monkeypatch):
    from semantic_testkit.assets import training_job
    from semantic_trainer.qualification import diagnostic
    from semantic_trainer.jobs import resolve
    source, _, _ = training_job(tmp_path / 'source', monkeypatch)
    _, found = resolve(source)
    row = next(row for row in found if row['role'] == 'fit')
    result = diagnostic(source, tmp_path / 'diagnostic', target=row['fact'], answer=row['classes'][row['label']], families=1)
    resolved, subset = resolve(result['job'])
    assert resolved['populations']['fit']['families'] == 1
    assert resolved['populations']['model_select']['families'] == 1
    assert len(resolved['probe_ids']) == resolved['populations']['fit']['observations']
    assert {row['role'] for row in subset} == {'fit', 'model_select'}


def test_variant_job_relocation_preserves_relative_exports_and_fit_identity(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from semantic_testkit.assets import training_job
    from semantic_trainer.jobs import resolve
    from semantic_trainer.qualification import write_job
    source, job, _ = training_job(tmp_path / 'source', monkeypatch)
    original, _ = resolve(source)
    job['exports'] = [{**entry, 'path': os.path.relpath(entry['path'], source.parent)} for entry in job['exports']]
    files.publish(source, job)
    destination = tmp_path / 'elsewhere' / 'job.yaml'
    write_job(source, destination, {})
    moved, _ = resolve(destination)
    assert moved['fit'] == original['fit'] and moved['monitor'] == original['monitor']
    assert all(Path(entry['path']).is_absolute() for entry in files.read_yaml(destination)['exports'])


@pytest.mark.parametrize(('prompt', 'expected'), [('calibration_1', 'met'), ('calibration_2', 'unmet')])
def test_coverage_requests_select_authored_states_without_rewriting_flows(tmp_path, monkeypatch, prompt, expected):
    from bench import evidence, procedures
    request = tmp_path / 'request.json'
    files.publish(request, {'seed': 92, 'requests': [{'target': 'method_inconsistent', 'class': 'true',
                                                    'tags': [f'prompt:{prompt}'], 'families': 1}]})
    parent = {'reference': [], 'situations': [], 'manifest_sha256': 'a' * 64}
    pack = evidence.Pack('fixture', '1', {}, (), (), 'fixture', '2026-10-01T00:00:00Z', tmp_path)
    monkeypatch.setattr(evidence, 'read', lambda path: parent)
    monkeypatch.setattr(evidence, 'load_pack', lambda path: pack)
    monkeypatch.setattr(evidence, 'review', lambda pack, frames, records, reference, **settings: frames)
    def publish(directory, pack, frames, reference, **settings):
        for frame in frames:
            view = procedures.View(frame)
            for assertion in frame.assertions:
                if assertion.property == 'step' and assertion.status == 'current':
                    assert assertion.value == view.get('prompt')
        return settings['extra']
    monkeypatch.setattr(evidence, 'publish', publish)
    result = procedures.coverage_command(SimpleNamespace(parent=tmp_path, output=tmp_path / 'out', request=request,
                                                         condition='structured', per_template=1, variants=None, role='fit', exclude=[]))
    assert result['coverage'][0]['state'] == expected
