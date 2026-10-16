from pathlib import Path

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.tools.specialize import decode_request
from semantic_trainer import controls, jobs


def observation(identifier, label, role='fit', text='text'):
    return {'id': identifier, 'task': 'task', 'fact': 'decision', 'classes': ['true', 'false'], 'label': label,
            'component': identifier, 'role': role, 'text': text, 'tags': ['contrast'], 'pairs': ['pair']}


def test_control_uses_fit_only_and_declared_tie_order():
    rows = [observation('a', 1), observation('b', 0), observation('c', 1, 'model_select'),
            observation('d', 1, 'acceptance_fit'), observation('e', 1, 'check')]
    result, unavailable = controls.controls(rows)
    assert len(result) == 1 and result[0]['answer'] == 'true' and result[0]['correct'] is False
    assert result[0]['acceptance'] == 'unassessed' and unavailable == []


def test_lexical_control_reports_missing_classes_without_predicting():
    rows = [observation('a', 0), observation('b', 1, 'model_select')]
    result, unavailable = controls.controls(rows, lexical=True)
    lexical = next(row for row in result if row['candidate'] == 'lexical')
    assert lexical['answer'] is None and lexical['outcome'] == 'unavailable'
    assert unavailable[0]['reason'] == 'fewer than two fit classes'


@pytest.mark.parametrize('field,change', [('correct', {'correct': {'macro': {'observation': .9}}}),
                                        ('regressions', {'regressions': []}), ('nll', {'nll': .2})])
def test_reason_uses_actual_selector_key_despite_more_accepted_errors(field, change):
    base = {'name': 'base', 'base': True, 'step': 0, 'correct': {'macro': {'observation': .8}}, 'regressions': ['x'], 'nll': .3, 'wrong_accepted': 1}
    student = {**base, 'name': 'student', 'base': False, 'step': 4, 'wrong_accepted': 3, **change}
    result = jobs.selection_reason([base, student], 'observation', 'student')
    assert jobs.ranking([base, student], 'observation')[0]['name'] == 'student'
    assert result['first_differing_key'] == field and result['accepted_errors_determine_ranking'] is False


def test_diagnostic_preserves_rank_scale_and_closes_explicit_fit_families(tmp_path, monkeypatch):
    from semantic_testkit.assets import pipeline_base
    from semantic_testkit.labels import label_file, action_export
    from semantic_testkit.scoring import FakeScorer
    from semantic_switch import Runtime
    from semantic_trainer.qualification import diagnostic
    manifest = pipeline_base(tmp_path, monkeypatch)
    labels = label_file(tmp_path, n=40)
    with Runtime(FakeScorer()) as runtime:
        exported = action_export(runtime, tmp_path / 'task', labels, 'test:diagnosis')
    prepared = jobs.prepare(tmp_path / 'task/action.yaml', [exported['path']], tmp_path / 'job', assets_manifest=manifest,
                            recipe={'rank': 16, 'alpha': 32, 'eval_every_examples': 8})
    _, rows = jobs.resolve(prepared['job'])
    family = next(row['component'] for row in rows if row['role'] == 'fit')
    report = diagnostic(prepared['job'], tmp_path / 'diagnostic', family_ids=[family], rank=64)
    assert (report['rank'], report['alpha'], report['effective_scale']) == (64, 128, 2.)
    assert {row['family'] for row in report['cohort'] if row['role'] == 'fit'} == {family}
    assert report['assessment'] == 'unassessed' and report['baseline_errors'] is None
    with pytest.raises(ConfigError, match='distinct'):
        diagnostic(prepared['job'], tmp_path / 'duplicate', family_ids=[family, family])
    held = next(row['component'] for row in rows if row['role'] == 'model_select')
    with pytest.raises(ConfigError, match='not fit families'):
        diagnostic(prepared['job'], tmp_path / 'held', family_ids=[held])


def test_public_freeze_reports_missing_support_and_inputs_without_model_imports(tmp_path):
    import json
    import subprocess
    import sys
    from semantic_switch import Runtime
    from semantic_switch.training import Actor, LabelSource, Task
    from semantic_switch.tools.specialize import request
    definition = tmp_path / 'operations.yaml'
    files.publish(definition, {'name': 'operations', 'fields': {'message': {'type': 'text'}, 'mode': {'type': 'enum', 'values': ['live', 'offline']}},
                               'reads': ['message', 'mode'], 'facts': {'allowed': {'check': 'The operation is authorized.'}}})
    with Runtime() as runtime:
        task = Task(runtime.load(definition, draft=True))
        source = LabelSource('authored', 'operator', actor=Actor('human')).record()
        labels = [task.label({'message': message}, {'allowed': answer}, sources={'allowed': [source]}, groups=[role], role=role)
                  for message, answer, role in [('authorized', True, 'fit'), ('prohibited', False, 'model_select'), ('held', False, 'check')]]
    path = tmp_path / 'labels.jsonl'
    files.publish(path, files.jsonl(labels))
    work = tmp_path / 'work'
    files.write_new(work / 'request.json', request(definition, path, work, assets='a' * 64, devices={'fit': 'cpu', 'native': 'cpu'}))
    script = "from semantic_trainer.pipeline import Controller; import json,sys; r=Controller(sys.argv[1],until='freeze',emit=lambda **k:None).run(); print(json.dumps({'result':r,'loaded':[m for m in ('torch','llama_cpp') if m in sys.modules]}))"
    completed = subprocess.run([sys.executable, '-c', script, str(work / 'request.json')], capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    support = result['result']['training_support']
    assert result['loaded'] == [] and support['input_findings'] == 3
    assert list(support['unsupported_classes'].values()) == [{'allowed': ['false']}]
    evidence = files.read_json(support['path'])
    assert evidence['roles']['check']['families'] == 1
    assert evidence['input_contract']['findings'][0]['missing'] == {'allowed': ['mode']}


def test_explicit_missing_development_configuration_cannot_silently_skip_evaluation(tmp_path):
    from semantic_trainer.cli import development_settings
    with pytest.raises(ConfigError, match='configuration needs'):
        development_settings(tmp_path / 'missing.json')
    assert development_settings(None) is None
