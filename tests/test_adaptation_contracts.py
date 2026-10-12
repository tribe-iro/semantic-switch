from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_switch import ConfigError, Runtime, api, files, tools
from semantic_switch.artifacts import RECEIPT_SCHEMA, assess_lineage, artifact_digest, load_receipt, read_export, validate_native_export
from semantic_switch.tools.data import Labels
from semantic_switch.tools.lock import build_lock
from bench.continuations import joint_log_probability
from .conftest import FakeScorer
from .test_artifact_pipeline import definition, dataset
from .conftest import envelopes


def optional_trainer(name):
    return pytest.importorskip('semantic_trainer.' + name)


def membership(uses=('adaptation', 'selection')):
    return [{'id': 'a' * 64, 'fact': 'other', 'projection': 'same', 'group': 'g', 'family_seeds': [['group', 'g']], 'uses': list(uses)}]


def test_unknown_adapter_provenance_is_unknown():
    labels = SimpleNamespace(parent={}, group=lambda node: node[1])
    assert assess_lineage(None, labels, []).state == 'unknown'


def test_verified_empty_membership_preserves_disjoint_scope():
    assert assess_lineage((), SimpleNamespace(parent={}), []).state == 'declared_disjoint'


def test_adapter_lineage_applies_across_facts():
    labels = SimpleNamespace(families={('group', 'g')}, group=lambda node: 'resolved')
    rows = [SimpleNamespace(id='one', group='resolved', projection='new'), SimpleNamespace(id='two', group='other', projection='same')]
    assert assess_lineage(membership(), labels, rows).overlap == ('one', 'two')


@pytest.mark.parametrize('use', ['adaptation', 'selection'])
def test_adapter_fit_families_excluded_from_threshold_fit(tmp_path, monkeypatch, use):
    path = definition(tmp_path)
    dataset(path, 60)
    import semantic_switch.tools.lock as module
    from importlib import import_module
    module = import_module('semantic_switch.tools.lock')
    with Runtime({'judge': FakeScorer()}) as runtime:
        labels = Labels(runtime.load(path, draft=True))
        tune = labels.examples('refund', 'tune')
        row = {'id': tune[0].id, 'fact': 'different_fact', 'projection': tune[0].projection,
               'group': tune[0].group, 'family_seeds': labels.seeds(tune[0].group), 'uses': [use]}
        monkeypatch.setattr(module, 'specialization', lambda rt: (None, [row]))
        observed = []
        original = module.judged
        def score(sw, snap, f, examples):
            observed.append([case.id for case in examples])
            return original(sw, snap, f, examples)
        monkeypatch.setattr(module, 'judged', score)
        data, _, _ = build_lock(runtime, path)
        assert tune[0].id not in observed[0]
        assert tune[0].projection in data['facts']['refund']['membership']['judge']
        assert tune[0].projection not in data['facts']['refund']['membership']['tuned']
        assert data['lineage']['judge-development:refund']['excluded'] == 1


def test_adapter_calibration_empty_is_insufficient(tmp_path, monkeypatch):
    from importlib import import_module
    module = import_module('semantic_switch.tools.lock')
    path = definition(tmp_path)
    dataset(path, 100)
    with Runtime({'judge': FakeScorer()}) as runtime:
        labels = Labels(runtime.load(path, draft=True))
        monkeypatch.setattr(module, 'specialization', lambda rt: (None, [{'id': row.id, 'fact': 'any', 'projection': row.projection, 'group': row.group,
                                                                          'family_seeds': labels.seeds(row.group), 'uses': ['adaptation']} for row in labels.examples('refund', 'tune')]))
        data, _, _ = build_lock(runtime, path)
        assert all(side['state'] == 'insufficient' and 'uncalibrated' in side['fitting_evidence'] for side in data['facts']['refund']['policies']['judge'].values())


def native_row():
    entry = {'stage': 'full', 'variant': 'canonical', 'slot_map': [0, 1], 'label': 0, 'global_label': 0,
             'prefix': [1], 'input': [2], 'question': [3], 'answer_slots': [[4, 5], [6]], 'answer_text': [['A', ' A'], ['B']]}
    row = {'schema': 5, 'id': 'a' * 64, 'kind': 'pick', 'classes': ['x', 'y'], 'label': 0,
           'native': {'presentations': [entry], 'tokenizer': {'vocab_size': 10, 'vocab_sha256': 'b' * 64, 'probes': []},
                      'framing': {}, 'presentation': {}, 'authored_kind': 'pick'}}
    entry['id'] = artifact_digest([row['id'], entry])
    return row


@pytest.mark.parametrize('slots', [[], [[4], []], [[4], [4]], [[True], [6]], [[10], [6]], [[4, 4], [6]]])
def test_training_export_validates_answer_slots(slots):
    row = native_row()
    row['native']['presentations'][0]['answer_slots'] = slots
    with pytest.raises(ValueError):
        validate_native_export(row)


def test_lora_rejects_legacy_incomplete_export(tmp_path):
    export = tmp_path / 'legacy.export.jsonl'
    files.publish(Path(str(export) + '.manifest.json'), {'schema': 5, 'split': 'tune', 'rows': [{'id': 'a' * 64}]},
                  {export.name: [json.dumps(native_row()) + '\n']})
    with pytest.raises(ConfigError, match='re-export'):
        read_export(Path(str(export) + '.manifest.json'), native=True)


def test_multitoken_readout_uses_joint_continuation_probability():
    rows = [np.log([.6, .4]), np.log([.1, .9])]
    assert joint_log_probability(rows, [0, 1]) == pytest.approx(np.log(.6 * .9))
    assert joint_log_probability(rows, [0, 1]) != pytest.approx(np.log(.6 + .9))


def test_trainer_help_without_fitting_extras():
    optional_trainer('cli')
    code = "import sys; from semantic_trainer.cli import main; sys.argv=['semantic-trainer','--help']; main()"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0 and 'lora' in result.stdout


def test_lora_command_does_not_import_lexical():
    optional_trainer('lora')
    code = "import sys; import semantic_trainer.lora; assert 'semantic_trainer.lexical' not in sys.modules; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


def test_policy_pairs_require_instruction_dependence():
    from bench.policy_adaptation import pair_metrics
    rows = [{'pair': 'p', 'raw_value': 'x', 'raw_usable': True, 'expected': {'value': 'x'}},
            {'pair': 'p', 'raw_value': 'x', 'raw_usable': True, 'expected': {'value': 'y'}}]
    assert pair_metrics(rows)['success'] == 0
    rows[1]['raw_value'] = 'y'
    assert pair_metrics(rows)['success'] == 1


def test_regression_cohort_excluded_from_training():
    from bench.policy_adaptation import COHORT
    data = json.loads(COHORT.read_text())
    assert 'development' in data['population']
    assert all('semantic-decisions-v5' not in case['text'] for policy in data['policies'] for case in policy['cases'])


def authored_export(tmp_path, item, *, name='action', native=True):
    with Runtime({'judge': FakeScorer()}) as runtime:
        path = tmp_path / (name + '.yaml')
        item.save(path, rt=runtime)
        classes = runtime.load(path, draft=True).plan.fact(name).keys
        records = [{'input': {'message': f'case-{i}'}, 'facts': {name: classes[i % len(classes)] if classes[0] != 'true' else bool(i % 2)},
                    'group': f'case-{i}', 'selection': {'selected_by': 'manual'}} for i in range(24)]
        files.append(path.with_suffix('.test.jsonl'), envelopes(records))
        result = tools.export(runtime, path, native=native, task_family='example:policy')
        manifest, rows = read_export(result['path'], native=native)
        return runtime, path, manifest, rows


def test_pick_permutation_preserves_gold_and_family(tmp_path):
    from semantic_switch import Pick
    _, _, manifest, rows = authored_export(tmp_path, Pick('Choose.', {'x': 'One', 'y': 'Two', 'z': 'Three'}))
    assert manifest['schema'] == 12 and manifest['native'] is True and all('execution' in row['native'] for row in rows)
    for row in rows:
        canonical, permuted = row['native']['presentations']
        assert canonical['variant'] == 'canonical' and permuted['variant'] == 'permuted'
        assert canonical['slot_map'] != permuted['slot_map']
        assert permuted['slot_map'][permuted['label']] == row['label']
        assert canonical['prefix'] == permuted['prefix'] and canonical['input'] == permuted['input']


def test_rate_training_preserves_level_order(tmp_path):
    from semantic_switch import Rate
    _, _, _, rows = authored_export(tmp_path, Rate('How urgent?', ('low', 'medium', 'high')))
    assert all(len(row['native']['presentations']) == 1 for row in rows)
    assert all(row['native']['presentations'][0]['slot_map'] == [0, 1, 2] for row in rows)


def test_native_training_export_matches_deployed_request(tmp_path):
    from semantic_switch import Pick
    with Runtime({'judge': FakeScorer()}) as runtime:
        path = tmp_path / 'action.yaml'
        item = Pick('Choose.', {'x': 'One', 'y': 'Two'}, guidance='Use the second choice for report B.')
        switch = item.save(path, rt=runtime)
        switch.correct('report B', fact='action', value='y', group='case')
        exported = tools.export(runtime, path, split='all', native=True)
        _, rows = read_export(exported['path'], fitting=False, native=True)
        row = rows[0]
        call = switch.exec.start(switch.snapshot, switch.bind({'message': 'report B'}, switch.snapshot))
        question = switch.exec.presentation(switch.snapshot, call, 0).question
        request = runtime.judge().adapter.request(runtime.judge().adapter.state(row['text']), [question])
        entry = row['native']['presentations'][0]
        assert entry['prefix'] == list(request.prefix)
        assert entry['input'] == list(request.state)
        assert entry['question'] == list(request.suffixes[0])
        assert entry['answer_slots'] == [list(group) for group in request.slots[0]]


def test_lora_job_never_consumes_check_rows(tmp_path):
    from semantic_switch import Pick
    resolve = optional_trainer('jobs').resolve
    _, path, _, _ = authored_export(tmp_path, Pick('Choose.', {'x': 'One', 'y': 'Two'}))
    with Runtime({'judge': FakeScorer()}) as runtime:
        check = tools.export(runtime, path, split='check', to=tmp_path / 'check.jsonl', native=True)
    with pytest.raises(ConfigError, match='tune-only'):
        read_export(check['path'], native=True)


def test_training_benchmark_keeps_errors_and_negative_results():
    from bench.outcomes import classification_metrics
    row = {'case_id': 'bad', 'family': 'bad', 'expected': {'value': 'x', 'status': 'ok'}, 'result': {'status': 'failed', 'value': None}}
    report = classification_metrics([row])
    assert report['submitted'] == 1 and report['protocol_failures'] == 1 and report['accepted'] == 0


def specialist(tmp_path, rows, payload=b'specialist', kind='specialization', **change):
    from semantic_switch.artifacts import RECEIPT_NAME
    directory = tmp_path / 'deployment'
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (('specialist.gguf', payload), ('task.yaml', b'name: fixture\n'), ('task.lock.json', b'{}\n')):
        (directory / name).write_bytes(content)
    listed = {name: files.sha256_file(directory / name) for name in ('specialist.gguf', 'task.yaml', 'task.lock.json')}
    receipt = {'schema': RECEIPT_SCHEMA, 'kind': 'specialization-receipt', 'selected': 'epoch-1', 'gguf': 'specialist.gguf', 'task': 'task.yaml',
               'lock': {'path': 'task.lock.json', 'id': 'fixture', 'candidate_id': 'c' * 64}, 'files': listed,
               'native': {'device': 'cpu', 'n_ctx': 2048}, 'execution': {'device': 'cpu'}, 'provenance': {'kind': kind, 'rows': rows}, **change}
    receipt['id'] = artifact_digest(receipt)
    files.publish(directory / RECEIPT_NAME, receipt)
    return directory / 'specialist.gguf', directory / RECEIPT_NAME


def receipt_row(**change):
    return {'id': 'a' * 64, 'fact': 'refund', 'projection': '{}', 'family_seeds': [['group', 'g']], 'group': 'g', 'uses': ['adaptation', 'selection'],
            'component': 'c' * 64, 'sources': [{'export_id': 'b' * 64, 'path': 'fixture.export.jsonl.manifest.json', 'task_family': 'fixture:refund', 'namespace': 'fixture',
                                     'component': 'e' * 64, 'group': 'g', 'aliases': ['g'], 'family_seeds': [['group', 'g']], 'dataset_role': 'fit'}],
            'label_sources': ['d' * 64], **change}


def test_specialist_model_retains_verified_provenance_across_moves(tmp_path):
    from dataclasses import fields
    from semantic_switch import Model
    gguf, path = specialist(tmp_path, [receipt_row(), receipt_row(id='d' * 64, uses=['acceptance_fit'])])
    model = Model.from_specialization(path, device='gpu')
    assert model.source == str(gguf) and model.device == 'gpu' and model.n_ctx == 2048
    assert model != Model(str(gguf), device='gpu', n_ctx=2048) and Model(str(gguf)).provenance is None
    assert model.provenance.kind == 'specialization' and model.provenance.gguf_sha256 == files.sha256_file(gguf)
    assert [row['id'] for row in model.provenance.membership('adaptation')] == ['a' * 64]
    assert [row['id'] for row in model.provenance.membership('acceptance_fit')] == ['d' * 64]
    assert not {'lora', 'lora_scale', 'provenance'} & set(api.NATIVE_SETTINGS) and 'provenance' in {field.name for field in fields(Model)}
    moved = tmp_path / 'moved'
    (tmp_path / 'deployment').rename(moved)
    renamed = moved / 'other-name.json'
    (moved / 'receipt.json').rename(renamed)
    assert Model.from_specialization(renamed).provenance == model.provenance
    (moved / 'specialist.gguf').write_bytes(b'changed')
    with pytest.raises(ConfigError, match='changed since selection'):
        Model.from_specialization(renamed)


def test_pipeline_base_provenance_differs_from_unknown_and_rejects_adaptation():
    from semantic_switch import Model
    from semantic_switch.artifacts import Provenance
    base = Model('base.gguf', provenance=Provenance.base('a' * 64, 'b' * 64))
    assert base != Model('base.gguf') and base.provenance.kind == 'base' and base.provenance.membership('adaptation') == ()
    with pytest.raises(ConfigError, match='adaptation membership contradicts'):
        Provenance('base', 'b' * 64, receipt='c' * 64, rows=({**receipt_row()},))
    with pytest.raises(ConfigError, match='provenance must come from'):
        Model('base.gguf', provenance={'kind': 'base'})


@pytest.mark.parametrize('change', [{'schema': 1}, {'provenance': {'kind': 'specialization', 'rows': [receipt_row(uses=['tune'])]}},
                                    {'provenance': {'kind': 'specialization', 'rows': [receipt_row(family_seeds=None)]}},
                                    {'provenance': {'kind': 'base', 'rows': [receipt_row()]}}, {'files': {}}])
def test_malformed_receipt_is_rejected(tmp_path, change):
    _, path = specialist(tmp_path, **{'rows': [receipt_row()], **change})
    with pytest.raises(ConfigError, match='invalid deployment receipt'):
        load_receipt(path)


def test_lock_reads_lineage_from_the_model_provenance_not_filenames(tmp_path):
    from semantic_switch import Model
    from semantic_switch.tools.lock import LockError, specialization
    gguf, path = specialist(tmp_path, [receipt_row()])
    identity = {'sha256': files.sha256_file(str(gguf))}
    bound = SimpleNamespace(models={'judge': Model.from_specialization(path)}, identity=lambda: identity)
    provenance, rows = specialization(bound)
    assert provenance.receipt == load_receipt(path).id and rows[0]['id'] == 'a' * 64
    assert specialization(SimpleNamespace(models={'judge': Model(str(gguf))}, identity=lambda: identity)) == (None, None)
    assert specialization(SimpleNamespace(models={'judge': FakeScorer()}, identity=lambda: {})) == (None, ())
    identity['sha256'] = 'f' * 64
    with pytest.raises(LockError, match='differ from their verified provenance'):
        specialization(bound)


def test_cli_has_no_runtime_adapter_options():
    from semantic_switch import cli
    assert not {'lora', 'lora_scale'} & set(cli.NATIVE)
    result = subprocess.run([sys.executable, '-m', 'semantic_switch', '--help'], capture_output=True, text=True)
    assert '--lora' not in result.stdout


def test_lora_example_reports_unaccepted_outcomes():
    from examples.adapt_judgment import show
    assert show(SimpleNamespace(value=None, status='uncertain', prediction=None, error=None)) == {'value': None, 'status': 'uncertain', 'prediction': None, 'error': None}


def label_file(tmp_path, values=('escalate', 'continue', 'clarify'), n=30, **extra):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.examples import label_record
    from examples.adapt_judgment import ACTION
    from semantic_switch.examples import LabelSource
    with Runtime({'judge': FakeScorer()}) as runtime:
        path = ACTION.save(tmp_path / 'scratch' / 'action.yaml', rt=runtime).paths.yaml
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    records = tmp_path / 'labels.jsonl'
    teacher = [LabelSource('synthetic', 'openrouter:google/gemma-4-31b-it').record()]
    files.publish(records, ''.join(json.dumps(label_record(plan, {'message': f'report {i}'}, {'action': values[i % len(values)]}, groups=[f'situation-{i}'],
                                                           sources={'action': teacher}, tags=[('relayed', 'quoted')[i % 2]], **extra)) + '\n' for i in range(n)))
    return records


def test_public_judgment_to_native_export_workflow(tmp_path):
    from examples.adapt_judgment import prepare
    records = label_file(tmp_path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        result = prepare(runtime, tmp_path / 'workflow', records, 'maintenance:test')
        _, rows = read_export(result['path'], native=True)
    assert rows and all(row['task_family'] == 'maintenance:test' and row['fact'] == 'action' for row in rows)
    assert all(row['label_origin'] == 'synthetic' and row['tags'] in (['quoted'], ['relayed']) for row in rows)
    assert not (tmp_path / 'workflow' / 'action.test.jsonl').exists() and not (tmp_path / 'workflow' / 'action.examples.jsonl').exists()


def test_focused_training_escape_gold(tmp_path):
    from semantic_switch import Pick
    with Runtime({'judge': FakeScorer()}) as runtime:
        item = Pick('Choose the referenced part.', {'ax': 'AX', 'by': 'BY', 'cz': 'CZ'}, focus='mentions')
        path = tmp_path / 'part.yaml'
        switch = item.save(path, rt=runtime)
        switch.correct('AX is mentioned but CZ is required.', fact='part', value='cz')
        report = tools.export(runtime, path, split='all', native=True)
        _, rows = read_export(report['path'], fitting=False, native=True)
        for entry in rows[0]['native']['presentations']:
            wanted = rows[0]['label'] if rows[0]['label'] in entry['slot_map'] else -1
            assert entry['slot_map'][entry['label']] == wanted


def fake_assets(base, quant='Q4_0'):
    from semantic_trainer import assets
    repo, commit = 'example/tiny', 'a' * 40
    paths = assets.layout(repo, commit)
    for relative in assets.TOOLS:
        (base / paths['toolchain'] / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / paths['toolchain'] / relative).write_text(relative)
    (base / paths['hf']).mkdir(parents=True)
    files.publish(base / paths['hf'] / 'config.json', {'model_type': 'qwen3'})
    (base / paths['gguf']).mkdir(parents=True)
    for kind in ('F16', quant):
        (base / paths['gguf'] / f'{kind}.gguf').write_bytes(kind.encode())
    return assets.record(base, repo, commit, [quant]), base / paths['gguf'] / f'{quant}.gguf'


def training_job(tmp_path, monkeypatch):
    from semantic_switch import Pick, api
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    digest, gguf = fake_assets(tmp_path / 'assets')
    original = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**original(self, role), 'sha256': files.sha256_file(str(gguf))})
    _, path, manifest, rows = authored_export(tmp_path, Pick('Choose.', {'x': 'One', 'y': 'Two'}))
    from semantic_trainer.jobs import LOSS, SCHEMA
    export = str(path.with_suffix('.export.jsonl')) + '.manifest.json'
    from semantic_switch.training import freeze_task
    frozen = freeze_task(path, tmp_path / 'task')
    job = {'schema': SCHEMA, 'loss': LOSS, 'objective': 'specialization', 'task': {'source': str(path), 'bundle': 'task', 'behavior': frozen['behavior']},
           'assets': {'manifest': digest, 'quant': 'Q4_0'},
           'fitting': {'device': 'cpu'}, 'exports': [{'path': export, 'export_id': manifest['export_id'],
                                                      'roles': {'fit': [row['group'] for row in rows[:-4]], 'model_select': [row['group'] for row in rows[-4:]]}}],
           'recipe': {'eval_every_examples': 5}, 'budget': {'max_tokens': 100000, 'max_sequence_tokens': 1000, 'max_seconds': 100}}
    job_path = tmp_path / 'job.yaml'
    files.publish(job_path, job)
    return job_path, job, rows


def test_job_resolves_monitoring_recipe_schedule_and_token_work(tmp_path, monkeypatch):
    import math
    resolve = optional_trainer('jobs').resolve
    job_path, job, rows = training_job(tmp_path, monkeypatch)
    resolved, observations = resolve(job_path)
    fit = [row for row in observations if row['role'] == 'fit']
    recipe, n = resolved['recipe'], len(fit)
    assert len(fit) == len(rows) - 4 and recipe['epochs'] == 8 and recipe['min_epochs'] == 2 and recipe['selection_unit'] == 'observation'
    assert (recipe['eval_every_examples'], recipe['patience_examples'], recipe['warmup_examples']) == (5, n, -(-3 * n // 20))
    assert resolved['schedule']['max_optimizer_steps'] == 8 * math.ceil(n / 8) and resolved['schedule']['warmup_steps'] == math.ceil(recipe['warmup_examples'] / 8)
    job['recipe'] = {'epochs': 2, 'eval_every_examples': 5}
    files.publish(job_path, job)
    resolved, _ = resolve(job_path)
    assert resolved['recipe']['epochs'] == 2 and resolved['schedule']['max_optimizer_steps'] == 2 * math.ceil(n / 8)
    assert resolved['work']['training_tokens'] == 2 * sum(sum(len(p[k]) for k in ('prefix', 'input', 'question')) for row in rows[:-4] for p in row['native']['presentations'])
    for change, match in (({'min_epochs': 3, 'epochs': 2}, 'exceeds the epoch budget cap'), ({'min_improvement': 1}, 'min_improvement'),
                          ({'selection_unit': 'family'}, 'selection_unit'), ({'warmup_ratio': .05}, 'unsupported recipe fields')):
        files.publish(job_path, {**job, 'recipe': {'eval_every_examples': 5, **change}})
        with pytest.raises(ConfigError, match=match):
            resolve(job_path)


def test_training_rejects_implicit_truncation(tmp_path, monkeypatch):
    resolve = optional_trainer('jobs').resolve
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    job['budget']['max_sequence_tokens'] = 1
    files.publish(job_path, job)
    with pytest.raises(ConfigError, match='no implicit truncation'):
        resolve(job_path)


def test_policy_family_split_keeps_counterfactuals_together(tmp_path, monkeypatch):
    resolve = optional_trainer('jobs').resolve
    job_path, job, rows = training_job(tmp_path, monkeypatch)
    job['exports'][0]['roles']['fit'] = [next(node[1] for node in row['family_seeds'] if node[0] == 'group') for row in rows[:2]]
    files.publish(job_path, job)
    resolved, observations = resolve(job_path)
    assert [row['role'] for row in observations if row['role'] == 'fit'] == ['fit', 'fit'] and len(resolved['memberships']) == len(rows)
    assert {row['role'] for row in resolved['memberships']} == {'fit', 'model_select', 'unused'}


def test_unsplittable_training_cohort_is_rejected(tmp_path, monkeypatch):
    resolve = optional_trainer('jobs').resolve
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    job['exports'][0]['roles']['fit'] = ['absent']
    files.publish(job_path, job)
    with pytest.raises(ConfigError, match='unknown group'):
        resolve(job_path)


def test_multi_export_family_assignments_are_consistent(tmp_path, monkeypatch):
    resolve = optional_trainer('jobs').resolve
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    job['exports'][0]['roles']['acceptance_fit'] = job['exports'][0]['roles']['fit']
    files.publish(job_path, job)
    with pytest.raises(ConfigError, match='conflicting roles'):
        resolve(job_path)


def test_missing_extra_reports_install_command(monkeypatch):
    import builtins
    dependencies = optional_trainer('lora').dependencies
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError('fixture unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    with pytest.raises(ConfigError, match=r'semantic-trainer\[lora\]'):
        dependencies()


def test_adapter_audit_overlap_not_independent():
    manifest = membership()
    labels = SimpleNamespace(families=set(), group=lambda node: 'g')
    rows = [SimpleNamespace(id='audit', group='g', projection='other')]
    result = assess_lineage(manifest, labels, rows)
    assert result.state == 'known_overlap' and result.overlap == ('audit',)


def test_head_admission_counts_actual_avoided_work():
    from bench.head_admission import admission_work
    arms = [{'native_work': {'submissions': 10}, 'results': [{'result': {'facts': {'f': {'source': 'judge', 'status': 'ok'}}}}]},
            {'native_work': {'submissions': 10}, 'results': [{'result': {'facts': {'f': {'source': 'classifier', 'status': 'uncertain'}}}}]}]
    assert not admission_work(arms)[1]['admitted_speedup']
    arms[1]['native_work']['submissions'] = 9
    arms[1]['results'][0]['result']['facts']['f']['status'] = 'ok'
    assert admission_work(arms)[1]['admitted_speedup']
    assert arms[1]['avoided_native_work']['submissions'] == 1


def test_specialization_allows_policy_overlap_not_case_overlap(tmp_path, monkeypatch):
    resolve = optional_trainer('jobs').resolve
    path, job, rows = training_job(tmp_path, monkeypatch)
    job['exports'][0]['roles'] |= {'fit': [rows[0]['group'], rows[1]['group']], 'acceptance_fit': [rows[2]['group']]}
    files.publish(path, job)
    resolved, observations = resolve(path)
    assert sorted(row['role'] for row in observations) == ['acceptance_fit', 'fit', 'fit', *['model_select'] * 4]
    job['exports'][0]['roles']['fit'] = [rows[0]['group']]
    files.publish(path, job)
    with pytest.raises(ConfigError, match='without fit support'):
        resolve(path)
    job['exports'][0]['roles']['fit'] = [rows[0]['group'], rows[1]['group']]
    job['objective'] = 'instruction_transfer'
    files.publish(path, job)
    with pytest.raises(ConfigError, match='policy family overlaps'):
        resolve(path)


def test_multiple_pair_memberships_survive_export(tmp_path):
    from semantic_switch import Pick
    _, path, _, _ = authored_export(tmp_path, Pick('Choose.', {'x': 'One', 'y': 'Two'}))
    source = path.with_suffix('.test.jsonl')
    records = files.read_jsonl(source)
    records[0]['pairs'] = ['test:opposite', 'test:equivalent']
    files.publish(source, ''.join(json.dumps(row) + '\n' for row in records))
    with Runtime({'judge': FakeScorer()}) as runtime:
        report = tools.export(runtime, path, native=True, split='all')
        _, rows = read_export(report['path'], fitting=False, native=True)
    assert any(set(row['pairs']) == {'test:opposite', 'test:equivalent'} for row in rows)


def test_model_selection_rows_excluded_from_lock(tmp_path, monkeypatch):
    from importlib import import_module
    module = import_module('semantic_switch.tools.lock')
    path = definition(tmp_path)
    dataset(path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        labels = Labels(runtime.load(path, draft=True))
        example = labels.examples('refund', 'tune')[0]
        _, receipt = specialist(tmp_path, [receipt_row(id=example.id, projection=example.projection, group=example.group,
                                                       family_seeds=labels.seeds(example.group), uses=['selection'])], kind='base')
        provenance = load_receipt(receipt).provenance
        monkeypatch.setattr(module, 'specialization', lambda rt: (provenance, [dict(row) for row in provenance.rows]))
        locked = tools.lock(runtime, path)
        loaded = runtime.load(path)
        assert loaded.snapshot.lineage['judge-development:refund']['excluded'] == 1
        assert example.projection not in loaded.snapshot.membership['refund']['tuned']
        assert files.read_json(path.with_suffix('.lock.json'))['predictor_artifacts']['judge']['receipt'] == provenance.receipt


def test_adaptation_template_variants_share_partition():
    from bench.policy_adaptation import COHORT
    cohort = json.loads(COHORT.read_text())
    assert all('Case reference' not in row['text'] for policy in cohort['policies'] for row in policy['cases'])
    groups = {}
    for policy in cohort['policies']:
        for row in policy['cases']:
            key = (policy['task_family'], row['text'])
            groups.setdefault(key, set()).add(row['family'])
    assert all(len(families) == 1 for families in groups.values())


def test_native_answer_mass_stays_bounded_at_saturation():
    from semantic_switch.backend import reduce_row
    logits = np.full(262144, -30., dtype=np.float32)
    logits[3], logits[9], logits[12] = 40., 32., 28.
    logp, log_mass = reduce_row(logits, ((3, 12), (9,)))
    assert log_mass <= 0 and 0 < np.exp(log_mass) <= 1
    assert np.exp(logp).sum() == pytest.approx(1)
    _, all_mass = reduce_row(np.array([40., 32.], dtype=np.float32), ((0,), (1,)))
    assert all_mass == 0


def test_unknown_gold_operational_expectation():
    from bench.outcomes import classification_metrics
    report = classification_metrics([{'case_id': 'one', 'family': 'one', 'raw_value': 'unstated', 'raw_usable': True,
                                     'expected': {'value': None, 'raw_value': 'unstated', 'status': 'needs_input'},
                                     'result': {'value': None, 'status': 'needs_input'}}])
    assert report['raw_correct'] == 1 and report['operational_agreement'] == 1
    assert report['accepted'] == 0 and report['submitted'] == 1


def test_window_work_counted_once():
    from bench.policy_adaptation import audit_window
    counter = {'submissions': 0, 'submitted_tokens': 0}
    runtime = SimpleNamespace(info=lambda: {'capabilities': {'counters': dict(counter)}})
    class Installed:
        snapshot = SimpleNamespace(candidate_id='frozen')
        def run(self, cases):
            counter.update(submissions=counter['submissions'] + 1, submitted_tokens=counter['submitted_tokens'] + 30)
            return [SimpleNamespace(as_dict=lambda verbose: {'value': 'yes', 'status': 'ok'}) for case in cases]
    cases = [SimpleNamespace(id=str(i), data={'message': str(i)}) for i in range(3)]
    decisions, first = audit_window(runtime, Installed(), cases, 'first', 'invalidated')
    _, replay = audit_window(runtime, Installed(), cases, 'replay', 'immediate replay')
    assert len(decisions) == 3 and first['native_work']['submitted_tokens'] == 30
    assert replay['native_work']['submitted_tokens'] == 30 and replay['cache_state'] == 'immediate replay'
    assert first['members'] == ['0', '1', '2']
    assert 'native_work' not in vars(decisions[0]) and first['host_peak_bytes'] > 0


def test_comparison_rejects_missing_ordered_results():
    from bench.policy_adaptation import audit_window
    runtime = SimpleNamespace(info=lambda: {'capabilities': {'counters': {}}})
    installed = SimpleNamespace(snapshot=SimpleNamespace(candidate_id='frozen'), run=lambda datas: [])
    with pytest.raises(ConfigError, match='incomplete ordered'):
        audit_window(runtime, installed, [SimpleNamespace(id='one', data={})], 'window', 'fresh')


def test_comparison_recomputes_from_raw_rows():
    from bench.policy_adaptation import paired_changes, summarize
    path = Path(__file__).parents[1] / 'bench/results/policy-adaptation-gemma-v2.json'
    report = files.read_json(path)
    for arm in report['arms']:
        for key, value in summarize(arm['rows']).items():
            assert arm[key] == value
        by_window = {window['id']: window for window in arm['windows']}
        assert len(by_window) == len(arm['windows'])
        assert all(row['window_id'] in by_window for row in arm['rows'])
        assert all('native_work' not in row for row in arm['rows'])
        assert any('replay' in window['cache_state'] for window in arm['windows'])
    assert report['paired'] == paired_changes(report['arms'])


def test_comparison_rejects_mismatched_candidate_membership():
    from bench.policy_adaptation import paired_changes
    row = {'case_id': 'one', 'task_family': 'task'}
    with pytest.raises(ConfigError, match='observations differ'):
        paired_changes([{'name': 'base', 'rows': [row]}, {'name': 'adapter', 'rows': []}])


def test_nonfinite_answer_slot_rejected_with_finite_mass(monkeypatch):
    module = optional_trainer('lora')
    from semantic_switch.protocols import ScoreBlock
    block = ScoreBlock(np.array([[np.nan, -1.]], dtype=np.float32), np.array([-.1], dtype=np.float32), [None])
    class Native:
        def __init__(self, models, job):
            assert job == 'parent'
        def identity(self):
            return {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def score_native(self, requests):
            return block
    monkeypatch.setattr(module, 'Runtime', Native)
    monkeypatch.setattr(module, 'native_model', lambda resolved: None)
    row = {'id': 'a', 'native': {'presentations': [{'id': 'p', 'variant': 'canonical', 'slot_map': [0, 1], 'prefix': [1], 'input': [2], 'question': [3], 'answer_slots': [[4], [5]]}]}}
    with pytest.raises(ConfigError, match='nonfinite answer slots'):
        module.native_answers({'native': {}}, [row], 'parent')


def test_no_effect_adapter_publication_is_diagnostic():
    report = files.read_json(Path(__file__).parents[1] / 'bench/results/lora-noop-publication.json')
    assert report['metrics']['effect_diagnostic'] == 'no_measurable_effect'
    assert report['metrics']['max_native_logp_change'] == 0
    assert report['metrics']['native_adapted']['finite']
    assert report['quality_claim'] is False
    assert report['resolved']['recipe']['epochs'] == 2
    assert {row['epoch'] for row in report['metrics']['fit']} == {1, 2}


def test_locked_evaluation_is_plain_json(tmp_path):
    path = definition(tmp_path)
    dataset(path)
    with Runtime({'judge': FakeScorer()}) as runtime:
        tools.lock(runtime, path)
        report = tools.run_test(runtime, path, intervals=True)
    encoded = json.dumps(report, allow_nan=False)
    assert 'mappingproxy' not in encoded
    assert all(isinstance(policy, dict) for population in report['fact_populations'].values()
               for fact in population.values() for policy in fact['policy'].values())


def test_selected_fact_export_keeps_dataset_families(tmp_path):
    path = tmp_path / 'two.yaml'
    files.publish(path, {'name': 'two', 'fields': {'message': {'type': 'text'}}, 'facts': {'first': {'check': 'First claim.'}, 'second': {'check': 'Second claim.'}}})
    files.append(path.with_suffix('.test.jsonl'), envelopes([{'input': {'message': str(i)}, 'facts': {'first': True, 'second': False}, 'group': f'situation:{i}'} for i in range(20)]))
    with Runtime({'judge': FakeScorer()}) as runtime:
        whole = tools.export(runtime, path, to=tmp_path / 'whole.jsonl', native=True, split='all')
        selected = tools.export(runtime, path, to=tmp_path / 'first.jsonl', native=True, split='all', fact='first')
    _, all_rows = read_export(whole['path'], fitting=False, native=True)
    _, rows = read_export(selected['path'], fitting=False, native=True)
    assert rows and {row['fact'] for row in rows} == {'first'}
    assert rows == [row for row in all_rows if row['fact'] == 'first']
    assert selected['dataset_id'] == whole['dataset_id']


def pipeline_base(tmp_path, monkeypatch):
    from semantic_switch import api
    optional_trainer('jobs')
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    digest, gguf = fake_assets(tmp_path / 'assets')
    original = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**original(self, role), 'sha256': files.sha256_file(str(gguf))})
    return digest


def test_suite_pilot_prepares_through_the_trainer_and_refuses_unsupported_answers(tmp_path, monkeypatch):
    from bench.policy_adaptation import suites
    digest = pipeline_base(tmp_path, monkeypatch)
    with Runtime({'judge': FakeScorer()}) as runtime, pytest.raises(ConfigError, match='without fit support: .*refund: true, false'):
        suites(runtime, tmp_path / 'pilot', assets=digest)
    assert not (tmp_path / 'pilot' / 'job.yaml').exists()


def test_suite_pilot_keeps_families_in_one_role_and_audits_only_accept_and_check(tmp_path, monkeypatch):
    import bench.policy_adaptation as pilot_module
    prepared = []
    monkeypatch.setattr(pilot_module, 'lora_job', lambda definition, export, directory, **settings: prepared.append(export) or
                        {'job': str(directory / 'job.yaml'), 'commands': {}, 'support': {}})
    with Runtime({'judge': FakeScorer()}) as runtime:
        pilot = pilot_module.suites(runtime, tmp_path / 'pilot', assets='a' * 64)
        audit = Labels(runtime.load(pilot['definitions'][0]['path'], draft=True))
    groups = pilot['groups']
    owners = [group for values in groups.values() for group in values]
    assert len(owners) == len(set(owners)) and all(groups.values())
    assert sum(pilot['observations'].values()) == 400
    _, rows = read_export(prepared[0], fitting=False, native=True)
    folds = {'fit': (0, 1, 2), 'model_select': (3,), 'acceptance_fit': (4,)}
    assert all(row['group'] in groups[role] for row in rows for role, values in folds.items() if row['fold'] in values)
    assert {row['group'] for row in rows} == set(groups['fit'] + groups['model_select'] + groups['acceptance_fit'])
    for row in rows:
        endpoints = [other for other in rows if set(other['pairs']) & set(row['pairs'])]
        assert len(endpoints) == 2 and {other['group'] for other in endpoints} == {row['group']}
    held = {e.group for fact in pilot['definitions'] for e in audit.examples(fact['fact'])}
    assert held == set(groups['acceptance_fit'] + groups['check'])
    assert {e.split for fact in pilot['definitions'] for e in audit.examples(fact['fact']) if e.group in groups['check']} == {'check'}
