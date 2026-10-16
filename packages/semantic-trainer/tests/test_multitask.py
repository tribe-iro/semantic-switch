import hashlib
import json
from pathlib import Path

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import artifact_digest, read_export
from semantic_switch.training import Resources
from semantic_switch.compiler import compile_spec
from semantic_switch.examples import each_target, label_record
from semantic_testkit.scoring import FakeScorer
from semantic_testkit.assets import pipeline_base

RECIPE = {'eval_every_examples': 8}
OPTIONS = {'calibrate': 'Run sensor calibration', 'resume': 'Resume printing'}
SOURCE = [{'derivation': 'computed', 'producer': 'mixture-reference', 'revision': '1'}]
SHARED = FakeScorer()


def definition(tmp_path, name, question, urgent=False):
    path = tmp_path / f'{name}.yaml'
    facts = {'action': {'pick': question, 'options': OPTIONS}, **({'urgent': {'check': 'The report describes an urgent stoppage.'}} if urgent else {})}
    files.publish(path, {'name': name, 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'], 'facts': facts})
    return path


def exported(tmp_path, path, name, family, n, role, message=lambda i: None, split='tune'):
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    rows = []
    for i in range(n):
        facts = {'action': ('calibrate', 'resume')[i % 2], **({'urgent': i % 4 < 2} if 'urgent' in plan.spec['facts'] else {})}
        rows.append(label_record(plan, {'message': message(i) or f'{name} report {i}'}, facts, sources=each_target(facts, SOURCE), groups=[f'{name}-{i}'],
                                 role=role(i)))
    labels = tmp_path / f'{name}.jsonl'
    files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(FakeScorer()) as runtime:
        return tools.export(runtime, path, to=tmp_path / f'{name}.export.jsonl', native=True, labels=labels, task_family=family, split=split)['path']


def cyclic(i):
    return ('fit', 'fit', 'model_select')[i % 3]


def two_tasks(tmp_path, monkeypatch, bridge=None):
    manifest = pipeline_base(tmp_path, monkeypatch)
    a = definition(tmp_path, 'manual_a', 'Which action does manual A require?', urgent=True)
    b = definition(tmp_path, 'manual_b', 'Which action does manual B require?')
    ea = exported(tmp_path, a, 'a', 'printer:a', 12, cyclic)
    eb = exported(tmp_path, b, 'b', 'printer:b', 6, cyclic, message=bridge or (lambda i: None))
    return manifest, a, b, ea, eb


def sources(a, b, ea, eb, weights=(1, 1)):
    return [{'definition': a, 'exports': [ea], 'weight': weights[0], 'contract': None}, {'definition': b, 'exports': [eb], 'weight': weights[1], 'contract': None}]


def test_task_map_scopes_colliding_addresses_and_weights_each_task_equally_per_example(tmp_path, monkeypatch):
    from semantic_trainer.jobs import fit_inputs, monitor_plan, prepare_tasks, resolve
    manifest, a, b, ea, eb = two_tasks(tmp_path, monkeypatch)
    result = prepare_tasks(sources(a, b, ea, eb), tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    resolved, observations = resolve(result['job'])
    by_task = {}
    for row in observations:
        by_task.setdefault(Path(resolved['tasks'][row['task']]['source']).stem, []).append(row)
    ca, cb = (by_task[name][0]['task'] for name in ('manual_a', 'manual_b'))
    assert set(resolved['tasks']) == set(result['tasks']) == {ca, cb} and files.read_yaml(result['job'])['tasks'][ca]['bundle'] == f'tasks/{ca}'
    assert resolved['tasks'][ca]['revisions']['action'] != resolved['tasks'][cb]['revisions']['action']
    assert set(resolved['support']['roles']['fit'][ca]) == {'action', 'urgent'} and set(resolved['support']['roles']['fit'][cb]) == {'action'}
    assert all(key.startswith(f"{entry['task']}:") for key, entry in resolved['situations'].items())
    assert {entry['task'] for entry in resolved['situations'].values()} == {ca, cb}
    fit = {row['id']: row for row in observations if row['role'] == 'fit'}
    selecting = [row for row in observations if row['role'] == 'model_select']
    assert sum(row['task'] == ca for row in fit.values()) == 16 and sum(row['task'] == cb for row in fit.values()) == 4
    assert {row['objective'] for row in fit.values() if row['task'] == ca} == {.5 / 8 / 2 * 20} and {row['objective'] for row in fit.values() if row['task'] == cb} == {.5 / 4 * 20}
    assert {row['objective'] for row in selecting if row['task'] == ca} == {.5 / 4 / 2 * 10} and {row['objective'] for row in selecting if row['task'] == cb} == {.5 / 2 * 10}
    report = resolved['objective_weights']['tasks']
    assert report[ca]['fit'] == {'weight': .5, 'examples': 8, 'observations': 16, 'presentations': sum(len(r['presentations']) for r in fit.values() if r['task'] == ca),
                                 'tokens': report[ca]['fit']['tokens']} and report[cb]['model_select']['examples'] == 2 and report[cb]['fit']['weight'] == .5
    assert resolved['fit']['fields']['observations'] == artifact_digest(fit_inputs(resolved, list(fit.values()))['observations'])
    assert monitor_plan(resolved, selecting, [fit[i] for i in resolved['probe_ids']])['tasks'] == [0 if row['task'] == selecting[0]['task'] else 1 for row in selecting]
    again = prepare_tasks(sources(a, b, ea, eb)[::-1], tmp_path / 'again', assets_manifest=manifest, recipe=RECIPE)
    assert (again['fit_id'], again['monitor_id']) == (result['fit_id'], result['monitor_id'])
    weighted, _ = resolve(prepare_tasks(sources(a, b, ea, eb, (3, 1)), tmp_path / 'weighted', assets_manifest=manifest, recipe=RECIPE)['job'])
    assert weighted['objective_weights']['tasks'][ca]['fit']['weight'] == .75 and weighted['fit']['id'] != resolved['fit']['id']
    assert weighted['monitor']['fields']['model_select'] != resolved['monitor']['fields']['model_select']


def test_task_maps_reject_repeated_exports_derivative_contracts_and_cross_task_role_bridges(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare_tasks
    manifest, a, b, ea, eb = two_tasks(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match='listed under two task contracts'):
        prepare_tasks([*sources(a, b, ea, eb)[:1], {'definition': b, 'exports': [ea, eb], 'weight': 1, 'contract': None}], tmp_path / 'repeated',
                      assets_manifest=manifest, recipe=RECIPE)
    renamed = tmp_path / 'renamed'
    renamed.mkdir()
    (renamed / a.name).write_text(a.read_text())
    with pytest.raises(ConfigError, match='same task contract'):
        prepare_tasks([sources(a, b, ea, eb)[0], {'definition': renamed / a.name, 'exports': [eb], 'weight': 1, 'contract': None}], tmp_path / 'derivative',
                      assets_manifest=manifest, recipe=RECIPE)
    with pytest.raises(ConfigError, match='not the declared'):
        prepare_tasks([{**sources(a, b, ea, eb)[0], 'contract': 'f' * 64}], tmp_path / 'contract', assets_manifest=manifest, recipe=RECIPE)
    assert not any((tmp_path / name / 'job.yaml').exists() or (tmp_path / name / 'tasks').exists() for name in ('repeated', 'derivative', 'contract'))
    bridged = tmp_path / 'bridged'
    bridged.mkdir()
    manifest, a, b, ea, eb = two_tasks(bridged, monkeypatch, bridge=lambda i: 'a report 2' if i == 0 else None)
    with pytest.raises(ConfigError, match='crosses roles'):
        prepare_tasks(sources(a, b, ea, eb), bridged / 'job', assets_manifest=manifest, recipe=RECIPE)


def test_task_macro_ranking_cannot_be_won_by_the_larger_task():
    from semantic_trainer.jobs import ranking, unit_correct
    tasks = {**{f'big{i}': 'big' for i in range(10)}, 'small0': 'small', 'small1': 'small'}
    situations = {f'{task}:{i}': {'task': task, 'role': 'model_select', 'complete': True, 'targets': {'action': i}} for i, task in tasks.items()}
    def point(name, step, wrong):
        correct = {i: i not in wrong for i in tasks}
        return {'name': name, 'step': step, 'correct': unit_correct(correct, situations, 'model_select', tasks), 'nll': None}
    pooled = point('step-1', 1, {'small0', 'small1'})
    balanced = point('step-2', 2, {'big0', 'big1', 'big2'})
    assert (pooled['correct']['observation'], balanced['correct']['observation']) == (10, 9)
    assert pooled['correct']['macro'] == {'observation': (1 + 0) / 2, 'complete_situation': (1 + 0) / 2}
    assert balanced['correct']['macro'] == {'observation': (.7 + 1) / 2, 'complete_situation': (.7 + 1) / 2}
    assert balanced['correct']['tasks']['big'] == {'observation': 7, 'complete_situation': 7, 'situations': 10, 'observations': 10}
    assert [p['name'] for p in ranking([pooled, balanced], 'observation')] == ['step-2', 'step-1']
    assert [p['name'] for p in ranking([pooled, balanced], 'complete_situation')] == ['step-2', 'step-1']


def licence(directory):
    path = directory / 'LICENSE'
    if not path.exists():
        path.write_text('Creative Commons Attribution 4.0; training and evaluation use permitted\n')
    return files.sha256_file(path)


def manifest_task(role, definition_path, exports, contract, revisions, weight=None, permits=None, sha=None, underlying=()):
    leaf = {'dataset': f'{definition_path.stem}-data', 'revision': 'r1', 'sha256': sha or artifact_digest([definition_path.stem]), 'licence': 'CC-BY-4.0',
            'permits': permits or ['adaptation', 'selection', 'evaluation'], 'evidence': {'locator': 'LICENSE', 'sha256': licence(definition_path.parent)},
            'underlying': list(underlying)}
    return {'role': role, 'definition': definition_path.name, 'contract': contract, 'revisions': revisions,
            'exports': [{'path': Path(path).name, 'export_id': read_export(path, fitting=False, native=True)[0]['export_id']} for path in exports],
            'mapping': {'action': {'CAL': 'calibrate', 'RES': 'resume'}}, 'sources': [leaf], 'quota': 500, 'weight': weight}


def frozen_contract(tmp_path, path):
    from semantic_switch.training import freeze_task
    frozen = freeze_task(path, tmp_path / 'frozen' / path.stem)
    return frozen['behavior'], {address: target['meaning_rev'] for address, target in frozen['targets'].items()}


def published(path, manifest):
    files.publish(path, {**manifest, 'id': artifact_digest(manifest)})
    return path


def test_mixture_manifest_validates_permissions_derivatives_bridges_and_prepares_one_task_map(tmp_path, monkeypatch):
    from semantic_trainer.mixture import prepare_mixture, validate
    manifest = pipeline_base(tmp_path, monkeypatch)
    paths = {name: definition(tmp_path, name, f'Which action does {name} require?') for name in ('fit_a', 'fit_b', 'select_c', 'held_d')}
    roles = {'fit_a': 'fit', 'fit_b': 'fit', 'select_c': 'model_select', 'held_d': 'check'}
    exports = {name: exported(tmp_path, path, name, f'pilot:{name}', 6 if name != 'fit_a' else 8, lambda i, name=name: roles[name],
                              split='all' if name == 'held_d' else 'tune') for name, path in paths.items()}
    contracts = {name: frozen_contract(tmp_path, path) for name, path in paths.items()}
    mixed = exported(tmp_path, paths['fit_b'], 'fit_b_mixed', 'pilot:fit_b', 6, cyclic)
    def mixture(**changes):
        tasks = {f'pilot:{name}': manifest_task({'fit': 'adaptation', 'model_select': 'model_select', 'check': 'evaluation'}[roles[name]], path, [exports[name]],
                                                *contracts[name], weight=1 if roles[name] == 'fit' else None) for name, path in paths.items()}
        for name, change in changes.items():
            tasks[name] = {**tasks[name], **change}
        return {'schema': 1, 'kind': 'task-mixture', 'claim': {'statement': 'decision skill transfers to held-out manuals', 'holdout': ['document', 'rule']}, 'tasks': tasks}
    good = published(tmp_path / 'mixture.json', mixture())
    checked = validate(good)
    fit_a = checked['coverage']['tasks']['pilot:fit_a']
    assert fit_a == {'role': 'adaptation', 'complete_examples': 8, 'incomplete_examples': 0, 'observations': 8, 'families': 8,
                     'presentations': fit_a['presentations'], 'tokens': fit_a['tokens'], 'targets': ['action']} and fit_a['presentations'] >= 8
    assert checked['coverage']['tasks']['pilot:held_d']['role'] == 'evaluation' and checked['coverage']['roles']['adaptation']['complete_examples'] == 8 + 6
    tampered = tmp_path / 'tampered.json'
    files.publish(tampered, {**files.read_json(good), 'claim': {'statement': 'changed', 'holdout': []}})
    with pytest.raises(ConfigError, match='identity mismatch'):
        validate(tampered)
    aggregated = manifest_task('adaptation', paths['fit_a'], [exports['fit_a']], *contracts['fit_a'], weight=1)
    aggregated['sources'][0]['underlying'] = [{**aggregated['sources'][0], 'dataset': 'inner', 'permits': ['evaluation'], 'underlying': []}]
    cases = {'not permit adaptation': mixture(**{'pilot:fit_a': aggregated}),
             'cannot be held out': mixture(**{'pilot:held_d': {'sources': manifest_task('evaluation', paths['fit_a'], [], *contracts['fit_a'])['sources']}}),
             'quota': mixture(**{'pilot:fit_a': {'quota': 7}}),
             'positive objective weight': mixture(**{'pilot:select_c': {'weight': 2}}),
             'freezes to task contract': mixture(**{'pilot:fit_b': {'contract': 'f' * 64}}),
             'freeze to one task contract': mixture(**{'pilot:fit_b': {'contract': contracts['fit_a'][0], 'revisions': contracts['fit_a'][1]}}),
             'one role': mixture(**{'pilot:fit_b': {'exports': [{'path': Path(mixed).name, 'export_id': read_export(mixed, native=True)[0]['export_id']}]}}),
             'maps source labels': mixture(**{'pilot:fit_b': {'mapping': {'action': {'CAL': 'stop'}}}})}
    for match, broken in cases.items():
        with pytest.raises(ConfigError, match=match):
            validate(published(tmp_path / 'broken.json', broken))
    prepared = prepare_mixture(good, tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    from semantic_trainer.jobs import resolve
    resolved, observations = resolve(prepared['job'])
    assert set(resolved['tasks']) == {contracts[name][0] for name in ('fit_a', 'fit_b', 'select_c')} and resolved['objective'] == 'instruction_transfer'
    assert prepared['mixture']['id'] == checked['id'] and not any(row['task'] == contracts['held_d'][0] for row in observations)
    assert {resolved['objective_weights']['tasks'][contracts[name][0]]['fit']['weight'] for name in ('fit_a', 'fit_b')} == {.5}


@pytest.mark.parametrize('microbatch', [1, 2, 5])
def test_objective_weighted_updates_reproduce_the_float64_loss_and_gradients(microbatch):
    torch = pytest.importorskip('torch')
    from semantic_trainer.jobs import presentation_weights
    from semantic_trainer.objective import accumulate, answer_logits, bind, prepared
    from test_lora import adapted_tiny, fit_rows, sequences
    rows = [{**row, 'objective': value} for row, value in zip(fit_rows(), (.625, 2.5, .125))]
    model = adapted_tiny()
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = 0
    for row in rows:
        for entry, weight in zip(row['presentations'], presentation_weights(row)):
            z = answer_logits(bind(model), sequences([entry]), 0)[0].double()
            gold = list(entry.answers.slots[entry.label])
            expected = expected + (torch.logsumexp(z, 0) - torch.logsumexp(z[gold], 0)) * row['objective'] * weight / len(rows)
    expected.backward()
    wanted = [p.grad.clone() for p in trainable]
    model.zero_grad(set_to_none=True)
    stats = accumulate(bind(model), rows[::-1], prepared(rows, bind(model).vocabulary), 0, microbatch)
    assert float(stats['loss']) == pytest.approx(float(expected.detach()), rel=1e-5)
    assert all(torch.allclose(p.grad, r, atol=1e-6, rtol=1e-4) for p, r in zip(trainable, wanted))


def test_monitoring_reduces_model_select_nll_with_the_same_objective_weights(tmp_path):
    torch = pytest.importorskip('torch')
    from semantic_trainer.jobs import presentation_weights
    from semantic_trainer.monitor import Monitor
    from semantic_trainer.objective import answer_logits, bind
    from test_lora import adapted_tiny, fit_rows, sequences
    rows = [{**row, 'role': role, 'objective': value, 'situations': [], 'component': row['id'], 'fact': 'action', 'kind': 'pick'}
            for row, role, value in zip(fit_rows(), ('fit', 'model_select', 'model_select'), (1., .5, 1.5))]
    model = adapted_tiny().eval()
    def nll(row):
        total = 0.
        for entry, weight in zip(row['presentations'], presentation_weights(row)):
            with torch.no_grad():
                z = answer_logits(bind(model), sequences([entry]), 0)[0].double()
            total += float(torch.logsumexp(z, 0) - torch.logsumexp(z[list(entry.answers.slots[entry.label])], 0)) * weight
        return total
    resolved = {'fit': {'id': 'fixture'}, 'observations': rows, 'probe_ids': ['a'], 'recipe': {'selection_unit': 'observation', 'microbatch': 1, 'evaluation_batch_size': 1, 'max_padded_tokens': 1024}, 'work': {'point_tokens': 100},
                'budget': {'max_monitor_tokens': 1000, 'max_sequence_tokens': 1024}, 'schedule': {}, 'situations': {}}
    deadline = type('Deadline', (), {'check': lambda self, operation: None, 'elapsed': lambda self: 0., 'expired': lambda self: False})()
    point = Monitor(tmp_path, resolved, model, 0, None, None, deadline).point(0, 0)
    assert point['nll'] == pytest.approx((.5 * nll(rows[1]) + 1.5 * nll(rows[2])) / 2, rel=1e-5)
    assert point['correct']['tasks'] == {'t': {**point['correct']['tasks']['t'], 'observations': 2}}


def test_warm_start_loads_the_verified_adapter_before_the_first_update_and_publishes_one_delta(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import shutil
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    from semantic_trainer import jobs, lora, objective
    from test_lora import fake_native, monitored_job, records
    recipe = {'epochs': 1, 'eval_every_examples': 8, 'min_epochs': 1, 'patience_examples': 1000}
    from semantic_switch import api
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe=recipe)
    fake_native(monkeypatch)
    base = files.sha256_file(jobs.resolve(job)[0]['model']['gguf'])
    monkeypatch.setattr(api, '_local_identity', lambda spec, *args: {'sha256': base, 'template': 'fixture', 'prompt': 'p', 'presentation': {'revision': 1}})
    first = Path(lora.train(job, tmp_path / 'first')['path'])
    listed = files.read_json(first / 'candidates.json')
    chosen = listed['candidates'][1]
    before = {path: files.sha256_file(path) for path in first.rglob('*') if path.is_file()}
    declared = files.read_yaml(job)
    (contract, task), = declared['tasks'].items()
    def warm(to, run=first, candidate=chosen['name'], **changes):
        return jobs.prepare(task['source'], [entry['path'] for entry in declared['exports']], tmp_path / to, assets_manifest=declared['assets']['manifest'],
                            recipe={'batch_size': 8, **recipe, **changes}, initialization={'run': str(run), 'candidate': candidate})
    fresh, _ = jobs.resolve(job)
    resolved, observations = jobs.resolve(warm('warm')['job'])
    initialization = resolved['initialization']
    assert initialization['kind'] == 'adapter' and initialization['peft_sha256'] == chosen['peft_sha256'] and initialization['job_id'] == fresh['id']
    assert initialization['null']['peft_sha256'] == listed['candidates'][0]['peft_sha256']
    assert [field for field, value in resolved['fit']['fields'].items() if fresh['fit']['fields'][field] != value] == ['initialization']
    assert {row['id']: row['uses'] for row in initialization['inherited']} == {row['id']: [jobs.INHERITED[row['role']]] for row in fresh['memberships'] if row['role'] != 'unused'}
    assert {row['inherited'] for row in initialization['inherited']} == {fresh['id']}

    _, model, _, _, matched = lora.adapted(resolved, AutoConfig.from_pretrained(resolved['model']['hf']))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert list(model.peft_config) == ['default'] and trainable and all('lora_' in name and '.default.' in name for name in trainable)
    pristine = dict(AutoModelForCausalLM.from_pretrained(resolved['model']['hf']).named_parameters())
    frozen = {name.replace('base_model.model.', '').replace('.base_layer', ''): parameter for name, parameter in model.named_parameters() if 'lora_' not in name}
    assert frozen.keys() == pristine.keys() and all(torch.equal(parameter, pristine[name]) for name, parameter in frozen.items())

    second = Path(lora.train(warm('warm-run')['job'], tmp_path / 'second')['path'])
    initial, zero = (load_file(path / 'adapter_model.safetensors') for path in (first / chosen['peft'], second / 'candidates' / 'zero'))
    assert initial.keys() == zero.keys() and all(torch.equal(initial[key], zero[key]) for key in initial)
    points = [{p['step']: p for p in records(run / 'progress.jsonl') if p['kind'] == 'evaluation'} for run in (first, second)]
    assert points[1][0]['model_select']['nll'] == pytest.approx(points[0][chosen['step']]['model_select']['nll'], rel=1e-5)
    assert {path: files.sha256_file(path) for path in first.rglob('*') if path.is_file()} == before
    listed2 = files.read_json(second / 'candidates.json')
    best = listed2['candidates'][1]
    delta = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(resolved['model']['hf']), str(second / best['peft'])).eval()
    selecting = [row for row in observations if row['role'] == 'model_select']
    binding = objective.bind(delta)
    results = objective.evaluate(binding, selecting, objective.prepared(selecting, binding.vocabulary), 0, 1024, lambda: None)
    assert sum(row['objective'] * results[row['id']]['nll'] for row in selecting) / len(selecting) == pytest.approx(points[1][best['step']]['model_select']['nll'], rel=1e-4)
    assert {key: value['predicted'] for key, value in results.items()} == {key: value['answer'] for key, value in best['predictions'].items()}

    with pytest.raises(ConfigError, match='differs from this job in recipe'):
        warm('resized', rank=8)
    with pytest.raises(ConfigError, match='no retained candidate'):
        warm('missing', candidate='step-999')
    for name, edit, match in (('tensors', lambda peft: (peft / 'adapter_model.safetensors').write_bytes((peft / 'adapter_model.safetensors').read_bytes()[:-1] + b'\0'), 'drift'),
                              ('config', lambda peft: files.publish(peft / 'adapter_config.json', {**files.read_json(peft / 'adapter_config.json'), 'lora_alpha': 999}), 'drift')):
        copy = tmp_path / f'corrupt-{name}'
        shutil.copytree(first, copy)
        for path in copy.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o644)
        edit(copy / chosen['peft'])
        with pytest.raises(ConfigError, match=match):
            warm(f'warm-{name}', run=copy)
    with pytest.raises(ConfigError, match='files'):
        jobs.compatible(initialization, {**resolved['model'], 'files': {**resolved['model']['files'], 'config.json': 'f' * 64}}, resolved['recipe'],
                        initialization['base']['sha256'])
    loaded = jobs.load([entry['path'] for entry in declared['exports']], contract)
    trained_selection = [{**row, 'uses': ['adaptation']} for row in initialization['inherited'] if row['uses'] == ['selection']]
    rows, families = jobs.cohort(loaded, trained_selection)
    jobs.assign(rows, jobs.intended(rows, False))
    with pytest.raises(ConfigError, match='trained the initializer adapter'):
        jobs.separated(rows, families)


def test_warm_started_selection_publishes_inherited_model_wide_lineage_that_disqualifies_overlapping_checks(tmp_path, monkeypatch):
    from semantic_switch.artifacts import load_receipt
    from semantic_switch.results import thaw
    from semantic_switch.tools import deployment as checking
    from semantic_switch.tools.common import input_key
    from semantic_trainer.jobs import prepare, resolve
    from test_specialize import RECIPE as SPECIALIZED, assigned, check_labels, fitted_run, selection
    module, job, _, _, _, deployed, path = selection(tmp_path, monkeypatch)
    resolved, observations = resolve(job)
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    labels = check_labels(tmp_path)
    template = {key: value for key, value in resolved['memberships'][0].items() if key != 'member'}
    ghost = {**template, 'id': 'e' * 64, 'fact': 'ghost', 'projection': 'ghost', 'role': 'fit', 'component': 'd' * 64, 'group': 'd' * 64, 'situations': [],
             'family_seeds': [['key', input_key(plan, {'message': 'report 1000'})]]}
    base = files.sha256_file(resolved['model']['gguf'])
    broad = fitted_run(tmp_path / 'broad', job, {**resolved, 'memberships': [*resolved['memberships'], ghost]}, observations, base)
    warm = prepare(path, [resolved['sources'][0]['path']], tmp_path / 'warm', assets_manifest=resolved['assets']['manifest'], recipe=SPECIALIZED,
                   native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048}, initialization={'run': str(broad), 'candidate': 'step-8'})['job']
    warmed, warm_rows = resolve(warm)
    run = fitted_run(tmp_path / 'warm-fit', warm, warmed, warm_rows, base)
    deployed.clear()
    result = module.select(warm, run)
    assert result['selected'] == 'step-8' and {'zero', 'step-8', 'step-16'} <= set(result['candidates']) and deployed[0][0] == 'zero'
    data = thaw(load_receipt(result['receipt']).data)
    rows = {row['id']: row for row in data['provenance']['rows']}
    assert rows['e' * 64]['uses'] == ['adaptation'] and rows['e' * 64]['inherited'] == resolved['id'] and data['job_id'] == warmed['id']
    assert all('inherited' not in row for key, row in rows.items() if key != 'e' * 64)
    assert data['initialization'] == {'kind': 'adapter', 'evidence': {'kind': 'named'}, 'candidate': 'step-8',
                                      **{key: warmed['initialization'][key] for key in ('step', 'job_id', 'fit_id', 'fitted_job_id', 'peft_sha256', 'config_sha256')}}
    assert data['initialization']['job_id'] == resolved['id']
    labelled, evidence = assigned(tmp_path / 'evidence', path, files.read_jsonl(labels))
    report = checking.check_deployment(result['receipt'], labelled, evidence, tmp_path / 'check')
    assert report['lineage']['targets']['judge:action']['state'] == 'known_overlap'



def partitioned(tmp_path, path, name, family, indices, role, file, partial=lambda i: False):
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    rows = []
    for i in indices:
        facts = {'action': ('calibrate', 'resume')[i % 2], **({'urgent': i % 4 < 2} if 'urgent' in plan.spec['facts'] and not partial(i) else {})}
        rows.append(label_record(plan, {'message': f'{name} report {i}'}, facts, sources=each_target(facts, SOURCE), groups=[f'{name}-{i}'], role=role))
    labels = tmp_path / f'{file}.jsonl'
    files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(SHARED) as runtime:
        return tools.export(runtime, path, to=tmp_path / f'{file}.export.jsonl', native=True, labels=labels, task_family=family, split='tune')['path']


def pilot(tmp_path, monkeypatch):
    manifest = pipeline_base(tmp_path, monkeypatch)
    paths = {'fit_a': definition(tmp_path, 'fit_a', 'Which action does fit_a require?', urgent=True),
             'select_c': definition(tmp_path, 'select_c', 'Which action does select_c require?'),
             'held_d': definition(tmp_path, 'held_d', 'Which action does held_d require?')}
    contracts = {name: frozen_contract(tmp_path, path) for name, path in paths.items()}
    select = exported(tmp_path, paths['select_c'], 'select_c', 'pilot:select_c', 4, lambda i: 'model_select')
    held = exported(tmp_path, paths['held_d'], 'held_d', 'pilot:held_d', 4, lambda i: 'check', split='all')
    def mixture(fit_exports, quota=500, claim='decision skill transfers to held-out manuals', weight=1):
        tasks = {'pilot:fit_a': {**manifest_task('adaptation', paths['fit_a'], fit_exports, *contracts['fit_a'], weight=weight), 'quota': quota},
                 'pilot:select_c': manifest_task('model_select', paths['select_c'], [select], *contracts['select_c']),
                 'pilot:held_d': manifest_task('evaluation', paths['held_d'], [held], *contracts['held_d'])}
        return {'schema': 1, 'kind': 'task-mixture', 'claim': {'statement': claim, 'holdout': ['document']}, 'tasks': tasks}
    return manifest, paths, contracts, held, mixture


def test_mixture_quotas_count_the_canonical_union_of_complete_examples_across_exports(tmp_path, monkeypatch):
    from semantic_trainer import mixture as module
    manifest, paths, contracts, _, mixture = pilot(tmp_path, monkeypatch)
    first = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(0, 3), 'fit', 'first')
    second = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(3, 6), 'fit', 'second')
    overlapping = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(2, 6), 'fit', 'overlapping')
    whole = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(0, 6), 'fit', 'whole')
    with pytest.raises(ConfigError, match=r'pilot:fit_a 6 > 4'):
        module.validate(published(tmp_path / 'split.json', mixture([first, second], quota=4)))
    boundary = module.validate(published(tmp_path / 'boundary.json', mixture([first, second], quota=6)))['coverage']
    union = module.validate(published(tmp_path / 'union.json', mixture([overlapping, first], quota=6)))['coverage']
    single = module.validate(published(tmp_path / 'single.json', mixture([whole], quota=6)))['coverage']
    assert boundary['tasks']['pilot:fit_a']['complete_examples'] == 6 and boundary['tasks']['pilot:fit_a']['observations'] == 12
    assert boundary == union == single and boundary['roles']['adaptation']['tasks'] == 1
    reordered = module.validate(published(tmp_path / 'reordered.json', mixture([second, first], quota=6)))['coverage']
    assert reordered == boundary
    with pytest.raises(ConfigError, match='declare each immutable export once'):
        module.validate(published(tmp_path / 'repeated.json', mixture([first, first], quota=6)))
    partial = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(0, 6), 'fit', 'partial', partial=lambda i: i >= 4)
    counted = module.validate(published(tmp_path / 'partial.json', mixture([partial], quota=4)))['coverage']['tasks']['pilot:fit_a']
    assert (counted['complete_examples'], counted['incomplete_examples'], counted['observations']) == (4, 2, 10)
    monkeypatch.setattr(module, 'MAX_ADAPTATION', 5)
    with pytest.raises(ConfigError, match='exceed the pilot cap 5'):
        module.validate(published(tmp_path / 'capped.json', mixture([whole], quota=6)))


def test_prepared_mixture_bundles_verified_permission_evidence_and_separates_audit_from_numerical_identity(tmp_path, monkeypatch):
    import shutil
    from semantic_trainer.jobs import resolve
    from semantic_trainer.mixture import prepare_mixture
    manifest, paths, contracts, held, mixture = pilot(tmp_path, monkeypatch)
    fit = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(0, 6), 'fit', 'whole')
    source = published(tmp_path / 'mixture.json', mixture([fit]))
    prepared = prepare_mixture(source, tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    resolved, observations = resolve(prepared['job'])
    licence_sha = files.sha256_file(tmp_path / 'LICENSE')
    bundle = tmp_path / 'job' / 'mixture'
    assert files.read_json(bundle / 'manifest.json') == files.read_json(source) and files.sha256_file(bundle / 'evidence' / licence_sha) == licence_sha
    audit = files.read_json(bundle / 'audit.json')
    assert resolved['mixture'] == {'audit': audit['id'], 'mixture': files.read_json(source)['id'], 'claim': files.read_json(source)['claim'],
                                   'evaluation': {'pilot:held_d': {'contract': contracts['held_d'][0], 'exports': [read_export(held, fitting=False, native=True)[0]['export_id']]}}}
    assert audit['coverage']['tasks']['pilot:held_d']['observations'] == 4 and audit['locators']['manifest'] == str(source)
    assert not any(row['task'] == contracts['held_d'][0] for row in observations) and prepared['mixture']['audit'] == audit['id']
    (tmp_path / 'LICENSE').unlink()
    source.unlink()
    moved = tmp_path / 'moved'
    shutil.copytree(tmp_path / 'job', moved)
    relocated, _ = resolve(moved / 'job.yaml')
    assert (relocated['id'], relocated['fit']['id'], relocated['monitor']['id'], relocated['mixture']) == (resolved['id'], resolved['fit']['id'], resolved['monitor']['id'],
                                                                                                    resolved['mixture'])
    (moved / 'mixture' / 'evidence' / licence_sha).write_text('edited permission\n')
    with pytest.raises(ConfigError, match='missing or altered'):
        resolve(moved / 'job.yaml')
    files.publish(moved / 'mixture' / 'audit.json', {**audit, 'claim': {'statement': 'laundered', 'holdout': []}})
    with pytest.raises(ConfigError, match='audit changed'):
        resolve(moved / 'job.yaml')
    licence(tmp_path)
    claimed = prepare_mixture(published(tmp_path / 'claimed.json', mixture([fit], claim='a narrower held-out manual claim')), tmp_path / 'claimed',
                              assets_manifest=manifest, recipe=RECIPE)
    audited, _ = resolve(claimed['job'])
    assert (audited['fit']['id'], audited['monitor']['id']) == (resolved['fit']['id'], resolved['monitor']['id'])
    assert audited['id'] != resolved['id'] and audited['mixture']['audit'] != resolved['mixture']['audit']
    smaller = partitioned(tmp_path, paths['fit_a'], 'fit_a', 'pilot:fit_a', range(0, 4), 'fit', 'smaller')
    numerical = prepare_mixture(published(tmp_path / 'smaller.json', mixture([smaller])), tmp_path / 'smaller', assets_manifest=manifest, recipe=RECIPE)
    assert resolve(numerical['job'])[0]['fit']['id'] != resolved['fit']['id']
    edited = tmp_path / 'claimed' / 'mixture' / 'manifest.json'
    files.publish(edited, {**files.read_json(edited), 'claim': {'statement': 'edited', 'holdout': []}})
    with pytest.raises(ConfigError, match='identity mismatch'):
        resolve(claimed['job'])


DECISIONS = {'escalate': 'Request a technician for a current unresolved incident', 'continue': 'Continue operation when the incident is resolved',
             'clarify': 'Ask whether the incident remains unresolved'}


class Ranked(FakeScorer):
    table, realized, built = {}, None, []

    def __init__(self, path, **kwargs):
        from types import SimpleNamespace
        from semantic_testkit.scoring import letters_of
        content = Path(path).read_bytes()
        name = content.decode().split(' ', 1)[1] if content.startswith(b'merged ') else 'base'
        def oracle(state, question, width):
            import re
            n = int(re.search(r'report (\d+)', state).group(1))
            task = re.search(r'manual (\w+)', question).group(1)
            if name == 'base':
                return [1 / width] * width
            options = letters_of(question)
            available = [key for key, text in DECISIONS.items() if text in options]
            wanted = available[(n + (not Ranked.table[name][task](n))) % len(available)]
            return [0.98 if chr(65 + k) == options[DECISIONS[wanted]] else 0.02 / max(width - 1, 1) for k in range(width)]
        super().__init__(oracle)
        self.vocab, self.inv = SHARED.vocab, SHARED.inv
        self.path, self.shared, self.weights, self.closed = path, True, SimpleNamespace(hybrid=False), 0
        Ranked.built.append(self)

    def capacity(self):
        from semantic_testkit.candidates import REALIZED
        return dict(Ranked.realized(self.path) if Ranked.realized else REALIZED)

    def close(self):
        self.closed += 1


def decision(tmp_path, name, options, binding=None):
    path = tmp_path / f'{name}.yaml'
    files.publish(path, {'name': name, 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'action': {'pick': f'Which response does manual {name} require?', 'options': {key: DECISIONS[key] for key in options},
                                              **({'binding': binding} if binding else {})}}})
    return path


def decided(tmp_path, path, name, indices, role, split='tune'):
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    keys = list(plan.spec['facts']['action']['options'])
    rows = [label_record(plan, {'message': f'report {i}'}, {'action': keys[i % len(keys)]},
                         sources=each_target({'action': None}, SOURCE), groups=[f'{name}-{i}'], role=role(i) if callable(role) else role) for i in indices]
    labels = tmp_path / f'{name}.jsonl'
    files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(bindings={'judge': SHARED, 'expert': SHARED}) as runtime:
        return tools.export(runtime, path, to=tmp_path / f'{name}.export.jsonl', native=True, labels=labels, task_family=f'pilot:{name}', split=split)['path']


def broad(tmp_path, monkeypatch):
    from semantic_switch import api, backend
    from semantic_trainer import representation, select as module
    from semantic_trainer.jobs import resolve
    from semantic_trainer.mixture import prepare_mixture
    from test_specialize import fitted_run
    identity = api.Runtime.identity
    manifest = pipeline_base(tmp_path, monkeypatch)
    paths = {'A': decision(tmp_path, 'A', DECISIONS), 'S1': decision(tmp_path, 'S1', DECISIONS, binding='expert'),
             'S2': decision(tmp_path, 'S2', ('escalate', 'continue')), 'H': decision(tmp_path, 'H', DECISIONS)}
    exports = {'A': decided(tmp_path, paths['A'], 'A', range(0, 12), 'fit'), 'S1': decided(tmp_path, paths['S1'], 'S1', range(100, 109), 'model_select'),
               'S2': decided(tmp_path, paths['S2'], 'S2', range(200, 203), 'model_select'), 'H': decided(tmp_path, paths['H'], 'H', range(300, 306), 'check', split='all')}
    roles = {'A': 'adaptation', 'S1': 'model_select', 'S2': 'model_select', 'H': 'evaluation'}
    tasks = {f'pilot:{name}': manifest_task(role, paths[name], [exports[name]], *frozen_contract(tmp_path, paths[name]), weight=1 if role == 'adaptation' else None)
             for name, role in roles.items()}
    for name, task in tasks.items():
        task['mapping'] = {'action': {'E': 'escalate'}}
    source = published(tmp_path / 'mixture.json', {'schema': 1, 'kind': 'task-mixture', 'claim': {'statement': 'decisions transfer', 'holdout': ['document']}, 'tasks': tasks})
    job = prepare_mixture(source, tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048})['job']
    target = decision(tmp_path, 'T', DECISIONS)
    exports['T'] = (target, decided(tmp_path, target, 'T', range(400, 412), lambda i: 'model_select' if i % 4 == 3 else 'fit'))
    monkeypatch.setattr(api.Runtime, 'identity', identity)
    resolved, observations = resolve(job)
    run = fitted_run(tmp_path, job, resolved, observations, files.sha256_file(resolved['model']['gguf']))
    deployed = []
    def deploy(fitted, located, peft, destination, quant, handle):
        deployed.append(Path(peft).name)
        Path(destination).write_bytes(b'merged ' + Path(peft).name.encode())
        return files.sha256_file(destination)
    monkeypatch.setattr(representation, 'deploy', deploy)
    monkeypatch.setattr(module, 'verify_zero', lambda fitted, located, peft, quant, work, handle: deployed.append('zero'))
    monkeypatch.setattr(backend, 'LlamaScorer', Ranked)
    monkeypatch.setattr(api.Model, 'resources', lambda self: {'weights': Resources(1)})
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    monkeypatch.setattr(module, 'conversion_bytes', lambda located: {'conversion': 1})
    Ranked.built, Ranked.realized = [], None
    Ranked.table = {'step-8': {'S1': lambda n: True, 'S2': lambda n: False}, 'step-16': {'S1': lambda n: n % 9 < 5, 'S2': lambda n: True}}
    return module, job, run, resolved, observations, deployed, paths, exports, source


def test_initializer_selection_ranks_each_held_out_task_under_its_own_definition_and_publishes_no_lock(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare, resolve
    module, job, run, resolved, observations, deployed, paths, exports, source = broad(tmp_path, monkeypatch)
    contracts = {name: next(c for c, task in resolved['tasks'].items() if Path(task['source']).stem == name) for name in ('A', 'S1', 'S2')}
    assert resolved['scope'][contracts['S1']]['binding'] == 'expert' and resolved['scope'][contracts['S2']]['binding'] == 'judge'
    with pytest.raises(ConfigError, match='its candidates are initializers'):
        module.select(job, run)
    held = [item for item in tmp_path.iterdir() if item.name.startswith('H.') or item.name.startswith(Path(exports['H']).name.split('.')[0] + '.')]
    assert held
    for item in held:
        item.unlink()
    before = {item: files.sha256_file(item) for item in run.rglob('*') if item.is_file()}
    result = module.select_initializer(job, run)
    assert deployed == ['zero', 'step-8', 'step-16'] and result['selected'] == 'step-16'
    candidates = result['candidates']
    assert (candidates['step-8']['raw_correct'], candidates['step-16']['raw_correct']) == (9, 5 + 3)
    assert candidates['step-8']['correct']['macro']['observation'] == (1 + 0) / 2 and candidates['step-16']['correct']['macro']['observation'] == (5 / 9 + 1) / 2
    assert candidates['step-16']['correct']['tasks'][contracts['S1']]['observation'] == 5 and candidates['step-16']['correct']['tasks'][contracts['S2']]['observations'] == 3
    record = files.read_json(result['record'])
    assert record['id'] == artifact_digest({key: value for key, value in record.items() if key != 'id'}) and record['kind'] == 'initializer-selection'
    assert record['selected'] == {'name': 'step-16', 'step': 16, **{key: next(c for c in files.read_json(run / 'candidates.json')['candidates'] if c['name'] == 'step-16')[key]
                                                                     for key in ('peft', 'peft_sha256', 'config_sha256', 'config_digest')}}
    assert set(record['plan']['tasks']) == {contracts['S1'], contracts['S2']} and record['plan']['tasks'][contracts['S1']]['observations'] == 9
    assert record['mixture'] == resolved['mixture'] == record['fitted_mixture'] and record['representation'] == 'merged'
    assert record['native']['sha256'] == hashlib.sha256(b'merged step-16').hexdigest() and record['native']['adapter'] is None
    trial = next(t for t in record['candidates'] if t['name'] == 'step-16')
    assert set(trial['hf_agreement']['tasks']) == {contracts['S1'], contracts['S2']} and set(trial['selection_work']['tasks']) == {contracts['S1'], contracts['S2']}
    assert any(key.startswith(f"{contracts['S2']}:") for key in trial['per_class']) and trial['probabilities']['tasks']
    roles = {row['id']: row['role'] for row in observations}
    assert {row['id']: row['uses'] for row in record['memberships']} == {i: [{'fit': 'adaptation', 'model_select': 'selection'}[role]] for i, role in roles.items()}
    assert not [item for item in run.rglob('*') if item.name in ('task.lock.json', 'receipt.json')] and not (run / 'deployment').exists()
    assert {item: files.sha256_file(item) for item in before} == before and not [item for item in run.iterdir() if item.name.startswith('.')]
    assert all(scorer.closed == 1 for scorer in Ranked.built)

    target, target_export = exports['T']
    warm = prepare(target, [target_export], tmp_path / 'warm', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE,
                   initialization={'selection': result['record']})['job']
    warmed, warm_rows = resolve(warm)
    assert warmed['initialization']['evidence'] == {'kind': 'native-selection', 'record': record['id'], 'mixture': resolved['mixture']}
    assert warmed['initialization']['candidate'] == 'step-16' and warmed['initialization']['inherited'] == record['memberships']
    named, _ = resolve(prepare(target, [target_export], tmp_path / 'named', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE,
                               initialization={'run': str(run), 'candidate': 'step-16'})['job'])
    assert named['initialization']['evidence'] == {'kind': 'named'}
    assert named['fit']['id'] == warmed['fit']['id'] and named['id'] != warmed['id']
    forged = tmp_path / 'forged.json'
    body = {**{key: value for key, value in record.items() if key != 'id'}, 'selected': {**record['selected'], 'name': 'step-8', 'step': 8}}
    files.publish(forged, {**body, 'id': artifact_digest(body)})
    with pytest.raises(ConfigError, match='does not describe candidate step-8'):
        prepare(target, [target_export], tmp_path / 'forged-job', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE, initialization={'selection': str(forged)})
    files.publish(forged, {**record, 'fit_stop': {'reason': 'edited'}})
    with pytest.raises(ConfigError, match='not an intact initializer-selection record'):
        prepare(target, [target_export], tmp_path / 'edited-job', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE, initialization={'selection': str(forged)})
    with pytest.raises(ConfigError, match='already selected'):
        module.select_initializer(job, run)


def test_initializer_selection_reports_base_victory_and_refuses_bad_proof_execution_drift_and_interruption(tmp_path, monkeypatch):
    import shutil
    from semantic_trainer.jobs import prepare, resolve
    from semantic_trainer.mixture import prepare_mixture
    module, job, run, resolved, observations, deployed, paths, exports, source = broad(tmp_path, monkeypatch)
    from semantic_trainer import assets
    located = assets.load(resolved['assets']['manifest'])
    base = located['gguf'][resolved['assets']['quant']]
    licence(tmp_path)
    claimed = files.read_json(source)
    body = {**{key: value for key, value in claimed.items() if key != 'id'}, 'claim': {'statement': 'a narrower claim', 'holdout': ['document']}}
    audited_job = prepare_mixture(published(tmp_path / 'claimed.json', body), tmp_path / 'claimed', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE,
                                  native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048})['job']
    audited, _ = resolve(audited_job)
    assert audited['id'] != resolved['id'] and audited['fit']['id'] == resolved['fit']['id']
    with pytest.raises(ConfigError, match='changed since'):
        module.admitted(audited_job, run, audited, base)
    assert module.admitted(audited_job, run, audited, base, audited=('mixture',))[0]['id'] == resolved['id']

    class Cancelled(BaseException):
        pass
    publish = module.files.publish_directory
    def interrupted(staged, output):
        raise Cancelled()
    monkeypatch.setattr(module.files, 'publish_directory', interrupted)
    with pytest.raises(Cancelled):
        module.select_initializer(job, run)
    assert not (run / 'initializer-selection').exists() and not [item for item in run.iterdir() if item.name.startswith('.')]
    monkeypatch.setattr(module.files, 'publish_directory', publish)

    from semantic_testkit.candidates import REALIZED
    Ranked.realized = lambda path: {**REALIZED, 'n_batch': 64 if 'step-16' in str(path) else 512}
    with pytest.raises(ConfigError, match='different'):
        module.select_initializer(job, run)
    Ranked.realized = None

    config = run / 'candidates' / 'step-8' / 'adapter_config.json'
    original = config.read_text()
    config.write_text(json.dumps({**json.loads(original), 'lora_alpha': 999}))
    deployed.clear()
    with pytest.raises(ConfigError, match='drift'):
        module.select_initializer(job, run)
    assert not deployed
    config.write_text(original)

    Ranked.table = {'step-8': {'S1': lambda n: False, 'S2': lambda n: False}, 'step-16': {'S1': lambda n: False, 'S2': lambda n: False}}
    result = module.select_initializer(job, run)
    assert result['selected'] == 'base' and result['warm_start'] is None and files.read_json(result['record'])['selected'] == {'name': 'base', 'step': 0}
    target, target_export = exports['T']
    with pytest.raises(ConfigError, match='chose the unadapted base'):
        prepare(target, [target_export], tmp_path / 'warm', assets_manifest=resolved['assets']['manifest'], recipe=RECIPE,
                initialization={'selection': result['record']})
