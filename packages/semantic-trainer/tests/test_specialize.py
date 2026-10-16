
from semantic_switch import model_jobs
from contextlib import ExitStack
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, Runtime, api, files, tools
from semantic_switch.api import execution_identity
from semantic_switch.artifacts import artifact_digest, load_receipt, read_export
from semantic_switch.evidence import role_of
from semantic_switch.results import thaw
from semantic_switch.tools.dataset import EvidenceContext
from semantic_testkit.assets import fake_assets, pipeline_base
from semantic_testkit.candidates import ACTION_TEXT, REALIZED, VALUES, Native
from semantic_testkit.labels import TEACHER, action_export, declared, label_file
from semantic_testkit.scoring import FakeScorer, letters_of

RECIPE = {'eval_every_examples': 8}


def task(tmp_path, monkeypatch, values=VALUES, n=40, author=None):
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, values, n=n)
    Native.tokens = FakeScorer()
    if author is not None:
        author(tmp_path / 'task' / 'action.yaml')
    with Runtime(Native.tokens) as runtime:
        exported = action_export(runtime, tmp_path / 'task', records, 'maintenance:test')
    return manifest, tmp_path / 'task' / 'action.yaml', exported['path'], records


def test_prepare_assigns_family_roles_once_and_reports_support(tmp_path, monkeypatch):
    from semantic_trainer.jobs import TUNING, prepare, resolve, single_task
    manifest, definition, export, _ = task(tmp_path, monkeypatch)
    result = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, acceptance_fit=True)
    resolved, observations = resolve(result['job'])
    contract, frozen = single_task(resolved)
    assert result['job_id'] == resolved['id'] and frozen['revisions'] == {'action': observations[0]['meaning_rev']} and set(result['tasks']) == {contract}
    assert files.read_yaml(result['job'])['tasks'] == {contract: {'source': str(definition), 'bundle': f'tasks/{contract}', 'weight': 1}}
    assert {row['task'] for row in observations} == {contract} and files.read_yaml(result['job'])['initialization'] == 'fresh'
    assert resolved['execution'] == observations[0]['native']['execution'] and {row['task_family'] for row in observations} == {'maintenance:test'}
    _, rows = read_export(export, native=True)
    for role in TUNING:
        assert result['roles'][role]['observations'] == sum(role_of(row['split'], row['fold']) == role for row in rows)
        assert result['roles'][role]['situations'] == result['roles'][role]['observations'] == result['roles'][role]['complete_situations']
    assert {row['id'] for row in observations if row['role'] == 'fit'} == {row['id'] for row in rows if role_of(row['split'], row['fold']) == 'fit'}
    fit = result['support']['roles']['fit'][contract]['action']
    assert set(fit) == set(VALUES) and all(set(cell['tags']) <= {'relayed', 'quoted'} for cell in fit.values())
    assert sum(cell['observations'] for cell in fit.values()) == result['roles']['fit']['observations']
    assert all(0 < cell['families'] <= cell['observations'] for cell in fit.values())
    assert sum(cell['objective'] for cell in fit.values()) == pytest.approx(1.)
    assert all(len(row['sources']) == 1 and row['sources'][0]['task_family'] == 'maintenance:test' for row in observations)
    assert result['commands']['select'] == ['semantic-trainer', 'select', result['job'], '--run', str(tmp_path / 'job' / 'run')]
    with pytest.raises(ConfigError, match='already assigned'):
        prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    definition.write_text(definition.read_text().replace('Which response does the policy require?', 'Which response does the changed policy require?'))
    assert resolve(result['job'])[0]['id'] == resolved['id']
    with pytest.raises(ConfigError, match='changed or left the task definition'):
        prepare(definition, [export], tmp_path / 'changed', assets_manifest=manifest, recipe=RECIPE)
    assert not (tmp_path / 'changed' / 'task').exists() and not (tmp_path / 'changed' / 'job.yaml').exists()
    frozen = tmp_path / 'job' / 'tasks' / contract / 'task.yaml'
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + '\n')
    with pytest.raises(ConfigError, match='frozen task changed since preparation'):
        resolve(result['job'])


def test_prepare_reports_a_declared_answer_without_fit_support(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare
    manifest, definition, export, _ = task(tmp_path, monkeypatch, values=VALUES[:2])
    result = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    assert list(result['support']['unsupported_classes'].values()) == [{'action': ['clarify']}]
    assert (tmp_path / 'job' / 'job.yaml').exists()


@pytest.mark.parametrize('history, match', [
    ([{'derivation': 'judged', 'producer': TEACHER}], 'declares no training use; unknown permission cannot produce training labels'),
    ([{'derivation': 'judged', 'producer': 'openrouter:typesafe/jev-1.13'}, {'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}, 'replaces': 'continue'}],
     r'the judged label from openrouter:typesafe/jev-1.13 in its history cannot label training data.*TypeSafe agreement'),
    ([{'derivation': 'authored', 'producer': 'openrouter:typesafe/jev-1.13', 'actor': {'kind': 'agent', 'model': 'typesafe/jev-1.13', 'session': 's'}}],
     r'the authored label from openrouter:typesafe/jev-1.13 in its history cannot label training data'),
    ([{'derivation': 'judged', 'producer': 'openrouter:qwen/qwen3.6-35b-a3b', 'training_use': declared()}], 'a declaration covers only the producer route it names'),
    ([{'derivation': 'judged', 'producer': TEACHER, 'training_use': declared(permission='unknown')}], 'is declared unknown for training use'),
    ([{'derivation': 'judged', 'producer': TEACHER, 'training_use': declared(permission='forbidden')}], 'is declared forbidden for training use')])
def test_prepare_rechecks_judged_training_provenance_without_semantic_learn(tmp_path, monkeypatch, history, match):
    from semantic_trainer.jobs import prepare
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, n=40, history=history)
    Native.tokens = FakeScorer()
    with Runtime(Native.tokens) as runtime:
        exported = action_export(runtime, tmp_path / 'task', records, 'maintenance:test')
    with pytest.raises(ConfigError, match=match):
        prepare(tmp_path / 'task' / 'action.yaml', [exported['path']], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    assert not (tmp_path / 'job' / 'job.yaml').exists()


def test_provenance_only_changes_admit_training_and_keep_fit_and_monitor_identities(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare, resolve
    from semantic_switch.training import TrainingUse
    manifest = pipeline_base(tmp_path, monkeypatch)
    histories = {'computed': [{'derivation': 'computed', 'producer': 'maintenance-reference', 'revision': 'r1'}],
                 'authored': [{'derivation': 'authored', 'producer': 'codex', 'actor': {'kind': 'agent', 'model': 'gpt-6.1-sol', 'session': 's-1'}}],
                 'judged': [{'derivation': 'judged', 'producer': TrainingUse.parse(declared()).route, 'training_use': declared()}],
                 'relabelled': [{'derivation': 'judged', 'producer': TrainingUse.parse(declared()).route, 'training_use': declared()},
                                {'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}, 'replaces': 'continue'}]}
    identities = {}
    for name, history in histories.items():
        records = label_file(tmp_path / name, n=40, history=history)
        Native.tokens = FakeScorer()
        with Runtime(Native.tokens) as runtime:
            exported = action_export(runtime, tmp_path / name / 'task', records, 'maintenance:test')
        resolved, _ = resolve(prepare(tmp_path / name / 'task' / 'action.yaml', [exported['path']], tmp_path / name / 'job', assets_manifest=manifest, recipe=RECIPE)['job'])
        identities[name] = (resolved['fit']['id'], resolved['monitor']['id'])
    assert len(set(identities.values())) == 1, identities


def test_role_crossing_family_join_fails(tmp_path, monkeypatch):
    from semantic_trainer.jobs import TUNING, prepare
    manifest, definition, export, records = task(tmp_path, monkeypatch)
    rows, roles = files.read_jsonl(records), {row['group']: role_of(row['split'], row['fold']) for row in read_export(export, native=True)[1]}
    index = next(i for i, row in enumerate(rows) if roles.get(f"group:{row['groups'][0]}") in TUNING)
    role = next(role for role in TUNING if role != roles[f"group:{rows[index]['groups'][0]}"])
    rows[index] = {**rows[index], 'groups': ['moved'], 'role': role}
    files.publish(tmp_path / 'other.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(FakeScorer()) as runtime:
        other = tools.export(runtime, definition, to=tmp_path / 'other.export.jsonl', native=True, task_family='maintenance:test', labels=tmp_path / 'other.jsonl')
    with pytest.raises(ConfigError, match='conflicting roles'):
        prepare(definition, [export, other['path']], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)


def test_prepare_takes_explicit_roles_and_never_fits_or_selects_on_check(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare, resolve, role_rows
    manifest, explicit = pipeline_base(tmp_path, monkeypatch), ('fit', 'model_select', 'fit', 'acceptance_fit', 'check')
    records = label_file(tmp_path, n=40)
    files.publish(records, ''.join(json.dumps({**row, 'role': explicit[i % 5]}) + '\n' for i, row in enumerate(files.read_jsonl(records))))
    with Runtime(FakeScorer()) as runtime:
        tuned = action_export(runtime, tmp_path / 'task', records, 'maintenance:test')
        everything = tools.export(runtime, tmp_path / 'task' / 'action.yaml', split='all', to=tmp_path / 'all.jsonl', native=True, task_family='maintenance:test', labels=records)
    _, rows = read_export(everything['path'], fitting=False, native=True)
    given = {f"group:situation-{i}": explicit[i % 5] for i in range(40)}
    assert all(role_of(row['split'], row['fold']) == given[row['group']] for row in rows) and {row['split'] for row in rows} == {'tune', 'check'}
    assert {role for role, _ in role_rows(rows)} == {'fit', 'model_select', 'acceptance_fit'}
    with pytest.raises(ConfigError, match='tune-only export'):
        prepare(tmp_path / 'task' / 'action.yaml', [everything['path']], tmp_path / 'all-job', assets_manifest=manifest, recipe=RECIPE)
    with pytest.raises(TypeError, match='task_family'):
        prepare(tmp_path / 'task' / 'action.yaml', [tuned['path']], tmp_path / 'other-job', assets_manifest=manifest, recipe=RECIPE, task_family='other:override')
    result = prepare(tmp_path / 'task' / 'action.yaml', [tuned['path']], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, acceptance_fit=True)
    assert {role: counts['families'] for role, counts in result['roles'].items()} == {'fit': 16, 'model_select': 8, 'acceptance_fit': 8}
    resolved, observations = resolve(result['job'])
    checked = {row['id'] for row in rows if row['split'] == 'check'}
    assert len(checked) == 8 and not checked & {row['id'] for row in observations}
    assert {row['id'] for row in rows if given[row['group']] == 'fit'} == {row['id'] for row in observations if row['role'] == 'fit'}


def fake_toolchain(monkeypatch, quantized, tensors=None):
    from semantic_trainer import assets
    calls = []
    def run(command, **kwargs):
        command = [str(part) for part in command]
        calls.append(command)
        if command[1].endswith('convert_lora_to_gguf.py'):
            Path(command[command.index('--outfile') + 1]).write_bytes(b'adapter')
        elif command[1] == '-c':
            return json.dumps(tensors or {'blk.0.attn_q.weight.lora_a': {'shape': [2, 4], 'finite': True}, 'blk.0.attn_q.weight.lora_b': {'shape': [8, 2], 'finite': True}})
        elif command[0].endswith('llama-export-lora'):
            Path(command[command.index('-o') + 1]).write_bytes(b'merged-f16')
        elif command[0].endswith('llama-quantize'):
            Path(command[2]).write_bytes(quantized(command))
        return ''
    monkeypatch.setattr(assets, 'run', run)
    return calls


PEFT_CONFIG = {'task_type': 'CAUSAL_LM', 'peft_type': 'LORA', 'auto_mapping': None, 'peft_version': '0.21.2', 'base_model_name_or_path': '/assets/hf/gemma',
               'revision': None, 'inference_mode': True, 'r': 8, 'target_modules': [], 'exclude_modules': None, 'lora_alpha': 16, 'lora_dropout': 0.0,
               'fan_in_fan_out': False, 'bias': 'none', 'use_rslora': False, 'modules_to_save': None, 'init_lora_weights': True, 'layers_to_transform': None,
               'layers_pattern': None, 'rank_pattern': {}, 'alpha_pattern': {}, 'megatron_config': None, 'megatron_core': 'megatron.core',
               'trainable_token_indices': None, 'loftq_config': {}, 'eva_config': None, 'corda_config': None, 'lora_ga_config': None, 'use_dora': False,
               'velora_config': None, 'alora_invocation_tokens': None, 'use_qalora': False, 'qalora_group_size': 16, 'monteclora_config': None,
               'layer_replication': None, 'lora_bias': False, 'target_parameters': None, 'use_bdlora': None, 'arrow_config': None, 'kasa_config': None,
               'ensure_weight_tying': False}


def lora_config(fitted):
    return {**PEFT_CONFIG, 'r': fitted['recipe']['rank'], 'lora_alpha': fitted['recipe']['alpha'], 'lora_dropout': fitted['recipe']['dropout'],
            'target_modules': sorted(fitted['targets'], reverse=True)}


def recorded(fitted):
    from semantic_trainer.adapter import effective
    return {**fitted, 'dependencies': {'peft': '0.21.2'}, 'adapter': effective(lora_config(fitted), '0.21.2')}


FITTED = recorded({'targets': {'model.language_model.layers.0.self_attn.q_proj': [8, 4]}, 'recipe': {'rank': 2, 'alpha': 4, 'dropout': 0.0}, 'model': {'revision': 'c' * 40}})


def safetensors(path, shapes, seed=0):
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = 4 * shape[0] * shape[1]
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, 'little') + encoded + bytes([seed % 256]) * offset)


def peft_adapter(directory, fitted, *, seed=0, rank=None, transposed=False, **config):
    rank = rank or fitted['recipe']['rank']
    directory.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, (out, inner) in fitted['targets'].items():
        shapes[f'base_model.model.{name}.lora_A.weight'] = [inner, rank] if transposed else [rank, inner]
        shapes[f'base_model.model.{name}.lora_B.weight'] = [rank, out] if transposed else [out, rank]
    safetensors(directory / 'adapter_model.safetensors', shapes, seed)
    written = {**lora_config(fitted), **config}
    files.publish(directory / 'adapter_config.json', {key: value for key, value in written.items() if value is not ...})
    return directory


def test_deployment_merges_into_the_f16_base_then_quantizes_and_checks_the_zero_adapter(tmp_path, monkeypatch):
    from semantic_switch.training import model_job
    from semantic_trainer import assets
    from semantic_trainer.deploy import conversion_bytes, verify_zero
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    manifest, base = fake_assets(tmp_path / 'assets')
    located = assets.load(manifest)
    calls = fake_toolchain(monkeypatch, lambda command: b'Q4_0')
    stack = ExitStack()
    job = stack.enter_context(model_job(extra=conversion_bytes(located)))
    peft_adapter(tmp_path / 'peft', FITTED)
    assert verify_zero(FITTED, located, tmp_path / 'peft', 'Q4_0', tmp_path, job) == files.sha256_file(str(base))
    assert job.diagnostics()['resident_peak_bytes']['host'] == Path(located['gguf']['F16']['path']).stat().st_size and job.diagnostics()['participants'] == 0
    convert, inspect, merge, quantize = calls
    assert convert[convert.index('--outtype') + 1] == 'f16' and convert[convert.index('--base') + 1] == located['hf'] and convert[-1] == str(tmp_path / 'peft')
    assert merge[merge.index('-m') + 1] == located['gguf']['F16']['path'] and merge[merge.index('--lora') + 1] == convert[convert.index('--outfile') + 1]
    assert quantize[1] == merge[merge.index('-o') + 1] and quantize[3] == 'Q4_0' and quantize[1].endswith('merged-F16.gguf')
    assert not list((tmp_path / 'assets' / 'tmp').iterdir()) and not (tmp_path / 'zero.gguf').exists()
    fake_toolchain(monkeypatch, lambda command: b'requantized')
    with pytest.raises(ConfigError, match='not the pipeline-built base'):
        verify_zero(FITTED, located, tmp_path / 'peft', 'Q4_0', tmp_path, job)
    fake_toolchain(monkeypatch, lambda command: b'Q4_0', {'blk.0.attn_q.weight.lora_a': {'shape': [2, 4], 'finite': False}})
    with pytest.raises(ConfigError, match='do not match the fitted architecture'):
        verify_zero(FITTED, located, tmp_path / 'peft', 'Q4_0', tmp_path, job)
    embedded = recorded({**FITTED, 'targets': {'model.embed_tokens': [8, 4]}})
    with pytest.raises(ConfigError, match='unsupported conversion target'):
        verify_zero(embedded, located, peft_adapter(tmp_path / 'embedded', embedded), 'Q4_0', tmp_path, job)
    stack.close()


def test_safetensors_descriptor_is_cached_by_file_identity_and_replacement_invalidates_it(tmp_path):
    import os
    from semantic_trainer.adapter import descriptor, tensor_shapes
    path = tmp_path / 'weights.safetensors'
    safetensors(path, {'a': [2, 3]})
    first = descriptor(path)
    assert descriptor(path) is first and tensor_shapes(path) == {'a': [2, 3]} and first.tensors['a'].dtype == 'F32' and first.tensors['a'].size == 24
    replacement = tmp_path / 'replacement.safetensors'
    safetensors(replacement, {'a': [4, 3], 'b': [1, 1]})
    os.replace(replacement, path)
    assert descriptor(path) is not first and tensor_shapes(path) == {'a': [4, 3], 'b': [1, 1]}
    path.write_bytes(b'\xff' * 16)
    with pytest.raises(ConfigError, match='not a readable safetensors file'):
        descriptor(path)


def test_runtime_adapter_zero_proof_reads_lora_factors_without_merging_or_quantizing(tmp_path, monkeypatch):
    from semantic_trainer import assets
    from semantic_trainer.deploy import verify_runtime_zero
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    manifest, _ = fake_assets(tmp_path / 'assets')
    located = assets.load(manifest)
    calls = fake_toolchain(monkeypatch, lambda command: b'Q4_0')
    sha = located['gguf']['Q4_0']['sha256']
    assert verify_runtime_zero(FITTED, located, peft_adapter(tmp_path / 'zero', FITTED, seed=0), 'Q4_0', sha) == sha
    with pytest.raises(ConfigError, match='nonzero lora_B'):
        verify_runtime_zero(FITTED, located, peft_adapter(tmp_path / 'trained', FITTED, seed=3), 'Q4_0', sha)
    with pytest.raises(ConfigError, match='not the pipeline-built'):
        verify_runtime_zero(FITTED, located, tmp_path / 'zero', 'Q4_0', 'f' * 64)
    assert calls == []


def trial(name, step, observation, situation, regressions=(), base=False, nll=None):
    return {'name': name, 'step': step, 'correct': {'macro': {'observation': observation / 10, 'complete_situation': situation / 10}}, 'regressions': list(regressions),
            'base': base, 'nll': nll}


def test_one_ranking_orders_native_and_hf_points_by_the_declared_unit():
    from semantic_trainer.jobs import ranking
    base = trial('base', 0, 5, 3, base=True)
    native = [base, trial('step-8', 8, 7, 2), trial('step-16', 16, 6, 4)]
    assert ranking(native, 'observation')[0]['name'] == 'step-8' and ranking(native, 'complete_situation')[0]['name'] == 'step-16'
    assert ranking([base, trial('step-8', 8, 5, 3)], 'observation')[0]['name'] == 'base'
    assert ranking([trial('step-8', 8, 6, 3, ['action:clarify']), trial('step-16', 16, 6, 3)], 'observation')[0]['name'] == 'step-16'
    assert ranking([trial('step-16', 16, 6, 3), trial('step-8', 8, 6, 3)], 'observation')[0]['name'] == 'step-8'
    hf = [trial('step-8', 8, 6, 3, nll=.4), trial('step-16', 16, 6, 3, nll=.3)]
    assert ranking(hf, 'observation')[0]['name'] == 'step-16'


def test_complete_situation_correctness_needs_every_target_and_counts_failures():
    from semantic_trainer.jobs import unit_correct
    situations = {'t:a': {'task': 't', 'role': 'model_select', 'complete': True, 'targets': {'x': 'o1', 'y': 'o2', 'z': 'exact'}},
                  't:b': {'task': 't', 'role': 'model_select', 'complete': True, 'targets': {'x': 'o3', 'y': 'o4'}},
                  't:c': {'task': 't', 'role': 'model_select', 'complete': False, 'targets': {'x': 'o5', 'y': 'unlabelled'}},
                  't:d': {'task': 't', 'role': 'fit', 'complete': True, 'targets': {'x': 'o6'}}}
    correct = {'o1': True, 'o2': True, 'o3': True, 'o4': None, 'o5': True}
    tasks = dict.fromkeys(correct, 't')
    assert unit_correct(correct, situations, 'model_select', tasks) == {
        'observation': 4, 'complete_situation': 1, 'situations': 2, 'observations': 5, 'macro': {'observation': .8, 'complete_situation': .5},
        'tasks': {'t': {'observation': 4, 'complete_situation': 1, 'situations': 2, 'observations': 5}}}


def test_summary_keeps_member_addressed_pairs_and_reports_prior_drift():
    from semantic_trainer.select import summary
    rows = [{'id': f'{member}{i}', 'task': 't', 'fact': 'issues', 'member': member, 'classes': ['true', 'false'], 'label': i, 'group': f'raw{i}', 'component': f'g{i}',
             'pairs': ['p']}
            for member in ('refund', 'billing') for i in (0, 1)]
    def scored(correct):
        return [{'id': row['id'], 'fact': 'issues', 'member': row['member'], 'correct': ok, 'accepted': False, 'gold_class': row['classes'][row['label']],
                 'answer': row['classes'][row['label']] if ok else row['classes'][1 - row['label']], 'ms': 1.} for row, ok in zip(rows, correct)]
    base = summary('base', 0, 'a' * 64, scored([True, True, True, False]), rows, {})
    candidate = summary('step-1', 1, 'b' * 64, scored([False, True, True, True]), rows, {}, base)
    assert base['pairs'] == {'pairs': 2, 'both_correct': 1} and candidate['pairs'] == {'pairs': 2, 'both_correct': 1} and base['base'] and not candidate['base']
    assert candidate['per_class']['issues/refund'] == {'true': {'gold': 1, 'predicted': 0, 'correct': 0, 'recall': 0.}, 'false': {'gold': 1, 'predicted': 2, 'correct': 1, 'recall': 1.}}
    assert candidate['class_regressions'] == candidate['regressions'] == ['issues/refund:true']
    assert candidate['family_fixes'] == ['g1'] and candidate['family_regressions'] == ['g0'] and candidate['raw_correct'] == 3 == candidate['correct']['observation']


def fitted_run(tmp_path, job, resolved, observations, base_sha):
    from semantic_trainer.adapter import identify
    from semantic_trainer.jobs import ranking_rule
    from semantic_trainer.jobs import CANDIDATES_SCHEMA, RUN_SCHEMA
    run = tmp_path / 'run'
    fitted = recorded({**resolved, 'targets': FITTED['targets'], 'placement': {'': 'cpu'}, 'trainable_parameters': 48, 'tokenizer': observations[0]['native']['tokenizer'],
                       'profile': {'model_type': 'qwen3', 'device': 'cpu'}})
    selected = {row['id']: row['label'] for row in observations if row['role'] == 'model_select'}
    candidates = []
    for step, name in ((0, 'zero'), (8, 'step-8'), (16, 'step-16')):
        (run / 'candidates' / name).mkdir(parents=True)
        candidates.append({'name': name, 'step': step, 'examples_seen': 8 * step, 'correct': {}, 'nll': 1., 'completed_after_deadline': False, 'peft': f'candidates/{name}',
                           'predictions': {i: {'answer': label if step == 8 else 0, 'stages': {}} for i, label in selected.items()},
                           **identify(fitted, peft_adapter(run / 'candidates' / name, fitted, seed=step))})
    files.publish(run / 'resolved-job.json', fitted)
    identities = {'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id']}
    files.publish(run / 'candidates.json', {'schema': CANDIDATES_SCHEMA, 'kind': 'lora-candidates', 'state': 'ready', **identities, 'loss': resolved['loss'],
                                            'fitted_job_id': artifact_digest(fitted), 'assets': resolved['assets'], 'base_sha256': base_sha,
                                            'unit': resolved['recipe']['selection_unit'], 'rule': ranking_rule(candidates[1:], resolved['recipe']['selection_unit']),
                                            'stop': {'reason': 'patience'}, 'candidates': candidates})
    files.publish(run / 'run.json', {'schema': RUN_SCHEMA, 'state': 'ready', 'at': files.now(), 'loss': resolved['loss'], **identities})
    return run


def selection(tmp_path, monkeypatch, *, acceptance_fit=False, author=None, amend=None, scope=None, fit_probe_observations=0):
    from semantic_switch import backend
    from semantic_trainer import select as module
    from semantic_trainer.jobs import prepare, resolve
    identity = api.Runtime.identity
    manifest, definition, export, records = task(tmp_path, monkeypatch, n=60, author=author)
    monkeypatch.setattr(api.Runtime, 'identity', identity)
    if amend is not None:
        amend(definition)
    job = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe={**RECIPE, 'fit_probe_observations': fit_probe_observations},
                  native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048}, acceptance_fit=acceptance_fit, scope=scope)['job']
    resolved, observations = resolve(job)
    run = fitted_run(tmp_path, job, resolved, observations, files.sha256_file(resolved['model']['gguf']))
    deployed = []
    def deploy(fitted, located, peft, destination, quant, handle):
        deployed.append((Path(peft).name, quant, handle.state))
        Path(destination).write_bytes(b'merged ' + Path(peft).name.encode())
        return files.sha256_file(destination)
    from semantic_trainer import representation
    monkeypatch.setattr(representation, 'deploy', deploy)
    monkeypatch.setattr(module, 'verify_zero', lambda fitted, located, peft, quant, work, handle: deployed.append(('zero', quant, handle.state)))
    monkeypatch.setattr(backend, 'LlamaScorer', Native)
    monkeypatch.setattr(api.Model, 'resources', lambda self: {'weights': api.Resources(1)})
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    monkeypatch.setattr(module, 'conversion_bytes', lambda located: {'conversion': 1})
    Native.built, Native.realized = [], REALIZED
    return module, job, run, {**resolved, 'roles': {role: [row['id'] for row in observations if row['role'] == role] for role in ('fit', 'model_select', 'acceptance_fit')}}, records, deployed, definition


def check_labels(tmp_path, n=12):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.evidence import label_record
    definition = tmp_path / 'task' / 'action.yaml'
    plan = compile_spec(files.read_yaml(definition), definition.name, definition.parent)
    path = tmp_path / 'check.jsonl'
    files.publish(path, ''.join(json.dumps(label_record(plan, {'message': f'report {1000 + i}'}, {'action': VALUES[(1000 + i) % 3]}, groups=[f'check-{i}'],
                                                        sources={'action': [{'derivation': 'authored', 'producer': 'reviewer', 'actor': {'kind': 'human'}}]}, role='check')) + '\n' for i in range(n)))
    return path


def resolved_rows(job, role):
    from semantic_trainer.jobs import resolve
    return [row for row in resolve(job)[1] if row['role'] == role]


def test_select_publishes_a_self_contained_deployment_with_retained_provenance(tmp_path, monkeypatch):
    import shutil
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    result = module.select(job, run)
    assert [name for name, _, _ in deployed] == ['zero', 'step-8', 'step-16'] and {state for _, _, state in deployed} == {'open'}
    assert result['selected'] == 'step-8' and result['candidates']['step-8']['raw_correct'] == len(resolved['roles']['model_select']) and result['unit'] == 'observation'
    trials = {t['name']: t for t in load_receipt(result['receipt']).data['selection']['candidates']}
    presentations = sum(len(row['presentations']) for row in resolved_rows(job, 'model_select'))
    for trial in trials.values():
        work = trial['selection_work']
        assert work['total_seconds'] == pytest.approx(work['public_seconds'] + work['diagnostic_seconds']) and work['public_seconds'] > 0
        reuse = work['diagnostic_reuse']
        assert reuse['requests'] == reuse['rows'] == presentations and reuse['reused'] + reuse['scored'] == presentations
    assert result['candidates']['step-8']['selection_work'] == trials['step-8']['selection_work']
    compatibility = trials['step-8']['hf_agreement']
    assert compatibility['specialist']['agreed'] == compatibility['difference']['targets'] == len(resolved['roles']['model_select']) and 'hf_agreement' not in trials['base']
    assert compatibility['gate'] is False and compatibility['tolerance'] is None and compatibility['intermediate'] is None
    assert compatibility['difference']['specialist_only_agreed'] - compatibility['difference']['base_only_agreed'] == len(resolved['roles']['model_select']) - compatibility['base']['agreed']
    assert all(scorer.closed == 1 for scorer in Native.built)
    receipt = load_receipt(result['receipt'])
    data = thaw(receipt.data)
    assert data['selection']['rule']['order'] == ['correct', 'regressions', 'base', 'step'] and data['selection']['rule']['nll'] == 'unavailable'
    assert data['selection']['loss'] == resolved['loss']
    assert set(data['files']) == {'specialist.gguf', 'task.yaml', 'task.lock.json', 'guidance/0.txt', 'selection.rows.jsonl'}
    assert data['execution'] == execution_identity(REALIZED) and data['native'] == {**data['native'], 'adapter': 'qwen3', 'device': 'cpu'}
    assert data['acceptance'] == {**data['acceptance'], 'policy': 'authored', 'calibration': 'absent', 'acceptance_fit': {}} and not resolved['roles']['acceptance_fit']
    lock = files.read_json(run / 'deployment' / 'task.lock.json')
    assert data['lock'] == {'path': 'task.lock.json', 'id': lock['id'], 'candidate_id': lock['candidate_id']}
    assert {side['status'] for fact in lock['facts'].values() for side in fact['policies']['judge'].values()} == {'uncalibrated_default'}
    uses = {row['id']: row['uses'] for row in data['provenance']['rows']}
    assert receipt.provenance.kind == 'specialization' and all(uses[i] == ['adaptation', 'selection'] for i in resolved['roles']['fit'])
    assert all(uses[i] == ['selection'] for i in resolved['roles']['model_select'])
    assert {source['export_id'] for row in receipt.provenance.rows for source in row['sources']} == {resolved['sources'][0]['export_id']}
    memberships = {row['id']: row['label_sources'] for row in resolved['memberships']}
    assert all(row['label_sources'] == memberships[row['id']] and row['label_sources'] for row in data['provenance']['rows'])
    moved = tmp_path / 'moved'
    shutil.move(run / 'deployment', moved)
    shutil.rmtree(tmp_path / 'task')
    with Runtime() as runtime:
        switch = runtime.load(moved / 'receipt.json')
        assert switch.snapshot.lock_id == lock['id'] and runtime.models['judge'].provenance == receipt.provenance
        assert switch({'message': 'report 4'}).facts['action'].value == VALUES[1]
    assert api.judge(str(moved / 'receipt.json')).provenance == receipt.provenance
    with Runtime(FakeScorer()) as other:
        with pytest.raises(ConfigError, match='leaves that role to the lock'):
            other.load(moved / 'receipt.json')
    Native.realized = {**REALIZED, 'n_batch': 64}
    with Runtime() as runtime:
        with pytest.raises(ConfigError, match='judge execution changed'):
            runtime.load(moved / 'receipt.json').snapshot


def test_deployment_loads_and_verifies_with_read_only_cache_storage(tmp_path, monkeypatch):
    import stat
    import warnings
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    receipt = Path(module.select(job, run)['receipt'])
    root = tmp_path / 'readonly-cache'
    root.mkdir()
    root.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(root))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert load_receipt(receipt).id
            with Runtime() as runtime:
                assert runtime.load(receipt)({'message': 'report 4'}).facts['action'].value == VALUES[1]
            gguf = receipt.parent / load_receipt(receipt).data['gguf']
            gguf.write_bytes(gguf.read_bytes() + b'changed')
            with Runtime() as runtime, pytest.raises(ConfigError, match='changed since selection'):
                runtime.load(receipt)
        assert list(root.iterdir()) == []
    finally:
        root.chmod(stat.S_IRWXU)


def test_deployment_check_records_exposure_first_and_never_mutates_the_deployment(tmp_path, monkeypatch):
    from semantic_switch.tools import deployment as checking
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch, acceptance_fit=True)
    result = module.select(job, run)
    receipt = load_receipt(result['receipt'])
    assert receipt.data['acceptance']['policy'] == 'fitted' and receipt.provenance.membership('acceptance_fit')
    frozen = files.read_json(run / 'deployment' / 'task.lock.json')['facts']['action']['policies']['judge']
    assert {side['calibration'] for side in frozen.values()} <= {'in_sample', 'absent'} and all(side['fit'] for side in frozen.values())
    with Runtime() as runtime:
        assert thaw(runtime.load(result['receipt']).snapshot.sides['judge']['action']) == frozen
    labels, evidence = assigned(tmp_path / 'evidence', definition, files.read_jsonl(check_labels(tmp_path)))
    before = {path.name: files.sha256_file(path) for path in (run / 'deployment').iterdir() if path.is_file()}
    audit = Path(EvidenceContext.from_manifest(evidence).audit)
    def failing(*args, **kwargs):
        assert files.read_jsonl(audit)[-1]['plan']['arms']['specialist']['entry'] == receipt.id
        raise RuntimeError('scoring interrupted')
    from semantic_switch.tools import evaluate
    original = evaluate.evaluate_facts
    monkeypatch.setattr(evaluate, 'evaluate_facts', failing)
    unbound = tmp_path / 'unbound.jsonl'
    unbound.write_text(''.join(labels.read_text().splitlines(True)[:5]))
    with pytest.raises(ConfigError, match='not the one bound by evidence'):
        checking.check_deployment(result['receipt'], unbound, evidence, tmp_path / 'unbound')
    assert not audit.read_text()
    with pytest.raises(RuntimeError, match='interrupted'):
        checking.check_deployment(result['receipt'], labels, evidence, tmp_path / 'failed')
    assert not (tmp_path / 'failed' / 'check.json').exists() and load_receipt(result['receipt']).id == receipt.id
    monkeypatch.setattr(evaluate, 'evaluate_facts', original)
    report = checking.check_deployment(result['receipt'], labels, evidence, tmp_path / 'check')
    assert report['entry'] == receipt.id and report['lock'] == receipt.data['lock']['id'] and report['situations'] == 12 and report['families'] == 12
    assert report['facts']['action']['labelled'] == 12 and report['lineage']['targets']['judge:action']['state'] == 'declared_disjoint'
    assert report['exposure']['state'] == 'repeat' and report['exposure']['independent'] and len(files.read_jsonl(audit)) == 1
    assert {path.name: files.sha256_file(path) for path in (run / 'deployment').iterdir() if path.is_file()} == before
    overlapping, overlap_evidence = assigned(tmp_path / 'overlap-evidence', definition, [{**row, 'role': 'check'} for row in files.read_jsonl(records)[:3]])
    assert checking.check_deployment(result['receipt'], overlapping, overlap_evidence, tmp_path / 'overlap')['lineage']['targets']['judge:action']['state'] == 'known_overlap'


def assigned(directory, definition, rows):
    from semantic_switch.tools.data import Labels
    paths = files.Paths.of(directory / Path(definition).name)
    files.write_new(paths.yaml, Path(definition).read_text())
    files.write_new(paths.test, ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(FakeScorer()) as runtime:
        Labels(runtime.load(paths.yaml, draft=True))
    return paths.test, paths.dataset


def test_deployment_comparison_journals_one_call_per_input_and_arm_and_a_verified_pointer_rolls_back(tmp_path, monkeypatch):
    from semantic_switch.tools import deployment
    from semantic_switch.training import compare_deployments, current, declaration, point
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    first, second = module.select(job, run)['receipt'], module.select(job, run, to=tmp_path / 'second')['receipt']
    labels, evidence = assigned(tmp_path / 'evidence', definition, files.read_jsonl(check_labels(tmp_path)))
    declared_ = declaration(claim='improvement', margin=.05, forbidden={'action': ['continue']})
    original, calls = deployment.compared_row, []
    def interrupted(plan, item, family, result, review):
        calls.append(item['input']['message'])
        if len(calls) == 5:
            raise KeyboardInterrupt('interrupted')
        return original(plan, item, family, result, review)
    monkeypatch.setattr(deployment, 'compared_row', interrupted)
    with pytest.raises(KeyboardInterrupt):
        compare_deployments(first, second, labels, evidence, tmp_path / 'compare', declaration=declared_, count=50)
    assert len(files.read_jsonl(tmp_path / 'compare' / 'rows.journal.jsonl')) == 5 and not (tmp_path / 'compare' / 'comparison.json').exists()
    report = compare_deployments(first, second, labels, evidence, tmp_path / 'compare', declaration=declared_, count=50)
    assert len(calls) == 5 + 24 - 4 and len(files.read_jsonl(tmp_path / 'compare' / 'rows.journal.jsonl')) == 1 + 24
    assert report['situations'] == 12 and report['arms']['baseline']['calls'] == report['arms']['candidate']['calls'] == 12
    assert report['exposure']['independent'] and report['lineage']['candidate']['state'] == 'declared_disjoint' and report['reference']['state'] == 'agreement'
    assert report['reference']['targets'] == 12 and report['reference']['within_scope'] == 0 and report['decision']['verdict'] == 'reference_agreement'
    assert report['paired']['support']['matched'] == 12 and report['paired']['fixes'] == report['paired']['regressions'] == 0
    assert report['arms']['candidate']['targets']['submitted'] == 12 and report['arms']['candidate']['forbidden_accepted'] == \
        sum(row['targets']['action']['accepted'] and row['targets']['action']['prediction']['key'] == 'continue' for row in report['rows']['candidate'])
    assert files.read_published(tmp_path / 'compare' / 'comparison.json')['id'] == report['id']
    with pytest.raises(ConfigError, match='already published'):
        compare_deployments(first, second, labels, evidence, tmp_path / 'compare', declaration=declared_, count=50)
    again = compare_deployments(first, second, labels, evidence, tmp_path / 'repeat', declaration={**declared_, 'margin': .1}, count=50)
    assert again['exposure']['state'] == 'consulted' and not again['exposure']['independent'] and 'descriptive only' in again['population']
    pointer = tmp_path / 'current.json'
    point(pointer, first, reason='initial', expected=None)
    point(pointer, second, reason='promoted', expected=load_receipt(first).id)
    assert current(pointer) == Path(second)
    rolled = point(pointer, files.read_json(pointer)['history'][0]['entry'], reason='rollback', expected=load_receipt(second).id)
    assert current(pointer) == Path(first) and [entry['reason'] for entry in rolled['history']] == ['initial', 'promoted', 'rollback']
    Path(first).parent.joinpath(load_receipt(first).data['gguf']).write_bytes(b'changed')
    with pytest.raises(ConfigError, match='changed since selection'):
        current(pointer)
    with pytest.raises(ConfigError, match='changed since selection'):
        point(pointer, first, reason='retry', expected=load_receipt(first).id)
    assert files.read_json(pointer)['history'][-1]['reason'] == 'rollback'


def test_interrupted_selection_publishes_nothing(tmp_path, monkeypatch):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    def interrupted(path):
        raise ConfigError('interrupted before publication')
    monkeypatch.setattr(module, 'load_receipt', interrupted)
    with pytest.raises(ConfigError, match='interrupted'):
        module.select(job, run)
    assert not (run / 'deployment').exists() and not list(run.glob('.deployment-*')) and not list(run.glob('.select-*'))


def test_heterogeneous_native_trials_are_rejected(tmp_path, monkeypatch):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    realized = iter([REALIZED, {**REALIZED, 'device': 'gpu'}, REALIZED])
    original = Native.__init__
    def varying(self, path, **kwargs):
        original(self, path, **kwargs)
        self.realized = next(realized)
    monkeypatch.setattr(Native, '__init__', varying)
    with pytest.raises(ConfigError, match='different execution'):
        module.select(job, run)
    assert not (run / 'deployment').exists()


def test_select_refuses_stale_runs_and_wrong_bases(tmp_path, monkeypatch):
    from semantic_trainer import select as module
    from semantic_trainer.jobs import prepare, resolve
    manifest, definition, export, _ = task(tmp_path, monkeypatch)
    job = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)['job']
    resolved, observations = resolve(job)
    run = fitted_run(tmp_path, job, resolved, observations, 'f' * 64)
    with pytest.raises(ConfigError, match='not the pipeline-built Q4_0 base'):
        module.select(job, run)
    listed = files.read_json(run / 'candidates.json')
    files.publish(run / 'candidates.json', {**listed, 'job_id': 'stale', 'base_sha256': files.sha256_file(resolved['model']['gguf'])})
    with pytest.raises(ConfigError, match='name different job, fit or monitor identities'):
        module.select(job, run)
    files.publish(run / 'candidates.json', {**listed, 'loss': {**listed['loss'], 'version': 1}, 'base_sha256': files.sha256_file(resolved['model']['gguf'])})
    with pytest.raises(ConfigError, match='fitted under training objective .*refit'):
        module.select(job, run)


def test_prepare_chooses_whole_fit_families_for_an_exact_situation_count(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare, resolve
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, n=90)
    files.publish(records, ''.join(json.dumps({**row, 'groups': [f'family-{i // 3}'], 'role': ('fit', 'fit', 'model_select')[i // 3 % 3]}) + '\n'
                                   for i, row in enumerate(files.read_jsonl(records))))
    with Runtime(FakeScorer()) as runtime:
        export = action_export(runtime, tmp_path / 'task', records, 'maintenance:test')['path']
    definition = tmp_path / 'task' / 'action.yaml'
    result = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, fit_situations=30)
    resolved, observations = resolve(result['job'])
    assert result['roles']['fit']['situations'] == 30 and result['roles']['fit']['families'] == 10
    fit = {row['component'] for row in observations if row['role'] == 'fit'}
    assert not fit & {row['component'] for row in observations if row['role'] != 'fit'}
    with pytest.raises(ConfigError, match='exactly 31 situations'):
        prepare(definition, [export], tmp_path / 'odd', assets_manifest=manifest, recipe=RECIPE, fit_situations=31)


def test_exact_fit_situation_budget_spans_every_export_of_the_job(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare, resolve
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, n=90)
    rows = [{**row, 'groups': [f'family-{i // 3}'], 'role': ('fit', 'fit', 'model_select')[i // 3 % 3]} for i, row in enumerate(files.read_jsonl(records))]
    with Runtime(FakeScorer()) as runtime:
        action_export(runtime, tmp_path / 'task', records, 'maintenance:test')
    definition = tmp_path / 'task' / 'action.yaml'
    exports = []
    for name, part in (('first', rows[:45]), ('second', rows[45:])):
        files.publish(tmp_path / f'{name}.jsonl', ''.join(json.dumps(row) + '\n' for row in part))
        with Runtime(FakeScorer()) as runtime:
            exports.append(tools.export(runtime, definition, to=tmp_path / f'{name}.export.jsonl', native=True, task_family='maintenance:test',
                                        labels=tmp_path / f'{name}.jsonl')['path'])
    result = prepare(definition, exports, tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, fit_situations=45)
    resolved, observations = resolve(result['job'])
    fit = [row for row in observations if row['role'] == 'fit']
    assert result['roles']['fit']['situations'] == 45 == len({s for row in fit for s in row['situations']}) == len(result['selection']['situation_ids'])
    assert result['roles']['fit']['families'] == 15 and {source['path'] for row in fit for source in row['sources']} == set(map(str, exports))
    assert result['roles']['fit']['targets'] == result['roles']['fit']['observations'] == len(fit)
    assert resolved['selection'] == result['selection']
    with pytest.raises(ConfigError, match='exactly 61 situations'):
        prepare(definition, exports, tmp_path / 'too-many', assets_manifest=manifest, recipe=RECIPE, fit_situations=61)
    assert not (tmp_path / 'too-many' / 'job.yaml').exists()


def test_cli_recipe_and_budget_are_the_sole_typed_setters(monkeypatch, capsys):
    import sys
    from semantic_trainer.cli import main, settings
    from semantic_trainer.jobs import BUDGET, DEFAULTS
    assert settings(['selection_unit=complete_situation', 'epochs=4', 'target_modules=[q_proj, v_proj]', 'learning_rate=1e-4'], DEFAULTS, 'recipe') == \
        {'selection_unit': 'complete_situation', 'epochs': 4, 'target_modules': ['q_proj', 'v_proj'], 'learning_rate': 1e-4}
    for pairs, match in ((['epochs=4', 'epochs=2'], 'more than once'), (['epochs=4', 'epochs=4'], 'more than once'), (['warmup_ratio=.05'], 'unknown --recipe key'),
                         (['epochs'], 'KEY=VALUE'), (['epochs=2.5'], 'expects int'), (['learning_rate=fast'], 'expects float')):
        with pytest.raises(ConfigError, match=match):
            settings(pairs, DEFAULTS, 'recipe')
    with pytest.raises(ConfigError, match='unknown --budget key'):
        settings(['max_steps=3'], BUDGET, 'budget')
    monkeypatch.setattr(sys, 'argv', ['semantic-trainer', 'prepare', 'a.yaml', 'b.json', '--assets', 'x', '--to', 'job', '--selection-unit', 'complete_situation'])
    with pytest.raises(SystemExit):
        main()
    assert 'unrecognized arguments: --selection-unit' in capsys.readouterr().err


def test_complete_situation_selection_rejects_incomplete_selection_gold_before_fitting(tmp_path, monkeypatch):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.evidence import each_target, label_record
    from semantic_trainer.jobs import prepare
    manifest = pipeline_base(tmp_path, monkeypatch)
    path = tmp_path / 'pair.yaml'
    files.publish(path, {'name': 'pair', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}, 'legal': {'check': 'The customer threatens legal action.'}}})
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    source = [{'derivation': 'authored', 'producer': 'fixture', 'actor': {'kind': 'human'}}]
    rows = []
    for i in range(30):
        facts = {'refund': bool(i % 2), **({'legal': bool(i // 2 % 2)} if i % 5 else {})}
        rows.append(label_record(plan, {'message': f'case {i}'}, facts, sources=each_target(facts, source), groups=[f'g{i}'],
                                 role=('fit', 'fit', 'model_select')[i % 3]))
    files.publish(tmp_path / 'labels.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime(FakeScorer()) as runtime:
        export = tools.export(runtime, path, native=True, task_family='support:pair', labels=tmp_path / 'labels.jsonl')['path']
    with pytest.raises(ConfigError, match='complete_situation selection needs gold-complete model_select situations'):
        prepare(path, [export], tmp_path / 'job', assets_manifest=manifest, recipe={**RECIPE, 'selection_unit': 'complete_situation'})
    assert prepare(path, [export], tmp_path / 'observation-job', assets_manifest=manifest, recipe=RECIPE)['recipe']['selection_unit'] == 'observation'


def test_transfer_preparation_keeps_per_source_task_families_and_objective_aware_support(tmp_path, monkeypatch):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.evidence import each_target, label_record
    from semantic_trainer.jobs import prepare, resolve
    manifest = pipeline_base(tmp_path, monkeypatch)
    path = tmp_path / 'printer.yaml'
    options = {'calibrate': 'Run sensor calibration', 'resume': 'Resume printing'}
    files.publish(path, {'name': 'printer', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'guidance': {'a': {'text': 'Document A: calibrate after loading new media.'}, 'c': {'text': 'Document C: calibrate only after an alarm.'}},
                         'facts': {'next_a': {'pick': 'Which action does the document require next?', 'options': options, 'guidance': ['a']},
                                   'next_c': {'pick': 'Which action does the document require next?', 'options': options, 'guidance': ['c']}}})
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    assert plan.fact('next_a').meaning_rev != plan.fact('next_c').meaning_rev
    source = [{'derivation': 'computed', 'producer': 'printer-reference', 'revision': '1'}]
    def export(name, fact, role, family, values):
        rows = [label_record(plan, {'message': f'{name} report {i}'}, {fact: values[i % 2]}, sources=each_target({fact: values[i % 2]}, source),
                             groups=[f'{name}-{i}'], role=role) for i in range(12)]
        labels = tmp_path / f'{name}.jsonl'
        files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
        with Runtime(FakeScorer()) as runtime:
            return tools.export(runtime, path, to=tmp_path / f'{name}.export.jsonl', native=True, labels=labels, scope=fact,
                                **({'task_family': family} if family else {}))['path']
    fit = export('a', 'next_a', 'fit', 'printer:a', ('calibrate', 'resume'))
    held = export('c', 'next_c', 'model_select', 'printer:c', ('resume', 'calibrate'))
    result = prepare(path, [fit, held], tmp_path / 'transfer', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')
    resolved, observations = resolve(result['job'])
    assert list(result['support']['absent_fit_support'].values()) == [['next_c']] and {row['task_family'] for row in observations if row['role'] == 'fit'} == {'printer:a'}
    assert {(source['task_family'], role) for source in resolved['sources'] for role in source['roles']} == {('printer:a', 'fit'), ('printer:c', 'model_select')}
    assert all('Document C' not in row['text'] and 'Document A' not in row['text'] for row in observations)
    specialization = prepare(path, [fit, held], tmp_path / 'specialization', assets_manifest=manifest, recipe=RECIPE)
    assert list(specialization['support']['absent_fit_support'].values()) == [['next_c']]
    bare = export('b', 'next_c', 'model_select', None, ('resume', 'calibrate'))
    with pytest.raises(ConfigError, match='carries one namespaced task family'):
        prepare(path, [fit, bare], tmp_path / 'bare', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')
    files.publish(path, {**files.read_yaml(path), 'guidance': {'a': {'text': 'Document A: calibrate after loading new media.'},
                                                               'c': {'text': 'Document C revised: calibrate after every job.'}}})
    assert resolve(result['job'])[0]['id'] == resolved['id'] and 'Document C revised' not in (tmp_path / 'transfer' / 'tasks' / next(iter(resolved['tasks'])) / 'guidance' / '1.txt').read_text()
    with pytest.raises(ConfigError, match='next_c changed or left the task definition'):
        prepare(path, [fit, held], tmp_path / 'revised', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')


class ResidentNative(Native):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.weights, self.active, self.activated = SimpleNamespace(hybrid=False, offload=False), None, []
        def oracle(state, question, width):
            found, options = re.search(r'report (\d+)', state), letters_of(question)
            wanted = VALUES[int(found.group(1)) % len(VALUES)] if found and self.active == b'adapter step-8' else None
            hits = [letter for text, letter in options.items() if wanted and text.startswith(dict(zip(VALUES, ACTION_TEXT))[wanted])]
            return [0.98 if chr(65 + k) in hits else 0.02 / max(width - 1, 1) for k in range(width)] if hits else [1 / width] * width
        self.oracle = oracle

    def score(self, requests, *, reference=False, lora=None, binding=None):
        self.active = None if lora is None else Path(lora.path).read_bytes()
        self.activated.append(self.active)
        return super().score(requests)

    def forget(self, lora):
        pass


def test_runtime_adapter_selection_publishes_independently_hashed_base_and_adapter(tmp_path, monkeypatch):
    from semantic_switch import backend, resident
    from semantic_trainer import representation
    module, job, run, resolved, _, deployed, _ = selection(tmp_path, monkeypatch)
    recipe = files.read_json(run / 'resolved-job.json')['recipe']
    monkeypatch.setattr(backend, 'LlamaScorer', ResidentNative)
    monkeypatch.setattr(api.Model, 'resources', lambda self: {'weights': api.Resources(1), **({'adapter': api.Resources(1 << 14)} if self.lora else {})})
    monkeypatch.setattr(api.Model, 'placement', lambda self: backend.Placement(1, 0))
    monkeypatch.setattr(backend, 'Lora', lambda weights, path, key: SimpleNamespace(path=path, key=key, buffers={'CPU LoRA': 64}, close=lambda: None))
    monkeypatch.setattr(representation, 'adapter', lambda fitted, located, peft, destination: Path(destination).write_bytes(b'adapter ' + Path(peft).name.encode()))
    monkeypatch.setattr(resident.Adapter, 'verified', classmethod(lambda cls, path, base, sha256=None: cls(
        str(path), files.sha256_file(path), recipe['rank'], float(recipe['alpha']), ('blk.0.attn_q.weight',), Path(path).stat().st_size,
        (('blk.0.attn_q.weight.lora_a', 32), ('blk.0.attn_q.weight.lora_b', 32)), 0)))
    ResidentNative.built = Native.built = []
    result = module.select(job, run, 'runtime-adapter')
    assert result['selected'] == 'step-8' and result['representation'] == 'runtime-adapter' and deployed == []
    selecting, freezing = Native.built
    assert set(selecting.activated) == {None, b'adapter step-8', b'adapter step-16'} and set(freezing.activated) <= {b'adapter step-8'}
    data = thaw(load_receipt(result['receipt']).data)
    base_sha = files.sha256_file(files.read_json(run / 'resolved-job.json')['model']['gguf'])
    assert data['gguf'] == 'base.gguf' and data['lora'] == 'adapter.gguf' and data['files']['base.gguf'] == base_sha
    assert data['files']['adapter.gguf'] == files.sha256_file(run / 'deployment' / 'adapter.gguf') and data['scope']['whole_task']
    lock = files.read_json(run / 'deployment' / 'task.lock.json')
    assert lock['bindings']['judge']['representation'] == 'runtime-adapter' and lock['models']['judge']['lora']['scale'] == 1.0
    assert lock['models']['judge']['sha256'] == base_sha and lock['models']['judge']['lora']['sha256'] == data['files']['adapter.gguf']
    with Runtime() as runtime:
        assert runtime.load(result['receipt'])({'message': 'report 4'}).facts['action'].value == VALUES[1]
        assert runtime.models['judge'].representation == 'runtime-adapter' and runtime.models['judge'].lora.endswith('adapter.gguf')
    (run / 'deployment' / 'adapter.gguf').write_bytes(b'adapter changed')
    with Runtime() as runtime, pytest.raises(ConfigError, match='changed'):
        runtime.load(result['receipt'])


def test_prepare_resolves_an_explicit_target_scope_through_the_integration_surface(tmp_path, monkeypatch):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.evidence import each_target, label_record
    from semantic_trainer.jobs import prepare, resolve
    manifest = pipeline_base(tmp_path, monkeypatch)
    path = tmp_path / 'printer.yaml'
    options = {'calibrate': 'Run sensor calibration', 'resume': 'Resume printing'}
    files.publish(path, {'name': 'printer', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'next_a': {'pick': 'Which action is required next?', 'options': options},
                                   'next_c': {'pick': 'Which action is required after an alarm?', 'options': options}}})
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    source = [{'derivation': 'computed', 'producer': 'printer-reference', 'revision': '1'}]
    labels = tmp_path / 'labels.jsonl'
    values = ('calibrate', 'resume')
    files.publish(labels, ''.join(json.dumps(label_record(plan, {'message': f'report {i}'}, {'next_a': values[i % 2], 'next_c': values[i // 2 % 2]},
                                                         sources=each_target({'next_a': values[i % 2], 'next_c': values[i // 2 % 2]}, source),
                                                         groups=[f'family-{i}'], role=('fit', 'model_select')[i // 4 % 2])) + '\n' for i in range(16)))

    def export(name, scope):
        with Runtime(FakeScorer()) as runtime:
            return tools.export(runtime, path, to=tmp_path / f'{name}.export.jsonl', native=True, labels=labels, scope=scope, task_family='printer:a')['path']
    scoped, whole = export('scoped', 'next_a'), export('whole', None)
    result = prepare(path, [scoped], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, scope=['next_a'])
    resolved, observations = resolve(result['job'])
    assert list(resolved['scope'].values()) == list(result['scope'].values()) == [{'targets': ['next_a'], 'whole_task': False, 'binding': 'judge'}]
    assert {row['fact'] for row in observations} == {'next_a'} and all(set(entry['targets']) == {'next_a'} for entry in resolved['situations'].values())
    assert [entry['whole_task'] for entry in resolve(prepare(path, [whole], tmp_path / 'whole', assets_manifest=manifest, recipe=RECIPE)['job'])[0]['scope'].values()] == [True]
    with pytest.raises(ConfigError, match=r"exports do not cover scoped targets \['next_c'\]"):
        prepare(path, [scoped], tmp_path / 'uncovered', assets_manifest=manifest, recipe=RECIPE, scope=['next_c'])
    with pytest.raises(ConfigError, match='outside the declared target scope'):
        prepare(path, [whole], tmp_path / 'outside', assets_manifest=manifest, recipe=RECIPE, scope=['next_a'])
    with pytest.raises(ConfigError, match='unknown fact'):
        prepare(path, [scoped], tmp_path / 'unknown', assets_manifest=manifest, recipe=RECIPE, scope=['next_b'])


class Forbidden:
    def __init__(self):
        self.calls = []

    async def __call__(self, url, body, timeout):
        self.calls.append(body)
        raise AssertionError('target-only selection must not run an unrelated hosted binding')


def test_scoped_selection_leaves_unrelated_hosted_bindings_unloaded_and_complete_publication_pins_them(tmp_path, monkeypatch):
    def amend(definition):
        spec = files.read_yaml(definition)
        spec['facts']['action']['binding'] = 'actions'
        spec['facts']['issues'] = {'select': 'Which issues does the report raise?', 'binding': 'hosted', 'min_score': 0.8,
                                   'labels': {'jam': 'A paper jam', 'ink': 'Low ink'}}
        files.publish(definition, spec)
    module, job, run, resolved, _, deployed, definition = selection(tmp_path, monkeypatch, amend=amend, scope=['action'])
    assert resolved['scope'][next(iter(resolved['scope']))] == {'targets': ['action'], 'whole_task': False, 'binding': 'actions'}
    transport = Forbidden()
    hosted = api.judge('openrouter:typesafe/jev-1.13', transport=transport)
    for wrong, missing in (({'other': hosted}, r"missing \['hosted'\], unused \['other'\]"), ({'actions': hosted, 'hosted': hosted}, 'unused')):
        with pytest.raises(ConfigError, match=missing):
            module.select(job, run, to=tmp_path / 'wrong', bindings=wrong)
    assert deployed == [] and Native.built == [] and not (tmp_path / 'wrong').exists()
    scoped = module.select(job, run, to=tmp_path / 'scoped')
    data = thaw(load_receipt(scoped['receipt']).data)
    assert scoped['publication'] == data['publication'] == 'scoped' and scoped['binding'] == 'actions' and scoped['loading']['deployment'] is None
    assert data['scope'] == {'targets': ['action'], 'whole_task': False, 'binding': 'actions'} and transport.calls == []
    lock = files.read_json(tmp_path / 'scoped' / 'task.lock.json')
    assert sorted(lock['bindings']) == sorted(lock['models']) == ['actions']
    with pytest.raises(ConfigError, match="scoped support for targets \\['action'\\] under binding 'actions', not a runnable application"), Runtime() as runtime:
        runtime.load(scoped['receipt'])
    complete = module.select(job, run, to=tmp_path / 'complete', bindings={'hosted': hosted})
    data = thaw(load_receipt(complete['receipt']).data)
    assert complete['publication'] == data['publication'] == 'application' and transport.calls == []
    lock = files.read_json(tmp_path / 'complete' / 'task.lock.json')
    assert sorted(lock['bindings']) == ['actions', 'hosted'] and lock['models']['hosted']['model'] == 'typesafe/jev-1.13'
    members = sorted(name for name, role in compile_spec_of(definition).targets.items() if role == 'hosted')
    assert len(members) == 2 and lock['bindings']['hosted']['scope'] == members and lock['bindings']['actions']['scope'] == ['action']
    with pytest.raises(ConfigError, match=r"bound to model roles \['hosted'\] that this runtime does not bind"):
        with Runtime() as runtime:
            runtime.load(complete['receipt'])
    with Runtime(bindings={'hosted': hosted}) as runtime:
        loaded = runtime.load(complete['receipt'])
        assert loaded.plan.binding_of(loaded.plan.fact_index['issues']) == 'hosted' and loaded.plan.fact('issues').policy.thresholds('select') == \
            compile_spec_of(definition).fact('issues').policy.thresholds('select')
    assert transport.calls == []


def compile_spec_of(definition):
    from semantic_switch.compiler import compile_spec
    return compile_spec(files.read_yaml(definition), definition.name, definition.parent)


def test_publication_retry_reuses_native_selection_rows_and_keeps_receipt_compact(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_switch.results import thaw
    module, job, run, resolved, *_ = selection(tmp_path, monkeypatch)
    freeze = module.freeze_specialization
    monkeypatch.setattr(module, 'freeze_specialization', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('publication interrupted')))
    with pytest.raises(RuntimeError, match='publication interrupted'):
        module.select(job, run)
    assert list((run / 'native-selection' / 'merged').glob('*.jsonl'))
    monkeypatch.setattr(module, 'freeze_specialization', freeze)
    monkeypatch.setattr(module, 'score_observations', lambda *args, **kwargs: pytest.fail('replayed a public observation'))
    monkeypatch.setattr(module, 'native_predictions', lambda *args, **kwargs: pytest.fail('replayed a native diagnostic'))
    result = module.select(job, run)
    receipt = load_receipt(result['receipt'])
    data = thaw(receipt.data)
    assert all('rows' not in candidate for candidate in data['selection']['candidates'])
    record = data['selection']['rows']
    rows = list(files.read_jsonl(receipt.directory / record['file']))
    assert len(rows) == record['rows'] and all('result' not in row and 'probability' in row for row in rows)
    assert files.sha256_file(receipt.directory / record['file']) == record['sha256']
    assert Path(result['receipt']).stat().st_size < 1024 * 1024
