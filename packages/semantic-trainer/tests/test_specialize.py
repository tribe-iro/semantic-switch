from contextlib import ExitStack
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, Runtime, api, files, tools
from semantic_switch.api import execution_identity
from semantic_switch.artifacts import load_receipt, read_export
from semantic_switch.examples import role_of
from semantic_switch.results import thaw
from semantic_switch.tools.dataset import EvidenceContext
from tests.conftest import FakeScorer, letters_of
from tests.test_adaptation_contracts import fake_assets, label_file, pipeline_base

VALUES = ('escalate', 'continue', 'clarify')
RECIPE = {'eval_every_examples': 8}


def task(tmp_path, monkeypatch, values=VALUES, n=40, author=None):
    from examples import adapt_judgment
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, values, n=n)
    Native.tokens = FakeScorer()
    if author is not None:
        author(tmp_path / 'task' / 'action.yaml')
    with Runtime({'judge': Native.tokens}) as runtime:
        exported = adapt_judgment.prepare(runtime, tmp_path / 'task', records, 'maintenance:test')
    return manifest, tmp_path / 'task' / 'action.yaml', exported['path'], records


def oracle(state, question, width):
    text = state.lower()
    if 'crash' in text:
        raise KeyboardInterrupt('process killed')
    if 'boom' in text:
        raise RuntimeError('backend exploded')
    if 'maybe' in text:
        return [1 / width] * width
    wanted = 'refund' if 'money back' in question.lower() else 'invoice' if 'billing' in question.lower() else None
    return [.96, .04] if wanted and wanted in text else [.04, .96]


def collection_task(tmp_path, rows):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'},
                                   'issues': {'select': 'Which issues does the customer raise?', 'max_claims': 1,
                                              'labels': {'refund': 'Money back for a charge', 'billing': 'A billing or invoice error'}}}})
    situations = tmp_path / 'situations.jsonl'
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in rows))
    return path, situations


def envelope(message, **meta):
    return {'schema': 2, 'input': {'message': message}, **meta}


def test_collection_accounts_every_target_and_publishes_reviewable_labels(tmp_path, monkeypatch):
    from semantic_switch.examples import ROLES
    from semantic_trainer import collect as module
    path, situations = collection_task(tmp_path, [envelope('please refund me', groups=['a'], tags=['direct'], role='check', pairs=['p']),
                                                  envelope('wrong invoice total', groups=['b']), envelope('maybe something', groups=['c']),
                                                  envelope('boom', groups=['d']), envelope('please refund me', groups=['e'], role='check')])
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge_from', lambda source, provider=None: teacher)
    joined, runtime = [], module.Runtime
    monkeypatch.setattr(module, 'Runtime', lambda models, **kwargs: joined.append(kwargs['job']) or runtime(models, **kwargs))
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    to = tmp_path / 'labels.collection.json'
    with api.model_job() as parent:
        report = module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', provider='Example', job=parent, window=2)
    counts = report['counts']
    assert joined == [parent] and counts['situations'] == 4 and counts['targets'] == 12 and counts['settled'] == 4
    assert counts['unresolved'] == {'unexamined': 4, 'missing': 0, 'failed': 2, 'rule': 0, 'unstated': 0, 'uncertain': 2, 'drift': 0}
    assert counts['settled'] + sum(counts['unresolved'].values()) == counts['targets'] and counts['complete_situations'] == 0 and report['review'][:2] == ['ss', 'label']
    labels, unresolved = files.read_jsonl(report['labels']), files.read_jsonl(report['unresolved'])
    first = next(row for row in labels if row['input'] == {'message': 'please refund me'})
    assert first['facts']['refund'] is True and first['groups'] == ['a', 'e'] and first['tags'] == ['direct'] and first['role'] == 'check' and first['pairs'] == ['p']
    assert all(source['origin'] == 'synthetic' and source['producer'] == 'openrouter:google/gemma-4-31b-it' for row in labels for value in row['label_sources'].values()
               for history in (value.values() if isinstance(value, dict) else [value]) for source in history)
    assert {row['outcome'] for row in unresolved} <= {'unexamined', 'failed', 'uncertain'} and all(row['address'][0] in ('refund', 'issues') for row in unresolved)
    reviewed = tmp_path / 'reviewed.jsonl'
    with Runtime({'judge': FakeScorer(oracle)}) as rt:
        assert tools.label(rt, path, report['labels'], to=str(reviewed), ask=lambda prompt: '', show=lambda text: None, reviewer='ana') == len(labels)
        exported = tools.export(rt, path, split='all', labels=reviewed)
    rows = read_export(exported['path'], fitting=False)[1]
    assert {row['label_origin'] for row in rows} == {'synthetic'} and any(row['tags'] == ['direct'] for row in rows)
    assert {(row['split'], row['fold']) for row in rows if row['group'] == 'group:a'} <= {(ROLES['check'][0], fold) for fold in ROLES['check'][1]}
    before = report['labels']
    again = module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', provider='Example')
    assert again['labels'] == before and Path(before).exists()


def test_collection_resumes_after_a_crash_and_failed_materialization_without_repeating_committed_work(tmp_path, monkeypatch):
    from semantic_trainer import collect as module
    rows = [envelope(f'please refund me {i}', groups=[f'g{i}']) for i in range(5)]
    path, situations = collection_task(tmp_path, [*rows[:3], envelope('crash now', groups=['x']), *rows[3:]])
    crashing = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge_from', lambda source, provider=None: crashing)
    to = tmp_path / 'labels.collection.json'
    with pytest.raises(KeyboardInterrupt):
        module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', window=1)
    committed = files.read_jsonl(tmp_path / 'labels.journal.jsonl')[1:]
    assert len(committed) == 3 and not to.exists()
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in [*rows[:3], envelope('calm now', groups=['x']), *rows[3:]]))
    with pytest.raises(ConfigError, match='different run'):
        module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', window=1)
    files.publish(situations, ''.join(json.dumps(row) + '\n' for row in [*rows[:3], envelope('crash now', groups=['x']), *rows[3:]]))
    def resuming(state, question, width):
        return oracle(state.replace('crash', 'calm'), question, width)
    resumed = FakeScorer(resuming)
    monkeypatch.setattr(module, 'judge_from', lambda source, provider=None: resumed)
    original = module.materialize
    def failing(*args):
        raise ConfigError('materialization failed')
    monkeypatch.setattr(module, 'materialize', failing)
    with pytest.raises(ConfigError, match='materialization failed'):
        module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', window=1)
    scored = len(resumed.requests)
    assert scored and len(files.read_jsonl(tmp_path / 'labels.journal.jsonl')) == 7 and not to.exists()
    monkeypatch.setattr(module, 'materialize', original)
    report = module.collect(path, situations, to, teacher='openrouter:google/gemma-4-31b-it', window=1)
    assert len(resumed.requests) == scored and report['counts']['situations'] == 6 and files.read_published(to)['execution']['completed'] == 6


def test_invalid_or_unversioned_sources_fail_before_any_teacher_request(tmp_path, monkeypatch):
    from semantic_trainer import collect as module
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge_from', lambda source, provider=None: teacher)
    for rows, match in (([envelope('fine'), {'message': 'raw domain input'}], r'situations.jsonl:2: not a schema-2 situation envelope'),
                        ([envelope('fine'), {'input': {'message': 'x'}, 'facts': {'refund': True}}], r'situations.jsonl:2: not a schema-2'),
                        ([envelope('x', role='train')], r'situations.jsonl:1: role must be one of'),
                        ([envelope('x', role='fit'), envelope('x', role='check')], 'duplicate decision input with roles fit and check')):
        path, situations = collection_task(tmp_path, rows)
        with pytest.raises(ConfigError, match=match):
            module.collect(path, situations, tmp_path / 'labels.collection.json', teacher='openrouter:google/gemma-4-31b-it')
        assert not teacher.requests
    path, situations = collection_task(tmp_path, [envelope(f'case {i}') for i in range(3)])
    with pytest.raises(ConfigError, match='more than 2 situations exceed the acquisition budget'):
        module.collect(path, situations, tmp_path / 'labels.collection.json', teacher='openrouter:google/gemma-4-31b-it', max_situations=2)
    assert not teacher.requests


def acquired(tmp_path, rows, **limits):
    from semantic_switch.training import Task
    path, situations = collection_task(tmp_path, rows)
    with Runtime({'judge': FakeScorer()}) as runtime:
        return Task(runtime.load(path, draft=True)).situations(situations, **{'max_situations': 2, 'max_bytes': 1 << 20, **limits})


def test_unique_situation_limit_counts_keys_and_merges_duplicate_lineage(tmp_path):
    merged = acquired(tmp_path, [envelope('a', groups=['g1'], tags=['first']), envelope('b', groups=['g2']), envelope('a', groups=['g3'], pairs=['p'], tags=['again'])])
    assert [row['input'] for row in merged] == [{'message': 'a'}, {'message': 'b'}]
    assert merged[0]['groups'] == ['g1', 'g3'] and merged[0]['pairs'] == ['p'] and merged[0]['tags'] == ['again', 'first'] and merged[1]['groups'] == ['g2']
    assert [row['input'] for row in acquired(tmp_path, [envelope('a', groups=['g1']), envelope('a', groups=['g2']), envelope('b')])] == [{'message': 'a'}, {'message': 'b'}]
    with pytest.raises(ConfigError, match=r'situations.jsonl:3: more than 2 situations exceed'):
        acquired(tmp_path, [envelope('a'), envelope('b'), envelope('c')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:5: more than 2 situations exceed'):
        acquired(tmp_path, [envelope('a'), envelope('b'), envelope('a'), envelope('b'), envelope('c')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:3: duplicate decision input with roles fit and check'):
        acquired(tmp_path, [envelope('a', role='fit'), envelope('b'), envelope('a', role='check')])
    with pytest.raises(ConfigError, match=r'situations.jsonl:2: duplicate decision input with roles fit and check'):
        acquired(tmp_path, [envelope('a', role='fit'), envelope('a', role='check')], max_situations=1)
    with pytest.raises(ConfigError, match='exceed the acquisition budget of 10'):
        acquired(tmp_path, [envelope('a'), envelope('a')], max_bytes=10)
    with pytest.raises(ConfigError, match='no situations to label'):
        acquired(tmp_path, [])


def test_collection_materializes_duplicates_at_the_unique_limit(tmp_path, monkeypatch):
    from semantic_trainer import collect as module
    path, situations = collection_task(tmp_path, [envelope('please refund me', groups=['a']), envelope('wrong invoice total', groups=['b']),
                                                  envelope('please refund me', groups=['c'], tags=['repeat'])])
    teacher = FakeScorer(oracle)
    monkeypatch.setattr(module, 'judge_from', lambda source, provider=None: teacher)
    report = module.collect(path, situations, tmp_path / 'labels.collection.json', teacher='openrouter:google/gemma-4-31b-it', max_situations=2)
    labels = files.read_jsonl(report['labels'])
    assert report['counts']['situations'] == 2 and len(labels) == 2
    refund = next(row for row in labels if row['input'] == {'message': 'please refund me'})
    assert refund['groups'] == ['a', 'c'] and refund['tags'] == ['repeat']


def test_shared_requests_count_once_and_returned_model_drift_is_an_outcome(tmp_path):
    from semantic_switch.judge import Request
    from semantic_switch.results import Fact, Prediction
    from semantic_switch.training import describe_task
    from semantic_trainer.collect import unit
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text'}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'Money back.'}, 'legal': {'check': 'Legal action.'}}})
    targets = describe_task(path)['targets']
    shared = Request('r1', 1., 1, 200, {'prompt_tokens': 3})
    def fact(model):
        return Fact(True, .9, True, 'judge', prediction=Prediction('value', 'true', True, .9), observed_models=(model,), requests=(shared,), status='ok')
    record = {'input': {'message': 'x'}}
    payload = unit(targets, record, {'refund': fact('m1'), 'legal': fact('m2')}, 'm1')
    assert len(payload['requests']) == 1 and payload['targets']['refund'] == {'value': True, 'models': ['m1']}
    assert payload['targets']['legal'] == {'outcome': 'drift', 'models': ['m2'], 'pinned': 'm1'} and payload['returned_models'] == ['m1', 'm2']


@pytest.mark.parametrize('teacher', ['openrouter:typesafe/jev-1.13', 'openrouter:TypeSafe/Jev-2'])
def test_collect_refuses_a_teacher_whose_terms_forbid_training_use(tmp_path, monkeypatch, teacher):
    from semantic_trainer import collect as module
    monkeypatch.setattr(module, 'judge_from', lambda *args, **kwargs: pytest.fail('a restricted teacher was contacted'))
    with pytest.raises(ConfigError, match=r'TypeSafe agreement §2.3\(b\).*gemma-4-31b-it'):
        module.collect(tmp_path / 'x.yaml', tmp_path / 'x.jsonl', tmp_path / 'labels.collection.json', teacher=teacher)


def test_prepare_assigns_family_roles_once_and_reports_support(tmp_path, monkeypatch):
    from semantic_trainer.jobs import TUNING, prepare, resolve
    manifest, definition, export, _ = task(tmp_path, monkeypatch)
    result = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE, acceptance_fit=True)
    resolved, observations = resolve(result['job'])
    assert result['job_id'] == resolved['id'] and resolved['task']['revisions'] == {'action': observations[0]['meaning_rev']}
    assert resolved['execution'] == observations[0]['native']['execution'] and {row['task_family'] for row in observations} == {'maintenance:test'}
    _, rows = read_export(export, native=True)
    for role in TUNING:
        assert result['roles'][role]['observations'] == sum(role_of(row['split'], row['fold']) == role for row in rows)
        assert result['roles'][role]['situations'] == result['roles'][role]['observations'] == result['roles'][role]['complete_situations']
    assert {row['id'] for row in observations if row['role'] == 'fit'} == {row['id'] for row in rows if role_of(row['split'], row['fold']) == 'fit'}
    fit = result['support']['roles']['fit']['action']
    assert set(fit) == set(VALUES) and all(set(tags) <= {'relayed', 'quoted'} for tags in fit.values())
    assert all(len(row['sources']) == 1 and row['sources'][0]['task_family'] == 'maintenance:test' for row in observations)
    assert result['commands']['select'] == ['semantic-trainer', 'select', result['job'], '--run', str(tmp_path / 'job' / 'run')]
    with pytest.raises(ConfigError, match='already assigned'):
        prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    definition.write_text(definition.read_text().replace('Which response does the policy require?', 'Which response does the changed policy require?'))
    assert resolve(result['job'])[0]['id'] == resolved['id']
    with pytest.raises(ConfigError, match='changed or left the task definition'):
        prepare(definition, [export], tmp_path / 'changed', assets_manifest=manifest, recipe=RECIPE)
    assert not (tmp_path / 'changed' / 'task').exists() and not (tmp_path / 'changed' / 'job.yaml').exists()
    frozen = tmp_path / 'job' / 'task' / 'task.yaml'
    frozen.chmod(0o644)
    frozen.write_text(frozen.read_text() + '\n')
    with pytest.raises(ConfigError, match='frozen task changed since preparation'):
        resolve(result['job'])


def test_prepare_fails_on_a_declared_answer_without_fit_support(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare
    manifest, definition, export, _ = task(tmp_path, monkeypatch, values=VALUES[:2])
    with pytest.raises(ConfigError, match='without fit support: action: clarify'):
        prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)
    assert not (tmp_path / 'job' / 'job.yaml').exists()


def test_role_crossing_family_join_fails(tmp_path, monkeypatch):
    from semantic_trainer.jobs import TUNING, prepare
    manifest, definition, export, records = task(tmp_path, monkeypatch)
    rows, roles = files.read_jsonl(records), {row['group']: role_of(row['split'], row['fold']) for row in read_export(export, native=True)[1]}
    index = next(i for i, row in enumerate(rows) if roles.get(f"group:{row['groups'][0]}") in TUNING)
    role = next(role for role in TUNING if role != roles[f"group:{rows[index]['groups'][0]}"])
    rows[index] = {**rows[index], 'groups': ['moved'], 'role': role}
    files.publish(tmp_path / 'other.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime({'judge': FakeScorer()}) as runtime:
        other = tools.export(runtime, definition, to=tmp_path / 'other.export.jsonl', native=True, task_family='maintenance:test', labels=tmp_path / 'other.jsonl')
    with pytest.raises(ConfigError, match='conflicting roles'):
        prepare(definition, [export, other['path']], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE)


def test_prepare_takes_explicit_roles_and_never_fits_or_selects_on_check(tmp_path, monkeypatch):
    from examples import adapt_judgment
    from semantic_trainer.jobs import prepare, resolve, role_rows
    manifest, explicit = pipeline_base(tmp_path, monkeypatch), ('fit', 'model_select', 'fit', 'acceptance_fit', 'check')
    records = label_file(tmp_path, n=40)
    files.publish(records, ''.join(json.dumps({**row, 'role': explicit[i % 5]}) + '\n' for i, row in enumerate(files.read_jsonl(records))))
    with Runtime({'judge': FakeScorer()}) as runtime:
        tuned = adapt_judgment.prepare(runtime, tmp_path / 'task', records, 'maintenance:test')
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


FITTED = {'targets': {'model.language_model.layers.0.self_attn.q_proj': [8, 4]},
          'recipe': {'rank': 2, 'alpha': 4, 'dropout': 0.0}, 'model': {'revision': 'c' * 40}, 'dependencies': {'peft': '0.21.2'}}
PEFT_CONFIG = {'task_type': 'CAUSAL_LM', 'peft_type': 'LORA', 'auto_mapping': None, 'peft_version': '0.21.2', 'base_model_name_or_path': '/assets/hf/gemma',
               'revision': None, 'inference_mode': True, 'r': 8, 'target_modules': [], 'exclude_modules': None, 'lora_alpha': 16, 'lora_dropout': 0.0,
               'fan_in_fan_out': False, 'bias': 'none', 'use_rslora': False, 'modules_to_save': None, 'init_lora_weights': True, 'layers_to_transform': None,
               'layers_pattern': None, 'rank_pattern': {}, 'alpha_pattern': {}, 'megatron_config': None, 'megatron_core': 'megatron.core',
               'trainable_token_indices': None, 'loftq_config': {}, 'eva_config': None, 'corda_config': None, 'lora_ga_config': None, 'use_dora': False,
               'velora_config': None, 'alora_invocation_tokens': None, 'use_qalora': False, 'qalora_group_size': 16, 'monteclora_config': None,
               'layer_replication': None, 'lora_bias': False, 'target_parameters': None, 'use_bdlora': None, 'arrow_config': None, 'kasa_config': None,
               'ensure_weight_tying': False}


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
    written = {**PEFT_CONFIG, 'r': fitted['recipe']['rank'], 'lora_alpha': fitted['recipe']['alpha'], 'lora_dropout': fitted['recipe']['dropout'],
               'target_modules': sorted(fitted['targets'], reverse=True), **config}
    files.publish(directory / 'adapter_config.json', {key: value for key, value in written.items() if value is not ...})
    return directory


def test_deployment_merges_into_the_f16_base_then_quantizes_and_checks_the_zero_adapter(tmp_path, monkeypatch):
    from semantic_switch import model_job
    from semantic_trainer import assets
    from semantic_trainer.deploy import conversion_bytes, verify_zero
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
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
    embedded = {**FITTED, 'targets': {'model.embed_tokens': [8, 4]}}
    with pytest.raises(ConfigError, match='unsupported conversion target'):
        verify_zero(embedded, located, peft_adapter(tmp_path / 'embedded', embedded), 'Q4_0', tmp_path, job)
    stack.close()


def trial(name, step, observation, situation, regressions=(), base=False, nll=None):
    return {'name': name, 'step': step, 'correct': {'observation': observation, 'complete_situation': situation}, 'regressions': list(regressions), 'base': base, 'nll': nll}


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
    situations = {'a': {'role': 'model_select', 'complete': True, 'targets': {'x': 'o1', 'y': 'o2', 'z': 'exact'}},
                  'b': {'role': 'model_select', 'complete': True, 'targets': {'x': 'o3', 'y': 'o4'}},
                  'c': {'role': 'model_select', 'complete': False, 'targets': {'x': 'o5', 'y': 'unlabelled'}},
                  'd': {'role': 'fit', 'complete': True, 'targets': {'x': 'o6'}}}
    correct = {'o1': True, 'o2': True, 'o3': True, 'o4': None, 'o5': True}
    assert unit_correct(correct, situations, 'model_select') == {'observation': 4, 'complete_situation': 1, 'situations': 2, 'observations': 5}


def test_summary_keeps_member_addressed_pairs_and_reports_prior_drift():
    from semantic_trainer.select import summary
    rows = [{'id': f'{member}{i}', 'fact': 'issues', 'member': member, 'classes': ['true', 'false'], 'label': i, 'group': f'raw{i}', 'component': f'g{i}', 'pairs': ['p']}
            for member in ('refund', 'billing') for i in (0, 1)]
    def scored(correct):
        return [{'id': row['id'], 'fact': 'issues', 'member': row['member'], 'correct': ok, 'accepted': False, 'gold_class': row['classes'][row['label']],
                 'answer': row['classes'][row['label']] if ok else row['classes'][1 - row['label']], 'ms': 1.} for row, ok in zip(rows, correct)]
    base = summary('base', 0, 'a' * 64, scored([True, True, True, False]), rows, {})
    candidate = summary('step-1', 1, 'b' * 64, scored([False, True, True, True]), rows, {}, base)
    assert base['pairs'] == {'pairs': 2, 'both_correct': 1} and candidate['pairs'] == {'pairs': 2, 'both_correct': 1} and base['base'] and not candidate['base']
    assert candidate['per_class']['issues/refund'] == {'true': {'gold': 1, 'predicted': 0, 'correct': 0}, 'false': {'gold': 1, 'predicted': 2, 'correct': 1}}
    assert candidate['class_regressions'] == candidate['regressions'] == ['issues/refund:true']
    assert candidate['family_fixes'] == ['g1'] and candidate['family_regressions'] == ['g0'] and candidate['raw_correct'] == 3 == candidate['correct']['observation']


ACTION_TEXT = ('Request a technician', 'Continue operation', 'Ask whether')
REALIZED = {'device': 'cpu', 'tensor_types': ['q4_0'], 'flash_attn': 'enabled', 'type_k': 'f16', 'type_v': 'f16', 'use_extra_bufts': True, 'repacked': [],
            'n_ubatch': 64, 'n_ctx': 2048, 'n_batch': 512, 'n_seq_max': 6, 'n_outputs_max': 1, 'swa_full': True, 'kv_unified': True, 'prefix_slots': 4,
            'scheduling': 'serial@1', 'shared': True, 'grouping': 'serial', 'serial': 1}


class Native(FakeScorer):
    realized = REALIZED
    built = []

    def __init__(self, path, **kwargs):
        adapted = Path(path).read_bytes().startswith(b'merged step-8')
        def oracle(state, question, width):
            found = re.search(r'report (\d+)', state)
            options = letters_of(question)
            wanted = VALUES[int(found.group(1)) % len(VALUES)] if found and adapted else None
            hits = [letter for text, letter in options.items() if wanted and text.startswith(dict(zip(VALUES, ACTION_TEXT))[wanted])]
            return [0.98 if chr(65 + k) in hits else 0.02 / max(width - 1, 1) for k in range(width)] if hits else [1 / width] * width
        super().__init__(oracle)
        self.vocab, self.inv = Native.tokens.vocab, Native.tokens.inv
        self.path, self.shared, self.weights, self.closed = path, True, SimpleNamespace(hybrid=False), 0
        Native.built.append(self)

    def capacity(self):
        return dict(self.realized)

    def close(self):
        self.closed += 1


def fitted_run(tmp_path, job, resolved, observations, base_sha):
    from semantic_trainer.adapter import identify
    run = tmp_path / 'run'
    fitted = {**resolved, **{key: FITTED[key] for key in ('targets', 'dependencies')}}
    selected = {row['id']: row['label'] for row in observations if row['role'] == 'model_select'}
    candidates = []
    for step, name in ((0, 'zero'), (8, 'step-8'), (16, 'step-16')):
        (run / 'candidates' / name).mkdir(parents=True)
        candidates.append({'name': name, 'step': step, 'examples_seen': 8 * step, 'correct': {}, 'nll': 1., 'peft': f'candidates/{name}',
                           'predictions': {i: {'answer': label if step == 8 else 0, 'stages': {}} for i, label in selected.items()},
                           **identify(fitted, peft_adapter(run / 'candidates' / name, fitted, seed=step))})
    files.publish(run / 'resolved-job.json', fitted)
    from semantic_trainer.lora import CANDIDATES_SCHEMA, RUN_SCHEMA
    files.publish(run / 'candidates.json', {'schema': CANDIDATES_SCHEMA, 'kind': 'lora-candidates', 'state': 'ready', 'job_id': resolved['id'], 'fit_id': resolved['fit']['id'],
                                            'monitor_id': resolved['monitor']['id'], 'assets': resolved['assets'], 'loss': resolved['loss'],
                                            'base_sha256': base_sha, 'unit': resolved['recipe']['selection_unit'], 'stop': {'reason': 'patience'}, 'candidates': candidates})
    files.publish(run / 'run.json', {'schema': RUN_SCHEMA, 'state': 'ready', 'job_id': resolved['id']})
    return run


def selection(tmp_path, monkeypatch, *, acceptance_fit=False, author=None, amend=None):
    from semantic_switch import backend
    from semantic_trainer import select as module
    from semantic_trainer.jobs import prepare, resolve
    identity = api.Runtime.identity
    manifest, definition, export, records = task(tmp_path, monkeypatch, n=60, author=author)
    monkeypatch.setattr(api.Runtime, 'identity', identity)
    if amend is not None:
        amend(definition)
    job = prepare(definition, [export], tmp_path / 'job', assets_manifest=manifest, recipe=RECIPE,
                  native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048}, acceptance_fit=acceptance_fit)['job']
    resolved, observations = resolve(job)
    run = fitted_run(tmp_path, job, resolved, observations, files.sha256_file(resolved['model']['gguf']))
    deployed = []
    def deploy(fitted, located, peft, destination, quant, handle):
        deployed.append((Path(peft).name, quant, handle.state))
        Path(destination).write_bytes(b'merged ' + Path(peft).name.encode())
        return files.sha256_file(destination)
    monkeypatch.setattr(module, 'deploy', deploy)
    monkeypatch.setattr(module, 'verify_zero', lambda fitted, located, peft, quant, work, handle: deployed.append(('zero', quant, handle.state)))
    monkeypatch.setattr(backend, 'LlamaScorer', Native)
    monkeypatch.setattr(api.Model, '_host_bytes', lambda self: {'weights': 1})
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    monkeypatch.setattr(module, 'conversion_bytes', lambda located: {'conversion': 1})
    Native.built, Native.realized = [], REALIZED
    return module, job, run, {**resolved, 'roles': {role: [row['id'] for row in observations if row['role'] == role] for role in ('fit', 'model_select', 'acceptance_fit')}}, records, deployed, definition


def check_labels(tmp_path, n=12):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.examples import label_record
    definition = tmp_path / 'task' / 'action.yaml'
    plan = compile_spec(files.read_yaml(definition), definition.name, definition.parent)
    path = tmp_path / 'check.jsonl'
    files.publish(path, ''.join(json.dumps(label_record(plan, {'message': f'report {1000 + i}'}, {'action': VALUES[(1000 + i) % 3]}, groups=[f'check-{i}'],
                                                        sources={'action': [{'origin': 'human', 'producer': 'reviewer'}]}, role='check')) + '\n' for i in range(n)))
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
    presentations = sum(len(row['native']['presentations']) for row in resolved_rows(job, 'model_select'))
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
    assert set(data['files']) == {'specialist.gguf', 'task.yaml', 'task.lock.json', 'guidance/0.txt'}
    assert data['execution'] == execution_identity(REALIZED) and data['native'] == {**data['native'], 'adapter': 'qwen3', 'device': 'cpu'}
    assert data['acceptance'] == {**data['acceptance'], 'policy': 'authored', 'calibration': 'absent', 'acceptance_fit': {}} and not resolved['roles']['acceptance_fit']
    lock = files.read_json(run / 'deployment' / 'task.lock.json')
    assert data['lock'] == {'path': 'task.lock.json', 'id': lock['id'], 'candidate_id': lock['candidate_id']}
    assert {side['status'] for fact in lock['facts'].values() for side in fact['policies']['judge'].values()} == {'uncalibrated_default'}
    uses = {row['id']: row['uses'] for row in data['provenance']['rows']}
    assert receipt.provenance.kind == 'specialization' and all(uses[i] == ['adaptation', 'selection'] for i in resolved['roles']['fit'])
    assert all(uses[i] == ['selection'] for i in resolved['roles']['model_select'])
    assert {source['export_id'] for row in data['provenance']['rows'] for source in row['sources']} == {resolved['sources'][0]['export_id']}
    memberships = {row['id']: row['label_sources'] for row in resolved['memberships']}
    assert all(row['label_sources'] == memberships[row['id']] and row['label_sources'] for row in data['provenance']['rows'])
    moved = tmp_path / 'moved'
    shutil.move(run / 'deployment', moved)
    shutil.rmtree(tmp_path / 'task')
    with Runtime.from_specialization(moved / 'receipt.json') as runtime:
        switch = runtime.load()
        assert switch.snapshot.lock_id == lock['id'] and runtime.models['judge'].provenance == receipt.provenance
        assert switch({'message': 'report 4'}).facts['action'].value == VALUES[1]
    with Runtime({'judge': api.Model.from_specialization(moved / 'receipt.json')}) as model_only:
        with pytest.raises(ConfigError, match='Runtime.from_specialization'):
            model_only.load()
    Native.realized = {**REALIZED, 'n_batch': 64}
    with Runtime.from_specialization(moved / 'receipt.json') as runtime:
        with pytest.raises(ConfigError, match='judge execution changed'):
            runtime.load()


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
            with Runtime.from_specialization(receipt) as runtime:
                assert runtime.load()({'message': 'report 4'}).facts['action'].value == VALUES[1]
            gguf = receipt.parent / load_receipt(receipt).data['gguf']
            gguf.write_bytes(gguf.read_bytes() + b'changed')
            with pytest.raises(ConfigError, match='changed since selection'):
                Runtime.from_specialization(receipt)
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
    with Runtime.from_specialization(result['receipt']) as runtime:
        assert thaw(runtime.load().snapshot.sides['judge']['action']) == frozen
    labels, evidence = assigned(tmp_path / 'evidence', definition, files.read_jsonl(check_labels(tmp_path)))
    before = {path.name: files.sha256_file(path) for path in (run / 'deployment').iterdir() if path.is_file()}
    audit = Path(EvidenceContext.from_manifest(evidence).audit)
    def failing(*args, **kwargs):
        assert files.read_jsonl(audit)[-1]['plan']['arms']['specialist']['receipt'] == receipt.id
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
    assert report['receipt'] == receipt.id and report['lock'] == receipt.data['lock']['id'] and report['situations'] == 12 and report['families'] == 12
    assert report['facts']['action']['labelled'] == 12 and report['lineage']['judge:action']['state'] == 'declared_disjoint'
    assert report['exposure']['state'] == 'repeat' and report['exposure']['independent'] and len(files.read_jsonl(audit)) == 1
    assert {path.name: files.sha256_file(path) for path in (run / 'deployment').iterdir() if path.is_file()} == before
    overlapping, overlap_evidence = assigned(tmp_path / 'overlap-evidence', definition, [{**row, 'role': 'check'} for row in files.read_jsonl(records)[:3]])
    assert checking.check_deployment(result['receipt'], overlapping, overlap_evidence, tmp_path / 'overlap')['lineage']['judge:action']['state'] == 'known_overlap'


def assigned(directory, definition, rows):
    from semantic_switch.tools.data import Labels
    paths = files.Paths.of(directory / Path(definition).name)
    files.write_new(paths.yaml, Path(definition).read_text())
    files.write_new(paths.test, ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime({'judge': FakeScorer()}) as runtime:
        Labels(runtime.load(paths.yaml, draft=True))
    return paths.test, paths.dataset


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
    with pytest.raises(ConfigError, match='changed since .* was fitted'):
        module.select(job, run)
    files.publish(run / 'candidates.json', {**listed, 'loss': {**listed['loss'], 'version': 1}, 'base_sha256': files.sha256_file(resolved['model']['gguf'])})
    with pytest.raises(ConfigError, match='fitted under training objective .*refit'):
        module.select(job, run)


def test_prepare_chooses_whole_fit_families_for_an_exact_situation_count(tmp_path, monkeypatch):
    from examples import adapt_judgment
    from semantic_trainer.jobs import prepare, resolve
    manifest = pipeline_base(tmp_path, monkeypatch)
    records = label_file(tmp_path, n=90)
    files.publish(records, ''.join(json.dumps({**row, 'groups': [f'family-{i // 3}'], 'role': ('fit', 'fit', 'model_select')[i // 3 % 3]}) + '\n'
                                   for i, row in enumerate(files.read_jsonl(records))))
    with Runtime({'judge': FakeScorer()}) as runtime:
        export = adapt_judgment.prepare(runtime, tmp_path / 'task', records, 'maintenance:test')['path']
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
    from examples import adapt_judgment
    with Runtime({'judge': FakeScorer()}) as runtime:
        adapt_judgment.prepare(runtime, tmp_path / 'task', records, 'maintenance:test')
    definition = tmp_path / 'task' / 'action.yaml'
    exports = []
    for name, part in (('first', rows[:45]), ('second', rows[45:])):
        files.publish(tmp_path / f'{name}.jsonl', ''.join(json.dumps(row) + '\n' for row in part))
        with Runtime({'judge': FakeScorer()}) as runtime:
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
    from semantic_switch.examples import each_target, label_record
    from semantic_trainer.jobs import prepare
    manifest = pipeline_base(tmp_path, monkeypatch)
    path = tmp_path / 'pair.yaml'
    files.publish(path, {'name': 'pair', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}, 'legal': {'check': 'The customer threatens legal action.'}}})
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    source = [{'origin': 'human', 'producer': 'fixture'}]
    rows = []
    for i in range(30):
        facts = {'refund': bool(i % 2), **({'legal': bool(i // 2 % 2)} if i % 5 else {})}
        rows.append(label_record(plan, {'message': f'case {i}'}, facts, sources=each_target(facts, source), groups=[f'g{i}'],
                                 role=('fit', 'fit', 'model_select')[i % 3]))
    files.publish(tmp_path / 'labels.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    with Runtime({'judge': FakeScorer()}) as runtime:
        export = tools.export(runtime, path, native=True, task_family='support:pair', labels=tmp_path / 'labels.jsonl')['path']
    with pytest.raises(ConfigError, match='complete_situation selection needs gold-complete model_select situations'):
        prepare(path, [export], tmp_path / 'job', assets_manifest=manifest, recipe={**RECIPE, 'selection_unit': 'complete_situation'})
    assert prepare(path, [export], tmp_path / 'observation-job', assets_manifest=manifest, recipe=RECIPE)['recipe']['selection_unit'] == 'observation'


def test_transfer_preparation_keeps_per_source_task_families_and_objective_aware_support(tmp_path, monkeypatch):
    from semantic_switch.compiler import compile_spec
    from semantic_switch.examples import each_target, label_record
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
    source = [{'origin': 'rule', 'producer': 'printer-reference'}]
    def export(name, fact, role, family, values):
        rows = [label_record(plan, {'message': f'{name} report {i}'}, {fact: values[i % 2]}, sources=each_target({fact: values[i % 2]}, source),
                             groups=[f'{name}-{i}'], role=role) for i in range(12)]
        labels = tmp_path / f'{name}.jsonl'
        files.publish(labels, ''.join(json.dumps(row) + '\n' for row in rows))
        with Runtime({'judge': FakeScorer()}) as runtime:
            return tools.export(runtime, path, to=tmp_path / f'{name}.export.jsonl', native=True, labels=labels, fact=fact,
                                **({'task_family': family} if family else {}))['path']
    fit = export('a', 'next_a', 'fit', 'printer:a', ('calibrate', 'resume'))
    held = export('c', 'next_c', 'model_select', 'printer:c', ('resume', 'calibrate'))
    result = prepare(path, [fit, held], tmp_path / 'transfer', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')
    resolved, observations = resolve(result['job'])
    assert result['support']['absent_fit_support'] == ['next_c'] and {row['task_family'] for row in observations if row['role'] == 'fit'} == {'printer:a'}
    assert {(source['task_family'], role) for source in resolved['sources'] for role in source['roles']} == {('printer:a', 'fit'), ('printer:c', 'model_select')}
    assert all('Document C' not in row['text'] and 'Document A' not in row['text'] for row in observations)
    with pytest.raises(ConfigError, match='without fit support: next_c'):
        prepare(path, [fit, held], tmp_path / 'specialization', assets_manifest=manifest, recipe=RECIPE)
    bare = export('b', 'next_c', 'model_select', None, ('resume', 'calibrate'))
    with pytest.raises(ConfigError, match='exactly one namespaced task family'):
        prepare(path, [fit, bare], tmp_path / 'bare', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')
    files.publish(path, {**files.read_yaml(path), 'guidance': {'a': {'text': 'Document A: calibrate after loading new media.'},
                                                               'c': {'text': 'Document C revised: calibrate after every job.'}}})
    assert resolve(result['job'])[0]['id'] == resolved['id'] and 'Document C revised' not in (tmp_path / 'transfer' / 'task' / 'guidance' / '1.txt').read_text()
    with pytest.raises(ConfigError, match='next_c changed or left the task definition'):
        prepare(path, [fit, held], tmp_path / 'revised', assets_manifest=manifest, recipe=RECIPE, objective='instruction_transfer')
