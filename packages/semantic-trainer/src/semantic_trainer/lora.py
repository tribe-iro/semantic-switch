from __future__ import annotations

from importlib.metadata import version
import math
import platform
from pathlib import Path
import shutil
import tempfile
import time

from semantic_switch import ConfigError
from semantic_switch.training import Engine, artifact_digest, estimate, files, model_job, native_model, native_score_requests, peak

from . import recovery
from .adapter import CONFIG, WEIGHTS, descriptor, effective, identify, tensor_shapes
from .jobs import (CANDIDATES_SCHEMA, LOSS, MODULES, PROFILE_KIND, PROFILE_SCHEMA, RUN_SCHEMA, advance, due, fitting_device, plateaued, population_of, profile_configuration,
                   profile_workload, projection, ranking_rule, resolve)
from .monitor import Deadline, Exhausted, Infeasible, Monitor, checkpoint, feasibility, peak_host_bytes, retain, save, stopped
from .objective import NumericalFailure, Phases, batches, distribution, input_bytes, optimizer_of, staged, update

HOST_TABLE = 'model.language_model.embed_tokens_per_layer'
SPARSE_FAMILIES = 10


def dependencies():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ConfigError('install fitting dependencies: pip install "semantic-trainer[lora]"') from error
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model


def hf_tensors(resolved):
    for path in Path(resolved['model']['hf']).glob('*.safetensors'):
        yield from tensor_shapes(path).items()


def hf_bytes(resolved):
    return sum(math.prod(shape) for _, shape in hf_tensors(resolved)) * {'float32': 4, 'bfloat16': 2}[resolved['fitting']['precision']]


def fitting_envelope(resolved, observations):
    return {**fit_bytes(resolved), 'fit:prepared_inputs': input_bytes(row for row in observations if row['role'] in ('fit', 'model_select'))}


def fit_bytes(resolved):
    recipe = resolved['recipe']
    suffixes = tuple(f'.{MODULES[module][0]}.{module}.weight' for module in recipe['target_modules'])
    trainable = sum(recipe['rank'] * sum(shape) for name, shape in hf_tensors(resolved) if name.endswith(suffixes) and len(shape) == 2)
    return {'fit:hf_weights': hf_bytes(resolved), 'fit:adapter_parameters_gradients_adamw': 16 * trainable}


def place(model, device):
    owner, name = HOST_TABLE.rsplit('.', 1)
    owner = model.get_submodule(owner)
    table = owner.get_submodule(name)
    delattr(owner, name)
    model.to(device)
    owner.add_module(name, table)
    return {'': str(device), HOST_TABLE: 'cpu'}


def map_host_table(model, hf):
    import numpy as np
    import torch
    table, name = model.get_submodule(HOST_TABLE), HOST_TABLE + '.weight'
    found = next((found for found in map(descriptor, sorted(Path(hf).glob('*.safetensors'))) if name in found.tensors), None)
    if found is None:
        raise ConfigError(f'{hf}: {name} is not in the safetensors snapshot')
    entry = found.tensors[name]
    if entry.dtype != 'BF16' or table.weight.dtype != torch.bfloat16 or tuple(table.weight.shape) != entry.shape:
        return 'cpu'
    table.weight = torch.nn.Parameter(torch.from_numpy(found.view(name, np.uint16)).view(torch.bfloat16), requires_grad=False)
    return 'cpu:file-mapped'


def verify_tokenizer(tokenizer, rows):
    descriptors = {artifact_digest(native['tokenizer']): native['tokenizer'] for native in {id(row['native']): row['native'] for row in rows}.values()}
    if len(descriptors) != 1:
        raise ConfigError('exports have different tokenizers')
    (descriptor_,) = descriptors.values()
    vocab = tokenizer.get_vocab()
    ordered = [None] * descriptor_['vocab_size']
    for token, index in vocab.items():
        if not 0 <= index < len(ordered) or ordered[index] is not None:
            raise ConfigError('HF/deployment tokenizer vocabulary mismatch')
        ordered[index] = token
    for index, token in descriptor_.get('padding_tokens', ()):
        if ordered[index] is None:
            ordered[index] = token
    if None in ordered or artifact_digest(ordered) != descriptor_['vocab_sha256']:
        raise ConfigError('HF/deployment tokenizer vocabulary mismatch')
    for probe in descriptor_['probes']:
        if tokenizer.encode(probe['text'], add_special_tokens=False) != probe['ids']:
            raise ConfigError('HF/deployment tokenizer token IDs mismatch')
    pairs = {}
    for answers in {id(entry.answers): entry.answers for row in rows for entry in row['presentations']}.values():
        for group, expected in zip(answers.slots, answers.text):
            for token, text in zip(group, expected):
                if pairs.setdefault(token, text) != text:
                    raise ConfigError('HF/deployment answer-token decoding mismatch')
    if any(tokenizer.decode([token], skip_special_tokens=False) != text for token, text in pairs.items()):
        raise ConfigError('HF/deployment answer-token decoding mismatch')
    for name, token in descriptor_['special'].items():
        if token >= 0 and (name != 'bos' or tokenizer.bos_token_id is not None) and getattr(tokenizer, name + '_token_id', None) != token:
            raise ConfigError(f'HF/deployment {name} token mismatch')
    return descriptor_


def targets(model, modules=tuple(MODULES)):
    from collections import Counter
    model_type = model.config.model_type
    if model_type not in ('qwen3', 'gemma4'):
        raise ConfigError(f'unsupported LoRA conversion profile {model_type!r}; supported: qwen3 and gemma4 language projections')
    suffixes = tuple(f'.{MODULES[module][0]}.{module}' for module in modules)
    matched = {name: list(module.weight.shape) for name, module in model.named_modules() if name.endswith(suffixes) and hasattr(module, 'weight')}
    if model_type == 'gemma4':
        matched = {name: shape for name, shape in matched.items() if '.language_model.' in name}
        config = model.config.text_config
        owned = config.num_hidden_layers - config.num_kv_shared_layers
        expected = {'k_proj': owned, 'v_proj': sum(not (config.attention_k_eq_v and kind != 'sliding_attention') for kind in config.layer_types[:owned])}
    else:
        config, expected = model.config, {}
    counts = Counter(name.rsplit('.', 1)[1] for name in matched)
    if not matched or any(counts[module] != expected.get(module, config.num_hidden_layers) for module in modules):
        raise ConfigError(f'{list(modules)} target resolution does not cover the pinned architecture')
    return matched


def native_predictions(runtime, rows, replay=False, role='judge'):
    import numpy as np
    requests = native_score_requests(rows)
    block, reuse = runtime.replay(requests, role) if replay else (runtime.score_native(requests, role=role), None)
    if any(block.error):
        raise ConfigError(f'native scoring failed: {block.error}')
    if not np.isfinite(block.log_mass).all() or any(not np.isfinite(block.logp[i, :len(req.slots[0])]).all() for i, req in enumerate(requests)):
        raise ConfigError('native scoring produced nonfinite answer slots or mass')
    logp = [block.logp[i, :len(req.slots[0])].astype(float).tolist() for i, req in enumerate(requests)]
    mass, out, at = block.log_mass[:len(requests)].astype(float).tolist(), {}, 0
    for row in rows:
        count = len(row['presentations'])
        chosen = {entry.id: int(np.argmax(values)) for entry, values in zip(row['presentations'], logp[at:at + count])}
        out[row['id']] = {**staged(row['presentations'], chosen), 'probability': distribution(row, logp[at:at + count], mass[at:at + count])}
        at += count
    return out, reuse


def native_answers(resolved, rows, job):
    with Engine(native_model(resolved), job=job) as rt:
        return native_predictions(rt, rows)[0], rt.identity()


def agreed(rows, native, monitored):
    out = {}
    for row in rows:
        a, b = native.get(row['id']), monitored.get(row['id'])
        stages = sorted({entry.stage for entry in row['presentations'] if entry.variant == 'canonical'})
        out[row['id']] = {'answer': a is not None and b is not None and a['answer'] == b['answer'],
                          'stages': {stage: a is not None and b is not None and stage in a['stages'] and stage in b['stages'] and a['stages'][stage] == b['stages'][stage]
                                     for stage in stages}}
    return out


def agreement(rows, native, monitored):
    flags, stages = agreed(rows, native, monitored), {}
    for value in flags.values():
        for stage, same in value['stages'].items():
            counts = stages.setdefault(stage, {'agreed': 0, 'submitted': 0})
            counts['submitted'] += 1
            counts['agreed'] += same
    disagreements = sorted(key for key, value in flags.items() if not value['answer'])
    return {'agreed': len(flags) - len(disagreements), 'submitted': len(flags), 'disagreements': disagreements, 'stages': stages}


def compatibility(rows, base, specialist):
    before, after = agreed(rows, *base), agreed(rows, *specialist)
    families, changed = {}, set()
    for row in rows:
        delta = after[row['id']]['answer'] - before[row['id']]['answer']
        families.setdefault(row['component'], []).append(delta)
        if delta:
            changed.add(row['component'])
    native, monitored = specialist
    def right(outputs, row):
        return outputs.get(row['id']) is not None and outputs[row['id']]['answer'] == row['label']
    disagreeing = [row for row in rows if not after[row['id']]['answer']]
    return {'base': agreement(rows, *base), 'specialist': agreement(rows, *specialist),
            'difference': {'family_mean': sum(sum(values) / len(values) for values in families.values()) / len(families) if families else None,
                           'families': len(families), 'targets': len(rows), 'changed_families': len(changed),
                           'specialist_only_agreed': sum(after[i]['answer'] and not before[i]['answer'] for i in after),
                           'base_only_agreed': sum(before[i]['answer'] and not after[i]['answer'] for i in after)},
            'status': 'unchanged' if not changed else 'sparse' if len(changed) < SPARSE_FAMILIES else 'measured', 'tolerance': None, 'gate': False,
            'disagreements_against_gold': {'targets': len(disagreeing), 'native_fixes': sum(right(native, r) and not right(monitored, r) for r in disagreeing),
                                           'native_regressions': sum(right(monitored, r) and not right(native, r) for r in disagreeing)},
            'intermediate': None}


def accelerator(torch, device):
    return getattr(torch, device.type) if device.type in ('xpu', 'cuda') else None


def adapted(resolved, config):
    torch, _, AutoModel, _, LoraConfig, get_peft_model = dependencies()
    recipe, fitting = resolved['recipe'], resolved['fitting']
    device = torch.device(fitting_device(fitting))
    torch.manual_seed(recipe['seed'])
    torch.set_num_threads(fitting['threads'])
    if config.model_type == 'gemma4':
        from transformers import AutoModelForImageTextToText
        AutoModel = AutoModelForImageTextToText
    host_table = config.model_type == 'gemma4' and device.type != 'cpu'
    base = AutoModel.from_pretrained(resolved['model']['hf'], local_files_only=True, dtype=getattr(torch, fitting['precision']),
                                     attn_implementation=fitting['attention'], device_map={'': 'cpu' if host_table else fitting['device']})
    placed = {**place(base, device), HOST_TABLE: map_host_table(base, resolved['model']['hf'])} if host_table else {'': fitting['device']}
    matched = targets(base, recipe['target_modules'])
    base.requires_grad_(False)
    if fitting['gradient_checkpointing']:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    initialization = resolved['initialization']
    if initialization['kind'] == 'fresh':
        model = get_peft_model(base, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'], lora_dropout=recipe['dropout'],
                                              target_modules=list(matched), bias='none', task_type='CAUSAL_LM'))
    else:
        model = initialized(base, initialization, matched)
    return torch, model, device, placed, matched


def initialized(base, initialization, matched):
    from peft import PeftModel
    if initialization['targets'] != matched:
        raise ConfigError(f"initializer {initialization['candidate']} adapts {sorted(initialization['targets'])[:3]}..., not the resolved targets of this base; "
                          'warm start never resizes or retargets an adapter')
    peft = Path(initialization['run']) / initialization['path']
    if files.sha256_file(peft / WEIGHTS) != initialization['peft_sha256'] or files.sha256_file(peft / CONFIG) != initialization['config_sha256']:
        raise ConfigError(f'{peft}: initializer adapter changed since preparation; prepare a new job')
    model = PeftModel.from_pretrained(base, str(peft), is_trainable=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if list(model.peft_config) != ['default'] or not trainable or any('lora_' not in name or '.default.' not in name for name in trainable):
        raise ConfigError(f'{peft}: warm start must load exactly one trainable LoRA adapter over a frozen base')
    return model


def fit(to, resolved, config, pad, descriptor, deadline, checkpoint_seconds=300.):
    deadline.check('model_loading')
    torch, model, device, placed, matched = adapted(resolved, config)
    deadline.check('model_loading')
    recipe, budget, backend = resolved['recipe'], resolved['budget'], accelerator(torch, device)
    fitted = {**{k: v for k, v in resolved.items() if k != 'observations'}, 'placement': placed, 'targets': matched,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'tokenizer': descriptor,
              'dependencies': {'python': platform.python_version(), **{name: version(name) for name in ('torch', 'transformers', 'peft', 'safetensors', 'semantic-trainer')}},
              'profile': {'model_type': config.model_type, 'device': str(device)}, 'adapter': effective(model.peft_config['default'].to_dict(), version('peft'))}
    files.publish(to / 'resolved-job.json', fitted)
    if backend:
        backend.reset_peak_memory_stats(device)
    monitor = Monitor(to, resolved, model, pad, backend, device, deadline)
    if monitor.binding.vocabulary < descriptor['vocab_size']:
        raise ConfigError('base logits vocabulary is smaller than exported tokenizer')
    rows = monitor.fit
    (to / 'candidates').mkdir(exist_ok=True)
    trained = optimizer_of(torch, model, resolved)
    state = {'retained': [], 'step': 0, 'examples': 0, 'measured': 0, 'control': None,
             'epoch': 0, 'offset': 0, 'pending': None, 'zero': None, 'finished': False, 'checkpoint_seconds': checkpoint_seconds}
    payload = recovery.load(to, resolved)
    if payload is not None:
        state = recovery.restore(payload, model, trained, monitor)
    saved = deadline.elapsed()
    cadence, total = recipe['eval_every_examples'], resolved['schedule']['max_optimizer_steps']
    reason, late = 'epoch_cap', None
    model.train()

    def persist():
        nonlocal saved
        monitor.flush()
        saved = recovery.save(to, model, trained, monitor, state)

    def point():
        nonlocal late
        measured = monitor.point(state['step'], state['examples'])
        if state['step'] == 0:
            state['zero'] = measured
        else:
            state['retained'] = retain(to, model, monitor.unit, state['retained'], measured, monitor)
            state['measured'] = state['examples']
        state['control'] = advance(state['control'], measured, monitor.unit, recipe['min_improvement'])
        state['pending'] = None
        persist()
        if measured['completed_after_deadline']:
            late = {'point': measured['name']}
            deadline.crossed = deadline.crossed or 'monitor_point'
            raise Exhausted('time_budget')
        deadline.check('checkpoint')

    try:
        if state['zero'] is None:
            if not (to / 'candidates' / 'zero').exists():
                deadline.check('checkpoint')
                staged = to / 'candidates' / '.zero.staging'
                if staged.exists():
                    shutil.rmtree(staged)
                checkpoint(model, staged, monitor)
                files.publish_directory(staged, to / 'candidates' / 'zero')
            state['pending'] = 0
            persist()
            point()
        elif state['pending'] is not None:
            point()
        for batch in batches(rows, recipe, epoch=state['epoch'], offset=state['offset']):
            if state['finished']:
                break
            deadline.check('optimizer_update')
            if budget['max_tokens'] is not None and monitor.work.training_tokens + sum(monitor.inputs[row].tokens for row in batch) > budget['max_tokens']:
                raise Exhausted('token_budget')
            began = deadline.elapsed()
            try:
                stats = update(monitor.binding, batch, monitor.inputs, pad, recipe, trained)
            except NumericalFailure as error:
                monitor.flush()
                monitor.record({'kind': 'numerical_failure', 'step': state['step'] + 1, 'examples_seen': state['examples'], 'error': str(error)})
                raise
            state['step'] += 1
            state['examples'] += len(batch)
            state['offset'] += len(batch)
            if state['offset'] == len(rows):
                state['epoch'] += 1
                state['offset'] = 0
            monitor.consumed(batch, {'step': state['step'], 'examples_seen': state['examples'], **stats}, deadline.elapsed() - began)
            deadline.check('optimizer_update')
            if state['step'] == min(budget['feasibility_updates'], total):
                monitor.feasibility = feasibility(resolved, monitor.work, deadline.elapsed())
                monitor.record(monitor.feasibility)
            if due(state['examples'], state['measured'], cadence, state['step'], total):
                if any(not torch.isfinite(parameter).all() for parameter in trained[0]):
                    raise NumericalFailure('nonfinite trained adapter tensors; no candidate published')
                state['pending'] = state['step']
                persist()
                point()
                if state['step'] < total and plateaued(state['control'], state['examples'], recipe, len(rows)):
                    raise Exhausted('score_plateau')
            elif deadline.elapsed() - saved >= checkpoint_seconds:
                persist()
            deadline.check('optimizer_update')
    except Exhausted as stop:
        reason = stop.reason
    state['finished'] = reason in ('epoch_cap', 'score_plateau')
    persist()
    stop = stopped(reason, monitor, state['retained'], state['step'], state['examples'], total, late, state['control'])
    return fitted, [state['zero'], *state['retained']], stop, monitor


def publish_run(to, state, **values):
    files.publish(to / 'run.json', {'schema': RUN_SCHEMA, 'state': state, 'at': files.now(), 'loss': LOSS, **values})


def tokenizer_and_pad(resolved, observations):
    _, AutoConfig, _, AutoTokenizer, _, _ = dependencies()
    config = AutoConfig.from_pretrained(resolved['model']['hf'], local_files_only=True)
    if config.model_type not in ('qwen3', 'gemma4'):
        raise ConfigError(f'unsupported LoRA profile {config.model_type}; supported: qwen3 and gemma4')
    tokenizer = AutoTokenizer.from_pretrained(resolved['model']['hf'], local_files_only=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return config, verify_tokenizer(tokenizer, observations), pad


def train(job, to, *, dry_run=False, clock=time.monotonic, checkpoint_seconds=300.):
    return trained_run(*resolve(job), to, dry_run=dry_run, clock=clock, checkpoint_seconds=checkpoint_seconds)



def finalized_run(output, resolved):
    from .jobs import ready_run
    pending = files.read_json(output / 'finalization.json')
    if pending is None:
        return None
    if pending['identity']['job_id'] != resolved['id']:
        raise ConfigError(f'{output}: finalization belongs to another job')
    files.publish(output / 'candidates.json', pending['candidates'])
    files.publish(output / 'metrics.json', pending['metrics'])
    publish_run(output, 'ready', **pending['identity'], stop=pending['stop'], feasibility=pending['feasibility'])
    ready_run(output, 'resume the identified fit')
    return {'path': str(output), 'candidates': str(output / 'candidates.json'), 'names': [c['name'] for c in pending['candidates']['candidates']],
            **pending['identity'], 'stop': pending['stop'], 'quality_claim': False}

def trained_run(resolved, observations, to, *, dry_run=False, clock=time.monotonic, consumed=0., checkpoint_seconds=300.):
    if not math.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ConfigError('checkpoint_seconds must be positive and finite')
    output = Path(to).resolve()
    if not dry_run and (output / 'finalization.json').exists():
        return finalized_run(output, resolved)
    pointer = files.read_json(Path(to) / 'recovery.json') or {}
    elapsed = files.read_json(Path(to) / 'elapsed.json') or {}
    deadline = Deadline(resolved['budget']['max_seconds'], clock, consumed=max(pointer.get('elapsed', consumed), elapsed.get('elapsed', 0.)))
    config, descriptor, pad = tokenizer_and_pad(resolved, observations)
    output = Path(to).resolve()
    existing = files.read_json(output / 'run.json')
    if existing is not None and existing.get('job_id') != resolved['id']:
        raise ConfigError(f'{output}: output belongs to another job')
    if output.exists() and existing is None:
        raise ConfigError(f'{output}: existing directory has no run identity')
    fitting = fitting_envelope(resolved, observations)
    with model_job(extra=peak(estimate(native_model(resolved)), fitting)) as job_:
        if dry_run:
            return {'dry_run': True, 'job': resolved, 'admission': job_.diagnostics()}
        output.mkdir(parents=True, exist_ok=True)
        identities = {'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id']}
        publish_run(output, 'fitting', **identities)
        selecting = [row for row in observations if row['role'] == 'model_select']
        try:
            deadline.check('native_baseline')
            baseline = files.read_json(output / 'baseline.json')
            if baseline is None:
                base_answers, identity = native_answers(resolved, selecting, job_)
                files.publish(output / 'baseline.json', {'answers': base_answers, 'identity': identity})
            else:
                base_answers, identity = baseline['answers'], baseline['identity']
            deadline.check('native_baseline')
            with job_.join(sum(fitting.values()), use='fit'):
                fitted, candidates, stop, monitor = fit(output, {**resolved, 'observations': observations}, config, pad, descriptor, deadline, checkpoint_seconds)
                backend = monitor.backend
                monitor.binding = monitor.inputs = None
            began = time.monotonic()
            if backend:
                backend.empty_cache()
            published = [{**{k: point[k] for k in ('name', 'step', 'examples_seen', 'correct', 'nll', 'completed_after_deadline', 'per_class', 'regressions', 'predictions')},
                          **({'qualification': point['qualification']} if 'qualification' in point else {}),
                          'peft': f"candidates/{point['name']}",
                          **identify(fitted, output / 'candidates' / point['name'])} for point in candidates]
            rule = ranking_rule(candidates[1:], resolved['recipe']['selection_unit'])
            candidate_record = {'schema': CANDIDATES_SCHEMA, 'kind': 'lora-candidates', 'state': 'ready', **identities, 'loss': LOSS,
                                                       'fitted_job_id': artifact_digest(fitted), 'assets': resolved['assets'], 'base_sha256': identity['sha256'],
                                                       'unit': resolved['recipe']['selection_unit'], 'rule': rule, 'stop': stop, 'candidates': published}
            metric_record = {'trajectory': 'progress.jsonl', 'updates': 'updates.jsonl', 'stop': stop, 'rule': rule, 'loss': LOSS,
                                                     'feasibility': monitor.feasibility, 'base_hf_native_agreement': agreement(selecting, base_answers, candidates[0]['predictions']),
                                                     'seconds': deadline.elapsed(), 'publication_seconds': time.monotonic() - began,
                                                     'peak_host_bytes': peak_host_bytes(), 'quality_claim': False}
            files.publish(output / 'finalization.json', {'identity': identities, 'candidates': candidate_record, 'metrics': metric_record,
                                                        'stop': stop, 'feasibility': monitor.feasibility})
            finalized_run(output, resolved)
        except BaseException as error:
            try:
                files.publish(output / 'elapsed.json', {'elapsed': deadline.elapsed(), 'interrupted': True})
                publish_run(output, 'failed', **identities, error=f'{type(error).__name__}: {error}', elapsed_seconds=deadline.elapsed())
            except Exception:
                pass
            raise
    return {'path': str(output), 'candidates': str(output / 'candidates.json'), 'names': [c['name'] for c in candidates], **identities,
            'stop': stop, 'quality_claim': False}


def cadences_of(population, given):
    batch, count = population['recipe']['batch_size'], population['populations']['fit']['observations']
    return sorted(set(given) or {max(batch, math.ceil(count * fraction / batch) * batch) for fraction in (.25, .5, 1, 2)})


def profile(job, to, *, steps=3, warmup=1, cadences=()):
    return profiled(*population_of(job), to, steps=steps, warmup=warmup, cadences=cadences)


def profiled(population, observations, to, *, steps=3, warmup=1, cadences=(), sample_observations=32):
    from .jobs import profile_record, profile_applies
    to = Path(to)
    if to.exists():
        completed = profile_record(to)
        profile_applies(completed, population)
        return completed
    if steps < 1 or warmup < 0 or sample_observations < 1:
        raise ConfigError('profile needs positive measured updates/sample size and nonnegative warmup')
    scratch = to.parent / f'{to.stem}.work'
    scratch.mkdir(parents=True, exist_ok=True)
    pointer = files.read_json(scratch / 'recovery.json') or {}
    elapsed = files.read_json(scratch / 'elapsed.json') or {}
    deadline = Deadline(population['budget']['max_seconds'], consumed=max(pointer.get('elapsed', 0.), elapsed.get('elapsed', 0.)))
    updates, points = [], []
    try:
        deadline.check('profile_loading')
        config, descriptor_, pad = tokenizer_and_pad(population, observations)
        recipe = population['recipe']
        with model_job(extra=fitting_envelope(population, observations)):
            began = deadline.elapsed()
            torch, model, device, placed, _ = adapted(population, config)
            backend = accelerator(torch, device)
            synchronize = (lambda: backend.synchronize(device)) if backend else (lambda: None)
            synchronize()
            loading = deadline.elapsed() - began
            resident = backend.memory_allocated(device) if backend else None
            monitor = Monitor(scratch, {**population, 'observations': observations}, model, pad, backend, device, deadline)
            monitor.select = sorted(monitor.select, key=lambda row: artifact_digest(['profile', recipe['seed'], row['id']]))[:sample_observations]
            monitor.probe = monitor.probe[:sample_observations]
            monitor.point_tokens = sum(monitor.inputs[row].tokens for row in [*monitor.select, *monitor.probe])
            trained = optimizer_of(torch, model, population)
            payload = recovery.load(scratch, population)
            state = {'step': 0, 'examples': 0, 'updates': [], 'points': []}
            if payload is not None:
                state = recovery.restore(payload, model, trained, monitor)
            updates, points = state['updates'], state['points']
            model.train()

            def measure(action):
                deadline.check('profile_operation')
                if backend:
                    backend.reset_peak_memory_stats(device)
                synchronize()
                began = deadline.elapsed()
                result = action()
                synchronize()
                return result, {'seconds': deadline.elapsed() - began, 'peak_host_bytes': peak_host_bytes(),
                                'peak_device_bytes': backend.max_memory_allocated(device) if backend else None,
                                'peak_reserved_device_bytes': backend.max_memory_reserved(device) if backend else None}

            for index, batch in enumerate(batches(monitor.fit, recipe)):
                if index < state['step']:
                    continue
                if index >= warmup + steps:
                    break
                stats, cost = measure(lambda: update(monitor.binding, batch, monitor.inputs, pad, recipe, trained))
                state['step'] += 1
                state['examples'] += len(batch)
                monitor.consumed(batch, {'step': state['step'], 'examples_seen': state['examples'], **stats}, cost['seconds'])
                if index >= warmup:
                    updates.append({**cost, **stats, 'observations': len(batch)})
                monitor.flush()
                recovery.save(scratch, model, trained, monitor, state)
                files.publish(scratch / 'profile-progress.json', {'status': 'partial', 'step': state['step'], 'updates': updates, 'points': points})
            if not updates:
                raise ConfigError('the fit population yields no measured updates; lower warmup')
            if not points:
                _, checkpoint_cost = measure(lambda: save(model, scratch / f"trial-{state['step']}"))
                _, cost = measure(lambda: monitor.point(state['step'], state['examples']))
                points.append({'state': 'measured', **cost, 'checkpoint_seconds': checkpoint_cost['seconds'], 'tokens': monitor.point_tokens,
                               'observations': len(monitor.select) + len(monitor.probe),
                               'presentations': sum(len(monitor.inputs[row].presentations) for row in [*monitor.select, *monitor.probe])})
                recovery.save(scratch, model, trained, monitor, state)
            inputs, restored = monitor.inputs.record(), model.training
        measured = points[-1]
        costs = {'loading_seconds': loading, 'update_seconds_per_token': sum(u['seconds'] for u in updates) / sum(u['tokens'] for u in updates),
                 'point_seconds_per_token': measured['seconds'] / measured['tokens'], 'checkpoint_seconds': measured['checkpoint_seconds']}
        record = {'schema': PROFILE_SCHEMA, 'kind': PROFILE_KIND, 'status': 'completed', 'population_id': population['id'], 'fit_id': population['fit']['id'],
                  'configuration': profile_configuration(population), 'workload': profile_workload(population), 'device': str(device), 'placement': placed,
                  'synchronized': backend is not None, 'instrumentation': 'complete_operation_boundaries', 'loss': LOSS,
                  'sample': {'selection_ids': [row['id'] for row in monitor.select], 'fit_probe_ids': [row['id'] for row in monitor.probe],
                             'quality_claim': False, 'extrapolation': 'seconds per observed token applied to declared full population'},
                  'dependencies': {'python': platform.python_version(), **{name: version(name) for name in ('torch', 'transformers', 'peft', 'semantic-trainer')}},
                  'allocations': {'resident_device_bytes_after_loading': resident, 'prepared_inputs': inputs,
                                  'peak_device_bytes': max((entry['peak_device_bytes'] or 0) for entry in [*updates, *points]) if backend else None,
                                  'peak_reserved_device_bytes': max((entry['peak_reserved_device_bytes'] or 0) for entry in [*updates, *points]) if backend else None,
                                  'peak_host_bytes': max(entry['peak_host_bytes'] for entry in [*updates, *points])},
                  'elapsed_seconds': deadline.elapsed(), 'warmup_excluded': warmup, 'updates': updates, 'points': points, 'costs': costs,
                  'work': monitor.work.totals(), 'cadences': [projection(population, costs, cadence) for cadence in cadences_of(population, cadences)],
                  'training_mode_restored': restored}
        record['id'] = artifact_digest(record)
        files.write_new(to, record)
        files.publish(scratch / 'profile-progress.json', {'status': 'completed', 'profile': str(to), 'id': record['id']})
        return record
    except BaseException as error:
        try:
            files.publish(scratch / 'profile-progress.json', {'status': 'failed', 'updates': updates, 'points': points, 'elapsed_seconds': deadline.elapsed(),
                                                            'error': f'{type(error).__name__}: {error}'})
        except OSError:
            pass
        raise
