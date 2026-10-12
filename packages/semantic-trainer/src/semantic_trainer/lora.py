from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from importlib.metadata import version
import json
import math
import os
import platform
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time
from typing import NamedTuple

from semantic_switch import Runtime, files, model_job
from semantic_switch.artifacts import artifact_digest, canonical_answer, native_model, native_score_requests
from semantic_switch.errors import ConfigError
from semantic_switch.examples import record_path

from .adapter import identify, tensor_shapes
from .jobs import (LOSS, MODULES, PROFILE_KIND, PROFILE_SCHEMA, answer_groups, due, presentation_weights, profile_configuration, profile_workload, projection,
                   ranking, ranking_rule, resolve, unit_correct)

CANDIDATES_SCHEMA, RUN_SCHEMA = 6, 4
HOST_TABLE = 'model.language_model.embed_tokens_per_layer'
RETAINED = 2
SPARSE_FAMILIES = 10


class Exhausted(ConfigError):
    def __init__(self, reason):
        super().__init__(f'{reason} reached')
        self.reason = reason


class NumericalFailure(ConfigError):
    pass


class Infeasible(ConfigError):
    def __init__(self, record):
        super().__init__(f"projected plan exceeds the declared budgets: {json.dumps(record['projection'])}; no further fitting")
        self.record = record


class Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        self.seconds, self.clock, self.started, self.crossed = seconds, clock, clock(), None

    def elapsed(self):
        return self.clock() - self.started

    def expired(self):
        return self.elapsed() >= self.seconds

    def check(self, operation):
        if self.expired():
            self.crossed = self.crossed or operation
            raise Exhausted('time_budget')

    def record(self, last):
        elapsed = self.elapsed()
        return {'limit_seconds': self.seconds, 'elapsed_seconds': elapsed, 'overrun_seconds': max(0., elapsed - self.seconds), 'crossed': self.crossed,
                'last_complete_point': last}


@dataclass
class Work:
    updates: int = 0
    points: int = 0
    checkpoints: int = 0
    training_tokens: int = 0
    monitor_tokens: int = 0
    seconds: dict = field(default_factory=lambda: {'update': [], 'point': [], 'checkpoint': []})

    def totals(self):
        return asdict(self)


class Phases:
    def __init__(self, synchronize=None):
        self.synchronize, self.seconds = synchronize, {}

    @contextmanager
    def __call__(self, name):
        if self.synchronize is None:
            yield
            return
        self.synchronize()
        began = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - began


UNTIMED = Phases()


def dependencies():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ConfigError('install fitting dependencies: pip install "semantic-trainer[lora]"') from error
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model


@dataclass(frozen=True)
class PackedSlots:
    index: object
    valid: object
    present: object
    gold: object
    legal: object
    golden: object

    def to(self, device):
        return PackedSlots(**{field.name: getattr(self, field.name).to(device, non_blocking=True) for field in fields(self)})


class SlotStatistics(NamedTuple):
    answers: object
    chosen: object
    complement: object
    union: object
    total: object


class SlotLoss(NamedTuple):
    nll: object
    conditional: object
    mass: object


class Answers(NamedTuple):
    groups: tuple
    label: int


class Presented(NamedTuple):
    tokens: object
    answers: Answers

    @property
    def length(self):
        return self.tokens.shape[0]


class Prepared(NamedTuple):
    presentations: tuple
    weights: tuple
    tokens: int


@dataclass(frozen=True)
class Inputs:
    rows: dict
    bytes: int
    presentations: int
    seconds: float

    def __getitem__(self, row):
        return self.rows[row['id']]

    def record(self):
        return {'rows': len(self.rows), 'presentations': self.presentations, 'bytes': self.bytes, 'seconds': self.seconds, 'device': 'cpu',
                'lifetime': 'built once after model loading; released with the admitted fitting stage'}


def answers_of(entry, vocabulary):
    return Answers(tuple(map(tuple, answer_groups(entry, vocabulary))), entry['label'])


def presentation(entry, vocabulary):
    import torch
    return Presented(torch.tensor(entry['prefix'] + entry['input'] + entry['question'], dtype=torch.long), answers_of(entry, vocabulary))


def input_bytes(rows):
    return 8 * sum(length(entry) for row in {row['id']: row for row in rows}.values() for entry in row['native']['presentations'])


def prepared(rows, vocabulary):
    began, out = time.perf_counter(), {}
    for row in rows:
        if row['id'] not in out:
            presentations = tuple(presentation(entry, vocabulary) for entry in row['native']['presentations'])
            out[row['id']] = Prepared(presentations, tuple(presentation_weights(row)), sum(entry.length for entry in presentations))
    held = sum(entry.tokens.nbytes for row in out.values() for entry in row.presentations)
    return Inputs(out, held, sum(len(row.presentations) for row in out.values()), time.perf_counter() - began)


def packed(answers):
    import torch
    count, width = max(len(entry.groups) for entry in answers), max(len(group) for entry in answers for group in entry.groups)
    index = torch.zeros((len(answers), count, width), dtype=torch.long)
    valid = torch.zeros((len(answers), count, width), dtype=torch.bool)
    present = torch.zeros((len(answers), count), dtype=torch.bool)
    legal, golden = [], []
    for row, entry in enumerate(answers):
        present[row, :len(entry.groups)] = True
        for column, group in enumerate(entry.groups):
            index[row, column, :len(group)] = torch.tensor(group)
            valid[row, column, :len(group)] = True
            legal += [(row, token) for token in group]
        golden += [(row, token) for token in entry.groups[entry.label]]
    gold = torch.tensor([entry.label for entry in answers], dtype=torch.long)
    return PackedSlots(index, valid, present, gold, torch.tensor(legal, dtype=torch.long), torch.tensor(golden, dtype=torch.long))


def membership(values, coordinates):
    import torch
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[coordinates[:, 0], coordinates[:, 1]] = True
    return mask


def slot_statistics(logits, slots):
    import torch
    values, excluded = logits.float(), float('-inf')
    answers = values.gather(1, slots.index.flatten(1)).view(slots.index.shape).masked_fill(~slots.valid, excluded).logsumexp(-1).masked_fill(~slots.present, excluded)
    legal, golden = membership(values, slots.legal), membership(values, slots.golden)
    whole = golden.all(-1)
    complement = torch.where(whole, excluded, values.masked_fill(golden & ~whole[:, None], excluded).logsumexp(-1))
    return SlotStatistics(answers, values.masked_fill(~golden, excluded).logsumexp(-1), complement,
                          values.masked_fill(~legal, excluded).logsumexp(-1), values.logsumexp(-1))


def slot_loss(statistics, gold):
    import torch
    softplus = torch.nn.functional.softplus
    other = statistics.answers.scatter(1, gold[:, None], float('-inf')).logsumexp(-1)
    return SlotLoss(softplus(statistics.complement - statistics.chosen), softplus(other - statistics.chosen), (statistics.union - statistics.total).exp())


def group_loss(logits, entries):
    slots = packed([answers_of(entry, logits.shape[-1]) for entry in entries]).to(logits.device)
    return slot_loss(slot_statistics(logits, slots), slots.gold)


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


def stages(*envelopes):
    return max(envelopes, key=lambda envelope: sum(envelope.values()))


def native_bytes(model):
    return {f'native:{use}': value for use, value in model._host_bytes().items()}


def length(entry):
    return len(entry['prefix']) + len(entry['input']) + len(entry['question'])


def collate(entries, pad):
    import torch
    lengths = [entry.length for entry in entries]
    tokens = torch.full((len(entries), max(lengths)), pad, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for index, entry in enumerate(entries):
        tokens[index, :lengths[index]] = entry.tokens
        mask[index, :lengths[index]] = 1
    return tokens, mask, lengths


def place(model, device):
    owner, name = HOST_TABLE.rsplit('.', 1)
    owner = model.get_submodule(owner)
    table = owner.get_submodule(name)
    delattr(owner, name)
    model.to(device)
    owner.add_module(name, table)
    return {'': str(device), HOST_TABLE: 'cpu'}


def map_host_table(model, hf, rows=16384):
    import numpy as np
    import torch
    table, name = model.get_submodule(HOST_TABLE), HOST_TABLE + '.weight'
    for path in sorted(Path(hf).glob('*.safetensors')):
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            entry = json.loads(stream.read(size)).get(name)
        if entry is not None:
            break
    else:
        raise ConfigError(f'{hf}: {name} is not in the safetensors snapshot')
    if entry['dtype'] != 'BF16' or table.weight.dtype != torch.bfloat16 or list(table.weight.shape) != entry['shape']:
        return 'cpu'
    mapped = torch.from_numpy(np.memmap(path, dtype=np.uint16, mode='c', offset=8 + size + entry['data_offsets'][0], shape=tuple(entry['shape']))).view(torch.bfloat16)
    if not all(torch.equal(mapped[i:i + rows], table.weight[i:i + rows]) for i in range(0, len(mapped), rows)):
        raise ConfigError(f'{path}: {name} differs from the loaded per-layer table')
    table.weight = torch.nn.Parameter(mapped, requires_grad=False)
    return 'cpu:file-mapped'


def per_layer_model(model):
    return next((module for name, module in model.named_modules() if name.endswith('language_model') and hasattr(module, 'embed_tokens_per_layer')), None)


def host_table_inputs(language, tokens):
    table = language.embed_tokens_per_layer.weight.device
    return {'inputs_embeds': language.embed_tokens(tokens), 'per_layer_inputs': language.get_per_layer_inputs(tokens.to(table), None).to(tokens.device)}


def answer_logits(model, entries, pad, phases=UNTIMED):
    import torch
    with phases('collate'):
        tokens, mask, lengths = collate(entries, pad)
    with phases('transfer'):
        device = model.get_input_embeddings().weight.device
        tokens, mask = tokens.to(device, non_blocking=True), mask.to(device, non_blocking=True)
    with phases('host_embedding'):
        language = per_layer_model(model)
        inputs = host_table_inputs(language, tokens) if language is not None and language.embed_tokens_per_layer.weight.device != tokens.device else {'input_ids': tokens}
    with phases('forward'):
        positions = sorted({size - 1 for size in lengths})
        output = model(**inputs, attention_mask=mask, use_cache=False, logits_to_keep=torch.tensor(positions, device=device)).logits
        return output[torch.arange(len(entries), device=output.device), torch.tensor([positions.index(size - 1) for size in lengths], device=output.device)]


def vocabulary_of(model):
    return getattr(model.config, 'text_config', model.config).vocab_size


def presented(model, entries, pad, phases=UNTIMED):
    with phases('slot_packing'):
        slots = packed([entry.answers for entry in entries])
    with phases('transfer'):
        slots = slots.to(model.get_input_embeddings().weight.device)
    return answer_logits(model, entries, pad, phases), slots


def reduced(logits, slots, weights, phases=UNTIMED):
    import torch
    with phases('reduction'):
        statistics = slot_statistics(logits, slots)
        losses = slot_loss(statistics, slots.gold)
        return statistics, losses, (losses.nll * logits.new_tensor(weights, dtype=torch.float32)).sum(), torch.isfinite(logits).all()


def verify_tokenizer(tokenizer, rows):
    descriptor = rows[0]['native']['tokenizer']
    vocab = tokenizer.get_vocab()
    ordered = [None] * descriptor['vocab_size']
    for token, index in vocab.items():
        if not 0 <= index < len(ordered) or ordered[index] is not None:
            raise ConfigError('HF/deployment tokenizer vocabulary mismatch')
        ordered[index] = token
    for index, token in descriptor.get('padding_tokens', ()):
        if ordered[index] is None:
            ordered[index] = token
    if None in ordered or artifact_digest(ordered) != descriptor['vocab_sha256']:
        raise ConfigError('HF/deployment tokenizer vocabulary mismatch')
    for row in rows:
        if row['native']['tokenizer'] != descriptor:
            raise ConfigError('exports have different tokenizers')
        for probe in descriptor['probes']:
            if tokenizer.encode(probe['text'], add_special_tokens=False) != probe['ids']:
                raise ConfigError('HF/deployment tokenizer token IDs mismatch')
        for entry in row['native']['presentations']:
            for group, expected in zip(entry['answer_slots'], entry['answer_text']):
                if [tokenizer.decode([token], skip_special_tokens=False) for token in group] != expected:
                    raise ConfigError('HF/deployment answer-token decoding mismatch')
    for name, token in descriptor['special'].items():
        if token >= 0 and (name != 'bos' or tokenizer.bos_token_id is not None) and getattr(tokenizer, name + '_token_id', None) != token:
            raise ConfigError(f'HF/deployment {name} token mismatch')
    return descriptor


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


def staged(presentations, slots):
    stages = {}
    for entry in presentations:
        if entry['variant'] == 'canonical':
            mapped = entry['slot_map'][slots[entry['id']]]
            stages[entry['stage']] = mapped if mapped >= 0 else None
    return {'answer': canonical_answer(presentations, slots), 'stages': stages}


def native_predictions(runtime, rows, ledger=None):
    import numpy as np
    requests = native_score_requests(rows)
    block, reuse = ledger.replay(requests, runtime.score_native) if ledger is not None else (runtime.score_native(requests), None)
    if any(block.error):
        raise ConfigError(f'native scoring failed: {block.error}')
    if not np.isfinite(block.log_mass).all() or any(not np.isfinite(block.logp[i, :len(req.slots[0])]).all() for i, req in enumerate(requests)):
        raise ConfigError('native scoring produced nonfinite answer slots or mass')
    slots = iter(int(np.argmax(block.logp[i, :len(req.slots[0])])) for i, req in enumerate(requests))
    return {row['id']: staged(row['native']['presentations'], {entry['id']: next(slots) for entry in row['native']['presentations']}) for row in rows}, reuse


def native_answers(resolved, rows, job):
    with Runtime({'judge': native_model(resolved)}, job=job) as rt:
        return native_predictions(rt, rows)[0], rt.identity()


def agreed(rows, native, monitored):
    out = {}
    for row in rows:
        a, b = native.get(row['id']), monitored.get(row['id'])
        stages = sorted({entry['stage'] for entry in row['native']['presentations'] if entry['variant'] == 'canonical'})
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
    return {'agreed': len(flags) - len(disagreements), 'submitted': len(flags), 'disagreements': disagreements, 'stages': stages,
            'scope': 'HF monitoring versus native scores of the same exported presentations per target and stage; a missing output is a disagreement'}


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
            'intermediate': None,
            'scope': 'compatibility diagnostic, not an eligibility gate: specialist minus base HF/native answer agreement on identical targets and stages, '
                     'paired within canonical families; includes backend and precision effects; quantization alone needs a measured same-backend F16-to-Q8 '
                     'comparison, not measured here; deployed native correctness owns selection'}


def learning_rate_factor(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    return 1. if schedule == 'constant' else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def scheduler(torch, optimizer, resolved):
    plan, recipe = resolved['schedule'], resolved['recipe']
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, plan['max_optimizer_steps'], plan['warmup_steps'], recipe['schedule']))


def accelerator(torch, device):
    return getattr(torch, device.type) if device.type in ('xpu', 'cuda') else None


def accumulate(model, batch, inputs, pad, microbatch, phases=UNTIMED):
    items = sorted(((entry, weight / len(batch)) for row in batch for entry, weight in zip(inputs[row].presentations, inputs[row].weights)),
                   key=lambda item: item[0].length)
    loss, finite = 0., True
    for start in range(0, len(items), microbatch):
        chunk = items[start:start + microbatch]
        logits, slots = presented(model, [entry for entry, _ in chunk], pad, phases)
        _, _, total, ok = reduced(logits, slots, [weight for _, weight in chunk], phases)
        with phases('backward'):
            total.backward()
        loss, finite = loss + total.detach(), ok & finite
    return {'loss': loss, 'finite': finite, 'tokens': sum(entry.length for entry, _ in items), 'presentations': len(items)}


def optimizer_of(torch, model, resolved):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=resolved['recipe']['learning_rate'], weight_decay=resolved['recipe']['weight_decay'])
    return trainable, optimizer, scheduler(torch, optimizer, resolved)


def update(model, batch, inputs, pad, recipe, trained, phases=UNTIMED):
    import torch
    trainable, optimizer, rates = trained
    stats = accumulate(model, batch, inputs, pad, recipe['microbatch'], phases)
    loss = float(stats['loss'])
    if not math.isfinite(loss) or not bool(stats['finite']):
        raise NumericalFailure(f'nonfinite model output or loss ({loss}); no candidate published')
    with phases('clip_optimizer'):
        try:
            norm = torch.nn.utils.clip_grad_norm_(trainable, recipe['gradient_clip'], error_if_nonfinite=True)
        except RuntimeError as error:
            raise NumericalFailure('nonfinite gradient norm; no candidate published') from error
        rate = optimizer.param_groups[0]['lr']
        optimizer.step()
        rates.step()
        optimizer.zero_grad(set_to_none=True)
    return {'learning_rate': rate, 'loss': loss, 'gradient_norm': float(norm), 'tokens': stats['tokens'], 'presentations': stats['presentations']}


def batches(rows, recipe):
    for epoch in range(recipe['epochs']):
        order = list(range(len(rows)))
        random.Random(artifact_digest([recipe['seed'], 'epoch', epoch])).shuffle(order)
        for start in range(0, len(order), recipe['batch_size']):
            yield [rows[index] for index in order[start:start + recipe['batch_size']]]


def evaluate(model, rows, inputs, pad, poll, charge=lambda tokens: None, phases=UNTIMED):
    import torch
    was = model.training
    model.eval()
    try:
        out = {}
        with torch.inference_mode():
            for row in rows:
                poll()
                entries, prepared_ = row['native']['presentations'], inputs[row]
                logits, slots = presented(model, prepared_.presentations, pad, phases)
                statistics, _, weighted, finite = reduced(logits, slots, prepared_.weights, phases)
                nll = float(weighted)
                if not math.isfinite(nll) or not bool(finite):
                    raise NumericalFailure(f"nonfinite model output evaluating {row['id']}; no candidate published")
                predicted = staged(entries, {entry['id']: int(index) for entry, index in zip(entries, statistics.answers.argmax(-1).tolist())})
                out[row['id']] = {'nll': nll, 'predicted': predicted['answer'], 'stages': predicted['stages'], 'correct': predicted['answer'] == row['label']}
                charge(prepared_.tokens)
        return out
    finally:
        model.train(was)


def per_class(rows, results):
    out = {}
    for row in rows:
        counts = out.setdefault('/'.join(record_path(row)), {key: {'gold': 0, 'predicted': 0, 'correct': 0} for key in row['classes']})
        gold, predicted = row['classes'][row['label']], results[row['id']]['predicted']
        counts[gold]['gold'] += 1
        counts[gold]['correct'] += results[row['id']]['correct']
        if predicted is not None:
            counts[row['classes'][predicted]]['predicted'] += 1
    return out


def peak_host_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Monitor:
    def __init__(self, to, resolved, model, pad, backend, device, deadline):
        self.to, self.resolved, self.model, self.pad, self.backend, self.device, self.deadline = to, resolved, model, pad, backend, device, deadline
        roles = {}
        for row in resolved['observations']:
            roles.setdefault(row['role'], []).append(row)
        self.fit, self.select = roles['fit'], roles['model_select']
        fitting = {row['id']: row for row in self.fit}
        self.probe = [fitting[identifier] for identifier in resolved['probe_ids']]
        self.inputs = prepared([*self.fit, *self.select], vocabulary_of(model))
        self.unit, self.point_tokens = resolved['recipe']['selection_unit'], resolved['work']['point_tokens']
        self.points, self.updates, self.window, self.flushed, self.work = [], [], [], 0, Work()
        self.phases, self.feasibility = UNTIMED, None
        self.seen = {'situations': set(), 'targets': set(), 'presentations': set(), 'families': set()}

    def consumed(self, batch, update, seconds):
        for row in batch:
            self.seen['situations'].update(row['situations'])
            self.seen['targets'].add(row['id'])
            self.seen['families'].add(row['component'])
            self.seen['presentations'].update(entry['id'] for entry in row['native']['presentations'])
        self.work.updates += 1
        self.work.training_tokens += update['tokens']
        self.work.seconds['update'].append(seconds)
        self.updates.append(update)
        self.window.append(update['loss'])

    def checkpointed(self, seconds):
        self.work.checkpoints += 1
        self.work.seconds['checkpoint'].append(seconds)

    def flush(self):
        files.append(self.to / 'updates.jsonl', self.updates[self.flushed:])
        self.flushed = len(self.updates)

    def record(self, record):
        files.append(self.to / 'progress.jsonl', [record])
        print(json.dumps(record), file=sys.stderr, flush=True)

    def charge(self, tokens):
        self.work.monitor_tokens += tokens

    def poll(self):
        self.deadline.check('evaluation_row')

    def token_work(self):
        return {'training': self.work.training_tokens, 'monitoring': self.work.monitor_tokens}

    def point(self, step, examples):
        if self.work.monitor_tokens + self.point_tokens > self.resolved['budget']['max_monitor_tokens']:
            raise Exhausted('monitor_budget')
        before, began = self.work.monitor_tokens, self.deadline.elapsed()
        try:
            with self.phases('monitor:model_select'):
                selected = evaluate(self.model, self.select, self.inputs, self.pad, self.poll, self.charge, self.phases)
            with self.phases('monitor:probe'):
                probed = evaluate(self.model, self.probe, self.inputs, self.pad, self.poll, self.charge, self.phases)
        except BaseException as error:
            self.flush()
            self.record({'kind': 'interrupted_evaluation', 'step': step, 'examples_seen': examples, 'reserved_tokens': self.point_tokens,
                         'charged_tokens': self.work.monitor_tokens - before, 'error': f'{type(error).__name__}: {error}', 'published': False,
                         'token_work': self.token_work()})
            raise
        seconds, late = self.deadline.elapsed() - began, self.deadline.expired()
        self.work.points += 1
        self.work.seconds['point'].append(seconds)
        record = {'kind': 'evaluation', 'loss': LOSS, 'step': step, 'examples_seen': examples, 'unit': self.resolved['schedule']['unit'], 'epoch': examples / len(self.fit),
                  'seen': {key: len(value) for key, value in self.seen.items()},
                  'train_loss': sum(self.window) / len(self.window) if self.window else None,
                  'model_select': {'nll': sum(r['nll'] for r in selected.values()) / len(selected),
                                   'correct': unit_correct({i: r['correct'] for i, r in selected.items()}, self.resolved['situations'], 'model_select'),
                                   'per_class': per_class(self.select, selected), 'population': 'selection proxy; not independent quality'},
                  'probe': {'correct': sum(r['correct'] for r in probed.values()), 'submitted': len(probed),
                            'population': 'in-sample fit probe; never independent quality'},
                  'seconds': self.deadline.elapsed(), 'point_seconds': seconds, 'completed_after_deadline': late, 'peak_host_bytes': peak_host_bytes(),
                  'peak_device_bytes': self.backend.max_memory_allocated(self.device) if self.backend else None,
                  'token_work': {**self.token_work(), 'point': self.work.monitor_tokens - before}}
        point = {'name': 'zero' if step == 0 else f'step-{step}', 'step': step, 'examples_seen': examples, 'correct': record['model_select']['correct'],
                 'nll': record['model_select']['nll'], 'completed_after_deadline': late,
                 'predictions': {i: {'answer': r['predicted'], 'stages': r['stages']} for i, r in selected.items()}}
        self.points.append(point)
        ranked = ranking([p for p in self.points if p['step']], self.unit)
        record['best_loss_step'] = min(self.points, key=lambda p: (p['nll'], p['step']))['step']
        record['best_correct_step'] = ranked[0]['step'] if ranked else None
        with self.phases('progress_io'):
            self.flush()
            self.record(record)
        self.window = []
        return point


def save(model, directory):
    model.save_pretrained(directory, safe_serialization=True)


def retain(to, model, unit, retained, point, monitor):
    ranked = ranking([*retained, point], unit)[:RETAINED]
    if point not in ranked:
        return retained
    candidates = to / 'candidates'
    staged = candidates / f".{point['name']}.staging"
    began = monitor.deadline.elapsed()
    save(model, staged)
    os.replace(staged, candidates / point['name'])
    monitor.checkpointed(monitor.deadline.elapsed() - began)
    files.publish(to / 'retained.json', {'unit': unit, 'retained': [p['name'] for p in ranked]})
    for evicted in retained:
        if evicted not in ranked:
            shutil.rmtree(candidates / evicted['name'])
    return ranked


def feasibility(resolved, work, elapsed):
    plan, budget, planned = resolved['schedule'], resolved['budget'], resolved['work']
    measured = {f'{stage}_seconds': sum(values) / len(values) if values else 0. for stage, values in work.seconds.items()}
    remaining = {'updates': plan['max_optimizer_steps'] - work.updates, 'points': planned['evaluation_points'] - work.points}
    remaining['checkpoints'] = remaining['points']
    projected = {'training_seconds': remaining['updates'] * measured['update_seconds'], 'monitoring_seconds': remaining['points'] * measured['point_seconds'],
                 'checkpoint_seconds': remaining['checkpoints'] * measured['checkpoint_seconds']}
    seconds = elapsed + sum(projected.values())
    projection = {'seconds': seconds, **projected, 'max_seconds': budget['max_seconds'], 'training_tokens': planned['training_tokens'], 'max_tokens': budget['max_tokens'],
                  'monitor_tokens': planned['monitor_tokens'], 'max_monitor_tokens': budget['max_monitor_tokens'],
                  'evaluation_points': planned['evaluation_points'], 'optimizer_steps': plan['max_optimizer_steps']}
    return {'kind': 'feasibility', 'feasible': seconds <= budget['max_seconds'] and planned['training_tokens'] <= budget['max_tokens']
            and planned['monitor_tokens'] <= budget['max_monitor_tokens'],
            'elapsed_seconds': elapsed, 'completed': {key: value for key, value in work.totals().items() if key != 'seconds'}, 'remaining': remaining,
            'measured': {**measured, 'point_tokens': planned['point_tokens'], 'updates_measured': len(work.seconds['update']),
                         'points_measured': len(work.seconds['point']), 'checkpoints_measured': len(work.seconds['checkpoint'])},
            'projection': projection,
            'assumptions': ['elapsed already contains every completed update, evaluation point (point zero included) and checkpoint save',
                            'remaining updates and points are the frozen plan totals minus completed work at this boundary; a due point not yet started is pending',
                            'each remaining update costs the mean measured update; each remaining point the mean measured complete point (model_select and fit probe)',
                            'one checkpoint allowance per remaining nonzero point at the mean measured save; retention can save fewer',
                            'maximum point count of the frozen plan with no early stop',
                            'a projection conditional on the observed stage costs, not a guarantee of convergence or elapsed time']}


def adapted(resolved, config):
    torch, _, AutoModel, _, LoraConfig, get_peft_model = dependencies()
    recipe, fitting = resolved['recipe'], resolved['fitting']
    device = torch.device(fitting['device'])
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
    model = get_peft_model(base, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'], lora_dropout=recipe['dropout'],
                                          target_modules=list(matched), bias='none', task_type='CAUSAL_LM'))
    return torch, model, device, placed, matched


def fit(to, resolved, config, pad, descriptor, deadline):
    deadline.check('model_loading')
    torch, model, device, placed, matched = adapted(resolved, config)
    deadline.check('model_loading')
    recipe, budget, backend = resolved['recipe'], resolved['budget'], accelerator(torch, device)
    if vocabulary_of(model) < descriptor['vocab_size']:
        raise ConfigError('base logits vocabulary is smaller than exported tokenizer')
    fitted = {**{k: v for k, v in resolved.items() if k != 'observations'}, 'placement': placed, 'targets': matched,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'tokenizer': descriptor,
              'dependencies': {'python': platform.python_version(), **{name: version(name) for name in ('torch', 'transformers', 'peft', 'safetensors', 'semantic-trainer')}},
              'profile': {'model_type': config.model_type, 'device': str(device)}}
    files.publish(to / 'resolved-job.json', fitted)
    if backend:
        backend.reset_peak_memory_stats(device)
    monitor = Monitor(to, resolved, model, pad, backend, device, deadline)
    rows = monitor.fit
    (to / 'candidates').mkdir()
    deadline.check('checkpoint')
    began = deadline.elapsed()
    save(model, to / 'candidates' / 'zero')
    monitor.checkpointed(deadline.elapsed() - began)
    deadline.check('checkpoint')
    zero = monitor.point(0, 0)
    trained = optimizer_of(torch, model, resolved)
    retained, step, examples, measured, reason, late = [], 0, 0, 0, 'epoch_cap', None
    best_loss, best_examples = zero['nll'], 0
    cadence, total = recipe['eval_every_examples'], resolved['schedule']['max_optimizer_steps']
    model.train()
    try:
        deadline.check('monitor_point')
        for batch in batches(rows, recipe):
            deadline.check('optimizer_update')
            if monitor.work.training_tokens + sum(monitor.inputs[row].tokens for row in batch) > budget['max_tokens']:
                raise Exhausted('token_budget')
            began = deadline.elapsed()
            try:
                stats = update(model, batch, monitor.inputs, pad, recipe, trained)
            except NumericalFailure as error:
                monitor.flush()
                monitor.record({'kind': 'numerical_failure', 'step': step + 1, 'examples_seen': examples, 'error': str(error)})
                raise
            step, examples = step + 1, examples + len(batch)
            monitor.consumed(batch, {'step': step, 'examples_seen': examples, **stats}, deadline.elapsed() - began)
            deadline.check('optimizer_update')
            if step == min(budget['feasibility_updates'], total):
                monitor.feasibility = feasibility(resolved, monitor.work, deadline.elapsed())
                monitor.flush()
                monitor.record(monitor.feasibility)
                if not monitor.feasibility['feasible']:
                    raise Infeasible(monitor.feasibility)
            if due(examples, measured, cadence, step, total):
                if any(not torch.isfinite(parameter).all() for parameter in trained[0]):
                    raise NumericalFailure('nonfinite trained adapter tensors; no candidate published')
                point = monitor.point(step, examples)
                measured = examples
                if point['completed_after_deadline']:
                    deadline.crossed = deadline.crossed or 'monitor_point'
                    began = deadline.elapsed()
                    retained = retain(to, model, monitor.unit, retained, point, monitor)
                    late = {'point': point['name'], 'finalization_seconds': deadline.elapsed() - began}
                    raise Exhausted('time_budget')
                retained = retain(to, model, monitor.unit, retained, point, monitor)
                deadline.check('checkpoint')
                if (best_loss - point['nll']) / abs(best_loss) > recipe['min_improvement']:
                    best_loss, best_examples = point['nll'], examples
                elif examples >= recipe['min_epochs'] * len(rows) and examples - best_examples >= recipe['patience_examples']:
                    raise Exhausted('loss_plateau')
    except Exhausted as stop:
        reason = stop.reason
    if reason in ('epoch_cap', 'loss_plateau') and deadline.expired():
        reason, deadline.crossed = 'time_budget', deadline.crossed or 'final_boundary'
    monitor.flush()
    if not retained:
        raise ConfigError(f'{reason}: no fully evaluated optimizer step to select; no candidate published')
    last, best = monitor.points[-1], retained[0]
    boundary = best['step'] == last['step']
    stop = {'reason': reason, 'convergence': 'loss_plateau' if reason == 'loss_plateau' and not boundary else 'not_demonstrated',
            'steps_trained': step, 'examples_trained': examples, 'max_optimizer_steps': total, 'evaluation_points': len(monitor.points),
            'last_evaluation': {'step': last['step'], 'examples_seen': last['examples_seen']},
            'unevaluated': {'steps': step - last['step'], 'examples': examples - last['examples_seen']},
            'best_monitored': {'name': best['name'], 'step': best['step'], 'examples_seen': best['examples_seen']}, 'best_at_boundary': boundary,
            'deadline': {**deadline.record(last['name']), 'completed_after_deadline': late} if reason == 'time_budget' else None,
            'work': monitor.work.totals(), 'inputs': monitor.inputs.record(),
            'scope': 'budget interruption, the epoch cap or a best point at the last evaluation boundary do not demonstrate convergence; '
                     'unevaluated trailing updates are discarded; native selection chooses among the retained points and base'}
    return fitted, [zero, *retained], stop, monitor


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


def train(job, to, *, dry_run=False, clock=time.monotonic):
    resolved, observations = resolve(job)
    deadline = Deadline(resolved['budget']['max_seconds'], clock)
    config, descriptor, pad = tokenizer_and_pad(resolved, observations)
    output = Path(to).resolve()
    if output.exists():
        raise ConfigError(f'{output}: output already exists; use a new identified run')
    fitting = fitting_envelope(resolved, observations)
    with model_job(extra=stages(native_bytes(native_model(resolved)), fitting)) as job_:
        if dry_run:
            return {'dry_run': True, 'job': resolved, 'admission': job_.diagnostics()}
        output.mkdir(parents=True)
        identities = {'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id']}
        publish_run(output, 'fitting', **identities)
        selecting = [row for row in observations if row['role'] == 'model_select']
        try:
            deadline.check('native_baseline')
            base_answers, identity = native_answers(resolved, selecting, job_)
            deadline.check('native_baseline')
            with job_.join(sum(fitting.values()), use='fit'):
                fitted, candidates, stop, monitor = fit(output, {**resolved, 'observations': observations}, config, pad, descriptor, deadline)
                backend = monitor.backend
                monitor.model = monitor.inputs = None
            began = time.monotonic()
            if backend:
                backend.empty_cache()
            published = [{**{k: point[k] for k in ('name', 'step', 'examples_seen', 'correct', 'nll', 'completed_after_deadline', 'predictions')},
                          'peft': f"candidates/{point['name']}",
                          **identify(fitted, output / 'candidates' / point['name'])} for point in candidates]
            rule = ranking_rule(candidates[1:], resolved['recipe']['selection_unit'])
            files.publish(output / 'candidates.json', {'schema': CANDIDATES_SCHEMA, 'kind': 'lora-candidates', 'state': 'ready', **identities, 'loss': LOSS,
                                                       'fitted_job_id': artifact_digest(fitted), 'assets': resolved['assets'], 'base_sha256': identity['sha256'],
                                                       'unit': resolved['recipe']['selection_unit'], 'rule': rule, 'stop': stop, 'candidates': published})
            files.publish(output / 'metrics.json', {'trajectory': 'progress.jsonl', 'updates': 'updates.jsonl', 'stop': stop, 'rule': rule, 'loss': LOSS,
                                                     'feasibility': monitor.feasibility, 'base_hf_native_agreement': agreement(selecting, base_answers, candidates[0]['predictions']),
                                                     'seconds': deadline.elapsed(), 'publication_seconds': time.monotonic() - began,
                                                     'peak_host_bytes': peak_host_bytes(), 'quality_claim': False})
            publish_run(output, 'ready', **identities, stop=stop, feasibility=monitor.feasibility)
        except BaseException as error:
            shutil.rmtree(output / 'candidates', ignore_errors=True)
            publish_run(output, 'failed', **identities, error=f'{type(error).__name__}: {error}',
                        **({'feasibility': error.record} if isinstance(error, Infeasible) else {}))
            raise
    return {'path': str(output), 'candidates': str(output / 'candidates.json'), 'names': [c['name'] for c in candidates], **identities,
            'stop': stop, 'quality_claim': False}


def cadences_of(resolved, given):
    batch, count = resolved['recipe']['batch_size'], resolved['populations']['fit']['observations']
    derived = {max(batch, math.ceil(count * fraction / batch) * batch) for fraction in (.25, .5, 1, 2)}
    return sorted({*(given or derived), resolved['recipe']['eval_every_examples']})


def profile(job, to, *, steps=3, warmup=1, cadences=()):
    to = Path(to)
    if to.exists():
        raise ConfigError(f'{to}: profile already exists; write a new profile')
    if steps < 1 or warmup < 0:
        raise ConfigError('profile needs at least one measured update and a nonnegative warmup')
    resolved, observations = resolve(job)
    config, descriptor, pad = tokenizer_and_pad(resolved, observations)
    recipe = resolved['recipe']
    with model_job(extra=fitting_envelope(resolved, observations)), tempfile.TemporaryDirectory(prefix='semantic-trainer-profile-') as scratch:
        began = time.perf_counter()
        torch, model, device, placed, _ = adapted(resolved, config)
        backend = accelerator(torch, device)
        synchronize = (lambda: backend.synchronize(device)) if backend else (lambda: None)
        synchronize()
        loading = time.perf_counter() - began
        resident = backend.memory_allocated(device) if backend else None
        monitor = Monitor(Path(scratch), {**resolved, 'observations': observations}, model, pad, backend, device, Deadline(math.inf))
        trained = optimizer_of(torch, model, resolved)
        model.train()
        def measure(action):
            phases = Phases(synchronize)
            if backend:
                backend.reset_peak_memory_stats(device)
            synchronize()
            began = time.perf_counter()
            result = action(phases)
            synchronize()
            return result, {'seconds': time.perf_counter() - began, 'phases': phases.seconds, 'peak_host_bytes': peak_host_bytes(),
                            'peak_device_bytes': backend.max_memory_allocated(device) if backend else None}
        updates = []
        for index, batch in zip(range(warmup + steps), batches(monitor.fit, recipe)):
            stats, cost = measure(lambda phases: update(model, batch, monitor.inputs, pad, recipe, trained, phases))
            if index >= warmup:
                updates.append({**cost, 'tokens': stats['tokens'], 'presentations': stats['presentations'], 'observations': len(batch)})
        if not updates:
            raise ConfigError(f'the fit population yields {warmup} or fewer updates; lower warmup')
        evaluated = [*monitor.select, *monitor.probe]
        points = []
        for index, state in enumerate(('cold', 'steady')):
            def point(phases):
                monitor.phases = phases
                with phases('checkpoint'):
                    save(model, Path(scratch) / state)
                return monitor.point(warmup + steps + index, (warmup + steps) * recipe['batch_size'])
            _, cost = measure(point)
            points.append({'state': state, **cost, 'tokens': monitor.point_tokens, 'observations': len(evaluated),
                           'presentations': sum(len(monitor.inputs[row].presentations) for row in evaluated)})
        restored = model.training
        inputs = monitor.inputs.record()
    steady = points[-1]
    costs = {'loading_seconds': loading, 'update_seconds_per_token': sum(u['seconds'] for u in updates) / sum(u['tokens'] for u in updates),
             'point_seconds_per_token': (steady['seconds'] - steady['phases']['checkpoint']) / steady['tokens'],
             'checkpoint_seconds': steady['phases']['checkpoint']}
    record = {'schema': PROFILE_SCHEMA, 'kind': PROFILE_KIND, 'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id'],
              'configuration': profile_configuration(resolved), 'workload': profile_workload(resolved),
              'device': str(device), 'placement': placed, 'synchronized': backend is not None, 'loss': LOSS,
              'dependencies': {'python': platform.python_version(), **{name: version(name) for name in ('torch', 'transformers', 'peft', 'semantic-trainer')}},
              'allocations': {'resident_device_bytes_after_loading': resident, 'prepared_inputs': inputs,
                              'peak_device_bytes': max((entry['peak_device_bytes'] or 0) for entry in [*updates, *points]) if backend else None,
                              'peak_host_bytes': max(entry['peak_host_bytes'] for entry in [*updates, *points])},
              'warmup_excluded': warmup, 'updates': updates, 'points': points, 'costs': costs,
              'cadences': [projection(resolved, costs, cadence) for cadence in cadences_of(resolved, cadences)],
              'training_mode_restored': restored,
              'phases': {'leaf': ['slot_packing', 'collate', 'transfer', 'host_embedding', 'forward', 'reduction', 'backward', 'clip_optimizer', 'checkpoint', 'progress_io'],
                         'inclusive': ['monitor:model_select', 'monitor:probe']},
              'scope': 'synchronized per-phase measurement of full optimizer updates over the first fit batches and complete cold/steady monitor points '
                       '(model_select and fit probe) on a disposable in-memory adapter; per-phase synchronization makes phase totals an upper bound '
                       'of unsynchronized production time; projections scale steady costs by token workload and exclude native baseline scoring; '
                       'nothing is published to the job'}
    record['id'] = artifact_digest(record)
    to.parent.mkdir(parents=True, exist_ok=True)
    files.write_new(to, record)
    return record
