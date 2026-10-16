from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
import math
import random
import time
from typing import NamedTuple

import numpy as np

from semantic_switch import ConfigError
from semantic_switch.training import ESCAPE, OPTION_SOFTMAX, artifact_digest, canonical_answer, observed, primitive, Tokens

from .jobs import presentation_weights


class NumericalFailure(ConfigError):
    pass


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
        yield
        self.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - began


UNTIMED = Phases()


@dataclass(frozen=True)
class PackedSlots:
    index: object
    valid: object
    present: object
    gold: object

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
    index: object
    valid: object


class Presented(NamedTuple):
    segments: tuple
    answers: Answers
    length: int


class Prepared(NamedTuple):
    presentations: tuple
    weights: tuple
    tokens: int
    objective: float


class Binding(NamedTuple):
    model: object
    trunk: object
    head: object
    softcap: float | None
    host: object
    device: object
    vocabulary: int


@dataclass(frozen=True)
class Inputs:
    rows: dict
    bytes: int
    presentations: int
    seconds: float

    def __getitem__(self, row):
        return self.rows[row['id']]

    def record(self):
        return {'rows': len(self.rows), 'presentations': self.presentations, 'bytes': self.bytes, 'seconds': self.seconds, 'device': 'cpu'}


def answers_of(groups, label, vocabulary):
    if max(token for group in groups for token in group) >= vocabulary:
        raise ConfigError(f'answer token outside the {vocabulary}-token model vocabulary')
    width = max(map(len, groups))
    index, valid = np.zeros((len(groups), width), dtype=np.int64), np.zeros((len(groups), width), dtype=bool)
    for column, group in enumerate(groups):
        index[column, :len(group)], valid[column, :len(group)] = group, True
    return Answers(groups, label, index, valid)


def segment_bytes(segments):
    return sum({id(segment): segment.nbytes if isinstance(segment, Tokens) else 8 * len(segment) for segment in segments}.values())


def input_bytes(rows):
    return segment_bytes(segment for row in {row['id']: row for row in rows}.values() for entry in row['presentations']
                         for segment in (entry.prefix, entry.input, entry.question))


def prepared(rows, vocabulary):
    began, out, answers = time.perf_counter(), {}, {}
    for row in rows:
        if row['id'] not in out:
            presentations = tuple(Presented((entry.prefix, entry.input, entry.question), answers[key] if (key := (id(entry.answers), entry.label)) in answers
                                            else answers.setdefault(key, answers_of(entry.answers.slots, entry.label, vocabulary)), entry.length)
                                  for entry in row['presentations'])
            out[row['id']] = Prepared(presentations, tuple(presentation_weights(row)), sum(entry.length for entry in presentations), row['objective'])
    held = segment_bytes(segment for row in out.values() for entry in row.presentations for segment in entry.segments)
    return Inputs(out, held, sum(len(row.presentations) for row in out.values()), time.perf_counter() - began)


def planned(lengths, presentations=None, tokens=None):
    if tokens is not None and any(length > tokens for length in lengths):
        raise ConfigError(f'a presentation exceeds max_padded_tokens {tokens}; no implicit truncation')
    batches, start = [], 0
    while start < len(lengths):
        stop, longest = start + 1, lengths[start]
        while stop < len(lengths) and (presentations is None or stop - start < presentations) \
                and (tokens is None or max(longest, lengths[stop]) * (stop + 1 - start) <= tokens):
            longest, stop = max(longest, lengths[stop]), stop + 1
        batches.append(range(start, stop))
        start = stop
    return batches


def packed(answers):
    import torch
    count, width = max(len(entry.groups) for entry in answers), max(entry.index.shape[1] for entry in answers)
    index, valid = np.zeros((len(answers), count, width), dtype=np.int64), np.zeros((len(answers), count, width), dtype=bool)
    present = np.zeros((len(answers), count), dtype=bool)
    for row, entry in enumerate(answers):
        groups, inner = entry.index.shape
        index[row, :groups, :inner], valid[row, :groups, :inner], present[row, :groups] = entry.index, entry.valid, True
    gold = np.fromiter((entry.label for entry in answers), dtype=np.int64, count=len(answers))
    return PackedSlots(*(torch.from_numpy(array) for array in (index, valid, present, gold)))


def bounded(shifted):
    return shifted.masked_fill_(shifted > 0, 0.).exp()


def slot_statistics(logits, slots):
    import torch
    values, excluded = logits.float(), float('-inf')
    rows = torch.arange(values.shape[0], device=values.device)
    answers = values.gather(1, slots.index.flatten(1)).view(slots.index.shape).masked_fill(~slots.valid, excluded).logsumexp(-1).masked_fill(~slots.present, excluded)
    gold, inside = slots.index[rows, slots.gold], slots.valid[rows, slots.gold]
    with torch.no_grad():
        top, at = values.topk(min(values.shape[-1], gold.shape[-1] + 1), -1)
        member = ((at[:, :, None] == gold[:, None, :]) & inside[:, None, :]).any(-1)
        peak = top.masked_fill(member, excluded).amax(-1)
        whole = torch.isneginf(peak)
        peak = torch.where(whole, torch.zeros_like(peak), peak)[:, None]
    every = bounded(values - peak).sum(-1)
    golden = (bounded(values.gather(1, gold) - peak) * inside).sum(-1)
    complement = torch.where(whole, excluded, peak[:, 0] + torch.where(whole, torch.ones_like(every), every - golden).log())
    return SlotStatistics(answers, answers[rows, slots.gold], complement, answers.logsumexp(-1), values.logsumexp(-1))


def slot_loss(statistics, gold):
    import torch
    softplus = torch.nn.functional.softplus
    other = statistics.answers.scatter(1, gold[:, None], float('-inf')).logsumexp(-1)
    return SlotLoss(softplus(statistics.complement - statistics.chosen), softplus(other - statistics.chosen), (statistics.union - statistics.total).exp())


def collate(entries, pad):
    import torch
    lengths = [entry.length for entry in entries]
    tokens = np.full((len(entries), max(lengths)), pad, dtype=np.int64)
    for index, entry in enumerate(entries):
        start = 0
        for segment in entry.segments:
            stop = start + len(segment)
            tokens[index, start:stop] = segment.array if isinstance(segment, Tokens) else segment
            start = stop
    mask = (np.arange(tokens.shape[1])[None, :] < np.asarray(lengths)[:, None]).astype(np.int64)
    return torch.from_numpy(tokens), torch.from_numpy(mask), lengths


def per_layer_model(model):
    return next((module for name, module in model.named_modules() if name.endswith('language_model') and hasattr(module, 'embed_tokens_per_layer')), None)


def bind(model):
    hf = model.get_base_model() if hasattr(model, 'get_base_model') else model
    config = hf.config.get_text_config() if hasattr(hf.config, 'get_text_config') else hf.config
    if config.model_type not in ('qwen3', 'gemma4_text', 'gemma4'):
        raise ConfigError(f'unsupported LoRA forward profile {config.model_type!r}; supported: qwen3 and gemma4 text trunks')
    device = hf.get_input_embeddings().weight.device
    language = per_layer_model(hf)
    host = language if language is not None and language.embed_tokens_per_layer.weight.device != device else None
    return Binding(model, hf.model, hf.get_output_embeddings(), getattr(config, 'final_logit_softcapping', None), host, device, config.vocab_size)


def host_table_inputs(language, tokens, placed):
    return {'inputs_embeds': language.embed_tokens(placed), 'per_layer_inputs': language.get_per_layer_inputs(tokens, None).to(placed.device, non_blocking=True)}


def answer_logits(binding, entries, pad, phases=UNTIMED):
    import torch
    with phases('collate'):
        tokens, mask, lengths = collate(entries, pad)
        last = torch.tensor(lengths) - 1
    with phases('transfer'):
        placed, mask, last = (value.to(binding.device, non_blocking=True) for value in (tokens, mask, last))
    with phases('host_embedding'):
        inputs = host_table_inputs(binding.host, tokens, placed) if binding.host is not None else {'input_ids': placed}
    with phases('forward'):
        hidden = binding.trunk(**inputs, attention_mask=mask, use_cache=False).last_hidden_state
        logits = binding.head(hidden[torch.arange(len(entries), device=hidden.device), last])
        if binding.softcap is not None:
            logits = torch.tanh(logits / binding.softcap) * binding.softcap
        return logits


def presented(binding, entries, pad, phases=UNTIMED):
    with phases('slot_packing'):
        slots = packed([entry.answers for entry in entries])
    with phases('transfer'):
        slots = slots.to(binding.device)
    return answer_logits(binding, entries, pad, phases), slots


def reduced(logits, slots, phases=UNTIMED):
    import torch
    with phases('reduction'):
        statistics = slot_statistics(logits, slots)
        return statistics, slot_loss(statistics, slots.gold), torch.isfinite(logits).all()


def staged(presentations, slots):
    stages = {}
    for entry in presentations:
        if entry.variant == 'canonical':
            mapped = entry.slot_map[slots[entry.id]]
            stages[entry.stage] = mapped if mapped >= 0 else None
    return {'answer': canonical_answer(presentations, slots), 'stages': stages}


def distribution(row, conditioned, log_mass):
    canonical = [primitive([row['classes'][slot] if slot >= 0 else ESCAPE for slot in entry.slot_map], [math.exp(value) for value in logp[:len(entry.slot_map)]],
                           stage=entry.stage, normalization=OPTION_SOFTMAX, mass=math.exp(mass))
                 for entry, logp, mass in zip(row['presentations'], conditioned, log_mass) if entry.variant == 'canonical']
    final = canonical[0] if len(canonical) == 1 and canonical[0]['stage'] == 'full' else None
    if final is None:
        return observed(row['classes'], None, stage='staged', normalization=OPTION_SOFTMAX, status='monitored', source='judge', stages=canonical)
    return observed(final['options'], final['probabilities'], stage='final', normalization=OPTION_SOFTMAX, status='monitored', source='judge',
                    mass=final['mass'])



def semantic_distribution(row, entry, logp):
    return {row['classes'][slot] if slot >= 0 else ESCAPE: math.exp(float(value)) for slot, value in zip(entry.slot_map, logp)}


def permutation_diagnostics(row, conditioned, log_mass):
    canonical = {entry.stage: (entry, logp, mass) for entry, logp, mass in zip(row['presentations'], conditioned, log_mass) if entry.variant == 'canonical'}
    out = []
    for entry, logp, mass in zip(row['presentations'], conditioned, log_mass):
        if entry.variant == 'canonical':
            continue
        result = {'stage': entry.stage, 'variant': entry.variant, 'comparable': False}
        if entry.stage not in canonical:
            out.append({**result, 'reason': 'missing_canonical'})
            continue
        original, reference, original_mass = canonical[entry.stage]
        left, right = semantic_distribution(row, original, reference), semantic_distribution(row, entry, logp)
        if left.keys() != right.keys():
            out.append({**result, 'reason': 'unequal_support'})
            continue
        keys = sorted(left)
        differences = [abs(left[key] - right[key]) for key in keys]
        if not all(math.isfinite(v) for v in [*left.values(), *right.values(), mass, original_mass]):
            out.append({**result, 'reason': 'invalid_distribution'})
            continue
        out.append({**result, 'comparable': True, 'disagreement': max(keys, key=left.get) != max(keys, key=right.get),
                    'mean_probability_delta': sum(differences) / len(keys), 'max_probability_delta': max(differences),
                    'mass_delta': abs(math.exp(mass) - math.exp(original_mass))})
    return out

def learning_rate_factor(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    return 1. if schedule == 'constant' else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def scheduler(torch, optimizer, resolved):
    plan, recipe = resolved['schedule'], resolved['recipe']
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, plan['max_optimizer_steps'], plan['warmup_steps'], recipe['schedule']))


def optimizer_of(torch, model, resolved):
    trainable = [p for p in model.parameters() if p.requires_grad]
    fused = resolved.get('fitting', {}).get('fused_optimizer', False)
    optimizer = torch.optim.AdamW(trainable, lr=resolved['recipe']['learning_rate'], weight_decay=resolved['recipe']['weight_decay'], **({'fused': True} if fused else {}))
    return trainable, optimizer, scheduler(torch, optimizer, resolved)


def batches(rows, recipe, *, epoch=0, offset=0):
    for current in range(epoch, recipe['epochs']):
        order = list(range(len(rows)))
        random.Random(artifact_digest([recipe['seed'], 'epoch', current])).shuffle(order)
        for start in range(offset if current == epoch else 0, len(order), recipe['batch_size']):
            yield [rows[index] for index in order[start:start + recipe['batch_size']]]


def accumulate(binding, batch, inputs, pad, microbatch, phases=UNTIMED, *, tokens=None):
    import torch
    items = sorted(((entry, inputs[row].objective * weight / len(batch)) for row in batch for entry, weight in zip(inputs[row].presentations, inputs[row].weights)),
                   key=lambda item: item[0].length)
    loss, finite, padded_tokens = 0., True, 0
    for chunk in planned([entry.length for entry, _ in items], microbatch, tokens=tokens):
        entries = [items[at][0] for at in chunk]
        padded_tokens += max(entry.length for entry in entries) * len(entries)
        logits, slots = presented(binding, entries, pad, phases)
        _, losses, ok = reduced(logits, slots, phases)
        with phases('reduction'):
            total = (losses.nll * torch.tensor([items[at][1] for at in chunk], dtype=torch.float32, device=losses.nll.device)).sum()
        with phases('backward'):
            total.backward()
        loss, finite = loss + total.detach(), ok & finite
    return {'loss': loss, 'finite': finite, 'tokens': sum(entry.length for entry, _ in items), 'presentations': len(items), 'padded_tokens': padded_tokens}


def update(binding, batch, inputs, pad, recipe, trained, phases=UNTIMED):
    import torch
    trainable, optimizer, rates = trained
    stats = accumulate(binding, batch, inputs, pad, recipe['microbatch'], phases, tokens=recipe['max_padded_tokens'])
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
    return {'learning_rate': rate, 'loss': loss, 'gradient_norm': float(norm), 'tokens': stats['tokens'], 'presentations': stats['presentations'], 'padded_tokens': stats['padded_tokens']}


def windows(rows, inputs, tokens):
    window, held = [], 0
    for row in rows:
        size = inputs[row].tokens
        if window and held + size > tokens:
            yield window
            window, held = [], 0
        window.append(row)
        held += size
    if window:
        yield window


def readout(binding, window, inputs, pad, tokens, phases=UNTIMED, *, presentations=None, telemetry=None):
    import torch
    items = sorted(((index, slot, entry) for index, row in enumerate(window) for slot, entry in enumerate(inputs[row].presentations)),
                   key=lambda item: item[2].length)
    width = max(len(entry.answers.groups) for _, _, entry in items)
    parts, chunks = [], planned([entry.length for _, _, entry in items], presentations, tokens=tokens)
    for chunk in chunks:
        entries = [items[at][2] for at in chunk]
        logits, slots = presented(binding, entries, pad, phases)
        statistics, losses, finite = reduced(logits, slots, phases)
        with phases('reduction'):
            conditioned = statistics.answers.log_softmax(-1)
            padded = torch.nn.functional.pad(conditioned, (0, width - conditioned.shape[1]), value=float('-inf'))
            parts.append(torch.cat([losses.nll[:, None], (statistics.union - statistics.total)[:, None], finite.float().expand(len(entries))[:, None], padded], 1).cpu().double().numpy())
            if telemetry is not None:
                telemetry['forwards'] = telemetry.get('forwards', 0) + 1
                telemetry['padded_tokens'] = telemetry.get('padded_tokens', 0) + max(entry.length for entry in entries) * len(entries)
                telemetry['max_length'] = max(telemetry.get('max_length', 0), max(entry.length for entry in entries))
                sizes = telemetry.setdefault('batch_sizes', {})
                sizes[str(len(entries))] = sizes.get(str(len(entries)), 0) + 1
        del logits, slots, statistics, losses, conditioned, padded
    with phases('readout'):
        values = np.concatenate(parts)
    out = [[None] * len(inputs[row].presentations) for row in window]
    for position, at in enumerate(at for chunk in chunks for at in chunk):
        index, slot, _ = items[at]
        out[index][slot] = values[position]
    return out


def evaluate(binding, rows, inputs, pad, tokens, poll, charge=lambda tokens: None, phases=UNTIMED, *, presentations=None, telemetry=None, journal=None):
    import torch
    model = binding.model
    was = model.training
    model.eval()
    try:
        out = {}
        with torch.inference_mode():
            for window in windows(rows, inputs, tokens):
                key = artifact_digest([row['id'] for row in window])
                if journal is not None and key in journal.current:
                    out.update(journal.payload(key))
                    charge(sum(inputs[row].tokens for row in window))
                    if telemetry is not None:
                        telemetry['reused_windows'] = telemetry.get('reused_windows', 0) + 1
                    continue
                poll()
                block = {}
                for row, values in zip(window, readout(binding, window, inputs, pad, tokens, phases, presentations=presentations, telemetry=telemetry)):
                    entries, prepared_ = row['presentations'], inputs[row]
                    nll = sum(weight * value[0] for weight, value in zip(prepared_.weights, values))
                    if not math.isfinite(nll) or not all(value[2] == 1 for value in values):
                        raise NumericalFailure(f"nonfinite model output evaluating {row['id']}; no candidate published")
                    widths = [len(entry.answers.groups) for entry in prepared_.presentations]
                    conditioned = [value[3:3 + groups] for value, groups in zip(values, widths)]
                    predicted = staged(entries, {entry.id: int(np.argmax(logp)) for entry, logp in zip(entries, conditioned)})
                    probability = distribution(row, [logp.tolist() for logp in conditioned], [float(value[1]) for value in values])
                    block[row['id']] = {'nll': float(nll), 'predicted': predicted['answer'], 'stages': predicted['stages'],
                                      'correct': predicted['answer'] == row['label'], 'probability': probability,
                                        'permutations': permutation_diagnostics(row, conditioned, [float(value[1]) for value in values])}
                charge(sum(inputs[row].tokens for row in window))
                if journal is not None:
                    journal.commit([(key, block)])
                out.update(block)
        return out
    finally:
        model.train(was)
