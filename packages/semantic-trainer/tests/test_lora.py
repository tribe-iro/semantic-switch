
from semantic_switch import model_jobs
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from semantic_testkit.candidates import fake_fitting, tiny_qwen
from semantic_trainer.jobs import presentation_weights
from semantic_trainer.lora import targets
from semantic_testkit.labels import presentation
from semantic_trainer.objective import Presented, answer_logits, answers_of, bind, collate, packed, prepared, slot_loss, slot_statistics

TOKENS = 1024


def slots_of(entry):
    return tuple(map(tuple, entry['answer_slots'])) if isinstance(entry, dict) else entry.answers.slots


def label_of(entry):
    return entry['label'] if isinstance(entry, dict) else entry.label


def answers(entry, vocabulary):
    return answers_of(slots_of(entry), label_of(entry), vocabulary)


def group_loss(logits, entries):
    slots = packed([answers(entry, logits.shape[-1]) for entry in entries])
    return slot_loss(slot_statistics(logits, slots), slots.gold)


def vocabulary_of(model):
    return bind(model).vocabulary


def sequences(entries):
    found = [(tuple(entry['prefix']), tuple(entry['input']), tuple(entry['question'])) if isinstance(entry, dict) else (entry.prefix, entry.input, entry.question)
             for entry in entries]
    return [Presented(segments, None, sum(map(len, segments))) for segments in found]


def reference(logits, entries):
    rows = []
    for row, entry in zip(logits.detach().double(), entries):
        z = row.clone().requires_grad_()
        groups = slots_of(entry)
        gold, union = list(groups[label_of(entry)]), [token for group in groups for token in group]
        total, chosen, legal = torch.logsumexp(z, 0), torch.logsumexp(z[gold], 0), torch.logsumexp(z[union], 0)
        rows.append((z, total - chosen, legal - chosen, torch.exp(legal - total)))
    return rows


def reference_gradients(logits, entries, weights):
    gradients = []
    for (z, nll, _, _), weight in zip(reference(logits, entries), weights):
        (nll * weight).backward()
        gradients.append(z.grad)
    return torch.stack(gradients)


CASES = [
    ([[[0, 1], [2]], [[3], [4, 5, 6], [7]], [[8], [9]]], [1, 2, 0]),
    ([[[0], [1, 2, 3, 4]], [[5, 6, 7], [8], [9], [10]], [[11, 12], [13]]], [0, 3, 1]),
    ([[[4, 5], [0, 1, 2], [3]], [[2], [1], [0]], [[9, 8, 7, 6], [5]]], [2, 1, 1]),
]


@pytest.mark.parametrize('groups,labels', CASES)
@pytest.mark.parametrize('scale', [1., 30., 3000.])
def test_full_vocabulary_loss_mass_and_every_logit_gradient_match_float64_reference(groups, labels, scale):
    torch.manual_seed(len(groups[0]) + int(scale))
    entries = [{'answer_slots': g, 'label': label} for g, label in zip(groups, labels)]
    logits = (torch.randn(len(entries), 16) * scale).requires_grad_()
    weights = [.5, .3, .2]
    losses = group_loss(logits, entries)
    expected = reference(logits, entries)
    tolerance = dict(rel=1e-5, abs=1e-5 * scale)
    for index, (_, nll, conditional, mass) in enumerate(expected):
        assert losses.nll[index].item() == pytest.approx(nll.item(), **tolerance)
        assert losses.conditional[index].item() == pytest.approx(conditional.item(), **tolerance)
        assert losses.mass[index].item() == pytest.approx(mass.item(), rel=1e-5, abs=1e-6)
    (losses.nll * torch.tensor(weights)).sum().backward()
    assert torch.allclose(logits.grad.double(), reference_gradients(logits, entries, weights), atol=1e-6, rtol=1e-4)


def test_padded_gold_group_keeps_the_dominant_wrong_token_in_the_complement():
    logits = torch.tensor([[4., 0., 0., 0.]], requires_grad=True)
    entry = {'answer_slots': [[0, 1], [2]], 'label': 1}
    losses = group_loss(logits, [entry])
    assert losses.nll.item() == pytest.approx(4.053491, abs=1e-6)
    losses.nll.sum().backward()
    assert logits.grad[0, 0].item() == pytest.approx(math.exp(4) / (math.exp(4) + 3), rel=1e-6)
    assert logits.grad[0, 2].item() == pytest.approx(1 / (math.exp(4) + 3) - 1, rel=1e-6)
    batch = torch.tensor([[4., 0., 0., 0.], [0., 0., 1., 2.]], requires_grad=True)
    paired = group_loss(batch, [entry, {'answer_slots': [[3], [1, 2]], 'label': 0}])
    assert paired.nll[0].item() == pytest.approx(4.053491, abs=1e-6)
    paired.nll[0].backward()
    assert batch.grad[0, 0] > .9 and batch.grad[1].abs().sum() == 0


def test_packed_slots_hold_only_padded_answer_coordinates():
    slots = packed([answers(entry, 8) for entry in [{'answer_slots': [[0, 1, 2], [3]], 'label': 1}, {'answer_slots': [[5], [4], [6, 7]], 'label': 2}]])
    assert slots.index.shape == (2, 3, 3) and slots.valid.sum().item() == 8 and slots.present.tolist() == [[True, True, False], [True, True, True]]
    assert sorted((row, token) for row, groups in enumerate(slots.index.tolist()) for column, group in enumerate(groups) for at, token in enumerate(group)
                  if slots.valid[row, column, at]) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7)]
    assert slots.gold.tolist() == [1, 2] and slots.index[1, 2, :2].tolist() == [6, 7]


@pytest.mark.parametrize('gap', [0., 1e-3, 40., 3000.])
def test_dominant_gold_complement_stays_stable_without_vocabulary_masks(gap):
    logits = torch.zeros(1, 4096, dtype=torch.float32)
    logits[0, :3] = gap
    logits = logits.requires_grad_()
    entry = {'answer_slots': [[0, 1, 2], [7]], 'label': 0}
    losses = group_loss(logits, [entry])
    (z, nll, conditional, mass), = reference(logits, [entry])
    assert losses.nll.item() == pytest.approx(nll.item(), rel=1e-5, abs=1e-12) and losses.mass.item() == pytest.approx(mass.item(), rel=1e-6)
    losses.nll.sum().backward()
    nll.backward()
    assert torch.allclose(logits.grad.double(), z.grad, atol=1e-9, rtol=1e-4) and torch.isfinite(logits.grad).all()


def test_answer_tokens_outside_the_model_vocabulary_are_configuration_errors():
    from semantic_switch.errors import ConfigError
    with pytest.raises(ConfigError, match='outside the 4-token model vocabulary'):
        group_loss(torch.zeros(1, 4), [{'answer_slots': [[0], [9]], 'label': 0}])


def test_all_vocabulary_gold_group_has_zero_loss_unit_mass_and_finite_gradients():
    logits = torch.tensor([[1., -2., 3.]], requires_grad=True)
    losses = group_loss(logits, [{'answer_slots': [[0, 1, 2]], 'label': 0}])
    assert losses.nll.item() == 0 and losses.conditional.item() == 0 and losses.mass.item() == pytest.approx(1)
    (losses.nll.sum() + losses.mass.sum()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().max() < 1e-6


def test_answer_probabilities_match_the_native_reducer_on_fixed_logits():
    import numpy as np
    from semantic_switch.backend import reduce_row
    from semantic_trainer.objective import packed, slot_statistics
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(3, 64, generator=generator) * 6
    entries = [{'answer_slots': [[0, 1], [2]], 'label': 1}, {'answer_slots': [[10], [11, 12, 13], [14]], 'label': 0}, {'answer_slots': [[40, 41], [42, 43]], 'label': 1}]
    statistics = slot_statistics(logits, packed([answers(entry, 64) for entry in entries]))
    for row, entry in enumerate(entries):
        logp, log_mass = reduce_row(logits[row].numpy(), tuple(tuple(group) for group in entry['answer_slots']))
        width = len(entry['answer_slots'])
        assert np.allclose(statistics.answers[row, :width].log_softmax(-1).numpy(), logp, atol=1e-5)
        assert (statistics.union[row] - statistics.total[row]).item() == pytest.approx(log_mass, abs=1e-5)


def test_nonfinite_model_outputs_are_numerical_failures():
    from semantic_trainer.objective import NumericalFailure, evaluate
    model = adapted_tiny()
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if 'lora_B' in n).fill_(float('nan'))
    with pytest.raises(NumericalFailure, match='nonfinite model output'):
        evaluate(bind(model), fit_rows(), prepared(fit_rows(), vocabulary_of(model)), 0, TOKENS, lambda: 1)


def test_lora_padding_has_no_loss():
    entries = [{'prefix': [1], 'input': [2], 'question': [3]}, {'prefix': [1], 'input': [], 'question': [4]}]
    tokens, mask, lengths = collate(sequences(entries), 0)
    assert tokens.tolist() == [[1, 2, 3], [1, 4, 0]] and tokens.device.type == 'cpu'
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert lengths == [3, 2]


def tiny_gemma():
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    torch.manual_seed(0)
    text = dict(vocab_size=64, vocab_size_per_layer_input=64, hidden_size=16, hidden_size_per_layer_input=4, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8, num_kv_shared_layers=0,
                layer_types=['sliding_attention', 'full_attention'], sliding_window=8, pad_token_id=0)
    vision = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=4)
    return Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision, audio_config=None, image_token_id=60, video_token_id=61,
                                                       audio_token_id=62, boi_token_id=58, eoi_token_id=59)).eval()


def test_accelerator_placement_keeps_only_the_gemma_per_layer_table_on_the_host():
    from semantic_trainer.lora import HOST_TABLE, place
    model = tiny_gemma()
    assert place(model, torch.device('meta')) == {'': 'meta', HOST_TABLE: 'cpu'}
    table = model.get_submodule(HOST_TABLE)
    assert {tensor.device.type for tensor in (*table.parameters(), *table.buffers())} == {'cpu'} and table.weight.shape == (64, 8)
    others = [tensor for name, tensor in (*model.named_parameters(), *model.named_buffers()) if not name.startswith(HOST_TABLE + '.')]
    assert others and {tensor.device.type for tensor in others} == {'meta'}


def test_gathered_head_matches_the_full_gemma_forward_with_softcapping_and_gradients():
    from peft import LoraConfig, get_peft_model
    model = tiny_gemma()
    model.config.get_text_config().final_logit_softcapping = 2.
    model.requires_grad_(False)
    adapted = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=list(targets(model)), task_type='CAUSAL_LM')).train()
    for name, parameter in adapted.named_parameters():
        if 'lora_B' in name:
            parameter.data.normal_()
    entries = [{'prefix': [2, 5], 'input': [7, 9, 13], 'question': [11]}, {'prefix': [2, 6], 'input': [], 'question': [10]}]
    tokens, mask, lengths = collate(sequences(entries), 0)
    full = adapted(input_ids=tokens, attention_mask=mask, use_cache=False).logits[torch.arange(2), torch.tensor(lengths) - 1]
    assert full.abs().max() <= 2.
    weights = torch.randn_like(full)
    (full * weights).sum().backward()
    expected = {name: parameter.grad.clone() for name, parameter in adapted.named_parameters() if parameter.requires_grad}
    adapted.zero_grad(set_to_none=True)
    binding = bind(adapted)
    assert binding.softcap == 2. and binding.host is None
    gathered = answer_logits(binding, sequences(entries), 0)
    assert gathered.shape == (2, 64) and torch.allclose(gathered, full.detach(), atol=1e-6, rtol=1e-5)
    (gathered * weights).sum().backward()
    assert expected and all(torch.allclose(parameter.grad, expected[name], atol=1e-6, rtol=1e-4) for name, parameter in adapted.named_parameters() if parameter.requires_grad)


def test_head_projects_one_row_per_presentation():
    model = adapted_tiny()
    binding = bind(model)
    rows = []
    handle = binding.head.register_forward_hook(lambda module, inputs, output: rows.append(tuple(inputs[0].shape)))
    entries = [entry for row in fit_rows() for entry in row['presentations']]
    answer_logits(binding, sequences(entries), 0)
    handle.remove()
    assert rows == [(len(entries), 16)]


def test_host_per_layer_table_is_file_mapped_from_the_snapshot(tmp_path):
    from semantic_trainer.lora import HOST_TABLE, map_host_table
    from semantic_trainer.objective import host_table_inputs, per_layer_model
    model = tiny_gemma().to(torch.bfloat16)
    model.save_pretrained(tmp_path / 'hf', safe_serialization=True)
    from transformers import Gemma4ForConditionalGeneration
    loaded = Gemma4ForConditionalGeneration.from_pretrained(tmp_path / 'hf', dtype=torch.bfloat16).eval()
    tokens, mask, _ = collate(sequences([{'prefix': [2, 5], 'input': [7, 9], 'question': [11]}]), 0)
    with torch.no_grad():
        expected = loaded(**host_table_inputs(per_layer_model(loaded), tokens, tokens), attention_mask=mask, use_cache=False).logits
        assert map_host_table(loaded, tmp_path / 'hf') == 'cpu:file-mapped'
        table = loaded.get_submodule(HOST_TABLE).weight
        assert not table.requires_grad and torch.equal(table, model.get_submodule(HOST_TABLE).weight)
        assert torch.equal(loaded(**host_table_inputs(per_layer_model(loaded), tokens, tokens), attention_mask=mask, use_cache=False).logits, expected)
    assert map_host_table(loaded.float(), tmp_path / 'hf') == 'cpu'
    with pytest.raises(Exception, match='not in the safetensors snapshot'):
        map_host_table(loaded, tmp_path)


def test_host_per_layer_table_inputs_reproduce_token_inputs_exactly():
    from peft import LoraConfig, get_peft_model
    from semantic_trainer.objective import host_table_inputs, per_layer_model
    model = tiny_gemma()
    model.requires_grad_(False)
    adapted = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=list(targets(model)), task_type='CAUSAL_LM')).eval()
    tokens, mask, _ = collate(sequences([{'prefix': [2, 5], 'input': [7, 9], 'question': [11]}, {'prefix': [2, 6], 'input': [8], 'question': [10]}]), 0)
    for wrapped in (model, adapted):
        language = per_layer_model(wrapped)
        assert language is model.model.language_model
        with torch.no_grad():
            expected = wrapped(input_ids=tokens, attention_mask=mask, use_cache=False).logits
            assert torch.equal(wrapped(**host_table_inputs(language, tokens, tokens), attention_mask=mask, use_cache=False).logits, expected)


def test_lora_observation_weight_independent_of_augmentation():
    row = {'presentations': [presentation(stage='mentions'), presentation(stage='mentions'), presentation(stage='fallback')]}
    assert presentation_weights(row) == [.25, .25, .5]
    assert sum(presentation_weights(row)) == 1


def test_lora_targets_resolve_before_optimizer():
    from semantic_switch.errors import ConfigError
    with pytest.raises(ConfigError, match='unsupported'):
        targets(SimpleNamespace(config=SimpleNamespace(model_type='unknown')))


@pytest.mark.parametrize('shared,equal,values', [(1, False, (0, 1)), (1, True, (0,))])
def test_gemma_targets_exclude_vision_and_respect_shared_kv(shared, equal, values):
    config = SimpleNamespace(num_hidden_layers=3, num_kv_shared_layers=shared,
                             attention_k_eq_v=equal, layer_types=['sliding_attention', 'full_attention', 'sliding_attention'])
    modules = [(f'model.language_model.layers.{i}.self_attn.q_proj', torch.nn.Linear(4, 4)) for i in range(3)]
    modules += [(f'model.language_model.layers.{i}.self_attn.v_proj', torch.nn.Linear(4, 2)) for i in values]
    modules.append(('model.vision_model.layers.0.self_attn.q_proj', torch.nn.Linear(4, 4)))
    model = SimpleNamespace(config=SimpleNamespace(model_type='gemma4', text_config=config), named_modules=lambda: modules)
    resolved = targets(model, ('q_proj', 'v_proj'))
    assert len(resolved) == 3 + len(values)
    assert all('.language_model.' in name for name in resolved)
    assert all(shape in ([4, 4], [2, 4]) for shape in resolved.values())


def test_gemma_all_linear_targets_follow_shared_kv_and_reject_missing_projections():
    from semantic_switch.errors import ConfigError
    config = SimpleNamespace(num_hidden_layers=3, num_kv_shared_layers=1, attention_k_eq_v=False, layer_types=['sliding_attention'] * 3)
    modules = [(f'model.language_model.layers.{i}.{parent}.{name}', torch.nn.Linear(4, 4)) for i in range(3)
               for parent, names in (('self_attn', ('q_proj', 'o_proj')), ('mlp', ('gate_proj', 'up_proj', 'down_proj'))) for name in names]
    modules += [(f'model.language_model.layers.{i}.self_attn.{name}', torch.nn.Linear(4, 2)) for i in range(2) for name in ('k_proj', 'v_proj')]
    modules.append(('model.audio_tower.layers.0.mlp.up_proj', torch.nn.Linear(4, 4)))
    model = SimpleNamespace(config=SimpleNamespace(model_type='gemma4', text_config=config), named_modules=lambda: modules)
    assert len(targets(model)) == 3 * 5 + 2 * 2
    model = SimpleNamespace(config=model.config, named_modules=lambda: [m for m in modules if not m[0].endswith('2.mlp.down_proj')])
    with pytest.raises(ConfigError, match='target resolution'):
        targets(model)


def test_only_lora_parameters_receive_gradients():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=list(targets(base)), task_type='CAUSAL_LM'))
    entry = {'prefix': [1], 'input': [2], 'question': [3]}
    loss, _, _ = group_loss(answer_logits(bind(model), sequences([entry]), 0), [{'answer_slots': [[4], [5]], 'label': 0}])
    loss.backward()
    assert all('lora_' in name for name, parameter in model.named_parameters() if parameter.grad is not None)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for name, parameter in model.named_parameters() if 'lora_' in name)


def scored(step, value, nll=1.):
    return {'step': step, 'examples_seen': step * 8, 'nll': nll, 'correct': {'macro': {'complete_situation': value}}}


def test_stopping_anchors_on_the_primary_score_accumulates_small_gains_and_ignores_nll():
    from semantic_trainer.jobs import advance, plateaued
    recipe = {'min_epochs': 1, 'patience_examples': 16}
    control = advance(None, scored(0, 0.), 'complete_situation', .01)
    assert control['anchor'] == control['best'] == 0. and control['improved_at']['examples'] == 0
    for step, value in ((1, .006), (2, .012)):
        control = advance(control, scored(step, value, nll=step), 'complete_situation', .01)
    assert control['anchor'] == .012 and control['improved_at'] == {'step': 2, 'examples': 16}
    control = advance(control, scored(3, .012, nll=.1), 'complete_situation', .01)
    assert control['improved_at']['step'] == 2 and not plateaued(control, 24, recipe, 8) and plateaued(control, 32, recipe, 8)
    worse_nll = advance(control, scored(4, .5, nll=9.), 'complete_situation', .01)
    assert worse_nll['improved_at']['step'] == 4 and not plateaued(worse_nll, 40, recipe, 8)
    again = advance(advance(control, scored(4, .5), 'complete_situation', .01), scored(4, .5), 'complete_situation', .01)
    assert again == advance(control, scored(4, .5), 'complete_situation', .01)


def test_previous_and_best_class_regressions_are_visible_when_the_base_scored_zero():
    from semantic_switch.training import class_changes, regressions
    cell = lambda correct, gold=22: {'gold': gold, 'predicted': 0, 'correct': correct, 'recall': correct / gold}
    history = [{'step': 0, 'per_class': {'t:ribbon_loaded': {'false': cell(0)}}}, {'step': 150, 'per_class': {'t:ribbon_loaded': {'false': cell(22)}}}]
    current = {'t:ribbon_loaded': {'false': cell(0)}}
    changes = class_changes(current, history)
    assert changes['base'] == [] and regressions(current, history[0]['per_class']) == []
    assert [(e['reference_step'], e['reference_correct'], e['correct']) for e in changes['previous']] == [(150, 22, 0)]
    assert [(e['reference_step'], e['reference_correct']) for e in changes['best']] == [(150, 22)]
    assert regressions({'t:x': {'a': cell(1, 5)}}, {'t:x': {'a': cell(3, 6)}}) == []


def test_tokenizer_probes_and_answer_tokens_are_checked_once_per_descriptor():
    from semantic_switch.artifacts import artifact_digest
    from semantic_trainer.lora import verify_tokenizer
    vocab = {'a': 0, 'b': 1, 'c': 2}
    descriptor = {'vocab_size': 3, 'vocab_sha256': artifact_digest(['a', 'b', 'c']), 'special': {}, 'probes': [{'text': 'ab', 'ids': [0, 1]}, {'text': 'c', 'ids': [2]}]}
    encoded, decoded = [], []
    tokenizer = SimpleNamespace(get_vocab=lambda: vocab, encode=lambda text, add_special_tokens: encoded.append(text) or [vocab[ch] for ch in text],
                                decode=lambda ids, skip_special_tokens: decoded.append(ids[0]) or 'abc'[ids[0]], bos_token_id=None)
    shown = presentation(answer_slots=[[0], [1, 2]], answer_text=[['a'], ['b', 'c']])
    rows = [{'native': {'tokenizer': dict(descriptor)}, 'presentations': [shown, shown]} for _ in range(4)]
    assert verify_tokenizer(tokenizer, rows) == descriptor
    assert sorted(encoded) == ['ab', 'c'] and sorted(decoded) == [0, 1, 2]
    from semantic_switch.errors import ConfigError
    with pytest.raises(ConfigError, match='different tokenizers'):
        verify_tokenizer(tokenizer, [*rows, {'native': {'tokenizer': {**descriptor, 'probes': []}}, 'presentations': []}])
    with pytest.raises(ConfigError, match='answer-token decoding'):
        verify_tokenizer(tokenizer, [*rows, {'native': {'tokenizer': descriptor}, 'presentations': [presentation(answer_slots=[[0]], answer_text=[['b']])]}])


def test_lora_training_rejects_tokenizer_mismatch():
    from semantic_switch.errors import ConfigError
    from semantic_trainer.lora import verify_tokenizer
    tokenizer = SimpleNamespace(get_vocab=lambda: {'x': 0, 'y': 1})
    with pytest.raises(ConfigError, match='vocabulary mismatch'):
        verify_tokenizer(tokenizer, [{'native': {'tokenizer': {'vocab_size': 2, 'vocab_sha256': 'f' * 64}}, 'presentations': []}])


def test_answer_only_projection_matches_full_logits_and_gradients():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()
    entries = [{'prefix': [1], 'input': [2], 'question': [3]}, {'prefix': [1], 'input': [], 'question': [4]}]
    tokens, mask, lengths = collate(sequences(entries), 0)
    full = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits[torch.arange(2), torch.tensor(lengths) - 1]
    full.sum().backward()
    gradients = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad()
    selected = answer_logits(bind(model), sequences(entries), 0)
    assert torch.allclose(full, selected, atol=1e-6, rtol=1e-5)
    selected.sum().backward()
    for parameter, expected in zip(model.parameters(), gradients):
        if expected is not None:
            assert torch.allclose(parameter.grad, expected, atol=1e-5, rtol=1e-4)


def monitored_job(tmp_path, monkeypatch, *, device='cpu', recipe=None, budget=None, roles=('fit', 'fit', 'model_select')):
    from semantic_switch import Pick, Runtime, api, files, tools
    from semantic_switch.artifacts import read_export
    from semantic_trainer import jobs
    from semantic_testkit.assets import fake_assets
    from semantic_testkit.labels import envelopes
    from semantic_testkit.scoring import FakeScorer
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    manifest, gguf = fake_assets(tmp_path / 'assets')
    original = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**original(self, role), 'sha256': files.sha256_file(str(gguf))})
    with Runtime(FakeScorer()) as runtime:
        path = tmp_path / 'action.yaml'
        Pick('Choose.', {'x': 'One', 'y': 'Two'}).save(path, rt=runtime)
        files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([{'input': {'message': f'case {i}'}, 'facts': {'action': 'xy'[i % 2]}, 'group': f'case-{i}',
                                                                               'role': roles[i // 2 % len(roles)]} for i in range(36)])))
        exported = tools.export(runtime, path, native=True, task_family='example:policy')
    monkeypatch.setattr(api.Runtime, 'identity', original)
    manifest, loaded, calls = fake_fitting(monkeypatch, manifest, read_export(exported['path'], native=True)[1])
    job = jobs.prepare(path, [exported['path']], tmp_path / 'job', assets_manifest=manifest, device=device, recipe={'batch_size': 8, 'eval_every_examples': 8, 'fit_probe_observations': 12, **(recipe or {})}, budget=budget)['job']
    return job, loaded, calls


def fake_native(monkeypatch):
    import numpy as np
    from semantic_switch import api
    from semantic_switch.protocols import ScoreBlock
    class Native:
        def score(self, requests):
            block = ScoreBlock.empty(len(requests), max(len(slots) for request in requests for slots in request.slots))
            block.logp[:] = np.log(1 / block.logp.shape[1])
            block.log_mass[:] = 0
            return block
    monkeypatch.setattr(api.Runtime, 'judge', lambda self, role='judge': SimpleNamespace(scorer=Native()))
    monkeypatch.setattr(api, '_local_identity', lambda spec, *args: {'sha256': 'f' * 64, 'template': 'fixture', 'prompt': 'p', 'presentation': {'revision': 1}})


def records(path):
    from semantic_switch import files
    return files.read_jsonl(path)


def test_trainer_dry_run_loads_no_weights(tmp_path, monkeypatch):
    from semantic_switch import api
    from semantic_trainer import lora
    job, loaded, calls = monitored_job(tmp_path, monkeypatch)
    result = lora.train(job, tmp_path / 'out', dry_run=True)
    assert loaded == [] and calls == [] and not (tmp_path / 'out').exists()
    assert result['dry_run'] and result['job']['schema'] == 11 and result['job']['loss']['version'] == 2 and result['job']['recipe']['epochs'] == 8
    assert result['job']['fitting'] == {'device': 'cpu', 'precision': 'float32', 'threads': 1, 'attention': 'eager', 'gradient_checkpointing': False,
                                        'fused_optimizer': False}
    admission = result['admission']
    estimated = {use: value['host'] for use, value in admission['estimated_bytes'].items()}
    assert estimated == {**lora.fit_bytes(result['job']), 'fit:prepared_inputs': estimated['fit:prepared_inputs']} and estimated['fit:adapter_parameters_gradients_adamw'] > 0
    assert admission['admitted_bytes'] == model_jobs.HOST_RESERVE + sum(estimated.values()) and admission['state'] == 'open'


def test_fit_monitors_from_point_zero_and_retains_the_two_best_plus_zero(tmp_path, monkeypatch, capsys):
    from semantic_switch import files
    from semantic_trainer import lora
    from semantic_trainer.jobs import ranking, resolve
    job, loaded, calls = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 2, 'eval_every_examples': 8, 'min_epochs': 1, 'patience_examples': 1000})
    fake_native(monkeypatch)
    result = lora.train(job, tmp_path / 'out')
    output = Path(result['path'])
    assert len(loaded) == 1 and loaded[0]() is None and calls[0]['device_map'] == {'': 'cpu'} and calls[0]['dtype'] == torch.float32
    trajectory, updates = records(output / 'progress.jsonl'), records(output / 'updates.jsonl')
    progress = [record for record in trajectory if record['kind'] == 'evaluation']
    resolved, observations = resolve(job)
    (feasible,) = [record for record in trajectory if record['kind'] == 'feasibility']
    assert feasible['feasible'] and feasible['measured']['updates_measured'] == 6 and feasible['projection']['evaluation_points'] == len(progress) == 7
    fit = sum(row['role'] == 'fit' for row in observations)
    assert progress[0]['step'] == 0 and progress[0]['examples_seen'] == 0 and progress[0]['train_loss'] is None
    seen = [point['examples_seen'] for point in progress]
    assert seen == sorted(set(seen)) and seen[-1] == 2 * fit and [point['step'] for point in progress] == [0, 1, 2, 3, 4, 5, 6]
    assert all(point['probe']['population'].startswith('in-sample') for point in progress)
    presentations = sum(len(row['presentations']) for row in observations if row['role'] == 'fit')
    assert progress[-1]['seen'] == {'situations': fit, 'targets': fit, 'presentations': presentations, 'families': fit} and presentations > fit
    assert all([set(paths) for paths in point['model_select']['per_class'].values()] == [{'action'}] for point in progress)
    assert [u['step'] for u in updates] == list(range(1, 7)) and all(u['learning_rate'] > 0 for u in updates)
    assert capsys.readouterr().err.count('"kind": "evaluation"') == len(progress)
    listed = files.read_json(output / 'candidates.json')
    names = [c['name'] for c in listed['candidates']]
    points = [{'name': f"step-{p['step']}", 'step': p['step'], 'correct': p['model_select']['correct'], 'nll': p['model_select']['nll']} for p in progress[1:]]
    assert names == ['zero', *(p['name'] for p in ranking(points, 'observation')[:2])] and listed['state'] == 'ready'
    assert set(names) <= {path.name for path in (output / 'candidates').iterdir()}
    assert len(list((output / 'candidates').iterdir())) <= 5
    for candidate in listed['candidates']:
        assert files.sha256_file(output / candidate['peft'] / 'adapter_model.safetensors') == candidate['peft_sha256']
        assert set(candidate['predictions']) == {row['id'] for row in observations if row['role'] == 'model_select'}
        assert all(set(p) == {'answer', 'stages'} and p['stages'] for p in candidate['predictions'].values())
    assert listed['job_id'] == result['job_id'] == resolved['id'] and listed['base_sha256'] == 'f' * 64 and listed['stop']['reason'] == 'epoch_cap'
    assert listed['stop']['convergence'] == 'not_demonstrated' and listed['stop']['unevaluated'] == {'steps': 0, 'examples': 0} and listed['loss'] == resolved['loss']
    assert listed['rule']['order'] == ['correct', 'regressions', 'nll', 'base', 'step'] and listed['stop']['best_monitored']['name'] == names[1]
    run = files.read_json(output / 'run.json')
    assert run['state'] == 'ready' and run['feasibility'] == feasible and run['loss'] == resolved['loss']
    from importlib.metadata import version
    from semantic_switch.artifacts import artifact_digest
    from semantic_trainer.adapter import COMPLETE
    fitted = files.read_json(output / 'resolved-job.json')
    assert listed['fitted_job_id'] == artifact_digest(fitted) and fitted['id'] == run['job_id'] == listed['job_id'] and fitted['dependencies']['peft'] == version('peft')
    assert list(fitted['adapter']) == list(COMPLETE) and fitted['adapter']['target_modules'] == sorted(fitted['targets'])
    assert (fitted['adapter']['r'], fitted['adapter']['lora_alpha'], fitted['adapter']['lora_dropout']) == (resolved['recipe']['rank'], resolved['recipe']['alpha'], resolved['recipe']['dropout'])
    assert files.read_json(output / 'metrics.json')['quality_claim'] is False


def test_fit_stops_on_patience_only_after_the_minimum_epochs(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import lora
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 8, 'min_epochs': 2, 'eval_every_examples': 8, 'patience_examples': 8, 'min_improvement': .999})
    fake_native(monkeypatch)
    result = lora.train(job, tmp_path / 'out')
    stop = files.read_json(Path(result['path']) / 'candidates.json')['stop']
    assert stop['reason'] == 'score_plateau' and stop['examples_trained'] == 2 * 24 and stop['steps_trained'] < stop['max_optimizer_steps']
    assert stop['last_evaluation']['examples_seen'] == stop['examples_trained'] and stop['convergence'] in ('score_plateau', 'not_demonstrated')
    assert stop['control']['improved_at']['examples'] <= stop['examples_trained'] - 8


def test_budget_exhaustion_finalizes_evaluated_points_or_fails_the_run(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_switch.errors import ConfigError
    from semantic_trainer import lora, monitor, objective
    with pytest.raises(ConfigError, match='predicted token work'):
        monitored_job(tmp_path / 'tight', monkeypatch, budget={'max_tokens': 400})
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 2, 'eval_every_examples': 8})
    fake_native(monkeypatch)
    original, calls = monitor.Monitor.point, []
    def interrupted(limit):
        def point(self, step, examples):
            calls.append(step)
            if len(calls) == limit:
                raise monitor.Exhausted('time_budget')
            return original(self, step, examples)
        return point
    monkeypatch.setattr(monitor.Monitor, 'point', interrupted(4))
    result = lora.train(job, tmp_path / 'out')
    listed = files.read_json(Path(result['path']) / 'candidates.json')
    assert listed['stop']['reason'] == 'time_budget' and [p['step'] for p in records(Path(result['path']) / 'progress.jsonl') if p['kind'] == 'evaluation'] == [0, 1, 2]
    assert {c['step'] for c in listed['candidates']} <= {0, 1, 2} and listed['stop']['convergence'] == 'not_demonstrated'
    assert listed['stop']['unevaluated'] == {'steps': 1, 'examples': 8} and listed['stop']['last_evaluation'] == {'step': 2, 'examples_seen': 16}
    calls.clear()
    monkeypatch.setattr(monitor.Monitor, 'point', interrupted(2))
    with pytest.raises(ConfigError, match='time_budget: no fully evaluated optimizer step'):
        lora.train(job, tmp_path / 'early')
    assert files.read_json(tmp_path / 'early' / 'run.json')['state'] == 'failed' and (tmp_path / 'early' / 'candidates' / 'zero').exists()
    monkeypatch.setattr(monitor.Monitor, 'point', original)
    def nonfinite(*args, **kwargs):
        return {'loss': torch.tensor(1.), 'finite': torch.tensor(False), 'tokens': 1, 'presentations': 1}
    monkeypatch.setattr(objective, 'accumulate', nonfinite)
    with pytest.raises(objective.NumericalFailure, match='nonfinite model output'):
        lora.train(job, tmp_path / 'failed')
    assert files.read_json(tmp_path / 'failed' / 'run.json')['state'] == 'failed'
    assert records(tmp_path / 'failed' / 'progress.jsonl')[-1]['kind'] == 'numerical_failure'
    assert not (tmp_path / 'failed' / 'candidates.json').exists() and (tmp_path / 'failed' / 'candidates' / 'zero').exists()


def test_evaluation_restores_training_mode_and_takes_no_gradients():
    from semantic_trainer.monitor import Exhausted
    from semantic_trainer.objective import evaluate
    model = adapted_tiny()
    rows = fit_rows()
    results = evaluate(bind(model), rows, prepared(rows, vocabulary_of(model)), 0, TOKENS, lambda: 1)
    assert model.training and all(p.grad is None for p in model.parameters())
    assert set(results) == {'a', 'b', 'c'} and all(set(r) == {'nll', 'predicted', 'stages', 'correct', 'probability', 'permutations'} for r in results.values())
    def exhausted():
        raise Exhausted('time_budget')
    with pytest.raises(Exhausted):
        evaluate(bind(model), rows, prepared(rows, vocabulary_of(model)), 0, TOKENS, exhausted)
    assert model.training


def test_point_update_cycles_retain_no_tensors():
    import gc
    from semantic_trainer.objective import evaluate, optimizer_of, update
    model, rows = adapted_tiny(), fit_rows()
    binding = bind(model)
    inputs = prepared(rows, binding.vocabulary)
    recipe = {'learning_rate': 1e-3, 'weight_decay': 0., 'schedule': 'constant', 'microbatch': 1, 'max_padded_tokens': 8192, 'gradient_clip': 1.}
    trained = optimizer_of(torch, model, {'recipe': recipe, 'schedule': {'max_optimizer_steps': 6, 'warmup_steps': 0}})
    def live():
        gc.collect()
        return sum(issubclass(type(value), torch.Tensor) for value in gc.get_objects())
    counts = []
    for _ in range(3):
        evaluate(binding, rows, inputs, 0, TOKENS, lambda: None)
        after_point = live()
        update(binding, rows, inputs, 0, recipe, trained)
        counts.append((after_point, live()))
    assert counts[1] == counts[2] and counts[0][1] == counts[1][1]
    assert all(p.grad is None for p in model.parameters())


def test_learning_rate_schedule_is_frozen_before_fitting_on_a_toy_optimizer():
    from semantic_trainer.objective import learning_rate_factor, scheduler
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    resolved = {'schedule': {'max_optimizer_steps': 10, 'warmup_steps': 4}, 'recipe': {'schedule': 'constant'}}
    rates = scheduler(torch, optimizer, resolved)
    used = []
    for _ in range(10):
        used.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        rates.step()
    assert used[:5] == pytest.approx([.025, .05, .075, .1, .1]) and used[-1] == pytest.approx(.1)
    cosine = [learning_rate_factor(step, 20, 4, 'cosine') for step in range(20)]
    assert cosine[:5] == [.25, .5, .75, 1., 1.] and all(a > b for a, b in zip(cosine[4:], cosine[5:])) and 0 < cosine[-1] < .02


def test_accelerator_job_places_model_and_reports_device_memory(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import lora
    monkeypatch.setattr(torch.xpu, 'is_available', lambda: True)
    monkeypatch.setattr(torch.xpu, 'device_count', lambda: 1)
    monkeypatch.setattr('semantic_trainer.jobs.physical_cores', lambda: 1)
    job, loaded, calls = monitored_job(tmp_path, monkeypatch, device='xpu:0', recipe={'epochs': 1, 'min_epochs': 1})
    fake_native(monkeypatch)
    events = []
    backend = SimpleNamespace(reset_peak_memory_stats=lambda device: events.append(('reset', str(device))),
                              max_memory_allocated=lambda device: 123, max_memory_reserved=lambda device: 456, empty_cache=lambda: events.append(('empty',)), get_rng_state=lambda device: torch.get_rng_state(), set_rng_state=lambda value, device: torch.set_rng_state(value))
    monkeypatch.setattr(lora, 'accelerator', lambda torch, device: backend if device.type == 'xpu' else None)
    result = lora.train(job, tmp_path / 'out')
    assert calls[0]['device_map'] == {'': 'xpu:0'} and calls[0]['dtype'] == torch.bfloat16
    assert events == [('reset', 'xpu:0'), ('empty',)]
    assert {(point['peak_device_bytes'], point['peak_reserved_device_bytes']) for point in records(Path(result['path']) / 'progress.jsonl')
            if point['kind'] == 'evaluation'} == {(123, 456)}


def adapted_tiny(vocab=32):
    from peft import LoraConfig, get_peft_model
    base = tiny_qwen(vocab)
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=list(targets(base)), task_type='CAUSAL_LM'))
    for name, parameter in model.named_parameters():
        if 'lora_B' in name:
            parameter.data.normal_()
    return model.train()


def fit_rows():
    def entry(index, tokens, label):
        return presentation(f'p{index}', 'full', 'canonical', [1], tokens, [3], [[4], [5, 6]], label, [0, 1])
    return [{'id': 'a', 'task': 't', 'objective': 1., 'classes': ['x', 'y'], 'label': 0, 'presentations': [entry(0, [7, 8, 9], 0), entry(1, [10], 0)]},
            {'id': 'b', 'task': 't', 'objective': 1., 'classes': ['x', 'y'], 'label': 1, 'presentations': [entry(2, [], 1)]},
            {'id': 'c', 'task': 't', 'objective': 1., 'classes': ['x', 'y'], 'label': 1, 'presentations': [entry(3, [11, 12], 1), entry(4, [13, 14, 15, 16], 1)]}]


def test_padded_forward_matches_unbatched_readout():
    model = adapted_tiny().eval()
    entries = [entry for row in fit_rows() for entry in row['presentations']]
    with torch.no_grad():
        padded = answer_logits(bind(model), sequences(entries), 0)
        single = torch.cat([answer_logits(bind(model), sequences([entry]), 0) for entry in entries])
    assert torch.allclose(padded, single, atol=1e-5, rtol=1e-4)


def test_monitor_windows_batch_rows_and_transfer_one_compact_block_per_window(monkeypatch):
    from semantic_trainer import objective
    model, rows = adapted_tiny(), fit_rows()
    binding = bind(model)
    inputs = prepared(rows, binding.vocabulary)
    forwards, transfers, original, cpu = [], [], objective.presented, torch.Tensor.cpu
    monkeypatch.setattr(objective, 'presented', lambda b, entries, *args, **kwargs: forwards.append(len(entries)) or original(b, entries, *args, **kwargs))
    monkeypatch.setattr(torch.Tensor, 'cpu', lambda self, *args, **kwargs: transfers.append(tuple(self.shape)) or cpu(self, *args, **kwargs))
    batched = objective.evaluate(binding, rows, inputs, 0, TOKENS, lambda: None)
    assert forwards == [5] and transfers == [(5, 5)]
    forwards.clear()
    transfers.clear()
    charged, polls = [], []
    windowed = objective.evaluate(binding, rows, inputs, 0, 9, lambda: polls.append(1), charged.append)
    assert len(polls) == 3 and len(transfers) == 5 and charged == [8, 2, 10] and sum(forwards) == 5
    for key in batched:
        assert windowed[key]['nll'] == pytest.approx(batched[key]['nll'], rel=1e-5) and windowed[key]['predicted'] == batched[key]['predicted']


@pytest.mark.parametrize('token_budget', [6, 12, None])
@pytest.mark.parametrize('microbatch', [1, 2, 5])
@pytest.mark.parametrize('reverse', [False, True])
def test_weighted_gradients_are_invariant_to_chunking_and_within_batch_order(microbatch, reverse, token_budget):
    from semantic_trainer.objective import accumulate
    rows = fit_rows()
    model = adapted_tiny()
    trainable = [p for p in model.parameters() if p.requires_grad]
    expected = 0
    for row in rows:
        for entry, weight in zip(row['presentations'], presentation_weights(row)):
            loss, _, _ = group_loss(answer_logits(bind(model), sequences([entry]), 0), [entry])
            expected = expected + loss * weight / len(rows)
    expected.backward()
    reference = [p.grad.clone() for p in trainable]
    model.zero_grad(set_to_none=True)
    stats = accumulate(bind(model), rows[::-1] if reverse else rows, prepared(rows, vocabulary_of(model)), 0, microbatch, tokens=token_budget)
    assert float(stats['loss']) == pytest.approx(float(expected.detach()), rel=1e-5)
    assert stats['presentations'] == 5 and stats['tokens'] == sum(e.length for r in rows for e in r['presentations'])
    assert all(torch.allclose(p.grad, r, atol=1e-6, rtol=1e-4) for p, r in zip(trainable, reference))
    assert all(p.grad is None for name, p in model.named_parameters() if 'lora_' not in name)


def test_missing_model_select_fails_before_any_allocation(tmp_path, monkeypatch):
    from semantic_switch.errors import ConfigError
    with pytest.raises(ConfigError, match='no model_select observations'):
        monitored_job(tmp_path, monkeypatch, roles=('fit',))
    assert not (tmp_path / 'job' / 'job.yaml').exists()


@pytest.mark.parametrize('count,expected', [(100, 15), (160, 24), (1100, 165)])
def test_default_warmup_uses_exact_integer_rounding(count, expected):
    from semantic_trainer.jobs import recipe_of, schedule_of
    recipe = recipe_of({'eval_every_examples': 8}, count)
    assert recipe['warmup_examples'] == expected == -(-3 * count // 20)
    schedule = schedule_of(recipe, {'eval_every_examples': 8}, count)
    assert schedule['warmup_steps'] == -(-expected // 8) and schedule['warmup'] == {'policy': 'reference_epochs_fraction', 'explicit': False, 'reference_epochs': 3,
                                                                                    'fraction': '1/20', 'examples': expected, 'steps': schedule['warmup_steps']}
    if count == 160:
        assert schedule['warmup_steps'] == 3


@pytest.mark.parametrize('epochs,steps,points', [(1, 3, 3), (2, 6, 5), (8, 24, 17)])
def test_horizons_count_partial_batches_and_evaluation_points_once(epochs, steps, points):
    from semantic_trainer.jobs import evaluation_points, recipe_of, schedule_of
    for microbatch in (1, 3):
        given = {'eval_every_examples': 5, 'epochs': epochs, 'min_epochs': 1, 'batch_size': 4, 'microbatch': microbatch}
        recipe = recipe_of(given, 10)
        assert schedule_of(recipe, given, 10)['max_optimizer_steps'] == steps and evaluation_points(10, {**recipe, 'eval_every_examples': 5}) == points


def test_constant_warmup_is_cap_invariant_and_explicit_warmup_is_the_only_override():
    from semantic_switch.errors import ConfigError
    from semantic_trainer.jobs import recipe_of, schedule_of
    from semantic_trainer.objective import learning_rate_factor
    plans = []
    for epochs in (2, 8):
        given = {'eval_every_examples': 8, 'epochs': epochs}
        plans.append(schedule_of(recipe_of(given, 160), given, 160))
    assert plans[0]['warmup'] == plans[1]['warmup'] and plans[0]['decay_horizon_steps'] is None
    shared = plans[0]['max_optimizer_steps']
    assert [learning_rate_factor(step, plans[0]['max_optimizer_steps'], plans[0]['warmup_steps'], 'constant') for step in range(shared)] == \
           [learning_rate_factor(step, plans[1]['max_optimizer_steps'], plans[1]['warmup_steps'], 'constant') for step in range(shared)]
    given = {'eval_every_examples': 8, 'warmup_examples': 0}
    zero = schedule_of(recipe_of(given, 160), given, 160)
    assert zero['warmup_steps'] == 0 and zero['warmup']['policy'] == 'explicit' and learning_rate_factor(0, 10, 0, 'constant') == 1
    given = {'eval_every_examples': 8, 'warmup_examples': 9, 'epochs': 1, 'min_epochs': 1, 'batch_size': 4}
    with pytest.raises(ConfigError, match='exceeds the 2 available steps'):
        schedule_of(recipe_of(given, 8), given, 8)
    from semantic_trainer.jobs import cadence_of
    with pytest.raises(ConfigError, match='eval_every_examples=N'):
        cadence_of({})
    assert cadence_of({'eval_every_examples': 8}) == 8 and 'eval_every_examples' not in recipe_of({'eval_every_examples': 8}, 160)
    cosine = {'eval_every_examples': 8, 'schedule': 'cosine'}
    assert schedule_of(recipe_of(cosine, 160), cosine, 160)['decay_horizon_steps'] == 160


def test_ranking_order_matches_the_published_rule():
    from semantic_switch.errors import ConfigError
    from semantic_trainer.jobs import ranking, ranking_rule
    monitored = [{'name': 'a', 'step': 1, 'correct': {'macro': {'observation': .3}}, 'nll': .5}, {'name': 'b', 'step': 2, 'correct': {'macro': {'observation': .3}}, 'nll': .4}]
    assert ranking_rule(monitored, 'observation')['order'] == ['correct', 'regressions', 'nll', 'base', 'step'] and ranking(monitored, 'observation')[0]['name'] == 'b'
    native = [{**point, 'nll': None} for point in monitored]
    rule = ranking_rule(native, 'observation')
    assert rule['order'] == ['correct', 'regressions', 'base', 'step'] and rule['nll'] == 'unavailable' and ranking(native, 'observation')[0]['name'] == 'a'
    with pytest.raises(ConfigError, match='mixes'):
        ranking([monitored[0], native[1]], 'observation')


def test_interrupted_evaluation_charges_completed_work_and_publishes_no_point(tmp_path):
    from semantic_trainer.monitor import Exhausted, Monitor
    rows = [{**row, 'role': role, 'situations': [row['id']], 'component': row['id']} for row, role in zip(fit_rows(), ('fit', 'model_select', 'model_select'))]
    tokens = sum(entry.length for row in rows[1:] for entry in row['presentations']) + sum(entry.length for entry in rows[0]['presentations'])
    resolved = {'fit': {'id': 'fixture'}, 'observations': rows, 'probe_ids': ['a'], 'recipe': {'selection_unit': 'observation', 'microbatch': 1, 'evaluation_batch_size': 1, 'max_padded_tokens': 6}, 'work': {'point_tokens': tokens},
                'budget': {'max_monitor_tokens': 10 * tokens, 'max_sequence_tokens': 6}, 'schedule': {}, 'situations': {}}
    class Deadline:
        calls = 0
        def check(self, operation):
            Deadline.calls += 1
            if Deadline.calls == 3:
                raise Exhausted('time_budget')
        def elapsed(self):
            return 1.
        def expired(self):
            return False
    monitor = Monitor(tmp_path, resolved, adapted_tiny(), 0, None, None, Deadline())
    with pytest.raises(Exhausted):
        monitor.point(4, 8)
    (record,) = records(tmp_path / 'progress.jsonl')
    charged = sum(entry.length for row in rows[1:] for entry in row['presentations'])
    assert record['kind'] == 'interrupted_evaluation' and record['reserved_tokens'] == tokens and record['charged_tokens'] == charged == monitor.work.monitor_tokens
    assert not record['published'] and monitor.points == [] and monitor.work.points == 0
    resolved['budget']['max_monitor_tokens'] = charged + tokens - 1
    with pytest.raises(Exhausted, match='monitor_budget'):
        monitor.point(5, 16)


def test_measured_infeasible_plan_stops_with_a_published_cost_breakdown(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import lora, monitor
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 2}, budget={'max_seconds': 900, 'feasibility_updates': 2})
    fake_native(monkeypatch)
    now = [0.]
    original_update, original_evaluate = lora.update, monitor.evaluate
    def update(*args, **kwargs):
        now[0] += 100
        return original_update(*args, **kwargs)
    def evaluate(*args, **kwargs):
        now[0] += 25
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(lora, 'update', update)
    monkeypatch.setattr(monitor, 'evaluate', evaluate)
    result = lora.train(job, tmp_path / 'out', clock=lambda: now[0])
    trajectory = records(tmp_path / 'out' / 'progress.jsonl')
    feasibility = next(row for row in trajectory if row['kind'] == 'feasibility')
    assert not feasibility['feasible'] and feasibility['measured']['updates_measured'] == 2
    assert result['stop']['steps_trained'] > 2
    assert files.read_json(tmp_path / 'out' / 'run.json')['state'] == 'ready'
    assert (tmp_path / 'out' / 'candidates').exists()


LEAF = {'slot_packing', 'collate', 'transfer', 'host_embedding', 'forward', 'reduction', 'backward', 'clip_optimizer'}


def test_profile_measures_full_updates_and_complete_monitor_points_per_phase(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import jobs, lora, objective
    job, loaded, _ = monitored_job(tmp_path, monkeypatch)
    calls = []
    original = lora.update
    monkeypatch.setattr(lora, 'update', lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    result = lora.profile(job, tmp_path / 'profile.json', steps=2, warmup=1, cadences=[8, 24])
    resolved, _ = jobs.resolve(job)
    assert files.read_json(tmp_path / 'profile.json') == result and jobs.profile_record(tmp_path / 'profile.json')['id'] == result['id']
    assert len(calls) == 3 and len(result['updates']) == 2 and result['training_mode_restored'] and result['loss']['version'] == 2
    assert result['instrumentation'] == 'complete_operation_boundaries'
    assert all(update['observations'] == 8 and update['tokens'] > 0 and update['seconds'] >= 0 for update in result['updates'])
    assert [point['state'] for point in result['points']] == ['measured']
    assert result['points'][0]['observations'] == 24
    assert result['work']['updates'] == 3
    assert result['configuration'] == jobs.profile_configuration(resolved) and result['workload'] == jobs.profile_workload(resolved)
    assert result['allocations']['prepared_inputs']['bytes'] == objective.input_bytes(row for row in resolved_rows(job) if row['role'] in ('fit', 'model_select'))
    assert result['costs']['checkpoint_seconds'] == result['points'][0]['checkpoint_seconds']
    assert result['costs']['point_seconds_per_token'] == pytest.approx(result['points'][0]['seconds'] / result['points'][0]['tokens'])
    assert lora.profile(job, tmp_path / 'profile.json') == result
    assert not (tmp_path / 'job' / 'run').exists()


def resolved_rows(job):
    from semantic_trainer import jobs
    return jobs.resolve(job)[1]


def reprepare(job, to, **settings):
    from semantic_switch import files
    from semantic_trainer import jobs
    declared = files.read_yaml(job)
    (task,) = declared['tasks'].values()
    return jobs.prepare(task['source'], [entry['path'] for entry in declared['exports']], to, assets_manifest=declared['assets']['manifest'],
                        device=declared['fitting']['device'], recipe={'batch_size': 8, 'eval_every_examples': 8, 'fit_probe_observations': 12, **settings.pop('recipe', {})}, **settings)


def costs(**given):
    return {'loading_seconds': 10., 'update_seconds_per_token': .001, 'point_seconds_per_token': .002, 'checkpoint_seconds': 1., **given}


def test_projection_counts_planned_points_tokens_and_checkpoints_against_budgets(tmp_path, monkeypatch):
    from semantic_trainer import jobs
    job, _, _ = monitored_job(tmp_path, monkeypatch)
    resolved, _ = jobs.resolve(job)
    work = resolved['work']
    projected = jobs.projection(resolved, costs(), 24)
    points = jobs.evaluation_points(24, {**resolved['recipe'], 'eval_every_examples': 24})
    assert projected['evaluation_points'] == points and projected['monitor_tokens'] == points * work['point_tokens']
    assert projected['seconds'] == pytest.approx(10 + work['training_tokens'] * .001 + points * work['point_tokens'] * .002 + points * 1.)
    assert projected['feasible'] is (projected['seconds'] <= resolved['budget']['max_seconds'])
    assert not jobs.projection(resolved, costs(update_seconds_per_token=1e6), 24)['feasible']


def test_prepare_records_a_profile_backed_cadence_and_rejects_mismatched_profiles(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_switch.artifacts import artifact_digest
    from semantic_switch.errors import ConfigError
    from semantic_trainer import jobs, lora
    job, _, _ = monitored_job(tmp_path, monkeypatch)
    record = lora.profile(job, tmp_path / 'profile.json', steps=1, warmup=0)
    prepared_ = reprepare(job, tmp_path / 'final', recipe={'eval_every_examples': 16}, profile=tmp_path / 'profile.json')
    resolved, _ = jobs.resolve(prepared_['job'])
    assert resolved['cadence']['profile'] == record['id'] and resolved['cadence']['eval_every_examples'] == 16
    assert resolved['cadence']['projection'] == jobs.projection(resolved, record['costs'], 16) == prepared_['cadence']['projection']
    assert files.read_json(tmp_path / 'final' / 'profile.json') == record
    unprofiled, _ = jobs.resolve(reprepare(job, tmp_path / 'plain', recipe={'eval_every_examples': 16})['job'])
    assert 'cadence' not in unprofiled and unprofiled['monitor']['id'] == resolved['monitor']['id'] and unprofiled['id'] != resolved['id']
    def variant(name, **changes):
        changed = {**record, **changes}
        changed['id'] = artifact_digest({key: value for key, value in changed.items() if key != 'id'})
        files.write_new(tmp_path / name, changed)
        return tmp_path / name
    with pytest.raises(ConfigError, match='another fitting configuration .*batching'):
        reprepare(job, tmp_path / 'batch', profile=variant('batch.json', configuration={**record['configuration'], 'batching': {'batch_size': 4, 'microbatch': 1}}))
    with pytest.raises(ConfigError, match='token workload changed'):
        reprepare(job, tmp_path / 'workload', profile=variant('workload.json', workload={**record['workload'], 'point_tokens': record['workload']['point_tokens'] * 3}))
    slow = reprepare(job, tmp_path / 'slow', profile=variant('slow.json', costs=costs(update_seconds_per_token=1e6)))
    assert not jobs.resolve(slow['job'])[0]['cadence']['projection']['feasible']
    tampered = {**record, 'costs': costs()}
    files.write_new(tmp_path / 'tampered.json', tampered)
    with pytest.raises(ConfigError, match='not an intact fitting profile'):
        reprepare(job, tmp_path / 'tampered', profile=tmp_path / 'tampered.json')
    assert not any((tmp_path / name / 'job.yaml').exists() for name in ('batch', 'workload', 'tampered'))
    assert not any((tmp_path / name / 'profile.json').exists() for name in ('batch', 'workload', 'tampered'))
    (tmp_path / 'final' / 'profile.json').unlink()
    files.write_new(tmp_path / 'final' / 'profile.json', files.read_json(variant('other.json', job_id='other')))
    with pytest.raises(ConfigError, match='profile changed since preparation'):
        jobs.resolve(prepared_['job'])


def staged_rows():
    def entry(index, tokens, label, stage, groups):
        return presentation(f'q{index}', stage, 'canonical', [1, 2], tokens, [3], groups, label)
    return [{'id': 'r', 'task': 't', 'objective': 1., 'classes': ['x', 'y', 'z'], 'label': 2,
             'presentations': [entry(0, [7], 2, 'full', [[4], [5, 6], [8, 9, 10]]), entry(1, [11, 12, 13, 14], 1, 'gate', [[4], [5]]),
                               entry(2, [], 0, 'gate', [[5, 6], [4]])]},
            {'id': 's', 'task': 't', 'objective': 1., 'classes': ['x', 'y'], 'label': 0, 'presentations': [entry(3, [15, 16], 0, 'full', [[4, 8], [5]])]}]


def test_prepared_inputs_are_built_once_and_reproduce_the_stage_weighted_reference():
    from semantic_trainer.objective import evaluate, input_bytes
    model = adapted_tiny().eval()
    rows = staged_rows()
    inputs = prepared([*rows, rows[0]], vocabulary_of(model))
    assert set(inputs.rows) == {'r', 's'} and inputs.presentations == 4 and inputs.bytes == input_bytes(rows) == 8 * (4 + 7 + 3 + 5)
    segments = [entry.segments for row in inputs.rows.values() for entry in row.presentations]
    expected = {}
    for row in rows:
        total = 0.
        for entry, weight in zip(row['presentations'], presentation_weights(row)):
            with torch.no_grad():
                nll = reference(answer_logits(bind(model), sequences([entry]), 0), [entry])[0][1]
            total += float(nll) * weight
        expected[row['id']] = total
    assert presentation_weights(rows[0]) == [.5, .25, .25]
    for _ in range(2):
        results = evaluate(bind(model), rows, inputs, 0, TOKENS, lambda: 1)
        assert {key: value['nll'] for key, value in results.items()} == pytest.approx(expected, rel=1e-5)
        assert results['r']['stages'] == {'full': results['r']['predicted'], 'gate': results['r']['stages']['gate']}
    assert all(a[1] is b.input for a, b in zip(segments, [entry for row in rows for entry in row['presentations']]))
    assert [sum(entry.segments, ()) for entry in inputs.rows['r'].presentations][1] == (1, 2, 11, 12, 13, 14, 3)


@pytest.mark.parametrize('microbatch', [1, 3])
def test_reused_inputs_reproduce_reference_gradients_across_repeated_updates(microbatch):
    from semantic_trainer.objective import accumulate
    rows = staged_rows()
    model = adapted_tiny()
    inputs = prepared(rows, vocabulary_of(model))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        expected = 0
        for row in rows:
            for entry, weight in zip(row['presentations'], presentation_weights(row)):
                z = answer_logits(bind(model), sequences([entry]), 0)[0].double()
                gold = list(entry.answers.slots[entry.label])
                expected = expected + (torch.logsumexp(z, 0) - torch.logsumexp(z[gold], 0)) * weight / len(rows)
        expected.backward()
        wanted = [p.grad.clone() for p in trainable]
        model.zero_grad(set_to_none=True)
        stats = accumulate(bind(model), rows, inputs, 0, microbatch)
        assert float(stats['loss']) == pytest.approx(float(expected.detach()), rel=1e-5) and stats['presentations'] == 4 and stats['tokens'] == 4 + 7 + 3 + 5
        assert all(torch.allclose(p.grad, r, atol=1e-6, rtol=1e-4) for p, r in zip(trainable, wanted))
        with torch.no_grad():
            for p in trainable:
                p.add_(p.grad, alpha=-.1)


def test_fitting_envelope_admits_the_prepared_inputs(tmp_path, monkeypatch):
    from semantic_trainer import jobs, lora, objective
    job, _, _ = monitored_job(tmp_path, monkeypatch)
    resolved, observations = jobs.resolve(job)
    envelope = lora.fitting_envelope(resolved, observations)
    assert envelope['fit:prepared_inputs'] == objective.input_bytes(row for row in observations if row['role'] in ('fit', 'model_select')) > 0
    assert set(envelope) - {'fit:prepared_inputs'} == set(lora.fit_bytes(resolved))


def plan(steps, points, budget):
    return {'schedule': {'max_optimizer_steps': steps}, 'budget': {'max_seconds': budget, 'max_tokens': 10, 'max_monitor_tokens': 10},
            'work': {'evaluation_points': points, 'training_tokens': 1, 'monitor_tokens': 1, 'point_tokens': 1}}


@pytest.mark.parametrize('steps,points,updates,completed,checkpoint,elapsed,budget,seconds,feasible', [
    (8, 9, 8, 8, 0., 88., 120, 98., True),
    (8, 9, 8, 8, 0., 88., 97, 98., False),
    (8, 3, 8, 2, 0., 28., 43, 38., True),
    (8, 4, 8, 3, 0., 38., 48, 48., True),
    (24, 3, 8, 1, 2., 18., 100, 18. + 16 * 1. + 2 * 10. + 2 * 2., True),
    (24, 3, 8, 1, 2., 18., 57, 58., False),
])
def test_feasibility_projects_only_remaining_work_from_completed_counters(steps, points, updates, completed, checkpoint, elapsed, budget, seconds, feasible):
    from semantic_trainer.monitor import Work, feasibility
    work = Work(updates=updates, points=completed, checkpoints=1, seconds={'update': [1.] * updates, 'point': [10.] * completed, 'checkpoint': [checkpoint]})
    record = feasibility(plan(steps, points, budget), work, elapsed)
    assert record['projection']['seconds'] == seconds and record['feasible'] is feasible
    assert record['remaining'] == {'updates': steps - updates, 'points': points - completed, 'checkpoints': points - completed}
    assert record['projection']['monitoring_seconds'] == (points - completed) * 10. and record['completed']['points'] == completed


def test_feasibility_after_update_eight_counts_completed_points_once(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import lora, monitor
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'batch_size': 3, 'epochs': 1, 'min_epochs': 1, 'eval_every_examples': 12},
                              budget={'max_seconds': 43, 'feasibility_updates': 8})
    fake_native(monkeypatch)
    now = [0.]
    original_update, original_evaluate = lora.update, monitor.evaluate
    monkeypatch.setattr(lora, 'update', lambda *args, **kwargs: now.__setitem__(0, now[0] + 1) or original_update(*args, **kwargs))
    monkeypatch.setattr(monitor, 'evaluate', lambda *args, **kwargs: now.__setitem__(0, now[0] + 5) or original_evaluate(*args, **kwargs))
    result = lora.train(job, tmp_path / 'out', clock=lambda: now[0])
    (feasible,) = [record for record in records(tmp_path / 'out' / 'progress.jsonl') if record['kind'] == 'feasibility']
    assert feasible['elapsed_seconds'] == 28 and feasible['completed']['points'] == 2 and feasible['remaining'] == {'updates': 0, 'points': 1, 'checkpoints': 1}
    assert feasible['projection']['seconds'] == 38 and feasible['feasible'] and result['stop']['reason'] == 'epoch_cap'
    assert files.read_json(tmp_path / 'out' / 'run.json')['state'] == 'ready'


def deadline_run(tmp_path, monkeypatch, trigger):
    from semantic_trainer import lora, monitor, objective
    job, loaded, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 2, 'eval_every_examples': 8, 'min_epochs': 1, 'patience_examples': 1000},
                                   budget={'max_seconds': 1000})
    fake_native(monkeypatch)
    now, starts, rows = [0.], [], [0]
    original, read = objective.presented, objective.readout
    def presented(*args, **kwargs):
        starts.append(now[0])
        return original(*args, **kwargs)
    def readout(binding, window, *args, **kwargs):
        out = read(binding, window, *args, **kwargs)
        rows[0] += len(window)
        if trigger.get('row') == rows[0]:
            now[0] = 1001.
        return out
    monkeypatch.setattr(objective, 'presented', presented)
    monkeypatch.setattr(objective, 'readout', readout)
    monkeypatch.setattr(objective, 'windows', lambda rows, inputs, tokens: ([row] for row in rows))
    for name, owner in (('update', lora), ('save', monitor), ('native_answers', lora), ('retain', lora)):
        if name in trigger:
            calls, wrapped = [0], getattr(owner, 'checkpoint' if name == 'save' else name)
            def counted(*args, _wrapped=wrapped, _calls=calls, _at=trigger[name], **kwargs):
                out = _wrapped(*args, **kwargs)
                _calls[0] += 1
                if _calls[0] == _at:
                    now[0] = 1001.
                return out
            if name == 'save':
                monkeypatch.setattr(monitor, 'checkpoint', counted)
                monkeypatch.setattr(lora, 'checkpoint', counted)
            else:
                monkeypatch.setattr(owner, name, counted)
    return job, loaded, now, starts, (lambda: lora.train(job, tmp_path / 'out', clock=lambda: now[0]))


def finished(tmp_path, starts):
    from semantic_switch import files
    output = tmp_path / 'out'
    listed, run, metrics = (files.read_json(output / name) for name in ('candidates.json', 'run.json', 'metrics.json'))
    assert listed['state'] == run['state'] == 'ready' and listed['stop'] == run['stop'] == metrics['stop'] and listed['stop']['reason'] == 'time_budget'
    assert listed['stop']['convergence'] == 'not_demonstrated' and listed['stop']['deadline']['limit_seconds'] == 1000
    assert listed['stop']['deadline']['elapsed_seconds'] == 1001 and listed['stop']['deadline']['overrun_seconds'] == 1
    assert all(start < 1000 for start in starts) and listed['candidates'][0]['name'] == 'zero' and len(listed['candidates']) > 1
    for candidate in listed['candidates']:
        assert files.sha256_file(output / candidate['peft'] / 'adapter_model.safetensors') == candidate['peft_sha256']
    return listed['stop'], [record for record in records(output / 'progress.jsonl') if record['kind'] in ('evaluation', 'interrupted_evaluation')]


def test_a_complete_point_finishing_after_the_deadline_stops_on_time_budget_with_a_valid_candidate(tmp_path, monkeypatch):
    job, _, _, starts, run = deadline_run(tmp_path, monkeypatch, {'row': 7 * 24})
    run()
    stop, points = finished(tmp_path, starts)
    assert stop['deadline']['crossed'] == 'monitor_point' and stop['deadline']['last_complete_point'] == 'step-6'
    assert stop['deadline']['completed_after_deadline']['point'] == 'step-6' and points[-1]['completed_after_deadline'] and points[-1]['step'] == 6
    assert stop['unevaluated'] == {'steps': 0, 'examples': 0} and stop['work']['points'] == 7


@pytest.mark.parametrize('trigger,crossed,last,unevaluated', [({'update': 6}, 'optimizer_update', 5, 1), ({'retain': 6}, 'checkpoint', 6, 0),
                                                              ({'save': 3}, 'checkpoint', 2, 0)])
def test_expiry_in_updates_or_checkpoints_keeps_retained_candidates_and_starts_no_model_call(tmp_path, monkeypatch, trigger, crossed, last, unevaluated):
    job, _, _, starts, run = deadline_run(tmp_path, monkeypatch, trigger)
    run()
    stop, points = finished(tmp_path, starts)
    assert stop['deadline']['crossed'] == crossed and stop['last_evaluation']['step'] == last and stop['unevaluated']['steps'] == unevaluated
    assert stop['reason'] != 'epoch_cap' and all(point['kind'] == 'evaluation' and not point['completed_after_deadline'] for point in points)


def test_a_partial_point_after_expiry_is_never_ranked(tmp_path, monkeypatch):
    from semantic_switch import files
    job, _, _, starts, run = deadline_run(tmp_path, monkeypatch, {'row': 3 * 24 + 13})
    run()
    stop, points = finished(tmp_path, starts)
    assert points[-1]['kind'] == 'interrupted_evaluation' and points[-1]['step'] == 3 and points[-1]['charged_tokens'] > 0
    assert stop['deadline']['crossed'] == 'evaluation_row' and stop['last_evaluation']['step'] == 2 and stop['unevaluated']['steps'] == 1
    assert {candidate['step'] for candidate in files.read_json(tmp_path / 'out' / 'candidates.json')['candidates']} <= {0, 1, 2}


def test_native_baseline_expiry_prevents_fitting(tmp_path, monkeypatch):
    from semantic_switch import files
    from semantic_trainer import monitor
    job, loaded, _, starts, run = deadline_run(tmp_path, monkeypatch, {'native_answers': 1})
    with pytest.raises(monitor.Exhausted, match='time_budget'):
        run()
    assert loaded == [] and starts == [] and files.read_json(tmp_path / 'out' / 'run.json')['state'] == 'failed'
    assert not (tmp_path / 'out' / 'candidates').exists()


def test_padded_token_planner_refuses_oversized_singletons():
    from semantic_switch import ConfigError
    from semantic_trainer.objective import planned
    assert [list(chunk) for chunk in planned([4, 9, 10, 12], presentations=3, tokens=20)] == [[0, 1], [2], [3]]
    with pytest.raises(ConfigError, match='max_padded_tokens 20'):
        planned([21], presentations=1, tokens=20)
