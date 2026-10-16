import sys

import numpy as np
import pytest

from semantic_switch import files
from semantic_switch.backend import LlamaScorer
from semantic_switch.protocols import ScoreRequest
from semantic_switch.training import model_job
from semantic_trainer import assets
from semantic_trainer.adapter import effective
from semantic_trainer.deploy import conversion_bytes, deploy, verify_zero

pytestmark = pytest.mark.slow


def test_nonzero_lora_survives_real_conversion_merge_quantization_and_native_scoring(tmp_path):
    import peft
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    sentencepiece = pytest.importorskip('sentencepiece')
    tools = assets.root() / 'llama.cpp' / assets.TOOLCHAIN
    missing = [name for name in assets.TOOLS if not (tools / name).is_file()]
    if missing:
        pytest.skip(f'real conversion needs llama.cpp {assets.TOOLCHAIN} under {tools}; missing {missing}')
    torch.manual_seed(17)
    hf = tmp_path / 'hf'
    hf.mkdir()
    sentencepiece.SentencePieceTrainer.train(sentence_iterator=iter([
        'The operator reports a loaded ribbon and ready printer.',
        'The printer is paused. Calibration is not complete.',
        'A missing ribbon prevents printing. Check the media configuration.',
    ]), model_prefix=str(hf / 'tokenizer'), vocab_size=64, hard_vocab_limit=False, minloglevel=2)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(hf / 'tokenizer.model'))
    files.publish(hf / 'tokenizer_config.json', {'tokenizer_class': 'LlamaTokenizer', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'})
    config = LlamaConfig(vocab_size=tokenizer.vocab_size(), hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                         bos_token_id=tokenizer.bos_id(), eos_token_id=tokenizer.eos_id())
    base = LlamaForCausalLM(config).eval()
    base.save_pretrained(hf)
    f16, quant = tmp_path / 'base-F16.gguf', tmp_path / 'base-Q8_0.gguf'
    assets.run([sys.executable, tools / 'convert_hf_to_gguf.py', hf, '--outtype', 'f16', '--outfile', f16])
    assets.run([tools / 'build/bin/llama-quantize', f16, quant, 'Q8_0'])
    target = 'model.layers.0.self_attn.v_proj'
    adapted = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', target_modules=[target], r=2, lora_alpha=4)).eval()
    zero = tmp_path / 'zero'
    adapted.save_pretrained(zero)
    fitted = {'targets': {target: [64, 64]}, 'recipe': {'rank': 2, 'alpha': 4, 'dropout': 0.},
              'model': {'revision': '0' * 40}, 'dependencies': {'peft': peft.__version__},
              'adapter': effective(files.read_json(zero / 'adapter_config.json'), peft.__version__)}
    located = {'root': str(tmp_path), 'hf': str(hf), 'toolchain': {'path': str(tools)},
               'gguf': {'F16': {'path': str(f16)}, 'Q8_0': {'path': str(quant), 'sha256': files.sha256_file(quant)}},
               'manifest': 'local-native-round-trip'}
    inputs = [[tokenizer.bos_id(), 5, 8, 12], [tokenizer.bos_id(), 9, 11, 15]]
    def probabilities():
        with torch.no_grad():
            return torch.stack([adapted(torch.tensor([ids])).logits[0, -1].softmax(-1) for ids in inputs]).numpy()
    base_probs = probabilities()
    with torch.no_grad():
        layer = adapted.base_model.model.model.layers[0].self_attn.v_proj
        layer.lora_A['default'].weight.normal_(std=.5)
        layer.lora_B['default'].weight.normal_(std=.5)
    expected = probabilities()
    assert np.max(np.abs(expected - base_probs)) > .005
    candidate = tmp_path / 'candidate'
    adapted.save_pretrained(candidate)
    destination = tmp_path / 'specialist.gguf'
    with model_job(extra=conversion_bytes(located)) as job:
        verify_zero(fitted, located, zero, 'Q8_0', tmp_path, job)
        digest = deploy(fitted, located, candidate, destination, 'Q8_0', job)
    assert digest == files.sha256_file(destination) and digest != located['gguf']['Q8_0']['sha256']
    slots = tuple((token,) for token in range(tokenizer.vocab_size()))
    requests = [ScoreRequest(tuple(ids[:1]), tuple(ids[1:2]), (tuple(ids[2:]),), (slots,)) for ids in inputs]
    observed = []
    for path in (quant, destination):
        native = LlamaScorer(str(path), n_ctx=128, n_batch=128, threads=1, n_gpu_layers=0)
        try:
            block = native.score(requests)
            assert not any(block.error), block.error
            observed.append(np.exp(block.logp))
        finally:
            native.close()
    np.testing.assert_allclose(observed[0], base_probs, rtol=.015, atol=.0003)
    np.testing.assert_allclose(observed[1], expected, rtol=.015, atol=.0003)
    assert np.max(np.abs(observed[1] - observed[0])) > .005
    assert np.array_equal(observed[1].argmax(axis=1), expected.argmax(axis=1))
