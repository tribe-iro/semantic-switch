from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

from .scoring import FakeScorer, letters_of

VALUES = ('escalate', 'continue', 'clarify')
ACTION_TEXT = ('Request a technician', 'Continue operation', 'Ask whether')
REALIZED = {'device': 'cpu', 'tensor_types': ['q4_0'], 'flash_attn': 'enabled', 'type_k': 'f16', 'type_v': 'f16', 'use_extra_bufts': True, 'repacked': [],
            'n_ubatch': 64, 'n_ctx': 12288, 'n_ctx_seq': 2048, 'sequence_limit': 2048, 'n_batch': 512, 'n_seq_max': 6, 'n_outputs_max': 1, 'swa_full': True, 'kv_unified': False, 'prefix_slots': 4,
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


def tiny_qwen(vocab):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()


def fake_base(manifest, vocab):
    from semantic_trainer import assets
    located = assets.load(manifest)
    tiny_qwen(vocab).save_pretrained(located['hf'])
    return assets.record(Path(located['root']), located['repo'], located['commit'], ['Q4_0'])


def fake_training(patch, vocab):
    import weakref
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from semantic_switch import api, model_jobs
    from semantic_trainer import jobs, lora
    loaded, calls = [], []
    class Loader:
        @staticmethod
        def from_pretrained(*args, device_map, **kwargs):
            calls.append({**kwargs, 'device_map': device_map})
            model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
            loaded.append(weakref.ref(model))
            return model
    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(pad_token_id=0))
    patch.setattr(lora, 'dependencies', lambda: (torch, AutoConfig, Loader, tokenizer, LoraConfig, get_peft_model))
    patch.setattr(jobs, 'physical_cores', lambda: 1)
    patch.setattr(lora, 'verify_tokenizer', lambda tokenizer, rows: {**rows[0]['native']['tokenizer'], 'vocab_size': vocab})
    patch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    patch.setattr(api.Model, 'resources', lambda self: {'weights': api.Resources()})
    return loaded, calls


def fake_fitting(monkeypatch, manifest, rows):
    vocab = 1 + max(token for row in rows for entry in row['presentations']
                    for token in (*entry.prefix, *entry.input, *entry.question, *(t for group in entry.answers.slots for t in group)))
    manifest = fake_base(manifest, vocab)
    loaded, calls = fake_training(monkeypatch, vocab)
    return manifest, loaded, calls


def fake_deployment(patch):
    from semantic_switch import backend, files
    from semantic_trainer import representation, select
    def deploy(fitted, located, peft, destination, quant, handle):
        Path(destination).write_bytes(b'merged ' + Path(peft).name.encode())
        return files.sha256_file(destination)
    patch.setattr(representation, 'deploy', deploy)
    patch.setattr(select, 'verify_zero', lambda *args: None)
    patch.setattr(select, 'conversion_bytes', lambda located: {'conversion': 1})
    patch.setattr(backend, 'LlamaScorer', Native)
    Native.built, Native.realized = [], REALIZED
