import inspect
from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, Model
from semantic_switch.training import native_settings
from semantic_trainer import jobs


def view(prefix=10, input_=20, question=30, stage='full'):
    return SimpleNamespace(prefix=(0,) * prefix, input=(0,) * input_, question=(0,) * question, stage=stage, length=prefix + input_ + question)


def row(*presentations):
    return {'id': 'obs-1', 'task': 'printer', 'path': ('ready_to_print',), 'fact': 'ready_to_print', 'presentations': presentations}


def test_trainer_native_defaults_come_from_model_defaults():
    source = inspect.getsource(jobs.populate)
    assert "max_sequence_tokens'] * 2" not in source and 'n_ctx=max(' not in source
    assert native_settings(Model('unused.gguf', device='auto', threads=4))['n_ctx'] == 3584
    explicit = native_settings(Model('unused.gguf', **{'device': 'auto', 'threads': 4, 'max_input_tokens': 2048}))
    assert explicit['n_ctx'] == 2048 + 1536 + 1024 and explicit['max_input_tokens'] == 2048


@pytest.mark.parametrize('segment, sizes', [('prefix', (1025, 1, 1)), ('input', (1, 1025, 1)), ('question', (1, 1, 1537))])
def test_oversized_exported_segment_names_task_target_and_segment(segment, sizes):
    native = native_settings(Model('unused.gguf'))
    with pytest.raises(ConfigError, match=rf"task printer, target .*observation obs-1, full presentation: {segment} segment is \d+ tokens"):
        jobs.native_fit([row(view(*sizes))], native)


def test_total_above_a_smaller_window_is_refused():
    native = native_settings(Model('unused.gguf', n_ctx=1024))
    with pytest.raises(ConfigError, match='total segment is 1100 tokens, above the native total limit 1024'):
        jobs.native_fit([row(view(100, 500, 500, stage='mentions'))], native)
    jobs.native_fit([row(view(100, 400, 500))], native)
