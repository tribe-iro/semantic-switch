import pytest

from semantic_switch import ConfigError, Model
from semantic_switch.backend import capacity, ceiling
from semantic_switch.protocols import MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS, MAX_QUESTION_TOKENS


def test_default_window_is_the_sum_of_the_request_caps():
    assert ceiling(MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS) == MAX_INPUT_TOKENS + MAX_QUESTION_TOKENS + MAX_PREFIX_TOKENS == 3584
    assert capacity(None, 512, 4).n_ctx == 3584
    model = Model("unused.gguf")
    assert model.n_ctx == 3584 and model.capacity() == capacity(None, 512, 4)


def test_omitted_and_default_equivalent_settings_normalize_alike():
    assert Model("unused.gguf") == Model("unused.gguf", n_ctx=3584)
    assert Model("unused.gguf").capacity().total_ctx == Model("unused.gguf", n_ctx=3584).capacity().total_ctx == 3584 * 6


def test_smaller_explicit_window_is_kept_and_padded_per_stream():
    layout = Model("unused.gguf", n_ctx=1536, prefix_slots=2).capacity()
    assert layout.n_ctx == 1536 and layout.n_ctx_seq == 1536 and layout.total_ctx == 1536 * 4
    odd = capacity(1000, 512, 4)
    assert odd.n_ctx == 1000 and odd.n_ctx_seq == 1024 and odd.total_ctx == 1024 * 6


def test_window_at_the_ceiling_passes_and_one_more_is_refused_before_any_load():
    assert Model("unused.gguf", n_ctx=3584).n_ctx == 3584
    with pytest.raises(ConfigError, match=r"n_ctx 3585 exceeds the useful request window 3584 = max_input_tokens 1024 \+ question suffix cap 1536 \+ max_prefix_tokens 1024"):
        Model("unused.gguf", n_ctx=3585)
    with pytest.raises(ConfigError, match="exceeds the useful request window"):
        Model("does-not-exist/anywhere:missing.gguf", n_ctx=8192)


def test_larger_window_needs_larger_caps():
    model = Model("unused.gguf", n_ctx=8192, max_input_tokens=4096, max_prefix_tokens=2560)
    assert model.n_ctx == 8192 and model.capacity().total_ctx == 8192 * 6
    assert Model("unused.gguf", max_input_tokens=4096, max_prefix_tokens=2560).n_ctx == 8192


@pytest.mark.parametrize("name", ["n_ctx", "max_input_tokens", "max_prefix_tokens", "n_batch", "prefix_slots"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budgets_are_refused(name, value):
    with pytest.raises(ConfigError):
        Model("unused.gguf", **{name: value})


def test_direct_capacity_uses_the_same_validation():
    with pytest.raises(ValueError, match="exceeds the useful request window"):
        capacity(4000, 512, 4)
    with pytest.raises(ValueError, match="max_input_tokens must be a positive integer"):
        capacity(None, 512, 4, max_input_tokens=0)
    assert capacity(None, 512, 4, max_input_tokens=2048).n_ctx == 2048 + MAX_QUESTION_TOKENS + MAX_PREFIX_TOKENS


def test_resource_estimate_follows_the_normalized_layout(monkeypatch):
    seen = []
    import semantic_switch.backend as backend
    real = backend.capacity
    monkeypatch.setattr(backend, "capacity", lambda *a, **k: seen.append((a, k)) or real(*a, **k))
    Model("unused.gguf", max_input_tokens=512).capacity()
    assert seen[-1][1] == {"max_input_tokens": 512, "max_prefix_tokens": 1024} and seen[-1][0][0] == 512 + MAX_QUESTION_TOKENS + 1024


def test_direct_scorer_refuses_before_native_allocation(monkeypatch):
    import semantic_switch.backend as backend
    monkeypatch.setattr(backend, "Weights", lambda *a, **k: pytest.fail("native weights loaded before capacity validation"))
    with pytest.raises(ValueError, match="exceeds the useful request window"):
        backend.LlamaScorer("unused.gguf", n_ctx=4096)
    with pytest.raises(ValueError, match="useful request window 2560"):
        backend.LlamaScorer("unused.gguf", n_ctx=3584, max_input_tokens=512, max_prefix_tokens=512)
