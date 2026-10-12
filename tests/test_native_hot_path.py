from __future__ import annotations

import numpy as np
import pytest

from semantic_switch import Model, Pick, Runtime, backend
from semantic_switch.context import Bm25

from .conftest import FakeScorer
from .test_native_resources import fake_llama, fresh, request, scorer, untracked_sequences_empty


def reference(row, slots):
    row = row.astype(np.float64)
    if np.isnan(row).any() or np.isposinf(row).any():
        raise RuntimeError("non-finite")

    def lse(x):
        m = float(np.max(x))
        if not np.isfinite(m):
            raise RuntimeError("non-finite")
        return m + float(np.log(np.exp(x - m).sum()))

    slot_lp = np.array([lse(row[list(ids)]) for ids in slots])
    total = lse(slot_lp)
    invalid = np.ones(len(row), dtype=bool)
    for ids in slots:
        invalid[list(ids)] = False
    complement = row[invalid]
    mass = -float(np.logaddexp(0., lse(complement) - total)) if np.isfinite(complement).any() else 0.
    return slot_lp - total, mass


@pytest.mark.parametrize("seed", range(6))
def test_reduction_matches_stable_reference_and_reuses_scratch(seed):
    rng = np.random.default_rng(seed)
    row = rng.normal(scale=[1, 30, 300][seed % 3], size=4096).astype(np.float32)
    row[rng.choice(4096, 50, replace=False)] = -np.inf
    slots = ((3, 12), (9,), (100, 101, 102))
    row[[3, 9, 100]] += [0, 80, -40][seed % 3]
    scratch = np.full(8192, np.nan)
    for buffer in (None, scratch, scratch):
        logp, mass = backend.reduce_row(row, slots, buffer)
        want, want_mass = reference(row, slots)
        assert logp.dtype == np.float32 and np.allclose(logp, want, atol=1e-6) and mass == pytest.approx(want_mass, abs=1e-9)
        assert mass <= 0


def test_reduction_rejects_nonfinite_and_overlapping_slots():
    row = np.zeros(16, dtype=np.float32)
    for bad in (np.nan, np.inf):
        broken = row.copy()
        broken[5] = bad
        with pytest.raises(RuntimeError, match="non-finite"):
            backend.reduce_row(broken, ((0,), (1,)))
    with pytest.raises(RuntimeError):
        backend.reduce_row(np.full(16, -np.inf, dtype=np.float32), ((0,), (1,)))
    with pytest.raises(ValueError, match="disjoint"):
        backend.reduce_row(row, ((0, 1), (1,)))
    saturated = np.full(262144, -30., dtype=np.float32)
    saturated[[3, 9]] = 40., 32.
    _, mass = backend.reduce_row(saturated, ((3,), (9,)))
    assert mass == pytest.approx(reference(saturated, ((3,), (9,)))[1], abs=1e-12)


def test_capacity_is_one_calculation_for_model_scorer_identity_and_estimate(fake_llama, monkeypatch):
    library = fake_llama()
    model = Model("fake.gguf", n_ctx=256, n_batch=64, prefix_slots=3)
    layout = model.capacity()
    with scorer(prefix_slots=3) as native:
        assert native.layout == layout and (layout.states, layout.branches) == (1, 1)
        assert (library.params.n_seq_max, library.params.n_outputs_max, layout.outputs_per_decode) == (layout.n_seq_max, layout.n_outputs_max, 1) == (5, 5, 1)
        assert {k: native.capacity()[k] for k in ("n_ctx", "n_batch", "n_seq_max", "n_outputs_max")} == \
            {"n_ctx": 256, "n_batch": 64, "n_seq_max": 5, "n_outputs_max": 5}
    batched = backend.capacity(None, 512, 8, 4, "batched")
    assert (batched.n_seq_max, batched.n_outputs_max, batched.outputs_per_decode, batched.n_ctx) == \
        (4 + 8 + backend.BRANCHES, 4 + 8 + backend.BRANCHES, backend.BRANCHES, 8192 + 7 * 1024)
    meta = {"general.architecture": "x", "x.block_count": 2, "x.attention.head_count_kv": 1, "x.embedding_length": 8,
            "x.attention.head_count": 1, "tokenizer.ggml.tokens": ["t"] * 100}
    monkeypatch.setattr(backend, "gguf_metadata", lambda path: meta)
    monkeypatch.setattr(Model, "path", lambda self: __file__)
    assert model._host_bytes()["output"] == 100 * layout.n_outputs_max * 4 * 2
    with pytest.raises(Exception, match="sequence slots"):
        Model("fake.gguf", prefix_slots=255)


def test_serial_layout_cache_rollback_matches_fresh(fake_llama):
    fake_llama()
    with scorer(prefix_slots=2) as s:
        prefixes = [(7, 8), (9, 9, 9), (4,), (7, 8)]
        for prefix in prefixes * 2:
            r = request(prefix, (3, 4), (5, 6), (6, 5))
            block = s.score([r])
            assert np.allclose(block.logp, fresh(r).logp) and untracked_sequences_empty(s)
        assert len(s._resident) <= 1 and s.stats["prefix_evictions"] > 0


def test_cached_named_pick_matches_dynamic_definitions():
    options = {f"unit-{i}": {"description": f"unit number {i} handles {'billing' if i % 3 else 'shipping'} topic {i}",
                              "aliases": [f"u{i}"]} for i in range(30)}
    objects = list(options)
    named = Pick("Which unit handles billing?", options, shortlist="lexical", eligible=lambda key: key != "unit-4")
    dynamic = Pick("Which unit handles billing?", objects, shortlist="lexical", eligible=lambda key: key != "unit-4",
                   definition=lambda key: options[key]["description"], ids=lambda key: (key, options[key]["aliases"]))
    results = [Runtime({"judge": FakeScorer()}).ask("u7 and billing for unit-12", {"x": item})["x"] for item in (named, dynamic)]
    assert results[0].removed == {key: [objects[i] for i in value] for key, value in results[1].removed.items()}
    assert results[0].removed["ineligible"] == ["unit-4"] and len(results[0].removed["not_examined"]) == 9
    assert [a["option"] for a in results[0].presentations[0]["answers"]] == [a["option"] for a in results[1].presentations[0]["answers"]]
    docs = [options[key]["description"] for key in objects]
    keep = [i for i in range(30) if i != 4]
    query = "Which unit handles billing? u7"
    assert Bm25(docs).subset(keep).rank(query, range(len(keep)), 10) == Bm25([docs[i] for i in keep]).rank(query, range(len(keep)), 10)
