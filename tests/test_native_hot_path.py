from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from semantic_switch import Model, Pick, Runtime, backend
from semantic_switch.context import Bm25

from semantic_testkit.scoring import FakeScorer
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
        assert native.layout == layout and layout.residents == range(3, 5)
        assert (library.params.n_seq_max, library.params.n_outputs_max, layout.outputs_per_decode) == (layout.n_seq_max, layout.n_outputs_max, 1) == (5, 5, 1)
        assert {k: native.capacity()[k] for k in ("n_ctx", "n_batch", "n_seq_max", "n_outputs_max")} == \
            {"n_ctx": layout.total_ctx, "n_batch": 64, "n_seq_max": 5, "n_outputs_max": 5}
    descriptor = SimpleNamespace(meta={"general.architecture": "x", "x.block_count": 2, "x.attention.head_count_kv": 1, "x.embedding_length": 8,
                                       "x.attention.head_count": 1}, architecture="x", tensors={}, size=0, vocab=100)
    monkeypatch.setattr(Model, "descriptor", lambda self: descriptor)
    monkeypatch.setattr(backend, "offload_device", lambda: None)
    assert model.resources()["output"].host == 100 * layout.n_outputs_max * 4 * 2
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
    results = [Runtime(FakeScorer()).ask("u7 and billing for unit-12", {"x": item})["x"] for item in (named, dynamic)]
    assert results[0].removed == {key: [objects[i] for i in value] for key, value in results[1].removed.items()}
    assert results[0].removed["ineligible"] == ["unit-4"] and len(results[0].removed["not_examined"]) == 9
    assert [a["option"] for a in results[0].presentations[0]["answers"]] == [a["option"] for a in results[1].presentations[0]["answers"]]
    docs = [options[key]["description"] for key in objects]
    keep = [i for i in range(30) if i != 4]
    query = "Which unit handles billing? u7"
    assert Bm25(docs).subset(keep).rank(query, range(len(keep)), 10) == Bm25([docs[i] for i in keep]).rank(query, range(len(keep)), 10)


def test_gguf_descriptor_is_parsed_once_per_unchanged_file_and_serves_identity_without_native_loading(tmp_path, monkeypatch):
    import os
    from semantic_switch import api
    from semantic_switch.adapters.chat import RenderContext, Template
    from .test_resident_adapters import gguf as write_gguf
    template = "{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}<a>{{ strftime_now('%Y-%m-%d') }}"
    meta = {"general.architecture": "x", "x.block_count": 2, "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.ggml.eos_token_id": 1, "tokenizer.chat_template": template, "tokenizer.ggml.tokens": ["<pad>", "<eos>", "<bos>", "a"]}
    path = write_gguf(tmp_path / "chat.gguf", meta, {"token_embd.weight": (8, 4)})
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path / "cache"))
    parses, parse = [], backend._parse_gguf
    monkeypatch.setattr(backend, "_parse_gguf", lambda *args: parses.append(1) or parse(*args))
    monkeypatch.setattr(backend, "_llama", lambda: pytest.fail("identity must not load native weights"))
    model = Model(path, device="cpu")
    first = model.descriptor()
    assert model.descriptor() is first and backend.gguf(path) is first and len(parses) == 1
    assert (first.vocab, first.bos, first.eos, first.size) == (4, "<bos>", "<eos>", os.path.getsize(path))
    assert first.chat() == {"template": template, "bos": "<bos>", "eos": "<eos>", "add_bos": False} and "tokenizer.ggml.tokens" not in str(first.chat())
    assert model.placement() == backend.Placement(2, 0)
    rt = Runtime(model)
    identity = rt.identity()
    assert identity["template"] == Template.of(first.chat()).name and len(parses) == 1
    assert Runtime(model).identity()["template"] == identity["template"]
    dated = Template.of(first.chat())
    assert dated.render(RenderContext(__import__("datetime").datetime(2026, 1, 2)))[1].endswith("2026-01-02")
    with open(path, "ab") as f:
        f.write(b"\0" * 32)
    assert backend.gguf(path) is not first and len(parses) == 2


def test_token_descriptors_cache_hash_and_compare_exactly():
    from semantic_switch.protocols import ScoreRequest, Tokens
    a, b = Tokens(range(10000)), Tokens(tuple(range(10000)))
    assert a == b and tuple(a) == tuple(range(10000)) and hash(a) == hash(b) and Tokens(a) is a
    assert a != Tokens(range(9999)) and {a: 1}[b] == 1 and a.array.dtype == np.int32 and not a.array.flags.writeable
    first, second = ScoreRequest(a, (1,), ((2,),), (((3,), (4,)),)), ScoreRequest(tuple(range(10000)), (1,), ((2,),), (((3,), (4,)),))
    assert first == second and hash(first) == hash(second) and isinstance(second.prefix, Tokens) and isinstance(second.suffixes[0], Tokens)
    assert first != ScoreRequest(a, (1,), ((2,),), (((4,), (3,)),))


def test_bounded_cache_evicts_by_bytes_not_by_clearing():
    from semantic_switch.protocols import Bounded
    cache = Bounded(100)
    for key in range(4):
        cache.put(key, key, 30)
    assert list(cache._entries) == [1, 2, 3] and cache.bytes == 90 and cache.evictions == 1
    assert cache.get(1) == 1 and cache.put(4, 4, 30) == 4 and list(cache._entries) == [3, 1, 4]
    assert cache.put("huge", 0, 101) == 0 and "huge" not in cache


def test_lexical_head_streams_features_and_reduces_vectorized():
    import unicodedata
    import zlib
    from semantic_switch.heads import N_FEATURES, LexicalHead, _WORDS, features

    def reference(text):
        words = _WORDS.findall(unicodedata.normalize("NFC", text).casefold())
        grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        grams += [f"#{w[i:i + n]}" for w in (f"<{w}>" for w in words) for n in (3, 4, 5) for i in range(len(w) - n + 1)]
        counts = {}
        for g in grams:
            h = zlib.crc32(g.encode()) % N_FEATURES
            counts[h] = counts.get(h, 0.0) + 1.0
        norm = sum(v * v for v in counts.values()) ** 0.5 or 1.0
        return {k: v / norm for k, v in counts.items()}

    texts = ["Refund my café order, não!", "x", "", "Straße STRASSE ＡＢＣ re-fund e.g. refund refund", "日本語のテキスト 返金"]
    for text in texts:
        assert list(features(text).items()) == list(reference(text).items())
    rng = np.random.default_rng(0)
    idx = rng.choice(N_FEATURES, 300, replace=False).astype(np.int32)
    idx[:20] = sorted(features(texts[0]))[:20]
    head = LexicalHead(("a", "b", "c"), idx, rng.normal(size=(300, 3)).astype(np.float32), rng.normal(size=3).astype(np.float32))
    rows = {int(j): r for r, j in enumerate(idx)}
    for text in texts:
        feats = features(text)
        hit = [(rows[j], v) for j, v in feats.items() if j in rows]
        z = head.b.astype(np.float64) + (np.asarray([v for _, v in hit]) @ head.w[[r for r, _ in hit]] if hit else 0)
        want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        assert np.array_equal(head.probs(text), want)
    loaded = LexicalHead.from_json(head.to_json())
    assert loaded.w.base is not None and not loaded.w.flags.writeable and np.array_equal(loaded.probs(texts[0]), head.probs(texts[0]))


def test_scored_stamps_only_missing_model_metadata_once():
    from semantic_switch.judge import Scored, Scores, scored
    from semantic_switch.protocols import Question
    stamped, kept = Scores(np.array([.9, .1]), 1.), Scores(np.array([.2, .8]), 1., model="own")
    complete = Scored((Scores(np.array([.5, .5]), 1., model="m"),), 3, "m")

    class Judge:
        identity, supports_guidance = {"source": "fake"}, False

        def score_many(self, items):
            return [Scored((stamped, kept), 4, "m"), complete, Scored((stamped,), 1)]
    question = Question("check", "claim")
    results = scored(Judge(), [("a", [question, question]), ("b", [question]), ("c", [question])])
    assert [s.model for s in results[0].scores] == ["m", "own"] and results[0].scores[1] is kept
    assert results[1] is complete and results[2].scores[0] is stamped


def test_adapter_prepares_framing_once_and_reuses_prepared_prefixes():
    from semantic_switch.adapters.qwen3 import Qwen3Adapter
    from semantic_switch.guidance import Guidance
    from semantic_switch.protocols import Question
    adapter = Qwen3Adapter(FakeScorer())
    assert adapter.framing is adapter.framing and adapter.prefix_for(None) is adapter.prefix
    guided = Question("check", "claim", guidance=Guidance("terms"))
    first = adapter.request(adapter.state("case"), [guided])
    assert adapter.request(adapter.state("case"), [guided]).prefix is first.prefix and len(adapter._prefixes) == 1
