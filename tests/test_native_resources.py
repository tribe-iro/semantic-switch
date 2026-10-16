
from semantic_switch import model_jobs
import ctypes
import gc
import json
import subprocess
import sys
import threading
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_switch import Model, Runtime, backend
from semantic_switch.model_jobs import estimate as api_estimate
from semantic_switch.resident import Resources
from semantic_switch.adapters.qwen3 import Qwen3Adapter
from semantic_switch.errors import ConfigError
from semantic_switch.guidance import REVISION, Guidance
from semantic_switch.protocols import Question, ScoreRequest
from semantic_switch.training import Engine

from semantic_testkit.scoring import FakeScorer

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
VOCAB = 16


class Batch(ctypes.Structure):
    _fields_ = [("n_tokens", ctypes.c_int32), ("token", ctypes.POINTER(ctypes.c_int32)), ("embd", ctypes.POINTER(ctypes.c_float)),
                ("pos", ctypes.POINTER(ctypes.c_int32)), ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
                ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))), ("logits", ctypes.POINTER(ctypes.c_int8))]


class FakeLlama:
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
    GGML_TYPE_F16 = 1

    def __init__(self, model_log=(), context_log=()):
        self.freed, self.decodes, self.memories, self.rows = [], [], {}, {}
        self.model_log, self.context_log = model_log, context_log

    def llama_model_default_params(self):
        return SimpleNamespace()

    def llama_context_default_params(self):
        return SimpleNamespace()

    def llama_model_load_from_file(self, path, params):
        for line in self.model_log:
            backend._log(2, line.encode(), None)
        return 1

    def llama_model_get_vocab(self, model):
        return 2

    def llama_vocab_n_tokens(self, vocab):
        return VOCAB

    def llama_model_is_hybrid(self, model):
        return False

    llama_model_is_recurrent = llama_model_is_hybrid

    def llama_init_from_model(self, model, params):
        assert params.n_outputs_max >= params.n_seq_max, "llama.cpp output_reserve(n_seq_max) aborts below one output row per sequence"
        for line in self.context_log:
            backend._log(2, line.encode(), None)
        self.params = params
        self.memories[len(self.memories) + 3] = {}
        return len(self.memories) + 2

    def llama_get_memory(self, ctx):
        return ctx

    def llama_batch_init(self, n, embd, sequences):
        rows = [(ctypes.c_int32 * sequences)() for _ in range(n)]
        arrays = ((ctypes.c_int32 * n)(), (ctypes.c_int32 * n)(), (ctypes.c_int32 * n)(), (ctypes.POINTER(ctypes.c_int32) * n)(*rows), (ctypes.c_int8 * n)())
        self.buffers = (rows, arrays)
        token, pos, n_seq_id, seq_id, logits = arrays
        return Batch(0, token, None, pos, n_seq_id, seq_id, logits)

    def llama_n_ctx(self, ctx):
        return self.params.n_ctx

    def llama_n_ctx_seq(self, ctx):
        return self.params.n_ctx // self.params.n_seq_max

    def llama_n_batch(self, ctx):
        return self.params.n_batch

    def llama_n_seq_max(self, ctx):
        return self.params.n_seq_max

    def llama_batch_free(self, batch):
        self.freed.append("batch")

    def llama_free(self, ctx):
        self.freed.append("ctx")

    def llama_model_free(self, model):
        self.freed.append("model")

    def llama_memory_seq_rm(self, memory, sequence, p0, p1):
        self.memories[memory].pop(sequence, None)

    def llama_memory_seq_cp(self, memory, source, target, p0, p1):
        cells = self.memories[memory]
        cells[target] = {**cells.get(target, {}), **cells.get(source, {})}

    def llama_memory_clear(self, memory, data):
        self.memories[memory].clear()

    def llama_memory_seq_pos_max(self, memory, sequence):
        return max(self.memories[memory].get(sequence, {}), default=-1)

    def llama_decode(self, ctx, b):
        for j in range(b.n_tokens):
            for sequence in b.seq_id[j][:b.n_seq_id[j]]:
                self.memories[ctx].setdefault(sequence, {})[b.pos[j]] = b.token[j]
        self.rows = {}
        for j in range(b.n_tokens):
            if b.logits[j]:
                cells = self.memories[ctx][b.seq_id[j][0]]
                history = tuple(cells[p] for p in sorted(cells) if p <= b.pos[j])
                logits = np.random.default_rng(hash(history) % 2 ** 32).normal(size=VOCAB)
                self.rows[j] = (ctypes.c_float * VOCAB)(*logits)
        self.decodes.append(sum(b.logits[:b.n_tokens]))
        return 0

    def llama_get_logits_ith(self, ctx, j):
        return ctypes.cast(self.rows[j], ctypes.POINTER(ctypes.c_float))


@pytest.fixture
def fake_llama(monkeypatch):
    def install(**logs):
        library = FakeLlama(**logs)
        fake = lambda: library
        fake.debug = False
        monkeypatch.setattr(backend, "_llama", fake)
        return library
    return install


def scorer(**kw):
    return backend.LlamaScorer("fake.gguf", **{"n_ctx": 256, "n_batch": 64, **kw})


def fresh(r):
    with scorer() as s:
        return s.score([r])


def request(prefix, state, *suffixes):
    return ScoreRequest(prefix, state, suffixes, tuple(((1,), (2,)) for _ in suffixes))


def untracked_sequences_empty(s):
    tracked = ({0} if s._prefix is not None else set()) | ({1} if s._state is not None else set()) | set(s._resident.values())
    return all(s.ctx.L.llama_memory_seq_pos_max(s.ctx.mem, sequence) == -1
               for sequence in range(s.ctx.n_seq_max) if sequence not in tracked)


def test_closed_scorer_tokenizer_identity_raises():
    code = """
import threading
from types import SimpleNamespace
from semantic_switch import backend
L = backend._llama()
handles = backend._NativeHandles(L)
weights = backend.Weights.__new__(backend.Weights)
weights.L, weights.handles, weights.vocab, weights.n_vocab = L, handles, None, 4
scorer = backend.LlamaScorer.__new__(backend.LlamaScorer)
scorer.weights, scorer.handles, scorer.ctx = weights, handles, SimpleNamespace(lock=threading.RLock())
try:
    scorer.tokenizer_identity()
except RuntimeError as error:
    print(error)
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "runtime is closed", done


def test_unclosed_scorer_finalized(fake_llama):
    library = fake_llama()
    s = scorer()
    with pytest.warns(ResourceWarning, match="unclosed LlamaScorer"):
        del s
        gc.collect()
    assert library.freed == ["batch", "ctx", "model"]
    closed = scorer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with closed:
            pass
        closed.close()
        del closed
        gc.collect()
    assert library.freed == ["batch", "ctx", "model"] * 2


def test_construction_failure_releases_acquired_handles(fake_llama):
    library = fake_llama()
    def failed(ctx):
        raise RuntimeError("context query failed")
    library.llama_n_ctx = failed
    with pytest.raises(RuntimeError, match="context query failed"):
        scorer()
    assert library.freed == ["batch", "ctx", "model"]


def test_shared_interrupt_invalidates_cache(fake_llama):
    fake_llama()
    s = scorer()
    first, second = request((7, 8, 9), (3, 4), (5, 6)), request((9, 9), (3, 4), (5, 6))
    expected = fresh(first)
    s.score([first])
    remove = s.ctx.rm

    def interrupted(sequence):
        remove(sequence)
        if sequence == 0:
            raise KeyboardInterrupt

    s.ctx.rm = interrupted
    with pytest.raises(KeyboardInterrupt):
        s.score([second])
    s.ctx.rm = remove
    assert untracked_sequences_empty(s)
    again = s.score([first])
    assert np.array_equal(again.logp, expected.logp) and np.array_equal(again.log_mass, expected.log_mass)
    s.close()


def test_reference_empty_suffixes_keeps_kv_consistent(fake_llama):
    fake_llama()
    s = scorer()
    first = request((7, 8, 9), (3, 4), (5, 6))
    expected = fresh(first)
    s.score([first])
    s.shared = False
    s.score([ScoreRequest((7, 8), (3,), (), ())])
    assert untracked_sequences_empty(s)
    s.shared = True
    s.score([request((1, 2), (3, 4), (5,))])
    assert untracked_sequences_empty(s)
    again = s.score([first])
    assert untracked_sequences_empty(s) and np.array_equal(again.logp, expected.logp)
    s.close()


def test_close_concurrent_with_score(fake_llama):
    fake_llama()
    s = scorer()
    errors, started = [], threading.Event()

    def work():
        try:
            while True:
                s.score([request((7, 8, 9), (3, 4), (5, 6))])
                started.set()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    started.wait(5)
    s.close()
    for thread in threads:
        thread.join(5)
    assert len(errors) == 3 and all(type(e) is RuntimeError and str(e) == "runtime is closed" for e in errors)
    s.invalidate()


def test_decode_writer_chunks_outputs_and_refreshes_memberships(fake_llama):
    library = fake_llama()
    s = scorer(n_batch=8, n_ctx=1024)
    seen = []
    s.ctx.decode([((1,), k, 1 + k % 2, k) for k in range(5)], lambda key, j: seen.append((key, j)))
    assert library.params.n_outputs_max == s.ctx.n_outputs_max == s.ctx.n_seq_max == 4 + 2
    assert library.decodes == [1] * 5 and seen == [(k, 0) for k in range(5)] and s.ctx.outputs_per_decode == 1
    assert [library.memories[s.ctx.mem][1 + k % 2][k] for k in range(5)] == [1] * 5
    library.decodes.clear()
    s.ctx.decode([(tuple(range(20)), 0, 3, "end")], lambda key, j: seen.append((key, j)))
    assert library.decodes == [0, 0, 1] and seen[-1] == ("end", 3) and s.ctx.submissions == 8
    assert sorted(library.memories[s.ctx.mem][3]) == list(range(20))
    writer = s.ctx.writer
    assert writer.flags[:4].tolist() == [0, 0, 0, 1] and (writer.member[:4] == 3).all()
    s.ctx.decode([((9, 9), 30, 3, None)])
    assert writer.flags[:2].tolist() == [0, 0] and library.decodes[-1] == 0
    s.close()


def test_realized_config_records_capacity_and_repack(fake_llama, monkeypatch):
    from semantic_switch import adapters
    fake_llama(model_log=["load_tensors:          CPU model buffer size =   167.90 MiB\n",
                          "load_tensors:   CPU_REPACK model buffer size =   204.75 MiB\n",
                          "repack: repack tensor blk.0.ffn_up.weight with q4_K_8x8\n",
                          "repack: repack tensor blk.1.ffn_up.weight with q4_K_8x8\n"],
               context_log=["llama_context:        CPU  output buffer size =    21.44 MiB\n",
                            "llama_kv_cache:        CPU KV buffer size =   448.00 MiB\n",
                            "llama_kv_cache:        CPU KV buffer size =   448.00 MiB\n",
                            "sched_reserve:        CPU compute buffer size =   352.45 MiB\n"])
    monkeypatch.setattr(adapters, "adapter_for", lambda name, s, context=None: SimpleNamespace(name="fake"))
    with scorer() as native, Runtime(native) as rt:
        rt.judge()
        realized = rt.info()["capacity"]
    assert {key: realized[key] for key in ("n_outputs_max", "swa_full", "kv_unified", "use_extra_bufts", "repacked")} == \
        {"n_outputs_max": realized["n_seq_max"], "swa_full": True, "kv_unified": False, "use_extra_bufts": True, "repacked": ["q4_K_8x8"]}
    assert realized["buffers"] == {"CPU model": round(167.90 * 2 ** 20), "CPU_REPACK model": round(204.75 * 2 ** 20),
                                   "CPU output": round(21.44 * 2 ** 20), "CPU KV": 896 * 2 ** 20,
                                   "CPU compute": round(352.45 * 2 ** 20)}


def test_suffix_cache_ignores_guidance():
    adapter = Qwen3Adapter(FakeScorer())
    plain = Question("check", "The customer wants a refund.")
    guided = Question("check", "The customer wants a refund.", guidance=Guidance("Refunds include store credit."))
    assert adapter.suffix(plain) == adapter.suffix(guided) and len(adapter._suffixes) == 1


def profile_file(tmp_path, **changes):
    path = tmp_path / "quality.json"
    path.write_text(json.dumps({"schema": 1, "guidance_revision": REVISION, "max_guidance_tokens": 10,
                                "cohort": "synthetic", "evidence": "measured.json#sha256", **changes}))
    return path


def test_model_quality_profile_accepts_str(tmp_path):
    assert Model("model.gguf", quality_profile=str(profile_file(tmp_path))).quality_profile
    with pytest.raises(ConfigError, match="unsupported guidance quality profile"):
        Model("model.gguf", quality_profile=str(profile_file(tmp_path, schema=2)))
    with pytest.raises(ConfigError, match="not found"):
        Model("model.gguf", quality_profile=str(tmp_path / "missing.json"))


class LoadedScorer(FakeScorer):
    built = []

    def __init__(self, path, **kw):
        super().__init__()
        self.path, self.closed, self.shared = path, 0, True
        self.weights = SimpleNamespace(hybrid=False)
        LoadedScorer.built.append(self)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("failure", ["fingerprint", "profile"])
def test_load_failure_closes_scorer(tmp_path, monkeypatch, failure):
    from semantic_switch import api
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path))
    monkeypatch.setattr(backend, "LlamaScorer", LoadedScorer)
    LoadedScorer.built = []
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")
    profile = profile_file(tmp_path)
    spec = Model(str(weights), adapter="qwen3", device="cpu", quality_profile=str(profile))
    monkeypatch.setattr(Model, "resources", lambda self: {"weights": api.Resources(1)})
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: 1 << 50)
    if failure == "fingerprint":
        def stat_failure(spec):
            raise OSError("stat failed")
        monkeypatch.setattr(api, "_file_fingerprint", stat_failure)
    else:
        profile.write_text("{}")
    rt = Runtime(spec)
    with pytest.raises((OSError, ConfigError)):
        rt.judge()
    assert [s.closed for s in LoadedScorer.built] == [1]
    assert "judge" not in rt._identities and "judge" not in rt._loaded and rt.job is None
    rt.close()


def test_score_native_orders_and_validates():
    fake = FakeScorer(lambda state, question, width: [0.9, 0.1] if "first" in question else [0.2, 0.8])
    with Engine(fake) as rt:
        first, second, state = fake.tokenize("first"), fake.tokenize("second"), fake.tokenize("case")
        block = rt.score_native([ScoreRequest((), state, (first, second), (((1,), (2,)), ((1,), (2,)))),
                                 ScoreRequest((), state, (second,), (((1,), (2,)),))])
        assert np.allclose(np.exp(block.logp), [[0.9, 0.1], [0.2, 0.8], [0.2, 0.8]]) and block.error == [None] * 3
        assert rt.detokenize(first) == "first"
        for invalid, message in ((ScoreRequest((), state, (first,), ()), "answer-slot"),
                                 (ScoreRequest((), state, (first,), (((1,), (1,)),)), "disjoint"),
                                 (ScoreRequest((-1,), state, (first,), (((1,), (2,)),)), "non-negative")):
            with pytest.raises(ConfigError, match=message):
                rt.score_native([invalid])
        assert len(fake.requests) == 3
    with pytest.raises(RuntimeError, match="runtime is closed"):
        rt.score_native([])


@pytest.mark.slow
def test_n_outputs_max_preserves_logits(monkeypatch):
    import llama_cpp
    from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
    path = Model(QWEN3).path()

    def scores():
        with backend.LlamaScorer(path, n_ctx=2048) as s:
            adapter = Qwen3Adapter(s)
            requests = [adapter.request(adapter.state(text), CONFORMANCE_QUESTIONS) for text in CONFORMANCE_PROBES]
            blocks = []
            for shared in (True, False):
                s.shared = shared
                blocks.append(s.score(requests))
            return blocks, s.capacity()["buffers"]

    explicit, explicit_buffers = scores()
    initialize = llama_cpp.llama_init_from_model

    def library_default(model, params):
        chosen, params.n_outputs_max = params.n_outputs_max, 0
        try:
            return initialize(model, params)
        finally:
            params.n_outputs_max = chosen

    monkeypatch.setattr(llama_cpp, "llama_init_from_model", library_default)
    inherited, inherited_buffers = scores()
    for a, b in zip(explicit, inherited):
        assert not any(a.error) and np.array_equal(a.logp, b.logp) and np.array_equal(a.log_mass, b.log_mass)
    assert sum(v for k, v in explicit_buffers.items() if k.endswith("compute")) < \
        sum(v for k, v in inherited_buffers.items() if k.endswith("compute"))


@pytest.mark.slow
@pytest.mark.parametrize("source", [QWEN3, "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf"])
def test_capacity_reports_native_buffers(source):
    spec = Model(source, device="cpu")
    estimate = {use: value.host for use, value in spec.resources().items()}
    assert all(not value.devices for value in spec.resources().values())
    with Runtime(spec) as rt:
        rt.judge()
        buffers = rt.info()["capacity"]["buffers"]
    measured = {kind: sum(v for k, v in buffers.items() if k.endswith(" " + kind)) for kind in ("KV", "output")}
    assert abs(measured["KV"] - estimate["kv"]) <= 0.05 * estimate["kv"]
    assert abs(measured["output"] - estimate["output"] / 2) <= 0.05 * estimate["output"] / 2
    assert sum(api_estimate(spec).values(), Resources()).host == sum(estimate.values())


def test_default_runtime_loads_without_a_model_job_and_explicit_jobs_charge(tmp_path, monkeypatch):
    from semantic_switch import api
    from semantic_switch.training import model_job
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_MODEL_LOCK", str(tmp_path / "model-job.lock"))
    monkeypatch.setattr(backend, "LlamaScorer", LoadedScorer)
    monkeypatch.setattr(Model, "resources", lambda self: {"weights": api.Resources(1)})
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: 1 << 50)
    LoadedScorer.built = []
    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"weights")
    spec = Model(str(weights), adapter="qwen3", device="cpu")
    with model_job() as held, Runtime(spec) as first, Runtime(spec) as second:
        first.judge()
        second.judge()
        assert first.job is None and second.job is None and held.diagnostics()["participants"] == 0
    assert [s.closed for s in LoadedScorer.built] == [1, 1]
    with model_job(spec) as parent:
        with Runtime(spec, job=parent) as joined:
            joined.judge()
            assert parent.diagnostics()["participants"] == 1
        assert parent.state == "open" and parent.diagnostics()["participants"] == 0
