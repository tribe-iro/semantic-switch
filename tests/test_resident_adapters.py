
from semantic_switch import model_jobs
import ctypes
import re
import struct
import threading
import zlib
from contextlib import contextmanager

import numpy as np
import pytest

from semantic_switch import Model, Runtime, api, backend
from semantic_switch.training import model_job
from semantic_switch.model_jobs import estimate
from semantic_switch.backend import Placement
from semantic_switch.errors import ConfigError
from semantic_switch.protocols import ScoreRequest
from semantic_switch.judge import Question
from semantic_switch.judgment import Judgment, Policy
from semantic_switch.protocols import ScoreBlock
from semantic_switch.resident import Adapter, Binding, ResidentBase, Resources, binding_identity

from bench.adapter_profile import SEQUENCE, compare, cycle, distribution, native, paired, probe_adapter, replay, timing, within
from .test_native_resources import VOCAB, FakeLlama

MiB = 2 ** 20
DEVICE = "Vulkan0"
CPU, OFFLOADED, HYBRID = Placement(2, 0), Placement(2, 999, DEVICE), Placement(2, 2, DEVICE)


class Native:
    def __init__(self, function, argtypes, restype):
        self.function, self.argtypes, self.restype = function, argtypes, restype

    def __call__(self, *args):
        return self.function(*args)


class AdapterLlama(FakeLlama):
    llama_model_p_ctypes = llama_context_p_ctypes = llama_adapter_lora_p_ctypes = ctypes.c_void_p

    def __init__(self):
        super().__init__()
        self.active, self.loaded, self.freed_adapters, self.setter_calls, self.models = None, {}, [], 0, 0
        self.report, self.fail = {"CPU": 1024}, False
        handle = self.llama_adapter_lora_p_ctypes
        self.llama_adapter_lora_init = Native(self._init, [ctypes.c_void_p, ctypes.c_char_p], handle)
        self.llama_adapter_lora_free = Native(self._free, [handle], None)
        self.llama_set_adapters_lora = Native(self._set, [ctypes.c_void_p, ctypes.POINTER(handle), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)],
                                              ctypes.c_int32)

    llama_token = ctypes.c_int32

    def llama_tokenize(self, vocab, data, size, buf, cap, add_special, special):
        pieces = re.findall(rb"\s?\w+|\s|[^\w\s]", data[:size])
        if len(pieces) > cap:
            return -len(pieces)
        for index, piece in enumerate(pieces):
            buf[index] = zlib.crc32(piece) % VOCAB
        return len(pieces)

    def llama_model_load_from_file(self, path, params):
        self.models += 1
        return super().llama_model_load_from_file(path, params)

    def _init(self, model, path):
        if self.fail:
            return None
        handle = 100 + len(self.loaded) + len(self.freed_adapters)
        self.loaded[handle] = path
        for device, size in self.report.items():
            backend._log(2, f"llama_adapter_lora_init_impl: {device:>12} LoRA buffer size = {size / MiB:8.2f} MiB\n".encode(), None)
        return handle

    def _free(self, handle):
        self.freed_adapters.append(handle)
        self.loaded.pop(handle)

    def _set(self, ctx, adapters, count, scales):
        self.setter_calls += 1
        if count:
            assert scales[0] == 1.0 and adapters[0] in self.loaded
        self.active = adapters[0] if count else None
        return 0

    def llama_decode(self, ctx, b):
        for j in range(b.n_tokens):
            for sequence in b.seq_id[j][:b.n_seq_id[j]]:
                self.memories[ctx].setdefault(sequence, {})[b.pos[j]] = (b.token[j], self.loaded.get(self.active))
        self.rows = {}
        for j in range(b.n_tokens):
            if b.logits[j]:
                cells = self.memories[ctx][b.seq_id[j][0]]
                history = (tuple(cells[p] for p in sorted(cells) if p <= b.pos[j]), self.loaded.get(self.active))
                logits = np.random.default_rng(hash(history) % 2 ** 32).normal(size=VOCAB)
                self.rows[j] = (ctypes.c_float * VOCAB)(*logits)
        self.decodes.append(sum(b.logits[:b.n_tokens]))
        return 0


def gguf(path, meta, tensors):
    def string(value):
        return struct.pack("<Q", len(value.encode())) + value.encode()

    def value(item):
        if isinstance(item, str):
            return struct.pack("<I", 8) + string(item)
        if isinstance(item, list):
            return struct.pack("<IIQ", 9, 8, len(item)) + b"".join(string(x) for x in item)
        return struct.pack("<II", 4, item)
    out, offset, blobs = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)), 0, b""
    for key, item in meta.items():
        out += string(key) + value(item)
    for name, dims in tensors.items():
        out += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, offset)
        size = 4 * int(np.prod(dims))
        blobs += b"\0" * (size + -size % 32)
        offset += size + -size % 32
    path.write_bytes(out + b"\0" * (-len(out) % 32) + blobs)
    return str(path)


@pytest.fixture
def lib(monkeypatch):
    library = AdapterLlama()
    fake = lambda: library
    fake.debug = False
    monkeypatch.setattr(backend, "_llama", fake)
    return library


BASE_META = {"general.architecture": "gemma4", "gemma4.block_count": 2, "gemma4.attention.head_count_kv": 1, "gemma4.embedding_length": 8,
             "gemma4.attention.head_count": 1, "tokenizer.ggml.tokens": ["t"] * VOCAB}
BASE_TENSORS = {"blk.0.attn_q.weight": (8, 8), "blk.1.ffn_down.weight": (12, 8), "blk.0.attn_norm.weight": (8,), "token_embd.weight": (8, VOCAB)}


@pytest.fixture
def assets(tmp_path):
    base = gguf(tmp_path / "base.gguf", BASE_META, BASE_TENSORS)
    made = {label: probe_adapter(base, tmp_path / f"{label}.gguf", rank=4, alpha=8., seed=0, label=label) for label in "AB"}
    return base, {label: Adapter.verified(entry["path"], base) for label, entry in made.items()}


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_MODEL_LOCK", str(tmp_path / "model-job.lock"))
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: 1 << 50)

    @contextmanager
    def reserve(envelope: Resources):
        with model_job(extra={"reserved": envelope}) as handle:
            try:
                yield handle
            finally:
                while OPEN:
                    OPEN.pop().close()
    return reserve


def overhead(adapter):
    return adapter.header + 2 * 4 * 512 + 96


def request(prefix, state, *suffixes):
    return ScoreRequest(prefix, state, suffixes, tuple(((1,), (2,)) for _ in suffixes))


REQUESTS = [request((7, 8, 9), (3, 4), (5, 6), (6,)), request((7, 8, 9), (4, 4), (5,)), request((9,), (3,), (5, 6))]


OPEN = []


def resident(max_adapters=2, job=None, placement=CPU):
    scorer = backend.LlamaScorer("fake.gguf", n_ctx=256, n_batch=64)
    OPEN.append(ResidentBase(scorer, max_adapters=max_adapters, job=job, placement=lambda: placement))
    return OPEN[-1]


def views(owner, adapters):
    return {"base": owner.view(None, "base"), **{label: owner.view(adapter, label) for label, adapter in adapters.items()}}


def fresh(view):
    view.resident.scorer.invalidate()
    return view.score(REQUESTS)


def test_probe_adapters_are_deterministic_valid_and_declared_untrained(tmp_path, assets):
    base, adapters = assets
    again = probe_adapter(base, tmp_path / "again.gguf", rank=4, alpha=8., seed=0, label="A")
    assert Adapter.verified(again["path"], base).sha256 == adapters["A"].sha256 != adapters["B"].sha256
    assert again["trained"] is False and again["kind"] == "deterministic-probe" and again["targets"] == 2
    assert (adapters["A"].rank, adapters["A"].alpha, adapters["A"].targets) == (4, 8., ("blk.0.attn_q.weight", "blk.1.ffn_down.weight"))
    other = probe_adapter(base, tmp_path / "seed.gguf", rank=4, alpha=8., seed=1, label="A")
    assert Adapter.verified(other["path"], base).sha256 != adapters["A"].sha256


def test_incompatible_base_and_targets_fail_before_loading(tmp_path, assets, lib):
    base, adapters = assets
    narrow = gguf(tmp_path / "narrow.gguf", {"general.architecture": "gemma4"}, {"blk.0.attn_q.weight": (6, 8), "blk.1.ffn_down.weight": (12, 8)})
    with pytest.raises(ConfigError, match="do not match base weight"):
        Adapter.verified(adapters["A"].path, narrow)
    missing = gguf(tmp_path / "missing.gguf", {"general.architecture": "gemma4"}, {"blk.0.attn_q.weight": (8, 8)})
    with pytest.raises(ConfigError, match="absent from the base"):
        Adapter.verified(adapters["A"].path, missing)
    other = gguf(tmp_path / "other.gguf", {"general.architecture": "llama"}, {"blk.0.attn_q.weight": (8, 8), "blk.1.ffn_down.weight": (12, 8)})
    with pytest.raises(ConfigError, match="architecture"):
        Adapter.verified(adapters["A"].path, other)
    with pytest.raises(ConfigError, match="not a LoRA adapter"):
        Adapter.verified(base, base)
    with pytest.raises(ConfigError, match="differs from its verified identity"):
        Adapter.verified(adapters["A"].path, base, "0" * 64)
    assert lib.loaded == {}


def test_abi_mismatch_is_reported_before_any_adapter_load(assets, lib):
    _, adapters = assets
    lib.llama_set_adapters_lora = Native(lib._set, [ctypes.c_void_p], ctypes.c_int32)
    owner = resident()
    with pytest.raises(ConfigError, match="runtime adapter ABI llama_set_adapters_lora"):
        owner.view(adapters["A"], "A").score(REQUESTS)
    assert lib.loaded == {}
    owner.close()


def test_one_base_two_adapters_switch_and_match_fresh_references(assets, lib):
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    references = {role: fresh(view) for role, view in bound.items()}
    assert not np.allclose(references["A"].logp, references["base"].logp) and not np.allclose(references["A"].logp, references["B"].logp)
    owner.scorer.invalidate()
    for role in ("A", "B", "base", "A", "A", "B", "base", "base", "A"):
        block = bound[role].score(REQUESTS)
        assert np.array_equal(block.logp, references[role].logp) and np.array_equal(block.log_mass, references[role].log_mass), role
        assert lib.active == (None if role == "base" else next(h for h, p in lib.loaded.items() if p.decode() == adapters[role].path))
    assert owner.pool.stats["loads"] == 2 and lib.freed == [] and len(lib.loaded) == 2
    assert {key.binding for key in owner.scorer._resident} <= {"A", "B", "base"}
    owner.close()
    assert sorted(lib.freed_adapters) == [100, 101] and lib.freed == ["batch", "ctx", "model"]


class LayoutLlama(AdapterLlama):
    def __init__(self, stale):
        super().__init__()
        self.stale, self.pending = stale, ()

    def _set(self, ctx, adapters, count, scales):
        previous = self.active
        rc = super()._set(ctx, adapters, count, scales)
        if self.stale:
            self.pending, self.active = (self.active,), previous
        return rc

    def llama_decode(self, ctx, b):
        rc = super().llama_decode(ctx, b)
        occupied = sum(map(len, self.memories[ctx].values()))
        for row in self.rows.values():
            row[1] += 1e-3 * occupied
        if self.pending:
            (self.active,), self.pending = self.pending, ()
        return rc


@pytest.mark.parametrize("stale", [False, True])
def test_switch_reference_replays_the_cache_layout_and_still_detects_stale_adapters(assets, monkeypatch, stale):
    library = LayoutLlama(stale)
    fake = lambda: library
    fake.debug = False
    monkeypatch.setattr(backend, "_llama", fake)
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    read = lambda block: native(block, [Judgment.check("refund")] * 4, Policy())
    fresh = {role: next(cycle(owner.scorer, bound, REQUESTS, (role,), read))["first"][0] for role in bound}
    controls = {role: replay(owner.scorer, bound, REQUESTS, SEQUENCE, role, read) for role in bound}
    steps = list(cycle(owner.scorer, bound, REQUESTS, SEQUENCE, read))
    entries = [paired(step, controls[step["binding"]][index], kind) for index, step in enumerate(steps) for kind in ("first", "steady")]
    numerics = max(compare(step["first"][0], fresh[step["binding"]])["max_abs_prob"] for step in steps)
    switching = max(entry["vs_control"]["max_abs_prob"] for entry in entries)
    assert all(entry["control"]["matched_work"] and entry["cache"] == entry["control"]["cache"] for entry in entries)
    assert [step["switched"] for step in steps] == [True, True, True, True, False, True, False, True, False, True]
    assert numerics > 0 and (switching > 0) == stale
    assert all(within(entry["vs_control"], 1e-3) for entry in entries) != stale
    owner.close()


def test_profile_timing_pairs_each_call_with_its_control_and_reports_distributions():
    def call(ms, control):
        return {"ms": ms, "control": {"ms": control}, "overhead_ms": ms - control}
    cycles = [{"switched": True, "setter": {"ms": .02}, "first": call(10., 7.), "steady": call(6., 6.)},
              {"switched": True, "setter": {"ms": .04}, "first": call(30., 20.), "steady": call(8., 7.)},
              {"switched": False, "setter": {"ms": 0.}, "first": call(5., 5.), "steady": call(5., 4.)}]
    out = timing(cycles)
    assert out["first_scoring"]["switched"]["paired_overhead_ms"] == {"n": 2, "min": 3., "median": 6.5, "p90": 9.3, "max": 10.}
    assert out["steady_scoring"]["unchanged"]["paired_overhead_ms"]["n"] == 1 and out["steady_scoring"]["unchanged"]["paired_overhead_ms"]["max"] == 1.
    assert out["setter"]["switched"]["max"] == .04 and out["setter"]["unchanged"]["n"] == 1
    assert distribution([]) is None


def profiled(base, tmp_path, monkeypatch, **requirements):
    from argparse import Namespace
    from functools import partial
    from bench import adapter_profile
    monkeypatch.setattr(adapter_profile, "Model", partial(Model, adapter="qwen3"))
    monkeypatch.setattr(adapter_profile, "CONFORMANCE_QUESTIONS", (Question("check", "Is this a refund?"),))
    args = Namespace(base=base, adapter=None, work=str(tmp_path / "probe"), device="cpu", rank=4, alpha=8., seed=0, repeats=2, tolerance=1e-3, accept=.9,
                     max_setter_ms=None, max_first_scoring_ms=None, max_public_call_ms=None, max_adapter_mib=None, texts=["first input", "second input"])
    for key, value in requirements.items():
        setattr(args, key, value)
    return adapter_profile.profile(args)


CORRECTNESS = ("application", "switch_correctness", "fresh_repeat", "state_reuse", "batch_geometry", "reference_path", "layout_numerics")
LATENCY = ("setter_latency", "first_scoring_latency", "public_latency", "adapter_memory")


def test_profile_receipt_keys_capability_and_scopes_every_measurement(assets, lib, jobs, tmp_path, monkeypatch):
    base, _ = assets
    report = profiled(base, tmp_path, monkeypatch)
    assert report["assessment"]["capability"] == "unproven" and report["assessment"]["max_switch_diff"] == 0.
    assert set(report["assessment"]["key"]) == {"build", "base", "placement", "workload", "requirements"}
    assert report["workload"]["inputs"] == 2 and report["workload"]["rows"] == 2 * report["workload"]["questions"]
    assert len(report["cycles"]) == 2 * len(SEQUENCE)
    for entry in report["cycles"]:
        assert entry["setter"]["scope"] == "setter" and {entry["first"]["scope"], entry["steady"]["scope"]} == {"first_scoring", "steady_scoring"}
        assert entry["first"]["control"]["matched_work"] and entry["steady"]["cache"]["prefix"] == "hit"
    assert {call["scope"] for call in report["public"]["calls"]} == {"public_call"} and report["eviction"]["calls"][0]["scope"] == "public_reload"
    assert all(set(entry["memory"]) >= {"file_bytes", "tensor_bytes", "allocated_bytes"} for entry in report["adapters"].values())
    assessments = report["assessment"]["assessments"]
    assert list(assessments) == [*CORRECTNESS, *LATENCY] and report["assessment"]["failures"] == []
    assert all(assessments[name]["status"] == "pass" for name in CORRECTNESS) and all(assessments[name]["status"] == "undeclared" for name in LATENCY)
    assert report["primitives"]["state_reuse"]["A"]["cache"]["prefix"] == "hit" and report["primitives"]["batch_geometry"]["A"]["rows"] == 2
    row = report["references"]["A"]["fresh"][0]
    assert set(row) == {"native", "observation", "public"} and row["native"]["finite"] == [True, True] and row["native"]["error"] is None
    assert np.isclose(np.exp(row["native"]["log_mass"]), row["observation"]["mass"]) and row["observation"]["normalization"] == "option-softmax@1"
    assert row["observation"]["probabilities"] == pytest.approx(np.exp(row["native"]["logp"]) / np.exp(row["native"]["logp"]).sum())
    import json
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("stale", [False, True])
def test_profile_assesses_switching_and_cache_equivalence_independently(assets, jobs, tmp_path, monkeypatch, stale):
    library = LayoutLlama(stale)
    fake = lambda: library
    fake.debug = False
    monkeypatch.setattr(backend, "_llama", fake)
    base, _ = assets
    report = profiled(base, tmp_path, monkeypatch, max_setter_ms=1e3, max_first_scoring_ms=1e4, max_public_call_ms=1e4, max_adapter_mib=1.)
    assessments = report["assessment"]["assessments"]
    assert assessments["switch_correctness"]["status"] == ("fail" if stale else "pass")
    assert assessments["layout_numerics"]["status"] == "fail" and assessments["application"]["status"] == "pass"
    layout = assessments["layout_numerics"]["measured"].values()
    assert max(d["max_abs_mass"] for d in layout) > 1e-3 and (stale or all(d["max_abs_prob"] == 0. and not d["errors"] for d in layout))
    assert {"layout_numerics", "switch_correctness"} & set(report["assessment"]["failures"]) == {"layout_numerics", *(("switch_correctness",) if stale else ())}
    assert report["assessment"]["capability"] == "unsupported" and all(assessments[name]["status"] == "pass" for name in LATENCY)


def block(probabilities, mass):
    logp = np.log(np.asarray(probabilities, dtype=np.float64) * mass)
    return ScoreBlock(logp[None].astype(np.float32), np.array([np.log(mass)], dtype=np.float32), [None])


def test_native_diagnostic_detects_changed_answer_mass_behind_equal_normalized_answers():
    check, policy = [Judgment.check("The customer asks for a refund.")], Policy(.9, .9)
    accepted, thinner, starved = (native(block([.95, .05], mass), check, policy) for mass in (.9, .6, .4))
    assert accepted[0]["observation"]["probabilities"] == pytest.approx([.95, .05]) == thinner[0]["observation"]["probabilities"]
    assert accepted[0]["public"] == {"status": "ok", "reason": None, "answer": "true", "accepted": True}
    assert starved[0]["public"] == {"status": "uncertain", "reason": "low_answer_mass", "answer": "true", "accepted": False}
    same = compare(accepted, thinner)
    assert same["max_abs_prob"] < 1e-6 and same["max_abs_mass"] == pytest.approx(.3, abs=1e-6) and same["max_abs_log_mass"] == pytest.approx(np.log(1.5), abs=1e-6)
    assert not any(same[key] for key in ("shape_changes", "answer_changes", "status_changes", "acceptance_changes"))
    assert not within(same, 1e-3) and within(same, .31)
    starving = compare(accepted, starved)
    assert (starving["answer_changes"], starving["status_changes"], starving["acceptance_changes"]) == (0, 1, 1) and not within(starving, 1.)
    failed = native(ScoreBlock(np.full((1, 2), -np.inf, dtype=np.float32), np.array([np.nan], dtype=np.float32), ["native: decode failed"]), check, policy)
    assert failed[0]["native"] == {"columns": 2, "finite": [False, False], "logp": [None, None], "log_mass": None, "error": "native: decode failed"}
    broken = compare(accepted, failed)
    assert broken["shape_changes"] == 1 and broken["errors"] == ["native: decode failed"] and not within(broken, 1.)


def test_saved_prefix_reuse_never_crosses_bindings(assets, lib):
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    first = bound["A"].score(REQUESTS[:1])
    hits = owner.scorer.stats["prefix_hits"]
    bound["B"].score(REQUESTS[:1])
    assert owner.scorer.stats["prefix_hits"] == hits
    again = bound["A"].score(REQUESTS[:1])
    assert owner.scorer.stats["prefix_hits"] == hits + 1 and np.array_equal(again.logp, first.logp)
    owner.close()


def test_unchanged_binding_skips_setter_and_base_resets_to_zero_adapters(assets, lib):
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    bound["A"].score(REQUESTS)
    calls = lib.setter_calls
    bound["A"].score(REQUESTS)
    assert lib.setter_calls == calls
    bound["base"].score(REQUESTS)
    assert lib.active is None and lib.setter_calls == calls + 1
    owner.close()


def test_namespace_cannot_be_reused_for_another_adapter(assets, lib):
    _, adapters = assets
    owner = resident()
    owner.view(adapters["A"], "shared").score(REQUESTS)
    with pytest.raises(RuntimeError, match="already names adapter"):
        owner.view(adapters["B"], "shared").score(REQUESTS)
    owner.close()


def test_pool_evicts_only_unreferenced_adapters_under_the_byte_cap(assets, lib, jobs):
    _, adapters = assets
    a, b = (overhead(adapters[k]) + 4 * 256 + 8192 for k in "AB")
    with jobs(Resources(a + b - 1)) as job:
        owner = resident(max_adapters=2, job=job)
        bound = views(owner, adapters)
        reference = fresh(bound["A"])
        assert job.diagnostics()["participants"] == 1 and owner.pool.resident == Resources(a)
        assert owner.pool.placement()[adapters["A"].sha256]["buffers"] == {"CPU LoRA": round(1024 / MiB, 2) * MiB}
        bound["B"].score(REQUESTS)
        assert owner.pool.stats["evictions"] == 1 and owner.pool.resident == Resources(b) and list(owner.pool.placement()) == [adapters["B"].sha256]
        assert not any(key.binding == "A" for key in owner.scorer._resident) and lib.freed_adapters == [100]
        assert np.array_equal(bound["A"].score(REQUESTS).logp, reference.logp) and owner.pool.stats["loads"] == 3
        with owner.scorer.operation(), owner.pool.use(adapters["A"]):
            with pytest.raises(ConfigError, match="needs"):
                bound["B"].score(REQUESTS)
        assert owner.pool.stats["evictions"] == 2 and list(owner.pool.placement()) == [adapters["A"].sha256]
        owner.close()
        assert job.diagnostics()["participants"] == 0 and job.diagnostics()["resident_peak_bytes"]["host"] <= a + b - 1


def test_entry_bound_rejects_in_use_eviction(assets, lib):
    _, adapters = assets
    owner = resident(max_adapters=1)
    bound = views(owner, adapters)
    with owner.scorer.operation(), owner.pool.use(adapters["A"]):
        with pytest.raises(ConfigError, match="without evicting an adapter in use"):
            bound["B"].score(REQUESTS)
    owner.close()


def test_failed_decode_does_not_leave_the_next_caller_on_the_wrong_adapter(assets, lib):
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    references = {role: fresh(view) for role, view in bound.items()}
    decode = lib.llama_decode

    def interrupted(ctx, b):
        raise KeyboardInterrupt
    lib.llama_decode = interrupted
    with pytest.raises(KeyboardInterrupt):
        bound["A"].score(REQUESTS)
    lib.llama_decode = decode
    assert np.array_equal(bound["base"].score(REQUESTS).logp, references["base"].logp) and lib.active is None
    assert np.array_equal(bound["B"].score(REQUESTS).logp, references["B"].logp)
    owner.close()


def test_interleaved_threads_keep_each_binding_atomic(assets, lib):
    _, adapters = assets
    owner = resident()
    bound = views(owner, adapters)
    references = {role: fresh(view) for role, view in bound.items()}
    errors = []

    def work(role):
        try:
            for _ in range(20):
                assert np.array_equal(bound[role].score(REQUESTS).logp, references[role].logp)
        except BaseException as error:
            errors.append(error)
    threads = [threading.Thread(target=work, args=(role,)) for role in ("A", "B", "base") * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors
    owner.close()


def test_close_waits_for_active_users_and_frees_adapters_first(assets, lib):
    _, adapters = assets
    order = []
    free = lib._free
    lib.llama_adapter_lora_free.function = lambda handle: (order.append("adapter"), free(handle))
    lib.llama_model_free = lambda model: order.append("model")
    owner = resident()
    bound = views(owner, adapters)
    bound["A"].score(REQUESTS)
    entered, release = threading.Event(), threading.Event()

    def hold():
        with owner.scorer.operation():
            entered.set()
            release.wait(5)
    holder = threading.Thread(target=hold)
    holder.start()
    entered.wait(5)
    closer = threading.Thread(target=owner.close)
    closer.start()
    closer.join(.2)
    assert closer.is_alive() and order == []
    release.set()
    holder.join(5)
    closer.join(5)
    assert order == ["adapter", "model"]
    with pytest.raises(RuntimeError, match="closed"):
        bound["A"].score(REQUESTS)


def test_binding_identity_and_representation_records(assets):
    _, adapters = assets
    identity = {"sha256": "a" * 64, "template": "t", "prompt": "p", "presentation": {"revision": 1}, "file": "x.gguf"}
    lora = {**identity, "lora": adapters["A"].record()}
    assert binding_identity(identity, None) == binding_identity({**identity, "file": "moved.gguf"}, None) != binding_identity(lora, None)
    assert binding_identity(identity, {"device": "cpu"}) != binding_identity(identity, {"device": "gpu"})
    record = Binding("refund", "runtime-adapter", lora, None, ("refund", "b")).record()
    assert record["scope"] == ["b", "refund"] and record["identity"]["lora"]["scale"] == 1.0 and "file" not in record["identity"]
    with pytest.raises(ConfigError, match="exactly one adapter"):
        Binding("refund", "base", lora)
    with pytest.raises(ConfigError, match="exactly one adapter"):
        Binding("refund", "runtime-adapter", identity)
    with pytest.raises(ConfigError, match="unknown representation"):
        Binding("refund", "blend", identity)


def test_adapter_estimates_follow_tensor_placement_and_memory_architecture(assets):
    _, adapters = assets
    adapter = adapters["A"]
    assert adapter.tensor_bytes == 64 + 64 + 96 + 64 and adapter.header == adapter.file_bytes - 288
    host_only = overhead(adapter) + 4 * 256 + 8192
    assert CPU.lora(adapter).resources == Resources(host_only)
    assert OFFLOADED.lora(adapter).resources == Resources(overhead(adapter), {DEVICE: 4 * 256 + 8192})
    assert HYBRID.lora(adapter).resources == Resources(overhead(adapter) + 2 * 256 + 8192, {DEVICE: 2 * 256 + 8192})
    unified = Placement(2, 2, DEVICE, True).lora(adapter).resources
    assert unified == Resources(overhead(adapter) + 2 * (2 * 256 + 8192), {DEVICE: 2 * 256 + 8192})
    assert Placement(2, 999, DEVICE, True).lora(adapter).resources == Resources(overhead(adapter) + 4 * 256 + 8192, {DEVICE: 4 * 256 + 8192})


def test_unknown_target_placement_cannot_certify_a_gpu_load(assets):
    _, adapters = assets
    odd = Adapter(**{**{name: getattr(adapters["A"], name) for name in Adapter.__slots__}, "tensors": (("altup_proj.lora_a", 64),)})
    with pytest.raises(ConfigError, match="no pinned placement"):
        OFFLOADED.lora(odd)
    assert CPU.lora(odd).resources.host > 0


def test_base_estimates_count_unified_device_bytes_once_in_physical_memory(assets, monkeypatch):
    base, _ = assets
    tensors = backend.gguf(base).tensors
    size, device = __import__("os").path.getsize(base), 256 + 384 + 32
    tied = 8 * VOCAB * 4
    assert OFFLOADED.weights(size, tensors) == Resources(size - device, {DEVICE: device + tied})
    assert Placement(2, 999, DEVICE, True).weights(size, tensors) == Resources(size + tied, {DEVICE: device + tied})
    assert CPU.weights(size, tensors) == Resources(size)


def test_admission_charges_a_shared_base_once_and_each_adapter_once(assets, monkeypatch):
    base, adapters = assets
    monkeypatch.setattr(backend, "offload_device", lambda: (DEVICE, False))
    models = [Model(base, device="gpu"), Model(base, device="gpu", lora=adapters["A"].path), Model(base, device="gpu", lora=adapters["B"].path),
              Model(base, device="gpu", lora=adapters["A"].path)]
    found = estimate(*models)
    assert sorted(found) == sorted([f"{base}:weights", f"{base}:kv", f"{base}:output", f"{adapters['A'].path}:adapter", f"{adapters['B'].path}:adapter"])
    for label in "AB":
        assert found[f"{adapters[label].path}:adapter"] == Resources(overhead(adapters[label]), {DEVICE: 4 * 256 + 8192})
    other = estimate(*models, Model(base, device="gpu", n_ctx=3072))
    assert other[f"{base}:weights"] == found[f"{base}:weights"] + found[f"{base}:weights"] and other[f"{adapters['A'].path}:adapter"] == found[f"{adapters['A'].path}:adapter"]


def test_model_representation_is_declared_and_never_interchanged(assets):
    base, adapters = assets
    assert Model(base).representation == "base" and Model(base, lora=adapters["A"].path).representation == "runtime-adapter"
    assert Model(base, representation="merged").representation == "merged"
    with pytest.raises(ConfigError, match="contradicts"):
        Model(base, lora=adapters["A"].path, representation="merged")
    with pytest.raises(ConfigError, match="contradicts"):
        Model(base, representation="runtime-adapter")


def test_offloaded_adapter_is_charged_per_device_before_native_allocation(assets, lib, jobs):
    _, adapters = assets
    lib.report = {"CPU": 512, DEVICE: 512}
    charge = HYBRID.lora(adapters["A"]).resources
    with jobs(Resources(charge.host)) as job:
        owner = resident(job=job, placement=HYBRID)
        with pytest.raises(ConfigError, match="needs"):
            owner.view(adapters["A"], "A").score(REQUESTS)
        assert lib.loaded == {} and lib.freed_adapters == [] and job.diagnostics()["participants"] == 0
        owner.close()
    with jobs(charge) as job:
        owner = resident(job=job, placement=HYBRID)
        owner.view(adapters["A"], "A").score(REQUESTS)
        assert job.diagnostics()["resident_peak_bytes"] == charge.record() and owner.pool.resident == charge
        assert owner.pool.placement()[adapters["A"].sha256]["buffers"] == {"CPU LoRA": 0, f"{DEVICE} LoRA": 0}
        owner.close()
        assert job.diagnostics()["participants"] == 0 and len(lib.freed_adapters) == 1


@pytest.mark.parametrize("report", [{DEVICE: 64 * 1024}, {"CPU": 64 * 1024}, {"Vulkan1": 512}, {}])
def test_underestimated_or_escaping_allocation_is_freed_and_reported(assets, lib, jobs, report):
    _, adapters = assets
    lib.report = report
    with jobs(Resources(1 << 30, {DEVICE: 1 << 30})) as job:
        owner = resident(job=job, placement=OFFLOADED)
        with pytest.raises(ConfigError, match="exceeds or escapes its estimate.*detached and freed"):
            owner.view(adapters["A"], "A").score(REQUESTS)
        assert lib.loaded == {} and lib.freed_adapters == [100] and owner.pool.stats["underestimates"] == 1
        assert job.diagnostics()["participants"] == 0 and owner.pool.resident == Resources() and job.envelope == Resources(1 << 30, {DEVICE: 1 << 30})
        owner.close()


def test_failed_adapter_load_releases_its_charge_once(assets, lib, jobs):
    _, adapters = assets
    lib.fail = True
    with jobs(Resources(1 << 30)) as job:
        owner = resident(job=job)
        with pytest.raises(RuntimeError, match="cannot load adapter"):
            owner.view(adapters["A"], "A").score(REQUESTS)
        assert job.diagnostics()["participants"] == 0 and owner.pool.placement() == {}
        lib.fail = False
        owner.view(adapters["A"], "A").score(REQUESTS)
        owner.close()
        assert job.diagnostics()["participants"] == 0 and lib.freed_adapters == [100]


@pytest.fixture
def gpu(lib, monkeypatch):
    monkeypatch.setattr(backend, "offload_device", lambda: (DEVICE, False))
    monkeypatch.setattr(api, "_gpu", lambda: True)
    lib.report = {DEVICE: 1024}
    return lib


def gpu_models(base, adapters):
    return {"judge": Model(base, adapter="qwen3", device="gpu"), "A": Model(base, adapter="qwen3", device="gpu", lora=adapters["A"].path)}


def test_zero_device_envelope_rejects_before_native_init(assets, gpu, jobs):
    base, adapters = assets
    with jobs(Resources(1 << 40)) as job:
        with Runtime(bindings=gpu_models(base, adapters), job=job) as rt:
            with pytest.raises(ConfigError, match=f"needs .*'{DEVICE}'"):
                rt.judge("A")
        assert gpu.models == 0 and gpu.loaded == {} and job.diagnostics()["participants"] == 0


def test_reserved_gpu_runtime_reaches_init_and_frees_every_participant(assets, gpu, jobs):
    base, adapters = assets
    models = gpu_models(base, adapters)
    with jobs(sum(estimate(*models.values()).values(), Resources())) as job:
        with Runtime(bindings=models, job=job) as rt:
            rt.judge("A").scorer.score(REQUESTS)
            rt.judge("judge").scorer.score(REQUESTS)
            during = (gpu.models, len(gpu.loaded), job.diagnostics()["participants"])
        assert during == (1, 1, 2) and job.diagnostics()["participants"] == 0 and gpu.loaded == {} and gpu.freed[-1] == "model"


def test_default_runtime_takes_no_job_and_keeps_an_adapter_whose_estimate_misses(assets, gpu, monkeypatch, tmp_path):
    base, adapters = assets
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(backend, "_ADVISED", set())
    gpu.report = {"Vulkan1": 512}
    models = gpu_models(base, adapters)
    with Runtime(bindings=models) as rt:
        with pytest.warns(RuntimeWarning, match="exceeds or escapes its estimate.*advisory"):
            first = rt.judge("A").scorer.score(REQUESTS)
        again = rt.judge("A").scorer.score(REQUESTS)
        pool = rt.judge("A").scorer.resident.pool
        assert rt.job is None and np.array_equal(first.logp, again.logp) and len(gpu.loaded) == 1
        assert pool.stats["underestimates"] == 1 and set(pool.placement()[adapters["A"].sha256]["buffers"]) == {"Vulkan1 LoRA"}
    assert gpu.loaded == {}


def test_unknown_placement_does_not_block_default_adapter_loading(assets, lib, monkeypatch):
    _, adapters = assets
    monkeypatch.setattr(backend, "_ADVISED", set())

    def unknown():
        raise ConfigError("offload devices ['Vulkan0', 'Vulkan1'] split layers")
    scorer = backend.LlamaScorer("fake.gguf", n_ctx=256, n_batch=64)
    owner = ResidentBase(scorer, max_adapters=2, job=None, placement=unknown)
    with pytest.warns(RuntimeWarning, match="estimate unavailable"):
        owner.view(adapters["A"], "A").score(REQUESTS)
    assert owner.pool.stats["unestimated"] == 1 and owner.pool.placement()[adapters["A"].sha256]["charged"] is None
    owner.close()
    assert lib.loaded == {}
