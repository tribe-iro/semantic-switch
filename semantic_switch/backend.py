from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import struct
import sys
import threading
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .errors import ConfigError
from .protocols import ScoreBlock, ScoreRequest

_lock = threading.Lock()
_init = threading.Lock()
_ready = False
_captured: list | None = None
BUFFER = re.compile(r"(\S+) +(model|output|KV|compute) buffer size = +([\d.]+) MiB")
REPACKED = re.compile(r"repack tensor \S+ with (\S+)")
TENSOR_TYPES = re.compile(r"- type +(\S+): +\d+ tensors")


def _log(level, text, data):
    if _captured is not None:
        _captured.append(text.decode(errors="replace"))
    if _llama.debug:
        sys.stderr.write(text.decode(errors="replace"))


def require_local(*packages):
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise ConfigError(f"local judges need the 'local' extra ({', '.join(missing)} not installed): pip install 'semantic-switch[local]'")


def _llama():
    global _ready
    require_local("llama_cpp")
    import llama_cpp as L

    with _lock:
        if not _ready:
            _llama.debug = bool(os.environ.get("SEMANTIC_SWITCH_DEBUG"))
            _llama.log = L.llama_log_callback(_log)
            L.llama_log_set(_llama.log, ctypes.c_void_p())
            L.llama_backend_init()
            _ready = True
    return L


@contextmanager
def _capture():
    global _captured
    with _init:
        _captured = []
        try:
            yield _captured
        finally:
            _captured = None


def _buffers(lines: list[str]) -> dict[str, int]:
    out = {}
    for device, kind, mib in BUFFER.findall("".join(lines)):
        out[f"{device} {kind}"] = out.get(f"{device} {kind}", 0) + round(float(mib) * 2 ** 20)
    return out


_GGUF = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        def read(code):
            return struct.unpack("<" + code, f.read(struct.calcsize(code)))[0]

        def value(kind):
            if kind == 8:
                return f.read(read("Q")).decode(errors="replace")
            if kind != 9:
                return read(_GGUF[kind])
            item, count = read("I"), read("Q")
            if item in _GGUF:
                return list(struct.unpack(f"<{count}{_GGUF[item]}", f.read(count * struct.calcsize(_GGUF[item]))))
            return [value(item) for _ in range(count)]

        if f.read(4) != b"GGUF" or read("I") < 2:
            raise ValueError(f"{path} is not a GGUF v2+ file")
        read("Q")
        return {value(8): value(read("I")) for _ in range(read("Q"))}


class _NativeHandles:
    def __init__(self, L):
        self.L = L
        self.batch = self.ctx = self.model = None
        self.free = (("batch", L.llama_batch_free), ("ctx", L.llama_free), ("model", L.llama_model_free))

    def close(self):
        for name, free in self.free:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, None)
                free(value)


def _collect(handles: _NativeHandles):
    warnings.warn("unclosed LlamaScorer released by its finalizer; call close()", ResourceWarning)
    handles.close()


def _lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    if not np.isfinite(m):
        raise RuntimeError("native logits are non-finite")
    return m + float(np.log(np.exp(x - m).sum()))


@lru_cache(maxsize=256)
def packed(slots: tuple[tuple[int, ...], ...]) -> tuple[np.ndarray, ...]:
    ids = tuple(np.asarray(group, dtype=np.intp) for group in slots)
    flat = np.concatenate(ids) if ids else np.empty(0, dtype=np.intp)
    if not all(len(group) for group in ids) or len(np.unique(flat)) != len(flat):
        raise ValueError("answer slots must be nonempty and disjoint")
    return ids


def reduce_row(row: np.ndarray, slots: tuple[tuple[int, ...], ...], scratch: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    top = float(np.max(row))
    if not np.isfinite(top):
        raise RuntimeError("native logits are non-finite")
    work = np.subtract(row, top, out=np.empty(len(row)) if scratch is None else scratch[:len(row)], dtype=np.float64)
    lse_all = top + float(np.log(np.sum(np.exp(work, out=work))))
    slot_lp = np.array([_lse(row[ids].astype(np.float64)) for ids in packed(slots)], dtype=np.float64)
    lse_slots = _lse(slot_lp)
    return (slot_lp - lse_slots).astype(np.float32), min(0., lse_slots - lse_all)


SHARED_CHECKS = ("D1 shared = reference", "D3 no leakage")


def shared_ok(checks: dict) -> bool:
    return all(checks.get(k, {"ok": False})["ok"] for k in SHARED_CHECKS)


def gpu_available() -> bool:
    return bool(_llama().llama_supports_gpu_offload())


class Weights:
    def __init__(self, path: str, n_gpu_layers: int = 0, vocab_only: bool = False, repack: bool = True):
        L = _llama()
        params = L.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        params.no_host = n_gpu_layers == 0
        params.vocab_only = vocab_only
        params.use_extra_bufts = repack
        self.L = L
        self.handles = _NativeHandles(L)
        with _capture() as lines:
            self.handles.model = L.llama_model_load_from_file(path.encode(), params)
        if not self.handles.model:
            raise RuntimeError(f"cannot load model {path}")
        self.vocab = L.llama_model_get_vocab(self.ptr)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.offload = n_gpu_layers > 0
        self.hybrid = bool(L.llama_model_is_hybrid(self.ptr) or L.llama_model_is_recurrent(self.ptr))
        self.buffers = _buffers(lines)
        self.realized = {"device": "gpu" if self.offload else "cpu", "tensor_types": sorted(set(TENSOR_TYPES.findall("".join(lines)))),
                         "use_extra_bufts": params.use_extra_bufts, "repacked": sorted(set(REPACKED.findall("".join(lines))))}

    @property
    def ptr(self):
        return self.handles.model

    def _live(self):
        if self.handles.model is None:
            raise RuntimeError("runtime is closed")

    def tokenize(self, text: str, special: bool = False, add_special: bool = False) -> tuple[int, ...]:
        self._live()
        data = text.encode()
        cap = len(data) + 16
        buf = (self.L.llama_token * cap)()
        n = self.L.llama_tokenize(self.vocab, data, len(data), buf, cap, add_special, special)
        if n < 0:
            buf = (self.L.llama_token * -n)()
            n = self.L.llama_tokenize(self.vocab, data, len(data), buf, -n, add_special, special)
        return tuple(buf[:n])

    def detokenize(self, tokens: Sequence[int], special: bool = False) -> str:
        self._live()
        piece = ctypes.create_string_buffer(256)
        out = bytearray()
        for t in tokens:
            n = self.L.llama_token_to_piece(self.vocab, t, piece, 256, 0, special)
            if n < 0:
                piece = ctypes.create_string_buffer(-n)
                n = self.L.llama_token_to_piece(self.vocab, t, piece, len(piece), 0, special)
            if n < 0:
                raise RuntimeError("cannot detokenize token")
            out += piece.raw[:n]
        return out.decode("utf-8", errors="replace")

    def chat(self) -> dict:
        self._live()
        tmpl = self.L.llama_model_chat_template(self.ptr, None)
        text = lambda tok: self.detokenize([tok], special=True) if tok >= 0 else ""
        return {"template": tmpl.decode() if tmpl else None, "bos": text(self.L.llama_vocab_bos(self.vocab)),
                "eos": text(self.L.llama_vocab_eos(self.vocab)),
                "add_bos": bool(self.L.llama_vocab_get_add_bos(self.vocab))}

    def close(self):
        self.handles.close()


MAX_SEQUENCES = 256


def validate_native_integer(name, value, *, optional=False):
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    maximum = (1 << 31) - 1 if name in ("threads", "n_batch") else (1 << 32) - 1
    if value > maximum:
        raise ValueError(f"{name} exceeds native integer limit {maximum}")


BRANCHES = 32
INPUT_TOKENS = 1024
GROUPINGS = ("serial", "batched")


@dataclass(frozen=True, slots=True)
class Capacity:
    n_ctx: int
    n_batch: int
    prefix_slots: int
    states: int
    branches: int

    @property
    def n_ubatch(self) -> int:
        return min(self.n_batch, 512)

    @property
    def n_seq_max(self) -> int:
        return self.prefix_slots + self.states + self.branches

    @property
    def n_outputs_max(self) -> int:
        return self.n_seq_max

    @property
    def outputs_per_decode(self) -> int:
        return self.branches

    @property
    def first_branch(self) -> int:
        return 1 + self.states

    @property
    def residents(self) -> range:
        return range(self.n_seq_max - self.prefix_slots + 1, self.n_seq_max)


def capacity(n_ctx: int | None, n_batch: int, inputs: int, prefix_slots: int, grouping: str = "serial") -> Capacity:
    for name, value in (("n_ctx", n_ctx), ("n_batch", n_batch), ("inputs", inputs), ("prefix_slots", prefix_slots)):
        validate_native_integer(name, value, optional=name == "n_ctx")
    if grouping not in GROUPINGS:
        raise ValueError(f"grouping must be one of {', '.join(GROUPINGS)}")
    states, branches = (inputs, BRANCHES) if grouping == "batched" else (1, 1)
    out = Capacity(n_ctx or 8192 + (states - 1) * INPUT_TOKENS, n_batch, prefix_slots, states, branches)
    validate_native_integer("n_ctx", out.n_ctx)
    if out.n_seq_max > MAX_SEQUENCES:
        raise ValueError(f"{out.n_seq_max} sequence slots exceed the native limit of {MAX_SEQUENCES}")
    if min(out.n_ctx, out.n_batch, 512) < out.n_seq_max:
        raise ValueError(f"native capacity requires n_ctx and n_batch >= {out.n_seq_max} for {out.n_seq_max} sequence slots; microbatch limit is 512")
    return out


class Context:
    def __init__(self, weights: Weights, cap: Capacity, threads: int | None):
        validate_native_integer("threads", threads, optional=True)
        L = weights.L
        p = L.llama_context_default_params()
        p.n_ctx = cap.n_ctx
        p.n_batch = cap.n_batch
        p.n_ubatch = cap.n_ubatch
        p.n_seq_max = cap.n_seq_max
        p.n_outputs_max = cap.n_outputs_max
        p.n_threads = p.n_threads_batch = threads or os.cpu_count()
        p.swa_full = True
        p.kv_unified = True
        p.op_offload = weights.offload
        p.flash_attn_type = L.LLAMA_FLASH_ATTN_TYPE_ENABLED
        p.type_k = p.type_v = L.GGML_TYPE_F16
        self.L = L
        self.w = weights
        self.handles = weights.handles
        with _capture() as lines:
            self.handles.ctx = L.llama_init_from_model(weights.ptr, p)
        if not self.handles.ctx:
            raise RuntimeError("cannot create llama context")
        self.mem = L.llama_get_memory(self.ptr)
        self.handles.batch = L.llama_batch_init(cap.n_batch, 0, cap.n_seq_max)
        self.n_ctx, self.n_batch, self.n_seq_max = L.llama_n_ctx(self.ptr), L.llama_n_batch(self.ptr), L.llama_n_seq_max(self.ptr)
        self.n_outputs_max, self.outputs_per_decode = p.n_outputs_max, cap.outputs_per_decode
        self.lock = threading.RLock()
        self.submissions = 0
        self.submitted_tokens = 0
        self.requested = {"n_ctx": cap.n_ctx, "n_batch": cap.n_batch, "n_seq_max": cap.n_seq_max}
        self.buffers = _buffers(lines)
        self.realized = {"n_outputs_max": p.n_outputs_max, "swa_full": p.swa_full, "kv_unified": p.kv_unified, "n_ubatch": p.n_ubatch,
                         "flash_attn": "enabled", "type_k": "f16", "type_v": "f16"}

    @property
    def ptr(self):
        return self.handles.ctx

    def rm(self, seq: int):
        if self.mem:
            self.L.llama_memory_seq_rm(self.mem, seq, -1, -1)

    def _live(self):
        if self.handles.ctx is None:
            raise RuntimeError("runtime is closed")

    def cp(self, src: int, dst: int):
        self._live()
        self.L.llama_memory_seq_cp(self.mem, src, dst, -1, -1)

    def clear(self):
        if self.mem:
            self.L.llama_memory_clear(self.mem, True)

    def run(self, entries: list, on_output):
        self._live()
        start = 0
        while start < len(entries):
            end, outputs = start, 0
            while end < len(entries) and end - start < self.n_batch and (entries[end][3] is None or outputs < self.outputs_per_decode):
                outputs += entries[end][3] is not None
                end += 1
            chunk, start = entries[start:end], end
            b = self.handles.batch
            b.n_tokens = len(chunk)
            for j, (tok, pos, seq, out) in enumerate(chunk):
                b.token[j] = tok
                b.pos[j] = pos
                sequences = (seq,) if isinstance(seq, int) else seq
                b.n_seq_id[j] = len(sequences)
                for index, sequence in enumerate(sequences):
                    b.seq_id[j][index] = sequence
                b.logits[j] = out is not None
            self.submissions += 1
            self.submitted_tokens += len(chunk)
            rc = self.L.llama_decode(self.ptr, b)
            if rc != 0:
                raise RuntimeError(f"llama decode returned {rc}")
            for j, (_, _, seq, out) in enumerate(chunk):
                if out is not None:
                    on_output(out, j, seq)

    def logits(self, j: int) -> np.ndarray:
        self._live()
        return np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ptr, j), shape=(self.w.n_vocab,))


SERIAL = 1
SCHEDULING = "resident-prefix-cache@1"


class LlamaScorer:
    def __init__(self, path: str, *, n_ctx: int | None = None, n_batch: int = 512, inputs: int = 1,
                 threads: int | None = None, n_gpu_layers: int = 0, shared: bool | None = None,
                 prefix_slots: int = 4, grouping: str = "serial",
                 repack: bool = True):
        validate_native_integer("threads", threads, optional=True)
        self.layout = capacity(n_ctx, n_batch, inputs, prefix_slots, grouping)
        self.weights = Weights(path, n_gpu_layers, repack=repack)
        self.handles = self.weights.handles
        self.inputs = inputs
        self.prefix_slots = prefix_slots
        self._resident = {}
        self._dirty = False
        try:
            self.ctx = Context(self.weights, self.layout, threads)
        except BaseException:
            self.handles.close()
            raise
        self._finalizer = weakref.finalize(self, _collect, self.handles)
        self.shared = (not self.weights.hybrid) if shared is None else shared
        self.path = path
        self.stats = {key: 0 for key in ("prefix_hits", "prefix_misses", "state_hits", "state_misses",
                                       "state_tokens", "suffix_tokens", "submissions", "branch_groups", "failures",
                                       "prefix_tokens", "prefix_evictions")}
        self.grouping = grouping
        self.width, self.branches = self.layout.states, self.layout.branches
        self._scratch = np.empty(self.weights.n_vocab, dtype=np.float64)
        self.fused = self.weights.offload and grouping == "batched"
        self._prefix: tuple | None = None
        self._states: dict[int, tuple] = {}

    def _live(self):
        if self.handles.ctx is None:
            raise RuntimeError("runtime is closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def invalidate(self):
        with self.ctx.lock:
            if self.handles.ctx is None:
                return
            self._resident = {}
            self._prefix = None
            self._states = {}
            self.ctx.clear()
            self._dirty = False

    def capacity(self) -> dict:
        with self.ctx.lock:
            self._live()
            c = self.ctx
            return {"n_ctx": c.n_ctx, "n_batch": c.n_batch, "n_seq_max": c.n_seq_max, "prefix_slots": self.prefix_slots,
                    "scheduling": SCHEDULING, **c.realized, **self.weights.realized,
                    "buffers": {**self.weights.buffers, **c.buffers}, "shared": self.shared, "grouping": self.grouping,
                    "serial": SERIAL if self.grouping == "serial" else None,
                    "fusion": self.fused if self.shared else self.weights.offload}

    def tokenizer_identity(self):
        from .compiler import digest
        weights = self.weights
        with self.operation():
            vocab = [weights.L.llama_vocab_get_text(weights.vocab, i).decode('utf-8', errors='replace') for i in range(weights.n_vocab)]
            probes = ['true False', 'A B C', 'Refund café não', '<input>\ncase\n</input>', '[1] [2]']
            special = {name: getattr(weights.L, 'llama_vocab_' + name)(weights.vocab) for name in ('bos', 'eos', 'pad')}
            return {'vocab_size': weights.n_vocab, 'vocab_sha256': digest(vocab),
                    'padding_tokens': [[i, token] for i, token in enumerate(vocab) if token == f'[PAD{i}]'], 'special': special,
                    'probes': [{'text': text, 'ids': list(self.tokenize(text))} for text in probes]}

    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]:
        with self.operation():
            return self.weights.tokenize(text, special)

    def detokenize(self, tokens: Sequence[int]) -> str:
        with self.operation():
            return self.weights.detokenize(tokens)

    def chat(self) -> dict:
        with self.operation():
            return self.weights.chat()

    @contextmanager
    def operation(self):
        with self.ctx.lock:
            self._live()
            yield

    def score(self, requests: Sequence[ScoreRequest], *, reference: bool = False) -> ScoreBlock:
        width = max((len(s) for r in requests for s in r.slots), default=1)
        out = ScoreBlock.empty(sum(len(r.suffixes) for r in requests), width)
        rows, row = [], 0
        for r in requests:
            rows.append(row)
            row += len(r.suffixes)
        with self.operation():
            if self._dirty:
                self.invalidate()
            if self.shared and not reference:
                for window in self._windows(requests):
                    self._shared([(requests[n], rows[n]) for n in window], out)
            else:
                for r, row0 in zip(requests, rows):
                    self._reference(r, row0, out)
        return out

    def _windows(self, requests: Sequence[ScoreRequest]) -> list[list[int]]:
        windows, cur, states, used, largest = [], [], set(), 0, 0
        for n, r in enumerate(requests):
            if not r.suffixes:
                continue
            new = r.state not in states
            cost = len(r.state) if new else 0
            if cur and (requests[cur[0]].prefix != r.prefix or (new and len(states) >= self.width)
                        or len(r.prefix) + used + cost + max(largest, max(map(len, r.suffixes), default=0)) > self.ctx.n_ctx):
                windows.append(cur)
                cur, states, used, new, cost, largest = [], set(), 0, True, len(r.state), 0
            cur.append(n)
            states.add(r.state)
            used += cost
            largest = max(largest, max(map(len, r.suffixes), default=0))
        return windows + ([cur] if cur else [])

    def _writer(self, slots: dict, out: ScoreBlock, done: set):
        def write(row, j, _seq):
            logp, log_mass = reduce_row(self.ctx.logits(j), slots[row], self._scratch)
            out.logp[row, :len(logp)] = logp
            out.log_mass[row] = log_mass
            done.add(row)

        return write

    @staticmethod
    def _fail(rows, out: ScoreBlock, done: set, err: Exception):
        for row in rows:
            if row not in done:
                out.error[row] = str(err)

    def _reference(self, r: ScoreRequest, row0: int, out: ScoreBlock):
        c = self.ctx
        slots = {row0 + i: s for i, s in enumerate(r.slots)}
        for i, suffix in enumerate(r.suffixes):
            done: set = set()
            try:
                if len(r.prefix) + len(r.state) + len(suffix) > c.n_ctx:
                    raise RuntimeError("sequence exceeds context")
                self.invalidate()
                self._dirty = True
                if self.weights.offload:
                    tokens = r.prefix + r.state + suffix
                    c.run([(token, position, 1, row0 + i if position == len(tokens) - 1 else None)
                           for position, token in enumerate(tokens)], self._writer(slots, out, done))
                else:
                    position = 0
                    for tokens, last in ((r.prefix, False), (r.state, False), (suffix, True)):
                        c.run([(token, position + k, 1, row0 + i if last and k == len(tokens) - 1 else None)
                               for k, token in enumerate(tokens)], self._writer(slots, out, done))
                        position += len(tokens)
            except RuntimeError as e:
                self._fail([row0 + i], out, done, e)
            except BaseException:
                self.invalidate()
                raise
            finally:
                c.rm(1)
        self._dirty = False

    def _ensure_prefix(self, prefix: tuple[int, ...], reserve=0):
        c = self.ctx
        if self._prefix != prefix:
            for sequence in tuple(self._states):
                c.rm(sequence)
            self._states = {}
            hit = self._resident.pop(prefix, None)
            old = self._prefix
            temporary = self.layout.first_branch
            if old is not None and self.prefix_slots > 1:
                c.cp(0, temporary)
            c.rm(0)
            if hit is not None:
                c.cp(hit, 0)
                c.rm(hit)
                self.stats["prefix_hits"] += 1
            else:
                self.stats["prefix_misses"] += 1
            if old is not None and self.prefix_slots > 1:
                free = next((sequence for sequence in self.layout.residents if sequence not in self._resident.values()), None)
                if free is None:
                    victim = next(iter(self._resident))
                    free = self._resident.pop(victim)
                    c.rm(free)
                    self.stats["prefix_evictions"] += 1
                c.cp(temporary, free)
                self._resident[old] = free
                c.rm(temporary)
            self._prefix = prefix if hit is not None else None
        else:
            self.stats["prefix_hits"] += 1
        while self._resident and len(prefix) + reserve + sum(map(len, self._resident)) > c.n_ctx:
            victim = next(iter(self._resident))
            c.rm(self._resident.pop(victim))
            self.stats["prefix_evictions"] += 1
        if self._prefix is None:
            c.run([(token, position, 0, None) for position, token in enumerate(prefix)], None)
            self.stats["prefix_tokens"] += len(prefix)
            self._prefix = prefix

    def _ensure_states(self, prefix: tuple[int, ...], states: list[tuple]) -> dict[tuple, int]:
        c = self.ctx
        seq_of = {st: s for s, st in self._states.items() if st in states}
        for s in [s for s, st in self._states.items() if st not in seq_of]:
            c.rm(s)
            del self._states[s]
        free = [s for s in range(1, self.width + 1) if s not in self._states]
        new = [st for st in dict.fromkeys(states) if st not in seq_of]
        for st, s in zip(new, free):
            c.cp(0, s)
            seq_of[st] = s
        c.run([(t, len(prefix) + k, seq_of[st], None) for st in new for k, t in enumerate(st)], None)
        self._states.update({seq_of[st]: st for st in new})
        return seq_of

    def _shared(self, window: list[tuple[ScoreRequest, int]], out: ScoreBlock):
        c = self.ctx
        done: set = set()
        prefix = window[0][0].prefix
        rows = [row0 + i for r, row0 in window for i in range(len(r.suffixes))]
        slots = {row0 + i: s for r, row0 in window for i, s in enumerate(r.slots)}
        write = self._writer(slots, out, done)
        try:
            if any(len(prefix) + len(r.state) + max(map(len, r.suffixes)) > c.n_ctx for r, _ in window):
                raise RuntimeError("sequence exceeds context")
            self._dirty = True
            states = list(dict.fromkeys(r.state for r, _ in window))
            reserve = sum(map(len, states)) + max(len(suffix) for r, _ in window for suffix in r.suffixes)
            self._ensure_prefix(prefix, reserve)
            seq_of = {state: sequence for sequence, state in self._states.items() if state in states}
            for sequence in list(self._states):
                if self._states[sequence] not in seq_of:
                    c.rm(sequence)
                    del self._states[sequence]
            free = iter(sequence for sequence in range(1, self.width + 1) if sequence not in self._states)
            new = [state for state in states if state not in seq_of]
            self.stats["state_hits"] += len(states) - len(new)
            self.stats["state_misses"] += len(new)
            self.stats["state_tokens"] += sum(map(len, new))
            for state in new:
                seq_of[state] = next(free)
                c.cp(0, seq_of[state])
            budget = c.n_ctx - len(prefix) - sum(map(len, self._resident)) - sum(map(len, states))
            branches = [(r, row0 + i, suffix) for r, row0 in window for i, suffix in enumerate(r.suffixes)]
            first = self.layout.first_branch
            groups, group, used = [], [], 0
            for branch in branches:
                if group and (used + len(branch[2]) > budget or len(group) >= self.branches):
                    groups.append(group)
                    group, used = [], 0
                if len(branch[2]) > budget:
                    raise RuntimeError("sequence exceeds context")
                group.append(branch)
                used += len(branch[2])
            if group:
                groups.append(group)
            if not self.fused:
                c.run([(token, len(prefix) + k, seq_of[state], None) for state in new for k, token in enumerate(state)], None)
                self._states.update({seq_of[state]: state for state in new})
                new = []
            for index, group in enumerate(groups):
                self.stats["branch_groups"] += 1
                self.stats["suffix_tokens"] += sum(len(suffix) for _, _, suffix in group)
                try:
                    for g, (request, _, _) in enumerate(group):
                        c.cp(0 if index == 0 and request.state in new else seq_of[request.state], first + g)
                    entries = []
                    if index == 0:
                        for state in new:
                            sequences = (seq_of[state],) + tuple(first + g for g, (r, _, _) in enumerate(group)
                                                                 if r.state == state)
                            entries.extend((token, len(prefix) + k, sequences, None) for k, token in enumerate(state))
                    entries.extend((token, len(prefix) + len(request.state) + k, first + g,
                                    row if k == len(suffix) - 1 else None)
                                   for g, (request, row, suffix) in enumerate(group)
                                   for k, token in enumerate(suffix))
                    c.run(entries, write)
                    if index == 0:
                        self._states.update({seq_of[state]: state for state in new})
                finally:
                    for g in range(len(group)):
                        c.rm(first + g)
            self._dirty = False
        except RuntimeError as e:
            self.stats["failures"] += len(rows) - len(done)
            self.invalidate()
            self._fail(rows, out, done, e)
        except BaseException:
            self.invalidate()
            raise

    def selftest(self, adapter, texts, questions) -> dict:
        with self.operation():
            n_batch = self.ctx.n_batch
            try:
                return self._selftest(adapter, texts, questions)
            except RuntimeError as error:
                self.shared = False
                return {"D1 shared = reference": {"ok": False, "error": str(error)}}
            finally:
                self.ctx.n_batch = n_batch
                self.invalidate()

    def _selftest(self, adapter, texts, questions):
        def probs(text, qs):
            block = self.score([adapter.request(adapter.state(text), list(qs))])
            if any(block.error) or any(not np.isfinite(block.logp[i, :q.width]).all() for i, q in enumerate(qs)):
                raise RuntimeError("conformance scoring failed: " + repr(block.error))
            return np.where(np.isfinite(block.logp), np.exp(block.logp), 0.)

        self.shared = True
        a = probs(texts[0], questions[:1])
        self.shared = False
        b = probs(texts[0], questions[:1])
        self.shared = True
        checks = {"D1 shared = reference": {"ok": float(np.abs(a - b).max()) < 1e-3,
                                            "max_diff": float(np.abs(a - b).max())}}
        if self.inputs > 1:
            requests = [adapter.request(adapter.state(text), questions[:1]) for text in texts[:2]]
            self.shared = True
            multi = self.score(requests)
            self.shared = False
            reference = self.score(requests)
            error = any(multi.error) or any(reference.error)
            difference = float(np.abs(np.exp(multi.logp) - np.exp(reference.logp)).max())
            checks["D1 shared = reference"].update(multi_input_diff=difference, inputs=2,
                ok=checks["D1 shared = reference"]["ok"] and not error and difference < .15 and
                   bool((multi.logp.argmax(axis=1) == reference.logp.argmax(axis=1)).all()))
            self.shared = True
        first = probs(texts[2], questions[:1])
        self.invalidate()
        fresh = probs(texts[2], questions[:1])
        checks["D3 no leakage"] = {"ok": float(np.abs(first - fresh).max()) < 1e-3,
                                   "max_diff": float(np.abs(first - fresh).max())}
        ref = probs(texts[0], questions)
        n_batch, self.ctx.n_batch = self.ctx.n_batch, 7
        tiny = probs(texts[0], questions)
        self.ctx.n_batch = n_batch
        same_top = bool((ref.argmax(axis=1) == tiny.argmax(axis=1)).all())
        checks["D6 chunking"] = {"ok": same_top and float(np.abs(ref - tiny).max()) < 0.25,
                                 "max_diff": float(np.abs(ref - tiny).max())}
        self.shared = shared_ok(checks)
        return checks

    def close(self):
        with self.ctx.lock:
            if self._finalizer.detach():
                self.handles.close()
