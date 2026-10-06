from __future__ import annotations

import ctypes
import os
import threading
from typing import Sequence

import numpy as np

from .protocols import ScoreBlock, ScoreRequest

_lock = threading.Lock()
_ready = False


def _llama():
    global _ready
    import llama_cpp as L

    with _lock:
        if not _ready:
            if not os.environ.get("SEMANTIC_SWITCH_DEBUG"):
                _llama.silence = L.llama_log_callback(lambda level, text, data: None)
                L.llama_log_set(_llama.silence, ctypes.c_void_p())
            L.llama_backend_init()
            _ready = True
    return L


def _lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.exp(x - m).sum()))


def reduce_row(row: np.ndarray, slots: tuple[tuple[int, ...], ...]) -> tuple[np.ndarray, float]:
    lse_all = _lse(row)
    slot_lp = np.array([_lse(row[list(ids)]) for ids in slots], dtype=np.float64)
    lse_slots = _lse(slot_lp)
    return (slot_lp - lse_slots).astype(np.float32), lse_slots - lse_all


SHARED_CHECKS = ("D1 shared = reference", "D3 no leakage")


def shared_ok(checks: dict) -> bool:
    return all(checks.get(k, {"ok": True})["ok"] for k in SHARED_CHECKS)


def gpu_available() -> bool:
    return bool(_llama().llama_supports_gpu_offload())


class Weights:
    def __init__(self, path: str, n_gpu_layers: int = 0, vocab_only: bool = False):
        L = _llama()
        params = L.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        params.no_host = n_gpu_layers == 0
        params.vocab_only = vocab_only
        self.L = L
        self.ptr = L.llama_model_load_from_file(path.encode(), params)
        if not self.ptr:
            raise RuntimeError(f"cannot load model {path}")
        self.vocab = L.llama_model_get_vocab(self.ptr)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.offload = n_gpu_layers > 0
        self.hybrid = bool(L.llama_model_is_hybrid(self.ptr) or L.llama_model_is_recurrent(self.ptr))

    def tokenize(self, text: str, special: bool = False, add_special: bool = False) -> tuple[int, ...]:
        if self.ptr is None:
            raise RuntimeError("runtime is closed")
        data = text.encode()
        cap = len(data) + 16
        buf = (self.L.llama_token * cap)()
        n = self.L.llama_tokenize(self.vocab, data, len(data), buf, cap, add_special, special)
        if n < 0:
            buf = (self.L.llama_token * -n)()
            n = self.L.llama_tokenize(self.vocab, data, len(data), buf, -n, add_special, special)
        return tuple(buf[:n])

    def detokenize(self, tokens: Sequence[int], special: bool = False) -> str:
        piece = ctypes.create_string_buffer(256)
        out = bytearray()
        for t in tokens:
            n = self.L.llama_token_to_piece(self.vocab, t, piece, 256, 0, special)
            out += piece.raw[:n]
        return out.decode("utf-8", errors="replace")

    def chat(self) -> dict:
        tmpl = self.L.llama_model_chat_template(self.ptr, None)
        text = lambda tok: self.detokenize([tok], special=True) if tok >= 0 else ""
        return {"template": tmpl.decode() if tmpl else None, "bos": text(self.L.llama_vocab_bos(self.vocab)),
                "eos": text(self.L.llama_vocab_eos(self.vocab)),
                "add_bos": bool(self.L.llama_vocab_get_add_bos(self.vocab))}

    def close(self):
        if self.ptr:
            self.L.llama_model_free(self.ptr)
            self.ptr = None


class Context:
    def __init__(self, weights: Weights, *, n_ctx: int, n_batch: int, n_seq_max: int, threads: int | None):
        L = weights.L
        p = L.llama_context_default_params()
        p.n_ctx = n_ctx
        p.n_batch = n_batch
        p.n_ubatch = min(n_batch, 512)
        p.n_seq_max = n_seq_max
        p.n_threads = p.n_threads_batch = threads or os.cpu_count()
        p.kv_unified = True
        p.op_offload = weights.offload
        self.L = L
        self.w = weights
        self.ptr = L.llama_init_from_model(weights.ptr, p)
        if not self.ptr:
            raise RuntimeError("cannot create llama context")
        self.mem = L.llama_get_memory(self.ptr)
        self.batch = L.llama_batch_init(n_batch, 0, 1)
        self.n_ctx, self.n_batch, self.n_seq_max = n_ctx, n_batch, n_seq_max
        self.lock = threading.Lock()

    def rm(self, seq: int):
        if self.mem:
            self.L.llama_memory_seq_rm(self.mem, seq, -1, -1)

    def cp(self, src: int, dst: int):
        self.L.llama_memory_seq_cp(self.mem, src, dst, -1, -1)

    def clear(self):
        if self.mem:
            self.L.llama_memory_clear(self.mem, True)

    def run(self, entries: list, on_output):
        for start in range(0, len(entries), self.n_batch):
            chunk = entries[start:start + self.n_batch]
            b = self.batch
            b.n_tokens = len(chunk)
            for j, (tok, pos, seq, out) in enumerate(chunk):
                b.token[j] = tok
                b.pos[j] = pos
                b.n_seq_id[j] = 1
                b.seq_id[j][0] = seq
                b.logits[j] = out is not None
            rc = self.L.llama_decode(self.ptr, b)
            if rc != 0:
                raise RuntimeError(f"llama decode returned {rc}")
            for j, (_, _, seq, out) in enumerate(chunk):
                if out is not None:
                    on_output(out, j, seq)

    def logits(self, j: int) -> np.ndarray:
        return np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ptr, j), shape=(self.w.n_vocab,))


    def close(self):
        if self.ptr:
            self.L.llama_batch_free(self.batch)
            self.L.llama_free(self.ptr)
            self.ptr = None


class LlamaScorer:
    def __init__(self, path: str, *, n_ctx: int = 8192, n_batch: int = 512, n_seq_max: int = 34,
                 threads: int | None = None, n_gpu_layers: int = 0, shared: bool | None = None):
        self.weights = Weights(path, n_gpu_layers)
        try:
            self.ctx = Context(self.weights, n_ctx=n_ctx, n_batch=n_batch, n_seq_max=n_seq_max, threads=threads)
        except RuntimeError:
            self.weights.close()
            raise
        self.shared = (not self.weights.hybrid) if shared is None else shared
        self.path = path
        self._prefix: tuple | None = None
        self._state: tuple | None = None

    def invalidate(self):
        self._prefix = self._state = None
        self.ctx.clear()

    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]:
        return self.weights.tokenize(text, special)

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.weights.detokenize(tokens)

    def chat(self) -> dict:
        return self.weights.chat()

    def score(self, requests: Sequence[ScoreRequest]) -> ScoreBlock:
        width = max((len(s) for r in requests for s in r.slots), default=1)
        out = ScoreBlock.empty(sum(len(r.suffixes) for r in requests), width)
        with self.ctx.lock:
            if self.ctx.ptr is None:
                raise RuntimeError("runtime is closed")
            row = 0
            for r in requests:
                (self._shared if self.shared else self._reference)(r, row, out)
                row += len(r.suffixes)
        return out

    def _writer(self, r: ScoreRequest, row0: int, out: ScoreBlock, done: set):
        def write(i, j, _seq):
            logp, log_mass = reduce_row(self.ctx.logits(j), r.slots[i])
            out.logp[row0 + i, :len(logp)] = logp
            out.log_mass[row0 + i] = log_mass
            done.add(i)

        return write

    def _fail(self, row0: int, indexes, out: ScoreBlock, done: set, err: Exception):
        for i in indexes:
            if i not in done:
                out.error[row0 + i] = str(err)

    def _reference(self, r: ScoreRequest, row0: int, out: ScoreBlock):
        c = self.ctx
        self._prefix = self._state = None
        for i, suffix in enumerate(r.suffixes):
            done: set = set()
            try:
                if len(r.prefix) + len(r.state) + len(suffix) > c.n_ctx:
                    raise RuntimeError("sequence exceeds context")
                c.clear()
                pos = 0
                for seg, last in ((r.prefix, False), (r.state, False), (suffix, True)):
                    c.run([(t, pos + k, 1, i if last and k == len(seg) - 1 else None) for k, t in enumerate(seg)],
                          self._writer(r, row0, out, done))
                    pos += len(seg)
            except RuntimeError as e:
                self._fail(row0, [i], out, done, e)
            finally:
                c.rm(1)

    def _ensure_prefix(self, prefix: tuple[int, ...]):
        if self._prefix == prefix:
            return
        c = self.ctx
        c.clear()
        self._state = None
        c.run([(t, k, 0, None) for k, t in enumerate(prefix)], None)
        self._prefix = prefix

    def _ensure_state(self, prefix: tuple[int, ...], state: tuple[int, ...]):
        if self._state == state:
            return
        c = self.ctx
        self._state = None
        c.rm(1)
        c.cp(0, 1)
        c.run([(t, len(prefix) + k, 1, None) for k, t in enumerate(state)], None)
        self._state = state

    def _groups(self, r: ScoreRequest, base: int) -> list[list[int]]:
        budget = self.ctx.n_ctx - base
        groups, cur, used = [], [], 0
        for i, s in enumerate(r.suffixes):
            if cur and (used + len(s) > budget or len(cur) >= self.ctx.n_seq_max - 2):
                groups.append(cur)
                cur, used = [], 0
            cur.append(i)
            used += len(s)
        return groups + ([cur] if cur else [])

    def _shared(self, r: ScoreRequest, row0: int, out: ScoreBlock):
        c = self.ctx
        done: set = set()
        base = len(r.prefix) + len(r.state)
        write = self._writer(r, row0, out, done)
        try:
            if base + max(map(len, r.suffixes)) > c.n_ctx:
                raise RuntimeError("sequence exceeds context")
            self._ensure_prefix(r.prefix)
            self._ensure_state(r.prefix, r.state)
            for group in self._groups(r, base):
                try:
                    for g in range(len(group)):
                        c.cp(1, 2 + g)
                    entries = [(t, base + k, 2 + g, i if k == len(r.suffixes[i]) - 1 else None)
                               for g, i in enumerate(group) for k, t in enumerate(r.suffixes[i])]
                    c.run(entries, write)
                finally:
                    for g in range(len(group)):
                        c.rm(2 + g)
        except RuntimeError as e:
            self.invalidate()
            self._fail(row0, range(len(r.suffixes)), out, done, e)

    def selftest(self, adapter, texts, questions) -> dict:
        def probs(text, qs):
            block = self.score([adapter.request(adapter.state(text), list(qs))])
            return np.where(np.isfinite(block.logp), np.exp(block.logp), 0.0)

        self.shared = True
        a = probs(texts[0], questions[:1])
        self.shared = False
        b = probs(texts[0], questions[:1])
        self.shared = True
        checks = {"D1 shared = reference": {"ok": float(np.abs(a - b).max()) < 1e-3,
                                            "max_diff": float(np.abs(a - b).max())}}
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
            self.ctx.close()
            self.weights.close()
