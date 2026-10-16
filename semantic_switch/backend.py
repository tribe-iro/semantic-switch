from __future__ import annotations

import ctypes
import importlib.util
import math
import mmap
import os
import re
import struct
import sys
import threading
import warnings
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .errors import ConfigError
from .guidance import Guidance
from .protocols import MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS, MAX_QUESTION_TOKENS, ScoreBlock, ScoreRequest, Tokens
from .resident import Resources

_lock = threading.Lock()
_init = threading.Lock()
_ready = False
_captured: list | None = None
BUFFER = re.compile(r"(\S+) +(model|output|KV|compute|LoRA) buffer size = +([\d.]+) MiB")
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
_NUMPY = {0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "?", 10: "<u8", 11: "<i8", 12: "<f8"}
GGUF_ALIGNMENT = 32
GGML_BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110),
               12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2)}
LORA_TYPES = (0, 1, 8, 30)
ADDS_BOS = {"model": ("llama", "bert"), "pre": ("llama3", "llama-v3", "llama-bpe")}


@dataclass(frozen=True, slots=True)
class Tensor:
    shape: tuple[int, ...]
    type: int
    span: int

    @property
    def nbytes(self) -> int | None:
        if self.type not in GGML_BLOCKS:
            return None
        block, size = GGML_BLOCKS[self.type]
        if not self.shape or self.shape[0] % block:
            raise ValueError(f"tensor shape {self.shape} does not tile ggml type {self.type} blocks of {block}")
        return math.prod(self.shape) // block * size

    @property
    def bytes(self) -> int:
        known = self.nbytes
        return self.span if known is None else known


@dataclass(frozen=True, slots=True)
class Strings:
    count: int
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class Gguf:
    path: str
    size: int
    meta: Mapping
    tensors: Mapping[str, Tensor]
    vocab: int
    bos: str
    eos: str

    @property
    def architecture(self) -> str | None:
        return self.meta.get("general.architecture")

    def chat(self) -> dict:
        meta = self.meta
        default = meta.get("tokenizer.ggml.model") in ADDS_BOS["model"] or meta.get("tokenizer.ggml.pre") in ADDS_BOS["pre"]
        return {"template": meta.get("tokenizer.chat_template"), "bos": self.bos, "eos": self.eos,
                "add_bos": bool(meta.get("tokenizer.ggml.add_bos_token", default))}


def _parse_gguf(path: str, data, size: int) -> Gguf:
    offset = 0

    def read(code):
        nonlocal offset
        found = struct.unpack_from("<" + code, data, offset)[0]
        offset += struct.calcsize(code)
        return found

    def string():
        nonlocal offset
        length = read("Q")
        found = bytes(data[offset:offset + length]).decode(errors="replace")
        offset += length
        return found

    def value(kind):
        nonlocal offset
        if kind == 8:
            return string()
        if kind != 9:
            return read(_GGUF[kind])
        item, count = read("I"), read("Q")
        if item in _NUMPY:
            found = np.frombuffer(data, dtype=_NUMPY[item], count=count, offset=offset).copy()
            found.setflags(write=False)
            offset += found.nbytes
            return found
        if item == 8:
            start = offset
            for _ in range(count):
                offset += 8 + struct.unpack_from("<Q", data, offset)[0]
            return Strings(count, start)
        return tuple(value(item) for _ in range(count))

    if bytes(data[:4]) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF v2+ file")
    offset = 4
    if read("I") < 2:
        raise ValueError(f"{path} is not a GGUF v2+ file")
    count = read("Q")
    meta = {}
    for _ in range(read("Q")):
        key = string()
        meta[key] = value(read("I"))
    infos = []
    for _ in range(count):
        name = string()
        shape = tuple(read("Q") for _ in range(read("I")))
        infos.append((name, shape, read("I"), read("Q")))
    alignment = meta.get("general.alignment", GGUF_ALIGNMENT)
    start = size - -(-offset // alignment) * alignment
    ends = sorted({where for *_, where in infos} | {start})
    following = dict(zip(ends, ends[1:]))
    tensors = {}
    for name, shape, kind, where in infos:
        tensors[name] = tensor = Tensor(shape, kind, following[where] - where if where < start else -1)
        if tensor.span < 0 or tensor.bytes > tensor.span:
            raise ValueError(f"{path}: tensor {name} data exceeds its GGUF layout")
    tokens = meta.get("tokenizer.ggml.tokens")
    special = {}
    if isinstance(tokens, Strings):
        wanted = {meta[f"tokenizer.ggml.{name}_token_id"]: name for name in ("bos", "eos")
                  if isinstance(meta.get(f"tokenizer.ggml.{name}_token_id"), int) and 0 <= meta[f"tokenizer.ggml.{name}_token_id"] < tokens.count}
        offset = tokens.offset
        for index in range(max(wanted, default=-1) + 1):
            text = string()
            if index in wanted:
                special[wanted[index]] = text
    return Gguf(path, size, MappingProxyType(meta), MappingProxyType(tensors), tokens.count if isinstance(tokens, Strings) else 0,
                special.get("bos", ""), special.get("eos", ""))


_GGUFS: OrderedDict = OrderedDict()
_GGUF_LOCK = threading.Lock()


def gguf(path: str) -> Gguf:
    with open(path, "rb") as f:
        stat = os.fstat(f.fileno())
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with _GGUF_LOCK:
            found = _GGUFS.get(key)
            if found is not None:
                _GGUFS.move_to_end(key)
                return found
        if stat.st_size < 24:
            raise ValueError(f"{path} is not a GGUF v2+ file")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            try:
                found = _parse_gguf(str(path), data, stat.st_size)
            except struct.error:
                raise ValueError(f"{path}: truncated GGUF header") from None
    with _GGUF_LOCK:
        _GGUFS[key] = found
        while len(_GGUFS) > 16:
            _GGUFS.popitem(last=False)
    return found


LORA_SUFFIXES = (".lora_a", ".lora_b")


def lora_layout(adapter: str, base: str) -> dict:
    found, reference = gguf(adapter), gguf(base)
    meta, tensors, base_tensors = found.meta, found.tensors, reference.tensors
    if meta.get("general.type") != "adapter" or meta.get("adapter.type") != "lora":
        raise ValueError(f"{adapter} is not a LoRA adapter GGUF")
    if found.architecture != reference.architecture:
        raise ValueError(f"{adapter}: adapter architecture {found.architecture} differs from base {reference.architecture}")
    alpha = meta.get("adapter.lora.alpha")
    stems = {name[:-len(suffix)] for name in tensors for suffix in LORA_SUFFIXES if name.endswith(suffix)}
    if not stems or len(tensors) != 2 * len(stems) or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not alpha > 0:
        raise ValueError(f"{adapter}: needs paired lora_a/lora_b tensors and a positive adapter.lora.alpha")
    unsupported = sorted(name for name, tensor in tensors.items() if tensor.type not in LORA_TYPES)
    if unsupported:
        raise ValueError(f"{adapter}: tensors {unsupported[:3]} have ggml types outside the supported adapter types {list(LORA_TYPES)}")
    ranks = set()
    for stem in sorted(stems):
        a, b = (getattr(tensors.get(stem + suffix), "shape", None) for suffix in LORA_SUFFIXES)
        weight = getattr(base_tensors.get(stem), "shape", None)
        if weight is None:
            raise ValueError(f"{adapter}: target {stem} is absent from the base")
        if a is None or b is None or len(a) != 2 or len(b) != 2 or len(weight) < 2 or a[0] != weight[0] or b[1] != weight[1] or a[1] != b[0]:
            raise ValueError(f"{adapter}: {stem} lora shapes {a}/{b} do not match base weight {weight}")
        ranks.add(a[1])
    if len(ranks) != 1:
        raise ValueError(f"{adapter}: mixed adapter ranks {sorted(ranks)}")
    return {"architecture": found.architecture, "rank": ranks.pop(), "alpha": float(alpha), "targets": sorted(stems),
            "tensors": {name: tensor.nbytes for name, tensor in tensors.items()}, "file_bytes": found.size,
            "header": found.size - sum(t.span for t in tensors.values())}


INPUT_TENSORS = ("token_embd", "per_layer_token_embd")
OUTPUT_TENSORS = ("output", "output_norm")
BUFFER_RESOLUTION = 5243
LORA_ALLOCATION = {"revision": 1, "build": "llama-cpp-python 0.3.36", "alignment": 256, "buffer_slack": 8192, "tensor_overhead": 512, "contexts": 2,
                   "placement": "tensor-by-tensor beside its base weight's layer device; repacked or host base weights keep the adapter on the host",
                   "transient": "one read buffer of the largest tensor"}


def host_buffer(name: str) -> bool:
    return name.startswith("CPU") or name.endswith("_Host")


@lru_cache(maxsize=1)
def offload_device() -> tuple[str, bool] | None:
    from pathlib import Path
    L = _llama()
    ggml = ctypes.CDLL(str(Path(L.__file__).parent / "lib" / "libggml.so"))
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes, ggml.ggml_backend_dev_get.restype = [ctypes.c_size_t], ctypes.c_void_p
    ggml.ggml_backend_dev_type.argtypes, ggml.ggml_backend_dev_type.restype = [ctypes.c_void_p], ctypes.c_int
    ggml.ggml_backend_dev_name.argtypes, ggml.ggml_backend_dev_name.restype = [ctypes.c_void_p], ctypes.c_char_p
    found = {1: [], 2: []}
    for index in range(ggml.ggml_backend_dev_count()):
        device = ggml.ggml_backend_dev_get(index)
        found.setdefault(ggml.ggml_backend_dev_type(device), []).append(ggml.ggml_backend_dev_name(device).decode())
    chosen, unified = (found[1], False) if found[1] else (found[2], True)
    if len(chosen) > 1:
        raise ConfigError(f"offload devices {chosen} split layers by free memory, which the pinned placement profile does not describe; "
                          "admission cannot certify a bounded GPU load: expose one device (e.g. GGML_VK_VISIBLE_DEVICES=0) or use device='cpu'")
    return (chosen[0], unified) if chosen else None


@dataclass(frozen=True, slots=True)
class Allocation:
    resources: Resources
    overhead: int
    unified: bool

    def escapes(self, observed: dict[str, int]) -> str | None:
        host, devices = self.overhead, {}
        for key, value in observed.items():
            name = key.rsplit(" ", 1)[0]
            if host_buffer(name):
                host += value + BUFFER_RESOLUTION
            else:
                devices[name] = devices.get(name, 0) + value + BUFFER_RESOLUTION
        physical = host + (sum(devices.values()) if self.unified else 0)
        if observed and self.resources.covers(Resources(physical, devices)):
            return None
        return (f"adapter allocation {observed or 'was not reported by the native build'} exceeds or escapes its estimate "
                f"{self.resources.record()} (overhead {self.overhead}, resolution {BUFFER_RESOLUTION} bytes per buffer); "
                f"the pinned allocation profile revision {LORA_ALLOCATION['revision']} does not describe this build")


_ADVISED: set = set()


def advise(kind: str, message: str):
    key = kind, getattr(sys.modules.get("llama_cpp"), "__version__", None)
    if key not in _ADVISED:
        _ADVISED.add(key)
        warnings.warn(f"{message} (advisory outside an explicit model job; reported once per native build)", RuntimeWarning, stacklevel=3)


@dataclass(frozen=True, slots=True)
class Placement:
    layers: int
    gpu_layers: int
    device: str | None = None
    unified: bool = False

    @classmethod
    def of(cls, model: Gguf, gpu_layers: int) -> Placement:
        device, unified = (offload_device() if gpu_layers else None) or (None, False)
        return cls(int(model.meta[f"{model.architecture}.block_count"]), gpu_layers, device, unified)

    def where(self, name: str) -> str:
        if self.device is None:
            return "host"
        start, parts = max(self.layers + 1 - self.gpu_layers, 0), name.split(".")
        if parts[0] == "blk" and len(parts) > 2 and parts[1].isdigit():
            return "device" if int(parts[1]) >= start else "host"
        if parts[0] in OUTPUT_TENSORS:
            return "device" if self.layers >= start else "host"
        return "host" if parts[0] in INPUT_TENSORS else "unknown"

    def charge(self, host: int, device: int = 0) -> Resources:
        return Resources(host + (device if self.unified else 0), {self.device: device} if device else {})

    def weights(self, size: int, tensors: dict) -> Resources:
        placed = {name: self.where(name) for name in tensors}
        device = sum(tensors[name].bytes for name, where in placed.items() if where != "host")
        if "output.weight" not in tensors and "token_embd.weight" in tensors and self.where("output.weight") == "device":
            device += tensors["token_embd.weight"].bytes
        return self.charge(size - sum(tensors[name].bytes for name, where in placed.items() if where == "device"), device)

    def kv(self, size: int) -> Resources:
        return self.charge(0, size) if self.device else Resources(size)

    def lora(self, adapter) -> Allocation:
        profile, buffers = LORA_ALLOCATION, {}
        for name, size in adapter.tensors:
            where = self.where(name.rsplit(".", 1)[0])
            if where == "unknown":
                raise ConfigError(f"adapter {adapter.sha256[:12]}: target {name} has no pinned placement on {self.device}; "
                                  "a bounded GPU load cannot be certified for it")
            buffers[where] = buffers.get(where, 0) + -(-size // profile["alignment"]) * profile["alignment"]
        buffers = {where: size + profile["buffer_slack"] for where, size in buffers.items()}
        overhead = adapter.header + profile["contexts"] * len(adapter.tensors) * profile["tensor_overhead"] + max((size for _, size in adapter.tensors), default=0)
        return Allocation(self.charge(overhead + buffers.get("host", 0), buffers.get("device", 0)), overhead, self.unified)


def lora_abi(L) -> dict:
    handle = L.llama_adapter_lora_p_ctypes
    expected = {"llama_adapter_lora_init": ([L.llama_model_p_ctypes, ctypes.c_char_p], handle),
                "llama_adapter_lora_free": ([handle], None),
                "llama_set_adapters_lora": ([L.llama_context_p_ctypes, ctypes.POINTER(handle), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)], ctypes.c_int32)}
    for name, (arguments, result) in expected.items():
        function = getattr(L, name, None)
        if function is None or list(getattr(function, "argtypes", None) or ()) != arguments or getattr(function, "restype", None) is not result:
            raise ConfigError(f"this llama.cpp build does not expose the runtime adapter ABI {name}({', '.join(t.__name__ for t in arguments)}) -> "
                              f"{getattr(result, '__name__', result)}; runtime-adapter bindings are unsupported on it")
    return {"version": getattr(L, "__version__", None), "functions": sorted(expected)}


class Lora:
    def __init__(self, weights: Weights, path: str, key: str):
        L = weights.L
        self.abi = lora_abi(L)
        self.L, self.path, self.key = L, path, key
        with _capture() as lines:
            self.ptr = L.llama_adapter_lora_init(weights.ptr, path.encode())
        if not self.ptr:
            raise RuntimeError(f"cannot load adapter {path}")
        self.buffers = _buffers(lines)

    def close(self):
        ptr, self.ptr = self.ptr, None
        if ptr:
            self.L.llama_adapter_lora_free(ptr)


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


def gpu_available() -> bool:
    return bool(_llama().llama_supports_gpu_offload())


class Weights:
    def __init__(self, path: str, n_gpu_layers: int = 0, repack: bool = True):
        L = _llama()
        params = L.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        params.no_host = n_gpu_layers == 0
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
        return Tokens.from_array(np.ctypeslib.as_array(buf)[:n])

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


@dataclass(frozen=True, slots=True)
class Capacity:
    n_ctx: int
    n_batch: int
    prefix_slots: int
    branch = 2
    outputs_per_decode = 1

    @property
    def n_ctx_seq(self):
        return ((self.n_ctx + 255) // 256) * 256

    @property
    def total_ctx(self):
        return self.n_ctx_seq * self.n_seq_max

    @property
    def n_ubatch(self) -> int:
        return min(self.n_batch, 512)

    @property
    def n_seq_max(self) -> int:
        return self.prefix_slots + 2

    @property
    def n_outputs_max(self) -> int:
        return self.n_seq_max

    @property
    def residents(self) -> range:
        return range(self.branch + 1, self.n_seq_max)


def ceiling(max_input_tokens: int, max_prefix_tokens: int) -> int:
    return max_input_tokens + MAX_QUESTION_TOKENS + max_prefix_tokens


def capacity(n_ctx: int | None, n_batch: int, prefix_slots: int, *, max_input_tokens: int = MAX_INPUT_TOKENS, max_prefix_tokens: int = MAX_PREFIX_TOKENS) -> Capacity:
    for name, value in (("n_ctx", n_ctx), ("n_batch", n_batch), ("prefix_slots", prefix_slots), ("max_input_tokens", max_input_tokens),
                        ("max_prefix_tokens", max_prefix_tokens)):
        validate_native_integer(name, value, optional=name == "n_ctx")
    useful = ceiling(max_input_tokens, max_prefix_tokens)
    if n_ctx is not None and n_ctx > useful:
        raise ValueError(f"n_ctx {n_ctx} exceeds the useful request window {useful} = max_input_tokens {max_input_tokens} + question suffix cap "
                         f"{MAX_QUESTION_TOKENS} + max_prefix_tokens {max_prefix_tokens}; omit n_ctx or raise those budgets explicitly")
    out = Capacity(useful if n_ctx is None else n_ctx, n_batch, prefix_slots)
    if out.n_seq_max > MAX_SEQUENCES:
        raise ValueError(f"{out.n_seq_max} sequence slots exceed the native limit of {MAX_SEQUENCES}")
    if out.total_ctx > (1 << 32) - 1:
        raise ValueError("total native context exceeds native integer limit")
    if min(out.n_batch, 512) < out.n_seq_max:
        raise ValueError(f"native capacity requires n_batch >= {out.n_seq_max} for {out.n_seq_max} sequence slots; microbatch limit is 512")
    return out


class Writer:
    def __init__(self, batch, size: int):
        self.batch, self.size, self.n, self.outputs = batch, size, 0, []
        self.token = np.ctypeslib.as_array(batch.token, shape=(size,))
        self.pos = np.ctypeslib.as_array(batch.pos, shape=(size,))
        self.flags = np.ctypeslib.as_array(batch.logits, shape=(size,))
        np.ctypeslib.as_array(batch.n_seq_id, shape=(size,))[:] = 1
        self.member = np.full(size, -1, dtype=np.int64)
        self.steps = np.arange(size, dtype=np.int32)

    def fill(self, tokens: Tokens, start: int, take: int, position: int, sequence: int):
        n, end = self.n, self.n + take
        self.token[n:end] = tokens.array[start:start + take]
        np.add(self.steps[:take], position, out=self.pos[n:end])
        self.flags[n:end] = 0
        stale = np.flatnonzero(self.member[n:end] != sequence)
        if len(stale):
            rows = self.batch.seq_id
            for j in stale.tolist():
                rows[n + j][0] = sequence
            self.member[n:end] = sequence
        self.n = end

    def output(self, key):
        self.flags[self.n - 1] = 1
        self.outputs.append((key, self.n - 1))


class Context:
    def __init__(self, weights: Weights, cap: Capacity, threads: int | None):
        validate_native_integer("threads", threads, optional=True)
        L = weights.L
        p = L.llama_context_default_params()
        p.n_ctx = cap.total_ctx
        p.n_batch = cap.n_batch
        p.n_ubatch = cap.n_ubatch
        p.n_seq_max = cap.n_seq_max
        p.n_outputs_max = cap.n_outputs_max
        p.n_threads = p.n_threads_batch = threads or os.cpu_count()
        p.swa_full = True
        p.kv_unified = False
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
        self.writer = Writer(self.handles.batch, cap.n_batch)
        self.n_ctx, self.n_batch, self.n_seq_max = L.llama_n_ctx(self.ptr), L.llama_n_batch(self.ptr), L.llama_n_seq_max(self.ptr)
        self.n_ctx_seq = L.llama_n_ctx_seq(self.ptr)
        self.sequence_limit = min(cap.n_ctx, self.n_ctx_seq)
        self.n_outputs_max, self.outputs_per_decode = p.n_outputs_max, cap.outputs_per_decode
        self.lock = threading.RLock()
        self.submissions = 0
        self.submitted_tokens = 0
        self.requested = {"n_ctx": cap.total_ctx, "n_ctx_seq": cap.n_ctx, "n_batch": cap.n_batch, "n_seq_max": cap.n_seq_max}
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

    def decode(self, spans, on_output=None):
        self._live()
        writer = self.writer
        writer.n, writer.outputs = 0, []
        limit = min(self.n_batch, writer.size)
        for tokens, position, sequence, output in spans:
            tokens, start = Tokens(tokens), 0
            total = len(tokens)
            while start < total:
                if writer.n == limit or (output is not None and len(writer.outputs) >= self.outputs_per_decode and total - start <= limit - writer.n):
                    self._submit(on_output)
                take = min(total - start, limit - writer.n)
                writer.fill(tokens, start, take, position + start, sequence)
                start += take
            if output is not None and total:
                writer.output(output)
        if writer.n:
            self._submit(on_output)

    def _submit(self, on_output):
        writer, batch = self.writer, self.handles.batch
        outputs, count = writer.outputs, writer.n
        writer.n, writer.outputs = 0, []
        batch.n_tokens = count
        self.submissions += 1
        self.submitted_tokens += count
        rc = self.L.llama_decode(self.ptr, batch)
        if rc != 0:
            raise RuntimeError(f"llama decode returned {rc}")
        for key, j in outputs:
            on_output(key, j)

    def logits(self, j: int) -> np.ndarray:
        self._live()
        return np.ctypeslib.as_array(self.L.llama_get_logits_ith(self.ptr, j), shape=(self.w.n_vocab,))


SCHEDULING = "resident-prefix-streams@3"
_UNSET = object()


@dataclass(frozen=True, slots=True)
class CacheKey:
    binding: str | None
    tokens: Tokens

    def __len__(self) -> int:
        return len(self.tokens)


class LlamaScorer:
    def __init__(self, path: str, *, n_ctx: int | None = None, n_batch: int = 512, inputs: int = 1,
                 threads: int | None = None, n_gpu_layers: int = 0, shared: bool | None = None,
                 prefix_slots: int = 4, repack: bool = True, max_input_tokens: int = MAX_INPUT_TOKENS, max_prefix_tokens: int = MAX_PREFIX_TOKENS):
        validate_native_integer("threads", threads, optional=True)
        validate_native_integer("inputs", inputs)
        self.layout = capacity(n_ctx, n_batch, prefix_slots, max_input_tokens=max_input_tokens, max_prefix_tokens=max_prefix_tokens)
        self.weights = Weights(path, n_gpu_layers, repack=repack)
        self.handles = self.weights.handles
        self.inputs = inputs
        self.prefix_slots = prefix_slots
        self._resident: dict[CacheKey, int] = {}
        self._dirty = False
        try:
            self.ctx = Context(self.weights, self.layout, threads)
        except BaseException:
            self.handles.close()
            raise
        self._finalizer = weakref.finalize(self, _collect, self.handles)
        self.shared = True if shared is None else shared
        self.path = path
        self.stats = {key: 0 for key in ("prefix_hits", "prefix_misses", "state_hits", "state_misses", "state_tokens", "suffix_tokens", "suffix_rows",
                                       "failures", "prefix_tokens", "prefix_evictions", "adapter_switches")}
        self._scratch = np.empty(self.weights.n_vocab, dtype=np.float64)
        self._prefix: CacheKey | None = None
        self._state: Tokens | None = None
        self._active, self.namespace, self._uses = None, None, {}

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
            self._state = None
            self.ctx.clear()
            self._dirty = False

    def capacity(self) -> dict:
        with self.ctx.lock:
            self._live()
            c = self.ctx
            return {"n_ctx": c.n_ctx, "n_ctx_seq": c.n_ctx_seq, "sequence_limit": c.sequence_limit, "n_batch": c.n_batch, "n_seq_max": c.n_seq_max, "prefix_slots": self.prefix_slots,
                    "scheduling": SCHEDULING, **c.realized, **self.weights.realized,
                    "buffers": {**self.weights.buffers, **c.buffers}, "shared": self.shared}

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
        return gguf(self.path).chat()

    @contextmanager
    def operation(self):
        with self.ctx.lock:
            self._live()
            yield

    def _activate(self, lora: Lora | None):
        if lora is self._active:
            return
        L, previous, self._active = self.ctx.L, self._active, _UNSET
        if lora is None:
            rc = L.llama_set_adapters_lora(self.ctx.ptr, None, 0, None)
        else:
            if not lora.ptr:
                raise RuntimeError(f"adapter {lora.key} was released")
            rc = L.llama_set_adapters_lora(self.ctx.ptr, (L.llama_adapter_lora_p_ctypes * 1)(lora.ptr), 1, (ctypes.c_float * 1)(1.0))
        if rc != 0:
            raise RuntimeError(f"llama_set_adapters_lora returned {rc}")
        self._active = lora
        self.stats["adapter_switches"] += previous is not _UNSET

    def _bind(self, lora: Lora | None, binding: str | None):
        namespace = binding if binding is not None else None if lora is None else lora.key
        used = None if lora is None else lora.key
        if self._uses.setdefault(namespace, used) != used:
            raise RuntimeError(f"binding namespace {namespace} already names adapter {self._uses[namespace]}, not {used}")
        self._activate(lora)
        self.namespace = namespace

    def forget(self, lora: Lora):
        with self.operation():
            if self._active is lora or self._active is _UNSET:
                self._activate(None)
            stale = {namespace for namespace, used in self._uses.items() if used == lora.key}
            for key in [key for key in self._resident if key.binding in stale]:
                self.ctx.rm(self._resident.pop(key))
            if self._prefix is not None and self._prefix.binding in stale:
                for sequence in (0, 1):
                    self.ctx.rm(sequence)
                self._prefix, self._state = None, None
            for namespace in stale:
                del self._uses[namespace]

    def score(self, requests: Sequence[ScoreRequest], *, reference: bool = False, lora: Lora | None = None, binding: str | None = None) -> ScoreBlock:
        width = max((len(s) for r in requests for s in r.slots), default=1)
        out = ScoreBlock.empty(sum(len(r.suffixes) for r in requests), width)
        with self.operation():
            live, row = [], 0
            for request in requests:
                if request.suffixes:
                    if len(request.prefix) + len(request.state) + max(map(len, request.suffixes)) > self.ctx.sequence_limit:
                        self._fail(range(row, row + len(request.suffixes)), out, set(), RuntimeError('sequence exceeds per-sequence context'))
                    else:
                        live.append((request, row))
                row += len(request.suffixes)
            if not live:
                return out
            if self._dirty:
                self.invalidate()
            try:
                self._bind(lora, binding)
            except BaseException:
                self.invalidate()
                raise
            run = self._shared if self.shared and not reference else self._reference
            for request, row in live:
                run(request, row, out)
        return out

    def _emit(self, out: ScoreBlock, done: set):
        def write(key, j):
            row, slots = key
            logp, log_mass = reduce_row(self.ctx.logits(j), slots, self._scratch)
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
        for i, (suffix, slots) in enumerate(zip(r.suffixes, r.slots)):
            done: set = set()
            try:
                self.invalidate()
                self._dirty = True
                spans = ((r.prefix, 0, 1, None), (r.state, len(r.prefix), 1, None), (suffix, len(r.prefix) + len(r.state), 1, (row0 + i, slots)))
                write = self._emit(out, done)
                if self.weights.offload:
                    c.decode(spans, write)
                else:
                    for span in spans:
                        c.decode((span,), write)
            except RuntimeError as e:
                self._fail([row0 + i], out, done, e)
            except BaseException:
                self.invalidate()
                raise
            finally:
                c.rm(1)
        self._dirty = False

    def _ensure_prefix(self, prefix: Tokens):
        c = self.ctx
        key = CacheKey(self.namespace, prefix)
        if self._prefix != key:
            c.rm(1)
            self._state = None
            hit = self._resident.pop(key, None)
            old = self._prefix
            temporary = self.layout.branch
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
            self._prefix = key if hit is not None else None
        else:
            self.stats["prefix_hits"] += 1
        if self._prefix is None:
            c.decode(((prefix, 0, 0, None),))
            self.stats["prefix_tokens"] += len(prefix)
            self._prefix = key

    def _ensure_state(self, prefix: Tokens, state: Tokens) -> int:
        c = self.ctx
        if self._state is not None and self._state == state:
            self.stats["state_hits"] += 1
            return 1
        c.rm(1)
        self._state = None
        c.cp(0, 1)
        c.decode(((state, len(prefix), 1, None),))
        self._state = state
        self.stats["state_misses"] += 1
        self.stats["state_tokens"] += len(state)
        return 1

    def _shared(self, r: ScoreRequest, row0: int, out: ScoreBlock):
        c = self.ctx
        done: set = set()
        rows = range(row0, row0 + len(r.suffixes))
        try:
            self._dirty = True
            self._ensure_prefix(r.prefix)
            sequence = self._ensure_state(r.prefix, r.state)
            branch, base, write = self.layout.branch, len(r.prefix) + len(r.state), self._emit(out, done)
            for row, suffix, slots in zip(rows, r.suffixes, r.slots):
                self.stats["suffix_tokens"] += len(suffix)
                self.stats["suffix_rows"] += 1
                try:
                    c.cp(sequence, branch)
                    c.decode(((suffix, base, branch, (row, slots)),), write)
                finally:
                    c.rm(branch)
            self._dirty = False
        except RuntimeError as e:
            self.stats["failures"] += len(rows) - len(done)
            self.invalidate()
            self._fail(rows, out, done, e)
        except BaseException:
            self.invalidate()
            raise

    def selftest(self, adapter, texts, questions, *, lora: Lora | None = None, binding: str | None = None) -> dict:
        with self.operation():
            n_batch, shared = self.ctx.n_batch, self.shared
            try:
                return self._selftest(adapter, texts, questions, {"lora": lora, "binding": binding})
            except RuntimeError as error:
                return {"D1 shared = reference": {"ok": False, "error": str(error)}}
            finally:
                self.ctx.n_batch, self.shared = n_batch, shared
                self.invalidate()

    def _selftest(self, adapter, texts, questions, bound):
        def probs(text, qs):
            block = self.score([adapter.request(adapter.state(text), list(qs))], **bound)
            if any(block.error) or any(not np.isfinite(block.logp[i, :q.width]).all() for i, q in enumerate(qs)):
                raise RuntimeError("conformance scoring failed: " + repr(block.error))
            return np.column_stack((np.where(np.isfinite(block.logp), np.exp(block.logp), 0.), np.exp(block.log_mass)))

        self.shared = True
        a = probs(texts[0], questions[:1])
        self.shared = False
        b = probs(texts[0], questions[:1])
        self.shared = True
        checks = {"D1 shared = reference": {"ok": bool(np.array_equal(a, b)),
                                            "max_diff": float(np.abs(a - b).max()), "includes_mass": True}}
        if self.inputs > 1:
            requests = [adapter.request(adapter.state(text), questions[:1]) for text in texts[:2]]
            self.shared = True
            multi = self.score(requests, **bound)
            self.shared = False
            reference = self.score(requests, **bound)
            error = any(multi.error) or any(reference.error)
            difference = max(float(np.abs(np.exp(multi.logp) - np.exp(reference.logp)).max()),
                             float(np.abs(np.exp(multi.log_mass) - np.exp(reference.log_mass)).max()))
            checks["D1 shared = reference"].update(multi_input_diff=difference, inputs=2,
                ok=checks["D1 shared = reference"]["ok"] and not error and difference == 0 and
                   bool(np.array_equal(multi.logp, reference.logp) and np.array_equal(multi.log_mass, reference.log_mass)))
            self.shared = True
        first = probs(texts[2], questions[:1])
        self.invalidate()
        fresh = probs(texts[2], questions[:1])
        checks["D3 no leakage"] = {"ok": bool(np.array_equal(first, fresh)),
                                   "max_diff": float(np.abs(first - fresh).max())}
        def observation(text, qs):
            block = self.score([adapter.request(adapter.state(text), list(qs))], **bound)
            if any(block.error):
                raise RuntimeError('residency conformance scoring failed: ' + repr(block.error))
            return np.column_stack((block.logp, block.log_mass))
        self.invalidate()
        target = observation(texts[1], questions)
        repeat = observation(texts[1], questions)
        for i in range(self.prefix_slots + 2):
            guided = [replace(q, guidance=Guidance(f'Use the supplied evidence. Residency probe {i}.')) for q in questions[:1]]
            observation(texts[i % len(texts)], guided)
        after = observation(texts[1], questions)
        delta = np.zeros_like(target)
        np.subtract(target, after, out=delta, where=np.isfinite(target) & np.isfinite(after))
        checks['D7 residency'] = {'ok': bool(np.array_equal(target, repeat) and np.array_equal(target, after)),
                                  'max_diff': float(np.abs(delta).max()), 'includes_mass': True}
        ref = probs(texts[0], questions)
        n_batch, self.ctx.n_batch = self.ctx.n_batch, 7
        tiny = probs(texts[0], questions)
        tiny_repeat = probs(texts[0], questions)
        self.ctx.n_batch = n_batch
        restored = probs(texts[0], questions)
        checks["D6 chunking"] = {"ok": bool(np.array_equal(tiny, tiny_repeat) and np.array_equal(ref, restored)),
                                 "scope": "repeatability within each execution shape; cross-shape probabilities are diagnostic",
                                 "shapes": [n_batch, 7], "max_diff": float(np.abs(ref - tiny).max()),
                                 "argmax_changes": int(np.sum(np.argmax(ref[:, :-1], axis=1) != np.argmax(tiny[:, :-1], axis=1))),
                                 "repeat_diff": float(np.abs(tiny - tiny_repeat).max()), "restored_diff": float(np.abs(ref - restored).max())}
        return checks

    def close(self):
        with self.ctx.lock:
            if self._finalizer.detach():
                self.handles.close()
