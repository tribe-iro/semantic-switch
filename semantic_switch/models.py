from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .errors import ConfigError, suggest
from .protocols import MAX_INPUT_TOKENS, MAX_PREFIX_TOKENS
from .resident import Resources

if TYPE_CHECKING:
    from .artifacts import Provenance

DEFAULT_JUDGE = "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf"
PRECISE_JUDGE = "google/gemma-4-E4B-it-qat-q4_0-gguf:gemma-4-E4B_q4_0-it.gguf"
ADAPTERS = ("chat", "qwen3")
DEVICES = ("auto", "cpu", "gpu")


@dataclass(frozen=True)
class Model:
    source: str
    adapter: str = "chat"
    device: str = "auto"
    n_ctx: int | None = None
    n_batch: int = 512
    threads: int | None = None
    max_input_tokens: int = MAX_INPUT_TOKENS
    max_prefix_tokens: int = MAX_PREFIX_TOKENS
    prefix_slots: int = 4
    quality_profile: str | None = None
    repack: bool = True
    lora: str | None = None
    max_adapters: int = 4
    representation: str | None = None
    provenance: Provenance | None = field(default=None, repr=False)

    def __post_init__(self):
        from .backend import validate_native_integer
        for name, value, allowed in (("adapter", self.adapter, ADAPTERS), ("device", self.device, DEVICES)):
            if value not in allowed:
                raise ConfigError(f"unknown {name} {value!r}{suggest(value, allowed)}")
        if not isinstance(self.repack, bool):
            raise ConfigError("repack must be boolean")
        if self.lora is not None and (not isinstance(self.lora, str) or not self.lora.strip()):
            raise ConfigError("lora must name an adapter GGUF path or repo:file")
        specialized = getattr(self.provenance, "kind", None) == "specialization"
        derived = "runtime-adapter" if self.lora else "merged" if specialized else "base"
        if self.representation is None:
            object.__setattr__(self, "representation", derived)
        elif self.representation != derived and not (self.representation == "merged" and self.provenance is None and not self.lora):
            raise ConfigError(f"representation {self.representation!r} contradicts this model: it is {derived}; "
                              "a runtime adapter needs lora, and merged weights carry no runtime adapter")
        for name in ("n_ctx", "n_batch", "threads", "max_input_tokens", "max_prefix_tokens", "prefix_slots", "max_adapters"):
            try:
                validate_native_integer(name, getattr(self, name), optional=name in ("n_ctx", "threads"))
            except ValueError as error:
                raise ConfigError(str(error)) from None
        try:
            object.__setattr__(self, "n_ctx", self.capacity().n_ctx)
        except ValueError as error:
            raise ConfigError(str(error)) from None
        if self.quality_profile:
            from .artifacts import load_quality_profile
            load_quality_profile(self.quality_profile)
        from .artifacts import Provenance
        if self.provenance is not None and not isinstance(self.provenance, Provenance):
            raise ConfigError("provenance must come from a verified fitted lock (ss.judge(LOCK)) or a verified pipeline-built base")

    def path(self) -> str:
        from .backend import require_local
        require_local("llama_cpp", "huggingface_hub")
        return _resolve(self.source)

    def lora_path(self) -> str | None:
        return None if self.lora is None else _resolve(self.lora)

    def resident_key(self) -> tuple:
        return (os.path.realpath(self.path()), self.gpu_layers, self.n_ctx, self.n_batch, self.threads, self.prefix_slots, self.repack)

    def adapter_content(self):
        from .resident import Adapter
        if self.lora is None:
            return None
        expected = self.provenance.adapter_sha256 if self.provenance is not None else None
        return Adapter.verified(self.lora_path(), self.path(), expected)

    @property
    def gpu_layers(self) -> int:
        return 0 if self.device == "cpu" else 999

    def capacity(self):
        from .backend import capacity
        return capacity(self.n_ctx, self.n_batch, self.prefix_slots, max_input_tokens=self.max_input_tokens, max_prefix_tokens=self.max_prefix_tokens)

    def descriptor(self):
        from .backend import gguf
        return gguf(self.path())

    def placement(self):
        from .backend import Placement
        return Placement.of(self.descriptor(), self.gpu_layers)

    def resources(self) -> dict[str, Resources]:
        from .backend import Placement
        model = self.descriptor()
        meta, arch, layout = model.meta, model.architecture, self.capacity()
        place = Placement.of(model, self.gpu_layers)
        value = lambda key, default=None: int(meta[key]) if key in meta else default
        layers = value(f"{arch}.block_count") - value(f"{arch}.attention.shared_kv_layers", 0)
        heads = meta[f"{arch}.attention.head_count_kv"]
        heads = [int(h) for h in heads] if isinstance(heads, (tuple, np.ndarray)) else [int(heads)] * layers
        dim = value(f"{arch}.attention.key_length", value(f"{arch}.embedding_length") // value(f"{arch}.attention.head_count"))
        full = dim + value(f"{arch}.attention.value_length", dim)
        swa = value(f"{arch}.attention.key_length_swa", dim) + value(f"{arch}.attention.value_length_swa", dim)
        pattern = meta.get(f"{arch}.attention.sliding_window_pattern", (False,) * layers)
        return {"weights": place.weights(model.size, model.tensors),
                "kv": place.kv(layout.total_ctx * sum(heads[i] * (swa if pattern[i] else full) * 2 for i in range(layers))),
                "output": Resources(model.vocab * layout.n_outputs_max * 4 * 2),
                **({"adapter": place.lora(self.adapter_content()).resources} if self.lora else {})}


NATIVE_SETTINGS = ("adapter", "device", "n_ctx", "n_batch", "threads", "max_input_tokens", "max_prefix_tokens", "prefix_slots", "quality_profile", "repack")


def native_settings(model: Model) -> dict:
    return {name: getattr(model, name) for name in NATIVE_SETTINGS}


def native_spec(source, settings, provenance: Provenance | None = None, lora=None) -> Model:
    if not isinstance(settings, dict) or set(settings) - set(NATIVE_SETTINGS):
        unknown = sorted(set(settings) - set(NATIVE_SETTINGS)) if isinstance(settings, dict) else settings
        raise ConfigError(f"unknown native settings {unknown}; supported: {', '.join(NATIVE_SETTINGS)}")
    return Model(str(source), **settings, lora=None if lora is None else str(lora), provenance=provenance)



@cache
def _resolve(source: str) -> str:
    if os.path.exists(source):
        return source
    from huggingface_hub import hf_hub_download, list_repo_files, try_to_load_from_cache

    repo, _, spec = source.partition(":")
    if spec.endswith(".gguf"):
        cached = try_to_load_from_cache(repo, spec)
        return cached if isinstance(cached, str) else hf_hub_download(repo, spec)
    names = sorted(f for f in list_repo_files(repo) if f.endswith(".gguf") and spec.lower() in f.lower()
                   and not f.startswith(("mmproj", "mtp")))
    if len(names) != 1:
        found = ", ".join(names) or "nothing"
        raise ConfigError(f"model source {source!r} must name exactly one GGUF file; it matches {found}")
    return hf_hub_download(repo, names[0])

