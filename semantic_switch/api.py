from __future__ import annotations

import json
import os
import threading
import warnings
from dataclasses import dataclass
from functools import cache

from . import files
from .errors import ConfigError, suggest
from .inline import Check, Pick, Rate, ask
from .judge import FIXTURE, FIXTURE_QUESTIONS, LlamaJudge
from .switch import Switch

DEFAULT_JUDGE = "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf"
PRECISE_JUDGE = "google/gemma-4-E4B-it-qat-q4_0-gguf:gemma-4-E4B_q4_0-it.gguf"
ADAPTERS = ("chat", "qwen3")
DEVICES = ("auto", "cpu", "gpu")
QUALITY_TARGET = 0.95


@dataclass(frozen=True)
class Model:
    source: str
    adapter: str = "chat"
    device: str = "auto"
    n_ctx: int = 8192
    n_batch: int = 512
    threads: int | None = None

    def __post_init__(self):
        if self.adapter not in ADAPTERS:
            raise ConfigError(f"unknown adapter {self.adapter!r}{suggest(self.adapter, ADAPTERS)}")
        if self.device not in DEVICES:
            raise ConfigError(f"unknown device {self.device!r}{suggest(self.device, DEVICES)}")

    def path(self) -> str:
        return _resolve(self.source)

    @property
    def gpu_layers(self) -> int:
        return 0 if self.device == "cpu" else 999


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


def _gpu() -> bool:
    from .backend import gpu_available

    return gpu_available()


def _build() -> str:
    try:
        import llama_cpp

        return llama_cpp.__version__
    except ImportError:
        return "none"


class Runtime:
    def __init__(self, models: dict | None = None):
        self.models = dict(models or {})
        self._loaded: dict = {}
        self._lock = threading.Lock()

    def _spec(self, role: str):
        if role not in self.models:
            if role != "judge":
                raise ConfigError(f"model role {role!r} is not bound in this runtime")
            self.models["judge"] = Model(DEFAULT_JUDGE)
        return self.models[role]

    def cache_key(self, role: str = "judge") -> str:
        backend = "gpu" if self.accelerated(role) else "cpu"
        return json.dumps({**self.identity(role), "backend": backend, "build": _build()}, sort_keys=True)

    def judge(self, role: str = "judge") -> LlamaJudge:
        with self._lock:
            if role not in self._loaded:
                spec = self._spec(role)
                if isinstance(spec, Model):
                    self._loaded[role] = self._load_judge(role, spec)
                elif hasattr(spec, "score_many") and hasattr(spec, "fit"):
                    self._loaded[role] = spec
                else:
                    from .adapters import adapter_for

                    adapter = adapter_for(getattr(spec, "adapter", "qwen3"), spec)
                    self._loaded[role] = LlamaJudge(spec, adapter, self.identity(role), self.accelerated(role))
            return self._loaded[role]

    def _load_judge(self, role: str, spec: Model) -> LlamaJudge:
        from .adapters import adapter_for
        from .backend import LlamaScorer, shared_ok

        if spec.device == "gpu" and not _gpu():
            raise ConfigError("device='gpu' was requested, but this llama.cpp build has no GPU offload; "
                              "rebuild with GPU support or use device='auto' or 'cpu'")
        try:
            scorer = LlamaScorer(spec.path(), n_ctx=spec.n_ctx, n_batch=spec.n_batch, threads=spec.threads,
                                 n_gpu_layers=spec.gpu_layers)
        except RuntimeError as e:
            raise ConfigError(f"cannot load model {spec.source}: {e}") from None
        try:
            adapter = adapter_for(spec.adapter, scorer)
        except ValueError as e:
            scorer.close()
            raise ConfigError(f"model {spec.source}: {e}") from None
        key = self.cache_key(role)
        store_path = files.cache_dir() / "doctor.json"
        known = (files.read_json(store_path) or {}).get(key)
        if known:
            scorer.shared = shared_ok(known.get("checks", {}))
        elif scorer.weights.hybrid:
            scorer.shared = True
            checks = scorer.selftest(adapter, FIXTURE, FIXTURE_QUESTIONS)
            store = files.read_json(store_path) or {}
            store[key] = {"ok": all(c["ok"] for c in checks.values()), "checks": checks, "at": files.now()}
            files.write_json_atomic(store_path, store)
        quality = (files.read_json(files.cache_dir() / "quality.json") or {}).get(key)
        if quality and (quality.get("precision_0.9") or 0) < QUALITY_TARGET:
            warnings.warn(f"judge {spec.source} settles untuned answers at {quality['precision_0.9']:.0%} precision "
                          f"on the quality probe; run `ss lock` with labelled data, or use the default judge",
                          stacklevel=4)
        return LlamaJudge(scorer, adapter, self.identity(role), self.accelerated(role))

    def identity(self, role: str) -> dict:
        spec = self._spec(role)
        if hasattr(spec, "score_many") and hasattr(spec, "fit"):
            return dict(spec.identity)
        if not isinstance(spec, Model):
            return {"source": getattr(spec, "name", type(spec).__name__),
                    "template": _template(getattr(spec, "adapter", "qwen3"))}
        path = spec.path()
        return {"file": os.path.basename(path), "sha256": files.sha256_file(path),
                "template": _template(spec.adapter, path)}

    def accelerated(self, role: str) -> bool:
        spec = self._spec(role)
        if isinstance(spec, Model):
            return spec.device != "cpu" and _gpu()
        return bool(getattr(spec, "accelerated", False))

    def ask(self, state, items: dict, *, fields=None) -> dict:
        return ask(self.judge, state, items, fields)

    def check(self, state, statement: str, *, excludes=(), fields=None, min_score=None, unstated: bool = False):
        return Check(statement, tuple(excludes), min_score, unstated)(state, self, fields)

    def pick(self, state, question: str, options, *, describe=None, eligible=None, allow_none=False, shortlist=None,
             fields=None, min_score=None, ids=None, focus=None):
        return Pick(question, options, describe, eligible, allow_none, shortlist, min_score, ids, focus)(state, self,
                                                                                                        fields)

    def rate(self, state, question: str, levels, *, fields=None, min_score=None):
        return Rate(question, tuple(levels), min_score)(state, self, fields)

    def load(self, path, draft: bool = False, allow_model_change: bool = False) -> Switch:
        return Switch(self, path, draft, allow_model_change)

    def close(self):
        with self._lock:
            for item in self._loaded.values():
                close = getattr(getattr(item, "scorer", item), "close", None)
                if close:
                    close()
            self._loaded.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@cache
def _template(adapter: str, path: str | None = None) -> str:
    from .adapters.chat import name_of
    from .adapters.qwen3 import Qwen3Adapter

    if adapter == "chat" and path:
        from .backend import Weights

        vocab = Weights(path, vocab_only=True)
        try:
            return name_of(vocab.chat())
        finally:
            vocab.close()
    return {"qwen3": Qwen3Adapter.name, "chat": "chat@2"}[adapter]


_default: Runtime | None = None


def default() -> Runtime:
    global _default
    if _default is None:
        _default = Runtime()
    return _default


def check(state, statement, **kw):
    return default().check(state, statement, **kw)


def pick(state, question, options, **kw):
    return default().pick(state, question, options, **kw)


def rate(state, question, levels, **kw):
    return default().rate(state, question, levels, **kw)


def ask_(state, items, **kw):
    return default().ask(state, items, **kw)


def load(path, draft: bool = False, allow_model_change: bool = False) -> Switch:
    return default().load(path, draft, allow_model_change)
