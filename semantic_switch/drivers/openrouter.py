from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType

from ..errors import ConfigError
from ..hosted import Envelope, HostedJudge, Reply, Wire, hosted_dependencies
from . import CONTRACT, DriverSpec
from .codecs import CODECS
from .profiles import ModelProfile, resolve

BASE = "https://openrouter.ai/api/v1"
PATHS = MappingProxyType({"chat": "chat/completions", "systemone": "systemone"})


def credentials(key: str | None = None) -> dict:
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ConfigError("OpenRouter judges need an API key: set OPENROUTER_API_KEY or pass key=...")
    return {"Authorization": f"Bearer {key}"}


@dataclass(frozen=True, slots=True)
class OpenRouter:
    route: str | None = None
    key: str | None = field(default=None, repr=False)
    name = "openrouter"
    revision = 1
    identity = MappingProxyType({"service": BASE})

    def endpoint(self, codec) -> str:
        if codec.name not in PATHS:
            raise ConfigError(f"OpenRouter has no endpoint for codec {codec.name!r}; supported: {', '.join(PATHS)}")
        return f"{BASE}/{PATHS[codec.name]}"

    def wire(self, connections: int, max_bytes: int) -> Wire:
        return Wire(hosted_dependencies(), credentials(self.key), connections, max_bytes)

    def envelope(self, model: str, codec, body: dict) -> dict:
        return {"model": model, **body, "provider": {**({"only": [self.route]} if self.route else {}), "allow_fallbacks": False,
                                                     **({"require_parameters": True} if codec.parameters else {})}}

    @staticmethod
    def usage(body: dict) -> dict | None:
        usage = body.get("usage")
        return usage if isinstance(usage, dict) else None

    @staticmethod
    def unwrap(body: dict) -> Envelope:
        model, route, usage = body.get("model"), body.get("provider"), body.get("usage")
        if not isinstance(model, str) or route is not None and not isinstance(route, str) or usage is not None and not isinstance(usage, dict):
            raise ValueError("returned model, provider or usage")
        tokens = (usage or {}).get("prompt_tokens", (usage or {}).get("input_tokens", 0))
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("prompt tokens must be a nonnegative integer")
        return Envelope(f"{model}@{route}" if route else model, route, tokens, body)

    def rejected(self, reply: Reply) -> str:
        error = reply.body.get("error") if isinstance(reply.body, dict) else None
        error = error if isinstance(error, dict) else {}
        message = error.get("message")[:300] if isinstance(error.get("message"), str) else None
        if reply.status != 400:
            return f"HTTP {reply.status}" + (f": {message}" if message else "")
        metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
        named = metadata.get("provider_name") if isinstance(metadata.get("provider_name"), str) else self.route
        return ((f"provider {named}" if named else "the routed provider") + " rejected the request shape (HTTP 400" + (f": {message}" if message else "") +
                "); pin a provider that accepts it with provider=PROVIDER or --provider PROVIDER")


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str | None = None
    key: str | None = field(default=None, repr=False)
    codec: str | None = None
    profile: str | ModelProfile | None = None
    timeout: float = 60.0
    batch_inputs: int = 8
    max_response_bytes: int | None = None
    transport: object = None
    own_transport: bool = False

    def __post_init__(self):
        for name in ("provider", "key", "codec"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{name} must be nonempty text")
        if self.codec is not None and self.codec not in CODECS:
            raise ConfigError(f"unknown codec {self.codec!r}; supported: {', '.join(CODECS)}")
        if self.key is not None and self.transport is not None:
            raise ConfigError("key applies only to the built-in transport; a supplied transport carries its own credentials")


def judge(model: str, settings: dict) -> HostedJudge:
    settings = Settings(**settings)
    if not model.strip() or "@" in model:
        raise ConfigError("use openrouter:MODEL and pass the provider separately (--provider PROVIDER or provider=...)")
    profile = resolve("openrouter", model, codec=settings.codec, profile=settings.profile)
    return HostedJudge(OpenRouter(settings.provider, settings.key), model, profile, transport=settings.transport,
                       own_transport=settings.own_transport, timeout=settings.timeout, batch_inputs=settings.batch_inputs,
                       max_response_bytes=settings.max_response_bytes)


DRIVER = DriverSpec("openrouter", CONTRACT, Settings, judge, 1, "OpenRouter hosted model: chat log probabilities or System One by model profile")
