from __future__ import annotations

import dataclasses
import importlib
import os
import re
import sys
import threading
import types
import typing
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Callable

from ..errors import ConfigError, suggest

CONTRACT = 1
GROUP = "semantic_switch.judges"
BUILTIN = types.MappingProxyType({"gguf": "semantic_switch.drivers.gguf:DRIVER", "openrouter": "semantic_switch.drivers.openrouter:DRIVER"})
SCHEME = re.compile(r"([a-z][a-z0-9+.-]*):(.*)", re.S)
TEXT = (str, int, float, bool)
TRUE, FALSE = ("true", "1", "yes", "on"), ("false", "0", "no", "off")


@dataclass(frozen=True, slots=True)
class DriverSpec:
    scheme: str
    contract: int
    settings: type
    factory: Callable[[str, dict], object]
    revision: int
    summary: str
    internal: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.scheme, str) or not SCHEME.fullmatch(f"{self.scheme}:"):
            raise ConfigError(f"driver scheme {self.scheme!r} must be lowercase letters, digits, '+', '-' or '.'")
        if not dataclasses.is_dataclass(self.settings) or not callable(self.factory):
            raise ConfigError(f"driver {self.scheme}: settings must be a dataclass and factory callable")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ConfigError(f"driver {self.scheme}: revision must be a positive integer")

    def fields(self) -> dict:
        return hints(self.settings, self.internal)

    def configure(self, settings: dict) -> dict:
        fields = self.fields()
        unknown = sorted(set(settings) - set(fields))
        if unknown:
            names = ", ".join(unknown)
            raise ConfigError(f"{names} {'does' if len(unknown) == 1 else 'do'} not apply to {self.scheme} judges"
                              f"{suggest(unknown[0], fields)}; supported: {', '.join(fields)}")
        return {name: coerce(name, fields[name], value) for name, value in settings.items()}

    def describe(self) -> str:
        names = [name for name, hint in self.fields().items() if textual(hint)]
        return f"{self.scheme}: {', '.join(names)}"


@lru_cache(maxsize=None)
def hints(settings: type, internal: tuple[str, ...]) -> types.MappingProxyType:
    scope = vars(sys.modules[settings.__module__])
    return types.MappingProxyType({f.name: eval(f.type, scope) if isinstance(f.type, str) else f.type
                                   for f in dataclasses.fields(settings) if f.name not in internal})


def textual(hint) -> tuple:
    return tuple(t for t in (typing.get_args(hint) or (hint,)) if t in TEXT)


def coerce(name: str, hint, value):
    if not isinstance(value, str):
        return value
    allowed = textual(hint)
    if not allowed:
        raise ConfigError(f"{name} cannot be given as text")
    if str in allowed:
        return value
    lowered = value.strip().lower()
    if bool in allowed and lowered in TRUE + FALSE:
        return lowered in TRUE
    for kind in (int, float):
        if kind in allowed:
            try:
                return kind(value)
            except ValueError:
                pass
    raise ConfigError(f"{name}={value!r} is not a valid {' or '.join(t.__name__ for t in allowed)}")


def pairs(items) -> dict:
    out = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"setting {item!r} must be KEY=VALUE")
        if key in out:
            raise ConfigError(f"setting {key} is given more than once")
        out[key] = value
    return out


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._explicit: dict[str, DriverSpec] = {}
        self._loaded: dict[str, DriverSpec] = {}
        self._plugins: dict[str, list] | None = None

    def plugins(self) -> dict[str, list]:
        with self._lock:
            if self._plugins is None:
                found: dict[str, list] = {}
                for point in metadata.entry_points(group=GROUP):
                    found.setdefault(point.name, []).append(point)
                self._plugins = found
            found = self._plugins
        shadowing = sorted(set(found) & set(BUILTIN))
        if shadowing:
            scheme = shadowing[0]
            raise ConfigError(f"judge scheme {scheme!r} is registered more than once: builtin {BUILTIN[scheme]}, {origins(found[scheme])}")
        return found

    def register(self, spec: DriverSpec) -> DriverSpec:
        if not isinstance(spec, DriverSpec):
            raise ConfigError("register takes a DriverSpec")
        check(spec)
        plugins = self.plugins()
        with self._lock:
            if spec.scheme in self._explicit or spec.scheme in BUILTIN or spec.scheme in plugins:
                raise ConfigError(f"judge scheme {spec.scheme!r} is already registered")
            self._explicit[spec.scheme] = spec
        return spec

    def schemes(self) -> tuple[str, ...]:
        return tuple(sorted({*BUILTIN, *self.plugins(), *self._explicit}))

    def driver(self, scheme: str) -> DriverSpec:
        with self._lock:
            known = self._explicit.get(scheme) or self._loaded.get(scheme)
        if known is not None:
            return known
        if scheme in BUILTIN:
            spec = load(BUILTIN[scheme])
        else:
            found = self.plugins().get(scheme)
            if not found:
                raise ConfigError(f"unknown judge scheme {scheme!r}{suggest(scheme, self.schemes())}; registered: {', '.join(self.schemes())}")
            if len(found) > 1:
                raise ConfigError(f"judge scheme {scheme!r} is registered more than once: {origins(found)}")
            spec = found[0].load()
        if not isinstance(spec, DriverSpec) or spec.scheme != scheme:
            raise ConfigError(f"entry point {scheme!r} in {GROUP} must name a DriverSpec for scheme {scheme!r}")
        check(spec)
        with self._lock:
            return self._loaded.setdefault(scheme, spec)

    def resolve(self, source) -> tuple[DriverSpec, str]:
        source = os.fspath(source) if isinstance(source, os.PathLike) else source
        if not isinstance(source, str) or not source.strip():
            raise ConfigError("judge source must be nonempty text: a GGUF path, repo:file or SCHEME:MODEL (e.g. openrouter:MODEL)")
        match = SCHEME.fullmatch(source)
        if match is None or Path(source).exists():
            return self.driver("gguf"), source
        return self.driver(match[1]), match[2]

    def build(self, source, settings: dict):
        spec, target = self.resolve(source)
        return spec.factory(target, spec.configure(settings))

    def help(self) -> str:
        return "; ".join(self.driver(scheme).describe() for scheme in BUILTIN) + f"; plugins through {GROUP}"


def origins(points) -> str:
    return ", ".join(f"{p.value} ({p.dist.name if p.dist else 'unknown distribution'})" for p in points)


def load(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def check(spec: DriverSpec):
    if spec.contract != CONTRACT:
        raise ConfigError(f"judge driver {spec.scheme!r} implements contract {spec.contract}; this semantic-switch supports contract {CONTRACT}")


REGISTRY = Registry()
register = REGISTRY.register
build = REGISTRY.build
