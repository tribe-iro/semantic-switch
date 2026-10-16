from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from jinja2.sandbox import ImmutableSandboxedEnvironment

from ..presentation import SYSTEM
from .base import NativeAdapter

SLOT = "STATE"


@dataclass(frozen=True, slots=True)
class RenderContext:
    date: datetime

    @classmethod
    def now(cls) -> RenderContext:
        return cls(datetime.now().replace(microsecond=0))


@dataclass(frozen=True, slots=True)
class Template:
    source: str | None
    bos: str
    eos: str
    add_bos: bool

    @classmethod
    def of(cls, chat: dict) -> Template:
        return cls(chat["template"], chat["bos"], chat["eos"], bool(chat["add_bos"]))

    @property
    def name(self) -> str:
        return "chat@3:" + hashlib.sha256((self.source or "").encode()).hexdigest()[:8]

    def render(self, context: RenderContext) -> tuple[str, str]:
        return _render(self, context)


def _raise(message):
    raise ValueError(message)


@lru_cache(maxsize=64)
def _render(template: Template, context: RenderContext) -> tuple[str, str]:
    if not template.source:
        raise ValueError("this model has no chat template; use a chat (instruct) model")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals.update(raise_exception=_raise, strftime_now=context.date.strftime)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **_: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
    text = env.from_string(template.source).render(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": SLOT}],
        add_generation_prompt=True, enable_thinking=False, bos_token=template.bos, eos_token=template.eos, tools=None)
    prefix, tail = text.split(SLOT)
    if template.add_bos and template.bos and not prefix.startswith(template.bos):
        prefix = template.bos + prefix
    return prefix, tail


class ChatAdapter(NativeAdapter):
    def __init__(self, scorer, context: RenderContext | None = None):
        template = Template.of(scorer.chat())
        super().__init__(scorer, template.name, *template.render(context or RenderContext.now()))
