from __future__ import annotations

import hashlib
import json
from datetime import datetime

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .qwen3 import SYSTEM, Qwen3Adapter

SLOT = "STATE"


def _raise(message):
    raise ValueError(message)


def render(chat: dict) -> tuple[str, str]:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals.update(raise_exception=_raise, strftime_now=lambda fmt: datetime.now().strftime(fmt))
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **_: json.dumps(x, indent=indent,
                                                                                    ensure_ascii=ensure_ascii)
    text = env.from_string(chat["template"]).render(
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": SLOT}],
        add_generation_prompt=True, enable_thinking=False, bos_token=chat["bos"], eos_token=chat["eos"], tools=None)
    prefix, tail = text.split(SLOT)
    if chat["add_bos"] and chat["bos"] and not prefix.startswith(chat["bos"]):
        prefix = chat["bos"] + prefix
    return prefix, tail


def name_of(chat: dict) -> str:
    return "chat@2:" + hashlib.sha256((chat["template"] or "").encode()).hexdigest()[:8]


class ChatAdapter(Qwen3Adapter):
    def __init__(self, scorer):
        chat = scorer.chat()
        if not chat.get("template"):
            raise ValueError("this model has no chat template; use a chat (instruct) model")
        self.prefix_text, self.tail_text = render(chat)
        self.name = name_of(chat)
        super().__init__(scorer)
