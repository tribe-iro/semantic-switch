from __future__ import annotations


def adapter_for(name: str, scorer, context=None):
    from .chat import ChatAdapter
    from .qwen3 import Qwen3Adapter

    return {"qwen3": Qwen3Adapter, "chat": ChatAdapter}[name](scorer, context)


def adapter_name(name: str, chat) -> str:
    from .chat import Template
    from .qwen3 import Qwen3Adapter

    return Qwen3Adapter.name if name == "qwen3" else Template.of(chat()).name
