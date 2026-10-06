from __future__ import annotations


def adapter_for(name: str, scorer):
    from .chat import ChatAdapter
    from .qwen3 import Qwen3Adapter

    return {"qwen3": Qwen3Adapter, "chat": ChatAdapter}[name](scorer)
