from __future__ import annotations

from ..presentation import SYSTEM
from .base import NativeAdapter

PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3Adapter(NativeAdapter):
    name = "qwen3@5"
    prefix_text, tail_text = PREFIX, TAIL

    def __init__(self, scorer, context=None):
        super().__init__(scorer, Qwen3Adapter.name, PREFIX, TAIL)
