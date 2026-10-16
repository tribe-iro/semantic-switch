from __future__ import annotations

from ..api import Model
from . import CONTRACT, DriverSpec


def judge(source: str, settings: dict) -> Model:
    from ..tools.deployment import fitted, frozen
    return Model(source, **settings) if frozen(source) is None else fitted(source, **settings)


DRIVER = DriverSpec("gguf", CONTRACT, Model, judge, 1, "local GGUF file, Hugging Face repo:file or fitted lock bundle through llama.cpp",
                    internal=("source", "provenance"))
