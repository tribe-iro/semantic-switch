from __future__ import annotations

import os
from pathlib import Path

import semantic_switch as ss

DATA = Path(__file__).resolve().parent / "data"
SETTINGS = (("provider", "OPENROUTER_PROVIDER"), ("codec", "OPENROUTER_CODEC"))


def judge(environ=os.environ):
    model = environ.get("OPENROUTER_MODEL")
    if not model:
        return ss.judge(ss.DEFAULT_JUDGE)
    return ss.judge(f"openrouter:{model}", **{key: environ[name] for key, name in SETTINGS if environ.get(name)})


def runtime(lock=None, environ=os.environ):
    return ss.Runtime() if lock else ss.Runtime(judge(environ), own_judges=True)
