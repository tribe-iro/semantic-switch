from __future__ import annotations

import difflib


class ConfigError(Exception):
    pass


def suggest(name: str, choices) -> str:
    close = difflib.get_close_matches(name, list(choices), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""
