from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Mapping

from .kernel import MISSING


class ContextError(Exception):
    def __init__(self, kind: str, field: str, detail: str = ""):
        super().__init__(f"{kind}: {field} {detail}".strip())
        self.kind = kind
        self.field = field


def lookup(data, path: str):
    if isinstance(data, Mapping) and path in data:
        return MISSING if data[path] is None else data[path]
    cur = data
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return MISSING
        cur = cur[part]
    return MISSING if cur is None else cur


_TYPES = {
    "text": lambda v, spec: isinstance(v, str),
    "number": lambda v, spec: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v, spec: isinstance(v, bool),
    "enum": lambda v, spec: isinstance(v, str) and v in spec.values,
    "json": lambda v, spec: True,
}


def project(fields: Mapping, data) -> dict:
    values = {}
    for name, spec in fields.items():
        v = lookup(data, name)
        if v is MISSING or (spec.required and spec.type == "text" and v == ""):
            if spec.required:
                raise ContextError("missing_context", name)
            continue
        if not _TYPES[spec.type](v, spec):
            raise ContextError("invalid_context", name, f"expected {spec.type}")
        values[name] = v
    return values


def truncate(text: str, limit: int | None, mode: str) -> str:
    if limit is None or len(text) <= limit:
        return text
    if mode == "middle":
        head = (limit - 3) // 2
        return text[:head] + " … " + text[len(text) - (limit - 3 - head):]
    return text[:limit]


def as_text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)


def render(fields: Mapping, values: Mapping, reads: tuple) -> str:
    shown = [n for n in reads if n in values]
    lines = []
    for name in shown:
        spec = fields[name]
        tag = spec.author or spec.source
        text = truncate(as_text(values[name]), spec.max_chars, spec.truncate)
        lines.append(f"[{tag}] {name}: {text}" if tag else f"{name}: {text}" if len(shown) > 1 else text)
    return "\n".join(lines)


def truncated(fields: Mapping, values: Mapping, reads) -> tuple[str, ...]:
    return tuple(n for n in reads if n in values and fields[n].max_chars is not None
                 and len(as_text(values[n])) > fields[n].max_chars)


def _nfc(v):
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if isinstance(v, Mapping):
        return {k: _nfc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nfc(x) for x in v]
    return v


def canonical(values: Mapping, reads: tuple) -> str:
    return json.dumps({k: _nfc(values[k]) for k in reads if k in values}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


_WORD = re.compile(r"\w+", re.UNICODE)


def terms(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return _WORD.findall("".join(c for c in folded if not unicodedata.combining(c)))


class Bm25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.tf = [Counter(terms(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(docs) if docs else 0.0) or 1.0
        df = Counter(t for tf in self.tf for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def score(self, query: set, i: int) -> float:
        tf, k1 = self.tf[i], self.k1
        norm = k1 * (1 - self.b + self.b * self.len[i] / self.avg)
        return sum(self.idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in query if t in tf)

    def rank(self, query: str, ids) -> list[int]:
        q = set(terms(query))
        return sorted(ids, key=lambda i: (-self.score(q, i), i))


def bm25_rank(query: str, docs: list[str]) -> list[int]:
    return Bm25(docs).rank(query, range(len(docs)))
