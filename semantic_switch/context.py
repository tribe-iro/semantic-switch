from __future__ import annotations

import json
import math
import heapq
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .kernel import MISSING

RENDERING_REVISION = "named-fields@2"


class ContextError(Exception):
    def __init__(self, kind: str, field: str, detail: str = ""):
        super().__init__(f"{kind}: {field} {detail}".strip())
        self.kind = kind
        self.field = field


def error_status(e: ContextError) -> str:
    return "needs_input" if e.kind == "missing_context" else "failed"


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
    "number": lambda v, spec: isinstance(v, (int, float)) and not isinstance(v, bool) and (isinstance(v, int) or math.isfinite(v)),
    "bool": lambda v, spec: isinstance(v, bool),
    "enum": lambda v, spec: isinstance(v, str) and v in spec.values,
    "json": lambda v, spec: True,
}


def bind_input(fields: Mapping, data):
    return {next(iter(fields)): data} if isinstance(data, str) and len(fields) == 1 else data


@dataclass(frozen=True, slots=True, eq=False, init=False)
class OwnedMapping(Mapping):
    _values: Mapping

    def __init__(self, values):
        from .results import freeze
        object.__setattr__(self, '_values', MappingProxyType({k: freeze(v) for k, v in values.items()}))

    @classmethod
    def _normalized(cls, values):
        result = object.__new__(cls)
        object.__setattr__(result, '_values', MappingProxyType(values))
        return result

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class OwnedJSON(OwnedMapping):
    __slots__ = ()

    def __init__(self, values):
        owned = json_value(values)
        object.__setattr__(self, '_values', owned._values)


class JSONArray(tuple):
    __slots__ = ()

    def __new__(cls, values=()):
        return tuple.__new__(cls, (json_value(value) for value in values))

    def __eq__(self, other):
        return tuple.__eq__(self, tuple(other)) if isinstance(other, (list, tuple)) else False

    def __ne__(self, other):
        return not self == other

    __hash__ = tuple.__hash__


def json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def json_value(value, path: str = "input", seen: set | None = None):
    if type(value) in (OwnedJSON, JSONArray):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        seen = set() if seen is None else seen
        if id(value) in seen:
            raise ContextError("invalid_context", path, "cyclic JSON")
        seen.add(id(value))
        try:
            if isinstance(value, Mapping):
                if not all(isinstance(k, str) for k in value):
                    raise ContextError("invalid_context", path, "JSON keys must be strings")
                return OwnedJSON._normalized({k: json_value(v, f"{path}.{k}", seen) for k, v in value.items()})
            return tuple.__new__(JSONArray, (json_value(v, f"{path}[{i}]", seen) for i, v in enumerate(value)))
        finally:
            seen.remove(id(value))
    raise ContextError("invalid_context", path, "expected finite JSON value")


def absent(v) -> bool:
    return v is MISSING or v is None or (isinstance(v, str) and not v.strip())


def project(fields: Mapping, data) -> dict:
    if not isinstance(data, Mapping):
        raise ContextError("invalid_context", "input", "expected mapping")
    values = {}
    for name, spec in fields.items():
        v = lookup(data, name)
        if v is MISSING or (spec.required and absent(v)):
            if spec.required:
                raise ContextError("missing_context", name)
            continue
        if not _TYPES[spec.type](v, spec):
            raise ContextError("invalid_context", name, f"expected {spec.type}")
        values[name] = json_value(v, name) if spec.type == "json" else v
    return values


def truncate(text: str, limit: int | None, mode: str) -> str:
    if limit is None or len(text) <= limit:
        return text
    if mode == "middle" and limit >= 3:
        head = (limit - 3) // 2
        return text[:head] + " … " + text[len(text) - (limit - 3 - head):]
    return text[:limit]


def as_text(v) -> str:
    return v if isinstance(v, str) else json.dumps(json_value(v), ensure_ascii=False, sort_keys=True, allow_nan=False, default=json_default)


def render(fields: Mapping, values: Mapping, reads: tuple) -> str:
    shown = [n for n in reads if n in values]
    lines = []
    for name in shown:
        spec = fields[name]
        tag = spec.author or spec.source
        text = truncate(as_text(values[name]), spec.max_chars, spec.truncate)
        lines.append(f"[{tag}] {name}: {text}" if tag else f"{name}: {text}")
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
                      separators=(",", ":"), allow_nan=False)


_WORD = re.compile(r"\w+", re.UNICODE)


def terms(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold())
    return _WORD.findall("".join(c for c in folded if not unicodedata.combining(c)))


@dataclass(frozen=True, slots=True, init=False)
class Bm25:
    tf: tuple
    len: tuple
    avg: float
    idf: Mapping
    k1: float
    b: float

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self._build(tuple(MappingProxyType(dict(Counter(terms(doc)))) for doc in docs), k1, b)

    def subset(self, ids) -> Bm25:
        out = object.__new__(Bm25)
        out._build(tuple(self.tf[i] for i in ids), self.k1, self.b)
        return out

    def _build(self, tf: tuple, k1: float, b: float):
        lengths = tuple(sum(counts.values()) for counts in tf)
        df = Counter(term for counts in tf for term in counts)
        n = len(tf)
        for name, value in (("tf", tf), ("len", lengths),
                            ("avg", (sum(lengths) / n if n else 0.) or 1.),
                            ("idf", MappingProxyType({term: math.log(1 + (n - count + .5) / (count + .5))
                                                      for term, count in df.items()})), ("k1", k1), ("b", b)):
            object.__setattr__(self, name, value)

    def score(self, query: set, i: int) -> float:
        tf, k1 = self.tf[i], self.k1
        norm = k1 * (1 - self.b + self.b * self.len[i] / self.avg)
        return sum(self.idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in query if t in tf)

    def rank(self, query: str, ids, limit=None) -> list[int]:
        q = set(terms(query))
        key = lambda i: (-self.score(q, i), i)
        return sorted(ids, key=key) if limit is None else heapq.nsmallest(limit, ids, key=key)

