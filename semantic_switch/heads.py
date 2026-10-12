from __future__ import annotations

import base64
import re
import unicodedata
import zlib
from dataclasses import dataclass, field

import numpy as np

N_FEATURES = 1 << 18
FEATURE_IDENTITY = f"lexical-nfc-casefold-words-bigrams-char345-crc32-utf8-l2-softmax-v1-unicode{unicodedata.unidata_version}"
_WORDS = re.compile(r"\w[\w\-.]*\w|\w")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _arr(s: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s, validate=True), dtype=dtype)


def features(text: str) -> dict[int, float]:
    words = _WORDS.findall(unicodedata.normalize("NFC", text).casefold())
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    grams += [f"#{w[i:i + n]}" for w in (f"<{w}>" for w in words) for n in (3, 4, 5) for i in range(len(w) - n + 1)]
    counts: dict[int, float] = {}
    for g in grams:
        h = zlib.crc32(g.encode()) % N_FEATURES
        counts[h] = counts.get(h, 0.0) + 1.0
    norm = sum(v * v for v in counts.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in counts.items()}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class LexicalHead:
    classes: tuple[str, ...]
    idx: np.ndarray
    w: np.ndarray
    b: np.ndarray
    rows: dict = field(default_factory=dict, compare=False)
    kind = "lexical"

    def __post_init__(self):
        from types import MappingProxyType
        for name in ("idx", "w", "b"):
            a = getattr(self, name)
            object.__setattr__(self, name, np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape))
        object.__setattr__(self, "rows", MappingProxyType({int(j): r for r, j in enumerate(self.idx)}))

    def probs(self, text: str) -> np.ndarray:
        return self.probs_features(features(text))

    def probs_features(self, feats: dict[int, float]) -> np.ndarray:
        z = self.b.astype(np.float64, copy=True)
        rows, vals = [], []
        for j, v in feats.items():
            r = self.rows.get(j)
            if r is not None:
                rows.append(r)
                vals.append(v)
        if rows:
            z += np.asarray(vals, dtype=np.float64) @ self.w[rows]
        return _softmax(z)

    def to_json(self) -> dict:
        return {"kind": self.kind, "feature_identity": FEATURE_IDENTITY, "classes": list(self.classes), "n": int(len(self.idx)),
                "idx": _b64(self.idx.astype("<i4")), "w": _b64(self.w.astype("<f4")),
                "b": _b64(self.b.astype("<f4"))}

    @classmethod
    def from_json(cls, d: dict) -> LexicalHead:
        from .errors import ConfigError
        try:
            if not isinstance(d, dict) or d.get("feature_identity") != FEATURE_IDENTITY:
                raise ValueError("incompatible or missing feature identity")
            if sum(len(d.get(k, "")) for k in ("idx", "w", "b")) > 64 * 1024 * 1024:
                raise ValueError("encoded head exceeds 64 MiB")
            classes, n = d["classes"], d["n"]
            if d.get("kind") != "lexical" or not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise ValueError("invalid head kind/size")
            if not classes or len(set(classes)) != len(classes) or not all(isinstance(x, str) for x in classes):
                raise ValueError("invalid classes")
            k = len(classes)
            idx, w, b = _arr(d["idx"], "<i4"), _arr(d["w"], "<f4"), _arr(d["b"], "<f4")
            if len(idx) != n or len(w) != n * k or len(b) != k:
                raise ValueError("head array lengths disagree")
            if len(np.unique(idx)) != n or ((idx < 0) | (idx >= N_FEATURES)).any():
                raise ValueError("invalid feature indices")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("non-finite head weights")
            return cls(tuple(classes), idx, w.reshape(n, k), b)
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f"invalid lexical head: {e}") from None

