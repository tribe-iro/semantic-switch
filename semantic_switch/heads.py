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
    counts: dict[int, float] = {}

    def add(gram: str):
        h = zlib.crc32(gram.encode()) % N_FEATURES
        counts[h] = counts.get(h, 0.0) + 1.0

    for w in words:
        add(w)
    for a, b in zip(words, words[1:]):
        add(f"{a} {b}")
    for w in words:
        w = f"<{w}>"
        for n in (3, 4, 5):
            for i in range(len(w) - n + 1):
                add("#" + w[i:i + n])
    norm = sum(v * v for v in counts.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in counts.items()}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _owned(a: np.ndarray) -> np.ndarray:
    base = a
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if not a.flags.writeable and isinstance(base, bytes):
        return a
    out = np.array(a, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class LexicalHead:
    classes: tuple[str, ...]
    idx: np.ndarray
    w: np.ndarray
    b: np.ndarray
    order: np.ndarray = field(init=False, repr=False, compare=False)
    keys: np.ndarray = field(init=False, repr=False, compare=False)
    kind = "lexical"

    def __post_init__(self):
        for name in ("idx", "w", "b"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        order = np.argsort(self.idx, kind="stable")
        keys = self.idx[order]
        order.setflags(write=False)
        keys.setflags(write=False)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "keys", keys)

    def probs(self, text: str) -> np.ndarray:
        return self.probs_features(features(text))

    def probs_features(self, feats: dict[int, float]) -> np.ndarray:
        z = self.b.astype(np.float64)
        if feats and len(self.keys):
            columns = np.fromiter(feats.keys(), dtype=np.int64, count=len(feats))
            values = np.fromiter(feats.values(), dtype=np.float64, count=len(feats))
            at = np.minimum(np.searchsorted(self.keys, columns), len(self.keys) - 1)
            known = self.keys[at] == columns
            if known.any():
                z += values[known] @ self.w[self.order[at[known]]]
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

