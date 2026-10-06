from __future__ import annotations

import base64
import re
import unicodedata
import zlib
from dataclasses import dataclass, field

import numpy as np

N_FEATURES = 1 << 18
_WORDS = re.compile(r"\w[\w\-.]*\w|\w")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _arr(s: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


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


def matrix(texts: list[str]):
    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        for c, v in features(t).items():
            rows.append(r)
            cols.append(c)
            vals.append(v)
    return csr_matrix((vals, (rows, cols)), shape=(len(texts), N_FEATURES), dtype=np.float32)


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
        self.rows.update({int(j): r for r, j in enumerate(self.idx)})

    def probs(self, text: str) -> np.ndarray:
        z = self.b.astype(np.float64).copy()
        for j, v in features(text).items():
            r = self.rows.get(j)
            if r is not None:
                z += v * self.w[r]
        return _softmax(z)

    def to_json(self) -> dict:
        return {"kind": self.kind, "classes": list(self.classes), "n": int(len(self.idx)),
                "idx": _b64(self.idx.astype(np.int32)), "w": _b64(self.w.astype(np.float32)),
                "b": _b64(self.b.astype(np.float32))}

    @classmethod
    def from_json(cls, d: dict) -> LexicalHead:
        k = len(d["classes"])
        return cls(tuple(d["classes"]), _arr(d["idx"], np.int32), _arr(d["w"], np.float32).reshape(d["n"], k),
                   _arr(d["b"], np.float32))


def _fit(x, y: list[int], k: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression

    if len(set(y)) < 2:
        raise ValueError("a head needs examples of at least two different values")
    clf = LogisticRegression(max_iter=5000, C=c)
    clf.fit(x, y)
    w = np.zeros((x.shape[1], k), dtype=np.float32)
    b = np.full(k, -30.0, dtype=np.float32)
    if len(clf.classes_) == 2:
        lo, hi = clf.classes_
        w[:, hi] = clf.coef_[0]
        b[lo], b[hi] = 0.0, clf.intercept_[0]
    else:
        for row, cls_ in enumerate(clf.classes_):
            w[:, cls_] = clf.coef_[row]
            b[cls_] = clf.intercept_[row]
    return w, b


def train_lexical(texts: list[str], y: list[int], classes: tuple[str, ...], c: float = 10.0) -> LexicalHead:
    w, b = _fit(matrix(texts), y, len(classes), c)
    idx = np.flatnonzero(np.abs(w).sum(axis=1) > 0).astype(np.int32)
    return LexicalHead(classes, idx, w[idx], b)

