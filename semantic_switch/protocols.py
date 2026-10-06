from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Question:
    kind: str
    text: str
    options: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()

    @property
    def width(self) -> int:
        return 2 if self.kind == "check" else len(self.options)


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    prefix: tuple[int, ...]
    state: tuple[int, ...]
    suffixes: tuple[tuple[int, ...], ...]
    slots: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(slots=True)
class ScoreBlock:
    logp: np.ndarray
    log_mass: np.ndarray
    error: list

    @classmethod
    def empty(cls, rows: int, width: int) -> ScoreBlock:
        return cls(np.full((rows, width), -np.inf, dtype=np.float32),
                   np.full(rows, np.nan, dtype=np.float32), [None] * rows)


class Scorer(Protocol):
    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]: ...

    def detokenize(self, tokens: Sequence[int]) -> str: ...

    def score(self, requests: Sequence[ScoreRequest]) -> ScoreBlock: ...


class Judge(Protocol):
    identity: dict
    accelerated: bool
    max_slots: int

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list: ...

    def fit(self, text: str, limit: int) -> tuple[str, bool]: ...
