from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from .guidance import Guidance

if TYPE_CHECKING:
    from .judge import Scored


@dataclass(frozen=True, slots=True)
class Framing:
    name: str
    prefix_text: str
    tail_text: str
    binary: tuple[tuple[int, ...], ...]
    polar: tuple[tuple[int, ...], ...]
    letters: tuple[tuple[int, ...], ...]
    revision: int = 2

    def as_dict(self):
        return {"name": self.name, "prefix_text": self.prefix_text, "tail_text": self.tail_text,
                "binary": [list(s) for s in self.binary], "polar": [list(s) for s in self.polar],
                "letters": [list(s) for s in self.letters], "revision": self.revision}

    @classmethod
    def parse(cls, value):
        from .errors import ConfigError
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ConfigError("invalid pinned framing; re-run ss lock")
        if value["revision"] != 2 or any(not isinstance(value[k], str) for k in ("name", "prefix_text", "tail_text")):
            raise ConfigError("unsupported pinned framing; re-run ss lock")
        for k in ("binary", "polar", "letters"):
            if not isinstance(value[k], list) or any(not isinstance(s, list) or not s or any(type(t) is not int or t < 0 for t in s) for s in value[k]):
                raise ConfigError("invalid pinned answer slots; re-run ss lock")
        return cls(**{**value, **{k: tuple(map(tuple, value[k])) for k in ("binary", "polar", "letters")}})


@dataclass(frozen=True, slots=True)
class Question:
    kind: str
    text: str
    options: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    guidance: Guidance = Guidance()
    framing: Framing | None = None
    candidate: str | None = None

    @property
    def width(self) -> int:
        return 2 if self.kind == "check" else len(self.options)

    @property
    def interrogative(self) -> bool:
        return self.candidate is not None or self.text.rstrip().endswith("?")


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


class ScoreLedger:
    def __init__(self, scorer):
        self.scorer, self.blocks = scorer, {}

    def __getattr__(self, name):
        return getattr(self.scorer, name)

    def score(self, requests: Sequence[ScoreRequest], **options) -> ScoreBlock:
        block = self.scorer.score(requests, **options)
        row = 0
        for request in requests:
            rows = slice(row, row + len(request.suffixes))
            if not options and all(error is None for error in block.error[rows]) and np.isfinite(block.log_mass[rows]).all():
                self.blocks.setdefault(request, ScoreBlock(block.logp[rows].copy(), block.log_mass[rows].copy(), list(block.error[rows])))
            row = rows.stop
        return block

    def replay(self, requests: Sequence[ScoreRequest], score) -> tuple[ScoreBlock, dict]:
        reused = [self.blocks.get(request) for request in requests]
        missing = [request for request, block in zip(requests, reused) if block is None]
        fresh, row = score(missing) if missing else None, 0
        parts = []
        for request, block in zip(requests, reused):
            if block is None:
                count = len(request.suffixes)
                block = ScoreBlock(fresh.logp[row:row + count], fresh.log_mass[row:row + count], list(fresh.error[row:row + count]))
                row += count
            parts.append(block)
        width = max((part.logp.shape[1] for part in parts), default=0)
        out = ScoreBlock.empty(sum(len(part.error) for part in parts), width)
        row = 0
        for part in parts:
            count = len(part.error)
            out.logp[row:row + count, :part.logp.shape[1]] = part.logp
            out.log_mass[row:row + count] = part.log_mass
            out.error[row:row + count] = part.error
            row += count
        return out, {'requests': len(requests), 'reused': len(requests) - len(missing), 'scored': len(missing),
                     'rows': row, 'scored_rows': sum(len(request.suffixes) for request in missing)}


class Scorer(Protocol):
    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]: ...

    def detokenize(self, tokens: Sequence[int]) -> str: ...

    def score(self, requests: Sequence[ScoreRequest]) -> ScoreBlock: ...


class Judge(Protocol):
    identity: dict
    accelerated: bool
    max_slots: int
    batch_inputs: int
    supports_guidance: bool

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]: ...
