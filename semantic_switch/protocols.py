from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Sequence as TokenSequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from .guidance import Guidance

MAX_INPUT_TOKENS = 1024
MAX_PREFIX_TOKENS = 1024
MAX_QUESTION_TOKENS = 1536

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
class Catalog:
    entries: tuple[tuple[str, str], ...]
    revision: str = "criteria-catalog@1"

    @property
    def text(self) -> str:
        return "".join(f"[{key}] {criterion}\n" for key, criterion in self.entries)


@dataclass(frozen=True, slots=True)
class Question:
    kind: str
    text: str
    options: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    guidance: Guidance = Guidance()
    framing: Framing | None = None
    candidate: str | None = None
    catalog: Catalog | None = None
    entry: str | None = None

    @property
    def width(self) -> int:
        return 2 if self.kind == "check" else len(self.options)

    @property
    def interrogative(self) -> bool:
        return self.candidate is not None or self.text.rstrip().endswith("?")


class Tokens(TokenSequence):
    __slots__ = ('_data', '_array')

    def __new__(cls, values=()):
        return values if type(values) is cls else object.__new__(cls)

    def __init__(self, values=()):
        if values is self:
            return
        def checked():
            for value in values:
                if type(value) is not int:
                    raise TypeError('token IDs must be integers')
                yield value
        data = array('i', checked()).tobytes()
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_array', np.frombuffer(data, dtype=np.int32))

    @classmethod
    def from_array(cls, values):
        result = object.__new__(cls)
        values = np.asarray(values)
        if values.ndim != 1 or values.dtype.kind not in 'iu' or values.size and (values.min() < 0 or values.max() > 2147483647):
            raise ValueError('tokens require a one-dimensional nonnegative int32 sequence')
        data = values.astype(np.int32, copy=False).tobytes()
        object.__setattr__(result, '_data', data)
        object.__setattr__(result, '_array', np.frombuffer(data, dtype=np.int32))
        return result

    def __setattr__(self, name, value):
        raise AttributeError('token segments are immutable')

    def __hash__(self):
        return hash(self._data)

    def __eq__(self, other):
        return isinstance(other, Tokens) and self._data == other._data

    def __lt__(self, other):
        return self._data < other._data if isinstance(other, Tokens) else NotImplemented

    def __len__(self):
        return len(self._data) // 4

    def __iter__(self):
        return map(int, self._array)

    def __getitem__(self, index):
        value = self._array[index]
        return Tokens.from_array(value) if isinstance(index, slice) else int(value)

    def __add__(self, other):
        other = Tokens(other)
        result = object.__new__(type(self))
        data = self._data + other._data
        object.__setattr__(result, '_data', data)
        object.__setattr__(result, '_array', np.frombuffer(data, dtype=np.int32))
        return result

    def __repr__(self):
        return f'Tokens({list(self)!r})'

    def __reduce__(self):
        return Tokens, (tuple(self),)

    @property
    def array(self):
        return self._array

    @property
    def nbytes(self):
        return len(self._data)


@dataclass(frozen=True, slots=True, eq=False)
class ScoreRequest:
    prefix: tuple[int, ...]
    state: tuple[int, ...]
    suffixes: tuple[tuple[int, ...], ...]
    slots: tuple[tuple[tuple[int, ...], ...], ...]
    _hash: int = field(init=False, repr=False)

    def __post_init__(self):
        prefix, state, suffixes = Tokens(self.prefix), Tokens(self.state), tuple(map(Tokens, self.suffixes))
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "suffixes", suffixes)
        object.__setattr__(self, "_hash", hash((prefix, state, suffixes, self.slots)))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ScoreRequest):
            return NotImplemented
        return self._hash == other._hash and self.prefix == other.prefix and self.state == other.state \
            and self.suffixes == other.suffixes and self.slots == other.slots


@dataclass(slots=True)
class ScoreBlock:
    logp: np.ndarray
    log_mass: np.ndarray
    error: list

    @classmethod
    def empty(cls, rows: int, width: int) -> ScoreBlock:
        return cls(np.full((rows, width), -np.inf, dtype=np.float32),
                   np.full(rows, np.nan, dtype=np.float32), [None] * rows)


class Bounded:
    def __init__(self, max_bytes: int):
        self.max_bytes, self.bytes, self.evictions, self._entries = max_bytes, 0, 0, OrderedDict()

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries

    def get(self, key, default=None):
        entry = self._entries.get(key)
        if entry is None:
            return default
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key, value, nbytes: int):
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if nbytes > self.max_bytes:
            return value
        while self._entries and self.bytes + nbytes > self.max_bytes:
            self.bytes -= self._entries.popitem(last=False)[1][1]
            self.evictions += 1
        self._entries[key] = (value, nbytes)
        self.bytes += nbytes
        return value

    def clear(self):
        self._entries.clear()
        self.bytes = 0


class ScoreLedger:
    def __init__(self, scorer, binding, population: Sequence[ScoreRequest]):
        self.scorer, self.binding, self.population, self.blocks, self._bound = scorer, binding, frozenset(population), {}, None

    def operation(self):
        return getattr(self.scorer, "operation", nullcontext)()

    def capacity(self) -> dict:
        return getattr(self.scorer, "capacity", dict)()

    def tokenizer_identity(self):
        if not hasattr(self.scorer, "tokenizer_identity"):
            from .errors import ConfigError
            raise ConfigError("native training export requires a scorer tokenizer identity")
        return self.scorer.tokenizer_identity()

    def detokenize(self, tokens: Sequence[int]) -> str:
        return self.scorer.detokenize(tokens)

    @property
    def ctx(self):
        return getattr(self.scorer, "ctx", None)

    @property
    def inputs(self) -> int:
        return getattr(self.scorer, "inputs", 1)

    @property
    def prefix_slots(self):
        return getattr(self.scorer, "prefix_slots", None)

    @property
    def stats(self) -> dict:
        return getattr(self.scorer, "stats", {})

    @property
    def shared(self):
        return getattr(self.scorer, "shared", None)

    @property
    def namespace(self):
        return getattr(self.scorer, "namespace", None)

    @property
    def resident(self):
        return getattr(self.scorer, "resident", None)

    @property
    def binding_identity(self):
        return getattr(self.scorer, "binding_identity", None)

    def invalidate(self):
        self.blocks, self._bound = {}, None
        self.scorer.invalidate()

    def release(self):
        self.blocks, self.population, self._bound = {}, frozenset(), None

    def score(self, requests: Sequence[ScoreRequest], *, reference: bool = False) -> ScoreBlock:
        if reference:
            return self.scorer.score(requests, reference=True)
        binding = self.binding()
        block = self.scorer.score(requests)
        rows = sum(len(request.suffixes) for request in requests)
        width = max((len(slots) for request in requests for slots in request.slots), default=0)
        if block.logp.ndim != 2 or block.logp.shape[0] != rows or block.logp.shape[1] < width or block.log_mass.shape != (rows,) or len(block.error) != rows:
            from .errors import ConfigError
            raise ConfigError("native scorer returned incomplete ordered scores")
        if self.binding() != binding:
            return block
        if self._bound != binding:
            self.blocks, self._bound = {}, binding
        row = 0
        for request in requests:
            span = slice(row, row + len(request.suffixes))
            row = span.stop
            key = (binding, request)
            if key in self.blocks or request not in self.population or any(error is not None for error in block.error[span]) \
                    or not np.isfinite(block.log_mass[span]).all() \
                    or not all(np.isfinite(block.logp[n, :len(slots)]).all() for n, slots in zip(range(span.start, span.stop), request.slots)):
                continue
            self.blocks[key] = ScoreBlock(block.logp[span].copy(), block.log_mass[span].copy(), list(block.error[span]))
        return block

    def replay(self, requests: Sequence[ScoreRequest], score) -> tuple[ScoreBlock, dict]:
        binding = self.binding()
        reused = [self.blocks.get((binding, request)) for request in requests]
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

    def score(self, requests: Sequence[ScoreRequest], *, reference: bool = False) -> ScoreBlock: ...


class Judge(Protocol):
    identity: dict
    accelerated: bool
    max_slots: int
    batch_inputs: int
    supports_guidance: bool
    normalization: str

    def score_many(self, items: Sequence[tuple[str, Sequence[Question]]]) -> list[Scored]: ...
