from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping
import math

from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class Selection:
    selected_by: str = 'unknown'
    selection_id: str | None = None
    frame_id: str | None = None
    unit: str = 'observation'
    window: str | None = None
    inclusion_probability: float | None = None
    independent: bool = False

    @classmethod
    def parse(cls, value=None, default='unknown'):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls(default)
        if not isinstance(value, Mapping) or set(value) - cls.__dataclass_fields__.keys():
            raise ConfigError('selection: expected a supported metadata object')
        data = dict(value)
        data.setdefault('selected_by', default)
        return cls(**data)

    def __post_init__(self):
        if self.selected_by not in ('random', 'uncertain', 'manual', 'unknown') or self.unit not in ('observation', 'family'):
            raise ConfigError('selection: invalid method or unit')
        for name in ('selection_id', 'frame_id', 'window'):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ConfigError(f'selection.{name}: expected nonempty text')
        p = self.inclusion_probability
        if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p <= 1):
            raise ConfigError('selection.inclusion_probability: expected finite probability in (0,1]')
        if not isinstance(self.independent, bool):
            raise ConfigError('selection.independent: expected boolean')

    def record(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def occurrences(values):
    if not isinstance(values, (list, tuple)):
        raise ConfigError('selections: expected a sequence')
    return tuple(dict.fromkeys(Selection.parse(v) for v in values))


def history(values=(), value=None, default="unknown"):
    rows = occurrences(values)
    if value is not None:
        rows = tuple(dict.fromkeys((*rows, Selection.parse(value))))
    return rows or (Selection(default),)
