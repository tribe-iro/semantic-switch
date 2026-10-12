from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    data: object
    label: int
    dist: object = None
    mass: float | None = None
    pending: frozenset = frozenset()
    error: str | None = None
    skip: str | None = None
    group: str | None = None
    observed_models: tuple = ()

    def __iter__(self):
        return iter((self.dist, self.mass, self.pending, self.label))
