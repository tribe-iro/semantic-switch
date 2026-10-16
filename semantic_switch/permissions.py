from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .errors import ConfigError

PERMISSIONS = ("allowed", "forbidden", "unknown")
RESTRICTED = MappingProxyType({"typesafe/jev": "TypeSafe agreement §2.3(b) keeps Jev outputs out of training data"})
FIELDS = ("permission", "model", "provider", "contract", "revision", "use")


def restriction(model: str) -> str | None:
    name = model.lower().removeprefix("openrouter:")
    return next((reason for prefix, reason in RESTRICTED.items() if name.startswith(prefix)), None)


@dataclass(frozen=True, slots=True)
class TrainingUse:
    permission: str
    model: str
    provider: str | None
    contract: str
    revision: str
    use: str

    def __post_init__(self):
        if self.permission not in PERMISSIONS:
            raise ConfigError(f"training use permission must be one of {', '.join(PERMISSIONS)}")
        for name in ("model", "contract", "revision", "use"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ConfigError(f"training use {name} must be nonempty text")
        if self.provider is not None and (not isinstance(self.provider, str) or not self.provider.strip() or "@" in self.provider):
            raise ConfigError("training use provider must be nonempty text without '@' or None")
        reason = restriction(self.model)
        if reason and self.permission == "allowed":
            raise ConfigError(f"{self.model} cannot be declared allowed for training use: {reason}")

    @property
    def route(self) -> str:
        return f"{self.model}@{self.provider}" if self.provider else self.model

    def record(self) -> dict:
        return {name: getattr(self, name) for name in FIELDS}

    @classmethod
    def parse(cls, value) -> TrainingUse:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != set(FIELDS):
            raise ConfigError(f"training use declaration needs exactly {', '.join(FIELDS)}")
        return cls(**value)

    @property
    def digest(self) -> str:
        from .compiler import digest
        return digest(self.record())

    def require(self, observed: str) -> None:
        if not isinstance(observed, str) or not observed:
            raise ConfigError("training labels need the observed producer route")
        reason = restriction(observed)
        if reason:
            raise ConfigError(f"{observed} cannot label training data: {reason}")
        if self.permission != "allowed":
            raise ConfigError(f"{self.route} is declared {self.permission} for training use under {self.contract}@{self.revision}; it cannot produce training labels")
        if observed != self.route:
            raise ConfigError(f"observed producer route {observed} differs from the declared {self.route}; declare the observed route explicitly")


def admit(addressed) -> int:
    from .examples import LabelSource
    count = 0
    for where, history in addressed:
        for source in map(LabelSource.parse, history):
            _admitted(source, where)
        count += 1
    return count


def _admitted(source, where: str) -> None:
    reason = restriction(source.producer) or (source.training_use and restriction(source.training_use.model))
    if reason:
        raise ConfigError(f"{where}: the {source.derivation} label from {source.producer} in its history cannot label training data, whatever its later "
                          f"verification or replacement: {reason}")
    if source.derivation != "judged":
        return
    if source.training_use is None:
        raise ConfigError(f"{where}: the judged label from {source.producer} declares no training use; unknown permission cannot produce training labels. "
                          "Acquire it again with semantic-learn collect --training-use DECLARATION, or record a computed or authored label")
    if source.training_use.permission != "allowed":
        raise ConfigError(f"{where}: {source.training_use.route} is declared {source.training_use.permission} for training use under "
                          f"{source.training_use.contract}@{source.training_use.revision}; its labels cannot be used for fitting or selection")
    if source.training_use.route != source.producer:
        raise ConfigError(f"{where}: the judged label from {source.producer} carries a training use declaration for {source.training_use.route}; "
                          "a declaration covers only the producer route it names")
