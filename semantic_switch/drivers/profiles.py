from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ..errors import ConfigError, suggest

SUMMATION = 2.0 ** -52
TRUNCATION = ("rejects", "truncates")
BASES = ("documented", "empirical")
PINNING = ("observed-identity", "pinned-snapshot")
UTF8 = "utf8-bytes-upper-bound@1"
ROUNDING = "nearest"
INPUTS = ("text",)


@dataclass(frozen=True, slots=True)
class Precision:
    decimals: int
    basis: str

    def __post_init__(self):
        if isinstance(self.decimals, bool) or not isinstance(self.decimals, int) or not 0 < self.decimals <= 15:
            raise ConfigError("precision decimals must be an integer from 1 to 15")
        if self.basis not in BASES:
            raise ConfigError(f"precision basis must be one of {', '.join(BASES)}")

    def bound(self, width: int) -> float:
        return width * (0.5 * 10.0 ** -self.decimals + SUMMATION)

    def record(self) -> dict:
        return {"decimals": self.decimals, "rounding": ROUNDING, "basis": self.basis, "bound": "width*(0.5*10**-decimals+2**-52)"}


@dataclass(frozen=True, slots=True)
class Budget:
    context: int
    overhead: int
    reserve: int = 1

    def __post_init__(self):
        for name in ("context", "overhead", "reserve"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (name == "context"):
                raise ConfigError(f"token budget {name} must be a {'positive' if name == 'context' else 'nonnegative'} integer")
        if self.capacity < 1:
            raise ConfigError(f"token budget overhead {self.overhead} and reserve {self.reserve} leave no room in a {self.context}-token context")

    @property
    def capacity(self) -> int:
        return self.context - self.overhead - self.reserve

    def record(self) -> dict:
        return {"method": UTF8, "context": self.context, "overhead": self.overhead, "reserve": self.reserve,
                "capacity": self.capacity}


@dataclass(frozen=True, slots=True)
class Fixture:
    path: str
    sha256: str
    responses: int
    records: str
    distributions: int
    widths: tuple[tuple[str, tuple[int, ...]], ...]
    observed: tuple[str, ...]
    lattice_error: float
    mass_error: float

    def record(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "responses": self.responses, "records": self.records,
                "distributions": self.distributions, "widths": {kind: list(sizes) for kind, sizes in self.widths},
                "observed": list(self.observed), "lattice_error": self.lattice_error, "mass_error": self.mass_error}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    revision: int
    codec: str
    kinds: MappingProxyType
    max_questions: int
    max_request_bytes: int
    semantics: str
    normalization: str
    providers: tuple[str, ...] = ("openrouter",)
    models: tuple[str, ...] = ()
    precision: Precision | None = None
    options: str = "letters"
    truncation: str = "rejects"
    budget: Budget | None = None
    evidence: tuple[Fixture, ...] = ()
    observed: tuple[str, ...] = ()
    pinning: str = "observed-identity"
    weights: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "kinds", MappingProxyType(dict(self.kinds)))
        if not isinstance(self.name, str) or not self.name or isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ConfigError("a model profile needs a name and a positive integer revision")
        for kind, (low, high) in self.kinds.items():
            if kind not in ("check", "pick", "rate", "select") or not 2 <= low <= high:
                raise ConfigError(f"profile {self.name}: invalid {kind} option limits ({low}, {high})")
        for name in ("max_questions", "max_request_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"profile {self.name}: {name} must be a positive integer")
        if self.truncation not in TRUNCATION:
            raise ConfigError(f"profile {self.name}: truncation must be one of {', '.join(TRUNCATION)}")
        if self.budget is not None and not isinstance(self.budget, Budget):
            raise ConfigError(f"profile {self.name}: budget must be a Budget")
        if self.truncation == "truncates" and self.budget is None:
            raise ConfigError(f"profile {self.name}: a truncating service needs a tested token budget; requests cannot be shown complete")
        if self.pinning not in PINNING:
            raise ConfigError(f"profile {self.name}: pinning must be one of {', '.join(PINNING)}")
        if (self.pinning == "pinned-snapshot") != (self.weights is not None):
            raise ConfigError(f"profile {self.name}: only a service contract guaranteeing a pinned snapshot names content-pinned weights; "
                              "an alias or observed response model is not proof of frozen weights")
        if self.options not in OPTION_IDS:
            raise ConfigError(f"profile {self.name}: option ids must be one of {', '.join(OPTION_IDS)}")
        if self.precision is not None and self.precision.bound(max(high for _, high in self.kinds.values())) >= 1:
            raise ConfigError(f"profile {self.name}: {self.precision.decimals}-decimal rounding cannot bound the mass of its widest question")

    @property
    def max_slots(self) -> int:
        return max(high for _, high in self.kinds.values())

    def ids(self, width: int) -> tuple[str, ...]:
        return OPTION_IDS[self.options](width)

    def validate(self, questions) -> None:
        if len(questions) > self.max_questions:
            raise ConfigError(f"profile {self.name} takes at most {self.max_questions} questions per request, got {len(questions)}; "
                              "joint grouping is part of execution identity and is not split silently")
        for question in questions:
            if question.kind not in self.kinds:
                raise ConfigError(f"profile {self.name} does not support {question.kind} questions{suggest(question.kind, self.kinds)}")
            low, high = self.kinds[question.kind]
            if not low <= question.width <= high:
                noun = "levels" if question.kind == "rate" else "options"
                raise ConfigError(f"profile {self.name} takes {low} to {high} {noun} for {question.kind}, got {question.width}; "
                                  "choose a local judge or a profile declaring that width")

    @property
    def limit(self) -> int:
        return min(self.max_request_bytes, self.budget.capacity) if self.budget else self.max_request_bytes

    def admission(self) -> dict:
        budget = self.budget.record() if self.budget else {"method": "service-rejects-oversize@1"}
        return {**budget, "truncation": self.truncation, "max_request_bytes": self.max_request_bytes, "limit": self.limit}

    def unsupported(self, size: int) -> str | None:
        if size <= self.limit:
            return None
        bound = f"{UTF8} bound of {self.budget.capacity} tokens after {self.budget.overhead} overhead and {self.budget.reserve} reserved" \
            if self.budget and self.budget.capacity == self.limit else f"{self.max_request_bytes}-byte request limit"
        return f"serialized request is {size} bytes, beyond profile {self.name}'s {bound}; completeness cannot be established, so it is not sent"

    def record(self) -> dict:
        return {"name": self.name, "revision": self.revision, "codec": self.codec, "kinds": {k: list(v) for k, v in sorted(self.kinds.items())},
                "max_questions": self.max_questions, "admission": self.admission(), "semantics": self.semantics,
                "normalization": self.normalization, "precision": self.precision and self.precision.record(), "options": self.options,
                "inputs": list(INPUTS), "observed": list(self.observed),
                "pinning": self.pinning, "weights": self.weights,
                "evidence": [fixture.record() for fixture in self.evidence]}


def letters(width: int) -> tuple[str, ...]:
    if width > 26:
        raise ConfigError(f"letter option ids cover at most 26 options, got {width}")
    return tuple(chr(65 + i) for i in range(width))


def ordinals(width: int) -> tuple[str, ...]:
    return tuple(f"o{i}" for i in range(width))


OPTION_IDS = MappingProxyType({"letters": letters, "ordinal": ordinals})

CHAT = ModelProfile("chat-logprobs", 2, "chat", {"check": (2, 2), "pick": (2, 20), "rate": (2, 20)}, 1, 65536,
                    "answer-token-lower-bound@2:top20", "token-mass-lower-bound@1")

JEV = ModelProfile(
    "jev-1.13-systemone-2dp", 2, "systemone", {"check": (2, 2), "pick": (2, 20), "rate": (2, 10)}, 16, 65536,
    "provider-distribution@1", "renormalize-within-rounding-bound@1", models=("typesafe/jev-1.13",),
    precision=Precision(2, "empirical"), observed=("typesafe/jev-1.13-20260917@TypeSafe",),
    evidence=(
        Fixture("bench/results/openrouter-jev-policy-v3.json", "a1682f3e637b0d7acbae7639ced6eb6c461524dd289a8d51bacb79f9787ed397", 56,
                "b3d16bbe040fb1d7823adeacc94ddd108fce7f4cda011a8b316d8d2f18ef2d2b", 90, (("choice", (2, 3)), ("score", (3,))),
                ("typesafe/jev-1.13-20260917@TypeSafe",), 0.0, 0.0),
        Fixture("bench/results/semantic-decisions-jev-v5.json", "9e1a9a042951a34476c3cf1aaaae2a4b76826a5dda4befc5a052c42f737cacbd", 76,
                "93e873ce4cbe43a138fa1c7aa030ff0219578a94fd52a088e97b8dc9fad2ad31", 90, (("choice", (2, 3, 4)),),
                ("typesafe/jev-1.13-20260917@TypeSafe",), 1.1102230246251565e-16, 0.009999999999999898)))

LUNA = ModelProfile(
    "luna-decisions-systemone-2dp", 2, "systemone", {"check": (2, 2), "pick": (2, 20), "rate": (2, 10)}, 16, 65536,
    "provider-distribution@1", "renormalize-within-rounding-bound@1", models=("openai/gpt-6-luna-decisions",),
    precision=Precision(2, "empirical"), observed=("openai/gpt-6-luna-decisions-20261006@OpenAI",),
    evidence=(
        Fixture("bench/results/semantic-decisions-luna-v5.json", "387eafa8d783fa8ae30cd0953363b9c9dad79d96747f920333582bd06f813b8c", 66,
                "c1f1c2a8c59992f692f12c76652bd31bf04330e028c87b8d42573ff39cc11091", 80, (("choice", (2, 3, 4)),),
                ("openai/gpt-6-luna-decisions-20261006@OpenAI",), 0.0, 0.0),
        Fixture("bench/results/luna-smoke.json", "0f5320df4cc250310c6a5fa37395c120866757ed25d11d2c73b0b156daff1f1b", 10,
                "06a8fcba4964c1dfc66e31d0f8a9b31d8f62aad302558878226b33a4f088dbe5", 8, (("choice", (3,)), ("score", (3,))),
                ("openai/gpt-6-luna-decisions-20261006@OpenAI",), 0.0, 1.1102230246251565e-16)))

PROFILES = MappingProxyType({profile.name: profile for profile in (CHAT, JEV, LUNA)})
MODELS = MappingProxyType({
    ("openrouter", "typesafe/jev-1.13"): JEV,
    ("openrouter", "openai/gpt-6-luna-decisions"): LUNA,
    **{("openrouter", model): CHAT for model in ("google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it", "qwen/qwen3.7-flash",
                                                  "nvidia/nemotron-3.5-lightning")}})
GENERIC = MappingProxyType({"chat": CHAT})


def resolve(provider: str, model: str, *, codec: str | None = None, profile=None) -> ModelProfile:
    if profile is not None:
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ConfigError(f"unknown model profile {profile!r}{suggest(profile, PROFILES)}; built-in: {', '.join(PROFILES)}")
            profile = PROFILES[profile]
        if not isinstance(profile, ModelProfile):
            raise ConfigError("profile must be a built-in profile name or a ModelProfile")
        if provider not in profile.providers:
            raise ConfigError(f"profile {profile.name} declares providers {', '.join(profile.providers)}, not {provider}")
        if profile.models and model not in profile.models:
            raise ConfigError(f"profile {profile.name} declares models {', '.join(profile.models)}, not {model}")
    else:
        profile = MODELS.get((provider, model))
        if profile is None:
            if codec in GENERIC:
                profile = GENERIC[codec]
            elif codec is None:
                raise ConfigError(f"{provider}:{model} has no built-in profile; select one explicitly with codec='chat' "
                                  f"(--setting codec=chat) or profile=NAME (built-in: {', '.join(PROFILES)})")
            else:
                raise ConfigError(f"codec {codec!r} needs a declared profile with its serialization precision and limits for {model}; pass profile=ModelProfile(...)")
    if codec is not None and codec != profile.codec:
        raise ConfigError(f"profile {profile.name} uses codec {profile.codec!r}, not {codec!r}")
    return profile
