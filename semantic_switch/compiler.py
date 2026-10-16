from __future__ import annotations

import hashlib
from collections import Counter
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, field_serializer, model_validator

from .context import Bm25
from .guidance import bind, resolve_catalog
from .errors import ConfigError, suggest
from .judgment import MAX_OPTIONS, Judgment, Label, Policy, authored_budget, authored_levels, authored_text, authored_bool, authored_strings
from .kernel import F, T, U, Card, Exact, Node, Sem, atom, atoms_of, conj, const, disj, k_and, k_not, k_or, neg
from .presentation import candidate, definition
from .protocols import Catalog
from .results import freeze, thaw

NO_OPTION = "\u2205"
MAX_FACTS = 32
DEFAULT_BINDING = "judge"
BINDING_NAME = re.compile(r"[A-Za-z_][\w.-]*")


def binding_name(value):
    if value is not None and (not isinstance(value, str) or not BINDING_NAME.fullmatch(value)):
        raise ValueError("binding names a runtime model role: a letter or underscore, then letters, digits, '_', '.' or '-'")
    return value


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldSpec(Spec):
    type: Literal["text", "number", "bool", "enum", "json"] = "text"
    required: bool = False
    max_chars: int | None = Field(None, gt=0, strict=True)
    truncate: Literal["end", "middle"] = "end"
    values: list[str] | None = None
    author: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _enum_values(self):
        if self.type == "enum" and not self.values:
            raise ValueError("an enum field needs values")
        return self


class OptionSpec(Spec):
    description: str
    requires: str | None = None
    excludes: list[str] = []
    aliases: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _label(cls, value):
        if isinstance(value, dict):
            Label.of({k: v for k, v in value.items() if k != "requires"})
        return value

    @property
    def label(self) -> Label:
        return Label(self.description, tuple(self.excludes), tuple(self.aliases))


class FactRef(Spec):
    fact: str


Claim = str | FactRef


class CardSpec(Spec):
    id: str
    title: str = ""
    requires: str | None = None
    aliases: list[str] = []
    applies_when: list[Claim]
    excludes: list[Claim] = []
    supersedes: list[str] = []
    source: dict = {}
    payload: Any = None
    guidance: list[str] | None = None

    @field_validator("applies_when", "excludes", mode="before")
    @classmethod
    def _listify(cls, v):
        return [v] if isinstance(v, (str, dict)) else v


class ExampleSpec(Spec):
    input: dict
    value: Any


def _cycle(cards: list[CardSpec]) -> list[str] | None:
    edges = {c.id: c.supersedes for c in cards}
    done, active = set(), set()
    for root in edges:
        if root in done:
            continue
        path, stack = [root], [(root, iter(edges[root]))]
        active.add(root)
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop(); path.pop(); active.remove(node); done.add(node)
            elif child in active:
                return path[path.index(child):] + [child]
            elif child not in done:
                active.add(child); path.append(child); stack.append((child, iter(edges[child])))
    return None


_COMMON = {"guidance", "reads", "excludes", "min_score", "target", "examples", "binding"}
_KIND_FIELDS = {
    "check": _COMMON | {"check", "unstated"},
    "pick": _COMMON | {"pick", "options", "allow_none", "focus"},
    "rate": _COMMON | {"rate", "levels"},
    "select": {"guidance", "select", "reads", "catalog", "cards", "max_claims", "exclusive", "binding"},
    "labels": {"guidance", "select", "reads", "labels", "excludes", "unstated", "min_score", "target", "max_claims", "binding"},
}


def _kind(fs) -> str:
    kind = next(k for k in ("check", "pick", "rate", "select") if getattr(fs, k) is not None)
    return "labels" if kind == "select" and fs.labels is not None else kind


def _specs(options) -> dict:
    return {k: v if isinstance(v, OptionSpec) else OptionSpec(description=v) for k, v in options.items()}


def _candidates(fs) -> list[str]:
    return list(fs.labels) if fs.labels is not None else [c.id for c in fs.cards]


class FactSpec(Spec):
    guidance: list[str] | None = None
    check: str | None = None
    pick: str | None = None
    rate: str | None = None
    select: str | None = None
    excludes: list[str] = []
    options: dict[str, str | OptionSpec] = {}
    labels: dict[str, str | OptionSpec] | None = None
    levels: list[str] = []
    allow_none: bool = False
    unstated: bool = False
    focus: Literal["mentions"] | None = None
    catalog: str | None = None
    cards: list[CardSpec] = []
    max_claims: int | None = None
    exclusive: bool = False
    min_score: float | None = None
    reads: list[str] | None = None
    target: float = 0.95
    examples: list[ExampleSpec] = []
    binding: str | None = None

    @field_validator("binding", mode="before")
    @classmethod
    def _binding(cls, value):
        return binding_name(value)

    @field_validator("check", "pick", "rate", "select", mode="before")
    @classmethod
    def _criterion(cls, value):
        return None if value is None else authored_text(value)

    @field_validator("max_claims", mode="before")
    @classmethod
    def _budget(cls, value):
        return authored_budget(value)

    @field_validator("unstated", "allow_none", "exclusive", mode="before")
    @classmethod
    def _boolean(cls, value, info):
        return authored_bool(value, info.field_name)

    @field_validator("excludes", "levels", mode="before")
    @classmethod
    def _strings(cls, value, info):
        return list(authored_strings(value, info.field_name))

    @field_validator("options", "labels", mode="before")
    @classmethod
    def _descriptions(cls, value):
        if isinstance(value, dict):
            for key, description in value.items():
                authored_text(key, "label key")
                if not isinstance(description, (dict, OptionSpec)):
                    Label.of(description)
        return value

    @field_validator("min_score", "target", mode="before")
    @classmethod
    def _finite_policy(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("policy values must be finite numbers")
        return value

    @model_validator(mode="after")
    def _one_kind(self):
        kinds = [k for k in ("check", "pick", "rate", "select") if getattr(self, k) is not None]
        if len(kinds) != 1:
            raise ValueError("a fact needs exactly one of check, pick, rate, select")
        kind = _kind(self)
        inactive = self.model_fields_set - _KIND_FIELDS[kind]
        if inactive:
            name = sorted(inactive)[0]
            raise ValueError(f"{name} does not apply to {'select labels' if kind == 'labels' else kind}; use a named check fact for claim policies")
        if self.focus and self.pick is None:
            raise ValueError("focus applies to pick facts only")
        if kind == "labels" and not self.labels:
            raise ValueError("select labels must be a nonempty mapping")
        if kind == "select":
            self.max_claims = 16 if self.max_claims is None else self.max_claims
            if not self.cards:
                raise ValueError("select needs labels, cards or a catalog with at least one card")
            ids = [c.id for c in self.cards]
            dupes = sorted(i for i, count in Counter(ids).items() if count > 1)
            if dupes:
                raise ValueError(f"duplicate card ids {dupes}")
            unknown = sorted({s for c in self.cards for s in c.supersedes} - set(ids))
            if unknown:
                raise ValueError(f"supersedes names unknown cards {unknown}")
            cycle = _cycle(self.cards)
            if cycle:
                raise ValueError(f"supersedes cycle: {' → '.join(cycle)}")
            if self.examples:
                raise ValueError("select facts take no examples")
        if self.pick is not None and len(self.options) < 2 - self.allow_none:
            raise ValueError("pick needs options")
        if self.allow_none and "none" in self.options:
            raise ValueError("'none' is reserved for allow_none; rename that option")
        if len(self.options) + self.allow_none > MAX_OPTIONS:
            raise ValueError(f"a pick fact can have at most {MAX_OPTIONS} options (including none)")
        if self.rate is not None:
            authored_levels(self.levels)
        if not 0.5 <= self.target <= 1:
            raise ValueError("target must be within [0.5, 1]")
        if self.min_score is not None and not 0.5 < self.min_score <= 1:
            raise ValueError("min_score must be above 0.5 and at most 1")
        return self


def _payload(value):
    from .context import ContextError, json_value
    try:
        return json_value(value)
    except ContextError as error:
        raise ValueError(str(error)) from None


class RuleSpec(Spec):
    id: str | None = None
    when: str
    then: Any = None

    @field_serializer("then")
    def _serialized_action(self, value):
        return thaw(value)

    @field_validator("then", mode="before")
    @classmethod
    def _action(cls, value):
        return _payload(value)


class TestSpec(Spec):
    input: dict
    expect: Any = None
    expect_status: Literal["ok", "none", "uncertain", "needs_input", "failed"] | None = None
    expect_rule: Any = None
    expect_facts: dict[str, Any] | None = None
    expect_complete: dict[str, bool] | None = None

    @model_validator(mode="after")
    def _expectation(self):
        if not self.model_fields_set - {"input"}:
            raise ValueError("test needs an expectation")
        return self


class SwitchSpec(Spec):
    guidance: dict[str, dict[str, str]] = {}
    name: str
    fields: dict[str, FieldSpec] = {}
    reads: list[str] = []
    facts: dict[str, FactSpec] = Field(max_length=MAX_FACTS)
    rules: list[RuleSpec] | dict[str, str] = []
    otherwise: Any = None
    resolve: Literal["first", "all"] = "first"
    exact_override: bool = True
    tests: list[TestSpec] = []
    binding: str = DEFAULT_BINDING
    retired: list[str] = []
    presentation: Literal["inline", "catalog"] = "inline"

    @model_validator(mode="after")
    def _retired(self):
        clashes = sorted(set(self.retired) & set(self.facts))
        if clashes or len(set(self.retired)) != len(self.retired):
            raise ValueError(f"retired names are distinct facts this task no longer asks; {clashes or self.retired} "
                             f"{'are current facts' if clashes else 'repeat'}")
        return self

    @field_validator("binding", mode="before")
    @classmethod
    def _binding(cls, value):
        return binding_name(value)

    @field_serializer("otherwise")
    def _serialized_otherwise(self, value):
        return thaw(value)

    @field_validator("otherwise", mode="before")
    @classmethod
    def _fallback(cls, value):
        return _payload(value)


@dataclass(frozen=True, slots=True)
class TestPlan:
    input: object
    expect: object = None
    expect_status: str | None = None
    expect_rule: object = None
    expect_facts: object = None
    expect_complete: object = None
    model_fields_set: frozenset = frozenset()

    @classmethod
    def of(cls, spec):
        return cls(freeze(spec.input), freeze(spec.expect), spec.expect_status, freeze(spec.expect_rule),
                   freeze(spec.expect_facts), freeze(spec.expect_complete), frozenset(spec.model_fields_set))


@dataclass(frozen=True, slots=True)
class FieldPlan:
    type: str
    required: bool
    max_chars: int | None
    truncate: str
    values: tuple | None
    author: str | None
    source: str | None

    @classmethod
    def of(cls, spec):
        return cls(spec.type, spec.required, spec.max_chars, spec.truncate,
                   tuple(spec.values) if spec.values is not None else None, spec.author, spec.source)


@dataclass(frozen=True, slots=True)
class CardPlan:
    id: str
    title: str
    requires: Node | None
    aliases: tuple[str, ...]
    claims: tuple[int, ...]
    supersedes: tuple[int, ...]
    base: Node
    has: Node
    raw: dict = field(compare=False)


@dataclass(frozen=True, slots=True)
class FactPlan:
    name: str
    judgment: Judgment
    requires: tuple
    allow_none: bool
    reads: tuple[str, ...]
    target: float
    policy: Policy
    meaning_rev: str
    examples: tuple
    aliases: tuple = ()
    focus: str | None = None
    cards: tuple = ()
    max_claims: int = 16
    exclusive: bool = False
    hidden: bool = False
    ranking: object = field(default=None, compare=False)
    context_reads: tuple[str, ...] = ()
    incoming: tuple = ()
    claim_cards: object = field(default=None, compare=False)
    member: tuple[str, str] | None = None
    labels: bool = False
    binding: str | None = None
    composition_rev: str = ""
    execution_rev: str = ""

    @property
    def kind(self) -> str:
        return "select" if self.cards else self.judgment.kind

    @property
    def text(self) -> str:
        return self.judgment.text

    @property
    def path(self) -> tuple[str, ...]:
        return self.member or (self.name,)

    @property
    def shown(self) -> str:
        return f'{self.member[0]} has "{self.member[1]}"' if self.member else self.text if self.hidden else self.name

    @property
    def keys(self) -> tuple[str, ...]:
        return self.judgment.keys

    @property
    def descriptions(self) -> tuple[str, ...]:
        return self.judgment.descriptions

    @property
    def excludes(self) -> tuple[str, ...]:
        return self.judgment.excludes

    @property
    def unstated(self) -> bool:
        return self.judgment.unstated

    def index(self, value) -> int:
        return self.judgment.index(value, self.name)

    def value(self, i: int):
        return self.judgment.value(i)


@dataclass(frozen=True, slots=True)
class RulePlan:
    id: str
    expr: Node
    then: Any
    when: str
    facts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Plan:
    name: str
    fields: dict
    facts: tuple[FactPlan, ...]
    fact_index: dict
    atoms: tuple
    atom_keys: tuple[str, ...]
    atom_fact: tuple[int, ...]
    exact_atoms: tuple[int, ...]
    fact_atoms: tuple
    rules: tuple[RulePlan, ...]
    otherwise: Any
    first: bool
    exact_override: bool
    tests: tuple
    switch_rev: str
    claim_of: dict
    spec: dict = field(compare=False)
    binding: str = DEFAULT_BINDING
    execution_rev: str = ""
    catalog: Catalog | None = field(default=None, compare=False)
    catalog_ids: dict = field(default_factory=dict, compare=False)

    def binding_of(self, f: int) -> str:
        return self.facts[f].binding or self.binding

    @property
    def retired(self) -> frozenset:
        return frozenset(self.spec.get("retired", ()))

    @property
    def targets(self) -> dict:
        return {fact.name: self.binding_of(f) for f, fact in enumerate(self.facts) if not fact.cards}

    @property
    def bindings(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.targets.values())))

    def questions(self, binding: str) -> dict:
        return {fact.shown: fact.judgment.question() for f, fact in enumerate(self.facts) if not fact.cards and self.binding_of(f) == binding}

    def rebound(self, binding: str) -> Plan:
        return replace(self, binding=binding_name(binding))

    def scoped(self, addresses, binding: str) -> Plan:
        binding = binding_name(binding)
        chosen = {self.fact_index[self.at(*address).name] for address in addresses}
        for k in sorted(chosen):
            fact = self.facts[k]
            if k not in self.observed:
                raise ConfigError(f"{'/'.join(fact.path)} is not a directly scored target; scope a check/pick/rate fact or a labels member")
            for g in sorted(self.claim_of.get(k, ())):
                if (fact.member is None or fact.member[0] != self.facts[g].name) and self.binding_of(g) != binding:
                    raise ConfigError(f"{'/'.join(fact.path)} is a shared semantic atom of {self.facts[g].name}, which answers under {self.binding_of(g)!r}; "
                                      f"binding it to {binding!r} alone would give one claim two scores: scope every fact sharing it together")
        return replace(self, facts=tuple(replace(fact, binding=binding) if k in chosen and self.binding_of(k) != binding else fact
                                         for k, fact in enumerate(self.facts)))

    def fact(self, name: str) -> FactPlan:
        if name not in self.fact_index:
            raise ConfigError(f"unknown fact {name!r}{suggest(name, [n for n in self.fact_index if not n.startswith('~')])}")
        return self.facts[self.fact_index[name]]

    @property
    def visible(self) -> tuple[FactPlan, ...]:
        return tuple(f for f in self.facts if not f.hidden)

    @property
    def observed(self) -> tuple[int, ...]:
        return tuple(f for f, fact in enumerate(self.facts) if fact.member or not (fact.hidden or fact.cards))

    def at(self, name: str, member: str | None = None) -> FactPlan:
        fact = self.fact(name)
        if member is None:
            return fact
        keys = [card.id for card in fact.cards] if fact.labels else []
        if member not in keys:
            raise ConfigError(f"{member!r} is not a label of {name}{suggest(str(member), keys)}")
        return self.facts[fact.cards[keys.index(member)].claims[0]]


HEX = frozenset("0123456789abcdef")


def is_digest(value) -> bool:
    return type(value) is str and len(value) == 64 and HEX.issuperset(value)


def _typed(value):
    from .protocols import Tokens
    if isinstance(value, Tokens):
        return list(value)
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return dict(value)
    if type(value).__module__ == "numpy":
        import numpy
        if isinstance(value, numpy.ndarray):
            if value.dtype.kind not in "biuf" or value.dtype.kind == "f" and not numpy.isfinite(value).all():
                raise ValueError(f"digest needs a finite boolean or numeric array, not dtype {value.dtype}")
            data = numpy.ascontiguousarray(value)
            return {"ndarray": data.dtype.str, "shape": list(value.shape), "sha256": hashlib.sha256(memoryview(data).cast("B")).hexdigest()}
        if isinstance(value, numpy.generic) and value.dtype.kind in "biuf":
            return value.item()
    raise TypeError(f"digest covers finite JSON and typed numeric arrays, not {type(value).__name__}")


_CANONICAL = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False, default=_typed)


def canonical_bytes(obj) -> bytes:
    return _CANONICAL.encode(obj).encode()


def digest(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


_TOKEN = re.compile(r"""\s*(?:(?P<str>"[^"]*"|'[^']*')|(?P<num>-?\d+(?:\.\d+)?)|(?P<op>==|!=|<=|>=|<|>|\(|\)|\[|\]|,)|(?P<name>[A-Za-z_][\w.]*))""")
_KEYWORDS = {"and", "or", "not", "in", "exists", "mentions", "has", "is", "empty"}


def tokenize(src: str) -> list[tuple[str, Any]]:
    out, pos = [], 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m or m.end() == pos:
            if src[pos:].strip():
                raise ConfigError(f"cannot parse {src[pos:]!r}")
            break
        pos = m.end()
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "str":
            out.append(("lit", text[1:-1]))
        elif kind == "num":
            out.append(("lit", float(text) if "." in text else int(text)))
        elif kind == "name" and text in ("true", "false"):
            out.append(("lit", text == "true"))
        elif kind == "name" and text in _KEYWORDS:
            out.append(("kw", text))
        else:
            out.append((kind, text))
    return out


def _flatten(node: Node) -> Node:
    kind = node[0]
    if kind == "not":
        return ("not", _flatten(node[1]))
    if kind not in ("and", "or"):
        return node
    parts = []
    for child in map(_flatten, node[1]):
        parts.extend(child[1] if child[0] == kind else (child,))
    return (kind, tuple(parts))


class Builder:
    def __init__(self, spec: SwitchSpec, facts: dict[str, int], fact_specs: list[FactSpec]):
        self.spec = spec
        self.facts = facts
        self.fact_specs = fact_specs
        self.card_ids = [{key: k for k, key in enumerate(_candidates(fact))} for fact in fact_specs]
        self.atoms: list = []
        self.keys: list[str] = []
        self.index: dict = {}

    def register(self, a, key: str) -> Node:
        if a not in self.index:
            self.index[a] = len(self.atoms)
            self.atoms.append(a)
            self.keys.append(key)
        return atom(self.index[a])

    def parse(self, src: str, fields_only: bool = False) -> Node:
        self.src = src
        self.toks = tokenize(src)
        self.pos = 0
        self.fields_only = fields_only
        node = self.expr()
        if self.pos != len(self.toks):
            raise ConfigError(f"unexpected {self.toks[self.pos][1]!r}")
        return node if fields_only else self.lower(node)

    def peek(self, kind=None, text=None):
        if self.pos >= len(self.toks):
            return None
        k, t = self.toks[self.pos]
        return (k, t) if (kind is None or k == kind) and (text is None or t == text) else None

    def take(self, kind=None, text=None):
        tok = self.peek(kind, text)
        if tok is None:
            raise ConfigError(f"expected {text or kind} in {self.src!r}")
        self.pos += 1
        return tok[1]

    def expr(self) -> Node:
        parts = [self.conj()]
        while self.peek("kw", "or"):
            self.take()
            parts.append(self.conj())
        return parts[0] if len(parts) == 1 else disj(*parts)

    def conj(self) -> Node:
        parts = [self.unary()]
        while self.peek("kw", "and"):
            self.take()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else conj(*parts)

    def unary(self) -> Node:
        if self.peek("kw", "not"):
            self.take()
            return neg(self.unary())
        if self.peek("op", "("):
            self.take()
            node = self.expr()
            self.take("op", ")")
            return node
        if self.peek("kw", "exists"):
            self.take()
            self.take("op", "(")
            name = self.take("name")
            self.take("op", ")")
            self.field_spec(name)
            return self.register(Exact(name, "exists"), f"exists({name})")
        if self.peek("kw", "mentions"):
            self.take()
            self.take("op", "(")
            name = self.take("name")
            self.take("op", ",")
            terms = tuple(self.literal_list()) if self.peek("op", "[") else (self.take("lit"),)
            self.take("op", ")")
            if self.field_spec(name).type not in ("text", "json") or not all(isinstance(t, str) for t in terms):
                raise ConfigError(f"mentions needs a text field and string terms, got {name} and {list(terms)}")
            return self.register(Exact(name, "mentions", terms), f"mentions({name}, {json.dumps(list(terms))})")
        return self.comparison()

    def literal_list(self) -> list:
        self.take("op", "[")
        items = [self.take("lit")]
        while self.peek("op", ","):
            self.take()
            items.append(self.take("lit"))
        self.take("op", "]")
        return items

    def comparison(self) -> Node:
        name = self.take("name")
        if name in self.facts and not self.fields_only and self.fact_specs[self.facts[name]].select is not None:
            return self.select_atom(name)
        op = None
        if self.peek("op") and self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self.take()
        elif self.peek("kw", "in"):
            self.take()
            op = "in"
        elif self.peek("kw", "not") and self.pos + 1 < len(self.toks) and self.toks[self.pos + 1] == ("kw", "in"):
            self.pos += 2
            op = "not in"
        arg = None if op is None else self.literal_list() if op in ("in", "not in") else self.take("lit")
        if name in self.facts and not self.fields_only:
            return self.fact_atom(name, op, arg)
        return self.field_atom(name, op, arg)

    def field_spec(self, name: str) -> FieldSpec:
        if name not in self.spec.fields:
            names = list(self.spec.fields) + ([] if self.fields_only else list(self.facts))
            raise ConfigError(f"unknown name {name!r}{suggest(name, names)}")
        return self.spec.fields[name]

    def field_atom(self, name: str, op, arg) -> Node:
        spec = self.field_spec(name)
        if op is None:
            if spec.type != "bool":
                raise ConfigError(f"{name} is not a bool field; compare it with a value")
            op, arg = "==", True
        if op in ("<", "<=", ">", ">=") and spec.type not in ("number", "text", "enum"):
            raise ConfigError(f"{name} ({spec.type}) does not support ordering")
        for v in arg if op in ("in", "not in") else [arg]:
            ok = {
                "number": isinstance(v, (int, float)) and not isinstance(v, bool) and (isinstance(v, int) or math.isfinite(v)),
                "bool": isinstance(v, bool),
                "text": isinstance(v, str),
                "enum": isinstance(v, str) and v in (spec.values or []),
                "json": True,
            }[spec.type]
            if not ok:
                hint = suggest(str(v), spec.values or []) if spec.type == "enum" else ""
                raise ConfigError(f"{name} ({spec.type}) cannot be compared with {v!r}{hint}")
        value = tuple(arg) if op in ("in", "not in") else arg
        return self.register(Exact(name, op, value), f"{name} {op} {json.dumps(value)}")

    def select_atom(self, name: str) -> Node:
        f = self.facts[name]
        ids = self.card_ids[f]
        if self.peek("kw", "has"):
            self.take()
            card = self.take("lit")
            if card not in ids:
                raise ConfigError(f"{card!r} is not a card of {name}{suggest(str(card), ids)}")
            return self.register(Card(f, ids[card]), f'{name} has "{card}"')
        self.take("kw", "is")
        self.take("kw", "empty")
        return self.register(Card(f, None), f"{name} is empty")

    def implicit(self, name: str) -> None:
        f = self.facts[name]
        for k, key in enumerate(_candidates(self.fact_specs[f])):
            self.register(Card(f, k), f'{name} has "{key}"')
        self.register(Card(f, None), f"{name} is empty")

    def values(self, f: int) -> list[str]:
        fs = self.fact_specs[f]
        if fs.check is not None:
            return ["true", "false"] + (["unstated"] if fs.unstated else [])
        if fs.pick is not None:
            return list(fs.options) + (["none"] if fs.allow_none else []) + [NO_OPTION]
        return list(fs.levels)

    def pred(self, f: int, holds) -> Node:
        silent = {2} if self.fact_specs[f].unstated else set()
        return ("pred", f, tuple(U if k in silent else T if holds(k) else F for k in range(len(self.values(f)))))

    def fact_atom(self, name: str, op, arg) -> Node:
        f = self.facts[name]
        fs = self.fact_specs[f]
        keys = self.values(f)
        if fs.check is not None:
            if op is not None:
                raise ConfigError(f"{name} is a check fact; use it on its own (or with not)")
            return self.pred(f, lambda k: k == 0)
        what = "option" if fs.pick is not None else "level"
        keys = [k for k in keys if k != NO_OPTION]

        def idx(v):
            if v not in keys:
                raise ConfigError(f"{v!r} is not a {what} of {name}{suggest(str(v), keys)}")
            return keys.index(v)

        if op in ("in", "not in"):
            members = {idx(v) for v in arg}
            node = self.pred(f, lambda k: k in members)
            return ("not", node) if op == "not in" else node
        if op is None:
            raise ConfigError(f"{name} is a {fs.pick and 'pick' or 'rate'} fact; compare it with a value")
        if fs.pick is not None and op not in ("==", "!="):
            raise ConfigError(f"{name} is a pick fact; use ==, != or in")
        i = idx(arg)
        return self.pred(f, {"==": lambda k: k == i, "!=": lambda k: k != i, ">=": lambda k: k >= i,
                             ">": lambda k: k > i, "<": lambda k: k < i, "<=": lambda k: k <= i}[op])

    def lower(self, node: Node) -> Node:
        node = self._lower(_flatten(node))
        return self.emit(node) if node[0] == "pred" else node

    def _lower(self, node: Node) -> Node:
        kind = node[0]
        if kind in ("atom", "const", "pred"):
            return node
        if kind == "not":
            inner = self._lower(node[1])
            return ("pred", inner[1], tuple(k_not(v) for v in inner[2])) if inner[0] == "pred" else neg(inner)
        merged: dict = {}
        rest = []
        combine = k_and if kind == "and" else k_or
        for child in (self._lower(c) for c in node[1]):
            if child[0] == "pred":
                merged[child[1]] = child[2] if child[1] not in merged else \
                    tuple(combine(pair) for pair in zip(merged[child[1]], child[2]))
            else:
                rest.append(child)
        parts = [("pred", f, t) for f, t in merged.items()] + rest
        if len(parts) == 1:
            return parts[0]
        parts = [self.emit(c) if c[0] == "pred" else c for c in parts]
        return conj(*parts) if kind == "and" else disj(*parts)

    def emit(self, node: Node) -> Node:
        _, f, truth = node
        if all(v == truth[0] for v in truth):
            return const(truth[0])
        keys = self.values(f)
        name = next(n for n, i in self.facts.items() if i == f)
        shown = [k for k, v in zip(keys, truth) if v == T and k != NO_OPTION]
        return self.register(Sem(f, truth), f"{name} in {shown}")


def _rules(spec: SwitchSpec) -> list[RuleSpec]:
    if isinstance(spec.rules, dict):
        return [RuleSpec(id=k, when=v, then=k) for k, v in spec.rules.items()]
    out = []
    for k, c in enumerate(spec.rules):
        given = "then" in c.model_fields_set
        if c.id is None and (not given or c.then is None):
            raise ConfigError(f"rule {k + 1} needs an id or a then")
        out.append(RuleSpec(id=c.id or str(c.then), when=c.when, then=c.then if given else c.id))
    ids = [c.id for c in out]
    dupes = sorted(i for i, count in Counter(ids).items() if count > 1)
    if dupes:
        raise ConfigError(f"duplicate rule ids {dupes}; give each rule a unique id")
    return out


def _meaning(spec: SwitchSpec, kind: str, text: str, keys, descs, excludes, unstated: bool, reqs, reads, guidance=None, member=None) -> str:
    return digest({**({"guidance": guidance.digest} if guidance else {}), **({"candidate": member} if member else {}),
                   "kind": kind, "text": text, "excludes": list(excludes), "unstated": unstated, "keys": list(keys),
                   "descriptions": list(descs), "requires": reqs, "reads": list(reads),
                   "render": {r: spec.fields[r].model_dump(mode="json") for r in reads}})[:8]


def _fact_plan(b: Builder, spec: SwitchSpec, name: str, fs: FactSpec) -> FactPlan:
    kind = next(k for k in ("check", "pick", "rate", "select") if getattr(fs, k) is not None)
    reads = tuple(fs.reads if fs.reads is not None else spec.reads)
    text = fs.check or fs.pick or fs.rate or fs.select
    aliases, reqs, req_src = (), (), []
    if kind == "check":
        j = Judgment.check(text, fs.excludes, fs.unstated)
        reqs = (None,) * len(j.keys)
    elif kind == "pick":
        opts = _specs(fs.options)
        keys = tuple(opts) + (("none",) if fs.allow_none else ())
        descs = tuple(definition(o.description, o.excludes) for o in opts.values()) + (("none of the above",) if fs.allow_none else ())
        j = Judgment("pick", text, keys, descs, tuple(fs.excludes))
        reqs = tuple(b.parse(o.requires, fields_only=True) if o.requires else None for o in opts.values())
        reqs += (None,) if fs.allow_none else ()
        req_src = [o.requires for o in opts.values()]
        aliases = tuple((k, *o.aliases) for k, o in opts.items())
    elif kind == "rate":
        j = Judgment("rate", text, tuple(fs.levels), tuple(fs.levels), tuple(fs.excludes))
        reqs = (None,) * len(fs.levels)
    elif fs.labels is not None:
        opts = _specs(fs.labels)
        j = Judgment("select", text, tuple(opts), tuple(definition(o.description, o.excludes) for o in opts.values()),
                     tuple(fs.excludes), fs.unstated)
        reqs = tuple(b.parse(o.requires, fields_only=True) if o.requires else None for o in opts.values())
        req_src = [o.requires for o in opts.values()]
    else:
        j = Judgment("select", text, tuple(c.id for c in fs.cards), tuple(c.id for c in fs.cards))
        reqs = tuple(b.parse(c.requires, fields_only=True) if c.requires else None for c in fs.cards)
        req_src = [c.model_dump(mode="json", exclude={"payload", "source", "title"}) for c in fs.cards]
    j = replace(j, guidance=bind(b.guidance, fs.guidance))
    used = set(reads) | {b.atoms[i].field for r in reqs if r for i in atoms_of(r)}
    context_reads = tuple(reads) + tuple(sorted(used - set(reads)))
    rev = _meaning(spec, kind, text, j.keys, j.descriptions, j.excludes, j.unstated, req_src, context_reads, j.guidance)
    return FactPlan(name, j, reqs, fs.allow_none, reads, fs.target, Policy.of(fs.min_score), rev,
                    tuple((freeze(e.input), freeze(e.value)) for e in fs.examples), aliases, fs.focus, (), fs.max_claims,
                    fs.exclusive, context_reads=context_reads, binding=fs.binding, composition_rev=rev,
                    execution_rev=digest({"focus": fs.focus})[:8])


def _claim_fact(spec: SwitchSpec, text: str, exclusion: bool, reads, guidance, binding=None) -> FactPlan:
    j = Judgment.check(text, (), exclusion, guidance)
    rev = _meaning(spec, "check", text, j.keys, j.descriptions, (), exclusion, [], reads, guidance)
    return FactPlan(f"~{rev}", j, (None,) * len(j.keys), False, tuple(reads), 0.95, Policy(), rev, (),
                    hidden=True, context_reads=tuple(reads), binding=binding, composition_rev=rev, execution_rev=digest({"focus": None})[:8])


def _shared_binding(spec: SwitchSpec, plan: FactPlan, claim: FactPlan) -> None:
    mine, theirs = plan.binding or spec.binding, claim.binding or spec.binding
    if mine != theirs:
        raise ConfigError(f"claim {claim.shown!r} is bound to {theirs!r} but {plan.name} inherits {mine!r}; "
                          "a shared semantic atom answers under one binding, so bind both facts to the same model role")


def memberships(instruction: str, labels, excludes=(), unstated: bool = False, guidance=None) -> tuple[Judgment, ...]:
    return tuple(Judgment.check(instruction, (), unstated, guidance, candidate(key, label.description, (*label.excludes, *excludes)))
                 for key, label in labels.items())


def membership_cards(labels, nodes, claims, requires) -> tuple[CardPlan, ...]:
    return tuple(CardPlan(key, label.description, req, (key, *label.aliases), (claim,), (), base, base, freeze({"id": key}))
                 for (key, label), node, claim, req in zip(labels.items(), nodes, claims, requires)
                 for base in (conj(req, node) if req else node,))


def document(card: CardPlan, texts) -> str:
    return " ".join((card.title, *card.aliases, *texts))


def _members(b: Builder, spec: SwitchSpec, facts: list[FactPlan], index: dict, plan: FactPlan, fs: FactSpec):
    labels = {key: o.label for key, o in _specs(fs.labels).items()}
    nodes, claims = [], []
    for key, j in zip(labels, memberships(plan.text, labels, plan.excludes, plan.unstated, plan.judgment.guidance)):
        rev = _meaning(spec, "check", j.text, j.keys, j.descriptions, (), j.unstated, [], plan.context_reads, j.guidance, [plan.name, key, j.candidate])
        member = FactPlan(f"~{rev}", j, (None,) * len(j.keys), False, plan.reads, plan.target, plan.policy, rev, (),
                          hidden=True, context_reads=plan.context_reads, member=(plan.name, key), binding=plan.binding, composition_rev=rev,
                          execution_rev=plan.execution_rev)
        index[member.name] = len(facts)
        claims.append(len(facts))
        facts.append(member)
        nodes.append(b.register(Sem(claims[-1], j.main), member.shown))
    return membership_cards(labels, nodes, claims, plan.requires), [[] for _ in labels]


def _compile_select(b: Builder, spec: SwitchSpec, facts: list[FactPlan], index: dict, f: int, fs: FactSpec) -> FactPlan:
    plan = facts[f]
    cards, over = (_members if fs.labels is not None else _cards)(b, spec, facts, index, plan, fs)
    rev = digest([plan.meaning_rev, [[facts[k].meaning_rev for k in card.claims] for card in cards]])[:8]
    composition = digest({"criterion": rev, "supersedes": [list(card.supersedes) for card in cards], "exclusive": fs.exclusive})[:8]
    execution = digest({"focus": fs.focus, "max_claims": fs.max_claims, "titles": [card.title for card in cards],
                        "aliases": [list(card.aliases) for card in cards]})[:8]
    claim_cards = {}
    for j, card in enumerate(cards):
        for k in card.claims:
            claim_cards.setdefault(k, []).append(j)
    ranking = Bm25([document(card, (facts[k].text for k in card.claims)) for card in cards])
    return replace(plan, cards=cards, meaning_rev=rev, composition_rev=composition, execution_rev=execution, ranking=ranking, incoming=tuple(map(tuple, over)),
                   claim_cards=freeze(claim_cards), labels=fs.labels is not None)


def _cards(b: Builder, spec: SwitchSpec, facts: list[FactPlan], index: dict, plan: FactPlan, fs: FactSpec):
    by_rev = {x.meaning_rev: k for k, x in enumerate(facts) if x.hidden}

    def claim(c, exclusion: bool, guidance, explicit=False) -> Node:
        if isinstance(c, FactRef):
            if c.fact not in index or facts[index[c.fact]].kind != "check":
                raise ConfigError(f"card claim {{fact: {c.fact}}} must name a check fact"
                                  f"{suggest(c.fact, [n for n in index if not n.startswith('~')])}")
            k = index[c.fact]
            if explicit and facts[k].judgment.guidance.digest != guidance.digest:
                raise ConfigError(f"card claim {c.fact!r} has an incompatible guidance binding")
            if exclusion and not facts[k].unstated:
                raise ConfigError(f"card exclusion {{fact: {c.fact}}} must name a three-way check (unstated: true): "
                                  f"an exclusion fires only when the input states it, and silence must not exclude")
        else:
            made = _claim_fact(spec, c, exclusion, plan.reads, guidance, plan.binding)
            if made.meaning_rev not in by_rev:
                by_rev[made.meaning_rev] = index[made.name] = len(facts)
                facts.append(made)
            k = by_rev[made.meaning_rev]
        _shared_binding(spec, plan, facts[k])
        return b.register(Sem(k, facts[k].judgment.main), facts[k].name)

    ids = [c.id for c in fs.cards]
    bases, claims = [], []
    for c, req in zip(fs.cards, plan.requires):
        guidance = bind(b.guidance, c.guidance) if c.guidance is not None else plan.judgment.guidance
        when = [claim(x, False, guidance, c.guidance is not None) for x in c.applies_when]
        unless = [neg(claim(x, True, guidance, c.guidance is not None)) for x in c.excludes]
        bases.append(conj(req or const(T), *when, *unless))
        claims.append(tuple(dict.fromkeys(b.atoms[n[1]].fact for n in when + [u[1] for u in unless])))
    id_index = {name: k for k, name in enumerate(ids)}
    over = [[] for _ in fs.cards]
    for j, c in enumerate(fs.cards):
        for target in c.supersedes:
            over[id_index[target]].append(j)
    return tuple(CardPlan(c.id, c.title, req, tuple(c.aliases), claims[k], tuple(id_index[s] for s in c.supersedes),
                          bases[k], conj(bases[k], neg(disj(*(bases[j] for j in over[k])))) if over[k] else bases[k],
                          freeze(c.model_dump(mode="json")))
                 for k, (c, req) in enumerate(zip(fs.cards, plan.requires))), over


def _load_catalogs(data: dict, base: Path | None) -> dict:
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, dict):
        return data
    out = dict(facts)
    from .files import read_yaml
    for name, fs in facts.items():
        if not isinstance(fs, dict):
            continue
        if sum(fs.get(key) is not None for key in ("labels", "catalog", "cards")) > 1:
            raise ConfigError(f"facts.{name}: select needs exactly one candidate source: labels, cards or catalog")
        for key in ("catalog", "labels"):
            if isinstance(fs.get(key), str):
                path = Path(fs[key]) if base is None else base / fs[key]
                if not path.exists():
                    raise ConfigError(f"facts.{name}: {key} {fs[key]!r} not found")
                loaded = read_yaml(path) or []
                out[name] = {**fs, "cards": loaded.get("cards", loaded) if isinstance(loaded, dict) else loaded} if key == "catalog" \
                    else {**fs, "labels": loaded}
    return {**data, "facts": out}


def _validated(data: dict, base: Path | None) -> tuple[SwitchSpec, Any]:
    try:
        resolved = _load_catalogs(data, base)
        guidance = resolve_catalog(resolved.get("guidance", {}), base)
        spec = SwitchSpec.model_validate({**resolved, "guidance": guidance.as_dict()})
        for n in spec.reads + [r for fs in spec.facts.values() for r in fs.reads or []]:
            if n not in spec.fields:
                raise ConfigError(f"reads references unknown field {n!r}{suggest(n, spec.fields)}")
    except ConfigError as e:
        raise ConfigError(f"spec: {e}") from None
    return spec, guidance


def _lowered(spec: SwitchSpec, guidance) -> tuple[Builder, list[FactPlan], list[RulePlan], dict]:
    names = list(spec.facts)
    fact_specs = [spec.facts[n] for n in names]
    index = {n: i for i, n in enumerate(names)}
    b = Builder(spec, index, fact_specs)
    b.guidance = guidance
    where, facts, rules = "spec", [], []
    try:
        for n, fs in zip(names, fact_specs):
            where = f"facts.{n}"
            facts.append(_fact_plan(b, spec, n, fs))
        for f, fs in enumerate(fact_specs):
            if fs.select is not None:
                where = f"facts.{names[f]}"
                facts[f] = _compile_select(b, spec, facts, index, f, fs)
        for c in _rules(spec):
            where = f"rules.{c.id}"
            expr = b.parse(c.when)
            refs = tuple(sorted({b.atoms[i].fact for i in atoms_of(expr) if isinstance(b.atoms[i], (Sem, Card))}))
            rules.append(RulePlan(c.id, expr, freeze(c.then), c.when, refs))
        if not rules:
            for n, fs in zip(names, fact_specs):
                if fs.select is not None:
                    b.implicit(n)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from None
    claim_of = {}
    for f, plan in enumerate(facts):
        for card in plan.cards:
            for k in card.claims:
                claim_of.setdefault(k, set()).add(f)
    return b, facts, rules, claim_of


def catalogue(facts) -> tuple[Catalog, dict]:
    entries, ids = [], {}
    for f, fact in enumerate(facts):
        if fact.cards or fact.judgment is None:
            continue
        question = fact.judgment.question() if fact.judgment.kind == "check" else fact.judgment.question(tuple(fact.judgment.descriptions))
        if question.candidate is not None:
            continue
        ids[f] = f"C{len(entries) + 1}"
        entries.append((ids[f], definition(question.text, question.excludes)))
    return Catalog(tuple(entries)), freeze(ids)


def _assembled(spec: SwitchSpec, b: Builder, facts: list[FactPlan], rules: list[RulePlan], claim_of: dict) -> Plan:
    index = b.facts
    atom_fact = tuple(a.fact if isinstance(a, (Sem, Card)) else -1 for a in b.atoms)
    indexed_atoms = [[] for _ in facts]
    for i, a in enumerate(b.atoms):
        if isinstance(a, (Sem, Card)):
            indexed_atoms[a.fact].append((i, a))
    fact_atoms = tuple(map(tuple, indexed_atoms))
    dependencies = set()
    pending = [f for rule in rules for f in rule.facts]
    while pending:
        f = pending.pop()
        if f in dependencies:
            continue
        dependencies.add(f)
        fact = facts[f]
        pending.extend(k for card in fact.cards for k in card.claims)
        pending.extend(b.atoms[i].fact for required in fact.requires if required is not None
                       for i in atoms_of(required) if isinstance(b.atoms[i], (Sem, Card)))
    meanings = {facts[f].name: facts[f].composition_rev for f in sorted(dependencies)}
    switch_rev = digest({"rules": [(c.id, c.when, c.then) for c in rules], "otherwise": spec.otherwise,
                         "resolve": spec.resolve, "compositions": meanings})[:8]
    execution_rev = digest({"switch": switch_rev, "facts": {fact.name: [fact.composition_rev, fact.execution_rev] for fact in facts}})[:8]
    canonical = spec.model_dump(mode="json")
    if canonical["binding"] == DEFAULT_BINDING:
        del canonical["binding"]
    if not canonical["retired"]:
        del canonical["retired"]
    if canonical["presentation"] == "inline":
        del canonical["presentation"]
    catalog, catalog_ids = catalogue(facts) if spec.presentation == "catalog" else (None, {})
    canonical["facts"] = {n: {k: v for k, v in canonical["facts"][n].items() if k in _KIND_FIELDS[_kind(fs)] and (k != "binding" or v is not None)}
                          for n, fs in spec.facts.items()}
    canonical["tests"] = [t.model_dump(mode="json", exclude_unset=True) for t in spec.tests]
    if not isinstance(spec.rules, dict):
        canonical["rules"] = [c.model_dump(mode="json", exclude=set() if "then" in c.model_fields_set else {"then"}) for c in spec.rules]
    for t in spec.tests:
        if "expect_rule" in t.model_fields_set:
            expected = (t.expect_rule,) if isinstance(t.expect_rule, str) else t.expect_rule or ()
            if any(name not in {rule.id for rule in rules} | {"otherwise"} for name in expected):
                raise ConfigError(f"test expectation names unknown rule {t.expect_rule!r}")
        for name in (t.expect_facts or {}) | (t.expect_complete or {}):
            if name not in index:
                raise ConfigError(f"test expectation names unknown fact {name!r}")
    return Plan(spec.name, freeze({n: FieldPlan.of(f) for n, f in spec.fields.items()}), tuple(facts), freeze(index), tuple(b.atoms), tuple(b.keys), atom_fact,
                tuple(i for i, f in enumerate(atom_fact) if f < 0), fact_atoms, tuple(rules), freeze(spec.otherwise),
                spec.resolve == "first", spec.exact_override, tuple(TestPlan.of(test) for test in spec.tests), switch_rev,
                freeze({k: frozenset(v) for k, v in claim_of.items()}), freeze(canonical), spec.binding, execution_rev, catalog, catalog_ids)


def compile_spec(data: dict, source: str = "switch", base: Path | None = None) -> Plan:
    try:
        spec, guidance = _validated(data, base)
        lowered = _lowered(spec, guidance)
    except ConfigError as e:
        raise ConfigError(f"{source}: {e}") from None
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise ConfigError(f"{source}: {loc}: {first['msg']}") from None
    return _assembled(spec, *lowered)
