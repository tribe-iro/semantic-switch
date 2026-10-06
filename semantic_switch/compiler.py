from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .context import Bm25
from .errors import ConfigError, suggest
from .judgment import Judgment, Policy
from .kernel import F, T, U, Card, Exact, Node, Sem, atom, atoms_of, conj, const, disj, k_and, k_not, k_or, neg

MAX_OPTIONS = 20
NO_OPTION = "\u2205"
MAX_FACTS = 32


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldSpec(Spec):
    type: Literal["text", "number", "bool", "enum", "json"] = "text"
    required: bool = False
    max_chars: int | None = None
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
    aliases: list[str] = []


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

    @field_validator("applies_when", "excludes", mode="before")
    @classmethod
    def _listify(cls, v):
        return [v] if isinstance(v, (str, dict)) else v


class ExampleSpec(Spec):
    input: dict
    value: Any


def _cycle(cards: list[CardSpec]) -> list[str] | None:
    edges = {c.id: c.supersedes for c in cards}
    state: dict[str, int] = {}

    def visit(n: str, path: list[str]) -> list[str] | None:
        state[n] = 1
        for m in edges.get(n, ()):
            if state.get(m) == 1:
                return path + [n, m]
            if m not in state and (found := visit(m, path + [n])):
                return found
        state[n] = 2
        return None

    for c in cards:
        if c.id not in state and (found := visit(c.id, [])):
            return found[found.index(found[-1]):]
    return None


class FactSpec(Spec):
    check: str | None = None
    pick: str | None = None
    rate: str | None = None
    select: str | None = None
    excludes: list[str] = []
    options: dict[str, str | OptionSpec] = {}
    levels: list[str] = []
    allow_none: bool = False
    unstated: bool = False
    focus: Literal["mentions"] | None = None
    catalog: str | None = None
    cards: list[CardSpec] = []
    max_claims: int = Field(16, ge=1, le=256)
    exclusive: bool = False
    specialize: Literal["lexical"] | None = None
    min_score: float | None = None
    reads: list[str] | None = None
    target: float = 0.95
    examples: list[ExampleSpec] = []

    @model_validator(mode="after")
    def _one_kind(self):
        kinds = [k for k in ("check", "pick", "rate", "select") if getattr(self, k) is not None]
        if len(kinds) != 1:
            raise ValueError("a fact needs exactly one of check, pick, rate, select")
        if self.unstated and self.check is None:
            raise ValueError("unstated applies to check facts only")
        if self.focus and self.pick is None:
            raise ValueError("focus applies to pick facts only")
        if self.select is not None:
            if not self.cards:
                raise ValueError("select needs a catalog with at least one card")
            ids = [c.id for c in self.cards]
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"duplicate card ids {dupes}")
            unknown = sorted({s for c in self.cards for s in c.supersedes} - set(ids))
            if unknown:
                raise ValueError(f"supersedes names unknown cards {unknown}")
            cycle = _cycle(self.cards)
            if cycle:
                raise ValueError(f"supersedes cycle: {' → '.join(cycle)}")
            if self.examples or self.specialize:
                raise ValueError("select facts take no examples and no specialize")
        if self.pick is not None and len(self.options) < 2 - self.allow_none:
            raise ValueError("pick needs options")
        if self.allow_none and "none" in self.options:
            raise ValueError("'none' is reserved for allow_none; rename that option")
        if len(self.options) + self.allow_none > MAX_OPTIONS:
            raise ValueError(f"a pick fact can have at most {MAX_OPTIONS} options (including none)")
        if self.rate is not None and (len(self.levels) < 2 or len(set(self.levels)) != len(self.levels)):
            raise ValueError("rate needs at least two distinct levels")
        if len(self.levels) > MAX_OPTIONS:
            raise ValueError(f"rate takes at most {MAX_OPTIONS} levels")
        if not 0.5 <= self.target <= 1:
            raise ValueError("target must be within [0.5, 1]")
        if self.min_score is not None and not 0.5 < self.min_score <= 1:
            raise ValueError("min_score must be above 0.5 and at most 1")
        return self


class CaseSpec(Spec):
    id: str | None = None
    when: str
    then: Any = None


class TestSpec(Spec):
    input: dict
    expect: Any


class SwitchSpec(Spec):
    name: str
    fields: dict[str, FieldSpec] = {}
    reads: list[str] = []
    facts: dict[str, FactSpec] = Field(max_length=MAX_FACTS)
    cases: list[CaseSpec] | dict[str, str] = []
    otherwise: Any = None
    resolve: Literal["first", "all"] = "first"
    exact_override: bool = True
    tests: list[TestSpec] = []


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
    specialize: str | None
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

    @property
    def kind(self) -> str:
        return "select" if self.cards else self.judgment.kind

    @property
    def text(self) -> str:
        return self.judgment.text

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
class CasePlan:
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
    cases: tuple[CasePlan, ...]
    otherwise: Any
    first: bool
    exact_override: bool
    tests: tuple
    switch_rev: str
    claim_of: dict
    spec: dict = field(compare=False)

    def fact(self, name: str) -> FactPlan:
        if name not in self.fact_index:
            raise ConfigError(f"unknown fact {name!r}{suggest(name, [n for n in self.fact_index if not n.startswith('~')])}")
        return self.facts[self.fact_index[name]]

    @property
    def visible(self) -> tuple[FactPlan, ...]:
        return tuple(f for f in self.facts if not f.hidden)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


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


class Builder:
    def __init__(self, spec: SwitchSpec, facts: dict[str, int], fact_specs: list[FactSpec]):
        self.spec = spec
        self.facts = facts
        self.fact_specs = fact_specs
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
        for v in arg if op in ("in", "not in") else [arg]:
            ok = {
                "number": isinstance(v, (int, float)) and not isinstance(v, bool),
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
        ids = [c.id for c in self.fact_specs[f].cards]
        if self.peek("kw", "has"):
            self.take()
            card = self.take("lit")
            if card not in ids:
                raise ConfigError(f"{card!r} is not a card of {name}{suggest(str(card), ids)}")
            return self.register(Card(f, ids.index(card)), f'{name} has "{card}"')
        self.take("kw", "is")
        self.take("kw", "empty")
        return self.register(Card(f, None), f"{name} is empty")

    def implicit(self, name: str) -> None:
        f = self.facts[name]
        for k, c in enumerate(self.fact_specs[f].cards):
            self.register(Card(f, k), f'{name} has "{c.id}"')
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
        node = self._lower(node)
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


def _cases(spec: SwitchSpec) -> list[CaseSpec]:
    if isinstance(spec.cases, dict):
        return [CaseSpec(id=k, when=v, then=k) for k, v in spec.cases.items()]
    out = []
    for k, c in enumerate(spec.cases):
        if c.id is None and c.then is None:
            raise ConfigError(f"case {k + 1} needs an id or a then")
        out.append(CaseSpec(id=c.id or str(c.then), when=c.when, then=c.id if c.then is None else c.then))
    ids = [c.id for c in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(f"duplicate case ids {dupes}; give each case a unique id")
    return out


def _meaning(spec: SwitchSpec, kind: str, text: str, keys, descs, excludes, unstated: bool, reqs, reads) -> str:
    return digest({"kind": kind, "text": text, "excludes": list(excludes), "unstated": unstated, "keys": list(keys),
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
        opts = {k: v if isinstance(v, OptionSpec) else OptionSpec(description=v) for k, v in fs.options.items()}
        keys = tuple(opts) + (("none",) if fs.allow_none else ())
        descs = tuple(o.description for o in opts.values()) + (("none of the above",) if fs.allow_none else ())
        j = Judgment("pick", text, keys, descs, tuple(fs.excludes))
        reqs = tuple(b.parse(o.requires, fields_only=True) if o.requires else None for o in opts.values())
        reqs += (None,) if fs.allow_none else ()
        req_src = [o.requires for o in opts.values()]
        aliases = tuple((k, *o.aliases) for k, o in opts.items())
    elif kind == "rate":
        j = Judgment("rate", text, tuple(fs.levels), tuple(fs.levels), tuple(fs.excludes))
        reqs = (None,) * len(fs.levels)
    else:
        j = Judgment("select", text, tuple(c.id for c in fs.cards), tuple(c.id for c in fs.cards))
        reqs = tuple(b.parse(c.requires, fields_only=True) if c.requires else None for c in fs.cards)
        req_src = [c.model_dump(mode="json", exclude={"payload", "source", "title"}) for c in fs.cards]
    rev = _meaning(spec, kind, text, j.keys, j.descriptions, j.excludes, j.unstated, req_src, reads)
    return FactPlan(name, j, reqs, fs.allow_none, reads, fs.target, fs.specialize, Policy.of(fs.min_score), rev,
                    tuple((e.input, e.value) for e in fs.examples), aliases, fs.focus, (), fs.max_claims,
                    fs.exclusive)


def _claim_fact(spec: SwitchSpec, text: str, exclusion: bool, reads) -> FactPlan:
    j = Judgment.check(text, (), exclusion)
    rev = _meaning(spec, "check", text, j.keys, j.descriptions, (), exclusion, [], reads)
    return FactPlan(f"~{rev}", j, (None,) * len(j.keys), False, tuple(reads), 0.95, None, Policy(), rev, (),
                    hidden=True)


def _compile_cards(b: Builder, spec: SwitchSpec, facts: list[FactPlan], index: dict, f: int,
                   fs: FactSpec) -> FactPlan:
    plan = facts[f]
    by_rev = {x.meaning_rev: k for k, x in enumerate(facts) if x.hidden}

    def claim(c, exclusion: bool) -> Node:
        if isinstance(c, FactRef):
            if c.fact not in index or facts[index[c.fact]].kind != "check":
                raise ConfigError(f"card claim {{fact: {c.fact}}} must name a check fact"
                                  f"{suggest(c.fact, [n for n in index if not n.startswith('~')])}")
            k = index[c.fact]
        else:
            made = _claim_fact(spec, c, exclusion, plan.reads)
            if made.meaning_rev not in by_rev:
                by_rev[made.meaning_rev] = index[made.name] = len(facts)
                facts.append(made)
            k = by_rev[made.meaning_rev]
        return b.register(Sem(k, facts[k].judgment.main), facts[k].name)

    ids = [c.id for c in fs.cards]
    bases, claims = [], []
    for c, req in zip(fs.cards, plan.requires):
        when = [claim(x, False) for x in c.applies_when]
        unless = [neg(claim(x, True)) for x in c.excludes]
        bases.append(conj(req or const(T), *when, *unless))
        claims.append(tuple(dict.fromkeys(b.atoms[n[1]].fact for n in when + [u[1] for u in unless])))
    over = [[j for j, o in enumerate(fs.cards) if c.id in o.supersedes] for c in fs.cards]
    cards = tuple(CardPlan(c.id, c.title, req, tuple(c.aliases), claims[k], tuple(ids.index(s) for s in c.supersedes),
                           bases[k], conj(bases[k], neg(disj(*(bases[j] for j in over[k])))) if over[k] else bases[k],
                           c.model_dump(mode="json"))
                  for k, (c, req) in enumerate(zip(fs.cards, plan.requires)))
    rev = digest([plan.meaning_rev, [[facts[k].meaning_rev for k in card.claims] for card in cards],
                  [card.supersedes for card in cards]])[:8]
    index = Bm25([" ".join((c.title, *c.aliases, *(facts[k].text for k in c.claims))) for c in cards])
    return replace(plan, cards=cards, meaning_rev=rev, ranking=index)


def _load_catalogs(data: dict, base: Path | None) -> dict:
    facts = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(facts, dict):
        return data
    out = dict(facts)
    for name, fs in facts.items():
        if isinstance(fs, dict) and fs.get("catalog") and not fs.get("cards"):
            path = Path(fs["catalog"]) if base is None else base / fs["catalog"]
            if not path.exists():
                raise ConfigError(f"facts.{name}: catalog {fs['catalog']!r} not found")
            cards = yaml.safe_load(path.read_text(encoding="utf-8")) or []
            out[name] = {**fs, "cards": cards.get("cards", cards) if isinstance(cards, dict) else cards}
    return {**data, "facts": out}


def compile_spec(data: dict, source: str = "switch", base: Path | None = None) -> Plan:
    where = "spec"
    try:
        spec = SwitchSpec.model_validate(_load_catalogs(data, base))
        names = list(spec.facts)
        fact_specs = [spec.facts[n] for n in names]
        index = {n: i for i, n in enumerate(names)}
        for n in spec.reads + [r for fs in fact_specs for r in fs.reads or []]:
            if n not in spec.fields:
                raise ConfigError(f"reads references unknown field {n!r}{suggest(n, spec.fields)}")
        b = Builder(spec, index, fact_specs)
        facts = []
        for n, fs in zip(names, fact_specs):
            where = f"facts.{n}"
            facts.append(_fact_plan(b, spec, n, fs))
        for f, fs in enumerate(fact_specs):
            if fs.select is not None:
                where = f"facts.{names[f]}"
                facts[f] = _compile_cards(b, spec, facts, index, f, fs)
        cases = []
        for c in _cases(spec):
            where = f"cases.{c.id}"
            expr = b.parse(c.when)
            refs = tuple(sorted({b.atoms[i].fact for i in atoms_of(expr) if isinstance(b.atoms[i], (Sem, Card))}))
            cases.append(CasePlan(c.id, expr, c.then, c.when, refs))
        if not cases:
            for n, fs in zip(names, fact_specs):
                if fs.select is not None:
                    b.implicit(n)
        claim_of = {}
        for f, plan in enumerate(facts):
            for card in plan.cards:
                for k in card.claims:
                    claim_of.setdefault(k, set()).add(f)
        where = "spec"
    except ConfigError as e:
        raise ConfigError(f"{source}: {where}: {e}") from None
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise ConfigError(f"{source}: {loc}: {first['msg']}") from None
    atom_fact = tuple(a.fact if isinstance(a, (Sem, Card)) else -1 for a in b.atoms)
    fact_atoms = tuple(tuple((i, a) for i, a in enumerate(b.atoms) if isinstance(a, (Sem, Card)) and a.fact == f)
                       for f in range(len(facts)))
    switch_rev = digest({"cases": [(c.id, c.when, c.then) for c in cases], "otherwise": spec.otherwise,
                         "resolve": spec.resolve})[:8]
    return Plan(spec.name, dict(spec.fields), tuple(facts), index, tuple(b.atoms), tuple(b.keys), atom_fact,
                tuple(i for i, f in enumerate(atom_fact) if f < 0), fact_atoms, tuple(cases), spec.otherwise,
                spec.resolve == "first", spec.exact_override, tuple(spec.tests), switch_rev,
                {k: frozenset(v) for k, v in claim_of.items()}, spec.model_dump(mode="json"))
