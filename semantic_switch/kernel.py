from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

F, T, U = 0, 1, 2
STATE_NAMES = {F: "F", T: "T", U: "U"}
MISSING = object()
NEEDS_INPUT = frozenset({"missing_input", "not_stated", "insufficient_evidence"})
SUPERSEDED = ("superseded",)
CONFLICT = ("exclusive_conflict",)


@dataclass(frozen=True, slots=True)
class Sem:
    fact: int
    truth: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Card:
    fact: int
    card: int | None


@dataclass(frozen=True, slots=True)
class Exact:
    field: str
    op: str
    value: object = None


Node = tuple


def atom(i: int) -> Node:
    return ("atom", i)


def const(v: int) -> Node:
    return ("const", v)


def neg(n: Node) -> Node:
    return ("not", n)


def conj(*ns: Node) -> Node:
    return ("and", tuple(ns))


def disj(*ns: Node) -> Node:
    return ("or", tuple(ns))


def k_not(v: int) -> int:
    return U if v == U else 1 - v


def k_and(vals) -> int:
    out = T
    for v in vals:
        if v == F:
            return F
        if v == U:
            out = U
    return out


def k_or(vals) -> int:
    out = F
    for v in vals:
        if v == T:
            return T
        if v == U:
            out = U
    return out


def evaluate(node: Node, st: list[int]) -> int:
    kind = node[0]
    if kind == "atom":
        return st[node[1]]
    if kind == "const":
        return node[1]
    if kind == "not":
        return k_not(evaluate(node[1], st))
    vals = (evaluate(c, st) for c in node[1])
    return k_and(vals) if kind == "and" else k_or(vals)


def relevant(node: Node, st: list[int]) -> set[int]:
    kind = node[0]
    if kind == "atom":
        return {node[1]} if st[node[1]] == U else set()
    if kind == "const":
        return set()
    if kind == "not":
        return relevant(node[1], st)
    vals = [evaluate(c, st) for c in node[1]]
    if (kind == "and" and F in vals) or (kind == "or" and T in vals):
        return set()
    return set().union(*(relevant(c, st) for c, v in zip(node[1], vals) if v == U))


def atoms_of(node: Node) -> set[int]:
    kind = node[0]
    if kind == "atom":
        return {node[1]}
    if kind == "const":
        return set()
    if kind == "not":
        return atoms_of(node[1])
    return set().union(*(atoms_of(c) for c in node[1]))


_CMP = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}


BEFORE, AFTER = r"(?<![\w-])(?<!\w\.)", r"(?![\w-])(?!\.\w)"


def _mention_char(char):
    if char in 'İı':
        return 'i'
    folded = char.casefold()
    return folded if len(folded) == 1 else char.lower()


def _word(char):
    return char == '_' or char.isalnum()


class Mentions:
    __slots__ = ('root',)

    def __init__(self, aliases):
        self.root = {}
        for alias in aliases:
            node = self.root
            for char in unicodedata.normalize('NFC', alias):
                node = node.setdefault(_mention_char(char), {})
            node.setdefault(None, []).append(alias)

    def found(self, normalized):
        hits, size = set(), len(normalized)
        folded = [_mention_char(char) for char in normalized]
        for start in range(size):
            if start and (_word(normalized[start - 1]) or normalized[start - 1] == '-'):
                continue
            if start > 1 and normalized[start - 1] == '.' and _word(normalized[start - 2]):
                continue
            node, stop = self.root, start
            while stop < size and folded[stop] in node:
                node = node[folded[stop]]
                stop += 1
                if None not in node:
                    continue
                if stop < size and (_word(normalized[stop]) or normalized[stop] == '-'):
                    continue
                if stop + 1 < size and normalized[stop] == '.' and _word(normalized[stop + 1]):
                    continue
                hits.update(node[None])
        return hits


@lru_cache(maxsize=1024)
def mentions(aliases: tuple[str, ...]) -> Mentions:
    return Mentions(aliases)


def matched_mentions(text, identities, cache=None):
    if cache is None:
        normalized, hits = unicodedata.normalize("NFC", text), {}
    else:
        if text not in cache:
            cache[text] = (unicodedata.normalize("NFC", text), {})
        normalized, hits = cache[text]
    missing = tuple(dict.fromkeys(alias for aliases in identities for alias in aliases if alias not in hits))
    if missing:
        found = mentions(missing).found(normalized)
        hits.update((alias, alias in found) for alias in missing)
    return [index for index, aliases in enumerate(identities) if any(hits[alias] for alias in aliases)]


def mentioned(text: str, terms: tuple[str, ...]) -> bool:
    return bool(terms) and bool(mentions(tuple(terms)).found(unicodedata.normalize("NFC", text)))


def _text(v) -> str:
    from .context import as_text
    return as_text(v)


def exact_state(a: Exact, values: dict) -> int:
    v = values.get(a.field, MISSING)
    if a.op == "exists":
        return F if v is MISSING else T
    if v is MISSING:
        return U
    if a.op == "mentions":
        return T if mentioned(_text(v), a.value) else F
    try:
        return T if _CMP[a.op](v, a.value) else F
    except (TypeError, ValueError):
        return U


@dataclass(frozen=True, slots=True)
class Resolution:
    status: str
    answer: object = None
    rule: str | None = None
    blocked_by: str | None = None
    blocking: frozenset = frozenset()
    partial: tuple = ()


def resolve(rules: tuple, otherwise: object, st: list[int], first: bool) -> Resolution:
    vals = []
    for rule in rules:
        v = evaluate(rule.expr, st)
        if first and v == U:
            return Resolution("blocked", blocked_by=rule.id, blocking=frozenset(relevant(rule.expr, st)))
        if first and v == T:
            return Resolution("accepted", rule.then, rule.id)
        vals.append(v)
    if first:
        return Resolution("accepted", otherwise, "otherwise")
    trues = tuple((c.id, c.then) for c, v in zip(rules, vals) if v == T)
    for rule, v in zip(rules, vals):
        if v == U:
            return Resolution("blocked", blocked_by=rule.id, blocking=frozenset(relevant(rule.expr, st)),
                              partial=trues)
    if trues:
        return Resolution("accepted", tuple(t for _, t in trues), tuple(c for c, _ in trues))
    return Resolution("accepted", otherwise, "otherwise")


def card_state(requires: int, base: int, has: int, unjudged: bool) -> str:
    if requires == F:
        return "ineligible"
    if base == F:
        return "rejected"
    if base == T:
        return "applicable" if has == T else "superseded" if has == F else "may_be_superseded"
    return "unexamined" if requires == T and unjudged else "unresolved"


@dataclass(frozen=True, slots=True)
class SelectionState:
    values: tuple[int, ...]
    states: tuple[str, ...]
    reasons: tuple[tuple | None, ...]
    superseding: tuple[int | None, ...]
    selected: tuple[int, ...]
    complete: bool
    settled: bool

    @property
    def empty(self) -> int:
        return F if self.selected else T if self.settled else U

    @property
    def status(self) -> str:
        if self.settled:
            return "ok" if self.selected else "none"
        kinds = {r[0] for r in self.reasons if r is not None}
        return "failed" if "model_error" in kinds else "needs_input" if kinds & NEEDS_INPUT else "uncertain"


def compose(requires, bases, incoming, exclusive: bool, unjudged, reasons) -> SelectionState:
    values = [k_and((b, k_not(k_or(bases[j] for j in inc)))) if inc else b for b, inc in zip(bases, incoming)]
    states = [card_state(r, b, v, u) for r, b, v, u in zip(requires, bases, values, unjudged)]
    why = [None if v != U else SUPERSEDED if b == T else reasons[k] for k, (b, v) in enumerate(zip(bases, values))]
    if exclusive and T in values and (values.count(T) > 1 or U in values):
        for k, v in enumerate(values):
            if v == T:
                values[k], states[k], why[k] = U, "conflict", CONFLICT
    superseding = tuple(next(j for j in inc if bases[j] == T) if s == "superseded" else None for inc, s in zip(incoming, states))
    return SelectionState(tuple(values), tuple(states), tuple(why), superseding, tuple(k for k, v in enumerate(values) if v == T),
                          "unexamined" not in states, U not in values)


def ranked(text: str, instruction: str, open_: list[int], aliases, ranking, memo=None) -> list[int]:
    named = [open_[i] for i in matched_mentions(text, [aliases[k] for k in open_], memo)]
    chosen = set(named)
    return named + ranking.rank(f"{instruction} {text}", [k for k in open_ if k not in chosen])


def budgeted(order, claims, known: set, limit: int | None) -> set[int]:
    spent, out = set(), set()
    for k in order:
        new = set(claims[k]) - known - spent
        if limit is None or len(spent) + len(new) <= limit:
            spent |= new
            out.add(k)
    return out
