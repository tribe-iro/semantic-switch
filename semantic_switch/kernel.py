from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

F, T, U = 0, 1, 2
STATE_NAMES = {F: "F", T: "T", U: "U"}
MISSING = object()


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
    vals = list(vals)
    return F if F in vals else U if U in vals else T


def k_or(vals) -> int:
    vals = list(vals)
    return T if T in vals else U if U in vals else F


def evaluate(node: Node, st: list[int]) -> int:
    kind = node[0]
    if kind == "atom":
        return st[node[1]]
    if kind == "const":
        return node[1]
    if kind == "not":
        return k_not(evaluate(node[1], st))
    vals = [evaluate(c, st) for c in node[1]]
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


@lru_cache(maxsize=4096)
def _mention(term: str) -> re.Pattern:
    body = re.escape(unicodedata.normalize("NFC", term))
    return re.compile(r"(?<![\w-])(?<!\w\.)" + body + r"(?![\w-])(?!\.\w)", re.IGNORECASE)


def mentioned(text: str, term: str) -> bool:
    return bool(_mention(term).search(unicodedata.normalize("NFC", text)))


def _text(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)


def exact_state(a: Exact, values: dict) -> int:
    v = values.get(a.field, MISSING)
    if a.op == "exists":
        return F if v is MISSING else T
    if v is MISSING:
        return U
    if a.op == "mentions":
        return T if any(mentioned(_text(v), t) for t in a.value) else F
    return T if _CMP[a.op](v, a.value) else F


@dataclass(frozen=True, slots=True)
class Resolution:
    status: str
    answer: object = None
    case: str | None = None
    blocked_by: str | None = None
    blocking: frozenset = frozenset()
    partial: tuple = ()


def resolve(cases: tuple, otherwise: object, st: list[int], first: bool) -> Resolution:
    vals = []
    for case in cases:
        v = evaluate(case.expr, st)
        if first and v == U:
            return Resolution("blocked", blocked_by=case.id, blocking=frozenset(relevant(case.expr, st)))
        if first and v == T:
            return Resolution("accepted", case.then, case.id)
        vals.append(v)
    if first:
        return Resolution("accepted", otherwise, "otherwise")
    trues = tuple((c.id, c.then) for c, v in zip(cases, vals) if v == T)
    for case, v in zip(cases, vals):
        if v == U:
            return Resolution("blocked", blocked_by=case.id, blocking=frozenset(relevant(case.expr, st)),
                              partial=trues)
    if trues:
        return Resolution("accepted", tuple(t for _, t in trues), tuple(c for c, _ in trues))
    return Resolution("accepted", otherwise, "otherwise")
