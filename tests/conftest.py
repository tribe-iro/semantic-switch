from __future__ import annotations

import re

import numpy as np
import pytest

from semantic_switch import Runtime
from semantic_switch.protocols import ScoreBlock

TOKEN = re.compile(r"<\|[^|]+\|>|\w+|\s|[^\w\s]")
PLAIN = re.compile(r"\w+|\s|[^\w\s]")


def letters_of(question: str) -> dict:
    return {m.group(2).strip(): m.group(1) for m in re.finditer(r"^([A-T])\) (.+)$", question, re.M)}


def default_oracle(state: str, question: str, width: int):
    s = state.lower()
    if "boom" in s and "legal action" not in question:
        raise RuntimeError("backend exploded")
    if "maybe" in s and "money back" in question:
        return [0.5, 0.5]
    if "garbled" in s and "money back" in question:
        return "lowmass"
    if "legal action" in question:
        return [0.97, 0.03] if ("lawyer will" in s or "lawsuit" in s) else [0.03, 0.97]
    if "money back" in question:
        return [0.96, 0.04] if "refund" in s and "do not refund" not in s else [0.04, 0.96]
    if "disputes the recorded payment" in question:
        return [0.95, 0.05] if "i did pay" in s else [0.05, 0.95]
    if "mainly about" in question:
        opts = letters_of(question)
        want = ("the product is broken" if "broken" in s or "error" in s else
                "a physical device" if "device" in s else
                "charges" if "charge" in s or "invoice" in s else "login" if "login" in s else "none of the above")
        p = np.full(width, 0.02 / max(width - 1, 1))
        hits = [letter for desc, letter in opts.items() if desc.startswith(want)] or \
               [letter for desc, letter in opts.items() if desc.startswith("none of the above")]
        for letter in hits:
            p[ord(letter) - 65] = 0.98
        return list(p / p.sum())
    if "urgent" in question:
        return [0.02, 0.03, 0.95] if "now" in s else [0.05, 0.9, 0.05] if "today" in s else [0.9, 0.05, 0.05]
    return [1.0 / width] * width


class FakeScorer:
    adapter = "qwen3"

    def __init__(self, oracle=default_oracle, name="fake-judge", accelerated=False):
        self.oracle = oracle
        self.name = name
        self.accelerated = accelerated
        self.vocab: dict = {}
        self.inv: list = []
        self.requests: list = []

    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]:
        out = []
        for piece in (TOKEN if special else PLAIN).findall(text):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.inv)
                self.inv.append(piece)
            out.append(self.vocab[piece])
        return tuple(out)

    def detokenize(self, tokens) -> str:
        return "".join(self.inv[t] for t in tokens)

    def score(self, requests):
        width = max(len(s) for r in requests for s in r.slots)
        block = ScoreBlock.empty(sum(len(r.suffixes) for r in requests), width)
        row = 0
        for r in requests:
            state = self.detokenize(r.state)
            for suffix, slots in zip(r.suffixes, r.slots):
                q = self.detokenize(suffix)
                self.requests.append((state, q))
                try:
                    p = self.oracle(state, q, len(slots))
                    if p == "lowmass":
                        block.logp[row, :2] = np.log([0.6, 0.4])
                        block.log_mass[row] = np.log(0.2)
                    else:
                        block.logp[row, :len(p)] = np.log(np.clip(p, 1e-9, 1))
                        block.log_mass[row] = 0.0
                except RuntimeError as e:
                    block.error[row] = str(e)
                row += 1
        return block

    def asked(self, needle: str) -> bool:
        return any(needle in q for _, q in self.requests)


@pytest.fixture
def fake():
    return FakeScorer()


@pytest.fixture
def rt(fake):
    return Runtime({"judge": fake})
