from __future__ import annotations

from ..presentation import CLAIM, GUIDED, LETTERS, POLAR, SYSTEM, evidence, guided, text
from ..protocols import Framing, Question, ScoreRequest

PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3Adapter:
    name = "qwen3@5"
    max_slots = len(LETTERS)
    prefix_text, tail_text = PREFIX, TAIL

    def __init__(self, scorer):
        self.scorer = scorer
        self.prefix = self.template(self.prefix_text)
        self.tail = self.template(self.tail_text)
        self.letters = tuple(self._variants(c) for c in LETTERS)
        self.binary = tuple(self._variants(w, w.capitalize()) for w in CLAIM)
        self.polar = tuple(self._variants(w, w.capitalize()) for w in POLAR)
        self._cache: dict = {}
        self._prefix_cache: dict = {}

    @property
    def framing(self):
        return Framing(self.name, self.prefix_text, self.tail_text, self.binary, self.polar, self.letters)

    def validate_framing(self, framing):
        from ..errors import ConfigError
        if (framing.name, framing.binary, framing.polar, framing.letters, framing.revision) != (self.name, self.binary, self.polar, self.letters, 2):
            raise ConfigError("framing/tokenizer mismatch; re-run ss lock")

    def template(self, text: str) -> tuple[int, ...]:
        return self.scorer.tokenize(text, special=True)

    def content(self, text: str) -> tuple[int, ...]:
        return self.scorer.tokenize(text, special=False)

    def _variants(self, *words: str) -> tuple[int, ...]:
        ids = []
        for w in words:
            for v in (w, " " + w):
                toks = self.content(v)
                if len(toks) == 1 and toks[0] not in ids:
                    ids.append(toks[0])
        if not ids:
            raise ValueError(f"answer token {words[0]!r} is not a single token for this model")
        return tuple(ids)

    def state(self, rendered: str) -> tuple[int, ...]:
        return self.content(evidence(rendered))

    def suffix(self, q: Question) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        framing = q.framing or self.framing
        key = q.kind, q.text, q.options, q.excludes, q.candidate, framing
        hit = self._cache.get(key)
        if hit is None:
            self.validate_framing(framing)
            slots = (framing.polar if q.interrogative else framing.binary) if q.kind == "check" else framing.letters[:len(q.options)]
            hit = (self.content(text(q)) + self.template(framing.tail_text), slots)
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def prefix_for(self, guidance, framing=None):
        framing = framing or self.framing
        self.validate_framing(framing)
        if not guidance:
            return self.prefix if framing == self.framing else self.template(framing.prefix_text)
        key = guidance.digest, framing
        if key not in self._prefix_cache:
            if SYSTEM not in framing.prefix_text:
                raise ValueError("chat template does not preserve the system instruction")
            tokens = self.template(framing.prefix_text.replace(SYSTEM, GUIDED)) + self.content(guided(guidance))
            if len(self._prefix_cache) >= 64:
                self._prefix_cache.clear()
            self._prefix_cache[key] = tokens
        return self._prefix_cache[key]

    def request(self, state: tuple[int, ...], questions) -> ScoreRequest:
        questions = tuple(questions)
        guidance = questions[0].guidance if questions else None
        if any(q.guidance.digest != guidance.digest or q.framing != questions[0].framing for q in questions):
            raise ValueError("native request requires one guidance and framing binding")
        parts = [self.suffix(q) for q in questions]
        return ScoreRequest(self.prefix_for(guidance, questions[0].framing if questions else None), state, tuple(p[0] for p in parts), tuple(p[1] for p in parts))
