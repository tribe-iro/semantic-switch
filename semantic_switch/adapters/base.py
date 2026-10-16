from __future__ import annotations

from ..presentation import CLAIM, GUIDED, LETTERS, POLAR, SYSTEM, catalogued, evidence, guided, text
from ..protocols import Bounded, Framing, Question, ScoreRequest, Tokens

SUFFIX_BYTES = 8 << 20
PREFIX_BYTES = 32 << 20


class NativeAdapter:
    max_slots = len(LETTERS)

    def __init__(self, scorer, name: str, prefix_text: str, tail_text: str):
        self.scorer, self.name, self.prefix_text, self.tail_text = scorer, name, prefix_text, tail_text
        self.prefix = self.template(prefix_text)
        self.tail = self.template(tail_text)
        self.letters = tuple(self._variants(c) for c in LETTERS)
        self.binary = tuple(self._variants(w, w.capitalize()) for w in CLAIM)
        self.polar = tuple(self._variants(w, w.capitalize()) for w in POLAR)
        self.framing = Framing(name, prefix_text, tail_text, self.binary, self.polar, self.letters)
        self._suffixes = Bounded(SUFFIX_BYTES)
        self._prefixes = Bounded(PREFIX_BYTES)

    def validate_framing(self, framing):
        if framing is self.framing:
            return
        from ..errors import ConfigError
        if (framing.name, framing.binary, framing.polar, framing.letters, framing.revision) != (self.name, self.binary, self.polar, self.letters, 2):
            raise ConfigError("framing/tokenizer mismatch; re-run ss lock")

    def template(self, text: str) -> Tokens:
        return Tokens(self.scorer.tokenize(text, special=True))

    def content(self, text: str) -> Tokens:
        return Tokens(self.scorer.tokenize(text, special=False))

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

    def state(self, rendered: str) -> Tokens:
        return self.content(evidence(rendered))

    def suffix(self, q: Question) -> tuple[Tokens, tuple[tuple[int, ...], ...]]:
        framing = q.framing or self.framing
        key = q.kind, q.text, q.options, q.excludes, q.candidate, framing, q.entry
        hit = self._suffixes.get(key)
        if hit is None:
            self.validate_framing(framing)
            slots = (framing.polar if q.interrogative else framing.binary) if q.kind == "check" else framing.letters[:len(q.options)]
            tokens = Tokens(self.content(text(q)) + self.template(framing.tail_text))
            hit = self._suffixes.put(key, (tokens, slots), tokens.nbytes + sum(len(value.encode("utf-8")) for value in (q.text, *q.options, *q.excludes, q.candidate or "")))
        return hit

    def prefix_for(self, guidance, framing=None, catalog=None) -> Tokens:
        framing = framing or self.framing
        self.validate_framing(framing)
        if not guidance and catalog is None and framing is self.framing:
            return self.prefix
        key = guidance.digest if guidance else None, framing, catalog
        hit = self._prefixes.get(key)
        if hit is None:
            if not guidance:
                hit = self.template(framing.prefix_text)
            else:
                if SYSTEM not in framing.prefix_text:
                    raise ValueError("chat template does not preserve the system instruction")
                hit = Tokens(self.template(framing.prefix_text.replace(SYSTEM, GUIDED)) + self.content(guided(guidance)))
            if catalog is not None:
                hit = Tokens(hit + self.content(catalogued(catalog)))
            self._prefixes.put(key, hit, hit.nbytes + (len(key[0]) if key[0] else 0) + (len(catalog.text.encode("utf-8")) if catalog else 0))
        return hit

    def request(self, state: tuple[int, ...], questions) -> ScoreRequest:
        questions = tuple(questions)
        guidance = questions[0].guidance if questions else None
        if any(q.guidance.digest != guidance.digest or q.framing != questions[0].framing or q.catalog != questions[0].catalog for q in questions):
            raise ValueError("native request requires one guidance, framing and catalog binding")
        parts = [self.suffix(q) for q in questions]
        first = questions[0] if questions else None
        return ScoreRequest(self.prefix_for(guidance, first and first.framing, first and first.catalog), state, tuple(p[0] for p in parts), tuple(p[1] for p in parts))
