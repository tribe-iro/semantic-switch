from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .errors import ConfigError

REVISION = "guidance@1"


@dataclass(frozen=True, slots=True)
class Section:
    id: str
    text: str
    content_digest: str
    source_location: str | None = None


@dataclass(frozen=True, slots=True, init=False)
class Guidance:
    sections: tuple[Section, ...]
    digest: str = field(compare=False, repr=False)
    text: str = field(compare=False, repr=False)

    def __init__(self, text: str | None = None, *, sections=()):
        if text is not None and sections:
            raise ConfigError("guidance accepts text or sections, not both")
        rows = (("guidance", text, None),) if text is not None else sections
        resolved, seen = [], set()
        for row in rows:
            if not isinstance(row, Section) and (not isinstance(row, (tuple, list)) or len(row) != 3):
                raise ConfigError("guidance sections must contain id, text and source location")
            identifier, content, location = (row.id, row.text, row.source_location) if isinstance(row, Section) else row
            if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
                raise ConfigError("guidance section IDs must be unique nonempty strings")
            if not isinstance(content, str) or not content.strip():
                raise ConfigError(f"guidance.{identifier}: text must be a nonempty string")
            if location is not None and not isinstance(location, str):
                raise ConfigError("guidance source location must be text")
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            try:
                identifier.encode("utf-8")
                encoded = content.encode("utf-8")
            except UnicodeError:
                raise ConfigError(f"guidance.{identifier}: text must be valid UTF-8") from None
            resolved.append(Section(identifier, content, hashlib.sha256(encoded).hexdigest(), location))
            seen.add(identifier)
        object.__setattr__(self, "sections", tuple(resolved))
        payload = [REVISION, [(s.id, s.content_digest) for s in resolved]]
        object.__setattr__(self, "digest", hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest())
        object.__setattr__(self, "text", "\n\n".join(f"[{s.id}]\n{s.text}" for s in resolved))

    def __bool__(self):
        return bool(self.sections)

    @classmethod
    def load(cls, path, id="guidance"):
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ConfigError(f"guidance.{id}: cannot read {path}: {error}") from None
        return cls(sections=((id, text, str(path)),))

    def as_dict(self):
        return {s.id: {"text": s.text} for s in self.sections}

    @property
    def references(self):
        return tuple((s.id, s.content_digest) for s in self.sections)


def normalize(value):
    if value is None:
        return Guidance()
    if isinstance(value, Guidance):
        return value
    if isinstance(value, str):
        return Guidance(value)
    raise ConfigError("guidance must be text or a Guidance bundle")


def resolve_catalog(catalog, base=None):
    if not isinstance(catalog, dict):
        raise ConfigError("guidance must be a mapping of named text/file sections")
    sections = []
    for identifier, entry in catalog.items():
        if not isinstance(entry, dict) or set(entry) not in ({"text"}, {"file"}):
            raise ConfigError(f"guidance.{identifier}: specify exactly one of text or file")
        if "text" in entry:
            sections.append((identifier, entry["text"], None))
        else:
            if not isinstance(entry["file"], str):
                raise ConfigError(f"guidance.{identifier}: file must be a path")
            path = Path(entry["file"])
            loaded = Guidance.load(path if base is None else Path(base) / path, id=identifier)
            sections.extend(loaded.sections)
    return Guidance(sections=sections)


def bind(catalog, identifiers=None):
    bundle = catalog if isinstance(catalog, Guidance) else resolve_catalog(catalog)
    if identifiers is None:
        return bundle
    available = {s.id: s for s in bundle.sections}
    if not isinstance(identifiers, (list, tuple)) or any(not isinstance(i, str) or i not in available for i in identifiers):
        raise ConfigError(f"guidance references unknown sections: {identifiers!r}")
    return Guidance(sections=tuple(available[i] for i in identifiers))
