from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
import sys
import urllib.request
import uuid

import yaml

from semantic_switch import Actor, ConfigError, files
from semantic_switch.compiler import compile_spec, digest
from semantic_switch.families import FamilyIndex
from semantic_switch.evidence import LABEL_SCHEMA, role_of, LabelSource, Verification, label_record, reviewed_content, situation_record

SCHEMA = 2
REVIEW_SCHEMA = 5
CONVENTION = "benchmark convention"
UNPINNED = "unpinned"
CITED = "cited"
INDICATIONS = frozenset({"reading", "display", "configuration", "operator_report", "handover_note", "command", "verification",
                         "customer_statement", "agent_statement", "quoted", "account_field"})
STATUSES = frozenset({"current", "stale", "resolved", "negated", "hypothetical", "ambiguous", "superseded"})
DECISIONS = frozenset({"confirm", "replace", "skip"})
WITHDRAW = "withdraw"
KINDS = frozenset({"label", "evidence"})
ACTORS = frozenset({"human", "agent"})
FORBIDDEN = frozenset({"gold", "reference", "annotation", "annotations", "assertions", "scenario", "private", "physical",
                       "rule", "rules", "justified", "prohibited", "unresolved", "label_sources", "facts", "answer"})
REVIEW_FRACTION = .2
UNITS = {"frame": "one rendered condition/language evidence item",
         "situation": "a frame under the declared trained condition with complete supervision (situation_frames)",
         "family": "one connected assignment component",
         "target": "one (frame, reference target) membership"}
MECHANICAL = ("evidence", "provenance", "reference", "scope", "rules", "invariance", "assignment")
PROTOCOL = {"name": "two-tier-frame-review", "version": 3, "unit": "frame",
            "mechanical": {"scope": "every frame", "checks": list(MECHANICAL),
                           "invariance": "frames presenting one observable problem (public input under one effective criterion contract, reference revision "
                                         "and target scope; hidden annotations excluded) share one semantic answer: target values, assessed scope, unresolved "
                                         "state, prohibited actions and composed response; rule and assertion citations are provenance",
                           "limits": "mechanical checks establish rendering, provenance, recomputation and assignment consistency, not the prose's domain interpretation"},
            "review": {"scope": "every rule/composition branch, unresolved, ambiguous or contradictory frame and condition/language/role/safety stratum, "
                               "then a seeded sample of the remaining frames",
                      "fraction": REVIEW_FRACTION,
                      "actors": "human and agent reviewers use one review operation; each record names its reviewer and actor kind (and an agent's "
                                "model and session) for traceability; actor kind never weights or completes a decision",
                      "completion": "a current evidence confirmation with an empty target confirms the frame's observable evidence, reference memberships "
                                    "and composed answer against the cited sources/rules; only it completes a required frame",
                      "labels": "a label confirmation supports its declared target, or every target when empty; it never certifies evidence fidelity",
                      "corrections": "skip and replace are frame-scoped evidence decisions on the generated frame's binding; a replacement recomputes "
                                     "gold and needs a current evidence confirmation of the corrected binding",
                      "binding": ["frame evidence", "effective gold and composition", "criterion contract and reference revision",
                                  "source contract and authored rules", "protocol"]},
            "reviewed": "the used-source contract is complete, every frame passes mechanical validation, every required human frame has a current "
                        "evidence confirmation, and no dispute or unconfirmed replacement remains; sampled review never means every frame was read"}
PROTOCOL_DIGEST = digest(PROTOCOL)
EXTRACTORS = {"pdftotext-layout": ("pdftotext", "-v")}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    section: str
    edition: str
    licence: str
    use: str = CITED
    document: dict | None = None
    locator: tuple = ()
    extraction: dict | None = None
    excerpt: str | None = None
    sha256: str | None = None
    access: tuple = ()


@dataclass(frozen=True)
class Rule:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class Pack:
    name: str
    version: str
    subject: dict
    sources: tuple[Source, ...]
    rules: tuple[Rule, ...]
    extraction: str
    created: str
    root: Path
    authored: dict = field(default_factory=dict)
    applicability: dict | None = None

    def rule(self, rule_id) -> Rule:
        found = next((rule for rule in self.rules if rule.id == rule_id), None)
        if found is None:
            raise ConfigError(f"{self.name}: unknown rule {rule_id}; add it to the pack manifest")
        return found

    def source(self, source_id) -> Source:
        found = next((source for source in self.sources if source.id == source_id), None)
        if found is None:
            raise ConfigError(f"{self.name}: unknown source {source_id}")
        return found


@dataclass(frozen=True)
class Span:
    field: str
    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class Assertion:
    id: str
    subject: str
    property: str
    value: object
    event: str
    time: int
    indication: str
    status: str
    spans: tuple[Span, ...] = ()
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Frame:
    id: str
    family: str
    scenario: str
    condition: str
    input: dict
    assertions: tuple[Assertion, ...]
    tags: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    pairs: tuple[str, ...] = ()
    role: str | None = None

    @property
    def public(self) -> str:
        return digest(self.input)

    @property
    def content(self) -> str:
        return digest({"input": self.input, "assertions": [asdict(a) for a in self.assertions]})


@dataclass(frozen=True)
class Decision:
    value: object
    rules: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()
    branch: str | None = None


@dataclass(frozen=True)
class Gold:
    revision: str
    scope: tuple[str, ...]
    decisions: dict = field(default_factory=dict)
    prohibited: tuple[str, ...] = ()
    primitives: dict = field(default_factory=dict)

    @property
    def supervision(self) -> dict:
        return self.primitives or self.decisions

    @property
    def complete(self) -> bool:
        return all(d.value is not None for d in self.decisions.values())

    @property
    def targets(self) -> int:
        return len(self.decisions or self.primitives)

    def values(self) -> dict:
        return {target: d.value for target, d in self.decisions.items()}

    def record(self) -> dict:
        return {"revision": self.revision, "scope": list(self.scope), "complete": self.complete, "prohibited": list(self.prohibited),
                "decisions": {t: asdict(d) for t, d in self.decisions.items()}, "primitives": {t: asdict(d) for t, d in self.primitives.items()}}


@dataclass(frozen=True)
class Review:
    frame: str
    reviewer: str
    kind: str
    decision: str
    binding: str | None
    target: tuple = ()
    replacement: tuple | None = None
    actor: Actor | None = None
    note: str | None = None


@dataclass(frozen=True)
class Reference:
    evaluate: Callable[[Frame], Gold]
    revision: str
    scope: tuple[str, ...]
    fields: tuple[str, ...]
    criteria: dict
    compose: Callable[[Frame, Gold], Decision] | None = None
    private: tuple[str, ...] = ()

    def criterion(self, frame: Frame) -> str:
        if frame.condition not in self.criteria:
            raise ConfigError(f"{frame.id}: condition {frame.condition!r} has no effective criterion contract")
        return self.criteria[frame.condition]

    def problem(self, frame: Frame) -> str:
        return digest({"input": frame.input, "criterion": self.criterion(frame), "revision": self.revision, "scope": list(self.scope)})

    def composed(self, frame: Frame, gold: Gold) -> Decision | None:
        return None if self.compose is None else self.compose(frame, gold)


@dataclass(frozen=True)
class Reviewed:
    frames: list
    golds: dict
    composition: dict
    terms: dict
    bindings: dict
    original: dict
    reviews: tuple
    skipped: tuple = ()
    replaced: tuple = ()
    disputed: tuple = ()
    withdrawn: tuple = ()

    @property
    def stale(self) -> list[str]:
        return sorted({r.frame for r in self.reviews if r.binding not in (self.original.get(r.frame), self.bindings.get(r.frame))})

    def application(self) -> dict:
        return {"stale": self.stale, "skipped": list(self.skipped), "replaced": list(self.replaced), "disputed": list(self.disputed),
                "withdrawn": sorted({decision.frame for decision, _ in self.withdrawn})}


def assertions_of(rows) -> tuple[Assertion, ...]:
    return tuple(Assertion(**{**a, "spans": tuple(Span(**s) for s in a.get("spans", ())), "paths": tuple(a.get("paths", ()))}) for a in rows)


def load_pack(path) -> Pack:
    path = Path(path)
    data = files.read_json(path)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ConfigError(f"{path}: not a schema {SCHEMA} document pack manifest")
    sources = tuple(Source(**{**s, "locator": tuple(s.get("locator", ())), "access": tuple(s.get("access", ()))}) for s in data["sources"])
    rules = tuple(Rule(**r) for r in data["rules"])
    known = {s.id for s in sources}
    for kind, ids in (("source", [s.id for s in sources]), ("rule", [r.id for r in rules])):
        if len(ids) != len(set(ids)):
            raise ConfigError(f"{path}: duplicate {kind} IDs")
    unknown = sorted(r.id for r in rules if r.source != CONVENTION and r.source not in known)
    if unknown:
        raise ConfigError(f"{path}: rules {unknown} name no source section; cite a pinned section or {CONVENTION!r}")
    undeclared = sorted(s.id for s in sources if s.use != CITED and not s.use.startswith("unused:"))
    if undeclared:
        raise ConfigError(f"{path}: sources {undeclared} declare neither {CITED!r} nor 'unused: <reason>'")
    return Pack(data["name"], data["version"], data["subject"], sources, rules,
                data["extraction"], data["created"], path.parent, dict(data.get("authored", {})), data.get("applicability"))


def rules_digest(rules) -> str:
    return digest([asdict(rule) if isinstance(rule, Rule) else rule for rule in rules])


def tool_version(method: str) -> str:
    command = EXTRACTORS.get(method)
    if command is None:
        raise ConfigError(f"unknown extraction method {method!r}; use one of {sorted(EXTRACTORS)} or a manual excerpt")
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise ConfigError(f"{command[0]} is not installed; it is needed for {method}") from None
    return (done.stdout or done.stderr).splitlines()[0].strip()


def extract(document: Path, method: str, locator) -> str:
    if method != "pdftotext-layout":
        raise ConfigError(f"{method}: no automatic extractor")
    parts = []
    for entry in locator:
        first, last = entry["pages"]
        done = subprocess.run(["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(document), "-"], capture_output=True, text=True, check=False)
        if done.returncode:
            raise ConfigError(f"{document}: pdftotext pages {first}-{last} failed: {done.stderr.strip()}")
        parts.append(f"[{entry['section']}; pages {first}-{last}]\n{done.stdout}")
    return "\n".join(parts)


def source_state(pack: Pack, source: Source) -> tuple[str, list[str]]:
    missing = [name for name, ok in (("edition", source.edition and not source.edition.startswith(UNPINNED)),
                                     ("document", isinstance(source.document, dict) and source.document.get("path") and source.document.get("sha256")),
                                     ("locator", bool(source.locator) and all(entry.get("section") for entry in source.locator)),
                                     ("extraction", isinstance(source.extraction, dict) and source.extraction.get("method") and source.extraction.get("tool")),
                                     ("excerpt", source.excerpt and source.sha256)) if not ok]
    if missing:
        return UNPINNED, missing
    document, excerpt = pack.root / source.document["path"], pack.root / source.excerpt
    absent = [str(path.relative_to(pack.root)) for path in (document, excerpt) if not path.is_file()]
    if excerpt.is_file() and files.sha256_file(excerpt) != source.sha256:
        raise ConfigError(f"{pack.name}: source {source.id} excerpt {excerpt} differs from its pinned digest; review the changed document and re-pin")
    if document.is_file() and files.sha256_file(document) != source.document["sha256"]:
        raise ConfigError(f"{pack.name}: source {source.id} document {document} differs from its pinned digest; review the new edition and re-pin")
    if absent:
        return "absent", [f"local copy {path}: run python -m bench.evidence fetch {pack.root / 'pack.json'} {source.id}" for path in absent]
    method = source.extraction["method"]
    if method in EXTRACTORS and tool_version(method) == source.extraction["tool"] and extract(document, method, source.locator) != excerpt.read_text():
        raise ConfigError(f"{pack.name}: source {source.id} excerpt no longer reproduces from its document with {source.extraction['tool']}; re-pin")
    return "pinned", []


def contract(pack: Pack, used_rules, *, slices=(), authored=None) -> dict:
    used_rules = set(used_rules)
    check_rules(pack, used_rules)
    cited = {pack.rule(r).source for r in used_rules} - {CONVENTION}
    used = cited | set(slices)
    for source_id in used:
        pack.source(source_id)
    wrong = sorted(s.id for s in pack.sources if s.id in used and s.use != CITED)
    idle = sorted(s.id for s in pack.sources if s.id not in used and s.use == CITED)
    if wrong:
        raise ConfigError(f"{pack.name}: sources {wrong} are used by the frozen rules or slices but declared unused; pin them and declare {CITED!r}")
    if idle:
        raise ConfigError(f"{pack.name}: sources {idle} are declared {CITED!r} but no used rule or slice cites them; declare 'unused: <reason>'")
    unrecorded, unverified, states = [], [], {}
    for source in pack.sources:
        if source.id in used:
            state, needs = source_state(pack, source)
            states[source.id] = state
            (unrecorded if state == UNPINNED else unverified).extend(f"source {source.id}: {need}" for need in needs)
    actual = {"rules": rules_digest(pack.rules), **(authored or {})}
    for name, value in actual.items():
        if name not in pack.authored:
            unrecorded.append(f"authored {name}: {UNPINNED}")
        elif pack.authored[name] != value:
            raise ConfigError(f"{pack.name}: authored {name} changed since it was pinned; review it and re-pin with pin-authored")
    if "equipment" in pack.subject:
        if not isinstance(pack.applicability, dict) or not pack.applicability:
            unrecorded.append("applicability: record the documented model/option/firmware applicability")
        else:
            unrecorded += [f"applicability {key}: {value}" for key, value in pack.applicability.items() if isinstance(value, str) and value.startswith(UNPINNED)]
    skeleton = "skeleton" in pack.version
    if not skeleton and unrecorded:
        raise ConfigError(f"{pack.name} {pack.version}: a non-skeleton pack needs every used source, authored digest and applicability pinned: {unrecorded}")
    missing = [*unrecorded, *unverified, *([f"version {pack.version}: the used-source contract is recorded; bump the pack version from skeleton"]
                                           if skeleton and not unrecorded else [])]
    pins = {"sources": [asdict(s) for s in pack.sources if s.id in used], "authored": pack.authored, "applicability": pack.applicability, "version": pack.version}
    return {"sources": states, "unused": {s.id: s.use for s in pack.sources if s.id not in used}, "authored": sorted(pack.authored),
            "applicability": pack.applicability, "missing": missing, "complete": not missing, "digest": digest(pins)}


def update_source(manifest, source_id, **fields) -> dict:
    manifest = Path(manifest)
    data = files.read_json(manifest)
    entry = next((s for s in data["sources"] if s["id"] == source_id), None)
    if entry is None:
        raise ConfigError(f"{manifest}: no source {source_id}")
    entry.update(fields)
    files.publish(manifest, data)
    return entry


def local(manifest: Path, path) -> Path:
    path = Path(path)
    resolved = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(manifest.parent.resolve()):
        raise ConfigError(f"{path}: keep local source copies beside the pack manifest")
    return resolved


AGENT = "Mozilla/5.0 (X11; Linux x86_64)"


def download(url, target, sha256=None, *, timeout=120) -> str:
    target = Path(target)
    if not target.is_file():
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=timeout) as response:
            body = response.read()
        found = hashlib.sha256(body).hexdigest()
        if sha256 and found != sha256:
            raise ConfigError(f"{url}: downloaded content {found} differs from the pinned {sha256}; the publisher changed it: review and re-pin")
        files.write_new(target, body)
    found = files.sha256_file(target)
    if sha256 and found != sha256:
        raise ConfigError(f"{target}: local copy {found} differs from the pinned {sha256}; remove it or review and re-pin")
    return found


def fetch(manifest, source_id) -> dict:
    manifest = Path(manifest)
    entry = next((s for s in files.read_json(manifest)["sources"] if s["id"] == source_id), None)
    if entry is None or not isinstance(entry.get("document"), dict) or not entry["document"].get("path"):
        raise ConfigError(f"{manifest}: source {source_id} records no document path")
    target = local(manifest, entry["document"]["path"])
    found = download(entry["url"], target, entry["document"].get("sha256"))
    method = (entry.get("extraction") or {}).get("method")
    if entry.get("excerpt") and method in EXTRACTORS and not local(manifest, entry["excerpt"]).is_file():
        excerpt = local(manifest, entry["excerpt"])
        excerpt.parent.mkdir(parents=True, exist_ok=True)
        files.write_new(excerpt, extract(target, method, entry["locator"]))
    return {"source": source_id, "document": str(target), "sha256": found}


def pin(manifest, source_id, excerpt=None) -> dict:
    manifest = Path(manifest)
    entry = next((s for s in files.read_json(manifest)["sources"] if s["id"] == source_id), None)
    if entry is None:
        raise ConfigError(f"{manifest}: no source {source_id}")
    if not isinstance(entry.get("document"), dict) or not entry["document"].get("path") or not entry.get("locator"):
        raise ConfigError(f"{manifest}: source {source_id} needs a document path and page/section locator before pinning")
    document = local(manifest, entry["document"]["path"])
    if not document.is_file():
        raise ConfigError(f"{document}: obtain the source document first (python -m bench.evidence fetch {manifest} {source_id})")
    if excerpt is None:
        method = (entry.get("extraction") or {}).get("method", "pdftotext-layout")
        excerpt = local(manifest, f"sources/{source_id}.txt")
        excerpt.parent.mkdir(parents=True, exist_ok=True)
        text = extract(document, method, entry["locator"])
        excerpt.write_text(text)
        extraction = {"method": method, "tool": tool_version(method)}
    else:
        excerpt = local(manifest, excerpt)
        extraction = {"method": "manual", "tool": "reviewer-copied excerpt"}
    root = manifest.parent.resolve()
    return update_source(manifest, source_id, document={**entry["document"], "sha256": files.sha256_file(document), "bytes": document.stat().st_size},
                         extraction=extraction, excerpt=str(excerpt.relative_to(root)), sha256=files.sha256_file(excerpt))


def pin_authored(manifest, authored: dict) -> dict:
    manifest = Path(manifest)
    data = files.read_json(manifest)
    data["authored"] = {"rules": rules_digest(data["rules"]), **authored}
    files.publish(manifest, data)
    return data["authored"]


def check_rules(pack: Pack, used) -> dict:
    used = set(used)
    defined = {rule.id for rule in pack.rules}
    if used - defined:
        raise ConfigError(f"{pack.name}: reference uses undefined rules {sorted(used - defined)}")
    return {"defined": len(defined), "used": len(used), "orphans": sorted(defined - used),
            "conventions": sorted(rule.id for rule in pack.rules if rule.source == CONVENTION)}


class Prose:
    def __init__(self, field_name):
        self.field, self.parts, self.spans = field_name, [], {}

    def add(self, text, key=None):
        start = sum(map(len, self.parts))
        self.parts.append(text)
        if key is not None:
            self.spans.setdefault(key, []).append(Span(self.field, start, start + len(text), sha256_text(text)))
        return self

    @property
    def text(self) -> str:
        return "".join(self.parts)


def permuted(values, seed="menu-order"):
    values = list(values)
    shuffled = values[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled if shuffled != values else values[::-1]


def counterpart(frame_id: str) -> str:
    return ":".join(frame_id.split(":")[:2])


def lookup(data, path):
    for part in path.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def walk(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield (*path, key), key
            yield from walk(item, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from walk(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def leakage(frame: Frame, private=()) -> list[str]:
    found = [f"{'.'.join(map(str, path))}: private field name {key!r}" for path, key in walk(frame.input)
             if path and path[-1] == key and isinstance(key, str) and key.lower() in FORBIDDEN]
    terms = {frame.scenario, *(a.id for a in frame.assertions), *private}
    strings = [(path, value) for path, value in walk(frame.input) if not (path and path[-1] == value)]
    found += [f"{'.'.join(map(str, path))}: contains private term {term!r}" for path, value in strings for term in terms
              if term and re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", value)]
    return found


def validate_frame(frame: Frame, fields, private=()) -> Frame:
    unknown = sorted(set(frame.input) - set(fields))
    if unknown:
        raise ConfigError(f"{frame.id}: public input has undeclared fields {unknown}")
    leaked = leakage(frame, private)
    if leaked:
        raise ConfigError(f"{frame.id}: leakage: {'; '.join(leaked[:3])}")
    ids = [a.id for a in frame.assertions]
    if len(ids) != len(set(ids)):
        raise ConfigError(f"{frame.id}: duplicate assertion IDs")
    for a in frame.assertions:
        if a.indication not in INDICATIONS or a.status not in STATUSES:
            raise ConfigError(f"{frame.id}/{a.id}: unknown indication {a.indication!r} or status {a.status!r}")
        if a.value is None:
            raise ConfigError(f"{frame.id}/{a.id}: an assertion needs a stated value; missing evidence has no assertion")
        if not a.spans and not a.paths:
            raise ConfigError(f"{frame.id}/{a.id}: an assertion needs source spans or structured field paths")
        for s in a.spans:
            text = frame.input.get(s.field)
            if not isinstance(text, str) or not 0 <= s.start < s.end <= len(text):
                raise ConfigError(f"{frame.id}/{a.id}: span {s.field}[{s.start}:{s.end}] is outside the public text")
            if sha256_text(text[s.start:s.end]) != s.sha256:
                raise ConfigError(f"{frame.id}/{a.id}: span {s.field}[{s.start}:{s.end}] text changed since annotation")
        for p in a.paths:
            if lookup(frame.input, p) is None:
                raise ConfigError(f"{frame.id}/{a.id}: structured path {p} is absent from the public input")
    return frame


def validate_gold(frame: Frame, gold: Gold, scope, revision) -> Gold:
    if gold.revision != revision:
        raise ConfigError(f"{frame.id}: reference revision {gold.revision!r} is not the frozen {revision!r}")
    if tuple(gold.scope) != tuple(scope) or (set(gold.decisions) != set(scope) if gold.decisions else not set(gold.primitives) <= set(scope)):
        raise ConfigError(f"{frame.id}: reference target scope differs from the declared scope")
    missing = sorted(t for t, d in {**gold.decisions, **gold.primitives}.items() if not d.rules)
    if missing:
        raise ConfigError(f"{frame.id}: targets {missing} have no criterion; a reference without justification is invalid input")
    return gold


def criterion(definition: dict) -> str:
    facts = {name: {**fact, "cards": sorted(fact["cards"], key=lambda card: card["id"])} if "cards" in fact else fact
             for name, fact in definition.get("facts", {}).items()}
    return digest({**definition, "facts": facts})


def answer(gold: Gold, composed: Decision | None = None) -> dict:
    def semantic(decision):
        return {"value": decision.value, "scope": list(decision.scope), "unresolved": list(decision.unresolved)}
    return {**{target: semantic(d) for target, d in {**gold.decisions, **gold.primitives}.items()}, "prohibited": sorted(gold.prohibited),
            **({"composition": {**semantic(composed), "branch": composed.branch}} if composed else {})}


def conflicts(ref: Reference, frames, golds) -> list[tuple[str, str, list[str]]]:
    seen, found = {}, []
    for frame in frames:
        projected = answer(golds[frame.id], ref.composed(frame, golds[frame.id]))
        first, before = seen.setdefault(ref.problem(frame), (frame.id, projected))
        differing = sorted(a for a in before.keys() | projected.keys() if before.get(a) != projected.get(a))
        if differing:
            found.append((frame.id, first, differing))
    return found


def invariant(ref: Reference, frames, golds) -> dict:
    found = conflicts(ref, frames, golds)
    if found:
        frame, other, addresses = found[0]
        raise ConfigError(f"{frame} and {other}: one observable problem (public input under one effective criterion) has different reference answers "
                          f"at {addresses}; {len(found)} conflicting frames")
    return {"observable_problems": len({ref.problem(f) for f in frames}), "frames": len(frames)}


def references(ref: Reference, frames) -> dict:
    for frame in frames:
        validate_frame(frame, ref.fields, ref.private)
    golds = {f.id: validate_gold(f, ref.evaluate(f), ref.scope, ref.revision) for f in frames}
    invariant(ref, frames, golds)
    return golds


def review_draft(planned, annotated) -> dict:
    planned = {(a.subject, a.property, json.dumps(a.value), a.status) for a in planned}
    annotated = {(a.subject, a.property, json.dumps(a.value), a.status) for a in annotated}
    return {"omissions": sorted(planned - annotated), "additions": sorted(annotated - planned),
            "decision": "accept" if planned == annotated else "reject"}


def components(frames) -> dict:
    index = FamilyIndex()
    for frame in frames:
        index.add(("group", frame.family), frame.family)
        for other in (("key", frame.public), *(("group", s) for s in frame.seeds), ("pair", frame.id), *(("pair", p) for p in frame.pairs)):
            index.join(("group", frame.family), other)
    return {frame.id: index.name(("group", frame.family)) for frame in frames}


def _pick(sizes, order, target, minimum):
    reach = {(0, 0): ()}
    for family in order:
        for (total, count), chosen in list(reach.items()):
            key = (total + sizes[family], count + 1)
            if key[0] <= target and key not in reach:
                reach[key] = (*chosen, family)
        found = [chosen for (total, count), chosen in reach.items() if total == target and count >= minimum]
        if found:
            return max(found, key=len)
    return None


def contrast_quota(variants, quota, key) -> list:
    if quota is None:
        return variants
    rng = random.Random(f"{key}:variants")
    return [variants[0], *rng.sample(variants[1:], rng.randrange(min(quota, len(variants))))]


def situation_frames(frames, golds, condition) -> set:
    return {f.id for f in frames if f.condition == condition and all(d.value is not None for d in golds[f.id].supervision.values())}


def tally(group, counted, component, golds=None) -> dict:
    return {"frames": len(group), "situations": sum(f.id in counted for f in group), "families": len({component[f.id] for f in group}),
            **({"targets": sum(golds[f.id].targets for f in group)} if golds is not None else {})}


def assign(frames, quotas, *, seed, counted=None, families_minimum=None, check_only=lambda frame: False) -> dict:
    component = components(frames)
    counted = {frame.id for frame in frames} if counted is None else set(counted)
    sizes, forced = dict.fromkeys(component.values(), 0), set()
    for frame in frames:
        sizes[component[frame.id]] += frame.id in counted
        if check_only(frame):
            forced.add(component[frame.id])
    forced |= {family for family, size in sizes.items() if not size}
    order = sorted(sizes, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest())
    roles, free = {}, [family for family in order if family not in forced]
    families_minimum = families_minimum or {}
    for role in ("fit", "model_select", "acceptance_fit"):
        if not quotas.get(role):
            continue
        chosen = _pick(sizes, free, quotas[role], families_minimum.get(role, 1))
        if chosen is None:
            raise ConfigError(f"no whole-family selection yields exactly {quotas[role]} {role} situations across at least {families_minimum.get(role, 1)} families; "
                              f"available families {len(free)} with sizes {dict(sorted(Counter(sizes[f] for f in free).items()))}")
        roles |= dict.fromkeys(chosen, role)
        free = [family for family in free if family not in roles]
    roles |= dict.fromkeys([*free, *forced], "check")
    check = sum(sizes[f] for f, role in roles.items() if role == "check")
    if check < quotas.get("check", 0) or sum(r == "check" for r in roles.values()) < families_minimum.get("check", 1):
        raise ConfigError(f"check has {check} situations in {sum(r == 'check' for r in roles.values())} families; need at least {quotas.get('check', 0)} situations")
    return {"seed": seed, "quotas": dict(quotas), "families": dict(sorted(roles.items())), "components": component, "counted": sorted(counted),
            "counts": {role: tally([frame for frame in frames if roles[component[frame.id]] == role], counted, component)
                       for role in ("fit", "model_select", "acceptance_fit", "check")},
            "units": UNITS}


def targets(gold: Gold) -> set[tuple[str, ...]]:
    return {tuple(target.split("/")) for target in {**gold.decisions, **gold.primitives}}


def covers(review: Review, target=None) -> bool:
    return review.kind == "evidence" if target is None else review.target in ((), tuple(target))


def holds(review: Review, bindings, target=None) -> bool:
    return review.binding is not None and review.binding == bindings.get(review.frame) and covers(review, target)


def sources_for(gold: Gold, data, frame: Frame, producer: str, producer_digest: str, confirmed=(), bindings=None) -> dict:
    out = {}
    for target, decision in gold.supervision.items():
        if decision.value is None:
            continue
        path = tuple(target.split("/"))
        checked = tuple(Verification(r.kind, r.reviewer, r.actor,
                                              "source-check" if r.kind == "evidence" else "review", r.binding,
                                              reviewed_content(r.kind, data, path, decision.value), r.decision)
                        for r in confirmed if holds(r, bindings or {}, path))
        history = [LabelSource("computed", producer, producer_digest, artifact=frame.content, address=(frame.id, *path), verifications=checked).record()]
        fact, *member = path
        if member:
            out.setdefault(fact, {})[member[0]] = history
        else:
            out[fact] = history
    return out


def facts_of(gold: Gold) -> dict:
    out = {}
    for target, decision in gold.supervision.items():
        if decision.value is None:
            continue
        fact, *member = target.split("/")
        if member:
            out.setdefault(fact, {})[member[0]] = decision.value
        else:
            out[fact] = decision.value
    return out


def situation(frame: Frame, role) -> dict:
    return {"schema": LABEL_SCHEMA, "input": frame.input, "role": role, "groups": [frame.family, *frame.seeds], "pairs": list(frame.pairs), "tags": sorted(frame.tags)}


def label(plan, frame: Frame, gold: Gold, role, producer, producer_digest, created, selection, confirmed=(), bindings=None) -> dict | None:
    facts = {name: value for name, value in facts_of(gold).items() if name not in plan.retired}
    if not facts:
        return None
    sources = sources_for(gold, situation_record(plan, frame.input)["input"], frame, producer, producer_digest, confirmed, bindings)
    return label_record(plan, frame.input, facts, sources={name: value for name, value in sources.items() if name in facts},
                        role=role, groups=[frame.family, *frame.seeds], pairs=list(frame.pairs), tags=sorted(frame.tags),
                        selections=[{"selected_by": "manual", "selection_id": selection, "unit": "family"}], created=created)


def declared_bindings(plan, frames, bindings) -> dict:
    out = {}
    for frame in frames:
        out.setdefault(reviewed_content("evidence", situation_record(plan, frame.input)["input"], ()), set()).add(bindings[frame.id])
    return {key: sorted(found) for key, found in sorted(out.items())}


def binding(frame: Frame, gold: Gold, composed: Decision | None, criterion_: str, sources: str) -> str:
    return digest({"content": frame.content, "gold": gold.record(), "composition": asdict(composed) if composed else None,
                     "criterion": criterion_, "sources": sources, "protocol": PROTOCOL_DIGEST})


def cited(gold: Gold, composed: Decision | None) -> dict:
    return {**gold.decisions, **gold.primitives, **({"composition": composed} if composed else {})}


def review_plan(frames, golds, roles, *, seed, composition=None, stratum=None, fraction=REVIEW_FRACTION) -> dict:
    order = sorted(frames, key=lambda frame: hashlib.sha256(f"{seed}:{frame.id}".encode()).hexdigest())
    reasons, first = {}, {}
    for frame in order:
        gold = golds[frame.id]
        composed = (composition or {}).get(frame.id)
        keys = [("branch", decision.rules) for decision in {**gold.decisions, **gold.primitives}.values()]
        keys += [("composition", composed.branch, composed.rules)] if composed else []
        keys += [("stratum", stratum(frame, gold, roles[frame.id]))] if stratum else []
        for key in keys:
            if key not in first:
                first[key] = frame.id
                reasons.setdefault(frame.id, set()).add(key[0])
        if not gold.complete or any(d.value is None for d in gold.primitives.values()) or (composed and composed.value is None):
            reasons.setdefault(frame.id, set()).add("unresolved")
        if any(a.status == "ambiguous" for a in frame.assertions) or "contradiction" in frame.tags:
            reasons.setdefault(frame.id, set()).add("ambiguous_or_contradictory")
    pool = [frame.id for frame in order if frame.id not in reasons]
    sample = pool[:round(len(pool) * fraction)]
    for frame_id in sample:
        reasons[frame_id] = {"sample"}
    return {"protocol": PROTOCOL_DIGEST, "seed": seed, "fraction": fraction, "mechanical": len(frames),
            "review": {frame_id: sorted(reasons[frame_id]) for frame_id in sorted(reasons)},
            "mandatory": len(reasons) - len(sample), "branches": sum(key[0] == "branch" for key in first),
            "compositions": sum(key[0] == "composition" for key in first), "strata": sum(key[0] == "stratum" for key in first),
            "sampled": len(sample), "pool": len(pool)}


def review_of(row, where="reviews") -> Review:
    unknown = sorted(set(row) - set(Review.__dataclass_fields__))
    if unknown:
        raise ConfigError(f"{where}: review schema {REVIEW_SCHEMA} rows carry frame, reviewer, kind, decision, binding, target, replacement, actor and note; "
                          f"{row.get('frame')} has {unknown}")
    if not isinstance(row.get("actor"), dict):
        raise ConfigError(f"{where}: review schema {REVIEW_SCHEMA} rows record their actor (kind human, or kind agent with model and session); "
                          f"{row.get('frame')} has none")
    return Review(**{**row, "target": tuple(row.get("target", ())), "replacement": assertions_of(row["replacement"]) if row.get("replacement") else None,
                     "actor": Actor.parse(row["actor"])})


def load_reviews(path) -> tuple:
    return tuple(review_of(row, path) for row in files.read_jsonl(path))


def append_review(path, review: Review) -> dict:
    row = {**asdict(admit(review)), "target": list(review.target), "replacement": [asdict(a) for a in review.replacement] if review.replacement else None,
           "actor": review.actor.record()}
    row = json.loads(json.dumps(row))
    if review_of(row, path) != review:
        raise ConfigError(f"{review.frame}: the review does not round-trip through the review schema")
    files.append(path, files.jsonl([row]))
    return row


def check_actor(actor, where) -> Actor:
    if not isinstance(actor, Actor) or actor.kind not in ACTORS:
        raise ConfigError(f"{where}: a review records its actor: human, or agent with its model and session")
    return actor


def admit(review: Review) -> Review:
    if review.decision not in DECISIONS | {WITHDRAW} or review.kind not in KINDS:
        raise ConfigError(f"{review.frame}: review decision must be confirm/replace/skip on label or evidence, or withdraw one")
    if not review.binding:
        raise ConfigError(f"{review.frame}: a review names the frame binding it decides (see review_worksheet.jsonl)")
    if review.kind == "evidence" and review.target:
        raise ConfigError(f"{review.frame}: an evidence review is frame-scoped; review one target with kind label")
    if review.kind == "label" and review.decision not in ("confirm", WITHDRAW):
        raise ConfigError(f"{review.frame}: a label review confirms; skip or correct the frame with an evidence decision")
    if (review.decision == "replace") != bool(review.replacement):
        raise ConfigError(f"{review.frame}: a replacement, and only a replacement, carries the corrected assertions")
    if not isinstance(review.reviewer, str) or not review.reviewer.strip():
        raise ConfigError(f"{review.frame}: a review names its reviewer")
    check_actor(review.actor, review.frame)
    if review.decision in ("skip", WITHDRAW) and not (review.note or "").strip():
        raise ConfigError(f"{review.frame}: a {review.decision} records its reason in note")
    return review


def withdraws(withdrawal: Review, decision: Review) -> bool:
    return decision.decision != WITHDRAW and all(getattr(decision, key) == getattr(withdrawal, key) for key in ("frame", "reviewer", "kind", "target", "binding"))


def standing(log) -> tuple[tuple, tuple]:
    active, withdrawn = [], []
    for r in log:
        if r.decision != WITHDRAW:
            active.append(r)
            continue
        n = next((n for n in reversed(range(len(active))) if withdraws(r, active[n])), None)
        if n is None:
            raise ConfigError(f"{r.frame}: {r.reviewer} has no standing {scope_of(r)} decision on binding {r.binding[:16]} to withdraw; "
                              "only a decision's own reviewer withdraws it")
        withdrawn.append((active.pop(n), r))
    return tuple(active), tuple(withdrawn)


def act(review: Review) -> tuple:
    return review.decision, digest([asdict(a) for a in review.replacement]) if review.replacement else None


def review(pack: Pack, frames, reviews, ref: Reference, *, rules=(), slices=(), authored=None) -> Reviewed:
    (reviews, withdrawn), memo = standing(admit(r) for r in reviews), {}

    def bind(group, golds):
        composition = {f.id: c for f in group if (c := ref.composed(f, golds[f.id])) is not None}
        used = frozenset({*rules, *(r for f in group for d in cited(golds[f.id], composition.get(f.id)).values() for r in d.rules)})
        terms = memo[used] = memo.get(used) or contract(pack, used, slices=slices, authored=authored)
        return composition, terms, {f.id: binding(f, golds[f.id], composition.get(f.id), ref.criterion(f), terms["digest"]) for f in group}
    golds = {f.id: ref.evaluate(f) for f in frames}
    _, _, original = bind(frames, golds)
    by_frame = {}
    for r in reviews:
        by_frame.setdefault(r.frame, []).append(r)
    out, skipped, replaced, disputed = [], [], [], []
    for frame in frames:
        mine = by_frame.get(frame.id, ())
        unknown = sorted({"/".join(r.target) for r in mine if r.target and r.target not in targets(golds[frame.id])})
        if unknown:
            raise ConfigError(f"{frame.id}: label targets {unknown} are not targets of the task")
        live = [r for r in mine if holds(r, original, r.target)]
        if any(act(a) != act(b) and holds(b, original, a.target) for a in live for b in live):
            disputed.append(frame.id)
            out.append(frame)
            continue
        correction = next((r for r in live if r.decision != "confirm"), None)
        if correction is None:
            out.append(frame)
        elif correction.decision == "skip":
            skipped.append(frame.id)
        else:
            out.append(replace(frame, assertions=correction.replacement))
            replaced.append(frame.id)
    golds = references(ref, out)
    composition, terms, bindings = bind(out, golds)
    misplaced = sorted(r.frame for r in reviews if r.decision != "confirm" and r.binding == bindings.get(r.frame) != original.get(r.frame))
    if misplaced:
        raise ConfigError(f"{misplaced[:3]}: a skip or replacement decides the generated frame's worksheet binding, not a corrected one")
    return Reviewed(out, golds, composition, terms, bindings, original, reviews, tuple(skipped), tuple(replaced), tuple(disputed), withdrawn)


def scope_of(review: Review) -> str:
    return "frame evidence" if review.kind == "evidence" else "label " + "/".join(review.target) if review.target else "all labels"


def review_records(author: str, reviewed: Reviewed) -> list[dict]:
    def record(r, withdrawal=None):
        return {"frame": r.frame, "kind": r.kind, "target": list(r.target), "scope": scope_of(r), "author": author, "reviewer": r.reviewer,
                "actor": r.actor.record(), "self_review": r.reviewer == author, "reviewer_distinct": r.reviewer != author, "binding": r.binding, "note": r.note,
                "current": withdrawal is None and (holds(r, reviewed.bindings, r.target) if r.decision == "confirm" else r.binding == reviewed.original.get(r.frame)),
                "verdict": r.decision, "corrections": [asdict(a) for a in r.replacement] if r.replacement else None,
                "withdrawn": withdrawal and {"actor": withdrawal.actor.record(), "note": withdrawal.note}}
    return [record(r) for r in reviewed.reviews] + [record(r, withdrawal) for r, withdrawal in reviewed.withdrawn]


def mechanical(pack: Pack, reviewed: Reviewed, ref: Reference, roles) -> dict:
    failures, used = {}, set(reviewed.terms.get("sources", {}))
    for frame in reviewed.frames:
        found = []
        restored = replace(frame, input=json.loads(json.dumps(frame.input)), assertions=assertions_of(json.loads(json.dumps([asdict(a) for a in frame.assertions]))))
        gold, composed = reviewed.golds[frame.id], reviewed.composition.get(frame.id)
        for check, run in (("evidence", lambda: validate_frame(restored, ref.fields, ref.private)),
                           ("scope", lambda: validate_gold(frame, gold, ref.scope, ref.revision))):
            try:
                run()
            except ConfigError as error:
                found.append(f"{check}: {error}")
        recomputed = ref.evaluate(restored)
        if recomputed != gold:
            found.append("reference: the independently recomputed reference differs from the published gold")
        elif ref.composed(restored, recomputed) != composed:
            found.append("reference: the recomputed composition differs from the published composition")
        ids = {a.id for a in frame.assertions}
        for target, decision in cited(gold, composed).items():
            found += [f"provenance: {target} cites absent assertion {i}" for i in decision.evidence if i not in ids]
            found += [f"provenance: {target} cites absent structured path {p}" for p in decision.paths if lookup(frame.input, p) is None]
            for rule_id in decision.rules:
                rule = next((r for r in pack.rules if r.id == rule_id), None)
                if rule is None:
                    found.append(f"rules: {target} cites undefined rule {rule_id}")
                elif rule.source != CONVENTION and rule.source not in used:
                    found.append(f"rules: {target} cites {rule_id} whose source {rule.source} is outside the used-source contract")
        if roles.get(frame.id) is None or any(roles.get(p, roles[frame.id]) != roles[frame.id] for p in frame.pairs):
            found.append("assignment: the frame and its pairs do not share one assigned role")
        if found:
            failures[frame.id] = found
    for left, right, addresses in conflicts(ref, reviewed.frames, reviewed.golds):
        failures.setdefault(left, []).append(f"invariance: the observable problem of {right} has a different reference answer at {addresses}")
    return {"protocol": PROTOCOL_DIGEST, "checks": list(MECHANICAL), "frames": len(reviewed.frames), "passed": len(reviewed.frames) - len(failures),
            "failed": len(failures), "observable_problems": len({ref.problem(f) for f in reviewed.frames}),
            "failures": {frame: failures[frame] for frame in sorted(failures)[:50]}}


def coverage(author: str, reviewed: Reviewed, roles, *, plan, checked, counted=(), component=None) -> dict:
    frames, golds, bindings = reviewed.frames, reviewed.golds, reviewed.bindings
    counted, component = set(counted), component or {f.id: f.family for f in frames}
    records = review_records(author, reviewed)
    current = [r for r in records if r["current"]]
    live = [r for r in reviewed.reviews if r.decision == "confirm" and r.frame in golds and r.frame not in reviewed.disputed]
    confirmed = {r.frame for r in live if holds(r, bindings)}
    supported = {(r.frame, t) for r in live for t in targets(golds[r.frame]) if holds(r, bindings, t)}
    required = set(plan["review"]) & {f.id for f in frames}

    def count(group):
        need = [f for f in group if f.id in required]
        done = [f for f in need if f.id in confirmed]
        return {**tally(group, counted, component, golds), "required": tally(need, counted, component, golds),
                "confirmed": tally(done, counted, component, golds), "remaining_frames": len(need) - len(done),
                "labels": {"targets": sum(len(targets(golds[f.id])) for f in group), "confirmed": sum((f.id, t) in supported for f in group for t in targets(golds[f.id]))}}
    unconfirmed = sorted(set(reviewed.replaced) - confirmed)
    terms = reviewed.terms
    reasons = [f"sources: {need}" for need in terms["missing"]]
    reasons += [f"mechanical: {checked['failed']} of {checked['frames']} frames fail"] if checked["failed"] else []
    reasons += [f"review: {len(required - confirmed)} of {len(required)} required frames lack a current evidence confirmation"] if required - confirmed else []
    reasons += [f"disputes: {len(reviewed.disputed)} frames have conflicting current records"] if reviewed.disputed else []
    reasons += [f"replacements: {len(unconfirmed)} corrected frames lack a confirmation of the correction"] if unconfirmed else []
    self_review, distinct = any(r["self_review"] for r in current), any(r["reviewer_distinct"] for r in current)
    phenomena = sorted({tag for f in frames for tag in f.tags})
    actors = sorted({(r["reviewer"], str(Actor.parse(r["actor"]))) for r in current})
    return {"schema": REVIEW_SCHEMA, "protocol": PROTOCOL_DIGEST, "units": UNITS, "author": author,
            "reviewers": sorted({r["reviewer"] for r in current}), "actors": [{"reviewer": reviewer, "actor": actor} for reviewer, actor in actors],
            "self_review": self_review, "reviewer_distinct": distinct,
            "sources": {"complete": terms["complete"], "digest": terms["digest"], "states": terms["sources"], "missing": terms["missing"]},
            "mechanical": {key: checked[key] for key in ("frames", "passed", "failed", "observable_problems")},
            "frames": count(frames), "roles": {role: count([f for f in frames if roles.get(f.id) == role]) for role in sorted(set(roles.values()))},
            "phenomena": {tag: count([f for f in frames if tag in f.tags]) for tag in phenomena},
            "branches": plan["branches"], "compositions": plan["compositions"], "strata": plan["strata"],
            "records": {"current": len(current), "stale": sum(not r["current"] and not r["withdrawn"] for r in records), "withdrawn": len(reviewed.withdrawn)},
            "disputes": list(reviewed.disputed), "unconfirmed_replacements": unconfirmed, "stale": reviewed.stale, "skipped": list(reviewed.skipped),
            "status": "reviewed" if not reasons else f"unreviewed: {'; '.join(reasons[:4])}",
            "disclosure": ("sampled review: " + f"{len(confirmed & required)} of {len(frames)} frames received a current evidence confirmation; "
                           + (f"reviewers {', '.join(f'{reviewer} ({actor})' for reviewer, actor in actors)}; " if actors else "no current review; ")
                           + ("includes author self-review; " if self_review else "")
                           + "actor kind is disclosed, not validation")}


def lines(rows):
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows]


def document(pack: Pack) -> str:
    return " ".join(f"{rule.id}: {rule.text}" for rule in pack.rules)


def code_digest(*modules) -> str:
    return digest([[m.__name__, files.sha256_file(m.__file__)] for m in modules])


def author_of(producer: str, revision: str) -> str:
    return f"{producer}@{revision}"


def publish(directory, pack: Pack, reviewed: Reviewed, ref: Reference, *, scenarios, definition: dict, name: str, condition, assignment, producer,
            producer_digest, stratum=None, seed=0, extra=None) -> dict:
    directory = Path(directory)
    author = author_of(producer, ref.revision)
    plan = compile_spec(definition, name)
    text = yaml.safe_dump(definition, sort_keys=False, allow_unicode=True)
    frames, golds, composition, terms, bindings = reviewed.frames, reviewed.golds, reviewed.composition, reviewed.terms, reviewed.bindings
    roles = {f.id: assignment["families"][assignment["components"][f.id]] for f in frames}
    counted, component = set(assignment["counted"]), assignment["components"]
    confirmed = {}
    for r in reviewed.reviews:
        if r.decision == "confirm" and r.frame not in reviewed.disputed and holds(r, bindings, r.target):
            confirmed.setdefault(r.frame, []).append(r)
    review_plan_ = review_plan(frames, golds, roles, seed=seed, composition=composition, stratum=stratum)
    checked = mechanical(pack, reviewed, ref, roles)
    labels = [row for f in frames if f.condition == condition and (f.id in counted or roles[f.id] == "check")
              and (row := label(plan, f, golds[f.id], roles[f.id], producer, producer_digest, pack.created,
                                f"{pack.name}:{pack.version}:planned", confirmed.get(f.id, ()), bindings)) is not None]
    by_id = {f.id: f for f in frames}
    worksheet = [{"frame": frame_id, "reasons": reasons, "role": roles[frame_id], "family": component[frame_id], "condition": by_id[frame_id].condition,
                  "content_sha256": by_id[frame_id].content, "binding": bindings[frame_id], "criterion": ref.criterion(by_id[frame_id]),
                  "input": by_id[frame_id].input, "assertions": [asdict(a) for a in by_id[frame_id].assertions], "gold": golds[frame_id].record(),
                  "composition": asdict(composition[frame_id]) if frame_id in composition else None,
                  "targets": sorted("/".join(t) for t in targets(golds[frame_id])),
                  "rules": {r: asdict(pack.rule(r)) for r in sorted({r for d in cited(golds[frame_id], composition.get(frame_id)).values() for r in d.rules})}}
                 for frame_id, reasons in review_plan_["review"].items()]
    payloads = {"definition.yaml": [text], "scenarios.jsonl": lines(scenarios),
                "annotations.jsonl": lines({"frame": f.id, "public_sha256": f.public, "condition": f.condition,
                                            "assertions": [asdict(a) for a in f.assertions]} for f in frames),
                "situations.jsonl": lines(situation(f, roles[f.id]) for f in frames),
                "labels.jsonl": lines(labels), "reviews.jsonl": lines(review_records(author, reviewed)),
                "review_worksheet.jsonl": lines(worksheet),
                "reference.jsonl": lines({"frame": f.id, "family": component[f.id], "role": roles[f.id], "condition": f.condition,
                                          "public_sha256": f.public, "problem": ref.problem(f), "binding": bindings[f.id], **golds[f.id].record(),
                                          **({"composition": asdict(composition[f.id])} if f.id in composition else {})} for f in frames)}
    counts = {role: tally([f for f in frames if roles[f.id] == role], counted, component, golds) for role in assignment["counts"]}
    manifest = {"schema": SCHEMA, "pack": {"name": pack.name, "version": pack.version, "subject": pack.subject, "extraction": pack.extraction,
                                            "sources": [asdict(s) for s in pack.sources], "rules": [asdict(r) for r in pack.rules],
                                            "authored": pack.authored, "applicability": pack.applicability, "contract": terms},
                "reference": {"producer": producer, "code": producer_digest, "revision": ref.revision, "criteria": ref.criteria},
                "definition": {"file": name, "sha256": hashlib.sha256(text.encode()).hexdigest(), "condition": condition},
                "assignment": {"seed": assignment["seed"], "quotas": assignment["quotas"], "counts": counts, "units": UNITS},
                "unresolved_reference": {role: sum(roles[f.id] == role and f.condition == condition and f.id not in counted for f in frames)
                                         for role in ("fit", "model_select", "acceptance_fit", "check")}, "assignment_sha256": digest(assignment["families"]),
                "review_protocol": {**PROTOCOL, "digest": PROTOCOL_DIGEST},
                "review_plan": {key: value for key, value in review_plan_.items() if key != "review"} | {"review_frames": len(review_plan_["review"])},
                "review_application": reviewed.application(),
                "mechanical": checked,
                "review": {**coverage(author, reviewed, roles, plan=review_plan_, checked=checked, counted=counted, component=component),
                           "bindings": declared_bindings(plan, [f for f in frames if f.condition == condition], bindings)}, **(extra or {})}
    return files.publish(directory / "benchmark.json", manifest, payloads)


QUOTAS = {"fit": 100, "model_select": 150, "check": 200}
FAMILIES_MINIMUM = {"fit": 30, "model_select": 40, "check": 60}


def pack_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", type=int, default=3, help="per scenario: base plus a seeded sample of up to N-1 contrast variants, keeping families small")
    for role, size in QUOTAS.items():
        parser.add_argument(f"--{role.replace('_', '-')}", type=int, default=size)
    parser.add_argument("--output", type=Path, required=True)


def quota_assignment(frames, args, *, counted, check_only=lambda frame: False) -> dict:
    return assign(frames, {role: getattr(args, role) for role in QUOTAS}, seed=args.seed, counted=counted,
                  families_minimum=FAMILIES_MINIMUM if args.fit == QUOTAS["fit"] else None, check_only=check_only)


def publish_domain(args, module, pack: Pack, reviewed: Reviewed, ref: Reference, *, scenarios, assignment, producer, definition=None, name=None, extra=None) -> dict:
    return publish(args.output, pack, reviewed, ref, scenarios=scenarios, definition=definition or files.read_yaml(module.DEFINITION),
                   name=name or module.DEFINITION.name, condition=module.TRAINED, assignment=assignment, producer=producer,
                   producer_digest=code_digest(module, sys.modules[__name__]), stratum=module.stratum, seed=args.seed, extra=extra)


def reviewed_pack(args, module, pack: Pack, frames, ref: Reference, *, slices=()) -> Reviewed:
    return review(pack, frames, load_reviews(args.reviews) if getattr(args, "reviews", None) else (), ref, slices=slices, authored=module.authored())


def pack_main(module, description, commands):
    parser = argparse.ArgumentParser(prog=module.__name__, description=description)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (text, _, extend) in commands.items():
        p = sub.add_parser(name, help=text)
        extend(p)
    sub.add_parser("pin-authored", help="record the reviewed authored rule and catalogue digests in the pack manifest")
    args = parser.parse_args()
    try:
        if args.command == "pin-authored":
            print(json.dumps(pin_authored(module.PACK, module.authored()), indent=1))
            return
        manifest = commands[args.command][1](args)
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(manifest if manifest.get('kind') == 'label-view' else
                     {"assignment": manifest["assignment"], "review_plan": manifest["review_plan"], "review": manifest["review"]["status"]}, indent=1))


def normalized(value):
    return json.loads(json.dumps(value, default=list))


def cited_rules(gold: Gold, composed: Decision | None, targets) -> set:
    decisions = cited(gold, composed)
    return {r for target in (*targets, "composition") if target in decisions for r in decisions[target].rules}


def scoped_sources(pack: Pack, manifest, rules, authored) -> tuple[dict, list[str], list[str]]:
    published = {s["id"]: s for s in manifest["pack"]["sources"]}
    states, missing, stale = {}, [], []
    for source_id in sorted({pack.rule(r).source for r in rules} - {CONVENTION}):
        source = pack.source(source_id)
        if source.use != CITED:
            stale.append(f"source {source_id}: used by the comparison's rules but declared {source.use!r}")
        if normalized(asdict(source)) != published.get(source_id):
            stale.append(f"source {source_id}: pin changed since the pack was published")
        try:
            states[source_id], needs = source_state(pack, source)
        except ConfigError as error:
            states[source_id], needs = "stale", []
            stale.append(str(error))
        missing += [f"source {source_id}: {need}" for need in needs]
    if "skeleton" in pack.version:
        missing.append(f"version {pack.version}: skeleton pack")
    actual = {"rules": rules_digest(pack.rules), **(authored or {})}
    stale += [f"authored {name}: changed since it was pinned" for name, value in actual.items() if name in pack.authored and pack.authored[name] != value]
    missing += [f"authored {name}: {UNPINNED}" for name in actual if name not in pack.authored]
    return states, missing, stale


def scoped_review(data, pack: Pack, frames, golds, targets, criteria, *, compose=None, authored=None) -> dict:
    manifest, used, targets = data["manifest"], {f.id for f in frames}, tuple(targets)
    published = {row["frame"]: row for row in data["reference"] if row["frame"] in used}
    composition = {f.id: compose(f, golds[f.id]) if compose else None for f in frames}
    stale_criteria = sorted({f.condition for f in frames if criteria.get(f.condition) != manifest["reference"]["criteria"].get(f.condition)})
    changed = sorted(f.id for f in frames if f.id not in published or published[f.id]["public_sha256"] != f.public
                     or {k: published[f.id][k] for k in ("revision", "scope", "prohibited", "decisions", "primitives")}
                     != {k: v for k, v in normalized(golds[f.id].record()).items() if k in ("revision", "scope", "prohibited", "decisions", "primitives")}
                     or published[f.id].get("composition") != (normalized(asdict(composition[f.id])) if composition[f.id] else None))
    stale = set(changed) | {f.id for f in frames if f.condition in stale_criteria}
    rules = set().union(*(cited_rules(golds[f.id], composition[f.id], targets) for f in frames)) if frames else set()
    states, missing, stale_sources = scoped_sources(pack, manifest, rules, authored)
    failures = manifest["mechanical"]["failures"]
    mechanical = sorted(used & set(failures))
    unattributed = manifest["mechanical"]["failed"] - len(failures)
    records = [r for r in data["reviews"] if r["frame"] in used]
    current = [r for r in records if r["current"] and not r["withdrawn"] and r["frame"] not in stale]
    confirmed = {r["frame"] for r in current if r["kind"] == "evidence" and r["verdict"] == "confirm"}
    supported = {(r["frame"], t) for r in current if r["verdict"] == "confirm" for t in targets if r["kind"] == "evidence" or r["target"] in ([], t.split("/"))}
    required = {row["frame"] for row in data["review_worksheet"]} & used
    partial = sorted(f for f in required - confirmed if any((f, t) in supported for t in targets))
    disputes = sorted(used & set(manifest["review"]["disputes"]))
    unconfirmed = sorted(used & set(manifest["review"]["unconfirmed_replacements"]))
    reasons = [f"sources: {need}" for need in [*stale_sources, *missing]]
    reasons += [f"criteria: conditions {stale_criteria} are scored under criteria other than the reviewed ones"] if stale_criteria else []
    reasons += [f"reference: {len(changed)} frames' current reference differs from the published reviewed reference"] if changed else []
    reasons += [f"mechanical: {len(mechanical)} used frames fail"] if mechanical else []
    reasons += [f"mechanical: {unattributed} failing frames are not listed in the manifest"] if unattributed > 0 else []
    reasons += [f"review: {len(required - confirmed)} of {len(required)} required frames lack a current evidence confirmation"
                + (f" ({len(partial)} only partially label-reviewed)" if partial else "")] if required - confirmed else []
    reasons += [f"disputes: {len(disputes)} frames"] if disputes else []
    reasons += [f"replacements: {len(unconfirmed)} corrected frames lack a confirmation"] if unconfirmed else []
    actors = sorted({(r["reviewer"], str(Actor.parse(r["actor"]))) for r in current})
    self_review = any(r["self_review"] for r in current)
    return {"schema": REVIEW_SCHEMA, "protocol": PROTOCOL_DIGEST, "author": manifest["review"]["author"],
            "scope": {"frames": len(used), "targets": list(targets), "conditions": sorted({f.condition for f in frames}), "rules": sorted(rules),
                      "conventions": sorted(r for r in rules if pack.rule(r).source == CONVENTION), "criteria": {c: criteria.get(c) for c in sorted({f.condition for f in frames})}},
            "sources": {"states": states, "missing": missing, "stale": stale_sources,
                        "unused": sorted(s.id for s in pack.sources if s.id not in states)},
            "stale_criteria": stale_criteria, "stale_reference": changed, "mechanical_failures": mechanical,
            "frames": {"used": len(used), "required": len(required), "confirmed": len(confirmed & required), "partial": partial,
                       "remaining": len(required - confirmed)},
            "targets": {"scored": len(used) * len(targets), "supported": len(supported)},
            "reviewers": sorted({r["reviewer"] for r in current}), "actors": [{"reviewer": a, "actor": b} for a, b in actors],
            "self_review": self_review, "reviewer_distinct": any(r["reviewer_distinct"] for r in current),
            "records": {"current": len(current), "stale": sum(not r["current"] and not r["withdrawn"] or r["frame"] in stale for r in records),
                        "withdrawn": sum(bool(r["withdrawn"]) for r in records)},
            "disputes": disputes, "unconfirmed_replacements": unconfirmed,
            "status": "reviewed" if not reasons else f"unreviewed: {'; '.join(reasons[:4])}",
            "disclosure": (f"declared comparison review: {len(confirmed & required)} of {len(used)} used frames received a current evidence confirmation; "
                           + (f"reviewers {', '.join(f'{a} ({b})' for a, b in actors)}; " if actors else "no current review; ")
                           + ("includes author self-review; " if self_review else "")
                           + (f"authored conventions {sorted(r for r in rules if pack.rule(r).source == CONVENTION)}; " if any(pack.rule(r).source == CONVENTION for r in rules) else "")
                           + "actor kind is disclosed, not validation")}


def manifest_sha256(directory) -> str:
    return hashlib.sha256((Path(directory) / "benchmark.json").read_bytes()).hexdigest()


def read(directory, expected: str | None = None) -> dict:
    directory = Path(directory)
    observed = manifest_sha256(directory)
    if expected is not None and observed != expected:
        raise ConfigError(f"{directory}/benchmark.json: corpus manifest {observed} differs from the bound corpus {expected}; "
                          "the run stays bound to its original publication; use a different run directory for a different corpus")
    manifest = files.read_published(directory / "benchmark.json")
    if manifest is None or manifest.get("schema") != SCHEMA:
        raise ConfigError(f"{directory}: no schema {SCHEMA} benchmark manifest; regenerate the pack")
    out = {"manifest": manifest, "manifest_sha256": observed}
    for name, entry in manifest["payloads"].items():
        path = directory / entry["file"]
        if not path.is_file():
            raise ConfigError(f"{path}: payload named by {directory}/benchmark.json is missing; restore the publication")
        if files.sha256_file(path) != entry["sha256"]:
            raise ConfigError(f"{path}: payload digest differs from the benchmark manifest; restore the publication")
        out[name.removesuffix(".jsonl")] = files.read_jsonl(path) if name.endswith(".jsonl") else path.read_text()
    if manifest_sha256(directory) != observed:
        raise ConfigError(f"{directory}/benchmark.json: changed while it was read; retry once the publication is quiescent")
    return out


def materialize(directory, to, *, condition, definition, derive=lambda frame, gold: gold, tags=lambda frame: (), producer, producer_digest):
    with files.committing(to):
        to = Path(to).resolve()
        manifest_path = to / 'view.json'
        if manifest_path.exists():
            raise ConfigError(f'{manifest_path}: input view is already published; reuse its labels or choose a new destination')
        data = read(directory)
        parent, frames, meta = frames_of(data, directory)
        plan = compile_spec(definition, 'definition.yaml')
        labeled = {(digest(row['input']), row['role']) for row in data['labels']}
        counted = {frame.id for frame in frames if (frame.public, meta[frame.id]['role']) in labeled}
        references = {row['frame']: row for row in data['reference']}
        labels, mappings, support = [], [], {}
        selection = f"view:{data['manifest_sha256']}:{condition}"
        created = next((row['created'] for row in data['labels'] if row.get('created')), files.now())
        for frame in frames:
            if frame.condition != condition:
                continue
            role = meta[frame.id]['role']
            source = next((name for name in (frame.id, *frame.pairs) if name in counted), None)
            if source is None and role != 'check':
                continue
            record = references[frame.id]
            def decisions(values):
                return {name: Decision(**{key: tuple(value) if isinstance(value, list) else value for key, value in entry.items()})
                        for name, entry in values.items()}
            gold = Gold(record['revision'], tuple(record['scope']), decisions(record['decisions']), tuple(record['prohibited']), decisions(record['primitives']))
            current = replace(frame, tags=tuple(sorted({*frame.tags, *tags(frame)})))
            gold = derive(current, gold)
            row = label(plan, current, gold, role, producer, producer_digest, created, selection)
            if row is None:
                continue
            labels.append(row)
            mappings.append({'frame': frame.id, 'source_frame': source, 'family': meta[frame.id]['family'], 'role': role,
                             'reference': record['binding'], 'input_sha256': frame.public, 'targets': sorted(row['facts'])})
            support[role] = support.get(role, 0) + 1
        if not labels:
            raise ConfigError(f'{directory}: no labeled evidence for input view {condition}')
        text = yaml.safe_dump(definition, sort_keys=False, allow_unicode=True)
        declaration = {'schema': 1, 'kind': 'label-view', 'parent': {'path': str(Path(directory).resolve()), 'sha256': data['manifest_sha256']},
                       'condition': condition, 'definition_sha256': digest(definition), 'producer': {'name': producer, 'code': producer_digest},
                       'roles': support, 'review': 'unreviewed'}
        declaration['id'] = digest(declaration)
        result = files.publish(manifest_path, declaration, {'definition.yaml': [text], 'labels.jsonl': lines(labels), 'mapping.jsonl': lines(mappings)})
        return {**result, 'manifest': str(manifest_path),
                **{name: str(to / result['payloads'][name + suffix]['file']) for name, suffix in (('definition', '.yaml'), ('labels', '.jsonl'))}}

def stage(directory, to, expected: str | None = None, data: dict | None = None) -> dict:
    from semantic_switch.api import Runtime
    from semantic_switch.tools.data import Labels
    data = data or read(directory, expected)
    manifest, to = data["manifest"], Path(to)
    if to.exists():
        raise ConfigError(f"{to}: already exists; stage the pack into a new directory")
    definition = to / manifest["definition"]["file"]
    files.write_new(definition, data["definition.yaml"])
    paths = files.Paths.of(definition)
    files.write_new(paths.test, "".join(lines(data["labels"])))
    with Runtime() as runtime:
        labels = Labels(runtime.load(definition, draft=True))
    assignment = {k: v for k, v in labels.manifest.items() if k != "id"} | {"review": manifest["review"]}
    files.publish(paths.dataset, {**assignment, "id": digest(assignment)})
    return staged_record(manifest, definition, data["manifest_sha256"], directory)


def staged_record(manifest, definition, sha256, source) -> dict:
    paths = files.Paths.of(definition)
    published = files.read_json(paths.dataset)
    assignment = {k: v for k, v in published.items() if k != "id"} if isinstance(published, dict) else None
    if assignment is None or published["id"] != digest(assignment) or assignment["review"] != manifest["review"]:
        raise ConfigError(f"{paths.dataset}: not the evidence assignment staged from this pack; stage it into a new directory")
    roles = Counter(role_of(record["split"], record["fold"]) or record["split"] for record in assignment["families"].values())
    return {"definition": str(definition), "labels": str(paths.test), "evidence": str(paths.dataset), "audit": str(paths.audit),
            "dataset_id": published["id"], "families": dict(sorted(roles.items())), "pack": manifest["pack"]["name"], "review": manifest["review"]["status"],
            "assignment": manifest["assignment"]["counts"], "units": manifest["assignment"]["units"], "manifest_sha256": sha256,
            "corpus": str(Path(source).resolve())}


def stage_under(directory, root, expected: str | None = None) -> dict:
    data = read(directory, expected)
    manifest, sha256 = data["manifest"], data["manifest_sha256"]
    target = Path(root) / sha256[:16]
    if not target.exists():
        partial = Path(root) / f".{target.name}.partial.{uuid.uuid4().hex[:12]}"
        stage(directory, partial, sha256, data)
        partial.replace(target)
    definition = target / manifest["definition"]["file"]
    paths = files.Paths.of(definition)
    if not definition.is_file() or definition.read_text() != data["definition.yaml"] or not paths.test.is_file() \
            or paths.test.read_text() != "".join(lines(data["labels"])):
        raise ConfigError(f"{target}: its staged definition or labels differ from {directory}; keep its audit ledger and stage the pack under a new root")
    return staged_record(manifest, definition, sha256, directory)


def bind_corpus(directory, root, path) -> dict:
    path = Path(path)
    retained = files.read_json(path) if path.exists() else None
    if retained is not None and not (isinstance(retained, dict) and isinstance(retained.get("manifest_sha256"), str)):
        raise ConfigError(f"{path}: this run records no corpus manifest digest; start a new run directory")
    record = stage_under(directory, root, retained and retained["manifest_sha256"])
    if retained is not None and retained != record:
        raise ConfigError(f"{path}: the run is bound to staged record {retained['definition']}; the pack now stages {record['definition']}; "
                          "keep the original binding and use a different run directory")
    if retained is None:
        files.publish(path, record)
    return record


def load_frames(directory, expected: str | None = None) -> tuple[dict, list[Frame], dict]:
    return frames_of(read(directory, expected), directory)


def frames_of(data, directory) -> tuple[dict, list[Frame], dict]:
    frames, meta = [], {}
    for situation_, annotation, reference_ in zip(data["situations"], data["annotations"], data["reference"], strict=True):
        if annotation["frame"] != reference_["frame"] or annotation["public_sha256"] != digest(situation_["input"]):
            raise ConfigError(f"{directory}: situation, annotation and reference rows are not aligned; regenerate the pack")
        frames.append(Frame(reference_["frame"], situation_["groups"][0], reference_["frame"], reference_["condition"], situation_["input"],
                            assertions_of(annotation["assertions"]), tuple(situation_["tags"]), tuple(situation_["groups"][1:]), tuple(situation_["pairs"])))
        meta[reference_["frame"]] = {"role": reference_["role"], "family": reference_["family"]}
    return data["manifest"], frames, meta


def main():
    parser = argparse.ArgumentParser(prog="bench.evidence", description="explicit document-pack commands; only fetch reaches the network")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fetch", help="download a source's recorded document beside the pack manifest and verify its pinned digest")
    p.add_argument("manifest", type=Path)
    p.add_argument("source")
    p = commands.add_parser("pin", help="record a source's document, extraction and excerpt digests from the local copy")
    p.add_argument("manifest", type=Path)
    p.add_argument("source")
    p.add_argument("--excerpt", type=Path, help="a reviewer-copied excerpt instead of the automatic extraction")
    p = commands.add_parser("verify", help="verify the used-source contract against the local copies")
    p.add_argument("manifest", type=Path)
    p.add_argument("--rules", nargs="*", help="the used rules (default: every rule)")
    p = commands.add_parser("stage", help="write a published pack's definition and role-assigned labels and publish its evidence assignment and audit ledger")
    p.add_argument("pack", type=Path)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--to", type=Path, help="new stage directory")
    where.add_argument("--under", type=Path, help="stage into UNDER/<benchmark digest>, reusing that verified stage and its audit ledger")
    p.add_argument("--binding", type=Path, help="with --under: the run's corpus binding; created once, then every restart must stage the same publication")
    args = parser.parse_args()
    try:
        out = {"fetch": lambda: fetch(args.manifest, args.source), "pin": lambda: pin(args.manifest, args.source, args.excerpt),
               "verify": lambda: (lambda pack: contract(pack, args.rules or [r.id for r in pack.rules]))(load_pack(args.manifest)),
               "stage": lambda: stage(args.pack, args.to) if args.to else bind_corpus(args.pack, args.under, args.binding) if args.binding else stage_under(args.pack, args.under)}[args.command]()
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
