from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
import sys
import urllib.request

import yaml

from semantic_switch import ConfigError, files
from semantic_switch.compiler import compile_spec, digest
from semantic_switch import examples
from semantic_switch.families import FamilyIndex
from semantic_switch.examples import LABEL_SCHEMA, role_of, LabelSource, label_record, reviewed_content, situation_record
from semantic_switch.metrics import identity

SCHEMA = 2
REVIEW_SCHEMA = 2
CONVENTION = "benchmark convention"
UNPINNED = "unpinned"
CITED = "cited"
INDICATIONS = frozenset({"reading", "display", "configuration", "operator_report", "handover_note", "command", "verification",
                         "customer_statement", "agent_statement", "quoted", "account_field"})
STATUSES = frozenset({"current", "stale", "resolved", "negated", "hypothetical", "ambiguous", "superseded"})
DECISIONS = frozenset({"confirm", "replace", "skip"})
KINDS = frozenset({"label", "evidence"})
FORBIDDEN = frozenset({"gold", "reference", "annotation", "annotations", "assertions", "scenario", "private", "physical",
                       "rule", "rules", "justified", "prohibited", "unresolved", "label_sources", "facts", "answer"})
REVIEW_FRACTION = .2
UNITS = {"frame": "one rendered condition/language evidence item",
         "situation": "a frame under the declared trained condition with complete supervision (situation_frames)",
         "family": "one connected assignment component",
         "target": "one (frame, reference target) membership"}
MECHANICAL = ("evidence", "provenance", "reference", "scope", "rules", "invariance", "assignment")
PROTOCOL = {"name": "two-tier-frame-review", "version": 1, "unit": "frame",
            "mechanical": {"scope": "every frame", "checks": list(MECHANICAL),
                           "limits": "mechanical checks establish rendering, provenance, recomputation and assignment consistency, not the prose's domain interpretation"},
            "human": {"scope": "every rule/composition branch, unresolved, ambiguous or contradictory frame and condition/language/role/safety stratum, "
                               "then a seeded sample of the remaining frames",
                      "fraction": REVIEW_FRACTION,
                      "binding": ["frame evidence", "effective gold and composition", "reference revision", "source contract and authored rules", "protocol"]},
            "reviewed": "the used-source contract is complete, every frame passes mechanical validation, every required human frame has a current "
                        "confirmation, and no dispute or unconfirmed replacement remains; sampled human review never means every frame was read"}
PROTOCOL_DIGEST = identity(PROTOCOL)
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
    author: str
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
        return identity(self.input)

    @property
    def content(self) -> str:
        return identity({"input": self.input, "assertions": [asdict(a) for a in self.assertions]})


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
    content_sha256: str
    decision: str
    target: tuple = ()
    replacement: object = None
    binding: str | None = None


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
    return Pack(data["name"], data["version"], data["subject"], sources, rules, data["author"],
                data["extraction"], data["created"], path.parent, dict(data.get("authored", {})), data.get("applicability"))


def rules_digest(rules) -> str:
    return identity([asdict(rule) if isinstance(rule, Rule) else rule for rule in rules])


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
            "applicability": pack.applicability, "missing": missing, "complete": not missing, "digest": identity(pins)}


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


def fetch(manifest, source_id) -> dict:
    manifest = Path(manifest)
    entry = next((s for s in files.read_json(manifest)["sources"] if s["id"] == source_id), None)
    if entry is None or not isinstance(entry.get("document"), dict) or not entry["document"].get("path"):
        raise ConfigError(f"{manifest}: source {source_id} records no document path")
    target = local(manifest, entry["document"]["path"])
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
        files.write_new(target, body)
    found = files.sha256_file(target)
    if entry["document"].get("sha256") and found != entry["document"]["sha256"]:
        raise ConfigError(f"{target}: fetched document {found} differs from the pinned {entry['document']['sha256']}; the publisher changed it: review and re-pin")
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


def conflicts(reference, frames) -> list[tuple[str, str]]:
    seen, found = {}, []
    for frame in frames:
        key = identity({"input": frame.input, "assertions": [asdict(a) for a in frame.assertions], "condition": frame.condition})
        gold = reference(frame)
        if key in seen and seen[key][1] != gold:
            found.append((frame.id, seen[key][0]))
        seen.setdefault(key, (frame.id, gold))
    return found


def invariant(reference, frames) -> dict:
    found = conflicts(reference, frames)
    if found:
        raise ConfigError(f"{found[0][0]} and {found[0][1]}: identical public evidence produced different reference labels")
    return {"public_frames": len({identity({"input": f.input, "assertions": [asdict(a) for a in f.assertions], "condition": f.condition}) for f in frames}),
            "frames": len(frames)}


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


def sources_for(gold: Gold, data, frame: Frame, producer: str, producer_digest: str, confirmed=()) -> dict:
    out = {}
    for target, decision in gold.supervision.items():
        if decision.value is None:
            continue
        path = tuple(target.split("/"))
        checked = tuple(examples.Review(r.kind, r.reviewer, reviewed_content(r.kind, data, path, decision.value), r.decision)
                        for r in confirmed if r.target in ((), path))
        history = [LabelSource("rule", producer, producer_digest, frame.content, (frame.id, *path), checked).record()]
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


def label(plan, frame: Frame, gold: Gold, role, producer, producer_digest, created, selection, confirmed=()) -> dict | None:
    facts = facts_of(gold)
    if not facts:
        return None
    return label_record(plan, frame.input, facts, sources=sources_for(gold, situation_record(plan, frame.input)["input"], frame, producer, producer_digest, confirmed),
                        role=role, groups=[frame.family, *frame.seeds], pairs=list(frame.pairs), tags=sorted(frame.tags),
                        selections=[{"selected_by": "manual", "selection_id": selection, "unit": "family"}], created=created)


def binding(frame: Frame, gold: Gold, composed: Decision | None, sources: str) -> str:
    return identity({"content": frame.content, "gold": gold.record(), "composition": asdict(composed) if composed else None,
                     "sources": sources, "protocol": PROTOCOL_DIGEST})


def cited(frame: Frame, gold: Gold, composed: Decision | None) -> dict:
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
            "human": {frame_id: sorted(reasons[frame_id]) for frame_id in sorted(reasons)},
            "mandatory": len(reasons) - len(sample), "branches": sum(key[0] == "branch" for key in first),
            "compositions": sum(key[0] == "composition" for key in first), "strata": sum(key[0] == "stratum" for key in first),
            "sampled": len(sample), "pool": len(pool)}


def load_reviews(path) -> tuple:
    return tuple(Review(**{**row, "target": tuple(row.get("target", ())),
                           "replacement": assertions_of(row["replacement"]) if row.get("replacement") else None})
                 for row in files.read_jsonl(path))


def apply_reviews(frames, reviews) -> tuple[list[Frame], dict]:
    by_frame = {}
    for review in reviews:
        if review.decision not in DECISIONS or review.kind not in KINDS:
            raise ConfigError(f"{review.frame}: review decision must be confirm/replace/skip on label or evidence")
        if review.decision == "replace" and (review.kind != "evidence" or not review.replacement):
            raise ConfigError(f"{review.frame}: a replacement corrects evidence and carries the corrected assertions")
        if review.decision == "confirm" and not review.binding:
            raise ConfigError(f"{review.frame}: a confirmation names the frame binding it confirms (see review_worksheet.jsonl)")
        by_frame.setdefault(review.frame, []).append(review)
    out, original, stale, skipped, replaced, disputed = [], {}, [], [], [], []
    for frame in frames:
        original[frame.id] = frame.content
        mine = by_frame.get(frame.id, ())
        on = [r for r in mine if r.content_sha256 == frame.content]
        acts = {("skip",) if r.decision == "skip" else ("replace", identity([asdict(a) for a in r.replacement])) for r in on if r.decision != "confirm"}
        if len(acts) > 1 or (acts and any(r.decision == "confirm" for r in on)):
            disputed.append(frame.id)
            out.append(frame)
            continue
        if acts == {("skip",)}:
            skipped.append(frame.id)
            continue
        if acts:
            frame = replace(frame, assertions=next(r.replacement for r in on if r.decision == "replace"))
            replaced.append(frame.id)
        if any(r.content_sha256 not in (original[frame.id], frame.content) for r in mine):
            stale.append(frame.id)
        out.append(frame)
    return out, {"original": original, "stale": sorted(set(stale) | (set(by_frame) - set(original))), "skipped": skipped, "replaced": replaced,
                 "disputed": disputed}


def review_records(pack: Pack, reviews, original, bindings) -> list[dict]:
    return [{"frame": r.frame, "kind": r.kind, "target": list(r.target), "author": pack.author, "reviewer": r.reviewer,
             "self_review": r.reviewer == pack.author, "reviewer_distinct": r.reviewer != pack.author, "content_sha256": r.content_sha256,
             "binding": r.binding,
             "current": r.binding == bindings.get(r.frame) if r.decision == "confirm" else original.get(r.frame) == r.content_sha256,
             "verdict": r.decision, "corrections": [asdict(a) for a in r.replacement] if r.replacement else None} for r in reviews]


def mechanical(pack: Pack, frames, golds, *, reference, scope, revision, fields, roles, composition=None, contract_=None, private=()) -> dict:
    failures = {}
    used = set((contract_ or {}).get("sources", {}))
    for frame in frames:
        found = []
        restored = replace(frame, input=json.loads(json.dumps(frame.input)), assertions=assertions_of(json.loads(json.dumps([asdict(a) for a in frame.assertions]))))
        gold = golds[frame.id]
        for check, run in (("evidence", lambda: validate_frame(restored, fields, private)), ("scope", lambda: validate_gold(frame, gold, scope, revision))):
            try:
                run()
            except ConfigError as error:
                found.append(f"{check}: {error}")
        if reference(restored) != gold:
            found.append("reference: the independently recomputed reference differs from the published gold")
        ids = {a.id for a in frame.assertions}
        for target, decision in cited(frame, gold, (composition or {}).get(frame.id)).items():
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
    for left, right in conflicts(reference, frames):
        failures.setdefault(left, []).append(f"invariance: identical public evidence to {right} produced a different reference")
    return {"protocol": PROTOCOL_DIGEST, "checks": list(MECHANICAL), "frames": len(frames), "passed": len(frames) - len(failures),
            "failed": len(failures), "failures": {frame: failures[frame] for frame in sorted(failures)[:50]}}


def coverage(pack: Pack, frames, golds, reviews, roles, *, plan, checked, contract_, bindings, applied, counted=(), component=None) -> dict:
    counted, component = set(counted), component or {f.id: f.family for f in frames}
    records = review_records(pack, reviews, applied["original"], bindings)
    current = [r for r in records if r["current"]]
    confirmed = {r["frame"] for r in current if r["verdict"] == "confirm"}
    required = set(plan["human"]) & {f.id for f in frames}

    def count(group):
        need = [f for f in group if f.id in required]
        done = [f for f in need if f.id in confirmed]
        return {**tally(group, counted, component, golds), "required": tally(need, counted, component, golds),
                "confirmed": tally(done, counted, component, golds), "remaining_frames": len(need) - len(done)}
    unconfirmed = sorted(set(applied["replaced"]) - confirmed)
    reasons = [f"sources: {need}" for need in contract_["missing"]]
    reasons += [f"mechanical: {checked['failed']} of {checked['frames']} frames fail"] if checked["failed"] else []
    reasons += [f"human: {len(required - confirmed)} of {len(required)} required frames lack a current confirmation"] if required - confirmed else []
    reasons += [f"disputes: {len(applied['disputed'])} frames have conflicting current records"] if applied["disputed"] else []
    reasons += [f"replacements: {len(unconfirmed)} corrected frames lack a confirmation of the correction"] if unconfirmed else []
    self_review, distinct = any(r["self_review"] for r in current), any(r["reviewer_distinct"] for r in current)
    phenomena = sorted({tag for f in frames for tag in f.tags})
    return {"schema": REVIEW_SCHEMA, "protocol": PROTOCOL_DIGEST, "units": UNITS, "author": pack.author,
            "reviewers": sorted({r["reviewer"] for r in current}), "self_review": self_review, "reviewer_distinct": distinct,
            "sources": {"complete": contract_["complete"], "digest": contract_["digest"], "states": contract_["sources"], "missing": contract_["missing"]},
            "mechanical": {key: checked[key] for key in ("frames", "passed", "failed")},
            "frames": count(frames), "roles": {role: count([f for f in frames if roles.get(f.id) == role]) for role in sorted(set(roles.values()))},
            "phenomena": {tag: count([f for f in frames if tag in f.tags]) for tag in phenomena},
            "branches": plan["branches"], "compositions": plan["compositions"], "strata": plan["strata"],
            "disputes": applied["disputed"], "unconfirmed_replacements": unconfirmed, "stale": applied["stale"], "skipped": applied["skipped"],
            "status": "reviewed" if not reasons else f"unreviewed: {'; '.join(reasons[:4])}",
            "disclosure": ("sampled human review: " + f"{len(confirmed & required)} of {len(frames)} frames received a current human confirmation; "
                           + ("distinct reviewer recorded" if distinct and not self_review else
                              "includes author self-review; not independent expert validation" if self_review else "no current review"))}


def lines(rows):
    return [json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n" for row in rows]


def document(pack: Pack) -> str:
    return " ".join(f"{rule.id}: {rule.text}" for rule in pack.rules)


def code_digest(*modules) -> str:
    return identity([[m.__name__, files.sha256_file(m.__file__)] for m in modules])


def publish(directory, pack: Pack, *, scenarios, frames, golds, definition: dict, name: str, condition, assignment, producer, producer_digest,
            reference, fields, reviews=(), applied=None, composition=None, stratum=None, slices=(), rules=(), authored=None, seed=0, private=(), extra=None) -> dict:
    directory = Path(directory)
    plan = compile_spec(definition, name)
    text = yaml.safe_dump(definition, sort_keys=False, allow_unicode=True)
    roles = {f.id: assignment["families"][assignment["components"][f.id]] for f in frames}
    counted, component = set(assignment["counted"]), assignment["components"]
    applied = applied or {"original": {f.id: f.content for f in frames}, "stale": [], "skipped": [], "replaced": [], "disputed": []}
    composition = composition or {}
    used_rules = {*rules, *(r for f in frames for d in cited(f, golds[f.id], composition.get(f.id)).values() for r in d.rules)}
    terms = contract(pack, used_rules, slices=slices, authored=authored)
    bindings = {f.id: binding(f, golds[f.id], composition.get(f.id), terms["digest"]) for f in frames}
    confirmed = {}
    for r in reviews:
        if r.decision == "confirm" and r.binding == bindings.get(r.frame):
            confirmed.setdefault(r.frame, []).append(r)
    review_plan_ = review_plan(frames, golds, roles, seed=seed, composition=composition, stratum=stratum)
    checked = mechanical(pack, frames, golds, reference=reference, scope=next(iter(golds.values())).scope, revision=next(iter(golds.values())).revision,
                         fields=fields, roles=roles, composition=composition, contract_=terms, private=private)
    labels = [row for f in frames if f.condition == condition and (f.id in counted or roles[f.id] == "check")
              and (row := label(plan, f, golds[f.id], roles[f.id], producer, producer_digest, pack.created,
                                f"{pack.name}:{pack.version}:planned", confirmed.get(f.id, ()))) is not None]
    by_id = {f.id: f for f in frames}
    worksheet = [{"frame": frame_id, "reasons": reasons, "role": roles[frame_id], "family": component[frame_id], "condition": by_id[frame_id].condition,
                  "content_sha256": by_id[frame_id].content, "binding": bindings[frame_id], "input": by_id[frame_id].input,
                  "assertions": [asdict(a) for a in by_id[frame_id].assertions], "gold": golds[frame_id].record(),
                  "composition": asdict(composition[frame_id]) if frame_id in composition else None,
                  "rules": {r: asdict(pack.rule(r)) for r in sorted({r for d in cited(by_id[frame_id], golds[frame_id], composition.get(frame_id)).values() for r in d.rules})}}
                 for frame_id, reasons in review_plan_["human"].items()]
    payloads = {"definition.yaml": [text], "scenarios.jsonl": lines(scenarios),
                "annotations.jsonl": lines({"frame": f.id, "public_sha256": f.public, "condition": f.condition,
                                            "assertions": [asdict(a) for a in f.assertions]} for f in frames),
                "situations.jsonl": lines(situation(f, roles[f.id]) for f in frames),
                "labels.jsonl": lines(labels), "reviews.jsonl": lines(review_records(pack, reviews, applied["original"], bindings)),
                "review_worksheet.jsonl": lines(worksheet),
                "reference.jsonl": lines({"frame": f.id, "family": component[f.id], "role": roles[f.id], "condition": f.condition,
                                          "public_sha256": f.public, "binding": bindings[f.id], **golds[f.id].record(),
                                          **({"composition": asdict(composition[f.id])} if f.id in composition else {})} for f in frames)}
    counts = {role: tally([f for f in frames if roles[f.id] == role], counted, component, golds) for role in assignment["counts"]}
    manifest = {"schema": SCHEMA, "pack": {"name": pack.name, "version": pack.version, "subject": pack.subject, "extraction": pack.extraction,
                                            "sources": [asdict(s) for s in pack.sources], "rules": [asdict(r) for r in pack.rules],
                                            "authored": pack.authored, "applicability": pack.applicability, "contract": terms},
                "reference": {"producer": producer, "code": producer_digest}, "definition": {"file": name, "sha256": hashlib.sha256(text.encode()).hexdigest(), "condition": condition},
                "assignment": {"seed": assignment["seed"], "quotas": assignment["quotas"], "counts": counts, "units": UNITS},
                "unresolved_reference": {role: sum(roles[f.id] == role and f.condition == condition and f.id not in counted for f in frames)
                                         for role in ("fit", "model_select", "acceptance_fit", "check")}, "assignment_sha256": identity(assignment["families"]),
                "review_protocol": {**PROTOCOL, "digest": PROTOCOL_DIGEST},
                "review_plan": {key: value for key, value in review_plan_.items() if key != "human"} | {"human_frames": len(review_plan_["human"])},
                "review_application": {key: value for key, value in applied.items() if key != "original"},
                "mechanical": checked,
                "review": coverage(pack, frames, golds, reviews, roles, plan=review_plan_, checked=checked, contract_=terms, bindings=bindings, applied=applied,
                                   counted=counted, component=component), **(extra or {})}
    return files.publish(directory / "benchmark.json", manifest, payloads)


def read(directory) -> dict:
    directory = Path(directory)
    manifest = files.read_published(directory / "benchmark.json")
    if manifest is None or manifest.get("schema") != SCHEMA:
        raise ConfigError(f"{directory}: no schema {SCHEMA} benchmark manifest; regenerate the pack")
    out = {"manifest": manifest}
    for name, entry in manifest["payloads"].items():
        path = directory / entry["file"]
        if files.sha256_file(path) != entry["sha256"]:
            raise ConfigError(f"{path}: payload digest differs from the benchmark manifest; regenerate the pack")
        out[name.removesuffix(".jsonl")] = files.read_jsonl(path) if name.endswith(".jsonl") else path.read_text()
    return out


def stage(directory, to) -> dict:
    from semantic_switch.api import Runtime
    from semantic_switch.tools.data import Labels
    data = read(directory)
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
    roles = Counter(role_of(record["split"], record["fold"]) or record["split"] for record in assignment["families"].values())
    return {"definition": str(definition), "labels": str(paths.test), "evidence": str(paths.dataset), "audit": str(paths.audit),
            "dataset_id": digest(assignment), "families": dict(sorted(roles.items())), "pack": manifest["pack"]["name"], "review": manifest["review"]["status"],
            "assignment": manifest["assignment"]["counts"], "units": manifest["assignment"]["units"]}


def load_frames(directory) -> tuple[dict, list[Frame], dict]:
    data = read(directory)
    frames, meta = [], {}
    for situation_, annotation, reference_ in zip(data["situations"], data["annotations"], data["reference"], strict=True):
        if annotation["frame"] != reference_["frame"] or annotation["public_sha256"] != identity(situation_["input"]):
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
    p.add_argument("--to", type=Path, required=True)
    args = parser.parse_args()
    try:
        out = {"fetch": lambda: fetch(args.manifest, args.source), "pin": lambda: pin(args.manifest, args.source, args.excerpt),
               "verify": lambda: (lambda pack: contract(pack, args.rules or [r.id for r in pack.rules]))(load_pack(args.manifest)),
               "stage": lambda: stage(args.pack, args.to)}[args.command]()
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
