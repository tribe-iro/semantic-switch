from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

from . import files
from .compiler import digest
from .errors import ConfigError
from .families import valid_seeds as _valid_seeds
from .heads import FEATURE_IDENTITY, LexicalHead
from .results import freeze, thaw
from .selection import occurrences

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
RECEIPT_SCHEMA = 3
USES = ('adaptation', 'selection', 'acceptance_fit', 'override')
TUNING_USES = USES[:3]
EXPORT_SCHEMA = 12
BUNDLE_SCHEMA = 4
ROW_SCHEMA = 11


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _pairs(values):
    out = {}
    for key, value in values:
        if key in out:
            raise ValueError(f'duplicate field {key}')
        out[key] = value
    return out


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    kind: str
    sha256: str
    feature_identity: str | None = None


@dataclass(frozen=True, slots=True)
class LineageAssessment:
    state: str
    overlap: tuple = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class HeadBundle:
    identity: ArtifactIdentity
    manifest: Mapping
    head: LexicalHead
    folds: Mapping

    def fold(self, number):
        return self.folds.get(number)


def _object(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ConfigError(f'{path}: missing or oversized artifact')
    try:
        out = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_pairs)
        if not isinstance(out, dict):
            raise ValueError('expected object')
        return out
    except (ValueError, UnicodeError) as e:
        raise ConfigError(f'{path}: {e}') from None


def _head(stream):
    if os.fstat(stream.fileno()).st_size > MAX_ARTIFACT_BYTES:
        raise ValueError('oversized head payload')
    return LexicalHead.from_json(json.load(stream, object_pairs_hook=_pairs))


def load_bundle(path, fact=None, dataset=None):
    directory = Path(path)
    with files.open_published(directory / 'manifest.json') as (manifest, heads):
        try:
            return _bundle(manifest, {name: _head(stream) for name, stream in heads.items()}, fact, dataset)
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigError(f'{directory}: invalid head bundle: {e}') from None


def _bundle(manifest, heads, fact, dataset):
    if manifest is None or manifest.get('schema') != BUNDLE_SCHEMA or manifest.get('kind') != 'lexical-bundle' or manifest.get('feature_identity') != FEATURE_IDENTITY:
        raise ValueError('missing or unsupported schema, kind or feature identity')
    if any(not _is_digest(manifest.get(k)) for k in ('dataset_id', 'training_id', 'export_id')) or not isinstance(manifest.get('meaning_rev'), str) or not manifest['meaning_rev'] or not isinstance(manifest.get('trainer'), dict) or any(not isinstance(manifest['trainer'].get(k), str) or not manifest['trainer'][k] for k in ('name', 'version')):
        raise ValueError('missing dataset/export/meaning/trainer identity')
    if not isinstance(manifest.get('folds'), dict) or not isinstance(manifest.get('rows'), list):
        raise ValueError('invalid folds/fit rows')
    rows = manifest['rows']
    if set(manifest['folds']) != {'0', '1', '2', '3', '4'}:
        raise ValueError('bundle must describe every fold')
    if any(not isinstance(r, dict) or not {'id', 'group', 'fold', 'projection', 'fact'} <= r.keys() or
           type(r['fold']) is not int or not 0 <= r['fold'] < 5 or any(not isinstance(r.get(k), str) for k in ('id', 'group', 'projection', 'fact')) for r in rows):
        raise ValueError('malformed fit lineage')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate fit row IDs')
    if any(not _is_digest(row['id']) or not _valid_seeds(row.get('family_seeds')) or row['fact'] != manifest.get('fact') for row in rows):
        raise ValueError('invalid fit row identity/family seeds')
    for number, entry in manifest['folds'].items():
        if number not in ('0', '1', '2', '3', '4') or not isinstance(entry, dict):
            raise ValueError('invalid fold artifact')
        expected_fit = sorted(r['id'] for r in rows if r['fold'] != int(number))
        expected_held = sorted(r['id'] for r in rows if r['fold'] == int(number))
        if entry.get('fit_ids') != expected_fit or entry.get('held_ids') != expected_held:
            raise ValueError(f'fold {number}: fit/held membership mismatch')
        if {r['group'] for r in rows if r['fold'] != int(number)} & {r['group'] for r in rows if r['fold'] == int(number)}:
            raise ValueError(f'fold {number}: overlapping families')
    if fact is not None and (manifest.get('fact') != fact.name or manifest.get('meaning_rev') != fact.meaning_rev or tuple(manifest.get('classes', ())) != fact.keys):
        raise ValueError('fact, meaning or class mismatch')
    if dataset is not None and manifest.get('training_id') != dataset:
        raise ValueError('dataset identity mismatch; re-export and fit this dataset')
    head = heads[manifest['final']['head']]
    if tuple(head.classes) != tuple(manifest['classes']):
        raise ValueError('final head class mismatch')
    folds = {int(number): None if entry.get('skip') else heads[entry['head']] for number, entry in manifest['folds'].items()}
    return HeadBundle(ArtifactIdentity('lexical', digest(manifest), FEATURE_IDENTITY), freeze(manifest), head, folds)


def assess_lineage(rows, labels, examples):
    if rows is None:
        return LineageAssessment('unknown', reason='adaptation provenance unknown; load weights with Model.from_specialization or a verified pipeline-built base')
    seeds = [tuple(n) for r in rows for n in r.get('family_seeds', ())]
    families = {labels.group(n) for n in seeds if n in labels.families}
    families.update(r.get('group') for r in rows)
    projections = {r.get('projection') for r in rows}
    overlap = tuple(e.id for e in examples if e.group in families or e.projection in projections)
    return LineageAssessment('known_overlap', overlap) if overlap else LineageAssessment('declared_disjoint')


def validate_membership(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid declared training rows')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ('id', 'fact', 'projection', 'group')) or not _is_digest(row['id']) \
                or not _valid_seeds(row.get('family_seeds')) or not row['family_seeds'] or not _is_digest(row.get('component')) \
                or 'member' in row and (not isinstance(row['member'], str) or not row['member']):
            raise ValueError('invalid or incomplete training observation membership')
        if not isinstance(row.get('sources'), list) or not row['sources'] and row.get('uses') != ['override'] or any(not isinstance(source, dict) or not _is_digest(source.get('export_id'))
                or any(not isinstance(source.get(k), str) or not source[k] for k in ('path', 'task_family', 'namespace', 'group'))
                or not _is_digest(source.get('component')) or not isinstance(source.get('aliases'), list) or not _valid_seeds(source.get('family_seeds'))
                for source in row['sources']):
            raise ValueError('membership needs every export occurrence with export_id, path, task_family, namespace, source component, group aliases and family seeds; '
                             'only an override-only correction may have none')
        if not isinstance(row.get('label_sources'), list) or not row['label_sources'] or not all(map(_is_digest, row['label_sources'])):
            raise ValueError('membership needs the label-source digests of every target')
        if not isinstance(row.get('uses'), list) or not row['uses'] or row['uses'] != sorted(set(row['uses']) & set(USES), key=USES.index) \
                or len(set(row['uses'])) != len(row['uses']):
            raise ValueError(f'membership uses must be distinct values from {USES} in that order')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate training observation declarations')


@dataclass(frozen=True)
class Provenance:
    kind: str
    gguf_sha256: str
    receipt: str | None = None
    asset: str | None = None
    rows: tuple = field(default=(), compare=False, hash=False)

    def __post_init__(self):
        if self.kind not in ('base', 'specialization') or not _is_digest(self.gguf_sha256):
            raise ConfigError('provenance needs kind base or specialization and the verified GGUF digest')
        if (self.receipt is None) == (self.asset is None) or not _is_digest(self.receipt or self.asset):
            raise ConfigError('provenance comes from exactly one verified receipt or pipeline asset manifest')
        adapted = any('adaptation' in row['uses'] for row in self.rows)
        if adapted != (self.kind == 'specialization') or self.asset is not None and self.rows:
            raise ConfigError('adaptation membership contradicts the provenance kind')
        object.__setattr__(self, 'rows', freeze(list(self.rows)))

    @classmethod
    def base(cls, asset, gguf_sha256):
        return cls('base', gguf_sha256, asset=asset)

    def membership(self, *uses):
        return tuple(row for row in self.rows if set(uses) & set(row['uses']))

    def record(self):
        return {'kind': self.kind, 'gguf_sha256': self.gguf_sha256, 'receipt': self.receipt, 'asset': self.asset, 'rows': [thaw(row) for row in self.rows]}


def canonical_answer(presentations, slots):
    for entry in presentations:
        if entry['variant'] == 'canonical':
            mapped = entry['slot_map'][slots[entry['id']]]
            if mapped >= 0:
                return mapped
    return None


def native_score_requests(rows):
    from .protocols import ScoreRequest
    return [ScoreRequest(tuple(p['prefix']), tuple(p['input']), (tuple(p['question']),), (tuple(tuple(group) for group in p['answer_slots']),))
            for row in rows for p in row['native']['presentations']]


def native_model(resolved, source=None, provenance=None):
    from .api import native_spec
    return native_spec(source or resolved['model']['gguf'], dict(resolved['native']), provenance)


RECEIPT_NAME = 'receipt.json'


def bundle_task(plan, directory) -> dict:
    from .compiler import compile_spec
    directory = Path(directory)
    spec, listed, guidance = thaw(plan.spec), {}, {}
    for index, (name, entry) in enumerate(spec.get('guidance', {}).items()):
        relative = f'guidance/{index}.txt'
        listed[relative] = files.write_new(directory / relative, entry['text'])
        guidance[name] = {'file': relative}
    listed['task.yaml'] = files.write_new(directory / 'task.yaml', {**spec, **({'guidance': guidance} if guidance else {})})
    rebuilt = compile_spec(files.read_yaml(directory / 'task.yaml'), 'task.yaml', directory)
    if {f.name: f.meaning_rev for f in rebuilt.facts} != {f.name: f.meaning_rev for f in plan.facts} or rebuilt.switch_rev != plan.switch_rev:
        raise ConfigError('bundled task definition does not reproduce the selected task meanings; no deployment published')
    return listed


@dataclass(frozen=True)
class Receipt:
    path: Path
    data: Mapping
    provenance: Provenance

    @property
    def directory(self):
        return self.path.parent

    @property
    def id(self):
        return self.data['id']

    @property
    def gguf(self):
        return str(self.directory / self.data['gguf'])

    @property
    def native(self):
        return thaw(self.data['native'])

    def verified_task(self):
        verify_files(self.directory, self.data['files'])
        return self.directory / self.data['task']


def verify_files(directory, listed, failure='deployed file missing or changed since selection; re-run semantic-trainer select'):
    root = directory.resolve()
    for name, sha in listed.items():
        path = (directory / name).resolve()
        if not isinstance(name, str) or Path(name).is_absolute() or root not in path.parents or not _is_digest(sha):
            raise ValueError(f'{name}: deployment files need bundle-relative paths and sha256 digests')
        if not path.is_file() or files.sha256_file(path) != sha:
            raise ConfigError(f'{path}: {failure}')


def load_receipt(path) -> Receipt:
    path = Path(path)
    data = _object(path)
    try:
        if data.get('schema') != RECEIPT_SCHEMA or data.get('kind') != 'specialization-receipt':
            raise ValueError(f'unsupported deployment receipt (current schema {RECEIPT_SCHEMA}); re-run semantic-trainer select')
        if data.get('id') != artifact_digest({k: v for k, v in data.items() if k != 'id'}):
            raise ValueError('receipt digest mismatch')
        if not isinstance(data.get('files'), dict) or {data['gguf'], data['task'], data['lock']['path']} - set(data['files']):
            raise ValueError('receipt must list the deployed GGUF, task and lock')
        verify_files(path.parent, data['files'])
        if not isinstance(data.get('native'), dict) or not isinstance(data.get('execution'), dict):
            raise ValueError('missing requested native settings or realized execution')
        recorded = data['provenance']
        validate_membership(recorded['rows'])
        provenance = Provenance(recorded['kind'], data['files'][data['gguf']], receipt=data['id'], rows=tuple(recorded['rows']))
        return Receipt(path, freeze(data), provenance)
    except (KeyError, TypeError, ValueError, OSError) as e:
        raise ConfigError(f'{path}: invalid deployment receipt: {e}') from None


IDENTITY_FIELDS = ('schema', 'native', 'dataset_id', 'training_id', 'namespace', 'split', 'scope', 'situations', 'rows')


def export_identity(manifest) -> str:
    return digest({key: manifest[key] for key in IDENTITY_FIELDS})


def _export_rows(stream, manifest, fitting):
    from .examples import history_of
    seen = {}
    for line, entry in zip(stream, manifest['rows'], strict=True):
        if hashlib.sha256(line).hexdigest() != entry['sha256']:
            raise ValueError('row bytes differ from the export identity')
        row = json.loads(line, object_pairs_hook=_pairs)
        if not isinstance(row, dict) or row.get('schema') != ROW_SCHEMA or not {'id', 'fact', 'meaning_rev', 'classes', 'label', 'group', 'component', 'aliases', 'fold', 'split', 'projection', 'text', 'selections', 'label_sources', 'situations'} <= row.keys():
            raise ValueError(f'malformed or obsolete observation (current row schema {ROW_SCHEMA}); re-export')
        if row['id'] in seen or not isinstance(row['id'], str) or len(row['id']) != 64:
            raise ValueError('invalid or duplicate observation ID')
        seen[row['id']] = row
        if fitting and (row['split'] != 'tune' or manifest.get('split') != 'tune'):
            raise ValueError('fitting requires tune-only export')
        if row.get('kind') not in ('check', 'pick', 'rate') or not isinstance(row['classes'], list) or len(row['classes']) < 2:
            raise ValueError('invalid kind/class keys')
        if not all(isinstance(row.get(k), str) and row[k] for k in ('fact', 'group', 'meaning_rev')) or not _is_digest(row['component']) \
                or not _valid_seeds(row.get('family_seeds')) or not isinstance(row['aliases'], list) or row['group'] not in row['aliases']:
            raise ValueError('invalid fact/canonical family/meaning identity')
        if row['split'] not in ('tune', 'check') or type(row.get('taught')) is not bool:
            raise ValueError('invalid partition/taught status')
        if not isinstance(row.get('tags'), list) or any(not isinstance(tag, str) or not tag for tag in row['tags']):
            raise ValueError('invalid phenomenon tags')
        if type(row['fold']) is not int or not 0 <= row['fold'] < 5 or type(row['label']) is not int or not 0 <= row['label'] < len(row['classes']):
            raise ValueError('invalid fold/label')
        if len(set(row['classes'])) != len(row['classes']) or not all(isinstance(v, str) for v in row['classes']):
            raise ValueError('invalid class keys')
        if not isinstance(row.get('text'), str) or not isinstance(row.get('projection'), str):
            raise ValueError('invalid evidence/projection')
        if 'member' in row and (not isinstance(row['member'], str) or not row['member'] or row['kind'] != 'check' or not isinstance(row.get('candidate'), str)):
            raise ValueError('invalid membership path')
        if digest([[row['fact'], *([row['member']] if 'member' in row else [])], row['meaning_rev'], row['projection']]) != row['id']:
            raise ValueError('observation identity mismatch')
        if ('native' in row) != manifest['native']:
            raise ValueError('native presentations must match the manifest declaration')
        if manifest['native']:
            validate_native_export(row)
        occurrences(row['selections'])
        history_of(row['label_sources'])
        if any(row.get(k) != v for k, v in entry.items() if k != 'sha256'):
            raise ValueError('rows/manifest lineage mismatch')
    linked = {}
    for situation in manifest['situations']:
        if [tuple(target['address']) for target in situation['targets']] != [tuple(address) for address in manifest['scope']]:
            raise ValueError(f"situation {situation['id']}: targets differ from the declared scope")
        for target in situation['targets']:
            if 'observation' in target:
                row = seen.get(target['observation'])
                if row is None or (row['fact'], *([row['member']] if 'member' in row else [])) != tuple(target['address']) or situation['id'] not in row['situations']:
                    raise ValueError(f"situation {situation['id']}: index link {target['observation']} does not match an exported observation")
                linked.setdefault(row['id'], set()).add(situation['id'])
            elif target.get('outcome') not in ('exact', 'missing', 'unlabelled'):
                raise ValueError(f"situation {situation['id']}: unknown target outcome")
        if situation['complete'] != all('observation' in t or t['outcome'] == 'exact' for t in situation['targets']):
            raise ValueError(f"situation {situation['id']}: gold-completeness does not match its targets")
    if any(set(row['situations']) != linked.get(row['id'], set()) for row in seen.values()):
        raise ValueError('observation situation memberships differ from the source-situation index')
    return list(seen.values())


def read_export(path, *, fitting=True, native=False):
    path = Path(path)
    with files.open_published(path) as (manifest, payloads):
        if manifest is None or manifest.get('schema') != EXPORT_SCHEMA or type(manifest.get('native')) is not bool or not isinstance(manifest.get('rows'), list) or len(payloads) != 1:
            raise ConfigError(f'{path}: missing or unsupported export manifest (current schema {EXPORT_SCHEMA}); re-export')
        if native and not manifest['native']:
            raise ConfigError(f'{path}: LoRA requires a native export; re-export with --native')
        try:
            if manifest.get('export_id') != export_identity(manifest):
                raise ValueError('export identity mismatch; re-export')
            (stream,) = payloads.values()
            return manifest, _export_rows(stream, manifest, fitting)
        except (OSError, KeyError, TypeError, ValueError, UnicodeError) as e:
            raise ConfigError(f'{path}: invalid export: {e}') from None


def validate_native_export(row):
    native = row.get('native')
    if not isinstance(native, dict) or not isinstance(native.get('presentations'), list) or not native['presentations']:
        raise ValueError('invalid native presentations')
    tokenizer = native.get('tokenizer', {})
    vocab = tokenizer.get('vocab_size')
    if type(vocab) is not int or vocab < 2 or not _is_digest(tokenizer.get('vocab_sha256')) or not isinstance(tokenizer.get('probes'), list):
        raise ValueError('missing native tokenizer identity')
    if not isinstance(native.get('framing'), dict) or not isinstance(native.get('presentation'), dict) or native.get('authored_kind') != row['kind']:
        raise ValueError('missing deployed presentation identity')
    if 'execution' not in native or not isinstance(native['execution'], (dict, type(None))):
        raise ValueError('missing native execution record')
    for entry in native['presentations']:
        slots = entry.get('slot_map')
        if not isinstance(slots, list) or len(set(slots)) != len(slots):
            raise ValueError('invalid native slot map')
        if type(entry.get('label')) is not int or not 0 <= entry['label'] < len(slots) or any(type(s) is not int or not -1 <= s < len(row['classes']) for s in slots):
            raise ValueError('invalid native slot map/label')
        mapped = row['label'] if row['label'] in slots else -1
        if entry.get('global_label') != row['label'] or slots[entry['label']] != mapped:
            raise ValueError('native gold label mapping mismatch')
        if entry.get('id') != digest([row['id'], {k: v for k, v in entry.items() if k != 'id'}]):
            raise ValueError('native presentation digest mismatch')
        groups = entry.get('answer_slots')
        if not isinstance(groups, list) or not groups or len(groups) != len(slots):
            raise ValueError('answer slot width mismatch')
        texts = entry.get('answer_text')
        if not isinstance(texts, list) or len(texts) != len(groups) or any(not isinstance(words, list) or len(words) != len(group) or any(not isinstance(word, str) for word in words) for group, words in zip(groups, texts)):
            raise ValueError('missing native answer-token decoding')
        ids = []
        for group in groups:
            if not isinstance(group, list) or not group or any(type(t) is not int or not 0 <= t < vocab for t in group):
                raise ValueError('invalid answer slots')
            ids.extend(group)
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate answer token IDs')
        if entry.get('stage') not in ('full', 'mentions', 'fallback') or entry.get('variant') not in ('canonical', 'permuted'):
            raise ValueError('invalid presentation stage/variant')
        for segment in ('prefix', 'input', 'question'):
            values = entry.get(segment)
            if not isinstance(values, list) or any(type(t) is not int or not 0 <= t < vocab for t in values):
                raise ValueError('invalid native token range')
        if not entry['prefix'] + entry['input'] + entry['question']:
            raise ValueError('empty training sequence')
    return native


def fact_records(plan, record) -> dict:
    return {f.name: {**record(f), **({'members': {c.id: record(plan.facts[c.claims[0]]) for c in f.cards}} if f.labels else {})}
            for f in plan.visible}


def _candidate_fact(value):
    return {'head': value['head'], 'examples': [{k: r[k] for k in ('input', 'value')} for r in value['examples']],
            'policies': {source: {side: record['theta'] for side, record in sides.items()} for source, sides in value['policies'].items()},
            **({'members': {key: _candidate_fact(member) for key, member in value['members'].items()}} if 'members' in value else {})}


def candidate_digest(data):
    from .api import behavior_identity
    return digest({'spec': data['spec'], 'model': behavior_identity(data['models']['judge']),
                   'facts': {name: _candidate_fact(value) for name, value in data['facts'].items()},
                   'framing': data['framing'], 'returned_models': data['returned_models'], 'execution': data['execution']})


def snapshot_digest(snap, runtime):
    return candidate_digest({'returned_models': {'judge': list(snap.returned_models)}, 'framing': snap.framing.as_dict() if snap.framing else None, 'spec': snap.plan.spec,
                             'models': {'judge': runtime.identity('judge')}, 'execution': {'judge': runtime.execution('judge')},
                             'facts': fact_records(snap.plan, lambda fact: {
                                 'head': snap.heads[fact.name].to_json() if fact.name in snap.heads else None,
                                 'examples': [e.record() for e in snap.examples[fact.name]],
                                 'policies': {source: {side: {'theta': theta} for side, theta in by_fact[fact.name].thresholds(fact.kind).items()}
                                              for source, by_fact in snap.policies.items() if fact.name in by_fact}})})


def publish_bundle(directory, manifest, heads):
    return digest(files.publish(Path(directory) / 'manifest.json', manifest,
                                {name: [json.dumps(head.to_json(), allow_nan=False)] for name, head in heads.items()}))


artifact_digest = digest


def load_quality_profile(path) -> dict:
    from .guidance import REVISION
    profile = files.read_json(path)
    if profile is None:
        raise ConfigError(f"quality profile {path} not found")
    if not isinstance(profile, dict) or profile.get("schema") != 1 or profile.get("guidance_revision") != REVISION:
        raise ConfigError("unsupported guidance quality profile")
    maximum = profile.get("max_guidance_tokens")
    if type(maximum) is not int or maximum < 1 or not isinstance(profile.get("cohort"), str) or not isinstance(profile.get("evidence"), str):
        raise ConfigError("invalid guidance quality profile")
    return profile
