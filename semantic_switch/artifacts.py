from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

from . import files
from .compiler import digest, is_digest
from .context import json_value
from .errors import ConfigError
from .families import valid_seeds as _valid_seeds
from .heads import FEATURE_IDENTITY, LexicalHead
from .results import freeze, thaw
from .protocols import Tokens
from .selection import occurrences

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
RECEIPT_SCHEMA = 4
USES = ('adaptation', 'selection', 'acceptance_fit', 'override')
TUNING_USES = USES[:3]
EXPORT_SCHEMA = 14
EXPORT_TABLES = ('inputs', 'families', 'segments', 'answers')
BUNDLE_SCHEMA = 4


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
    if any(not is_digest(manifest.get(k)) for k in ('dataset_id', 'training_id', 'export_id')) or not isinstance(manifest.get('meaning_rev'), str) or not manifest['meaning_rev'] or not isinstance(manifest.get('trainer'), dict) or any(not isinstance(manifest['trainer'].get(k), str) or not manifest['trainer'][k] for k in ('name', 'version')):
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
    if any(not is_digest(row['id']) or not _valid_seeds(row.get('family_seeds')) or row['fact'] != manifest.get('fact') for row in rows):
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


def family_overlap(rows, scope, items, *, seeds=(), used=()) -> LineageAssessment:
    if rows is None:
        return LineageAssessment('unknown', reason='adaptation provenance unknown; load weights from a verified fitted lock (ss.judge(LOCK)) or a verified pipeline-built base')
    nodes = [*(n for r in rows for n in r.get('family_seeds', ())), *seeds]
    families = {scope.family(n) for n in nodes} | {scope.resolve(r['group']) for r in rows if r.get('group')}
    families.discard(None)
    projections = {r.get('projection') for r in rows} | set(used)
    overlap = tuple(dict.fromkeys(key for key, family, projection in items if scope.resolve(family) in families or projection in projections))
    return LineageAssessment('known_overlap', overlap) if overlap else LineageAssessment('declared_disjoint')


def validate_membership(rows):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('invalid declared training rows')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k] for k in ('id', 'fact', 'projection', 'group')) or not is_digest(row['id']) \
                or not _valid_seeds(row.get('family_seeds')) or not row['family_seeds'] or not is_digest(row.get('component')) \
                or 'member' in row and (not isinstance(row['member'], str) or not row['member']):
            raise ValueError('invalid or incomplete training observation membership')
        if not isinstance(row.get('sources'), list) or not row['sources'] and row.get('uses') != ['override'] or any(not isinstance(source, dict) or not is_digest(source.get('export_id'))
                or any(not isinstance(source.get(k), str) or not source[k] for k in ('path', 'task_family', 'namespace', 'group'))
                or not is_digest(source.get('component')) or not isinstance(source.get('aliases'), list) or not _valid_seeds(source.get('family_seeds'))
                for source in row['sources']):
            raise ValueError('membership needs every export occurrence with export_id, path, task_family, namespace, source component, group aliases and family seeds; '
                             'only an override-only correction may have none')
        if not isinstance(row.get('label_sources'), list) or not row['label_sources'] or not all(map(is_digest, row['label_sources'])):
            raise ValueError('membership needs the label-source digests of every target')
        if not isinstance(row.get('uses'), list) or not row['uses'] or row['uses'] != sorted(set(row['uses']) & set(USES), key=USES.index) \
                or len(set(row['uses'])) != len(row['uses']):
            raise ValueError(f'membership uses must be distinct values from {USES} in that order')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate training observation declarations')


def membership_record(rows):
    families, recorded = {}, []
    for row in rows:
        family = {key: row[key] for key in ('family_seeds', 'sources') if key in row}
        identity = artifact_digest(family)
        families.setdefault(identity, family)
        recorded.append({**{key: value for key, value in row.items() if key not in family}, 'family': identity})
    return {'families': families, 'rows': recorded}


def membership_rows(record):
    families = record.get('families')
    if families is None:
        return list(record['rows'])
    if not isinstance(families, Mapping) or any(not isinstance(value, Mapping) or set(value) - {'family_seeds', 'sources'} for value in families.values()):
        raise ValueError('invalid membership family table')
    rows = []
    for row in record['rows']:
        if not isinstance(row, Mapping) or row.get('family') not in families or {'family_seeds', 'sources'} & row.keys():
            raise ValueError('invalid membership family reference')
        rows.append({**families[row['family']], **{key: value for key, value in row.items() if key != 'family'}})
    return rows


@dataclass(frozen=True)
class Provenance:
    kind: str
    gguf_sha256: str
    receipt: str | None = None
    asset: str | None = None
    rows: tuple = field(default=(), compare=False, hash=False)
    adapter_sha256: str | None = None

    def __post_init__(self):
        if self.kind not in ('base', 'specialization') or not is_digest(self.gguf_sha256):
            raise ConfigError('provenance needs kind base or specialization and the verified GGUF digest')
        if (self.receipt is None) == (self.asset is None) or not is_digest(self.receipt or self.asset):
            raise ConfigError('provenance comes from exactly one verified receipt or pipeline asset manifest')
        adapted = any('adaptation' in row['uses'] for row in self.rows)
        if adapted != (self.kind == 'specialization') or self.asset is not None and self.rows:
            raise ConfigError('adaptation membership contradicts the provenance kind')
        if self.adapter_sha256 is not None and (self.kind != 'specialization' or not is_digest(self.adapter_sha256)):
            raise ConfigError('a runtime adapter digest belongs to a verified specialization receipt')
        object.__setattr__(self, 'rows', freeze(list(self.rows)))

    @classmethod
    def base(cls, asset, gguf_sha256):
        return cls('base', gguf_sha256, asset=asset)

    def membership(self, *uses):
        return tuple(row for row in self.rows if set(uses) & set(row['uses']))

    def record(self):
        return {'kind': self.kind, 'gguf_sha256': self.gguf_sha256, 'receipt': self.receipt, 'asset': self.asset, **thaw(membership_record(self.rows)),
                **({'adapter_sha256': self.adapter_sha256} if self.adapter_sha256 else {})}


def canonical_answer(presentations, slots):
    for entry in presentations:
        if entry.variant == 'canonical':
            mapped = entry.slot_map[slots[entry.id]]
            if mapped >= 0:
                return mapped
    return None


def native_score_requests(rows):
    from .protocols import ScoreRequest
    return [ScoreRequest(p.prefix, p.input, (p.question,), (p.answers.slots,)) for row in rows for p in row['presentations']]

def native_model(resolved, source=None, provenance=None, lora=None):
    from .api import native_spec
    return native_spec(source or resolved['model']['gguf'], dict(resolved['native']), provenance, lora)


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
    def lora(self):
        return str(self.directory / self.data['lora']) if self.data.get('lora') else None

    @property
    def native(self):
        return thaw(self.data['native'])

    def verified_task(self):
        verify_files(self.directory, self.data['files'])
        return self.directory / self.data['task']

    def model(self, **settings):
        from .api import native_spec
        return native_spec(self.gguf, {**self.native, **settings}, self.provenance, self.lora)


def verify_files(directory, listed, failure='deployed file missing or changed since selection; re-run semantic-trainer select'):
    root = directory.resolve()
    for name, sha in listed.items():
        path = (directory / name).resolve()
        if not isinstance(name, str) or Path(name).is_absolute() or root not in path.parents or not is_digest(sha):
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
        if not isinstance(data.get('files'), dict) or {data['gguf'], data['task'], data['lock']['path'], *([data['lora']] if data.get('lora') else [])} - set(data['files']):
            raise ValueError('receipt must list the deployed GGUF, adapter, task and lock')
        verify_files(path.parent, data['files'])
        if not isinstance(data.get('native'), dict) or not isinstance(data.get('execution'), dict):
            raise ValueError('missing requested native settings or realized execution')
        scope = data.get('scope')
        if not isinstance(scope, dict) or not isinstance(scope.get('targets'), list) or not scope['targets'] or type(scope.get('whole_task')) is not bool \
                or not isinstance(scope.get('binding'), str) or data.get('publication') not in ('application', 'scoped') \
                or (scope['whole_task'] and data['publication'] != 'application'):
            raise ValueError('missing target scope, binding or publication kind')
        recorded = data['provenance']
        rows = membership_rows(recorded)
        validate_membership(rows)
        if 'families' in recorded:
            families = {key: json_value(value) for key, value in recorded['families'].items()}
            rows = [{**families[row['family']], **{key: value for key, value in row.items() if key != 'family'}} for row in recorded['rows']]
        provenance = Provenance(recorded['kind'], data['files'][data['gguf']], receipt=data['id'], rows=tuple(rows),
                                adapter_sha256=data['files'][data['lora']] if data.get('lora') else None)
        return Receipt(path, freeze(data), provenance)
    except (KeyError, TypeError, ValueError, OSError) as e:
        raise ConfigError(f'{path}: invalid deployment receipt: {e}') from None


EXPORT_IDENTITY = ('schema', 'dataset_id', 'training_id', 'namespace', 'split', 'scope', 'task_family', 'native', 'targets')
EXPORT_PAYLOADS = ('observations', *EXPORT_TABLES, 'situations')
STAGES, VARIANTS = ('full', 'mentions', 'fallback'), ('canonical', 'permuted')
OWNED = ('label', 'fold', 'split', 'taught', 'pairs', 'tags', 'situations', 'selections', 'label_sources', 'label_derivation')
OBSERVATION = {'id', 'digest', 'target', 'input', 'family', *OWNED}


@dataclass(frozen=True, slots=True)
class AnswerGroups:
    slots: tuple
    text: tuple
    claim: str
    options: tuple
    excludes: tuple
    candidate: str | None


@dataclass(frozen=True, slots=True)
class Presentation:
    id: str
    stage: str
    variant: str
    prefix: tuple
    input: tuple
    question: tuple
    answers: AnswerGroups
    slot_map: tuple
    label: int

    @property
    def length(self):
        return len(self.prefix) + len(self.input) + len(self.question)


def export_payload(out, kind) -> str:
    return f'{Path(out).with_suffix("").name}.{kind}.jsonl'


def export_identity(manifest, observations, situations) -> str:
    return digest({**{key: manifest[key] for key in EXPORT_IDENTITY}, 'observations': list(observations), 'situations': list(situations)})


def export_line(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'


class Interned:
    def __init__(self, scope):
        self.scope, self.ordinals, self.values, self.digests = scope, {}, [], []

    def add(self, key, build=None):
        found = self.ordinals.get(key)
        if found is None:
            value = key if build is None else build()
            found = self.ordinals[key] = len(self.values)
            self.values.append(value)
            self.digests.append(digest([self.scope, value]))
        return found

    def record(self, value):
        return self.add(export_line(value), lambda: value)

    def lines(self):
        for value in self.values:
            yield export_line(list(value) if isinstance(value, (tuple, Tokens)) else value)


class ExportEncoder:
    def __init__(self, header):
        self.header, self.tables, self.digests, self.situations = dict(header), {kind: Interned(kind) for kind in ('targets', *EXPORT_TABLES)}, [], []

    def target(self, key, build):
        return self.tables['targets'].add(key, build)

    def presentation(self, observation, label, stage, variant, prefix, input_, question, groups, rendered, text, slot_map):
        segments = [self.tables['segments'].add(Tokens(tokens)) for tokens in (prefix, input_, question)]
        groups = tuple(tuple(group) for group in groups)
        answers = self.tables['answers'].add((groups, *rendered), lambda: {
            'slots': [list(group) for group in groups], 'text': [list(words) for words in text(groups)], 'claim': rendered[0], 'options': list(rendered[1]),
            'excludes': list(rendered[2]), **({'candidate': rendered[3]} if rendered[3] else {})})
        slot_map = list(slot_map)
        gold = slot_map.index(label if label in slot_map else -1)
        identity = digest([observation, stage, variant, slot_map, gold, *(self.tables['segments'].digests[s] for s in segments), self.tables['answers'].digests[answers]])
        return [identity, *segments, answers, slot_map, stage, variant]

    def observation(self, owned, target, projection, family, presentations=None):
        row = {**owned, 'target': target, 'input': self.tables['inputs'].record(projection),
               'family': self.tables['families'].add(family['group'], lambda: family), **({'presentations': presentations} if presentations is not None else {})}
        row['digest'] = digest([{key: row[key] for key in ('id', *OWNED)}, self.tables['targets'].digests[target], self.tables['inputs'].digests[row['input']],
                                self.tables['families'].digests[row['family']], [entry[0] for entry in presentations or ()]])
        self.digests.append(row['digest'])
        return export_line(row)

    def publish(self, path, observations, situations):
        path = Path(path)
        out = path.name.removesuffix('.manifest.json')

        def indexed():
            for entry in situations():
                self.situations.append(entry)
                yield export_line(entry)
            self.header.update(schema=EXPORT_SCHEMA, targets=self.tables['targets'].values,
                               counts={'observations': len(self.digests), 'situations': len(self.situations), **{kind: len(self.tables[kind].values) for kind in EXPORT_TABLES}})
            self.header['export_id'] = export_identity(self.header, self.digests, self.situations)
        files.publish(path, self.header, {export_payload(out, 'observations'): observations, **{export_payload(out, kind): self.tables[kind].lines() for kind in EXPORT_TABLES},
                                          export_payload(out, 'situations'): indexed()})
        return self.header


def _lines(stream):
    for raw in stream:
        yield json.loads(raw, object_pairs_hook=_pairs)


def _ordinal(value, table, what):
    if type(value) is not int or not 0 <= value < len(table):
        raise ValueError(f'{what} reference {value!r} outside its table')
    return table[value]


def _strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _target(entry):
    if not isinstance(entry, dict) or entry.get('kind') not in ('check', 'pick', 'rate') or not all(isinstance(entry.get(k), str) and entry[k] for k in ('fact', 'meaning_rev', 'claim')):
        raise ValueError('invalid target descriptor kind/fact/meaning')
    classes = entry.get('classes')
    if not _strings(classes) or len(classes) < 2 or len(set(classes)) != len(classes) or not isinstance(entry.get('values'), list) or len(entry['values']) != len(classes):
        raise ValueError('invalid target class keys/values')
    if 'member' in entry and (not isinstance(entry['member'], str) or not entry['member'] or entry['kind'] != 'check' or not isinstance(entry.get('candidate'), str)):
        raise ValueError('invalid membership path')
    if not _strings(entry.get('excludes')) or not isinstance(entry.get('guidance'), dict) or not isinstance(entry.get('guidance_digest'), str):
        raise ValueError('invalid target excludes/guidance')
    return entry


def _input(entry):
    if not isinstance(entry, dict) or set(entry) != {'bound_input', 'projection', 'text', 'decision_key'} or not isinstance(entry['bound_input'], dict) \
            or not all(isinstance(entry[k], str) for k in ('projection', 'text', 'decision_key')):
        raise ValueError('invalid input projection')
    return entry


def _family(entry):
    if not isinstance(entry, dict) or set(entry) != {'group', 'component', 'aliases', 'family_seeds'} or not isinstance(entry['group'], str) or not entry['group'] \
            or not is_digest(entry['component']) or not _valid_seeds(entry['family_seeds']) or not _strings(entry['aliases']) or entry['group'] not in entry['aliases']:
        raise ValueError('invalid canonical family')
    return entry


def _segment(tokens, vocab):
    if not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < vocab for t in tokens):
        raise ValueError('invalid native token range')
    return Tokens(tokens)


def _answers(entry, vocab):
    if not isinstance(entry, dict) or set(entry) - {'candidate'} != {'slots', 'text', 'claim', 'options', 'excludes'}:
        raise ValueError('invalid answer groups')
    groups, texts = entry['slots'], entry['text']
    if not isinstance(groups, list) or not groups or any(not isinstance(group, list) or not group for group in groups):
        raise ValueError('invalid answer slots')
    slots = tuple(tuple(_segment(group, vocab)) for group in groups)
    if len({t for group in slots for t in group}) != sum(map(len, slots)):
        raise ValueError('duplicate answer token IDs')
    if not isinstance(texts, list) or len(texts) != len(slots) or any(not _strings(words) or len(words) != len(group) for group, words in zip(slots, texts)):
        raise ValueError('missing native answer-token decoding')
    if not isinstance(entry['claim'], str) or not _strings(entry['options']) or not _strings(entry['excludes']) or not isinstance(entry.get('candidate', ''), str):
        raise ValueError('invalid rendered question')
    return AnswerGroups(slots, tuple(map(tuple, texts)), entry['claim'], tuple(entry['options']), tuple(entry['excludes']), entry.get('candidate'))


def _native(native):
    tokenizer = native.get('tokenizer') if isinstance(native, dict) else None
    if set(native or ()) != {'judge', 'execution', 'template', 'framing', 'tokenizer', 'presentation'} or not isinstance(tokenizer, dict):
        raise ValueError('invalid native descriptors')
    vocab = tokenizer.get('vocab_size')
    if type(vocab) is not int or vocab < 2 or not is_digest(tokenizer.get('vocab_sha256')) or not isinstance(tokenizer.get('probes'), list):
        raise ValueError('missing native tokenizer identity')
    if not isinstance(native['judge'], dict) or not isinstance(native['framing'], dict) or not isinstance(native['presentation'], dict) \
            or not isinstance(native['execution'], (dict, type(None))):
        raise ValueError('missing deployed judge/presentation/execution identity')
    return vocab


def _presentations(found, target, label, segments, answers):
    if not isinstance(found, list) or not found:
        raise ValueError('invalid native presentations')
    out = []
    for entry in found:
        if not isinstance(entry, list) or len(entry) != 8:
            raise ValueError('invalid presentation reference')
        identity, prefix, input_, suffix, groups, slots, stage, variant = entry
        groups = _ordinal(groups, answers, 'answer-group')
        if not is_digest(identity) or stage not in STAGES or variant not in VARIANTS:
            raise ValueError('invalid presentation identity/stage/variant')
        if not isinstance(slots, list) or len(slots) != len(groups.slots) or len(set(slots)) != len(slots) \
                or any(type(s) is not int or not -1 <= s < len(target['classes']) for s in slots):
            raise ValueError('invalid native slot map')
        if label not in slots and -1 not in slots:
            raise ValueError(f'presentation {identity[:12]} ({stage}) shows neither the gold answer nor an escape')
        tokens = tuple(_ordinal(value, segments, 'segment') for value in (prefix, input_, suffix))
        if not any(tokens):
            raise ValueError('empty training sequence')
        out.append(Presentation(identity, stage, variant, *tokens, groups, tuple(slots), slots.index(label if label in slots else -1)))
    return tuple(out)


def _observations(stream, manifest, tables, fitting):
    from .examples import history_of
    targets, (inputs, families, segments, answers) = manifest['targets'], tables
    native, rows, digests = manifest['native'], {}, []
    for row in _lines(stream):
        if not isinstance(row, dict) or set(row) - {'presentations'} != OBSERVATION or ('presentations' in row) != (native is not None):
            raise ValueError(f'malformed or obsolete observation (current export schema {EXPORT_SCHEMA}); re-export')
        if not is_digest(row['id']) or row['id'] in rows or not is_digest(row['digest']):
            raise ValueError('invalid or duplicate observation ID')
        target = _ordinal(row['target'], targets, 'target')
        if fitting and row['split'] != 'tune':
            raise ValueError('fitting requires tune-only export')
        if row['split'] not in ('tune', 'check') or type(row['taught']) is not bool or type(row['fold']) is not int or not 0 <= row['fold'] < 5:
            raise ValueError('invalid partition/fold/taught status')
        if type(row['label']) is not int or not 0 <= row['label'] < len(target['classes']):
            raise ValueError('invalid label')
        if not _strings(row['tags']) or not all(row['tags']) or not isinstance(row['pairs'], list) or not _strings(row['situations']):
            raise ValueError('invalid tags/pairs/situations')
        occurrences(row['selections'])
        history_of(row['label_sources'])
        view = {**target, **_ordinal(row['input'], inputs, 'input'), **_ordinal(row['family'], families, 'family'),
                **{key: row[key] for key in OBSERVATION if key not in ('target', 'input', 'family')}, 'value': target['values'][row['label']],
                'task_family': manifest['task_family']}
        del view['values']
        if native is not None:
            view['native'], view['presentations'] = native, _presentations(row['presentations'], target, row['label'], segments, answers)
        rows[row['id']] = view
        digests.append(row['digest'])
    return rows, digests


def _situations(stream, scope, rows):
    linked, out = {}, []
    for situation in _lines(stream):
        if not isinstance(situation, dict) or set(situation) != {'id', 'records', 'targets', 'complete'} or not isinstance(situation['targets'], list):
            raise ValueError('malformed situation index entry')
        if [tuple(target.get('address', ())) for target in situation['targets']] != [tuple(address) for address in scope]:
            raise ValueError(f"situation {situation['id']}: targets differ from the declared scope")
        for target in situation['targets']:
            if 'observation' in target:
                row = rows.get(target['observation'])
                if row is None or (row['fact'], *([row['member']] if 'member' in row else [])) != tuple(target['address']) or situation['id'] not in row['situations']:
                    raise ValueError(f"situation {situation['id']}: index link {target['observation']} does not match an exported observation")
                linked.setdefault(row['id'], set()).add(situation['id'])
            elif target.get('outcome') not in ('exact', 'missing', 'unlabelled'):
                raise ValueError(f"situation {situation['id']}: unknown target outcome")
        if situation['complete'] != all('observation' in t or t['outcome'] == 'exact' for t in situation['targets']):
            raise ValueError(f"situation {situation['id']}: gold-completeness does not match its targets")
        out.append(situation)
    if any(set(row['situations']) != linked.get(row['id'], set()) for row in rows.values()):
        raise ValueError('observation situation memberships differ from the source-situation index')
    return out


def _table(stream, decode, count):
    found = [decode(value) for value in _lines(stream)]
    if len(found) != count:
        raise ValueError('table length differs from the manifest count')
    return found


def read_export(path, *, fitting=True, native=False):
    path = Path(path)
    with files.open_published(path) as (manifest, payloads):
        streams = {name.rsplit('.', 2)[-2]: stream for name, stream in payloads.items()}
        if manifest is None or manifest.get('schema') != EXPORT_SCHEMA or set(streams) != set(EXPORT_PAYLOADS) or len(payloads) != len(EXPORT_PAYLOADS) \
                or not isinstance(manifest.get('targets'), list) or not isinstance(manifest.get('counts'), dict):
            raise ConfigError(f'{path}: missing or unsupported export manifest (current schema {EXPORT_SCHEMA}); re-export')
        if native and manifest.get('native') is None:
            raise ConfigError(f'{path}: LoRA requires a native export; re-export with --native')
        try:
            if fitting and manifest['split'] != 'tune':
                raise ValueError('fitting requires tune-only export')
            if manifest['task_family'] is not None and (not isinstance(manifest['task_family'], str) or not manifest['task_family'].strip()):
                raise ValueError('invalid task family')
            for target in manifest['targets']:
                _target(target)
            vocab = None if manifest['native'] is None else _native(manifest['native'])
            if vocab is None and (manifest['counts']['segments'] or manifest['counts']['answers']):
                raise ValueError('token tables belong to native exports')
            decoders = {'inputs': _input, 'families': _family, 'segments': lambda tokens: _segment(tokens, vocab), 'answers': lambda entry: _answers(entry, vocab)}
            tables = [_table(streams[kind], decoders[kind], manifest['counts'][kind]) for kind in EXPORT_TABLES]
            rows, digests = _observations(streams['observations'], manifest, tables, fitting)
            situations = _situations(streams['situations'], manifest['scope'], rows)
            if len(rows) != manifest['counts']['observations'] or len(situations) != manifest['counts']['situations']:
                raise ValueError('observation/situation counts differ from the manifest')
            if manifest.get('export_id') != export_identity(manifest, digests, situations):
                raise ValueError('export identity mismatch; re-export')
            return {**manifest, 'situations': situations}, list(rows.values())
        except (OSError, KeyError, TypeError, ValueError, UnicodeError, ConfigError) as e:
            raise ConfigError(f'{path}: invalid export: {e}') from None

def fact_records(plan, record) -> dict:
    return {f.name: {**record(f), **({'members': {c.id: record(plan.facts[c.claims[0]]) for c in f.cards}} if f.labels else {})}
            for f in plan.visible}


def _candidate_fact(value):
    return {'head': value['head'], 'examples': [{k: r[k] for k in ('input', 'value')} for r in value['examples']],
            'policies': {source: {side: record['theta'] for side, record in sides.items()} for source, sides in value['policies'].items()},
            **({'members': {key: _candidate_fact(member) for key, member in value['members'].items()}} if 'members' in value else {})}


def candidate_digest(data):
    from .api import behavior_identity
    return digest({'spec': data['spec'], 'models': {name: behavior_identity(identity) for name, identity in data['models'].items()},
                   'facts': {name: _candidate_fact(value) for name, value in data['facts'].items()}, 'bindings': data['bindings'],
                   'framing': data['framing'], 'returned_models': data['returned_models'], 'execution': data['execution']})


BINDING_PLACEMENT = ('representation', 'scope')


def binding_pins(plan, runtime, targets=None) -> dict:
    scopes = {}
    for target, name in (plan.targets if targets is None else targets).items():
        scopes.setdefault(name, []).append(target)
    records = {name: runtime.binding(name, scope).record() for name, scope in sorted(scopes.items())}
    return {'models': {name: runtime.identity(name) for name in records}, 'execution': {name: record['execution'] for name, record in records.items()},
            'bindings': {name: {key: record[key] for key in BINDING_PLACEMENT} for name, record in records.items()}}


def snapshot_digest(snap, runtime):
    pins = binding_pins(snap.plan, runtime)
    return candidate_digest({'returned_models': {name: list(snap.returned_models.get(name, ())) for name in pins['models']},
                             'framing': snap.framing.as_dict() if snap.framing else None, 'spec': snap.plan.spec, **pins,
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
