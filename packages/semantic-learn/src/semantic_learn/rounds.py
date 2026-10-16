from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from semantic_switch import ConfigError
from semantic_switch.training import ASSESSMENT, LabelSource, artifact_digest, current, decode_request, files, load_receipt, sources, specialize

from .common import BUDGET, Families, addresses, available, task_of

SCHEMA = 2
ENVELOPES = ('stale', 'after_cutoff')
SITUATIONS = ('deleted', 'unauthorized', 'capped')
SERVING = ('separate_host', 'interrupt', 'fallback')
LABELS, RECORD, PIPELINE = 'round.labels.jsonl', 'round.json', 'pipeline'
SCOPES = {'situations': 'situation', 'families': 'projection'}


def tombstone(store, definition, data, *, reason, scope='situation') -> dict:
    if not isinstance(reason, str) or not reason.strip():
        raise ConfigError('a deletion tombstone records its reason')
    task = task_of(definition)
    record = {'schema': SCHEMA, 'kind': 'tombstone', 'scope': scope, 'contract': task.contract(scope), 'key': task.identity(scope)(data),
              'projection': task.identity('projection')(data), 'reason': reason, 'at': files.now()}
    files.append(Path(store), files.jsonl([record]))
    return record


def tombstones(store) -> list:
    if store is None:
        return []
    found = files.read_jsonl(Path(store))
    fields = {'schema', 'kind', 'scope', 'contract', 'key', 'projection', 'reason', 'at'}
    if any(not isinstance(record, dict) or set(record) != fields or record['schema'] != SCHEMA or record['kind'] != 'tombstone'
           or record['scope'] not in SCOPES.values() or not all(isinstance(record[name], str) for name in ('contract', 'key', 'projection')) for record in found):
        raise ConfigError(f'{store}: not a schema-{SCHEMA} scoped tombstone store; record deletions again with semantic-learn tombstone')
    return found


def deletions(task, store) -> dict:
    out = {scope: set() for scope in SCOPES.values()}
    for record in tombstones(store):
        if record['contract'] != task.contract(record['scope']):
            raise ConfigError(f"{store}: a {record['scope']} tombstone was recorded under another task contract; its identity cannot be matched to this task: "
                              're-record the deletion against the current definition')
        out[record['scope']].add(record['key'])
    return out


def affected(store, receipts, pointer=None) -> dict:
    keys, active = {record['projection'] for record in tombstones(store)}, current(pointer) if pointer is not None else None
    out = []
    for path in receipts:
        deployment = load_receipt(path)
        rows = [row for row in deployment.provenance.rows if any(node[0] == 'key' and node[1] in keys for node in row['family_seeds'])]
        if rows:
            out.append({'receipt': deployment.id, 'path': str(deployment.path), 'lock': deployment.data['lock']['id'], 'active': deployment.path == active,
                        'observations': sorted(row['id'] for row in rows), 'uses': sorted({use for row in rows for use in row['uses']})})
    return {'tombstones': len(keys), 'affected': out, 'matched': 'read projections of tombstoned inputs; a situation tombstone conservatively flags every '
                                                                'deployment that used its projection',
            'remedy': 'deleting source files does not unlearn adapter weights or embedded corrections: run a round without the deleted examples and promote '
                      'its verified lock to remove their influence from active serving'}


def _stale(meanings, record) -> bool:
    given = addresses(record.get('revs', {}))
    return any(meanings.get(address) != rev for address, rev in given.items()) or any(address not in meanings for address in addresses(record['facts']))


def _histories(task, envelope) -> dict | None:
    try:
        task.admit([envelope])
    except ConfigError:
        return None
    return {address: tuple(map(LabelSource.parse, history)) for address, history in addresses(envelope['label_sources']).items()}


def origins(histories) -> set:
    return {f'{source.derivation}:{source.producer}' for history in histories.values() for source in history if not source.boundary}


def eligible(task, sources, *, cutoff, deleted, caps, budget=BUDGET) -> tuple[list, dict]:
    situation, projection = task.identity(SCOPES['situations']), task.identity('projection')
    meanings = {address: target['meaning_rev'] for address, target in task.targets().items()}
    envelopes, situations, removed = Counter(), Counter(), set()
    merged = task.accumulator(**budget)
    families = Families(task, **budget, retained=lambda: merged.retained[2])
    for path in sources:
        for line, record in files.iter_records(Path(path), with_lines=True):
            where = f'{path}:{line}'
            if not isinstance(record, dict) or not isinstance(record.get('facts'), dict) or not isinstance(record.get('label_sources'), dict) or 'input' not in record:
                raise ConfigError(f'{where}: not a labelled situation envelope')
            reason = 'stale' if _stale(meanings, record) else 'after_cutoff' if available(record.get('created'), where) > cutoff else None
            gone = reason is None and (situation(record['input']) in deleted['situation'] or projection(record['input']) in deleted['projection'])
            try:
                families.add(record, removed=reason is not None or gone)
                if reason:
                    envelopes[reason] += 1
                elif gone:
                    removed.add(situation(record['input']))
                else:
                    merged.add(task.unit(record))
            except ConfigError as error:
                raise ConfigError(f'{where}: {error}') from None
    situations['deleted'] = len(removed)
    seen, kept = Counter(), []
    for key in sorted(merged.keys(), key=artifact_digest):
        try:
            envelope = merged.settle(key, strict=True)[0]
        except ConfigError as error:
            raise ConfigError(f'round preparation: {error}; nothing was started') from None
        histories = _histories(task, envelope)
        found = origins(histories) if histories is not None else set()
        reason = 'unauthorized' if histories is None else 'capped' if any(seen[origin] >= caps[origin] for origin in found if origin in caps) else None
        if reason:
            situations[reason] += 1
            families.remove(envelope['input'])
            continue
        seen.update(found)
        kept.append(envelope)
    rows = [{**envelope, 'groups': sorted({*envelope['groups'], families.bridge(envelope['input'], 'round')}, key=json.dumps)}
            if families.touched(envelope['input']) else envelope for envelope in kept]
    return rows, {'envelopes': {reason: envelopes[reason] for reason in ENVELOPES}, 'situations': {reason: situations[reason] for reason in SITUATIONS}}


def coverage(task, records, *, min_reviewed, min_per_class) -> dict:
    classes, reviewed = Counter(), Counter()
    for record in records:
        assessed = task.assess(record)
        for address, value in addresses(record['facts']).items():
            classes[(address, json.dumps(value))] += 1
            reviewed[address] += assessed[address].established
    targets = {}
    for address, target in task.targets().items():
        counts = {key: classes[(address, json.dumps(value))] for key, value in target['answers'].items()}
        short = sorted(key for key, count in counts.items() if count < min_per_class)
        targets[address] = {'classes': counts, 'reviewed': reviewed[address], 'missing_classes': short,
                            'feasible': not short and reviewed[address] >= min_reviewed}
    return {'min_reviewed': min_reviewed, 'min_per_class': min_per_class, 'targets': targets, 'feasible': all(entry['feasible'] for entry in targets.values()),
            'assessment': {'revision': ASSESSMENT, 'review': None, 'criteria': {address: [target['meaning_rev']] for address, target in task.targets().items()},
                           'rule': 'reviewed counts targets whose label is computed or carries a current confirmation bound to its input, answer and '
                                   'this task meaning; old-criterion and stale confirmations count as unreviewed'}}


def recipe(path) -> dict:
    record = files.read_json(Path(path))
    if record is None:
        raise ConfigError(f'{path}: no specialization request; write one with ss specialize or by hand under the specialization request schema')
    return {key: value for key, value in decode_request(record, str(path)).items() if key not in ('sources', 'work')}


def specialization_request(definition, labels, work, settings) -> dict:
    return decode_request({**settings, 'sources': sources(definition, labels), 'work': str(Path(work).resolve())}, 'round')


def freeze_round(definition, labels, work, *, cutoff, serving, min_reviewed, min_per_class, settings, incumbent=None, deleted=None, caps=None) -> dict:
    work = Path(work).resolve()
    if serving not in SERVING:
        raise ConfigError(f"declare the serving policy during fitting: one of {', '.join(SERVING)}; a background thread never bypasses the GPU envelope")
    if any(type(value) is not int or value < 1 for value in (min_reviewed, min_per_class)):
        raise ConfigError('declare positive min_reviewed and min_per_class feasibility requirements; there is no production default')
    cutoff = available(cutoff, 'round cutoff')
    task = task_of(definition)
    caps = dict(caps or {})
    records, excluded = eligible(task, labels, cutoff=cutoff, deleted=deletions(task, deleted), caps=caps)
    covered = coverage(task, records, min_reviewed=min_reviewed, min_per_class=min_per_class)
    if not covered['feasible']:
        raise ConfigError(f"round is infeasible under the declared support: {json.dumps({a: e for a, e in covered['targets'].items() if not e['feasible']})}; "
                          'review or acquire more examples; nothing was started')
    body = ''.join(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n' for record in records)
    record = {'schema': SCHEMA, 'kind': 'learning-round', 'definition': {'path': str(Path(definition).resolve()), 'sha256': files.sha256_file(definition)},
              'meanings': {address: target['meaning_rev'] for address, target in task.targets().items()}, 'cutoff': cutoff.isoformat(),
              'sources': {str(Path(path).resolve()): files.sha256_file(path) for path in labels}, 'identity': SCOPES,
              'tombstones': files.sha256_file(deleted) if deleted is not None else None, 'caps': caps, 'serving': serving,
              'incumbent': load_receipt(incumbent).id if incumbent is not None else None, 'initialization': 'pinned base; no adapter is inherited',
              'eligible': {'file': LABELS, 'sha256': artifact_digest(body), 'rows': len(records)}, 'excluded': excluded, 'coverage': covered,
              'specialization': settings}
    record['id'] = artifact_digest(record)
    existing = files.read_json(work / RECORD)
    if existing is None:
        files.publish(work / LABELS, body)
        files.write_new(work / RECORD, record)
    elif existing != record:
        raise ConfigError(f'{work} holds round {existing.get("id")} with other inputs; resume it unchanged or start a new round directory')
    elif (work / LABELS).read_text() != body:
        raise ConfigError(f'{work / LABELS}: the frozen round labels changed; start a new round directory')
    return record


def run_round(definition, labels, work, *, trainer, request, cutoff, serving, min_reviewed, min_per_class, incumbent=None, deleted=None, caps=None,
              until=None, attempt=None, cadence=None) -> dict:
    work, settings = Path(work).resolve(), recipe(request)
    record = freeze_round(definition, labels, work, cutoff=cutoff, serving=serving, min_reviewed=min_reviewed, min_per_class=min_per_class, settings=settings,
                          incumbent=incumbent, deleted=deleted, caps=caps)
    result = specialize(trainer, work / PIPELINE, specialization_request(definition, work / LABELS, work / PIPELINE, settings), until=until, attempt=attempt,
                        cadence=cadence)
    return {'round': record['id'], 'record': str(work / RECORD), 'state': result['state'], 'lock': result.get('lock'), 'receipt': result.get('receipt'),
            'excluded': record['excluded'], 'eligible': record['eligible']['rows'], 'pipeline': result,
            'publication': 'publishing a candidate never changes the active lock; compare it and promote explicitly'}
