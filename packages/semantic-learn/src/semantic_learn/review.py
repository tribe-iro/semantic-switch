from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import random

from semantic_switch import ConfigError
from semantic_switch.training import artifact_digest, files

from .capture import captured
from .common import BUDGET, Families, addresses, predictions, task_of

SCHEMA = 2
ALLOCATOR = 'audit-then-seeded-round-robin@1'
STRATA = ('teacher_uncertain', 'disagreement', 'inconsistency', 'feedback', 'conflict')
FAILURES = {'input_view': 'materialize', 'coverage': 'produce', 'label': 'review', 'relation': 'task', 'optimization': 'diagnose', 'acceptance': 'calibrate'}
ESTIMATES = {'audit': 'uniform random sample without replacement of the eligible population; reviewed accuracy on it estimates that population',
             'queue': 'triage strata (teacher uncertainty, student/teacher disagreement, repeated-presentation inconsistency, weak agent feedback, '
                      'unresolved label conflicts) form a '
                      'biased review queue; their reviewed accuracy is not a population estimate and confidence or consistency is not label correctness'}


def collection(manifest) -> dict:
    manifest = Path(manifest)
    published = files.read_published(manifest)
    if published is None or published.get('kind') != 'collection':
        raise ConfigError(f'{manifest}: not a published collection manifest from semantic-learn collect')
    payloads = {name: manifest.parent / entry['file'] for name, entry in published['payloads'].items()}
    for name, path in payloads.items():
        if files.sha256_file(path) != published['payloads'][name]['sha256']:
            raise ConfigError(f'{path}: collection payload changed since publication')
    return payloads


class Population:
    def __init__(self, task, budget):
        self.task, self.units, self.events, self.stale, self.scope = task, {}, {}, 0, 'situation'
        self.claims = task.accumulator(**budget)
        self.meanings = {address: target['meaning_rev'] for address, target in task.targets().items()}

    def unit(self, data) -> dict:
        key = self.task.key(data)
        return self.units.setdefault(key, {'key': key, 'envelope': None, 'unresolved': {}, 'outcomes': {}, 'student': [], 'feedback': []})

    def merge(self, found, envelope):
        self.claims.add(self.task.unit(envelope))
        found['held'] = True

    def settle(self):
        for unit in self.units.values():
            if unit.get('held'):
                unit['envelope'], unit['unresolved'] = self.claims.settle(unit['key'], strict=False)

    def capture(self, record):
        if record['kind'] == 'feedback':
            self.events.setdefault(record['event'], {}).setdefault('feedback', []).append(record)
            return
        situation = record['situation']
        if situation is None:
            return
        current = {address: target['meaning_rev'] for address, target in record['task']['targets'].items()}
        if current != self.meanings:
            self.stale += 1
            return
        unit = self.unit(situation['input'])
        self.merge(unit, situation)
        unit['student'].append(predictions(record['decision']['facts']))
        self.events.setdefault(record['event'], {})['unit'] = (unit, len(unit['student']) - 1)

    def label(self, record):
        if any(self.meanings.get(address) != rev for address, rev in addresses(record.get('revs', {})).items()):
            self.stale += 1
            return
        self.merge(self.unit(record['input']), record)

    def unresolved(self, record):
        self.unit(record['input'])['outcomes']['/'.join(record['address'])] = record['outcome']

    def link(self):
        for found in self.events.values():
            if 'unit' in found:
                unit, index = found['unit']
                unit['feedback'] += [(record['address'], record['value'], unit['student'][index].get(record['address'])) for record in found.get('feedback', ())]

    def strata(self, unit) -> set:
        latest = unit['student'][-1] if unit['student'] else {}
        disagree = any(address in latest and latest[address] != value for address, value in teacher(unit).items())
        inconsistent = any(len({json.dumps(seen.get(address), sort_keys=True) for seen in unit['student'] if address in seen}) > 1
                           for address in {address for seen in unit['student'] for address in seen})
        return {name for name, hit in (('teacher_uncertain', 'uncertain' in unit['outcomes'].values()), ('disagreement', disagree),
                                       ('inconsistency', inconsistent), ('feedback', any(value != predicted for _, value, predicted in unit['feedback'])),
                                       ('conflict', bool(unit['unresolved'])))
                if hit}


def teacher(unit) -> dict:
    return addresses(unit['envelope'].get('facts', {})) if unit['envelope'] is not None else {}


def support(units, index) -> dict:
    classes = Counter(f'{address}={json.dumps(value)}' for unit in units for address, value in teacher(unit).items())
    return {'rows': len(units), 'families': len({index.component(unit['envelope']['input']) for unit in units}), 'classes': dict(sorted(classes.items())),
            'teacher_labelled': sum(bool(teacher(unit)) for unit in units), 'conflicted': sum(bool(unit['unresolved']) for unit in units)}


def review_export(definition, to, *, size, audit, captures=(), collected=(), strata=(), seed=0, budget=BUDGET) -> dict:
    to = Path(to)
    if not to.name.endswith('.review.json'):
        raise ConfigError(f'{to}: review exports are named NAME.review.json')
    if to.exists():
        raise ConfigError(f'{to}: a review export is already published there; export into a new name')
    if any(type(value) is not int or value < 0 for value in (size, audit, seed)) or audit > size or size < 1:
        raise ConfigError('size is a positive review budget, audit a random stratum within it and seed a nonnegative integer')
    unknown = set(strata) - set(STRATA)
    if unknown or len(set(strata)) != len(strata):
        raise ConfigError(f"strata are distinct names among {', '.join(STRATA)}, not {sorted(unknown) or list(strata)}")
    task = task_of(definition)
    population = Population(task, budget)
    for path in captures:
        for record in captured(path):
            population.capture(record)
    for manifest in collected:
        payloads = collection(manifest)
        for record in files.iter_records(payloads['labels.jsonl']):
            population.label(record)
        for record in files.iter_records(payloads['unresolved.jsonl']):
            population.unresolved(record)
    population.settle()
    population.link()
    eligible = [unit for unit in population.units.values() if unit['envelope'] is not None]
    if not eligible:
        raise ConfigError('no current situations to review; capture or collect for the current task meanings first')
    index = Families(task, **budget)
    for unit in eligible:
        index.add(unit['envelope'])
    declared = set(strata)
    for unit in eligible:
        unit['reasons'] = sorted(population.strata(unit) & declared)
        unit['digest'] = artifact_digest({'envelope': unit['envelope'], 'unresolved': {address: [claim.record() for claim in found.alternatives]
                                                                                       for address, found in unit['unresolved'].items()}, 'family': index.component(unit['envelope']['input']),
                                          'reasons': unit['reasons']})
    units = sorted(eligible, key=lambda unit: unit['digest'])
    run = artifact_digest({'definition': files.sha256_file(definition), 'size': size, 'audit': audit, 'strata': list(strata), 'seed': seed,
                           'allocator': ALLOCATOR, 'population': [unit['digest'] for unit in units]})
    drawn = random.Random(seed).sample(units, min(audit, len(units)))
    chosen = {unit['key']: ('audit', {'selected_by': 'random', 'inclusion_probability': len(drawn) / len(units), 'selection_id': f'review:audit:{run}'})
              for unit in drawn}
    queues = {name: deque() for name in strata}
    for unit in units:
        for name in unit['reasons']:
            queues[name].append(unit)
    for name, queue in queues.items():
        random.Random(artifact_digest([seed, ALLOCATOR, name])).shuffle(queue)
    queued, allocation = {name: [] for name in strata}, []
    active = deque(name for name in strata if queues[name])
    while active and len(chosen) < size:
        name = active.popleft()
        queue = queues[name]
        while queue and queue[0]['key'] in chosen:
            queue.popleft()
        if not queue:
            continue
        unit = queue.popleft()
        chosen[unit['key']] = (name, {'selected_by': 'targeted', 'selection_id': f'review:{name}:{run}'})
        queued[name].append(unit)
        allocation.append({'unit': unit['digest'], 'stratum': name, 'reasons': unit['reasons']})
        if queue:
            active.append(name)
    rows, claims, by_key = [], [], {unit['key']: unit for unit in units}
    for key, (stratum, selection) in sorted(chosen.items(), key=lambda item: by_key[item[0]]['digest']):
        envelope = by_key[key]['envelope']
        claims += [{'schema': SCHEMA, 'input': envelope['input'], 'address': address, 'alternatives': [claim.record() for claim in found.alternatives]}
                   for address, found in sorted(by_key[key]['unresolved'].items())]
        bridged = [index.bridge(envelope['input'], 'review')] if index.shared(envelope['input']) else []
        common = dict(groups=[*envelope['groups'], *bridged], pairs=envelope['pairs'], role=envelope.get('role'), tags=envelope.get('tags', ()),
                      selections=[*envelope['selections'], selection])
        rows.append(task.label(envelope['input'], envelope['facts'], sources=envelope['label_sources'], **common) if 'facts' in envelope
                    else task.situation(envelope['input'], **common))
    reviewed = to.with_name(to.name.removesuffix('.review.json') + '.reviewed.jsonl')
    manifest = {'schema': SCHEMA, 'kind': 'review-export', 'id': run, 'definition': str(Path(definition).resolve()), 'seed': seed, 'budget': size,
                'identity': {'situations': population.scope, 'families': index.scope},
                'population': support(units, index), 'stale': population.stale,
                'conflicts': sorted({address for unit in units for address in unit['unresolved']}),
                'audit': {**support(drawn, index), 'inclusion_probability': len(drawn) / len(units) if drawn else None},
                'strata': {name: support(found, index) for name, found in queued.items()}, 'allocator': ALLOCATOR, 'allocation': allocation,
                'units': [{'id': by_key[key]['digest'], 'stratum': stratum, 'reasons': by_key[key]['reasons']}
                          for key, (stratum, _) in sorted(chosen.items(), key=lambda item: by_key[item[0]]['digest'])],
                'estimates': ESTIMATES,
                'families': 'closed over the complete eligible population before sampling; rows whose family reaches other situations carry its review-bridge group',
                'acquisition': 'random audit only' if not strata else f"random audit plus declared triage strata {list(strata)}; active selection is not the "
                                                                       'default until it beats random acquisition on matched budgets',
                'weak_feedback': 'agent feedback only orders the queue; it never becomes a label',
                'unresolved': 'queued envelopes omit targets whose retained claims no recorded correction lineage orders; claims.jsonl lists every '
                              'alternative with its complete history for an explicit authored resolution (Task.resolve); it is review evidence, never '
                              'model input or inferred gold'}
    published = files.publish(to, manifest, {'queue.jsonl': [json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows],
                                             'claims.jsonl': [json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in claims]})
    queue, alternatives = (str(to.parent / published['payloads'][name]['file']) for name in ('queue.jsonl', 'claims.jsonl'))
    return {**manifest, 'queue': queue, 'claims': alternatives, 'review': ['ss', 'label', str(definition), queue, '--to', str(reviewed)]}


def diagnose(review, unit, *, reason, reviewer, evidence, note, to):
    if reason not in FAILURES or not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(note, str) or not note.strip():
        raise ConfigError(f'diagnosis requires a reviewer, note and reason among {list(FAILURES)}')
    review, evidence, to = Path(review).resolve(), Path(evidence).resolve(), Path(to)
    manifest = files.read_published(review)
    if manifest is None or manifest.get('kind') != 'review-export' or unit not in {row['id'] for row in manifest.get('units', ())}:
        raise ConfigError('diagnosis must name a unit of the published review queue')
    if not evidence.is_file():
        raise ConfigError('diagnosis evidence must be an existing review or analysis artifact')
    record = {'schema': 1, 'kind': 'failure-diagnosis', 'review': manifest['id'], 'unit': unit, 'reason': reason, 'next': FAILURES[reason],
              'reviewer': reviewer, 'evidence': {'path': str(evidence), 'sha256': files.sha256_file(evidence)}, 'note': note}
    record['id'] = artifact_digest(record)
    with files.locked(to):
        existing = {entry['id'] for entry in files.iter_records(to)} if to.exists() else set()
        if record['id'] not in existing:
            files.append(to, files.jsonl([record]))
    return record


def situations(definition, captures, to, budget=BUDGET) -> dict:
    population = Population(task_of(definition), budget)
    for path in captures:
        for record in captured(path):
            population.capture(record)
    population.settle()
    rows = [unit['envelope'] for unit in sorted(population.units.values(), key=lambda unit: artifact_digest(unit['key']))]
    if not rows:
        raise ConfigError('no captured situations hold the current task meanings')
    files.write_new(Path(to), ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
    return {'situations': str(to), 'rows': len(rows), 'stale': population.stale,
            'collect': 'the teacher receives exactly these redacted inputs: semantic-learn collect DEFINITION SITUATIONS --teacher ... --training-use ...'}
