from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from collections import Counter
import heapq
import math
import json
from pathlib import Path

from .. import files
from ..compiler import digest
from ..errors import ConfigError
from ..evidence import DATASET_SCHEMA, ROLES, path_record, role_of
from ..families import FamilyIndex, seed, valid_seeds
from ..results import thaw
from .common import input_key
from .data import merged_sources, observation_id

AUDIT_SCHEMA = 2
DISCLOSURE = 'tool-recorded exposure; external inspections require caller disclosure'
REGENERATE = f'delete it and re-run ss lock (or bench.evidence stage) to rebuild a schema-{DATASET_SCHEMA} evidence assignment; keep its audit ledger'


ALLOCATION = 'grouped-iterative-v2'


def allocate(families, targets, *, pinned=None, seed=0):
    if type(seed) is not int or seed < 0:
        raise ConfigError('family allocation seed must be a nonnegative integer')
    targets = dict(sorted(targets.items()))
    if not targets or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in targets.values()) or sum(targets.values()) <= 0:
        raise ConfigError('family allocation needs finite nonnegative target sizes with a positive total')
    assigned = dict(pinned or {})
    if set(assigned) - set(families) or set(assigned.values()) - set(targets):
        raise ConfigError('pinned family assignments must name known families and target roles')
    sizes, cells = {}, {}
    for family, record in sorted(families.items()):
        size = record['size']
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size < 0:
            raise ConfigError(f'{family}: family size must be finite and nonnegative')
        sizes[family], cells[family] = size, frozenset(record.get('cells', ()))
        if any(not isinstance(cell, str) or not cell for cell in cells[family]):
            raise ConfigError(f'{family}: cells must be nonempty canonical strings')
    support = Counter(cell for found in cells.values() for cell in found)
    counts = {role: Counter() for role in targets}
    actual = Counter()
    for family, role in assigned.items():
        actual[role] += sizes[family]
        counts[role].update(cells[family])
    total = sum(targets.values())
    desired = {role: {cell: number * targets[role] / total for cell, number in support.items()} for role in targets}
    priority = {family: digest([ALLOCATION, seed, family]) for family in families}
    members, remaining = {}, Counter()
    for family in sizes.keys() - assigned.keys():
        for cell in cells[family]:
            members.setdefault(cell, []).append((priority[family], family))
            remaining[cell] += 1
    for found in members.values():
        heapq.heapify(found)
    queue = [(number, cell) for cell, number in remaining.items()]
    heapq.heapify(queue)
    free = [(priority[family], family) for family in sizes.keys() - assigned.keys()]
    heapq.heapify(free)
    while len(assigned) < len(families):
        while queue and (queue[0][0] != remaining[queue[0][1]] or not queue[0][0]):
            heapq.heappop(queue)
        cell = queue[0][1] if queue else None
        available = members[cell] if cell is not None else free
        while available and available[0][1] in assigned:
            heapq.heappop(available)
        _, family = heapq.heappop(available)
        candidates = [role for role in targets if targets[role] > actual[role]] or [role for role in targets if targets[role] > 0]
        def deficit(role):
            missing = sum(max(0., desired[role][entry] - counts[role][entry]) for entry in cells[family])
            empty_fit = role == 'fit' and any(counts[role][entry] == 0 for entry in cells[family])
            return (actual[role] == 0, missing, empty_fit, desired[role].get(cell, 0.) - counts[role][cell] if cell is not None else 0.,
                    (targets[role] - actual[role]) / max(targets[role], 1.))
        role = max(candidates, key=deficit)
        assigned[family] = role
        actual[role] += sizes[family]
        counts[role].update(cells[family])
        for entry in cells[family]:
            remaining[entry] -= 1
            heapq.heappush(queue, (remaining[entry], entry))
    report = {'algorithm': ALLOCATION, 'seed': seed, 'targets': targets, 'achieved': {role: actual[role] for role in targets},
              'families': dict(sorted(assigned.items())),
              'support': {cell: {role: counts[role][cell] for role in targets} for cell in sorted(support)},
              'deficits': {role: [cell for cell in sorted(support) if targets[role] > 0 and not counts[role][cell]] for role in targets}}
    return {**report, 'id': digest(report)}


def source_digests(sw, data=None):
    paths = {'definition': sw.paths.yaml, 'examples': sw.paths.examples, 'labels': Path(data) if data else sw.paths.test}
    return {role: files.sha256_file(p) if p.exists() else None for role, p in paths.items()}


def validate(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != DATASET_SCHEMA or not isinstance(manifest.get('families'), dict) \
            or not isinstance(manifest.get('namespace'), str) or not isinstance(manifest.get('audit'), str) or not isinstance(manifest.get('sources'), dict):
        raise ConfigError(f'dataset: unsupported or malformed assignment manifest (current schema {DATASET_SCHEMA}); {REGENERATE}')
    if manifest.get('id') != digest({k: v for k, v in manifest.items() if k != 'id'}):
        raise ConfigError('dataset: manifest digest mismatch')
    for family, record in manifest['families'].items():
        if not isinstance(record, dict) or record.get('split') not in ('tune', 'check', 'conflict') or type(record.get('fold')) is not int or not 0 <= record['fold'] < 5:
            raise ConfigError(f'dataset: invalid assignment for {family}')
        if not valid_seeds(record.get('seeds')) or not record['seeds'] or not isinstance(record.get('aliases'), list) or family not in record['aliases'] \
                or not isinstance(record.get('conflicts', {}), dict):
            raise ConfigError(f'dataset: invalid seeds or aliases for {family}')
    return manifest


def restore(index, manifest):
    if not manifest:
        return index
    for family, record in validate(manifest)['families'].items():
        first = index.add(record['seeds'][0], *record['aliases'])
        for node in record['seeds'][1:]:
            index.join(first, node)
    return index


def _role(record):
    return 'conflict' if record['split'] == 'conflict' else role_of(record['split'], record['fold'])


def materialize(labels, previous, sources):
    index = labels.families
    prior = {}
    for family, record in (previous or {}).get('families', {}).items():
        prior[index.find(record['seeds'][0])] = {**prior.get(index.find(record['seeds'][0]), {}), family: record}
    roles = {}
    for node, role in labels._roles:
        roles.setdefault(index.find(node), set()).add(role)
    population = {index.name(root): {'size': 0, 'cells': set()} for root in index.roots()}
    pinned = {}
    for root in index.roots():
        family = index.name(root)
        inherited = prior.get(root, {})
        historical = {_role(record) for record in inherited.values()}
        explicit = roles.get(root, set())
        if inherited:
            pinned[family] = next(iter(historical)) if len(historical) == 1 else 'conflict'
        elif len(explicit) == 1:
            pinned[family] = next(iter(explicit))
    for (name, projection), record in labels._merged.items():
        family = index.name(record['nodes'][0])
        fact = labels.sw.plan.fact(name)
        cell = json.dumps([list(fact.path), fact.keys[record['label']]], ensure_ascii=False, separators=(',', ':'))
        population[family]['size'] += 1
        population[family]['cells'].add(cell)
        population[family]['cells'].update(json.dumps([list(fact.path), fact.keys[record['label']], tag], ensure_ascii=False, separators=(',', ':')) for tag in record['tags'])
    total = sum(row['size'] for row in population.values()) or len(population) or 1
    allocation = allocate(population, {'fit': total * .3, 'model_select': total * .1, 'acceptance_fit': total * .1, 'check': total * .5, 'conflict': 0}, pinned=pinned)
    families = {}
    for root in index.roots():
        inherited = prior.get(root, {})
        family = index.name(root)
        conflicts = {alias: value for record in inherited.values() for alias, value in record.get('conflicts', {}).items()}
        historical = {name: _role(record) for name, record in inherited.items()}
        if len(set(historical.values())) < 2 and not conflicts and len(roles.get(root, ())) > 1:
            raise ConfigError(f"dataset: family {family} has label records with roles {', '.join(sorted(roles[root]))}; "
                              f'give every record joined by its groups, pairs, inputs, projections or situations one role')
        role = next(iter(roles.get(root, ())), None)
        if inherited:
            if len(set(historical.values())) > 1 or conflicts:
                split, fold = 'conflict', min(r['fold'] for r in inherited.values())
                conflicts = {**{name: value for name, value in historical.items() if value != 'conflict'}, **conflicts}
            else:
                split, fold = next(iter(inherited.values()))['split'], min(r['fold'] for r in inherited.values())
            if role is not None and split != 'conflict' and role_of(split, fold) != role:
                raise ConfigError(f'dataset: family {family} was materialized as {role_of(split, fold)} in {labels.sw.paths.dataset} but its label records '
                                  f'now assign role {role}; connected evidence keeps its first role, so relabel the records or start a new evidence assignment')
        elif role is not None:
            split, folds = ROLES[role]
            fold = folds[int(digest(['fold', family])[:8], 16) % len(folds)]
        else:
            split, folds = ROLES[allocation['families'][family]]
            fold = folds[int(digest(['fold', family])[:8], 16) % len(folds)]
        index.add(root, family)
        families[family] = {'split': split, 'fold': fold, 'seeds': [list(n) for n in index.seeds(root)], 'aliases': list(index.history(root)),
                            **({'conflicts': dict(sorted(conflicts.items()))} if conflicts else {})}
        if split == 'conflict':
            labels.warnings += (f'{family}: connected historical roles conflict ({", ".join(f"{k}: {v}" for k, v in sorted(conflicts.items()))}); '
                                f'excluded from fitting and independent checks',)
    rows = sorted([{'id': observation_id(labels.sw.plan.fact(name), projection),
                    **path_record(labels.sw.plan.fact(name).path), 'meaning_rev': labels.sw.plan.fact(name).meaning_rev,
                    'family': index.name(entry['nodes'][0]),
                    'label': entry['label'], 'taught': entry['taught'],
                    'selections': [s.record() for s in entry['selections']], 'label_sources': [source.digest for source in merged_sources(entry)],
                    'situations': sorted(entry['situations'])}
                   for (name, projection), entry in labels._merged.items()], key=lambda r: r['id'])
    out = {'schema': DATASET_SCHEMA, 'namespace': labels.sw.plan.name, 'audit': labels.sw.paths.audit.name, 'sources': sources,
           'families': families, 'observations': rows,
           **({'allocation': previous['allocation']} if previous and set(previous['families']) == set(families) and 'allocation' in previous else
              {'allocation': allocation} if any(not prior.get(root) for root in index.roots()) else {}),
           **({'review': previous['review']} if previous and 'review' in previous and previous['sources'] == sources else {})}
    out['training_id'] = digest([r for r in rows if families[r['family']]['split'] == 'tune'])
    out['id'] = digest(out)
    labels.manifest = out
    labels.dataset_id = out['id']
    labels.training_id = out['training_id']
    return out


def evaluation_plan(arms, *, scope, conditions=(), acceptance=None, metrics=None, margin=None) -> dict:
    if not isinstance(arms, Mapping) or not arms or any(not isinstance(role, str) or not role or value is None for role, value in arms.items()):
        raise ConfigError('an evaluation plan declares every comparison arm as {role: identity} before reading check labels')
    plan = {'schema': AUDIT_SCHEMA, 'arms': thaw(dict(sorted(arms.items()))), 'conditions': sorted(conditions), 'scope': thaw(list(scope)),
            'acceptance': thaw(acceptance), 'metrics': thaw(metrics), 'margin': margin}
    return {**plan, 'id': digest(plan)}


def _consultation(record):
    return isinstance(record, dict) and record.get('schema') == AUDIT_SCHEMA and isinstance(record.get('plan'), dict) \
        and isinstance(record['plan'].get('id'), str) and all(isinstance(record.get(k), list) for k in ('families', 'aliases', 'seeds'))


@dataclass(frozen=True)
class EvidenceContext:
    path: Path
    manifest: Mapping
    index: FamilyIndex

    @classmethod
    def from_manifest(cls, path) -> EvidenceContext:
        path = Path(path)
        manifest = files.read_published(path)
        if manifest is None:
            raise ConfigError(f'{path}: no evidence assignment manifest; materialize it with ss lock or bench.evidence stage')
        return cls(path, manifest, restore(FamilyIndex(), manifest))

    @classmethod
    def of(cls, labels) -> EvidenceContext:
        return cls(labels.committed(), labels.manifest, labels.families)

    @property
    def dataset_id(self) -> str:
        return self.manifest['id']

    @property
    def audit(self) -> Path:
        return self.path.parent / self.manifest['audit']

    @property
    def review(self) -> dict:
        return thaw(self.manifest.get('review')) or {'status': 'undisclosed', 'self_review': None, 'reviewer_distinct': None}

    def bind(self, labels) -> Path:
        labels = Path(labels)
        if not labels.is_file() or files.sha256_file(labels) != self.manifest['sources'].get('labels'):
            raise ConfigError(f'{labels}: label file is not the one bound by evidence {self.path} (sha256 {self.manifest["sources"].get("labels")}); '
                              're-materialize the evidence assignment for these labels')
        return labels

    def family(self, node) -> str | None:
        node = seed(node)
        return self.index.name(node) if node in self.index else None

    def member(self, plan, data) -> str:
        family = self.family(('key', input_key(plan, data)))
        if family not in self.manifest['families']:
            raise ConfigError(f'input {input_key(plan, data)[:12]} has no family in evidence {self.path}; re-materialize the assignment for these labels')
        return family

    def role(self, family) -> str:
        return _role(self.manifest['families'][family])

    @cached_property
    def aliases(self) -> dict:
        return {alias: family for family, record in self.manifest['families'].items() for alias in record['aliases']}

    def resolve(self, name) -> str:
        return self.aliases.get(name) or (self.index.name(('group', name)) if ('group', name) in self.index else name)

    def consult(self, plan, families) -> dict:
        if not isinstance(plan, Mapping) or plan.get('id') != digest({k: v for k, v in plan.items() if k != 'id'}):
            raise ConfigError('consult a frozen evaluation_plan(...)')
        intended = sorted(map(self.resolve, set(families)))
        if not intended:
            return {'state': 'empty', 'plan': plan['id'], 'consulted_families': [], 'independent': False, 'scope': DISCLOSURE}
        records = {family: self.manifest['families'].get(family, {'aliases': [family], 'seeds': []}) for family in intended}
        with files.locked(self.audit):
            existed = self.audit.exists()
            history = files.read_jsonl(self.audit)
            if any(not _consultation(r) for r in history):
                raise ConfigError(f'{self.audit}: malformed audit exposure record')
            missing = not existed or any(r.get('history_missing') for r in history)
            consulted, covered = set(), set()
            for record in history:
                if record['plan']['id'] == plan['id']:
                    covered.update(map(self.resolve, record['families']))
                    continue
                touched = set(map(self.resolve, (*record['families'], *record.get('aliases', ())))) \
                    | {self.index.name(node) for node in record.get('seeds', ()) if tuple(node) in self.index}
                consulted |= touched & set(intended)
            repeat = set(intended) <= covered
            if not repeat:
                files.append(self.audit, files.jsonl([{'schema': AUDIT_SCHEMA, 'plan': thaw(plan), 'dataset_id': self.dataset_id, 'manifest': self.path.name,
                                                       'families': intended, 'aliases': sorted({a for r in records.values() for a in r['aliases']}),
                                                       'seeds': sorted({tuple(n) for r in records.values() for n in r['seeds']}),
                                                       **({'history_missing': True} if missing else {}), 'created': files.now()}]))
        state = 'unknown_history' if missing else 'consulted' if consulted else 'repeat' if repeat else 'fresh'
        return {'state': state, 'plan': plan['id'], 'families': len(intended), 'independent': state in ('fresh', 'repeat'),
                'consulted_families': intended if missing else sorted(consulted),
                'history': 'missing' if missing else 'recorded', 'scope': DISCLOSURE}
