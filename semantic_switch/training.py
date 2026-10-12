from __future__ import annotations

import json
from pathlib import Path

from . import files
from .artifacts import bundle_task, verify_files
from .compiler import compile_spec, digest
from .context import canonical
from .errors import ConfigError
from .examples import collect, label_record, path_record, read_envelopes, situation_record
from .results import thaw

INTEGRATION = 1
TASK_FREEZE, FROZEN_TASK, CORRECTIONS = 1, 'frozen-task.json', 'task.examples.jsonl'


def describe(plan) -> dict:
    return {'integration': INTEGRATION, 'switch_rev': plan.switch_rev,
            'fields': {name: {'type': field.type, 'required': field.required, 'values': list(field.values) if field.values is not None else None}
                       for name, field in plan.fields.items()},
            'targets': {'/'.join(fact.path): {'meaning_rev': fact.meaning_rev, 'kind': fact.kind, 'answers': {key: fact.value(index) for index, key in enumerate(fact.keys)},
                                              'reads': list(fact.context_reads or fact.reads), 'policy': fact.policy.thresholds(fact.kind)}
                        for fact in (plan.facts[f] for f in plan.observed)}}


def describe_task(definition) -> dict:
    path = Path(definition)
    return describe(compile_spec(files.read_yaml(path), path.name, path.parent))


def corrections(plan, examples) -> list:
    from .tools.common import input_key
    from .tools.data import observation_id
    return [{'observation': observation_id(fact, example.key), **path_record(fact.path), 'meaning_rev': fact.meaning_rev, 'projection': example.key,
             'value': example.value, 'example': example.id, 'label_sources': [source.digest for source in example.sources],
             'family_seeds': sorted([['observation', observation_id(fact, example.key)], ['key', input_key(plan, example.input)],
                                     ['rendered', digest([fact.name, example.rendered])], *(['group', str(group)] for group in example.groups),
                                     *(['pair', str(pair)] for pair in example.pairs)])}
            for fact in plan.facts for example in examples[fact.name]]


def task_record(plan, examples) -> dict:
    population = corrections(plan, examples)
    record = {'revision': TASK_FREEZE, 'switch': plan.name, 'switch_rev': plan.switch_rev, 'spec': digest(thaw(plan.spec)),
              'targets': {'/'.join(fact.path): {'meaning_rev': fact.meaning_rev, 'guidance': fact.judgment.guidance.digest}
                          for fact in (plan.facts[f] for f in plan.observed)},
              'guidance': sorted({section.content_digest for fact in plan.facts for section in fact.judgment.guidance.sections}),
              'exact_override': plan.exact_override, 'corrections': digest(population), 'overrides': population if plan.exact_override else []}
    return {**record, 'behavior': digest(record)}


def freeze_task(definition, directory) -> dict:
    definition, directory = Path(definition), Path(directory)
    plan = compile_spec(files.read_yaml(definition), definition.name, definition.parent)
    examples, _ = collect(plan, files.read_jsonl(files.Paths.of(definition).examples))
    listed = bundle_task(plan, directory)
    listed[CORRECTIONS] = files.write_new(directory / CORRECTIONS, ''.join(json.dumps(example.record(), sort_keys=True) + '\n'
                                                                           for fact in plan.facts for example in examples[fact.name]))
    record = {**task_record(plan, examples), 'files': listed}
    files.write_new(directory / FROZEN_TASK, record)
    return frozen_task(directory, record['behavior'])


def frozen_task(directory, behavior) -> dict:
    directory = Path(directory)
    remedy = 'the frozen task changed since preparation; prepare a new job'
    record = files.read_json(directory / FROZEN_TASK)
    if not isinstance(record, dict) or record.get('revision') != TASK_FREEZE or not isinstance(record.get('files'), dict) \
            or {'task.yaml', CORRECTIONS} - set(record['files']):
        raise ConfigError(f'{directory}: not a frozen task bundle (revision {TASK_FREEZE}); prepare a new job')
    try:
        verify_files(directory, record['files'], remedy)
    except ValueError as error:
        raise ConfigError(f'{directory}: {error}') from None
    plan = compile_spec(files.read_yaml(directory / 'task.yaml'), 'task.yaml', directory)
    examples, _ = collect(plan, files.read_jsonl(directory / CORRECTIONS))
    if {**task_record(plan, examples), 'files': record['files']} != record or record['behavior'] != behavior:
        raise ConfigError(f'{directory}: {remedy}')
    return {**record, 'bundle': str(directory), 'definition': str(directory / 'task.yaml')}


class Task:
    def __init__(self, switch):
        self._plan = switch.plan
        self.description = describe(self._plan)

    def targets(self, fact=None) -> dict:
        if fact is not None:
            self._plan.fact(fact)
        return {address: entry for address, entry in self.description['targets'].items() if fact is None or address.split('/')[0] == fact}

    def key(self, data) -> str:
        return canonical(data, tuple(self._plan.fields))

    def situations(self, path, *, max_situations, max_bytes) -> list:
        path = Path(path)
        if path.stat().st_size > max_bytes:
            raise ConfigError(f'{path}: {path.stat().st_size} bytes exceed the acquisition budget of {max_bytes}; split the source or raise max_bytes explicitly')
        merged = {}
        for line, record in read_envelopes(self._plan, path, labelled=False):
            key = self.key(record['input'])
            previous = merged.get(key)
            if previous is None:
                if len(merged) >= max_situations:
                    raise ConfigError(f'{path}:{line}: more than {max_situations} situations exceed the acquisition budget; split the source or raise max_situations explicitly')
                merged[key] = record
            elif previous.get('role') != record.get('role'):
                raise ConfigError(f"{path}:{line}: duplicate decision input with roles {previous.get('role')} and {record.get('role')}; give it one role")
            else:
                merged[key] = situation_record(
                    self._plan, record['input'], groups=[*previous['groups'], *record['groups']], pairs=[*previous['pairs'], *record['pairs']],
                    role=record.get('role'), tags=[*previous.get('tags', ()), *record.get('tags', ())], selections=[*previous['selections'], *record['selections']])
        if not merged:
            raise ConfigError(f'{path}: no situations to label')
        return list(merged.values())

    def label(self, data, facts, **metadata) -> dict:
        return label_record(self._plan, data, facts, **metadata)
