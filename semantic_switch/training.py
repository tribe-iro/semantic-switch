from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

from . import files
from .api import NATIVE_SETTINGS, ModelJob, Runtime, estimate, execution_identity, judge, model_job, native_settings, peak
from .artifacts import (BUNDLE_SCHEMA, RECEIPT_NAME, RECEIPT_SCHEMA, USES, AnswerGroups, Presentation, artifact_digest, bundle_task, canonical_answer,
                        load_receipt, membership_record, native_model, native_score_requests, publish_bundle, read_export, verify_files)
from .backend import gpu_available
from .compiler import compile_spec, digest, is_digest
from .context import ContextError, json_value
from .errors import ConfigError
from .examples import (ASSESSMENT, DATASET_SCHEMA, ROLES, Accumulator, Actor, Frame, LabelSource, Verification, addressed, assessments, collect, iter_units,
                       label_record, merge_envelopes, merge_sources, path_record, record_path, resolution, role_of, situation_record, unit_of)
from .families import FamilyIndex, exact_families
from .heads import FEATURE_IDENTITY, N_FEATURES, LexicalHead, features
from .journal import Journal
from .journal import replay as read_journal
from .judge import validate_requests
from .judgment import ESCAPE
from .metrics import OPTION_SOFTMAX, Progress, class_changes, correctness, nested, observed, per_class, primitive, probabilities, recall_counts, recall_floor, regressions, scored_answer, semantic_answer
from .permissions import TrainingUse, admit, restriction
from .protocols import MAX_QUESTION_TOKENS, ScoreBlock, ScoreLedger, ScoreRequest, Tokens
from .resident import Adapter, Resources, binding_identity
from .results import thaw
from .tools.dataset import EvidenceContext, evaluation_plan
from .tools.deployment import (DECISION_SCHEMA, Entry, arm_summary, assess_lineage, check_deployment, comparable, compare_deployments, current, decide,
                               declaration, entry, fitted, freeze_specialization, frozen, point, promote, promotion_decision, realized, recompute_comparison,
                               reference_support, rollback, unproven)
from .tools.evaluate import contrast_pairs, score_observations
from .tools.specialize import REQUEST_SCHEMA, RESULT_SCHEMA, decode_request, read_request, sources, specialize, terminal
from .tools.specialize import request as specialization_request

TASK_FREEZE, FROZEN_TASK, CORRECTIONS = 1, 'frozen-task.json', 'task.examples.jsonl'


def evidence_snapshot(value, *, mutable=True):
    try:
        owned = json_value(value)
        return thaw(owned) if mutable else owned
    except ContextError as error:
        raise ConfigError(f'not supported finite JSON evidence ({error})') from None


def target_scope(plan, scope=None) -> list:
    observed = [list(plan.facts[f].path) for f in plan.observed]
    if scope is None:
        return observed
    resolved = []
    for address in [scope] if isinstance(scope, str) else scope:
        parts = address.split('/') if isinstance(address, str) else list(address)
        if not 1 <= len(parts) <= 2 or not all(isinstance(part, str) and part for part in parts) or plan.fact(parts[0]).hidden:
            raise ConfigError(f'target scope {address!r} must name a visible fact or FACT/MEMBER')
        plan.at(*parts)
        paths = [path for path in observed if path[:len(parts)] == parts]
        if not paths:
            raise ConfigError(f'target scope {address!r} has no directly scored targets; scope a check/pick/rate fact or a labels member')
        resolved.extend(path for path in paths if path not in resolved)
    return sorted(resolved)


def describe(plan) -> dict:
    return {'switch_rev': plan.switch_rev,
            'fields': {name: {'type': field.type, 'required': field.required, 'values': list(field.values) if field.values is not None else None}
                       for name, field in plan.fields.items()},
            'targets': {'/'.join(fact.path): {'meaning_rev': fact.meaning_rev, 'kind': fact.kind, 'answers': {key: fact.value(index) for index, key in enumerate(fact.keys)},
                                              'reads': list(fact.context_reads or fact.reads), 'policy': fact.policy.thresholds(fact.kind), 'binding': plan.binding_of(f)}
                        for f, fact in ((f, plan.facts[f]) for f in plan.observed)}}


def describe_task(definition, scope=None) -> dict:
    path = Path(definition)
    plan = compile_spec(files.read_yaml(path), path.name, path.parent)
    return {**describe(plan), **({'scope': ['/'.join(target) for target in target_scope(plan, scope)]} if scope is not None else {})}


def assigned_evidence(definition, labels, scope=None) -> dict:
    from .tools.data import Labels
    with Runtime() as runtime:
        switch = runtime.load(Path(definition), draft=True)
        assigned = Labels(switch, labels)
    plan = switch.plan
    addresses = target_scope(plan, scope)
    rows = [{'id': example.id, **path_record(fact.path), 'classes': list(fact.keys), 'label': example.label, 'role': role_of(example.split, example.fold),
             'tags': list(example.tags), 'component': assigned.component(example.group)}
            for name, examples in assigned.by_fact.items() if list((fact := plan.fact(name)).path) in addresses for example in examples]
    return {'dataset_id': assigned.dataset_id, 'scope': ['/'.join(address) for address in addresses], 'rows': rows}


def export_native(definition, to, *, judge, labels, task_family, scope=None) -> dict:
    from .tools.export import export
    with Runtime(judge, own_judges=True) as runtime:
        return export(runtime, Path(definition), to, native=True, task_family=task_family, scope=scope, labels=labels)


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
    return {**record, 'bundle': str(directory), 'definition': str(directory / 'task.yaml')}


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


SCOPES = ('situation', 'projection')


class Task:
    def __init__(self, switch):
        self._plan = switch.plan
        self.description = describe(self._plan)

    def targets(self, fact=None) -> dict:
        if fact is not None:
            self._plan.fact(fact)
        return {address: entry for address, entry in self.description['targets'].items() if fact is None or address.split('/')[0] == fact}

    def frame(self, data) -> Frame:
        return Frame(self._plan, data)

    def key(self, data) -> str:
        return Frame(self._plan, data).full

    def input_key(self, data) -> str:
        return Frame(self._plan, data).family

    def observation(self, data, address) -> str:
        return Frame(self._plan, data).observation(self._plan.at(*(address.split('/') if isinstance(address, str) else address)))

    def identity(self, scope):
        if scope == 'situation':
            return lambda data: digest(Frame(self._plan, data).full)
        if scope == 'projection':
            return self.input_key
        raise ConfigError(f"identity scope is one of {', '.join(SCOPES)}, not {scope!r}")

    def contract(self, scope) -> str:
        self.identity(scope)
        reads = sorted({read for target in self.description['targets'].values() for read in target['reads']}) if scope == 'projection' else None
        return digest({'scope': scope, 'fields': sorted(self.description['fields']), 'reads': reads})

    def situation(self, data, **metadata) -> dict:
        return situation_record(self._plan, data, **metadata)

    def unit(self, envelope):
        return unit_of(self._plan, envelope)

    def situations(self, path, *, max_situations, max_bytes) -> list:
        path = Path(path)
        found = self.accumulator(max_situations=max_situations, max_claims=0, max_bytes=max_bytes)
        for line, unit in iter_units(self._plan, path, labelled=False):
            try:
                found.add(unit)
            except ConfigError as error:
                raise ConfigError(f'{path}:{line}: {error}') from None
        if not len(found):
            raise ConfigError(f'{path}: no situations to label')
        return [found.settle(key, strict=True)[0] for key in found.keys()]

    def label(self, data, facts, **metadata) -> dict:
        return label_record(self._plan, data, facts, **metadata)

    def accumulator(self, *, max_situations, max_claims, max_bytes) -> Accumulator:
        return Accumulator(self._plan, max_situations=max_situations, max_claims=max_claims, max_bytes=max_bytes)

    def merge(self, previous, record) -> dict:
        return merge_envelopes(self._plan, previous, record)

    def resolve(self, data, address, alternatives, value, *, reviewer, actor) -> dict:
        path = tuple(address.split('/')) if isinstance(address, str) else tuple(address)
        return resolution(self._plan, data, path, alternatives, value, LabelSource('authored', reviewer, actor=Actor.parse(actor)))

    def assess(self, record, review=None) -> dict:
        return assessments(self._plan, record, review)

    def admit(self, records) -> int:
        return admit((f"situation {digest(Frame(self._plan, record['input']).full)[:12]} {address}", history)
                     for record in records for address, history in addressed(self._plan, record))


def publication(definition, scope, bindings=None) -> dict:
    from .tools.deployment import composition
    path = Path(definition)
    role, kind, _, remaining = composition(compile_spec(files.read_yaml(path), path.name, path.parent), scope, bindings)
    return {'binding': role, 'publication': kind, 'remaining': sorted(set(remaining.values()))}


def trial(runtime, definition, targets, binding):
    from .switch import Switch
    return Switch(runtime, Path(definition), draft=True, scope=(tuple(address.split('/') if isinstance(address, str) else tuple(address) for address in targets),
                                                               binding))


class Engine(Runtime):
    def _scorer(self, role):
        scorer = getattr(self.judge(role), "scorer", None)
        if scorer is None:
            raise ConfigError(f"model role {role!r} has no native scorer")
        return scorer

    def score_native(self, requests: Sequence[ScoreRequest], *, role: str = "judge", reference: bool = False) -> ScoreBlock:
        with self.operation():
            requests = validate_requests(requests)
            scorer = self._scorer(role)
            with scorer.operation() if hasattr(scorer, "operation") else nullcontext():
                block = scorer.score(requests, reference=True) if reference else scorer.score(requests)
            rows = sum(len(r.suffixes) for r in requests)
            if block.logp.shape[0] != rows or len(block.log_mass) != rows or len(block.error) != rows:
                raise ConfigError("native scorer returned incomplete ordered scores")
            return block

    def invalidate(self, role: str = "judge") -> None:
        with self.operation():
            self._scorer(role).invalidate()

    def detokenize(self, tokens: Sequence[int], *, role: str = "judge") -> str:
        with self.operation():
            return self._scorer(role).detokenize(tokens)


class Selection(Engine):
    def __init__(self, models, population, *, job=None):
        super().__init__(bindings=models if isinstance(models, dict) else {'judge': models}, job=job)
        self._population, self._ledgers = tuple(population), {}

    def _view(self, role, scorer):
        identity, key = self.identity(role), getattr(scorer, 'binding_identity', None)
        self._ledgers[role] = ScoreLedger(scorer, (lambda: key) if key else lambda: binding_identity(identity, execution_identity(getattr(scorer, 'capacity', dict)())),
                                          self._population)
        return self._ledgers[role]

    def replay(self, requests, role='judge'):
        with self.operation():
            self.judge(role)
            return self._ledgers[role].replay(validate_requests(requests), lambda missing: self.score_native(missing, role=role))

    def close(self):
        try:
            super().close()
        finally:
            for ledger in self._ledgers.values():
                ledger.release()
