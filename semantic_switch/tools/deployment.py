from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

from .. import files
from ..artifacts import RECEIPT_NAME, Receipt, artifact_digest, bundle_task, family_overlap, load_receipt, membership_rows, validate_membership
from ..compiler import compile_spec, digest
from ..context import bind_input, canonical, project
from ..errors import ConfigError
from ..evidence import ASSESSMENT, assessments, collect, label_provenance, labelled, partial, read_envelopes, record_path
from ..metrics import METRICS, recall_counts, recall_floor
from ..results import freeze, thaw
from ..snapshot import LOCK_SCHEMA, lock_id
from .common import input_key
from .dataset import EvidenceContext, evaluation_plan

CHECK_SCHEMA = 6
LINEAGE = ('known_overlap', 'undeclared_overrides')
FREEZE_REVISION = 3
COMPARED = ('sha256', 'template', 'prompt', 'presentation')


REFERENCES = ('verified', 'computed', 'authored')
ESTABLISHED = ('verified', 'computed')
REFERENCE_VERDICTS = {'agreement': 'reference_agreement', 'partial': 'partial_reference', 'unreviewed': 'unreviewed_reference',
                      'unavailable': 'unassessed_reference', 'empty': 'no_check_evidence'}


def target_references(plan, item, review) -> dict:
    return {address: assessment.record() for address, assessment in assessments(plan, item, review).items()}


def reference_support(records, *, scope, review) -> dict:
    records = list(records)
    known = [record for record in records if record is not None and record.get('binding') is not None]
    states = Counter(record['state'] for record in known)
    within = sum(record['state'] in scope for record in known)
    protocol = (review or {}).get('status', 'undisclosed')
    state = 'empty' if not records else 'unavailable' if len(known) < len(records) else 'unreviewed' if protocol not in ('undisclosed', 'reviewed') \
        else 'established' if within == len(records) else 'partial' if within else 'agreement'
    return {'state': state, 'scope': list(scope), 'review': protocol, 'targets': len(records), 'assessed': len(known), 'within_scope': within,
            'strata': dict(sorted(states.items())), 'derivations': dict(sorted(Counter(record['derivation'] for record in known).items())),
            'established': sum(record['state'] in ESTABLISHED for record in known), 'verified': states['verified'],
            'unverified': len(known) - states['verified'], 'stale': sum(record['verifications']['stale'] > 0 for record in known),
            'conflicting': states['conflicting'], 'procedures': sorted({tuple(p) for record in known for p in record['procedures']}),
            'fidelity': dict(sorted(Counter(record['fidelity'] for record in known).items())),
            'bindings': len({json.dumps(record['binding'], sort_keys=True) for record in known}), 'unassessed': len(records) - len(known) + states['unassessed'],
            'denominator': 'scored target observations; each is assessed from its own current label-source history, replacements and verifications '
                           'bound to its input, answer and the frozen assessment binding (task meaning plus declared review bindings); a stored reference '
                           'without its binding is unassessed',
            'rule': 'quality claims need every scored target inside the predeclared reference scope and a complete review protocol when the evidence declares one; '
                    'one authored, computed or verified target never upgrades the others, judged or unknown references remain agreement, and actor kind '
                    'establishes nothing; training permission is checked separately'}


@dataclass(frozen=True)
class Entry:
    path: Path
    lock: Path
    data: Mapping
    receipt: Receipt | None = None

    @property
    def id(self) -> str:
        return self.receipt.id if self.receipt is not None else self.data['id']

    @property
    def plan(self):
        return compile_spec(thaw(self.data['spec']), self.lock.name)

    @property
    def rows(self) -> tuple:
        if self.receipt is not None:
            return tuple(self.receipt.provenance.rows)
        return tuple(row for record in self.data.get('predictor_artifacts', {}).values() for row in membership_rows(record))

    def identity(self) -> dict:
        common = {'entry': self.id, 'lock': self.data['id'], 'models': thaw(self.data['models']), 'execution': thaw(self.data['execution'])}
        return {**common, 'receipt': self.receipt.id, 'native': self.receipt.native} if self.receipt is not None else common


def frozen(path) -> Entry | None:
    path = Path(path)
    if path.is_dir():
        if not (path / RECEIPT_NAME).is_file():
            raise ConfigError(f'{path}: a directory loads as a fitted lock bundle, but it holds no {RECEIPT_NAME}')
        path = path / RECEIPT_NAME
    if path.suffix != '.json':
        return None
    data = files.read_json(path)
    if not isinstance(data, dict):
        raise ConfigError(f'{path}: no frozen lock, fitted lock bundle or current pointer here')
    if data.get('kind') == 'current-deployment':
        return _current(path)
    if data.get('kind') == 'specialization-receipt':
        receipt = load_receipt(path)
        lock = receipt.directory / receipt.data['lock']['path']
        return Entry(receipt.path, lock, freeze(files.read_json(lock)), receipt)
    sibling = files.read_json(path.parent / RECEIPT_NAME)
    if isinstance(sibling, dict) and sibling.get('kind') == 'specialization-receipt' and (sibling.get('lock') or {}).get('path') == path.name:
        return frozen(path.parent / RECEIPT_NAME)
    if data.get('schema') != LOCK_SCHEMA:
        raise ConfigError(f"{path}: unsupported lock schema {data.get('schema')} (current {LOCK_SCHEMA}); regenerate this lock with ss lock")
    if not isinstance(data.get('spec'), dict) or data.get('id') != lock_id(data):
        raise ConfigError(f'{path}: not an intact frozen lock; regenerate it with ss lock')
    return Entry(path, path, freeze(data))


def entry(path) -> Entry:
    found = frozen(path)
    if found is None:
        raise ConfigError(f'{path}: name a frozen lock (X.lock.json), a fitted lock bundle or a current pointer, not an authored definition')
    return found


def fitted(path, **settings):
    found = entry(path)
    if found.receipt is None:
        raise ConfigError(f'{path}: an ordinary lock pins a model identity but carries no fitted weights; bind the model it was locked with')
    return found.receipt.model(**settings)


def assess_lineage(deployment, context, items) -> dict:
    plan, lock, rows = deployment.plan, deployment.data, deployment.rows
    seeds = [node for row in rows for node in row['family_seeds']]
    if deployment.receipt is None:
        seeds += [node for stages in lock['family_seeds'].values() for stage, nodes in stages.items() if stage != 'validated' for node in nodes]
    frames = [(input_key(plan, data), family, project(plan.fields, bind_input(plan.fields, data))) for data, family in items]
    embedded, _ = collect(plan, locked_examples(lock))
    declared = {(record_path(row), row['projection']) for row in rows if 'override' in row['uses']}
    targets = {}
    for f in plan.observed:
        fact = plan.facts[f]
        active = {example.key for example in embedded[fact.name]} if plan.exact_override else set()
        undeclared = sorted(key for key in active if (fact.path, key) not in declared) if deployment.receipt is not None else []
        reads = fact.context_reads or fact.reads
        found = family_overlap([row for row in rows if record_path(row) == fact.path], context,
                               [(key, family, canonical(partial(values, reads), reads)) for key, family, values in frames], seeds=seeds, used=active)
        overlap = sorted(found.overlap)
        state = 'known_overlap' if overlap else 'undeclared_overrides' if undeclared else 'declared_disjoint'
        targets[f'judge:{fact.shown}'] = {'state': state, 'overlap': overlap, 'undeclared_overrides': len(undeclared)}
    states = {entry['state'] for entry in targets.values()}
    return {'state': next((state for state in LINEAGE if state in states), 'declared_disjoint'), 'entry': deployment.id, 'dataset_id': context.dataset_id,
            'situations': len(frames), 'targets': targets,
            'scope': "the lock's frozen task targets against this evidence assignment: adaptation, selection, acceptance_fit, tuning and override memberships, "
                     'their family closure and undeclared embedded corrections; pretraining of a base or hosted model is outside it'}


def unproven(reason) -> dict:
    return {'state': 'unproven', 'reason': reason, 'targets': {}}


def check_items(plan, labels, evidence):
    context = EvidenceContext.from_manifest(evidence)
    labels = context.bind(labels)
    assigned = [(record, context.member(plan, record['input'])) for _, record in read_envelopes(plan, labels, labelled=True)]
    check = [(item, family) for item, family in assigned if context.role(family) == 'check']
    if not check:
        raise ConfigError(f'{labels}: no labelled situations in check-role families of evidence {context.path}')
    return context, labels, check


def check_deployment(lock, labels, evidence, to, *, bindings=None, job=None) -> dict:
    from ..api import Runtime
    from .evaluate import evaluate_facts, fact_rows, withhold
    to = Path(to)
    if (to / 'check.json').exists():
        raise ConfigError(f'{to}: a check report is already published there; evaluate into a new directory')
    deployment = entry(lock)
    plan = deployment.plan
    scope = [list(plan.facts[f].path) for f in plan.observed]
    frozen_plan = evaluation_plan({'specialist': deployment.identity()}, scope=scope, metrics={'check_schema': CHECK_SCHEMA, 'metrics': METRICS,
                                                                                          'assessment': ASSESSMENT})
    context, labels, check = check_items(plan, labels, evidence)
    items, families = [item for item, _ in check], [family for _, family in check]
    lineage = assess_lineage(deployment, context, [(item['input'], family) for item, family in check])
    exposure = context.consult(frozen_plan, families)
    with Runtime(bindings=bindings, job=job) as runtime:
        switch = runtime.load(deployment.path)
        snap = switch.snapshot
        report = {'schema': CHECK_SCHEMA, 'kind': 'deployment-check', 'entry': deployment.id, 'candidate_id': snap.candidate_id, 'lock': snap.lock_id,
                  'execution': runtime.execution(plan.binding), 'plan': frozen_plan,
                  'evidence': {'dataset_id': context.dataset_id, 'manifest': str(context.path), 'audit': str(context.audit), 'labels_sha256': files.sha256_file(labels)},
                  'exposure': exposure, 'review': context.review, 'label_provenance': label_provenance(plan, items, review=context.review), 'lineage': lineage,
                  'reference': reference_support([record for item in items for record in target_references(plan, item, context.review).values()],
                                                 scope=ESTABLISHED, review=context.review),
                  'situations': len(items), 'families': len(set(families)),
                  'facts': evaluate_facts(switch, replace(snap, lineage=lineage['targets']), items, families, review=context.review),
                  'population': 'check of the frozen deployment on check-role families; selection and calibration are not repeated'
                                if exposure['independent'] else f"reused check evidence ({exposure['state']}); descriptive only, not an independent adoption claim"}
    withhold(list(fact_rows(report['facts'])), exposure)
    report['id'] = digest(report)
    files.publish(to / 'check.json', report)
    return report


def realized(runtime, role='judge') -> dict:
    judge = runtime.judge(role)
    return {'identity': thaw(runtime.identity(role)), 'execution': thaw(runtime.execution(role)),
            'framing': judge.adapter.framing.as_dict() if hasattr(judge, 'adapter') else None}


def comparable(realized_, compared=COMPARED) -> dict:
    return {'judge': {key: realized_['identity'].get(key) for key in compared}, 'framing': realized_['framing'],
            'execution': {key: value for key, value in (realized_['execution'] or {}).items() if key != 'build'}}


def _acceptance(plan, rows, provenance):
    from .data import Example
    uses = {row['id']: list(row['uses']) for row in provenance}
    by_path = {}
    for row in rows:
        try:
            fact = plan.at(*record_path(row))
        except ConfigError as error:
            raise ConfigError(f"acceptance observation {row['id']}: unresolved address {'/'.join(record_path(row))} ({error})") from None
        if row['meaning_rev'] != fact.meaning_rev or list(row['classes']) != list(fact.keys):
            raise ConfigError(f"acceptance observation {row['id']}: meaning revision or label classes differ from the frozen task; re-export")
        if [use for use in uses.get(row['id'], ()) if use != 'override'] != ['acceptance_fit']:
            raise ConfigError(f"acceptance observation {row['id']}: provenance declares uses {uses.get(row['id'])}, not acceptance_fit")
        by_path.setdefault(fact.path, []).append(Example(row['id'], fact.name, row['bound_input'], row['label'], row['projection'], row['component'], row['split'],
                                                         row['taught'], row['text'], row['fold'], path=fact.path))
    return by_path


def locked_examples(lock) -> list:
    def records(entry):
        return [*entry['examples'], *(record for member in entry.get('members', {}).values() for record in records(member))]
    return [record for entry in lock['facts'].values() for record in records(entry)]


def composition(plan, scope, bindings) -> tuple[str, str, frozenset, dict]:
    role = scope['binding'] if scope else 'judge'
    whole = scope is None or scope['whole_task']
    chosen = frozenset(plan.facts[f].name for f in plan.observed) if whole else \
        frozenset(plan.at(*address.split('/')).name for address in scope['targets'])
    remaining = {} if whole else {name: binding for name, binding in plan.targets.items() if name not in chosen}
    roles, supplied = set(remaining.values()), set(bindings or {})
    if supplied:
        shared = sorted('/'.join(plan.facts[plan.fact_index[name]].path) for name, binding in remaining.items() if binding == role)
        if role in roles:
            raise ConfigError(f'targets {shared} outside the specialized scope share its binding {role!r}; a complete application cannot serve them with '
                              'an adapter fitted for other targets: bind them to another model role in the definition, or publish scoped support')
        if supplied != roles:
            raise ConfigError(f'complete-application publication binds every remaining target role {sorted(roles)}; missing {sorted(roles - supplied)}, '
                              f'unused {sorted(supplied - roles)}. Nothing was allocated or staged')
    return role, 'application' if supplied or not roles else 'scoped', chosen, remaining


def freeze_specialization(model, task, acceptance, provenance, population, expected, staging, *, job, scope=None, bindings=None) -> dict:
    from ..api import Runtime
    from ..artifacts import binding_pins
    from ..training import frozen_task, task_record, trial
    from .lock import judged, lock_payload, policy_sides
    from .policy import fit_report
    staging = Path(staging)
    if not staging.is_dir() or any(staging.iterdir()):
        raise ConfigError(f'{staging}: freezing needs an empty caller-owned staging directory')
    frozen = frozen_task(task['bundle'], task['behavior'])
    try:
        validate_membership(list(provenance))
    except ValueError as error:
        raise ConfigError(f'specialization provenance: {error}') from None
    declared = {row['id'] for row in provenance if 'override' in row['uses']}
    if declared != {item['observation'] for item in frozen['overrides']}:
        raise ConfigError('provenance override declarations differ from the frozen correction population; prepare a new job')
    authored = compile_spec(files.read_yaml(frozen['definition']), 'task.yaml', Path(frozen['definition']).parent)
    role, publication, chosen, remaining = composition(authored, scope, bindings)
    with Runtime(bindings={role: model, **(bindings or {})}, job=job) as runtime:
        switch = runtime.load(frozen['definition'], draft=True) if publication == 'application' else trial(runtime, frozen['definition'], scope['targets'], role)
        plan = switch.plan
        if task_record(plan, switch.snapshot.examples)['behavior'] != frozen['behavior']:
            raise ConfigError(f"{frozen['bundle']}: the loaded task differs from its frozen behavior record; prepare a new job")
        by_path = _acceptance(plan, acceptance, provenance)
        found = realized(runtime, role)
        if comparable(found) != comparable(expected):
            raise ConfigError('the freezing model differs from the selected trial in identity, framing or execution; no deployment staged')
        sides, fitted = {}, {}
        for f in plan.observed:
            fact = plan.facts[f]
            if publication == 'scoped' and fact.name not in chosen:
                continue
            examples = by_path.get(fact.path, [])
            report = fit_report(fact, judged(switch, switch.snapshot, f, examples), fact.policy) if examples else None
            sides[fact.name] = policy_sides(fact, fact.policy, report)
            if report is not None:
                fitted[fact.name] = {'observations': len(examples), 'families': len({example.group for example in examples}), 'sides': report}
        membership, seeds = {}, {}
        for row in provenance:
            tuning = [use for use in row['uses'] if use != 'override']
            if not tuning or 'inherited' in row:
                continue
            use = 'tuned' if tuning == ['acceptance_fit'] else 'judge'
            membership.setdefault(plan.at(*record_path(row)).name, {}).setdefault(use, set()).add(row['projection'])
            seeds.setdefault(row['fact'], {}).setdefault(use, set()).update(tuple(node) for node in row['family_seeds'])
        listed = bundle_task(plan, staging)
        pinned = binding_pins(plan, runtime, None if publication == 'application' else {name: role for name in plan.targets if name not in remaining})
        lock = lock_payload(plan, switch.snapshot.examples, **pinned, returned={name: [] for name in pinned['models']},
                            framing=found['framing'], dataset={**population, 'id': artifact_digest(population)}, sides={'judge': sides},
                            membership=membership, lineage={}, family_seeds={fact: {use: sorted(nodes) for use, nodes in uses.items()} for fact, uses in seeds.items()},
                            tests='not run during selection; the independent check evaluates the frozen deployment')
        listed['task.lock.json'] = files.write_new(staging / 'task.lock.json', lock)
        staged = compile_spec(files.read_yaml(staging / 'task.yaml'), 'task.yaml', staging)
        embedded, _ = collect(staged, locked_examples(lock))
        if task_record(staged, embedded)['behavior'] != frozen['behavior']:
            raise ConfigError('the staged task and lock do not reproduce the frozen definition, guidance and corrections; no deployment staged')
    behavior = {**{key: frozen[key] for key in ('behavior', 'spec', 'switch_rev', 'targets', 'guidance', 'exact_override', 'corrections')},
                'overrides': artifact_digest(frozen['overrides']), 'override_count': len(frozen['overrides'])}
    return {'revision': FREEZE_REVISION, 'files': listed, 'task': 'task.yaml', 'lock': {'path': 'task.lock.json', 'id': lock['id'], 'candidate_id': lock['candidate_id']},
            'behavior': behavior, 'publication': publication, 'bindings': sorted(pinned['models']),
            'realized': found,
            'acceptance': {'policy': 'fitted' if fitted else 'authored', 'calibration': 'in_sample' if fitted else 'absent', 'acceptance_fit': fitted,
                           'targets': sorted(sides), 'evidence': 'in-sample threshold-fit diagnostics; the independent check supplies held-out evidence'}}


POINTER_SCHEMA, COMPARE_SCHEMA, DECISION_SCHEMA = 2, 5, 2
UNRESOLVED = ('uncertain', 'needs_input')
CLAIMS = ('improvement', 'non_inferiority')
BENEFITS = ('latency', 'cost')
CONSTRAINTS = {'failed': 'failed', 'unresolved': 'unresolved', 'forbidden': 'forbidden_accepted'}


def _pointer(pointer) -> dict | None:
    record = files.read_json(Path(pointer))
    if record is not None and (record.get('schema') != POINTER_SCHEMA or record.get('kind') != 'current-deployment'):
        raise ConfigError(f'{pointer}: not a schema-{POINTER_SCHEMA} current-deployment pointer; point to a new path')
    return record


def point(pointer, target, *, reason, expected) -> dict:
    pointer = Path(pointer)
    if not isinstance(reason, str) or not reason.strip():
        raise ConfigError('a current-pointer change records its reason')
    if expected is not None and not isinstance(expected, str):
        raise ConfigError('expected names the incumbent entry id the change was decided against, or None for a first adoption')
    found = entry(Path(target).resolve())
    with files.locked(str(pointer) + '.transaction'):
        previous = _pointer(pointer)
        observed = previous and previous['id']
        if observed != expected:
            raise ConfigError(f"{pointer} serves {observed or 'nothing'}, not the expected {expected or 'absent pointer'}; nothing changed. "
                              'Compare against the observed incumbent and act again')
        record = {'entry': str(found.path), 'id': found.id, 'lock_id': found.data['id'], 'reason': reason, 'at': files.now()}
        published = {'schema': POINTER_SCHEMA, 'kind': 'current-deployment', **{key: record[key] for key in ('entry', 'id', 'lock_id')},
                     'history': [*(previous or {}).get('history', []), record]}
        files.publish(pointer, published)
    return published


def _current(pointer) -> Entry:
    record = _pointer(pointer)
    if record is None:
        raise ConfigError(f'{pointer}: no current-deployment pointer; promote a verified lock first')
    found = frozen(record['entry'])
    if found is None or found.id != record['id']:
        raise ConfigError(f"{pointer}: points to {record['id']}, but {record['entry']} now holds {found and found.id}; restore it or point again")
    return found


def current(pointer) -> Path:
    return _current(pointer).path


def promotion_decision(report) -> dict:
    decision = {'schema': DECISION_SCHEMA, 'kind': 'promotion-decision', 'comparison': report['id'], 'baseline': report['arms']['baseline']['entry'],
                'candidate': report['arms']['candidate']['entry'], **report['decision']}
    return {**decision, 'id': artifact_digest(decision)}


def promote(pointer, lock, *, reason, decision=None) -> dict:
    found = entry(lock)
    if decision is None:
        return point(pointer, found.path, reason=reason, expected=None)
    record = files.read_json(Path(decision))
    if not isinstance(record, dict) or record.get('kind') != 'promotion-decision' or record.get('schema') != DECISION_SCHEMA or \
            record.get('id') != artifact_digest({k: v for k, v in record.items() if k != 'id'}):
        raise ConfigError(f'{decision}: not an intact schema-{DECISION_SCHEMA} promotion decision')
    if record['verdict'] != 'supported' or record['candidate'] != found.id:
        raise ConfigError(f"{decision}: verdict {record['verdict']} compares {record['candidate']} with {record['baseline']}, not a supported claim for "
                          f'{found.id}; nothing changed')
    return point(pointer, found.path, reason=f"{reason} (decision {record['id']})", expected=record['baseline'])


def rollback(pointer, *, reason, expected, to=None) -> dict:
    record = _pointer(pointer)
    if record is None:
        raise ConfigError(f'{pointer}: no current-deployment pointer to roll back')
    target = next((item for item in reversed(record['history']) if item['id'] != expected and to in (None, item['id'])), None)
    if target is None:
        raise ConfigError(f'{pointer}: no previously verified lock {to or ""} in its history to roll back to')
    return point(pointer, target['entry'], reason=f'rollback: {reason}', expected=expected)


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ConfigError(f'{name} must be a number in [0, 1)')
    return float(value)


RECALLS = ('raw', 'operational')


def _recall(declared) -> dict:
    declared = dict(declared or {})
    for address, classes in declared.items():
        if not isinstance(address, str) or not isinstance(classes, dict) or not classes:
            raise ConfigError('recall requirements map a target address to {class: {raw|operational: minimum recall}}')
        for name, floors in classes.items():
            if not isinstance(floors, dict) or not floors or set(floors) - set(RECALLS) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= 1 for v in floors.values()):
                raise ConfigError(f'{address} {name}: declare raw and/or operational minimum recall in (0, 1]')
    return {address: {name: {kind: float(floors[kind]) for kind in RECALLS if kind in floors} for name, floors in sorted(classes.items())}
            for address, classes in sorted(declared.items())}


def class_recall(rows, required) -> dict:
    if not required:
        return {}
    counts = recall_counts(((address,), target['gold'], target['family'], target['correct'], target['accepted'])
                           for row in rows for address, target in row['targets'].items() if address in required)
    out = {}
    for address, classes in required.items():
        for name, floors in classes.items():
            cell = counts.get((address, name), {'references': 0, 'families': 0, 'raw': 0, 'operational': 0})
            out.setdefault(address, {})[name] = {key: cell[key] for key in ('references', 'families')}
            out[address][name].update({kind: recall_floor(cell[kind], cell['references'], floor) for kind, floor in floors.items()})
    return out


def declaration(*, claim='improvement', margin, required_precision=None, tolerated=None, benefit=None, forbidden=None, reference=ESTABLISHED,
                recall=None) -> dict:
    if claim not in CLAIMS:
        raise ConfigError(f"declare the claim before comparing: one of {', '.join(CLAIMS)}")
    tolerated = {name: 0 for name in CONSTRAINTS} | dict(tolerated or {})
    if set(tolerated) != set(CONSTRAINTS) or any(type(value) is not int or value < 0 for value in tolerated.values()):
        raise ConfigError(f"tolerated regressions are nonnegative counts of {', '.join(CONSTRAINTS)}")
    if (claim == 'non_inferiority') != (benefit is not None) or benefit is not None and benefit not in BENEFITS:
        raise ConfigError(f"a non-inferiority claim declares its operational benefit ({', '.join(BENEFITS)}); an improvement claim declares none")
    forbidden = dict(forbidden or {})
    if any(not isinstance(keys, list) or not all(isinstance(key, str) for key in keys) for keys in forbidden.values()):
        raise ConfigError('forbidden answers map a target address to a list of answer keys')
    if required_precision is not None and (isinstance(required_precision, bool) or not isinstance(required_precision, (int, float))
                                           or not 0 < required_precision <= 1):
        raise ConfigError('required_precision is an accepted precision in (0, 1]')
    if isinstance(reference, str) or not reference or not set(reference) <= set(REFERENCES) or len(set(reference)) != len(tuple(reference)):
        raise ConfigError(f"the reference scope names distinct assessment states among {', '.join(REFERENCES)}; judged, unknown, mixed or conflicting "
                          'references only measure agreement')
    return {'claim': claim, 'margin': _fraction(margin, 'margin'), 'required_precision': required_precision, 'tolerated': tolerated, 'benefit': benefit,
            'forbidden': forbidden, 'reference': [state for state in REFERENCES if state in reference],
            **({'recall': _recall(recall)} if recall else {})}


def _benefit(declared, baseline, candidate):
    if declared is None:
        return None
    measured = {'latency': lambda arm: arm['latency_ms'] and arm['latency_ms']['p50'], 'cost': lambda arm: arm['provider_cost'] and arm['provider_cost']['total']}[declared]
    left, right = measured(baseline), measured(candidate)
    return {'metric': declared, 'baseline': left, 'candidate': right, 'met': left is not None and right is not None and right < left}


def decide(report) -> dict:
    declared, exposure, lineage = report['declaration'], report['exposure'], report['lineage']['candidate']
    baseline, candidate, difference = report['arms']['baseline'], report['arms']['candidate'], report['paired']['difference']
    constraints = {name: {'baseline': baseline[field], 'candidate': candidate[field], 'tolerated': declared['tolerated'][name],
                          'met': candidate[field] <= baseline[field] + declared['tolerated'][name]} for name, field in CONSTRAINTS.items()}
    precision = candidate['targets']['precision']
    benefit = _benefit(declared['benefit'], baseline, candidate)
    recalled = [floor for classes in candidate.get('class_recall', {}).values() for found in classes.values() for kind, floor in found.items() if kind in RECALLS]
    lower = None if difference['interval'] is None else difference['interval'][0]
    if exposure['state'] == 'empty':
        verdict = 'no_check_evidence'
    elif not exposure['independent']:
        verdict = 'reused_evidence'
    elif lineage['state'] != 'declared_disjoint':
        verdict = 'unproven_lineage' if lineage['state'] == 'unproven' else 'overlapping_lineage'
    elif report['reference']['state'] != 'established':
        verdict = REFERENCE_VERDICTS[report['reference']['state']]
    elif lower is None or difference['support'] != 'reported':
        verdict = 'inconclusive'
    elif not all(item['met'] for item in constraints.values()):
        verdict = 'constraint_violated'
    elif declared['required_precision'] is not None and (precision is None or precision < declared['required_precision']):
        verdict = 'precision_below_requirement'
    elif any(entry['state'] != 'met' for entry in recalled):
        verdict = 'class_recall_unassessed' if any(entry['state'] == 'unassessed' for entry in recalled) else 'class_recall_below_requirement'
    elif declared['claim'] == 'improvement':
        verdict = 'supported' if lower > declared['margin'] else 'not_supported'
    else:
        verdict = 'supported' if lower > -declared['margin'] and benefit['met'] else 'not_supported'
    return {'claim': declared['claim'], 'verdict': verdict, 'lower_bound': lower, 'support': difference['support'], 'constraints': constraints,
            'precision': {'candidate': precision, 'baseline': baseline['targets']['precision'], 'required': declared['required_precision']},
            'benefit': benefit, 'class_recall': candidate.get('class_recall', {}), 'exposure': exposure['state'], 'lineage': lineage['state'],
            'reference': report['reference']['state'],
            'rule': 'improvement: the family-resampling lower bound of candidate - baseline exceeds the margin; non-inferiority: it exceeds -margin and '
                    'the declared operational benefit is measured; both on fresh disjoint reviewed evidence with no failure, unresolved or forbidden-answer '
                    'regression beyond the declared tolerance, the declared accepted precision and every scored target inside the declared reference scope'}


def _targets(plan) -> dict:
    return {'/'.join(plan.facts[f].path): plan.facts[f].meaning_rev for f in plan.observed}


def _public(plan, result, address):
    found = result.facts.get(address[0])
    if found is not None and len(address) > 1:
        found = found.members.get(address[1])
    return found


def compared_row(plan, item, family, result, review) -> dict:
    from ..metrics import comparison_identity, scored_answer
    from ..results import predicted
    projection = input_key(plan, item['input'])
    requests = getattr(result.cost, 'requests', ())
    usage = [request.usage for request in requests if request.usage]
    references = target_references(plan, item, review)
    targets = {}
    for name, value in item['facts'].items():
        for fact, gold in labelled(plan, name, value):
            address = '/'.join(fact.path)
            found, label = _public(plan, result, fact.path), fact.keys[fact.index(gold)]
            outcome = 'unexamined' if found is None else found.status
            prediction = None if found is None else found.prediction
            targets[address] = {'key': digest([projection, list(fact.path)]), 'family': family, 'gold': label, **scored_answer(prediction, label, outcome, fact.keys),
                                'outcome': outcome, 'prediction': predicted(prediction), 'options': list(fact.keys),
                                'scores': {} if found is None else dict(found.scores), 'source': None if found is None else found.source,
                                'reference': references.get(address),
                                'comparison': comparison_identity(task=fact.meaning_rev, projection=projection, gold=label, answers=list(fact.keys),
                                                                  scope=list(fact.path), condition=None)}
    return {'input': projection, 'family': family, 'status': result.status, 'elapsed_ms': result.cost.get('elapsed_ms'),
            'amortized_ms': (result.cost.get('operation') or {}).get('amortized_ms'), 'requests': len(requests),
            'attempts': sum(request.attempts or 0 for request in requests), 'logical_tokens': result.cost.get('logical_tokens'),
            'provider_cost': sum(entry.get('cost') or 0 for entry in usage) if usage else None, 'usage_known': len(usage), 'targets': targets}


def rescored(row) -> dict:
    from ..metrics import scored_answer
    return {**row, 'targets': {address: {**target, **scored_answer(target['prediction'], target['gold'], target['outcome'], target.get('options'))}
                               for address, target in row['targets'].items()}}


def _measured(rows, field) -> dict:
    from ..metrics import ratio
    known = [row[field] for row in rows if row.get(field) is not None]
    return {'total': sum(known), 'calls': len(known), 'per_call': ratio(sum(known), len(known)), 'bound': 'total' if len(known) == len(rows) else 'lower_bound'}


def _latency(rows) -> dict | None:
    from ..metrics import percentile
    measured = [row for row in rows if row['elapsed_ms'] is not None]
    if not measured:
        return None
    def spread(values):
        return {'calls': len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95)} if values else None
    everything = spread([row['elapsed_ms'] for row in measured])
    return {**everything, 'cold': spread([row['elapsed_ms'] for row in measured if row.get('cold') is True]),
            'steady': spread([row['elapsed_ms'] for row in measured if row.get('cold') is False]),
            'unscoped': sum('cold' not in row for row in measured),
            'scope': 'per-input admission-to-completion wall time inside one bounded scoring operation, including retries; cold is the first admitted '
                     'input after an arm is loaded in a session, steady every later input; amortized operation time is reported separately'}


def call_cost(rows) -> dict:
    return {'submitted': len(rows), **{field: _measured(rows, field) for field in ('elapsed_ms', 'amortized_ms', 'requests', 'attempts', 'logical_tokens',
                                                                                   'provider_cost')},
            'scope': 'measured per submitted input: admission-to-completion and amortized operation wall time, provider requests and attempts including '
                     'retries, logical tokens including repeated '
                     'context and provider-reported cost where usage is known; never a token-price estimate'}


def arm_summary(rows, forbidden) -> dict:
    from ..metrics import correctness
    targets = [target | {'address': address} for row in rows for address, target in row['targets'].items()]
    costs = [row['provider_cost'] for row in rows if row['provider_cost'] is not None]
    statuses = Counter(row['status'] for row in rows)
    return {'calls': len(rows), 'statuses': dict(sorted(statuses.items())), 'unresolved': sum(statuses[s] for s in UNRESOLVED), 'failed': statuses['failed'],
            'targets': correctness((target['correct'], target['accepted']) for target in targets),
            'forbidden_accepted': sum(target['accepted'] and target['answer'] in forbidden.get(target['address'], ()) for target in targets),
            'latency_ms': _latency(rows), 'cost': call_cost(rows),
            'provider_cost': {'total': sum(costs), 'calls_with_usage': len(costs)} if costs else None}


ROBUSTNESS_SCHEMA = 1
TRANSFORMATIONS = {'permutation': 'invariant', 'rename': 'invariant', 'format': 'invariant', 'wording': 'invariant', 'contrast': 'changed'}
ROBUSTNESS = ('descriptive diagnostic for every arm from the stored rows of this plan; it never changes the predeclared baseline/candidate decision, and extra '
              'arms stay descriptive unless separately predeclared. Choosing an engine from these check results exposes this check set for selection; '
              'a later claim needs fresh evidence. Option paraphrases and criterion revisions are not presumed equivalent')


def _addressed(plan) -> dict:
    return {'/'.join(plan.facts[f].path): plan.facts[f] for f in plan.observed}


def _golds(plan, item) -> dict:
    return {'/'.join(fact.path): fact.keys[fact.index(gold)] for name, value in item['facts'].items() for fact, gold in labelled(plan, name, value)}


def _presentation(parent, plan, mapping, transformation, where) -> None:
    origin, shown = _addressed(parent), _addressed(plan)
    if set(origin) != set(shown):
        raise ConfigError(f'{where}: the transformed lock answers targets {sorted(shown)}, not {sorted(origin)}')
    if not isinstance(mapping, dict) or not mapping or set(mapping) - set(shown):
        raise ConfigError(f'{where}: mapping names the transformed Pick targets and maps each transformed option key to its parent key')
    for address, fact in shown.items():
        before = origin[address]
        if address not in mapping:
            if fact.meaning_rev != before.meaning_rev:
                raise ConfigError(f'{where}: {address} changed meaning without a declared option mapping')
            continue
        keys, mine, theirs = mapping[address], fact.judgment, before.judgment
        if fact.kind != 'pick' or before.kind != 'pick':
            raise ConfigError(f'{where}: {address}: permutation and rename apply to Pick options')
        if (mine.text, mine.guidance, mine.excludes) != (theirs.text, theirs.guidance, theirs.excludes):
            raise ConfigError(f'{where}: {address}: the criterion differs; a criterion revision is not presumed equivalent')
        if not isinstance(keys, dict) or sorted(keys) != sorted(fact.keys) or sorted(keys.values()) != sorted(before.keys):
            raise ConfigError(f'{where}: {address}: the mapping is a bijection from {list(fact.keys)} onto {list(before.keys)}')
        if any(mine.descriptions[fact.keys.index(k)] != theirs.descriptions[before.keys.index(v)] for k, v in keys.items()):
            raise ConfigError(f'{where}: {address}: an option description differs; option paraphrases are not presumed equivalent')
        renamed = any(k != v for k, v in keys.items())
        if renamed != (transformation == 'rename') or transformation == 'permutation' and fact.keys == before.keys:
            raise ConfigError(f'{where}: {address}: a permutation reorders the same keys and a rename changes keys')


def transforms(path, plan, arms, check, context) -> dict:
    manifest = files.read_json(Path(path))
    if not isinstance(manifest, dict) or manifest.get('kind') != 'robustness-transforms' or manifest.get('schema') != ROBUSTNESS_SCHEMA \
            or not isinstance(manifest.get('transforms'), list) or not manifest['transforms']:
        raise ConfigError(f'{path}: not a schema-{ROBUSTNESS_SCHEMA} robustness-transforms manifest with a nonempty transforms list')
    items = {input_key(plan, item['input']): (item, family) for item, family in check}

    def member(data, where, name):
        key = input_key(plan, data) if isinstance(data, dict) else None
        if key not in items:
            raise ConfigError(f'{where}: its {name} is not a labelled check-role input of this comparison')
        return key, *items[key]
    out = []
    for t in manifest['transforms']:
        kind = t.get('transformation') if isinstance(t, dict) else None
        if kind not in TRANSFORMATIONS or not isinstance(t.get('id'), str) or not t['id'] or t.get('expected') != TRANSFORMATIONS[kind]:
            raise ConfigError(f"{path}: each transform has an id, a transformation among {', '.join(TRANSFORMATIONS)} and its expected answer: "
                              'changed for contrast, invariant otherwise')
        where = f"{path}: transform {t['id']}"
        parent, item, family = member(t.get('parent'), where, 'parent')
        record = {'id': t['id'], 'transformation': kind, 'expected': t['expected'], 'parent': parent, 'family': context.resolve(family),
                  'task': _targets(plan)}
        if kind in ('permutation', 'rename'):
            locks = t.get('locks')
            if not isinstance(locks, dict) or set(locks) != set(arms):
                raise ConfigError(f'{where}: names one transformed lock for every arm {sorted(arms)}')
            record['locks'] = {}
            for name, lock in sorted(locks.items()):
                found = entry(lock)
                if (thaw(found.data['models']), thaw(found.data['execution'])) != (thaw(arms[name].data['models']), thaw(arms[name].data['execution'])):
                    raise ConfigError(f"{where}: arm {name}'s transformed lock binds another model or execution; only the presentation may change")
                _presentation(plan, found.plan, t.get('mapping'), kind, where)
                record['locks'][name] = {'path': str(found.path), 'entry': found.id, 'lock': found.data['id']}
            record['mapping'] = t['mapping']
        else:
            sibling, other, kin = member(t.get('input'), where, 'input')
            if context.resolve(kin) != record['family'] or sibling == parent:
                raise ConfigError(f'{where}: a transformed sibling is a distinct input of the parent family')
            changed = sorted(k for k in set(item['input']) | set(other['input']) if item['input'].get(k) != other['input'].get(k))
            before, after = _golds(plan, item), _golds(plan, other)
            if kind == 'contrast':
                targets = t.get('targets')
                if not isinstance(targets, list) or not targets or any(a not in before or a not in after for a in targets):
                    raise ConfigError(f'{where}: a contrast names the targets it changes, labelled on both inputs')
                if any(before[a] == after[a] for a in targets):
                    raise ConfigError(f'{where}: the reference gives the same answer on both inputs; a contrast must change its declared targets')
                if len(changed) != 1:
                    raise ConfigError(f'{where}: a contrast changes one decisive observed field, not {changed}')
            else:
                if before != after:
                    raise ConfigError(f'{where}: the reference answers differ, so the {kind} change is not meaning-preserving')
                targets = sorted(before)
            record |= {'input': sibling, 'targets': targets, 'changed_fields': changed}
        out.append(record)
    if len({record['id'] for record in out}) != len(out):
        raise ConfigError(f'{path}: transform ids repeat')
    return {'schema': ROBUSTNESS_SCHEMA, 'manifest': str(path), 'transforms': out, 'id': digest(out)}


def _side(target, mapping=None) -> dict:
    if target is None:
        return {'answer': None, 'correct': None, 'accepted': False, 'outcome': 'unavailable', 'scores': {}}
    back = mapping or {}
    return {'answer': back.get(target['answer'], target['answer']), 'correct': target['correct'], 'accepted': target['accepted'], 'outcome': target['outcome'],
            'scores': {back.get(key, key): value for key, value in target.get('scores', {}).items()}}


def robustness_pairs(record, primary, transformed) -> list:
    by_input = {row['input']: row for row in primary}
    by_transform = {row['transform']: row for row in transformed}
    pairs = []
    for t in record['transforms']:
        parent = by_input[t['parent']]
        sibling = by_transform.get(t['id']) if 'locks' in t else by_input.get(t['input'])
        for address in t.get('targets') or sorted(parent['targets']):
            mapping = (t.get('mapping') or {}).get(address)
            pairs.append({'transform': t['id'], 'transformation': t['transformation'], 'expected': t['expected'], 'family': t['family'], 'address': address,
                          'options': parent['targets'][address]['options'], 'parent': _side(parent['targets'][address]),
                          'sibling': _side(None if sibling is None else sibling['targets'].get(address), mapping)})
    return pairs


def _presented(item, mapping) -> dict:
    return {'input': item['input'], 'facts': {name: {v: k for k, v in mapping[name].items()}[value] if name in mapping else value
                                              for name, value in item['facts'].items()}}


def scored(switch, todo) -> list:
    rows = switch.score([item['input'] for _, item in todo], demand=[(sorted(item['facts']), True) for _, item in todo])
    return [(key, row.decision) for (key, _), row in zip(todo, rows)]


def _transform_rows(journal, plan_id, robust, items, bindings, job, review) -> list:
    from ..api import Runtime
    if robust is None:
        return []
    units = {digest([plan_id, name, 'transform', t['id']]): (name, t) for t in robust['transforms'] if 'locks' in t for name in t['locks']}
    grouped = {}
    for key in journal.pending(units):
        name, t = units[key]
        grouped.setdefault((name, t['locks'][name]['path']), []).append((key, t))
    for (name, path), todo in grouped.items():
        with Runtime(bindings=bindings.get(name), job=job) as runtime:
            switch = runtime.load(path)
            transforms = dict(todo)
            for n, (key, result) in enumerate(scored(switch, [(key, items[t['parent']][0]) for key, t in todo])):
                t = transforms[key]
                item, family = items[t['parent']]
                journal.commit([(key, {'arm': name, 'transform': t['id'], 'cold': n == 0,
                                       **compared_row(switch.plan, _presented(item, t['mapping']), family, result, review)})])
    return [journal.payload(key) for key in units]


BASE = ('schema', 'kind', 'plan', 'declaration', 'evidence', 'exposure', 'lineage', 'review', 'label_provenance', 'situations', 'families', 'population',
        'robustness', 'recomputed')


def summarized(base, rows) -> dict:
    from ..metrics import paired, robustness
    declared, settings = base['declaration'], base['plan']['metrics']
    primary = {name: [row for row in rows if row['arm'] == name and 'transform' not in row] for name in base['arms']}

    def flat(name):
        return [target for row in primary[name] for target in row['targets'].values()]
    expected = [{key: target[key] for key in ('key', 'family', 'gold', 'comparison')} for target in flat('baseline')]
    report = {**base, 'arms': {name: {**base['arms'][name], **arm_summary(primary[name], declared['forbidden']),
                                      'class_recall': class_recall(primary[name], declared.get('recall', {}))} for name in base['arms']},
              'paired': paired(expected, flat('baseline'), flat('candidate'), seed=settings['seed'], count=settings['replicates'], names=('baseline', 'candidate')),
              'reference': reference_support([target.get('reference') for target in flat('baseline')], scope=declared.get('reference', ()), review=base['review'])}
    if base.get('robustness') is not None:
        transformed = {name: [row for row in rows if row['arm'] == name and 'transform' in row] for name in base['arms']}
        report['robustness'] = {**base['robustness'], 'scope': ROBUSTNESS,
                                'arms': {name: {**robustness(robustness_pairs(base['robustness'], primary[name], transformed[name])),
                                                'transformed_calls': call_cost(transformed[name]), 'latency_ms': _latency(transformed[name])}
                                         for name in base['arms']}}
    report['decision'] = decide(report)
    report['id'] = digest(report)
    return report


def compare_deployments(baseline, candidate, labels, evidence, to, *, declaration, seed=0, count=2000, arms=None, bindings=None, job=None,
                        robustness=None) -> dict:
    from ..api import Runtime
    from ..journal import Journal
    to = Path(to)
    if (to / 'comparison.json').exists():
        raise ConfigError(f'{to}: a comparison is already published there; compare into a new directory')
    if not isinstance(declaration, dict) or set(declaration) != {'claim', 'margin', 'required_precision', 'tolerated', 'benefit', 'forbidden', 'reference'}:
        raise ConfigError('declare the comparison claim, margin, tolerated regressions, forbidden answers and reference scope with declaration(...) before scoring')
    extra = dict(arms or {})
    if {'baseline', 'candidate'} & set(extra) or not all(isinstance(name, str) and name for name in extra):
        raise ConfigError('additional arms carry their own nonempty names besides baseline and candidate')
    arms = {'baseline': entry(baseline), 'candidate': entry(candidate), **{name: entry(path) for name, path in extra.items()}}
    bindings = dict(bindings or {})
    if set(bindings) - set(arms):
        raise ConfigError(f'bindings name the compared arms {sorted(arms)}, not {sorted(set(bindings) - set(arms))}')
    plans = {}
    for name, deployment in arms.items():
        if deployment.receipt is not None and deployment.receipt.data['publication'] != 'application':
            raise ConfigError(f'{deployment.path}: scoped support is not a complete application; compare complete locks')
        plans[name] = deployment.plan
    for name, other in plans.items():
        if _targets(other) != _targets(plans['baseline']):
            raise ConfigError(f"the locks answer different targets or meanings ({_targets(plans['baseline'])} versus {name} {_targets(other)}); "
                              'only behavior on identical task semantics is comparable')
    plan = plans['candidate']
    scope = sorted(map(list, (address.split('/') for address in _targets(plan))))
    context, labels, check = check_items(plan, labels, evidence)
    robust = transforms(robustness, plan, arms, check, context) if robustness is not None else None
    frozen_plan = evaluation_plan({name: deployment.identity() for name, deployment in arms.items()},
                                  scope=scope, metrics={'compare_schema': COMPARE_SCHEMA, 'metrics': METRICS, 'assessment': ASSESSMENT, 'seed': seed,
                                                        'replicates': count,
                                                        'declaration': declaration, 'robustness': robust and robust['id']},
                                  margin=declaration['margin'])
    items = [item for item, _ in check]
    lineage = {name: assess_lineage(deployment, context, [(item['input'], family) for item, family in check]) for name, deployment in arms.items()}
    exposure = context.consult(frozen_plan, [family for _, family in check])
    with Journal(to / 'rows.journal.jsonl', {'kind': 'deployment-comparison', 'plan': frozen_plan}) as journal:
        rows = {}
        for name, deployment in arms.items():
            units = {digest([frozen_plan['id'], name, input_key(plan, item['input'])]): (item, family) for item, family in check}
            pending = journal.pending(units)
            if pending:
                with Runtime(bindings=bindings.get(name), job=job) as runtime:
                    switch = runtime.load(deployment.path)
                    for n, (key, result) in enumerate(scored(switch, [(key, units[key][0]) for key in pending])):
                        journal.commit([(key, {'arm': name, 'cold': n == 0, **compared_row(plan, *units[key], result, context.review)})])
            rows[name] = [journal.payload(key) for key in units]
        transformed = _transform_rows(journal, frozen_plan['id'], robust, {input_key(plan, item['input']): (item, family) for item, family in check},
                                      bindings, job, context.review)
        base = {'schema': COMPARE_SCHEMA, 'kind': 'deployment-comparison', 'plan': frozen_plan, 'declaration': declaration,
                'arms': {name: {'entry': deployment.id, 'lock': deployment.data['id']} for name, deployment in arms.items()},
                'evidence': {'dataset_id': context.dataset_id, 'manifest': str(context.path), 'audit': str(context.audit), 'labels_sha256': files.sha256_file(labels)},
                'exposure': exposure, 'lineage': lineage, 'review': context.review, 'label_provenance': label_provenance(plan, items, review=context.review),
                'situations': len(items), 'families': len({family for _, family in check}), **({'robustness': robust} if robust else {}),
                'population': 'one bounded scoring operation per arm over the check inputs on the frozen locks, '
                              'demanding each input\'s labelled targets'
                              if exposure['independent'] else
                              f"reused check evidence ({exposure['state']}); descriptive only, not an independent adoption claim"}
        stored = [row for name in arms for row in rows[name]] + transformed
        report = summarized(base, stored)
        journal.publish(to / 'comparison.json', report, {'rows.jsonl': [json.dumps(row, sort_keys=True) + '\n' for row in stored]})
    decision = promotion_decision(report)
    files.publish(to / 'decision.json', decision)
    return {**report, 'promotion': decision, 'rows': {name: rows[name] for name in arms}}


def recompute_comparison(source, to) -> dict:
    source, to = Path(source), Path(to)
    if to.exists():
        raise ConfigError(f'{to}: already exists; recompute into a new path')
    with files.open_published(source) as (stored, handles):
        if not isinstance(stored, dict) or stored.get('kind') != 'deployment-comparison' or 'rows.jsonl' not in handles:
            raise ConfigError(f'{source}: not a published deployment comparison with stored rows')
        rows = [rescored(json.loads(line)) for line in handles['rows.jsonl']]
    base = {key: stored[key] for key in BASE if key in stored} | {
        'schema': COMPARE_SCHEMA, 'arms': {name: {key: arm[key] for key in ('entry', 'lock')} for name, arm in stored['arms'].items()},
        'recomputed': {'source': str(source), 'sha256': files.sha256_file(source), 'schema': stored['schema'], 'id': stored['id'],
                       'scope': 'aggregates recomputed from the verified stored rows with the current answer reduction; no inference and no exposure '
                                'consultation, so exposure, lineage and population stay as recorded'}}
    report = summarized(base, rows)
    files.publish(to, report)
    return report
