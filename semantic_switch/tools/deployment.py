from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .. import files
from ..artifacts import artifact_digest, bundle_task, load_receipt, validate_membership
from ..compiler import compile_spec, digest
from ..context import bind_input, canonical, project
from ..errors import ConfigError
from ..examples import collect, partial, read_envelopes, record_path
from ..results import thaw
from .common import input_key
from .dataset import EvidenceContext, evaluation_plan

CHECK_SCHEMA = 2
FREEZE_REVISION = 2
COMPARED = ('sha256', 'template', 'prompt', 'presentation')


def review_summary(items) -> dict:
    histories = [history for item in items for value in item['label_sources'].values() for history in (value.values() if isinstance(value, dict) else (value,))]
    reviewers = sorted({review['reviewer'] for history in histories for source in history for review in source.get('reviews', ())})
    reviewed = sum(any(source.get('reviews') for source in history) for history in histories)
    return {'targets': len(histories), 'reviewed_targets': reviewed, 'reviewers': reviewers}


def check_deployment(receipt, labels, evidence, to, *, job=None) -> dict:
    from ..api import Runtime
    from .evaluate import evaluate_facts, fact_rows, withhold
    to = Path(to)
    if (to / 'check.json').exists():
        raise ConfigError(f'{to}: a check report is already published there; evaluate into a new directory')
    deployment = load_receipt(receipt)
    task = deployment.verified_task()
    plan = compile_spec(files.read_yaml(task), task.name, task.parent)
    scope = [list(plan.facts[f].path) for f in plan.observed]
    frozen = evaluation_plan({'specialist': {'receipt': deployment.id, 'execution': thaw(deployment.data['execution']), 'native': deployment.native}},
                             scope=scope, metrics={'check_schema': CHECK_SCHEMA})
    context = EvidenceContext.from_manifest(evidence)
    labels = context.bind(labels)
    assigned = [(record, context.member(plan, record['input'])) for _, record in read_envelopes(plan, labels, labelled=True)]
    check = [(item, family) for item, family in assigned if context.role(family) == 'check']
    if not check:
        raise ConfigError(f'{labels}: no labelled situations in check-role families of evidence {context.path}')
    items, families = [item for item, _ in check], [family for _, family in check]
    rows = deployment.provenance.rows
    trained = {context.family(node) for row in rows for node in row['family_seeds'] if tuple(node) in context.index}
    projected = [project(plan.fields, bind_input(plan.fields, item['input'])) for item in items]
    embedded, _ = collect(plan, locked_examples(files.read_json(deployment.directory / deployment.data['lock']['path'])))
    declared = {(record_path(row), row['projection']) for row in rows if 'override' in row['uses']}
    lineage = {}
    for f in plan.observed:
        fact = plan.facts[f]
        active = {example.key for example in embedded[fact.name]} if plan.exact_override else set()
        undeclared = sorted(key for key in active if (fact.path, key) not in declared)
        used = {row['projection'] for row in rows if record_path(row) == fact.path} | active
        reads = fact.context_reads or fact.reads
        overlap = sorted({input_key(plan, item['input']) for item, family, values in zip(items, families, projected)
                          if family in trained or canonical(partial(values, reads), reads) in used})
        state = 'known_overlap' if overlap else 'undeclared_overrides' if undeclared else 'declared_disjoint'
        lineage[f'judge:{fact.shown}'] = {'state': state, 'overlap': overlap, 'undeclared_overrides': len(undeclared)}
    exposure = context.consult(frozen, families)
    with Runtime.from_specialization(receipt, job=job) as runtime:
        switch = runtime.load()
        snap = switch.snapshot
        report = {'schema': CHECK_SCHEMA, 'kind': 'deployment-check', 'receipt': deployment.id, 'candidate_id': snap.candidate_id, 'lock': snap.lock_id,
                  'task_sha256': deployment.data['files'][deployment.data['task']], 'execution': runtime.execution(), 'plan': frozen,
                  'evidence': {'dataset_id': context.dataset_id, 'manifest': str(context.path), 'audit': str(context.audit), 'labels_sha256': files.sha256_file(labels)},
                  'exposure': exposure, 'review': {**context.review, **review_summary(items)}, 'lineage': lineage,
                  'situations': len(items), 'families': len(set(families)),
                  'facts': evaluate_facts(switch, replace(snap, lineage=lineage), items, families),
                  'population': 'check of the frozen deployment on check-role families; selection and calibration are not repeated'
                                if exposure['independent'] else f"reused check evidence ({exposure['state']}); descriptive only, not an independent adoption claim"}
    withhold(list(fact_rows(report['facts'])), exposure)
    report['id'] = digest(report)
    files.publish(to / 'check.json', report)
    return report


def realized(runtime) -> dict:
    judge = runtime.judge()
    return {'identity': thaw(runtime.identity()), 'execution': thaw(runtime.execution()),
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


def freeze_specialization(model, task, acceptance, provenance, population, expected, staging, *, job) -> dict:
    from ..api import Runtime
    from ..training import frozen_task, task_record
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
    with Runtime({'judge': model}, job=job) as runtime:
        switch = runtime.load(frozen['definition'], draft=True)
        plan = switch.plan
        if task_record(plan, switch.snapshot.examples)['behavior'] != frozen['behavior']:
            raise ConfigError(f"{frozen['bundle']}: the loaded task differs from its frozen behavior record; prepare a new job")
        by_path = _acceptance(plan, acceptance, provenance)
        found = realized(runtime)
        if comparable(found) != comparable(expected):
            raise ConfigError('the freezing model differs from the selected trial in identity, framing or execution; no deployment staged')
        sides, fitted = {}, {}
        for f in plan.observed:
            fact = plan.facts[f]
            examples = by_path.get(fact.path, [])
            report = fit_report(fact, judged(switch, switch.snapshot, f, examples), fact.policy) if examples else None
            sides[fact.name] = policy_sides(fact, fact.policy, report)
            if report is not None:
                fitted[fact.name] = {'observations': len(examples), 'families': len({example.group for example in examples}), 'sides': report}
        membership, seeds = {}, {}
        for row in provenance:
            tuning = [use for use in row['uses'] if use != 'override']
            if not tuning:
                continue
            use = 'tuned' if tuning == ['acceptance_fit'] else 'judge'
            membership.setdefault(plan.at(*record_path(row)).name, {}).setdefault(use, set()).add(row['projection'])
            seeds.setdefault(row['fact'], {}).setdefault(use, set()).update(tuple(node) for node in row['family_seeds'])
        listed = bundle_task(plan, staging)
        lock = lock_payload(plan, switch.snapshot.examples, models={'judge': found['identity']}, execution={'judge': found['execution']}, returned={'judge': []},
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
            'behavior': behavior,
            'realized': found,
            'acceptance': {'policy': 'fitted' if fitted else 'authored', 'calibration': 'in_sample' if fitted else 'absent', 'acceptance_fit': fitted,
                           'evidence': 'in-sample threshold-fit diagnostics; the independent check supplies held-out evidence'}}
