from __future__ import annotations

from pathlib import Path

from semantic_switch import files
from semantic_switch.artifacts import artifact_digest, read_export
from semantic_switch.errors import ConfigError

from . import assets
from .adapter import verified
from .jobs import LOSS, evidence, fit_inputs, identity, monitor_plan, resolve
from .lora import CANDIDATES_SCHEMA

SCHEMA, KIND = 1, 'candidate-compatibility'
PRECEDING = {'resolved-job.json': 7, 'candidates.json': 5, 'run.json': 3}
FITTED = ('model', 'targets', 'loss', 'objective', 'recipe', 'schedule', 'fitting', 'work', 'situations', 'probe_ids', 'memberships', 'sources')
LISTED = ('job_id', 'fitted_job_id', 'base_sha256', 'unit', 'rule', 'stop', 'candidates')
CANDIDATE = ('name', 'step', 'examples_seen', 'correct', 'nll', 'predictions', 'peft', 'peft_sha256', 'config_sha256')
RECORDS = ('resolved-job.json', 'candidates.json', 'run.json', 'progress.jsonl', 'updates.jsonl', 'metrics.json')


def original(run):
    records = {name: files.read_json(run / name) for name in PRECEDING}
    missing = [name for name in RECORDS if not (run / name).is_file()]
    if missing:
        raise ConfigError(f'{run}: missing original run records {missing}; migration needs every one of {list(RECORDS)}')
    fitted, listed, state = records['resolved-job.json'], records['candidates.json'], records['run.json']
    if any(record.get('loss') != LOSS for record in records.values()):
        raise ConfigError(f'{run}: fitted under {listed.get("loss")}; loss-v1 artifacts cannot be relabelled {LOSS["name"]} v{LOSS["version"]}: refit')
    if any(records[name].get('schema') != schema for name, schema in PRECEDING.items()) or state.get('state') != 'ready':
        raise ConfigError(f'{run}: migrate reads only a ready run of the immediately preceding schemas {PRECEDING}')
    needed = [f'resolved-job.json:{key}' for key in FITTED if key not in fitted] + [f'candidates.json:{key}' for key in LISTED if key not in listed] \
        + [f"candidates.json:{candidate.get('name')}.{key}" for candidate in listed.get('candidates', ()) for key in CANDIDATE if key not in candidate]
    if needed:
        raise ConfigError(f'{run}: missing equivalence evidence {needed}')
    if listed['fitted_job_id'] != artifact_digest(fitted):
        raise ConfigError(f'{run}: resolved-job.json differs from the job its candidates were fitted under')
    adapters = {candidate['name']: verified(fitted, run / candidate['peft'], candidate, ('config_sha256',)) for candidate in listed['candidates']}
    return fitted, {**listed, 'candidates': [{**candidate, **adapters[candidate['name']]} for candidate in listed['candidates']]}


def original_rows(fitted, resolved):
    current = {source['export_id']: source['path'] for source in resolved['sources']}
    ordered, exports = {}, []
    for source in fitted['sources']:
        if source['export_id'] in exports:
            continue
        exports.append(source['export_id'])
        path = current.get(source['export_id'], source['path'])
        if not Path(path).is_file():
            raise ConfigError(f"original export {source['export_id']} ({source['path']}) is unavailable; migration needs its rows to reconstruct the fitted and monitored "
                              'presentations, so eligibility is unproven')
        manifest, rows = read_export(path, native=True)
        if manifest['export_id'] != source['export_id']:
            raise ConfigError(f"{path}: export changed since fitting ({source['export_id']} → {manifest['export_id']}); eligibility is unproven")
        for row in rows:
            known = ordered.setdefault(row['id'], row)
            if evidence(known) != evidence(row):
                raise ConfigError(f"original observation {row['id']} has inconsistent repeated evidence across its exports; eligibility is unproven")
    roles = {}
    for member in fitted['memberships']:
        roles.setdefault(member['role'], set()).add(member['id'])
    missing = sorted({*roles.get('fit', ()), *roles.get('model_select', ()), *fitted['probe_ids']} - set(ordered))
    if missing:
        raise ConfigError(f'original fit, model_select or probe observations {missing[:5]} are absent from their recorded exports; eligibility is unproven')
    def consumed(role):
        return [row for identifier, row in ordered.items() if identifier in roles.get(role, ())]
    return consumed('fit'), consumed('model_select'), [ordered[identifier] for identifier in fitted['probe_ids']]


def original_model(fitted):
    try:
        located = assets.load(fitted['assets']['manifest'])
    except ConfigError as error:
        raise ConfigError(f"original pinned assets {fitted['assets']['manifest']} cannot be verified ({error}); base content evidence is missing, so eligibility is unproven") from None
    if (located['repo'], located['commit']) != (fitted['model']['repo'], fitted['model']['revision']):
        raise ConfigError('the original asset manifest names another base repository or revision; eligibility is unproven')
    return {**fitted['model'], 'files': located['files']}


def migrate(job, run, to):
    to, run = Path(to), Path(run).resolve()
    if to.exists():
        raise ConfigError(f'{to}: compatibility record already exists; write a new record')
    resolved, observations = resolve(job)
    fitted, listed = original(run)
    fitting, selecting, probe = original_rows(fitted, resolved)
    roles = {}
    for member in fitted['memberships']:
        roles.setdefault(member['role'], set()).add(member['id'])
    reconstructed = {**fitted, 'model': original_model(fitted)}
    before = {'fit': identity(fit_inputs(reconstructed, fitting)), 'monitor': identity(monitor_plan(reconstructed, selecting, probe))}
    changed = {kind: sorted(field for field, value in before[kind]['fields'].items() if resolved[kind]['fields'].get(field) != value) for kind in before}
    if changed['fit']:
        raise ConfigError(f"changed numerical fitting inputs {changed['fit']}; the original adapters do not fit the new job: refit")
    accepting = {row['id'] for row in observations if row['role'] == 'acceptance_fit'}
    eligible = not changed['monitor']
    limitations = [] if eligible else [
        f"monitoring or selection plan changed ({', '.join(changed['monitor'])}); the adapter weights remain compatible, but the original shortlist was not "
        'produced by the new selection protocol and cannot be claimed for it; a new fit is needed to reproduce its retained shortlist']
    record = {'schema': SCHEMA, 'kind': KIND, 'outcome': 'eligible' if eligible else 'selection_evidence_required', 'eligible': eligible,
              'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id'],
              'original': {'run': str(run), 'job_id': listed['job_id'], 'fitted_job_id': listed['fitted_job_id'], 'fit_id': before['fit']['id'],
                           'monitor_id': before['monitor']['id'], 'files': {name: files.sha256_file(run / name) for name in RECORDS}},
              'changes': {'fit': [], 'monitor': changed['monitor'], 'acceptance_fit': accepting != roles.get('acceptance_fit', set()),
                          'families': {'original': len({member['component'] for member in fitted['memberships'] if member['role'] != 'unused'}),
                                       'resolved': sum(population['families'] for population in resolved['populations'].values())}},
              'base_sha256': listed['base_sha256'], 'unit': listed['unit'], 'rule': listed['rule'], 'stop': listed['stop'],
              'candidates': listed['candidates'],
              'limitations': limitations,
              'scope': 'provenance, schema and path changes with equal fitting and monitoring inputs preserve candidate eligibility; acceptance-only changes rerun '
                       'policy fitting at selection; historical check exposure stays in its evidence audit ledgers; originals are not modified'}
    record['id'] = artifact_digest(record)
    to.parent.mkdir(parents=True, exist_ok=True)
    files.write_new(to, record)
    return {'record': str(to), 'outcome': record['outcome'], 'eligible': eligible, 'changes': record['changes'], 'limitations': limitations,
            'select': ['semantic-trainer', 'select', str(job), '--run', str(run), '--compatibility', str(to)] if eligible else None}


def accepted(path, run, resolved):
    record = files.read_json(path)
    if not isinstance(record, dict) or record.get('schema') != SCHEMA or record.get('kind') != KIND \
            or record.get('id') != artifact_digest({key: value for key, value in record.items() if key != 'id'}):
        raise ConfigError(f'{path}: not a valid candidate compatibility record; write one with semantic-trainer migrate')
    if not record['eligible']:
        raise ConfigError(f"{path}: {record['outcome']}: {'; '.join(record['limitations'])}")
    if record['job_id'] != resolved['id']:
        raise ConfigError(f'{path}: the record validates another job; migrate against this job')
    if Path(record['original']['run']) != run or any(files.sha256_file(run / name) != sha for name, sha in record['original']['files'].items()):
        raise ConfigError(f'{path}: the original run changed since migration; migrate again')
    return {'schema': CANDIDATES_SCHEMA, 'kind': 'lora-candidates', 'state': 'ready', 'loss': LOSS,
            **{key: record[key] for key in ('job_id', 'fit_id', 'monitor_id', 'base_sha256', 'unit', 'rule', 'stop', 'candidates')}}
