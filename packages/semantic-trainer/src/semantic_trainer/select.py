from __future__ import annotations

import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from semantic_switch import Runtime, files, model_job
from semantic_switch.api import native_settings
from semantic_switch.artifacts import RECEIPT_NAME, RECEIPT_SCHEMA, artifact_digest, load_receipt, native_model
from semantic_switch.errors import ConfigError
from semantic_switch.examples import DATASET_SCHEMA, LabelSource, path_record, record_path
from semantic_switch.metrics import Progress, correctness
from semantic_switch.tools import freeze_specialization, score_observations
from semantic_switch.tools.deployment import comparable, realized
from semantic_switch.tools.evaluate import contrast_pairs

from . import assets
from .adapter import verified
from .deploy import conversion_bytes, deploy, verify_zero
from .jobs import LOSS, ranking, ranking_rule, resolve, unit_correct
from .lora import CANDIDATES_SCHEMA, compatibility, native_bytes, native_predictions, stages
from .migrate import accepted

COMPARED = ('template', 'prompt', 'presentation')
RANKED = ('judge',)


def answer(row):
    prediction = row['result']['prediction']
    return None if prediction is None else prediction['key'] if prediction['kind'] == 'value' else prediction['kind']


def evaluate(resolved, source, rows, job, name):
    with Runtime({'judge': native_model(resolved, source)}, job=job) as runtime:
        switch = runtime.load(resolved['task']['definition'], draft=True)
        progress, scored = Progress(f'select {name}', len(rows), unit='model_select observations'), []
        with runtime.recording() as ledger:
            began = time.perf_counter()
            for row in rows:
                started = time.perf_counter()
                (result,) = score_observations(switch, [row], sources=RANKED)
                scored.append({**result, 'answer': answer(result), 'gold_class': row['classes'][row['label']], 'ms': (time.perf_counter() - started) * 1000})
                progress.record([{**scored[-1], 'outcome': 'failed' if scored[-1]['answer'] is None else 'scored'}])
            progress.finish()
            public = time.perf_counter() - began
            predictions, reuse = native_predictions(runtime, rows, ledger)
            total = time.perf_counter() - began
        work = {'public_seconds': public, 'diagnostic_seconds': total - public, 'total_seconds': total, 'diagnostic_reuse': reuse,
                'scope': 'public: judge-only scoring of the frozen task, whose per-observation latencies give p50_ms; diagnostic: exported-presentation '
                         'HF/native compatibility scoring, reusing a public score block only for an identical token request under this runtime; '
                         'total: both passes, excluding model loading and conversion'}
        return scored, realized(runtime), predictions, work


def summary(name, step, digest, scored, rows, situations, base=None):
    by_id = {row['id']: row for row in rows}
    per_class = {}
    for result in scored:
        counts = per_class.setdefault('/'.join(record_path(result)), {key: {'gold': 0, 'predicted': 0, 'correct': 0} for key in by_id[result['id']]['classes']})
        counts[result['gold_class']]['gold'] += 1
        counts[result['gold_class']]['correct'] += bool(result['correct'])
        if result['answer'] is not None:
            counts[result['answer']]['predicted'] += 1
    families = {}
    for result in scored:
        families.setdefault(by_id[result['id']]['component'], []).append(bool(result['correct']))
    correct = {r['id']: r['correct'] for r in scored}
    pairs = [[correct[i] for i in ids] for ids in contrast_pairs(rows).values() if len(ids) >= 2]
    out = {'name': name, 'step': step, 'base': base is None, 'gguf_sha256': digest, **correctness((r['correct'], r['accepted']) for r in scored),
           'correct': unit_correct({r['id']: r['correct'] for r in scored}, situations, 'model_select'),
           'failures': sum(r['answer'] is None for r in scored), 'per_class': per_class,
           'pairs': {'pairs': len(pairs), 'both_correct': sum(all(values) for values in pairs)},
           'p50_ms': statistics.median(r['ms'] for r in scored), 'rows': scored}
    out |= {'class_regressions': [], 'family_fixes': [], 'family_regressions': []}
    if base is not None:
        before = {family: all(values) for family, values in base['families'].items()}
        after = {family: all(values) for family, values in families.items()}
        out['class_regressions'] = out['regressions'] = sorted(f'{fact}:{key}' for fact, classes in per_class.items() for key, counts in classes.items()
                                                               if counts['correct'] < base['per_class'][fact][key]['correct'])
        out['family_fixes'] = sorted(f for f in after if after[f] and not before[f])
        out['family_regressions'] = sorted(f for f in after if before[f] and not after[f])
    out['families'] = families
    return out


def require_gpu(resolved, trial):
    if resolved['native']['device'] == 'gpu' and (trial['realized']['execution'] or {}).get('device') in (None, 'cpu'):
        raise ConfigError(f"{trial['name']}: GPU native selection was requested but the realized execution is {trial['realized']['execution']}; no CPU fallback")


def homogeneous(trials):
    settings = [comparable(trial['realized'], COMPARED) for trial in trials]
    differing = sorted({key for setting in settings[1:] for key in setting if setting[key] != settings[0][key]})
    if differing:
        raise ConfigError(f"native trials ran under different {', '.join(differing)}; rerun selection under one explicit native configuration")


def membership(rows, uses):
    return [{**{k: row[k] for k in ('id', 'projection', 'group', 'family_seeds', 'component', 'sources')}, **path_record(record_path(row)),
             'label_sources': [LabelSource.parse(source).digest for source in row['label_sources']], 'uses': list(uses)} for row in rows]


def provenance(winner, fit_rows, select_rows, accept_rows, overrides):
    adapted = not winner['base']
    rows = membership(fit_rows, ('adaptation', 'selection') if adapted else ('selection',)) + membership(select_rows, ('selection',)) \
        + membership(accept_rows, ('acceptance_fit',))
    declared = {row['id']: row for row in rows}
    for item in overrides:
        row = declared.get(item['id'])
        if row is None:
            rows.append({**{key: item[key] for key in ('id', 'projection', 'group', 'family_seeds', 'component')}, **path_record(record_path(item)),
                         'sources': [], 'label_sources': list(item['label_sources']), 'uses': ['override']})
        else:
            row['uses'] = [*row['uses'], 'override']
            row['label_sources'] = list(dict.fromkeys([*row['label_sources'], *item['label_sources']]))
    return {'kind': 'specialization' if adapted else 'base', 'rows': rows}


def link(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def select(job, run, compatible=None):
    resolved, observations = resolve(job)
    run = Path(run).resolve()
    fitted, state = files.read_json(run / 'resolved-job.json'), files.read_json(run / 'run.json')
    if not isinstance(fitted, dict) or (state or {}).get('state') != 'ready':
        raise ConfigError(f'{run}: not a ready fitted run (state {(state or {}).get("state")}); fit with semantic-trainer lora {job} --to NEW_RUN')
    listed = files.read_json(run / 'candidates.json') if compatible is None else accepted(compatible, run, resolved)
    if not isinstance(listed, dict) or listed.get('schema') != CANDIDATES_SCHEMA or listed.get('kind') != 'lora-candidates' or listed.get('state') != 'ready':
        raise ConfigError(f'{run}: unsupported candidate list (current schema {CANDIDATES_SCHEMA}); refit with semantic-trainer lora, '
                          'or validate a run of the preceding schema with semantic-trainer migrate')
    if listed.get('loss') != resolved['loss'] or resolved['loss'] != LOSS:
        raise ConfigError(f"{run}: candidates were fitted under training objective {listed.get('loss')}, not the prepared job's {resolved['loss']}; "
                          'refit under a new job; fit-derived selection claims cannot be relabelled')
    if listed['job_id'] != resolved['id']:
        raise ConfigError(f'{job}: the job, its task, exports or assets changed since {run} was fitted; refit or select with the original job')
    if listed['fit_id'] != resolved['fit']['id'] or listed['monitor_id'] != resolved['monitor']['id']:
        raise ConfigError(f'{run}: candidates were fitted from other numerical inputs or under another monitoring plan; refit')
    quant = resolved['assets']['quant']
    located = assets.load(resolved['assets']['manifest'])
    base = located['gguf'][quant]
    if listed['base_sha256'] != base['sha256']:
        raise ConfigError(f'{run}: candidates were fitted against base {listed["base_sha256"]}, not the pipeline-built {quant} base {base["sha256"]}; refit')
    for candidate in listed['candidates']:
        verified(fitted, run / candidate['peft'], candidate)
    output = run / 'deployment'
    roles = {role: [row for row in observations if row['role'] == role] for role in ('fit', 'model_select', 'acceptance_fit')}
    rows, unit = roles['model_select'], resolved['recipe']['selection_unit']
    if not rows:
        raise ConfigError('selection requires model_select families; prepare the job with semantic-trainer prepare')
    finalists = [candidate for candidate in listed['candidates'] if candidate['step']]
    zero = next((candidate for candidate in listed['candidates'] if candidate['name'] == 'zero'), None)
    if zero is None or not finalists:
        raise ConfigError(f'{run}: candidate list needs the zero adapter and at least one retained step; refit')
    with files.Lease(output), model_job(extra=stages(conversion_bytes(located), native_bytes(native_model(resolved)))) as handle:
        if output.exists():
            raise ConfigError(f'{output}: already selected; inspect its receipt or select a new run')
        work = Path(tempfile.mkdtemp(prefix='.select-', dir=run))
        staged = Path(tempfile.mkdtemp(prefix='.deployment-', dir=run))
        try:
            verify_zero(fitted, located, run / zero['peft'], quant, work, handle)
            deployed = {'base': (0, base['path'], base['sha256'], None)}
            for candidate in finalists:
                path = work / f"{candidate['name']}.gguf"
                deployed[candidate['name']] = (candidate['step'], str(path), deploy(fitted, located, run / candidate['peft'], path, quant, handle), candidate)
            trials, natives = [], {}
            for name, (step, source, digest, candidate) in deployed.items():
                scored, realized_, natives[name], work_ = evaluate(resolved, source, rows, handle, name)
                if realized_['identity']['sha256'] != digest:
                    raise ConfigError(f'{source}: scored model differs from the deployed {name} artifact {digest}')
                trial = {**summary(name, step, digest, scored, rows, resolved['situations'], trials[0] if trials else None), 'realized': realized_, 'selection_work': work_}
                require_gpu(resolved, trial)
                trials.append({**trial, **({'hf_agreement': compatibility(rows, (natives['base'], zero['predictions']), (natives[name], candidate['predictions']))}
                                           if candidate else {})})
            homogeneous(trials)
            winner = ranking(trials, unit)[0]
            _, source, digest, _ = deployed[winner['name']]
            history = provenance(winner, roles['fit'], rows, roles['acceptance_fit'], resolved['corrections'])
            published = len({row['component'] for row in history['rows'] if row['uses'] != ['override']})
            if published != sum(population['families'] for population in resolved['populations'].values()):
                raise ConfigError(f'{published} published canonical families differ from the resolved populations {resolved["populations"]}; prepare a new job')
            population = {'schema': DATASET_SCHEMA, 'kind': 'specialization-population', 'job_id': resolved['id'],
                          'exports': sorted({source['export_id'] for source in resolved['sources']})}
            frozen = freeze_specialization(native_model(resolved, source), resolved['task'], roles['acceptance_fit'],
                                           history['rows'], population, winner['realized'], staged, job=handle)
            link(source, staged / 'specialist.gguf')
            if files.sha256_file(staged / 'specialist.gguf') != digest:
                raise ConfigError('deployed specialist changed during selection')
            report = {'rule': ranking_rule(trials, unit), 'unit': unit, 'loss': resolved['loss'], 'selected': winner['name'], 'populations': resolved['populations'],
                      'cohort': resolved['cohort'], 'sources': list(RANKED),
                      'scope': 'chosen within the retained shortlist and base, not every discarded training point; ranked on judge answers of the frozen task and guidance, '
                               'without exact corrections or fitted acceptance policies, which the receipt behavior and the independent check cover',
                      'fit_stop': listed['stop'], 'candidates': [{k: v for k, v in t.items() if k != 'families'} for t in trials]}
            receipt = {'schema': RECEIPT_SCHEMA, 'kind': 'specialization-receipt', 'selected': winner['name'], 'gguf': 'specialist.gguf', 'task': frozen['task'],
                       'lock': frozen['lock'], 'files': {'specialist.gguf': digest, **frozen['files']}, 'freeze': {'revision': frozen['revision']},
                       'base': {'manifest': located['manifest'], 'quant': quant, 'sha256': base['sha256'], 'repo': located['repo'], 'commit': located['commit']},
                       'provenance': history, 'native': native_settings(native_model(resolved)), 'execution': winner['realized']['execution'] or {},
                       'judge': {key: winner['realized']['identity'][key] for key in COMPARED},
                       'definition': {'source': resolved['task']['source'], 'sha256': resolved['task']['sha256'], 'revisions': resolved['task']['revisions']},
                       'behavior': frozen['behavior'],
                       'recipe': resolved['recipe'], 'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id'],
                       'selection': report, 'acceptance': frozen['acceptance']}
            receipt['id'] = artifact_digest(receipt)
            files.write_new(staged / RECEIPT_NAME, receipt)
            load_receipt(staged / RECEIPT_NAME)
            files.publish_directory(staged, output)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(staged, ignore_errors=True)
    path = output / RECEIPT_NAME
    return {'receipt': str(path), 'gguf': str(output / 'specialist.gguf'), 'selected': winner['name'], 'unit': unit,
            'candidates': {t['name']: {k: t[k] for k in ('correct', 'raw_correct', 'submitted', 'failures', 'class_regressions', 'p50_ms', 'selection_work')} for t in trials},
            'loading': {'deployment': f'Runtime.from_specialization({str(path)!r}).load()', 'model_only': f'Model.from_specialization({str(path)!r})'},
            'check': ['semantic-trainer', 'check', str(path), 'LABELS.jsonl', '--evidence', 'DATASET_MANIFEST', '--to', 'NEW_CHECK_DIRECTORY']}
