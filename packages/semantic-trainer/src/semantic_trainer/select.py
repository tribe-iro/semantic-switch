from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import NamedTuple

from semantic_switch import ConfigError
from semantic_switch.training import (per_class as reduce_classes, regressions, DATASET_SCHEMA, RECEIPT_NAME, RECEIPT_SCHEMA, Journal, LabelSource, Progress, Selection, artifact_digest, comparable,
                                      contrast_pairs, correctness, files, freeze_specialization, load_receipt, model_job, native_model, native_score_requests,
                                      membership_record, native_settings, path_record, probabilities, publication, realized, record_path, score_observations, trial)

from . import assets, jobs
from .adapter import verified
from .deploy import conversion_bytes, representation_of, verify_runtime_zero, verify_zero
from .jobs import (SELECTION_KIND, SELECTION_SCHEMA, inheritance, merged_uses, portable, ranking, ranking_rule, ready_run, resolve, single_task,
                   unit_correct)
from .lora import compatibility, native_predictions
from .monitor import probability_entries
from .representation import FILES, deployable, realized_binding, residency

COMPARED = ('template', 'prompt', 'presentation')
RANKED = ('judge',)
INITIALIZER = ('kind', 'evidence', 'candidate', 'step', 'job_id', 'fit_id', 'fitted_job_id', 'peft_sha256', 'config_sha256')


def observation_metadata(row):
    seeds = [*row['family_seeds'], *(node for origin in row['sources'] for node in origin['family_seeds'])]
    return {'input': row['bound_input'], 'meaning_rev': row['meaning_rev'], 'classes': row['classes'],
            'label_sources': row['label_sources'], 'role': row['role'], 'task': row['task'], 'family': row['component'],
            'groups': sorted({row['component'], *(value for kind, value in seeds if kind == 'group')}),
            'tags': row.get('tags', []), 'pairs': sorted({*row.get('pairs', []), *(value for kind, value in seeds if kind == 'pair')})}


def evaluate(runtime, switch, rows, name, role='judge', journal=None):
    keys = {row['id']: artifact_digest(['public', row['id']]) for row in rows}
    cached = {row['id']: journal.payload(keys[row['id']]) for row in rows if journal is not None and keys[row['id']] in journal.current}
    progress = Progress(f'select {name}', len(rows), unit='model_select observations', resumed=cached.values())
    scored, pending, began = [], [], time.perf_counter()
    for row in rows:
        metadata = observation_metadata(row)
        if row['id'] in cached:
            scored.append({**cached[row['id']], **metadata})
            continue
        started = time.perf_counter()
        (result,) = score_observations(switch, [row], sources=RANKED)
        compact = {key: value for key, value in result.items() if key != 'result'}
        compact |= {**metadata, 'gold_class': row['classes'][row['label']], 'ms': (time.perf_counter() - started) * 1000,
                    'outcome': result['result']['status']}
        if journal is not None:
            pending.append((keys[row['id']], compact))
            if len(pending) >= 16:
                journal.commit(pending)
                pending.clear()
        scored.append(compact)
        progress.record([{**compact, 'outcome': 'failed' if compact['answer'] is None else 'scored'}])
    if journal is not None and pending:
        journal.commit(pending)
    progress.finish()
    public_key = artifact_digest(['public_seconds'])
    public = journal.payload(public_key) if journal is not None and public_key in journal.current else time.perf_counter() - began + sum(row['ms'] / 1000 for row in cached.values())
    if journal is not None and public_key not in journal.current:
        journal.commit([(public_key, public)])
    native_keys = {row['id']: artifact_digest(['native', row['id']]) for row in rows}
    predictions = {row['id']: journal.payload(native_keys[row['id']]) for row in rows if journal is not None and native_keys[row['id']] in journal.current}
    missing = [row for row in rows if row['id'] not in predictions]
    diagnostic_key = artifact_digest(['diagnostic_work'])
    if missing:
        started = time.perf_counter()
        fresh, reuse = native_predictions(runtime, missing, replay=True, role=role)
        diagnostic = time.perf_counter() - started
        predictions.update(fresh)
        if journal is not None:
            journal.commit([(native_keys[row['id']], fresh[row['id']]) for row in missing])
            journal.commit([(diagnostic_key, {'seconds': diagnostic, 'reuse': reuse})])
    elif journal is not None and diagnostic_key in journal.current:
        diagnostic_record = journal.payload(diagnostic_key)
        diagnostic, reuse = diagnostic_record['seconds'], diagnostic_record['reuse']
    else:
        diagnostic, reuse = 0., {'requests': 0, 'rows': 0, 'scored': 0, 'reused': 0, 'scored_rows': 0}
    work = {'public_seconds': public, 'diagnostic_seconds': diagnostic, 'total_seconds': public + diagnostic, 'diagnostic_reuse': reuse,
            'reused_public_rows': len(cached), 'reused_native_rows': len(rows) - len(missing), 'invocation_seconds': time.perf_counter() - began}
    return scored, realized(runtime, role), predictions, work


def trial_models(representation, resolved, deployed):
    bound = max(1, sum(entry['lora'] is not None for _, entry, _ in deployed.values())) if representation == 'runtime-adapter' else 1
    return {name: replace(native_model(resolved, entry['source'], lora=entry['lora']), max_adapters=bound) for name, (_, entry, _) in deployed.items()}


def evaluations(representation, definition, targets, models, rows, job, *, journal_root=None, job_id=None, baseline=None):
    population = native_score_requests(rows)
    def measured(runtime, name, role):
        current = realized(runtime, role)
        if name == 'base' and baseline is not None and current == baseline['realized']:
            if [row['id'] for row in baseline['scored']] != [row['id'] for row in rows]:
                raise ConfigError('baseline observation order changed')
            return [{**scored, **observation_metadata(row)} for row, scored in zip(rows, baseline['scored'])], current, baseline['predictions'], {**baseline['work'], 'invocation_seconds': 0., 'reused_baseline': True}
        config = {'job': job_id, 'representation': representation, 'model': current, 'rows': [row['id'] for row in rows]}
        held = Journal(Path(journal_root) / f'{name}.jsonl', config) if journal_root is not None else nullcontext(None)
        with held as journal:
            return evaluate(runtime, trial(runtime, definition, targets, role), rows, name, role, journal)
    if representation == 'runtime-adapter':
        roles = {name: 'judge' if name == 'base' else name for name in models}
        with Selection({roles[name]: model for name, model in models.items()}, population, job=job) as runtime:
            for name in models:
                yield name, measured(runtime, name, roles[name])
        return
    for name, model in models.items():
        with Selection(model, population, job=job) as runtime:
            yield name, measured(runtime, name, 'judge')


def summary(name, step, digest, scored, rows, situations, base=None):
    by_id = {row['id']: row for row in rows}
    scoped = len({row['task'] for row in rows}) > 1
    per_class = {f'{task}:{target}' if scoped else target: counts for (task, target), counts in reduce_classes(
        ((by_id[result['id']]['task'], '/'.join(record_path(result))), by_id[result['id']]['classes'], result['gold_class'], result['answer'], result['correct'])
        for result in scored).items()}
    families = {}
    for result in scored:
        families.setdefault(by_id[result['id']]['component'], []).append(bool(result['correct']))
    correct = {r['id']: r['correct'] for r in scored}
    pairs = [[correct[i] for i in ids] for ids in contrast_pairs(rows).values() if len(ids) >= 2]
    out = {'name': name, 'step': step, 'base': base is None, 'gguf_sha256': digest, **correctness((r['correct'], r['accepted']) for r in scored),
           'correct': unit_correct({r['id']: r['correct'] for r in scored}, situations, 'model_select', {row['id']: row['task'] for row in rows}),
           'failures': sum(r['answer'] is None for r in scored), 'per_class': per_class,
           'pairs': {'pairs': len(pairs), 'both_correct': sum(all(values) for values in pairs)},
           'p50_ms': statistics.median(r['ms'] for r in scored), 'rows': scored}
    out |= {'class_regressions': [], 'family_fixes': [], 'family_regressions': []}
    if base is not None:
        before = {family: all(values) for family, values in base['families'].items()}
        after = {family: all(values) for family, values in families.items()}
        out['class_regressions'] = out['regressions'] = [f'{fact}:{key}' for fact, key in regressions(per_class, base['per_class'])]
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


def provenance(winner, fit_rows, select_rows, accept_rows, overrides, inherited):
    adapted = bool(winner['step'])
    rows = merged_uses(membership(fit_rows, ('adaptation', 'selection') if adapted else ('selection',)) + membership(select_rows, ('selection',))
                       + membership(accept_rows, ('acceptance_fit',)) + ([] if winner['base'] else list(inherited)))
    declared = {row['id']: row for row in rows}
    for item in overrides:
        row = declared.get(item['id'])
        if row is None:
            rows.append({**{key: item[key] for key in ('id', 'projection', 'group', 'family_seeds', 'component')}, **path_record(record_path(item)),
                         'sources': [], 'label_sources': list(item['label_sources']), 'uses': ['override']})
        else:
            row['uses'] = [*row['uses'], 'override']
            row['label_sources'] = list(dict.fromkeys([*row['label_sources'], *item['label_sources']]))
    return {'kind': 'specialization' if any('adaptation' in row['uses'] for row in rows) else 'base', 'rows': rows}


def changed(fitted, resolved, kind):
    return sorted(field for field, value in resolved[kind]['fields'].items() if fitted[kind]['fields'].get(field) != value)


def admitted(job, run, resolved, base, audited=()):
    refit = f'semantic-trainer lora {job} --to NEW_RUN'
    _, listed, fitted = ready_run(run, refit)
    if changed(fitted, resolved, 'fit'):
        raise ConfigError(f"{run}: changed numerical fitting inputs {changed(fitted, resolved, 'fit')}; the candidates do not fit this job: refit with {refit}")
    if changed(fitted, resolved, 'monitor'):
        raise ConfigError(f"{run}: monitoring or selection plan changed ({', '.join(changed(fitted, resolved, 'monitor'))}); the shortlist was not produced "
                          f'under this protocol: refit with {refit}')
    described = [artifact_digest(portable({key: record.get(key) for key in resolved if key not in ('id', 'population', *audited)})) for record in (fitted, resolved)]
    if fitted['id'] != resolved['id'] and not (audited and described[0] == described[1]):
        raise ConfigError(f'{job}: the job, its task, exports or assets changed since {run} was fitted; refit or select with the original job')
    if described[0] != described[1]:
        raise ConfigError(f"{run}: resolved-job.json does not describe job {resolved['id']}; restore the original fitted record or refit with {refit}")
    if listed['base_sha256'] != base['sha256']:
        raise ConfigError(f"{run}: candidates were fitted against base {listed['base_sha256']}, not the pipeline-built {resolved['assets']['quant']} base {base['sha256']}; refit")
    for candidate in listed['candidates']:
        verified(fitted, run / candidate['peft'], candidate)
    return fitted, listed


def shortlist(run, listed, warm):
    finalists = [candidate for candidate in listed['candidates'] if candidate['step'] or warm]
    zero = next((candidate for candidate in listed['candidates'] if candidate['name'] == 'zero'), None)
    if zero is None or not finalists:
        raise ConfigError(f'{run}: candidate list needs the zero adapter and at least one retained step; refit')
    return finalists, zero


def native_device(resolved):
    if resolved['native']['device'] == 'gpu':
        from semantic_switch.training import gpu_available
        if not gpu_available():
            raise ConfigError('GPU native selection was requested but the installed llama.cpp offers no GPU offload; no CPU fallback')


def proven(representation, fitted, located, null, quant, work, handle, base_sha256):
    peft = Path(null['run']) / null['peft']
    verified(fitted, peft, null)
    if representation == 'merged':
        return verify_zero(fitted, located, peft, quant, work, handle)
    return verify_runtime_zero(fitted, located, peft, quant, base_sha256)


def converted(representation, resolved, fitted, located, run, finalists, zero, work, handle, base_sha256):
    quant = resolved['assets']['quant']
    base = located['gguf'][quant]
    null = resolved['initialization']['null'] if resolved['initialization']['kind'] == 'adapter' else {'run': run, **zero}
    proven(representation, fitted, located, null, quant, work, handle, base_sha256)
    deployed = {'base': (0, {'source': base['path'], 'sha256': base['sha256'], 'lora': None, 'adapter': None}, None)}
    for candidate in finalists:
        deployed[candidate['name']] = (candidate['step'], deployable(representation, fitted, located, run / candidate['peft'], work, candidate['name'], quant, handle),
                                       candidate)
    return deployed


def trial_of(name, deployed, scored, realized_, work_, rows, resolved, trials, natives, zero):
    step, entry, candidate = deployed[name]
    for result in scored:
        result['probability'] = natives[name][result['id']]['probability']
    realized_binding(entry, realized_)
    trial = {**summary(name, step, entry['sha256'], scored, rows, resolved['situations'], trials[0] if trials else None), 'realized': realized_, 'selection_work': work_,
             'probabilities': {**probabilities(probability_entries(rows, natives[name]), targets=False),
                               'population': 'exported model_select presentations scored natively; selection diagnostics, never independent quality'}}
    if required := resolved['recipe'].get('development_recall'):
        trial['qualification'] = jobs.development_recall(rows, {row['id']: row for row in scored}, required)
    require_gpu(resolved, trial)
    if candidate is None:
        return trial
    reference = natives['zero' if resolved['initialization']['kind'] == 'adapter' else 'base']
    tasks = sorted({row['task'] for row in rows})
    agreement = {task: compatibility(members, (reference, zero['predictions']), (natives[name], candidate['predictions']))
                 for task in tasks for members in ([row for row in rows if row['task'] == task],)}
    return {**trial, 'hf_agreement': agreement[tasks[0]] if len(tasks) == 1 else {'tasks': agreement}}


class Admitted(NamedTuple):
    resolved: dict
    observations: list
    representation: str
    run: Path
    located: dict
    base: dict
    fitted: dict
    listed: dict
    finalists: list
    zero: dict
    rows: list
    unit: str


def admission(resolved, observations, job, run, representation, audited=()):
    representation = representation_of(representation)
    native_device(resolved)
    run = Path(run).resolve()
    located = assets.load(resolved['assets']['manifest'])
    base = located['gguf'][resolved['assets']['quant']]
    fitted, listed = admitted(job, run, resolved, base, audited)
    rows = [row for row in observations if row['role'] == 'model_select']
    if not rows:
        raise ConfigError('selection requires model_select families; prepare the job with semantic-trainer prepare')
    finalists, zero = shortlist(run, listed, resolved['initialization']['kind'] == 'adapter')
    return Admitted(resolved, observations, representation, run, located, base, fitted, listed, finalists, zero, rows, resolved['recipe']['selection_unit'])


@contextmanager
def publishing(output, parent, prefix):
    with files.Lease(output):
        if output.exists():
            raise ConfigError(f'{output}: already selected; inspect its record or select a new run')
        parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='.select-', dir=parent))
        staged = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        try:
            yield work, staged
            files.publish_directory(staged, output)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            shutil.rmtree(staged, ignore_errors=True)


@contextmanager
def trialled(chosen, work, evaluate, shared):
    extra = conversion_bytes(chosen.located) if chosen.representation == 'merged' else {}
    with model_job(extra=extra) as handle:
        deployed = converted(chosen.representation, chosen.resolved, chosen.fitted, chosen.located, chosen.run, chosen.finalists, chosen.zero, work, handle,
                             chosen.listed['base_sha256'])
    models = trial_models(chosen.representation, chosen.resolved, deployed)
    with model_job(extra=residency(models.values(), shared)) as handle:
        trials, natives = [], {}
        for name, (scored, realized_, natives[name], work_) in evaluate(deployed, models, handle):
            trials.append(trial_of(name, deployed, scored, realized_, work_, chosen.rows, chosen.resolved, trials, natives, chosen.zero))
        homogeneous(trials)
        yield deployed, trials, ranking(trials, chosen.unit)[0], handle



def selection_rows(directory, trials):
    path = Path(directory) / 'selection.rows.jsonl'
    with path.open('wb') as stream:
        for trial_ in trials:
            for row in trial_['rows']:
                stream.write(files.jsonl([{'candidate': trial_['name'], **row}]))
    return {'file': path.name, 'sha256': files.sha256_file(path), 'rows': sum(len(trial_['rows']) for trial_ in trials)}

def outcomes(trials):
    return {t['name']: {k: t[k] for k in ('correct', 'raw_correct', 'submitted', 'failures', 'class_regressions', 'p50_ms', 'selection_work', 'qualification') if k in t} for t in trials}


def select(job, run, representation='merged', to=None, bindings=None):
    return application(*resolve(job), job, run, representation, to=to, bindings=bindings)


def application(resolved, observations, job, run, representation='merged', to=None, bindings=None, baseline=None):
    chosen = admission(resolved, observations, job, run, representation)
    baseline_path = baseline
    if baseline_path is not None:
        from .controls import load, native_result
        baseline = native_result(load(baseline_path, resolved, observations), baseline_path, observations)
    contract, task = single_task(resolved)
    scope = resolved['scope'][contract]
    composed = publication(task['definition'], scope, bindings)
    roles = {role: [row for row in observations if row['role'] == role] for role in ('fit', 'acceptance_fit')}
    output = chosen.run / 'deployment' if to is None else Path(to).resolve()
    representation, rows, unit, located, base = chosen.representation, chosen.rows, chosen.unit, chosen.located, chosen.base
    def evaluate(deployed, models, handle):
        return evaluations(representation, task['definition'], scope['targets'], models, rows, handle,
                           journal_root=chosen.run / 'native-selection' / representation, job_id=resolved['id'], baseline=baseline)
    with publishing(output, output.parent, '.deployment-') as (work, staged), \
            trialled(chosen, work, evaluate, representation == 'runtime-adapter') as (deployed, trials, winner, handle):
        _, entry, _ = deployed[winner['name']]
        source, digest = entry['source'], entry['sha256']
        weights, adapter_file = FILES[representation] if entry['lora'] else (FILES['merged'][0], None)
        history = provenance(winner, roles['fit'], rows, roles['acceptance_fit'], resolved['corrections'], resolved['initialization'].get('inherited', ()))
        published = len({row['component'] for row in history['rows'] if row['uses'] != ['override'] and 'inherited' not in row})
        if published != sum(population['families'] for population in resolved['populations'].values()):
            raise ConfigError(f'{published} published canonical families differ from the resolved populations {resolved["populations"]}; prepare a new job')
        population = {'schema': DATASET_SCHEMA, 'kind': 'specialization-population', 'job_id': resolved['id'],
                      'exports': sorted({source['export_id'] for source in resolved['sources']})}
        deployed_model = native_model(resolved, source, lora=entry['lora'])
        if not (entry['lora'] or winner['base']):
            deployed_model = replace(deployed_model, representation='merged')
        frozen = freeze_specialization(deployed_model, task, roles['acceptance_fit'], history['rows'], population, winner['realized'], staged,
                                       job=handle, scope=scope, bindings=bindings)
        assets.link(source, staged / weights)
        listed_files = {weights: digest}
        if adapter_file:
            assets.link(entry['lora'], staged / adapter_file)
            listed_files[adapter_file] = entry['adapter']['sha256']
        if any(files.sha256_file(staged / name) != sha for name, sha in listed_files.items()):
            raise ConfigError('deployed specialist changed during selection')
        row_record = selection_rows(staged, trials)
        report = {'rule': ranking_rule(trials, unit), 'unit': unit, 'loss': resolved['loss'], 'selected': winner['name'], 'populations': resolved['populations'],
                  'cohort': resolved['cohort'], 'sources': list(RANKED), 'fit_stop': chosen.listed['stop'],
                  'rows': row_record, 'candidates': [{k: v for k, v in t.items() if k not in ('families', 'rows')} for t in trials]}
        receipt = {'schema': RECEIPT_SCHEMA, 'kind': 'specialization-receipt', 'selected': winner['name'], 'gguf': weights, **({'lora': adapter_file} if adapter_file else {}),
                   'task': frozen['task'], 'scope': scope, 'publication': frozen['publication'],
                   'lock': frozen['lock'], 'files': {**listed_files, **frozen['files'], row_record['file']: row_record['sha256']}, 'freeze': {'revision': frozen['revision']},
                   'base': {'manifest': located['manifest'], 'quant': resolved['assets']['quant'], 'sha256': base['sha256'], 'repo': located['repo'], 'commit': located['commit']},
                   'provenance': {'kind': history['kind'], **membership_record(history['rows'])}, 'native': native_settings(native_model(resolved)), 'execution': winner['realized']['execution'] or {},
                   'judge': {key: winner['realized']['identity'][key] for key in COMPARED},
                   'definition': {'source': task['source'], 'sha256': task['sha256'], 'revisions': task['revisions']},
                   'behavior': frozen['behavior'],
                   'initialization': {key: resolved['initialization'][key] for key in INITIALIZER if key in resolved['initialization']},
                   'recipe': resolved['recipe'], 'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id'],
                   'selection': report, 'acceptance': frozen['acceptance']}
        from .assessment import publish
        learning = publish(staged, resolved, observations, report, frozen['acceptance'], chosen.run, baseline_path)
        receipt['learning'] = {'file': 'learning.json', 'sha256': files.sha256_file(staged / 'learning.json')}
        receipt['files']['learning.json'] = receipt['learning']['sha256']
        receipt['id'] = artifact_digest(receipt)
        files.write_new(staged / RECEIPT_NAME, receipt)
        load_receipt(staged / RECEIPT_NAME)
    path = output / RECEIPT_NAME
    return {'receipt': str(path), 'gguf': str(output / weights), **({'lora': str(output / adapter_file)} if adapter_file else {}), 'selected': winner['name'], 'unit': unit,
            'representation': deployed_model.representation, 'candidates': outcomes(trials),
            'learning': {**learning['summary'], 'path': str(output / 'learning.json')},
            'publication': composed['publication'], 'binding': composed['binding'],
            'loading': {'deployment': f'Runtime({"bindings={...}" if bindings else ""}).load({str(output)!r})'
                        if composed['publication'] == 'application' else None,
                        'model_only': f'ss.judge({str(output)!r})',
                        'binding': f"Runtime(bindings={{{composed['binding']!r}: ss.judge({str(output)!r}), ...}}).load(APPLICATION)"},
            'check': ['semantic-trainer', 'check', str(path), 'LABELS.jsonl', '--evidence', 'DATASET_MANIFEST', '--to', 'NEW_CHECK_DIRECTORY']}


def task_evaluations(resolved, deployed, rows, job):
    tasks, groups = {}, {}
    for row in rows:
        tasks.setdefault(row['task'], []).append(row)
    for contract in sorted(tasks):
        groups.setdefault(resolved['scope'][contract]['binding'] or 'judge', []).append(contract)
    for name, (_, entry, _) in deployed.items():
        model = native_model(resolved, entry['source'], lora=entry['lora'])
        if entry['lora']:
            model = replace(model, max_adapters=1)
        results = {}
        for binding, contracts in sorted(groups.items()):
            with Selection({binding: model}, native_score_requests([row for contract in contracts for row in tasks[contract]]), job=job) as runtime:
                for contract in contracts:
                    switch = trial(runtime, resolved['tasks'][contract]['definition'], resolved['scope'][contract]['targets'], binding)
                    results[contract] = evaluate(runtime, switch, tasks[contract], f'{name} {contract[:12]}', binding)
        realized_ = [found for _, found, _, _ in results.values()]
        homogeneous([{'realized': found} for found in realized_])
        work = {key: sum(results[contract][3][key] for contract in results) for key in ('public_seconds', 'diagnostic_seconds', 'total_seconds')}
        yield name, ([scored for contract in sorted(results) for scored in results[contract][0]], realized_[0],
                     {key: value for contract in sorted(results) for key, value in results[contract][2].items()},
                     {**work, 'tasks': {contract: results[contract][3] for contract in sorted(results)}})


def select_initializer(job, run, representation='merged'):
    resolved, observations = resolve(job)
    chosen = admission(resolved, observations, job, run, representation, audited=('mixture',))
    representation, rows, unit, located, run = chosen.representation, chosen.rows, chosen.unit, chosen.located, chosen.run
    output = run / 'initializer-selection'
    def evaluate(deployed, models, handle):
        return task_evaluations(resolved, deployed, rows, handle)
    with publishing(output, run, '.initializer-selection-') as (work, staged), trialled(chosen, work, evaluate, False) as (deployed, trials, winner, _):
        _, entry, candidate = deployed[winner['name']]
        memberships, overrides = inheritance(chosen.fitted)
        row_record = selection_rows(staged, trials)
        record = {'schema': SELECTION_SCHEMA, 'kind': SELECTION_KIND, 'purpose': 'initializer', 'run': str(run), 'job_id': chosen.fitted['id'],
                  'fitted_job_id': chosen.listed['fitted_job_id'], 'fit_id': chosen.fitted['fit']['id'], 'monitor_id': chosen.fitted['monitor']['id'],
                  'selection_job_id': resolved['id'], 'mixture': resolved.get('mixture'), 'fitted_mixture': chosen.fitted.get('mixture'),
                  'selected': {'name': winner['name'], 'step': winner['step'],
                               **({key: candidate[key] for key in ('peft', 'peft_sha256', 'config_sha256', 'config_digest')} if candidate else {})},
                  'representation': representation, 'base': {'manifest': located['manifest'], 'quant': resolved['assets']['quant'], 'sha256': chosen.base['sha256']},
                  'native': {'sha256': entry['sha256'], 'adapter': entry['adapter'], 'execution': winner['realized']['execution'] or {},
                             'judge': {key: winner['realized']['identity'][key] for key in COMPARED}, 'settings': native_settings(native_model(resolved))},
                  'plan': {'unit': unit, 'rule': ranking_rule(trials, unit), 'sources': list(RANKED), 'population': resolved['populations']['model_select'],
                           'tasks': {contract: {'binding': resolved['scope'][contract]['binding'], 'sha256': resolved['tasks'][contract]['sha256'],
                                                'revisions': resolved['tasks'][contract]['revisions'],
                                                'observations': sum(row['task'] == contract for row in rows)}
                                     for contract in sorted({row['task'] for row in rows})}},
                  'fit_stop': chosen.listed['stop'], 'rows': row_record, 'candidates': [{k: v for k, v in t.items() if k not in ('families', 'rows')} for t in trials],
                  'memberships': memberships, 'overrides': overrides}
        record['id'] = artifact_digest(record)
        files.write_new(staged / 'record.json', record)
    path = output / 'record.json'
    return {'record': str(path), 'selected': winner['name'], 'unit': unit, 'representation': representation, 'candidates': outcomes(trials),
            'warm_start': None if winner['base'] else ['semantic-trainer', 'prepare', 'DEFINITION', 'EXPORTS...', '--init-selection', str(path), '--to', 'NEW_JOB']}
