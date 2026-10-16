from __future__ import annotations

from pathlib import Path
import statistics

from semantic_switch import ConfigError
from semantic_switch.training import artifact_digest, canonical_answer, files, load_receipt, record_path

from . import jobs


def variants(recipe):
    rate = recipe['learning_rate']
    return [{'name': f'seed-{seed}-lr-{scale:g}', 'seed': seed, 'learning_rate': rate * scale}
            for seed, scale in ((0, 1.), (1, 1.), (2, 1.), (0, .5), (0, 2.))]


def write_job(source, destination, overrides, *, selection=None, exports=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    declared = files.read_yaml(source)
    job = jobs.population_job(declared, source.parent)
    job['recipe'] = {**job.get('recipe', {}), jobs.CADENCE: jobs.cadence_of(declared.get('recipe')), **overrides}
    if selection is not None:
        job['selection'] = selection
    if exports is not None:
        job['exports'] = exports
    if destination.exists():
        if files.read_yaml(destination) != job:
            raise ConfigError(f'{destination}: experiment inputs changed; use another destination')
    else:
        files.publish(destination, job)
    return job


def curve(run):
    path = Path(run) / 'progress.jsonl'
    return [{'step': point['step'], 'examples_seen': point['examples_seen'], 'nll': point['model_select']['nll'],
             'correct': point['model_select']['correct'], 'per_class': point['model_select']['per_class'],
             'qualification': point['model_select'].get('qualification'), 'class_changes': point['model_select'].get('class_changes'),
             'probe': point['probe']} for point in files.iter_records(path) if point['kind'] == 'evaluation'] if path.exists() else []


def completed(directory, *, run=None, receipt=None):
    directory = Path(directory)
    run = Path(run) if run is not None else directory / 'run'
    receipt_path = Path(receipt) if receipt is not None else run / 'deployment' / 'receipt.json'
    published = files.read_json(receipt_path)
    state = files.read_json(run / 'run.json') or {}
    points = curve(run)
    if published is None:
        return {'state': state.get('state', 'not_started'), 'curve': points, 'stop': state.get('stop')}
    report = published['selection']
    candidates = report.get('candidates', report.get('trials', []))
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    name = report.get('selected', report.get('winner'))
    name = name.get('name') if isinstance(name, dict) else name
    winner = next((row for row in candidates if row.get('name') == name), None)
    fitted = files.read_json(run / 'resolved-job.json') or {}
    limited = (state.get('stop') or {}).get('reason') in ('time_budget', 'token_budget', 'monitor_budget')
    return {'state': 'budget_limited' if limited else 'completed', 'receipt': str(receipt_path), 'selected': name,
            'selection': report, 'winner': winner, 'curve': points, 'stop': state.get('stop'),
            'environment': {'dependencies': fitted.get('dependencies'), 'fitting': fitted.get('fitting'), 'execution': published.get('execution')},
            'operational_actions': {'state': 'unassessed', 'reason': 'target_only_evaluation'}}


def spread(values):
    values = [value for value in values if value is not None]
    return {'runs': len(values), 'worst': min(values) if values else None, 'median': statistics.median(values) if values else None,
            'range': [min(values), max(values)] if values else None}


def summarize(records):
    completed_ = [row for row in records if row.get('state') == 'completed']
    metrics, classes = {}, {}
    for row in completed_:
        winner = row.get('winner')
        if winner is None:
            continue
        for unit, value in winner['correct']['macro'].items():
            metrics.setdefault(unit, []).append(value)
        for key in ('precision', 'coverage', 'correct_accepted_rate'):
            metrics.setdefault(key, []).append(winner.get(key))
        for scope, answers in winner.get('per_class', {}).items():
            for name, cell in answers.items():
                classes.setdefault((scope, name), []).append(cell['recall'])
    slices = {}
    for row in completed_:
        for requirement in (row.get('winner') or {}).get('qualification', {}).get('requirements', ()):
            key = tuple(requirement.get(name) for name in ('task', 'target', 'class', 'slice'))
            slices.setdefault(key, []).append(requirement['recall'])
    qualified = [row.get('winner', {}).get('qualification', {}).get('state') for row in completed_ if row.get('winner')]
    comparable = len({artifact_digest(row.get('environment')) for row in completed_}) <= 1
    state = 'incomplete' if len(completed_) != len(records) else 'not_declared' if not qualified or all(value in (None, 'not_declared') for value in qualified) else 'met' if all(value == 'met' for value in qualified) else 'unmet'
    return {'state': state if comparable else 'incomparable', 'comparable': comparable, 'completed': len(completed_), 'declared': len(records),
            'metrics': {name: spread(values) for name, values in metrics.items()} if comparable else {},
            'per_class': [{'target': target, 'class': name, **spread(values)} for (target, name), values in sorted(classes.items())] if comparable else [],
            'requirements': [{'task': task, 'target': target, 'class': name, 'slice': slice_, **spread(values)}
                             for (task, target, name, slice_), values in sorted(slices.items(), key=lambda item: str(item[0]))] if comparable else [],
            'population': 'model_select'}


def qualify(source, to, *, execute=False, representation='merged', nominal_run=None, nominal_receipt=None):
    source, to = Path(source).resolve(), Path(to).resolve()
    resolved, _ = jobs.resolve(source)
    nominal = None
    if nominal_receipt is not None and nominal_run is None:
        raise ConfigError('nominal_receipt requires its nominal_run')
    if nominal_run is not None:
        nominal_run = Path(nominal_run).resolve()
        nominal_receipt = Path(nominal_receipt).resolve() if nominal_receipt is not None else nominal_run / 'deployment' / 'receipt.json'
        _, _, fitted = jobs.ready_run(nominal_run, 'the matching nominal seed-0 fit')
        receipt = load_receipt(nominal_receipt)
        if resolved['recipe']['seed'] != 0 or receipt.model().representation != representation or fitted['fit']['id'] != resolved['fit']['id'] or fitted['monitor']['id'] != resolved['monitor']['id'] or fitted['native'] != resolved['native'] or any(
                receipt.data[key] != resolved[field]['id'] for key, field in (('fit_id', 'fit'), ('monitor_id', 'monitor'))):
            raise ConfigError('nominal run/receipt differs in fit, monitoring, native settings or seed; no reuse')
        nominal = {'run': str(nominal_run), 'receipt': str(nominal_receipt), 'receipt_id': receipt.id}
    declaration = {'schema': 1, 'kind': 'recipe-qualification', 'job': str(source), 'job_sha256': files.sha256_file(source),
                   'nominal_fit': resolved['fit']['id'], 'monitor': resolved['monitor']['id'],
                   'population': resolved['cohort'], 'requirements': resolved['recipe']['development_recall'],
                   'variants': variants(resolved['recipe']), 'representation': representation, 'nominal': nominal}
    declaration['id'] = artifact_digest(declaration)
    path = to / 'qualification.json'
    with files.Lease(to):
        known = files.read_json(path)
        if known is not None and known['declaration'] != declaration:
            raise ConfigError(f'{path}: qualification declaration changed; use another destination')
        previous = {row['name']: row for row in known['runs']} if known else {}
        records = []
        for variant in declaration['variants']:
            directory = to / variant['name']
            job = directory / 'job.yaml'
            write_job(source, job, {key: variant[key] for key in ('seed', 'learning_rate')})
            result = completed(directory, run=nominal['run'], receipt=nominal['receipt']) if nominal and variant == declaration['variants'][0] else completed(directory)
            if result['state'] not in ('completed', 'budget_limited') and variant['name'] in previous:
                result = {**previous[variant['name']], **result, 'state': previous[variant['name']]['state']}
            records.append({**variant, **result, 'job': str(job)})
        def publish():
            report = {'declaration': declaration, 'runs': records, 'summary': summarize(records)}
            files.publish(path, report)
            return {'path': str(path), **report}
        publish()
        for index, record in enumerate(records):
            if not execute or record['state'] in ('completed', 'budget_limited'):
                continue
            directory = to / record['name']
            try:
                from .lora import train
                from .select import select
                train(record['job'], directory / 'run')
                select(record['job'], directory / 'run', representation)
                result = completed(directory)
            except (KeyboardInterrupt, SystemExit):
                records[index] = {**record, **completed(directory), 'state': 'interrupted'}
                publish()
                raise
            except Exception as error:
                result = {**completed(directory), 'state': 'failed', 'error': f'{type(error).__name__}: {error}'}
                if isinstance(error, RuntimeError):
                    records[index] = {**record, **result}
                    return publish()
            records[index] = {**record, **result}
            publish()
        return publish()


def diagnostic(source, to, *, target, answer, families, epochs=4, execute=False, rank=None):
    if type(families) is not int or families < 1 or type(epochs) is not int or epochs < 1:
        raise ConfigError('diagnostic families and epochs must be positive integers')
    if rank is not None and (type(rank) is not int or rank < 1):
        raise ConfigError('diagnostic rank must be a positive integer')
    resolved, rows = jobs.resolve(source)
    found = sorted({row['component'] for row in rows if row['role'] == 'fit' and '/'.join(record_path(row)) == target and row['classes'][row['label']] == answer})[:families]
    if not found:
        raise ConfigError('no fit families support the requested target/class; materialize or review evidence first')
    select = {row['component'] for row in rows if row['role'] == 'model_select'}
    selected = {'fit': set(found), 'model_select': {min(select)}}
    cohort = [row for row in rows if row['component'] in selected.get(row['role'], ())]
    slots, labels = [], {}
    for row in cohort:
        if row['role'] != 'fit':
            continue
        canonical = canonical_answer(row['presentations'], {entry.id: entry.label for entry in row['presentations'] if entry.variant == 'canonical'})
        if canonical != row['label']:
            slots.append(row['id'])
        labels.setdefault((row['task'], record_path(row), row['projection']), set()).add(row['classes'][row['label']])
    contradictions = [{'task': task, 'target': list(address), 'projection': projection, 'classes': sorted(values)}
                      for (task, address, projection), values in labels.items() if len(values) > 1]
    if slots or contradictions:
        raise ConfigError(f'fit diagnostic has slot mismatches {slots} or contradictory labels {contradictions}')
    declared = files.read_yaml(Path(source))
    exports = []
    for entry in declared['exports']:
        roles = {role: sorted({origin['group'] for row in cohort if row['role'] == role for origin in row['sources']
                               if origin['export_id'] == entry['export_id']}) for role in selected}
        exports.append({**entry, 'path': str((Path(source).resolve().parent / entry['path']).resolve()), 'roles': {key: values for key, values in roles.items() if values}})
    fitting = [row for row in cohort if row['role'] == 'fit']
    selection = {'fit_situations': None, 'families': found, 'situation_ids': sorted({value for row in fitting for value in row['situations']})}
    overrides = {'epochs': epochs, 'min_epochs': epochs, 'fit_probe_observations': len(fitting), jobs.CADENCE: len(fitting) * epochs,
                 'warmup_examples': min(resolved['recipe']['warmup_examples'], len(fitting)), 'development_recall': []}
    if rank is not None:
        overrides['rank'] = rank
    to = Path(to).resolve()
    job = to / 'job.yaml'
    write_job(source, job, overrides, selection=selection, exports=exports)
    record = {'kind': 'fit-diagnostic', 'target': target, 'class': answer, 'requested_families': families, 'fit_families': found, 'observations': len(fitting),
              'role': 'fit', 'probe': 'in_sample', 'rank': rank or resolved['recipe']['rank'], 'alpha': resolved['recipe']['alpha'], 'slot_errors': slots, 'contradictions': contradictions}
    files.publish(to / 'diagnostic.json', record)
    if execute:
        from .lora import train
        train(job, to / 'run')
    return {**record, 'job': str(job), 'curve': curve(to / 'run')}
