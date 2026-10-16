from __future__ import annotations

from collections import Counter
from pathlib import Path
from importlib import metadata

from semantic_switch import ConfigError
from semantic_switch.training import (artifact_digest, files, model_job, native_model, prediction_report, record_path)

from . import jobs


def configuration(resolved, rows, lexical=False):
    return {'base': resolved['assets'], 'sha256': files.sha256_file(resolved['model']['gguf']), 'native': resolved['native'],
            'tasks': {key: {'behavior': key, 'scope': resolved['scope'][key]} for key in sorted(resolved['tasks'])},
            'observations': artifact_digest([{key: row.get(key) for key in ('id', 'task', 'role', 'component', 'label', 'classes', 'tags', 'pairs', 'bound_input')} |
                                           {'presentations': [entry.id for entry in row['presentations']]} for row in rows if row['role'] in ('fit', 'model_select')]),
            'lexical': lexical}


def reported(row, scored):
    return {**scored, 'target': '/'.join(record_path(row)), 'task': row['task'], 'options': row['classes'],
            'gold': row['classes'][row['label']], 'family': row['component'], 'role': row['role'],
            'tags': row.get('tags', []), 'pairs': row.get('pairs', [])}


def controls(rows, lexical=False):
    fitting, selected = {}, {}
    for row in rows:
        if row['role'] in ('fit', 'model_select'):
            (fitting if row['role'] == 'fit' else selected).setdefault((row['task'], record_path(row)), []).append(row)
    lexical_module = None
    if lexical:
        try:
            from . import lexical as lexical_module
        except ModuleNotFoundError as error:
            if (error.name or '').partition('.')[0] not in ('sklearn', 'scipy'):
                raise
            raise ConfigError('lexical baseline needs pip install "semantic-trainer[lexical]" in the trainer environment') from error
    stored, unavailable = [], []
    for target, held in selected.items():
        fit = fitting.get(target, [])
        counts = Counter(row['label'] for row in fit)
        majority = min(counts, key=lambda label: (-counts[label], label)) if counts else None
        head, reason = None, None
        if lexical:
            if len(counts) < 2:
                reason = 'fewer than two fit classes'
            else:
                x, active = lexical_module.compact(lexical_module.matrix([row['text'] for row in fit]))
                if not len(active):
                    reason = 'no lexical features in fit inputs'
                else:
                    head = lexical_module.fit(x, active, [row['label'] for row in fit], held[0]['classes'])
            if reason:
                unavailable.append({'task': target[0], 'target': '/'.join(target[1]), 'reason': reason})
        for row in held:
            answers = {'majority': majority}
            if lexical:
                answers['lexical'] = int(head.probs(row['text']).argmax()) if head is not None else None
            for arm, answer in answers.items():
                stored.append(reported(row, {'candidate': arm, 'id': row['id'], 'answer': row['classes'][answer] if answer is not None else None,
                                             'correct': answer == row['label'] if answer is not None else None, 'accepted': False,
                                             'outcome': 'predicted' if answer is not None else 'unavailable', 'acceptance': 'unassessed'}))
    return stored, unavailable


def load(path, resolved=None, rows=None):
    path = Path(path)
    record = files.read_json(path)
    if not isinstance(record, dict) or record.get('kind') != 'training-baseline' or record.get('schema') != 1 \
            or record.get('id') != artifact_digest({key: value for key, value in record.items() if key != 'id'}):
        raise ConfigError(f'{path}: invalid training baseline')
    if files.sha256_file(path.parent / record['rows']['file']) != record['rows']['sha256']:
        raise ConfigError(f'{path}: baseline prediction rows changed')
    if resolved is not None and record['configuration'] != configuration(resolved, rows, record['configuration']['lexical']):
        raise ConfigError(f'{path}: baseline does not match this task, population, base or native settings')
    return record


def baseline(resolved, observations, directory, *, lexical=False):
    from .select import evaluations, trial_models
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'baseline.json'
    if path.exists():
        return load(path, resolved, observations)
    contract, task = jobs.single_task(resolved)
    rows = [row for row in observations if row['role'] == 'model_select']
    base = {'base': (0, {'source': resolved['model']['gguf'], 'lora': None}, None)}
    models = trial_models('merged', resolved, base)
    stored, unavailable = controls(observations, lexical)
    with model_job(*models.values()) as handle:
        _, evaluated = next(evaluations('merged', task['definition'], resolved['scope'][contract]['targets'], models, rows, handle,
                                         journal_root=directory / 'native', job_id=configuration(resolved, observations, lexical)))
    scored, realized, native, work = evaluated
    stored.extend(reported(row, {**result, 'candidate': 'base'}) for row, result in zip(rows, scored))
    row_path = directory / 'baseline.rows.jsonl'
    files.publish(row_path, files.jsonl(stored))
    record = {'schema': 1, 'kind': 'training-baseline', 'configuration': configuration(resolved, observations, lexical),
              'rows': {'file': row_path.name, 'sha256': files.sha256_file(row_path), 'rows': len(stored)},
              'native': {'realized': realized, 'predictions': native, 'work': work},
              'methods': {'majority': {'tie_break': 'declared class order', 'role': 'fit'},
                          'lexical': {'role': 'fit', 'c': 10., 'seed': 0, 'max_iter': 5000, 'tolerance': 1e-4, 'solver': 'lbfgs',
                                      'versions': {name: metadata.version(name) for name in ('scikit-learn', 'scipy')}} if lexical else None},
              'controls': {name: prediction_report([row for row in stored if row['candidate'] == name])
                           for name in ('majority', 'base', *(['lexical'] if lexical else []))}, 'unavailable': unavailable}
    record['id'] = artifact_digest(record)
    files.write_new(path, record)
    return record


def native_result(record, path, observations):
    values = {row['id']: row['value'] for row in observations}
    scored = [row for row in files.iter_records(Path(path).parent / record['rows']['file']) if row['candidate'] == 'base']
    return {**record['native'], 'scored': [{**row, 'gold': values[row['id']]} for row in scored]}
