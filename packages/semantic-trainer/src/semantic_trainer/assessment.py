from __future__ import annotations

import os
from pathlib import Path

from semantic_switch import ConfigError
from semantic_switch.training import artifact_digest, files, load_receipt, prediction_report, read_learning, record_path

from . import jobs


def predictions(rows):
    return [{**row, 'target': '/'.join(record_path(row)), 'options': row['classes'], 'gold': row['gold_class']}
            for row in rows]


def policies(facts, prefix=''):
    found = {}
    for name, fact in facts.items():
        address = f'{prefix}/{name}' if prefix else name
        if fact.get('policies'):
            found[address] = fact['policies']
        found.update(policies(fact.get('members', {}), address))
    return found


def publish(directory, resolved, observations, selection, acceptance, run, baseline=None):
    directory, run = Path(directory), Path(run)
    selected = selection['selected']
    rows = predictions(list(files.iter_records(directory / selection['rows']['file'])))
    arms = {candidate['name']: prediction_report([row for row in rows if row['candidate'] == candidate['name']]) for candidate in selection['candidates']}
    comparator = 'zero' if resolved['initialization']['kind'] == 'adapter' else 'base'
    before, after = arms.get(comparator, {}), arms[selected]
    controls = {}
    references = {'predictions': selection['rows'], 'lock': {'file': 'task.lock.json', 'sha256': files.sha256_file(directory / 'task.lock.json')}}
    if baseline is not None:
        from .controls import load
        base = load(baseline, resolved, observations)
        controls = base['controls']
        references['baseline'] = {'file': os.path.relpath(baseline, directory), 'sha256': files.sha256_file(baseline)}
        references['baseline_rows'] = {'file': os.path.relpath(Path(baseline).parent / base['rows']['file'], directory), 'sha256': base['rows']['sha256']}
    for name in ('progress.jsonl', 'updates.jsonl', 'resolved-job.json'):
        if (run / name).exists():
            references[name] = {'file': os.path.relpath(run / name, directory), 'sha256': files.sha256_file(run / name)}
    curves = [{'step': point['step'], 'examples_seen': point['examples_seen'], 'correct': point['model_select']['correct'],
               'class_changes': point['model_select'].get('class_changes'), 'nll': point['model_select']['nll']}
              for point in files.iter_records(run / 'progress.jsonl') if point.get('kind') == 'evaluation'] if (run / 'progress.jsonl').exists() else []
    outcome = 'base_retained' if selected == 'base' else 'incumbent_retained' if selected == comparator else 'student_selected'
    report = {'schema': 1, 'kind': 'learning-assessment', 'selected': selected, 'comparator': comparator,
              'summary': {'outcome': outcome, 'selected': selected, 'comparator': comparator,
                          'correct': after['raw_correct'], 'submitted': after['submitted'], 'wrong_accepted': after['wrong_accepted'],
                          'coverage': after['coverage'], 'independent': False},
              'reason': jobs.selection_reason(selection['candidates'], selection['unit'], selected),
              'criteria': {f"{row['task']}:{'/'.join(record_path(row))}": row['meaning_rev'] for row in observations},
              'selection': selection, 'support': resolved['support'], 'allocation': resolved['objective_weights'], 'stop': selection['fit_stop'],
              'controls': controls, 'candidates': arms, 'delta': {key: after[key] - before[key] if after.get(key) is not None and before.get(key) is not None else None
                                                               for key in ('raw_correct', 'wrong_accepted', 'precision', 'coverage')},
              'acceptance': {**acceptance, 'policies': policies(files.read_json(directory / 'task.lock.json')['facts'])}, 'monitoring': curves, 'files': references,
              'populations': {'selection': 'model_select development; never independent quality',
                              'acceptance': 'acceptance_fit, separate from selection', 'confirmation': 'unassessed'},
              'operational_actions': {'state': 'unassessed', 'reason': 'target selection; use evaluate_development with action references'},
              'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'initialization': resolved['initialization']['kind']}
    report['summary']['ranking_key'] = report['reason']['first_differing_key']
    report['summary']['compared_with'] = report['reason']['compared_with']
    report['id'] = artifact_digest(report)
    files.write_new(directory / 'learning.json', report)
    read_learning(directory / 'learning.json')
    return report


def rebuild(source, to=None, *, development=None):
    source = Path(source).resolve()
    if source.is_dir() or source.name == 'receipt.json':
        receipt = load_receipt(source)
        source = receipt.directory / receipt.data.get('learning', {}).get('file', 'learning.json')
    report = read_learning(source)
    selected = report['selected']
    rows = predictions(list(files.iter_records(source.parent / report['files']['predictions']['file'])))
    report['candidates'] = {name: prediction_report([row for row in rows if row['candidate'] == name]) for name in report['candidates']}
    report['reason'] = jobs.selection_reason(report['selection']['candidates'], report['selection']['unit'], selected)
    before, after = report['candidates'].get(report['comparator'], {}), report['candidates'][selected]
    report['summary'].update(correct=after['raw_correct'], submitted=after['submitted'], wrong_accepted=after['wrong_accepted'],
                             coverage=after['coverage'], ranking_key=report['reason']['first_differing_key'], compared_with=report['reason']['compared_with'])
    report['delta'] = {key: after[key] - before[key] if after.get(key) is not None and before.get(key) is not None else None
                       for key in report['delta']}
    lock = source.parent / report['files'].get('lock', {}).get('file', 'task.lock.json')
    if lock.exists():
        report['acceptance']['policies'] = policies(files.read_json(lock)['facts'])
    if development is not None:
        from semantic_switch.training import evaluate_development
        if to is None:
            raise ConfigError('composed development assessment needs --to for its journal')
        assessed = evaluate_development(lock.parent / 'receipt.json', to=Path(to).with_suffix('.development'), **development)
        report['operational_actions'] = assessed['operational_actions']
    if to is not None:
        to = Path(to).resolve()
        report['files'] = {name: {**entry, 'file': os.path.relpath(source.parent / entry['file'], to.parent)} for name, entry in report['files'].items()}
    report.pop('id')
    report['id'] = artifact_digest(report)
    if to is not None:
        files.write_new(to, report)
    return report
