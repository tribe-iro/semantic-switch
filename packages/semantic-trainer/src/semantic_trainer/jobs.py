from __future__ import annotations

from fractions import Fraction
import math
import os
import re
import shutil
from pathlib import Path

from semantic_switch import ConfigError, Model
from semantic_switch.training import (MAX_QUESTION_TOKENS, NATIVE_SETTINGS, ROLES, USES, FamilyIndex, LabelSource, admit, artifact_digest, describe_task, exact_families, files,
                                      freeze_task, frozen_task, merge_sources, native_model, native_settings, path_record, read_export, recall_counts, recall_floor, record_path, role_of)

from . import assets
from .adapter import verified

SCHEMA, RESOLVED_SCHEMA, FAMILIES, FITTING, IDENTITY = 8, 11, 1, 2, 3
CANDIDATES_SCHEMA, RUN_SCHEMA = 8, 4
RECORDS = {'run.json': RUN_SCHEMA, 'candidates.json': CANDIDATES_SCHEMA, 'resolved-job.json': RESOLVED_SCHEMA}
LOSS = {'name': 'full_vocabulary_group_nll', 'version': 2, 'reduction': 'float32'}
MODULES = {'q_proj': ('self_attn', 'attn_q'), 'k_proj': ('self_attn', 'attn_k'), 'v_proj': ('self_attn', 'attn_v'), 'o_proj': ('self_attn', 'attn_output'),
           'gate_proj': ('mlp', 'ffn_gate'), 'up_proj': ('mlp', 'ffn_up'), 'down_proj': ('mlp', 'ffn_down')}
DEFAULTS = {'rank': 16, 'alpha': 32, 'dropout': 0., 'learning_rate': 2e-4, 'schedule': 'constant', 'weight_decay': .01, 'gradient_clip': 1.,
            'epochs': 8, 'min_epochs': 2, 'seed': 0, 'batch_size': 8, 'microbatch': 1, 'evaluation_batch_size': 1, 'fit_probe_observations': 0, 'max_padded_tokens': 8192, 'target_modules': list(MODULES), 'warmup_examples': None,
            'patience_examples': None, 'min_improvement': .001, 'selection_unit': 'observation', 'objective_balance': 'situation',
            'behavior_slices': [], 'development_recall': []}
CADENCE = 'eval_every_examples'
FITTING_DEFAULTS = {'attention': 'eager', 'gradient_checkpointing': False, 'fused_optimizer': False}
WARMUP_REFERENCE_EPOCHS, WARMUP_FRACTION = 3, Fraction(1, 20)
UNITS = ('observation', 'complete_situation')
FIELDS = {'schema', 'loss', 'objective', 'tasks', 'initialization', 'assets', 'fitting', 'exports', 'selection', 'recipe', 'budget', 'native', 'profile', 'scope',
          'mixture'}
PROFILE_SCHEMA, PROFILE_KIND, PROFILE_TOLERANCE = 2, 'fitting-profile', .5
TUNING = tuple(role for role, (split, _) in ROLES.items() if split == 'tune')
BUDGET = {'max_tokens': None, 'max_monitor_tokens': None, 'max_sequence_tokens': 1024, 'max_seconds': 7200, 'feasibility_updates': 8}
RANKING = ('correct', 'regressions', 'nll', 'base', 'step')
SHARED = ('key', 'observation', 'rendered', 'situation')
ORDER = {'name': 'epoch_shuffle', 'revision': 1}
REDUCTION = {'revision': 2}
LOCATIONS = ('bundle', 'definition')
ADAPTER = ('rank', 'alpha', 'dropout', 'target_modules')
LOADER = {'name': 'peft_from_pretrained', 'revision': 1}
INHERITED = {'fit': 'adaptation', 'model_select': 'selection', 'acceptance_fit': 'acceptance_fit'}
SELECTION_SCHEMA, SELECTION_KIND = 1, 'initializer-selection'
INSTALL = 'install the torch wheel for that device in the trainer environment (packages/semantic-trainer/README.md)'


def physical_cores():
    cores = {path.read_text() for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology/core_cpus_list')}
    return len(cores) or os.cpu_count() or 1


def fitting_settings(settings):
    if not isinstance(settings, dict) or set(settings) - {'device', 'precision', 'threads', *FITTING_DEFAULTS}:
        raise ConfigError(f"fitting accepts device, precision, threads, {', '.join(FITTING_DEFAULTS)}")
    device = settings.get('device')
    if not isinstance(device, str) or not re.fullmatch(r'cpu|(xpu|cuda):(\d+)', device):
        raise ConfigError('fitting.device must be cpu, xpu:N or cuda:N')
    allowed = ('float32', 'bfloat16') if device == 'cpu' else ('bfloat16',)
    resolved = {'device': device, 'precision': settings.get('precision', allowed[0]), 'threads': settings.get('threads', physical_cores()),
                **{key: settings.get(key, value) for key, value in FITTING_DEFAULTS.items()}}
    if resolved['precision'] not in allowed:
        raise ConfigError(f"fitting.precision on {device} must be {' or '.join(allowed)}")
    if type(resolved['threads']) is not int or resolved['threads'] < 1:
        raise ConfigError('fitting.threads must be a positive integer')
    if resolved['attention'] not in ('eager', 'sdpa'):
        raise ConfigError('fitting.attention must be eager or sdpa')
    if type(resolved['gradient_checkpointing']) is not bool or type(resolved['fused_optimizer']) is not bool:
        raise ConfigError('fitting.gradient_checkpointing and fitting.fused_optimizer must be boolean')
    return resolved


def fitting_device(fitting):
    try:
        import torch
    except ImportError as error:
        raise ConfigError(f'fitting requires torch; {INSTALL}') from error
    kind, _, number = fitting['device'].partition(':')
    if number:
        backend = getattr(torch, kind, None)
        count = backend.device_count() if backend is not None and backend.is_available() else 0
        if int(number) >= count:
            raise ConfigError(f"{fitting['device']} is unavailable to torch {torch.__version__} ({count} {kind} devices); {INSTALL}")
    return fitting['device']


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def recipe_of(given, fit_count):
    given = given or {}
    if not isinstance(given, dict) or set(given) - {*DEFAULTS, CADENCE}:
        raise ConfigError(f'unsupported recipe fields {sorted(set(given) - {*DEFAULTS, CADENCE})}; supported: {sorted({*DEFAULTS, CADENCE})}')
    if any(value is None for value in given.values()):
        raise ConfigError(f'recipe overrides need values: {sorted(key for key, value in given.items() if value is None)}')
    recipe = {**DEFAULTS, **{key: value for key, value in given.items() if key != CADENCE}}
    derived = {'warmup_examples': ceil_div(WARMUP_REFERENCE_EPOCHS * fit_count * WARMUP_FRACTION.numerator, WARMUP_FRACTION.denominator),
               'patience_examples': fit_count}
    recipe |= {key: value for key, value in derived.items() if recipe[key] is None}
    for key in ('rank', 'alpha', 'batch_size', 'microbatch', 'max_padded_tokens', 'evaluation_batch_size', 'epochs', 'min_epochs', 'patience_examples'):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ConfigError(f'{key} must be a positive integer')
    if type(recipe['fit_probe_observations']) is not int or recipe['fit_probe_observations'] < 0:
        raise ConfigError('fit_probe_observations must be a nonnegative integer')
    if type(recipe['warmup_examples']) is not int or recipe['warmup_examples'] < 0:
        raise ConfigError('warmup_examples must be a nonnegative integer')
    if recipe['min_epochs'] > recipe['epochs']:
        raise ConfigError(f"min_epochs {recipe['min_epochs']} exceeds the epoch budget cap {recipe['epochs']}")
    modules = recipe['target_modules']
    if not isinstance(modules, list) or not modules or len(set(modules)) != len(modules) or not set(modules) <= set(MODULES):
        raise ConfigError(f'target_modules must be distinct projections from {list(MODULES)}')
    if type(recipe['seed']) is not int:
        raise ConfigError('seed must be integer')
    if recipe['schedule'] not in ('constant', 'cosine'):
        raise ConfigError('schedule must be constant or cosine')
    if recipe['selection_unit'] not in UNITS:
        raise ConfigError(f"selection_unit must be one of {', '.join(UNITS)}")
    if recipe['objective_balance'] not in ('situation', 'class_family'):
        raise ConfigError('objective_balance must be situation or class_family')
    behavior_contract(recipe)
    recipe['behavior_slices'] = sorted([{'name': item['name'], **{key: sorted(set(item[key])) for key in ('tags', 'any_tags') if item.get(key)}}
                                       for item in recipe['behavior_slices']], key=lambda item: item['name'])
    recipe['development_recall'] = sorted([{**item, 'minimum': float(item['minimum'])} for item in recipe['development_recall']],
                                           key=lambda item: tuple(item.get(key, '') for key in ('task', 'target', 'class', 'slice')))
    for key in ('learning_rate', 'gradient_clip', 'dropout', 'weight_decay', 'min_improvement'):
        if type(recipe[key]) not in (int, float) or not math.isfinite(recipe[key]):
            raise ConfigError(f'{key} must be finite')
    if recipe['learning_rate'] <= 0 or recipe['gradient_clip'] <= 0 or not 0 <= recipe['dropout'] < 1 or recipe['weight_decay'] < 0 or not 0 < recipe['min_improvement'] < 1:
        raise ConfigError('invalid learning rate, gradient cap, dropout, weight decay or absolute min_improvement (0, 1)')
    return recipe


def cadence_of(given):
    cadence = (given or {}).get(CADENCE)
    if cadence is None:
        raise ConfigError('choose the monitoring cadence explicitly: --recipe eval_every_examples=N (examples seen between evaluation points)')
    if type(cadence) is not int or cadence < 1:
        raise ConfigError(f'{CADENCE} must be a positive integer')
    return cadence


STOPPING = 'primary-score-anchor@1'


def primary(point, unit):
    return point['correct']['macro'][unit]


def advance(control, point, unit, threshold):
    value, at = primary(point, unit), {'step': point['step'], 'examples': point['examples_seen']}
    if control is None:
        return {'anchor': value, 'best': value, 'best_at': at, 'improved_at': at}
    out = dict(control)
    if value > out['best']:
        out['best'], out['best_at'] = value, at
    if out['best'] - out['anchor'] > threshold:
        out['anchor'], out['improved_at'] = out['best'], out['best_at']
    return out


def plateaued(control, examples, recipe, fit_count):
    return examples >= recipe['min_epochs'] * fit_count and examples - control['improved_at']['examples'] >= recipe['patience_examples']


def schedule_of(recipe, given, fit_count):
    batch = recipe['batch_size']
    per_epoch = ceil_div(fit_count, batch)
    total, warmup = recipe['epochs'] * per_epoch, ceil_div(recipe['warmup_examples'], batch)
    if warmup > total:
        raise ConfigError(f"warmup of {recipe['warmup_examples']} examples ({warmup} optimizer steps) exceeds the {total} available steps")
    explicit = 'warmup_examples' in given
    return {'max_optimizer_steps': total, 'steps_per_epoch': per_epoch, 'warmup_steps': warmup,
            'warmup': {'policy': 'explicit' if explicit else 'reference_epochs_fraction', 'explicit': explicit, 'reference_epochs': WARMUP_REFERENCE_EPOCHS,
                       'fraction': str(WARMUP_FRACTION), 'examples': recipe['warmup_examples'], 'steps': warmup},
            'decay_horizon_steps': total if recipe['schedule'] == 'cosine' else None}


def due(examples, measured, cadence, step, total):
    return examples // cadence > measured // cadence or step == total


def evaluation_points(count, recipe):
    batch, cadence = recipe['batch_size'], recipe['eval_every_examples']
    total = recipe['epochs'] * ceil_div(count, batch)
    points, step, examples, measured = 1, 0, 0, 0
    for _ in range(recipe['epochs']):
        for start in range(0, count, batch):
            step, examples = step + 1, examples + min(batch, count - start)
            if due(examples, measured, cadence, step, total):
                points, measured = points + 1, examples
    return points


def evidence(row):
    return (row['meaning_rev'], row['classes'], row['label'], row['text'], row['projection'], [entry.id for entry in row['presentations']])


def lengths(row):
    return [entry.length for entry in row['presentations']]


def tokens(row):
    return sum(lengths(row))


def source_family(path, manifest):
    family = manifest['task_family']
    if not isinstance(family, str) or ':' not in family:
        raise ConfigError(f'{path}: a native export carries one namespaced task family (ss export --native --task-family NAMESPACE:NAME); found {family!r}')
    return family


def behavior_contract(recipe):
    slices, requirements = recipe['behavior_slices'], recipe['development_recall']
    if not isinstance(slices, list) or not isinstance(requirements, list):
        raise ConfigError('behavior_slices and development_recall must be lists')
    names = set()
    for item in slices:
        if not isinstance(item, dict) or set(item) - {'name', 'tags', 'any_tags'} or not isinstance(item.get('name'), str) or not item['name'] or item['name'] in names:
            raise ConfigError('behavior_slices require distinct names with tags and/or any_tags')
        names.add(item['name'])
        for key in ('tags', 'any_tags'):
            if key in item and (not isinstance(item[key], list) or not all(isinstance(tag, str) and tag for tag in item[key])):
                raise ConfigError(f"behavior slice {item['name']}: {key} must list nonempty tags")
        if not item.get('tags') and not item.get('any_tags'):
            raise ConfigError(f"behavior slice {item['name']}: declare metadata tags")
    seen = set()
    for item in requirements:
        if not isinstance(item, dict) or set(item) - {'task', 'target', 'class', 'slice', 'minimum'} or not {'target', 'class', 'minimum'} <= set(item):
            raise ConfigError('development_recall entries require target, class, minimum; task and slice are optional')
        if any(not isinstance(item[key], str) or not item[key] for key in ('target', 'class', 'task') if key in item):
            raise ConfigError('development_recall task, target and class must be nonempty strings')
        if 'slice' in item and (not isinstance(item['slice'], str) or item['slice'] not in names):
            raise ConfigError(f"unknown development recall slice {item['slice']!r}")
        if type(item['minimum']) not in (int, float) or not 0 < item['minimum'] <= 1:
            raise ConfigError('development recall minimum must be in (0, 1]')
        key = tuple(item.get(field) for field in ('task', 'target', 'class', 'slice'))
        if key in seen:
            raise ConfigError('development recall requirements must have distinct scopes')
        seen.add(key)


def behavior_rows(rows, recipe):
    if not recipe['behavior_slices'] and not recipe['development_recall']:
        return
    for row in rows:
        tags = set(row['tags'])
        row['slices'] = [item['name'] for item in recipe['behavior_slices'] if set(item.get('tags', ())) <= tags
                         and (not item.get('any_tags') or tags.intersection(item['any_tags']))]
    available = {}
    for row in rows:
        available.setdefault((row['task'], '/'.join(record_path(row))), set()).update(row['classes'])
    for item in recipe['development_recall']:
        matches = [classes for (task, target), classes in available.items() if target == item['target'] and ('task' not in item or task == item['task'])]
        if not matches or any(item['class'] not in classes for classes in matches):
            raise ConfigError(f"development recall names an undeclared target/class: {item}")
        if 'task' not in item and len(matches) != 1:
            raise ConfigError(f"development recall target {item['target']} needs a task scope")


def behavior_support(rows, recipe):
    if not recipe['behavior_slices']:
        return {}
    out, families, totals = {}, {}, {}
    names = [item['name'] for item in recipe['behavior_slices']]
    for row in rows:
        role, task, target = row['role'], row['task'], '/'.join(record_path(row))
        totals[role] = totals.get(role, 0) + row.get('objective', 0.)
        declared = out.setdefault(role, {}).setdefault(task, {}).setdefault(target, {})
        for name in names:
            declared.setdefault(name, {value: {'observations': 0, 'families': 0, 'objective': 0.} for value in row['classes']})
        label = row['classes'][row['label']]
        for name in row['slices']:
            cell = declared[name][label]
            cell['observations'] += 1
            cell['objective'] += row.get('objective', 0.)
            families.setdefault((role, task, target, name, label), set()).add(row['component'])
    for role, tasks in out.items():
        for task, targets in tasks.items():
            for target, slices in targets.items():
                for name, classes in slices.items():
                    for label, cell in classes.items():
                        cell['families'] = len(families.get((role, task, target, name, label), ()))
                        cell['objective'] = cell['objective'] / totals[role] if totals[role] else None
    return out


def development_recall(rows, results, required):
    if not required:
        return {'state': 'not_declared', 'requirements': []}
    def entries():
        for row in rows:
            target, result = '/'.join(record_path(row)), results[row['id']]
            for name in (None, *row.get('slices', ())):
                yield (row['task'], target, name), row['classes'][row['label']], row['component'], result['correct'], result.get('accepted', False)
    counts, out = recall_counts(entries()), []
    for item in required:
        task = item.get('task') or next((row['task'] for row in rows if '/'.join(record_path(row)) == item['target']), None)
        cell = counts.get((task, item['target'], item.get('slice'), item['class']), {'references': 0, 'families': 0, 'raw': 0})
        out.append({**item, 'task': task, 'references': cell['references'], 'families': cell['families'],
                    **recall_floor(cell['raw'], cell['references'], item['minimum'])})
    return {'state': 'met' if all(item['state'] == 'met' for item in out) else 'unassessed' if any(item['state'] == 'unassessed' for item in out) else 'unmet',
            'population': 'model_select', 'readout': 'raw', 'requirements': out}


def allocation_totals(rows, shares):
    cells = {}
    for row in rows:
        key = (row['task'], '/'.join(record_path(row)), row['classes'][row['label']], row['component'])
        cells[key] = cells.get(key, Fraction(0)) + shares[row['id']]
    return [{'task': task, 'target': target, 'class': label, 'family': family, 'share': float(share)}
            for (task, target, label, family), share in sorted(cells.items())]


def fit_support(rows, objective):
    support, declared, families = {}, {}, {}
    for row in rows:
        path = '/'.join(record_path(row))
        declared.setdefault(row['task'], {})[path] = row['classes']
        classes = support.setdefault(row['role'], {}).setdefault(row['task'], {}).setdefault(
            path, {key: {'observations': 0, 'families': 0, 'objective': None, 'tags': {}} for key in row['classes']})
        label = row['classes'][row['label']]
        cell = classes[label]
        cell['observations'] += 1
        families.setdefault((row['role'], row['task'], path, label), set()).add(row['component'])
        for tag in row['tags'] or ['untagged']:
            cell['tags'][tag] = cell['tags'].get(tag, 0) + 1
    for (role, task, path, label), found in families.items():
        support[role][task][path][label]['families'] = len(found)
    fitted = support.get('fit', {})
    missing = {(task, path): [key for key in classes if not fitted.get(task, {}).get(path, {}).get(key, {}).get('observations')]
               for task, paths in declared.items() for path, classes in paths.items() if objective == 'specialization' or path in fitted.get(task, {})}
    missing = {where: keys for where, keys in missing.items() if keys}
    return {'roles': support, 'unsupported_classes': {task: {path: keys for (owner, path), keys in missing.items() if owner == task}
                                                    for task in sorted({task for task, _ in missing})},
            'absent_fit_support': {task: sorted(set(paths) - set(fitted.get(task, {}))) for task, paths in sorted(declared.items())}}


def support_objective(support, rows):
    totals, members = {}, {}
    for row in rows:
        if 'objective' not in row or row['role'] not in support['roles']:
            continue
        key = (row['role'], row['task'], '/'.join(record_path(row)), row['classes'][row['label']])
        totals[key] = totals.get(key, 0.) + row['objective']
        members[row['role']] = members.get(row['role'], 0) + 1
    for (role, task, path, label), total in totals.items():
        support['roles'][role][task][path][label]['objective'] = total / members[role]
    return support


def scoped(task, identifier):
    return f'{task}:{identifier}'


def situations_of(loaded, observations, scopes):
    merged = {}
    for _, manifest, _, task in loaded:
        for situation in manifest['situations']:
            entry = merged.setdefault(scoped(task, situation['id']), {'task': task, 'targets': {}, 'records': set()})
            entry['records'].update(situation['records'])
            for target in situation['targets']:
                address = '/'.join(target['address'])
                known = entry['targets'].get(address)
                value = target.get('observation') or target['outcome']
                if 'observation' in target and isinstance(known, str) and len(known) == 64 and known != value:
                    raise ConfigError(f"situation {situation['id']}: {address} links two observations across exports")
                if known is None or 'observation' in target:
                    entry['targets'][address] = value
    out = {}
    for identifier, entry in sorted(merged.items()):
        targets = {address: entry['targets'].get(address, 'unlabelled') for address in scopes[entry['task']]}
        linked = [value for value in targets.values() if value in observations]
        roles = {observations[value]['role'] for value in linked}
        if len(roles) > 1:
            raise ConfigError(f'situation {identifier} spans roles {sorted(roles)}; its observations share one family')
        out[identifier] = {'task': entry['task'], 'role': roles.pop() if roles else 'unused', 'targets': targets, 'records': sorted(entry['records']),
                           'complete': all(value in observations or value == 'exact' for value in targets.values())}
    return out


def objective_weights(rows, tasks, balance='situation'):
    report, allocation = {}, {'mode': balance, 'revision': 1, 'max_row_multiplier': 0., 'before': [], 'after': []}
    for role in ('fit', 'model_select'):
        members = [row for row in rows if row['role'] == role]
        examples = {}
        for row in members:
            for situation in row['situations'] or [scoped('observation', row['id'])]:
                examples.setdefault(row['task'], {}).setdefault(situation, []).append(row)
        total = sum(Fraction(tasks[task]['weight']) for task in examples)
        shares = {row['id']: Fraction(0) for row in members}
        for task, found in examples.items():
            for linked in found.values():
                for row in linked:
                    shares[row['id']] += Fraction(tasks[task]['weight']) / total / len(found) / len(linked)
        before = dict(shares)
        if role == 'fit' and balance == 'class_family':
            budgets, classes, families, cells = {}, {}, {}, {}
            for row in members:
                address = (row['task'], record_path(row))
                label = row['classes'][row['label']]
                cell = (*address, label, row['component'])
                budgets[address] = budgets.get(address, Fraction(0)) + shares[row['id']]
                classes.setdefault(address, set()).add(label)
                families.setdefault((*address, label), set()).add(row['component'])
                cells[cell] = cells.get(cell, Fraction(0)) + shares[row['id']]
            for row in members:
                address = (row['task'], record_path(row))
                label = row['classes'][row['label']]
                shares[row['id']] *= budgets[address] / len(classes[address]) / len(families[(*address, label)]) / cells[(*address, label, row['component'])]
        for row in members:
            row['objective'] = float(shares[row['id']] * len(members))
            entry = report.setdefault(row['task'], {}).setdefault(role, {'weight': float(Fraction(tasks[row['task']]['weight']) / total),
                                                                         'examples': len(examples[row['task']]), 'observations': 0, 'presentations': 0, 'tokens': 0})
            entry['observations'] += 1
            entry['presentations'] += len(row['presentations'])
            entry['tokens'] += tokens(row)
        if role == 'fit':
            allocation = {'mode': balance, 'revision': 1, 'max_row_multiplier': max((float(shares[key] / value) for key, value in before.items() if value), default=0.),
                          'before': allocation_totals(members, before), 'after': allocation_totals(members, shares)}
    return {'revision': REDUCTION['revision'], 'tasks': dict(sorted(report.items())), 'fit_allocation': allocation}


def presentation_weights(row):
    counts = {}
    for entry in row['presentations']:
        counts[entry.stage] = counts.get(entry.stage, 0) + 1
    return [1 / len(counts) / counts[entry.stage] for entry in row['presentations']]


def names(row):
    return {*row['aliases'], *(name for kind, name in row['family_seeds'] if kind == 'group')}


def load(paths, task):
    loaded = {}
    for path in sorted(set(paths)):
        manifest, exported = read_export(path, native=True)
        loaded.setdefault(manifest['export_id'], (path, manifest, exported, task))
    return [loaded[key] for key in sorted(loaded)]


def cohort(loaded, inherited=()):
    index, anchors, rows = FamilyIndex(), {}, {}
    for path, manifest, exported, task in loaded:
        task_family = source_family(path, manifest)
        for row in exported:
            shared = [tuple(node) for node in row['family_seeds'] if node[0] in SHARED]
            if not shared:
                raise ConfigError(f"{path}: observation {row['id']} carries no {', '.join(SHARED)} family seed; re-export it from its dataset assignment")
            anchor = ('group', artifact_digest([manifest['export_id'], row['component']]))
            for node in (*shared, anchors.setdefault(row['id'], anchor)):
                index.join(anchor, node)
            current = rows.setdefault(row['id'], {**row, 'task': task, 'task_family': task_family, 'role': 'unused', 'sources': [], 'situations': [], 'label_sources': []})
            if evidence(current) != evidence(row):
                raise ConfigError(f"observation {row['id']} has different meaning, labels, evidence or native presentations across exports")
            if current['task'] != task:
                raise ConfigError(f"observation {row['id']} is exported under task contracts {current['task']} and {task}; one supervised observation "
                                  'belongs to one task objective, so a renamed or derivative task cannot repeat it')
            if current['task_family'] != task_family:
                raise ConfigError(f"observation {row['id']} carries task families {current['task_family']} and {task_family}")
            current['sources'].append({'export_id': manifest['export_id'], 'path': path, 'task_family': task_family, 'namespace': manifest['namespace'],
                                       'component': row['component'], 'group': row['group'], 'aliases': row['aliases'], 'family_seeds': row['family_seeds'],
                                       'dataset_role': role_of(row['split'], row['fold'])})
            current['situations'] = sorted({*current['situations'], *(scoped(task, situation) for situation in row['situations'])})
            current['label_sources'] = [s.record() for s in merge_sources(map(LabelSource.parse, current['label_sources']), map(LabelSource.parse, row['label_sources']))]
    prior = {}
    for row in inherited:
        anchor = ('group', artifact_digest(['inherited', row['component']]))
        index.add(anchor)
        for node in row['family_seeds']:
            index.join(anchor, tuple(node))
        prior.setdefault(anchor, set()).update(row['uses'])
    closures, families = {}, {}
    for identifier, row in rows.items():
        root = index.find(anchors[identifier])
        if root not in closures:
            seeds = [list(node) for node in index.seeds(root) if node[0] in SHARED]
            uses = set().union(*(found for anchor, found in prior.items() if index.find(anchor) == root))
            closures[root] = artifact_digest([FAMILIES, seeds]), seeds, sorted(uses, key=USES.index)
        family, seeds, uses = closures[root]
        row.update(component=family, group=family, family_seeds=seeds)
        entry = families.setdefault(family, {'observations': [], 'situations': set(), 'seeds': seeds, 'inherited': uses})
        entry['observations'].append(identifier)
        entry['situations'].update(row['situations'])
    return rows, families


def assign(rows, roles):
    for identifier, found in roles.items():
        if len(found) > 1:
            raise ConfigError(f'observation {identifier} assigned to conflicting roles {sorted(found)}')
        rows[identifier]['role'] = role = next(iter(found))
        if role in TUNING:
            admit([(f"observation {identifier} ({'/'.join(record_path(rows[identifier]))}, {role})", rows[identifier]['label_sources'])])


def separated(rows, families):
    for family, entry in families.items():
        found = {}
        for identifier in entry['observations']:
            row = rows[identifier]
            if row['role'] != 'unused':
                found.setdefault(row['role'], set()).update(f"{source['export_id'][:12]}:{source['component'][:12]}" for source in row['sources'])
        if len(found) > 1:
            raise ConfigError(f"canonical family {family} crosses roles through source components {({role: sorted(found[role]) for role in sorted(found)})}; "
                              f"connecting seeds {[':'.join(node) for node in entry['seeds'][:6]]}; unsplittable cohort")
        if 'adaptation' in entry['inherited'] and set(found) - {'fit'}:
            raise ConfigError(f"canonical family {family} trained the initializer adapter but holds {sorted(set(found) - {'fit'})} observations; "
                              f"connecting seeds {[':'.join(node) for node in entry['seeds'][:6]]}; reassign that family to fit or leave it unused")


def intended(rows, acceptance_fit):
    return {identifier: found for identifier, row in rows.items()
            if (found := {source['dataset_role'] for source in row['sources'] if acceptance_fit or source['dataset_role'] != 'acceptance_fit'})}


def selected(loaded, entries):
    roles = {}
    for (path, _, exported, _), entry in zip(loaded, entries):
        matched = {role: set() for role in entry['roles']}
        for row in exported:
            for role, selectors in entry['roles'].items():
                if hit := names(row) & set(selectors):
                    matched[role] |= hit
                    roles.setdefault(row['id'], set()).add(role)
        for role, selectors in entry['roles'].items():
            if set(selectors) - matched[role]:
                raise ConfigError(f'{path}: unknown group selectors for {role}: {sorted(set(selectors) - matched[role])}')
    return roles


def exported_roles(loaded, rows):
    entries = []
    for path, manifest, exported, task in loaded:
        roles = {}
        for row in exported:
            if rows[row['id']]['role'] == (role := role_of(row['split'], row['fold'])):
                roles.setdefault(role, set()).add(row['group'])
        entries.append({'path': path, 'export_id': manifest['export_id'], 'task': task, 'roles': {role: sorted(roles[role]) for role in TUNING if role in roles}})
    return entries


def numerical(row):
    return {'id': row['id'], 'kind': row['kind'], 'classes': row['classes'], 'label': row['label'], 'objective': row['objective'],
            'weights': presentation_weights(row), 'presentations': [entry.id for entry in row['presentations']]}


def computation(rows):
    return {'revision': IDENTITY, 'reduction': REDUCTION, 'tokenizer': artifact_digest(rows[0]['native']['tokenizer']) if rows else None}


def fit_inputs(resolved, rows):
    recipe, schedule = resolved['recipe'], resolved['schedule']
    return {**computation(rows), 'base': {**{key: resolved['model'][key] for key in ('repo', 'revision')}, 'files': resolved['model']['files']},
            'loss': resolved['loss'], 'objective': resolved['objective'], 'observations': [numerical(row) for row in rows],
            'adapter': {key: recipe[key] for key in ADAPTER},
            'optimizer': {'algorithm': 'AdamW', **{key: recipe[key] for key in ('learning_rate', 'weight_decay', 'gradient_clip')}},
            'schedule': {'shape': recipe['schedule'], **{key: schedule[key] for key in ('max_optimizer_steps', 'steps_per_epoch', 'warmup_steps', 'decay_horizon_steps')}},
            'order': {'algorithm': ORDER, **{key: recipe[key] for key in ('seed', 'epochs', 'batch_size', 'microbatch', 'max_padded_tokens')}},
            **({'allocation': {'mode': recipe['objective_balance'], 'revision': 1}} if recipe['objective_balance'] != 'situation' else {}),
            'placement': resolved['fitting'], 'initialization': numerical_initialization(resolved['initialization']), 'implementation': FITTING}


def numerical_initialization(initialization):
    if initialization['kind'] == 'fresh':
        return {'kind': 'fresh'}
    return {'kind': 'adapter', 'loader': LOADER, **{key: initialization[key] for key in ('peft_sha256', 'config_sha256', 'config_digest', 'adapter', 'peft', 'precision', 'fit_id')},
            'inherited': artifact_digest(initialization['inherited']), 'overrides': artifact_digest(initialization['overrides'])}


def ordinals(rows):
    found = {}
    for row in rows:
        found.setdefault(row['task'], len(found))
    return found


def monitor_plan(resolved, selecting, probe):
    recipe, order = resolved['recipe'], ordinals(selecting)
    return {**computation([*selecting, *probe]), 'model_select': [numerical(row) for row in selecting], 'probe': [numerical(row) for row in probe],
            'tasks': [order[row['task']] for row in selecting],
            'situations': {f"{order[value['task']]}:{key.partition(':')[2]}": {'targets': value['targets'], 'complete': value['complete']}
                           for key, value in resolved['situations'].items() if value['role'] == 'model_select'},
            **({'behavior_slices': recipe['behavior_slices'], 'development_recall': recipe['development_recall'],
                'slice_memberships': [(row['id'], row.get('slices', [])) for row in selecting]}
               if recipe['behavior_slices'] or recipe['development_recall'] else {}),
            'evaluation_batch_size': recipe['evaluation_batch_size'], 'cadence': recipe[CADENCE], 'evaluation_points': resolved['work']['evaluation_points'], 'unit': recipe['selection_unit'], 'ranking': list(RANKING),
            'loss': resolved['loss'], 'stopping': {'rule': STOPPING, 'regressions': 'step-zero', **{key: recipe[key] for key in ('min_epochs', 'patience_examples', 'min_improvement')}}}


def identity(record):
    fields = {field: artifact_digest(value) for field, value in record.items()}
    return {'fields': fields, 'id': artifact_digest(fields)}


def resolve(path):
    path = Path(path).resolve()
    return resolved_job(files.read_yaml(path), path.parent)


def population_of(path):
    path = Path(path).resolve()
    return populate(files.read_yaml(path), path.parent)


def entries_of(exports, base, tasks):
    if not isinstance(exports, list) or not exports:
        raise ConfigError('LoRA job requires native tune exports')
    for entry in exports:
        if not isinstance(entry, dict) or set(entry) != {'path', 'export_id', 'task', 'roles'} or not isinstance(entry['path'], str) or not entry['path'] \
                or entry['task'] not in tasks or not isinstance(entry['roles'], dict) or set(entry['roles']) - set(TUNING) \
                or any(not isinstance(selectors, list) or not selectors or any(not isinstance(g, str) or not g for g in selectors) for selectors in entry['roles'].values()):
            raise ConfigError(f"each export entry names path, export_id, its task contract from tasks and roles {{role: [group selectors]}} with roles from "
                              f"{', '.join(TUNING)}; an empty roles object keeps a connectivity-only source")
    if len({entry['export_id'] for entry in exports}) != len(exports):
        raise ConfigError('the job lists one entry per immutable export; an export belongs to one task contract')
    if set(tasks) - {entry['task'] for entry in exports}:
        raise ConfigError(f"task contracts {sorted(set(tasks) - {entry['task'] for entry in exports})} have no export")
    entries = sorted(exports, key=lambda entry: entry['export_id'])
    loaded = []
    for entry in entries:
        path = str((Path(base) / entry['path']).resolve())
        manifest, exported = read_export(path, native=True)
        if manifest['export_id'] != entry['export_id']:
            raise ConfigError(f"{path}: export {entry['export_id']} changed (now {manifest['export_id']}); prepare a new job")
        loaded.append((path, manifest, exported, entry['task']))
    return entries, loaded


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def tasks_of(declared, base):
    if not isinstance(declared, dict) or not declared:
        raise ConfigError('tasks maps each frozen task contract ID to {source, bundle, weight}; prepare a new job')
    tasks = {}
    for contract, entry in sorted(declared.items()):
        if not isinstance(entry, dict) or set(entry) != {'source', 'bundle', 'weight'} or not all(isinstance(entry[key], str) and entry[key] for key in ('source', 'bundle')) \
                or not positive(entry['weight']):
            raise ConfigError(f'task {contract} requires the frozen bundle prepared from its definition, {{source, bundle, weight}} with a positive objective weight; '
                              'prepare a new job')
        frozen = frozen_task((Path(base) / entry['bundle']).resolve(), contract)
        tasks[contract] = {'frozen': frozen, 'described': describe_task(Path(frozen['definition']))['targets'], 'source': entry['source'], 'weight': entry['weight']}
    return tasks


def single_task(resolved):
    if len(resolved['tasks']) != 1:
        raise ConfigError(f"a deployment freezes one task, but this job adapts {len(resolved['tasks'])} task contracts; its candidates are initializers: "
                          'prepare a single-task job with --init-run RUN --init-candidate NAME')
    return next(iter(resolved['tasks'].items()))


def portable(resolved):
    tasks = {contract: {key: value for key, value in task.items() if key not in LOCATIONS} for contract, task in resolved['tasks'].items()}
    return {**resolved, 'tasks': tasks, 'initialization': {key: value for key, value in resolved['initialization'].items() if key not in ('run', 'null')}}


def ready_run(run, remedy):
    state, listed, fitted = (files.read_json(Path(run) / name) for name in RECORDS)
    if (state or {}).get('state') != 'ready':
        raise ConfigError(f'{run}: not a ready fitted run (state {(state or {}).get("state")}); fit with {remedy}')
    unsupported = [f'{name} schema {(record or {}).get("schema")}' for (name, schema), record in zip(RECORDS.items(), (state, listed, fitted))
                   if not isinstance(record, dict) or record.get('schema') != schema]
    if unsupported or listed.get('kind') != 'lora-candidates' or listed.get('state') != 'ready':
        raise ConfigError(f'{run}: unsupported fitting records ({", ".join(unsupported) or "candidate list not ready"}; current schemas {RECORDS}); '
                          f'refit the current job with {remedy}, or re-prepare with semantic-trainer prepare when the job itself is unsupported')
    if listed.get('loss') != LOSS:
        raise ConfigError(f"{run}: candidates were fitted under training objective {listed.get('loss')}, not {LOSS}; "
                          'refit under a new job; fit-derived selection claims cannot be relabelled')
    if not listed.get('fitted_job_id'):
        raise ConfigError(f'{run}: candidates.json records no fitted_job_id, so the fitted record cannot be proven; refit with {remedy}')
    if listed['fitted_job_id'] != artifact_digest(fitted):
        raise ConfigError(f"{run}: resolved-job.json is not the record its candidates were fitted under (fitted_job_id {listed['fitted_job_id']}); "
                          f'restore the original fitted record or refit with {remedy}; edited metadata alone does not establish changed weights')
    linked = {'run.json': [state.get(key) for key in ('job_id', 'fit_id', 'monitor_id')], 'candidates.json': [listed.get(key) for key in ('job_id', 'fit_id', 'monitor_id')]}
    if any(ids != [fitted['id'], fitted['fit']['id'], fitted['monitor']['id']] for ids in linked.values()):
        raise ConfigError(f'{run}: run.json, candidates.json and resolved-job.json name different job, fit or monitor identities; refit with {remedy}')
    return state, listed, fitted


def merged_uses(rows):
    out = {}
    for row in rows:
        known = out.setdefault(row['id'], {**row, 'uses': [], 'label_sources': []})
        known['uses'] = sorted({*known['uses'], *row['uses']}, key=USES.index)
        known['label_sources'] = list(dict.fromkeys([*known['label_sources'], *row['label_sources']]))
    return [out[key] for key in sorted(out)]


def inheritance(fitted):
    keys = ('id', 'fact', 'member', 'projection', 'group', 'component', 'family_seeds', 'sources', 'label_sources', 'task')
    direct = [{**{key: row[key] for key in keys if key in row}, 'uses': [INHERITED[row['role']]], 'inherited': fitted['id']}
              for row in fitted['memberships'] if row['role'] in INHERITED]
    return merged_uses([*direct, *fitted['initialization'].get('inherited', ())]), [*fitted['initialization'].get('overrides', ()), *fitted['corrections']]


def selection_record(path):
    record = files.read_json(path)
    if not isinstance(record, dict) or record.get('schema') != SELECTION_SCHEMA or record.get('kind') != SELECTION_KIND \
            or record.get('id') != artifact_digest({key: value for key, value in record.items() if key != 'id'}):
        raise ConfigError(f'{path}: not an intact initializer-selection record of schema {SELECTION_SCHEMA}; select with semantic-trainer select --purpose initializer')
    if record['selected']['name'] == 'base':
        raise ConfigError(f'{path}: initializer selection chose the unadapted base over every retained candidate; prepare a fresh job')
    return record


def initializer(declared, base):
    if declared == 'fresh':
        return {'kind': 'fresh'}
    shape = set(declared) if isinstance(declared, dict) else None
    if shape not in ({'run', 'candidate'}, {'selection'}) or not all(isinstance(value, str) and value for value in declared.values()):
        raise ConfigError("initialization is 'fresh', {selection} naming an initializer-selection record, or {run, candidate} naming a retained PEFT adapter "
                          'of a ready fitted run')
    record = selection_record((Path(base) / declared['selection']).resolve()) if shape == {'selection'} else None
    if record is not None:
        declared = {'run': record['run'], 'candidate': record['selected']['name']}
    run = (Path(base) / declared['run']).resolve()
    remedy = f'semantic-trainer lora on the initializer job, then name a retained candidate of {run}'
    _, listed, fitted = ready_run(run, remedy)
    found = {candidate['name']: candidate for candidate in listed['candidates']}
    if declared['candidate'] not in found or 'zero' not in found:
        raise ConfigError(f"{run}: no retained candidate {declared['candidate']!r} beside its zero adapter; retained: {sorted(found)}")
    candidate = found[declared['candidate']]
    adapter = verified(fitted, run / candidate['peft'], candidate)
    if fitted['initialization']['kind'] == 'fresh':
        null = {'run': str(run), 'peft': found['zero']['peft'], **verified(fitted, run / found['zero']['peft'], found['zero'])}
    else:
        null = fitted['initialization']['null']
    inherited, overrides = inheritance(fitted)
    if record is not None and ([record[key] for key in ('job_id', 'fitted_job_id', 'fit_id', 'monitor_id')] != [fitted['id'], listed['fitted_job_id'], fitted['fit']['id'],
                                                                                                         fitted['monitor']['id']]
                               or {key: record['selected'][key] for key in ('step', 'peft_sha256', 'config_sha256')} != {'step': candidate['step'], **{
                                   key: adapter[key] for key in ('peft_sha256', 'config_sha256')}} or record['memberships'] != inherited):
        raise ConfigError(f"initializer-selection record {record['id']} does not describe candidate {candidate['name']} of the ready run {run}; "
                          'select the initializer again or name the record of this run')
    evidence = {'kind': 'native-selection', 'record': record['id'], 'mixture': record['mixture']} if record else {'kind': 'named'}
    return {'kind': 'adapter', 'evidence': evidence, 'run': str(run), 'candidate': candidate['name'], 'path': candidate['peft'], 'step': candidate['step'], 'job_id': fitted['id'], 'fit_id': fitted['fit']['id'],
            'fitted_job_id': listed['fitted_job_id'], **adapter, 'adapter': fitted['adapter'], 'peft': fitted['dependencies']['peft'],
            'precision': fitted['fitting']['precision'], 'targets': fitted['targets'], 'recipe': {key: fitted['recipe'][key] for key in ADAPTER},
            'base': {'repo': fitted['model']['repo'], 'revision': fitted['model']['revision'], 'files': fitted['model']['files'], 'sha256': listed['base_sha256']},
            'null': null, 'inherited': inherited, 'overrides': overrides}


def compatible(initialization, model, recipe, base_sha256):
    if initialization['kind'] == 'fresh':
        return
    found = {**initialization['base'], 'recipe': initialization['recipe']}
    expected = {'repo': model['repo'], 'revision': model['revision'], 'files': model['files'], 'sha256': base_sha256, 'recipe': {key: recipe[key] for key in ADAPTER}}
    differing = sorted(key for key in expected if found[key] != expected[key])
    if differing:
        raise ConfigError(f"initializer adapter {initialization['candidate']} of {initialization['run']} differs from this job in {', '.join(differing)}: "
                          'warm start optimizes the same parameters on the same original base, never a resized or rebased adapter')


def job_envelope(job):
    if not isinstance(job, dict) or job.get('schema') != SCHEMA:
        raise ConfigError(f'LoRA job requires schema: {SCHEMA}; prepare a new job in a new directory with semantic-trainer prepare')
    if set(job) - FIELDS:
        raise ConfigError(f'unsupported job fields {sorted(set(job) - FIELDS)}; supported: {sorted(FIELDS)}')
    if job.get('loss') != LOSS:
        raise ConfigError(f"job was prepared for training objective {job.get('loss')}; the trainer fits {LOSS['name']} v{LOSS['version']}: "
                          'prepare a new job in a new directory and refit (earlier runs cannot certify this objective)')
    objective = job.get('objective')
    if objective not in ('specialization', 'instruction_transfer'):
        raise ConfigError('objective must be specialization or instruction_transfer')
    given_budget = job.get('budget', {})
    if not isinstance(given_budget, dict) or set(given_budget) - set(BUDGET):
        raise ConfigError(f'unsupported budget fields {sorted(set(given_budget) - set(BUDGET)) if isinstance(given_budget, dict) else given_budget}; supported: {sorted(BUDGET)}')
    budget = {**BUDGET, **given_budget}
    if any((budget[key] is None and key not in ('max_tokens', 'max_monitor_tokens')) or (budget[key] is not None and (type(budget[key]) is not int or budget[key] < 1)) for key in BUDGET):
        raise ConfigError(f'budget requires positive integer {", ".join(BUDGET)}')
    return objective, budget, given_budget


def pinned_base(pinned):
    if not isinstance(pinned, dict) or set(pinned) != {'manifest', 'quant'}:
        raise ConfigError('assets requires manifest and quant from the semantic-trainer assets receipt')
    located = assets.load(pinned['manifest'])
    if pinned['quant'] not in located['gguf']:
        raise ConfigError(f"asset manifest has no {pinned['quant']!r} base; available: {sorted(located['gguf'])}")
    base_model = located['gguf'][pinned['quant']]
    model = {'repo': located['repo'], 'revision': located['commit'], 'quant': pinned['quant'], 'hf': located['hf'], 'files': located['files'], 'gguf': base_model['path']}
    return located, base_model, model


def target_scopes(tasks, loaded, declared):
    if declared is not None and (not isinstance(declared, list) or not declared or len(tasks) != 1):
        raise ConfigError('scope lists the targeted facts or FACT/MEMBER addresses of a single-task job; omit it to adapt every exported target')
    scopes, bindings = {}, {}
    for contract, task in tasks.items():
        exported = sorted({'/'.join(address) for _, manifest, _, owner in loaded if owner == contract for address in manifest['scope']})
        scope = exported if declared is None else describe_task(Path(task['frozen']['definition']), declared)['scope']
        if set(scope) - set(exported):
            raise ConfigError(f'exports do not cover scoped targets {sorted(set(scope) - set(exported))}; export them with ss export --native --scope')
        bound = {task['described'][address]['binding'] for address in scope if address in task['described']}
        if len(bound) > 1:
            raise ConfigError(f'scoped targets {scope} are bound to different model roles {sorted(bound)}; an adapter serves one binding, so prepare one job per binding scope')
        scopes[contract], bindings[contract] = scope, bound.pop() if bound else None
    return scopes, bindings


def frozen_revisions(rows, tasks, scopes, pinned, base_model):
    revisions = {contract: {} for contract in tasks}
    for row in rows.values():
        address, scope, task = '/'.join(record_path(row)), scopes[row['task']], tasks[row['task']]
        definition, described = task['frozen']['definition'], task['described']
        if address not in scope:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} lies outside the declared target scope {scope}; export only the scoped targets")
        if address not in described:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} is not in the task definition {definition}; re-export")
        if described[address]['meaning_rev'] != row['meaning_rev']:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} changed or left the task definition {definition} since export; re-export")
        if task['frozen']['targets'][address]['guidance'] != row['guidance_digest']:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} was exported under other guidance than the frozen task; export and prepare from one definition")
        revisions[row['task']][address] = row['meaning_rev']
        if row['native']['judge'].get('sha256') != base_model['sha256']:
            raise ConfigError(f"native export is from another base; export with the pipeline-built {pinned['quant']} base {base_model['path']}")
    return revisions


def assigned_population(rows, families, loaded, entries, objective, scopes, declared):
    assign(rows, selected(loaded, entries))
    separated(rows, families)
    assigned = [row for row in rows.values() if row['role'] != 'unused']
    task_roles = {}
    for row in assigned:
        task_roles.setdefault(row['task_family'], set()).add(row['role'])
    if objective == 'instruction_transfer' and any('fit' in values and len(values) > 1 for values in task_roles.values()):
        raise ConfigError('target policy family overlaps predictor-fit policy family')
    fitting = [row for row in assigned if row['role'] == 'fit']
    selecting = [row for row in assigned if row['role'] == 'model_select']
    if not fitting:
        raise ConfigError('unsplittable cohort: no predictor-fit observations')
    if not selecting:
        raise ConfigError('no model_select observations: assign model_select families (role: model_select) and re-export; monitoring and selection need them')
    support = fit_support(assigned, objective)
    situations = situations_of(loaded, {row['id']: row for row in assigned}, scopes)
    return assigned, fitting, selecting, support, situations


def fit_population(job, fitting):
    chosen, covered = {row['component'] for row in fitting}, {s for row in fitting for s in row['situations']}
    selection = job.get('selection', {'fit_situations': None, 'families': sorted(chosen), 'situation_ids': sorted(covered)})
    if not isinstance(selection, dict) or set(selection) != {'fit_situations', 'families', 'situation_ids'}:
        raise ConfigError('selection records fit_situations, families and situation_ids')
    if chosen != set(selection['families']) or covered != set(selection['situation_ids']) or selection['fit_situations'] not in (None, len(covered)):
        raise ConfigError(f"fit observations no longer cover exactly the selected {len(selection['families'])} families and {len(selection['situation_ids'])} "
                          'situations; prepare a new job')
    return selection


def numerical_work(recipe, budget, assigned, fitting, selecting, situations):
    if recipe['selection_unit'] == 'complete_situation':
        incomplete = sorted(identifier for identifier, entry in situations.items() if entry['role'] == 'model_select' and not entry['complete'])
        uncovered = sorted(row['id'] for row in selecting if not any(situations[s]['complete'] for s in row['situations']))
        if incomplete or uncovered:
            raise ConfigError(f'complete_situation selection needs gold-complete model_select situations; incomplete: {incomplete[:5]}, '
                              f'observations outside complete situations: {uncovered[:5]}')
    long = [row['id'] for row in assigned if max(lengths(row)) > budget['max_sequence_tokens']]
    if long:
        raise ConfigError(f'{len(long)} presentations exceed max_sequence_tokens {budget["max_sequence_tokens"]}; no implicit truncation or subsampling')
    oversized = [row['id'] for row in assigned if max(lengths(row)) > recipe['max_padded_tokens']]
    if oversized:
        raise ConfigError(f'{len(oversized)} presentations exceed max_padded_tokens {recipe["max_padded_tokens"]}; no implicit truncation or subsampling')
    probe = sorted(fitting, key=lambda row: artifact_digest([recipe['seed'], 'probe', row['id']]))[:recipe['fit_probe_observations']]
    work = {'training_tokens': recipe['epochs'] * sum(map(tokens, fitting)), 'max_sequence_tokens': max(length for row in assigned for length in lengths(row)),
            'point_tokens': sum(map(tokens, selecting)) + sum(map(tokens, probe))}
    if budget['max_tokens'] is not None and work['training_tokens'] > budget['max_tokens']:
        raise ConfigError(f"predicted token work {work} exceeds max_tokens {budget['max_tokens']}; no implicit truncation or subsampling")
    return probe, work


def monitor_work(population, cadence):
    points = evaluation_points(population['populations']['fit']['observations'], {**population['recipe'], CADENCE: cadence})
    work = {**population['work'], 'monitor_tokens': points * population['work']['point_tokens'], 'evaluation_points': points}
    if population['budget']['max_monitor_tokens'] is not None and work['monitor_tokens'] > population['budget']['max_monitor_tokens']:
        raise ConfigError(f"predicted token work {work} at cadence {cadence} exceeds max_monitor_tokens {population['budget']['max_monitor_tokens']}; "
                          'no implicit truncation or subsampling')
    return work


def role_population(assigned, situations, role):
    members = [row for row in assigned if row['role'] == role]
    linked = {row['id'] for row in members}
    entries = [entry for entry in situations.values() if entry['role'] == role]
    return {'situations': len(entries), 'complete_situations': sum(entry['complete'] for entry in entries),
            'targets': sum(value in linked for entry in entries for value in entry['targets'].values()), 'observations': len(members),
            'presentations': sum(len(row['presentations']) for row in members), 'families': len({row['component'] for row in members}),
            'source_components': len({(source['export_id'], source['component']) for row in members for source in row['sources']})}


def populate(job, base):
    objective, budget, given_budget = job_envelope(job)
    tasks = tasks_of(job.get('tasks'), base)
    initialization = initializer(job.get('initialization'), base)
    pinned = job.get('assets')
    located, base_model, model = pinned_base(pinned)
    placement = fitting_settings(job.get('fitting'))
    entries, loaded = entries_of(job.get('exports'), base, tasks)
    if 'mixture' in job:
        from .mixture import bundled
        audit = {'mixture': bundled(job['mixture'], base, tasks, entries)}
    else:
        audit = {}
    rows, families = cohort(loaded, initialization.get('inherited', ()))
    declared = job.get('scope')
    scopes, bindings = target_scopes(tasks, loaded, declared)
    revisions = frozen_revisions(rows, tasks, scopes, pinned, base_model)
    assigned, fitting, selecting, support, situations = assigned_population(rows, families, loaded, entries, objective, scopes, declared)
    selection = fit_population(job, fitting)
    given = {key: value for key, value in (job.get('recipe') or {}).items() if key != CADENCE}
    recipe = recipe_of(given, len(fitting))
    schedule = schedule_of(recipe, given, len(fitting))
    compatible(initialization, model, recipe, base_model['sha256'])
    behavior_rows(assigned, recipe)
    weights = objective_weights(assigned, tasks, recipe['objective_balance'])
    support_objective(support, assigned)
    support['slices'] = behavior_support(assigned, recipe)
    probe, work = numerical_work(recipe, budget, assigned, fitting, selecting, situations)
    executions = {artifact_digest(native['execution']): native['execution'] for native in {id(row['native']): row['native'] for row in assigned}.values()}
    if len(executions) != 1:
        raise ConfigError('exports were made under different native execution records; re-export every source with one runtime configuration')
    settings = job.get('native', {})
    if not isinstance(settings, dict) or set(settings) - set(NATIVE_SETTINGS):
        raise ConfigError(f'native accepts {", ".join(NATIVE_SETTINGS)}')
    ordered = sorted(rows.values(), key=lambda row: row['id'])
    population = {'schema': RESOLVED_SCHEMA, 'loss': LOSS, 'objective': objective, 'assets': pinned, 'model': model, 'toolchain': located['toolchain'],
                  'tasks': {contract: {**{key: value for key, value in task['frozen'].items() if key != 'overrides'}, 'source': task['source'], 'weight': task['weight'],
                                       'sha256': task['frozen']['files']['task.yaml'], 'revisions': dict(sorted(revisions[contract].items()))}
                            for contract, task in tasks.items()},
                  'corrections': [{**item, 'task': contract} for contract, task in tasks.items()
                                  for item in override_population(task['frozen']['overrides'], rows)],
                  'scope': {contract: {'targets': scope, 'whole_task': set(scope) == set(tasks[contract]['described']), 'binding': bindings[contract]}
                            for contract, scope in scopes.items()},
                  'initialization': initialization, 'objective_weights': weights, **audit,
                  'execution': next(iter(executions.values())), 'fitting': placement, 'recipe': recipe, 'budget': budget,
                  'work': work, 'schedule': schedule, 'overrides': {'recipe': given, 'budget': given_budget}, 'selection': selection,
                  'sources': [{'path': path, 'export_id': manifest['export_id'], 'task': task, 'task_family': source_family(path, manifest), 'roles': entry['roles']}
                              for (path, manifest, exported, task), entry in zip(loaded, entries)],
                  'cohort': {'revision': FAMILIES, 'observations': len(rows), 'families': len(families),
                             'memberships': artifact_digest([[row['id'], row['component'], row['role']] for row in ordered])},
                  'support': support, 'populations': {role: role_population(assigned, situations, role) for role in TUNING},
                  'probe_ids': [row['id'] for row in probe], 'situations': situations,
                  'memberships': [{k: row[k] for k in ('id', 'task', 'component', 'role', 'group', 'family_seeds', 'projection', 'task_family', 'sources', 'situations')} |
                                  path_record(record_path(row)) | {'label_sources': [LabelSource.parse(s).digest for s in row['label_sources']]} for row in ordered],
                  'policy_families': sorted({row['task_family'] for row in fitting}),
                  'native': native_settings(Model(model['gguf'], **{'device': 'auto', 'threads': 4, **settings}))}
    native_model(population)
    native_fit(assigned, population['native'])
    population['fit'] = identity(fit_inputs(population, fitting))
    population['id'] = artifact_digest(portable(population))
    return population, assigned


def native_fit(rows, native):
    caps = (('prefix', native['max_prefix_tokens']), ('input', native['max_input_tokens']), ('question', MAX_QUESTION_TOKENS))
    for row in rows:
        for entry in row['presentations']:
            for segment, cap in (*caps, ('total', native['n_ctx'])):
                size = entry.length if segment == 'total' else len(getattr(entry, segment))
                if size > cap:
                    raise ConfigError(f"task {row['task']}, target {'/'.join(record_path(row))}, observation {row['id']}, {entry.stage} presentation: "
                                      f"{segment} segment is {size} tokens, above the native {segment} limit {cap}; raise the job's native settings "
                                      "explicitly or shorten the task; no truncation")


def finalize(population, assigned, cadence, profile=None, base=None):
    by_id = {row['id']: row for row in assigned}
    selecting = [row for row in assigned if row['role'] == 'model_select']
    resolved = {**{key: value for key, value in population.items() if key != 'id'}, 'population': population['id'],
                'recipe': {**population['recipe'], CADENCE: cadence}, 'work': monitor_work(population, cadence),
                'overrides': {**population['overrides'], 'recipe': {**population['overrides']['recipe'], CADENCE: cadence}}}
    resolved['monitor'] = identity(monitor_plan(resolved, selecting, [by_id[identifier] for identifier in population['probe_ids']]))
    if profile is not None:
        resolved['cadence'] = profiled_cadence(profile, population, base, cadence)
    resolved['id'] = artifact_digest(portable(resolved))
    return resolved


def resolved_job(job, base):
    population, assigned = populate(job, base)
    return finalize(population, assigned, cadence_of(job.get('recipe')), job.get('profile'), base), assigned


def population_job(job, base):
    tasks = {contract: {**task, 'bundle': str((Path(base) / task['bundle']).resolve())} for contract, task in job['tasks'].items()}
    found = {**{key: value for key, value in job.items() if key != 'profile'}, 'tasks': tasks,
             'exports': [{**entry, 'path': str((Path(base) / entry['path']).resolve())} for entry in job['exports']],
             'recipe': {key: value for key, value in (job.get('recipe') or {}).items() if key != CADENCE}}
    if isinstance(job['initialization'], dict):
        found['initialization'] = {key: str((Path(base) / value).resolve()) if key in ('run', 'selection') else value
                                   for key, value in job['initialization'].items()}
    if 'mixture' in job:
        found['mixture'] = {**job['mixture'], 'audit': str((Path(base) / job['mixture']['audit']).resolve())}
    return {key: value for key, value in found.items() if key != 'recipe' or value}


def final_job(job, base, cadence, profile, to):
    found = population_job(job, base)
    result = {**found, 'recipe': {**found.get('recipe', {}), CADENCE: cadence}}
    if profile is not None:
        record = profile_record(profile)
        files.write_new(Path(to) / 'profile.json', record)
        result['profile'] = {'file': 'profile.json', 'id': record['id']}
    return result


def override_population(overrides, rows):
    families = {tuple(node): row['component'] for row in rows.values() for node in row['family_seeds']}
    population = []
    for item in overrides:
        observed = rows.get(item['observation'])
        if observed is not None and (record_path(observed) != record_path(item) or observed['projection'] != item['projection'] or observed['value'] != item['value']):
            raise ConfigError(f"exact correction {item['example']} for {'/'.join(record_path(item))} contradicts exported observation {item['observation']} "
                              f"({observed['value']!r}); reconcile the correction or the label and prepare a new job")
        touched = sorted({families[tuple(node)] for node in item['family_seeds'] if tuple(node) in families})
        component = observed['component'] if observed is not None else touched[0] if len(touched) == 1 else artifact_digest([FAMILIES, touched or item['family_seeds']])
        population.append({'id': item['observation'], **path_record(record_path(item)), 'projection': item['projection'], 'value': item['value'],
                           'example': item['example'], 'group': component, 'component': component, 'family_seeds': item['family_seeds'],
                           'label_sources': item['label_sources'], 'observed': observed is not None})
    return population


def role_rows(exported):
    return [(role, row) for row in exported if (role := role_of(row['split'], row['fold'])) in TUNING]


def fit_selection(rows, count, seed):
    situations = {}
    for row in rows.values():
        if row['role'] == 'fit':
            situations.setdefault(row['component'], set()).update(row['situations'])
    chosen = set(situations) if count is None else exact_families({family: len(found) for family, found in situations.items()}, count, seed)
    for row in rows.values():
        if row['role'] == 'fit' and row['component'] not in chosen:
            row['role'] = 'unused'
    return {'fit_situations': count, 'families': sorted(chosen), 'situation_ids': sorted({s for family in chosen for s in situations[family]})}


def prepare(definition, exports, to, *, weight=1, **settings):
    return prepare_tasks([{'definition': definition, 'exports': exports, 'weight': weight, 'contract': None}], to, **settings)


def prepare_tasks(sources, to, **settings):
    return prepared_tasks(sources, to, **settings)[0]


def prepared_tasks(sources, to, *, assets_manifest, quant='Q4_0', device='cpu', objective='specialization', budget=None, recipe=None,
                   native=None, acceptance_fit=False, fit_situations=None, profile=None, scope=None, initialization='fresh', mixture=None):
    to = Path(to).resolve()
    job_path = to / 'job.yaml'
    if job_path.exists():
        raise ConfigError(f'{job_path}: families were already assigned to roles there; prepare a new job in a new directory')
    if not sources or any(not entry['exports'] for entry in sources):
        raise ConfigError('prepare requires native tune exports for every task: ss export DEFINITION --native --task-family NAMESPACE:NAME --labels FILE')
    if fit_situations is not None and (type(fit_situations) is not int or fit_situations < 1):
        raise ConfigError('fit_situations must be a positive integer')
    if isinstance(initialization, dict):
        initialization = {key: str(Path(value).resolve()) if key in ('run', 'selection') else value for key, value in initialization.items()}
    bundles, copied, audited = to / 'tasks', to / 'profile.json', to / 'mixture'
    for stale in (bundles, copied, audited):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    try:
        tasks, loaded = {}, {}
        for index, entry in enumerate(sources):
            staging = bundles / f'.staging-{index}'
            frozen = freeze_task(Path(entry['definition']).resolve(), staging)
            if entry['contract'] not in (None, frozen['behavior']):
                raise ConfigError(f"{entry['definition']} now freezes to task contract {frozen['behavior']}, not the declared {entry['contract']}")
            if frozen['behavior'] in tasks:
                raise ConfigError(f"{entry['definition']} freezes to the same task contract {frozen['behavior']} as another task; one contract is one task")
            staging.rename(bundles / frozen['behavior'])
            tasks[frozen['behavior']] = {'source': str(Path(entry['definition']).resolve()), 'bundle': f"tasks/{frozen['behavior']}", 'weight': entry['weight']}
            for item in load((str(Path(export).resolve()) for export in entry['exports']), frozen['behavior']):
                if item[1]['export_id'] in loaded:
                    raise ConfigError(f"{item[0]}: export {item[1]['export_id']} is listed under two task contracts; an export belongs to one task")
                loaded[item[1]['export_id']] = item
        loaded = [loaded[key] for key in sorted(loaded)]
        rows, families = cohort(loaded, initializer(initialization, to).get('inherited', ()))
        assign(rows, intended(rows, acceptance_fit))
        separated(rows, families)
        selection = fit_selection(rows, fit_situations, (recipe or {}).get('seed', DEFAULTS['seed']))
        given = dict(recipe or {})
        if profile is not None:
            cadence_of(given)
            record = profile_record(profile)
            files.write_new(copied, record)
        if mixture is not None:
            for relative, source in mixture['copies'].items():
                (audited / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, audited / relative)
            files.write_new(audited / 'audit.json', mixture['audit'])
        job = {'schema': SCHEMA, 'loss': LOSS, 'objective': objective, 'tasks': tasks, 'initialization': initialization,
               'assets': {'manifest': assets_manifest, 'quant': quant}, 'fitting': {'device': device}, 'exports': exported_roles(loaded, rows),
               'selection': selection, **({'budget': dict(budget)} if budget else {}), **({'recipe': given} if given else {}),
               **({'native': native} if native else {}), **({'profile': {'file': copied.name, 'id': record['id']}} if profile is not None else {}),
               **({'scope': list(scope)} if scope else {}), **({'mixture': {'audit': 'mixture/audit.json', 'id': mixture['audit']['id']}} if mixture else {})}
        population, assigned = populate(job, to)
        resolved = finalize(population, assigned, cadence_of(given), job.get('profile'), to) if CADENCE in given else None
    except BaseException:
        shutil.rmtree(bundles, ignore_errors=True)
        shutil.rmtree(audited, ignore_errors=True)
        copied.unlink(missing_ok=True)
        raise
    files.publish(job_path, job)
    run, final = str(to / 'run'), resolved or population
    summary = {'job': str(job_path), 'population_id': population['id'], 'fit_id': population['fit']['id'], 'cohort': population['cohort'],
               'tasks': {contract: {'bundle': task['bundle'], 'weight': task['weight'], 'overrides': sum(item['task'] == contract for item in population['corrections'])}
                         for contract, task in population['tasks'].items()},
               'scope': population['scope'], 'initialization': {key: len(value) if key in ('inherited', 'overrides') else value
                                                                for key, value in population['initialization'].items() if key != 'null'},
               'roles': population['populations'], 'support': population['support'], 'objective_weights': population['objective_weights'], 'selection': selection,
               'recipe': final['recipe'], 'schedule': population['schedule'], 'work': final['work'], 'loss': LOSS,
               'commands': {'profile': ['semantic-trainer', 'profile', str(job_path), '--to', 'NEW_PROFILE']}}
    if resolved is None:
        return {**summary, 'job_id': None, 'monitor_id': None, 'cadence': None}, population, assigned
    return {**summary, 'job_id': resolved['id'], 'monitor_id': resolved['monitor']['id'], 'cadence': resolved.get('cadence'),
            'commands': {**summary['commands'], 'dry_run': ['semantic-trainer', 'lora', str(job_path), '--to', run, '--dry-run'],
                         'fit': ['semantic-trainer', 'lora', str(job_path), '--to', run],
                         'select': ['semantic-trainer', 'select', str(job_path), '--run', run]}}, population, assigned


def profile_configuration(resolved):
    recipe = resolved['recipe']
    return {'assets': resolved['assets'], 'fitting': resolved['fitting'], 'loss': resolved['loss'], 'implementation': FITTING,
            'adapter': {key: recipe[key] for key in ADAPTER},
            'batching': {key: recipe[key] for key in ('batch_size', 'microbatch', 'max_padded_tokens', 'evaluation_batch_size', 'fit_probe_observations')}, 'max_sequence_tokens': resolved['budget']['max_sequence_tokens']}


def profile_workload(resolved):
    fit = resolved['populations']['fit']
    return {'point_tokens': resolved['work']['point_tokens'], 'fit_tokens_per_observation': resolved['work']['training_tokens'] / resolved['recipe']['epochs'] / fit['observations'],
            'fit_observations': fit['observations']}


def projection(population, costs, cadence):
    recipe, work, budget = {**population['recipe'], CADENCE: cadence}, population['work'], population['budget']
    points = evaluation_points(population['populations']['fit']['observations'], recipe)
    monitor_tokens = points * work['point_tokens']
    seconds = {'loading_seconds': costs['loading_seconds'], 'training_seconds': work['training_tokens'] * costs['update_seconds_per_token'],
               'monitoring_seconds': monitor_tokens * costs['point_seconds_per_token'], 'checkpoint_seconds': points * costs['checkpoint_seconds']}
    total = sum(seconds.values())
    return {CADENCE: cadence, 'evaluation_points': points, 'optimizer_steps': population['schedule']['max_optimizer_steps'],
            'training_tokens': work['training_tokens'], 'monitor_tokens': monitor_tokens, **seconds, 'seconds': total,
            'max_seconds': budget['max_seconds'], 'max_tokens': budget['max_tokens'], 'max_monitor_tokens': budget['max_monitor_tokens'],
            'feasible': total <= budget['max_seconds'] and (budget['max_tokens'] is None or work['training_tokens'] <= budget['max_tokens']) and (budget['max_monitor_tokens'] is None or monitor_tokens <= budget['max_monitor_tokens'])}


def profile_record(path):
    record = files.read_json(path)
    if not isinstance(record, dict) or record.get('schema') != PROFILE_SCHEMA or record.get('kind') != PROFILE_KIND \
            or record.get('id') != artifact_digest({key: value for key, value in record.items() if key != 'id'}):
        raise ConfigError(f'{path}: not an intact fitting profile of schema {PROFILE_SCHEMA}; run semantic-trainer profile JOB --to NEW_PROFILE')
    return record


def profile_applies(record, resolved):
    if record['configuration'] != profile_configuration(resolved):
        differing = sorted(key for key, value in profile_configuration(resolved).items() if record['configuration'].get(key) != value)
        raise ConfigError(f'fitting profile {record["id"]} measured another fitting configuration ({", ".join(differing)}); profile this configuration')
    current = profile_workload(resolved)
    ratios = {key: current[key] / record['workload'][key] for key in ('point_tokens', 'fit_tokens_per_observation')}
    if any(abs(ratio - 1) > PROFILE_TOLERANCE for ratio in ratios.values()):
        raise ConfigError(f'token workload changed by {ratios} since profile {record["id"]}; the measured costs no longer describe this job: re-profile')
    return ratios


def profiled_cadence(declared, population, base, cadence):
    if not isinstance(declared, dict) or set(declared) != {'file', 'id'}:
        raise ConfigError('profile records the copied fitting profile {file, id}; prepare with --profile PROFILE')
    record = profile_record(Path(base) / declared['file'])
    if record['id'] != declared['id']:
        raise ConfigError(f"{declared['file']}: fitting profile changed since preparation; prepare a new job")
    ratios = profile_applies(record, population)
    projected = feasible_cadence(population, record, cadence)
    return {CADENCE: cadence, 'profile': record['id'], 'workload_ratio': ratios, 'projection': projected}


def feasible_cadence(population, record, cadence):
    projected = projection(population, record['costs'], cadence)
    return projected


def ranking_rule(points, unit):
    losses = {point.get('nll') is not None for point in points}
    if len(losses) > 1:
        raise ConfigError('ranking mixes points with and without model_select NLL')
    order = [field for field in RANKING if field != 'nll' or losses == {True}]
    if any(point.get('qualification', {}).get('state', 'not_declared') != 'not_declared' for point in points):
        order.insert(0, 'qualified')
    return {'unit': unit, 'order': order, 'nll': 'available' if 'nll' in order else 'unavailable'}


def rank_value(point, unit, field):
    return {'qualified': lambda: point.get('qualification', {}).get('state') != 'met',
            'correct': lambda: -point['correct']['macro'][unit], 'regressions': lambda: len(point.get('regressions', ())),
            'nll': lambda: point['nll'], 'base': lambda: not point.get('base', False), 'step': lambda: point['step']}[field]()


def ranking(points, unit):
    order = ranking_rule(points, unit)['order']
    return sorted(points, key=lambda point: tuple(rank_value(point, unit, field) for field in order))


def selection_reason(points, unit, selected):
    ordered = ranking(points, unit)
    winner = next(point for point in ordered if point['name'] == selected)
    other = next((point for point in ordered if point['name'] != selected), None)
    field = next((field for field in ranking_rule(points, unit)['order']
                  if other is not None and rank_value(winner, unit, field) != rank_value(other, unit, field)), None)
    return {'selected': selected, 'compared_with': other['name'] if other else None, 'first_differing_key': field,
            'selected_value': rank_value(winner, unit, field) if field else None,
            'alternative_value': rank_value(other, unit, field) if field else None,
            'rule': ranking_rule(points, unit), 'accepted_errors_determine_ranking': False}


DENOMINATORS = {'observation': 'observations', 'complete_situation': 'situations'}


def unit_correct(correct, situations, role, tasks):
    per = {}
    def counts(task):
        return per.setdefault(task, {'observation': 0, 'complete_situation': 0, 'situations': 0, 'observations': 0})
    for identifier, value in correct.items():
        found = counts(tasks[identifier])
        found['observation'] += bool(value)
        found['observations'] += 1
    for entry in situations.values():
        if entry['role'] == role and entry['complete']:
            found = counts(entry['task'])
            found['complete_situation'] += all(correct.get(value) is True for value in entry['targets'].values() if value != 'exact')
            found['situations'] += 1
    macro = {}
    for unit, denominator in DENOMINATORS.items():
        fractions = [Fraction(found[unit], found[denominator]) for found in per.values() if found[denominator]]
        macro[unit] = float(sum(fractions) / len(fractions)) if fractions else None
    return {**{key: sum(found[key] for found in per.values()) for key in ('observation', 'complete_situation', 'situations', 'observations')},
            'macro': macro, 'tasks': dict(sorted(per.items()))}
