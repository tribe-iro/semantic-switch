from __future__ import annotations

from fractions import Fraction
import math
import os
import re
import shutil
from pathlib import Path

from semantic_switch import Model, files
from semantic_switch.api import NATIVE_SETTINGS, native_settings
from semantic_switch.artifacts import artifact_digest, native_model, read_export
from semantic_switch.errors import ConfigError
from semantic_switch.families import FamilyIndex
from semantic_switch.examples import ROLES, LabelSource, merge_sources, path_record, record_path, role_of
from semantic_switch.training import describe_task, freeze_task, frozen_task

from . import assets

SCHEMA, RESOLVED_SCHEMA, FAMILIES, FITTING, IDENTITY = 7, 8, 1, 1, 2
LOSS = {'name': 'full_vocabulary_group_nll', 'version': 2, 'reduction': 'float32', 'expression': 'softplus(logsumexp(z[not G]) - logsumexp(z[G]))'}
MODULES = {'q_proj': ('self_attn', 'attn_q'), 'k_proj': ('self_attn', 'attn_k'), 'v_proj': ('self_attn', 'attn_v'), 'o_proj': ('self_attn', 'attn_output'),
           'gate_proj': ('mlp', 'ffn_gate'), 'up_proj': ('mlp', 'ffn_up'), 'down_proj': ('mlp', 'ffn_down')}
DEFAULTS = {'rank': 16, 'alpha': 32, 'dropout': 0., 'learning_rate': 2e-4, 'schedule': 'constant', 'weight_decay': .01, 'gradient_clip': 1.,
            'epochs': 8, 'min_epochs': 2, 'seed': 0, 'batch_size': 8, 'microbatch': 1, 'target_modules': list(MODULES), 'warmup_examples': None,
            'eval_every_examples': None, 'patience_examples': None, 'min_improvement': .001, 'selection_unit': 'observation'}
WARMUP_REFERENCE_EPOCHS, WARMUP_FRACTION = 3, Fraction(1, 20)
UNITS = ('observation', 'complete_situation')
FIELDS = {'schema', 'loss', 'objective', 'task', 'assets', 'fitting', 'exports', 'selection', 'recipe', 'budget', 'native', 'profile'}
PROFILE_SCHEMA, PROFILE_KIND, PROFILE_TOLERANCE = 1, 'fitting-profile', .5
TUNING = tuple(role for role, (split, _) in ROLES.items() if split == 'tune')
BUDGET = {'max_tokens': 2000000, 'max_monitor_tokens': 20000000, 'max_sequence_tokens': 1024, 'max_seconds': 7200, 'feasibility_updates': 8}
RANKING = {'correct': 'more raw-correct model_select units of the declared selection unit', 'regressions': 'fewer path/class regressions against base',
           'nll': 'lower model_select full-vocabulary NLL', 'base': 'base before adapted candidates', 'step': 'earlier step'}
SHARED = ('key', 'observation', 'rendered', 'situation')
ORDER = 'per-epoch random.Random(artifact_digest([seed, "epoch", epoch])).shuffle of the resolved fit order, consecutive batches'
DEADLINE = {'max_seconds': 'elapsed compute-run time from job resolution: tokenizer/config loading, admission waiting, native baseline scoring, HF loading, '
                           'optimizer updates, monitoring and checkpoint retention; final publication and cleanup are reported separately',
            'polling': 'before each expensive stage and at its completion boundary (native baseline, model loading, optimizer update, evaluation row, '
                       'monitor point, checkpoint retention); cooperative, not a hard wall-clock limit'}
REDUCTION = {'revision': 1, 'answer': 'first canonical-variant presentation whose slot_map maps the argmax answer group to a global label',
             'stages': 'per-stage canonical-variant slot_map answer', 'weights': 'uniform over stages, uniform over presentations within a stage'}
LOCATIONS = ('bundle', 'definition')
PRESENTED = ('stage', 'variant', 'prefix', 'input', 'question', 'answer_slots', 'label', 'slot_map', 'global_label')
INSTALL = 'install the torch wheel for that device in the trainer environment (packages/semantic-trainer/README.md)'


def physical_cores():
    cores = {path.read_text() for path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology/core_cpus_list')}
    return len(cores) or os.cpu_count() or 1


def fitting_settings(settings):
    if not isinstance(settings, dict) or set(settings) - {'device', 'precision', 'threads', 'attention', 'gradient_checkpointing'}:
        raise ConfigError('fitting accepts device, precision, threads, attention and gradient_checkpointing')
    device = settings.get('device')
    match = re.fullmatch(r'cpu|(xpu|cuda):(\d+)', device) if isinstance(device, str) else None
    if not match:
        raise ConfigError('fitting.device must be cpu, xpu:N or cuda:N')
    try:
        import torch
    except ImportError as error:
        raise ConfigError(f'fitting requires torch; {INSTALL}') from error
    if match[1]:
        backend = getattr(torch, match[1], None)
        count = backend.device_count() if backend is not None and backend.is_available() else 0
        if int(match[2]) >= count:
            raise ConfigError(f'{device} is unavailable to torch {torch.__version__} ({count} {match[1]} devices); {INSTALL}')
    allowed = ('float32', 'bfloat16') if device == 'cpu' else ('bfloat16',)
    resolved = {'device': device, 'precision': settings.get('precision', allowed[0]), 'threads': settings.get('threads', physical_cores()),
                'attention': settings.get('attention', 'eager'), 'gradient_checkpointing': settings.get('gradient_checkpointing', False)}
    if resolved['precision'] not in allowed:
        raise ConfigError(f"fitting.precision on {device} must be {' or '.join(allowed)}")
    if type(resolved['threads']) is not int or resolved['threads'] < 1:
        raise ConfigError('fitting.threads must be a positive integer')
    if resolved['attention'] not in ('eager', 'sdpa'):
        raise ConfigError('fitting.attention must be eager or sdpa')
    if type(resolved['gradient_checkpointing']) is not bool:
        raise ConfigError('fitting.gradient_checkpointing must be boolean')
    return resolved


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def recipe_of(given, fit_count):
    given = given or {}
    if not isinstance(given, dict) or set(given) - set(DEFAULTS):
        raise ConfigError(f'unsupported recipe fields {sorted(set(given) - set(DEFAULTS))}; supported: {sorted(DEFAULTS)}')
    if any(value is None for value in given.values()):
        raise ConfigError(f'recipe overrides need values: {sorted(key for key, value in given.items() if value is None)}')
    if 'eval_every_examples' not in given:
        raise ConfigError('choose the monitoring cadence explicitly: --recipe eval_every_examples=N (examples seen between evaluation points)')
    recipe = {**DEFAULTS, **given}
    derived = {'warmup_examples': ceil_div(WARMUP_REFERENCE_EPOCHS * fit_count * WARMUP_FRACTION.numerator, WARMUP_FRACTION.denominator),
               'patience_examples': fit_count}
    recipe |= {key: value for key, value in derived.items() if recipe[key] is None}
    for key in ('rank', 'alpha', 'batch_size', 'microbatch', 'epochs', 'min_epochs', 'eval_every_examples', 'patience_examples'):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ConfigError(f'{key} must be a positive integer')
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
    for key in ('learning_rate', 'gradient_clip', 'dropout', 'weight_decay', 'min_improvement'):
        if type(recipe[key]) not in (int, float) or not math.isfinite(recipe[key]):
            raise ConfigError(f'{key} must be finite')
    if recipe['learning_rate'] <= 0 or recipe['gradient_clip'] <= 0 or not 0 <= recipe['dropout'] < 1 or recipe['weight_decay'] < 0 or not 0 < recipe['min_improvement'] < 1:
        raise ConfigError('invalid learning rate, gradient cap, dropout, weight decay or relative min_improvement (0, 1)')
    return recipe


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
            'decay_horizon_steps': total if recipe['schedule'] == 'cosine' else None,
            'unit': 'exported target observations consumed by completed optimizer steps'}


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


def answer_groups(entry, vocabulary):
    groups, gold = entry['answer_slots'], entry['label']
    tokens = [token for group in groups for token in group]
    if not groups or not all(groups):
        raise ConfigError('answer groups must be nonempty')
    if any(type(token) is not int or not 0 <= token < vocabulary for token in tokens):
        raise ConfigError(f'answer token outside the {vocabulary}-token vocabulary')
    if len(set(tokens)) != len(tokens):
        raise ConfigError('an answer token is repeated or shared by different answers; re-export with disjoint, deduplicated answer groups')
    if type(gold) is not int or not 0 <= gold < len(groups):
        raise ConfigError('gold answer group is absent')
    return groups


def evidence(row):
    return (row['meaning_rev'], row['classes'], row['label'], row['text'], row['projection'], [entry['id'] for entry in row['native']['presentations']])


def lengths(row):
    return [len(p['prefix']) + len(p['input']) + len(p['question']) for p in row['native']['presentations']]


def tokens(row):
    return sum(lengths(row))


def source_family(path, exported):
    families = {row.get('task_family') for row in exported}
    if len(families) != 1 or not isinstance(next(iter(families)), str) or ':' not in next(iter(families)):
        raise ConfigError(f'{path}: a native export carries exactly one namespaced task family (ss export --native --task-family NAMESPACE:NAME); found {sorted(map(str, families))}')
    return families.pop()


def fit_support(rows, objective):
    support, declared = {}, {}
    for row in rows:
        path = '/'.join(record_path(row))
        declared[path] = row['classes']
        for tag in row['tags'] or ['untagged']:
            classes = support.setdefault(row['role'], {}).setdefault(path, {key: {} for key in row['classes']})
            label = row['classes'][row['label']]
            classes[label][tag] = classes[label].get(tag, 0) + 1
    fitted = set(support.get('fit', {}))
    required = declared if objective == 'specialization' else {path: classes for path, classes in declared.items() if path in fitted}
    missing = {path: [key for key in classes if not support.get('fit', {}).get(path, {}).get(key)] for path, classes in required.items()}
    missing = {path: keys for path, keys in missing.items() if keys}
    if missing:
        raise ConfigError('declared answers without fit support: ' + '; '.join(f"{path}: {', '.join(keys)}" for path, keys in missing.items()) +
                          '; label situations for these answers (semantic-trainer collect or ss label) and re-export before fitting')
    return {'roles': support, 'absent_fit_support': sorted(set(declared) - fitted)}


def situations_of(manifests, observations):
    merged = {}
    for manifest in manifests:
        for situation in manifest['situations']:
            entry = merged.setdefault(situation['id'], {'targets': {}, 'records': set()})
            entry['records'].update(situation['records'])
            for target in situation['targets']:
                address = '/'.join(target['address'])
                known = entry['targets'].get(address)
                value = target.get('observation') or target['outcome']
                if 'observation' in target and isinstance(known, str) and len(known) == 64 and known != value:
                    raise ConfigError(f"situation {situation['id']}: {address} links two observations across exports")
                if known is None or 'observation' in target:
                    entry['targets'][address] = value
    scope = sorted({'/'.join(address) for manifest in manifests for address in manifest['scope']})
    out = {}
    for identifier, entry in sorted(merged.items()):
        targets = {address: entry['targets'].get(address, 'unlabelled') for address in scope}
        linked = [value for value in targets.values() if value in observations]
        roles = {observations[value]['role'] for value in linked}
        if len(roles) > 1:
            raise ConfigError(f'situation {identifier} spans roles {sorted(roles)}; its observations share one family')
        out[identifier] = {'role': roles.pop() if roles else 'unused', 'targets': targets, 'records': sorted(entry['records']),
                           'complete': all(value in observations or value == 'exact' for value in targets.values())}
    return out


def presentation_weights(row):
    counts = {}
    for entry in row['native']['presentations']:
        counts[entry['stage']] = counts.get(entry['stage'], 0) + 1
    return [1 / len(counts) / counts[entry['stage']] for entry in row['native']['presentations']]


def names(row):
    return {*row['aliases'], *(name for kind, name in row['family_seeds'] if kind == 'group')}


def load(paths):
    loaded = {}
    for path in sorted(set(paths)):
        manifest, exported = read_export(path, native=True)
        loaded.setdefault(manifest['export_id'], (path, manifest, exported))
    return [loaded[key] for key in sorted(loaded)]


def cohort(loaded):
    index, anchors, rows = FamilyIndex(), {}, {}
    for path, manifest, exported in loaded:
        task_family = source_family(path, exported)
        for row in exported:
            shared = [tuple(node) for node in row['family_seeds'] if node[0] in SHARED]
            if not shared:
                raise ConfigError(f"{path}: observation {row['id']} carries no {', '.join(SHARED)} family seed; re-export it from its dataset assignment")
            anchor = ('group', artifact_digest([manifest['export_id'], row['component']]))
            for node in (*shared, anchors.setdefault(row['id'], anchor)):
                index.join(anchor, node)
            current = rows.setdefault(row['id'], {**row, 'task_family': task_family, 'role': 'unused', 'sources': [], 'situations': [], 'label_sources': []})
            if evidence(current) != evidence(row):
                raise ConfigError(f"observation {row['id']} has different meaning, labels, evidence or native presentations across exports")
            if current['task_family'] != task_family:
                raise ConfigError(f"observation {row['id']} carries task families {current['task_family']} and {task_family}")
            current['sources'].append({'export_id': manifest['export_id'], 'path': path, 'task_family': task_family, 'namespace': manifest['namespace'],
                                       'component': row['component'], 'group': row['group'], 'aliases': row['aliases'], 'family_seeds': row['family_seeds'],
                                       'dataset_role': role_of(row['split'], row['fold'])})
            current['situations'] = sorted({*current['situations'], *row['situations']})
            current['label_sources'] = [s.record() for s in merge_sources(map(LabelSource.parse, current['label_sources']), map(LabelSource.parse, row['label_sources']))]
    closures, families = {}, {}
    for identifier, row in rows.items():
        root = index.find(anchors[identifier])
        if root not in closures:
            seeds = [list(node) for node in index.seeds(root) if node[0] in SHARED]
            closures[root] = artifact_digest([FAMILIES, seeds]), seeds
        family, seeds = closures[root]
        row.update(component=family, group=family, family_seeds=seeds)
        entry = families.setdefault(family, {'observations': [], 'situations': set(), 'seeds': seeds})
        entry['observations'].append(identifier)
        entry['situations'].update(row['situations'])
    return rows, families


def assign(rows, roles):
    for identifier, found in roles.items():
        if len(found) > 1:
            raise ConfigError(f'observation {identifier} assigned to conflicting roles {sorted(found)}')
        rows[identifier]['role'] = next(iter(found))


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


def intended(rows, acceptance_fit):
    return {identifier: found for identifier, row in rows.items()
            if (found := {source['dataset_role'] for source in row['sources'] if acceptance_fit or source['dataset_role'] != 'acceptance_fit'})}


def selected(loaded, entries):
    roles = {}
    for (path, _, exported), entry in zip(loaded, entries):
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
    for path, manifest, exported in loaded:
        roles = {}
        for row in exported:
            if rows[row['id']]['role'] == (role := role_of(row['split'], row['fold'])):
                roles.setdefault(role, set()).add(row['group'])
        entries.append({'path': path, 'export_id': manifest['export_id'], 'roles': {role: sorted(roles[role]) for role in TUNING if role in roles}})
    return entries


def numerical(row):
    return {'id': row['id'], 'kind': row['kind'], 'classes': row['classes'], 'label': row['label'], 'weights': presentation_weights(row),
            'presentations': [{key: entry[key] for key in PRESENTED} for entry in row['native']['presentations']]}


def computation(rows):
    return {'revision': IDENTITY, 'reduction': REDUCTION, 'tokenizer': artifact_digest(rows[0]['native']['tokenizer']) if rows else None}


def fit_inputs(resolved, rows):
    recipe, schedule = resolved['recipe'], resolved['schedule']
    return {**computation(rows), 'base': {**{key: resolved['model'][key] for key in ('repo', 'revision')}, 'files': resolved['model']['files']},
            'loss': resolved['loss'], 'objective': resolved['objective'], 'observations': [numerical(row) for row in rows],
            'adapter': {key: recipe[key] for key in ('rank', 'alpha', 'dropout', 'target_modules')},
            'optimizer': {'algorithm': 'AdamW', **{key: recipe[key] for key in ('learning_rate', 'weight_decay', 'gradient_clip')}},
            'schedule': {'shape': recipe['schedule'], **{key: schedule[key] for key in ('max_optimizer_steps', 'steps_per_epoch', 'warmup_steps', 'decay_horizon_steps')}},
            'order': {'algorithm': ORDER, **{key: recipe[key] for key in ('seed', 'epochs', 'batch_size', 'microbatch')}},
            'placement': resolved['fitting'], 'implementation': FITTING}


def monitor_plan(resolved, selecting, probe):
    recipe = resolved['recipe']
    return {**computation([*selecting, *probe]), 'model_select': [numerical(row) for row in selecting], 'probe': [numerical(row) for row in probe],
            'situations': {key: {'targets': value['targets'], 'complete': value['complete']} for key, value in resolved['situations'].items() if value['role'] == 'model_select'},
            'cadence': recipe['eval_every_examples'], 'evaluation_points': resolved['work']['evaluation_points'], 'unit': recipe['selection_unit'], 'ranking': RANKING,
            'loss': resolved['loss'], 'stopping': {key: recipe[key] for key in ('min_epochs', 'patience_examples', 'min_improvement')}}


def identity(record):
    fields = {field: artifact_digest(value) for field, value in record.items()}
    return {'fields': fields, 'id': artifact_digest(fields)}


def resolve(path):
    path = Path(path).resolve()
    return resolved_job(files.read_yaml(path), path.parent)


def entries_of(exports, base):
    if not isinstance(exports, list) or not exports:
        raise ConfigError('LoRA job requires native tune exports')
    for entry in exports:
        if not isinstance(entry, dict) or set(entry) != {'path', 'export_id', 'roles'} or not isinstance(entry['path'], str) or not entry['path'] \
                or not isinstance(entry['roles'], dict) or set(entry['roles']) - set(TUNING) \
                or any(not isinstance(selectors, list) or not selectors or any(not isinstance(g, str) or not g for g in selectors) for selectors in entry['roles'].values()):
            raise ConfigError(f"each export entry names path, export_id and roles {{role: [group selectors]}} with roles from {', '.join(TUNING)}; "
                              'an empty roles object keeps a connectivity-only source')
    if len({entry['export_id'] for entry in exports}) != len(exports):
        raise ConfigError('the job lists one entry per immutable export')
    entries = sorted(exports, key=lambda entry: entry['export_id'])
    loaded = []
    for entry in entries:
        path = str((Path(base) / entry['path']).resolve())
        manifest, exported = read_export(path, native=True)
        if manifest['export_id'] != entry['export_id']:
            raise ConfigError(f"{path}: export {entry['export_id']} changed (now {manifest['export_id']}); prepare a new job")
        loaded.append((path, manifest, exported))
    return entries, loaded


def resolved_job(job, base):
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
    if any(type(budget[key]) is not int or budget[key] < 1 for key in BUDGET):
        raise ConfigError(f'budget requires positive integer {", ".join(BUDGET)}')
    task = job.get('task')
    if not isinstance(task, dict) or set(task) != {'source', 'bundle', 'behavior'} or not all(isinstance(value, str) and value for value in task.values()):
        raise ConfigError('task requires the frozen task bundle prepared from its definition, {source, bundle, behavior}; prepare a new job')
    frozen = frozen_task((Path(base) / task['bundle']).resolve(), task['behavior'])
    definition = Path(frozen['definition'])
    described = describe_task(definition)['targets']
    pinned = job.get('assets')
    if not isinstance(pinned, dict) or set(pinned) != {'manifest', 'quant'}:
        raise ConfigError('assets requires manifest and quant from the semantic-trainer assets receipt')
    located = assets.load(pinned['manifest'])
    if pinned['quant'] not in located['gguf']:
        raise ConfigError(f"asset manifest has no {pinned['quant']!r} base; available: {sorted(located['gguf'])}")
    base_model = located['gguf'][pinned['quant']]
    model = {'repo': located['repo'], 'revision': located['commit'], 'quant': pinned['quant'], 'hf': located['hf'], 'files': located['files'], 'gguf': base_model['path']}
    placement = fitting_settings(job.get('fitting'))
    entries, loaded = entries_of(job.get('exports'), base)
    rows, families = cohort(loaded)
    revisions = {}
    for row in rows.values():
        address = '/'.join(record_path(row))
        if address not in described:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} is not in the task definition {definition}; re-export")
        if described[address]['meaning_rev'] != row['meaning_rev']:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} changed or left the task definition {definition} since export; re-export")
        if frozen['targets'][address]['guidance'] != row['guidance_digest']:
            raise ConfigError(f"{row['sources'][0]['path']}: {address} was exported under other guidance than the frozen task; export and prepare from one definition")
        revisions[address] = row['meaning_rev']
        if row['native']['judge'].get('sha256') != base_model['sha256']:
            raise ConfigError(f"native export is from another base; export with the pipeline-built {pinned['quant']} base {base_model['path']}")
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
    for row in assigned:
        for entry in row['native']['presentations']:
            answer_groups(entry, row['native']['tokenizer']['vocab_size'])
    support = fit_support(assigned, objective)
    situations = situations_of([manifest for _, manifest, _ in loaded], {row['id']: row for row in assigned})
    chosen, covered = {row['component'] for row in fitting}, {s for row in fitting for s in row['situations']}
    selection = job.get('selection', {'fit_situations': None, 'families': sorted(chosen), 'situation_ids': sorted(covered)})
    if not isinstance(selection, dict) or set(selection) != {'fit_situations', 'families', 'situation_ids'}:
        raise ConfigError('selection records fit_situations, families and situation_ids')
    if chosen != set(selection['families']) or covered != set(selection['situation_ids']) or selection['fit_situations'] not in (None, len(covered)):
        raise ConfigError(f"fit observations no longer cover exactly the selected {len(selection['families'])} families and {len(selection['situation_ids'])} "
                          'situations; prepare a new job')
    given = job.get('recipe') or {}
    recipe = recipe_of(given, len(fitting))
    schedule = schedule_of(recipe, given, len(fitting))
    if recipe['selection_unit'] == 'complete_situation':
        incomplete = sorted(identifier for identifier, entry in situations.items() if entry['role'] == 'model_select' and not entry['complete'])
        uncovered = sorted(row['id'] for row in selecting if not any(situations[s]['complete'] for s in row['situations']))
        if incomplete or uncovered:
            raise ConfigError(f'complete_situation selection needs gold-complete model_select situations; incomplete: {incomplete[:5]}, '
                              f'observations outside complete situations: {uncovered[:5]}')
    long = [row['id'] for row in assigned if max(lengths(row)) > budget['max_sequence_tokens']]
    if long:
        raise ConfigError(f'{len(long)} presentations exceed max_sequence_tokens {budget["max_sequence_tokens"]}; no implicit truncation or subsampling')
    probe = sorted(fitting, key=lambda row: artifact_digest([recipe['seed'], 'probe', row['id']]))[:len(selecting)]
    points = evaluation_points(len(fitting), recipe)
    point_tokens = sum(map(tokens, selecting)) + sum(map(tokens, probe))
    work = {'training_tokens': recipe['epochs'] * sum(map(tokens, fitting)), 'max_sequence_tokens': max(length for row in assigned for length in lengths(row)),
            'point_tokens': point_tokens, 'monitor_tokens': points * point_tokens, 'evaluation_points': points}
    if work['training_tokens'] > budget['max_tokens'] or work['monitor_tokens'] > budget['max_monitor_tokens']:
        raise ConfigError(f"predicted token work {work} exceeds max_tokens {budget['max_tokens']} or max_monitor_tokens {budget['max_monitor_tokens']}; "
                          'no implicit truncation or subsampling')
    executions = {artifact_digest(row['native']['execution']): row['native']['execution'] for row in assigned}
    if len(executions) != 1:
        raise ConfigError('exports were made under different native execution records; re-export every source with one runtime configuration')
    settings = job.get('native', {})
    if not isinstance(settings, dict) or set(settings) - set(NATIVE_SETTINGS):
        raise ConfigError(f'native accepts {", ".join(NATIVE_SETTINGS)}')
    def population(role):
        members = [row for row in assigned if row['role'] == role]
        linked = {row['id'] for row in members}
        entries_ = [entry for entry in situations.values() if entry['role'] == role]
        return {'situations': len(entries_), 'complete_situations': sum(entry['complete'] for entry in entries_),
                'targets': sum(value in linked for entry in entries_ for value in entry['targets'].values()), 'observations': len(members),
                'presentations': sum(len(row['native']['presentations']) for row in members), 'families': len({row['component'] for row in members}),
                'source_components': len({(source['export_id'], source['component']) for row in members for source in row['sources']})}
    ordered = sorted(rows.values(), key=lambda row: row['id'])
    resolved = {'schema': RESOLVED_SCHEMA, 'loss': LOSS, 'objective': objective, 'assets': pinned, 'model': model, 'toolchain': located['toolchain'],
                'task': {**{key: value for key, value in frozen.items() if key != 'overrides'}, 'source': task['source'], 'sha256': frozen['files']['task.yaml'],
                         'revisions': dict(sorted(revisions.items()))},
                'corrections': override_population(frozen['overrides'], rows),
                'execution': next(iter(executions.values())), 'fitting': placement, 'recipe': recipe, 'budget': budget, 'deadline': DEADLINE,
                'work': work, 'schedule': schedule, 'overrides': {'recipe': given, 'budget': given_budget}, 'selection': selection,
                'sources': [{'path': path, 'export_id': manifest['export_id'], 'task_family': source_family(path, exported), 'roles': entry['roles']}
                            for (path, manifest, exported), entry in zip(loaded, entries)],
                'cohort': {'revision': FAMILIES, 'observations': len(rows), 'families': len(families),
                           'memberships': artifact_digest([[row['id'], row['component'], row['role']] for row in ordered])},
                'support': support, 'populations': {role: population(role) for role in TUNING},
                'probe_ids': [row['id'] for row in probe], 'situations': situations,
                'memberships': [{k: row[k] for k in ('id', 'component', 'role', 'group', 'task_family', 'sources', 'situations')} |
                                {'label_sources': [LabelSource.parse(s).digest for s in row['label_sources']]} for row in ordered],
                'policy_families': sorted({row['task_family'] for row in fitting}),
                'native': {**native_settings(Model(model['gguf'], device='auto', threads=4, n_ctx=max(2048, budget['max_sequence_tokens'] * 2))), **settings}}
    native_model(resolved)
    resolved['fit'] = identity(fit_inputs(resolved, fitting))
    resolved['monitor'] = identity(monitor_plan(resolved, selecting, probe))
    if 'profile' in job:
        resolved['cadence'] = profiled_cadence(job['profile'], resolved, base)
    resolved['id'] = artifact_digest({**resolved, 'task': {key: value for key, value in resolved['task'].items() if key not in LOCATIONS}})
    return resolved, assigned


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


def exact_families(sizes, count, seed):
    order = sorted(sizes, key=lambda family: artifact_digest([seed, 'fit-family', family]))
    reachable = {0: ()}
    for family in order:
        for total, chosen in list(reachable.items()):
            if total + sizes[family] <= count:
                reachable.setdefault(total + sizes[family], (*chosen, family))
    if count not in reachable:
        raise ConfigError(f'cannot choose whole fit families totalling exactly {count} situations; family sizes {dict(sorted(sizes.items()))} '
                          f'reach {sorted(reachable)[-5:]} near the target; adjust the scenario quotas')
    return set(reachable[count])


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


def prepare(definition, exports, to, *, assets_manifest, quant='Q4_0', device='cpu', objective='specialization', budget=None, recipe=None,
            native=None, acceptance_fit=False, fit_situations=None, profile=None):
    to = Path(to).resolve()
    job_path = to / 'job.yaml'
    if job_path.exists():
        raise ConfigError(f'{job_path}: families were already assigned to roles there; prepare a new job in a new directory')
    if not exports:
        raise ConfigError('prepare requires native tune exports: ss export DEFINITION --native --task-family NAMESPACE:NAME --labels FILE')
    if fit_situations is not None and (type(fit_situations) is not int or fit_situations < 1):
        raise ConfigError('fit_situations must be a positive integer')
    loaded = load(str(Path(export).resolve()) for export in exports)
    rows, families = cohort(loaded)
    bundle = to / 'task'
    if bundle.exists():
        shutil.rmtree(bundle)
    assign(rows, intended(rows, acceptance_fit))
    separated(rows, families)
    selection = fit_selection(rows, fit_situations, (recipe or {}).get('seed', DEFAULTS['seed']))
    frozen = freeze_task(Path(definition).resolve(), bundle)
    copied = to / 'profile.json'
    if copied.exists():
        copied.unlink()
    if profile is not None:
        record = profile_record(profile)
        files.write_new(copied, record)
    job = {'schema': SCHEMA, 'loss': LOSS, 'objective': objective, 'task': {'source': str(Path(definition).resolve()), 'bundle': 'task', 'behavior': frozen['behavior']},
           'assets': {'manifest': assets_manifest, 'quant': quant}, 'fitting': {'device': device}, 'exports': exported_roles(loaded, rows),
           'selection': selection, **({'budget': dict(budget)} if budget else {}), **({'recipe': dict(recipe)} if recipe else {}),
           **({'native': native} if native else {}), **({'profile': {'file': copied.name, 'id': record['id']}} if profile is not None else {})}
    try:
        resolved, _ = resolved_job(job, to)
    except BaseException:
        shutil.rmtree(bundle, ignore_errors=True)
        copied.unlink(missing_ok=True)
        raise
    files.publish(job_path, job)
    run = str(to / 'run')
    return {'job': str(job_path), 'job_id': resolved['id'], 'fit_id': resolved['fit']['id'], 'monitor_id': resolved['monitor']['id'], 'cohort': resolved['cohort'],
            'task': {'bundle': resolved['task']['bundle'], 'behavior': resolved['task']['behavior'], 'overrides': len(resolved['corrections'])},
            'roles': resolved['populations'], 'support': resolved['support'], 'selection': selection,
            'recipe': resolved['recipe'], 'schedule': resolved['schedule'], 'work': resolved['work'], 'loss': LOSS, 'cadence': resolved.get('cadence'),
            'commands': {'dry_run': ['semantic-trainer', 'lora', str(job_path), '--to', run, '--dry-run'],
                         'fit': ['semantic-trainer', 'lora', str(job_path), '--to', run],
                         'select': ['semantic-trainer', 'select', str(job_path), '--run', run]}}


def profile_configuration(resolved):
    recipe = resolved['recipe']
    return {'assets': resolved['assets'], 'fitting': resolved['fitting'], 'loss': resolved['loss'], 'implementation': FITTING,
            'adapter': {key: recipe[key] for key in ('rank', 'alpha', 'dropout', 'target_modules')},
            'batching': {key: recipe[key] for key in ('batch_size', 'microbatch')}, 'max_sequence_tokens': resolved['budget']['max_sequence_tokens']}


def profile_workload(resolved):
    fit = resolved['populations']['fit']
    return {'point_tokens': resolved['work']['point_tokens'], 'fit_tokens_per_observation': resolved['work']['training_tokens'] / resolved['recipe']['epochs'] / fit['observations'],
            'fit_observations': fit['observations']}


def projection(resolved, costs, cadence):
    recipe, work, budget = {**resolved['recipe'], 'eval_every_examples': cadence}, resolved['work'], resolved['budget']
    points = evaluation_points(resolved['populations']['fit']['observations'], recipe)
    monitor_tokens = points * work['point_tokens']
    seconds = {'loading_seconds': costs['loading_seconds'], 'training_seconds': work['training_tokens'] * costs['update_seconds_per_token'],
               'monitoring_seconds': monitor_tokens * costs['point_seconds_per_token'], 'checkpoint_seconds': points * costs['checkpoint_seconds']}
    total = sum(seconds.values())
    return {'eval_every_examples': cadence, 'evaluation_points': points, 'optimizer_steps': resolved['schedule']['max_optimizer_steps'],
            'training_tokens': work['training_tokens'], 'monitor_tokens': monitor_tokens, **seconds, 'seconds': total,
            'max_seconds': budget['max_seconds'], 'max_tokens': budget['max_tokens'], 'max_monitor_tokens': budget['max_monitor_tokens'],
            'feasible': total <= budget['max_seconds'] and work['training_tokens'] <= budget['max_tokens'] and monitor_tokens <= budget['max_monitor_tokens']}


def profile_record(path):
    record = files.read_json(path)
    if not isinstance(record, dict) or record.get('schema') != PROFILE_SCHEMA or record.get('kind') != PROFILE_KIND \
            or record.get('id') != artifact_digest({key: value for key, value in record.items() if key != 'id'}):
        raise ConfigError(f'{path}: not an intact fitting profile of schema {PROFILE_SCHEMA}; run semantic-trainer profile JOB --to NEW_PROFILE')
    return record


def profiled_cadence(declared, resolved, base):
    if not isinstance(declared, dict) or set(declared) != {'file', 'id'}:
        raise ConfigError('profile records the copied fitting profile {file, id}; prepare with --profile PROFILE')
    record = profile_record(Path(base) / declared['file'])
    if record['id'] != declared['id']:
        raise ConfigError(f"{declared['file']}: fitting profile changed since preparation; prepare a new job")
    if record['configuration'] != profile_configuration(resolved):
        differing = sorted(key for key, value in profile_configuration(resolved).items() if record['configuration'].get(key) != value)
        raise ConfigError(f'fitting profile {record["id"]} measured another fitting configuration ({", ".join(differing)}); profile this configuration')
    current = profile_workload(resolved)
    ratios = {key: current[key] / record['workload'][key] for key in ('point_tokens', 'fit_tokens_per_observation')}
    if any(abs(ratio - 1) > PROFILE_TOLERANCE for ratio in ratios.values()):
        raise ConfigError(f'token workload changed by {ratios} since profile {record["id"]}; the measured costs no longer describe this job: re-profile')
    projected = projection(resolved, record['costs'], resolved['recipe']['eval_every_examples'])
    if not projected['feasible']:
        raise ConfigError(f'cadence {projected["eval_every_examples"]} projects {projected} from profile {record["id"]}, beyond the declared budgets; '
                          'choose a cadence or budget the profile shows feasible')
    return {'eval_every_examples': projected['eval_every_examples'], 'profile': record['id'], 'workload_ratio': ratios, 'projection': projected,
            'scope': 'projected from measured steady costs scaled by token workload; native baseline scoring and early stopping are not included'}


def ranking_rule(points, unit):
    losses = {point.get('nll') is not None for point in points}
    if len(losses) > 1:
        raise ConfigError('ranking mixes points with and without model_select NLL')
    order = [field for field in RANKING if field != 'nll' or losses == {True}]
    return {'unit': unit, 'order': order, 'nll': 'available' if 'nll' in order else 'unavailable', 'description': [RANKING[field] for field in order]}


def ranking(points, unit):
    keys = {'correct': lambda point: -point['correct'][unit], 'regressions': lambda point: len(point.get('regressions', ())), 'nll': lambda point: point['nll'],
            'base': lambda point: not point.get('base', False), 'step': lambda point: point['step']}
    order = ranking_rule(points, unit)['order']
    return sorted(points, key=lambda point: tuple(keys[field](point) for field in order))


def unit_correct(correct, situations, role):
    complete = [entry for entry in situations.values() if entry['role'] == role and entry['complete']]
    return {'observation': sum(bool(value) for value in correct.values()),
            'complete_situation': sum(all(correct.get(value) is True for value in entry['targets'].values() if value != 'exact') for entry in complete),
            'situations': len(complete), 'observations': len(correct)}
