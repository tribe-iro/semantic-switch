from __future__ import annotations

from pathlib import Path
import tempfile

from semantic_switch import ConfigError
from semantic_switch.training import artifact_digest, describe_task, files, freeze_task, is_digest, read_export, record_path, role_of

from .jobs import cohort, positive, prepare_tasks, situations_of, source_family, tokens

SCHEMA, KIND, AUDIT_SCHEMA, AUDIT_KIND, BUNDLE = 1, 'task-mixture', 1, 'mixture-audit', 'mixture'
ROLES = {'adaptation': 'fit', 'model_select': 'model_select', 'evaluation': 'check'}
USES = ('adaptation', 'selection', 'evaluation')
PERMITS = {'adaptation': 'adaptation', 'model_select': 'selection', 'evaluation': 'evaluation'}
MAX_EXAMPLES, MAX_ADAPTATION = 500, 2000
TASK = {'role', 'definition', 'contract', 'revisions', 'exports', 'mapping', 'sources', 'quota', 'weight'}
SOURCE = {'dataset', 'revision', 'sha256', 'licence', 'permits', 'evidence', 'underlying'}


def text(value):
    return isinstance(value, str) and bool(value)


def leaves(source, where):
    if not isinstance(source, dict) or set(source) != SOURCE or not all(text(source[key]) for key in ('dataset', 'revision', 'licence')) \
            or not is_digest(source['sha256']) or not isinstance(source['permits'], list) or set(source['permits']) - set(USES) \
            or not isinstance(source['evidence'], dict) or set(source['evidence']) != {'locator', 'sha256'} or not text(source['evidence']['locator']) \
            or not is_digest(source['evidence']['sha256']) or not isinstance(source['underlying'], list):
        raise ConfigError(f'{where}: a source records dataset, revision, content sha256, licence, permitted uses from {list(USES)}, '
                          'licence/permission evidence {locator, sha256} and the underlying datasets it aggregates')
    if not source['underlying']:
        return [source]
    return [leaf for inner in source['underlying'] for leaf in leaves(inner, f"{where} > {source['dataset']}")]


def described(name, task, definition):
    with tempfile.TemporaryDirectory(prefix='semantic-trainer-mixture-') as scratch:
        frozen = freeze_task(definition, Path(scratch) / 'task')
    if frozen['behavior'] != task['contract']:
        raise ConfigError(f"{name}: {definition} freezes to task contract {frozen['behavior']}, not the declared {task['contract']}; freeze a new mixture")
    revisions = {address: target['meaning_rev'] for address, target in frozen['targets'].items()}
    if task['revisions'] != revisions:
        raise ConfigError(f'{name}: declared meaning revisions differ from the frozen definition {revisions}; freeze a new mixture')
    return describe_task(definition)['targets']


def mapped(name, task, targets):
    if not isinstance(task['mapping'], dict) or not task['mapping']:
        raise ConfigError(f'{name}: mapping scopes each source label to an answer key per target address')
    for address, labels in task['mapping'].items():
        if address not in targets:
            raise ConfigError(f'{name}: mapping address {address!r} is not a target of the task; targets: {sorted(targets)}')
        answers = set(targets[address]['answers'])
        if not isinstance(labels, dict) or not labels or not all(text(label) for label in labels) or set(labels.values()) - answers:
            raise ConfigError(f'{name}: {address} maps source labels onto its declared answers {sorted(answers)}')


def exported(name, task, base, role):
    if not isinstance(task['exports'], list) or not task['exports'] \
            or any(not isinstance(entry, dict) or set(entry) != {'path', 'export_id'} or not text(entry['path']) for entry in task['exports']):
        raise ConfigError(f'{name}: exports lists the immutable native exports {{path, export_id}} of the task')
    loaded = []
    for entry in task['exports']:
        path = str((Path(base) / entry['path']).resolve())
        manifest, rows = read_export(path, fitting=role != 'evaluation', native=True)
        if manifest['export_id'] != entry['export_id']:
            raise ConfigError(f"{path}: export {entry['export_id']} changed (now {manifest['export_id']}); freeze a new mixture")
        if source_family(path, manifest) != name:
            raise ConfigError(f'{path}: exported under task family {source_family(path, manifest)}, not {name}')
        stray = sorted({role_of(row['split'], row['fold']) for row in rows} - {ROLES[role]})
        if stray:
            raise ConfigError(f'{path}: {role} task {name} assigns {stray} families; a mixture task holds one role')
        loaded.append((path, manifest, rows, task['contract']))
    return loaded


def contract_of(manifest, where):
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA or manifest.get('kind') != KIND or set(manifest) != {'schema', 'kind', 'claim', 'tasks', 'id'}:
        raise ConfigError(f'{where}: not a task mixture of schema {SCHEMA} {{schema, kind, claim, tasks, id}}')
    if manifest['id'] != artifact_digest({key: value for key, value in manifest.items() if key != 'id'}):
        raise ConfigError(f'{where}: mixture identity mismatch; a frozen mixture is never edited, freeze a new one')
    claim, tasks = manifest['claim'], manifest['tasks']
    if not isinstance(claim, dict) or set(claim) != {'statement', 'holdout'} or not text(claim['statement']) \
            or not isinstance(claim['holdout'], list) or not all(text(kind) for kind in claim['holdout']):
        raise ConfigError(f'{where}: claim states the transfer hypothesis and the held-out family kinds (documents, rules, source states)')
    if not isinstance(tasks, dict) or {task.get('role') for task in tasks.values() if isinstance(task, dict)} != set(ROLES):
        raise ConfigError(f'{where}: tasks maps task families to adaptation, model_select and evaluation tasks; every role is required')
    contracts, datasets, permissions = {}, {}, {}
    for name, task in sorted(tasks.items()):
        if ':' not in name or not isinstance(task, dict) or set(task) != TASK or task['role'] not in ROLES or not text(task['definition']) \
                or not is_digest(task['contract']) or not isinstance(task['revisions'], dict) or type(task['quota']) is not int or not 0 < task['quota'] <= MAX_EXAMPLES:
            raise ConfigError(f'{where}: task {name!r} needs a namespaced family and {sorted(TASK)} with a quota of 1..{MAX_EXAMPLES} complete examples')
        if (task['weight'] is None) == (task['role'] == 'adaptation') or task['weight'] is not None and not positive(task['weight']):
            raise ConfigError(f'{where}: {name}: adaptation tasks declare a positive objective weight and other roles none')
        if task['contract'] in contracts:
            raise ConfigError(f"{where}: {name} and {contracts[task['contract']]} freeze to one task contract; renaming does not create a task family")
        contracts[task['contract']] = name
        if not isinstance(task['sources'], list) or not task['sources']:
            raise ConfigError(f'{where}: {name} records the datasets its examples derive from')
        permissions[name] = [leaf for source in task['sources'] for leaf in leaves(source, f'{where}: {name}')]
        for leaf in permissions[name]:
            if PERMITS[task['role']] not in leaf['permits']:
                raise ConfigError(f"{where}: {name}: dataset {leaf['dataset']} {leaf['revision']} ({leaf['licence']}) does not permit {PERMITS[task['role']]}; "
                                  'an aggregation licence does not grant uses its underlying datasets withhold')
            known = datasets.setdefault(leaf['sha256'], (task['role'], name))
            if known[0] != task['role']:
                raise ConfigError(f"{where}: dataset {leaf['dataset']} content {leaf['sha256'][:12]} feeds {known[1]} ({known[0]}) and {name} ({task['role']}); "
                                  'a derivative of one task cannot be held out from it')
    return {'schema': AUDIT_SCHEMA, 'kind': AUDIT_KIND, 'mixture': manifest['id'], 'claim': claim, 'total_adaptation_cap': MAX_ADAPTATION,
            'tasks': {name: {'role': task['role'], 'contract': task['contract'], 'revisions': task['revisions'], 'quota': task['quota'], 'weight': task['weight'],
                             'mapping': task['mapping'], 'exports': sorted(entry['export_id'] for entry in task['exports']), 'permissions': permissions[name]}
                      for name, task in sorted(tasks.items())}}


def evidence_of(contract):
    return sorted({leaf['evidence']['sha256'] for task in contract['tasks'].values() for leaf in task['permissions']})


def canonical(loaded, rows, tasks):
    names = {task['contract']: name for name, task in tasks.items()}
    for row in rows.values():
        row['role'] = ROLES[tasks[names[row['task']]]['role']]
    scopes = {}
    for _, manifest, _, contract in loaded:
        scopes.setdefault(contract, set()).update('/'.join(address) for address in manifest['scope'])
    scopes = {contract: sorted(found) for contract, found in scopes.items()}
    situations = situations_of(loaded, rows, scopes)
    coverage = {}
    for name, task in sorted(tasks.items()):
        members = [row for row in rows.values() if row['task'] == task['contract']]
        examples = [entry['complete'] for entry in situations.values() if entry['task'] == task['contract']]
        examples += [scopes[task['contract']] == ['/'.join(record_path(row))] for row in members if not row['situations']]
        coverage[name] = {'role': task['role'], 'complete_examples': sum(examples), 'incomplete_examples': len(examples) - sum(examples),
                          'observations': len(members), 'families': len({row['component'] for row in members}),
                          'presentations': sum(len(row['presentations']) for row in members), 'tokens': sum(map(tokens, members)),
                          'targets': sorted({'/'.join(record_path(row)) for row in members})}
    roles = {}
    for found in coverage.values():
        total = roles.setdefault(found['role'], {key: 0 for key in ('tasks', 'complete_examples', 'incomplete_examples', 'observations', 'families', 'presentations', 'tokens')})
        total['tasks'] += 1
        for key in total.keys() - {'tasks'}:
            total[key] += found[key]
    return {'tasks': coverage, 'roles': dict(sorted(roles.items()))}


def validate(path):
    path = Path(path).resolve()
    manifest = files.read_json(path)
    contract = contract_of(manifest, path)
    tasks, loaded, declared = manifest['tasks'], [], {}
    for name, task in sorted(tasks.items()):
        definition = (path.parent / task['definition']).resolve()
        mapped(name, task, described(name, task, definition))
        for item in exported(name, task, path.parent, task['role']):
            if item[1]['export_id'] in declared:
                raise ConfigError(f"{path}: export {item[1]['export_id']} is declared by {declared[item[1]['export_id']]} and {name}; declare each immutable export once")
            declared[item[1]['export_id']] = name
            loaded.append(item)
    loaded.sort(key=lambda item: item[1]['export_id'])
    rows, families = cohort(loaded)
    names = {task['contract']: name for name, task in tasks.items()}
    for family, entry in families.items():
        roles = {tasks[names[rows[identifier]['task']]]['role'] for identifier in entry['observations']}
        if len(roles) > 1:
            bridged = sorted({names[rows[identifier]['task']] for identifier in entry['observations']})
            raise ConfigError(f"{path}: canonical family {family} bridges tasks {bridged} across roles {sorted(roles)} through seeds "
                              f"{[':'.join(node) for node in entry['seeds'][:6]]}; deduplicate the shared records before assignment")
    coverage = canonical(loaded, rows, tasks)
    over = {name: found['complete_examples'] for name, found in coverage['tasks'].items() if found['complete_examples'] > tasks[name]['quota']}
    if over:
        raise ConfigError(f'{path}: canonical complete examples exceed the task quotas: ' +
                          ', '.join(f"{name} {count} > {tasks[name]['quota']}" for name, count in sorted(over.items())) + '; no rows are truncated to fit a quota')
    adapted = coverage['roles'].get('adaptation', {}).get('complete_examples', 0)
    if adapted > MAX_ADAPTATION:
        raise ConfigError(f'{path}: {adapted} canonical complete adaptation examples exceed the pilot cap {MAX_ADAPTATION}')
    evidence = {}
    for task in contract['tasks'].values():
        for leaf in task['permissions']:
            located = (path.parent / leaf['evidence']['locator']).resolve()
            if not located.is_file() or files.sha256_file(located) != leaf['evidence']['sha256']:
                raise ConfigError(f"{path}: permission evidence {leaf['evidence']['locator']} of {leaf['dataset']} is missing or differs from its sha256")
            evidence.setdefault(leaf['evidence']['sha256'], located)
    return {'path': str(path), 'id': manifest['id'], 'claim': manifest['claim'], 'tasks': tasks, 'coverage': coverage, 'contract': contract, 'evidence': evidence}


def bundled(declared, base, tasks, entries):
    if not isinstance(declared, dict) or set(declared) != {'audit', 'id'} or declared['audit'] != f'{BUNDLE}/audit.json' or not is_digest(declared['id']):
        raise ConfigError(f'mixture records the bundled audit {{audit: {BUNDLE}/audit.json, id}}; prepare a new job with --mixture')
    root = Path(base) / BUNDLE
    audit = files.read_json(root / 'audit.json')
    if not isinstance(audit, dict) or audit.get('id') != declared['id'] or audit['id'] != artifact_digest({key: value for key, value in audit.items() if key != 'id'}):
        raise ConfigError(f'{root}: mixture audit changed since preparation; prepare a new job with the frozen mixture')
    contract = contract_of(files.read_json(root / 'manifest.json'), root / 'manifest.json')
    if {key: value for key, value in audit.items() if key not in ('coverage', 'locators', 'id')} != contract:
        raise ConfigError(f'{root}: the mixture audit does not describe its bundled manifest; prepare a new job')
    for sha in evidence_of(contract):
        if not (root / 'evidence' / sha).is_file() or files.sha256_file(root / 'evidence' / sha) != sha:
            raise ConfigError(f'{root}: bundled permission evidence {sha[:12]} is missing or altered; prepare a new job from the frozen mixture')
    trained = {task['contract']: task for task in contract['tasks'].values() if task['role'] != 'evaluation'}
    exports = {}
    for entry in entries:
        exports.setdefault(entry['task'], []).append(entry['export_id'])
    if set(tasks) != set(trained) or any(tasks[c]['weight'] != (task['weight'] or 1) or sorted(exports.get(c, ())) != task['exports'] for c, task in trained.items()):
        raise ConfigError('the job tasks, weights or exports differ from its bundled mixture; prepare a new job with --mixture')
    return {'audit': audit['id'], 'mixture': contract['mixture'], 'claim': contract['claim'],
            'evaluation': {name: {key: task[key] for key in ('contract', 'exports')} for name, task in contract['tasks'].items() if task['role'] == 'evaluation'}}


def prepare_mixture(path, to, **settings):
    checked = validate(path)
    base = Path(checked['path']).parent
    sources = [{'definition': (base / task['definition']).resolve(), 'exports': [(base / entry['path']).resolve() for entry in task['exports']],
                'weight': task['weight'] if task['role'] == 'adaptation' else 1, 'contract': task['contract']}
               for _, task in sorted(checked['tasks'].items()) if task['role'] != 'evaluation']
    audit = {**checked['contract'], 'coverage': checked['coverage'],
             'locators': {'manifest': checked['path'], 'evidence': {sha: str(located) for sha, located in sorted(checked['evidence'].items())}}}
    audit['id'] = artifact_digest(audit)
    copies = {'manifest.json': Path(checked['path']), **{f'evidence/{sha}': located for sha, located in checked['evidence'].items()}}
    prepared = prepare_tasks(sources, to, objective='instruction_transfer', mixture={'audit': audit, 'copies': copies}, **settings)
    return {**prepared, 'mixture': {'id': checked['id'], 'audit': audit['id'], 'claim': checked['claim'], 'coverage': checked['coverage']}}
