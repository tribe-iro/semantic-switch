from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import platform
import signal
import sys
import traceback
import uuid
from importlib import metadata, util
from pathlib import Path
from typing import Callable, NamedTuple

from semantic_switch import ConfigError
from semantic_switch.training import (RECEIPT_SCHEMA, REQUEST_SCHEMA, RESULT_SCHEMA, Journal, artifact_digest, assigned_evidence, describe_task, export_native, files,
                                      freeze_task, frozen_task, judge, load_receipt, read_export, read_request, role_of, sources, terminal, verify_files)

from . import __version__, assets

REVISION, JOURNAL = 3, 'pipeline.journal.jsonl'
BUNDLE, LABELS, MANIFEST = 'task', 'labels.jsonl', 'frozen-inputs.json'
SOURCES = {'definition': 'task definition', 'description': 'task meaning or guidance', 'corrections': 'corrections', 'labels': 'labels'}
STAGES = ('freeze', 'assets', 'export', 'population', 'profile', 'cadence', 'finalize', 'fit', 'select', 'lock')
NEEDS = {'freeze': (), 'assets': (), 'export': ('freeze', 'assets'), 'population': ('freeze', 'assets', 'export'), 'profile': ('population',),
         'cadence': ('population', 'profile'), 'finalize': ('population', 'profile', 'cadence'), 'fit': ('finalize',), 'select': ('finalize', 'fit'),
         'lock': ('select',)}
IGNORED = ('.write-lock', '.run')
PACKAGES = ('semantic-switch', 'semantic-trainer', 'torch', 'transformers', 'peft', 'llama-cpp-python')
BUILT = {'semantic-switch': 'semantic_switch', 'semantic-trainer': 'semantic_trainer'}


class Operation(NamedTuple):
    run: Callable
    verify: Callable


class Awaiting(Exception):
    def __init__(self, choices):
        super().__init__('the monitoring cadence is not chosen')
        self.choices = choices


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment(request) -> dict:
    return {'python': platform.python_version(), 'executable': sys.executable, 'packages': {name: version(name) for name in PACKAGES},
            'devices': request['devices'], 'toolchain': assets.TOOLCHAIN}


def imports(nodes, package):
    result = set()
    for tree in nodes:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                result.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                name = util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                if name:
                    result.add(name)
                    result.update(name + '.' + alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('semantic_switch.drivers.') and ':' in node.value:
                result.add(node.value.split(':')[0])
    return sorted(result)


def source(distribution, package):
    spec = util.find_spec(package)
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    listing = files.source_content(root, root.rglob('*.py'))
    graph = {}
    for path in listing:
        module = package + '.' + path.removesuffix('.py').replace('/', '.')
        module = module.removesuffix('.__init__')
        owner = module if path.endswith('__init__.py') else module.rpartition('.')[0]
        graph[module] = {'path': path, 'imports': imports([ast.parse((root / path).read_text())], owner)}
    return {'version': version(distribution), 'files': listing, 'imports': graph}


def producer(name):
    tree = ast.parse(Path(__file__).read_text())
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for child in ast.walk(node.targets[0]):
                if isinstance(child, ast.Name):
                    definitions[child.id] = node
    registry = definitions['OPERATIONS'].value
    entry = next(value for key, value in zip(registry.keys, registry.values) if ast.literal_eval(key) == name)
    pending, selected = [*(value.id for value in entry.args if isinstance(value, ast.Name)), 'Context'], {}
    while pending:
        key = pending.pop()
        if key in selected or key not in definitions:
            continue
        node = selected[key] = definitions[key]
        pending.extend(child.id for child in ast.walk(node) if isinstance(child, ast.Name) and child.id not in selected)
    nodes = list({id(node): node for node in selected.values()}.values())
    roots = imports([*nodes, *(node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))], __package__)
    return {'source': artifact_digest([ast.dump(node, include_attributes=False) for node in sorted(nodes, key=lambda node: node.lineno)]), 'roots': roots}


def build():
    built = {name: source(name, package) for name, package in BUILT.items()}
    built['producers'] = {name: producer(name) for name in STAGES}
    spec = util.find_spec('llama_cpp')
    if spec is not None:
        root = Path(spec.origin).resolve().parent
        built['native'] = {'version': version('llama-cpp-python'), 'libraries': files.source_content(root, root.rglob('*.so*'))}
    return built


def implementation(name, built):
    graph = {module: (package, info) for package in BUILT for module, info in built[package].get('imports', {}).items()}
    entry = built.get('producers', {}).get(name, {'source': None, 'roots': list(graph)})
    pending, visited, selected = list(entry['roots']), set(), {package: {'version': built[package]['version'], 'files': {}} for package in BUILT}
    while pending:
        module = pending.pop()
        if module in visited or module not in graph:
            continue
        visited.add(module)
        package, info = graph[module]
        if info['path'] == 'pipeline.py' and package == 'semantic-trainer':
            continue
        selected[package]['files'][info['path']] = built[package]['files'][info['path']]
        pending.extend(info['imports'])
    for package in BUILT:
        if 'imports' not in built[package]:
            selected[package] = dict(built[package])
    numerical = ['numpy', 'pyyaml', 'pydantic', 'jinja2']
    if name in ('profile', 'fit', 'select'):
        numerical.extend(('torch', 'transformers', 'peft'))
    if name in ('export', 'fit', 'select'):
        selected['native'] = built.get('native')
        numerical.append('llama-cpp-python')
    return {'producer': entry['source'], 'build': selected, 'numerical_builds': {package: version(package) for package in numerical}}


def computation(name, settings, inputs, built) -> str:
    return artifact_digest({'stage': name, 'inputs': {dep: inputs[dep]['id'] for dep in NEEDS[name] if dep != 'profile' or dep in inputs}, 'settings': settings, 'implementation': implementation(name, built)})


def tree(root) -> dict:
    root = Path(root)
    return {str(path.relative_to(root)): files.sha256_file(path) for path in sorted(root.rglob('*'))
            if path.is_file() and not path.name.endswith(IGNORED) and not path.name.startswith('.')}


def output(kind, identifier, path, root=None, **summary) -> dict:
    path = Path(path)
    root = Path(root) if root is not None else path if path.is_dir() else path.parent
    listed = tree(root) if root == path else {path.name: files.sha256_file(path)}
    return {'kind': kind, 'id': identifier, 'path': str(path), 'root': str(root), 'files': listed, 'summary': summary}


class Context:
    def __init__(self, request, recipe, budget, cadence, outputs):
        self.request, self.recipe, self.budget, self.cadence, self.outputs, self.cache = request, recipe, budget, cadence, outputs, {}

    def cached(self, name, record, make):
        key = (name, record['id'])
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def materialized(self) -> dict:
        record = self.outputs.get('freeze')
        if record is None:
            raise ConfigError('later stages consume the task and labels only from the completed freeze stage')
        def verified():
            try:
                verify_files(Path(record['root']), record['files'], 'frozen input changed or missing since the freeze stage; rerun into a new work root')
            except ValueError as error:
                raise ConfigError(str(error)) from None
            return materialized(record)
        return self.cached('freeze', record, verified)

    def population(self, record=None):
        from .jobs import population_of
        record = record or self.outputs['population']
        return self.cached('population', record, lambda: population_of(record['path']))

    def finalized(self, path):
        from .jobs import cadence_of, finalize, population_job
        path, source = Path(path), Path(self.outputs['population']['path'])
        job = files.read_yaml(path)
        if population_job(job, path.parent) != population_job(files.read_yaml(source), source.parent):
            raise ConfigError(f'{path}: the finalized job does not describe the committed population')
        population, assigned = self.population()
        return finalize(population, assigned, cadence_of(job.get('recipe')), job.get('profile'), path.parent), assigned

    def resolved(self, record=None):
        record = record or self.outputs['finalize']
        return self.cached('finalize', record, lambda: self.finalized(record['path']))


def observed(declared) -> None:
    try:
        found = sources(declared['definition']['path'], declared['labels']['path'])
    except (ConfigError, OSError) as error:
        raise ConfigError(f'a request source is unavailable before freeze ({error}); restore it, or write a new request into a new work root') from None
    changed = [SOURCES[key] for key in SOURCES if found[key] != declared[key]]
    if changed:
        raise ConfigError(f"the {', '.join(changed)} changed since this request; write a new request into a new work root")


def freeze(context, directory, settings):
    from .jobs import TUNING, fit_support
    declared = context.request['sources']
    observed(declared)
    frozen = freeze_task(declared['definition']['path'], directory / BUNDLE)
    labels = files.write_new(directory / LABELS, Path(declared['labels']['path']).read_bytes())
    observed(declared)
    described = artifact_digest(describe_task(frozen['definition']))
    if labels != declared['labels']['sha256'] or described != declared['description']:
        raise ConfigError('the task or labels changed while they were frozen; write a new request into a new work root')
    evidence = assigned_evidence(frozen['definition'], directory / LABELS, context.request['task']['scope'])
    rows = [{**row, 'task': frozen['behavior']} for row in evidence['rows']]
    tuned = [row for row in rows if row['role'] in TUNING]
    if not any(row['role'] == 'model_select' for row in tuned):
        raise ConfigError(f"{declared['labels']['path']}: no model_select families are assigned; selection needs whole model_select families before fitting")
    support = fit_support(tuned, 'specialization')
    roles = {}
    for row in rows:
        entry = roles.setdefault(row['role'] or 'unassigned', {'observations': 0, 'families': set()})
        entry['observations'] += 1
        entry['families'].add(row['component'])
    assignment = {'dataset_id': evidence['dataset_id'],
                  'roles': {role: {'observations': entry['observations'], 'families': len(entry['families'])} for role, entry in sorted(roles.items())},
                  'check_components': sorted(roles.get('check', {'families': set()})['families']), 'support': support['roles']}
    manifest = {'kind': 'frozen-inputs', 'sources': declared, 'scope': evidence['scope'], 'assignment': assignment,
                'task': {'bundle': BUNDLE, 'behavior': frozen['behavior'], 'switch': frozen['switch'], 'description': described},
                'labels': {'file': LABELS, 'sha256': labels}}
    files.write_new(directory / MANIFEST, manifest)
    return output('frozen-inputs', artifact_digest(manifest), directory, task=frozen['behavior'], roles=assignment['roles'])


def materialized(record) -> dict:
    root = Path(record['path'])
    manifest = files.read_json(root / MANIFEST)
    if not isinstance(manifest, dict) or artifact_digest(manifest) != record['id']:
        raise ConfigError(f'{root / MANIFEST}: the frozen input manifest changed')
    frozen = frozen_task(root / manifest['task']['bundle'], manifest['task']['behavior'])
    if files.sha256_file(root / manifest['labels']['file']) != manifest['labels']['sha256']:
        raise ConfigError(f"{root / manifest['labels']['file']}: the frozen labels changed since the freeze stage")
    return {'definition': frozen['definition'], 'labels': str(root / manifest['labels']['file']), 'switch': manifest['task']['switch'], 'manifest': manifest}


def verify_freeze(context, record):
    manifest = materialized(record)['manifest']
    if manifest['sources'] != context.request['sources'] or manifest['labels']['sha256'] != manifest['sources']['labels']['sha256'] \
            or manifest['task']['behavior'] != record['summary']['task']:
        raise ConfigError('the frozen inputs belong to another request')


def model(context, directory, settings):
    request = context.request['model']
    if 'assets' in request:
        located = assets.load(request['assets'])
    else:
        built = assets.build(request['repo'], [request['quant']], revision=settings['commit'])
        located = assets.load(built['manifest'])
    if request['quant'] not in located['gguf']:
        raise ConfigError(f"asset manifest {located['manifest']} has no {request['quant']} base; build it with semantic-trainer assets --quant {request['quant']}")
    base = located['gguf'][request['quant']]
    return {'kind': 'trainer-assets', 'id': located['manifest'], 'path': str(Path(located['root']) / 'manifests' / f"{located['manifest']}.json"),
            'root': None, 'files': {}, 'summary': {'base': base['path'], 'sha256': base['sha256'], 'repo': located['repo'], 'commit': located['commit']}}


def verify_model(context, record):
    located = assets.load(record['id'])
    if located['gguf'].get(context.request['model']['quant'], {}).get('sha256') != record['summary']['sha256']:
        raise ConfigError('the deployed base changed')


def native_judge(path, device):
    return judge(path, device=device)


def disjoint(context, manifest_path):
    _, rows = read_export(manifest_path, native=True)
    held = set(materialized(context.outputs['freeze'])['manifest']['assignment']['check_components'])
    leaked = sorted({row['component'] for row in rows if role_of(row['split'], row['fold']) == 'check' or row['component'] in held})
    if leaked:
        raise ConfigError(f'{manifest_path}: check families {leaked[:5]} entered the tune export; export only the tune split')


def export(context, directory, settings):
    request, inputs = context.request, context.materialized()
    target = directory / f"{inputs['switch']}.export.jsonl"
    base = context.outputs['assets']['summary']['base']
    export_native(inputs['definition'], target, judge=native_judge(base, request['devices']['native']), labels=inputs['labels'],
                  task_family=request['task']['family'], scope=request['task']['scope'])
    manifest_path = Path(f'{target}.manifest.json')
    manifest, _ = read_export(manifest_path, native=True)
    return output('native-export', manifest['export_id'], manifest_path, directory)


def verify_export(context, record):
    manifest, _ = read_export(record['path'], native=True)
    if manifest['export_id'] != record['id']:
        raise ConfigError(f"export {manifest['export_id']} is not the committed {record['id']}")
    disjoint(context, record['path'])


def population(context, directory, settings):
    from .jobs import prepared_tasks
    selection, model = context.request['selection'], context.request['model']
    source = {'definition': context.materialized()['definition'], 'exports': [context.outputs['export']['path']], 'weight': 1, 'contract': None}
    summary, found, assigned = prepared_tasks([source], directory, assets_manifest=context.outputs['assets']['id'], quant=model['quant'],
                                              device=context.request['devices']['fit'], budget=context.budget, recipe=context.recipe,
                                              native={'device': context.request['devices']['native']}, acceptance_fit=selection['acceptance_fit'],
                                              fit_situations=selection['fit_situations'], scope=context.request['task']['scope'])
    record = output('prepared-population', found['id'], summary['job'], directory, fit_id=found['fit']['id'], roles=found['populations'])
    context.cache[('population', found['id'])] = found, assigned
    return record


def verify_population(context, record):
    found, _ = context.population(record)
    if found['id'] != record['id']:
        raise ConfigError(f"population resolves to {found['id']}, not the committed {record['id']}")
    if set(found['tasks']) != {context.outputs['freeze']['summary']['task']}:
        raise ConfigError(f"population task contracts {sorted(found['tasks'])} differ from the frozen task {context.outputs['freeze']['summary']['task']}")


def profile(context, directory, settings):
    from .jobs import profile_record
    declared = context.request['profile']
    if 'path' in declared:
        if files.sha256_file(declared['path']) != declared['sha256']:
            raise ConfigError(f"{declared['path']}: the fitting profile changed since this request was written; write a new request into a new work root")
        record = profile_record(declared['path'])
        return output('fitting-profile', record['id'], declared['path'], measured=False)
    from .lora import profiled
    record = profiled(*context.population(), directory / 'profile.json', **declared['measure'])
    return output('fitting-profile', record['id'], directory / 'profile.json', directory, measured=True)


def verify_profile(context, record):
    from .jobs import profile_applies, profile_record
    measured = profile_record(record['path'])
    if measured['id'] != record['id']:
        raise ConfigError(f"profile {measured['id']} is not the committed {record['id']}")
    profile_applies(measured, context.population()[0])


def cadence(context, directory, settings):
    from .jobs import feasible_cadence, profile_record
    chosen = context.request['cadence'] or context.cadence
    measured = profile_record(context.outputs['profile']['path']) if 'profile' in context.outputs else None
    if measured is None:
        chosen = chosen or context.population()[0]['populations']['fit']['observations']
    if chosen is None:
        raise Awaiting([{key: entry[key] for key in ('eval_every_examples', 'evaluation_points', 'monitor_tokens', 'seconds', 'feasible')} for entry in measured['cadences']])
    projected = feasible_cadence(context.population()[0], measured, chosen) if measured is not None else None
    decision = {'cadence': chosen, 'profile': measured['id'] if measured else None}
    return {'kind': 'cadence-decision', 'id': artifact_digest(decision), 'path': None, 'root': None, 'files': {}, 'summary': {**decision, 'projection': projected}}


def verify_cadence(context, record):
    chosen = record['summary']['cadence']
    if context.cadence not in (None, chosen):
        raise ConfigError(f'cadence {chosen} is committed for this request; a different cadence needs a new request and work root')
    if record['summary']['profile'] != context.outputs.get('profile', {}).get('id') or record['id'] != artifact_digest({key: record['summary'][key] for key in ('cadence', 'profile')}):
        raise ConfigError('the cadence decision is bound to another profile')


def finalize(context, directory, settings):
    from .jobs import final_job
    source = Path(context.outputs['population']['path'])
    job = final_job(files.read_yaml(source), source.parent, context.outputs['cadence']['summary']['cadence'], context.outputs.get('profile', {}).get('path'), directory)
    files.publish(directory / 'job.yaml', job)
    found, assigned = context.finalized(directory / 'job.yaml')
    context.cache[('finalize', found['id'])] = found, assigned
    return output('prepared-job', found['id'], directory / 'job.yaml', directory, fit_id=found['fit']['id'], monitor_id=found['monitor']['id'],
                  roles=found['populations'], cadence=found.get('cadence'))


def verify_finalize(context, record):
    found, _ = context.resolved(record)
    if found['id'] != record['id'] or found['population'] != context.outputs['population']['id']:
        raise ConfigError(f"job resolves to {found['id']} over population {found['population']}, not the committed {record['id']}")


def fit(context, directory, settings):
    from .lora import trained_run
    measured = context.outputs.get('profile')
    consumed = files.read_json(measured['path']).get('elapsed_seconds', 0.) if measured and measured['summary'].get('measured') else 0.
    trained = trained_run(*context.resolved(), directory, consumed=consumed)
    return output('fitted-run', artifact_digest(files.read_json(directory / 'candidates.json')), directory, names=trained['names'], stop=trained['stop'])


def verify_fit(context, record):
    from .jobs import ready_run
    _, listed, fitted = ready_run(record['path'], f"ss specialize --to {context.request['work']} --attempt N")
    if artifact_digest(listed) != record['id'] or fitted['id'] != context.outputs['finalize']['id']:
        raise ConfigError(f"run {record['path']} does not hold the committed candidates of job {context.outputs['finalize']['id']}")


def select(context, directory, settings):
    from .select import application
    chosen = application(*context.resolved(), context.outputs['finalize']['path'], context.outputs['fit']['path'], context.request['representation'], to=directory)
    receipt = load_receipt(chosen['receipt'])
    return output('deployment', receipt.id, directory, receipt=chosen['receipt'], selected=chosen['selected'], representation=chosen['representation'],
                  candidates=chosen['candidates'])


def verify_select(context, record):
    receipt = load_receipt(record['summary']['receipt'])
    if receipt.id != record['id']:
        raise ConfigError(f'receipt {receipt.id} is not the committed {record["id"]}')


def lock(context, directory, settings):
    receipt = load_receipt(context.outputs['select']['summary']['receipt'])
    task = receipt.verified_task()
    return {'kind': 'frozen-lock', 'id': receipt.id, 'path': str(receipt.directory / receipt.data['lock']['path']), 'root': None, 'files': {},
            'summary': {'receipt': str(receipt.path), 'lock_id': receipt.data['lock']['id'], 'task': str(task),
                        'representation': context.outputs['select']['summary']['representation']}}


def verify_lock(context, record):
    receipt = load_receipt(record['summary']['receipt'])
    receipt.verified_task()
    if receipt.id != record['id']:
        raise ConfigError(f'receipt {receipt.id} is not the committed {record["id"]}')


OPERATIONS = {'freeze': Operation(freeze, verify_freeze), 'assets': Operation(model, verify_model), 'export': Operation(export, verify_export),
              'population': Operation(population, verify_population), 'profile': Operation(profile, verify_profile),
              'cadence': Operation(cadence, verify_cadence), 'finalize': Operation(finalize, verify_finalize), 'fit': Operation(fit, verify_fit),
              'select': Operation(select, verify_select), 'lock': Operation(lock, verify_lock)}


def message(**event):
    print(json.dumps({'pipeline': REVISION, **event}, default=str)[:4096], file=sys.stderr, flush=True)


class Controller:
    def __init__(self, request, *, until=None, attempt=None, cadence=None, operations=OPERATIONS, emit=message):
        from .cli import settings
        from .jobs import BUDGET, DEFAULTS
        if until is not None and until not in STAGES:
            raise ConfigError(f"--until {until!r} names no pipeline stage; stages: {', '.join(STAGES)}")
        if attempt is not None and (type(attempt) is not int or attempt < 1):
            raise ConfigError('--attempt is a positive fit attempt number')
        self.path = Path(request).resolve()
        self.request = read_request(self.path)
        if until == 'profile' and self.request['profile'] is None:
            raise ConfigError('profiling is disabled in this request; opt in with --profile-steps when creating the request')
        if cadence is not None and self.request['cadence'] not in (None, cadence):
            raise ConfigError(f"the request predeclares cadence {self.request['cadence']}; a different cadence needs a new request and work root")
        self.context = Context(self.request, settings(self.request['recipe'], DEFAULTS, 'recipe'), settings(self.request['budget'], BUDGET, 'budget'), cadence, {})
        self.until, self.attempt, self.operations, self.emit = until, attempt, operations, emit
        self.work, self.stage, self.states, self.built = Path(self.request['work']), None, {}, build()

    def settings(self, name, journal) -> dict:
        request = self.request
        model, devices, selection, scope = request['model'], request['devices'], request['selection'], request['task']['scope']
        numerical = {'quant': model['quant'], 'fit_device': devices['fit'], 'native_device': devices['native'], 'recipe': self.context.recipe,
                     'budget': self.context.budget, 'fit_situations': selection['fit_situations'], 'acceptance_fit': selection['acceptance_fit'], 'scope': scope}
        if name == 'assets':
            return {'manifest': model['assets'], 'quant': model['quant']} if 'assets' in model else {'repo': model['repo'], 'commit': self.revision(journal), 'quant': model['quant']}
        return {'freeze': {'sources': request['sources'], 'scope': scope}, 'export': {'family': request['task']['family'], 'scope': scope, 'native_device': devices['native']},
                'population': numerical, 'profile': request['profile'], 'cadence': {}, 'finalize': {},
                'fit': {'attempt': self.fit_attempt(journal)}, 'select': {'representation': request['representation']}, 'lock': {}}[name]

    def revision(self, journal) -> str:
        from .assets import COMMIT, revision_of
        model = self.request['model']
        if COMMIT.fullmatch(model['revision']):
            return model['revision']
        key = artifact_digest({'decision': 'revision', 'repo': model['repo'], 'revision': model['revision']})
        if key not in journal.current:
            commit = revision_of(model['repo'], model['revision'])
            journal.commit([(key, {'stage': 'assets', 'state': 'decided', 'repo': model['repo'], 'revision': model['revision'], 'commit': commit})])
            self.emit(stage='assets', state='decided', revision=model['revision'], commit=commit)
        return journal.payload(key)['commit']

    def fit_attempt(self, journal) -> int:
        if self.attempt is not None:
            return self.attempt
        return max((record['settings']['attempt'] for record in journal.payloads() if record.get('stage') == 'fit'), default=1)

    def completed(self, name, journal, key):
        record = journal.payload(key) if key in journal.current else None
        if record is None or record.get('state') != 'completed':
            return None
        found = record['output']
        try:
            if found['root'] is not None:
                verify_files(Path(found['root']), found['files'], f'committed {name} output changed or missing')
            self.operations[name].verify(self.context, found)
        except (ConfigError, ValueError, OSError) as error:
            raise ConfigError(f'stage {name}: committed output {found["path"]} no longer verifies ({error}); restore it, or rerun into a new work root') from None
        return found

    def execute(self, name, journal, key, settings):
        execution = uuid.uuid4().hex
        directory = self.work / name / f'{key[:16]}-{execution[:8]}'
        if name in ('fit', 'profile'):
            prior = [record for record in journal.payloads() if record.get('computation') == key and record.get('directory')]
            if prior:
                directory = Path(prior[-1]['directory'])
        if name in ('fit', 'profile'):
            journal.commit([(artifact_digest({'start': key, 'execution': execution}),
                             {'stage': name, 'state': 'started', 'computation': key, 'execution': execution, 'settings': settings, 'directory': str(directory)})])
        self.emit(stage=name, state='started', computation=key, execution=execution)
        try:
            found = self.operations[name].run(self.context, directory, settings)
            self.operations[name].verify(self.context, found)
        except Awaiting:
            raise
        except BaseException as error:
            state = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
            try:
                journal.commit([(artifact_digest({'failure': key, 'execution': execution}),
                                 {'stage': name, 'state': state, 'computation': key, 'execution': execution, 'settings': settings, 'directory': str(directory), 'recovery': str(directory / 'recovery.json') if (directory / 'recovery.json').exists() else None, 'error': f'{type(error).__name__}: {error}'[:4000]})])
                self.emit(stage=name, state=state, execution=execution)
            except Exception:
                pass
            raise
        finally:
            gc.collect()
        journal.commit([(key, {'stage': name, 'state': 'completed', 'computation': key, 'execution': execution, 'settings': settings,
                               'inputs': {dep: self.context.outputs[dep]['id'] for dep in NEEDS[name] if dep != 'profile' or dep in self.context.outputs}, 'output': found})])
        return found

    def run(self) -> dict:
        with Journal(self.work / JOURNAL, self.request) as journal:
            for name in STAGES:
                if name == 'profile' and self.request['profile'] is None:
                    continue
                self.stage = name
                settings = self.settings(name, journal)
                key = computation(name, settings, self.context.outputs, self.built)
                found = self.completed(name, journal, key)
                state = 'reused' if found is not None else 'completed'
                if found is None:
                    try:
                        found = self.execute(name, journal, key, settings)
                    except Awaiting as waiting:
                        return self.result('awaiting_configuration', choices=waiting.choices,
                                           resume={'ss': ['ss', 'specialize', '--to', str(self.work), '--cadence', 'N'],
                                                   'trainer': [sys.argv[0], 'pipeline', str(self.path), '--cadence', 'N']})
                self.context.outputs[name] = found
                self.states[name] = state
                self.emit(stage=name, state=state, id=found['id'])
                if name == self.until:
                    return self.result('stopped')
        return self.result('completed')

    def result(self, state, **values) -> dict:
        outputs = self.context.outputs
        def path(name, key=None):
            return None if name not in outputs else outputs[name]['summary'].get(key) if key else outputs[name]['path']
        return terminal(state, stage=self.stage, request=str(self.path), request_id=artifact_digest(self.request), work=str(self.work),
                        stages={name: {'state': self.states[name], 'kind': outputs[name]['kind'], 'id': outputs[name]['id'], 'path': outputs[name]['path']}
                                for name in STAGES if name in self.states},
                        lock=path('lock'), receipt=path('lock', 'receipt'), representation=self.request['representation'], profile=path('profile'),
                        job=path('finalize'), run=path('fit'), cadence=path('cadence', 'cadence'),
                        base={'path': path('assets', 'base'), 'sha256': path('assets', 'sha256'), 'manifest': outputs['assets']['id'] if 'assets' in outputs else None},
                        selection=path('select', 'candidates'), environment=environment(self.request), **values)


def _interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def command(request, *, until=None, attempt=None, cadence=None) -> int:
    sys.stdout.flush()
    stdout = os.dup(1)
    os.dup2(2, 1)
    previous = signal.signal(signal.SIGTERM, _interrupted)
    controller = None
    try:
        try:
            controller = Controller(request, until=until, attempt=attempt, cadence=cadence)
            result, code = controller.run(), 0
        except ConfigError as error:
            result, code = failure(controller, 'failed', error), 2
        except KeyboardInterrupt as error:
            result, code = failure(controller, 'cancelled', error), 130
        except Exception as error:
            traceback.print_exc()
            result, code = failure(controller, 'failed', error), 1
    finally:
        sys.stdout.flush()
        os.dup2(stdout, 1)
        os.close(stdout)
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(result, default=str), flush=True)
    return code


def failure(controller, state, error) -> dict:
    described = f'{type(error).__name__}: {error}'
    if controller is None:
        return terminal(state, stage=None, request=None, error=described)
    return controller.result(state, error=described)


def protocol() -> dict:
    return {'request': REQUEST_SCHEMA, 'result': RESULT_SCHEMA, 'version': __version__, 'stages': list(STAGES)}
