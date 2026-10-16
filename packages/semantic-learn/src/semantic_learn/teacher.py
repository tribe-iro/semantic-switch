from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import resource

from semantic_switch import ConfigError, Runtime
from semantic_switch.training import Journal, LabelSource, Task, TrainingUse, artifact_digest, files, judge, restriction, thaw

from .common import available

SCHEMA = 3
REASONS = ('unexamined', 'missing', 'failed', 'rule', 'uncertain', 'drift')
PROJECTION = 'declared-answers@2'
SPECIAL = {('needs_input', 'unstated'): 'unstated', ('none', 'none'): 'none'}


def permitted(teacher):
    reason = restriction(teacher)
    if reason:
        raise ConfigError(f'{teacher} cannot label training data: {reason}. Use a teacher whose terms permit training use, '
                          f'e.g. --teacher openrouter:google/gemma-4-31b-it (Apache-2.0) or a Qwen 3.5/3.6 model')


def requested(teacher, provider, identity):
    model = identity.get('model', teacher)
    return f'{model}@{provider}' if provider else model


def outcome(result, address, target):
    path = address.split('/')
    found = result.get(path[0])
    if found is not None and len(path) > 1:
        found = found.members.get(path[1])
    if found is None:
        return {'outcome': 'unexamined'}
    detail = {'status': found.status, 'why': found.why, 'prediction': thaw(found.prediction.as_dict()) if found.prediction else None}
    if found.status == 'failed':
        return {'outcome': 'failed', **detail}
    if found.source not in (None, 'judge'):
        return {'outcome': 'rule', **detail}
    answer = declared(found, target['answers'])
    if answer is not None:
        return {'value': target['answers'][answer], 'models': sorted(found.observed_models)}
    return {'outcome': 'missing' if found.status == 'needs_input' else 'uncertain', **detail}


def declared(found, answers):
    prediction = found.prediction
    if found.source != 'judge' or prediction is None:
        return None
    if prediction.kind == 'value':
        return prediction.key if found.settled and found.status != 'needs_input' and prediction.key in answers else None
    special = SPECIAL.get((found.status, prediction.kind))
    return special if special in answers and (prediction.kind != 'none' or found.settled) else None


def unit(targets, record, result, pin):
    outcomes, requests = {}, {}
    for address, target in targets.items():
        path = address.split('/')
        found = result.get(path[0])
        member = found.members.get(path[1]) if found is not None and len(path) > 1 else found
        for request in getattr(member, 'requests', ()) or ():
            requests[request.id] = request.as_dict()
        settled = outcome(result, address, target)
        if 'value' in settled and pin is not None and set(settled['models']) - {pin}:
            settled = {'outcome': 'drift', 'models': settled['models'], 'pinned': pin}
        outcomes[address] = settled
    models = sorted({model for value in outcomes.values() for model in value.get('models', ())})
    return {'record': record, 'targets': outcomes, 'requests': list(requests.values()), 'returned_models': models, 'collected_at': files.now()}


def materialize(task, targets, payloads, identity, use, route, *, emit=None):
    labels, unresolved, counts = [], [], {'situations': 0, 'targets': 0, 'settled': 0, 'complete_situations': 0,
                                          'unresolved': dict.fromkeys(REASONS, 0), 'requests': 0, 'usage_known': 0}
    def write(kind, row):
        if emit is None:
            (labels if kind == 'labels.jsonl' else unresolved).append(row)
        else:
            emit(kind, row)
    for payload in payloads:
        counts['situations'] += 1
        facts, sources = {}, {}
        for address in targets:
            path = address.split('/')
            settled = payload['targets'][address]
            counts['targets'] += 1
            if 'value' in settled:
                counts['settled'] += 1
                for observed in settled['models'] or [route]:
                    use.require(observed)
                producer = LabelSource('judged', route, artifact_digest({'identity': identity, 'returned_models': settled['models']}), address=tuple(path),
                                       training_use=use).record()
                if len(path) > 1:
                    facts.setdefault(path[0], {})[path[1]] = settled['value']
                    sources.setdefault(path[0], {})[path[1]] = [producer]
                else:
                    facts[path[0]], sources[path[0]] = settled['value'], [producer]
            else:
                counts['unresolved'][settled['outcome']] += 1
                write('unresolved.jsonl', {'schema': SCHEMA, 'input': payload['record']['input'], 'address': path, **settled})
        counts['complete_situations'] += all('value' in payload['targets'][address] for address in targets)
        counts['requests'] += len(payload['requests'])
        counts['usage_known'] += sum(request['usage'] is not None for request in payload['requests'])
        if facts:
            record = payload['record']
            write('labels.jsonl', task.label(record['input'], facts, sources=sources, groups=record['groups'], pairs=record['pairs'], role=record.get('role'),
                                       tags=record.get('tags', ()), selections=record['selections'], created=available(payload['collected_at'], 'collection unit').isoformat()))
    if counts['settled'] + sum(counts['unresolved'].values()) != counts['targets']:
        raise ConfigError('collection target accounting does not reconcile')
    return labels, unresolved, counts


def collect(definition, inputs, to, *, teacher, training_use, provider=None, fact=None, job=None, window=32, max_situations=100000, max_bytes=64 << 20,
            retry_failed=False):
    permitted(teacher)
    use = TrainingUse.parse(training_use)
    use.require(use.route)
    to = Path(to)
    if not to.name.endswith('.collection.json'):
        raise ConfigError(f'{to}: collection manifests are named NAME.collection.json')
    if type(window) is not int or window < 1:
        raise ConfigError('window must be a positive number of situations')
    journal_path = to.with_name(to.name.removesuffix('.collection.json') + '.journal.jsonl')
    with Runtime(judge(teacher, provider=provider), own_judges=True, job=job) as runtime:
        switch = runtime.load(definition, draft=True)
        task = Task(switch)
        targets = task.targets(fact)
        records = task.situations(inputs, max_situations=max_situations, max_bytes=max_bytes)
        meanings = {address: target['meaning_rev'] for address, target in targets.items()}
        keyed = {artifact_digest([task.key(record['input']), meanings]): record for record in records}
        del records
        configuration = {'schema': SCHEMA, 'task': meanings, 'switch_rev': task.description['switch_rev'],
                         'policies': {address: target['policy'] for address, target in targets.items()}, 'source': files.sha256_file(inputs),
                         'teacher': teacher, 'provider': provider, 'training_use': use.record(), 'projection': PROJECTION,
                         'code': {name: version(name) for name in ('semantic-switch', 'semantic-learn')}}
        with Journal(journal_path, configuration) as journal:
            pending = journal.pending(keyed, (lambda payload: any(v.get('outcome') == 'failed' for v in payload['targets'].values())) if retry_failed else None)
            if pending:
                use.require(requested(teacher, provider, runtime.identity()))
            pin = next((model for payload in journal.payloads() for model in payload['returned_models']), None)
            for start in range(0, len(pending), window):
                batch = pending[start:start + window]
                scored = switch.score_facts([keyed[key]['input'] for key in batch], sources=('judge',), facts=sorted({address.split('/')[0] for address in targets}))
                units = []
                for key, result in zip(batch, scored):
                    payload = unit(targets, keyed[key], result, pin)
                    pin = pin or next(iter(payload['returned_models']), None)
                    units.append((key, payload))
                journal.commit(units)
            identity = thaw(runtime.identity())
            with files.Publication(to) as publication:
                sinks = {name: publication.payload(name) for name in ('labels.jsonl', 'unresolved.jsonl')}
                def emit(name, row):
                    sinks[name].write(files.jsonl([row]))
                _, _, counts = materialize(task, targets, (journal.payload(key) for key in keyed), identity, use,
                                           requested(teacher, provider, identity), emit=emit)
                manifest = publication.commit({'schema': SCHEMA, 'kind': 'collection',
                    'teacher': {'source': teacher, 'provider': provider, 'identity': identity, 'pinned': pin, 'training_use': use.record()},
                    'counts': counts, 'source': {'path': str(inputs), 'sha256': configuration['source'], 'bytes': Path(inputs).stat().st_size},
                    'journal_bytes': journal_path.stat().st_size, 'peak_host_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    'execution': journal.summary()})

    paths = {name: str(to.parent / entry['file']) for name, entry in manifest['payloads'].items()}
    return {'labels': paths['labels.jsonl'], 'unresolved': paths['unresolved.jsonl'], 'manifest': str(to), 'journal': str(journal_path), 'counts': counts,
            'teacher': {'source': teacher, 'provider': provider}, 'review': ['ss', 'label', str(definition), paths['labels.jsonl'], '--to', str(to.with_name(to.name.removesuffix('.collection.json') + '.reviewed.jsonl'))]}
