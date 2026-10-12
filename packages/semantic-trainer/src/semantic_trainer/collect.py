from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import resource

from semantic_switch import Runtime, files, judge_from
from semantic_switch.artifacts import artifact_digest
from semantic_switch.errors import ConfigError
from semantic_switch.examples import LabelSource
from semantic_switch.journal import Journal
from semantic_switch.results import thaw
from semantic_switch.training import Task

SCHEMA = 1
RESTRICTED = {'typesafe/jev': 'TypeSafe agreement §2.3(b) keeps Jev outputs out of training data'}
REASONS = ('unexamined', 'missing', 'failed', 'rule', 'unstated', 'uncertain', 'drift')


def permitted(teacher):
    model = teacher.removeprefix('openrouter:').lower()
    for prefix, reason in RESTRICTED.items():
        if model.startswith(prefix):
            raise ConfigError(f'{teacher} cannot label training data: {reason}. Use a teacher whose terms permit training use, '
                              f'e.g. --teacher openrouter:google/gemma-4-31b-it (Apache-2.0) or a Qwen 3.5/3.6 model')


def outcome(result, address, target):
    path = address.split('/')
    found = result.get(path[0])
    if found is not None and len(path) > 1:
        found = found.members.get(path[1])
    if found is None:
        return {'outcome': 'unexamined'}
    detail = {'status': found.status, 'why': found.why, 'prediction': thaw(found.prediction.as_dict()) if found.prediction else None}
    if found.status == 'needs_input':
        return {'outcome': 'missing', **detail}
    if found.status == 'failed':
        return {'outcome': 'failed', **detail}
    if found.source not in (None, 'judge'):
        return {'outcome': 'rule', **detail}
    if found.prediction is not None and found.prediction.kind == 'unstated':
        return {'outcome': 'unstated', **detail}
    if not found.settled or found.prediction is None or found.prediction.kind != 'value':
        return {'outcome': 'uncertain', **detail}
    return {'value': target['answers'][found.prediction.key], 'models': sorted(found.observed_models)}


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
    return {'record': record, 'targets': outcomes, 'requests': list(requests.values()), 'returned_models': models}


def materialize(task, targets, payloads, teacher, identity):
    labels, unresolved, counts = [], [], {'situations': len(payloads), 'targets': 0, 'settled': 0, 'complete_situations': 0,
                                          'unresolved': dict.fromkeys(REASONS, 0), 'requests': 0, 'usage_known': 0}
    for payload in payloads:
        facts, sources = {}, {}
        for address in targets:
            path = address.split('/')
            settled = payload['targets'][address]
            counts['targets'] += 1
            if 'value' in settled:
                counts['settled'] += 1
                producer = LabelSource('synthetic', teacher, artifact_digest({'identity': identity, 'returned_models': settled['models']}), address=tuple(path)).record()
                if len(path) > 1:
                    facts.setdefault(path[0], {})[path[1]] = settled['value']
                    sources.setdefault(path[0], {})[path[1]] = [producer]
                else:
                    facts[path[0]], sources[path[0]] = settled['value'], [producer]
            else:
                counts['unresolved'][settled['outcome']] += 1
                unresolved.append({'schema': SCHEMA, 'input': payload['record']['input'], 'address': path, **settled})
        counts['complete_situations'] += all('value' in payload['targets'][address] for address in targets)
        counts['requests'] += len(payload['requests'])
        counts['usage_known'] += sum(request['usage'] is not None for request in payload['requests'])
        if facts:
            record = payload['record']
            labels.append(task.label(record['input'], facts, sources=sources, groups=record['groups'], pairs=record['pairs'], role=record.get('role'),
                                       tags=record.get('tags', ()), selections=record['selections'], created=f'collected:{teacher}'))
    if counts['settled'] + sum(counts['unresolved'].values()) != counts['targets']:
        raise ConfigError('collection target accounting does not reconcile')
    return labels, unresolved, counts


def collect(definition, inputs, to, *, teacher, provider=None, fact=None, job=None, window=32, max_situations=100000, max_bytes=64 << 20, retry_failed=False):
    permitted(teacher)
    to = Path(to)
    if not to.name.endswith('.collection.json'):
        raise ConfigError(f'{to}: collection manifests are named NAME.collection.json')
    if type(window) is not int or window < 1:
        raise ConfigError('window must be a positive number of situations')
    journal_path = to.with_name(to.name.removesuffix('.collection.json') + '.journal.jsonl')
    with Runtime({'judge': judge_from(teacher, provider=provider)}, own_judges=True, job=job) as runtime:
        switch = runtime.load(definition, draft=True)
        task = Task(switch)
        targets = task.targets(fact)
        records = task.situations(inputs, max_situations=max_situations, max_bytes=max_bytes)
        meanings = {address: target['meaning_rev'] for address, target in targets.items()}
        keyed = {artifact_digest([task.key(record['input']), meanings]): record for record in records}
        configuration = {'schema': SCHEMA, 'task': meanings, 'switch_rev': task.description['switch_rev'],
                         'policies': {address: target['policy'] for address, target in targets.items()}, 'source': files.sha256_file(inputs),
                         'teacher': teacher, 'provider': provider,
                         'code': {name: version(name) for name in ('semantic-switch', 'semantic-trainer')}}
        with Journal(journal_path, configuration) as journal:
            pending = journal.pending(keyed, (lambda payload: any(v.get('outcome') == 'failed' for v in payload['targets'].values())) if retry_failed else None)
            if pending:
                permitted(str(runtime.identity().get('model', teacher)))
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
            labels, unresolved, counts = materialize(task, targets, [journal.current[key]['payload'] for key in keyed], teacher, identity)
            manifest = journal.publish(to, {'schema': SCHEMA, 'kind': 'collection', 'teacher': {'source': teacher, 'provider': provider, 'identity': identity, 'pinned': pin},
                                            'counts': counts, 'source': {'path': str(inputs), 'sha256': configuration['source'], 'bytes': Path(inputs).stat().st_size},
                                            'journal_bytes': journal_path.stat().st_size, 'peak_host_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
                                       {'labels.jsonl': [json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in labels],
                                        'unresolved.jsonl': [json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in unresolved]})
    paths = {name: str(to.parent / entry['file']) for name, entry in manifest['payloads'].items()}
    return {'labels': paths['labels.jsonl'], 'unresolved': paths['unresolved.jsonl'], 'manifest': str(to), 'journal': str(journal_path), 'counts': counts,
            'teacher': {'source': teacher, 'provider': provider}, 'review': ['ss', 'label', str(definition), paths['labels.jsonl'], '--to', str(to.with_name(to.name.removesuffix('.collection.json') + '.reviewed.jsonl'))]}
