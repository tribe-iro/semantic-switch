from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from .. import files
from ..compiler import digest
from ..errors import ConfigError

REQUEST_SCHEMA, RESULT_SCHEMA = 2, 1
REQUEST_KIND, RESULT_KIND, REQUEST_NAME = 'specialization-request', 'specialization-result', 'request.json'
STATES = ('completed', 'stopped', 'awaiting_configuration', 'failed', 'cancelled')
REPRESENTATIONS, NATIVE_DEVICES = ('merged', 'runtime-adapter'), ('auto', 'cpu', 'gpu')
FIELDS = {'schema', 'kind', 'sources', 'task', 'model', 'devices', 'recipe', 'budget', 'selection', 'profile', 'cadence', 'representation', 'work'}
COMMIT, SETTING = re.compile('[0-9a-f]{40}'), re.compile(r'[A-Za-z_]+=.+')
GRACE, HANDSHAKE = 30., 60.
INSTALL = ('install semantic-trainer with its fitting extras in the fitting environment (pip install "semantic-trainer[lora]") and configure its absolute '
           'executable with --trainer PATH or SEMANTIC_TRAINER=PATH')


def sources(definition, labels) -> dict:
    from ..training import describe_task
    definition, labels = Path(definition).resolve(), Path(labels).resolve()
    if not labels.is_file():
        raise ConfigError(f'{labels}: label file not found')
    sidecar = files.Paths.of(definition).examples
    return {'definition': {'path': str(definition), 'sha256': files.sha256_file(definition)}, 'description': digest(describe_task(definition)),
            'corrections': files.sha256_file(sidecar) if sidecar.is_file() else None, 'labels': {'path': str(labels), 'sha256': files.sha256_file(labels)}}


def request(definition, labels, work, *, devices, model=None, assets=None, quant='Q8_0', recipe=(), budget=(), fit_situations=None, acceptance_fit=False,
            task_family=None, scope=None, profile=None, measure=None, cadence=None, representation='merged') -> dict:
    from ..api import located
    definition = located(definition)
    if (model is None) == (assets is None):
        raise ConfigError('specialize names either a base model {repo, revision} or a verified trainer asset manifest digest')
    built = {'schema': REQUEST_SCHEMA, 'kind': REQUEST_KIND, 'sources': sources(definition, labels),
             'task': {'family': task_family or f'local:{Path(definition).stem}', 'scope': sorted(scope) if scope else None},
             'model': {'assets': assets, 'quant': quant} if assets else {**model, 'quant': quant}, 'devices': dict(devices),
             'recipe': sorted(recipe), 'budget': sorted(budget), 'selection': {'fit_situations': fit_situations, 'acceptance_fit': bool(acceptance_fit)},
             'profile': {'path': str(Path(profile).resolve()), 'sha256': files.sha256_file(profile)} if profile else ({'measure': {'steps': 3, 'warmup': 1, 'cadences': [], **measure}} if measure is not None else None),
             'cadence': cadence, 'representation': representation, 'work': str(Path(work).resolve())}
    return decode_request(built, 'specialize')


def decode_request(record, where) -> dict:
    def fail(message):
        raise ConfigError(f'{where}: {message}; write a new request with ss specialize')
    if not isinstance(record, dict) or record.get('schema') != REQUEST_SCHEMA or record.get('kind') != REQUEST_KIND or set(record) != FIELDS:
        fail(f'not a schema-{REQUEST_SCHEMA} specialization request')
    model, devices, selection = record['model'], record['devices'], record['selection']
    if not isinstance(model, dict) or not isinstance(model.get('quant'), str) or set(model) not in ({'assets', 'quant'}, {'repo', 'revision', 'quant'}) \
            or not all(isinstance(value, str) and value for value in model.values()):
        fail('model is {repo, revision, quant} or {assets, quant}')
    if not isinstance(devices, dict) or set(devices) != {'fit', 'native'} or not isinstance(devices['fit'], str) or devices['native'] not in NATIVE_DEVICES:
        fail(f"devices are {{fit: cpu|xpu:N|cuda:N, native: {'|'.join(NATIVE_DEVICES)}}}")
    if any(not isinstance(values, list) or not all(isinstance(v, str) and SETTING.fullmatch(v) for v in values) for values in (record['recipe'], record['budget'])):
        fail('recipe and budget are KEY=VALUE lists')
    if any(value.startswith('eval_every_examples=') for value in record['recipe']):
        fail('the monitoring cadence is a separate decision: give --cadence N, not a recipe eval_every_examples')
    if not isinstance(selection, dict) or set(selection) != {'fit_situations', 'acceptance_fit'} or not isinstance(selection['acceptance_fit'], bool) \
            or not (selection['fit_situations'] is None or type(selection['fit_situations']) is int and selection['fit_situations'] > 0):
        fail('selection is {fit_situations: positive integer or null, acceptance_fit: boolean}')
    if record['cadence'] is not None and (type(record['cadence']) is not int or record['cadence'] < 1):
        fail('cadence is a positive number of examples between evaluation points')
    if record['representation'] not in REPRESENTATIONS:
        fail(f"representation must be one of {', '.join(REPRESENTATIONS)}")
    task, profile = record['task'], record['profile']
    if not isinstance(task, dict) or set(task) != {'family', 'scope'} or not isinstance(task['family'], str) or ':' not in task['family']:
        fail('task names a namespaced task family NAMESPACE:NAME and an optional target scope')
    if profile is not None and (not isinstance(profile, dict) or set(profile) not in ({'path', 'sha256'}, {'measure'})):
        fail('profile is null, an existing {path, sha256}, or {measure: {steps, warmup, cadences}}')
    if not isinstance(record['work'], str) or not Path(record['work']).is_absolute():
        fail('work is an absolute directory')
    return record


def read_request(path) -> dict:
    path = Path(path)
    record = files.read_json(path)
    if record is None:
        raise ConfigError(f'{path}: no specialization request; start one with ss specialize SWITCH --examples LABELS --to WORK')
    record = decode_request(record, str(path))
    if Path(record['work']) != path.resolve().parent:
        raise ConfigError(f"{path}: the request belongs to work root {record['work']}; requests are not relocated")
    return record


def terminal(state, **values) -> dict:
    if state not in STATES:
        raise ConfigError(f'unknown specialization state {state!r}')
    return {'schema': RESULT_SCHEMA, 'kind': RESULT_KIND, 'state': state, **values}


def read_result(text, code, argv) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    try:
        result = json.loads(lines[0]) if len(lines) == 1 else None
    except ValueError:
        result = None
    if not isinstance(result, dict) or result.get('schema') != RESULT_SCHEMA or result.get('kind') != RESULT_KIND or result.get('state') not in STATES:
        raise ConfigError(f'{argv[0]} exited {code} without one schema-{RESULT_SCHEMA} specialization result on stdout; see its stderr above')
    if (code == 0) != (result['state'] in ('completed', 'stopped', 'awaiting_configuration')):
        raise ConfigError(f"{argv[0]} exited {code} with result state {result['state']}; the worker broke the result protocol")
    return {**result, 'exit': code}


def executable(trainer) -> str:
    if not trainer:
        raise ConfigError(f'no trainer executable is configured; {INSTALL}')
    path = Path(trainer)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ConfigError(f'{trainer}: the trainer must be an absolute path to an executable file; {INSTALL}')
    try:
        answered = subprocess.run([str(path), 'protocol'], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=HANDSHAKE)
        declared = json.loads(answered.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise ConfigError(f'{path} does not answer the specialization protocol ({type(error).__name__}); {INSTALL}') from None
    expected = {'request': REQUEST_SCHEMA, 'result': RESULT_SCHEMA}
    if not isinstance(declared, dict) or {key: declared.get(key) for key in expected} != expected or not isinstance(declared.get('stages'), list):
        raise ConfigError(f'{path} speaks protocol {declared}, not {expected}; install the semantic-trainer release built for this semantic-switch; {INSTALL}')
    return str(path), declared['stages']


def _interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def terminate(child, grace):
    try:
        if os.name == 'posix':
            os.killpg(child.pid, signal.SIGTERM)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                child.poll()
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(min(.05, max(0., end - time.monotonic())))
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            child.terminate()
            try:
                child.wait(grace)
            except subprocess.TimeoutExpired:
                child.kill()
        child.wait()
    except OSError:
        pass


def invoke(argv, grace=GRACE) -> dict:
    main = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _interrupted) if main else None
    try:
        with subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, start_new_session=os.name == 'posix') as child:
            try:
                out, _ = child.communicate()
            except BaseException:
                terminate(child, grace)
                raise
    finally:
        if main:
            signal.signal(signal.SIGTERM, previous)
    if os.name == 'posix':
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            pass
        else:
            terminate(child, grace)
    return read_result(out.decode('utf-8', 'replace'), child.returncode, argv)


def specialize(trainer, work, specification=None, *, until=None, attempt=None, cadence=None, grace=GRACE) -> dict:
    trainer, stages = executable(trainer)
    if until is not None and until not in stages:
        raise ConfigError(f"--until {until!r} names no pipeline stage; stages: {', '.join(stages)}")
    path = Path(work).resolve() / REQUEST_NAME
    if specification is None:
        read_request(path)
    else:
        existing = files.read_json(path)
        if existing is None:
            files.write_new(path, specification)
        elif existing != specification:
            raise ConfigError(f'{path.parent} holds another specialization request; resume it with ss specialize --to {path.parent} and no SWITCH, '
                              'or write this request to a new --to')
    flags = [*(['--until', until] if until else []), *(['--attempt', str(attempt)] if attempt is not None else []),
             *(['--cadence', str(cadence)] if cadence is not None else [])]
    return invoke([trainer, 'pipeline', str(path), *flags], grace)
