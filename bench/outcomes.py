from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time

from semantic_switch import ConfigError, Model, Runtime, files, judge_from
from semantic_switch.api import native_settings
from semantic_switch.artifacts import artifact_digest
from semantic_switch.guidance import normalize
from semantic_switch.inline import rendered
from semantic_switch.journal import Journal, replay
from semantic_switch.results import thaw
from semantic_switch.metrics import correctness
from semantic_switch.presentation import identity as presentation_identity
from bench.diagnostics import fingerprint


def materialize(value):
    if hasattr(value, 'as_dict'):
        return materialize(value.as_dict(True))
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [materialize(item) for item in value]
    return value


@dataclass(frozen=True)
class Measurement:
    result: object
    output: object
    ms: float
    work: dict | None
    peak_rss_bytes: int | None
    error: dict | None = None
    unavailable: dict = field(default_factory=dict)


def counters(runtime):
    if runtime is None:
        return None, 'no runtime bound to the measurement'
    try:
        return dict(runtime.info()['capabilities']['counters']), None
    except Exception as error:
        return None, f'counter collection failed: {type(error).__name__}'


def reset_peak():
    try:
        Path('/proc/self/clear_refs').write_text('5')
        return None
    except OSError as error:
        return f'resident-set reset failed: {type(error).__name__}'


def peak_rss():
    try:
        return next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')), None
    except (OSError, StopIteration, ValueError) as error:
        return None, f'resident-set collection failed: {type(error).__name__}'


def measure(operation, *, runtime=None, clock=time.monotonic):
    before, missing = counters(runtime)
    unpeaked = reset_peak()
    result = output = error = None
    stage = 'execute'
    started = clock()
    try:
        result = operation()
        stage = 'materialize'
        output = materialize(result)
        stage = 'serialize'
        json.dumps(output, allow_nan=False)
    except Exception as caught:
        result = output = None
        error = {'type': type(caught).__name__, 'stage': stage}
    finally:
        elapsed = (clock() - started) * 1000
        after, late = counters(runtime) if before is not None else (None, missing)
        peak, unread = peak_rss()
    unavailable = {**({'native_work': late} if after is None else {}), **({'peak_rss_bytes': unpeaked or unread} if unpeaked or unread else {})}
    work = None if after is None else {key: after[key] - before.get(key, 0) for key in after}
    return Measurement(result, output, elapsed, work, None if unpeaked else peak, error, unavailable)


@contextmanager
def runtime(judge, *, job=None, **roles):
    with Runtime({'judge': judge, **roles}, own_judges=True, job=job) as bound:
        yield bound


JEV = 'openrouter:typesafe/jev-1.13'


def judge_spec(spec, *, provider=None, key_file=None, batch_inputs=None, **native):
    if not spec.startswith('openrouter:'):
        if batch_inputs is not None:
            raise ConfigError('batch_inputs applies only to openrouter:MODEL judges')
        native = {key: value for key, value in native.items() if value is not None}
        return (Model.from_specialization(spec, **native) if spec.endswith('.json') else judge_from(spec, **native)), None
    from bench.credentials import read_key
    from bench.receipts import RecordingTransport
    recording = RecordingTransport(key=read_key(key_file) if key_file else None)
    return judge_from(spec, provider=provider, transport=recording, own_transport=True,
                      **({} if batch_inputs is None else {'batch_inputs': batch_inputs})), recording


@contextmanager
def judged(spec, *, provider=None, key_file=None, batch_inputs=None, **native):
    judge, recording = judge_spec(spec, provider=provider, key_file=key_file, batch_inputs=batch_inputs, **native)
    with runtime(judge) as bound:
        yield bound, recording


def configuration(script, args, *models, **inputs):
    return {'script': script, 'args': json.loads(json.dumps({key: value for key, value in sorted(vars(args).items()) if key != 'fresh'}, default=str)),
            'models': [{'source': model.source, 'native': native_settings(model), 'sha256': files.sha256_file(model.path())}
                       for model in models],
            'inputs': inputs, 'code': fingerprint()}


def journal_path(output):
    return Path(f'{output}.journal.jsonl')


def work_key(*identity):
    return artifact_digest(['bench', *identity])


def resumable(output, config, *, fresh=False):
    path = journal_path(output)
    if fresh and path.exists():
        os.replace(path, Path(f'{path}.{files.now()}'))
    return Journal(path, config)


def resumed(journal, keys, todo):
    todo = set(todo)
    return [journal.current[key]['payload'] for key in keys if key not in todo]


def failure(measured, **known):
    return {'outcome': 'failed', 'error': measured.error, 'ms': measured.ms, 'native_work': measured.work, 'unavailable': dict(measured.unavailable), **known}


def finish(journal, output, summarize):
    return journal.publish(output, summarize(journal.payloads()))


def recorded(output):
    header, current, _, _ = replay(journal_path(output))
    if header is None:
        raise ConfigError(f'{journal_path(output)}: no recorded run')
    return header, {key: record['payload'] for key, record in current.items()}


def prediction_value(result):
    prediction = result.get('prediction')
    if prediction is None:
        return None, False
    return (prediction['value'] if prediction['kind'] == 'value' else None), True


def task(runtime, item, state, guidance=None, result=None):
    binding = normalize(item.guidance if item.guidance is not None else guidance)
    return {'presentation': presentation_identity(), 'rendered_evidence': rendered(state), 'definition': thaw(item.describe()),
            'guidance': binding.as_dict(), 'guidance_digest': binding.digest, 'presentations': (result or {}).get('presentations', []),
            'truncated': (result or {}).get('truncated', [])}


def classification_metrics(rows):
    raw = [(row['raw_value'], row['raw_usable']) if 'raw_value' in row else prediction_value(row['result']) for row in rows]
    gold = [row['expected'].get('raw_value', row['expected']['value']) for row in rows]
    accepted = [row['result']['status'] == 'ok' for row in rows]
    return {**correctness((value == expected if usable else row['result']['value'] == row['expected']['value'] if ok else None, ok)
                          for row, (value, usable), expected, ok in zip(rows, raw, gold, accepted)),
            'unique_cases': len({row['case_id'] for row in rows}), 'families': len({row['family'] for row in rows}),
            'status_agreement': sum(row['result']['status'] == row['expected']['status'] for row in rows),
            'operational_agreement': sum((row['result']['value'], row['result']['status']) ==
                                        (row['expected']['value'], row['expected']['status']) for row in rows),
            'statuses': {status: sum(row['result']['status'] == status for row in rows)
                         for status in ('ok', 'uncertain', 'needs_input', 'failed', 'none')},
            'protocol_failures': sum(row['result']['status'] == 'failed' for row in rows)}
