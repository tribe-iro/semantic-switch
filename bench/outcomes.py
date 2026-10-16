from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import warnings

import semantic_switch as ss
from semantic_switch import ConfigError, Runtime, files
from semantic_switch.api import native_settings
from semantic_switch.artifacts import artifact_digest
from semantic_switch.journal import Journal, replay
from semantic_switch.results import thaw, freeze
from semantic_switch.training import Progress
from semantic_switch.metrics import correctness, percentile

CODE_ROOT = Path(__file__).resolve().parents[1]


_SOURCES = ContextVar('benchmark_sources', default=None)


def fingerprint(root=CODE_ROOT):
    current = _SOURCES.get()
    if current is not None and current[0] == Path(root).resolve():
        return current[1]
    return _fingerprint(root)


def _fingerprint(root):
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    paths = [root / 'pyproject.toml']
    for directory in ('semantic_switch', 'bench'):
        paths.extend((root / directory).rglob('*.py'))
    listing = files.source_content(root, (path for path in paths if path.is_file()))
    return freeze({'revision': revision, 'sources': artifact_digest(listing)})


def provenance(rt, inputs=(), *, boundary="public SDK", snapshot=None):
    return {"schema": 2, **(snapshot if snapshot is not None else fingerprint()),
            "dataset": artifact_digest(inputs),
            "execution": rt.info(), "boundary": boundary}


QUANTILES = (.5, .95)


def latency(values, unit="ms") -> dict:
    values = sorted(float(value) for value in values)
    return {"unit": unit, "samples": len(values), "interpolation": "linear between closest ranks",
            **{f"p{round(q * 100)}": percentile(values, q) if values else None for q in QUANTILES}}


def ece(confidence, correct, *, low=0., bins=10) -> dict:
    pairs = [(float(p), bool(c)) for p, c in zip(confidence, correct, strict=True)]
    width = (1 - low) / bins
    table = [[] for _ in range(bins)]
    for p, c in pairs:
        if not low <= p <= 1:
            raise ValueError(f"confidence {p} outside [{low}, 1]")
        table[min(int((p - low) * bins / (1 - low)), bins - 1)].append((p, c))
    rows = [{"lower": low + k * width, "upper": low + (k + 1) * width, "count": len(group),
             "mean_confidence": math.fsum(p for p, _ in group) / len(group) if group else None,
             "accuracy": sum(c for _, c in group) / len(group) if group else None} for k, group in enumerate(table)]
    value = math.fsum(row["count"] / len(pairs) * abs(row["accuracy"] - row["mean_confidence"]) for row in rows if row["count"]) if pairs else None
    return {"value": value, "samples": len(pairs), "bins": rows, "edges": "equal-width over [low, 1], last bin closed"}


def output_argument(parser, name="--output", help="published report path; refused when it already exists"):
    parser.add_argument(name, type=Path, required=True, help=help)


def work_argument(parser):
    parser.add_argument("--work", type=Path, help="work/cache directory (default: OUTPUT.work beside the report)")


def work_directory(output, work=None) -> Path:
    return Path(work) if work is not None else Path(output).with_name(f"{Path(output).stem}.work")


def destination(path) -> Path:
    path = Path(path)
    if path.exists():
        raise ConfigError(f"{path}: already exists; name a new output")
    return path


def report(path, data):
    files.publish(destination(path), data)
    return data


def process(command, *, input=None, cwd=None, env=None, capture=True, check=False) -> dict:
    command = [str(part) for part in command]
    started = time.perf_counter()
    done = subprocess.run(command, input=input, text=True, cwd=cwd, env=env, capture_output=capture, stdout=None if capture else sys.stderr)
    out = {"command": command, "exit_code": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
           "elapsed_ms": (time.perf_counter() - started) * 1000}
    if check and done.returncode:
        raise RuntimeError(f"{' '.join(command[:4])} exited {done.returncode}: {(done.stderr or '')[-4000:]}")
    return out


def cli(arguments, value) -> dict:
    done = process([sys.executable, "-m", "semantic_switch", *arguments], input=json.dumps(value))
    try:
        result = json.loads(done["stdout"])
    except ValueError:
        raise RuntimeError(f"CLI emitted invalid JSON: {done['stdout']!r}; {done['stderr'][-1000:]}") from None
    return {"arguments": arguments, "result": result, "exit_code": done["exit_code"], "elapsed_ms": done["elapsed_ms"], "stderr": done["stderr"]}


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


def usage(rows):
    totals = {}
    for row in rows:
        for name, value in (row["usage"] if isinstance(row["usage"], dict) else {}).items():
            if type(value) is int:
                totals[name] = totals.get(name, 0) + value
    return {"reported": sum(isinstance(row["usage"], dict) for row in rows), **totals}


class Receipts:
    def __init__(self):
        self.rows, self._seen, self._lock = [], set(), threading.Lock()
        self._operation = ContextVar("bench_operation", default=None)

    def __len__(self):
        return len(self.rows)

    @contextmanager
    def operation(self, name, *, warmup=False):
        token = self._operation.set({"id": name, "warmup": warmup})
        try:
            yield
        finally:
            self._operation.reset(token)

    def __call__(self, event):
        operation, found = self._operation.get(), {}
        for name, fact in event.facts.items():
            for request in fact["requests"]:
                entry = found.setdefault(request["id"], {**thaw(request), "targets": [], "models": set()})
                entry["targets"].append(name)
                entry["models"].update(fact["observed_models"] or ())
        with self._lock:
            for key, entry in found.items():
                if key not in self._seen:
                    self._seen.add(key)
                    self.rows.append({**entry, "models": sorted(entry["models"]), "operation": operation, "call": event.call, "item": event.item})

    @staticmethod
    def failed(row) -> bool:
        return row["status"] is None or not 200 <= row["status"] < 300

    def since(self, start: int) -> list:
        with self._lock:
            return [dict(row) for row in self.rows[start:]]

    def summary(self) -> dict:
        rows = self.since(0)
        operations = {}
        for row in rows:
            operations.setdefault((row["operation"] or {}).get("id", "unidentified"), []).append(row)
        return {"source": "request receipts of public decision events", "requests": len(rows), "failed": sum(map(self.failed, rows)),
                "attempts": sum(row["attempts"] for row in rows), "usage": usage(rows), "latency": latency([row["seconds"] * 1000 for row in rows]),
                "models": sorted({model for row in rows for model in row["models"]}),
                "by_operation": {operation: {"requests": len(group), "usage": usage(group),
                                             "warmup": all((row["operation"] or {}).get("warmup", False) for row in group)}
                                 for operation, group in operations.items()},
                "receipts": rows}


@contextmanager
def runtime(judge, *, job=None, receipts=None, **roles):
    snapshot = fingerprint()
    token = _SOURCES.set((CODE_ROOT.resolve(), snapshot))
    try:
        with Runtime(bindings={"judge": judge, **roles}, own_judges=True, job=job, on_decision=receipts) as bound:
            yield bound
    finally:
        _SOURCES.reset(token)
        if _fingerprint(CODE_ROOT) != snapshot:
            warnings.warn('benchmark sources changed during the run; the receipt identifies its starting snapshot', RuntimeWarning, stacklevel=2)


JEV = "openrouter:typesafe/jev-1.13"


def judge_spec(spec, *, provider=None, key_file=None, batch_inputs=None, **native):
    if not spec.startswith("openrouter:"):
        if batch_inputs is not None:
            raise ConfigError("batch_inputs applies only to openrouter:MODEL judges")
        return ss.judge(spec, **{key: value for key, value in native.items() if value is not None})
    from bench.credentials import read_key
    return ss.judge(spec, provider=provider, key=read_key(key_file) if key_file else None,
                    **({} if batch_inputs is None else {"batch_inputs": batch_inputs}))


@contextmanager
def judged(spec, *, provider=None, key_file=None, batch_inputs=None, **native):
    receipts = Receipts()
    with runtime(judge_spec(spec, provider=provider, key_file=key_file, batch_inputs=batch_inputs, **native), receipts=receipts) as bound:
        yield bound, receipts


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
    return [journal.payload(key) for key in keys if key not in todo]


def failure(measured, **known):
    return {'outcome': 'failed', 'error': measured.error, 'ms': measured.ms, 'native_work': measured.work, 'unavailable': dict(measured.unavailable), **known}


def finish(journal, output, summarize):
    return journal.publish(output, summarize(list(journal.payloads())))


def recorded(output):
    header, current, _, _ = replay(journal_path(output))
    if header is None:
        raise ConfigError(f'{journal_path(output)}: no recorded run')
    with Journal(journal_path(output), header['configuration']) as journal:
        return header, {key: journal.payload(key) for key in current}


def prediction_value(result):
    prediction = result.get('prediction')
    if prediction is None:
        return None, False
    return (prediction['value'] if prediction['kind'] == 'value' else None), True


def task(runtime, item, state, guidance=None) -> dict:
    previewed = runtime.preview(state, {"item": item}, guidance=guidance).as_dict()
    return {"definition": thaw(item.describe()), "error": previewed["error"], "stages": previewed["stages"],
            "requests": [{**{key: value for key, value in request.items() if key in ("representation", "binding", "targets", "prompts", "body", "endpoint", "joint")},
                          "digest": artifact_digest(request)} for request in previewed["requests"]]}


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
