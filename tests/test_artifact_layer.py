from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

from semantic_switch import Actor, ConfigError, Runtime, files, tools
from semantic_switch.artifacts import candidate_digest, read_export
from semantic_switch.compiler import digest
from semantic_switch.tools.data import Labels
from semantic_switch.tools.dataset import EvidenceContext, evaluation_plan
from semantic_switch.tools.lock import build_lock

from semantic_testkit.scoring import FakeScorer
from .test_artifact_pipeline import dataset, definition
from semantic_testkit.labels import envelopes

RANDOM = {'selected_by': 'random', 'frame_id': 'audit', 'independent': True, 'inclusion_probability': .5}


def cohort(path, prefix, n):
    files.publish(path, ''.join(json.dumps(row) + '\n' for row in envelopes([{'input': {'message': f'{"refund" if i % 2 else "hello"} {prefix}{i}'}, 'facts': {'refund': bool(i % 2)},
                                                                                'group': f'{prefix}{i}', 'selection': RANDOM} for i in range(n)])))
    return path


def test_new_artifacts_respect_umask(tmp_path):
    previous = os.umask(0o027)
    try:
        files.publish(tmp_path / 'a.json', {'x': 1})
        files.append(tmp_path / 'a.jsonl', files.jsonl([{'x': 1}]))
        stored = files.publish(tmp_path / 'm.json', {'x': 1}, {'p.jsonl': ['{}\n']})['payloads']['p.jsonl']['file']
    finally:
        os.umask(previous)
    assert {(tmp_path / name).stat().st_mode & 0o777 for name in ('a.json', 'a.jsonl', 'm.json', stored)} == {0o640}
    os.chmod(tmp_path / 'a.json', 0o600)
    files.publish(tmp_path / 'a.json', {'x': 2})
    assert (tmp_path / 'a.json').stat().st_mode & 0o777 == 0o600


def test_locked_reentrant_same_thread(tmp_path):
    target = tmp_path / 'artifact.json'
    probe = '''import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(3)
'''
    outcome = {}

    def nested():
        with files.locked(target):
            with files.locked(target):
                files.publish(target, {'nested': True})
                outcome['probe'] = subprocess.run([sys.executable, '-c', probe, str(files.lock_file(target))]).returncode
                outcome['holder'] = subprocess.run([sys.executable, '-c', 'import sys\nfrom semantic_switch import files\nwith files.locked(sys.argv[1], blocking=False): pass',
                                                    str(target)], capture_output=True, text=True).stderr
    worker = threading.Thread(target=nested)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert outcome['probe'] == 3
    assert f'"pid": {os.getpid()}' in outcome['holder'] and 'ConfigError' in outcome['holder']
    with files.locked(target, blocking=False):
        assert files.read_published(target) == {'nested': True}


def test_publish_crash_before_manifest_keeps_previous_export(tmp_path, monkeypatch):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        exported = tools.export(rt, path)
        before = read_export(exported['path'])
        module = __import__('semantic_switch.tools.export', fromlist=['input_key'])
        calls = []

        def crash(plan, data):
            calls.append(data)
            if len(calls) > 3:
                raise RuntimeError("crash")
            return digest(data)
        monkeypatch.setattr(module, 'input_key', crash)
        with pytest.raises(RuntimeError, match="crash"):
            tools.export(rt, path)
        monkeypatch.undo()
        monkeypatch.setattr(files, '_encode', lambda path, data: (_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError):
            tools.export(rt, path)
    assert read_export(exported['path']) == before
    assert not [p for p in tmp_path.iterdir() if p.name.endswith('.tmp')]
    entry = next(value for name, value in before[0]['payloads'].items() if name.endswith('.observations.jsonl'))
    (tmp_path / entry['file']).write_text('')
    with pytest.raises(ConfigError, match='incomplete publication; re-run'):
        read_export(exported['path'])


def test_audit_log_appends_without_rewrite(tmp_path):
    path = definition(tmp_path)
    dataset(path, 20)
    with Runtime(FakeScorer()) as rt:
        labels = Labels(rt.load(path, draft=True))
        families = {e.group for e in labels.examples('refund', 'check')}
        context = EvidenceContext.of(labels)
        context.consult(evaluation_plan({'candidate': 'first'}, scope=[]), families)
        log = path.with_suffix('.audit.jsonl')
        inode, prefix = log.stat().st_ino, log.read_bytes()
        assert context.consult(evaluation_plan({'candidate': 'second'}, scope=[]), families)['state'] == 'consulted'
    assert log.stat().st_ino == inode and log.read_bytes().startswith(prefix) and len(files.read_jsonl(log)) == 2
    with log.open('ab') as stream:
        stream.write(b'{"schema": 1, "cand')
    assert len(files.read_jsonl(log)) == 2
    files.append(log, files.jsonl([{'schema': 1, 'candidate_id': 'third', 'families': []}]))
    assert [r.get('candidate_id') or r['plan']['arms']['candidate'] for r in files.read_jsonl(log)] == ['first', 'second', 'third']


def test_digest_cache_entry_write_is_constant_size(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    sizes = []
    for i in range(40):
        source = tmp_path / f'payload{i:03}.bin'
        source.write_bytes(os.urandom(64))
        before = {p: p.stat().st_mtime_ns for p in (tmp_path / 'cache').rglob('*.json')}
        assert files.sha256_file(source) == __import__('hashlib').sha256(source.read_bytes()).hexdigest()
        written = [p for p in (tmp_path / 'cache').rglob('*.json') if before.get(p) != p.stat().st_mtime_ns]
        assert len(written) == 1
        sizes.append(written[0].stat().st_size)
    assert len(set(sizes)) == 1 and sizes[0] < 512


def loaded(monkeypatch, path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(f"fingerprinted_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, spec.name, module)


def test_provenance_fingerprint_unambiguous_and_cwd_independent(tmp_path, monkeypatch):
    from bench.outcomes import fingerprint
    prints = []
    for name, content in (('ab', 'c = 1\n'), ('a', 'bc = 1\n')):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'x'], check=True)
        (root / 'bench').mkdir()
        (root / 'bench' / f'{name}.py').write_text(content)
        loaded(monkeypatch, root / 'bench' / f'{name}.py')
        prints.append(fingerprint(root))
    assert prints[0]['sources'] != prints[1]['sources']
    monkeypatch.chdir(tmp_path / 'ab')
    assert fingerprint(tmp_path / 'a') == prints[1] and prints[1]['revision']


def test_receipt_lineage_never_upgrades_unknown_membership(tmp_path, monkeypatch):
    from semantic_switch.artifacts import Provenance
    module = __import__('semantic_switch.tools.lock', fromlist=['specialization'])
    path = definition(tmp_path)
    dataset(path)
    with Runtime(FakeScorer()) as rt:
        monkeypatch.setattr(module, 'specialization', lambda rt, role: (None, None))
        data, _, _ = build_lock(rt, path)
        assert data['lineage']['judge:refund']['state'] == 'unknown' and not data['predictor_artifacts']
        assert all(side['interval_95'] is None and side['interval']['interval'] is None for side in data['facts']['refund']['policies']['judge'].values())
        rows = ({'id': 'b' * 64, 'fact': 'refund', 'projection': '{"message": "elsewhere"}', 'group': 'elsewhere', 'family_seeds': [['group', 'elsewhere']],
                 'uses': ['selection'], 'source': 'x.export.jsonl', 'task_family': 'x:y', 'export_id': 'c' * 64},)
        provenance = Provenance('base', 'f' * 64, receipt='e' * 64, rows=rows)
        monkeypatch.setattr(module, 'specialization', lambda rt, role: (provenance, rows))
        data, _, _ = build_lock(rt, path)
    assert data['schema'] == 20 and data['lineage']['judge:refund']['state'] == 'declared_disjoint'
    assert data['predictor_artifacts']['judge']['gguf_sha256'] == 'f' * 64 and data['predictor_artifacts']['judge']['rows'][0]['uses'] == ['selection']


@pytest.mark.parametrize('schema', [6, 10, 11, 13, 15])
def test_obsolete_lock_schema_is_rejected_before_policy_use(tmp_path, schema):
    from semantic_switch.snapshot import from_lock
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        data, _, _ = build_lock(rt, path)
        data = {k: v for k, v in data.items() if k not in ('id', 'created')} | {'schema': schema}
        data['candidate_id'] = candidate_digest(data)
        data['id'] = digest(data)[:12]
        with pytest.raises(ConfigError, match=f'unsupported lock schema {schema}.*regenerate this lock'):
            from_lock(data, 'lock')


def test_embedded_obsolete_dataset_is_rejected(tmp_path):
    from semantic_switch.snapshot import from_lock
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        data = files.read_json(path.with_suffix('.lock.json'))
        dataset_ = {k: v for k, v in data['dataset'].items() if k != 'id'} | {'schema': 4}
        data = {k: v for k, v in data.items() if k not in ('id', 'created')} | {'dataset': dataset_ | {'id': digest(dataset_)}}
        data['id'] = digest(data)[:12]
        with pytest.raises(ConfigError, match='embedded dataset is not a current schema-7'):
            from_lock(data, 'lock')
        files.publish(path.with_suffix('.dataset.json'), dataset_ | {'id': digest(dataset_)})
        with pytest.raises(ConfigError, match='current schema 7'):
            Labels(rt.load(path, draft=True))


def test_removed_fact_labels_ignored_with_warning(tmp_path):
    path = tmp_path / 'decision.yaml'
    spec = {'name': 'decision', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
            'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}, 'legal': {'check': 'The customer threatens legal action.'}}}
    files.publish(path, spec)
    files.append(path.with_suffix('.test.jsonl'), files.jsonl(envelopes([{'input': {'message': f'refund {i}'}, 'facts': {'refund': True, 'legal': False},
                                                    'group': f'g{i}'} for i in range(6)])))
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(path, draft=True)
        sw.correct({'message': 'refund a'}, fact='legal', value=False, actor=Actor('human'))
        del spec['facts']['legal']
        files.publish(path, spec)
        locked = tools.lock(rt, path, dry_run=True)
        exported = tools.export(rt, path, split='all')
        report = tools.run_test(rt, path)
    assert 'legal: 6 labels for a removed fact ignored' in locked['warnings']
    assert 'legal: 1 labels for a removed fact ignored' in locked['warnings']
    assert exported['rows'] and 'legal: 6 labels for a removed fact ignored' in report['warnings']


def test_relock_keeps_fresh_interval_for_unconsulted_families(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        first = tools.run_test(rt, path, cohort(tmp_path / 'first.jsonl', 'first', 30))
        assert first['facts']['refund']['interval']['interval'] is not None
        spec = files.read_yaml(path)
        spec['otherwise'] = 'changed action'
        files.publish(path, spec)
        tools.lock(rt, path)
        fresh = tools.run_test(rt, path, cohort(tmp_path / 'second.jsonl', 'second', 30))
        assert fresh['audit_use']['state'] == 'fresh' and fresh['audit_use']['independent']
        assert fresh['facts']['refund']['interval']['interval'] is not None
        consulted = tools.run_test(rt, path, tmp_path / 'first.jsonl')
    assert consulted['audit_use']['state'] == 'consulted'
    assert consulted['facts']['refund']['interval']['interval'] is None
    assert consulted['facts']['refund']['interval']['reason'] == 'check families consulted: reused evidence, not an independent adoption claim'


def test_lock_dry_run_records_exposure_without_lock(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        result = tools.lock(rt, path, dry_run=True)
    audit = path.with_suffix('.audit.jsonl')
    assert result['lock'] is None and not path.with_suffix('.lock.json').exists()
    assert result['written'] == [str(path.with_suffix('.dataset.json')), str(audit)]
    assert [len(r['families']) > 0 for r in files.read_jsonl(audit)] == [True]
    from inspect import signature
    from semantic_switch import cli
    assert signature(cli.lock).parameters['dry_run'].default.help == 'evaluate without publishing the lock; dataset assignments and audit exposure are recorded'


def test_test_data_does_not_rewrite_dataset_manifest(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        manifest = path.with_suffix('.dataset.json')
        before = manifest.read_bytes()
        report = tools.run_test(rt, path, cohort(tmp_path / 'cohort.jsonl', 'cohort', 10))
        assert manifest.read_bytes() == before
        assert rt.load(path).snapshot.dataset_id == files.read_json(manifest)['id'] != report['dataset_id']


def test_dataset_id_location_independent(tmp_path):
    ids = []
    for name in ('one', 'two'):
        (tmp_path / name).mkdir()
        path = definition(tmp_path / name)
        dataset(path, 20)
        with Runtime(FakeScorer()) as rt:
            ids.append(Labels(rt.load(path, draft=True)).dataset_id)
        manifest = files.read_json(path.with_suffix('.dataset.json'))
        assert manifest['schema'] == 7 and set(manifest['sources']) == {'definition', 'examples', 'labels'}
    assert ids[0] == ids[1]
    with Runtime(FakeScorer()) as rt:
        exported = tools.export(rt, path)
    assert all('definition' not in row and str(tmp_path) not in json.dumps(row) for row in read_export(exported['path'])[1])


def test_label_rejects_unknown_destination(tmp_path):
    path = definition(tmp_path)
    inputs = tmp_path / 'inputs.jsonl'
    files.append(inputs, files.jsonl([{'message': 'refund please'}]))
    with Runtime(FakeScorer()) as rt:
        with pytest.raises(ConfigError, match='destination'):
            tools.label(rt, path, str(inputs), 'tset', ask=lambda _: 'y', show=lambda _: None, actor=Actor('human'))
    result = subprocess.run([sys.executable, '-m', 'semantic_switch', 'label', str(path), str(inputs), '--to', 'tset'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not path.with_suffix('.examples.jsonl').exists() and not path.with_suffix('.test.jsonl').exists()


def test_each_and_single_exit_codes_match(monkeypatch):
    import typer
    from semantic_switch import cli
    monkeypatch.setattr(cli, 'emit', lambda obj, raw=False: None)
    for statuses in (['ok'], ['ok', 'uncertain'], ['ok', 'failed'], ['needs_input', 'failed']):
        monkeypatch.setattr(cli, 'states', lambda: iter(statuses))
        with pytest.raises(typer.Exit) as stopped:
            cli.run_stream(lambda chunk: [type('R', (), {'status': s})() for s in chunk], 2)
        assert stopped.value.exit_code == cli.worst(statuses) == max(cli.EXIT[s] for s in statuses)
    assert 'failed 3' in cli.app.info.help
