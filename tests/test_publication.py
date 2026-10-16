from __future__ import annotations

import os

import pytest

from semantic_switch import ConfigError, api, files
from semantic_switch.training import model_job


def publication(tmp_path, rows):
    return files.publish(tmp_path / 'data.manifest.json', {'schema': 1}, {'data.jsonl': [f'{row}\n' for row in rows]})


def contents(commit_path):
    with files.open_published(commit_path) as (manifest, handles):
        return manifest, handles['data.jsonl'].read().decode().split()


def stored(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir() if p.name.startswith('data.') and p.suffix == '.jsonl')


def test_interrupted_republish_keeps_previous_generation_readable(tmp_path, monkeypatch):
    commit = tmp_path / 'data.manifest.json'
    previous = publication(tmp_path, ['a', 'b'])
    replace = os.replace
    def interrupted(source, target):
        if os.fspath(target) == os.fspath(commit):
            raise OSError('interrupted')
        replace(source, target)
    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        publication(tmp_path, ['c', 'd', 'e'])
    monkeypatch.undo()
    assert contents(commit) == (previous, ['a', 'b'])
    assert len(stored(tmp_path)) == 2


def test_republish_retains_generations_until_an_explicit_prune(tmp_path):
    commit = tmp_path / 'data.manifest.json'
    published = [publication(tmp_path, rows) for rows in (['a'], ['b'], ['c', 'd'])]
    (tmp_path / 'data.0123456789abcdef.jsonl').write_text('orphan\n')
    (tmp_path / 'data.other.jsonl').write_text('kept\n')
    manifest = publication(tmp_path, ['e'])
    generations = [entry['payloads']['data.jsonl']['file'] for entry in (*published, manifest)]
    assert stored(tmp_path) == sorted([*generations, 'data.0123456789abcdef.jsonl', 'data.other.jsonl'])
    assert sorted(files.prune(tmp_path)) == sorted([*generations[:3], 'data.0123456789abcdef.jsonl'])
    assert stored(tmp_path) == sorted([generations[3], 'data.other.jsonl']) and contents(commit) == (manifest, ['e'])


def test_readers_of_an_older_manifest_keep_their_generation(tmp_path):
    commit = tmp_path / 'data.manifest.json'
    publication(tmp_path, ['a', 'b'])
    with files.open_published(commit) as (manifest, handles):
        current = publication(tmp_path, ['c'])
        assert manifest['payloads']['data.jsonl']['file'] in stored(tmp_path)
        assert handles['data.jsonl'].read() == b'a\nb\n'
    assert contents(commit) == (current, ['c'])
    old = tmp_path / manifest['payloads']['data.jsonl']['file']
    assert old.read_text() == 'a\nb\n'


def test_short_writes_complete(tmp_path, monkeypatch):
    write = os.write
    monkeypatch.setattr(os, 'write', lambda fd, data: write(fd, bytes(data[:3])))
    rows = [f'row-{i}' for i in range(50)]
    manifest = publication(tmp_path, rows)
    files.append(tmp_path / 'log.jsonl', files.jsonl([{'n': i, 'text': 'x' * 40} for i in range(20)]))
    files.append(tmp_path / 'log.jsonl', files.jsonl([{'n': 20}]))
    monkeypatch.undo()
    assert manifest['payloads']['data.jsonl']['count'] == 50
    assert contents(tmp_path / 'data.manifest.json') == (manifest, rows)
    assert [r['n'] for r in files.read_jsonl(tmp_path / 'log.jsonl')] == list(range(21))


def test_model_lock_without_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.delenv('SEMANTIC_MODEL_LOCK', raising=False)
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    monkeypatch.setattr(api, 'available_host_bytes', lambda: 1 << 50)
    directory = tmp_path / f'semantic-switch-{os.getuid()}'
    assert api._model_lock() == directory / 'model-job.lock'
    assert directory.stat().st_mode & 0o777 == 0o700
    with model_job() as job:
        assert job.available_bytes == 1 << 50


def meminfo(tmp_path, monkeypatch):
    path = tmp_path / 'meminfo'
    path.write_text('MemTotal:       33554432 kB\nMemAvailable:    8388608 kB\n')
    monkeypatch.setattr(api, 'MEMINFO', path)
    monkeypatch.setattr(api, 'CGROUP_ROOT', tmp_path / 'cgroup')


def test_admission_without_cgroup_v2(tmp_path, monkeypatch):
    meminfo(tmp_path, monkeypatch)
    cgroup = tmp_path / 'self-cgroup'
    cgroup.write_text('12:memory:/user.slice\n1:name=systemd:/user.slice\n')
    monkeypatch.setattr(api, 'SELF_CGROUP', cgroup)
    assert api.available_host_bytes() == 8 << 30
    monkeypatch.setattr(api, 'SELF_CGROUP', tmp_path / 'absent')
    assert api.available_host_bytes() == 8 << 30


def test_admission_unsupported_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'MEMINFO', tmp_path / 'absent')
    with pytest.raises(ConfigError, match='memory admission is unsupported on this platform'):
        api.available_host_bytes()
