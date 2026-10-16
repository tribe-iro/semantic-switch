from __future__ import annotations

import errno
import hashlib
import os
import stat
import warnings

import pytest

from semantic_switch import ConfigError, files
from .test_adaptation_contracts import receipt_row, specialist


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(tmp_path, content=b'weights' * 1000):
    path = tmp_path / 'artifact.bin'
    path.write_bytes(content)
    return path


def entries(root):
    return [path for path in root.rglob('*.json')] if root.exists() else []


@pytest.fixture
def read_only_cache(tmp_path, monkeypatch):
    root = tmp_path / 'readonly-cache'
    root.mkdir()
    root.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(root / 'nested'))
    yield root
    root.chmod(stat.S_IRWXU)


def unavailable(error):
    def fail(*args, **kwargs):
        raise error
    return fail


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'denied'), PermissionError('denied'), OSError(errno.EROFS, 'read-only file system')])
def test_digest_survives_cache_lookup_and_publication_failures(tmp_path, monkeypatch, error):
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    path = artifact(tmp_path)
    monkeypatch.setattr(files, 'read_json', unavailable(error))
    monkeypatch.setattr(files, 'publish', unavailable(error))
    with pytest.warns(UserWarning, match='continuing without the cache'):
        assert files.sha256_file(path) == digest(path)
    assert not entries(tmp_path / 'cache')


def test_unwritable_cache_root_still_hashes_and_creates_nothing(tmp_path, read_only_cache):
    path = artifact(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert files.sha256_file(path) == digest(path)
        path.write_bytes(b'changed')
        assert files.sha256_file(path) == digest(path)
    assert list(read_only_cache.iterdir()) == []


def test_other_cache_errors_and_artifact_errors_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    path = artifact(tmp_path)
    monkeypatch.setattr(files, 'publish', unavailable(OSError(errno.EIO, 'device error')))
    with pytest.raises(OSError, match='device error'):
        files.sha256_file(path)
    monkeypatch.undo()
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    with pytest.raises(FileNotFoundError):
        files.sha256_file(tmp_path / 'missing.bin')
    if os.geteuid():
        path.chmod(0)
        with pytest.raises(PermissionError):
            files.sha256_file(path)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_writable_cache_reuses_and_invalidates_on_change(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    path = artifact(tmp_path)
    assert files.sha256_file(path) == digest(path) and len(entries(tmp_path / 'cache')) == 1
    reads = []
    real_open = open
    monkeypatch.setattr('builtins.open', lambda *a, **k: reads.append(a[0]) or real_open(*a, **k))
    assert files.sha256_file(path) == digest(path) and str(path) not in map(str, reads)
    monkeypatch.undo()
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
    path.write_bytes(b'different content')
    assert files.sha256_file(path) == digest(path) and len(entries(tmp_path / 'cache')) == 2


def test_receipt_validation_with_read_only_cache_storage_still_rejects_changes(tmp_path, monkeypatch):
    from semantic_switch.training import fitted
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'build-cache'))
    gguf, path = specialist(tmp_path, [receipt_row()])
    root = tmp_path / 'readonly-cache'
    root.mkdir()
    root.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(root))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = fitted(path)
            assert model.provenance.gguf_sha256 == digest(gguf)
            gguf.write_bytes(b'changed')
            with pytest.raises(ConfigError, match='changed since selection'):
                fitted(path)
        assert list(root.iterdir()) == []
    finally:
        root.chmod(stat.S_IRWXU)
