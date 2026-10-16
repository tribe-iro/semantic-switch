import pytest


@pytest.fixture(autouse=True)
def isolated_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_MODEL_LOCK', str(tmp_path / 'model-job.lock'))
    monkeypatch.setenv('SEMANTIC_SWITCH_CACHE', str(tmp_path / 'cache'))
