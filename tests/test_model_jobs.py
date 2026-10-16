from __future__ import annotations

from semantic_switch import model_jobs

import ast
import inspect
import os
import subprocess
import sys
import threading
import time

import pytest

from semantic_switch import ConfigError, Model, Runtime, api, backend
from semantic_switch.training import model_job
from semantic_testkit.scoring import FakeScorer

GiB = 1 << 30


@pytest.fixture
def lock(tmp_path, monkeypatch):
    path = tmp_path / "model-job.lock"
    monkeypatch.setenv("SEMANTIC_MODEL_LOCK", str(path))
    return path


def test_model_job_lock_exclusive_across_processes(lock, monkeypatch):
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: 1 << 50)
    code = ("import sys, time; from semantic_switch import model_jobs; model_jobs.available_host_bytes = lambda: 1 << 50\n"
            "with model_jobs.model_job():\n    print('held', flush=True)\n    time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(ConfigError, match=f'"pid": {child.pid}'):
            with model_job():
                pass
    finally:
        child.kill()
        child.wait()
    with model_job() as job:
        assert job.diagnostics()["admitted_bytes"] == model_jobs.HOST_RESERVE


def test_explicit_job_admission_refuses_but_default_runtime_never_admits(lock, tmp_path, monkeypatch):
    from .test_native_resources import LoadedScorer
    weights = tmp_path / "base.gguf"
    weights.write_bytes(b"gguf")
    admitted = []
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(backend, "LlamaScorer", LoadedScorer)
    monkeypatch.setattr(Model, "resources", lambda self: {"weights": api.Resources(1 << 50)})
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: admitted.append(1) or 8 * GiB)
    model = Model(str(weights), adapter="qwen3", device="cpu")
    with pytest.raises(ConfigError, match=r"requires \d+ bytes.*8589934592 bytes are available.*systemd-run --user --scope -q -p MemoryMax=\d+G -p MemorySwapMax=0"):
        with model_job(model):
            pass
    admitted.clear()
    with Runtime(model) as runtime:
        runtime.judge()
        assert runtime.job is None and not admitted


def test_default_runtimes_in_separate_processes_load_concurrently(lock, tmp_path, monkeypatch):
    from .test_native_resources import LoadedScorer
    weights = tmp_path / "base.gguf"
    weights.write_bytes(b"gguf")
    monkeypatch.setenv("SEMANTIC_SWITCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(backend, "LlamaScorer", LoadedScorer)
    code = ("import time; from semantic_switch import Model, Runtime, backend; from tests.test_native_resources import LoadedScorer\n"
            "backend.LlamaScorer = LoadedScorer\n"
            f"with Runtime(Model({str(weights)!r}, adapter='qwen3', device='cpu')) as rt:\n"
            "    rt.judge()\n    print('held', flush=True)\n    time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env={**os.environ, "SEMANTIC_SWITCH_CACHE": str(tmp_path / "child")})
    try:
        assert child.stdout.readline().strip() == "held"
        with Runtime(Model(str(weights), adapter="qwen3", device="cpu")) as first, Runtime(Model(str(weights), adapter="qwen3", device="cpu")) as second:
            first.judge()
            second.judge()
            assert first.job is None and second.job is None
    finally:
        child.kill()
        child.wait()


def test_admission_uses_lower_cgroup_limit(lock, tmp_path, monkeypatch):
    meminfo, root = tmp_path / "meminfo", tmp_path / "cgroup"
    meminfo.write_text("MemTotal:       33554432 kB\nMemAvailable:    8388608 kB\n")
    leaf = root / "user.slice" / "job.scope"
    leaf.mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text("max\n")
    (root / "user.slice" / "memory.current").write_text(f"{5 * GiB}\n")
    (leaf / "memory.max").write_text(f"{4 * GiB}\n")
    (leaf / "memory.current").write_text(f"{2 * GiB}\n")
    (leaf / "memory.stat").write_text(f"anon {GiB}\ninactive_file {GiB}\n")
    cgroup = tmp_path / "self-cgroup"
    cgroup.write_text("0::/user.slice/job.scope\n")
    monkeypatch.setattr(model_jobs, "MEMINFO", meminfo)
    monkeypatch.setattr(model_jobs, "CGROUP_ROOT", root)
    monkeypatch.setattr(model_jobs, "SELF_CGROUP", cgroup)
    assert model_jobs.available_host_bytes() == 3 * GiB
    with model_job(extra={"working_set": GiB // 2}) as job:
        assert (job.diagnostics()["admitted_bytes"], job.available_bytes) == (2 * GiB + GiB // 2, 3 * GiB)
    with pytest.raises(ConfigError, match=f"requires {4 * GiB} bytes.*{3 * GiB} bytes are available"):
        with model_job(extra={"working_set": 2 * GiB}):
            pass
    (leaf / "memory.max").write_text("max\n")
    assert model_jobs.available_host_bytes() == 8 * GiB


def test_admission_counts_reclaimable_gpu_pool(lock, tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       33554432 kB\nMemAvailable:    7340032 kB\nGPUActive:        655360 kB\nGPUReclaim:      9437184 kB\n")
    monkeypatch.setattr(model_jobs, "MEMINFO", meminfo)
    monkeypatch.setattr(model_jobs, "SELF_CGROUP", tmp_path / "absent")
    assert model_jobs.available_host_bytes() == 16 * GiB
    with model_job(extra={"working_set": 12 * GiB}) as job:
        assert job.available_bytes == 16 * GiB


def test_slow_fixtures_single_resident_model(monkeypatch):
    from tests import test_backend
    events = []
    class Recorded:
        def __init__(self, models=None):
            self.name = "default" if models is None else "qwen"
            events.append(("open", self.name))
        def close(self):
            events.append(("close", self.name))
    monkeypatch.setattr(test_backend, "Runtime", Recorded)
    real = test_backend.real.__wrapped__()
    gemma = test_backend.gemma.__wrapped__(next(real))
    next(gemma)
    assert events == [("open", "qwen"), ("close", "qwen"), ("open", "default")]
    order = [inspect.signature(function).parameters for name, function in vars(test_backend).items()
             if name.startswith("test_") and callable(function)]
    first = next(index for index, parameters in enumerate(order) if "gemma" in parameters)
    assert all("real" not in parameters for parameters in order[first:])
    tree = ast.parse(inspect.getsource(test_backend))
    assert not any(isinstance(node, ast.FunctionDef) and {"real", "gemma"} <= {arg.arg for arg in node.args.args} for node in tree.body)


@pytest.fixture
def roomy(lock, monkeypatch):
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: 1 << 50)
    monkeypatch.setattr(Model, "resources", lambda self: {"weights": api.Resources(GiB)})


def test_independent_threads_contend_and_path_never_implies_participation(roomy):
    held, done = threading.Event(), threading.Event()
    def owner():
        with model_job():
            held.set()
            done.wait(5)
    thread = threading.Thread(target=owner)
    thread.start()
    held.wait(5)
    try:
        with pytest.raises(ConfigError, match="locked by"):
            with model_job():
                pass
    finally:
        done.set()
        thread.join()
    with model_job() as again:
        assert again.state == "open"


def test_explicit_child_joins_once_and_release_crosses_threads(roomy):
    manager = model_job(extra={"fit": 4 * GiB})
    job = manager.__enter__()
    with model_job(extra={"probe": GiB}, job=job) as joined:
        assert joined is job and job.diagnostics()["participants"] == 1
    assert job.diagnostics()["participants"] == 0 and job.diagnostics()["resident_peak_bytes"]["host"] == GiB
    thread = threading.Thread(target=manager.__exit__, args=(None, None, None))
    thread.start()
    thread.join()
    assert job.state == "closed"
    job.close()
    with pytest.raises(ConfigError, match="is closed"):
        with model_job(job=job):
            pass
    with model_job() as again:
        assert again.owner != job.owner


def test_live_child_prevents_owner_release(roomy):
    job_manager = model_job(extra={"envelope": 2 * GiB})
    job = job_manager.__enter__()
    joined, leave, closed = threading.Event(), threading.Event(), threading.Event()
    def child():
        with job.join(GiB, use="child"):
            joined.set()
            leave.wait(5)
    worker = threading.Thread(target=child)
    worker.start()
    joined.wait(5)
    closer = threading.Thread(target=lambda: (job_manager.__exit__(None, None, None), closed.set()))
    closer.start()
    time.sleep(.05)
    assert not closed.is_set() and job.state == "closing"
    with pytest.raises(ConfigError, match="is closing"):
        with job.join(1):
            pass
    with pytest.raises(ConfigError, match="locked by"):
        with model_job():
            pass
    leave.set()
    worker.join()
    closer.join()
    assert closed.is_set() and job.state == "closed"


def test_forked_handle_and_over_budget_child_reject_before_construction(roomy, tmp_path, monkeypatch):
    weights = tmp_path / "base.gguf"
    weights.write_bytes(b"gguf")
    constructed = []
    monkeypatch.setattr(backend, "LlamaScorer", lambda *args, **kwargs: constructed.append(args))
    with model_job(extra={"envelope": GiB // 2}) as job:
        with pytest.raises(ConfigError, match="declared envelope"):
            with Runtime(Model(str(weights), device="cpu"), job=job) as runtime:
                runtime.judge()
        real = os.getpid()
        monkeypatch.setattr(os, "getpid", lambda: real + 1)
        with pytest.raises(ConfigError, match=f"belongs to process {real}"):
            with job.join(1):
                pass
        monkeypatch.undo()
        assert job.diagnostics()["participants"] == 0
    assert constructed == []


def test_load_failure_releases_owned_job_and_hosted_takes_no_lease(roomy, tmp_path, monkeypatch):
    weights = tmp_path / "base.gguf"
    weights.write_bytes(b"gguf")
    def failing(*args, **kwargs):
        raise RuntimeError("load failed")
    monkeypatch.setattr(backend, "LlamaScorer", failing)
    runtime = Runtime(Model(str(weights), device="cpu"))
    with pytest.raises(ConfigError, match="load failed"):
        runtime.judge()
    assert runtime.job is None
    with model_job():
        pass
    runtime.close()
    with Runtime(FakeScorer()) as hosted:
        hosted.judge()
        assert hosted.job is None
        with model_job():
            pass


def test_cli_sessions_admit_lazily(lock, monkeypatch):
    from semantic_switch import cli
    admitted = []
    monkeypatch.setattr(model_jobs, "available_host_bytes", lambda: admitted.append(1) or 1 << 50)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with cli.session({"judge": "openrouter:m", "provider": "p", "settings": {"codec": "chat"}}) as rt:
        assert rt.job is None and not admitted
    proc = subprocess.run([sys.executable, "-m", "semantic_switch", "check", "x", "--each"], input="not json\n", capture_output=True, text=True,
                          env={**os.environ, "SEMANTIC_MODEL_LOCK": str(lock)})
    assert proc.returncode == 5 and "stdin line 1: invalid JSON" in proc.stderr
