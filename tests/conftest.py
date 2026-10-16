from __future__ import annotations

from importlib import metadata
import json
import platform
import sys

import numpy as np
import pytest

from semantic_switch import Runtime
from semantic_testkit.scoring import FakeScorer

DEPENDENCIES = ("numpy", "pyyaml", "pydantic", "typer", "jinja2", "httpx", "llama-cpp-python", "huggingface_hub", "pytest")


def pytest_addoption(parser):
    parser.addoption("--baseline", help="write this run's verification baseline JSON to PATH")


def classify(report) -> str:
    if report.when != "call":
        return f"{report.when}_error"
    crash = getattr(report.longrepr, "reprcrash", None)
    message = getattr(crash, "message", "") or ""
    return "assertion" if message.startswith("AssertionError") or message.startswith("assert ") else f"error:{message.split(':', 1)[0] or 'unknown'}"


class Baseline:
    def __init__(self, path):
        self.path, self.selected, self.outcomes, self.failures = path, [], {}, {}

    def pytest_collection_finish(self, session):
        self.selected = [{"id": item.nodeid, "slow": bool(item.get_closest_marker("slow"))}
                         for item in session.items]

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.outcomes[report.nodeid] = "failed"
            self.failures[report.nodeid] = classify(report)
        elif report.skipped:
            self.outcomes.setdefault(report.nodeid, "skipped")
        elif report.when == "call":
            self.outcomes.setdefault(report.nodeid, "passed")

    def pytest_sessionfinish(self, session, exitstatus):
        from bench.outcomes import fingerprint
        from semantic_switch import files

        def version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None
        counts = {state: sum(value == state for value in self.outcomes.values()) for state in ("passed", "failed", "skipped")}
        files.publish(self.path, {"schema": 2, "command": sys.argv, "code": fingerprint(), "exit_status": int(exitstatus),
                                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                  "dependencies": {name: version(name) for name in DEPENDENCIES}},
                                  "selected": self.selected, **counts, "failures": self.failures,
                                  "scope": "behavior checks of this code identity; counts from another code identity are not this baseline"})


def pytest_configure(config):
    if config.getoption("--baseline"):
        config.pluginmanager.register(Baseline(config.getoption("--baseline")), "verification-baseline")


def transport(reply):
    async def send(url, body, remaining):
        return reply(url, json.loads(body), remaining)
    return send


@pytest.fixture(scope="session", autouse=True)
def model_lock(request, tmp_path_factory):
    if any(item.get_closest_marker("slow") for item in request.session.items):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SEMANTIC_MODEL_LOCK", str(tmp_path_factory.mktemp("model-job") / "model-job.lock"))
        yield


@pytest.fixture
def fake():
    return FakeScorer()


@pytest.fixture
def rt(fake):
    return Runtime(fake)


@pytest.fixture
def head_bundle(tmp_path):
    def create(rt, path, fact_name='refund_requested'):
        import zlib
        from semantic_switch.artifacts import publish_bundle
        from semantic_switch.compiler import digest
        from semantic_switch.heads import LexicalHead, FEATURE_IDENTITY
        from semantic_switch.tools.data import Labels
        labels = Labels(rt.load(path, draft=True))
        fact = labels.sw.plan.fact(fact_name)
        rows = labels.examples(fact_name, 'tune')
        head = LexicalHead(fact.keys, np.array([zlib.crc32(b'refund') % (1 << 18)], dtype=np.int32),
                           np.array([[30., -30.]], dtype=np.float32), np.array([0., 0.], dtype=np.float32))
        directory = tmp_path / f'{fact_name}-bundle'
        entry = {'head': 'head.json'}
        fit_rows = [{'id': e.id, 'fact': e.fact, 'group': e.group, 'fold': e.fold, 'projection': e.projection,
                     'family_seeds': labels.seeds(e.group)} for e in rows]
        folds = {str(k): {**entry, 'fit_ids': sorted(e.id for e in rows if e.fold != k),
                         'held_ids': sorted(e.id for e in rows if e.fold == k)} for k in range(5)}
        manifest = {'schema': 4, 'kind': 'lexical-bundle', 'feature_identity': FEATURE_IDENTITY,
                    'classes': list(fact.keys), 'meaning_rev': fact.meaning_rev, 'fact': fact_name,
                    'dataset_id': labels.dataset_id, 'training_id': labels.training_id,
                    'export_id': digest(fit_rows), 'trainer': {'name': 'deterministic-fixture', 'version': '1'},
                    'rows': fit_rows, 'folds': folds, 'final': entry, 'unobserved_classes': []}
        publish_bundle(directory, manifest, {'head.json': head})
        return directory
    return create
