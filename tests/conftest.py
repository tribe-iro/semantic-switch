from __future__ import annotations

from importlib import metadata
import platform
import re
import sys

import numpy as np
import pytest

from semantic_switch import Runtime
from semantic_switch.protocols import ScoreBlock

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
        self.selected = [{"id": item.nodeid, "kind": "native_or_hosted_experiment" if item.get_closest_marker("slow") else "fake_behavior"}
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
        from bench.diagnostics import fingerprint
        from semantic_switch import files

        def version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None
        counts = {state: sum(value == state for value in self.outcomes.values()) for state in ("passed", "failed", "skipped")}
        files.publish(self.path, {"schema": 1, "command": sys.argv, "code": fingerprint(), "exit_status": int(exitstatus),
                                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                                  "dependencies": {name: version(name) for name in DEPENDENCIES}},
                                  "selected": self.selected, **counts, "failures": self.failures,
                                  "scope": "behavior checks of this code identity; counts from another code identity are not this baseline"})


def pytest_configure(config):
    if config.getoption("--baseline"):
        config.pluginmanager.register(Baseline(config.getoption("--baseline")), "verification-baseline")


TOKEN = re.compile(r"<\|[^|]+\|>|\w+|\s|[^\w\s]")
PLAIN = re.compile(r"\w+|\s|[^\w\s]")


def letters_of(question: str) -> dict:
    return {m.group(2).strip(): m.group(1) for m in re.finditer(r"^([A-T])\) (.+)$", question, re.M)}


def default_oracle(state: str, question: str, width: int):
    s = state.lower()
    if "boom" in s and "legal action" not in question:
        raise RuntimeError("backend exploded")
    if "maybe" in s and "money back" in question:
        return [0.5, 0.5]
    if "garbled" in s and "money back" in question:
        return "lowmass"
    if "legal action" in question:
        return [0.97, 0.03] if ("lawyer will" in s or "lawsuit" in s) else [0.03, 0.97]
    if "money back" in question:
        return [0.96, 0.04] if "refund" in s and "do not refund" not in s else [0.04, 0.96]
    if "disputes the recorded payment" in question:
        return [0.95, 0.05] if "i did pay" in s else [0.05, 0.95]
    if "mainly about" in question:
        opts = letters_of(question)
        want = ("the product is broken" if "broken" in s or "error" in s else
                "a physical device" if "device" in s else
                "charges" if "charge" in s or "invoice" in s else "login" if "login" in s else "none of the above")
        p = np.full(width, 0.02 / max(width - 1, 1))
        hits = [letter for desc, letter in opts.items() if desc.startswith(want)] or \
               [letter for desc, letter in opts.items() if desc.startswith("none of the above")]
        for letter in hits:
            p[ord(letter) - 65] = 0.98
        return list(p / p.sum())
    if "urgent" in question:
        return [0.02, 0.03, 0.95] if "now" in s else [0.05, 0.9, 0.05] if "today" in s else [0.9, 0.05, 0.05]
    return [1.0 / width] * width


class FakeScorer:
    adapter = "qwen3"

    def __init__(self, oracle=default_oracle, name="fake-judge", accelerated=False):
        self.oracle = oracle
        self.name = name
        self.accelerated = accelerated
        self.vocab: dict = {}
        self.inv: list = []
        self.requests: list = []

    def tokenize(self, text: str, special: bool = False) -> tuple[int, ...]:
        out = []
        for piece in (TOKEN if special else PLAIN).findall(text):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.inv)
                self.inv.append(piece)
            out.append(self.vocab[piece])
        return tuple(out)

    def tokenizer_identity(self):
        from semantic_switch.compiler import digest
        return {'vocab_size': 1000000, 'vocab_sha256': digest('fake-tokenizer'),
                'probes': [], 'special': {}, 'padding_tokens': []}

    def detokenize(self, tokens) -> str:
        return "".join(self.inv[t] for t in tokens)

    def score(self, requests):
        width = max(len(s) for r in requests for s in r.slots)
        block = ScoreBlock.empty(sum(len(r.suffixes) for r in requests), width)
        row = 0
        for r in requests:
            state = self.detokenize(r.state)
            for suffix, slots in zip(r.suffixes, r.slots):
                q = self.detokenize(suffix)
                self.requests.append((state, q))
                try:
                    p = self.oracle(state, q, len(slots))
                    if p == "lowmass":
                        block.logp[row, :2] = np.log([0.6, 0.4])
                        block.log_mass[row] = np.log(0.2)
                    else:
                        block.logp[row, :len(p)] = np.log(np.clip(p, 1e-9, 1))
                        block.log_mass[row] = 0.0
                except RuntimeError as e:
                    block.error[row] = str(e)
                row += 1
        return block

    def asked(self, needle: str) -> bool:
        return any(needle in q for _, q in self.requests)


def transport(reply):
    async def send(url, body, remaining):
        return reply(url, body, remaining)
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
    return Runtime({"judge": fake})


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


def envelopes(records, origin="human", producer="fixture"):
    out = []
    for record in records:
        record = dict(record)
        facts = record.pop("facts", {})
        source = [{"origin": record.pop("label_origin", origin), "producer": producer}]
        groups = [*record.pop("groups", []), *([record.pop("group")] if "group" in record else [])]
        pairs = [*record.pop("pairs", []), *([record.pop("pair")] if "pair" in record else [])]
        selections = [*record.pop("selections", []), *([record.pop("selection")] if "selection" in record else [])]
        out.append({"schema": 2, **record, "facts": facts, "groups": groups, "pairs": pairs, "selections": selections,
                    "label_sources": {name: {key: source for key in value} if isinstance(value, dict) else source for name, value in facts.items()}})
    return out
