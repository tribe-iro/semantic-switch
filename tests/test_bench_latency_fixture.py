import inspect
import json

import semantic_switch as ss
from semantic_switch.judge import Scored, Scores

from bench import latency, outcomes


class Stub:
    identity = {"source": "stub"}
    accelerated = False
    max_slots = 20
    batch_inputs = 1
    supports_guidance = True
    normalization = "option-softmax"

    def __init__(self):
        self.questions = []

    def score_many(self, items):
        self.questions.extend(q for _, qs in items for q in qs)
        return [Scored(tuple(Scores((0.9,) + (0.1 / (q.width - 1),) * (q.width - 1), 1.0) for q in qs), 10) for _, qs in items]


def test_product_bench_imports_no_runtime_fixtures_or_metrics_internals():
    source = inspect.getsource(latency)
    assert "CONFORMANCE" not in source and "semantic_switch.judge" not in source and "semantic_switch.metrics" not in source
    assert latency.Progress is outcomes.Progress is ss.training.Progress


def test_option_edit_changes_workload_identity_and_order_is_preserved(tmp_path):
    fixture = json.loads(latency.FIXTURE.read_text())
    edited = json.loads(json.dumps(fixture))
    next(item for item in edited["items"] if item["kind"] == "pick")["options"][0] = "billing questions"
    path = tmp_path / "latency.json"
    path.write_text(json.dumps(edited))
    original, changed = latency.workload(), latency.workload(path)
    assert original["inputs"] == changed["inputs"] and original["digest"] != changed["digest"]
    assert list(original["items"]) == [item["name"] for item in fixture["items"]]
    assert original["items"]["topic"].options == tuple(next(i for i in fixture["items"] if i["name"] == "topic")["options"])


def test_latency_run_loads_the_fixture_once_and_records_its_digest(monkeypatch):
    loads = []
    real = latency.workload
    monkeypatch.setattr(latency, "workload", lambda path=latency.FIXTURE: loads.append(path) or real(path))
    judge = Stub()
    with ss.Runtime(judge) as rt:
        report = latency.latency(rt, n=2)
    assert loads == [latency.FIXTURE] and report["workload"]["digest"] == real()["digest"]
    asked = [q.text for q in judge.questions]
    names = {item["question"] for item in json.loads(latency.FIXTURE.read_text())["items"]}
    assert set(asked) == names and report["new_input_4_questions_ms"]["failed"] == 0
