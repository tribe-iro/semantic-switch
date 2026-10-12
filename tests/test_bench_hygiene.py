from __future__ import annotations

from contextlib import contextmanager
import json
import zlib

from semantic_switch import Runtime, files
from .conftest import FakeScorer


class Invalidating(FakeScorer):
    def invalidate(self):
        pass


def peaked(state, question, width):
    top = zlib.crc32(state.encode()) % width
    return [0.97 if k == top else 0.03 / (width - 1) for k in range(width)]


def failing(state, question, width):
    raise RuntimeError("backend down")


def test_all_failure_latency_has_no_non_failed_summary():
    from bench.latency import latency
    with Runtime({"judge": FakeScorer(failing)}) as rt:
        report = latency(rt, n=3)
    for key in ("same_input_ms", "new_input_ms", "new_input_4_questions_ms"):
        stats = report[key]
        assert stats["non_failed"] is None and stats["all"]["count"] == 3
        assert all("failed" in sample["statuses"] and sample["ms"] > 0 for sample in stats["samples"])
        assert stats["statuses"]["failed"] == sum(len(sample["statuses"]) for sample in stats["samples"])


def test_adaptation_rows_come_from_frozen_execution(tmp_path, monkeypatch):
    from bench import policy_adaptation
    @contextmanager
    def fake_runtime(model):
        with Runtime({"judge": Invalidating(peaked)}) as bound:
            yield bound
    monkeypatch.setattr(policy_adaptation, "runtime", fake_runtime)
    monkeypatch.setattr(policy_adaptation.tools, "export", lambda *a, **k: {"path": "unused"})
    monkeypatch.setattr("semantic_switch.switch.Switch.score_facts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("separate scoring")))
    with Runtime({"judge": FakeScorer()}) as bound:
        definitions = policy_adaptation.materialize(bound, tmp_path / "defs")["definitions"]
    report = policy_adaptation.compare(definitions, {"base": None}, tmp_path / "report.json")
    rows = report["arms"][0]["rows"]
    assert rows and report == json.loads(json.dumps(files.read_json(tmp_path / "report.json")))
    for row in rows:
        scores = row["result"]["scores"]
        top = max(scores, key=scores.get)
        assert row["raw_mass"] == row["result"]["mass"]
        assert row["raw_value"] == top and row["raw_usable"]
        assert row["result"]["status"] != "ok" or row["result"]["value"] == row["raw_value"]
    assert {row["result"]["status"] for row in rows} >= {"ok"} and len({row["raw_value"] for row in rows}) > 1
    for key, value in policy_adaptation.summarize(rows).items():
        assert report["arms"][0][key] == value


def test_raised_audit_window_becomes_failed_rows_with_elapsed_time():
    from types import SimpleNamespace
    from bench import policy_adaptation
    from bench.model_comparison import timings

    def run(datas):
        raise RuntimeError("backend down")
    installed = SimpleNamespace(snapshot=SimpleNamespace(candidate_id="c"), run=run)
    with Runtime({"judge": FakeScorer()}) as rt:
        results, window = policy_adaptation.audit_window(rt, installed, [SimpleNamespace(id="a", data={})], "w", "cold")
    assert results is None and window["outcome"] == "failed" and window["error"] == {"type": "RuntimeError", "stage": "execute"}
    assert window["ms"] == window["total_ms"] > 0 and window["members"] == ["a"]
    timed = timings(lambda n: 1 / (n % 2), 4)
    assert len(timed["samples_ms"]) == 2 and [row["index"] for row in timed["failed"]] == [0, 2] and timed["p50_ms"] is not None
