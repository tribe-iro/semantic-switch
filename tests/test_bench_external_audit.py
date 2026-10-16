from __future__ import annotations

import hashlib
import json

from semantic_switch import Pick, Runtime, files
from bench import external_audit
from semantic_testkit.scoring import FakeScorer


def cached(tmp_path, monkeypatch):
    document = {"instructions": "Which field is this abstract from?", "labels": {"physics": "Physics", "biology": "Biology"}}
    rows = [{"item_id": "a", "label": "physics", "text": "Quarks bind into hadrons."}, {"item_id": "b", "label": "biology", "text": "Cells divide by mitosis."}]
    task = json.dumps(document).encode()
    main = "\n".join(json.dumps(row) for row in rows).encode()
    files.publish(tmp_path / "toy" / "task.json", task)
    files.publish(tmp_path / "toy" / "main.jsonl", main)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cached copies are not refetched")))
    return hashlib.sha256(task).hexdigest(), hashlib.sha256(main).hexdigest()


def test_audit_rows_serialize_public_results_and_keep_failures_submitted(tmp_path, monkeypatch):
    digests = cached(tmp_path, monkeypatch)
    with Runtime(FakeScorer(lambda state, question, width: [.97, .03][:width])) as rt:
        report = external_audit.audit(rt, "toy", tmp_path)
        json.dumps(report, allow_nan=False)
        assert (report["task_sha256"], report["rows_sha256"]) == digests and report["outcome"] == "predicted"
        assert report["submitted"] == 2 and report["failed"] == 0 and len(report["observations"]) == 2
        assert all(row["outcome"] == "predicted" and row["task"]["definition"] and row["raw_answer"] == "physics" for row in report["observations"])
        assert report["progress"]["submitted"] == 2 and report["progress"]["raw_correct"] == 1
        assert report["settled"] == report["progress"]["accepted"] and report["progress"]["correct_accepted"] == sum(
            row["result"]["value"] == row["label"] for row in report["observations"] if row["result"]["status"] == "ok")

        def unavailable(self, inputs, rt=None):
            raise RuntimeError("backend down")
        monkeypatch.setattr(Pick, "run", unavailable)
        failed = external_audit.audit(rt, "toy", tmp_path)
    json.dumps(failed, allow_nan=False)
    assert failed["outcome"] == "failed" and failed["error"] == {"type": "RuntimeError", "stage": "execute"} and failed["ms"] > 0
    assert (failed["task_sha256"], failed["rows_sha256"]) == digests and failed["submitted"] == failed["failed"] == 2 and failed["settled"] == 0
    assert failed["progress"]["failed"] == 2 and failed["progress"]["accepted"] == failed["progress"]["raw_correct"] == 0
    assert all(row["result"]["status"] == "failed" and row["task"]["definition"] for row in failed["observations"])
