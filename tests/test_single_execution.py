from __future__ import annotations

from collections import Counter

import pytest
import yaml

from semantic_switch import ConfigError, Runtime

from .conftest import FakeScorer

SPEC = {
    "name": "single",
    "fields": {"text": {"type": "text", "required": True}},
    "reads": ["text"],
    "facts": {
        "urgent": {"check": "The message is urgent."},
        "refund": {"check": "The customer asks for money back."},
        "weather": {"check": "The message mentions the weather."},
        "route": {"select": "Which desk handles it?", "exclusive": True,
                  "cards": [{"id": "billing", "title": "Billing desk", "applies_when": [{"fact": "refund"}]},
                            {"id": "triage", "title": "Triage desk", "applies_when": [{"fact": "urgent"}]}]},
        "issues": {"select": "Which issues apply?", "labels": {"late": "Late delivery", "broken": "Broken item", "invoice": "Invoice"},
                   "max_claims": 1},
    },
    "cases": [{"id": "urgent", "when": "urgent", "then": "escalate"}, {"id": "refund", "when": "refund", "then": "billing"}],
    "otherwise": "general",
}


class Flipping:
    def __init__(self, truths, failing=()):
        self.truths, self.failing, self.asked = truths, failing, Counter()

    def __call__(self, state, question, width):
        key = next((k for k in (*self.truths, *self.failing) if k in question), question)
        self.asked[key, state] += 1
        if key in self.failing:
            raise RuntimeError("extra demand failed")
        truth = self.truths.get(key, False) if self.asked[key, state] == 1 else not self.truths.get(key, False)
        return [.97, .03, 0.][:width] if truth else [.03, .97, 0.][:width]


def switch(tmp_path, oracle, spec=SPEC):
    path = tmp_path / "single.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    rt = Runtime({"judge": FakeScorer(oracle)})
    return rt, rt.load(path, draft=True), path


def test_case_facts_and_cost_come_from_one_call_with_each_judgment_once(tmp_path):
    oracle = Flipping({"urgent": True, "money back": True, "weather": False})
    _, sw, _ = switch(tmp_path, oracle)
    result = sw({"text": "urgent refund"}, facts=["refund", "weather", "route"])
    assert (result.status, result.case, result.answer) == ("ok", "urgent", "escalate")
    assert result.facts["refund"].value is True and result.facts["weather"].value is False and result.facts["urgent"].value is True
    assert not result.facts["route"].settled and set(result.facts["route"].unresolved) == {"billing", "triage"}
    assert set(oracle.asked.values()) == {1} and {"urgent", "money back", "weather"} <= {key for key, _ in oracle.asked}
    assert result.cost["judge"] == sum(oracle.asked.values()) == 3 and result.cost["rounds"] == 2 and result.cost["logical_tokens"] > 0


def test_omitted_facts_keep_demand_driven_behavior(tmp_path):
    oracle = Flipping({"urgent": True})
    _, sw, _ = switch(tmp_path, oracle)
    result = sw({"text": "urgent"})
    assert result.case == "urgent" and {key for key, _ in oracle.asked} == {"urgent"} and "weather" not in result.facts


def test_irrelevant_fact_does_not_change_the_case_and_failed_extra_demand_is_explicit(tmp_path):
    oracle = Flipping({"money back": True}, failing=("weather",))
    _, sw, _ = switch(tmp_path, oracle)
    result = sw({"text": "refund"}, facts=["weather"])
    assert (result.status, result.case) == ("ok", "refund")
    assert result.facts["weather"].status == "failed" and not result.facts["weather"].settled and result.errors["weather"]


def test_partial_budget_reports_unexamined_members(tmp_path):
    oracle = Flipping({"Late delivery": True})
    _, sw, _ = switch(tmp_path, oracle)
    issues = sw({"text": "urgent"}, facts=["issues"]).facts["issues"]
    assert issues.complete is False and not issues.settled and sum(state == "unexamined" for state in issues.states.values()) == 2


def test_windowed_run_and_failed_binding_represent_requested_facts(tmp_path):
    oracle = Flipping({"money back": True})
    _, sw, _ = switch(tmp_path, oracle)
    first, second = sw.run([{"text": "a"}, {"text": "b"}], facts=["weather"])
    assert first.case == second.case == "refund" and first.facts["weather"].value is False and second.facts["weather"].value is False
    failed = sw({}, facts=["weather"])
    assert failed.status != "ok" and failed.facts["weather"].status in ("needs_input", "failed") and not failed.facts["weather"].settled


def test_requested_facts_are_validated_before_execution(tmp_path):
    oracle = Flipping({})
    _, sw, _ = switch(tmp_path, oracle)
    for bad in ("weather", ["nope"], [1]):
        with pytest.raises(ConfigError):
            sw({"text": "x"}, facts=bad)
    assert not oracle.asked


def test_snapshot_replacement_during_execution_keeps_the_captured_snapshot(tmp_path):
    renamed = {**SPEC, "facts": {("climate" if k == "weather" else k): v for k, v in SPEC["facts"].items()}}
    state = {}

    def oracle(text, question, width):
        if "weather" in question and "done" not in state:
            state["path"].write_text(yaml.safe_dump(renamed, sort_keys=False))
            state["switch"].reload()
            state["done"] = True
        return [.03, .97, 0.][:width]
    _, sw, path = switch(tmp_path, oracle)
    state.update(path=path, switch=sw)
    result = sw({"text": "x"}, facts=["weather"])
    assert "weather" in result.facts and "climate" not in result.facts and "climate" in sw.plan.fact_index


def test_describe_exposes_snapshot_bound_documents(tmp_path):
    spec = {**SPEC, "guidance": {"policy": {"text": "Refunds within 30 days."}},
            "facts": {**SPEC["facts"], "refund": {**SPEC["facts"]["refund"], "guidance": ["policy"]}}}
    _, sw, path = switch(tmp_path, Flipping({}), spec)
    (document,) = sw.describe()["documents"]
    assert dict(document) == {"reference": "policy", "digest": sw.plan.fact("refund").judgment.guidance.sections[0].content_digest,
                              "text": "Refunds within 30 days."}
    path.write_text(yaml.safe_dump(SPEC, sort_keys=False))
    assert sw.describe()["documents"][0]["text"] == "Refunds within 30 days."


def frozen_printer(tmp_path):
    from examples import printer_snapshot
    from semantic_switch import files, tools
    from semantic_switch.artifacts import RECEIPT_NAME, RECEIPT_SCHEMA, artifact_digest
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    spec = yaml.safe_load(printer_snapshot.DEFINITION.read_text())
    spec["guidance"] = {"manual": {"text": "Calibrate only after verified loading."}}
    spec["facts"]["actions"]["guidance"] = ["manual"]
    task = deployment / "printer.yaml"
    task.write_text(yaml.safe_dump(spec, sort_keys=False))
    with Runtime({"judge": FakeScorer()}) as rt:
        tools.lock(rt, task)
    (deployment / "specialist.gguf").write_bytes(b"specialist")
    listed = {name: files.sha256_file(deployment / name) for name in ("specialist.gguf", "printer.yaml", "printer.lock.json")}
    receipt = {"schema": RECEIPT_SCHEMA, "kind": "specialization-receipt", "selected": "epoch-1", "gguf": "specialist.gguf", "task": "printer.yaml",
               "lock": {"path": "printer.lock.json", "id": "fixture", "candidate_id": "c" * 64}, "files": listed, "native": {"device": "cpu"},
               "execution": {"device": "cpu"}, "provenance": {"kind": "base", "rows": []}}
    files.publish(deployment / RECEIPT_NAME, {**receipt, "id": artifact_digest(receipt)})
    return deployment / RECEIPT_NAME, spec


def test_receipt_example_executes_the_frozen_task_not_working_tree_files(tmp_path, monkeypatch):
    from examples import printer_snapshot
    from semantic_switch.artifacts import load_receipt
    receipt, spec = frozen_printer(tmp_path)
    monkeypatch.setattr(printer_snapshot, "DEFINITION", tmp_path / "deleted.yaml")
    with Runtime({"judge": FakeScorer()}) as rt:
        rt._deployment = load_receipt(receipt)
        printed = printer_snapshot.run(rt.load())
        assert [d["reference"] for d in printed["documents"]] == ["manual"] and printed["documents"][0]["text"] == "Calibrate only after verified loading."
        assert len(printed["decisions"]) == len(printer_snapshot.EVIDENCE) and all("justified" in row for row in printed["decisions"])
        draft = tmp_path / "draft.yaml"
        draft.write_text(yaml.safe_dump({**spec, "guidance": {"manual": {"text": "Edited working tree."}}}, sort_keys=False))
        assert printer_snapshot.run(rt.load(draft, draft=True))["documents"][0]["text"] == "Edited working tree."
        (receipt.parent / "printer.lock.json").unlink()
        with pytest.raises(ConfigError, match="missing or changed"):
            rt.load()
