import json
import shutil

import pytest

from semantic_switch import Actor, Check, ConfigError, files, tools
from semantic_switch.judge import BATCH_QUESTIONS
from semantic_switch.protocols import MAX_INPUT_TOKENS

from .test_switch import ROOT, edit, t
from .test_tools import write_test_file


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "route_ticket.yaml"
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", p)
    return p


def test_oversized_input_fails_with_limit(rt):
    r = rt.check("word " * MAX_INPUT_TOKENS, "The text is about food.")
    assert r.status == "failed" and r.error.startswith("limit: input is")


def test_oversized_input_fails_the_switch_not_the_process(rt, path):
    edit(path, lambda s: s["fields"]["body"].pop("max_chars"))
    r = rt.load(path)(t("refund " * MAX_INPUT_TOKENS))
    assert r.status == "failed" and any(m.startswith("limit") for m in r.errors.values())


def test_large_asks_are_split_into_bounded_batches(rt, fake):
    calls = []
    score = fake.score
    fake.score = lambda reqs: calls.append(sum(len(r.suffixes) for r in reqs)) or score(reqs)
    n = BATCH_QUESTIONS + 6
    out = rt.ask("my lawyer will file a lawsuit", {f"q{i}": Check("The customer threatens legal action.")
                                                   for i in range(n)})
    assert calls == [BATCH_QUESTIONS, 6] and all(r.value is True for r in out.values())


def test_fact_and_candidate_limits(rt, path):
    edit(path, lambda s: s["facts"].update({f"extra{i}": {"check": f"Claim {i}."} for i in range(30)}))
    with pytest.raises(ConfigError, match="facts"):
        rt.load(path)
    r = rt.pick("x", "q", [f"option {i}" for i in range(1001)], shortlist="lexical")
    assert r.status == "failed" and "1000" in r.error


def test_answer_labels_go_stale_when_rules_change(rt, path):
    sw = rt.load(path)
    sw.correct(t("my lawyer will file a lawsuit"), rule="legal", actor=Actor('human'))
    assert tools.run_test(rt, path)["fresh"]["items"] + tools.run_test(rt, path)["taught"]["items"] == 1
    edit(path, lambda s: s.update(otherwise="other"))
    report = tools.run_test(rt, path)
    assert report["fresh"]["items"] + report["taught"]["items"] == 0
    assert "stale answer labels" in report["warnings"][0] and sum("stale" in w for w in report["warnings"]) == 1


def test_fact_labels_go_stale_when_meaning_changes(rt, path, tmp_path):
    inputs = tmp_path / "in.jsonl"
    files.append(inputs, files.jsonl([{"schema": 3, "input": t("refund please")}]))
    answers = iter(["y", "", "", "", "", ""])
    tools.label(rt, path, str(inputs), to="test", ask=lambda _: next(answers), show=lambda _: None, actor=Actor('human'))
    assert tools.run_test(rt, path)["facts"]["legal_threat"]["labelled"] == 1
    edit(path, lambda s: s["facts"]["legal_threat"].update(check="The customer explicitly threatens legal action."))
    report = tools.run_test(rt, path)
    assert "legal_threat" not in report["facts"] and report["warnings"][0].startswith("legal_threat: 1 stale")
    assert sum("stale" in w for w in report["warnings"]) == 1
    assert any("stale" in w for w in tools.lock(rt, path)["warnings"])


def test_test_report_reads_as_sentences_with_intervals(rt, path):
    write_test_file(path, 30)
    records = files.read_jsonl(path.with_suffix(".test.jsonl"))
    for row in records:
        row["selections"] = [{"selected_by": "random", "frame_id": "test-frame", "independent": True, "inclusion_probability": .1}]
    files.publish(path.with_suffix(".test.jsonl"), ''.join([json.dumps(r) + "\n" for r in records]))
    report = tools.run_test(rt, path)
    lo, hi = report["facts"]["legal_threat"]["precision_95"]
    assert 0 < lo < hi == 1.0
    text = tools.explain_test(report)
    assert "Fresh (30): ok" in text and "legal_threat (fresh): 10 actually true" in text
    assert "95% interval" in text and "can't be told apart" not in text
