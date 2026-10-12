import json
import shutil

import pytest

from semantic_switch import files, tools

from .test_switch import ROOT, edit, t
from .conftest import envelopes


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "route_ticket.yaml"
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", p)
    return p


def write_test_file(path, n=60):
    items = []
    for i in range(n):
        if i % 3 == 0:
            items.append({"input": t(f"my lawyer will file a lawsuit {i}"), "facts": {"legal_threat": True},
                          "case": "legal"})
        elif i % 3 == 1:
            items.append({"input": t(f"please refund charge {i}", payment={"status": "pending"}),
                          "facts": {"legal_threat": False, "refund_requested": True}, "case": "refund_other"})
        else:
            items.append({"input": t(f"just saying hi {i}"), "facts": {"legal_threat": False,
                                                                        "refund_requested": False},
                          "case": "otherwise"})
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes(items))
    return items


def test_lock_dry_run_reports_without_writing(rt, path):
    write_test_file(path, 90)
    report = tools.lock(rt, path, dry_run=True)
    side = report["policies"]["legal_threat"]["judge"]
    assert report["lock"] is None and side["true"]["state"] == "met" and side["true"]["theta"] == 0.9
    assert side["false"]["state"] == "met" and side["false"]["held_out_precision"] >= 0.95
    assert side["false"]["interval_95"] is None
    assert side["false"]["interval_reason"]
    assert not path.with_name("route_ticket.lock.json").exists()


def test_lock_tunes_and_test_never_counts_tuning_inputs_as_unseen(rt, path):
    items = write_test_file(path)
    tools.lock(rt, path)
    data = json.loads(path.with_name("route_ticket.lock.json").read_text())
    assert data["tuned_keys"] and data["facts"]["legal_threat"]["policies"]["judge"]["true"]["state"]
    report = tools.run_test(rt, path)
    assert report["used"]["items"] == 60 and report["fresh"]["items"] == 0
    assert report["used_by"]["tuned"] == len(data["tuned_keys"]) and report["used_by"]["validated"] > 0
    assert report["used"]["ok"] == report["used"]["correct"] > 0
    assert "Used by the lock (60" in tools.explain_test(report)
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes([{"input": t("brand new"), "case": "otherwise"}]))
    again = tools.run_test(rt, path)
    assert again["used"]["items"] == report["used"]["items"] and again["fresh"]["items"] == \
        report["fresh"]["items"] + 1
    assert again["fresh"]["items"] == 1 and "legal_threat" not in again["facts"]


def test_test_separates_taught_inputs(rt, path):
    items = write_test_file(path, 9)
    sw = rt.load(path)
    sw.correct(items[2]["input"], fact="legal_threat", value=False)
    report = tools.run_test(rt, path)
    assert report["taught"]["items"] == 1 and report["fresh"]["items"] == 8


def test_diff_lists_changed_answers(rt, path):
    write_test_file(path, 9)
    tools.lock(rt, path)
    edit(path, lambda s: s.update(otherwise="other"))
    report = tools.diff(rt, path)
    assert report["changed"] >= 3 and all(c["new"][1] == "other" for c in report["changes"])


def test_specialize_supplied_folds_do_not_leak(rt, path, fake, head_bundle):
    sw = rt.load(path)
    texts = []
    for i in range(10):
        texts.append((t(f"please refund my charge alpha{i}"), True))
        texts.append((t(f"hello there beta{i}"), False))
    for d, v in texts:
        sw.correct(d, fact="refund_requested", value=v)
    fake.requests.clear()
    report = tools.specialize(rt, path, "refund_requested", head_bundle=head_bundle(rt, path))
    assert report["judge"]["settled"] <= 20 and "judge_calls_avoided" in report["lexical"]
    assert "admit" not in report and report["bundle"]
    for state, q in fake.requests:
        token = next(w for w in state.split() if w.startswith(("alpha", "beta")))
        assert token not in q
    assert not path.with_name("route_ticket.lock.json").exists()


def test_label_writes_examples(rt, path, tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    files.append(inputs, [{"schema": 2, "input": t("refund please")}, {"schema": 2, "input": t("hello")}])
    answers = iter(["y", "", "", "", "", "n", "", "", "", ""])
    n = tools.label(rt, path, str(inputs), ask=lambda _: next(answers), show=lambda _: None)
    recs = files.read_jsonl(path.with_name("route_ticket.examples.jsonl"))
    assert n == 2 and [r["value"] for r in recs] == [True, False] and {r["fact"] for r in recs} == {"legal_threat"}
