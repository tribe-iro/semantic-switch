import json
import shutil

import pytest

from semantic_switch import files, tools

from .test_switch import ROOT, edit, t


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
    files.append_jsonl(path.with_name("route_ticket.test.jsonl"), items)
    return items


def test_lock_dry_run_reports_without_writing(rt, path):
    write_test_file(path, 90)
    report = tools.lock(rt, path, dry_run=True)
    side = report["policies"]["legal_threat"]["judge"]
    assert report["lock"] is None and side["true"]["state"] == "untuned" and side["true"]["theta"] == 0.9
    assert side["false"]["state"] == "met" and side["false"]["held_out_precision"] >= 0.95
    assert side["false"]["interval_95"][0] < side["false"]["held_out_precision"]
    assert not path.with_name("route_ticket.lock.json").exists()


def test_lock_tunes_and_test_never_counts_tuning_inputs_as_unseen(rt, path):
    items = write_test_file(path)
    tools.lock(rt, path)
    data = json.loads(path.with_name("route_ticket.lock.json").read_text())
    assert data["tuned_keys"] and data["facts"]["legal_threat"]["policies"]["judge"]["true"]["state"]
    report = tools.run_test(rt, path)
    assert report["trained_on"]["items"] == len(data["tuned_keys"]) and report["unseen"]["items"] == 60 - len(
        data["tuned_keys"])
    assert report["unseen"]["ok"] == report["unseen"]["correct"] > 0
    assert "legal_threat judge policy: TRUE side" in tools.explain_test(report)
    files.append_jsonl(path.with_name("route_ticket.test.jsonl"), [{"input": t("brand new"), "case": "otherwise"}])
    again = tools.run_test(rt, path)
    assert again["trained_on"]["items"] == report["trained_on"]["items"] and again["unseen"]["items"] == \
        report["unseen"]["items"] + 1
    assert report["facts"]["legal_threat"]["silent_misses"] == 0


def test_test_separates_taught_inputs(rt, path):
    items = write_test_file(path, 9)
    sw = rt.load(path)
    sw.correct(items[2]["input"], fact="legal_threat", value=False)
    report = tools.run_test(rt, path)
    assert report["taught"]["items"] == 1 and report["unseen"]["items"] == 8


def test_diff_lists_changed_answers(rt, path):
    write_test_file(path, 9)
    tools.lock(rt, path)
    edit(path, lambda s: s.update(otherwise="other"))
    report = tools.diff(rt, path)
    assert report["changed"] >= 3 and all(c["new"][1] == "other" for c in report["changes"])


def test_specialize_cross_validation_does_not_leak(rt, path, fake):
    sw = rt.load(path)
    texts = []
    for i in range(10):
        texts.append((t(f"please refund my charge alpha{i}"), True))
        texts.append((t(f"hello there beta{i}"), False))
    for d, v in texts:
        sw.correct(d, fact="refund_requested", value=v)
    fake.requests.clear()
    report = tools.specialize(rt, path, "refund_requested")
    assert report["judge"]["settled"] <= 20 and {"gained", "avoided", "introduced", "judge_calls_avoided"} <= set(
        report["lexical"])
    assert report["judge"]["auc"] is not None and any("lexical head" in line for line in report["report"])
    for state, q in fake.requests:
        token = next(w for w in state.split() if w.startswith(("alpha", "beta")))
        assert token not in q
    assert not path.with_name("route_ticket.lock.json").exists()


def test_label_writes_examples(rt, path, tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    files.append_jsonl(inputs, [t("refund please"), t("hello")])
    answers = iter(["y", "", "", "", "", "n", "", "", "", ""])
    n = tools.label(rt, path, str(inputs), ask=lambda _: next(answers), show=lambda _: None)
    recs = files.read_jsonl(path.with_name("route_ticket.examples.jsonl"))
    assert n == 2 and [r["value"] for r in recs] == [True, False] and {r["fact"] for r in recs} == {"legal_threat"}
