import shutil
from pathlib import Path

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.judgment import Policy
from semantic_switch.tools.data import Labels
from bench import policy_adaptation, quality
from bench.question_form import paired
from tests.conftest import FakeScorer
from tests.test_switch import ROOT, t
from .conftest import envelopes

RESULTS = Path(__file__).parents[1] / "bench/results"


@pytest.fixture
def path(tmp_path):
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", tmp_path / "route_ticket.yaml")
    return tmp_path / "route_ticket.yaml"


def label(text, value=True, **metadata):
    return {"input": t(text), "facts": {"refund_requested": value}, **metadata}


def examples(rt, path, rows):
    path.with_name("route_ticket.test.jsonl").unlink(missing_ok=True)
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes(rows))
    return Labels(rt.load(path, draft=True)).examples("refund_requested")


def test_adaptation_fixture_materializes_and_exports_synthetic_labels(tmp_path, monkeypatch):
    exported, export = [], tools.export
    monkeypatch.setattr(policy_adaptation.tools, "export", lambda rt, path, **k: exported.append(export(rt, path, split=None)) or {"path": "unused"})
    with Runtime({"judge": FakeScorer()}) as bound:
        definitions = policy_adaptation.materialize(bound, tmp_path)["definitions"]
    assert exported and len(exported) == len(definitions)
    for definition in definitions:
        rows = files.read_jsonl(Path(definition["path"]).with_suffix(".test.jsonl"))
        assert {source["origin"] for row in rows for history in row["label_sources"].values() for source in history} == {"synthetic"}


def test_malformed_origins_fail_on_raw_test_and_correction_paths(path):
    rt = Runtime({"judge": FakeScorer()})
    with pytest.raises(ConfigError, match="label source origin must be one of"):
        examples(rt, path, [label("refund please", label_origin="authored")])
    path.with_name("route_ticket.test.jsonl").unlink()
    with pytest.raises(ConfigError, match="label source origin must be one of"):
        rt.load(path).correct(t("refund please"), fact="refund_requested", value=True, sources=[{"origin": "authored", "producer": "x"}])
    sw = rt.load(path)
    files.append(sw.paths.examples, [{"fact": "refund_requested", "meaning_rev": sw.plan.fact("refund_requested").meaning_rev,
                                      "input": t("refund please"), "value": True, "label_origin": "human"}])
    with pytest.raises(ConfigError, match="without label_sources are obsolete"):
        rt.load(path)


def test_reordered_duplicates_give_identical_records_or_conflicts(path):
    rt = Runtime({"judge": FakeScorer()})
    rows = [label("refund please", group="a", label_origin="unknown", selection={"selected_by": "random"}),
            label("refund please", group="b", label_origin="human", selection={"selected_by": "manual"}),
            label("hello there", value=False, label_origin="rule")]
    forward, backward = examples(rt, path, rows), examples(rt, path, rows[::-1])
    assert sorted(forward, key=lambda e: e.id) == sorted(backward, key=lambda e: e.id)
    merged = next(e for e in forward if e.label == 0)
    assert merged.label_origin == "mixed" and {s.origin for s in merged.sources} == {"unknown", "human"}
    assert next(e for e in forward if e.label == 1).label_origin == "rule"
    conflicting = [label("refund please"), label("refund please", value=False)]
    messages = []
    for order in (conflicting, conflicting[::-1]):
        with pytest.raises(ConfigError, match="conflicting labels") as error:
            examples(rt, path, order)
        messages.append(str(error.value))
    assert messages[0] == messages[1]


def test_correction_log_merges_contributors_and_carries_sources_into_the_dataset(path):
    rt = Runtime({"judge": FakeScorer()})
    sw = rt.load(path)
    teacher = {"origin": "synthetic", "producer": "teacher"}
    sw.correct(t("refund please"), fact="refund_requested", value=True, sources=[teacher])
    sw.correct(t("refund please"), fact="refund_requested", value=True)
    (example,) = [e for e in rt.load(path).snapshot.examples["refund_requested"]]
    assert [s.origin for s in example.sources] == ["synthetic", "human"] and [s.producer for s in example.sources] == ["teacher", "ss correct"]
    labels = Labels(rt.load(path, draft=True))
    (row,) = [r for r in labels.manifest["observations"] if r["fact"] == "refund_requested"]
    assert row["label_sources"] == [s.digest for s in example.sources]


def test_provenance_never_changes_model_evidence(path):
    rt = Runtime({"judge": FakeScorer()})
    evidence = lambda origin: [(e.id, e.data, e.projection, e.text) for e in examples(rt, path, [label("refund please", label_origin=origin)])]
    assert evidence("synthetic") == evidence("human") == evidence("unknown")


def test_evaluation_reports_origin_counts(path):
    rt = Runtime({"judge": FakeScorer()})
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes([label("refund please", label_origin="synthetic"), label("hello there", value=False)]))
    report = tools.run_test(rt, path)
    assert report["fresh"]["label_origins"] == {} and report["facts"]["refund_requested"]["label_origins"] == {"synthetic": 1, "human": 1}


def check_row(p, mass=None, label=True):
    return {"p": p, "mass": mass, "label": label, "criterion": "The customer asks for money back.", "excludes": [], "error": None}


def test_sweeps_use_deployed_acceptance_boundaries():
    assert quality.decision(check_row(.1)) == ("ok", False)
    assert quality.decision(check_row(.9)) == ("ok", True)
    assert quality.decision(check_row(.5)) == ("uncertain", None)
    assert quality.decision(check_row(.99, mass=.4)) == ("uncertain", None)
    assert quality.decision(check_row(None)) == ("failed", None)
    asymmetric = Policy(.95, .85)
    assert quality.decision(check_row(.12), asymmetric) == ("ok", False)
    assert quality.decision(check_row(.9), asymmetric) == ("uncertain", None)
    assert quality.selective([check_row(.1, label=False), check_row(.9), check_row(.5)])["accepted"] == 2


def test_summary_matches_runtime_statuses_and_reconciles():
    def oracle(state, question, width):
        return [.1, .9] if "twice" in state else [.9, .1] if "money" in state else [.5, .5]
    with Runtime({"judge": FakeScorer(oracle)}) as rt:
        result = quality.run(rt, checks_spec={"refund": quality.CHECKS["refund"]}, picks_spec=[])
    rows = result["checks"]
    recomputed = [quality.decision({k: v for k, v in row.items() if k != "value"}, Policy(.9, .9)) for row in rows]
    assert recomputed == [(row["status"], row["value"]) for row in rows]
    summary = quality.summarize(result)
    assert sum(summary["check_statuses"].values()) == summary["items"] == len(rows)
    assert summary["coverage_0.9"] == round(sum(row["status"] == "ok" for row in rows) / len(rows), 3)


def test_jev_selection_rows_recompute_accepted_answers():
    arms = files.read_json(RESULTS / "question-form-jev.json")["arms"]
    counts = {name: quality.selective(arm["checks"]) for name, arm in arms.items()}
    assert [(counts[name]["accepted"], counts[name]["correct"]) for name in ("bundled:claim", "bundled:question", "fresh:claim", "fresh:question")] == \
        [(183, 183), (205, 205), (111, 111), (121, 121)]
    assert quality.summarize({"checks": arms["bundled:question"]["checks"], "picks": []})["coverage_0.9"] == round(205 / 240, 3)


def test_question_form_separates_wording_from_request_availability():
    arms = files.read_json(RESULTS / "question-form-gemma26b.json")["arms"]
    out = paired(arms["fresh:claim"]["checks"], arms["fresh:question"]["checks"])
    assert (out["both_scored"], out["fixes"], out["regressions"]) == (143, 8, 1)
    assert out["failed_to_scored"] == {"correct": 15, "incorrect": 0} and out["scored_to_failed"]["correct"] == 1
    assert (out["claim_correct"], out["question_correct"]) == (133, 154)
    assert out["question_correct"] == out["claim_correct"] + out["fixes"] - out["regressions"] + \
        out["failed_to_scored"]["correct"] - out["scored_to_failed"]["correct"]


@pytest.mark.parametrize("change", [{"id": "other"}, {"label": False}, {"input": "different evidence"}])
def test_question_form_requires_matching_items(change):
    row = {"id": "a", "label": True, "input": "text", "p": .9, "error": None}
    with pytest.raises(ConfigError, match="same unique item IDs"):
        paired([row], [dict(row, **change)])
    with pytest.raises(ConfigError, match="same unique item IDs"):
        paired([row, row], [row, row])
