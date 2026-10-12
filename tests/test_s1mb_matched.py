from __future__ import annotations

import copy

import pytest

from bench import s1mb_matched
from bench.outcomes import work_key
from .test_external_primitives import arms as pairs, prediction


def three():
    rows = pairs()
    rows["third"] = copy.deepcopy(rows["other"])
    third = rows["third"]
    third[work_key("bench-choice", "c")]["predictions"] = [prediction("q", None, None, "failed", "failed: timeout")]
    third[work_key("bench-score", "s")]["predictions"] = [prediction(q, None, None, "unsupported", "unsupported: x") for q in ("q1", "q2", "q3")]
    third[work_key("bench-noul", "n")]["requests"] = [{"elapsed_ms": 90.0, "usage": {"input_tokens": 10}, "returned_model": "m", "completion": "ok"}]
    third[work_key("bench-choice", "c")]["requests"] = [{"elapsed_ms": 900.0, "usage": None, "returned_model": None, "completion": "failed"}]
    third[work_key("bench-score", "s")]["requests"] = []
    for arm in rows.values():
        for index, case in enumerate(arm.values()):
            case["ms"] = 100.0 * (index + 1)
            for q in case["predictions"]:
                q["mass"] = None
    return {name: ({"digest": name, "configuration": {"inputs": {"provenance": {"evaluator": "pinned"}, "cases": "digest",
                                                                **({"keys": "k"} if name == "third" else {})}}}, arm)
            for name, arm in rows.items()}


def test_join_by_case_and_question_id_is_order_free_and_counts_outcomes():
    recorded = three()
    report = s1mb_matched.analyze(recorded)
    assert (report["matched_cases"], report["judgments"], report["jointly_scored"], report["jointly_scored_hard"]) == (3, 6, 2, 1)
    assert report["unscored"] == {"base": 2, "other": 1, "third": 4} and report["availability"]["third"] == {"failed": 1, "ok": 1, "uncertain": 1, "unsupported": 3}
    row = report["hard_label"]["noul/in_scope"]["base"]["curve"]["0.9"]
    assert row == {"accepted": 1, "correct": 1, "wrong": 0, "abstain": 0, "selective_precision": 1.0, "coverage": 1.0, "benchmarks": 1}
    assert report["hard_label"]["noul/in_scope"]["other"]["curve"]["0.999"]["abstain"] == 1 and set(report["paired"]) == {"base->other", "base->third", "other->third"}
    reordered = {name: (header, {key: {**case, "predictions": case["predictions"][::-1]} for key, case in reversed(arm.items())})
                 for name, (header, arm) in recorded.items()}
    assert s1mb_matched.analyze(reordered) == report


@pytest.mark.parametrize("change,needle", [
    (lambda h, a: h["configuration"]["inputs"].update(cases="other"), "provenance or case definitions"),
    (lambda h, a: a[work_key("bench-noul", "n")].update(targets={**a[work_key("bench-noul", "n")]["targets"], "q1": {"false": 1, "true": 0}}), "targets"),
    (lambda h, a: a[work_key("bench-noul", "n")]["predictions"].append(a[work_key("bench-noul", "n")]["predictions"][0]), "duplicate question IDs"),
    (lambda h, a: a[work_key("bench-noul", "n")]["predictions"][0].update(probabilities={"yes": .9, "no": .1}), "option IDs"),
])
def test_changed_definitions_provenance_duplicates_and_gold_fail(change, needle):
    recorded = three()
    change(*recorded["third"])
    with pytest.raises(ValueError, match=needle):
        s1mb_matched.analyze(recorded)


def test_timing_separates_failed_request_free_and_unrecorded_strata():
    recorded = three()
    timing = s1mb_matched.analyze(recorded)["timing"]
    third = timing["third"]["all"]
    assert (third["cases"], third["case_ms"]["count"], third["non_failed_case_ms"]["count"], third["model_work_case_ms"]["count"]) == (3, 3, 2, 2)
    assert (third["requests"], third["failed_requests"], third["request_ms"], third["usage"]) == (2, 1, {"count": 2, "p50": 495.0, "p95": 900.0}, {"input_tokens": 10})
    assert timing["third"]["question_count"]["3"]["model_work_case_ms"] is None and timing["base"]["all"]["model_work_case_ms"]["count"] == 3
    assert "requests" not in timing["base"]["all"] and timing["third"]["input_characters_from"] is None
    for name, (_, arm) in recorded.items():
        for size, case in zip((10, 2000, 50000), arm.values()):
            case["input_characters"] = size
    strata = s1mb_matched.analyze(recorded)["timing"]["base"]["input_characters_from"]
    assert {key: value["cases"] for key, value in strata.items()} == {"0": 1, "1000": 1, "16000": 1}
