from __future__ import annotations

import pytest

from semantic_switch import Runtime, files
from semantic_testkit.scoring import FakeScorer

from bench import outcomes, quality
from bench.latency import _summary
from bench.model_comparison import isolated, suite, timings, tuned
from bench.select_quality import timing

SAMPLES = [5., 1., 3., 2., 4., 100.]
EXPECTED = {"p50": 3.5, "p95": 76.25}


def test_latency_is_linear_interpolation_with_units_and_count():
    found = outcomes.latency(SAMPLES)
    assert {key: found[key] for key in EXPECTED} == pytest.approx(EXPECTED)
    assert found["samples"] == 6 and found["unit"] == "ms"
    assert outcomes.latency([]) == {**found, "samples": 0, "p50": None, "p95": None}


def test_every_script_uses_the_same_timing_definition():
    values = iter(SAMPLES)
    timed = timings(lambda n: next(values), len(SAMPLES))
    rows = [{"arms": {"select": {"ms": ms, "requests": []}}} for ms in SAMPLES]
    for found in (_summary([{"ms": ms} for ms in SAMPLES]), timing(rows, "select")["case"]):
        assert {key: found[key] for key in EXPECTED} == pytest.approx(EXPECTED) and found["samples"] == 6
    assert timed["latency"]["samples"] == len(SAMPLES)


def test_ece_fixture_matches_the_quality_summary():
    rows = [{"p": p, "label": label, "mass": 1., "pair": f"f:{n}", "tag": "t", "lang": "en", "ms": 1., "status": "ok", "value": p >= .5,
             "criterion": "c", "excludes": [], "error": None}
            for n, (p, label) in enumerate([(.95, True), (.7, False), (.2, False), (.55, True), (.05, True)])]
    expected = outcomes.ece([.95, .7, .8, .55, .95], [True, False, True, True, False], low=.5)
    assert quality.summarize({"checks": rows, "picks": []})["ece"] == expected
    assert expected["value"] == pytest.approx((2 * abs(.95 - .5) + abs(.7 - 0) + abs(.8 - 1) + abs(.55 - 1)) / 5)


def test_threshold_tuning_is_cross_fitted_over_whole_pairs():
    rows = [{"pair": f"f:{n // 2}", "p": .99 if n % 2 == 0 else .01, "label": n % 2 == 0, "mass": 1., "criterion": "c", "excludes": []}
            for n in range(40)]
    found = tuned(rows, target=.95, minimum=5)
    assert [fold["threshold"] for fold in found["folds"]] == [.5, .5] and found["precision"] == 1 and found["coverage"] == 1
    assert sum(fold["held"]["total"] for fold in found["folds"]) == len(rows)


def test_suite_runner_returns_raw_summary_and_provenance():
    with Runtime(FakeScorer()) as rt:
        result = suite(rt, checks_spec={"refund": quality.CHECKS["refund"]}, picks_spec=[])
    assert set(result) == {"raw", "summary", "provenance"} and result["summary"]["tuned"]["folds"]
    assert len(result["raw"]["checks"]) == 2 * len(quality.CHECKS["refund"][2])


def test_isolated_arm_runs_through_the_shared_process_runner(tmp_path, monkeypatch):
    calls = []

    def process(command, **kwargs):
        calls.append(command)
        files.publish(command[command.index("--output") + 1], {"model_preparation_ms": 1.})
        return {"exit_code": 0, "stderr": ""}
    monkeypatch.setattr("bench.model_comparison.process", process)
    result = isolated("model", tmp_path / "arm.json", suites=False)
    assert "bench.model_comparison" in calls[0] and "--skip-suites" in calls[0]
    assert result["memory_boundary"].startswith("fresh subprocess")


def test_report_refuses_an_existing_destination(tmp_path):
    outcomes.report(tmp_path / "r.json", {"a": 1})
    with pytest.raises(Exception, match="already exists"):
        outcomes.report(tmp_path / "r.json", {"a": 2})
