from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, Runtime, files
from semantic_switch.metrics import (Counts, Progress, comparison_identity, correctness, membership, paired, reconcile, resample, success)
from semantic_switch.presentation import identity as presentation_identity
from bench.outcomes import fingerprint
from bench.outcomes import resumable, work_key
from semantic_testkit.scoring import FakeScorer
from tests.conftest import classify


def scope(n=4, families=None, condition="workflow"):
    families = families or [f"f{i // 2}" for i in range(n)]
    return [{"key": f"k{i}", "family": families[i], "gold": i % 2,
             "comparison": comparison_identity(task="t", projection=f"p{i}", gold=i % 2, answers=["a", "b"], scope="fact", condition=condition)}
            for i in range(n)]


def arm(expected, correct, *, drop=(), condition=None):
    rows = []
    for row, ok in zip(expected, correct):
        if row["key"] in drop:
            continue
        comparison = row["comparison"] if condition is None else comparison_identity(task="t", projection=f"p{row['key'][1:]}", gold=row["gold"],
                                                                                    answers=["a", "b"], scope="fact", condition=condition)
        rows.append({"key": row["key"], "comparison": comparison, "gold": row["gold"], "outcome": "predicted", "correct": ok, "accepted": True,
                     "family": "source-alias"})
    return rows


def test_unordered_rows_pair_by_key_and_missing_counts_as_failure():
    expected = scope(2)
    reconciled, support = reconcile(expected, list(reversed(arm(expected, [True, True], drop={"k1"}))))
    assert [row["key"] for row in reconciled] == ["k0", "k1"] and reconciled[1]["outcome"] == "unavailable"
    summary = success(reconciled)
    assert summary["correct_per_submission"] == .5 and summary["accuracy_given_prediction"] == 1.
    assert support == {"expected": 2, "submitted": 1, "unavailable": 1, "predicted": 1, "families": 1}


def test_pairing_rejects_changed_gold_scope_condition_and_duplicates():
    expected = scope(2)
    with pytest.raises(ConfigError, match="comparison identity"):
        reconcile(expected, arm(expected, [True, True], condition="direct"))
    changed = arm(expected, [True, True])
    changed[0]["gold"] = 1
    with pytest.raises(ConfigError, match="gold differs"):
        reconcile(expected, changed)
    with pytest.raises(ConfigError, match="repeats keys"):
        reconcile(expected, arm(expected, [True, True]) * 2)
    with pytest.raises(ConfigError, match="outside the frozen expected scope"):
        reconcile(expected[:1], arm(expected, [True, True]))


def test_identical_arms_have_zero_difference_and_bootstrap_keeps_canonical_components():
    expected = scope(8)
    rows = arm(expected, [True, False] * 4)
    report = paired(expected, rows, rows, seed=3, count=200)
    assert report["difference"]["estimate"] == 0 and report["difference"]["interval"] == [0, 0]
    assert report["fixes"] == report["regressions"] == 0 and report["difference"]["families"] == 4
    assert "degenerate_interval" in report["difference"]["flags"]
    assert report["difference"]["weighting"] == "row" and report["difference"]["scope"].startswith("within-pack family-resampling interval")
    better = paired(expected, rows, arm(expected, [True] * 8), seed=3, count=200)
    assert better["fixes"] == 4 and better["difference"]["estimate"] == .5
    assert better["difference"] == paired(expected, rows, arm(expected, [True] * 8), seed=3, count=200)["difference"]
    one = paired(scope(2, ["f", "f"]), arm(scope(2, ["f", "f"]), [True, True]), arm(scope(2, ["f", "f"]), [True, False]), seed=0, count=10)
    assert one["difference"]["interval"] is None and one["difference"]["support"] == "insufficient"


def _row_bootstrap(rows, statistic, seed, count):
    import random
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    names, values = sorted(groups), []
    import numpy as np
    draws = np.random.Generator(np.random.PCG64(seed)).integers(len(names), size=(count, len(names)), dtype=np.int64)
    for draw in draws:
        value = statistic([row for index in draw for row in groups[names[index]]])
        if value is not None:
            values.append(value)
    return values


def test_resample_of_family_sufficient_statistics_matches_the_row_reference():
    import random
    rnd = random.Random(7)
    rows = [{"family": f"f{rnd.randrange(9)}", "accepted": rnd.random() < .7, "correct": rnd.random() < .6} for _ in range(80)]
    def precision(sample):
        chosen = [row["correct"] for row in sample if row["accepted"]]
        return sum(chosen) / len(chosen) if chosen else None
    found = resample(rows, lambda row: (row["accepted"] and row["correct"], row["accepted"]), seed=5, count=300)
    reference = sorted(_row_bootstrap(rows, precision, 5, 300))
    assert found["estimate"] == precision(rows) and found["valid_replicates"] == len(reference) and found["rows"] == 80
    from semantic_switch.metrics import percentile
    assert found["interval"] == [percentile(reference, .025), percentile(reference, .975)]


def test_mostly_negative_membership_exposes_positive_precision_and_soft_gold():
    rows = [{"gold": False, "raw": False, "value": False, "accepted": True}] * 95 + \
           [{"gold": False, "raw": True, "value": True, "accepted": True}] * 4 + [{"gold": True, "raw": True, "value": True, "accepted": True}] + \
           [{"gold": None, "raw": True, "value": True, "accepted": True}] * 3
    report = membership(rows)
    assert report["micro_accepted_precision"] == .96 and report["accepted_outcomes"]["positive_precision"] == .2
    assert report["soft_or_unlabelled"] == 3 and report["hard"] == 100 and report["accepted_outcomes"]["false_positive"] == 4


def test_one_count_reducer_serves_correctness_success_and_progress():
    rows = [(True, True, "predicted"), (False, True, "predicted"), (True, False, "predicted"), (None, False, "failed"), (None, False, "unavailable")]
    counts = Counts(rows).counts
    assert counts == {"submitted": 5, "predicted": 3, "raw_correct": 2, "raw_wrong": 1, "accepted": 2, "correct_accepted": 1, "wrong_accepted": 1,
                      "correct_rejected": 1, "wrong_rejected": 0, "failed": 1, "unavailable": 1}
    assert correctness((c, a) for c, a, _ in rows) == Counts(rows).rates()
    found = success({"correct": c, "outcome": o} for c, _, o in rows)
    assert (found["expected"], found["submitted"], found["predicted"], found["correct"], found["failed"]) == (5, 4, 3, 2, 1)
    with pytest.raises(ConfigError, match="accepted answer needs a prediction"):
        Counts([(None, True)])
    assert Counts([(None, True)], strict=False).counts["accepted"] == 1


def test_progress_formatter_separates_fresh_and_resumed_counts():
    stream, now = io.StringIO(), [0.]
    resumed = [{"correct": True, "accepted": False, "outcome": "predicted"}] * 3 + [{"correct": None, "accepted": False, "outcome": "failed"}]
    progress = Progress("bench.x", 10, unit="situations", resumed=resumed, every_rows=2, every_seconds=60, stream=stream, clock=lambda: now[0])
    progress.record([{"correct": True, "accepted": True, "outcome": "predicted"}])
    assert stream.getvalue() == ""
    progress.record([{"correct": None, "accepted": False, "outcome": "failed"}])
    now[0] = 3.
    final = progress.finish()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2 and "completed 6/10 (fresh 2, resumed 4)" in lines[0] and lines[1].startswith("bench.x [situations] final")
    assert final["failed"] == 2 and final["raw_correct"] == 4 and final["correct_accepted"] == 1 and final["elapsed_s"] == 3.
    assert final["completed"] == 6 and final["fresh"] == 2 and final["resumed"] == 4
    with pytest.raises(ConfigError, match="unknown progress"):
        progress.add(accuracy=1)
    progress.add(submitted=2, failed=1)
    assert progress.counts["completed"] == 8 and progress.counts["failed"] == 3


def test_fit_report_persists_support_and_disables_unsupported_search():
    from semantic_switch.compiler import compile_spec
    from semantic_switch.judgment import Policy
    from semantic_switch.observations import Observation
    from semantic_switch.tools.policy import fit_report
    plan = compile_spec({"name": "s", "fields": {"t": {"type": "text"}}, "reads": ["t"], "facts": {"c": {"check": "It holds.", "target": .9}}})
    fact = plan.fact("c")
    rows = [Observation(f"o{i}", {}, 0 if i < 25 else 1, [.99, .01], group=f"g{i % 5}") for i in range(30)]
    report = fit_report(fact, rows, Policy(.9, .9))
    assert report["true"]["status"] == "disabled" and report["true"]["accepted"] == 0 and report["true"]["searched_thresholds"] >= 2
    assert report["false"]["status"] == "uncalibrated_default" and report["false"]["n_min_meaning"].startswith("heuristic")
    good = fit_report(fact, [Observation(f"o{i}", {}, 0, [.99, .01], group=f"g{i % 7}") for i in range(30)], Policy(.9, .9))
    assert good["true"]["status"] == "fitted" and good["true"]["accepted_families"] == 7 and good["true"]["in_sample_coverage"] == 1


def test_latency_reports_nonzero_logical_work():
    from bench.latency import logical_tokens
    rt = Runtime(FakeScorer())
    sw = rt.load("switches/refund.yaml", draft=True)
    results = [sw({"message": "please refund my last charge"}), sw({"message": "hello"})]
    assert logical_tokens(results) > 0
    assert logical_tokens([SimpleNamespace(cost={})]) is None


def test_changed_code_or_presentation_refuses_resume(tmp_path):
    output = tmp_path / "run.json"
    with resumable(output, {"code": fingerprint(), "presentation": presentation_identity()}) as journal:
        journal.commit([(work_key("a"), {"v": 1})])
    with pytest.raises(ConfigError, match="different run"):
        resumable(output, {"code": {**fingerprint(), "sources": "changed"}, "presentation": presentation_identity()})


def test_baseline_plugin_classifies_failures(tmp_path):
    from tests.conftest import Baseline
    report = SimpleNamespace(when="call", longrepr=SimpleNamespace(reprcrash=SimpleNamespace(message="AssertionError: x")), failed=True, skipped=False, nodeid="t::a")
    assert classify(report) == "assertion"
    plugin = Baseline(tmp_path / "baseline.json")
    plugin.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid="t::a", get_closest_marker=lambda name: None)]))
    plugin.pytest_runtest_logreport(report)
    plugin.pytest_runtest_logreport(SimpleNamespace(when="setup", failed=False, skipped=True, nodeid="t::b"))
    plugin.pytest_sessionfinish(SimpleNamespace(), 1)
    written = files.read_json(tmp_path / "baseline.json")
    assert (written["failed"], written["skipped"], written["passed"]) == (1, 1, 0) and written["failures"] == {"t::a": "assertion"}
    assert written["selected"] == [{"id": "t::a", "kind": "fake_behavior"}] and written["code"]["sources"]


def exact(**row):
    return {"scope": ["a", "b"], "selected": [], "unresolved": [], "gold": [], "reference_unresolved": [], "prohibited": [], "complete_gold": True,
            "prediction_complete": True, "outcome": "predicted", **row}


def test_exact_sets_require_complete_reference_and_successful_complete_predictions():
    from semantic_switch.metrics import METRICS, exact_sets
    rows = [exact(), exact(unresolved=["a", "b"], prediction_complete=False), exact(unresolved=["b"], prediction_complete=False),
            exact(outcome="failed", prediction_complete=False), exact(outcome="unavailable", prediction_complete=False),
            exact(selected=["a"], gold=["a"], unresolved=["b"], prediction_complete=False),
            exact(selected=["a"], gold=["a"]), exact(selected=["a"], gold=["b"]), exact(scope=[]),
            exact(selected=["a"], gold=["a"], reference_unresolved=["b"], complete_gold=False)]
    report = exact_sets(rows)
    assert report["exact"] == 3 and report["gold_complete"] == 9 and report["incomplete_reference"] == 1 and report["failed"] == 2
    assert report["version"] == METRICS == 5 and report["exact_rate"] == 3 / 9
    for bad in (exact(selected=["a", "a"]), exact(selected=["z"]), exact(selected=["a"], unresolved=["a"], prediction_complete=False),
                exact(unresolved=["a"]), exact(outcome="failed"), exact(prediction_complete=None), exact(gold=["c"]),
                exact(complete_gold=False), exact(reference_unresolved=["a"]), exact(gold=["a"], reference_unresolved=["a"], complete_gold=False),
                exact(reference_unresolved=["z"], complete_gold=False), exact(reference_unresolved=["b", "b"], complete_gold=False),
                exact(prohibited=["a"], gold=["a"]), exact(prohibited=["b"], reference_unresolved=["b"], complete_gold=False)):
        with pytest.raises(ConfigError):
            exact_sets([bad])
    for missing in ("reference_unresolved", "prediction_complete", "complete_gold", "outcome", "prohibited"):
        with pytest.raises(KeyError):
            exact_sets([{k: v for k, v in exact().items() if k != missing}])


def test_action_sets_score_known_reference_and_report_unknown_selections_as_unassessed():
    from semantic_switch.metrics import action_sets
    rows = [exact(scope=["a", "b", "c"], selected=["a", "c"], gold=["a"], reference_unresolved=["c"], complete_gold=False, prohibited=["b"]),
            exact(scope=["a", "b", "c"], selected=["b"], gold=["a"], prohibited=["b"]),
            exact(scope=["a", "b", "c"], gold=["a"], unresolved=["a"], prediction_complete=False)]
    report = action_sets(rows)
    assert "exact" not in report and "exact_rate" not in report
    assert report["known_reference_support"] == 8 and report["unassessed_selected"] == 1 and report["reference_unresolved"] == 1
    assert report["precision"] == 1 / 2 and report["recall"] == 1 / 3 and report["known_reference"]["false_positive"] == 1
    assert (report["prohibited_selected"], report["frames_with_prohibited"], report["required_omissions"], report["prediction_unresolved"]) == (1, 1, 2, 1)


def test_snapshot_set_rows_mark_unresolved_and_failed_predictions_incomplete():
    from bench import snapshots
    from semantic_switch.metrics import exact_sets
    domain = snapshots.DOMAINS["zt411"]
    keys = domain.sets["actions"]
    gold = {"values": {f"actions/{k}": False for k in keys}, "prohibited": []}
    def row(states, outcome="predicted"):
        return {"gold": gold, "outcome": outcome, "targets": snapshots.targets(domain, {"actions": {"states": states}})}
    settled = row({k: "rejected" for k in keys})
    unresolved = row({k: "unexamined" for k in keys})
    failed = row({k: "rejected" for k in keys}, "failed")
    sets = snapshots.set_rows(domain, [settled, unresolved, failed], "actions")
    assert [r["prediction_complete"] for r in sets] == [True, False, False]
    assert exact_sets(sets)["exact"] == 1
