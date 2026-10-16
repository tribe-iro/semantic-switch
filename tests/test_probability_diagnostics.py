from __future__ import annotations

import json
import math

import pytest

import semantic_switch as ss
from semantic_switch import files
from semantic_switch.judgment import ESCAPE, Judgment
from semantic_switch.metrics import OPTION_SOFTMAX, observation, observed, primitive, probabilities, reliability
from bench import snapshots
from semantic_testkit.scoring import FakeScorer

ROUNDED = "renormalize-within-rounding-bound@1"
LOWER = "token-mass-lower-bound@1"


def entry(probs, gold, *, options=("a", "b", "c"), task="t", target="x", kind="pick", family=None, mass=None, normalization=OPTION_SOFTMAX,
          source="judge", status="uncertain", accepted=None, outcome="predicted"):
    return {"task": task, "target": target, "kind": kind, "family": family or f"{task}-{target}-{gold}-{probs}", "gold": gold, "outcome": outcome,
            "observation": observed(options, probs, stage="final", normalization=normalization, status=status, source=source, mass=mass, accepted=accepted)}


def micro(*entries):
    report = probabilities(entries)
    json.dumps(report, allow_nan=False)
    return report["micro"]


def test_perfect_wrong_and_uniform_distributions_match_hand_calculations():
    perfect = micro(entry([1., 0., 0.], "a"))
    assert perfect["brier"]["value"] == 0 and perfect["nll"]["value"] == 0 and perfect["zero_gold"] == 0
    wrong = micro(entry([0., 1., 0.], "a"))
    assert wrong["brier"]["value"] == 2 and wrong["nll"] == {"value": None, "nonfinite": "+inf", "infinite": 1, "finite_mean": None, "rows": 1}
    assert wrong["zero_gold"] == 1 and wrong["scored"] == 1
    uniform = micro(entry([1 / 3] * 3, "a"))
    assert uniform["brier"]["value"] == pytest.approx(4 / 9 + 1 / 9 + 1 / 9) and uniform["nll"]["value"] == pytest.approx(math.log(3))
    table = uniform["reliability"]["table"]
    assert table[3]["count"] == 1 and table[3]["observed_correct"] == 1 and table[3]["mean_predicted"] == pytest.approx(1 / 3)


def test_zero_gold_mass_is_an_explicit_nonfinite_aggregate_and_serializes_without_nan():
    report = probabilities([entry([0., 1., 0.], "a", family="f1"), entry([.5, .5, 0.], "a", family="f2")])
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    nll = report["micro"]["nll"]
    assert nll["value"] is None and nll["nonfinite"] == "+inf" and nll["infinite"] == 1 and nll["finite_mean"] == pytest.approx(math.log(2))
    assert report["micro"]["brier"]["value"] == pytest.approx((2 + .5) / 2)
    assert report["macro"]["nll"]["nonfinite"] == "+inf" and report["macro"]["brier"]["value"] == pytest.approx(1.25)


def test_unequal_widths_true_false_unstated_and_one_convention_across_widths():
    binary = entry([.8, .2], "true", options=("true", "false"), kind="check", target="b")
    three = entry([.1, .2, .7], "unstated", options=("true", "false", "unstated"), kind="check", target="u")
    rate = entry([.25, .25, .25, .25], "high", options=("low", "mid", "high", "max"), kind="rate", target="r")
    report = probabilities([binary, three, rate])
    targets = report["targets"]["t"]
    assert targets["b"]["brier"]["value"] == pytest.approx(.04 + .04) and targets["b"]["nll"]["value"] == pytest.approx(-math.log(.8))
    assert targets["u"]["brier"]["value"] == pytest.approx(.01 + .04 + .09) and targets["u"]["nll"]["value"] == pytest.approx(-math.log(.7))
    assert targets["r"]["brier"]["value"] == pytest.approx(.5625 + 3 * .0625) and targets["r"]["nll"]["value"] == pytest.approx(math.log(4))
    assert report["micro"]["brier"]["value"] == pytest.approx((.08 + .14 + .75) / 3)
    assert set(report["kinds"]) == {"check", "rate"} and report["kinds"]["check"]["scored"] == 2


def test_rounded_provider_values_are_used_as_recorded_and_never_renormalized():
    exact = micro(entry([.7, .2, .1], "a", normalization=ROUNDED))
    assert exact["scored"] == 1 and exact["brier"]["value"] == pytest.approx(.09 + .04 + .01)
    short = micro(entry([.33, .33, .33], "a", normalization=ROUNDED))
    assert short["scored"] == 0 and short["unscored"]["incomplete"] == 1
    rounded_zero = micro(entry([0., .5, .5], "a", normalization=ROUNDED))
    assert rounded_zero["zero_gold"] == 1 and rounded_zero["nll"]["nonfinite"] == "+inf"
    lower = micro(entry([.6, .3, .05], "a", normalization=LOWER, mass=.95))
    assert lower["scored"] == 0 and lower["unscored"]["normalization"] == 1
    undeclared = micro(entry([.6, .3, .1], "a", normalization=None))
    assert undeclared["unscored"]["normalization"] == 1


def test_answer_mass_changes_only_the_unconditioned_nll():
    native, hf = entry([.8, .2], "a", options=("a", "b"), mass=1., family="f"), entry([.8, .2], "a", options=("a", "b"), mass=.5, family="f")
    first, second = micro(native), micro(hf)
    assert first["brier"] == second["brier"] and first["nll"] == second["nll"] and first["nll"]["value"] == pytest.approx(-math.log(.8))
    assert first["unconditioned_nll"]["value"] == pytest.approx(-math.log(.8)) and second["unconditioned_nll"]["value"] == pytest.approx(-math.log(.4))
    hosted = micro(entry([.8, .2], "a", options=("a", "b"), mass=.5, normalization=ROUNDED))
    assert hosted["unconditioned_nll"]["rows"] == 0 and hosted["unconditioned_nll"]["unavailable"] == 1
    massless = micro(entry([.8, .2], "a", options=("a", "b"), mass=0.))
    assert massless["nll"]["value"] == pytest.approx(-math.log(.8)) and massless["unconditioned_nll"]["nonfinite"] == "+inf"


def test_reliability_bins_include_probability_one_in_the_last_bin_and_show_empty_support():
    table = reliability([(1., True, "f"), (.9, False, "g"), (.1, True, "h"), (.0999, True, "h"), (0., False, "i")])
    assert [row["count"] for row in table["table"]] == [2, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    last = table["table"][-1]
    assert last["upper_included"] and last["lower"] == .9 and last["upper"] == 1. and last["families"] == 2
    assert last["mean_predicted"] == pytest.approx(.95) and last["observed_correct"] == .5 and last["support"] == "sparse"
    assert table["table"][4] == {"lower": .4, "upper": .5, "upper_included": False, "count": 0, "families": 0, "mean_predicted": None,
                                 "observed_correct": None, "support": "empty"}
    assert table["edges"] == [k / 10 for k in range(11)] and table["bins"] == 10
    expected = 2 / 5 * abs(.5 - .95) + 1 / 5 * abs(1 - .1) + 2 / 5 * abs(.5 - .0999 / 2)
    assert table["ece"]["value"] == pytest.approx(expected)
    assert reliability([])["ece"]["value"] is None and all(row["support"] == "empty" for row in reliability([])["table"])


def test_incomplete_gold_failed_unexamined_missing_and_rule_rows_count_against_expected_submissions():
    rows = [entry([.6, .4, 0.], "a"), entry([.6, .4, 0.], None), entry([.6, .4, 0.], "a", outcome="failed"),
            entry([.6, .4, 0.], "a", source="rule", status="ok", accepted="a"),
            {"task": "t", "target": "m", "kind": "membership", "family": "u", "gold": "true", "outcome": "predicted",
             "observation": observation(None, Judgment.check("Applies?"), normalization=OPTION_SOFTMAX)},
            {"task": "t", "target": "z", "kind": "pick", "family": "z", "gold": "a", "outcome": "predicted", "observation": None},
            entry([.6, None, .4], "a"), entry([.5, .5], "c", options=("a", "b")), entry([1.2, -.2, 0.], "a")]
    report = micro(*rows)
    assert report["submitted"] == 9 and report["scored"] == 1
    assert {k: v for k, v in report["unscored"].items() if v} == {"unknown_gold": 1, "failed": 1, "not_judge": 1, "unexamined": 1, "missing": 1,
                                                                    "partial_options": 1, "gold_outside_options": 1, "invalid": 1}
    assert report["brier"]["rows"] == 1 and report["accepted"]["accepted"] == 0


def test_model_reliability_and_policy_accepted_correctness_are_separate_populations():
    rows = [entry([.9, .1, 0.], "a", status="ok", accepted="a"), entry([.4, .6, 0.], "a", status="ok", accepted="b"),
            entry([.5, .3, .2], "b", status="uncertain"), entry([1., 0., 0.], "a", source="example", status="ok", accepted="a")]
    report = micro(*rows)
    assert report["accepted"] == {"accepted": 2, "correct": 1, "precision": .5}
    assert report["scored"] == 3 and report["reliability"]["rows"] == 3 and sum(r["observed_correct"] * r["count"] for r in report["reliability"]["table"] if r["count"]) == 1


def test_task_macro_differs_from_submitted_unit_micro_with_counts():
    rows = [entry([1., 0., 0.], "a", task="A", family="A1"), *(entry([1 / 3] * 3, "a", task="B", target=f"x{k}", family=f"B{k % 2}") for k in range(3))]
    report = probabilities(rows)
    assert report["micro"]["brier"]["value"] == pytest.approx((0 + 3 * 2 / 3) / 4)
    assert report["macro"]["brier"]["value"] == pytest.approx((0 + 2 / 3) / 2) and report["macro"]["tasks"] == report["macro"]["contributing"] == 2
    assert report["tasks"]["B"]["targets"] == 3 and report["tasks"]["B"]["families"] == 2 and report["micro"]["submitted"] == 4
    assert "targets" not in probabilities(rows, targets=False)


def result(scores, stages, *, source="judge", status="uncertain", value=None, mass=1.):
    return {"status": status, "value": value, "source": source, "score": max(scores.values()) if scores else None, "mass": mass, "scores": scores,
            "observed_models": [], "presentations": [{"stage": stage, "answers": []} for stage in stages]}


def test_semantic_keys_are_aligned_to_declared_options_regardless_of_score_order():
    pick = Judgment("pick", "Which?", ("a", "b", "c"), ("A", "B", "C"))
    canonical = observation(result({"a": .2, "b": .7, "c": .1}, ["full"]), pick, normalization=OPTION_SOFTMAX)
    permuted = observation(result({"c": .1, "b": .7, "a": .2}, ["full"]), pick, normalization=OPTION_SOFTMAX)
    assert canonical["probabilities"] == permuted["probabilities"] == [.2, .7, .1]
    narrowed = observation(result({"a": .3, "b": .7}, ["full"]), pick, normalization=OPTION_SOFTMAX)
    assert narrowed["probabilities"] == [.3, .7, None]
    assert micro({"task": "t", "target": "p", "kind": "pick", "family": "f", "gold": "b", "observation": narrowed})["unscored"]["partial_options"] == 1


def test_staged_final_distributions_are_unassessed_and_primitive_stages_report_their_own_scope():
    pick = Judgment("pick", "Which?", ("a", "b"), ("A", "B"))
    fallback = observation(result({"a": .25, "b": .75}, ["mentions", "fallback"]), pick, normalization=OPTION_SOFTMAX)
    mentions = observation(result({"a": 1.}, ["mentions"]), pick, normalization=OPTION_SOFTMAX)
    assert fallback["probabilities"] is None and fallback["stage"] == "staged" and fallback["stages"][0]["stage"] == "fallback"
    assert mentions["stages"] == []
    escape = {"task": "t", "target": "e", "kind": "pick", "family": "e", "gold": "b",
              "observation": observed(("a", "b"), None, stage="staged", normalization=OPTION_SOFTMAX, status="monitored", source="judge",
                                      stages=[primitive(("a", ESCAPE), (.4, .6), stage="mentions", normalization=OPTION_SOFTMAX, mass=.9)])}
    report = probabilities([{"task": "t", "target": "f", "kind": "pick", "family": "f", "gold": "b", "observation": fallback},
                            {"task": "t", "target": "m", "kind": "pick", "family": "m", "gold": "b", "observation": mentions}, escape])
    assert report["micro"]["scored"] == 0 and report["micro"]["unscored"]["staged"] == 3
    assert report["stages"]["fallback"]["brier"]["value"] == pytest.approx(2 * .25 ** 2) and report["stages"]["fallback"]["scored"] == 1
    assert report["stages"]["mentions"]["nll"]["value"] == pytest.approx(-math.log(.6))
    assert report["stages"]["mentions"]["unconditioned_nll"]["value"] == pytest.approx(-math.log(.54))


def test_check_observation_records_acceptance_separately_and_rule_facts_are_not_model_rows():
    check = Judgment.check("It is on.", unstated=True)
    ok = observation({"status": "ok", "value": True, "p_true": .9, "mass": .8, "scores": {"true": .9, "false": .05, "unstated": .05},
                      "presentations": [{"stage": "full", "answers": []}], "observed_models": ["m"]}, check, normalization=OPTION_SOFTMAX, execution="e")
    assert ok["options"] == ["true", "false", "unstated"] and ok["accepted"] == "true" and ok["score"] == .9 and ok["source"] == "judge"
    assert ok["execution"] == "e" and ok["models"] == ["m"] and ok["mass"] == .8 and ok["revision"] == 1
    rule = observation(result({"true": 1., "false": 0.}, [], source="rule", status="none"), Judgment.check("It is on."), normalization=OPTION_SOFTMAX)
    assert micro({"task": "t", "target": "r", "kind": "check", "family": "r", "gold": "true", "observation": rule})["unscored"]["not_judge"] == 1


def published(tmp_path):
    from tests.test_snapshot_packs import published as publish, staged
    publish(tmp_path)
    return staged(tmp_path)


def test_snapshot_rows_store_observations_and_recompute_from_stored_rows_without_scoring(tmp_path, monkeypatch):
    from bench import recompute
    evidence = published(tmp_path)
    scorer = FakeScorer(lambda state, question, width: [.03, .97, 0.][:width])
    output = tmp_path / "out" / "eval.json"
    artifact = snapshots.evaluate("zt411", tmp_path / "pack", evidence, {"base": scorer}, output, baseline="base", margin=.02, kinds=("workflow", "direct"))
    json.dumps(artifact, allow_nan=False)
    report = artifact["probabilities"]
    assert report["state"] == "reported" and set(report["arms"]) == {"base|workflow", "base|direct"}
    assert report["plan"] == artifact["plan"]["id"] and report["exposure"] == artifact["exposure"]["state"] and report["review"] == artifact["review"]["status"]
    rows = [row for result_ in artifact["results"].values() for row in result_["rows"]]
    targets = [value for row in rows for value in row["targets"].values()]
    assert all(value["probability"]["revision"] == 1 for value in targets)
    workflow = report["arms"]["base|workflow"]["micro"]
    composed = sum(row["kind"] == "workflow" for row in rows) * len(snapshots.DOMAINS["zt411"].sets["actions"])
    assert workflow["unscored"]["not_judge"] == composed and workflow["submitted"] == sum(len(row["targets"]) for row in rows if row["kind"] == "workflow")
    stored = next(row["targets"]["tt_media"]["probability"] for row in rows if row["kind"] == "workflow")
    assert stored["normalization"] == OPTION_SOFTMAX and stored["options"] == ["true", "false", "unstated"] and stored["stage"] == "final"
    asked = len(scorer.requests)
    monkeypatch.setattr(FakeScorer, "score", lambda self, requests: pytest.fail("stored-row recomputation scored a request"))
    states = recompute.recompute(output, tmp_path / "out" / "recomputed.json")
    again = files.read_json(tmp_path / "out" / "recomputed.json")
    assert states["probabilities"] == "recomputed" and again["probabilities"] == json.loads(json.dumps(report)) and len(scorer.requests) == asked
    old = files.read_json(output)
    for result_ in old["results"].values():
        for row in result_["rows"]:
            row["targets"] = {t: {k: v for k, v in p.items() if k != "probability"} for t, p in row["targets"].items()}
    files.publish(tmp_path / "out" / "old.json", old)
    states = recompute.recompute(tmp_path / "out" / "old.json", tmp_path / "out" / "old-recomputed.json")
    repaired = files.read_json(tmp_path / "out" / "old-recomputed.json")
    assert states["probabilities"] == "unrecoverable" and repaired["probabilities"]["state"] == "unrecoverable"
    assert {k for k, v in states.items() if v == "recomputed"} and all(repaired["results"][k]["scored"] == v["scored"] for k, v in old["results"].items())


def test_judges_declare_the_normalization_their_distributions_carry():
    from semantic_switch import Runtime
    from semantic_switch.drivers.profiles import CHAT, JEV, LUNA
    from semantic_switch.metrics import COMPLETE

    async def unused(url, body, timeout):
        raise AssertionError("no request is sent")
    assert JEV.normalization in COMPLETE and LUNA.normalization in COMPLETE and CHAT.normalization not in COMPLETE
    assert ss.judge("openrouter:typesafe/jev-1.13", transport=unused).normalization == JEV.normalization
    chat = ss.judge("openrouter:m", codec="chat", transport=unused)
    assert chat.normalization == CHAT.normalization
    lower = micro(entry([.6, .3, .1], "a", normalization=chat.normalization, mass=.9))
    assert lower["unscored"]["normalization"] == 1 and lower["scored"] == 0
    with Runtime(FakeScorer()) as runtime:
        assert runtime.judge().normalization == OPTION_SOFTMAX
