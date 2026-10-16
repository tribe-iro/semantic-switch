from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

from semantic_switch import Runtime, files, judge
from bench import external_primitives as s1mb
from bench.outcomes import work_key
from semantic_testkit.scoring import FakeScorer
from .test_openrouter_judge import Provider

SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "s1mb-sample.json").read_text())["cases"]


def present(case, state):
    evidence = s1mb.state(case)
    return all(text in state for text in s1mb.leaves(evidence if not isinstance(evidence, str) else [evidence]))


def perfect(sample):
    def oracle(state, question, width):
        for row in sample:
            if not present(row["case"], state):
                continue
            for q in row["case"]["questions"]:
                target = row["targets"][q["id"]]
                if q["instructions"] in question and (q["task"] == "noul" or all(o["description"] in question for o in q["options"])):
                    return [target["true"], target["false"]] if q["task"] == "noul" else [target[o["id"]] for o in q["options"]]
        raise RuntimeError("unknown question")
    return oracle


def test_perfect_judge_reproduces_gold_and_marks_unsupported_mappings():
    fake = FakeScorer(perfect(SAMPLE))
    with Runtime(fake) as rt:
        rows = {row["benchmark"]: s1mb.predict(rt, row["case"]) for row in SAMPLE}
    for row in SAMPLE:
        for prediction in rows[row["benchmark"]]:
            question = next(q for q in row["case"]["questions"] if q["id"] == prediction["question_id"])
            if len(question["options"]) > s1mb.MAX_OPTIONS:
                assert prediction["status"] == "unsupported" and prediction["probabilities"] is None
                assert prediction["error"].startswith("unsupported: choice: 77 options")
                continue
            target = row["targets"][prediction["question_id"]]
            assert prediction["error"] is None and list(prediction["probabilities"]) == [o["id"] for o in question["options"]]
            assert prediction["probabilities"] == pytest.approx(target, abs=1e-6)
            assert sum(prediction["probabilities"].values()) == pytest.approx(1, abs=1e-6)
            if prediction["status"] == "ok":
                assert target[prediction["accepted"]] == pytest.approx(max(target.values()))
    assert {p["status"] for preds in rows.values() for p in preds} >= {"ok", "unsupported"}
    noul = next(row for row in SAMPLE if row["task"] == "noul")["case"]["questions"][0]
    assert [(s.id, s.text) for s in s1mb.primitive(noul).guidance.sections] == [("false", noul["options"][0]["description"]),
                                                                              ("true", noul["options"][1]["description"])]


def test_score_order_system_prompt_input_and_backend_failures_stay_distinct():
    row = copy.deepcopy(next(row for row in SAMPLE if row["task"] == "score" and isinstance(row["case"]["state"], dict)))
    question = row["case"]["questions"][0]
    descending = {**question, "id": "descending", "options": question["options"][::-1]}
    prompted = {**question, "id": "prompted", "system_prompt": "Be strict."}
    assert s1mb.unsupported(descending).startswith("score: stored levels are not strictly ascending")
    assert s1mb.unsupported(prompted) == "system prompt has no Check/Pick/Rate field"
    row["case"]["questions"] = [question, descending, prompted]
    missing = copy.deepcopy(row)
    missing["case"]["state"][next(iter(missing["case"]["state"]))] = None
    missing["case"]["questions"] = [question]

    def failing(state, q, width):
        raise RuntimeError("backend down")
    with Runtime(FakeScorer(perfect([row]))) as rt:
        statuses = [p["status"] for p in s1mb.predict(rt, row["case"])]
        absent = s1mb.predict(rt, missing["case"])[0]
    with Runtime(FakeScorer(failing)) as rt:
        failed = s1mb.predict(rt, row["case"])[0]
    assert statuses[1:] == ["unsupported", "unsupported"] and statuses[0] != "unsupported"
    assert absent["status"] == "needs_input" and absent["error"].startswith("needs_input:") and absent["probabilities"] is None
    assert failed["status"] == "failed" and failed["error"] == "failed: backend down" and failed["probabilities"] is None
    assert set(s1mb.FAILURES) == {error.split(":")[0] for error in ("unsupported: x", absent["error"], failed["error"])}


def hosted(rows):
    def answer(body):
        user = body["messages"][1]["content"]
        words = ("yes", "no") if "yes or no" in user else ("true", "false") if "true or false" in user else tuple("ABCD")
        return {word: math.log(p) for word, p in zip(words, next(v for k, v in rows.items() if k in user)) if p}
    return Runtime(judge("openrouter:m", codec="chat", transport=Provider(answer)))


def test_public_scores_map_by_authored_id_and_lower_bounds_stay_unsupported():
    noul, choice, score = (copy.deepcopy(next(row for row in SAMPLE if row["task"] == task)) for task in ("noul", "choice", "score"))
    noul["case"]["questions"] = noul["case"]["questions"][:2]
    first, second = (q["instructions"] for q in noul["case"]["questions"])
    topic, = (q["instructions"] for q in choice["case"]["questions"])
    severity, = (q["instructions"] for q in score["case"]["questions"])
    rows = {first: (.3, .7), second: (.08, .72), topic: (.4, .1, .4, .1), severity: (.5, .2)}
    with hosted(rows) as rt:
        complete, partial = s1mb.predict(rt, noul["case"])
        tied, = s1mb.predict(rt, choice["case"])
        rated, = s1mb.predict(rt, score["case"])
    assert complete["probabilities"] == complete["scores"] == pytest.approx({"false": .7, "true": .3}) and complete["error"] is None
    assert partial["scores"] == pytest.approx({"false": .72, "true": .08}) and partial["probabilities"] is None
    assert partial["error"].startswith("unsupported: public scores sum to 0.800000") and partial["status"] != "unsupported"
    assert s1mb.failure([complete, partial]) == partial["error"] and s1mb.FAILURES[partial["error"].split(":")[0]] is s1mb.UnsupportedMapping
    options = [o["id"] for o in choice["case"]["questions"][0]["options"]]
    assert list(tied["probabilities"]) == options and list(tied["probabilities"].values()) == pytest.approx([.4, .1, .4, .1])
    levels = [o["id"] for o in score["case"]["questions"][0]["options"]]
    assert rated["scores"] == pytest.approx(dict(zip(levels, (.5, .2, 0)))) and rated["probabilities"] is None and rated["error"].startswith("unsupported")


@pytest.mark.parametrize("scores,valid", [
    ({"sports": .25, "business": .25, "sci_tech": .25, "world": .25 + 9e-6}, True),
    ({"sports": .25, "business": .25, "sci_tech": .25, "world": .25 + 2e-5}, False),
    ({"sports": math.nan, "business": .5, "sci_tech": .25, "world": .25}, False),
    ({"sports": 1.5, "business": -.5, "sci_tech": 0, "world": 0}, False),
    ({"sports": .5, "business": .5, "sci_tech": 0}, False),
    ({"sports": .5, "business": .5, "sci_tech": 0, "world": 0, "other": 0}, False),
])
def test_distribution_validates_finite_complete_authored_scores(scores, valid):
    question = next(row for row in SAMPLE if row["task"] == "choice")["case"]["questions"][0]
    found, error = s1mb.distribution(question, SimpleNamespace(scores=scores))
    assert (error is None) == valid and all(math.isfinite(v) for v in found.values()) and set(found) <= set(scores)
    assert found == {key: value for key, value in scores.items() if key in found}


def prediction(qid, accepted, probabilities=None, status="ok", error=None):
    return {"question_id": qid, "status": status, "accepted": accepted, "probabilities": probabilities, "error": error, "scores": probabilities or {}, "mass": None}


def arms():
    p = {"true": .9, "false": .1}
    tie = {"a": .4, "b": .4, "c": .2}
    level = {"option_0": .1, "option_1": .8, "option_2": .1}
    targets = {"n": {"q1": {"false": 0, "true": 1}, "q2": {"false": .4, "true": .6}}, "c": {"q": {"a": 1, "b": 0, "c": 0}},
               "s": {"q1": {"option_0": 1, "option_1": 0, "option_2": 0}, "q2": {"option_0": 0, "option_1": 1, "option_2": 0},
                     "q3": {"option_0": 0, "option_1": 0, "option_2": 1}}}
    cases = {"n": ("bench-noul", "noul"), "c": ("bench-choice", "choice"), "s": ("bench-score", "score"), "x": ("bench-noul", "noul")}
    predictions = {
        "base": {"n": [prediction("q1", "true", p), prediction("q2", "true", p)], "c": [prediction("q", None, tie, "uncertain")],
                 "s": [prediction("q1", "option_1", level), prediction("q2", None, None, "failed", "failed: down"),
                       prediction("q3", None, None, "unsupported", "unsupported: x")], "x": [prediction("q1", "true", p)]},
        "other": {"n": [prediction("q1", "false", p), prediction("q2", None, p, "uncertain")], "c": [prediction("q", "a", tie)],
                  "s": [prediction("q1", "option_0", level), prediction("q2", "option_1", level),
                        prediction("q3", None, None, "unsupported", "unsupported: x")]}}
    from bench.outcomes import work_key
    return {name: {work_key(cases[k][0], k): {"benchmark": cases[k][0], "task": cases[k][1], "case_id": k, "predictions": rows, "ms": 1.0,
                                          "targets": targets.get(k, targets["n"])} for k, rows in arms.items()} for name, arms in predictions.items()}


def test_paired_table_separates_hard_labels_soft_targets_and_unscored_rows():
    rows = arms()
    report = s1mb.paired(rows["base"], rows["other"])
    assert (report["matched_cases"], report["judgments"], report["jointly_scored"]) == (3, 6, 4)
    assert report["unscored"] == {"base_only": 1, "other_only": 0, "both": 1}
    assert report["availability"]["base"] == {"failed": 1, "ok": 3, "uncertain": 1, "unsupported": 1}
    hard = report["table"]["all"]["hard_label"]
    assert (hard["items"], hard["fixes"], hard["regressions"]) == (3, 1, 1)
    assert hard["base"] == {"accepted": 2, "correct": 1, "wrong": 1, "abstain": 1, "selective_precision": .5, "raw_correct": 2}
    assert (hard["other"]["accepted"], hard["other"]["correct"], hard["other"]["wrong"]) == (3, 2, 1)
    assert hard["accepted_transitions"] == {"abstain->correct": 1, "correct->wrong": 1, "wrong->correct": 1}
    assert hard["raw_transitions"] == {"correct->correct": 2, "wrong->wrong": 1}
    soft = report["table"]["all"]["soft_target"]
    assert soft["base"] == {"items": 1, "accepted": 1, "expected_agreement": .6} and soft["other"] == {"items": 1, "accepted": 0, "expected_agreement": None}
    assert report["table"]["score"]["hard_label"]["items"] == 1 and report["pooled_soft_and_hard"]["base"]["accepted"] == 3
    assert report["table"]["in_scope"]["hard_label"]["items"] == 3 and report["table"]["out_of_scope"]["hard_label"]["items"] == 0


def test_paired_matches_questions_by_id_and_rejects_mismatches():
    rows = arms()
    report = s1mb.paired(rows["base"], rows["other"])
    for case in rows["other"].values():
        case["predictions"].reverse()
    assert s1mb.paired(rows["base"], rows["other"]) == report
    from bench.outcomes import work_key
    score = work_key("bench-score", "s")
    scoped = {name: {(work_key("gsm8k", "s") if key == score else key): {**case, "benchmark": "gsm8k"} if key == score else case
                     for key, case in arm.items()} for name, arm in rows.items()}
    table = s1mb.paired(scoped["base"], scoped["other"])["table"]
    assert (table["in_scope"]["hard_label"]["items"], table["out_of_scope"]["hard_label"]["items"]) == (2, 1)
    for change, needle in ((lambda c: c["predictions"].append(c["predictions"][0]), "duplicate question IDs"),
                           (lambda c: c["predictions"][0].update(question_id="q9"), "question IDs"),
                           (lambda c: c.update(targets={**c["targets"], "q1": {"false": 1, "true": 0}}), "targets"),
                           (lambda c: c.update(task="choice"), "task")):
        changed = copy.deepcopy(rows["other"])
        change(changed[work_key("bench-noul", "n")])
        with pytest.raises(ValueError, match=needle):
            s1mb.paired(rows["base"], changed)


def test_compare_is_model_free_and_publishes_one_report(tmp_path, monkeypatch):
    import semantic_switch
    rows = arms()
    for name, entries in rows.items():
        from bench.outcomes import resumable
        with resumable(tmp_path / f"{name}.json", {"args": {"judge": name}, "models": [{"source": name, "native": {}, "sha256": name}],
                                                   "inputs": {"provenance": {"evaluator": "pinned"}, "cases": "digest"}}) as journal:
            journal.commit(list(entries.items()))
    calls = []

    def score(root, command, predictions, info, run_id, output):
        lines = files.read_jsonl(predictions)
        calls.append((command, run_id, [(line["benchmark"], line["case_id"]) for line in lines]))
        metrics = {"bench-noul": ("noul", "binary_brier", .2), "bench-choice": ("choice", "target_mass_at_prediction", None),
                   "bench-score": ("score", "normalized_expected_score_mae", .3)}
        output.write_text(json.dumps({"benchmarks": {name: {"task": task, "primary_metric": metric, "metrics": {metric: value},
                                                            "counts": {"failed": int(name == "bench-score")}, "status": "partial"}
                                                     for name, (task, metric, value) in metrics.items()},
                                      "task_scores": {"noul": None, "choice": None, "score": None}}))

    def refuse(*args, **kwargs):
        raise AssertionError("analysis must not construct models, spawn inference or open sockets")
    monkeypatch.setattr(s1mb, "evaluator", score)
    monkeypatch.setattr(semantic_switch, "judge", refuse)
    monkeypatch.setattr(semantic_switch.Model, "__init__", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    s1mb.compare({"base": tmp_path / "base.json", "other": tmp_path / "other.json"}, tmp_path / "report.json", tmp_path)
    report = files.read_json(tmp_path / "report.json")
    assert report["schema"] == 4 and "first N cases" in report["boundary"] and report["paired"] == {"base->other": s1mb.paired(rows["base"], rows["other"])}
    assert [call[:2] for call in calls] == [("score", "base-base-matched"), ("score", "other-other-matched")]
    assert all(call[2] == [("bench-noul", "n"), ("bench-choice", "c"), ("bench-score", "s")] for call in calls)
    tasks = report["arms"]["base"]["tasks"]
    assert tasks["noul"]["primary_metric_benchmark_mean"] == .2 and tasks["choice"]["primary_metric_benchmark_mean"] is None
    assert tasks["score"]["benchmarks_with_failures"] == 1 and tasks["score"]["statuses"] == {"ok": 1, "failed": 1, "unsupported": 1}
    assert len(files.read_jsonl(tmp_path / "base.json.journal.jsonl")) == 5
    with pytest.raises(semantic_switch.ConfigError, match="already exists"):
        s1mb.compare({"base": tmp_path / "base.json", "other": tmp_path / "other.json"}, tmp_path / "report.json", tmp_path)


def test_audit_transport_comes_from_the_unsent_public_preview():
    with s1mb.previewing() as rt:
        for row in SAMPLE:
            question = row["case"]["questions"][0]
            found = s1mb.reconstruct(rt, row["case"], question)
            if s1mb.unsupported(question):
                assert found["transport"] is None and found["error"]
            else:
                assert found["transport"] == {"instructions": True, "options_in_authored_order": True, "input": True} and question["instructions"] in found["user"]


def three():
    rows = arms()
    rows["third"] = copy.deepcopy(rows["other"])
    third = rows["third"]
    third[work_key("bench-choice", "c")]["predictions"] = [prediction("q", None, None, "failed", "failed: timeout")]
    third[work_key("bench-score", "s")]["predictions"] = [prediction(q, None, None, "unsupported", "unsupported: x") for q in ("q1", "q2", "q3")]
    third[work_key("bench-noul", "n")]["requests"] = [{"id": "r1", "seconds": .09, "attempts": 1, "status": 200, "usage": {"input_tokens": 10}, "models": ["m"]}]
    third[work_key("bench-choice", "c")]["requests"] = [{"id": "r2", "seconds": .9, "attempts": 2, "status": None, "usage": None, "models": []}]
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
    report = s1mb.analyze(recorded)
    assert (report["matched_cases"], report["judgments"], report["jointly_scored"], report["jointly_scored_hard"]) == (3, 6, 2, 1)
    assert report["unscored"] == {"base": 2, "other": 1, "third": 4} and report["availability"]["third"] == {"failed": 1, "ok": 1, "uncertain": 1, "unsupported": 3}
    row = report["hard_label"]["noul/in_scope"]["base"]["curve"]["0.9"]
    assert row == {"accepted": 1, "correct": 1, "wrong": 0, "abstain": 0, "selective_precision": 1.0, "coverage": 1.0, "benchmarks": 1}
    assert report["hard_label"]["noul/in_scope"]["other"]["curve"]["0.999"]["abstain"] == 1 and set(report["paired"]) == {"base->other", "base->third", "other->third"}
    reordered = {name: (header, {key: {**case, "predictions": case["predictions"][::-1]} for key, case in reversed(arm.items())})
                 for name, (header, arm) in recorded.items()}
    assert s1mb.analyze(reordered) == report


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
        s1mb.analyze(recorded)


def test_timing_separates_failed_request_free_and_unrecorded_strata():
    recorded = three()
    timing = s1mb.analyze(recorded)["timing"]
    third = timing["third"]["all"]
    assert (third["cases"], third["case_ms"]["samples"], third["non_failed_case_ms"]["samples"], third["model_work_case_ms"]["samples"]) == (3, 3, 2, 2)
    assert (third["requests"], third["failed_requests"], third["attempts"], third["usage"]) == (2, 1, 3, {"input_tokens": 10})
    assert (third["request_ms"]["samples"], third["request_ms"]["p50"], third["request_ms"]["p95"]) == (2, pytest.approx(495.0), pytest.approx(859.5))
    assert third["returned_models"] == {"m": 1} and third["case_ms"]["unit"] == "ms"
    assert timing["third"]["question_count"]["3"]["model_work_case_ms"]["samples"] == 0 and timing["base"]["all"]["model_work_case_ms"]["samples"] == 3
    assert "requests" not in timing["base"]["all"] and timing["third"]["input_characters_from"] is None
    for name, (_, arm) in recorded.items():
        for size, case in zip((10, 2000, 50000), arm.values()):
            case["input_characters"] = size
    strata = s1mb.analyze(recorded)["timing"]["base"]["input_characters_from"]
    assert {key: value["cases"] for key, value in strata.items()} == {"0": 1, "1000": 1, "16000": 1}
