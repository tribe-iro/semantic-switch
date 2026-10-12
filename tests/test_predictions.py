from __future__ import annotations

import json

import numpy as np
import pytest

import semantic_switch as ss
from semantic_switch import ConfigError, Guidance, Runtime
from semantic_switch.judge import Scored, Scores
from semantic_switch.metrics import correctness
from semantic_switch.tools.evaluate import contrast_pairs, score_observations

from .conftest import FakeScorer
from .test_classification import LABELS, member, oracle, saved

OPTIONS = {"ax": "AX part", "by": "BY part", "cz": "CZ part"}


def fixed(p):
    return lambda state, question, width: p[:width] if len(p) >= width else [1 / width] * width


def run(p, call):
    fake = FakeScorer(fixed(p) if isinstance(p, list) else p)
    with Runtime({"judge": fake}) as rt:
        return call(rt), fake


def test_uncertain_checks_keep_correct_and_wrong_predictions():
    r, _ = run([.7, .3], lambda rt: rt.check("x", "The customer asks for a refund."))
    assert (r.status, r.value) == ("uncertain", None)
    assert (r.prediction.kind, r.prediction.key, r.prediction.value) == ("value", "true", True) and r.prediction.score == pytest.approx(.7)
    r, _ = run([.2, .8], lambda rt: rt.check("x", "The customer asks for a refund.", min_score=.9))
    assert r.status == "uncertain" and r.prediction.value is False


def test_none_unstated_and_object_identity():
    r, _ = run([.02, .02, .02, .94], lambda rt: rt.pick("x", "Which part?", OPTIONS, allow_none=True))
    assert r.status == "none" and (r.prediction.kind, r.prediction.key, r.prediction.value) == ("none", None, None)
    r, _ = run([.1, .2, .7], lambda rt: rt.check("x", "The customer asks for a refund.", unstated=True, min_score=.95))
    assert r.status == "uncertain" and (r.prediction.kind, r.prediction.value) == ("unstated", None)
    parts = [object(), object(), object()]
    r, _ = run([.2, .6, .2], lambda rt: rt.pick("x", "Which part?", parts, definition=lambda o: f"part {parts.index(o)}"))
    assert r.status == "uncertain" and r.prediction.value is parts[1] and r.prediction.key == 1


def test_ties_resolve_in_declaration_order_after_slot_mapping():
    r, _ = run([.4, .4, .2], lambda rt: rt.pick("x", "Which part?", OPTIONS))
    assert r.prediction.key == "ax"
    r, _ = run([.45, .45, .1], lambda rt: rt.rate("x", "How urgent?", ["low", "mid", "high"]))
    assert r.prediction.key == "low"


def test_escaped_shortlist_predicts_from_the_full_list():
    def answers(state, question, width):
        return [.1, .9] if "another one" in question else [.1, .7, .2]
    r, fake = run(answers, lambda rt: rt.pick("ax failed", "Which part?", OPTIONS, focus="mentions", ids=None))
    assert r.focus == "fallback" and len(fake.requests) == 2 and r.prediction.key == "by" and r.status == "uncertain"
    assert set(r.scores) == {"ax", "by", "cz"}


def test_failures_have_no_prediction_and_json_events_carry_it():
    def broken(state, question, width):
        raise RuntimeError("backend exploded")
    r, _ = run(broken, lambda rt: rt.pick("x", "Which part?", OPTIONS))
    assert r.status == "failed" and r.prediction is None and r.as_dict()["prediction"] is None
    r, _ = run([.1, .9], lambda rt: rt.check({"a": ""}, "The customer asks for a refund.", fields=["a"]))
    assert r.status == "needs_input" and r.prediction is None
    events = []
    with Runtime({"judge": FakeScorer(fixed([.6, .4]))}, on_decision=events.append) as rt:
        r = rt.ask("x", {"c": ss.Check("The customer asks for a refund.")})["c"]
    record = json.loads(json.dumps(r.as_dict()))
    assert record["prediction"] == {"kind": "value", "key": "true", "value": True, "score": pytest.approx(.6)}
    assert events[0].schema == 5 and events[0].as_dict()["facts"]["c"]["prediction"]["value"] is True


def test_threshold_only_changes_issue_identical_requests_and_predictions():
    seen = []
    for theta in (.6, .99):
        r, fake = run([.1, .7, .2], lambda rt: rt.pick("ax failed", "Which part?", OPTIONS, focus="mentions", min_score=theta))
        seen.append((r.status, r.prediction, r.scores, fake.requests))
    (first, guess, scores, asked), (second, again, rescored, reasked) = seen
    assert (first, second) == ("ok", "uncertain") and guess == again and scores == rescored and asked == reasked


def test_saved_facts_and_select_members_expose_predictions(tmp_path):
    path = saved(tmp_path, {"c": {"check": "The customer asks for a refund."}, "issues": {"select": "Which issues?", "labels": LABELS}})
    with Runtime({"judge": FakeScorer(lambda s, q, w: oracle(s, q, w) if member(q) else [.7, .3])}) as rt:
        result = rt.load(path)({"message": "refund"})
        inline = rt.check("refund", "The customer asks for a refund.")
    fact, issues = result.facts["c"], result.facts["issues"]
    assert fact.status == "uncertain" and fact.prediction == inline.prediction and fact.as_dict()["prediction"]["value"] is True
    assert issues.prediction is None and issues.members["refund"].prediction.value is True
    assert issues.members["delivery"].prediction.value is False


class LowerBounds:
    identity = {"source": "lower-bounds"}
    max_slots = 20

    def __init__(self, p, mass):
        self.p, self.mass = p, mass

    def score_many(self, items):
        return [Scored(tuple(Scores(np.array(self.p), self.mass) for _ in questions), 1) for _, questions in items]


def test_rate_expected_requires_a_complete_distribution():
    levels = ["low", "mid", "high"]
    with Runtime({"judge": LowerBounds([.1, .2, .7], 1.)}) as rt:
        assert rt.rate("x", "How urgent?", levels).expected == pytest.approx(1.6)
    with Runtime({"judge": LowerBounds([.1, .2, .5], .8)}) as rt:
        partial = rt.rate("x", "How urgent?", levels)
    assert partial.expected is None and partial.prediction.key == "high" and partial.scores["high"] == pytest.approx(.5)


def test_guided_select_members_record_guidance_even_on_failure():
    def answers(state, question, width):
        if member(question) == "delivery":
            raise RuntimeError("backend exploded")
        return oracle(state, question, width)
    guidance = Guidance("Refunds cover duplicate charges.")
    r, _ = run(answers, lambda rt: rt.select("refund", "Which issues?", LABELS, guidance=guidance))
    assert r.members["delivery"].status == "failed" and r.members["delivery"].prediction is None
    assert all(m.guidance == r.guidance and m.guidance for m in r.members.values())


def test_correctness_counts_rejected_rankings_and_failures():
    report = correctness([(True, False), (True, True), (False, True), (None, False), (False, False)])
    assert (report["submitted"], report["predicted"], report["raw_correct"], report["raw_wrong"]) == (5, 4, 2, 2)
    assert (report["accepted"], report["correct_accepted"], report["correct_rejected"], report["wrong_rejected"]) == (2, 1, 1, 1)
    assert report["correct_accepted_rate"] == .2 and report["accuracy_given_prediction"] == .5 and report["correct_per_submission"] == .4
    empty = correctness([])
    assert empty["accuracy_given_prediction"] is None and empty["coverage"] is None and empty["precision"] is None
    with pytest.raises(ConfigError, match="needs a prediction"):
        correctness([(None, True)])


def two_members(tmp_path):
    path = saved(tmp_path, {"issues": {"select": "Which issues?", "labels": {"refund": LABELS["refund"], "billing": LABELS["billing"]}}})
    return path


def rows_for(switch, inputs):
    rows = []
    for n, (text, key, label) in enumerate(inputs):
        fact = switch.plan.at("issues", key)
        rows.append({"id": f"o{n}", "fact": "issues", "member": key, "meaning_rev": fact.meaning_rev, "label": label,
                     "bound_input": {"message": text}, "pairs": ["p"], "classes": list(fact.keys)})
    return rows


def test_addressed_member_rows_score_only_their_member(tmp_path):
    path = two_members(tmp_path)
    fake = FakeScorer(oracle)
    with Runtime({"judge": fake}) as rt:
        switch = rt.load(path, draft=True)
        rows = rows_for(switch, [("refund", "refund", 0), ("refund", "billing", 0), ("invoice", "billing", 0), ("hello", "refund", 1)])
        scored = score_observations(switch, rows)
        assert len(fake.requests) == 4 and all(member(q) in ("refund", "billing") for _, q in fake.requests)
        assert [(r["id"], r["member"], r["correct"]) for r in scored] == [("o0", "refund", True), ("o1", "billing", False),
                                                                         ("o2", "billing", True), ("o3", "refund", True)]
        assert scored[1]["result"]["prediction"]["value"] is False and scored[1]["accepted"]
        again = score_observations(switch, list(reversed(rows)))
    assert [r["id"] for r in again] == ["o3", "o2", "o1", "o0"] and {r["id"]: r for r in again} == {r["id"]: r for r in scored}
    pairs = contrast_pairs(rows)
    assert pairs == {("p", "issues", "refund"): ["o0", "o3"], ("p", "issues", "billing"): ["o1", "o2"]}


def test_addressed_rows_fail_before_scoring(tmp_path):
    path = two_members(tmp_path)
    fake = FakeScorer(oracle)
    with Runtime({"judge": fake}) as rt:
        switch = rt.load(path, draft=True)
        row = rows_for(switch, [("refund", "refund", 0)])[0]
        hidden = switch.plan.at("issues", "refund").name
        for bad, match in (([row, row], "unique ids"), ([{**row, "id": None}], "unique ids"),
                           ([{**row, "meaning_rev": "old"}], "changed meaning since export"),
                           ([{**row, "member": "delivery"}], "not a label of issues"),
                           ([{**row, "fact": "nope"}], "unknown fact"),
                           ([{**row, "fact": hidden}], "not a visible fact"),
                           ([{**row, "label": 5}], "label must index")):
            with pytest.raises(ConfigError, match=match):
                score_observations(switch, bad)
        for names, match in ((["nope"], "unknown fact"), ([hidden], "not a visible fact"), ("issues", "sequence")):
            with pytest.raises(ConfigError, match=match):
                switch.score_facts([{"message": "refund"}], facts=names)
    assert not fake.requests


def test_lock_receipt_lists_every_target_state(tmp_path):
    from .test_artifact_pipeline import dataset, definition
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": FakeScorer()}) as rt:
        receipt = ss.tools.lock(rt, path)
    targets = receipt["targets"]
    assert set(targets) == {"met", "missed", "insufficient", "absent"} and any(targets.values()) and not targets["absent"]
    for state in ("missed", "insufficient"):
        for entry in targets[state]:
            name, rest = entry.split(": ")
            assert any(warning.startswith(f"{name}: {rest} side") for warning in receipt["warnings"])
    assert any(entry.split(": ")[1].startswith("judge") for entries in targets.values() for entry in entries)


def test_prediction_example_dispatches_directly_or_on_acceptance():
    import ast
    import importlib
    from pathlib import Path
    example = importlib.import_module("examples.predictions")
    source = Path(example.__file__).read_text()
    assert "plan" not in source and "~" not in source and not any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("bench") for node in ast.walk(ast.parse(source)))

    def answers(state, question, width):
        if "explicitly approves" in question:
            return [.1, .9]
        return [.8, .2] if width == 2 else [.6] + [.4 / (width - 1)] * (width - 1)
    with Runtime({"judge": FakeScorer(answers)}) as rt:
        out = example.run(rt)
    direct, routed = out["direct"], out["routed"]
    assert direct["status"] == "uncertain" and direct["accepted"] is None and direct["direct"] == "refund"
    assert routed["status"] == "uncertain" and routed["facts"]["approved"]["value"] is False
    assert routed["facts"]["duplicate"]["status"] == "uncertain" and routed["facts"]["duplicate"]["prediction"] is True
