from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.compiler import digest as identity
from bench import evidence, procedures, support


def message(statements, account=None, frame_id="a"):
    frame = support.render("s", statements, account or {}, frame_id=frame_id, family="s", order="A10000")
    evidence.validate_frame(frame, support.FIELDS)
    return frame


def twin(frame, frame_id="b", **changes):
    return replace(frame, id=frame_id, assertions=tuple(replace(a, **changes) for a in frame.assertions))


def golds(reference, *frames):
    return {frame.id: reference(frame) for frame in frames}


def printer(variant, template, condition="structured"):
    rows = [row for row in procedures.scenarios(0, 1) if row["template"] == template and row["variant"] == variant]
    (frame,) = procedures.frames(rows, (condition,), ("en",))
    return frame


def test_observable_key_is_input_under_the_effective_criterion_without_annotations():
    spec = files.read_yaml(procedures.DEFINITION)
    actions = spec["facts"]["actions"]
    by_hand = identity({**spec, "facts": {**spec["facts"], "actions": {**actions, "cards": sorted(actions["cards"], key=lambda card: card["id"])}}})
    ref = procedures.standard()
    assert ref.criteria["structured"] == ref.criteria["structured_permuted"] == by_hand != ref.criteria["prose"]
    frame = printer("base", "ribbon_out")
    assert ref.problem(frame) == identity({"input": frame.input, "criterion": by_hand, "revision": procedures.REVISION, "scope": list(procedures.SCOPE)})
    hidden = twin(frame, status="stale")
    assert hidden.content != frame.content and ref.problem(hidden) == ref.problem(frame)
    assert ref.problem(replace(frame, condition="structured_permuted")) == ref.problem(frame)
    labels = support.definition("message_permuted")["facts"]["issues"]["labels"]
    assert list(labels) != list(support.ISSUES) and support.standard().criteria["message"] == support.standard().criteria["message_permuted"]
    with pytest.raises(ConfigError, match="no effective criterion"):
        ref.problem(replace(frame, condition="unknown"))


def test_opposing_hidden_status_on_identical_input_is_a_conflict_with_answer_addresses():
    current = message([{"property": "request:refund_initiate", "status": "current"}])
    negated = twin(current, status="negated")
    for frame in (current, negated):
        evidence.validate_frame(frame, support.FIELDS)
    assert current.input == negated.input == {"order": "A10000", "message": "Please refund me."}
    ref = support.standard()
    found = evidence.conflicts(ref, [current, negated], golds(support.reference, current, negated))
    assert found == [("b", "a", ["composition", "requests/refund_initiate"])]
    with pytest.raises(ConfigError, match=r"b and a: one observable problem .* at \['composition', 'requests/refund_initiate'\]"):
        support.build([current, negated])


def test_same_semantic_answer_with_different_attribution_is_one_problem():
    resolved = message([{"property": "issue:size_mismatch", "status": "resolved"}])
    negated = twin(resolved, status="negated", id="x0")
    left, right = support.reference(resolved), support.reference(negated)
    assert left.decisions["issues/size_mismatch"].rules == ("S-RESOLVED",) and right.decisions["issues/size_mismatch"].rules == ("S-NEGATED",)
    assert left.decisions["issues/size_mismatch"].evidence == ("s0",) and right.decisions["issues/size_mismatch"].evidence == ("x0",)
    assert left != right and evidence.answer(left) == evidence.answer(right)
    assert evidence.answer(left)["issues/size_mismatch"] == {"value": False, "scope": [], "unresolved": []}
    assert evidence.invariant(support.standard(), [resolved, negated], golds(support.reference, resolved, negated)) == {"observable_problems": 1, "frames": 2}


def test_unknown_versus_false_and_composed_response_differ():
    ambiguous = message([{"property": "request:replacement", "status": "ambiguous"}])
    negated = twin(ambiguous, status="negated")
    left, right = support.reference(ambiguous), support.reference(negated)
    assert left.values()["requests/replacement"] is None and right.values()["requests/replacement"] is False
    assert support.response(ambiguous, left).value == "clarify" and support.response(negated, right).value == "no_action"
    assert evidence.conflicts(support.standard(), [ambiguous, negated], {"a": left, "b": right}) == [("b", "a", ["composition", "requests/replacement"])]
    assert evidence.conflicts(replace(support.standard(), compose=None), [ambiguous, negated], {"a": left, "b": right}) == [("b", "a", ["requests/replacement"])]


def test_prohibited_actions_are_part_of_the_semantic_answer():
    frame = printer("base", "ribbon_out")
    reading = next(a for a in frame.assertions if a.indication == "reading")
    assert reading.paths == ("alerts",) and frame.input["alerts"] == ["08:02 RIBBON OUT"]
    stale = replace(frame, id="stale", assertions=tuple(replace(a, status="stale") if a is reading else a for a in frame.assertions))
    evidence.validate_frame(stale, procedures.FIELDS)
    left, right = procedures.reference(frame), procedures.reference(stale)
    assert left.prohibited == ("resume_printing",) and right.prohibited == ()
    ((_, _, addresses),) = evidence.conflicts(procedures.standard(), [frame, stale], {frame.id: left, "stale": right})
    assert {"prohibited", "actions/resume_printing", "actions/load_ribbon", "ribbon_absent"} <= set(addresses)


def test_menu_permutation_cannot_change_gold_but_a_changed_criterion_is_a_different_problem():
    frame = printer("base", "detection")
    permuted = replace(frame, id="permuted", condition="structured_permuted")
    gold = procedures.reference(frame)
    flipped = replace(gold, decisions={**gold.decisions, "actions/verify_loading": replace(gold.decisions["actions/verify_loading"], value=False)})
    ref = procedures.standard()
    assert evidence.conflicts(ref, [frame, permuted], {frame.id: gold, "permuted": flipped}) == [("permuted", frame.id, ["actions/verify_loading"])]
    spec = procedures.definition("structured")
    edited = {**spec, "facts": {**spec["facts"], "unverified_step": {"check": "A step was attempted; a report alone counts as verification."}}}
    counterfactual = replace(ref, criteria={**ref.criteria, "structured_permuted": evidence.criterion(edited)})
    assert counterfactual.problem(permuted) != counterfactual.problem(frame)
    assert evidence.invariant(counterfactual, [frame, permuted], {frame.id: gold, "permuted": flipped}) == {"observable_problems": 2, "frames": 2}


def test_generated_packs_have_one_answer_per_observable_problem():
    built, golds_ = procedures.build(procedures.scenarios(0, 2))
    counted = evidence.invariant(procedures.standard(), built, golds_)
    structured = {(f.scenario, f.id.split(":")[1]) for f in built}
    assert counted["frames"] == len(built) == 6 * len(structured)
    assert counted["observable_problems"] == len({procedures.standard().problem(f) for f in built}) <= 5 * len(structured)
    frames = support.frames(support.scenarios(0, 4))
    assert evidence.invariant(support.standard(), frames, support.build(frames))["observable_problems"] == len(frames) // 2


def test_support_composition_traces_cite_spans_paths_and_refund_boundaries():
    def traced(statements, account):
        frame = message(statements, account)
        gold = support.reference(frame)
        composed = support.response(frame, gold)
        spans = {a.id: a.spans for a in frame.assertions}
        assert all(spans[i] and all(frame.input[s.field][s.start:s.end] for s in spans[i]) for i in composed.evidence)
        assert all(evidence.lookup(frame.input, p) is not None for p in composed.paths)
        return composed
    refund = [{"property": "issue:stain_damage", "status": "current"}, {"property": "request:refund_initiate", "status": "current"}]
    edge = traced(refund, {"delivered_days_ago": 30, "amount": 500.0, "refund_blocked": False})
    assert (edge.value, edge.branch, edge.paths) == ("initiate_refund", "initiate_refund", ("refund_blocked", "delivered_days_ago", "amount"))
    assert edge.rules == ("S-RESPONSE", "S-REFUND", "S-CURRENT", "S-ABSENT") and edge.evidence == ("s1", "s0")
    assert edge.scope == ("requests/refund_initiate", "issues/missing_delivery", "issues/stain_damage")
    late = traced(refund, {"delivered_days_ago": 31, "amount": 500.01, "refund_blocked": False})
    assert (late.value, late.branch, late.paths) == ("refund_ineligible", "refund_ineligible:delivered_days_ago+amount", ("delivered_days_ago", "amount"))
    blocked = traced(refund, {"refund_blocked": True})
    assert blocked.branch == "refund_ineligible:refund_blocked" and blocked.paths == ("refund_blocked",)
    missing = traced([{"property": "issue:missing_delivery", "status": "current"}, {"property": "request:refund_initiate", "status": "current"}],
                     {"delivered_days_ago": 45, "amount": 40.0})
    assert missing.value == "initiate_refund" and "delivered_days_ago" in missing.paths
    conflicting = traced([{"property": "request:return_item", "status": "current"}, {"property": "request:return_item", "status": "negated"}], {})
    assert (conflicting.value, conflicting.branch, conflicting.unresolved) == ("clarify", "clarify", ("requests/return_item",))
    assert "S-CONFLICT" in conflicting.rules and conflicting.evidence == ("s0", "s1")
    superseded = traced([{"property": "request:return_item", "status": "superseded"}, {"property": "request:replacement", "status": "superseded_by"}], {})
    assert superseded.branch == "return_or_replace:replacement" and {"S-LATEST", "S-CURRENT"} <= set(superseded.rules) and superseded.evidence == ("s0", "s1")
    quoted = traced([{"property": "request:return_item", "status": "hypothetical"}], {})
    assert quoted.value == "no_action" and "S-QUOTED" in quoted.rules and quoted.evidence == ("s0",)
    other = traced([{"property": "request:return_item", "status": "current", "other": "A20000"}], {})
    assert other.value == "no_action" and "S-ENTITY" in other.rules and other.evidence == ()
    absent = traced([], {})
    assert absent.value == "no_action" and absent.evidence == () and absent.paths == () and set(absent.rules) == {"S-RESPONSE", "S-ABSENT"}
    assert absent.scope == tuple(f"requests/{k}" for k in support.REQUESTS)
