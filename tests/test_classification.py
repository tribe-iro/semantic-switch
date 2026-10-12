from __future__ import annotations

import json
import re

import numpy as np
import pytest

import semantic_switch as ss
from semantic_switch import ConfigError, Label, Pick, Rate, Runtime, Select, files, tools
from semantic_switch.artifacts import load_receipt, read_export
from semantic_switch.heads import LexicalHead
from semantic_switch.judgment import accept
from semantic_switch.tools.data import Labels
from semantic_switch.tools.evaluate import confusion, distances

from .conftest import FakeScorer
from .conftest import envelopes

LABELS = {"refund": "Money back for a charge", "delivery": {"description": "Shipping problems", "excludes": ["damaged items"]},
          "billing": "Billing"}
TOKENS = {"refund": "refund", "delivery": "late", "billing": "invoice"}


def member(question):
    found = re.search(r"^Candidate: (\S+) — ", question, re.M)
    return found and found.group(1)


def oracle(state, question, width, scale=.97):
    key = member(question)
    if key is None:
        return [1 / width] * width
    if "boom" in state and key == "delivery":
        raise RuntimeError("backend exploded")
    if width == 3:
        return [.01, .01, .98] if "silent" in state else [.97, .02, .01] if TOKENS.get(key, key) in state else [.02, .97, .01]
    if "maybe" in state:
        return [.5, .5]
    return [scale, 1 - scale] if TOKENS.get(key, key) in state else [1 - scale, scale]


def judge(fn=oracle):
    fake = FakeScorer(fn)
    return fake, Runtime({"judge": fake})


def questions(fake):
    return [q.split("<|im_end|>")[0] for _, q in fake.requests]


def saved(tmp_path, facts, name="issues"):
    path = tmp_path / f"{name}.yaml"
    files.publish(path, {"name": name, "fields": {"message": {"type": "text", "required": True}, "vip": {"type": "bool"}},
                         "reads": ["message"], "facts": facts})
    return path


def test_membership_template_matches_inline_and_saved_and_follows_the_instruction(tmp_path):
    fake, rt = judge()
    rt.select("refund please", "Which issues does the customer raise?", LABELS)
    inline = questions(fake)
    rt.select("refund please", "Which teams must act?", LABELS)
    assert questions(fake)[3:] != inline
    assert inline[1] == ("Question: Which issues does the customer raise?\nCandidate: delivery — Shipping problems. Except: damaged items.\n"
                         "Does this candidate apply to the input? Answer with only yes or no.")
    fake.requests.clear()
    sw = rt.load(saved(tmp_path, {"issues": {"select": "Which issues does the customer raise?", "labels": LABELS}}))
    sw({"message": "refund please"})
    assert questions(fake) == inline
    assert not any(q.startswith(("Claim about", "Question about")) or "Billing?" in q for q in inline)


def test_unstated_membership_uses_three_way_answers():
    fake, rt = judge()
    result = rt.select("silent", "Which issues?", {"refund": "Money back"}, unstated=True)
    assert result.status == "needs_input" and result.members["refund"].error == "not stated in the input"
    assert questions(fake)[0].endswith("A) yes\nB) no\nC) the input does not say\nAnswer with only the letter (A, B, C) of the best option.")


def test_per_label_exclusions_survive_pick_and_select(tmp_path):
    fake, rt = judge(lambda state, question, width: [1 / width] * width)
    rt.pick("x", "Which issue?", LABELS)
    assert "B) Shipping problems. Except: damaged items." in questions(fake)[0]
    rt.select("x", "Which issues?", LABELS, excludes=["test orders"])
    assert "Candidate: delivery — Shipping problems. Except: damaged items; test orders." in questions(fake)[2]
    pick = Pick("Which issue?", LABELS).save(tmp_path / "route.yaml", rt=rt)
    assert pick.plan.fact("route").descriptions[1] == "Shipping problems. Except: damaged items."


def test_definitions_are_immutable_validated_and_loaded_from_yaml(tmp_path):
    given = {"refund": {"description": "Money back", "aliases": ["chargeback"]}}
    select = Select("Which issues?", given)
    given["refund"]["aliases"].append("changed")
    given["other"] = "Other"
    assert select.labels == {"refund": Label("Money back", (), ("chargeback",))} and select.compiled[0][0].candidate == "refund — Money back"
    for bad in ({}, {"a": ""}, {"a": {"description": "x", "requires": "vip"}}, {"a": {"aliases": ["x"]}}, {"": "x"}, {"a": {"description": "x", "excludes": "y"}}):
        with pytest.raises(ConfigError):
            Select("Which?", bad)
    with pytest.raises(ConfigError):
        Select("Which?", LABELS, max_claims=0)
    files.publish(tmp_path / "labels.yaml", LABELS)
    loaded = ss.labels(tmp_path / "labels.yaml")
    assert loaded["delivery"] == Label("Shipping problems", ("damaged items",)) and Pick("Which?", loaded).options == loaded
    with pytest.raises(ConfigError, match="exactly one candidate source"):
        saved(tmp_path, {"issues": {"select": "Which?", "labels": LABELS, "cards": [{"id": "a", "applies_when": ["x"]}]}})
        Runtime({"judge": FakeScorer()}).load(tmp_path / "issues.yaml")
    for fact in ({"select": "Which?", "labels": LABELS, "exclusive": True}, {"select": "Which?", "cards": [{"id": "a", "applies_when": ["x"]}], "unstated": True}):
        with pytest.raises(ConfigError, match="does not apply"):
            Runtime({"judge": FakeScorer()}).load(saved(tmp_path, {"issues": fact}))


def test_saved_labels_file_and_requires(tmp_path):
    files.publish(tmp_path / "issues.labels.yaml", {"refund": "Money back", "delivery": {"description": "Shipping", "requires": "vip"}})
    fake, rt = judge()
    sw = rt.load(saved(tmp_path, {"issues": {"select": "Which issues?", "labels": "issues.labels.yaml"}}))
    fact = sw({"message": "refund late", "vip": False}).facts["issues"]
    assert (fact.value, fact.states["delivery"], fact.status) == (("refund",), "ineligible", "ok") and len(fake.requests) == 1


def test_mixed_ask_preserves_order_rounds_and_partial_failure():
    calls = []
    fake, rt = judge()
    original = rt.judge
    rt.judge = lambda role="judge": original(role)
    score_many = original().score_many
    original().score_many = lambda items: calls.append([len(qs) for _, qs in items]) or score_many(items)
    items = {"route": Pick("Which machine?", {"refund": {"description": "Money", "aliases": ["MX"]}, "billing": "Billing"}, focus="mentions"),
             "issues": Select("Which issues?", LABELS), "urgency": Rate("How urgent?", ("low", "high"))}
    rows = rt.ask_many(["MX refund boom", "refund late"], items)
    assert [list(row) for row in rows] == [["route", "issues", "urgency"]] * 2
    failed = rows[0]["issues"]
    assert failed.status == "failed" and failed.value == ("refund",) and failed.members["delivery"].status == "failed"
    assert failed.members["billing"].status == "ok" and rows[0]["urgency"].status == "uncertain"
    assert rows[1]["issues"].value == ("refund", "delivery") and rows[1]["issues"].status == "ok"
    assert calls == [[5], [5]] and rows[0]["route"].focus == rows[1]["route"].focus == "mentions"
    assert len(fake.requests) == 10 and rows[0]["route"].prediction.key == "refund"


def test_simple_labels_examine_everything_and_cards_spend_a_bounded_budget(tmp_path):
    many = {f"l{k}": f"Label number {k}" for k in range(20)}
    fake, rt = judge(lambda state, question, width: [.97, .03] if member(question) == "l3" else [.03, .97])
    result = rt.select("text", "Which labels?", many)
    assert (result.status, result.value, result.complete, len(fake.requests)) == ("ok", ("l3",), True, 20)
    path = saved(tmp_path, {"issues": {"select": "Which labels?", "labels": many}})
    assert rt.load(path)({"message": "text"}).facts["issues"].value == ("l3",) and len(fake.requests) == 40
    bounded = rt.select("text", "Which labels?", many, max_claims=4)
    assert bounded.status == "uncertain" and not bounded.complete and list(bounded.states.values()).count("unexamined") == 16
    cards = [{"id": f"c{k}", "applies_when": [f"Claim {k}.", "Shared claim."]} for k in range(30)]
    fake, rt = judge()
    sw = rt.load(saved(tmp_path, {"procs": {"select": "Which procedures?", "cards": cards}}, "procs"))
    fact = sw({"message": "text"}).facts["procs"]
    assert len(fake.requests) == 16 and not fact.complete and list(fact.states.values()).count("unexamined") == 15


def test_large_catalogues(tmp_path):
    fake, rt = judge()
    exact = [{"id": f"c{k}", "requires": "vip", "applies_when": [f"Claim {k}."]} for k in range(1000)]
    result = rt.load(saved(tmp_path, {"procs": {"select": "Which?", "cards": exact}}, "exact"))({"message": "x", "vip": False})
    assert result.status == "none" and result.cost["judge"] == 0 and not fake.requests
    semantic = [{"id": f"c{k}", "applies_when": [f"Claim {k}."]} for k in range(1000)]
    fact = rt.load(saved(tmp_path, {"procs": {"select": "Which?", "cards": semantic}}, "semantic"))({"message": "x"}).facts["procs"]
    assert len(fake.requests) == 16 and not fact.complete


def test_all_ineligible_select_never_loads_a_judge():
    class Exploding(Runtime):
        def judge(self, role="judge"):
            raise AssertionError("judge loaded")
    result = Exploding().select("x", "Which?", LABELS, eligible=lambda key: False)
    assert (result.status, result.value, result.complete, result.settled) == ("none", (), True, True)
    assert set(result.states.values()) == {"ineligible"}


def test_partial_search_agrees_with_saved_execution(tmp_path):
    fake, rt = judge()
    inline = rt.select("refund invoice", "Which issues?", LABELS, max_claims=2)
    sw = rt.load(saved(tmp_path, {"issues": {"select": "Which issues?", "labels": LABELS, "max_claims": 2}}))
    fact = sw({"message": "refund invoice"}).facts["issues"]
    assert (inline.value, inline.complete, dict(inline.states)) == (fact.value, fact.complete, dict(fact.states))
    assert inline.as_dict()["states"] == fact.as_dict()["states"] and inline.as_dict()["value"] == fact.as_dict()["value"]


@pytest.mark.parametrize("text,status,value,complete,settled", [
    ("refund late", "ok", ["refund", "delivery"], True, True),
    ("nothing", "none", [], True, True),
    ("maybe", "uncertain", [], True, False),
    ("refund boom", "failed", ["refund"], True, False),
])
def test_selection_outcomes_round_trip(tmp_path, text, status, value, complete, settled):
    _, rt = judge()
    inline = rt.select(text, "Which issues?", LABELS)
    fact = rt.load(saved(tmp_path, {"issues": {"select": "Which issues?", "labels": LABELS}}))({"message": text}).facts["issues"]
    for record in (json.loads(json.dumps(inline.as_dict())), json.loads(json.dumps(fact.as_dict()))):
        assert (record["status"], record["value"], record["complete"], record["settled"]) == (status, value, complete, settled)
    with pytest.raises(TypeError):
        bool(inline)


def test_budget_incomplete_and_events_record_members():
    events = []
    with Runtime({"judge": FakeScorer(oracle)}, on_decision=events.append) as rt:
        result = rt.ask("refund", {"issues": Select("Which issues?", LABELS, max_claims=1), "legal": ss.Check("Legal action.")})
    assert result["issues"].value == ("refund",) and not result["issues"].complete and result["issues"].status == "uncertain"
    assert result["issues"].unresolved == {"delivery": "not examined (over max_claims)", "billing": "not examined (over max_claims)"}
    record = events[0].as_dict()
    assert events[0].schema == 5 and record["facts"]["issues"]["score"] is None
    assert set(record["facts"]["issues"]["members"]) == {"refund"} and record["facts"]["issues"]["members"]["refund"]["score"] > .9


def lock_dataset(path, n=120):
    files.append(path.with_suffix(".test.jsonl"), envelopes([
        {"input": {"message": f'{"refund" if i % 2 else "hello"} item{i}'},
         "facts": {"issues": {"refund": bool(i % 2)} | ({"delivery": False} if i % 3 == 0 else {})}, "group": f"family{i}"}
        for i in range(n)]))


def test_member_fitting_binds_back_to_deployed_acceptance(tmp_path):
    path = saved(tmp_path, {"issues": {"select": "Which issues?", "labels": {"refund": "Money back", "delivery": "Shipping"}}})
    lock_dataset(path)
    fake, rt = judge(lambda state, question, width: oracle(state, question, width, .8))
    labels = Labels(rt.load(path, draft=True))
    plan = labels.sw.plan
    refund, delivery = plan.at("issues", "refund"), plan.at("issues", "delivery")
    assert len(labels.examples(refund.name)) == 120 and len(labels.examples(delivery.name)) == 40
    report = tools.lock(rt, path)
    theta = report["policies"]["issues"]["members"]["refund"]["judge"]["true"]["theta"]
    assert theta is not None and theta < .9
    snap = rt.load(path).snapshot
    fact = rt.load(path)({"message": "refund now"}).facts["issues"]
    assert fact.value == ("refund",) and fact.members["refund"].settled
    assert accept(refund.judgment, list(fact.members["refund"].scores.values()), snap.policy("judge", refund))[0] == 0
    assert rt.select("refund now", "Which issues?", {"refund": "Money back", "delivery": "Shipping"}).status == "uncertain"
    tested = tools.run_test(rt, path)
    issues = tested["fact_populations"]["used"]["issues"]
    assert issues["accepted"] == sum(row["accepted"] for row in issues["members"].values())
    assert issues["full_gold"] == 40 and issues["labelled"] == 160
    files.publish(path, {**files.read_yaml(path), "facts": {"issues": {"select": "Which issues?",
                                                                       "labels": {"refund": "Money back", "delivery": "Late shipping"}}}})
    changed = Labels(rt.load(path, draft=True))
    assert len(changed.examples(refund.name)) == 120 and changed.sw.plan.at("issues", "delivery").name != delivery.name


def test_corrections_and_paths_avoid_separator_collisions(tmp_path):
    path = saved(tmp_path, {"a": {"select": "Which?", "labels": {"b.c": "First"}}, "a.b": {"select": "Which?", "labels": {"c": "Second"}}})
    _, rt = judge()
    sw = rt.load(path)
    receipt = sw.correct({"message": "x"}, fact="a", value={"b.c": True})
    sw.correct({"message": "x"}, fact="a.b", value={"c": False})
    records = files.read_jsonl(path.with_suffix(".examples.jsonl"))
    assert receipt["facts"] == [["a", "b.c"]] and [(r["fact"], r["member"], r["value"]) for r in records] == [("a", "b.c", True), ("a.b", "c", False)]
    assert sw({"message": "x"}).facts["a"].value == ("b.c",) and sw({"message": "x"}).facts["a.b"].value == ()
    labels = Labels(rt.load(path, draft=True))
    ids = {e.path: e.id for exs in labels.by_fact.values() for e in exs}
    assert set(ids) == {("a", "b.c"), ("a.b", "c")} and len(set(ids.values())) == 2


def test_member_export_matches_runtime_presentation(tmp_path):
    path = saved(tmp_path, {"issues": {"select": "Which issues?", "labels": {"refund": "Money back", "delivery": "Shipping"}}})
    lock_dataset(path, 20)
    fake, rt = judge()
    exported = tools.export(rt, path, split="all", native=True)
    _, rows = read_export(exported["path"], fitting=False, native=True)
    row = next(r for r in rows if r.get("member") == "refund")
    assert row["fact"] == "issues" and row["candidate"] == "refund — Money back" and row["kind"] == "check"
    fake.requests.clear()
    rt.load(path)(row["bound_input"])
    asked = next(q for q in questions(fake) if member(q) == "refund")
    assert rt.detokenize(row["native"]["presentations"][0]["question"]).startswith(asked)


def test_pick_confusion_and_rate_distance():
    class Fact:
        keys = ("a", "b", "none")
    table = confusion(Fact, [(0, 0), (None, 0), (2, 1), (1, 1)])
    assert table["confusion"] == {"a": {"a": 1, "abstain": 1}, "b": {"none": 1, "b": 1}, "none": {}}
    assert table["per_label"]["b"] == {"gold": 2, "accepted": 1, "correct": 1}
    assert distances([(0, 2), (1, 1), (None, 0)]) == {"index_distance": {"0": 1, "2": 1}, "mean_index_distance": 1.0}


def test_obsolete_receipt_and_head_artifacts_are_rejected(tmp_path):
    files.publish(tmp_path / "receipt.json", {"schema": 2, "kind": "specialization-receipt", "rows": []})
    with pytest.raises(ConfigError, match="current schema 3"):
        load_receipt(tmp_path / "receipt.json")
    head = LexicalHead(("true", "false"), np.array([1], dtype=np.int32), np.zeros((1, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)).to_json()
    with pytest.raises(ConfigError, match="feature identity"):
        LexicalHead.from_json({k: v for k, v in head.items() if k != "feature_identity"})


def test_cli_question_files_accept_select_with_label_files(tmp_path):
    from semantic_switch.cli import question_specs
    files.publish(tmp_path / "issues.labels.yaml", LABELS)
    files.publish(tmp_path / "questions.yaml", {"issues": {"select": "Which issues?", "labels": "issues.labels.yaml", "max_claims": 2},
                                                "urgency": {"rate": "How urgent?", "levels": ["low", "high"]}})
    specs = question_specs(tmp_path / "questions.yaml")
    assert specs["issues"] == Select("Which issues?", LABELS, max_claims=2)
    _, rt = judge()
    assert rt.ask("refund", specs)["issues"].value == ("refund",)
    files.publish(tmp_path / "bad.yaml", {"issues": {"select": "Which issues?", "labels": {"a": {"description": "x", "requires": "vip"}}}})
    with pytest.raises(ConfigError, match="issues: unknown label field"):
        question_specs(tmp_path / "bad.yaml")
