import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import compile_spec
from semantic_switch.training import describe, target_scope

from semantic_testkit.labels import envelopes
from semantic_testkit.scoring import FakeScorer


def spec(**changes):
    base = {"name": "routing", "fields": {"message": {"type": "text", "required": True}}, "reads": ["message"],
            "facts": {"refund": {"check": "The customer asks to receive money back for a charge.", "binding": "refund"},
                      "legal": {"check": "The customer threatens legal action against us."},
                      "issues": {"select": "Which issues are raised?", "labels": {"billing": "billing problem", "outage": "service outage"},
                                 "binding": "refund"}},
            "rules": [{"id": "escalate", "when": "legal", "then": "escalate"}, {"id": "money", "when": "refund", "then": "refund"}],
            "otherwise": "other"}
    base.update(changes)
    return base


def write(tmp_path, data):
    path = tmp_path / "routing.yaml"
    files.publish(path, data)
    return path


def test_targets_inherit_static_bindings_and_meanings_ignore_them():
    plan = compile_spec(spec())
    bound = {plan.facts[f].shown: plan.binding_of(f) for f in plan.observed}
    assert bound == {"refund": "refund", "legal": "judge", 'issues has "billing"': "refund", 'issues has "outage"': "refund"}
    assert plan.bindings == ("judge", "refund") and plan.rebound("other").binding_of(plan.fact_index["legal"]) == "other"
    assert plan.rebound("other").binding_of(plan.fact_index["refund"]) == "refund"
    unbound = compile_spec(spec(facts={name: {k: v for k, v in fact.items() if k != "binding"} for name, fact in spec()["facts"].items()}))
    assert [f.meaning_rev for f in unbound.facts] == [f.meaning_rev for f in plan.facts] and unbound.switch_rev == plan.switch_rev
    assert "binding" not in unbound.spec and all("binding" not in fact for fact in unbound.spec["facts"].values())
    assert plan.spec["facts"]["refund"]["binding"] == "refund" and compile_spec(spec(binding="local")).bindings == ("local", "refund")
    assert describe(plan)["targets"]["legal"]["binding"] == "judge"


def test_card_claims_inherit_and_shared_atoms_reject_conflicting_bindings():
    cards = {"cards": [{"id": "late", "applies_when": "The parcel is late."}]}
    plan = compile_spec(spec(facts={"a": {"select": "Which?", **cards, "binding": "fast"}}, rules=[]))
    claim = next(f for f, fact in enumerate(plan.facts) if fact.hidden)
    assert plan.binding_of(claim) == "fast" and plan.bindings == ("fast",)
    with pytest.raises(ConfigError, match="shared semantic atom"):
        compile_spec(spec(facts={"a": {"select": "Which?", **cards, "binding": "fast"}, "b": {"select": "Which?", **cards}}, rules=[]))
    compile_spec(spec(facts={"a": {"select": "Which?", **cards, "binding": "judge"}, "b": {"select": "Which?", **cards}}, rules=[]))
    referenced = {"late": {"check": "The parcel is late.", "binding": "fast"},
                  "a": {"select": "Which?", "cards": [{"id": "late", "applies_when": {"fact": "late"}}]}}
    with pytest.raises(ConfigError, match="shared semantic atom"):
        compile_spec(spec(facts=referenced, rules=[]))
    with pytest.raises(ConfigError, match="binding names a runtime model role"):
        compile_spec(spec(binding="two words"))


def oracle(answer):
    def respond(state, question, width):
        return [0.97, 0.03] if answer in question else [0.03, 0.97] if width == 2 else [1 / width] * width
    return respond


def test_executor_scores_each_binding_on_its_judge_and_restores_order(tmp_path):
    path = write(tmp_path, spec())
    general, specialist = FakeScorer(oracle("legal action"), name="general"), FakeScorer(oracle("money back"), name="specialist")
    with Runtime(bindings={"judge": general, "refund": specialist}) as rt:
        switch = rt.load(path, draft=True)
        results = switch.run([{"message": "first"}, {"message": "second"}, {"message": "third"}], facts=["refund", "legal", "issues"])
    assert [r.answer for r in results] == ["escalate"] * 3
    assert all(r.facts["refund"].value is True and r.facts["legal"].value is True for r in results)
    assert {q for _, q in general.requests} and all("legal action" in q for _, q in general.requests)
    assert all("legal action" not in q for _, q in specialist.requests) and any("billing problem" in q for _, q in specialist.requests)
    for fake in (general, specialist):
        seen = list(dict.fromkeys(next(word for word in ("first", "second", "third") if word in state) for state, _ in fake.requests))
        assert seen == ["first", "second", "third"]
    assert results[0].cost["bindings"] == {"legal": "judge", "refund": "refund", 'issues has "billing"': "refund", 'issues has "outage"': "refund"}


def test_exact_rules_bypass_every_binding(tmp_path):
    data = spec(fields={"message": {"type": "text", "required": True}, "kind": {"type": "enum", "values": ["a", "b"]}},
                facts={"route": {"pick": "Where?", "options": {"x": {"description": "x", "requires": "kind == 'a'"},
                                                                 "y": {"description": "y", "requires": "kind == 'a'"}},
                                 "allow_none": True, "binding": "refund"}},
                rules=[], otherwise=None)
    path = write(tmp_path, data)
    specialist = FakeScorer(name="specialist")
    with Runtime(bindings={"judge": FakeScorer(), "refund": specialist}) as rt:
        result = rt.load(path, draft=True)({"message": "m", "kind": "b"})
    assert result.facts["route"].source == "rule" and specialist.requests == [] and result.cost["bindings"] == {}


def test_unbound_model_role_fails_at_load(tmp_path):
    path = write(tmp_path, spec())
    with Runtime(FakeScorer()) as rt:
        with pytest.raises(ConfigError, match=r"model roles \['refund'\]"):
            rt.load(path, draft=True)


def labelled(tmp_path):
    path = write(tmp_path, spec(rules=[], otherwise=None, facts={"refund": spec()["facts"]["refund"], "legal": spec()["facts"]["legal"]}))
    files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes([
        {"input": {"message": f"case {i}"}, "facts": {"refund": True, "legal": False}, "group": f"family-{i}"} for i in range(24)])))
    return path


def test_lock_pins_the_target_binding_map_and_resets_only_the_changed_binding(tmp_path):
    path = labelled(tmp_path)
    with Runtime(bindings={"judge": FakeScorer(name="general"), "refund": FakeScorer(name="specialist")}) as rt:
        tools.lock(rt, path)
        data = files.read_json(path.with_suffix(".lock.json"))
        assert set(data["models"]) == set(data["execution"]) == set(data["returned_models"]) == {"judge", "refund"}
        assert data["bindings"] == {"judge": {"representation": "base", "scope": ["legal"]}, "refund": {"representation": "base", "scope": ["refund"]}}
        assert data["models"]["refund"]["source"] == "specialist"
        locked = rt.load(path)
        assert set(locked.snapshot.policies["judge"]) == {"refund", "legal"}
    with Runtime(bindings={"judge": FakeScorer(name="general"), "refund": FakeScorer(name="replacement")}) as rt:
        with pytest.raises(ConfigError, match="binding refund identity changed"):
            rt.load(path).snapshot
        changed = rt.load(path, allow_model_change=True)
        assert set(changed.snapshot.policies["judge"]) == {"legal"} and changed.snapshot.models["refund"]["source"] == "replacement"
        assert changed.snapshot.candidate_id != data["candidate_id"] and "judge:legal" in changed.snapshot.lineage
        assert "judge:refund" not in changed.snapshot.lineage
    tampered = {**data, "bindings": {**data["bindings"], "refund": {"representation": "base", "scope": ["legal"]}}}
    from semantic_switch.compiler import digest
    tampered["id"] = digest({k: v for k, v in tampered.items() if k not in ("id", "created")})[:12]
    files.publish(path.with_suffix(".lock.json"), tampered)
    with Runtime(bindings={"judge": FakeScorer(name="general"), "refund": FakeScorer(name="specialist")}) as rt:
        with pytest.raises(ConfigError, match="target-binding map differs"):
            rt.load(path)


def test_events_identify_the_answering_binding(tmp_path):
    path = write(tmp_path, spec())
    events = []
    with Runtime(bindings={"judge": FakeScorer(), "refund": FakeScorer(name="specialist")}, on_decision=events.append) as rt:
        rt.load(path, draft=True)({"message": "a refund please"})
    assert events[0].schema == 9 and events[0].facts["refund"]["binding"] == "refund" and set(events[0].predictor) == {"judge", "refund"}


def test_target_scope_resolves_whole_task_fact_and_member_scopes():
    plan = compile_spec(spec())
    assert target_scope(plan) == [["refund"], ["legal"], ["issues", "billing"], ["issues", "outage"]]
    assert target_scope(plan, "issues") == [["issues", "billing"], ["issues", "outage"]]
    assert target_scope(plan, ["issues/outage", "refund", "refund"]) == [["issues", "outage"], ["refund"]]
    with pytest.raises(ConfigError, match="not a label"):
        target_scope(plan, "issues/shipping")
    with pytest.raises(ConfigError, match="unknown fact"):
        target_scope(plan, "shipping")
    cards = compile_spec(spec(facts={"a": {"select": "Which?", "cards": [{"id": "late", "applies_when": "The parcel is late."}]}}, rules=[]))
    with pytest.raises(ConfigError, match="no directly scored targets"):
        target_scope(cards, "a")
    with pytest.raises(ConfigError, match="visible fact"):
        target_scope(cards, next(f.name for f in cards.facts if f.hidden))


def test_scoped_export_covers_only_its_targets_and_rejects_mixed_binding_native_scope(tmp_path):
    path = labelled(tmp_path)
    with Runtime(bindings={"judge": FakeScorer(name="general"), "refund": FakeScorer(name="specialist")}) as rt:
        exported = tools.export(rt, path, to=tmp_path / "refund.jsonl", scope="refund")
        manifest, rows = read_export(exported["path"], fitting=False)
        assert manifest["scope"] == [["refund"]] and {row["fact"] for row in rows} == {"refund"}
        assert all(situation["complete"] for situation in manifest["situations"])
        with pytest.raises(ConfigError, match="spans bindings"):
            tools.export(rt, path, to=tmp_path / "all.jsonl", native=True)
