import json
import shutil
from pathlib import Path

import pytest
import yaml

from semantic_switch import ConfigError, Runtime, tools
from semantic_switch.context import canonical, project
from semantic_switch.examples import derive_fact_labels

from .conftest import FakeScorer

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "route_ticket.yaml"
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", p)
    return p


def edit(path, fn):
    spec = yaml.safe_load(path.read_text())
    fn(spec)
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


def t(body, subject="Help", **extra):
    return {"subject": subject, "body": body, **extra}


def test_routing(rt, path):
    sw = rt.load(path)
    assert sw(t("I will see you in court, my lawyer will call")).answer == "legal"
    r = sw(t("Please refund my last charge", payment={"status": "settled"}))
    assert (r.status, r.answer, r.case) == ("ok", "refund_review", "refund_settled")
    assert sw(t("Please refund my last charge", payment={"status": "pending"})).answer == "billing"
    assert sw(t("The app is broken and I need it fixed now")).answer == "tech_priority"
    assert sw(t("Just saying hello")).answer == "general"


def test_repr_and_explain_speak_the_users_language(rt, path):
    r = rt.load(path)(t("Please refund my last charge", payment={"status": "pending"}))
    assert repr(r).startswith("<ok · 'billing' · case refund_other")
    text = r.explain()
    assert "✓ refund_other" in text and "refund_requested" in text and "==" not in text.split("facts")[1]
    assert "atoms" in r.explain(raw=True)
    fact = r.facts["refund_requested"]
    assert (fact.value, fact.settled, fact.source) == (True, True, "judge") and set(fact.scores) == {"true", "false"}


def test_missing_and_invalid_context_fail(rt, path):
    r = rt.load(path)({"subject": "x"})
    assert (r.status, r.missing, r.invalid) == ("failed", ("body",), ())
    r = rt.load(path)(t("hi", payment={"status": "weird"}))
    assert (r.status, r.invalid, r.missing) == ("failed", ("payment.status",), ())


def test_needs_input_lists_only_blocking_fields(rt, path):
    r = rt.load(path)(t("Please refund my last charge"))
    assert r.status == "needs_input" and r.missing == ("payment.status",) and r.blocked_by == "refund_settled"


def test_masked_missing_field_is_irrelevant(rt, path):
    assert rt.load(path)(t("Refund now or my lawyer will file a lawsuit")).answer == "legal"


def test_semantic_fact_with_missing_read_is_not_scored(rt, path, fake):
    edit(path, lambda s: s["cases"].insert(0, {"id": "dispute_first", "when": "disputes_payment_status",
                                               "then": "billing_escalation"}))
    r = rt.load(path)(t("I did pay, your records are wrong"))
    assert not fake.asked("disputes the recorded payment")
    assert r.status == "needs_input" and r.missing == ("payment.status",)
    assert r.facts["disputes_payment_status"].why == "missing: payment.status"


def test_uncertain_blocks_lower_cases(rt, path):
    r = rt.load(path)(t("maybe a refund?", payment={"status": "settled"}))
    assert r.status == "uncertain" and r.uncertain == ("refund_requested",)
    fact = r.facts["refund_requested"]
    assert fact.value is None and not fact.settled and fact.why == "uncertain"


def test_false_masks_uncertain(rt, path):
    edit(path, lambda s: s.update(cases={"x": 'refund_requested and payment.status == "settled"'}, otherwise="y"))
    assert rt.load(path)(t("maybe a refund?", payment={"status": "pending"})).answer == "y"


def test_low_mass_is_unknown_and_maskable(rt, path):
    r = rt.load(path)(t("garbled text", payment={"status": "settled"}))
    assert r.status == "uncertain" and r.facts["refund_requested"].why == "low answer mass"
    edit(path, lambda s: s.update(cases={"x": 'refund_requested and payment.status == "settled"'}, otherwise="y"))
    assert rt.load(path)(t("garbled text", payment={"status": "pending"})).answer == "y"


def test_error_in_a_fact_that_does_not_decide_is_reported_not_fatal(path):
    from .conftest import default_oracle

    def oracle(state, q, width):
        if "urgent" in q:
            raise RuntimeError("backend exploded")
        return default_oracle(state, q, width)

    r = Runtime({"judge": FakeScorer(oracle)}).load(path)(t("please refund", payment={"status": "pending"}))
    assert (r.status, r.answer) == ("ok", "billing") and "urgency" in r.errors


def test_blocking_error_fails(rt, path):
    r = rt.load(path)(t("BOOM nothing else", payment={"status": "settled"}))
    assert r.status == "failed" and "refund_requested" in r.errors


def test_staged_schedule_scores_only_what_the_first_case_needs(rt, path, fake):
    r = rt.load(path)(t("my lawyer will file a lawsuit"))
    assert r.answer == "legal" and r.cost["judge"] == 1 and r.cost["rounds"] == 1
    r = rt.load(path)(t("The app is broken and I need it fixed now"))
    assert r.answer == "tech_priority" and r.cost["rounds"] == 2


def test_each_input_runs_its_rounds_before_the_next(rt, path, fake):
    sw = rt.load(path)
    sw.run([t("Please refund my last charge", payment={"status": "pending"}), t("hello there")])
    states = [s for s, _ in fake.requests]
    first = [i for i, s in enumerate(states) if "refund" in s]
    assert len(first) > 1 and first == list(range(first[0], first[-1] + 1))


def test_eligibility_removes_options(rt, path, fake):
    sw = rt.load(path)
    assert sw(t("my device is damaged", product="laptop")).answer == "general"
    assert not fake.asked("a physical device")
    assert sw(t("my device is damaged", product="router")).answer == "tech"
    assert fake.asked("a physical device")


def test_ineligible_override_is_ignored(rt, path):
    sw = rt.load(path)
    sw.correct(t("my device is damaged", product="laptop"), fact="topic", value="hardware")
    r = sw(t("my device is damaged", product="laptop"))
    assert r.facts["topic"].source == "judge" and r.answer == "general"


def test_correct_on_draft_applies_now(rt, path):
    sw = rt.load(path)
    msg = t("I would like my money returned please", payment={"status": "settled"})
    assert sw(msg).answer == "general"
    assert sw.correct(msg, fact="refund_requested", value=True)["effective"] == "now"
    r = sw(msg)
    assert r.answer == "refund_review" and r.facts["refund_requested"].source == "example"


def test_correct_accepts_true_false_strings(rt, path):
    sw = rt.load(path)
    msg = t("refund please", payment={"status": "settled"})
    sw.correct(msg, fact="refund_requested", value="false")
    assert sw(msg).facts["refund_requested"].value is False


def test_correct_on_lock_waits_for_next_lock(rt, path):
    tools.lock(rt, path)
    sw = rt.load(path)
    msg = t("I would like my money returned please", payment={"status": "settled"})
    assert sw.correct(msg, fact="refund_requested", value=True)["effective"] == "after ss lock"
    assert sw(msg).answer == "general"
    tools.lock(rt, path)
    assert rt.load(path)(msg).answer == "refund_review"


def test_canonical_key_uses_untruncated_projection(rt, path):
    sw = rt.load(path)
    head = "x" * 5000
    reads = sw.plan.fact("refund_requested").reads
    a, b = (project(sw.plan.fields, t(head + s)) for s in (" please refund", " please do NOT refund"))
    assert canonical(a, reads) != canonical(b, reads)
    sw.correct(t(head + " please refund"), fact="refund_requested", value=True)
    assert sw(t(head + " please do NOT refund", payment={"status": "settled"})).facts["refund_requested"].source == "judge"


def test_override_key_is_per_fact_reads(rt, path):
    sw = rt.load(path)
    sw.correct(t("money back please", subject="A"), fact="refund_requested", value=True)
    assert sw(t("money back please", subject="A", payment={"status": "settled"})).answer == "refund_review"
    assert sw(t("money back please", subject="B", payment={"status": "settled"})).facts["refund_requested"].source == "judge"


def test_lock_runs_tests_against_frozen_plan(rt, path):
    edit(path, lambda s: s["tests"].append({"input": t("hello"), "expect": "legal"}))
    with pytest.raises(tools.LockError):
        tools.lock(rt, path)
    assert not path.with_name("route_ticket.lock.json").exists()


def test_lock_is_self_contained_and_not_mixed_with_draft(rt, path):
    sw = rt.load(path)
    msg = t("I would like my money returned please", payment={"status": "settled"})
    sw.correct(msg, fact="refund_requested", value=True)
    tools.lock(rt, path)
    path.with_name("route_ticket.examples.jsonl").unlink()
    edit(path, lambda s: s.update(otherwise="changed"))
    locked = rt.load(path)
    assert locked.snapshot.locked and locked(msg).answer == "refund_review" and locked(t("hello")).answer == "general"
    assert rt.load(path, draft=True)(t("hello")).answer == "changed"


def test_lock_pins_its_models(rt, path):
    tools.lock(rt, path)
    other = Runtime({"judge": FakeScorer(name="other-judge")})
    with pytest.raises(ConfigError, match="re-run `ss lock`|Re-run `ss lock`"):
        other.load(path)
    sw = other.load(path, allow_model_change=True)
    assert sw.warnings and "were reset" in sw.warnings[0]


def test_stale_examples_are_ignored_after_meaning_change(rt, path):
    sw = rt.load(path)
    msg = t("I would like my money returned please", payment={"status": "settled"})
    sw.correct(msg, fact="refund_requested", value=True)
    edit(path, lambda s: s["facts"]["refund_requested"].update(check="The customer wants a refund."))
    sw2 = rt.load(path)
    assert sw2.warnings and sw2(msg).facts["refund_requested"].source == "judge"


def test_specialize_without_lock_warns(rt, path):
    edit(path, lambda s: s["facts"]["refund_requested"].update(specialize="lexical"))
    assert any("specialize: lexical" in w for w in rt.load(path).warnings)


def test_many_preserves_order_and_is_lazy(rt, path):
    gen = rt.load(path).many(iter([t("my lawyer will sue"), t("hello"), t("broken now")]))
    assert next(gen).answer == "legal"
    assert [r.answer for r in gen] == ["general", "tech_priority"]


def test_resolve_all_always_returns_tuples(rt, path):
    edit(path, lambda s: s.update(resolve="all"))
    sw = rt.load(path)
    r = sw(t("refund me now or my lawyer will file a lawsuit", payment={"status": "pending"}))
    assert r.status == "ok" and set(r.answer) == {"legal", "billing"}
    assert sw(t("hello")).answer == ("general",)


def test_answer_label_derivation(rt, path):
    plan = rt.load(path).plan
    assert derive_fact_labels(plan, "refund_other") == {"legal_threat": False, "refund_requested": True}
    assert derive_fact_labels(plan, "tech") == {"legal_threat": False, "refund_requested": False}
    assert derive_fact_labels(plan, "refund_settled") == {"legal_threat": False, "refund_requested": True}


def test_lexical_head_settles_and_falls_back(rt, path, fake):
    sw = rt.load(path)
    for i in range(12):
        sw.correct(t(f"please refund my charge number {i}"), fact="refund_requested", value=True)
        sw.correct(t(f"hello there friend number {i}"), fact="refund_requested", value=False)
    edit(path, lambda s: s["facts"]["refund_requested"].update(specialize="lexical"))
    tools.lock(rt, path)
    locked = rt.load(path)
    assert "refund_requested" in locked.snapshot.heads
    r = locked(t("please refund my charge number 99", payment={"status": "settled"}))
    assert r.facts["refund_requested"].source in ("classifier", "judge") and r.answer == "refund_review"
    assert locked(t("zebra quantum", payment={"status": "settled"})).facts["refund_requested"].source == "judge"


def test_lock_contents(rt, path):
    data = json.loads(Path(tools.lock(rt, path)["lock"]).read_text())
    assert data["facts"]["refund_requested"]["examples"] == [] and data["tests"].startswith("1/1")
    assert data["models"]["judge"]["template"].startswith("qwen3@")
