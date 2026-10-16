import math

import pytest

from semantic_switch import Check, Pick, Rate

TICKET = {"subject": "Double charge", "body": "I was charged twice, please refund one."}


def test_check_value_and_p_true(rt):
    r = rt.check(TICKET, "The customer asks to receive money back for a charge.")
    assert r.status == "ok" and r.value is True and r.p_true > 0.9 and r.mass == pytest.approx(1.0)


def test_check_gate_and_tie(fake):
    from semantic_switch import Runtime

    rt = Runtime(type(fake)(lambda s, q, w: [0.5, 0.5]))
    r = rt.check("x", "anything")
    assert r.status == "uncertain" and r.value is None
    assert rt.check("x", "anything", min_score=0.51).status == "uncertain"
    rt = Runtime(type(fake)(lambda s, q, w: [0.8, 0.2]))
    assert rt.check("x", "anything").status == "uncertain"
    assert rt.check("x", "anything", min_score=0.8 - 1e-3).value is True


@pytest.mark.parametrize("bad", [0.4, 1.1, math.nan])
def test_min_score_validation(rt, bad):
    with pytest.raises(ValueError):
        rt.check(TICKET, "x", min_score=bad)


def test_results_have_no_truth_value(rt):
    with pytest.raises(TypeError):
        bool(rt.check(TICKET, "The customer asks to receive money back for a charge."))


class Address:
    def __init__(self, text):
        self.text = text


def test_pick_returns_original_object(rt):
    objs = [Address("login, profile or access"), Address("charges and invoices")]
    r = rt.pick(TICKET, "What is the message mainly about?", objs, definition=lambda a: a.text)
    assert r.status == "ok" and r.value is objs[1] and r.index == 1


def test_pick_dict_options_return_keys(rt):
    r = rt.pick(TICKET, "What is the message mainly about?", {"billing": "charges and invoices", "acct": "login"})
    assert r.value == "billing" and set(r.scores) == {"billing", "acct"}


def test_pick_no_eligible(rt):
    r = rt.pick(TICKET, "q", ["a", "b"], eligible=lambda o: False)
    assert r.status == "none" and r.removed["ineligible"] == [0, 1]
    r = rt.pick(TICKET, "q", ["a", "b"], eligible=lambda o: False, allow_none=True)
    assert r.status == "none"


def test_pick_too_many_candidates_needs_shortlist(rt, fake):
    options = [f"option number {i}" for i in range(25)] + ["charges and invoices"]
    r = rt.pick(TICKET, "What is the message mainly about?", options)
    assert r.status == "failed" and "too many" in r.error
    r = rt.pick(TICKET, "What is the message mainly about? charges", options, shortlist="lexical")
    assert r.status == "ok" and r.value == "charges and invoices"
    assert len(r.removed["not_examined"]) == 26 - 20 and not r.complete


def test_pick_single_candidate_is_scored(rt, fake):
    before = len(fake.requests)
    rt.pick(TICKET, "What is the message mainly about?", ["charges and invoices"])
    assert len(fake.requests) == before + 1


def test_pick_preserves_long_candidates(rt):
    long = "charges " + "word " * 200
    r = rt.pick(TICKET, "What is the message mainly about?", [long, "login"])
    assert r.presentations[-1]["answers"][0]["option"] == long


def test_rate(rt):
    r = rt.rate({"body": "it is broken and I need it today"}, "How urgent is it?", ["can wait", "today", "immediately"])
    assert r.value == "today" and r.index == 1 and 0.9 < r.expected < 1.1


def test_ask_scores_all_questions_in_one_call(rt, fake):
    calls = []
    original = fake.score
    fake.score = lambda reqs: calls.append(reqs) or original(reqs)
    out = rt.ask(TICKET, {"refund": Check("The customer asks to receive money back for a charge."),
                          "legal": Check("The customer threatens legal action against us."),
                          "urgency": Rate("How urgent is it?", ("can wait", "today", "immediately"))})
    assert len(calls) == 1 and len(calls[0]) == 1 and len(calls[0][0].suffixes) == 3
    assert out["refund"].value is True and out["legal"].value is False


def test_callable_judgments_run_on_a_runtime(rt):
    refund = Check("The customer asks to receive money back for a charge.")
    assert refund(TICKET, rt=rt).value is True
    topic = Pick("What is the message mainly about?", {"billing": "charges and invoices", "acct": "login"})
    assert topic(TICKET, rt=rt).value == "billing"


def test_pick_uncertain_below_policy(fake):
    from semantic_switch import Runtime

    rt = Runtime(type(fake)(lambda s, q, w: [0.6, 0.4]))
    r = rt.pick(TICKET, "q", ["a", "b"])
    assert r.status == "uncertain" and r.value is None and r.score == pytest.approx(0.6)
    assert rt.pick(TICKET, "q", ["a", "b"], min_score=0.6).value == "a"


def test_inline_reprs(rt):
    assert repr(rt.check(TICKET, "The customer asks to receive money back for a charge.")).startswith("<ok · True")
    assert repr(rt.pick(TICKET, "What is the message mainly about?", {"b": "charges and invoices", "a": "login"})) \
        .startswith("<ok · 'b'")
