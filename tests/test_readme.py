import re
from pathlib import Path

import semantic_switch as ss

from semantic_testkit.scoring import FakeScorer
from tests.conftest import transport

README = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
MESSAGE = "Charged twice AGAIN. Fix it by Friday or my lawyer files suit."


def answers(state, question, width):
    if width == 3 and "refund" in question:
        return [0.96, 0.02, 0.02]
    if "legal action" in question:
        return [0.97, 0.03]
    if width == 2:
        return [0.02, 0.98]
    return [0.94] + [0.06 / (width - 1)] * (width - 1)


def test_readme_inline_api_with_scripted_scores():
    scorer = FakeScorer(answers)
    with ss.Runtime(scorer) as rt:
        r = rt.check(MESSAGE, "Is the customer threatening legal action against us?", excludes=["merely mentioning a lawyer"])
        assert (r.status, r.value) == ("ok", True)
        assert rt.pick(MESSAGE, "What should support do?", {"legal": "Escalate to the legal team", "billing": "Fix a billing problem",
                                                            "tech": "Fix a broken product"}, allow_none=True).value == "legal"
        assert rt.rate(MESSAGE, "How urgent is this?", ["can wait", "today", "immediately"]).value == "can wait"
        shared = rt.ask(MESSAGE, {"legal": ss.Check("Is the customer threatening legal action against us?"),
                                  "refund": ss.Check("Is the customer asking to get money back?")})
        assert shared["legal"].value is True and shared["refund"].value is False
        assert rt.check({"subject": "", "body": " "}, "Is the customer asking to get money back?", fields=["subject", "body"]).status == "needs_input"
    assert scorer.requests and all(MESSAGE in state for state, question in scorer.requests)


def test_readme_switch_dispatch_with_scripted_scores(tmp_path):
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "refunds.md").write_text("Duplicate charges qualify for refunds.", encoding="utf-8")
    definition = tmp_path / "support_route.yaml"
    definition.write_text(re.search(r"```yaml\n(name: support_route.*?)```", README, re.S).group(1), encoding="utf-8")
    with ss.Runtime(FakeScorer(answers)) as rt:
        r = rt.load(definition)({"subject": "Double charge", "body": MESSAGE, "payment": {"status": "settled"}})
    assert (r.status, r.answer, r.rule) == ("ok", "legal_team", "legal")


def test_readme_judge_choice_is_constructible():
    judge = ss.judge("openrouter:google/gemma-4-26b-a4b-it", transport=transport(lambda url, body, timeout: None))
    with ss.Runtime(judge, own_judges=True) as rt:
        assert rt.identity()["model"] == "google/gemma-4-26b-a4b-it"


def test_readme_dispatch_uses_evidence_fields_and_settled_results():
    from semantic_switch.guidance import Guidance
    handled = []
    handlers = {"refund": handled.append, "escalate": handled.append, "ask_details": handled.append}
    options = {"refund": "Refund the duplicate charge", "escalate": "Escalate to a supervisor",
               "ask_details": "Ask the customer for the missing order number"}
    case = {"message": "Charged twice for order 7.", "reference": "Order 7: two identical charges.", "internal": "do not show"}
    for oracle, expect in ((lambda s, q, w: [0.94] + [0.06 / (w - 1)] * (w - 1), 1), (lambda s, q, w: [1 / w] * w, 0)):
        handled.clear()
        scorer = FakeScorer(oracle)
        with ss.Runtime(scorer) as rt:
            r = rt.pick(case, "What should support do next?", options, guidance=Guidance("Refund duplicate charges."),
                        fields=["message", "reference"], allow_none=True)
        if r.status == "ok":
            handlers[r.value](case)
        assert len(handled) == expect
        state, question = scorer.requests[0]
        assert "Order 7: two identical charges." in state and "do not show" not in state and "Refund duplicate charges." not in state


TOPICS = {"billing": ss.Label("charges, invoices, payments or refunds", excludes=["a question about future prices"]),
          "delivery": "an order that is late, missing or damaged", "account": "login, profile or data access"}


def oracle_for(answers_by_label, pick=None):
    def oracle(state, question, width):
        for label, p in answers_by_label.items():
            if f"Candidate: {label}" in question:
                return [p, 1 - p] if width == 2 else [p, 1 - p - .01, .01]
        return pick if pick is not None else [0.94] + [0.06 / (width - 1)] * (width - 1)
    return oracle


def test_readme_classification_shapes_and_dispatch():
    message = "Charged twice and my parcel never came."
    oracle = oracle_for({"billing": .97, "delivery": .96, "account": .02})
    with ss.Runtime(FakeScorer(oracle)) as rt:
        r = rt.ask(message, {"topic": ss.Pick("What is the message mainly about?", TOPICS, allow_none=True),
                             "issues": ss.Select("Which issues does the customer raise?", TOPICS),
                             "urgency": ss.Rate("How urgent is this?", ["can wait", "today", "immediately"])})
    assert r["topic"].value == "billing" and r["urgency"].status == "ok"
    issues = r["issues"]
    assert (issues.status, issues.value, issues.complete, issues.settled) == ("ok", ("billing", "delivery"), True, True)


def test_readme_select_unresolved_member_keeps_partial_value():
    with ss.Runtime(FakeScorer(oracle_for({"billing": .97, "delivery": .5, "account": .02}))) as rt:
        issues = rt.select("Charged twice; maybe the parcel is late.", "Which issues does the customer raise?", TOPICS)
    assert issues.status == "uncertain" and issues.value == ("billing",) and issues.complete and not issues.settled


def test_proposer_runs_only_on_complete_none():
    import importlib
    example = importlib.import_module("examples.propose_category")
    calls = []
    propose = lambda message, instruction, categories: calls.append(categories) or {"name": "praise", "definition": "compliments"}
    for pick, expect in (([0.02, 0.02, 0.02, 0.94], 1), ([0.94, 0.02, 0.02, 0.02], 0), ([0.25, 0.25, 0.25, 0.25], 0)):
        calls.clear()
        with ss.Runtime(FakeScorer(lambda s, q, w, pick=pick: pick)) as rt:
            result, proposal = example.classify("Great service, thanks!", propose, rt)
        assert len(calls) == expect and (proposal is not None) == bool(expect)
        assert not calls or calls[0] == example.CATEGORIES
