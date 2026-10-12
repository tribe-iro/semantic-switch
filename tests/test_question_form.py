import pytest

from semantic_switch import ConfigError, Runtime
from semantic_switch.adapters.qwen3 import Qwen3Adapter
from semantic_switch.judgment import Judgment
from semantic_switch.protocols import Framing, Question

from tests.conftest import FakeScorer


@pytest.fixture
def adapter():
    return Qwen3Adapter(FakeScorer())


def rendered(adapter, question):
    tokens, slots = adapter.suffix(question)
    return adapter.scorer.detokenize(tokens), slots


def test_question_check_asks_yes_or_no(adapter):
    text, slots = rendered(adapter, Question("check", "Is the customer threatening legal action?", (), ("quoting someone else",)))
    assert text.startswith("Question about the input above: Is the customer threatening legal action?\nExcept: quoting someone else.\n"
                           "Answer with only yes or no.")
    assert slots == adapter.polar


def test_claim_check_rendering_unchanged(adapter):
    text, slots = rendered(adapter, Question("check", "The customer threatens legal action."))
    assert text.startswith("Claim about the input above: The customer threatens legal action.\n"
                           "Is the claim true or false? Answer with only true or false.")
    assert slots == adapter.binary


@pytest.mark.parametrize("text,guidance,options", [
    ("Is there a leak?", None, ("yes", "no", "the input does not say")),
    ("Is there a leak?", "Leaks are visible coolant on the floor.", ("yes", "no", "the evidence is insufficient to answer")),
    ("There is a leak.", None, ("the input states it is true", "the input states it is false", "the input does not say")),
])
def test_three_way_wording_follows_form(text, guidance, options):
    question = Judgment.check(text, unstated=True, guidance=guidance).question()
    assert question.kind == "pick" and question.options == options and question.text.startswith(text if text.endswith("?") else "Claim: ")


def test_question_and_claim_settle_the_same_way():
    for text in ("Is the customer asking for a refund?", "The customer asks for a refund."):
        scorer = FakeScorer(lambda state, question, width: [0.97, 0.03])
        with Runtime({"judge": scorer}) as rt:
            result = rt.check("I want my money back.", text)
        assert (result.status, result.value) == ("ok", True)


def test_previous_framing_revision_requires_relock(adapter):
    pinned = adapter.framing.as_dict() | {"revision": 1}
    with pytest.raises(ConfigError, match="re-run ss lock"):
        Framing.parse(pinned)
