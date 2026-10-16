import math

import numpy as np
import pytest
import yaml

from semantic_switch import Check, Runtime, judge
from semantic_switch.events import inline_event
from semantic_switch.drivers.codecs import token_scores
from semantic_switch.judge import SCORE_SEMANTICS, Scores
from semantic_switch.judgment import Judgment, Policy, answer, settle, unpresent
from semantic_switch.kernel import F, T, U
from semantic_switch.protocols import Question
from tests.test_openrouter_judge import Provider, response

CLAIM = Question("check", "The customer asks for a refund.")


def tokens(rows):
    return {token: math.log(p) for token, p in rows.items()}


def vertices(p):
    shortfall = 1 - sum(p)
    return [np.asarray(p) + shortfall * np.eye(len(p))[j] for j in range(len(p))]


def floor(p, indices):
    return min(sum(c[i] for i in indices) for c in vertices(p))


def old_sets(dist, truth, theta, pending):
    p_true = sum(p for i, (p, t) in enumerate(zip(dist, truth)) if t == T and i not in pending)
    p_false = sum(p for i, (p, t) in enumerate(zip(dist, truth)) if t == F and i not in pending)
    if p_true >= theta:
        return T, None
    if p_false >= theta:
        return F, None
    held = sum(dist[i] for i in pending if truth[i] in (T, F))
    return (U, "missing_input") if pending and max(p_true, p_false) + held >= theta else (U, "uncertain")


def cases(seed, complete):
    rng = np.random.default_rng(seed)
    for _ in range(400):
        width = int(rng.integers(2, 6))
        p = rng.dirichlet(np.ones(width) * .4) * (1 if complete else rng.uniform(.5, 1))
        truth = tuple(int(x) for x in rng.choice([T, F, U], width))
        pending = frozenset(int(i) for i in np.flatnonzero(rng.random(width) < .25))
        yield p, truth, pending, float(rng.uniform(.55, .99))


def pick(width):
    keys = tuple("abcde"[:width])
    return Judgment("pick", "Which?", keys, keys)


def test_reported_answer_mass_with_omitted_competitor_abstains():
    scores = token_scores(CLAIM, response(tokens({"true": .465, "false": .040, "maybe": .475})))
    assert scores.mass == pytest.approx(.505) and np.allclose(scores.probs, [.465 / .525, .04 / .525])
    assert answer(Judgment.check("c"), scores.probs, Policy(), scores.mass) == (None, "uncertain")


def test_non_answer_vocabulary_does_not_inflate_the_answer():
    rows = {"true": .54, **{f"x{k}": .04 / 19 for k in range(19)}}
    scores = token_scores(CLAIM, response(tokens(rows)))
    assert scores.mass == pytest.approx(.54) and scores.probs[0] == pytest.approx(.54 / .96)
    assert answer(Judgment.check("c"), scores.probs, Policy(), scores.mass) == (None, "uncertain")


def test_nearly_complete_decisive_evidence_settles():
    scores = token_scores(CLAIM, response(tokens({"true": .97, "false": .02})))
    assert np.allclose(scores.probs, [.97, .02]) and answer(Judgment.check("c"), scores.probs, Policy(), scores.mass) == (0, None)


@pytest.mark.parametrize("rows,error", [
    ({"true": .7, "false": .4}, "exceeds one"),
    ({"maybe": .9}, "no answer token"),
])
def test_invalid_token_evidence_is_rejected(rows, error):
    with pytest.raises(ValueError, match=error):
        token_scores(CLAIM, response(tokens(rows)))
    with pytest.raises(ValueError, match="invalid answer log probability"):
        token_scores(CLAIM, response({"true": -math.inf}))


def test_subset_acceptance_is_the_minimum_over_missing_mass_allocations():
    for p, truth, pending, theta in cases(0, complete=False):
        true = {i for i, t in enumerate(truth) if t == T and i not in pending}
        false = {i for i, t in enumerate(truth) if t == F and i not in pending}
        held = {i for i in pending if truth[i] in (T, F)}
        state = T if floor(p, true) >= theta else F if floor(p, false) >= theta else U
        why = None if state != U else "missing_input" if pending and max(floor(p, true | held), floor(p, false | held)) >= theta else "uncertain"
        assert settle(pick(len(p)), list(p), truth, Policy(theta, theta), .9, pending) == (state, why)


def test_choice_ordinal_and_silence_accept_only_guaranteed_answers():
    for p, _, pending, theta in cases(1, complete=False):
        top = int(np.argmax(p))
        for j in (pick(len(p)), Judgment("rate", "How much?", pick(len(p)).keys, pick(len(p)).keys)):
            accepted = floor(p, {top}) >= theta and top not in pending
            assert answer(j, p, Policy(theta, theta), .9, pending)[0] == (top if accepted else None)
    silent = Judgment.check("c", unstated=True)
    for p in ([.05, .02, .9], [.03, .02, .88]):
        assert (answer(silent, p, Policy(), .9)[0] == 2) == (floor(p, {2}) >= .9)


def test_escape_removal_conditions_on_the_escape_lower_bound():
    rng = np.random.default_rng(2)
    for _ in range(400):
        p = rng.dirichlet(np.ones(4) * .4) * rng.uniform(.5, 1)
        scores = Scores(p, .8)
        dist = unpresent((3, 0, 1, -1), scores, 5)
        if dist is None:
            assert int(np.argmax(p)) == 3
            continue
        conditional = [np.array([c[0], c[1], c[2]]) / (1 - c[3]) for c in vertices(p)]
        for subset in ({0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}):
            assert sum(dist[(3, 0, 1)[i]] for i in subset) == pytest.approx(min(sum(c[i] for i in subset) for c in conditional))
        assert dist.sum() == pytest.approx(p[:3].sum() / (1 - p[3])) and dist[[2, 4]].tolist() == [0, 0]


def test_complete_scores_keep_their_outcomes():
    for p, truth, pending, theta in cases(3, complete=True):
        assert settle(pick(len(p)), list(p), truth, Policy(theta, theta), None, pending) == old_sets(p, truth, theta, pending)
        if len(p) > 2 and int(np.argmax(p)) != len(p) - 1 and p.max() >= theta:
            slots = tuple(range(len(p) - 1)) + (-1,)
            old = np.zeros(len(p))
            old[:-1] = p[:-1] / p[:-1].sum()
            assert np.allclose(unpresent(slots, Scores(p, None), len(p)), old)


OPTIONS = {"ax": "AX part", "by": "BY part", "cz": "CZ part"}


def focused_rows(body):
    user = body["messages"][1]["content"]
    return tokens({"A": .80, "B": .03, "zz": .10} if "another one" in user else {"A": .80, "B": .02, "C": .01, "zz": .10})


def test_threshold_changes_acceptance_without_changing_requests_or_scores(tmp_path):
    path = tmp_path / "s.yaml"
    observations = []
    for theta in (.85, .95):
        provider = Provider(focused_rows)
        with Runtime(judge("openrouter:m", codec="chat", transport=provider)) as rt:
            inline = rt.pick("ax failed", "Which part failed?", OPTIONS, focus="mentions", min_score=theta)
            path.write_text(yaml.safe_dump({"name": "s", "fields": {"r": {"type": "text", "required": True}}, "reads": ["r"],
                                            "facts": {"part": {"pick": "Which part failed?", "options": OPTIONS, "focus": "mentions", "min_score": theta}}}))
            saved = rt.load(path)("ax failed").facts["part"]
            claim = Check("The customer asks for a refund.")
            event = inline_event(rt, {"c": claim("x", rt)}, {"c": claim}, {}, "call", 0)
        assert (inline.status, inline.value, inline.score) == (saved.status, saved.value, saved.score)
        assert inline.scores == dict(saved.scores) == {"ax": pytest.approx(.8 / .87)} and inline.prediction == saved.prediction
        observations.append((inline, [body["messages"][1]["content"] for body in provider.bodies]))
    (accepted, asked), (rejected, again) = observations
    assert accepted.status == "ok" and rejected.status == "uncertain" and asked == again and len(asked) == 3
    assert accepted.prediction == rejected.prediction and rejected.prediction.key == "ax" and rejected.focus == "mentions"
    assert event.schema == 9 and event.score_semantics == SCORE_SEMANTICS