from __future__ import annotations

import re

import pytest

from semantic_switch import Runtime, Select, files, tools
from semantic_switch.kernel import F, T, U, compose

from semantic_testkit.scoring import FakeScorer
from semantic_testkit.labels import envelopes

LABELS = {"alpha": "Alpha label", "beta": "Beta label", "gamma": {"description": "Gamma label", "requires": 'tier == "gold"'}}
OPEN = {"applicable": T, "ineligible": F, "rejected": F, "superseded": F}


def member(question):
    found = re.search(r"^Candidate: (\S+) — ", question, re.M)
    return found and found.group(1)


def oracle(state, question, width):
    found = re.search(r"The (\w+) (?:issue applies|unit)", question)
    key = member(question) or (found and found.group(1))
    if key and f"{key}!" in state:
        raise RuntimeError("backend exploded")
    if key and f"{key}?" in state:
        return [.5, .5]
    return [.97, .03] if key and f"{key}+" in state else [.03, .97]


def variants(cards):
    first = cards[0]
    return {"none": {}, "empty": {"rules": [{"id": "empty", "when": "issues is empty", "then": "empty"}], "otherwise": "some"},
            "one": {"rules": [{"id": "hit", "when": f'issues has "{first}"', "then": "hit"}], "otherwise": "miss"},
            "all": {"rules": [{"id": k, "when": f'issues has "{k}"', "then": k} for k in cards], "otherwise": "otherwise", "resolve": "all"}}


def run(tmp_path, name, fact, extra, data):
    path = tmp_path / f"{name}.yaml"
    files.publish(path, {"name": name, "fields": {"message": {"type": "text", "required": True}, "tier": {"type": "text"}},
                         "reads": ["message"], "facts": {"issues": fact}, **extra})
    events = []
    rt = Runtime(FakeScorer(oracle), on_decision=events.append)
    return rt.load(path)(data), events[0].as_dict()


def truth(fact, key):
    return OPEN.get(fact.states[key], U)


def predicate(result, expected, yes, no):
    if expected == U:
        assert result.status != "ok"
    else:
        assert result.status == "ok" and result.answer == (yes if expected == T else no)


LABEL_FACT = {"select": "Which issues apply?", "labels": LABELS}
EXCLUSIVE = {"select": "Which issue applies?", "exclusive": True, "cards": [
    {"id": "alpha", "applies_when": ["The alpha issue applies."]}, {"id": "beta", "applies_when": ["The beta issue applies."]},
    {"id": "gamma", "applies_when": ["The gamma issue applies."], "requires": 'tier == "gold"'}]}
HOT = {"id": "hot", "applies_when": ["The hot unit overheats."]}
HOT2 = {"id": "hot2", "applies_when": ["The hot unit overheats."], "requires": 'tier == "gold"'}
RATTLE = {"id": "rattle", "applies_when": ["The rattle unit shakes."], "supersedes": ["hot"]}
CARD_FACT = {"select": "Which procedures apply?", "cards": [HOT, HOT2, RATTLE]}

SCENARIOS = {
    "two_exclusive_positives": (EXCLUSIVE, "alpha+ beta+", "silver"),
    "positive_plus_unknown": (EXCLUSIVE, "alpha+ beta?", "silver"),
    "all_negative": (EXCLUSIVE, "quiet", "silver"),
    "exact_ineligible": (EXCLUSIVE, "gamma+", "silver"),
    "eligible_single": (EXCLUSIVE, "gamma+", "gold"),
    "failure": (LABEL_FACT, "alpha! beta+", "silver"),
    "budgeted": (LABEL_FACT | {"max_claims": 1}, "alpha+ beta+", "silver"),
    "shared_claim": (CARD_FACT, "hot+", "gold"),
    "shared_claim_exclusive": (CARD_FACT | {"exclusive": True}, "hot+", "gold"),
    "superseded": (CARD_FACT, "hot+ rattle+", "silver"),
    "unresolved_superseder": (CARD_FACT, "hot+ rattle?", "silver"),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_effective_selection_is_identical_under_every_reference(tmp_path, scenario):
    fact, message, tier = SCENARIOS[scenario]
    keys = list(fact["labels"]) if "labels" in fact else [c["id"] for c in fact["cards"]]
    runs = {name: run(tmp_path, name, fact, extra, {"message": message, "tier": tier}) for name, extra in variants(keys).items()}
    base = runs["none"][0].facts["issues"]
    for name, (result, event) in runs.items():
        got = result.facts["issues"]
        full = name != "one" or fact.get("exclusive")
        if full:
            assert got.as_dict(True) | {"requests": None} == base.as_dict(True) | {"requests": None}
        else:
            assert got.states[keys[0]] == base.states[keys[0]]
        record = event["facts"]["issues"]
        assert (record["status"], record["settled"], record["complete"]) == (got.status, got.settled, got.complete)
    predicate(runs["one"][0], truth(base, keys[0]), "hit", "miss")
    empty = F if any(s == "applicable" for s in base.states.values()) else T if base.settled else U
    predicate(runs["empty"][0], empty, "empty", "some")
    hits = tuple(k for k in keys if truth(base, k) == T)
    predicate(runs["all"][0], U if not base.settled else T, hits or ("otherwise",), None)
    if fact.get("exclusive") and base.settled:
        assert sum(s == "applicable" for s in base.states.values()) <= 1 and not isinstance(base.value, tuple)


def test_exclusive_conflict_demotes_without_claiming_supersession(tmp_path):
    fact, message, tier = SCENARIOS["two_exclusive_positives"]
    result, _ = run(tmp_path, "x", fact, variants(["alpha"])["empty"], {"message": message, "tier": tier})
    issues = result.facts["issues"]
    assert result.status == "uncertain" and issues.value is None and not issues.settled
    assert issues.states["alpha"] == issues.states["beta"] == "conflict" and issues.unresolved["alpha"] == "exclusive conflict"
    assert dict(issues.superseded) == {}
    unknown, _ = run(tmp_path, "y", *SCENARIOS["positive_plus_unknown"][:1], {}, {"message": "alpha+ beta?", "tier": "silver"})
    assert unknown.facts["issues"].states == {"alpha": "conflict", "beta": "unresolved", "gamma": "ineligible"}


def test_composer_is_pure_and_index_addressed():
    state = compose((T, T, F), (T, T, F), ((), (), ()), True, (False, False, False), (None, None, None))
    assert state.values == (U, U, F) and state.states == ("conflict", "conflict", "ineligible") and not state.settled
    assert state.selected == () and state.empty == U and state.status == "uncertain"
    state = compose((T, T), (T, T), ((), (0,)), False, (False, False), (None, None))
    assert state.values == (T, F) and state.superseding == (None, 0) and state.selected == (0,) and state.empty == F
    state = compose((T, T), (U, T), ((), (0,)), False, (True, False), (("not_examined",), None))
    assert state.states == ("unexamined", "may_be_superseded") and state.reasons[1] == ("superseded",) and not state.complete
    assert compose((T,), (U,), ((),), False, (False,), (("model_error", "x"),)).status == "failed"


def test_inline_select_uses_the_same_composer(tmp_path):
    labels = {"alpha": "Alpha label", "beta": "Beta label", "gamma": "Gamma label"}
    for message in ("alpha+ beta?", "alpha! beta+", "quiet", "alpha+ gamma+"):
        saved, _ = run(tmp_path, "inline", {"select": "Which issues apply?", "labels": LABELS}, {}, {"message": message, "tier": "silver"})
        rt = Runtime(FakeScorer(oracle))
        inline = rt.ask({"message": message}, {"issues": Select("Which issues apply?", labels, eligible=lambda key: key != "gamma")})["issues"]
        fact = saved.facts["issues"]
        assert (inline.status, inline.value, inline.settled, inline.complete) == (fact.status, fact.value, fact.settled, fact.complete)
        assert dict(inline.states) == dict(fact.states) and dict(inline.unresolved) == dict(fact.unresolved)


def test_evaluation_reads_the_effective_selection(tmp_path):
    fact = LABEL_FACT | {"max_claims": 2}
    path = tmp_path / "ev.yaml"
    files.publish(path, {"name": "ev", "fields": {"message": {"type": "text", "required": True}, "tier": {"type": "text"}},
                         "reads": ["message"], "facts": {"issues": fact}})
    rows = [("alpha+ beta?", {"alpha": True, "beta": True, "gamma": False}), ("alpha+", {"alpha": True, "beta": False, "gamma": False}),
            ("alpha! beta+", {"alpha": False, "beta": True, "gamma": False})]
    files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes([{"input": {"message": m, "tier": "silver"}, "facts": {"issues": g}, "group": f"g{n}"}
                                                   for n, (m, g) in enumerate(rows)])))
    report = tools.run_test(Runtime(FakeScorer(oracle)), path)["fact_populations"]["fresh"]["issues"]
    assert report["full_gold"] == 3 and report["full_gold_settled"] == 1 and report["exact_set_correct"] == 1
