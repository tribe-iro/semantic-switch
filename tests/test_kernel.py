import pytest

from semantic_switch.compiler import compile_spec
from semantic_switch.errors import ConfigError
from semantic_switch.judgment import Judgment, Policy, settle
from semantic_switch.kernel import F, T, U, Exact, Sem, conj, disj, evaluate, exact_state, neg, relevant
from semantic_switch.kernel import atom as A

KLEENE = [(T, T, T, T), (T, U, U, T), (T, F, F, T), (U, U, U, U), (U, F, F, U), (F, F, F, F)]


@pytest.mark.parametrize("a,b,and_,or_", KLEENE)
def test_kleene_truth_table(a, b, and_, or_):
    st = [a, b]
    assert evaluate(conj(A(0), A(1)), st) == and_
    assert evaluate(conj(A(1), A(0)), st) == and_
    assert evaluate(disj(A(0), A(1)), st) == or_
    assert evaluate(neg(A(0)), st) == {T: F, F: T, U: U}[a]


def test_false_masks_unknown_in_relevance():
    st = [U, F]
    assert evaluate(conj(A(0), A(1)), st) == F
    assert relevant(conj(A(0), A(1)), st) == set()
    assert relevant(disj(A(0), A(1)), st) == {0}


def test_exists_is_never_unknown():
    assert exact_state(Exact("x", "exists"), {}) == F
    assert exact_state(Exact("x", "exists"), {"x": 1}) == T
    assert exact_state(Exact("x", "==", 1), {}) == U


CHECK = Judgment.check("C.")
THREE = Judgment.check("C.", unstated=True)
PICK = Judgment("pick", "Which?", ("a", "b", "c"))


def test_checks_settle_their_answer_first():
    policy = Policy()
    assert settle(CHECK, [0.95, 0.05], (T, F), policy) == (T, None)
    assert settle(CHECK, [0.05, 0.95], (T, F), policy) == (F, None)
    assert settle(CHECK, [0.05, 0.95], (F, T), policy) == (T, None)
    assert settle(CHECK, [0.6, 0.4], (T, F), policy) == (U, "uncertain")
    assert settle(CHECK, [0.6, 0.4], (T, F), policy, mass=0.3) == (U, "low_answer_mass")


def test_negation_cannot_reach_a_disabled_side():
    off_false = Policy(0.9, None)
    assert settle(CHECK, [0.03, 0.97], (T, F), off_false) == (U, "uncertain")
    assert settle(CHECK, [0.03, 0.97], (F, T), off_false) == (U, "uncertain")


def test_three_way_silence_needs_evidence():
    assert settle(THREE, [0.33, 0.33, 0.34], (T, F, U), Policy()) == (U, "uncertain")
    assert settle(THREE, [0.02, 0.03, 0.95], (T, F, U), Policy()) == (U, "not_stated")
    assert settle(THREE, [0.02, 0.93, 0.05], (T, F, U), Policy()) == (F, None)


def test_set_events_use_one_symmetric_threshold():
    assert settle(PICK, [0.46, 0.46, 0.08], (T, T, F), Policy()) == (T, None)
    assert settle(PICK, [0.46, 0.46, 0.08], (F, F, T), Policy()) == (F, None)
    assert settle(PICK, [0.95, 0.05, 0.0], (T, F, F), Policy(), pending=frozenset({0})) == (U, "missing_input")


SPEC = {
    "name": "s",
    "fields": {"t": {"type": "text", "required": True}, "n": {"type": "number"},
               "e": {"type": "enum", "values": ["a", "b"]}},
    "reads": ["t"],
    "facts": {"c": {"check": "is it?"},
              "p": {"pick": "which?", "options": {"x": "X", "y": "Y", "z": "Z"}},
              "r": {"rate": "how much?", "levels": ["low", "mid", "high"]}},
    "rules": [{"when": "c", "then": 1}],
}


def compiled(when: str):
    return compile_spec({**SPEC, "rules": [{"when": when, "then": 1}]})


def test_not_equal_is_the_negated_atom():
    plan = compiled('p == "x" or p != "x"')
    assert plan.rules[0].expr == ("const", T)
    plan = compiled('p != "x"')
    assert plan.atoms == (Sem(1, (F, T, T, T)),)


def test_one_judgment_conditions_become_one_atom():
    eq = compiled('r == "mid"')
    assert eq.atoms == (Sem(2, (F, T, F)),)
    assert compiled('r >= "mid" and r < "high"').atoms == eq.atoms
    assert compiled('r > "low" and r <= "mid" and r < "high" and r >= "low"').atoms == eq.atoms
    assert compiled('r in ["low", "mid"]').atoms == (Sem(2, (T, T, F)),)
    assert settle(Judgment("rate", "q", ("low", "mid", "high")), [0.08, 0.84, 0.08], eq.atoms[0].truth, Policy())[0] == U


def test_parentheses_never_change_the_atoms():
    flat = compiled('r >= "mid" and r < "high" and c')
    for when in ('(r >= "mid" and c) and r < "high"', 'r >= "mid" and (c and r < "high")'):
        assert set(compiled(when).atoms) == set(flat.atoms)


def test_atoms_across_judgments_stay_separate():
    plan = compiled('c and r >= "mid" and r < "high"')
    assert set(plan.atoms) == {Sem(0, (T, F)), Sem(2, (F, T, F))}


def test_three_way_negation_reads_the_false_side():
    plan = compile_spec({**SPEC, "facts": {**SPEC["facts"], "c": {"check": "is it?", "unstated": True}},
                         "rules": [{"when": "not c", "then": 1}]})
    assert plan.atoms == (Sem(0, (F, T, U)),)


@pytest.mark.parametrize("when,needle", [
    ("cc", "did you mean 'c'"), ('p == "xx"', "did you mean 'x'"), ('n == "text"', "number"),
    ('e == "aa"', "did you mean 'a'"), ('c == "x"', "check fact"), ("c and (", "expected"),
    ('r >= "medium"', "not a level")])
def test_compile_errors_are_config_errors_with_hints(when, needle):
    with pytest.raises(ConfigError, match=needle):
        compiled(when)


def test_rules_shorthand_and_then_defaults_to_id():
    plan = compile_spec({**SPEC, "rules": {"yes": "c", "big": 'r == "high"'}, "otherwise": "no"})
    assert [(c.id, c.then, c.when) for c in plan.rules] == [("yes", "yes", "c"), ("big", "big", 'r == "high"')]
    plan = compile_spec({**SPEC, "rules": [{"id": "x", "when": "c"}, {"when": "not c", "then": "y"}]})
    assert [(c.id, c.then) for c in plan.rules] == [("x", "x"), ("y", "y")]


def test_meaning_rev_tracks_meaning_only():
    a = compile_spec(SPEC).facts[0].meaning_rev
    b = compile_spec({**SPEC, "rules": [{"when": "not c", "then": 2}]}).facts[0].meaning_rev
    c = compile_spec({**SPEC, "facts": {**SPEC["facts"], "c": {"check": "is it really?"}}}).facts[0].meaning_rev
    assert a == b != c


def test_one_mention_matcher_per_catalog_matches_an_independent_per_alias_reference():
    import re
    import unicodedata
    from semantic_switch import kernel

    def reference(text, identities):
        normalized = unicodedata.normalize("NFC", text)
        def hit(alias):
            body = re.escape(unicodedata.normalize("NFC", alias))
            return re.search(r"(?<![\w-])(?<!\w\.)" + body + r"(?![\w-])(?!\.\w)", normalized, re.IGNORECASE) is not None
        return [index for index, aliases in enumerate(identities) if any(hit(alias) for alias in aliases)]

    identities = [("card",), ("credit card", "cc"), ("credit",), ("card fee",), ("e.g",), ("re-fund",), ("café",), ("São Paulo",),
                  ("日本",), ("a+b",), ("ZT411",), ("zt411 printer",), ("fee",)]
    texts = ["My CREDIT CARD fee was charged", "cards and credit-card", "café in São Paulo", "pay with cc.", "e.g. a+b works",
             "re-fund vs refund", "日本 語", "zt411 printer jammed; ZT411.", "a credit card fee", "x.card", "card-holder", ""]
    kernel.mentions.cache_clear()
    for text in texts:
        assert kernel.matched_mentions(text, identities) == reference(text, identities), text
        memo = {}
        assert kernel.matched_mentions(text, identities[:5], memo) == reference(text, identities[:5])
        assert kernel.matched_mentions(text, identities, memo) == reference(text, identities)
    assert kernel.mentions.cache_info().currsize <= 3
    assert kernel.exact_state(Exact("m", "mentions", ("credit card", "fee")), {"m": "a credit card"}) == T
    assert kernel.exact_state(Exact("m", "mentions", ("fees",)), {"m": "a fee"}) == F
