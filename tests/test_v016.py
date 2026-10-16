import json
import io
import shutil
import sys
from dataclasses import replace

import numpy as np
import pytest
import yaml

from semantic_switch import Actor, ConfigError, Model, Runtime, cli, files, tools
from semantic_switch.api import _resolve
from semantic_switch.compiler import compile_spec
from semantic_switch.examples import derive_fact_labels
from semantic_switch.heads import LexicalHead
from semantic_switch.judgment import Policy
from semantic_switch.judgment import ESCAPE
from semantic_switch.judge import Scored, Scores

from semantic_testkit.scoring import FakeScorer, letters_of
from .test_switch import ROOT, edit, t


def write(tmp_path, spec, name="s.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


def fake_rt(oracle=None):
    return Runtime(FakeScorer(oracle) if oracle else FakeScorer())


def peaked(q: str, width: int, wanted) -> list:
    p = np.full(width, 0.01)
    for desc, letter in letters_of(q).items():
        if wanted(desc):
            p[ord(letter) - 65] = 0.97
    return list(p / p.sum())


@pytest.fixture
def route(tmp_path):
    p = tmp_path / "route_ticket.yaml"
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", p)
    return p


@pytest.fixture
def procedures(tmp_path):
    for name in ("procedures.yaml", "procedures.cards.yaml"):
        shutil.copy(ROOT / "switches" / name, tmp_path / name)
    return tmp_path / "procedures.yaml"


def plant(state: str, q: str, width: int):
    s = state.lower()
    if "Claim about the input above" in q:
        hit = ("smoke" in q and "smoke" in s) or ("fluid or oil" in q and "oil" in s) or \
              ("fails to start" in q and "won't start" in s) or ("sensor fault" in q and "sensor" in s)
        return [0.97, 0.03] if hit else [0.03, 0.97]
    if "does not say" in q:
        want = "states it is true" if "oil" in s else "states it is false" if "no leak" in s else "does not say"
        return peaked(q, width, lambda d: want in d)
    return [1.0 / width] * width


def test_group_atom_does_not_settle_the_category(tmp_path):
    spec = {"name": "g", "fields": {"m": {"type": "text", "required": True}}, "reads": ["m"],
            "facts": {"topic": {"pick": "What is it about?", "options": {"billing": "b", "technical": "t",
                                                                          "account": "a"}}},
            "rules": [{"id": "spec", "when": 'topic in ["billing", "technical"]', "then": "specialist"}],
            "otherwise": "general"}
    rt = fake_rt(lambda s, q, w: [0.48, 0.48, 0.04] if "about" in q else [0.5, 0.5])
    r = rt.load(write(tmp_path, spec))({"m": "x"})
    assert r.answer == "specialist" and r.facts["topic"].value is None and not r.facts["topic"].settled


def test_resolve_all_derives_no_false_labels_for_other_rules():
    spec = {"name": "u", "fields": {"m": {"type": "text"}}, "reads": ["m"], "resolve": "all",
            "facts": {"a": {"check": "A."}, "b": {"check": "B."}},
            "rules": [{"id": "a", "when": "a"}, {"id": "b", "when": "b"}], "otherwise": "none_"}
    assert derive_fact_labels(compile_spec(spec), "b") == {"b": True}


def test_low_answer_mass_is_uncertain_inline(rt):
    r = rt.check("garbled", "The customer asks to receive money back for a charge.", min_score=0.55)
    assert (r.status, r.error) == ("uncertain", "low answer mass")


def test_cheap_paths_never_acquire_the_judge():
    class NoJudge(Runtime):
        def judge(self, role="judge"):
            raise AssertionError("judge acquired")

    rt = NoJudge({"judge": FakeScorer()})
    with pytest.raises(ValueError):
        rt.check("x", "y", min_score=2)
    assert rt.ask("x", {}) == {}
    assert rt.pick("x", "q", ["a", "b"], eligible=lambda o: False, allow_none=True).status == "none"


def test_limits_are_checked_before_describe_and_slots(rt):
    seen = []
    r = rt.pick("x", "q", [f"o{i}" for i in range(1001)], definition=lambda o: seen.append(o) or o, shortlist="lexical")
    assert r.status == "failed" and seen == []
    with pytest.raises(ConfigError, match="at most 20 levels"):
        rt.rate("x", "q", [f"l{i}" for i in range(21)])


def test_tuned_lexical_head_settles_and_untuned_defers(route):
    msg = t("please refund my charge", payment={"status": "pending"})
    head = LexicalHead(("true", "false"), np.array([], dtype=np.int32),
                       np.zeros((0, 2), dtype=np.float32), np.array([3., 0.], dtype=np.float32))
    for table, source in (({"refund_requested": Policy(0.6, 0.6)}, "classifier"), ({}, "judge")):
        sw = fake_rt().load(route)
        sw._snap = replace(sw.snapshot, heads={"refund_requested": head}, policies={"classifier": table})
        assert sw(msg).facts["refund_requested"].source == source


def test_changing_reads_makes_labels_stale(route):
    spec = files.read_yaml(route)
    before = compile_spec(spec).fact("legal_threat").meaning_rev
    spec["facts"]["legal_threat"]["reads"] = ["subject"]
    assert compile_spec(spec).fact("legal_threat").meaning_rev != before


def test_cache_key_names_execution_identity(rt):
    assert json.loads(rt.cache_key())["execution"] == rt.execution() is None


def test_model_sources_and_names_are_validated(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo: ["a-Q4_0.gguf", "b-Q4_0.gguf"])
    _resolve.cache_clear()
    with pytest.raises(ConfigError, match="exactly one"):
        _resolve("org/repo:Q4_0")
    with pytest.raises(ConfigError, match="chat"):
        Model("x", adapter="chatt")
    with pytest.raises(ConfigError, match="gpu"):
        Model("x", device="gpuu")


def test_exit_codes_and_prompts(monkeypatch, capsys):
    assert (cli.worst(["ok", "uncertain"]), cli.worst(["ok", "failed", "needs_input"]), cli.worst([])) == (2, 3, 0)
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    assert cli.prompt("Refund? ") == "y"
    out = capsys.readouterr()
    assert out.out == "" and "Refund?" in out.err


def test_mentions_atom(tmp_path):
    spec = {"name": "m", "fields": {"report": {"type": "text"}}, "reads": ["report"],
            "facts": {"x": {"check": "X."}},
            "rules": [{"id": "plain", "when": 'mentions(report, "MX-120")', "then": "a"},
                      {"id": "alias", "when": 'mentions(report, ["MX-120A", "Mixer 120 Advanced"])', "then": "b"}],
            "otherwise": "c"}
    sw = fake_rt().load(write(tmp_path, spec))
    assert sw({"report": "MX-120A failed"}).answer == "b"
    assert sw({"report": "the MX-120 failed."}).answer == "a"
    assert sw({"report": "the mixer 120 advanced is dead"}).answer == "b"
    assert sw({}).status == "needs_input" and sw({}).missing == ("report",)


MACHINES = {"MX-120": "MX-120 standard mixer", "MX-120A": {"description": "MX-120A advanced mixer",
                                                            "aliases": ["Mixer 120 Advanced"]},
            "MX-210": "MX-210 industrial mixer", "MX-300": "MX-300 compact mixer"}


def machine_oracle(state, q, width):
    if "other one broke" in state:
        return peaked(q, width, lambda d: d == ESCAPE) if ESCAPE in q else peaked(q, width, lambda d: "MX-210" in d)
    return peaked(q, width, lambda d: "MX-120A" in d)


def test_focus_shows_mentioned_candidates_plus_escape():
    rt = fake_rt(machine_oracle)
    fake = rt.models["judge"]
    r = rt.pick("MX-120 is fine; MX-120A failed.", "Which machine failed?", MACHINES, focus="mentions")
    assert (r.value, r.focus) == ("MX-120A", "mentions")
    assert ESCAPE in fake.requests[-1][1] and "MX-210" not in fake.requests[-1][1]


def test_focus_never_excludes_an_unmentioned_answer():
    rt = fake_rt(machine_oracle)
    r = rt.pick("MX-120 is fine, the other one broke.", "Which machine failed?", MACHINES, focus="mentions")
    assert (r.value, r.focus) == ("MX-210", "fallback")


def test_switch_without_rules_returns_its_facts(rt, tmp_path):
    shutil.copy(ROOT / "switches" / "refund.yaml", tmp_path / "refund.yaml")
    sw = rt.load(tmp_path / "refund.yaml")
    r = sw({"message": "Please refund my charge"})
    assert (r.status, r.answer, r.facts["refund"].value) == ("ok", None, True)
    sw.correct({"message": "Reverse it"}, fact="refund", value=True, actor=Actor('human'))
    assert sw({"message": "Reverse it"}).facts["refund"].source == "example"
    assert tools.lock(rt, tmp_path / "refund.yaml")["id"]


def leak_oracle(state, q, width):
    return plant(state.replace("leaking", "oil").replace("dry floor", "no leak"), q, width)


def test_three_way_check_reports_not_stated(tmp_path):
    rt = fake_rt(leak_oracle)
    assert rt.check("E17 again", "There is a leak.", unstated=True).status == "needs_input"
    spec = {"name": "l", "fields": {"report": {"type": "text", "required": True}}, "reads": ["report"],
            "facts": {"leak": {"check": "There is a visible leak.", "unstated": True}},
            "rules": {"inspect": "leak"}, "otherwise": "fine"}
    sw = rt.load(write(tmp_path, spec))
    silent = sw({"report": "E17 again"})
    assert silent.status == "needs_input" and silent.missing == ("fact:leak",)
    assert silent.facts["leak"].value is None and silent.facts["leak"].why == "not stated in the input"
    assert sw({"report": "dry floor, E17"}).answer == "fine"
    assert sw({"report": "it is leaking"}).answer == "inspect"


TITLES = ("startup fault", "Hydraulic leak", "sensor fault E71", "Emergency shutdown", "P0")


def test_select_judges_situation_claims_never_titles_and_returns_payloads(procedures):
    rt = fake_rt(plant)
    fake = rt.models["judge"]
    r = rt.load(procedures)({"machine": "MX-120A", "active_alarms": ["E17"], "report": "It won't start. No leak."})
    assert r.answer == "follow_procedure" and r.facts["procedures"].value == ("P02",)
    fact = r.facts["procedures"]
    assert fact.cards[0]["source"]["section"] == "7.3" and fact.complete
    assert fact.states["P01"] == "ineligible" and fact.states["P02"] == "applicable"
    assert not any(title in q for _, q in fake.requests for title in TITLES)
    assert not any("Maintenance manual" in s + q for s, q in fake.requests)
    assert r.as_dict()["facts"]["procedures"]["cards"][0]["id"] == "P02"


def test_select_supersedes_and_exclusive(procedures):
    rt = fake_rt(plant)
    r = rt.load(procedures)({"machine": "MX-120A", "active_alarms": ["E17"],
                             "report": "It won't start and there is smoke. No leak."})
    assert r.answer == "emergency_shutdown" and r.facts["procedures"].value == ("P09",)
    assert r.facts["procedures"].states["P02"] == "unexamined" and r.cost["rounds"] == 1
    edit(procedures, lambda s: s["facts"]["procedures"].update(exclusive=True))
    r = rt.load(procedures)({"machine": "MX-120A", "active_alarms": ["E71"], "report": "Sensor fault and oil below."})
    assert r.status == "uncertain" and not r.facts["procedures"].settled


def catalogue(tmp_path, cards, rules=None, **fact):
    spec = {"name": "c", "fields": {"report": {"type": "text", "required": True}, "machine": {"type": "text"}},
            "reads": ["report"],
            "facts": {"leak": {"check": "There is a visible leak.", "unstated": True},
                      "procs": {"select": "Which procedures apply?", "cards": cards, **fact}}}
    if rules is not None:
        spec |= {"rules": rules, "otherwise": "nothing"}
    return write(tmp_path, spec)


def claim_oracle(state, q, width):
    s = state.lower()
    if "does not say" in q:
        want = "states it is true" if "oil" in s else "states it is false" if "dry" in s else "does not say"
        return peaked(q, width, lambda d: want in d)
    if "overheats" in q:
        return [0.97, 0.03] if "hot" in s else [0.03, 0.97]
    if "rattles" in q:
        return [0.97, 0.03] if "rattle" in s else [0.5, 0.5] if "maybe" in s else [0.03, 0.97]
    return [0.03, 0.97]


HOT = {"id": "hot", "applies_when": ["The unit overheats."], "excludes": [{"fact": "leak"}]}
HOT2 = {"id": "hot2", "applies_when": ["The unit overheats."], "supersedes": ["hot"]}
RATTLE = {"id": "rattle", "applies_when": ["The unit rattles."], "supersedes": ["hot"]}


def test_identical_claims_are_judged_once(tmp_path):
    rt = fake_rt(claim_oracle)
    r = rt.load(catalogue(tmp_path, [HOT, HOT2]))({"report": "it runs hot, floor is dry"})
    assert r.facts["procs"].value == ("hot2",) and dict(r.facts["procs"].superseded) == {"hot": "hot2"}
    assert sum("overheats" in q for _, q in rt.models["judge"].requests) == 1


def test_unstated_exclusion_needs_input_by_card(tmp_path):
    rt = fake_rt(claim_oracle)
    r = rt.load(catalogue(tmp_path, [HOT], {"go": 'procs has "hot"'}))({"report": "it runs hot"})
    assert r.status == "needs_input" and r.missing == ("hot: fact:leak",)
    assert rt.load(catalogue(tmp_path, [HOT], {"go": 'procs has "hot"'}))({"report": "hot, oil below"}).answer == \
        "nothing"


def test_may_be_superseded_by_an_unresolved_card(tmp_path):
    rt = fake_rt(claim_oracle)
    r = rt.load(catalogue(tmp_path, [HOT, RATTLE]))({"report": "hot, maybe something, dry"})
    fact = r.facts["procs"]
    assert r.status == "uncertain" and fact.states["hot"] == "may_be_superseded" and fact.states["rattle"] == "unresolved"
    assert "may be superseded" in fact.unresolved["hot"]


def test_is_empty_needs_a_complete_search(tmp_path):
    rt = fake_rt(claim_oracle)
    rules = [{"id": "empty", "when": "procs is empty", "then": "none_apply"}]
    sw = rt.load(catalogue(tmp_path, [HOT, RATTLE], rules))
    assert sw({"report": "all quiet, dry"}).answer == "none_apply"
    assert sw({"report": "maybe something, dry"}).status == "uncertain"
    ruleless = rt.load(catalogue(tmp_path, [HOT, RATTLE]))
    assert ruleless({"report": "all quiet, dry"}).status == "none"
    assert ruleless({"report": "maybe something, dry"}).status == "uncertain"


def test_over_budget_cards_stay_unexamined(tmp_path):
    rt = fake_rt(claim_oracle)
    cards = [HOT, {"id": "rattle", "aliases": ["R-9"], "applies_when": ["The unit rattles."]}]
    sw = rt.load(catalogue(tmp_path, cards, max_claims=1))
    r = sw({"report": "R-9 is hot and dry"})
    fact = r.facts["procs"]
    assert r.status == "uncertain" and not fact.complete
    assert fact.states == {"hot": "unexamined", "rattle": "rejected"}
    assert "not examined" in fact.unresolved["hot"]
    assert fact.why == "incomplete search: hot not examined"


def test_exact_requires_rule_cards_out_without_the_model(tmp_path):
    rt = fake_rt(claim_oracle)
    card = {**HOT, "requires": 'machine == "AX-410B"'}
    r = rt.load(catalogue(tmp_path, [card], {"go": 'procs has "hot"'}))({"report": "it runs hot", "machine": "AX-410"})
    assert r.answer == "nothing" and r.facts["procs"].states == {"hot": "ineligible"} and not rt.models["judge"].requests


@pytest.mark.parametrize("cards,needle", [
    ([{**HOT, "supersedes": ["hot"]}], "cycle: hot → hot"),
    ([{**HOT, "supersedes": ["hot2"]}, HOT2], "cycle: hot → hot2 → hot"),
    ([HOT, {**HOT, "applies_when": ["x"]}], "duplicate card ids"),
    ([{**HOT, "supersedes": ["nope"]}], "unknown cards"),
    ([{**HOT, "codes": ["E17"]}], "codes"),
])
def test_catalogue_validation(tmp_path, cards, needle):
    with pytest.raises(ConfigError, match=needle):
        fake_rt().load(catalogue(tmp_path, cards))


def test_catalogue_equals_hand_written_rules(tmp_path):
    rt = fake_rt(claim_oracle)
    cat = rt.load(catalogue(tmp_path, [HOT], [{"id": "hot", "when": 'procs has "hot"', "then": "cool"}]))
    hand = {"name": "h", "fields": {"report": {"type": "text", "required": True}}, "reads": ["report"],
            "facts": {"leak": {"check": "There is a visible leak.", "unstated": True},
                      "hot": {"check": "The unit overheats."}},
            "rules": [{"id": "hot", "when": "hot and not leak", "then": "cool"}], "otherwise": "nothing"}
    sw = rt.load(write(tmp_path, hand, "h.yaml"))
    for report in ("hot and dry", "hot, oil below", "quiet and dry", "hot"):
        a, b = cat({"report": report}), sw({"report": report})
        assert (a.status, a.answer) == (b.status, b.answer)


def test_saved_judgment_answers_like_the_inline_one(tmp_path):
    from semantic_switch import Check

    rt = fake_rt(lambda s, q, w: [0.8, 0.2])
    loose = Check("The customer asks to receive money back for a charge.", min_score=0.75)
    sw = loose.save(tmp_path / "refund.yaml", rt=rt)
    assert loose("refund please", rt=rt).value is True
    assert sw({"message": "refund please"}).facts["refund"].value is True
    strict = Check("The customer asks to receive money back for a charge.")
    assert strict("refund please", rt=rt).status == "uncertain"
    assert strict.save(tmp_path / "strict.yaml", rt=rt)({"message": "refund please"}).status == "uncertain"
    assert loose.save(tmp_path / "refund.yaml", rt=rt).plan.name == "refund"
    with pytest.raises(ConfigError, match="different definition"):
        strict.save(tmp_path / "refund.yaml", rt=rt)


def test_engine_runs_on_a_minimal_non_decoder_judge(tmp_path):
    class TinyJudge:
        identity = {"source": "tiny"}
        accelerated = False
        max_slots = 20

        def fit(self, text, limit):
            return text, False

        def score_many(self, items):
            return [Scored(tuple(Scores(np.array([0.95, 0.05]) if q.kind == "check" else
                                        np.full(len(q.options), 1 / len(q.options)), None) for q in qs), 0)
                    for _, qs in items]

    rt = Runtime(TinyJudge())
    assert rt.check("x", "y").value is True
    shutil.copy(ROOT / "switches" / "refund.yaml", tmp_path / "refund.yaml")
    assert rt.load(tmp_path / "refund.yaml")({"message": "hi"}).facts["refund"].value is True


def test_threshold_choice_accepts_numpy_truths():
    from semantic_switch.tools.policy import choose

    rows = list(zip([0.97] * 30 + [0.03] * 30, np.array([True] * 30 + [False] * 30)))
    enough, theta = choose(rows, 0.95)
    assert enough and theta <= 0.97


def test_threshold_scan_takes_highest_coverage_not_first_failure():
    from semantic_switch.tools.policy import choose

    rows = list(zip([0.99] * 25 + [0.95] * 2 + [0.9] * 40, [True] * 25 + [False] * 2 + [True] * 40))
    assert choose(rows, 0.95) == (True, 0.9)


def test_policy_sides_are_validated_without_mutating_thresholds():
    from semantic_switch.tools.policy import validate
    good, bad = [(0.97, True)] * 30, [(0.97, True)] * 20 + [(0.97, False)] * 10
    assert validate(good, .95, .9, .9)['state'] == 'met'
    missed = validate(bad, .95, .9, .9)
    assert missed['state'] == 'missed' and missed['theta'] == .9
    assert missed['held_out_precision'] == pytest.approx(2/3)
    assert validate(good[:5], .95, .9, .9)['state'] == 'insufficient'
