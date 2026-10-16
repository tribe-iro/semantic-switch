import shutil

import numpy as np
import pytest
import yaml

from semantic_switch.artifacts import read_export
from semantic_switch import Actor, Check, ConfigError, Runtime, files, tools
from bench import quality
from bench.latency import latency
from semantic_switch.compiler import compile_spec
from semantic_switch.judge import Scored, Scores

from semantic_testkit.scoring import FakeScorer
from .test_switch import ROOT, t
from semantic_testkit.labels import envelopes


def write(tmp_path, spec, name="s.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p


RATE = {"name": "r", "fields": {"m": {"type": "text", "required": True}}, "reads": ["m"],
        "facts": {"u": {"rate": "How urgent?", "levels": ["low", "today", "high"]}}}


def peaked_rate(state, q, width):
    return [0.08, 0.84, 0.08] if width == 3 else [0.5, 0.5]


def test_rate_equality_is_judged_on_its_own_probability(tmp_path):
    rt = Runtime(FakeScorer(peaked_rate))
    ruled = rt.load(write(tmp_path, {**RATE, "rules": {"t": 'u == "today"'}, "otherwise": "no"}))({"m": "x"})
    assert ruled.status == "uncertain" and ruled.facts["u"].value is None
    ranged = rt.load(write(tmp_path, {**RATE, "rules": {"t": 'u >= "today" and u < "high"'}, "otherwise": "no"},
                           "b.yaml"))({"m": "x"})
    assert ranged.status == "uncertain"


def test_ruleless_switch_is_ok_only_when_its_outputs_settle(tmp_path):
    rt = Runtime(FakeScorer(peaked_rate))
    r = rt.load(write(tmp_path, RATE))({"m": "x"})
    assert r.status == "uncertain" and r.uncertain == ("u",) and not r.facts["u"].settled


def test_three_way_lean_towards_silence_is_uncertain():
    rt = Runtime(FakeScorer(lambda s, q, w: [0.33, 0.33, 0.34] if w == 3 else [0.5, 0.5]))
    assert rt.check("x", "There is a leak.", unstated=True).status == "uncertain"
    rt = Runtime(FakeScorer(lambda s, q, w: [0.02, 0.03, 0.95] if w == 3 else [0.5, 0.5]))
    assert rt.check("x", "There is a leak.", unstated=True).status == "needs_input"


def test_named_claims_obey_the_catalogue_budget(tmp_path):
    cards = [{"id": f"c{i}", "applies_when": [{"fact": f"f{i}"}]} for i in range(3)]
    spec = {**RATE, "facts": {**{f"f{i}": {"check": f"Claim {i}."} for i in range(3)},
                              "p": {"select": "Which?", "cards": cards, "max_claims": 1}},
            "rules": {"any": "not p is empty"}, "otherwise": "none"}
    fake = FakeScorer(lambda s, q, w: [0.97, 0.03])
    Runtime(fake).load(write(tmp_path, spec))({"m": "x"})
    assert len(fake.requests) == 1


def test_saved_judgment_takes_the_same_bare_string(tmp_path):
    rt = Runtime(FakeScorer())
    refund = Check("The customer asks to receive money back for a charge.")
    sw = refund.save(tmp_path / "refund.yaml", rt=rt)
    assert refund("please refund", rt=rt).value is True and sw("please refund").facts["refund"].value is True


def test_inline_requires_is_rejected(rt):
    with pytest.raises(ConfigError, match="requires is exact eligibility"):
        rt.pick("x", "q", {"a": {"description": "A", "requires": "x == 1"}, "b": "B"})


def test_rendering_changes_change_meaning():
    spec = {**RATE, "fields": {"m": {"type": "text", "required": True, "max_chars": 100}}}
    a = compile_spec(spec).fact("u").meaning_rev
    spec["fields"]["m"]["max_chars"] = 1
    assert compile_spec(spec).fact("u").meaning_rev != a


def test_shortlisted_pick_never_answers_none():
    def none_wins(s, q, w):
        p = np.full(w, 0.001)
        p[w - 1] = 1
        return list(p / p.sum())

    rt = Runtime(FakeScorer(none_wins))
    r = rt.pick("option 7 please", "Which option?", [f"option {i}" for i in range(30)], shortlist="lexical",
                allow_none=True)
    assert r.status == "uncertain" and not r.complete and len(r.removed["not_examined"]) == 11


def test_malformed_judge_fails_loudly():
    class Short:
        identity = {"source": "short"}
        accelerated = False
        max_slots = 20

        def fit(self, text, limit):
            return text, False

        def score_many(self, items):
            return [Scored((Scores(np.array([0.97, 0.03]), None),), 0) for _ in items]

    with pytest.raises(TypeError, match="1 scores for 2 questions"):
        Runtime(Short()).ask("x", {"a": Check("A."), "b": Check("B.")})


def test_correct_reports_its_scope(rt, tmp_path):
    shutil.copy(ROOT / "switches" / "refund.yaml", tmp_path / "refund.yaml")
    sw = rt.load(tmp_path / "refund.yaml")
    assert sw.correct({"message": "x"}, fact="refund", value=True, actor=Actor('human'))["scope"] == "this exact input"


def test_fact_level_specialize_fails_compilation(tmp_path):
    shutil.copy(ROOT / "switches" / "refund.yaml", tmp_path / "refund.yaml")
    path = tmp_path / "refund.yaml"
    rt = Runtime(FakeScorer())
    rt.load(path).correct({"message": "refund"}, fact="refund", value=True, actor=Actor('human'))
    spec = yaml.safe_load(path.read_text())
    spec["facts"]["refund"]["specialize"] = "lexical"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ConfigError, match="specialize"):
        tools.lock(rt, path)


def test_export_rows_carry_runtime_token_ids(tmp_path):
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", tmp_path / "route_ticket.yaml")
    path = tmp_path / "route_ticket.yaml"
    rt = Runtime(FakeScorer())
    report = tools.export(rt, path, split=None, native=True)
    manifest, rows = read_export(report["path"], fitting=False, native=True)
    assert report["rows"] == len(rows) == 2
    assert manifest["split"] == "all" and manifest["counts"]["observations"] == len({r["id"] for r in rows})
    judge = rt.judge()
    row = rows[0]["presentations"][0]
    assert judge.scorer.detokenize(row.input).startswith("<input>")
    assert rows[0]["fact"] == "legal_threat" and row.label in (0, 1) and rows[0]["split"] in ("tune", "check")
    assert row.answers.slots and all(isinstance(t, int) for t in row.question)


def test_quality_probe_sets_run(rt):
    pos, comp = quality.positions(rt), quality.comparative(rt)
    assert set(pos["top1_by_slot"]) == {1, 10, 20} and comp["items"] == 24


def test_missing_switch_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Runtime(FakeScorer()).load(tmp_path / "nope.yaml")


def test_nan_answer_mass_is_rejected_at_the_boundary():
    class NanMass:
        identity = {"source": "nan"}
        accelerated = False
        max_slots = 20

        def fit(self, text, limit):
            return text, False

        def score_many(self, items):
            return [Scored(tuple(Scores(np.array([0.97, 0.03]), float("nan")) for _ in qs), 0) for _, qs in items]

    with pytest.raises(TypeError, match="answer mass"):
        Runtime(NanMass()).check("x", "A.")


def test_failed_export_keeps_the_previous_file(tmp_path):
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", tmp_path / "route_ticket.yaml")

    class Broken(FakeScorer):
        def tokenize(self, *a, **k):
            raise RuntimeError("tokenizer down")

    out = tmp_path / "rows.jsonl"
    out.write_text("previous\n")
    with pytest.raises(RuntimeError):
        tools.export(Runtime(Broken()), tmp_path / "route_ticket.yaml", str(out), native=True)
    assert out.read_text() == "previous\n"
    with pytest.raises(ConfigError, match="collides"):
        tools.export(Runtime(FakeScorer()), tmp_path / "route_ticket.yaml",
                     str(tmp_path / "route_ticket.yaml"))


def test_export_keeps_groups_and_rejects_conflicts(tmp_path):
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", tmp_path / "route_ticket.yaml")
    path = tmp_path / "route_ticket.yaml"
    rt = Runtime(FakeScorer())
    sw = rt.load(path)
    sw.correct(t("refund please"), fact="refund_requested", value=True, actor=Actor('human'))
    files.append(tmp_path / "route_ticket.test.jsonl",
                       files.jsonl(envelopes([{"input": t("refund please"), "facts": {"refund_requested": True}, "group": "g1"}])))
    rows = [r for r in read_export(tools.export(rt, path, split=None)["path"], fitting=False)[1] if r["fact"] == "refund_requested"]
    assert len({r["group"] for r in rows if r["label"] == 0}) == 1
    assert rows[0]["taught"] and rows[0]["text"] and rows[0]["claim"]
    files.append(tmp_path / "route_ticket.test.jsonl",
                       files.jsonl(envelopes([{"input": t("refund please"), "facts": {"refund_requested": False}}])))
    with pytest.raises(ConfigError, match="conflicting labels"):
        tools.export(rt, path)


def test_inline_and_saved_focus_share_the_escape_decision(tmp_path):
    from semantic_testkit.scoring import letters_of

    machines = {"MX-120": "MX-120 standard mixer", "MX-120A": "MX-120A advanced mixer",
                "MX-210": "MX-210 industrial mixer"}

    def oracle(state, q, width):
        if "another one" in q:
            return "lowmass"
        p = np.full(width, 0.01)
        for desc, letter in letters_of(q).items():
            if "MX-120A" in desc:
                p[ord(letter) - 65] = 0.97
        return list(p / p.sum())

    rt = Runtime(FakeScorer(oracle))
    inline = rt.pick("MX-120A failed", "Which machine failed?", machines, focus="mentions")
    spec = {"name": "m", "fields": {"r": {"type": "text", "required": True}}, "reads": ["r"],
            "facts": {"m": {"pick": "Which machine failed?", "options": machines, "focus": "mentions"}}}
    saved = rt.load(write(tmp_path, spec))("MX-120A failed")
    assert (inline.status, inline.value, inline.focus) == ("ok", "MX-120A", "fallback")
    assert (saved.status, saved.facts["m"].value) == ("ok", "MX-120A")


def test_bench_without_inputs_explains_itself(tmp_path):
    spec = {"name": "e", "fields": {"r": {"type": "text"}}, "reads": ["r"], "facts": {"c": {"check": "C."}}}
    with pytest.raises(ConfigError, match="no test inputs"):
        latency(Runtime(FakeScorer()), write(tmp_path, spec), n=1)
