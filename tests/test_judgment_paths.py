from __future__ import annotations

import re
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_switch import Check, ConfigError, Pick, Rate, Runtime, files
from semantic_switch.compiler import compile_spec
from semantic_switch.executor import Executor
from semantic_switch.heads import LexicalHead

from semantic_testkit.scoring import FakeScorer, letters_of


def marked(state, question, width):
    wanted = re.findall(r"#(\w+)", state)
    options = letters_of(question)
    p = np.full(width, .01)
    if options:
        for description, letter in options.items():
            if any(word in description for word in wanted):
                p[ord(letter) - 65] = 1.
    else:
        p[0 if "yes" in wanted else 1] = 1.
    return list(p / p.sum())


def switch_file(tmp_path, fact, guidance=None, fields=None, name="compiled"):
    spec = {"name": name, "fields": {"message": {"type": "text", "required": True}, **(fields or {})},
            "reads": ["message"], "facts": {"f": fact}}
    if guidance:
        spec["guidance"] = {"guidance": {"text": guidance}}
    path = tmp_path / f"{name}.yaml"
    files.publish(path, spec)
    return path


MACHINES = {"a1": {"description": "alpha unit", "aliases": ["AX-1"]},
            "b2": {"description": "beta unit", "aliases": ["BX-2"]},
            "c3": {"description": "gamma unit", "aliases": ["CX-3"]}}
SHAPES = {
    "check": (Check("The customer wants money back."), {"check": "The customer wants money back."}, None, "#yes refund"),
    "guided_unstated": (Check("The customer wants money back.", unstated=True, guidance="Use only the message."),
                        {"check": "The customer wants money back.", "unstated": True}, "Use only the message.",
                        "#insufficient"),
    "pick_excludes": (Pick("Which topic?", {"billing": "billing questions", "login": "login problems"},
                           excludes=("refund requests",)),
                      {"pick": "Which topic?", "options": {"billing": "billing questions", "login": "login problems"},
                       "excludes": ["refund requests"]}, None, "#login"),
    "focused_none": (Pick("Which machine failed?", MACHINES, allow_none=True, focus="mentions"),
                     {"pick": "Which machine failed?", "options": MACHINES, "allow_none": True, "focus": "mentions"},
                     None, "AX-1 is fine #another #none"),
    "rate": (Rate("How urgent is it?", ["low", "medium", "high"]),
             {"rate": "How urgent is it?", "levels": ["low", "medium", "high"]}, None, "#high"),
}


@pytest.mark.parametrize("shape", [*SHAPES, "no_eligible"])
def test_inline_saved_compiled_share_questions_and_statuses(tmp_path, shape):
    runs = []
    if shape == "no_eligible":
        inline = Pick("Which topic?", {"billing": "billing questions", "login": "login problems"},
                      eligible=lambda option: False)
        compiled = switch_file(tmp_path, {"pick": "Which topic?", "options": {
            "billing": {"description": "billing questions", "requires": "enabled == true"},
            "login": {"description": "login problems", "requires": "enabled == true"}}},
            fields={"enabled": {"type": "bool"}})
        arms = [lambda rt: inline({"message": "#login"}, rt),
                lambda rt: rt.load(compiled)({"message": "#login", "enabled": False}).facts["f"]]
    else:
        inline, fact, guidance, text = SHAPES[shape]
        compiled = switch_file(tmp_path, fact, guidance)
        arms = [lambda rt: inline({"message": text}, rt),
                lambda rt: inline.save(tmp_path / "saved.yaml", rt=rt)({"message": text}).facts["saved"],
                lambda rt: rt.load(compiled)({"message": text}).facts["f"]]
    for arm in arms:
        scorer = FakeScorer(marked)
        with Runtime(scorer) as rt:
            result = arm(rt)
        runs.append((scorer.requests, result.status, result.value))
    assert all(run == runs[0] for run in runs)
    expected = {"check": ("ok", 1), "guided_unstated": ("needs_input", 1), "pick_excludes": ("ok", 1),
                "focused_none": ("none", 2), "rate": ("ok", 1), "no_eligible": ("none", 0)}[shape]
    assert (runs[0][1], len(runs[0][0])) == expected


@pytest.mark.parametrize("state", [{"other": "x"}, {"message": None}, "", "  "])
def test_inline_missing_field_needs_input(rt, fake, state):
    fields = None if isinstance(state, str) else ["message"]
    assert rt.check(state, "The customer wants money back.", fields=fields).status == "needs_input"
    assert rt.pick(state, "Which?", {"a": "A", "b": "B"}, fields=fields).status == "needs_input"
    assert rt.rate(state, "How urgent?", ["low", "high"], fields=fields).status == "needs_input"
    assert fake.requests == []


def test_compiled_pick_rate_transmit_excludes(tmp_path):
    for fact in ({"pick": "Which topic?", "options": {"a": "billing", "b": "login"}, "excludes": ["refund requests"]},
                 {"rate": "How urgent?", "levels": ["low", "high"], "excludes": ["refund requests"]}):
        scorer = FakeScorer(marked)
        with Runtime(scorer) as rt:
            rt.load(switch_file(tmp_path, fact))({"message": "x"})
        assert len(scorer.requests) == 1 and "Except: refund requests." in scorer.requests[0][1]


def test_select_card_guided_insufficient_evidence_needs_input(tmp_path):
    def oracle(state, question, width):
        if "What follows" in question:
            return marked("#insufficient", question, width)
        return [.97, .03]
    spec = {"name": "cards", "fields": {"report": {"type": "text", "required": True}}, "reads": ["report"],
            "guidance": {"rules": {"text": "Use only the report."}},
            "facts": {"leak": {"check": "There is a visible leak.", "unstated": True},
                      "procs": {"select": "Which procedures apply?", "cards": [
                          {"id": "hot", "applies_when": ["The unit overheats."], "excludes": ["There is a visible leak."]}]}},
            "rules": [{"id": "go", "when": 'procs has "hot"', "then": "go"}], "otherwise": "nothing"}
    path = tmp_path / "cards.yaml"
    files.publish(path, spec)
    with Runtime(FakeScorer(oracle)) as rt:
        sw = rt.load(path)
        result = sw({"report": "it runs hot"})
        assert result.status == "needs_input"
        assert result.facts["procs"].status == "needs_input"
        files.publish(path, {**spec, "rules": [{"id": "go", "when": "leak", "then": "go"}]})
        assert rt.load(path)({"report": "it runs hot"}).facts["leak"].status == "needs_input"


def test_focus_none_requires_full_stage(tmp_path):
    pick = Pick("Which machine failed?", MACHINES, allow_none=True, focus="mentions")
    path = switch_file(tmp_path, {"pick": "Which machine failed?", "options": MACHINES, "allow_none": True,
                                  "focus": "mentions"})
    for arm in (lambda rt: pick({"message": "AX-1 is fine #none #another"}, rt),
                lambda rt: rt.load(path)({"message": "AX-1 is fine #none #another"}).facts["f"]):
        scorer = FakeScorer(marked)
        with Runtime(scorer) as rt:
            result = arm(rt)
        (_, focused), (_, full) = scorer.requests
        assert "none of the above" not in focused and "another one, not listed above" in focused
        assert "none of the above" in full and result.status == "none"


def test_no_eligible_options_is_none_everywhere(tmp_path):
    options = {"billing": {"description": "billing questions", "requires": "enabled == true"},
               "login": {"description": "login problems", "requires": "enabled == true"}}
    for allow_none in (False, True):
        scorer = FakeScorer(marked)
        with Runtime(scorer) as rt:
            inline = rt.pick("#login", "Which topic?", ["billing", "login"], eligible=lambda option: False,
                             allow_none=allow_none)
            path = switch_file(tmp_path, {"pick": "Which topic?", "options": options, "allow_none": allow_none},
                               fields={"enabled": {"type": "bool"}})
            fact = rt.load(path)({"message": "#login", "enabled": False}).facts["f"]
        assert inline.status == fact.status == "none" and fact.settled and fact.source == "rule"
        assert scorer.requests == []


def test_save_invalid_definition_writes_nothing(tmp_path):
    path = tmp_path / "bad.yaml"
    with Runtime(FakeScorer()) as rt, pytest.raises(ConfigError):
        Pick("Which?", {"none": "nothing", "a": "A"}, allow_none=True).save(path, rt=rt)
    assert not path.exists()


def test_focus_mentions_requires_identities():
    with pytest.raises(ConfigError, match="identities"):
        Pick("Which?", ["a", "b"], focus="mentions")
    Pick("Which?", ["a", "b"], focus="mentions", ids=str)


def test_switch_rev_tracks_dependency_meanings():
    def rev(claim, applies):
        return compile_spec({"name": "s", "fields": {"m": {"type": "text"}}, "reads": ["m"],
                             "facts": {"refund": {"check": claim},
                                       "procs": {"select": "Which apply?", "cards": [{"id": "hot", "applies_when": [applies]}]}},
                             "rules": [{"id": "r", "when": 'refund and procs has "hot"', "then": 1}],
                             "otherwise": 0}).switch_rev
    base = rev("Wants money back.", "The unit overheats.")
    assert rev("Wants a refund.", "The unit overheats.") != base
    assert rev("Wants money back.", "The unit runs hot.") != base


def test_cost_counts_discarded_classifier_and_escape_rounds(tmp_path):
    with Runtime(FakeScorer(marked)) as rt:
        path = switch_file(tmp_path, {"pick": "Which machine failed?", "options": MACHINES, "allow_none": True,
                                      "focus": "mentions"})
        cost = rt.load(path)({"message": "AX-1 is fine #another #beta"}).cost
        assert cost["judge"] == 2 and cost["settled_by"]["judge"] == 1
        sw = rt.load(switch_file(tmp_path, {"check": "The customer wants money back."}, name="check"))
        head = LexicalHead(("true", "false"), np.array([0], dtype=np.int32), np.zeros((1, 2), dtype=np.float32),
                           np.zeros(2, dtype=np.float32))
        result = sw.run([{"message": "#yes"}], replace(sw.snapshot, heads={"f": head}))[0]
        assert result.cost["classifier"] == 1 and result.cost["judge"] == 1
        assert result.cost["settled_by"]["judge"] == 1 and result.cost["settled_by"]["classifier"] == 0


def test_switch_elapsed_includes_materialization(tmp_path, monkeypatch):
    built = Executor.result

    def slow(self, snap, call, requested=frozenset()):
        time.sleep(.05)
        return built(self, snap, call, requested)

    monkeypatch.setattr(Executor, "result", slow)
    with Runtime(FakeScorer(marked)) as rt:
        result = rt.load(switch_file(tmp_path, {"check": "The customer wants money back."}))({"message": "#yes"})
    assert result.cost["elapsed_ms"] >= 50


class CountingScorer(FakeScorer):
    def __init__(self):
        super().__init__(marked)
        self.batch_inputs = 8
        self.ctx = SimpleNamespace(submissions=0, submitted_tokens=0, n_ctx=1 << 20)

    def score(self, requests):
        self.ctx.submissions += 1
        self.ctx.submitted_tokens += sum(len(r.state) + sum(map(len, r.suffixes)) for r in requests)
        return super().score(requests)


def test_batch_native_work_not_multiplied(tmp_path):
    scorer = CountingScorer()
    with Runtime(scorer) as rt:
        sw = rt.load(switch_file(tmp_path, {"check": "The customer wants money back."}))
        results = sw.run([{"message": f"#yes {n}"} for n in range(3)]) + sw.run([{"message": "#no"}])
    batches = {id(r.cost["operation"]["native_work"]): r.cost["operation"]["native_work"] for r in results}
    assert [r.cost["operation"]["inputs"] for r in results] == [3, 3, 3, 1] and len(batches) == 2
    assert sum(work["submissions"] for work in batches.values()) == scorer.ctx.submissions
    assert sum(work["submitted_tokens"] for work in batches.values()) == scorer.ctx.submitted_tokens


@pytest.mark.parametrize("kind", ["pick", "rate"])
def test_runtime_calls_forward_excludes(kind):
    options = {"refund": "Refund the charge", "other": "Anything else"} if kind == "pick" else ("low", "high")
    excludes = ("a request quoted from someone else",)
    typed = (Pick if kind == "pick" else Rate)("What applies?", options, excludes=excludes)
    runs = []
    for call in (lambda rt: typed("Refund me, says my friend.", rt),
                 lambda rt: getattr(rt, kind)("Refund me, says my friend.", "What applies?", options, excludes=excludes)):
        scorer = FakeScorer(marked)
        with Runtime(scorer) as rt:
            result = call(rt)
        runs.append((scorer.requests, result.status, result.value))
    assert runs[0] == runs[1] and "Except: a request quoted from someone else." in runs[0][0][0][1]
