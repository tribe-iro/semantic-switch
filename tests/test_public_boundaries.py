from __future__ import annotations

import json

import pytest

from semantic_switch import Check, ConfigError, Pick, Rate, Runtime, Select, files
from semantic_switch.cli import question_specs
from semantic_switch.compiler import compile_spec
from semantic_switch.results import thaw

from semantic_testkit.scoring import FakeScorer, letters_of
from semantic_testkit.labels import envelopes

OPTIONS = {"AX-1": {"description": "First unit", "excludes": ["spare parts"], "aliases": ["ax one"]}, "AX-2": "Second unit"}
LABELS = {"refund": {"description": "Money back", "excludes": ["store credit"]}, "delivery": "Shipping"}
QUESTIONS = {
    "claim": ({"check": "Refund requested.", "excludes": ["store credit"], "unstated": True},
              Check("Refund requested.", excludes=("store credit",), unstated=True)),
    "unit": ({"pick": "Which unit?", "options": OPTIONS, "excludes": ["accessories"]},
             Pick("Which unit?", OPTIONS, excludes=("accessories",))),
    "level": ({"rate": "How severe?", "levels": ["low", "high"], "excludes": ["tone"]}, Rate("How severe?", ("low", "high"), excludes=("tone",))),
    "issues": ({"select": "Which issues?", "labels": LABELS, "excludes": ["greetings"]}, Select("Which issues?", LABELS, excludes=("greetings",))),
}


def asked(fake):
    return sorted(q.split("<|im_end|>")[0] for _, q in fake.requests)


def saved(tmp_path, facts, name="s"):
    path = tmp_path / f"{name}.yaml"
    files.publish(path, {"name": name, "fields": {"message": {"type": "text", "required": True}}, "reads": ["message"], "facts": facts})
    return path


@pytest.mark.parametrize("name", sorted(QUESTIONS))
def test_sdk_cli_and_saved_definitions_present_the_same_meaning(tmp_path, name):
    authored, sdk = QUESTIONS[name]
    files.publish(tmp_path / "questions.yaml", {name: authored})
    cli = question_specs(tmp_path / "questions.yaml")[name]
    assert cli == sdk and cli.describe() == sdk.describe()
    sent = []
    for item in (sdk, cli):
        fake = FakeScorer()
        Runtime(fake).ask("case", {name: item})
        sent.append(asked(fake))
    fake = FakeScorer()
    Runtime(fake).load(saved(tmp_path, {name: authored}), draft=True).score_facts([{"message": "case"}])
    assert sent[0] == sent[1] == asked(fake) and sent[0]


@pytest.mark.parametrize("change", [{"aliases": "ax"}, {"excludes": [""]}, {"description": ""}, {"requires": "x"}, {"typo": "x"}])
def test_option_validation_is_shared(tmp_path, change):
    options = {"AX-1": {"description": "First unit", **change}, "AX-2": "Second unit"}
    with pytest.raises(ConfigError):
        Pick("Which unit?", options)
    files.publish(tmp_path / "questions.yaml", {"unit": {"pick": "Which unit?", "options": options}})
    with pytest.raises(ConfigError, match="questions.yaml:unit"):
        question_specs(tmp_path / "questions.yaml")
    if "requires" not in change:
        with pytest.raises(ConfigError):
            compile_spec({"name": "s", "facts": {"unit": {"pick": "Which unit?", "options": options}}})


@pytest.mark.parametrize("make", [lambda: Rate("q", ["low"]), lambda: Pick("q", {"a": "A"}, focus="names"),
                                  lambda: Pick("q", ["a"], shortlist="vector"), lambda: Pick("q", "ab"),
                                  lambda: Pick("q", {"none": "x", "a": "A"}, allow_none=True), lambda: Select("q", LABELS, max_claims=0)])
def test_construction_validates_once_without_prepare(make):
    with pytest.raises(ConfigError):
        make()


class Unloadable(FakeScorer):
    def tokenize(self, text, special=False):
        raise AssertionError("describe touched the model")


def test_describe_is_deterministic_authored_and_model_free(tmp_path):
    path = saved(tmp_path, {"claim": QUESTIONS["claim"][0], "issues": QUESTIONS["issues"][0],
                            "procs": {"select": "Which procedures?", "cards": [{"id": "hot", "applies_when": ["It overheats."]},
                                                                               {"id": "cold", "applies_when": [{"fact": "claim"}], "supersedes": ["hot"]}]}})
    rt = Runtime(Unloadable())
    first, second = rt.load(path, draft=True).describe(), rt.load(path, draft=True).describe()
    assert first == second and json.loads(json.dumps(thaw(first))) == thaw(first)
    assert first["facts"]["claim"]["address"] == ("s", "claim") and first["facts"]["claim"]["definition"]["excludes"] == ("store credit",)
    assert set(first["facts"]["issues"]["members"]) == {"refund", "delivery"} and len(first["facts"]["procs"]["claims"]["cold"]) == 1
    assert first["facts"]["claim"]["revision"] == rt.load(path, draft=True).plan.fact("claim").meaning_rev
    with pytest.raises(TypeError):
        first["facts"]["claim"]["kind"] = "pick"
    for _, item in QUESTIONS.values():
        assert item.describe() == type(item)(**{k: getattr(item, k) for k in item.__dataclass_fields__ if item.__dataclass_fields__[k].init}).describe()
    assert Check("A.", excludes=("b",)).describe()["revision"] != Check("A.").describe()["revision"]
    dynamic = Pick("Which?", [object(), object()], definition=str, eligible=bool).describe()
    assert dynamic["definition"]["candidates"] == 2 and dynamic["definition"]["callbacks"] == ("definition", "eligible")
    with pytest.raises(ConfigError, match="does not save"):
        Pick("Which?", ["a", "b"], eligible=bool).fact()


def escaping(state, question, width):
    letters = letters_of(question)
    p = [.01] * width
    target = "another one, not listed above" if "B) another one" in question else "Second unit"
    p[ord(next(letter for desc, letter in letters.items() if desc.startswith(target))) - 65] = .97
    return p


def test_focus_and_fallback_presentations_match_the_transport(tmp_path):
    item = Pick("Which unit?", OPTIONS, focus="mentions")
    fake = FakeScorer(escaping)
    result = Runtime(fake).ask("AX-1 is broken", {"unit": item})["unit"]
    assert result.value == "AX-2" and [r["stage"] for r in result.presentations] == ["mentions", "fallback"]
    for record, (_, question) in zip(result.presentations, fake.requests):
        assert {a["answer"]: a["option"] for a in record["answers"]} == {v: k for k, v in letters_of(question).items()}
    assert result.as_dict(True)["presentations"][0]["answers"][-1] == {"answer": "B", "key": None, "option": "another one, not listed above"}
    fake = FakeScorer(escaping)
    fact = Runtime(fake).load(saved(tmp_path, {"unit": {"pick": "Which unit?", "options": OPTIONS, "focus": "mentions"}}),
                                         draft=True)({"message": "AX-1 is broken"}).facts["unit"]
    assert thaw(fact.presentations) == thaw(result.presentations)
    check = Runtime(FakeScorer()).ask("case", {"c": Check("Refund requested.")})["c"]
    assert check.as_dict(True)["presentations"] == [{"stage": "full", "answers": [{"answer": "true", "key": "true", "option": None},
                                                                                 {"answer": "false", "key": "false", "option": None}]}]


def test_lexical_example_runs_through_public_tools(tmp_path):
    pytest.importorskip("sklearn")
    from examples.lexical_head import run
    path = saved(tmp_path, {"refund": {"check": "The customer asks to receive money back for a charge."}}, "refund")
    files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes([
        {"input": {"message": f'{"please refund my payment" if i % 2 else "hello thanks friend"} number {i}'},
         "facts": {"refund": bool(i % 2)}, "group": str(i)} for i in range(240)])))
    with Runtime(FakeScorer()) as rt:
        report, tested = run(rt, path, "refund", tmp_path / "bundle")
        assert report["heads"] == ["refund"] and tested
        assert rt.load(path)("please refund my payment").facts["refund"].value is True
