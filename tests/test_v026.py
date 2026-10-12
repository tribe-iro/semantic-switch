import json
import shutil
import warnings

import numpy as np
import pytest
import yaml

from semantic_switch.artifacts import read_export
from semantic_switch.hosted import Reply
from semantic_switch import ConfigError, Model, Runtime, files, judge_from, tools
from semantic_switch.judge import Scored, Scores, scored
from semantic_switch.judgment import Judgment, Policy, settle
from semantic_switch.kernel import F, T, U
from semantic_switch.protocols import Question
from semantic_switch.tools.data import Labels
from semantic_switch.tools.lock import side_rows

from .conftest import FakeScorer
from .test_switch import ROOT, t
from .test_tools import write_test_file
from .test_v016 import fake_rt, plant
from .conftest import envelopes


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "route_ticket.yaml"
    shutil.copy(ROOT / "switches" / "route_ticket.yaml", p)
    return p


class Counting(FakeScorer):
    def __init__(self, batch_inputs=1):
        super().__init__(accelerated=batch_inputs > 1)
        self.batch_inputs = batch_inputs
        self.inputs = batch_inputs
        self.calls = 0

    def score(self, requests):
        self.calls += 1
        return super().score(requests)


INPUTS = [t("my lawyer will file a lawsuit"), t("please refund my charge", payment={"status": "pending"}),
          t("the app shows an error today"), t("just saying hi"), t("refund please", payment={"status": "settled"}),
          t("login broken now")]


def outcomes(results):
    return [(r.status, r.answer, {n: f.value for n, f in r.facts.items()}) for r in results]


def test_batch_runs_rounds_across_inputs_with_identical_results(path):
    stream, batch = Counting(1), Counting(8)
    a = list(Runtime({"judge": stream}).load(path).many(INPUTS))
    b = Runtime({"judge": batch}).load(path).run(INPUTS)
    assert outcomes(a) == outcomes(b)
    assert batch.calls < stream.calls and batch.calls <= max(r.cost["rounds"] for r in b) + 2


def test_elapsed_and_amortized_time(path):
    rt = Runtime({"judge": Counting(8)})
    results = rt.load(path).run(INPUTS[:4])
    assert {r.cost["elapsed_ms"] for r in results} == {results[0].cost["elapsed_ms"]}
    assert results[0].cost["amortized_ms"] == pytest.approx(results[0].cost["elapsed_ms"] / 4, abs=0.01)
    one = rt.load(path)(INPUTS[0])
    assert one.cost["elapsed_ms"] == one.cost["amortized_ms"]


def test_cpu_windows_hold_one_input(path):
    fake = Counting(1)
    rt = Runtime({"judge": fake})
    assert rt.batch_inputs() == 1 and rt.judge().batch_inputs == 1
    rt.load(path).run(INPUTS[:3])
    assert fake.calls >= 3


def test_negation_respects_a_disabled_side():
    j = Judgment.check("A.")
    assert settle(j, [0.03, 0.97], (T, F), Policy(0.9, None))[0] == U
    assert settle(j, [0.03, 0.97], (F, T), Policy(0.9, None))[0] == U
    assert settle(j, [0.97, 0.03], (F, T), Policy(0.9, None))[0] == F


def test_low_mass_answers_never_count_as_accepted_in_tuning():
    fact = type("F", (), {"kind": "check"})
    rows = side_rows(fact, [([0.99, 0.01], 0.2, frozenset(), 1), ([0.99, 0.01], 0.9, frozenset(), 0)])
    assert rows["true"] == [(-1.0, False), (0.99, True)]


def test_collector_merges_and_rejects_conflicts(rt, path):
    sw = rt.load(path)
    sw.correct(t("refund please"), fact="refund_requested", value=True)
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes([
        {"input": t("refund please"), "facts": {"refund_requested": True}, "group": "a"},
        {"input": t("refund please now"), "facts": {"refund_requested": True}, "group": "a"},
        {"input": t("refund please now"), "facts": {"refund_requested": True}, "group": "b"}]))
    labels = Labels(rt.load(path, draft=True))
    rows = labels.examples("refund_requested")
    assert len({e.group for e in rows}) == 1 and len({e.split for e in rows}) == 1
    assert any(e.taught for e in rows) and any("share an input" in w for w in labels.warnings)
    files.append(path.with_name("route_ticket.test.jsonl"),
                       envelopes([{"input": t("refund please"), "facts": {"refund_requested": False}}]))
    with pytest.raises(ConfigError, match="conflicting labels"):
        Labels(rt.load(path, draft=True))


def test_taught_inputs_are_excluded_from_validation(rt, path):
    items = write_test_file(path, 60)
    labels = Labels(rt.load(path, draft=True))
    held = labels.examples("legal_threat", "check")[0]
    rt.load(path).correct(held.data, fact="legal_threat", value=held.label == 0)
    tools.lock(rt, path)
    data = json.loads(path.with_name("route_ticket.lock.json").read_text())
    member = data["facts"]["legal_threat"]["membership"]
    assert held.projection not in member["validated"] and held.projection not in member["tuned"]
    assert len(items) and member["validated"]


def test_card_exclusion_must_be_a_three_way_check(tmp_path):
    for name in ("procedures.yaml", "procedures.cards.yaml"):
        shutil.copy(ROOT / "switches" / name, tmp_path / name)
    path = tmp_path / "procedures.yaml"
    rt = Runtime({"judge": FakeScorer()})
    spec = yaml.safe_load(path.read_text())
    spec["facts"]["dry"] = {"check": "The floor is dry.", "reads": ["report"]}
    cards = yaml.safe_load((tmp_path / "procedures.cards.yaml").read_text())
    cards[0]["excludes"] = [{"fact": "dry"}]
    (tmp_path / "procedures.cards.yaml").write_text(yaml.safe_dump(cards, sort_keys=False))
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    with pytest.raises(ConfigError, match="three-way"):
        rt.load(path)
    cards[0]["excludes"] = [{"fact": "leak"}]
    (tmp_path / "procedures.cards.yaml").write_text(yaml.safe_dump(cards, sort_keys=False))
    rt.load(path)


def test_card_payloads_are_frozen(tmp_path):
    for name in ("procedures.yaml", "procedures.cards.yaml"):
        shutil.copy(ROOT / "switches" / name, tmp_path / name)
    rt = fake_rt(plant)
    r = rt.load(tmp_path / "procedures.yaml")({"machine": "MX-120A", "active_alarms": ["E17"],
                                               "report": "It won't start. No leak."})
    card = r.facts["procedures"].cards[0]
    with pytest.raises(TypeError):
        card["payload"]["instructions"] = "x"
    json.dumps(r.as_dict())


def test_export_defaults_to_the_tune_split_with_a_manifest(path):
    write_test_file(path, 30)
    rt = Runtime({"judge": FakeScorer()})
    report = tools.export(rt, path)
    manifest, rows = read_export(report["path"])
    assert report["split"] == "tune" and rows and all(r["split"] == "tune" for r in rows)
    assert [{k: v for k, v in entry.items() if k != "sha256"} for entry in manifest["rows"]] == [{k: r[k] for k in ("id", "fact", "projection", "group", "component", "fold", "split", "meaning_rev", "family_seeds")} for r in rows]
    assert all(r["text"] and r["claim"] for r in rows)


def test_judge_identity_has_no_runtime_adapter_fields(tmp_path):
    base = tmp_path / "m.gguf"
    base.write_bytes(b"model")
    ident = Runtime({"judge": Model(str(base), adapter="qwen3")}).identity("judge")
    assert ident["sha256"] == files.sha256_file(str(base)) and not {"lora_sha256", "lora_file", "lora_scale"} & set(ident)
    with pytest.raises(TypeError):
        Model(str(base), lora=str(base))


def test_receipt_marks_judge_training(rt, path, tmp_path, monkeypatch):
    from importlib import import_module
    module = import_module("semantic_switch.tools.lock")
    write_test_file(path, 60)
    labels = Labels(rt.load(path, draft=True))
    held = labels.examples("legal_threat", "check", taught=False)
    monkeypatch.setattr(module, "specialization", lambda rt: (None, [
        {"id": held[0].id, "fact": "legal_threat", "projection": held[0].projection, "group": held[0].group,
         "family_seeds": labels.seeds(held[0].group), "uses": ["adaptation", "selection"], "export_id": "a" * 64,
         "source": "fixture.export.jsonl", "task_family": "fixture"}]))
    report = tools.lock(rt, path)
    assert any("not independent" in w for w in report["warnings"])


def test_confidence_outside_range_breaks_the_contract():
    class Bad:
        identity = {"source": "bad"}

        def score_many(self, items):
            return [Scored((Scores(np.array([0.9, 0.1]), None, None, 1.5),), 0) for _ in items]

    with pytest.raises(TypeError, match="confidence"):
        scored(Bad(), [("x", [Question("check", "A.")])])


def test_contrast_pairs_and_invariant_groups(rt, path):
    files.append(path.with_name("route_ticket.test.jsonl"), envelopes([
        {"input": t("my lawyer will file a lawsuit"), "facts": {"legal_threat": True}, "pair": "p1"},
        {"input": t("my friend mentioned a lawyer"), "facts": {"legal_threat": False}, "pair": "p1"},
        {"input": t("lawsuit coming from my lawyer will"), "facts": {"legal_threat": True}, "group": "g"},
        {"input": t("my lawyer will sue, lawsuit"), "facts": {"legal_threat": True}, "group": "g"}]))
    report = tools.run_test(rt, path)
    fact = report["facts"]["legal_threat"]
    assert fact["pairs"] == fact["pairs_both_correct"] == 1
    assert fact["invariant_groups"] == fact["invariant_consistent"] == 1
    assert "contrast pairs" in tools.explain_test(report)


class JevTransport:
    def __init__(self, model="typesafe/jev-1.13"):
        self.model = model

    async def __call__(self, url, body, timeout):
        if "refuse" in body["state"]:
            return Reply(200, {"model": self.model, "refusal": "policy"})
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.96 if "lawyer" in body["state"] and "legal" in q["instructions"] else 0.03}
            else:
                keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
                answers[name] = {"type": q["type"], "confidence": 0.8, "probabilities": {
                    k: 0.9 if i == 0 else 0.1 / (len(keys) - 1) for i, k in enumerate(keys)}}
        return Reply(200, {"model": self.model, "answers": answers, "usage": {"prompt_tokens": 7}})


def test_structured_judge_scores_refuses_and_tracks_the_returned_model(path):
    send = JevTransport()
    judge = judge_from("openrouter:typesafe/jev-1.13", transport=send)
    out = scored(judge, [("my lawyer", [Question("check", "The customer threatens legal action."), Question("pick", "Q?", ("a", "b", "c"))]),
                         ("refuse this", [Question("check", "A.")])])
    assert np.allclose(out[0].scores[0].probs, [0.96, 0.04]) and np.allclose(out[0].scores[1].probs, [0.9, 0.05, 0.05])
    assert out[0].scores[1].confidence == 0.8 and out[0].tokens == 7 and out[1].scores[0].error.startswith("hosted refusal")
    assert judge.returned == send.model
    send.model = "typesafe/jev-1.13-2026-11-01"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        judge.score_many([("x", [Question("check", "A.")])])
    assert not caught and send.model in judge.observed_models
    r = Runtime({"judge": judge_from("openrouter:typesafe/jev-1.13", transport=JevTransport())}).load(path)(t("my lawyer will file a lawsuit"))
    assert r.answer == "legal" and r.facts["legal_threat"].source == "judge"
    with pytest.raises(ConfigError, match="local tokenizer"):
        tools.export(Runtime({"judge": judge_from("openrouter:typesafe/jev-1.13", transport=JevTransport())}), path, native=True)


def test_read_key_accepts_assignments(tmp_path):
    from bench.credentials import read_key

    f = tmp_path / "k"
    f.write_text('export KEY="abc123"\n')
    assert read_key(f) == "abc123"
