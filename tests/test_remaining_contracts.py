from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_switch import Actor, Check, ConfigError, Guidance, Model, Pick, Runtime, files, tools
from semantic_switch import backend
from semantic_switch.cli import question_specs
from semantic_switch.judge import Scored, Scores
from semantic_switch.judgment import Policy
from semantic_switch.tools.evaluate import contrasts
from semantic_testkit.scoring import FakeScorer


def definition(tmp_path, **changes):
    path = tmp_path / "decision.yaml"
    spec = {"name": "decision", "guidance": {"terms": {"text": "Store credit is not a refund."}},
            "fields": {"message": {"type": "text", "required": True}}, "reads": ["message"],
            "facts": {"refund": {"check": "Refund requested.", "unstated": True}}}
    spec.update(changes)
    files.publish(path, spec)
    return path


@pytest.mark.parametrize("name,value", [("n_ctx", 2**32), ("n_batch", 2**31), ("threads", 2**31), ("prefix_slots", 2**32)])
def test_native_integer_overflow_rejected_before_resolution(monkeypatch, name, value):
    monkeypatch.setattr("semantic_switch.api._resolve", lambda _: pytest.fail("resolved invalid model"))
    with pytest.raises(ConfigError, match="native integer limit"):
        Model("unresolved:model.gguf", **{name: value})


@pytest.mark.parametrize("settings", [{"n_batch": 2**32}, {"threads": 2**31}, {"n_ctx": 2**32}])
def test_direct_backend_integer_overflow_rejected_before_weights(monkeypatch, settings):
    monkeypatch.setattr(backend, "Weights", lambda *args: pytest.fail("allocated invalid model"))
    with pytest.raises(ValueError, match="native integer limit"):
        backend.LlamaScorer("unresolved.gguf", **settings)


def test_native_derived_sequence_overflow_rejected(monkeypatch):
    monkeypatch.setattr(backend, "Weights", lambda *args: pytest.fail("allocated invalid model"))
    with pytest.raises(ValueError, match="sequence slots"):
        backend.LlamaScorer("unresolved.gguf", prefix_slots=2**32-1)


def test_native_integer_boundaries():
    for name, maximum in (("n_ctx", 2**32-1), ("n_batch", 2**31-1), ("threads", 2**31-1)):
        backend.validate_native_integer(name, maximum)
        with pytest.raises(ValueError):
            backend.validate_native_integer(name, maximum + 1)


def test_cli_question_guidance_matches_sdk(tmp_path):
    path = tmp_path / "questions.yaml"
    files.publish(path, {"refund": {"check": "Refund requested.", "guidance": "Store credit is not a refund.", "unstated": True}})
    item = question_specs(path)["refund"]
    sdk = Check("Refund requested.", unstated=True, guidance="Store credit is not a refund.")
    with Runtime(FakeScorer(lambda *args: [.99, .005, .005])) as rt:
        assert rt.ask("case", {"refund": item})["refund"].as_dict() == rt.ask("case", {"refund": sdk})["refund"].as_dict()


@pytest.mark.parametrize("spec", [{}, {"x": "bad"}, {"x": {"check": "x", "pick": "y"}},
    {"x": {"check": "x", "typo": True}}, {"x": {"rate": "x", "levels": "bad"}},
    {"x": {"pick": "x", "options": {"a": "A"}, "focus": "bad"}}, {"x": {"check": "x", "unstated": "false"}}])
def test_cli_question_kind_and_keys_are_strict(tmp_path, spec):
    path = tmp_path / "questions.yaml"
    files.publish(path, spec)
    with pytest.raises(ConfigError, match="questions.yaml"):
        question_specs(path)


def test_cli_question_rejects_callbacks_before_model_load(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from semantic_switch import cli
    path = tmp_path / "questions.yaml"
    files.publish(path, {"x": {"pick": "Choose.", "options": {"a": "A"}, "eligible": "callback"}})
    monkeypatch.setattr(cli, "session", lambda: pytest.fail("loaded model before validation"))
    result = CliRunner().invoke(cli.app, ["ask", str(path)], input="case")
    assert result.exit_code != 0 and isinstance(result.exception, ConfigError)


def test_cli_explicit_empty_guidance_replaces_operation_binding(tmp_path):
    path = tmp_path / "questions.yaml"
    files.publish(path, {"inherit": {"check": "Claim."}, "empty": {"check": "Claim.", "guidance": []}})
    with Runtime(FakeScorer(lambda *args: [.99, .01])) as rt:
        results = rt.ask("case", question_specs(path), guidance="Definitions.")
    assert results["inherit"].guidance and not results["empty"].guidance


def test_cli_pick_focus_matches_sdk(tmp_path):
    path = tmp_path / "questions.yaml"
    options = {"AX-1": "first", "AX-2": "second"}
    files.publish(path, {"x": {"pick": "Choose.", "options": options, "focus": "mentions"}})
    with Runtime(FakeScorer(lambda *args: [.99, .01])) as rt:
        cli = rt.ask("AX-1", question_specs(path))["x"]
        sdk = rt.ask("AX-1", {"x": Pick("Choose.", options, focus="mentions")})["x"]
    assert cli.as_dict(True) == sdk.as_dict(True)


def test_corrected_guided_fact_retains_binding(tmp_path):
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(definition(tmp_path))
        sw.correct("case", fact="refund", value=True, actor=Actor('human'))
        result = sw("case")
        assert result.facts["refund"].source == "example"
        assert result.facts["refund"].guidance == sw.plan.fact("refund").judgment.guidance.references


def test_admitted_head_guided_fact_retains_binding(tmp_path):
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(definition(tmp_path))
        sw._snap = replace(sw.snapshot, heads={"refund": SimpleNamespace(probs_features=lambda _: [.99, .005, .005])},
                           policies={"classifier": {"refund": Policy()}})
        result = sw("case")
        assert result.facts["refund"].source == "classifier" and result.facts["refund"].guidance


def test_immediate_inline_result_retains_guidance():
    with Runtime(FakeScorer()) as rt:
        result = rt.ask("case", {"x": Pick("Choose.", {"a": "first"}, eligible=lambda key: False, allow_none=True)}, guidance="Definitions.")["x"]
        assert result.status == "none" and result.guidance == Guidance("Definitions.").references


def test_guidance_binding_event_parity(tmp_path):
    events = []
    with Runtime(FakeScorer(), on_decision=events.append) as rt:
        sw = rt.load(definition(tmp_path))
        sw.correct("case", fact="refund", value=True, actor=Actor('human'))
        result = sw("case")
    assert events[-1].facts["refund"]["guidance"] == result.facts["refund"].guidance


def test_switch_insufficient_evidence_render_names_unresolved_fact(tmp_path):
    with Runtime(FakeScorer(lambda *args: [.005, .005, .99])) as rt:
        result = rt.load(definition(tmp_path))("case")
    assert result.status == "needs_input" and not result.missing
    assert "refund" in str(result) and "insufficient" in str(result)
    assert "refund" in repr(result) and "missing " not in repr(result)


def test_contrast_multiple_memberships_count_once_per_membership():
    result = contrasts([(0, 0), (1, 1)], [{"pairs": ["p", "q", "q"]}] * 2)
    assert result["pairs"] == result["pairs_both_correct"] == 2


def test_contrast_plural_invariant_groups():
    result = contrasts([(0, 0), (1, 0)], [{"groups": ["g"]}] * 2)
    assert result["invariant_groups"] == 1 and result["invariant_consistent"] == 0


class ReceiptJudge:
    identity = {"source": "test-hosted", "model": "requested-v1"}
    batch_inputs = 8
    max_slots = 20
    supports_guidance = True
    accelerated = False

    def __init__(self, models):
        self.models = iter(models)
        self.calls = 0

    def score_many(self, items):
        out = []
        for _, questions in items:
            model = next(self.models)
            self.calls += 1
            scores = []
            for q in questions:
                p = [.99] + [.01 / (q.width - 1)] * (q.width - 1)
                if self.calls == 1 and q.kind == "pick" and len(q.options) == 2:
                    p = [.01, .99]
                scores.append(Scores(np.array(p), None, None))
            out.append(Scored(tuple(scores), 17, model))
        return out

    def fit(self, text, limit):
        return text, False


def locked_switch(tmp_path, rt, **changes):
    sw = rt.load(definition(tmp_path, **changes))
    sw._snap = replace(sw.snapshot, locked=True, returned_models={"judge": ("old-v1",)})
    return sw


@pytest.mark.parametrize("model,error", [("new-v2", "predictor_identity_changed"), (None, "predictor_identity_unverified")])
def test_hosted_drift_blocks_old_lock_policy(tmp_path, model, error):
    with Runtime(ReceiptJudge([model])) as rt:
        result = locked_switch(tmp_path, rt)("case")
    assert result.status == "failed" and result.errors["refund"] == error
    assert result.cost["logical_tokens"] == 17
    assert result.facts["refund"].observed_models == ((model,) if model else ())


def test_hosted_matching_receipt_preserves_acceptance(tmp_path):
    with Runtime(ReceiptJudge(["old-v1"])) as rt:
        assert locked_switch(tmp_path, rt)("case").status == "ok"


def test_hosted_focus_fallback_checks_each_receipt(tmp_path):
    judge = ReceiptJudge(["old-v1", "new-v2"])
    with Runtime(judge) as rt:
        sw = locked_switch(tmp_path, rt, facts={"route": {"pick": "Choose.", "options": {"AX-1": "first", "AX-2": "second"}, "focus": "mentions"}})
        result = sw("AX-1")
    assert judge.calls == 2 and result.status == "failed"
    assert result.facts["route"].observed_models == ("old-v1", "new-v2")


def test_unrelated_drift_preserves_exact_settled_action(tmp_path):
    judge = ReceiptJudge([])
    with Runtime(judge) as rt:
        sw = locked_switch(tmp_path, rt, rules=[{"id": "exact", "when": "message == 'case'", "then": "accepted"},
                                             {"id": "semantic", "when": "refund", "then": "other"}])
        result = sw("case")
    assert result.status == "ok" and result.answer == "accepted" and judge.calls == 0


def test_allow_model_change_resets_receipt_baseline_and_policy(tmp_path):
    path = definition(tmp_path, guidance={}, facts={"refund": {"check": "Refund requested."}})
    with Runtime(ReceiptJudge(['old-v1'])) as rt:
        tools.lock(rt, path)
    payload = files.read_json(path.with_suffix(".lock.json"))
    payload["returned_models"] = {"judge": ["old-v1"]}
    from semantic_switch.compiler import digest
    payload["id"] = digest({k: v for k, v in payload.items() if k not in ("id", "created")})[:12]
    files.publish(path.with_suffix(".lock.json"), payload)
    changed = ReceiptJudge(["new-v2"])
    changed.identity = {**changed.identity, "model": "requested-v2"}
    with Runtime(changed) as rt:
        sw = rt.load(path, allow_model_change=True)
        assert sw.snapshot.returned_models == {"judge": ()} and "judge" not in sw.snapshot.policies
        assert sw.warnings and sw("case").status == "ok"


def test_policy_audit_replays_frozen_labels_and_denominators(tmp_path):
    from bench.policy_audit import run, summarize
    from semantic_switch.examples import label_record
    path = definition(tmp_path)
    audit = tmp_path / 'audit.jsonl'
    with Runtime(FakeScorer(lambda *args: [.99, .005, .005])) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        row = label_record(sw.plan, {'message': 'case'}, {'refund': True}, groups=['family'], sources={'refund': [{'derivation': 'authored', 'producer': 'fixture', 'actor': {'kind': 'human'}}]})
        files.publish(audit, json.dumps(row) + '\n')
        candidate = sw.snapshot.candidate_id
        artifact = run(rt, path, audit)
        assert artifact['candidate_id'] == candidate
        assert artifact['metrics'] == summarize(artifact['observations'])
        assert artifact['metrics']['fact:refund']['submitted'] == 1
        assert artifact['memory']['device_memory'] == 'unmeasured'
        assert artifact['observations'][0]['result']['facts']['refund']['guidance']


def test_model_derived_sequence_overflow_rejected_before_resolution():
    with pytest.raises(ConfigError, match='sequence slots'):
        Model('unresolved.gguf', prefix_slots=2**32 - 1)


def test_native_topology_rejected_before_resolution(monkeypatch):
    monkeypatch.setattr('semantic_switch.api._resolve', lambda _: pytest.fail('resolved invalid topology'))
    with pytest.raises(ConfigError, match='sequence slots'):
        Model('unresolved.gguf', device='cpu', n_batch=4)


def test_policy_audit_case_and_fact_denominators_separate():
    from bench.policy_audit import summarize
    rows = [{'case_labelled': True, 'expected_case': 'refund', 'labels': {'refund': True},
             'result': {'status': 'ok', 'rule': 'support', 'facts': {'refund': {'settled': True, 'value': True}}}},
            {'case_labelled': True, 'expected_case': 'refund', 'labels': {'refund': True},
             'result': {'status': 'failed', 'rule': None, 'facts': {}}}]
    result = summarize(rows)
    assert result['cases']['submitted'] == result['fact:refund']['submitted'] == 2
    assert result['cases']['wrong'] == 1 and result['cases']['failed'] == 1
    assert result['fact:refund']['correct'] == 1 and result['fact:refund']['unexamined'] == 1


def test_policy_audit_raw_argmax_keeps_abstentions_in_denominator(tmp_path):
    from bench.policy_audit import run
    from semantic_switch.examples import label_record
    path = definition(tmp_path)
    audit = tmp_path / 'audit.jsonl'
    with Runtime(FakeScorer(lambda *args: [.6, .2, .2])) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        row = label_record(sw.plan, {'message': 'case'}, {'refund': True}, groups=['family'], sources={'refund': [{'derivation': 'authored', 'producer': 'fixture', 'actor': {'kind': 'human'}}]})
        files.publish(audit, json.dumps(row) + '\n')
        artifact = run(rt, path, audit)
    metric = artifact['metrics']['fact:refund']
    assert metric['submitted'] == metric['uncertain'] == metric['raw_correct'] == 1
    assert metric['accepted'] == 0 and metric['accepted_precision'] is None
    assert metric['all_row_agreement'] == 1
    assert artifact['observations'][0]['raw_answers']['refund']['value'] is True


def test_policy_audit_invalid_metadata_fails_before_predictor(tmp_path):
    from bench.policy_audit import run
    from semantic_switch.examples import label_record
    path = definition(tmp_path)
    audit = tmp_path / 'audit.jsonl'
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        sw = rt.load(path)
        row = label_record(sw.plan, {'message': 'case'}, {'refund': True}, groups=['family'], sources={'refund': [{'derivation': 'authored', 'producer': 'fixture', 'actor': {'kind': 'human'}}]})
        row['selections'] = [{'selected_by': 'invented'}]
        files.publish(audit, json.dumps(row) + '\n')
        with pytest.raises(ConfigError, match='audit.jsonl:1:.*selection'):
            run(rt, path, audit)


def test_policy_audit_pairs_identical_cohort_and_rejects_label_changes():
    from copy import deepcopy
    from bench.policy_audit import compare
    left = {'schema': 1, 'audit_sha256': 'cohort', 'guidance': {}, 'candidate_id': 'left', 'runtime': {},
            'switch_revision': 'same', 'meanings': {'x': 'same'},
            'observations': [{'id': 'a', 'input': 'case', 'labels': {'x': True}, 'elapsed_ms': 10,
                              'case_labelled': False, 'expected_case': None, 'result': {'status': 'ok', 'facts': {}}}]}
    from bench.policy_audit import summarize
    left['metrics'] = summarize(left['observations'])
    right = deepcopy(left)
    right['candidate_id'] = 'right'
    right['observations'][0]['elapsed_ms'] = 15
    assert compare(left, right)['paired_latency'][0]['delta_ms'] == 5
    right['observations'][0]['labels']['x'] = False
    with pytest.raises(ConfigError, match='order or labels'):
        compare(left, right)


def test_policy_audit_all_case_labels_match_serialized_tuple_outcomes():
    from bench.policy_audit import summarize
    summary = summarize([{'case_labelled': True, 'expected_case': ['a', 'b'], 'labels': {},
                          'result': {'status': 'ok', 'rule': ('a', 'b'), 'facts': {}}}])
    assert summary['cases']['correct'] == 1 and summary['cases']['wrong'] == 0
    assert summary['cases']['all_row_agreement'] is None
