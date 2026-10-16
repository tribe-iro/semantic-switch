from __future__ import annotations

from dataclasses import replace

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import semantic_switch as ss
from semantic_switch.artifacts import read_export
from semantic_switch.hosted import Reply
from semantic_switch import Actor, Check, ConfigError, Pick, Runtime, files, tools
from tests.support.batching import BatchingJudge
from semantic_switch.compiler import compile_spec
from semantic_switch.context import ContextError, project
from semantic_switch.judge import Scored, Scores
from semantic_switch.protocols import Question
from semantic_switch.snapshot import _same, from_lock
from semantic_switch.tools.data import Labels
from semantic_switch.tools.evaluate import evaluate_facts
from semantic_testkit.scoring import FakeScorer
from tests.conftest import transport
from semantic_testkit.labels import envelopes


def definition(tmp_path, *, pick=False):
    spec = {"name": "sample", "fields": {"message": {"type": "text", "required": True}},
            "reads": ["message"], "facts": {"decision": {"check": "The customer wants money back."}}}
    if pick:
        spec["fields"]["enabled"] = {"type": "bool"}
        spec["facts"] = {"decision": {"pick": "Choose", "options": {
            "z": {"description": "first " * 200, "requires": "enabled == true"},
            "a": {"description": "second " * 200}}}}
    path = tmp_path / "sample.yaml"
    files.publish(path, spec)
    return path


def structured(send):
    return ss.judge("openrouter:typesafe/jev-1.13", transport=transport(lambda url, body, timeout: Reply(200, send(body))))


@pytest.mark.parametrize("probabilities", [{}, {"A": .2}, {"A": -2, "B": -1}, {"A": 0, "B": 0}, {"A": float("nan"), "B": 1},
                                           {"A": True, "B": 0}, [.5, .5]])
def test_remote_distributions_fail_locally(probabilities):
    with Runtime(structured(lambda body: {"model": "m", "answers": {"q0": {"type": "choice", "probabilities": probabilities}}})) as rt:
        assert rt.pick("input", "choose", ["a", "b"]).status == "failed"


@pytest.mark.parametrize("response", [[], {"model": "m", "answers": [], "usage": None}, {"answers": {"q0": {"type": "noul", "noul": .8}}},
                                      {"model": "m", "answers": {"unexpected": {"type": "noul", "noul": .8}}}])
def test_remote_envelopes_fail_per_input(response):
    judge = structured(lambda body: response if body["state"] == "bad" else {"model": "m", "answers": {"q0": {"type": "noul", "noul": .99}}})
    with Runtime(judge) as rt:
        out = Check("claim").run(["bad", "good"], rt)
        assert [result.status for result in out] == ["failed", "ok"]


def test_bind_and_own_correction(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(path)
        sw.correct("plain message", fact="decision", value=True, actor=Actor('human'))
        assert sw("plain message").facts["decision"].value is True
        with pytest.raises(TypeError):
            sw.snapshot.overrides["decision"]["bad"] = False
        tools.lock(rt, path)
        assert rt.load(path)("plain message").facts["decision"].value is True


def test_contextual_labels_share_evidence_family(tmp_path):
    path = definition(tmp_path, pick=True)
    rows = [{"input": {"message": "same", "enabled": enabled}, "facts": {"decision": value}}
            for enabled, value in ((True, "z"), (False, "a"))]
    files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes(rows)))
    with Runtime(FakeScorer()) as rt:
        labels = Labels(rt.load(path))
        a, b = labels.examples("decision")
        assert a.projection != b.projection and a.group == b.group and a.split == b.split
        sw = rt.load(path)
        sw.correct(rows[0]["input"], fact="decision", value="z", actor=Actor('human'))
        assert sw(rows[0]["input"]).facts["decision"].value == "z"
        assert sw(rows[1]["input"]).facts["decision"].value != "z"


def test_long_options_saved_inline_export_parity(tmp_path):
    options = {"z": "first " * 100, "a": "second " * 100}
    with Runtime(FakeScorer(lambda s, q, w: [.97, .03])) as rt:
        inline = rt.pick("text", "choose", options)
        switch = Pick("choose", options).save(tmp_path / "long.yaml", rt=rt)
        saved = switch("text").facts["long"]
        assert inline.value == saved.value == "z"
        switch.correct("text", fact="long", value="z", actor=Actor('human'))
        row = read_export(tools.export(rt, switch.paths.yaml, split=None, native=True)["path"], fitting=False, native=True)[1][0]
        assert list(row["presentations"][0].answers.options) == list(options.values())
        tools.lock(rt, switch.paths.yaml)
        assert rt.load(switch.paths.yaml)("text").facts["long"].value == "z"


def test_all_failed_fact_accounting(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer(lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable")))) as rt:
        sw = rt.load(path)
        row = evaluate_facts(sw, sw.snapshot, [{"input": "text", "facts": {"decision": True}}])["decision"]
        assert row["labelled"] == row["actual_positives"] == 1
        assert row["statuses"] == {"failed": 1}


def test_identity_every_behavior_field_and_rename():
    assert _same({"sha256": "x", "file": "a"}, {"sha256": "x", "file": "b"})
    for field in ("model", "sha256", "template", "prompt"):
        assert not _same({"source": "provider", field: "a"}, {"source": "provider", field: "b"})
        assert not _same({"source": "provider", field: None}, {"source": "provider"})


def test_lock_schema_and_payload_validation(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        data = files.read_json(path.with_suffix(".lock.json"))
        assert data["schema"] == 20
        data["evidence"]["new"] = "tampered"
        with pytest.raises(ConfigError, match="artifact ID"):
            from_lock(data, "lock")
        data["schema"] = 99
        with pytest.raises(ConfigError, match="unsupported lock schema"):
            from_lock(data, "lock")


def test_embedded_null_cannot_mask_failure(tmp_path):
    path = definition(tmp_path)
    spec = files.read_yaml(path)
    spec["tests"] = [{"input": {}, "expect": None}]
    files.publish(path, spec)
    with Runtime(FakeScorer()) as rt:
        with pytest.raises(ConfigError, match="expected status ok"):
            tools.lock(rt, path)
    spec["tests"] = [{"input": {}, "expect_status": "needs_input"}]
    files.publish(path, spec)
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)


def test_owned_numeric_json_and_immutable_fields(tmp_path):
    plan = compile_spec({"name": "x", "fields": {"n": {"type": "number"}, "j": {"type": "json"}}, "facts": {}})
    for value in (True, float('inf'), float('nan')):
        with pytest.raises(ContextError):
            project(plan.fields, {"n": value})
    with pytest.raises(ContextError):
        project(plan.fields, {"j": {"bad": float('nan')}})
    with pytest.raises((AttributeError, TypeError)):
        plan.fields["n"].required = True


class CountingJudge:
    identity = {"source": "counter"}
    max_slots = 20
    batch_inputs = 8
    accelerated = True

    def __init__(self):
        self.calls = []

    def fit(self, text, budget):
        return text, False

    def score_many(self, items):
        self.calls.append(len(items))
        return [Scored(tuple(Scores(np.array([.99, .01]), None) for q in questions), 1)
                for state, questions in items]


def test_batches_and_stream_no_readahead():
    judge = CountingJudge()
    with Runtime(judge) as rt:
        assert len(Check("claim").run(["a", "b", "c"], rt)) == 3
        assert judge.calls == [3]
        delivered = []
        def inputs():
            yield "a"
            assert delivered
            yield "b"
        output = Check("claim").many(inputs(), rt)
        delivered.append(next(output))
        assert next(output).status == "ok"
        assert rt.ask_many([], {}) == []
        assert rt.ask_many(["a", "b"], {}) == [{}, {}]
        with pytest.raises(TypeError):
            rt.models["judge"] = CountingJudge()


def test_serving_coalesces_and_preserves_order():
    judge = CountingJudge()
    wrapper = BatchingJudge(judge, wait_ms=20)
    try:
        futures = [wrapper.submit([(str(i), [Question("check", "claim")])]) for i in range(5)]
        assert all(future.result()[0].scores[0].probs[0] == .99 for future in futures)
        assert judge.calls == [5]
    finally:
        wrapper.close()
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.submit([("x", [Question("check", "claim")])])


def test_new_family_member_keeps_historical_use(tmp_path):
    path = definition(tmp_path)
    files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes([
        {"input": "original English", "facts": {"decision": True}, "group": "family"}])))
    with Runtime(FakeScorer()) as rt:
        tools.lock(rt, path)
        files.append(path.with_suffix(".test.jsonl"), files.jsonl(envelopes([
            {"input": "tradução nova", "facts": {"decision": True}, "group": "family"}])))
        report = tools.run_test(rt, path)
        assert report["fact_populations"]["used"]["decision"]["labelled"] == 2
        assert "decision" not in report["fact_populations"]["fresh"]


def test_unknown_eligibility_head_reaches_residue(tmp_path):
    from semantic_switch.heads import LexicalHead
    from semantic_switch.judgment import Policy
    from semantic_switch.tools.lock import head_scores
    path = definition(tmp_path, pick=True)
    with Runtime(FakeScorer()) as rt:
        sw = rt.load(path)
        fact = sw.plan.fact("decision")
        head = LexicalHead(fact.keys, np.array([], dtype=np.int32), np.empty((0, 2), dtype=np.float32),
                           np.log([.99, .01]).astype(np.float32))
        files.append(sw.paths.test, files.jsonl(envelopes([{"input": {"message": "unknown"}, "facts": {"decision": "z"}}])))
        example = Labels(sw).examples("decision")[0]
        observation = head_scores(fact, head, [example], sw, sw.snapshot)[0]
        assert observation.pending == {0}
        installed = replace(sw.snapshot, heads={fact.name: head}, policies={"classifier": {fact.name: Policy(.9, .9)}})
        (row,) = sw.score([example.data], [fact.name], sources=("classifier",), snap=installed)
        assert row.error is None and not row.targets[fact.name].settled


def test_concurrent_corrections_publish_both(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer()) as rt:
        switch = rt.load(path)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(switch.correct, message, fact="decision", value=True, actor=Actor("human"))
                       for message in ("first correction", "second correction")]
            for future in futures:
                future.result()
        assert all(switch(message).facts["decision"].value is True for message in
                   ("first correction", "second correction"))
        assert len(files.read_jsonl(switch.paths.examples)) == 2


def test_focused_policy_rows_do_not_depend_on_threshold():
    from semantic_switch.observations import Observation
    from semantic_switch.tools.policy import side_rows, choose
    plan = compile_spec({"name": "focus", "fields": {"message": {"type": "text"}}, "reads": ["message"],
                         "facts": {"pick": {"pick": "choose", "options": {"a": "A", "b": "B"}, "focus": "mentions"}}})
    rows = side_rows(plan.fact("pick"), [Observation("id", {}, 1, [.96, .04])])["answer"]
    assert rows == [(.96, False)] and choose(rows * 20, .5) == (True, None)


def test_malformed_head_bytes_and_indices():
    from semantic_switch.heads import LexicalHead
    head = LexicalHead(("true", "false"), np.array([0, 1], dtype=np.int32),
                       np.zeros((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32))
    encoded = head.to_json()
    for malformed in ({**encoded, "n": 3}, {**encoded, "idx": "not base64!"},
                      {**encoded, "w": encoded["w"][:-1]}):
        with pytest.raises(ConfigError, match="invalid lexical head"):
            LexicalHead.from_json(malformed)
    with pytest.raises(ValueError):
        head.w[0, 0] = 1


def test_quality_all_failed_has_complete_denominator():
    from bench.quality import summarize
    result = summarize({"checks": [{"p": None, "label": True, "mass": None, "error": "offline", "ms": 1}], "picks": []})
    assert result["items"] == result["errors"] == 1
    assert result["valid_scores"] == 0 and result["coverage_0.9"] == 0
    assert result["accuracy_given_prediction"] is None and result["correct_per_submission"] == 0


def test_explicit_empty_inline_fields_and_invalid_neighbor():
    judge = CountingJudge()
    with Runtime(judge) as runtime:
        assert runtime.ask({"hidden": object()}, {"c": Check("claim")}, fields=[])["c"].status == "needs_input"
        output = Check("claim").run([None, "valid"], runtime)
        assert [row.status for row in output] == ["failed", "ok"]


def test_metadata_survives_repeated_correction_and_lock(tmp_path):
    path = definition(tmp_path)
    with Runtime(FakeScorer()) as rt:
        switch = rt.load(path)
        switch.correct("message", fact="decision", value=True, groups=["english"], pairs=["contrast"], actor=Actor('human'))
        switch.correct("message", fact="decision", value=True, groups=["translation"], pairs=["contrast"], actor=Actor('human'))
        tools.lock(rt, path)
        example = rt.load(path).snapshot.examples["decision"][0]
        assert set(example.groups) == {"english", "translation"}
        assert example.pairs == ("contrast",)


def test_lazy_public_facts_depend_only_on_required_work(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    spec = {"name": "routes", "fields": {"message": {"type": "text"}}, "reads": ["message"],
            "facts": {"first": {"check": "first claim"}, "later": {"check": "later claim"}},
            "rules": [{"id": "first", "when": "first", "then": "first"},
                      {"id": "later", "when": "later", "then": "later"}]}
    path = tmp_path / "routes.yaml"
    files.publish(path, spec)
    def oracle(state, question, width):
        if "later claim" in question:
            raise RuntimeError("unused predictor failure")
        return [.99, .01]
    fake = FakeScorer(oracle)
    fake.operation = nullcontext
    fake.ctx = SimpleNamespace(n_batch=512, n_ctx=16384)
    fake.shared = True
    with Runtime(fake) as runtime:
        switch = runtime.load(path)
        result = switch("input")
        assert result.status == "ok" and result.rule == "first"
        assert "later" not in result.facts and not result.errors
        assert not fake.asked("later claim")
