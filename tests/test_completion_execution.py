import json
import threading

import anyio
import pytest

from semantic_switch import Check, Pick, Runtime, files, judge
from semantic_switch.executor import Reuse
from semantic_switch.hosted import Reply
from semantic_testkit.scoring import FakeScorer

from .test_openrouter_judge import default_rows, response

REFUND = "The customer asks to receive money back for a charge."


def saved(tmp_path, facts, **extra):
    path = tmp_path / "s.yaml"
    files.publish(path, {"name": "s", "fields": {"message": {"type": "text", "required": True}}, "reads": ["message"], "facts": facts, **extra})
    return path


def test_completion_refills_the_window_while_an_older_input_is_pending():
    sent, live, peak, release, waited = [], [0], [0], threading.Event(), []

    async def send(url, body, remaining):
        body = json.loads(body)
        user = body["messages"][1]["content"]
        sent.append(next(name for name in ("slow", "case a", "case b", "case c", "case d") if name in user))
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        try:
            if "slow" in user:
                for _ in range(300):
                    if release.is_set():
                        break
                    await anyio.sleep(.01)
                waited.append(release.is_set())
            if "case d" in user:
                release.set()
            return Reply(200, response(default_rows(body)))
        finally:
            live[0] -= 1
    with Runtime(judge("openrouter:m", codec="chat", transport=send, batch_inputs=2), own_judges=True) as rt:
        rows = rt.ask_many(["slow case", "case a", "case b", "case c", "case d"], {"c": Check(REFUND)})
    assert waited == [True] and peak[0] <= 2 and "case d" in sent
    assert [row["c"].status for row in rows] == ["ok"] * 5


def test_named_only_bindings_size_the_window_from_participating_roles(tmp_path):
    scorer = FakeScorer(name="specialist")
    path = saved(tmp_path, {"refund": {"check": REFUND, "binding": "refund"}})
    with Runtime(bindings={"refund": scorer}) as rt:
        switch = rt.load(path)
        results = switch.run([{"message": "refund me"}, {"message": "hello"}, {"message": "refund twice"}])
    assert [r.status for r in results] == ["ok", "ok", "ok"] and results[0].cost["operation"]["window"] == 2
    assert results[0].cost["operation"]["inputs"] == 3 and all(r.cost["elapsed_ms"] >= 0 for r in results)


def test_exact_only_work_admits_one_input(tmp_path):
    path = saved(tmp_path, {"refund": {"check": REFUND}}, rules=[{"id": "exact", "when": "message == 'case'", "then": "yes"}], otherwise="no")
    with Runtime(FakeScorer()) as rt:
        switch = rt.load(path)
        rows = switch.score([{"message": "x"}], sources=("example",))
        assert switch.exec.window(switch.exec.bindings(switch.snapshot, ("example",))) == 1
    assert rows[0].targets["refund"].source is None


def test_preview_matches_the_submitted_hosted_request():
    bodies = []

    async def send(url, body, remaining):
        bodies.append(json.loads(body))
        return Reply(200, response(default_rows(bodies[-1])))
    items = {"c": Check(REFUND), "p": Pick("Which part failed?", {"ax": "The axle", "hub": "The hub"})}
    with Runtime(judge("openrouter:m", codec="chat", transport=send), own_judges=True) as rt:
        preview = rt.preview("the axle broke, refund me", items)
        rt.ask("the axle broke, refund me", items)
    assert [request["body"] for request in preview.requests] == bodies
    assert all(request["representation"] == "hosted" and "key" not in json.dumps(request["body"]).lower() for request in preview.requests)
    assert [stage.target for stage in preview.stages] == [("c",), ("p",)] and not any(stage.conditional for stage in preview.stages)


def test_preview_marks_the_conditional_fallback_stage(tmp_path):
    path = saved(tmp_path, {"part": {"pick": "Which part failed?", "focus": "mentions",
                                     "options": {"AX-1": {"description": "The front axle", "aliases": ["AX-1"]},
                                                 "AX-2": {"description": "The rear axle", "aliases": ["AX-2"]},
                                                 "HB-1": {"description": "The hub", "aliases": ["HB-1"]}}}})
    fake = FakeScorer()
    with Runtime(fake) as rt:
        (preview,) = rt.load(path).preview([{"message": "AX-2 cracked"}])
    assert [(stage.stage, stage.conditional) for stage in preview.stages] == [("mentions", False), ("fallback", True)]
    assert preview.requests[0]["representation"] == "local" and preview.requests[0]["targets"] == [["part"]] and not fake.requests


def test_one_scoring_unit_serves_every_condition_of_a_pick(tmp_path):
    path = saved(tmp_path, {"topic": {"pick": "What is it about?", "options": {"bill": "billing", "tech": "technical", "acct": "account"}}},
                 rules=[{"id": "b", "when": "topic == 'bill'", "then": "b"}, {"id": "t", "when": "topic == 'tech'", "then": "t"},
                        {"id": "a", "when": "topic == 'acct'", "then": "a"}], otherwise="o")
    fake = FakeScorer()
    with Runtime(fake) as rt:
        switch = rt.load(path)
        reuse = Reuse()
        rows = switch.score([{"message": "billing problem"}, {"message": "login issue"}], ["topic"], reuse=reuse)
        assert len(fake.requests) == 2 and all(len(row.conditions) == 3 for row in rows)
        again = switch.score([{"message": "billing problem"}, {"message": "login issue"}], ["topic"], reuse=reuse)
    assert len(fake.requests) == 2 and reuse.reused == 2
    assert [row.targets["topic"].result.value for row in again] == [row.targets["topic"].result.value for row in rows]


def test_decided_rows_carry_the_decision_from_the_same_state(tmp_path):
    path = saved(tmp_path, {"refund": {"check": REFUND}}, rules=[{"id": "money", "when": "refund", "then": "refund"}], otherwise="other")
    fake = FakeScorer()
    with Runtime(fake) as rt:
        rows = rt.load(path).score([{"message": "refund me"}, {"message": "hello"}], ["refund"], decide=[True, False])
    assert rows[0].decision.status == "ok" and rows[1].decision is None and len(fake.requests) == 2
    assert rows[0].targets["refund"].result.value == rows[0].decision.facts["refund"].value


def test_stream_admits_one_input_and_cancellation_propagates(tmp_path):
    path = saved(tmp_path, {"refund": {"check": REFUND}})
    with Runtime(FakeScorer()) as rt:
        switch = rt.load(path)
        stream = switch.many({"message": text} for text in ("refund", "hello"))
        first = next(stream)
        assert first.status == "ok" and "operation" not in first.cost
        stream.close()
        assert switch({"message": "refund"}).status == "ok"
    with pytest.raises(RuntimeError, match="closed"):
        rt.ask("x", {"c": Check(REFUND)})


def test_inline_dynamic_callbacks_run_once_per_valid_input_and_errors_skip_work():
    described = []
    pick = Pick("Which option?", ["alpha", "beta"], definition=lambda o: described.append(o) or f"the {o} option")
    fake = FakeScorer()
    with Runtime(fake) as rt:
        rows = rt.ask_many(["alpha", "beta", ""], {"p": pick, "c": Check(REFUND)})
    assert sorted(described) == ["alpha", "alpha", "beta", "beta"] and len(fake.requests) == 4
    assert rows[2]["p"].status == rows[2]["c"].status == "needs_input" and rows[2]["p"].guidance == ()
