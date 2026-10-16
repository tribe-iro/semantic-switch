import ast
from email.utils import formatdate
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import anyio
import numpy as np
import pytest

import semantic_switch as ss
from semantic_switch import Check, ConfigError, Pick, Rate, Runtime, Select, files, tools
from semantic_switch.adapters.qwen3 import Qwen3Adapter
from semantic_switch.guidance import normalize
from semantic_switch.drivers import openrouter
from semantic_switch.drivers.codecs import distribution
from semantic_switch.drivers.profiles import CHAT, JEV as JEV_PROFILE
from semantic_switch.hosted import ATTEMPTS, IN_FLIGHT, HostedJudge, Reply, TransportError
from semantic_switch.judgment import Judgment
from semantic_switch.protocols import Question
from semantic_testkit.scoring import FakeScorer
from tests.conftest import transport
from tests.test_remaining_contracts import definition

EXAMPLE = Path(__file__).parents[1] / "examples" / "quickstart.py"
JEV = "typesafe/jev-1.13"
QUESTIONS = [
    Question("check", "The customer asks for a refund.", (), ("store credit",)),
    Question("check", "Is the customer asking for a refund?"),
    Judgment.check("Is there a leak?", unstated=True, guidance="Leaks are visible coolant.").question(),
    Question("pick", "Which team?", ("billing", "support"), ("spam",), normalize("Billing owns charges.")),
    Question("rate", "How urgent?", ("low", "high")),
]


def response(rows, *, model="m", provider="p", **usage):
    (token, logprob), *_ = rows.items()
    return {"model": model, "provider": provider, "usage": {"prompt_tokens": 7, **usage},
            "choices": [{"finish_reason": "length", "logprobs": {"content": [{"token": token, "logprob": logprob,
                         "top_logprobs": [{"token": t, "logprob": v} for t, v in rows.items()]}]}}]}


def default_rows(body):
    user = body["messages"][1]["content"]
    pair = ("true", "false") if "true or false" in user else ("yes", "no") if "yes or no" in user else ("A", "B")
    return {pair[0]: math.log(.97), pair[1]: math.log(.02)}


class Provider:
    def __init__(self, rows=default_rows, **extra):
        self.rows, self.extra, self.bodies, self.urls, self.closed = rows, extra, [], [], False

    async def __call__(self, url, body, timeout):
        body = json.loads(body)
        self.urls.append(url)
        self.bodies.append(body)
        return Reply(200, response(self.rows(body), **self.extra))

    async def close(self):
        self.closed = True


class Structured(Provider):
    async def __call__(self, url, body, timeout):
        body = json.loads(body)
        self.urls.append(url)
        self.bodies.append(body)
        answers = {}
        for name, question in body["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": .97}
            elif question["type"] == "score":
                top = len(question["criteria"]) - 1
                answers[name] = {"type": "score", "probabilities": {str(i): .9 if i == top else .1 / top for i in range(top + 1)}}
            else:
                letters = list(question["criteria"])
                answers[name] = {"type": "choice", "probabilities": {k: .9 if i == 0 else .1 / (len(letters) - 1) for i, k in enumerate(letters)}}
        return Reply(200, {"model": JEV, "provider": "TypeSafe", "answers": answers, "usage": {"prompt_tokens": 9}, **self.extra})


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def scripted(*replies, clock=None, cost=0.0):
    sent = []

    async def send(url, body, timeout):
        sent.append(timeout)
        if clock is not None:
            clock.now += cost
        reply = replies[len(sent) - 1]
        if isinstance(reply, BaseException):
            raise reply
        return reply
    return send, sent


def judge(send, clock, **settings):
    return HostedJudge(openrouter.OpenRouter("p"), "m", CHAT, transport=send, clock=clock, sleep=clock.sleep, **settings)


def one(judge_):
    return judge_.score_many([("x", [QUESTIONS[0]])])[0]


OK = Reply(200, response({"true": math.log(.97), "false": math.log(.02)}))


def test_public_configuration_selects_local_or_openrouter_without_bench():
    tree = ast.parse(EXAMPLE.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.split(".")[0] == "bench" for name in imported)
    script = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('quickstart', sys.argv[1]); "
              "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
              "local = module.judge({}); hosted = module.judge({'OPENROUTER_MODEL': 'm', 'OPENROUTER_PROVIDER': 'p', 'OPENROUTER_CODEC': 'chat'}); "
              "print(type(local).__name__, type(hosted).__name__, hosted.provider.route, any(m.split('.')[0] == 'bench' for m in sys.modules)); hosted.close()")
    out = subprocess.run([sys.executable, "-c", script, str(EXAMPLE)], capture_output=True, text=True, check=True,
                         env={**os.environ, "OPENROUTER_API_KEY": "k"}).stdout.split()
    assert out == ["Model", "HostedJudge", "p", "False"]


@pytest.mark.parametrize("question", QUESTIONS)
def test_rendered_question_matches_native_adapter(question):
    adapter = Qwen3Adapter(FakeScorer())
    request = adapter.request(adapter.state("I was charged twice."), [question])
    native = adapter.scorer.detokenize(request.prefix + request.state + request.suffixes[0])
    provider = Provider()
    ss.judge("openrouter:m", codec="chat", transport=provider).score_many([("I was charged twice.", [question])])
    system, user = (message["content"] for message in provider.bodies[0]["messages"])
    assert native == f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_answer_tokens_match_declared_spellings_with_lower_bound_mass():
    rows = {"true": math.log(.5), " True": math.log(.2), "TRUE ": math.log(.05), "false": math.log(.1), "maybe": math.log(.1)}
    scores = ss.judge("openrouter:m", codec="chat", transport=Provider(lambda _: rows)).score_many([("x", [QUESTIONS[0]])])[0].scores[0]
    assert scores.error is None and scores.mass == pytest.approx(.8) and np.allclose(scores.probs, [.7 / .85, .1 / .85])
    three = ss.judge("openrouter:m", codec="chat", transport=Provider(lambda _: {" a": math.log(.6), "C": math.log(.3)})).score_many([("x", [QUESTIONS[2]])])[0].scores[0]
    assert three.mass == pytest.approx(.3) and np.allclose(three.probs, [0, 0, .75])


def test_lowercase_letters_are_not_option_letters():
    nine = Question("pick", "Which one?", tuple(f"option {k}" for k in range(9)))
    rows = {" I": math.log(.6), "i": math.log(.2), "a": math.log(.1), " a": math.log(.05)}
    judge_ = ss.judge("openrouter:m", codec="chat", transport=Provider(lambda _: rows))
    score = judge_.score_many([("x", [nine])])[0].scores[0]
    assert score.mass == pytest.approx(.6) and score.probs[8] == pytest.approx(.6 / .65) and score.probs[:8].sum() == 0
    assert judge_.identity["codec"] == "chat@2" and judge_.identity["answers"] == "declared-exact-spellings@1"


@pytest.mark.parametrize("provider,error", [
    (Provider(lambda _: {"maybe": -.1}), "no answer token"),
    (Provider(provider="other"), "provider route changed"),
    (transport(lambda url, body, timeout: Reply(200, {"model": "m", "provider": "p", "choices": [{"finish_reason": "stop", "logprobs": None}]})), "log probabilities unavailable"),
    (transport(lambda url, body, timeout: Reply(200, {**response(default_rows(body)), "usage": {"completion_tokens_details": {"reasoning_tokens": 3}}})), "reasoning tokens"),
    (transport(lambda url, body, timeout: Reply(200, {"provider": "p"})), "returned model"),
])
def test_incomplete_or_unsupported_scores_are_explicit_errors(provider, error):
    scores = ss.judge("openrouter:m", codec="chat", provider="p", transport=provider).score_many([("x", [QUESTIONS[0]])])[0].scores[0]
    assert scores.probs is None and scores.mass is None and error in scores.error


def test_chat_request_pins_provider_and_records_protocol_and_returned_identity():
    provider = Provider()
    judge_ = ss.judge("openrouter:m", codec="chat", provider="p", transport=provider)
    scored = judge_.score_many([("x", QUESTIONS[:2])])[0]
    body = provider.bodies[0]
    assert len(provider.bodies) == 2 and scored.tokens == 14 and set(provider.urls) == {"https://openrouter.ai/api/v1/chat/completions"}
    assert body["provider"] == {"only": ["p"], "allow_fallbacks": False, "require_parameters": True}
    assert (body["max_tokens"], body["logprobs"], body["top_logprobs"], body["reasoning"]) == (1, True, 20, {"enabled": False})
    assert {key: judge_.identity[key] for key in ("source", "model", "remote", "protocol", "question_scope", "codec", "score_semantics")} == {
        "source": "openrouter", "model": "m", "remote": {"route": "p", "observed": "per-response", "weights": None, "guarantee": "observed-identity"},
        "protocol": "chat-logprobs", "question_scope": "individual", "codec": "chat@2",
        "score_semantics": "answer-token-lower-bound@2:top20"}
    assert scored.model == judge_.returned == "m@p" and {s.model for s in scored.scores} == {"m@p"}
    assert [(s.request.attempts, s.request.status, s.request.usage) for s in scored.scores] == [(1, 200, {"prompt_tokens": 7})] * 2 and scored.scores[0].request is not scored.scores[1].request


def test_structured_protocol_is_resolved_from_profile_data():
    provider = Structured()
    judge_ = ss.judge(f"openrouter:{JEV}", provider="TypeSafe", transport=provider)
    questions = [QUESTIONS[0], QUESTIONS[2], QUESTIONS[3], Question("rate", "How urgent?", ("low", "mid", "high"))]
    scored = judge_.score_many([("case", questions)])[0]
    body, = provider.bodies
    assert provider.urls == ["https://openrouter.ai/api/v1/systemone"] and body["model"] == JEV and body["state"] == "case"
    assert body["provider"] == {"only": ["TypeSafe"], "allow_fallbacks": False}
    assert [q["type"] for q in body["questions"].values()] == ["noul", "choice", "choice", "score"]
    assert body["questions"]["q0"]["instructions"].endswith("Except: store credit.")
    assert body["questions"]["q1"]["instructions"].startswith("Guidance:\n") and "Leaks are visible coolant." in body["questions"]["q1"]["instructions"]
    assert body["questions"]["q2"]["criteria"] == {"A": "billing", "B": "support"} and body["questions"]["q3"]["criteria"] == ["low", "mid", "high"]
    assert all(score.error is None for score in scored.scores) and np.allclose(scored.scores[3].probs, [.05, .05, .9])
    assert scored.model == f"{JEV}@TypeSafe" and scored.tokens == 9 and len({s.request.id for s in scored.scores}) == 1
    assert (judge_.identity["protocol"], judge_.identity["question_scope"]) == ("systemone", "joint")
    assert judge_.identity["profile"]["name"] == JEV_PROFILE.name and judge_.identity["precision"]["decimals"] == 2
    with pytest.raises(ConfigError, match="no built-in profile"):
        ss.judge("openrouter:typesafe/jev-1.13-preview", transport=Provider())
    with pytest.raises(ConfigError, match="needs a declared profile"):
        ss.judge("openrouter:typesafe/jev-1.13-preview", codec="systemone", transport=Provider())
    with pytest.raises(ConfigError, match="uses codec 'systemone', not 'chat'"):
        ss.judge(f"openrouter:{JEV}", codec="chat", transport=Provider())
    assert ss.judge("openrouter:typesafe/jev-1.13-preview", codec="chat", transport=Provider()).identity["protocol"] == "chat-logprobs"


@pytest.mark.parametrize("model,send", [("m", Provider()), (JEV, Structured())])
def test_every_primitive_runs_on_both_protocols(model, send):
    with Runtime(ss.judge(f"openrouter:{model}", provider="p" if model == "m" else "TypeSafe", transport=send, codec="chat" if model == "m" else None), own_judges=True) as rt:
        results = rt.ask("Refund the duplicate charge today.", {
            "claim": Check("The customer asks for a refund."),
            "silent": Check("The customer mentions a leak.", unstated=True),
            "team": Pick("Which team?", {"billing": "charges", "support": "other"}),
            "urgency": Rate("How urgent?", ("low", "mid", "high")),
            "issues": Select("Which issues apply?", {"refund": "money back", "broken": "device broken"})}, guidance="Refunds return money.")
    assert {name: result.status for name, result in results.items()} == dict.fromkeys(results, "ok")
    assert results["claim"].value is True and results["team"].value == "billing"
    assert results["urgency"].value == ("high" if model == JEV else "low")


def test_luna_decisions_answers_through_system_one_with_recorded_shapes():
    luna, bodies = "openai/gpt-6-luna-decisions", []

    async def send(url, body, timeout):
        body = json.loads(body)
        bodies.append((url, body))
        q = body["questions"]["q0"]
        answer = {"noul": {"type": "noul", "noul": 1},
                  "choice": {"type": "choice", "choice": "A", "probabilities": dict(zip(q.get("criteria", ()), (.99, .01, 0))), "confidence": .99},
                  "score": {"type": "score", "score": 2, "legend": {"0": "low", "1": "mid", "2": "high"}, "probabilities": {"0": 0, "1": 0, "2": 1}, "confidence": 1}}[q["type"]]
        return Reply(200, {"model": f"{luna}-20261006", "answers": {"q0": answer}, "usage": {"input_tokens": 150, "output_tokens": 0}, "provider": "OpenAI"})
    with Runtime(ss.judge(f"openrouter:{luna}", transport=send), own_judges=True) as rt:
        assert rt.judge().identity["protocol"] == "systemone"
        claim, silent, team, urgency = (item.run(["Refund my kettle."], rt)[0] for item in (
            Check("The customer asks for a refund."), Check("The customer asks for a refund.", unstated=True),
            Pick("Which team?", {"billing": "charges", "support": "faults", "security": "access"}), Rate("How urgent?", ("low", "mid", "high"))))
    assert {url for url, _ in bodies} == {"https://openrouter.ai/api/v1/systemone"} and {body["model"] for _, body in bodies} == {luna}
    assert [body["questions"]["q0"]["type"] for _, body in bodies] == ["noul", "choice", "choice", "score"]
    assert (claim.value, silent.value, team.value, urgency.value) == (True, True, "billing", "high")
    assert {r.status for r in (claim, silent, team, urgency)} == {"ok"}


def test_structured_limits_are_validated_before_sending():
    provider = Structured()
    with Runtime(ss.judge(f"openrouter:{JEV}", transport=provider), own_judges=True) as rt:
        with pytest.raises(ConfigError, match="2 to 10 levels"):
            rt.rate("case", "How urgent?", [f"level {i}" for i in range(11)])
    assert provider.bodies == []


@pytest.mark.parametrize("answers,error", [
    ({"q0": {"type": "choice", "probabilities": {"A": .9, "B": .2}}, "q1": {"type": "noul", "noul": .99}}, "differs from one by more than"),
    ({"q0": {"type": "choice", "probabilities": {"AA": .9, "B": .1}}, "q1": {"type": "noul", "noul": .99}}, "option ids A, B"),
    ({"q0": {"type": "noul", "noul": .9}, "q1": {"type": "noul", "noul": .99}}, "differs from the requested 'choice'"),
    ({"q0": {"probabilities": {"A": .9, "B": .1}}, "q1": {"type": "noul", "noul": .99}}, "answer type None"),
    ({"q1": {"type": "noul", "noul": .99}}, "no answer"),
    ({"q0": {"refusal": "policy"}, "q1": {"type": "noul", "noul": .99}}, "hosted refusal"),
])
def test_structured_malformed_answers_fail_alone(answers, error):
    send = transport(lambda url, body, timeout: Reply(200, {"model": JEV, "provider": "TypeSafe", "answers": answers}))
    scores = ss.judge(f"openrouter:{JEV}", transport=send).score_many([("x", [QUESTIONS[3], QUESTIONS[0]])])[0].scores
    assert error in scores[0].error and scores[1].error is None


@pytest.mark.parametrize("payload,error", [({"model": JEV, "provider": "TypeSafe", "answers": {"q9": {}}}, "requested question names"),
                                           ({"model": JEV, "provider": "TypeSafe", "refusal": "no"}, "hosted refusal"),
                                           ({"model": JEV, "provider": "Other", "answers": {}}, "provider route changed")])
def test_structured_envelopes_and_drift_fail_the_request(payload, error):
    send = transport(lambda url, body, timeout: Reply(200, payload))
    scores = ss.judge(f"openrouter:{JEV}", provider="TypeSafe", transport=send).score_many([("x", QUESTIONS[:2])])[0].scores
    assert all(error in score.error for score in scores)


@pytest.mark.parametrize("body,named", [({"error": {"message": "top_logprobs must be <= 5", "metadata": {"provider_name": "Alibaba"}}}, "provider Alibaba"),
                                        (None, "provider p")])
def test_rejected_request_shape_names_the_provider_without_retry(body, named):
    clock = Clock()
    send, sent = scripted(Reply(400, body), clock=clock)
    scored = one(judge(send, clock))
    assert len(sent) == 1 and not clock.slept and scored.scores[0].request.status == 400
    assert f"{named} rejected the request shape" in scored.scores[0].error and "--provider PROVIDER" in scored.scores[0].error


def test_retry_after_seconds_and_http_date_are_honoured():
    clock = Clock()
    date = formatdate(time.time() + 20, usegmt=True)
    send, sent = scripted(Reply(429, None, "3"), Reply(503, None, date), OK, clock=clock)
    scored = one(judge(send, clock))
    assert scored.scores[0].error is None and clock.slept[0] == 3 and 18 <= clock.slept[1] <= 20
    assert (scored.scores[0].request.attempts, scored.scores[0].request.status, scored.scores[0].request.usage) == (3, 200, {"prompt_tokens": 7})
    assert sent[0] == 60 and sent[1] == 57 and sent[2] == pytest.approx(60 - 3 - clock.slept[1])


def test_hint_beyond_the_deadline_fails_without_an_early_retry():
    clock = Clock()
    send, sent = scripted(Reply(429, None, "90"), OK, clock=clock)
    scored = one(judge(send, clock))
    assert len(sent) == 1 and not clock.slept and "would exceed the 60s deadline" in scored.scores[0].error


def test_retries_exhaust_after_five_attempts():
    clock = Clock()
    send, sent = scripted(*[Reply(502, None, "0")] * ATTEMPTS, OK, clock=clock)
    scored = one(judge(send, clock))
    assert len(sent) == ATTEMPTS == scored.scores[0].request.attempts and "HTTP 502 after 5 attempts" in scored.scores[0].error
    assert scored.scores[0].request.status == 502 and scored.scores[0].request.usage is None


def test_deadline_covers_attempts_and_waits():
    clock = Clock()
    send, sent = scripted(*[ConnectionError("reset")] * ATTEMPTS, clock=clock, cost=40)
    scored = one(judge(send, clock))
    assert len(sent) == 2 and sent[1] <= 20 and "would exceed the 60s deadline" in scored.scores[0].error
    assert scored.scores[0].request.status is None and scored.scores[0].request.seconds >= 80


@pytest.mark.parametrize("reply", [Reply(200, ["not", "an", "object"]), TransportError("invalid response JSON: x"), Reply(404, None), Reply(401, None)])
def test_malformed_successes_and_client_errors_are_never_retried(reply):
    clock = Clock()
    send, sent = scripted(reply, OK, clock=clock)
    scored = one(judge(send, clock))
    assert len(sent) == 1 and scored.scores[0].error and not clock.slept


def test_connection_errors_retry_and_missing_usage_is_unknown():
    clock = Clock()
    payload = response({"true": math.log(.97), "false": math.log(.02)})
    del payload["usage"]
    send, sent = scripted(ConnectionError("reset"), TimeoutError("slow"), Reply(200, payload), clock=clock)
    scored = one(judge(send, clock))
    assert len(sent) == 3 and len(clock.slept) == 2 and scored.scores[0].error is None
    assert (scored.scores[0].request.attempts, scored.scores[0].request.usage, scored.tokens) == (3, None, 0)


def admitted(judge_):
    semaphore = getattr(judge_, "_admission", None)
    return 0 if semaphore is None else IN_FLIGHT * judge_.batch_inputs - semaphore.value


async def held(release):
    while not release.is_set():
        await anyio.sleep(.005)


def test_admission_is_global_ordered_and_released():
    lock, live, peak, release = threading.Lock(), [0], [0], threading.Event()

    async def send(url, body, timeout):
        body = json.loads(body)
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        await held(release)
        with lock:
            live[0] -= 1
        return Reply(200, response(default_rows(body), prompt_tokens=int(body["messages"][1]["content"].split("\n")[1])))

    judge_ = ss.judge("openrouter:m", codec="chat", transport=send, batch_inputs=2)
    capacity = IN_FLIGHT * judge_.batch_inputs
    outputs = {}
    callers = [threading.Thread(target=lambda n=n: outputs.__setitem__(n, judge_.score_many(
        [(f"{n}{k}", [QUESTIONS[0]]) for k in range(6)]))) for n in range(1, 4)]
    for caller in callers:
        caller.start()
    deadline = time.monotonic() + 5
    while admitted(judge_) < capacity and time.monotonic() < deadline:
        time.sleep(.01)
    time.sleep(.05)
    assert admitted(judge_) == capacity and live[0] == peak[0] == judge_.batch_inputs
    release.set()
    for caller in callers:
        caller.join(5)
    assert peak[0] == judge_.batch_inputs and admitted(judge_) == 0
    assert all([r.tokens for r in outputs[n]] == [n * 10 + k for k in range(6)] for n in range(1, 4))
    judge_.close()


def blocked(timeout):
    release, entered = threading.Event(), threading.Event()

    async def send(url, body, timeout):
        entered.set()
        await held(release)
        return OK
    judge_ = ss.judge("openrouter:m", codec="chat", provider="p", transport=send, batch_inputs=1, timeout=timeout)
    outcome = []

    def run():
        try:
            outcome.append(judge_.score_many([("a", [QUESTIONS[0]]), ("b", [QUESTIONS[0]])]))
        except RuntimeError as error:
            outcome.append(error)
    first = threading.Thread(target=run)
    first.start()
    entered.wait(5)
    while admitted(judge_) < IN_FLIGHT:
        time.sleep(.01)
    return judge_, release, first, outcome


def test_queue_admission_and_service_share_one_total_deadline():
    judge_, release, first, outcome = blocked(.3)
    started = time.monotonic()
    late = judge_.score_many([("c", [QUESTIONS[0]])])[0].scores[0]
    first.join(5)
    assert time.monotonic() - started < 1 and "deadline" in late.error and late.request.queue_seconds > 0
    assert .1 < late.request.seconds < .45 and late.request.seconds == pytest.approx(late.request.queue_seconds + late.request.service_seconds)
    assert all("deadline" in result.scores[0].error for result in outcome[0]) and admitted(judge_) == 0
    release.set()
    judge_.close()


def test_close_wakes_waiters_and_refuses_new_work():
    judge_, release, first, outcome = blocked(60)
    errors = []
    waiter = threading.Thread(target=lambda: errors.append(pytest.raises(RuntimeError, judge_.score_many, [("d", [QUESTIONS[0]])])))
    waiter.start()
    time.sleep(.05)
    closer = threading.Thread(target=judge_.close)
    closer.start()
    waiter.join(5)
    closer.join(5)
    first.join(5)
    assert not waiter.is_alive() and not closer.is_alive() and "closed" in str(errors[0].value)
    assert "closed" in str(outcome[0]) and not release.is_set()
    judge_.close()
    with pytest.raises(RuntimeError, match="closed"):
        judge_.score_many([("e", [QUESTIONS[0]])])


def test_owns_created_transports_and_borrows_supplied(monkeypatch):
    supplied = Provider()
    used = ss.judge("openrouter:m", codec="chat", transport=supplied)
    one(used)
    used.close()
    assert not supplied.closed
    created, transferred = Provider(), Provider()
    monkeypatch.setattr(openrouter.OpenRouter, "wire", lambda *args: created)
    ss.judge("openrouter:m", codec="chat", key="k").close()
    owned = ss.judge("openrouter:m", codec="chat", transport=transferred, own_transport=True)
    one(owned)
    owned.close()
    owned.close()
    assert created.closed and transferred.closed


def test_missing_key_is_actionable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        ss.judge("openrouter:m", codec="chat")


def test_lock_requires_pinned_provider_and_relock(tmp_path):
    path = definition(tmp_path, guidance={}, facts={"refund": {"check": "Refund requested."}})
    with Runtime(FakeScorer(lambda *args: [.99, .01])) as rt:
        tools.lock(rt, path)
    local = files.read_json(path.with_suffix(".lock.json"))
    with Runtime(ss.judge("openrouter:m", codec="chat", transport=Provider()), own_judges=True) as rt:
        with pytest.raises(ConfigError, match="pinned provider"):
            tools.lock(rt, path)
    with Runtime(ss.judge("openrouter:m", codec="chat", provider="p", transport=Provider()), own_judges=True) as rt:
        with pytest.raises(ConfigError, match="re-run `ss lock`"):
            rt.load(path).snapshot
        tools.lock(rt, path)
        sw = rt.load(path)
        assert sw.snapshot.returned_models == {"judge": ("m@p",)} and sw("case").status == "ok"
    assert files.read_json(path.with_suffix(".lock.json"))["models"]["judge"] != local["models"]["judge"]


def test_bench_judge_spec_records_every_attempt(tmp_path, monkeypatch):
    from bench import quality
    from bench.outcomes import judged
    provider = Provider()
    monkeypatch.setattr("semantic_switch.drivers.openrouter.OpenRouter.wire", lambda *args: provider)
    (tmp_path / "key").write_text("OPENROUTER_API_KEY=k\n")
    with judged("openrouter:google/gemma-4-31b-it", provider="p", key_file=tmp_path / "key", device="cpu") as (rt, recording):
        assert rt.judge().provider.route == "p"
        result = quality.run(rt, checks_spec={"refund": quality.CHECKS["refund"]}, picks_spec=[])
        summary = recording.summary()
    assert result["checks"] and all(row["p"] is not None for row in result["checks"])
    assert summary["requests"] == len(provider.bodies) and summary["usage"]["prompt_tokens"] == 7 * summary["requests"]
    assert {row["status"] for row in summary["receipts"]} == {200}
    assert provider.closed


def test_hosted_connection_reuse_against_a_local_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    ports, statuses = [], [429, 200, 200]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            ports.append(self.client_address[1])
            assert self.headers["Authorization"] == "Bearer local-test-key"
            status = statuses.pop(0)
            body = json.dumps(response(default_rows({"messages": [{}, {"content": "true or false"}]}))).encode()
            self.send_response(status)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    wire = openrouter.OpenRouter(key="local-test-key").wire(8, 1 << 20)
    url = f"http://127.0.0.1:{server.server_port}"

    async def exchange():
        try:
            return [await wire(url, b"{}", 5) for _ in range(3)]
        finally:
            await wire.close()
    try:
        replies = anyio.run(exchange)
        assert [(reply.status, reply.retry_after) for reply in replies] == [(429, "0"), (200, "0"), (200, "0")]
        assert replies[1].body["model"] == "m" and len(set(ports)) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_request_records_reach_public_results_and_switch_cost(tmp_path):
    with Runtime(judge(Provider(), Clock()), own_judges=True) as rt:
        shared = rt.ask("Refund me", {"refund": Check("Is the customer asking for a refund?"), "team": Pick("Which team?", {"billing": "charges", "support": "faults"})})
        path = tmp_path / "route.yaml"
        files.publish(path, {"name": "route", "fields": {"message": {"type": "text", "required": True}}, "reads": ["message"],
                             "facts": {"refund": {"check": "Is the customer asking for a refund?"}}, "rules": [{"id": "r", "when": "refund", "then": "refund"}],
                             "otherwise": "other"})
        routed = rt.load(path)({"message": "Refund me"})
    for result in shared.values():
        assert len(result.requests) == 1 and result.requests[0].attempts == 1 and result.requests[0].status == 200
        assert result.as_dict(True)["requests"] == [result.requests[0].as_dict()] and "requests" not in result.as_dict()
    assert shared["refund"].requests[0] is not shared["team"].requests[0]
    assert [request["attempts"] for request in routed.cost["requests"]] == [1] and routed.facts["refund"].requests


def test_jev_profile_tolerates_its_declared_two_decimal_rounding():
    keys, precision = ("A", "B", "C", "D", "E"), JEV_PROFILE.precision
    assert distribution(dict(zip(keys, (.21, .21, .21, .21, .17))), keys, precision)[1].sum() == pytest.approx(1)
    raw, probs = distribution(dict(zip(keys, (.2, .2, .2, .2, .19))), keys, precision)
    assert raw.sum() == pytest.approx(.99) and probs.sum() == pytest.approx(1) and probs[0] == pytest.approx(.2 / .99)
    with pytest.raises(ValueError, match="differs from one"):
        distribution(dict(zip(keys, (.3, .3, .2, .2, .1))), keys, precision)
