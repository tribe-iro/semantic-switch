from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import pytest

from semantic_switch import Pick, Runtime, Select, judge
from semantic_switch.drivers import codecs, openrouter
from semantic_switch.drivers.profiles import CHAT, JEV as JEV_PROFILE
from semantic_switch.hosted import IN_FLIGHT, HostedJudge, Reply, TransportError
from semantic_switch.protocols import Question

from .test_openrouter_judge import JEV, OK, QUESTIONS, Clock, Structured, admitted, default_rows, held, one, response


def hosted_judge(**engine):
    return HostedJudge(openrouter.OpenRouter(), "m", CHAT, **engine)


class Server:
    def __init__(self, handler):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                handler(self)

            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class Local:
    def __init__(self, url, max_bytes=1 << 20):
        self.url, self.wire, self.closed = url, openrouter.OpenRouter(key="local-test-key").wire(2, max_bytes), False

    async def __call__(self, url, body, remaining):
        return await self.wire(self.url, body, remaining)

    async def close(self):
        self.closed = True
        await self.wire.close()


def trickle(request):
    body = json.dumps(response(default_rows({"messages": [{}, {"content": "true or false"}]}))).encode()
    request.send_response(200)
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    try:
        for byte in body:
            request.wfile.write(bytes([byte]))
            request.wfile.flush()
            time.sleep(.05)
    except OSError:
        pass


def test_trickled_body_cannot_outlast_the_total_deadline():
    with Server(trickle) as url:
        local = Local(url)
        judge_ = hosted_judge(transport=local, own_transport=True, timeout=.4)
        started = time.monotonic()
        scored = one(judge_)
        elapsed = time.monotonic() - started
        judge_.close()
    assert "deadline" in scored.scores[0].error and elapsed < 1.5 and local.closed
    assert scored.scores[0].request.attempts == 1 and scored.scores[0].request.seconds < 1.5


def test_response_bytes_are_bounded_before_decoding():
    def large(request):
        body = json.dumps({"padding": "x" * 5000}).encode()
        request.send_response(200)
        request.send_header("Content-Length", str(len(body)))
        request.end_headers()
        request.wfile.write(body)
    with Server(large) as url:
        local = Local(url, max_bytes=1024)
        with pytest.raises(TransportError, match="exceeds 1024 bytes"):
            anyio.run(local, url, b"{}", 5)
        anyio.run(local.close)
    assert hosted_judge(transport=local).max_response_bytes == codecs.Chat.response_bytes
    assert HostedJudge(openrouter.OpenRouter(), JEV, JEV_PROFILE, transport=local).max_response_bytes == codecs.SystemOne.response_bytes
    assert hosted_judge(transport=local, max_response_bytes=4096).max_response_bytes == 4096


@pytest.mark.parametrize("parse", [False, True])
def test_late_success_is_rejected_and_never_retried(parse, monkeypatch):
    clock, sent = Clock(), []

    async def send(url, body, remaining):
        sent.append(remaining)
        clock.now += 1 if parse else 61
        return OK
    if parse:
        decode = codecs.Chat.decode
        monkeypatch.setattr(codecs.Chat, "decode", staticmethod(lambda *args: (clock.__setattr__("now", 61), decode(*args))[1]))
    scored = one(hosted_judge(transport=send, clock=clock, sleep=clock.sleep))
    assert len(sent) == 1 and "after the 60s deadline" in scored.scores[0].error and scored.scores[0].probs is None
    assert scored.scores[0].request.attempts == 1 and scored.scores[0].request.status == 200


def test_retry_waits_consume_the_same_budget():
    clock, replies = Clock(), iter([Reply(503, None, "10"), Reply(503, None, "20"), OK])

    async def send(url, body, remaining):
        clock.now += 5
        return next(replies)
    scored = one(hosted_judge(transport=send, clock=clock, sleep=clock.sleep))
    request = scored.scores[0].request
    assert scored.scores[0].error is None and request.retry_seconds == 30 and request.seconds == 45 and request.service_seconds == 15
    assert request.queue_seconds == 0 and request.attempts == 3 and request.id


def test_cancelled_await_releases_slots_and_starts_no_next_round():
    sent, entered, release = [], threading.Event(), threading.Event()

    async def send(url, body, remaining):
        sent.append(body)
        entered.set()
        await held(release)
        return OK
    judge_ = judge("openrouter:m", codec="chat", transport=send, batch_inputs=1)
    item = Pick("Which unit?", {"AX-1": "first", "AX-2": "second"}, focus="mentions")

    async def scenario(rt):
        task = asyncio.create_task(rt.aask("AX-1 failed", {"unit": item}))
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return rt._active
    with Runtime(judge_, own_judges=True) as rt:
        active = asyncio.run(scenario(rt))
        assert active == 0 and admitted(judge_) == 0 and len(sent) == 1
        time.sleep(.05)
        assert len(sent) == 1
    assert not release.is_set()


def test_mixed_returned_identities_remain_visible():
    versions = iter(["m-v1", "m-v2"])

    async def send(url, body, remaining):
        return Reply(200, response(default_rows(json.loads(body)), model=next(versions)))
    judge_ = judge("openrouter:m", codec="chat", transport=send)
    with pytest.warns(UserWarning, match="mixed model versions"):
        scored = judge_.score_many([("x", QUESTIONS[:2])])[0]
    assert [score.model for score in scored.scores] == ["m-v1@p", "m-v2@p"] and scored.model is None
    assert judge_.observed_models == {"m-v1@p", "m-v2@p"} and judge_.returned is None
    judge_.close()


def test_joint_request_is_counted_once_by_id():
    provider = Structured()
    with Runtime(judge(f"openrouter:{JEV}", transport=provider), own_judges=True) as rt:
        result = rt.ask("refund", {"issues": Select("Which issues?", {"refund": "money back", "late": "delivery"})})["issues"]
    ids = {member.requests[0].id for member in result.members.values()}
    assert len(provider.bodies) == 1 and len(ids) == 1 and [r.id for r in result.requests] == list(ids)
    assert result.as_dict(True)["requests"][0]["id"] in ids


def test_owned_transport_is_closed_on_its_own_loop_after_use():
    with Server(lambda request: (request.send_response(200), request.send_header("Content-Length", "2"), request.end_headers(),
                                 request.wfile.write(b"{}"))) as url:
        local = Local(url)
        judge_ = hosted_judge(transport=local, own_transport=True, batch_inputs=IN_FLIGHT)
        assert "malformed" in one(judge_).scores[0].error
        judge_.close()
    assert local.closed and local.wire.client is None


def chat_reply(body):
    return Reply(200, response(default_rows(json.loads(body))))


def test_unit_failures_are_isolated_and_ordered(monkeypatch):
    import zlib
    decode = codecs.Chat.decode

    async def send(url, body, remaining):
        if b"gzip" in body:
            raise zlib.error("incorrect header check")
        return chat_reply(body)

    def flaky(payload, questions, profile, model, request):
        if "codec bug" in questions[0].text:
            raise RuntimeError("codec bug")
        return decode(payload, questions, profile, model, request)
    monkeypatch.setattr(codecs.Chat, "decode", staticmethod(flaky))
    questions = [QUESTIONS[0], Question("check", "Is this a gzip body?"), Question("check", "Is this a codec bug?"), QUESTIONS[1]]
    judge_ = hosted_judge(transport=send, batch_inputs=1)
    scores = judge_.score_many([("x", questions)])[0].scores
    assert [score.error is None for score in scores] == [True, False, False, True]
    assert "incorrect header check" in scores[1].error and scores[1].request.attempts == 1
    assert "RuntimeError: codec bug" in scores[2].error and all(score.request is not None for score in scores)
    assert admitted(judge_) == 0 and judge_._active.value == 1
    judge_.close()


def test_memory_errors_propagate_from_the_unit_boundary():
    async def send(url, body, remaining):
        raise MemoryError()
    judge_ = hosted_judge(transport=send)
    with pytest.raises(MemoryError):
        one(judge_)
    assert admitted(judge_) == 0
    judge_.close()


def test_body_is_encoded_once_and_reused_across_retries(monkeypatch):
    from semantic_switch import hosted
    clock, payloads, replies, encoded = Clock(), [], iter([Reply(503, None, "1"), Reply(503), OK]), []
    original = hosted.serialized
    monkeypatch.setattr(hosted, "serialized", lambda body: encoded.append(body) or original(body))

    async def send(url, body, remaining):
        payloads.append(body)
        return next(replies)
    scored = one(hosted_judge(transport=send, clock=clock, sleep=clock.sleep))
    assert scored.scores[0].error is None and len(encoded) == 1 and len(payloads) == 3
    assert all(payload is payloads[0] and isinstance(payload, bytes) for payload in payloads)


def test_private_backoff_releases_active_capacity():
    order, retried = [], threading.Event()

    async def send(url, body, remaining):
        name = json.loads(body)["messages"][1]["content"].split("\n")[1]
        order.append(name)
        if order == ["a"]:
            raise ConnectionError("reset")
        return chat_reply(body)

    async def sleep(seconds):
        while "b" not in order:
            await anyio.sleep(.005)
        retried.set()
    judge_ = hosted_judge(transport=send, batch_inputs=1, sleep=sleep, timeout=5)
    results = judge_.score_many([("a", [QUESTIONS[0]]), ("b", [QUESTIONS[0]])])
    assert retried.is_set() and order == ["a", "b", "a"] and all(r.scores[0].error is None for r in results)
    assert results[0].scores[0].request.attempts == 2 and results[0].scores[0].request.retry_seconds > 0
    judge_.close()


def test_shared_retry_after_blocks_every_submission():
    clock, sends = Clock(), []

    async def send(url, body, remaining):
        name = json.loads(body)["messages"][1]["content"].split("\n")[1]
        sends.append((name, clock.now))
        return Reply(429, None, "5") if len(sends) == 1 else chat_reply(body)
    judge_ = hosted_judge(transport=send, batch_inputs=1, clock=clock, sleep=clock.sleep)
    a, b = judge_.score_many([("a", [QUESTIONS[0]]), ("b", [QUESTIONS[0]])])
    assert sends[0] == ("a", 0) and all(at >= 5 for _, at in sends[1:]) and len(sends) == 3
    assert a.scores[0].error is None and b.scores[0].error is None and a.scores[0].request.attempts == 2
    assert b.scores[0].request.attempts == 1 and b.scores[0].request.queue_seconds >= 5
    for request in (a.scores[0].request, b.scores[0].request):
        assert request.seconds == pytest.approx(request.queue_seconds + request.service_seconds + request.retry_seconds)
    judge_.close()


def test_completions_are_posted_per_unit_to_the_caller_sink():
    posted = []

    async def send(url, body, remaining):
        return chat_reply(body)
    judge_ = hosted_judge(transport=send)
    future = judge_.start([("x", QUESTIONS[:2]), ("y", QUESTIONS[:1])], posted.append)
    future.result(5)
    assert sorted((done.owner, done.index) for done in posted) == [(0, (0,)), (0, (1,)), (1, (0,))]
    assert all(len(done.scored.scores) == 1 and done.scored.scores[0].model == "m@p" for done in posted)
    judge_.close()


class Closable:
    def __init__(self, asynchronous=True):
        self.closes, self.asynchronous = 0, asynchronous

    async def __call__(self, url, body, remaining):
        return chat_reply(body)

    def close(self):
        self.closes += 1
        if self.asynchronous:
            return anyio.sleep(0)


@pytest.mark.parametrize("used", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_owned_transport_close_runs_once_inside_a_running_loop(used, asynchronous):
    transport = Closable(asynchronous)
    judge_ = hosted_judge(transport=transport, own_transport=True)
    if used:
        assert one(judge_).scores[0].error is None

    async def inside():
        judge_.close()
        judge_.close()
    asyncio.run(inside())
    assert transport.closes == 1


def test_owned_plain_function_closes_and_aclose_is_idempotent():
    async def plain(url, body, remaining):
        return chat_reply(body)
    judge_ = hosted_judge(transport=plain, own_transport=True)
    assert one(judge_).scores[0].error is None

    async def closing():
        await judge_.aclose()
        await judge_.aclose()
    asyncio.run(closing())
    with pytest.raises(RuntimeError, match="closed"):
        one(judge_)
    borrowed = Closable()
    asyncio.run(hosted_judge(transport=borrowed).aclose())
    assert borrowed.closes == 0


class Stalled(Closable):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()

    async def __call__(self, url, body, remaining):
        self.entered.set()
        await anyio.sleep_forever()


def test_concurrent_close_cancels_active_units_and_closes_once():
    transport = Stalled()
    entered = transport.entered
    judge_ = hosted_judge(transport=transport, own_transport=True)
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(pytest.raises(RuntimeError, one, judge_)))
    worker.start()
    entered.wait(5)
    closers = [threading.Thread(target=judge_.close) for _ in range(3)]
    for closer in closers:
        closer.start()
    for thread in (*closers, worker):
        thread.join(5)
    assert transport.closes == 1 and "closed" in str(outcome[0].value) and admitted(judge_) == 0


def test_invalid_optional_fields_keep_the_valid_distribution():
    reply = {"type": "choice", "probabilities": {"A": .5, "B": .3, "C": .2}, "choice": "Z", "confidence": 7}
    rate = {"type": "score", "probabilities": {"0": .5, "1": .3, "2": .2}, "score": True, "confidence": .4}

    async def send(url, body, remaining):
        return Reply(200, {"model": JEV, "provider": "TypeSafe", "answers": {"q0": reply, "q1": rate}})
    judge_ = HostedJudge(openrouter.OpenRouter(), JEV, JEV_PROFILE, transport=send)
    pick, level = judge_.score_many([("x", [Question("pick", "Which?", ("a", "b", "c")), Question("rate", "How?", ("a", "b", "c"))])])[0].scores
    assert pick.error is None and list(pick.probs) == pytest.approx([.5, .3, .2]) and pick.confidence is None and pick.auxiliary is None
    assert level.error is None and level.confidence == .4 and level.auxiliary is None
    judge_.close()


@pytest.mark.parametrize('status', [429, 503])
def test_exhausted_response_throttles_the_next_unit(status):
    clock, calls = Clock(), []

    async def send(url, body, timeout):
        calls.append(clock())
        if len(calls) <= 5:
            return Reply(status, None, '7' if len(calls) == 5 else '0')
        return OK
    engine = hosted_judge(transport=send, clock=clock, sleep=clock.sleep)
    try:
        first = engine.score_many([('first', QUESTIONS[:1])])[0]
        assert first.scores[0].error is not None
        second = engine.score_many([('second', QUESTIONS[:1])])[0]
        assert second.scores[0].error is None
        assert calls[-1] == 7
    finally:
        engine.close()
