from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import pytest

from semantic_switch import Pick, Runtime, Select, hosted, judge_from
from semantic_switch.hosted import IN_FLIGHT, OpenRouterJudge, Reply, TransportError

from .test_openrouter_judge import JEV, OK, QUESTIONS, Clock, Structured, admitted, default_rows, held, one, response


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
        self.url, self.wire, self.closed = url, hosted.http("local-test-key", 2, max_bytes), False

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
        judge_ = OpenRouterJudge("m", transport=local, own_transport=True, timeout=.4)
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
            anyio.run(local, url, {}, 5)
        anyio.run(local.close)
    assert OpenRouterJudge("m", transport=local).max_response_bytes == hosted.Chat.response_bytes
    assert OpenRouterJudge(JEV, transport=local, max_response_bytes=4096).max_response_bytes == 4096


@pytest.mark.parametrize("parse", [False, True])
def test_late_success_is_rejected_and_never_retried(parse, monkeypatch):
    clock, sent = Clock(), []

    async def send(url, body, remaining):
        sent.append(remaining)
        clock.now += 1 if parse else 61
        return OK
    if parse:
        scores = hosted.Chat.scores
        monkeypatch.setattr(hosted.Chat, "scores", staticmethod(lambda response, questions: (clock.__setattr__("now", 61), scores(response, questions))[1]))
    scored = one(OpenRouterJudge("m", transport=send, clock=clock, sleep=clock.sleep))
    assert len(sent) == 1 and "after the 60s deadline" in scored.scores[0].error and scored.scores[0].probs is None
    assert scored.scores[0].request.attempts == 1 and scored.scores[0].request.status == 200


def test_retry_waits_consume_the_same_budget():
    clock, replies = Clock(), iter([Reply(503, None, "10"), Reply(503, None, "20"), OK])

    async def send(url, body, remaining):
        clock.now += 5
        return next(replies)
    scored = one(OpenRouterJudge("m", transport=send, clock=clock, sleep=clock.sleep))
    request = scored.scores[0].request
    assert scored.scores[0].error is None and request.retry_seconds == 30 and request.seconds == 45 == request.service_seconds
    assert request.attempts == 3 and request.id


def test_cancelled_await_releases_slots_and_starts_no_next_round():
    sent, entered, release = [], threading.Event(), threading.Event()

    async def send(url, body, remaining):
        sent.append(body)
        entered.set()
        await held(release)
        return OK
    judge_ = judge_from("openrouter:m", transport=send, batch_inputs=1)
    item = Pick("Which unit?", {"AX-1": "first", "AX-2": "second"}, focus="mentions")

    async def scenario(rt):
        task = asyncio.create_task(rt.aask("AX-1 failed", {"unit": item}))
        await asyncio.get_running_loop().run_in_executor(None, entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return rt._active
    with Runtime({"judge": judge_}, own_judges=True) as rt:
        active = asyncio.run(scenario(rt))
        assert active == 0 and admitted(judge_) == 0 and len(sent) == 1
        time.sleep(.05)
        assert len(sent) == 1
    assert not release.is_set()


def test_mixed_returned_identities_remain_visible():
    versions = iter(["m-v1", "m-v2"])

    async def send(url, body, remaining):
        return Reply(200, response(default_rows(body), model=next(versions)))
    judge_ = judge_from("openrouter:m", transport=send)
    with pytest.warns(UserWarning, match="mixed model versions"):
        scored = judge_.score_many([("x", QUESTIONS[:2])])[0]
    assert [score.model for score in scored.scores] == ["m-v1@p", "m-v2@p"] and scored.model is None
    assert judge_.observed_models == {"m-v1@p", "m-v2@p"} and judge_.returned is None
    judge_.close()


def test_joint_request_is_counted_once_by_id():
    provider = Structured()
    with Runtime({"judge": judge_from(f"openrouter:{JEV}", transport=provider)}, own_judges=True) as rt:
        result = rt.ask("refund", {"issues": Select("Which issues?", {"refund": "money back", "late": "delivery"})})["issues"]
    ids = {member.requests[0].id for member in result.members.values()}
    assert len(provider.bodies) == 1 and len(ids) == 1 and [r.id for r in result.requests] == list(ids)
    assert result.as_dict(True)["requests"][0]["id"] in ids


def test_owned_transport_is_closed_on_its_own_loop_after_use():
    with Server(lambda request: (request.send_response(200), request.send_header("Content-Length", "2"), request.end_headers(),
                                 request.wfile.write(b"{}"))) as url:
        local = Local(url)
        judge_ = OpenRouterJudge("m", transport=local, own_transport=True, batch_inputs=IN_FLIGHT)
        assert "malformed" in one(judge_).scores[0].error
        judge_.close()
    assert local.closed and local.wire.client is None
