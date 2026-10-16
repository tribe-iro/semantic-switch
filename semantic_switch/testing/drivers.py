from __future__ import annotations

import dataclasses
import inspect
import json
import math
import threading
from dataclasses import dataclass
from typing import Callable

from ..drivers.codecs import CODECS
from ..drivers.profiles import PINNING, UTF8, Budget, ModelProfile
from ..errors import ConfigError
from ..hosted import ATTEMPTS, HostedJudge, Reply, serialized, unique
from ..judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
from ..protocols import Question


class Scripted:
    def __init__(self, respond: Callable):
        self.respond, self.urls, self.payloads, self.bodies, self.remaining, self.closed = respond, [], [], [], [], False

    async def __call__(self, url, payload, remaining):
        body = json.loads(payload)
        self.urls.append(url)
        self.payloads.append(payload)
        self.bodies.append(body)
        self.remaining.append(remaining)
        reply = self.respond(url, body, remaining)
        if inspect.isawaitable(reply):
            reply = await reply
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def systemone(questions, answers) -> dict:
    return {"answers": {f"q{k}": answer for k, answer in enumerate(answers)}}


def systemone_answer(question: Question, probs, profile: ModelProfile) -> dict:
    if question.kind == "check":
        return {"type": "noul", "noul": probs[0]}
    keys = tuple(map(str, range(question.width))) if question.kind == "rate" else profile.ids(question.width)
    return {"type": "score" if question.kind == "rate" else "choice", "probabilities": dict(zip(keys, probs))}


def chat(questions, answers) -> dict:
    (rows,) = answers
    (token, logprob), *_ = rows.items()
    return {"choices": [{"finish_reason": "stop", "logprobs": {"content": [{"token": token, "logprob": logprob,
             "top_logprobs": [{"token": t, "logprob": v} for t, v in rows.items()]}]}}]}


def chat_answer(question: Question, probs, profile: ModelProfile) -> dict:
    from ..presentation import answers
    return {token: math.log(p) for token, p in zip(answers(question), probs) if p > 0}


PAYLOADS = {"systemone": (systemone, systemone_answer), "chat": (chat, chat_answer)}


@dataclass(frozen=True)
class Harness:
    provider: object
    model: str
    profile: ModelProfile
    wrap: Callable[[dict, str | None], dict]
    returned: str | None = None
    secret: str | None = None

    def judge(self, send, **engine) -> HostedJudge:
        return HostedJudge(self.provider, self.model, self.profile, transport=send, **engine)

    def payload(self, questions, distributions, raw=None) -> dict:
        encode, answer = PAYLOADS[self.profile.codec]
        return encode(questions, raw if raw is not None else [answer(q, p, self.profile) for q, p in zip(questions, distributions)])

    def reply(self, questions, distributions, raw=None, route=None) -> Reply:
        return Reply(200, self.wrap(self.payload(questions, distributions, raw), route or self.returned))



class Failed(AssertionError):
    pass


def expect(condition, message):
    if not condition:
        raise Failed(message)


def distributions(question: Question) -> tuple[float, ...]:
    if question.kind == "check":
        return (.75, .25)
    head = round(1 - .05 * (question.width - 1), 2)
    return (head, *(.05,) * (question.width - 1))


def answered(harness: Harness, questions, *, raw=None, route=None, mutate=None):
    def respond(url, body, remaining):
        unit = unit_questions(harness, questions, body)
        dist = [distributions(q) for q in unit]
        return harness.reply(unit, dist, raw=None if raw is None else raw(unit), route=route) if mutate is None else mutate(unit, dist)
    return Scripted(respond)


def unit_questions(harness, questions, body):
    if harness.profile.codec == "chat":
        text = json.dumps(body, ensure_ascii=False)
        return (max((q for q in questions if q.text in text), key=lambda q: len(q.text)),)
    return tuple(questions)


def supported(harness: Harness) -> tuple[Question, ...]:
    kinds = harness.profile.kinds
    return tuple(q for q in CONFORMANCE_QUESTIONS if q.kind in kinds and kinds[q.kind][0] <= q.width <= kinds[q.kind][1])


def ordering(harness: Harness):
    questions = supported(harness)
    send = answered(harness, questions)
    with harness.judge(send) as judge:
        results = judge.score_many([(probe, questions) for probe in CONFORMANCE_PROBES[:2]])
    expect(len(results) == 2 and all(len(r.scores) == len(questions) for r in results), "one ordered score per question and input")
    for result in results:
        for question, score in zip(questions, result.scores):
            expect(score.error is None, f"valid answer failed: {score.error}")
            expected = distributions(question)
            expect(score.probs.shape == (question.width,) and abs(score.probs[0] - expected[0] / sum(expected)) < 1e-9,
                   "scores keep question and option order")
            expect(score.request is not None and score.request.attempts == 1, "each score carries its request record")
    expect(all(url == send.urls[0] for url in send.urls), "one declared endpoint")


def envelope(harness: Harness):
    questions = supported(harness)[:1]
    send = answered(harness, questions)
    with harness.judge(send) as judge:
        judge.score_many([(CONFORMANCE_PROBES[0], questions)])
        expect(send.urls == [judge.url] and judge.url == harness.provider.endpoint(judge.codec), "provider owns the endpoint")
        body = harness.provider.envelope(harness.model, judge.codec, judge.codec.encode(CONFORMANCE_PROBES[0], questions, harness.profile))
        expect(send.payloads == [serialized(body)], "provider owns the request envelope, sent as canonical bytes")


def precision_cases(harness: Harness):
    precision = harness.profile.precision
    if precision is None:
        return
    width = min(harness.profile.kinds.get("pick", (2, 4))[1], 8)
    question = Question("pick", "Which option applies?", tuple(f"option {i}" for i in range(width)))
    bound = precision.bound(width)
    step = 10.0 ** -precision.decimals
    exact = [round(1 / width, precision.decimals)] * width
    exact[0] = round(1 - sum(exact[1:]), precision.decimals)
    for name, delta, ok in (("exact", 0.0, True), ("inside", math.floor(bound / step) * step, True),
                            ("outside", (math.floor(bound / step) + 1) * step, False), ("below", -(math.floor(bound / step) + 1) * step, False)):
        values = list(exact)
        values[-1 if delta < 0 else 0] = round(values[-1 if delta < 0 else 0] + delta, precision.decimals)
        expect(all(0 <= v <= 1 for v in values), f"{name}: fixture leaves [0, 1]")
        score = one(harness, question, raw=lambda unit, values=values: [systemone_answer(question, values, harness.profile)])
        expect((score.error is None) == ok, f"{name} mass {math.fsum(values)!r} with bound {bound!r}: {score.error}")
        if ok:
            expect(abs(float(score.probs.sum()) - 1) < 1e-12 and score.raw is not None and list(score.raw) == values,
                   "declared normalization keeps the raw distribution")
    keys = harness.profile.ids(width)
    for name, mapping in (("nonfinite", {**dict(zip(keys, exact)), keys[0]: math.nan}), ("negative", {**dict(zip(keys, exact)), keys[0]: -.01}),
                          ("missing", dict(zip(keys[1:], exact[1:]))), ("extra", {**dict(zip(keys, exact)), "extra": 0.0}),
                          ("zero", dict.fromkeys(keys, 0.0))):
        score = one(harness, question, raw=lambda unit, mapping=mapping: [{"type": "choice", "probabilities": mapping}])
        expect(score.error is not None and score.probs is None, f"{name} distribution must fail")
    try:
        json.loads('{"A": 0.5, "A": 0.5}', object_pairs_hook=unique)
    except ValueError:
        pass
    else:
        raise Failed("duplicate keys must fail at the wire")


def one(harness, question, raw):
    with harness.judge(answered(harness, (question,), raw=raw)) as judge:
        return judge.score_many([(CONFORMANCE_PROBES[0], [question])])[0].scores[0]


def refusal(harness: Harness):
    if harness.profile.codec != "systemone":
        return
    questions = supported(harness)
    send = Scripted(lambda url, body, remaining: Reply(200, harness.wrap({"refusal": "policy"}, harness.returned)))
    with harness.judge(send) as judge:
        scores = judge.score_many([(CONFORMANCE_PROBES[0], questions)])[0].scores
    expect(all(s.probs is None and "refusal" in s.error for s in scores), "a refusal is an explicit per-question failure")


def limits(harness: Harness):
    send = Scripted(lambda url, body, remaining: Failed("sent"))
    kinds = harness.profile.kinds
    kind = next(iter(k for k in ("pick", "rate") if k in kinds))
    wide = Question(kind, "Which?", tuple(f"o{i}" for i in range(kinds[kind][1] + 1)))
    too_many = (supported(harness)[0],) * (harness.profile.max_questions + 1)
    with harness.judge(send) as judge:
        for items in ([("x", [wide])], [("x", list(too_many))] if judge.codec.joint else None, [(b"image", [supported(harness)[0]])]):
            if items is None:
                continue
            try:
                judge.score_many(items)
            except ConfigError:
                pass
            else:
                raise Failed(f"unsupported request accepted: {items[0][1][0].kind} x{len(items[0][1])}")
    expect(send.bodies == [], "unsupported requests fail before submission")


def retries(harness: Harness):
    question = supported(harness)[0]
    clock = Clock()
    good = answered(harness, (question,))
    replies = iter([Reply(503, None, "1"), Reply(429, None, "2")])
    send = Scripted(lambda url, body, remaining: next(replies, None) or good.respond(url, body, remaining))
    with harness.judge(send, clock=clock, sleep=clock.sleep) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    expect(score.error is None and score.request.attempts == 3 and clock.slept == [1.0, 2.0], "retryable statuses honour Retry-After")
    clock = Clock()
    send = Scripted(lambda url, body, remaining: Reply(503, None, "0"))
    with harness.judge(send, clock=clock, sleep=clock.sleep) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    expect(len(send.bodies) == ATTEMPTS and score.error and "after" in score.error, "bounded attempts")
    send = Scripted(lambda url, body, remaining: Reply(401, {"error": {"message": "no"}}))
    with harness.judge(send, clock=clock, sleep=clock.sleep) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    expect(len(send.bodies) == 1 and score.error, "client errors are not retried")


def deadlines(harness: Harness):
    question = supported(harness)[0]
    clock = Clock()
    good = answered(harness, (question,))

    def late(url, body, remaining):
        clock.now += 61
        return good.respond(url, body, remaining)
    with harness.judge(Scripted(late), clock=clock, sleep=clock.sleep) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    expect(score.probs is None and "deadline" in score.error, "a response after the total deadline fails")
    import anyio
    release, entered = threading.Event(), threading.Event()

    async def held(url, body, remaining):
        entered.set()
        while not release.is_set():
            await anyio.sleep(.005)
        return good.respond(url, body, remaining)
    judge = harness.judge(Scripted(held), timeout=.2, batch_inputs=1)
    score = judge.score_many([("x", [question])])[0].scores[0]
    expect(entered.is_set() and score.probs is None and "deadline" in score.error, "the body deadline bounds a stalled exchange")
    release.set()
    judge.close()


def lifecycle(harness: Harness):
    question = supported(harness)[0]
    borrowed, owned = answered(harness, (question,)), answered(harness, (question,))
    judge = harness.judge(borrowed)
    judge.score_many([("x", [question])])
    judge.close()
    judge.close()
    expect(not borrowed.closed, "a supplied transport is borrowed")
    try:
        judge.score_many([("x", [question])])
    except RuntimeError as error:
        expect("closed" in str(error), "closed judges refuse work")
    else:
        raise Failed("a closed judge accepted work")
    harness.judge(owned, own_transport=True).close()
    expect(owned.closed, "a transferred transport is closed by its owner")

    async def plain(url, body, remaining):
        return answered(harness, (question,)).respond(url, json.loads(body), remaining)
    resource_free = harness.judge(plain, own_transport=True)
    resource_free.score_many([("x", [question])])
    resource_free.close()
    resource_free.close()


def drift(harness: Harness):
    if not getattr(harness.provider, "route", None):
        return
    question = supported(harness)[0]
    with harness.judge(answered(harness, (question,), route="drifted")) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    expect(score.probs is None and "route changed" in score.error, "a changed provider route fails")


def credentials(harness: Harness):
    with harness.judge(Scripted(lambda *args: Failed("sent"))) as judge:
        text = json.dumps(judge.identity, default=str)
    expect(harness.secret is None or harness.secret not in text, "credentials never enter identity")
    expect(all(key in judge.identity for key in ("driver", "codec", "profile", "endpoint", "model", "remote", "score_semantics")), "identity names driver, codec and profile")
    remote = judge.identity["remote"]
    expect(remote["route"] == getattr(harness.provider, "route", None) and remote["guarantee"] in PINNING and
           (remote["weights"] is None) == (remote["guarantee"] == "observed-identity"), "identity separates route, alias, observed model and pinned weights")


def size(harness: Harness, state: str, questions) -> int:
    codec = CODECS[harness.profile.codec]
    return len(serialized(harness.provider.envelope(harness.model, codec, codec.encode(state, tuple(questions), harness.profile))))


def fitting(harness: Harness, question: Question, limit: int) -> str:
    base = size(harness, "", (question,))
    expect(base < limit, "the empty request fits its limit")
    return "x" * (limit - base)


def admission(harness: Harness):
    question = supported(harness)[0]
    fit = fitting(harness, question, harness.profile.limit)
    send = answered(harness, (question,))
    with harness.judge(send) as judge:
        kept, refused = judge.score_many([(fit, [question]), (fit + "x", [question])])
    kept, refused = kept.scores[0], refused.scores[0]
    expect(len(send.payloads) == 1 and len(send.payloads[0]) == harness.profile.limit, "a request exactly at the bound is sent once, unsplit")
    expect(kept.error is None and kept.request.admission["unsupported"] is None and kept.request.admission["bytes"] == harness.profile.limit,
           "the sent request records its measured size")
    expect(refused.probs is None and "unsupported request" in refused.error and refused.request.attempts == 0, "a dynamically oversized input is refused before sending")
    record = refused.request.admission
    expect(record["bytes"] == harness.profile.limit + 1 and record["unsupported"] and record["method"] == harness.profile.admission()["method"],
           "the refusal records method, size and reason")


def overhead(harness: Harness):
    question = supported(harness)[0]
    state = "x" * 64
    measured = size(harness, state, (question,))
    for extra, ok in ((0, True), (1, False)):
        budget = Budget(measured + 7 + 1 - extra, 7, 1)
        bounded = dataclasses.replace(harness, profile=dataclasses.replace(harness.profile, truncation="truncates", budget=budget))
        send = answered(bounded, (question,))
        with bounded.judge(send) as judge:
            score = judge.score_many([(state, [question])])[0].scores[0]
        expect((score.error is None) == ok and len(send.bodies) == ok, f"overhead {'inside' if ok else 'crossing'} the context: {score.error}")
        expect(score.request.admission["capacity"] == measured - extra and score.request.admission["method"] == UTF8,
               "the budget method and capacity are recorded")


def grouping(harness: Harness):
    questions = supported(harness)[:2]
    expect(len(questions) == 2, "two supported questions")
    joint = CODECS[harness.profile.codec].joint
    if joint and harness.profile.max_questions < 2:
        return
    records = []
    for items in ([("x", list(questions))], [("x", [questions[0]]), ("x", [questions[1]])]):
        send = answered(harness, questions)
        with harness.judge(send) as judge:
            results = judge.score_many(items)
        records.append((len(send.bodies), [score.request.admission["grouping"]["questions"] for result in results for score in result.scores]))
    expected = [(1, [2, 2]), (2, [1, 1])] if joint else [(2, [1, 1]), (2, [1, 1])]
    expect(records == expected, f"grouping is recorded and never split silently: {records}")


def mapping(harness: Harness):
    kinds = harness.profile.kinds
    if "pick" not in kinds or kinds["pick"][1] < 3:
        return
    question = Question("pick", "Which option applies?", ("first", "second", "third"))
    values = (.2, .7, .1)
    _, answer = PAYLOADS[harness.profile.codec]

    def reversed_answer(unit):
        reply = answer(question, values, harness.profile)
        if "probabilities" in reply:
            return [{**reply, "probabilities": dict(reversed(reply["probabilities"].items()))}]
        return [dict(reversed(reply.items()))]
    send = answered(harness, (question,), raw=reversed_answer)
    with harness.judge(send) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
        encoded = judge.codec.encode("x", (question,), harness.profile)
    expect(score.error is None and [round(float(p), 9) for p in score.probs / score.probs.sum()] == [round(v / sum(values), 9) for v in values],
           "scores follow the option ids, not the response order")
    expect(send.payloads == [serialized(harness.provider.envelope(harness.model, judge.codec, encoded))], "the sent request is the canonical encoding")
    if harness.profile.codec == "systemone":
        expect(encoded["questions"]["q0"]["criteria"] == dict(zip(harness.profile.ids(3), question.options)), "the request maps each option id exactly")


def means(harness: Harness):
    kinds = harness.profile.kinds
    if "rate" not in kinds or not kinds["rate"][0] <= 3 <= kinds["rate"][1]:
        return
    from ..api import Runtime
    levels, values = ("1", "2", "5"), (.6, 0., .4)
    _, answer = PAYLOADS[harness.profile.codec]

    def reported(unit):
        reply = answer(unit[0], values, harness.profile)
        return [{**reply, "score": .8} if harness.profile.codec == "systemone" else reply]
    send = answered(harness, (Question("rate", "How severe?", levels),), raw=reported)
    with harness.judge(send) as judge, Runtime(judge) as rt:
        result = rt.rate("x", "How severe?", levels, min_score=.55)
        score = judge.score_many([("x", [Question("rate", "How severe?", levels)])])[0].scores[0]
    expect(result.status == "ok" and result.value == "1" and abs(result.expected - .8) < 1e-9,
           f"the categorical Rate reducer, not the rounded mean, chooses the level: {result!r}")
    expect([round(v, 9) for v in result.scores.values()] == [.6, 0., .4], "the full level distribution is retained")
    if harness.profile.codec == "systemone":
        expect(score.auxiliary == {"score": .8, "basis": "level-index-mean"}, "the provider mean is retained only as auxiliary")
        send = answered(harness, (Question("rate", "How severe?", levels),), raw=lambda unit: [{"type": "score", "score": .8}])
        with harness.judge(send) as judge:
            alone = judge.score_many([("x", [Question("rate", "How severe?", levels)])])[0].scores[0]
        expect(alone.probs is None and "per-level distribution" in alone.error, "a mean without its distribution fails")


CASES = (ordering, envelope, precision_cases, refusal, limits, admission, overhead, grouping, mapping, means, retries, deadlines, lifecycle, drift,
         credentials)


def conformance(harness: Harness) -> dict[str, str | None]:
    report = {}
    for case in CASES:
        try:
            case(harness)
        except Failed as error:
            report[case.__name__] = str(error)
        else:
            report[case.__name__] = None
    return report
