from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import semantic_switch as ss
from semantic_switch import ConfigError, Model, Runtime, cli
from semantic_switch.drivers import CONTRACT, GROUP, DriverSpec, Registry, openrouter, pairs
from semantic_switch.drivers.codecs import Chat as CHAT_CODEC, distribution
from semantic_switch.drivers.profiles import CHAT, JEV, LUNA, Budget, ModelProfile, Precision
from semantic_switch.hosted import Envelope, HostedJudge, Reply, serialized
from semantic_switch.permissions import TrainingUse
from semantic_switch.protocols import Question
from semantic_switch.testing.drivers import Harness, Scripted, conformance, systemone_answer

ROOT = Path(__file__).parents[1]
SECRET = "sk-conformance-secret"
FOUR = ModelProfile("conformance-4dp", 1, "systemone", {"check": (2, 2), "pick": (2, 255), "rate": (2, 10)}, 8, 1 << 20,
                    "provider-distribution@1", "renormalize-within-rounding-bound@1", providers=("openrouter", "alternate"),
                    precision=Precision(4, "documented"), options="ordinal")


def openrouter_wrap(payload, returned):
    return {"model": "m-v1", "provider": returned, "usage": {"prompt_tokens": 3}, **payload}


@dataclass(frozen=True)
class Alternate:
    key: str = field(default=SECRET, repr=False)
    route = None
    name = "alternate"
    revision = 1
    identity = {"service": "https://judge.example.invalid/v2"}

    def endpoint(self, codec):
        return f"https://judge.example.invalid/v2/{codec.name}/decide"

    def wire(self, connections, max_bytes):
        raise AssertionError("conformance supplies its transport")

    def envelope(self, model, codec, body):
        return {"target": model, "request": body}

    @staticmethod
    def usage(body):
        return body.get("meta")

    @staticmethod
    def unwrap(body):
        meta = body["meta"]
        return Envelope(meta["model"], None, meta["input_tokens"], body["result"])

    def rejected(self, reply):
        return f"HTTP {reply.status}"


HARNESSES = {
    "openrouter-chat": Harness(openrouter.OpenRouter("p", SECRET), "m", CHAT, openrouter_wrap, "p", SECRET),
    "openrouter-jev": Harness(openrouter.OpenRouter("TypeSafe", SECRET), "typesafe/jev-1.13", JEV, openrouter_wrap, "TypeSafe", SECRET),
    "openrouter-four-decimal": Harness(openrouter.OpenRouter("p", SECRET), "m", FOUR, openrouter_wrap, "p", SECRET),
    "alternate-systemone": Harness(Alternate(), "alt-1", FOUR, lambda payload, returned: {"result": payload, "meta": {"model": "alt-1-2026", "input_tokens": 5}},
                                   None, SECRET),
}


@pytest.mark.parametrize("name", HARNESSES)
def test_driver_conformance(name):
    report = conformance(HARNESSES[name])
    assert report == dict.fromkeys(report), {case: failure for case, failure in report.items() if failure}


def test_alternate_provider_keeps_codec_and_changes_only_the_envelope():
    question = Question("pick", "Which?", ("a", "b"))
    harness = HARNESSES["alternate-systemone"]
    send = Scripted(lambda url, body, remaining: harness.reply((question,), [(.6, .4)]))
    with harness.judge(send) as judge:
        score = judge.score_many([("case", [question])])[0]
    assert send.urls == ["https://judge.example.invalid/v2/systemone/decide"] and set(send.bodies[0]) == {"target", "request"}
    assert send.bodies[0]["request"]["questions"]["q0"]["criteria"] == {"o0": "a", "o1": "b"}
    assert score.model == "alt-1-2026" and score.tokens == 5 and np.allclose(score.scores[0].probs, [.6, .4])
    assert judge.identity["source"] == "alternate" and judge.identity["remote"]["route"] is None and "openrouter" not in json.dumps(judge.identity)


def wide(width):
    return Question("pick", "Which option applies?", tuple(f"option {i}" for i in range(width)))


def decoded(profile, width, values):
    question = wide(width)
    harness = Harness(openrouter.OpenRouter("p", SECRET), "m", profile, openrouter_wrap, "p")
    send = Scripted(lambda url, body, remaining: harness.reply((question,), (), raw=[{"type": "choice", "probabilities": dict(zip(profile.ids(width), values))}]))
    with harness.judge(send) as judge:
        return judge.score_many([("case", [question])])[0].scores[0]


def spread(width, total):
    values = [0.0] * width
    values[0] = total
    if total > 1:
        values[0], values[1] = 1.0, round(total - 1, 4)
    return values


@pytest.mark.parametrize("width,total", [(255, 2.0), (20, 1.09), (4, 1.019)])
def test_four_decimal_profile_rejects_excess_mass(width, total):
    values = [1.0, *[(total - 1) / (width - 1)] * (width - 1)] if width == 255 else spread(width, total)
    assert math.fsum(values) == pytest.approx(total)
    score = decoded(FOUR, width, values)
    assert score.probs is None and "declared 4-decimal bound" in score.error


def test_four_decimal_profile_accepts_valid_wide_rounding():
    values = [round(.9945 - 254 * .0039, 4), *[.0039] * 254]
    assert math.fsum(values) == pytest.approx(.9945) and FOUR.precision.bound(255) == pytest.approx(255 * (5e-5 + 2 ** -52))
    score = decoded(FOUR, 255, values)
    assert score.error is None and score.probs.sum() == pytest.approx(1) and score.raw.sum() == pytest.approx(.9945)
    assert score.probs[1] == pytest.approx(.0039 / .9945)


@pytest.mark.parametrize("excess,ok", [(0.0, True), (.0002, True), (.0003, False), (-.0002, True), (-.0003, False)])
def test_four_decimal_bound_is_width_times_half_unit(excess, ok):
    keys = FOUR.ids(4)
    values = dict(zip(keys, (round(.25 + excess, 4), .25, .25, .25)))
    if ok:
        raw, probs = distribution(values, keys, FOUR.precision)
        assert probs.sum() == pytest.approx(1) and raw[0] == round(.25 + excess, 4)
    else:
        with pytest.raises(ValueError, match="differs from one"):
            distribution(values, keys, FOUR.precision)


@pytest.mark.parametrize("values,error", [({"o0": math.inf, "o1": 0}, "finite"), ({"o0": -.1, "o1": 1.1}, "finite"), ({"o0": 1.0}, "exactly once"),
                                          ({"o0": .5, "o1": .5, "o2": 0}, "exactly once"), ({"o0": 0, "o1": 0}, "zero total"),
                                          ({"o0": True, "o1": 0}, "number")])
def test_distribution_rejects_malformed_entries_rather_than_repairing(values, error):
    with pytest.raises(ValueError, match=error):
        distribution(values, FOUR.ids(2), FOUR.precision)


def test_two_decimal_policy_bound_differs_from_four_decimal():
    keys = JEV.ids(3)
    assert distribution(dict(zip(keys, (.5, .3, .19))), keys, JEV.precision)[1].sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        distribution(dict(zip(keys, (.5, .3, .19))), keys, Precision(4, "documented"))
    assert JEV.precision.bound(3) == pytest.approx(3 * (.005 + 2 ** -52))
    with pytest.raises(ConfigError, match="cannot bound the mass"):
        ModelProfile("bad", 1, "systemone", {"pick": (2, 255)}, 1, 100, "s", "n", precision=Precision(2, "empirical"))
    assert Precision(2, "empirical").record()["rounding"] == "nearest"


def records(path):
    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("response"), dict) and "answers" in node["response"]:
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
    unique, routes = {}, set()
    for record in walk(json.loads(path.read_bytes())):
        text = json.dumps(record["response"], sort_keys=True, separators=(",", ":"))
        assert unique.setdefault(record["response"]["id"], text) == text
        routes.add(f"{record['response']['model']}@{record['response']['provider']}")
    return unique, routes


@pytest.mark.parametrize("fixture", [fixture for profile in (JEV, LUNA) for fixture in profile.evidence], ids=lambda f: Path(f.path).name)
def test_empirical_profiles_pin_their_response_fixtures(fixture):
    path = ROOT / fixture.path
    if not path.exists():
        pytest.skip(f"{fixture.path} is not in this checkout")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture.sha256
    unique, routes = records(path)
    digest = hashlib.sha256("\n".join(hashlib.sha256(text.encode()).hexdigest() for text in sorted(unique.values())).encode()).hexdigest()
    widths, lattice, mass, count = {}, 0.0, 0.0, 0
    for text in unique.values():
        for answer in json.loads(text)["answers"].values():
            if "probabilities" in answer:
                values = list(answer["probabilities"].values())
                count += 1
                widths.setdefault(answer["type"], set()).add(len(values))
                lattice = max(lattice, *(abs(v - round(v, 2)) for v in values))
                mass = max(mass, abs(math.fsum(values) - 1))
    assert (len(unique), digest, count, sorted(routes)) == (fixture.responses, fixture.records, fixture.distributions, list(fixture.observed))
    assert {kind: tuple(sorted(sizes)) for kind, sizes in widths.items()} == dict(fixture.widths)
    assert lattice == fixture.lattice_error <= 2 ** -52 and mass == fixture.mass_error <= .01


def test_jev_fixture_mass_error_fits_its_two_decimal_bound_and_is_one_response():
    fixture = JEV.evidence[1]
    unique, _ = records(ROOT / fixture.path) if (ROOT / fixture.path).exists() else pytest.skip("fixture absent")
    short = [answer for text in unique.values() for answer in json.loads(text)["answers"].values()
             if "probabilities" in answer and abs(math.fsum(answer["probabilities"].values()) - 1) > 1e-9]
    assert len(short) == 1 and len(short[0]["probabilities"]) == 3
    raw, probs = distribution(short[0]["probabilities"], JEV.ids(3), JEV.precision)
    assert raw.sum() == pytest.approx(.99) and probs.sum() == pytest.approx(1)


def test_identity_freezes_profile_and_excludes_credentials():
    judge = ss.judge("openrouter:typesafe/jev-1.13", provider="TypeSafe", key=SECRET)
    try:
        identity = judge.identity
        assert SECRET not in json.dumps(identity) and identity["driver"] == "openrouter@1" and identity["codec"] == "systemone@4"
        assert identity["profile"]["record"]["evidence"][0]["sha256"] == JEV.evidence[0].sha256 and identity["precision"]["basis"] == "empirical"
        assert identity["profile"]["record"]["options"] == "letters" and identity["normalization"] == "renormalize-within-rounding-bound@1"
    finally:
        judge.close()


def test_registry_resolves_sources_once_and_refuses_unknown_schemes(tmp_path):
    local = tmp_path / "weird:name.gguf"
    local.write_bytes(b"")
    assert isinstance(ss.judge("models/x.gguf"), Model) and ss.judge("gguf:models/x.gguf").source == "models/x.gguf"
    assert ss.judge("ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf").source == "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf"
    assert ss.judge(str(local)).source == str(local)
    with pytest.raises(ConfigError, match="unknown judge scheme 'openroute'.*did you mean 'openrouter'"):
        ss.judge("openroute:m")
    with pytest.raises(ConfigError, match="unknown judge scheme 'jev'"):
        ss.judge("jev:typesafe/jev-1.13")


def test_settings_are_typed_and_misapplied_settings_fail_before_requests(monkeypatch):
    monkeypatch.setattr(openrouter.OpenRouter, "wire", lambda *args: pytest.fail("transport created"))
    with pytest.raises(ConfigError, match="device does not apply to openrouter judges"):
        ss.judge("openrouter:m", codec="chat", device="cpu")
    with pytest.raises(ConfigError, match="provider does not apply to gguf judges"):
        ss.judge("x.gguf", provider="p")
    with pytest.raises(ConfigError, match="timeout='soon' is not a valid"):
        ss.judge("openrouter:m", codec="chat", timeout="soon")
    with pytest.raises(ConfigError, match="transport cannot be given as text"):
        ss.judge("openrouter:m", codec="chat", transport="http://x")
    with pytest.raises(ConfigError, match="unknown codec 'xml'"):
        ss.judge("openrouter:m", codec="xml")
    assert ss.judge("x.gguf", threads="4", repack="false") == Model("x.gguf", threads=4, repack=False)
    assert pairs(["codec=chat", "timeout=5", "key=a=b"]) == {"codec": "chat", "timeout": "5", "key": "a=b"}
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        pairs(["codec"])
    with pytest.raises(ConfigError, match="more than once"):
        pairs(["codec=chat", "codec=systemone"])


def test_cli_forwards_ordinary_options_and_settings_into_one_validator(monkeypatch):
    seen = []

    def judge(source, **settings):
        seen.append((source, settings))
        raise ConfigError("stop")
    monkeypatch.setattr(cli, "judge", judge)
    runner = CliRunner()
    runner.invoke(cli.app, ["--judge", "openrouter:m", "--provider", "p", "--setting", "codec=chat", "--setting", "timeout=5", "check", "Claim."], input="x")
    assert seen == [("openrouter:m", {"provider": "p", "codec": "chat", "timeout": "5"})]
    result = runner.invoke(cli.app, ["--judge", "openrouter:m", "--provider", "p", "--setting", "provider=q", "check", "Claim."], input="x")
    assert isinstance(result.exception, ConfigError) and "both as an option and as --setting" in str(result.exception)
    assert "--setting" in runner.invoke(cli.app, ["--help"]).output


def driver(scheme="example", contract=CONTRACT):
    @dataclass(frozen=True)
    class Settings:
        depth: int = 1
    return DriverSpec(scheme, contract, Settings, lambda target, settings: ("built", target, Settings(**settings)), 1, "example")


class Point:
    def __init__(self, name, value, loaded):
        self.name, self.value, self.dist, self.loaded = name, value, None, loaded

    def load(self):
        self.loaded.append(self.name)
        return self.value


def test_entry_point_drivers_load_lazily_and_duplicates_fail(monkeypatch):
    loaded, scans = [], []
    points = [Point("example", driver(), loaded), Point("twice", driver("twice"), loaded), Point("twice", driver("twice"), loaded),
              Point("old", driver("old", CONTRACT + 1), loaded)]
    monkeypatch.setattr("semantic_switch.drivers.metadata.entry_points", lambda group: scans.append(group) or [p for p in points if group == GROUP])
    registry = Registry()
    assert isinstance(registry.build("x.gguf", {}), Model) and "gguf:" in registry.help() and scans == []
    assert "example" in registry.schemes() and loaded == [] and scans == [GROUP]
    assert registry.build("example:target", {"depth": "3"})[:2] == ("built", "target") and loaded == ["example"]
    with pytest.raises(ConfigError, match="registered more than once"):
        registry.build("twice:x", {})
    with pytest.raises(ConfigError, match=f"implements contract {CONTRACT + 1}"):
        registry.build("old:x", {})
    assert scans == [GROUP]
    shadowed = Registry()
    points.append(Point("gguf", driver("gguf"), loaded))
    with pytest.raises(ConfigError, match="'gguf' is registered more than once: builtin"):
        shadowed.build("example:target", {})
    points.pop()
    with pytest.raises(ConfigError, match="already registered"):
        registry.register(driver("example"))
    with pytest.raises(ConfigError, match="contract"):
        registry.register(driver("fresh", CONTRACT + 1))
    registry.register(driver("fresh"))
    assert registry.build("fresh:a", {})[1] == "a" and loaded == ["example", "old"]


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)


def test_runtime_constructs_text_judges_through_the_registry_and_owns_them(key):
    send = Scripted(lambda url, body, remaining: Reply(200, {"model": "m", "provider": "p", "choices": []}))
    with Runtime(ss.judge("openrouter:m", codec="chat", transport=send, own_transport=True)) as rt:
        assert rt.identity()["codec"] == "chat@2"
    assert not send.closed
    with Runtime("openrouter:google/gemma-4-31b-it") as rt:
        judge = rt.judge()
        assert isinstance(judge, HostedJudge) and judge.profile is CHAT
    assert judge._closed
    with pytest.raises(ConfigError, match="no built-in profile"):
        Runtime("openrouter:unknown/model")


def test_profile_limits_are_checked_before_submission():
    send = Scripted(lambda *args: pytest.fail("sent"))
    judge = ss.judge("openrouter:typesafe/jev-1.13", transport=send)
    with pytest.raises(ConfigError, match="at most 16 questions"):
        judge.score_many([("case", [Question("check", "Claim.")] * 17)])
    with pytest.raises(ConfigError, match="at most 20 answer slots"):
        judge.score_many([("case", [wide(21)])])
    with pytest.raises(ConfigError, match="2 to 10 levels"):
        judge.score_many([("case", [Question("rate", "How?", tuple("abcdefghijk"))])])
    with pytest.raises(ConfigError, match="text evidence only"):
        judge.score_many([({"image": b"..."}, [Question("check", "Claim.")])])
    refused = judge.score_many([("x" * 70000, [Question("check", "Claim.")])])[0].scores[0]
    assert refused.probs is None and "65536-byte request limit" in refused.error and refused.request.attempts == 0
    with pytest.raises(ConfigError, match="tested token budget"):
        ModelProfile("t", 1, "chat", {"check": (2, 2)}, 1, 100, "s", "n", truncation="truncates")
    with pytest.raises(ConfigError, match="budget must be a Budget"):
        ModelProfile("t", 1, "chat", {"check": (2, 2)}, 1, 100, "s", "n", truncation="truncates", budget=lambda *a: None)
    judge.close()


def test_token_budget_capacity_is_context_minus_overhead_and_reserve():
    budget = Budget(4096, 300, 16)
    assert budget.capacity == 3780 and budget.record() == {"method": "utf8-bytes-upper-bound@1", "context": 4096, "overhead": 300,
                                                         "reserve": 16, "capacity": 3780}
    profile = ModelProfile("t", 1, "chat", {"check": (2, 2)}, 1, 1 << 16, "s", "n", truncation="truncates", budget=budget)
    assert profile.limit == 3780 and profile.unsupported(3780) is None and "3780 tokens after 300 overhead" in profile.unsupported(3781)
    assert ModelProfile("r", 1, "chat", {"check": (2, 2)}, 1, 2000, "s", "n", budget=budget).limit == 2000
    with pytest.raises(ConfigError, match="leave no room"):
        Budget(100, 99, 1)


def test_truncating_profile_refuses_when_overhead_crosses_the_context():
    question = Question("check", "Claim.")
    body = openrouter.OpenRouter("p").envelope("m", CHAT_CODEC, CHAT_CODEC.encode("a long input text", (question,), CHAT))
    measured = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())
    sent = []
    for overhead, ok in ((100, True), (101, False)):
        profile = ModelProfile("truncating", 1, "chat", {"check": (2, 2)}, 1, 1 << 16, "answer-token-lower-bound@2:top20", "token-mass-lower-bound@1",
                               truncation="truncates", budget=Budget(measured + 101, overhead, 1))
        send = Scripted(lambda url, body, remaining: sent.append(body) or Reply(500))
        with HostedJudge(openrouter.OpenRouter("p"), "m", profile, transport=send, timeout=.01) as judge:
            score = judge.score_many([("a long input text", [question])])[0].scores[0]
        assert score.request.admission["bytes"] == measured and score.request.admission["capacity"] == measured + 100 - overhead
        assert score.request.admission["method"] == "utf8-bytes-upper-bound@1" and score.request.admission["grouping"] == {"scope": "individual", "questions": 1}
        assert (score.request.admission["unsupported"] is None) == ok and (len(send.bodies) > 0) == ok
    assert "completeness cannot be established" in score.error


def test_built_in_wire_sends_the_measured_canonical_bytes(monkeypatch):
    import httpx
    seen = []

    def handler(request):
        seen.append(request.content)
        return httpx.Response(200, json={"model": "m", "provider": "p", "choices": []})
    wire = openrouter.OpenRouter(key="k").wire(8, 1 << 20)
    wire.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    body = serialized({"model": "m", "state": "é ü", "questions": {"q0": {"type": "noul"}}})
    import anyio
    anyio.run(wire, "https://x.invalid", body, 5.)
    anyio.run(wire.close)
    assert seen == ['{"model":"m","state":"é ü","questions":{"q0":{"type":"noul"}}}'.encode()]


def test_switch_load_checks_static_shapes_against_the_bound_profile(tmp_path):
    send = Scripted(lambda *args: pytest.fail("sent"))
    narrow = ModelProfile("narrow", 1, "systemone", {"check": (2, 2), "pick": (2, 4)}, 4, 1 << 16, "provider-distribution@1",
                          "renormalize-within-rounding-bound@1", precision=Precision(2, "documented"))
    jev, small = ss.judge("openrouter:typesafe/jev-1.13", transport=send), HostedJudge(openrouter.OpenRouter("p"), "m", narrow, transport=send)
    levels = [f"level {i}" for i in range(11)]
    cases = ((jev, {"rate": "How urgent?", "levels": levels}, "binding 'judge': urgency: profile jev-1.13-systemone-2dp takes 2 to 10 levels for rate, got 11"),
             (small, {"pick": "Which team?", "options": {f"t{i}": f"team {i}" for i in range(5)}}, "urgency: this judge takes at most 4 answer slots, got 5"),
             (small, {"rate": "How urgent?", "levels": levels[:3]}, "urgency: profile narrow does not support rate questions"))
    for n, (judge, fact, error) in enumerate(cases):
        path = tmp_path / f"case{n}.yaml"
        path.write_text(json.dumps({"name": "t", "fields": {"text": {"type": "text"}}, "reads": ["text"], "facts": {"urgency": fact}}))
        with pytest.raises(ConfigError, match=re.escape(f"case{n}.yaml: {error}" if n == 0 else error)):
            Runtime(judge).load(path, draft=True)
    path = tmp_path / "ok.yaml"
    path.write_text(json.dumps({"name": "t", "fields": {"text": {"type": "text"}}, "reads": ["text"],
                                "facts": {"urgency": {"rate": "How urgent?", "levels": levels[:10]}}}))
    Runtime(jev).load(path, draft=True)
    assert send.bodies == []
    jev.close()
    small.close()


def test_provider_mean_between_levels_never_chooses_the_class():
    levels = ("can wait", "today", "immediately")
    reply = {"type": "score", "score": 0.78, "legend": dict(zip("012", levels)), "probabilities": {"0": 0.59, "1": 0.04, "2": 0.37}, "confidence": 0}
    send = Scripted(lambda url, body, remaining: Reply(200, {"model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe", "answers": {"q0": reply}}))
    with Runtime(ss.judge("openrouter:typesafe/jev-1.13", provider="TypeSafe", transport=send)) as rt:
        result = rt.rate("Checkout is down.", "How soon?", levels, min_score=.55)
        raw = rt.judge().score_many([("Checkout is down.", [Question("rate", "How soon?", levels)])])[0].scores[0]
    assert result.value == "can wait" and result.index == 0 and round(0.78) == 1
    assert result.expected == pytest.approx(0 * .59 + 1 * .04 + 2 * .37) and result.scores == pytest.approx(dict(zip(levels, (.59, .04, .37))))
    assert raw.auxiliary == {"score": 0.78, "basis": "level-index-mean"} and list(raw.raw) == [.59, .04, .37]
    assert result.requests[0].admission["grouping"] == {"scope": "joint", "questions": 1} and result.requests[0].admission["method"] == "service-rejects-oversize@1"


@pytest.mark.parametrize("field,value", [("score", 2.5), ("score", True), ("choice", "Z"), ("confidence", 1.5), ("confidence", "high")])
def test_inconsistent_optional_fields_are_discarded_without_changing_the_distribution(field, value):
    kind = "rate" if field == "score" else "pick"
    question = Question(kind, "How?", ("a", "b", "c"))
    keys = {"rate": ("0", "1", "2"), "pick": ("A", "B", "C")}[kind]
    reply = {"type": "score" if kind == "rate" else "choice", "probabilities": dict(zip(keys, (.5, .3, .2))), field: value}
    send = Scripted(lambda url, body, remaining: Reply(200, {"model": "m", "provider": "TypeSafe", "answers": {"q0": reply}}))
    with ss.judge("openrouter:typesafe/jev-1.13", provider="TypeSafe", transport=send) as judge:
        score = judge.score_many([("x", [question])])[0].scores[0]
    assert score.error is None and list(score.probs) == pytest.approx([.5, .3, .2]) and score.auxiliary is None and score.confidence is None


def test_remote_identity_separates_route_alias_observed_model_and_weights():
    send = Scripted(lambda url, body, remaining: Reply(200, {"model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
                                                             "answers": {"q0": {"type": "noul", "noul": .9}}}))
    with Runtime(ss.judge("openrouter:typesafe/jev-1.13", provider="TypeSafe", transport=send)) as rt:
        identity = rt.identity()
        rt.check("x", "Claim.")
        observed = rt.info()["observed_models"]
    assert identity["model"] == "typesafe/jev-1.13" and identity["remote"] == {"route": "TypeSafe", "observed": "per-response", "weights": None,
                                                                              "guarantee": "observed-identity"}
    assert observed == ["typesafe/jev-1.13-20260917@TypeSafe"] and identity["profile"]["record"]["pinning"] == "observed-identity"
    with pytest.raises(ConfigError, match="not proof of frozen weights"):
        ModelProfile("p", 1, "systemone", {"check": (2, 2)}, 1, 100, "s", "n", precision=Precision(2, "documented"), pinning="pinned-snapshot")
    with pytest.raises(ConfigError, match="not proof of frozen weights"):
        ModelProfile("p", 1, "systemone", {"check": (2, 2)}, 1, 100, "s", "n", precision=Precision(2, "documented"), weights="sha256:" + "0" * 64)
    pinned = ModelProfile("p", 1, "systemone", {"check": (2, 2)}, 1, 100, "s", "n", providers=("alternate",), precision=Precision(2, "documented"),
                          pinning="pinned-snapshot", weights="sha256:" + "0" * 64)
    judge = HostedJudge(Alternate(), "alt-1", pinned, transport=Scripted(lambda *a: pytest.fail("sent")))
    assert judge.identity["remote"] == {"route": None, "observed": "per-response", "weights": "sha256:" + "0" * 64, "guarantee": "pinned-snapshot"}
    judge.close()


def test_training_use_declarations():
    allowed = TrainingUse("allowed", "google/gemma-4-31b-it", "Example", "https://example.invalid/terms", "2026-10-01", "teacher-labels")
    assert TrainingUse.parse(allowed.record()) == allowed and len(allowed.digest) == 64
    allowed.require("google/gemma-4-31b-it@Example")
    with pytest.raises(ConfigError, match="differs from the declared"):
        allowed.require("google/gemma-4-31b-it@Other")
    for permission in ("unknown", "forbidden"):
        with pytest.raises(ConfigError, match=f"declared {permission}"):
            TrainingUse(permission, "m", None, "terms", "1", "teacher-labels").require("m")
    for model in ("typesafe/jev-1.13", "openrouter:TypeSafe/Jev-2"):
        with pytest.raises(ConfigError, match="§2.3"):
            TrainingUse("allowed", model, "TypeSafe", "plugin-capability", "1", "teacher-labels")
    with pytest.raises(ConfigError, match="§2.3"):
        TrainingUse("allowed", "m", None, "terms", "1", "teacher-labels").require("typesafe/jev-1.13-20260917@TypeSafe")
    with pytest.raises(ConfigError, match="exactly"):
        TrainingUse.parse({**allowed.record(), "capability": "train"})
    assert systemone_answer(Question("check", "c"), (.4, .6), JEV) == {"type": "noul", "noul": .4}
