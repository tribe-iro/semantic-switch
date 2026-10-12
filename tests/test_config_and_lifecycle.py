from __future__ import annotations

import json
import subprocess
import sys
import threading

import pytest
from typer.testing import CliRunner

from semantic_switch import DEFAULT_JUDGE, ConfigError, Model, Runtime, api, cli, files, hosted, judge_from, tools
from semantic_switch.hosted import OpenRouterJudge

from .conftest import FakeScorer
from .test_artifact_pipeline import dataset
from .test_native_resources import QWEN3
from .test_openrouter_judge import Provider
from .test_remaining_contracts import definition

SECRET = "sk-or-secret-value"


class Closing(FakeScorer):
    def __init__(self, closed, name, error=None, oracle=None):
        super().__init__(*(() if oracle is None else (oracle,)), name=name)
        self.closed, self.error = closed, error

    def close(self):
        self.closed.append(self.name)
        if self.error:
            raise self.error


class Wrapper:
    def __init__(self, scorer):
        self.scorer, self.identity, self.accelerated, self.max_slots, self.batch_inputs = scorer, {"source": "wrapper"}, False, 20, 1

    def score_many(self, items):
        raise AssertionError("unused")


def test_close_releases_unused_owned_bindings_once(monkeypatch):
    monkeypatch.setattr(api, "_resolve", lambda source: pytest.fail("loaded an unused model to close it"))
    closed = []
    first, shared = Closing(closed, "first"), Closing(closed, "shared")
    rt = Runtime({"judge": first, "alias": first, "wrapped": Wrapper(shared), "raw": shared, "local": Model("unused.gguf")}, own_judges=True)
    rt.close()
    rt.close()
    assert closed == ["first", "shared"]


def test_close_keeps_borrowed_bindings_and_transports(monkeypatch):
    closed, supplied, created = [], Provider(), Provider()
    Runtime({"judge": Closing(closed, "borrowed")}).close()
    with Runtime({"judge": judge_from("openrouter:m", transport=supplied)}, own_judges=True):
        pass
    monkeypatch.setattr(hosted, "http", lambda *args: created)
    with Runtime({"judge": judge_from("openrouter:m")}, own_judges=True):
        pass
    assert closed == [] and not supplied.closed and created.closed


def test_close_attempts_every_resource_and_reports_all_failures():
    closed = []
    rt = Runtime({"judge": Closing(closed, "a", RuntimeError("a failed")), "b": Closing(closed, "b"),
                  "c": Closing(closed, "c", ValueError("c failed"))}, own_judges=True)
    with pytest.raises(ExceptionGroup) as raised:
        rt.close()
    assert closed == ["a", "b", "c"] and [str(error) for error in raised.value.exceptions] == ["a failed", "c failed"]
    rt.close()


def test_close_waits_for_active_operation_then_refuses_calls():
    closed, entered, release, calls = [], threading.Event(), threading.Event(), []

    def oracle(state, question, width):
        calls.append(state)
        entered.set()
        release.wait(5)
        return [.99, .01]

    rt = Runtime({"judge": Closing(closed, "judge", oracle=oracle)}, own_judges=True)
    results = []
    worker = threading.Thread(target=lambda: results.append(rt.check("case", "Claim.")))
    worker.start()
    entered.wait(5)
    closer = threading.Thread(target=rt.close)
    closer.start()
    closer.join(.2)
    assert closer.is_alive() and closed == []
    release.set()
    worker.join(5)
    closer.join(5)
    assert results[0].status == "ok" and closed == ["judge"]
    for call in (lambda: rt.check("case", "Claim."), rt.judge, lambda: rt.score_native([])):
        with pytest.raises(RuntimeError, match="runtime is closed"):
            call()
    assert len(calls) == 1


def test_closed_hosted_runtime_never_sends():
    provider = Provider()
    rt = Runtime({"judge": judge_from("openrouter:m", transport=provider)}, own_judges=True)
    rt.close()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        rt.check("case", "Claim.")
    assert provider.bodies == []


def test_judge_from_returns_existing_judges(monkeypatch):
    monkeypatch.setattr(hosted, "http", lambda *args: pytest.fail("supplied transport ignored"))
    local = judge_from("models/x.gguf", device="cpu", threads=2, repack=False)
    assert local == Model("models/x.gguf", device="cpu", threads=2, repack=False)
    routed = judge_from("openrouter:m", provider="p", transport=Provider(), timeout=5, batch_inputs=2)
    structured = judge_from("openrouter:typesafe/jev-1.13", transport=Provider(), batch_inputs=3)
    assert isinstance(routed, OpenRouterJudge) and (routed.model, routed.provider, routed.batch_inputs, routed.timeout) == ("m", "p", 2, 5)
    assert structured.identity["protocol"] == "systemone" and structured.batch_inputs == 3 and not routed.own_transport
    assert SECRET not in json.dumps(routed.identity)


@pytest.mark.parametrize("source, settings, message", [
    ("", {}, "judge source must be nonempty"),
    ("openrouter:", {}, "pass the provider separately"),
    ("openrouter:m@p", {}, "pass the provider separately"),
    ("openrouter:m", {"provider": ""}, "provider must be nonempty"),
    ("openrouter:m", {"key": " "}, "key must be nonempty"),
    ("openrouter:m", {"device": "cpu", "repack": False}, "device, repack apply only to local GGUF judges"),
    ("openrouter:m", {"timeout": 0}, "timeout must be a positive"),
    ("openrouter:m", {"batch_inputs": 0}, "batch_inputs must be a positive integer"),
    ("openrouter:m", {"key": SECRET, "transport": Provider()}, "key applies only to the built-in transport"),
    ("x.gguf", {"provider": "p", "key": SECRET}, "provider, key apply only to hosted judges"),
    ("x.gguf", {"timeout": 5}, "timeout apply only to hosted judges"),
    ("x.gguf", {"devise": "cpu"}, "unknown native setting 'devise' \\(did you mean 'device'\\?\\)"),
    ("x.gguf", {"device": "tpu"}, "unknown device"),
])
def test_judge_from_refuses_invalid_combinations_before_transport(monkeypatch, source, settings, message):
    monkeypatch.setattr(hosted, "http", lambda *args: pytest.fail("transport created for an invalid configuration"))
    monkeypatch.setattr(api, "_resolve", lambda source: pytest.fail("assets resolved for an invalid configuration"))
    with pytest.raises(ConfigError, match=message) as raised:
        judge_from(source, **settings)
    assert SECRET not in str(raised.value)


def test_cli_options_distinguish_unspecified_native_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "judge_from", lambda source, **settings: seen.append((source, settings)) or FakeScorer(lambda *a: [.99, .01]))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["--json", "--judge", "openrouter:m", "--provider", "p", "check", "Claim."], input="case").exit_code == 0
    assert runner.invoke(cli.app, ["--json", "--judge", "x.gguf", "--no-repack", "--device", "cpu", "check", "Claim."], input="case").exit_code == 0
    assert seen == [("openrouter:m", {"provider": "p"}), ("x.gguf", {"provider": None, "device": "cpu", "repack": False})]


def test_cli_rejects_native_flags_for_hosted_judges(monkeypatch):
    monkeypatch.setattr(hosted, "http", lambda *args: pytest.fail("transport created"))
    result = CliRunner().invoke(cli.app, ["--judge", "openrouter:m", "--threads", "4", "check", "Claim."], input="case")
    assert isinstance(result.exception, ConfigError) and "threads apply only to local GGUF judges" in str(result.exception)


def test_hosted_cli_matches_python_without_native_packages(tmp_path, monkeypatch):
    for name in ("llama_cpp", "huggingface_hub"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setattr(hosted, "http", lambda *args: Provider())
    paths = {}
    for name in ("python", "cli"):
        (tmp_path / name).mkdir()
        paths[name] = definition(tmp_path / name)
        dataset(paths[name], 40)
    case = json.dumps({"message": "refund item"})
    with Runtime({"judge": judge_from("openrouter:m", provider="p")}, own_judges=True) as rt:
        python = {"check": rt.check("refund please", "Refund requested.").as_dict(False)}
        tools.lock(rt, paths["python"])
        python["run"] = rt.load(paths["python"])(json.loads(case)).as_dict(False)
        python["test"] = tools.run_test(rt, paths["python"], None, False)
    runner, options = CliRunner(), ["--json", "--judge", "openrouter:m", "--provider", "p"]
    outputs = [runner.invoke(cli.app, options + command, input=stdin) for command, stdin in (
        (["check", "Refund requested."], "refund please"), (["lock", str(paths["cli"])], ""),
        (["run", str(paths["cli"])], case), (["test", str(paths["cli"])], ""))]
    assert all(output.exception is None or output.exit_code in (0, 1, 2) for output in outputs), [o.exception for o in outputs]
    check, _, run, test = (json.loads(output.stdout) for output in outputs)
    locks = [files.read_json(paths[name].with_suffix(".lock.json")) for name in ("python", "cli")]
    assert (check, run) == (python["check"], python["run"]) and locks[0]["candidate_id"] == locks[1]["candidate_id"]
    assert {key: test[key] for key in ("facts", "candidate_id")} == json.loads(json.dumps({key: python["test"][key] for key in ("facts", "candidate_id")}))
    assert sys.modules["llama_cpp"] is None and sys.modules["huggingface_hub"] is None
    assert SECRET not in "".join(output.stdout for output in outputs) + json.dumps(locks)


def test_native_export_on_hosted_judge_is_unsupported(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": judge_from("openrouter:m", provider="p", transport=Provider())}, own_judges=True) as rt:
        with pytest.raises(ConfigError, match="native export requires a local tokenizer"):
            tools.export(rt, path, native=True)


def test_quickstart_and_readme_select_judges_through_judge_from(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    from examples.quickstart import judge
    assert judge({}) == Model(api.DEFAULT_JUDGE)
    routed = judge({"OPENROUTER_MODEL": "m", "OPENROUTER_PROVIDER": "p"})
    assert (routed.model, routed.provider) == ("m", "p")
    routed.close()


BLOCKED = """
import sys
sys.modules["llama_cpp"] = sys.modules["huggingface_hub"] = None
import semantic_switch as ss
import semantic_switch.cli
import semantic_switch.hosted
try:
    ss.Runtime({"judge": ss.Model(ss.DEFAULT_JUDGE, device="cpu")}).check("case", "Claim.")
except ss.ConfigError as error:
    print(error)
"""


def test_sdk_imports_and_local_judges_fail_before_download_without_native_packages():
    lines = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True, check=True).stdout.splitlines()
    assert lines == [f"cannot load model {DEFAULT_JUDGE}: local judges need the 'local' extra (llama_cpp, huggingface_hub not installed): "
                     "pip install 'semantic-switch[local]'"]


def test_package_metadata_isolates_native_dependencies():
    import tomllib
    from pathlib import Path
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    trainer = tomllib.loads((root / "packages/semantic-trainer/pyproject.toml").read_text())["project"]
    native = {"llama-cpp-python>=0.3.30", "huggingface_hub"}
    assert not native & set(project["dependencies"]) and set(project["optional-dependencies"]["local"]) == native
    assert project["optional-dependencies"]["hosted"] == ["httpx", "anyio>=4.4"]
    assert any(requirement.startswith("semantic-switch[local]") for requirement in trainer["optional-dependencies"]["lora"])


@pytest.mark.slow
def test_serial_native_accepted_outcomes_do_not_depend_on_cache_composition_or_capacity():
    from semantic_switch import Check, Guidance, Pick
    guidance = Guidance("A refund means money returned to the customer. Store credit is not a refund.")
    items = {"refund": Check("The customer asks for a refund.", guidance=guidance), "legal": Check("The customer threatens legal action."),
             "team": Pick("Which team should handle this?", {"billing": "Charges and refunds", "support": "Product problems"},
                          focus="mentions", guidance=guidance)}
    texts = ["I was charged twice, give my money back.", "The app crashes when I log in.", "My lawyer will hear about this refund.", "Send this to billing, not support."]

    def outcomes(rt, mode):
        if mode == "composed":
            return [{key: (r.value, r.status) for key, r in row.items()} for row in rt.ask_many(texts, items)]
        rows = []
        for text in texts:
            row = {}
            for key, item in items.items():
                rt.invalidate()
                r = rt.ask(text, {key: item})[key]
                row[key] = (r.value, r.status)
            rows.append(row)
        return rows

    results = []
    for settings in ({}, {"n_batch": 1024}):
        with Runtime({"judge": Model(QWEN3, adapter="qwen3", device="cpu", **settings)}) as rt:
            results += [outcomes(rt, "composed"), outcomes(rt, "composed"), outcomes(rt, "cold")]
    assert all(result == results[0] for result in results) and any(status == "ok" for row in results[0] for _, status in row.values())
