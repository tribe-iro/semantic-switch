import numpy as np
import pytest

from semantic_switch import ConfigError, Model, Runtime, api, backend, judge_from
from semantic_switch.artifacts import candidate_digest
from semantic_switch.compiler import digest
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS

from .test_execution_identity import ExecutionScorer, admit, locked
from .test_native_resources import QWEN3, fake_llama, request, scorer


def test_serial_grouping_decodes_each_state_and_question_alone(fake_llama):
    library = fake_llama()
    requests = [request((7, 8), (3, 4), (5, 6), (6,)), request((7, 8), (4, 3), (5,))]
    with scorer(inputs=2, grouping="batched") as batched:
        expected = batched.score(requests)
    library.decodes.clear()
    with scorer(inputs=2, n_gpu_layers=1) as serial:
        assert (serial.capacity()["grouping"], serial.capacity()["serial"], serial.fused) == ("serial", backend.SERIAL, False)
        block = serial.score(requests)
    assert np.array_equal(block.logp, expected.logp) and library.decodes == [0, 0, 1, 1, 0, 1]


def test_ordinary_judges_have_no_grouping_option(monkeypatch):
    monkeypatch.setattr(api, "_gpu", lambda: True)
    assert "grouping" not in Model.__dataclass_fields__ and Runtime({"judge": Model("model.gguf")}).batch_inputs() == 1
    with pytest.raises(ConfigError, match="unknown native setting 'grouping'"):
        judge_from("model.gguf", grouping="batched")


def test_model_validates_execution_settings():
    with pytest.raises(ConfigError, match="repack"):
        Model("model.gguf", repack="no")


@pytest.mark.parametrize("mode, changed", [({"shared": False}, "shared"), ({"grouping": "batched", "serial": None}, "grouping, serial")])
def test_admission_pins_realized_mode(tmp_path, mode, changed):
    data = locked(tmp_path)
    assert {key: data["execution"]["judge"][key] for key in api.MODE} == {"shared": True, "grouping": "serial", "serial": backend.SERIAL}
    with pytest.raises(ConfigError, match=rf"judge execution changed \({changed}\)"):
        admit(data, **mode)
    snap = admit(data, True, **mode)
    assert "judge" not in snap.policies and any("tuned judge policies were reset" in w for w in snap.warnings)


def test_fit_without_serial_contract_is_refused(tmp_path):
    data = locked(tmp_path)
    data = {key: value for key, value in data.items() if key not in ("id", "created")}
    data["execution"] = {"judge": {key: value for key, value in data["execution"]["judge"].items() if key != "serial"}}
    data["candidate_id"] = candidate_digest(data)
    data["id"] = digest(data)[:12]
    with pytest.raises(ConfigError, match=r"judge execution changed \(serial\)"):
        admit(data)


def test_conformance_key_excludes_realized_mode():
    keys, executions = set(), []
    for mode in ({}, {"shared": False}, {"grouping": "batched", "serial": None}):
        with Runtime({"judge": ExecutionScorer(**mode)}) as rt:
            executions.append(rt.execution())
            keys.add(rt.cache_key())
    assert len(keys) == 1 and len({str(execution) for execution in executions}) == 3


@pytest.mark.slow
def test_serial_scores_do_not_depend_on_co_requests():
    with Runtime({"judge": Model(QWEN3, adapter="qwen3", device="cpu")}) as rt:
        adapter = rt.judge().adapter
        first, second = (adapter.request(adapter.state(text), CONFORMANCE_QUESTIONS) for text in CONFORMANCE_PROBES[:2])
        alone = rt.score_native([first])
        rt.invalidate()
        together = rt.score_native([second, first])
        assert (rt.execution()["grouping"], rt.execution()["serial"]) == ("serial", backend.SERIAL)
    assert np.array_equal(alone.logp, together.logp[len(second.suffixes):])


@pytest.mark.parametrize("name, sequences, residents", [("right-sized", 6, range(3, 6)), ("reserved-37", 4 + 1 + backend.BRANCHES, range(34, 37))])
def test_diagnostic_layouts_reserve_one_output_row_per_sequence(fake_llama, monkeypatch, name, sequences, residents):
    from bench.backend_diagnostics import layout
    library = fake_llama()
    canonical = backend.capacity
    requests = [request(prefix, (3, 4), (5, 6), (6,)) for prefix in ((7,), (8,), (9,), (10,), (11,), (7,))]
    with layout(name):
        with scorer() as s:
            block = s.score(requests)
            assert (library.params.n_seq_max, library.params.n_outputs_max, s.ctx.outputs_per_decode) == (sequences, sequences, 1)
            assert s.layout.residents == residents and set(s._resident.values()) <= set(residents) and s.stats["prefix_evictions"]
            assert max(library.decodes) <= 1
        meta = {"general.architecture": "x", "x.block_count": 2, "x.attention.head_count_kv": 1, "x.embedding_length": 8,
                "x.attention.head_count": 1, "tokenizer.ggml.tokens": ["t"] * 100}
        monkeypatch.setattr(backend, "gguf_metadata", lambda path: meta)
        monkeypatch.setattr(Model, "path", lambda self: __file__)
        assert Model("fake.gguf")._host_bytes()["output"] == 100 * sequences * 4 * 2
    assert backend.capacity is canonical
    with scorer() as s:
        assert np.array_equal(s.score(requests).logp, block.logp)


def test_canonical_resident_slots_follow_the_capacity_owner():
    assert backend.capacity(None, 512, 8, 4, "batched").residents == range(41, 44)
    assert backend.capacity(None, 512, 1, 2).residents == range(3, 4)


def test_phase_profile_attributes_decode_and_reduction(fake_llama):
    from bench.backend_diagnostics import phases, reducer_cost, split
    fake_llama()
    reduce = backend.reduce_row
    with scorer() as s:
        with phases(s) as events:
            s.score([request((7, 8), (3, 4), (5, 6), (6,))])
        assert backend.reduce_row is reduce and len(events["rows"]) == 2 and events["rows"][0][1] == ((1,), (2,))
        found = split(events)
        assert (found["outputs"], found["submissions"]) == (2, 4) and min(found.values()) >= 0
    cost = reducer_cost(*events["rows"][0], repeats=3)
    assert cost["slots"] == [1, 1] and cost["ms"]["n"] == 3 and cost["peak_bytes"] < cost["peak_bytes_without_scratch"]


def test_layout_agreement_separates_answer_acceptance_and_latency_populations():
    from bench.backend_diagnostics import agreement, latency, summarize
    call = lambda p, status, value, ms: {"results": {"q": {"value": value, "status": status, "p": [p], "mass": .99}}, "questions": ["q"],
                                         "ms": ms, "work": {"submissions": 3}}
    def run(name, p, status, value, ms):
        execution = {"n_seq_max": 6 if name == "right-sized" else 37, "n_outputs_max": 6 if name == "right-sized" else 37}
        return {"layout": name, "execution": execution, "buffers": {}, "estimated_host_bytes": {}, "peak_rss_bytes": 1,
                "cases": {"0": {"cold": call(p, status, value, ms), "one:q": call(p, status, value, ms / 2)}}, "histories": {}}
    a, b = run("right-sized", .91, "ok", True, 10.), run("reserved-37", .89, "uncertain", None, 30.)
    found = agreement(a, b)
    assert (found["answers"], found["identical"], found["answer_changes"], found["acceptance_changes"], found["accepted_changes"]) == (2, 0, 0, 2, 2)
    assert found["max_dp"] == pytest.approx(.02) and found["flips"][0]["a"]["status"] == "ok"
    assert agreement(a, a)["identical"] == 2 and latency([a])["one"]["ms"]["p50"] == 5.
    report = summarize([a, b, a, b])
    assert report["within"]["right-sized"]["accepted_changes"] == 0 and report["across"]["run 1"]["accepted_changes"] == 2
    assert report["latency"]["reserved-37"]["cold"]["ms"]["p50"] == 30. and report["identity_changed"] == ["n_outputs_max", "n_seq_max"]
