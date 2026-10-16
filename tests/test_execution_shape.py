import numpy as np
import pytest

from semantic_switch import ConfigError, Model, Runtime, api, backend, judge
from semantic_switch.artifacts import candidate_digest
from semantic_switch.compiler import digest
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
from semantic_switch.training import Engine

from .test_execution_identity import ExecutionScorer, admit, locked
from .test_native_resources import QWEN3, fake_llama, request, scorer


def test_serial_scorer_decodes_each_state_and_question_alone(fake_llama):
    library = fake_llama()
    requests = [request((7, 8), (3, 4), (5, 6), (6,)), request((7, 8), (4, 3), (5,))]
    with scorer(inputs=2) as cpu:
        expected = cpu.score(requests)
    library.decodes.clear()
    with scorer(inputs=2, n_gpu_layers=1) as offload:
        assert offload.capacity()["scheduling"] == backend.SCHEDULING and "grouping" not in offload.capacity()
        block = offload.score(requests)
    assert np.array_equal(block.logp, expected.logp) and library.decodes == [0, 0, 1, 1, 0, 1]


def test_ordinary_judges_have_no_grouping_option(monkeypatch):
    monkeypatch.setattr(api, "_gpu", lambda: True)
    assert "grouping" not in Model.__dataclass_fields__ and Runtime(Model("model.gguf")).batch_inputs() == 1
    with pytest.raises(ConfigError, match="grouping does not apply to gguf judges"):
        judge("model.gguf", grouping="batched")


def test_model_validates_execution_settings():
    with pytest.raises(ConfigError, match="repack"):
        Model("model.gguf", repack="no")


def test_admission_pins_realized_mode(tmp_path):
    data = locked(tmp_path)
    assert {key: data["execution"]["judge"][key] for key in api.MODE} == {"shared": True}
    with pytest.raises(ConfigError, match=r"judge execution changed \(shared\)"):
        admit(data, shared=False)
    snap = admit(data, True, shared=False)
    assert "judge" not in snap.policies and any("tuned judge policies were reset" in w for w in snap.warnings)


def test_fit_without_scheduling_contract_is_refused(tmp_path):
    data = locked(tmp_path)
    data = {key: value for key, value in data.items() if key not in ("id", "created")}
    data["execution"] = {"judge": {key: value for key, value in data["execution"]["judge"].items() if key != "scheduling"}}
    data["candidate_id"] = candidate_digest(data)
    data["id"] = digest(data)[:12]
    with pytest.raises(ConfigError, match=r"judge execution changed \(scheduling\)"):
        admit(data)


def test_conformance_key_excludes_realized_mode():
    keys, executions = set(), []
    for mode in ({}, {"shared": False}):
        with Runtime(ExecutionScorer(**mode)) as rt:
            executions.append(rt.execution())
            keys.add(rt.cache_key())
    assert len(keys) == 1 and len({str(execution) for execution in executions}) == 2


@pytest.mark.slow
def test_serial_scores_do_not_depend_on_co_requests():
    with Engine(Model(QWEN3, adapter="qwen3", device="cpu")) as rt:
        adapter = rt.judge().adapter
        first, second = (adapter.request(adapter.state(text), CONFORMANCE_QUESTIONS) for text in CONFORMANCE_PROBES[:2])
        alone = rt.score_native([first])
        rt.invalidate()
        together = rt.score_native([second, first])
        assert rt.execution()["scheduling"] == backend.SCHEDULING
    assert np.array_equal(alone.logp, together.logp[len(second.suffixes):])


def test_resident_prefix_slots_follow_the_capacity_owner(fake_llama):
    library = fake_llama()
    requests = [request(prefix, (3, 4), (5, 6), (6,)) for prefix in ((7,), (8,), (9,), (10,), (11,), (7,))]
    with scorer() as s:
        block = s.score(requests)
        assert (library.params.n_seq_max, library.params.n_outputs_max, s.ctx.outputs_per_decode) == (6, 6, 1)
        assert s.layout.residents == range(3, 6) and set(s._resident.values()) <= set(s.layout.residents) and s.stats["prefix_evictions"]
        assert max(library.decodes) <= 1
    with scorer() as s:
        assert np.array_equal(s.score(requests).logp, block.logp)
    assert backend.capacity(None, 512, 4).residents == range(3, 6) and backend.capacity(None, 512, 2).residents == range(3, 4)


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
    assert cost["slots"] == [1, 1] and cost["latency"]["samples"] == 3 and cost["peak_bytes"] < cost["peak_bytes_without_scratch"]
