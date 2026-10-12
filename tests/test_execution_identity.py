from __future__ import annotations

import pytest

from semantic_switch import ConfigError, Runtime, api, files
from semantic_switch.api import execution_identity
from semantic_switch.backend import SCHEDULING
from semantic_switch.compiler import digest
from semantic_switch.snapshot import from_lock
from semantic_switch.tools.lock import build_lock

from .conftest import FakeScorer
from .test_artifact_pipeline import dataset, definition
from .test_native_resources import fake_llama, scorer

REALIZED = {"device": "cpu", "tensor_types": ["f32", "q8_0"], "flash_attn": "enabled", "type_k": "f16", "type_v": "f16",
            "use_extra_bufts": True, "repacked": [], "n_ubatch": 64, "n_ctx": 8192, "n_batch": 64, "n_seq_max": 6,
            "n_outputs_max": 1, "swa_full": True, "kv_unified": True, "prefix_slots": 4, "scheduling": SCHEDULING,
            "buffers": {"CPU model": 1}, "shared": True, "grouping": "serial", "serial": 1, "fusion": False}


class ExecutionScorer(FakeScorer):
    def __init__(self, **realized):
        super().__init__()
        self.realized = {**REALIZED, **realized}

    def capacity(self):
        return dict(self.realized)


def locked(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": ExecutionScorer()}) as rt:
        data, _, _ = build_lock(rt, path)
    files.publish(tmp_path / "lock.json", data)
    data = files.read_json(tmp_path / "lock.json")
    assert any("judge" in fact["policies"] for fact in data["facts"].values())
    return data


def admit(data, allow=False, **realized):
    with Runtime({"judge": ExecutionScorer(**realized)}) as rt:
        return from_lock(data, rt.identity, allow, "lock")


def test_execution_identity_records_capacities_but_not_threads_or_buffers():
    same = execution_identity({**REALIZED, "threads": 8, "buffers": {}})
    assert same == execution_identity(REALIZED) == {**{key: REALIZED[key] for key in api.EXECUTION}, "build": api._build()}
    for key, value in (("n_ubatch", 512), ("n_ctx", 16384), ("n_batch", 2048), ("n_seq_max", 40), ("n_outputs_max", 40),
                       ("swa_full", False), ("kv_unified", False), ("prefix_slots", 1), ("scheduling", "other@1")):
        assert execution_identity({**REALIZED, key: value}) != same
    assert execution_identity({}) is None


def test_lock_records_execution_identity(tmp_path):
    data = locked(tmp_path)
    assert data["schema"] == 17 and data["execution"] == {"judge": execution_identity(REALIZED)}


def test_admission_rejects_material_execution_change(tmp_path):
    data = locked(tmp_path)
    with pytest.raises(ConfigError, match=r"judge execution changed \(device, n_ubatch\)"):
        admit(data, device="gpu", n_ubatch=512)
    for key, value in (("n_batch", 2048), ("n_seq_max", 40), ("swa_full", False), ("kv_unified", False), ("prefix_slots", 1)):
        with pytest.raises(ConfigError, match=rf"judge execution changed \({key}\)"):
            admit(data, **{key: value})
    assert admit(data, threads=8, buffers={}).policies.get("judge")


def test_admission_resets_policies_on_execution_change_when_allowed(tmp_path):
    data = locked(tmp_path)
    snap = admit(data, True, flash_attn="disabled")
    assert "judge" not in snap.policies and any("judge execution changed (flash_attn)" in w for w in snap.warnings)


def test_admission_warns_on_build_only_change(tmp_path, monkeypatch):
    data = locked(tmp_path)
    monkeypatch.setattr(api, "_build", lambda: "other build")
    snap = admit(data)
    assert snap.policies.get("judge") and any("native build changed" in w and "calibration has not been reverified" in w for w in snap.warnings)


def test_lock_without_execution_identity_is_rejected(tmp_path):
    data = {key: value for key, value in locked(tmp_path).items() if key not in ("id", "created", "execution")}
    data["id"] = digest(data)[:12]
    with pytest.raises(ConfigError, match="missing judge execution identity"):
        admit(data)


def test_context_sets_attention_and_kv_types_explicitly(fake_llama):
    library = fake_llama(model_log=["llama_model_loader: - type  f32:  113 tensors\n",
                                    "llama_model_loader: - type q8_0:  197 tensors\n"])
    with scorer() as native:
        realized = native.capacity()
    assert (library.params.flash_attn_type, library.params.type_k, library.params.type_v) == \
        (library.LLAMA_FLASH_ATTN_TYPE_ENABLED, library.GGML_TYPE_F16, library.GGML_TYPE_F16)
    assert execution_identity(realized) == {"device": "cpu", "tensor_types": ["f32", "q8_0"], "flash_attn": "enabled",
                                            "type_k": "f16", "type_v": "f16", "use_extra_bufts": True, "repacked": [],
                                            "n_ctx": realized["n_ctx"], "n_batch": 64, "n_ubatch": 64, "n_seq_max": 6, "n_outputs_max": 6,
                                            "swa_full": True, "kv_unified": True, "prefix_slots": 4, "scheduling": SCHEDULING,
                                            "shared": True, "grouping": "serial", "serial": 1, "build": api._build()}
