from __future__ import annotations

from pathlib import Path

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import candidate_digest, export_identity, read_export, snapshot_digest
from semantic_switch.tools.lock import build_lock

from .test_artifact_pipeline import dataset, definition
from .test_execution_identity import ExecutionScorer
from .test_v016 import RATTLE, catalogue, claim_oracle, fake_rt

HOT = {"id": "hot", "applies_when": ["The unit overheats."]}
REPORTS = [{"report": "it runs hot, floor is dry"}, {"report": "rattles, maybe hot, oil below"}, {"report": "all quiet, dry"}]


def test_selected_card_fact_resolves_like_normal_execution(tmp_path):
    rt = fake_rt(claim_oracle)
    caseless = rt.load(catalogue(tmp_path, [HOT, RATTLE]))
    cased = rt.load(catalogue(tmp_path, [HOT, RATTLE], [{"id": "wet", "when": "leak", "then": "mop"}]))
    assert "procs" not in cased(REPORTS[0]).facts
    for data, row in zip(REPORTS, cased.score_facts(REPORTS, facts=["procs"])):
        expected = caseless(data).facts["procs"]
        assert (row["procs"].value, row["procs"].settled, dict(row["procs"].states)) == \
            (expected.value, expected.settled, dict(expected.states))
    assert cased.score_facts(REPORTS[:1], facts=["procs"])[0]["procs"].value == ("hot",)


def test_selected_fact_absent_from_cases_is_judged(tmp_path):
    rt = fake_rt(claim_oracle)
    sw = rt.load(catalogue(tmp_path, [HOT], [{"id": "warm", "when": 'procs has "hot"', "then": "cool"}]))
    assert "leak" not in sw({"report": "it runs hot, oil below"}).facts
    leak = sw.score_facts([{"report": "it runs hot, oil below"}], sources=("judge",), facts=["leak"])[0]["leak"]
    assert (leak.value, leak.source, leak.status) == (True, "judge", "ok")


def test_export_reference_round_trips_through_reader(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": ExecutionScorer()}) as rt:
        exported = tools.export(rt, path)
    reference = Path(exported["path"])
    assert reference.is_file() and reference.name.endswith(".manifest.json")
    manifest, rows = read_export(reference)
    assert len(rows) == exported["rows"] == len(manifest["rows"]) and manifest['export_id'] == export_identity(manifest)


@pytest.mark.parametrize("trim", ["payload", "manifest"])
def test_export_rejects_row_count_mismatch(tmp_path, trim):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": ExecutionScorer()}) as rt:
        reference = Path(tools.export(rt, path)["path"])
    manifest, rows = read_export(reference)
    name = reference.name.removesuffix(".manifest.json")
    with files.open_published(reference) as (_, payloads):
        lines = payloads[name].readlines()
    files.publish(reference, {**{k: v for k, v in manifest.items() if k != "payloads"}, "rows": manifest["rows"][:-1] if trim == "manifest" else manifest["rows"]},
                  {name: lines[:-1] if trim == "payload" else lines})
    with pytest.raises(ConfigError, match="invalid export"):
        read_export(reference)


def test_candidate_id_binds_execution(tmp_path):
    path = definition(tmp_path)
    dataset(path, 40)
    with Runtime({"judge": ExecutionScorer()}) as rt:
        data, snap, _ = build_lock(rt, path)
        assert snapshot_digest(snap, rt) == snap.candidate_id == data["candidate_id"] == candidate_digest(data)
    changed = {**data, "execution": {"judge": {**data["execution"]["judge"], "n_ubatch": 512}}}
    assert candidate_digest(changed) != data["candidate_id"]
    with Runtime({"judge": ExecutionScorer(n_ubatch=512)}) as other:
        assert snapshot_digest(snap, other) == candidate_digest(changed)
