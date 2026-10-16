from __future__ import annotations

from dataclasses import replace

from semantic_switch import files
from bench import evidence, procedures, snapshots
from semantic_testkit.scoring import FakeScorer


def published(tmp_path):
    pack = evidence.load_pack(procedures.PACK)
    rows = procedures.scenarios(0, 1)
    ref = procedures.standard()
    reviewed = evidence.review(pack, procedures.frames(rows), (), ref, authored=procedures.authored())
    assignment = evidence.assign(reviewed.frames, {"fit": 8, "model_select": 6}, seed=0,
                                 counted=evidence.situation_frames(reviewed.frames, reviewed.golds, procedures.TRAINED))
    evidence.publish(tmp_path / "pack", pack, reviewed, ref, scenarios=rows, definition=files.read_yaml(procedures.DEFINITION), name=procedures.DEFINITION.name,
                     condition=procedures.TRAINED, assignment=assignment, producer="bench.procedures.reference",
                     producer_digest=evidence.code_digest(procedures, evidence), stratum=procedures.stratum)
    return pack


def scope(tmp_path, pack, *, data=None, criteria=None, frames=None):
    data = data or evidence.read(tmp_path / "pack")
    _, loaded, meta = evidence.load_frames(tmp_path / "pack")
    used = frames or [f for f in loaded if meta[f.id]["role"] == "check" and f.condition == procedures.TRAINED]
    golds = {f.id: procedures.reference(f) for f in used}
    criteria = criteria or {c: evidence.criterion(procedures.definition(c)) for c in procedures.CONDITIONS}
    return evidence.scoped_review(data, pack, used, golds, snapshots.scope_targets(snapshots.DOMAINS["zt411"]), criteria,
                                  authored=procedures.authored()), used, data


def test_stale_criteria_and_changed_excerpt_pins_are_not_current_support(tmp_path):
    pack = published(tmp_path)
    base, used, _ = scope(tmp_path, pack)
    assert not base["stale_criteria"] and not base["stale_reference"] and base["scope"]["frames"] == len(used)
    changed, _, _ = scope(tmp_path, pack, criteria={procedures.TRAINED: "0" * 64})
    assert changed["stale_criteria"] == [procedures.TRAINED] and "criteria" in changed["status"] and changed["status"] != "reviewed"
    cited = base["scope"]["rules"]
    source = next(pack.rule(r).source for r in cited if pack.rule(r).source != evidence.CONVENTION)
    repinned = replace(pack, sources=tuple(replace(s, sha256="f" * 64) if s.id == source else s for s in pack.sources))
    stale, _, _ = scope(tmp_path, repinned)
    assert any(source in reason for reason in stale["sources"]["stale"]) and stale["status"] != "reviewed"


def test_unused_sources_and_unrelated_frames_do_not_block(tmp_path):
    pack = published(tmp_path)
    base, used, _ = scope(tmp_path, pack)
    unused = [s for s in pack.sources if s.id not in base["sources"]["states"]]
    if unused:
        moved = replace(pack, sources=tuple(replace(s, sha256="e" * 64) if s.id == unused[0].id else s for s in pack.sources))
        again, _, _ = scope(tmp_path, moved)
        assert again["sources"]["stale"] == base["sources"]["stale"] and unused[0].id in again["sources"]["unused"]
    one, _, _ = scope(tmp_path, pack, frames=used[:1])
    assert one["scope"]["frames"] == 1 and one["frames"]["required"] <= 1


def test_partial_target_review_does_not_complete_a_required_frame(tmp_path):
    pack = published(tmp_path)
    base, used, data = scope(tmp_path, pack)
    required = {row["frame"] for row in data["review_worksheet"]} & {f.id for f in used}
    frame = sorted(required)[0]
    binding = next(row["binding"] for row in data["reference"] if row["frame"] == frame)
    record = {"frame": frame, "kind": "label", "target": ["actions", "load_ribbon"], "author": base["author"], "reviewer": "r", "actor": {"kind": "human"},
              "self_review": False, "reviewer_distinct": True, "binding": binding, "current": True, "verdict": "confirm", "withdrawn": None}
    partial, _, _ = scope(tmp_path, pack, data={**data, "reviews": [*data["reviews"], record]})
    assert frame in partial["frames"]["partial"] and partial["targets"]["supported"] == 1 and partial["status"] != "reviewed"
    confirmed, _, _ = scope(tmp_path, pack, data={**data, "reviews": [*data["reviews"], {**record, "kind": "evidence", "target": []}]})
    assert frame not in confirmed["frames"]["partial"] and confirmed["frames"]["confirmed"] == 1 and confirmed["reviewers"] == ["r"]
    stale, _, _ = scope(tmp_path, pack, data={**data, "reviews": [*data["reviews"], {**record, "kind": "evidence", "target": []}]},
                        criteria={procedures.TRAINED: "0" * 64})
    assert stale["frames"]["confirmed"] == 0


def test_exposed_families_are_not_current_independent_support():
    even = {"forbidden_action_situations": 0, "failures": 0}
    decision = snapshots.adoption({"support": "reported", "interval": [.5, .9]}, .02, even, even, exposure={"state": "consulted", "independent": False},
                                  lineage={"state": "declared_disjoint"}, review="reviewed")
    assert decision["verdict"] == "reused_evidence"


def test_corrected_reference_rebuilds_report_from_stored_rows_without_inference(tmp_path, monkeypatch):
    published(tmp_path)
    staged = evidence.stage(tmp_path / "pack", tmp_path / "stage")["evidence"]
    scorer, output = FakeScorer(), tmp_path / "out" / "eval.json"
    first = snapshots.evaluate("zt411", tmp_path / "pack", staged, {"base": scorer}, output, baseline="base", margin=.02, kinds=("workflow",))
    calls = len(scorer.requests)
    original = procedures.reference

    def corrected(frame):
        gold = original(frame)
        target = "actions/set_method_tt"
        decision = gold.decisions[target]
        return replace(gold, decisions={**gold.decisions, target: replace(decision, value=decision.value is not True)})
    monkeypatch.setattr(procedures, "reference", corrected)
    again = snapshots.evaluate("zt411", tmp_path / "pack", staged, {"base": scorer}, output, baseline="base", margin=.02, kinds=("workflow",))
    assert len(scorer.requests) == calls and again["plan"]["id"] == first["plan"]["id"] and again["exposure"]["state"] == "repeat"
    assert again["review"]["stale_reference"] and again["review"]["status"] != "reviewed"
    key = f"base|workflow|{procedures.TRAINED}"
    before = {row["key"]: row["gold"] for row in first["results"][key]["rows"]}
    after = {row["key"]: row["gold"] for row in again["results"][key]["rows"]}
    assert before.keys() == after.keys() and before != after
    assert [row["ms"] for row in first["results"][key]["rows"]] == [row["ms"] for row in again["results"][key]["rows"]]
