from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from semantic_switch import ConfigError, files
from bench import evidence, procedures, support
from bench.evidence import Review


def pinned(tmp_path, *, version="1.0", **changes):
    data = {**files.read_json(procedures.PACK), "version": version, **changes}
    (tmp_path / "doc.txt").write_text("printer guide pages")
    for source in data["sources"]:
        if source["use"] == evidence.CITED:
            excerpt = tmp_path / f"{source['id']}.txt"
            excerpt.write_text(f"excerpt {source['id']}")
            source.update(document={**source["document"], "path": "doc.txt", "sha256": files.sha256_file(tmp_path / "doc.txt")},
                          extraction={"method": "manual", "tool": "reviewer-copied excerpt"}, excerpt=excerpt.name, sha256=files.sha256_file(excerpt))
    files.publish(tmp_path / "pack.json", data)
    return tmp_path / "pack.json"


def used(pack):
    return {rule.id for rule in pack.rules}


def test_source_contract_pins_documents_excerpts_editions_and_declared_unused_sources(tmp_path):
    manifest = pinned(tmp_path)
    pack = evidence.load_pack(manifest)
    terms = evidence.contract(pack, used(pack), authored=procedures.authored())
    assert terms["complete"] and set(terms["sources"]) == {"zt-user-guide", "zt-calibration"} and terms["sources"]["zt-calibration"] == "pinned"
    assert terms["unused"]["zt-ribbon-mode"].startswith("unused:")
    (tmp_path / "zt-calibration.txt").write_text("edited excerpt")
    with pytest.raises(ConfigError, match="excerpt .* differs from its pinned digest"):
        evidence.contract(pack, used(pack), authored=procedures.authored())
    (tmp_path / "zt-calibration.txt").write_text("excerpt zt-calibration")
    (tmp_path / "doc.txt").write_text("a new edition")
    with pytest.raises(ConfigError, match="document .* differs from its pinned digest"):
        evidence.contract(pack, used(pack), authored=procedures.authored())
    (tmp_path / "doc.txt").unlink()
    absent = evidence.contract(pack, used(pack), authored=procedures.authored())
    assert not absent["complete"] and absent["sources"]["zt-calibration"] == "absent" and any("fetch" in need for need in absent["missing"])
    with pytest.raises(ConfigError, match="changed since it was pinned"):
        evidence.contract(pack, used(pack), authored={"catalogue": "0" * 64})
    with pytest.raises(ConfigError, match="no used rule or slice cites them"):
        evidence.contract(pack, {"ZT-DEFAULT"}, authored=procedures.authored())
    rules = [{**r, "source": "zt-ribbon-mode"} if r["id"] == "ZT-RIBBON" else r for r in files.read_json(manifest)["rules"]]
    files.publish(manifest, {**files.read_json(manifest), "rules": rules})
    with pytest.raises(ConfigError, match="declared unused"):
        evidence.contract(evidence.load_pack(manifest), used(pack), authored=procedures.authored())


def test_skeleton_versions_unpinned_fields_and_applicability(tmp_path):
    manifest = pinned(tmp_path, version="0.1-skeleton")
    pack = evidence.load_pack(manifest)
    assert any("bump the pack version" in need for need in evidence.contract(pack, used(pack), authored=procedures.authored())["missing"])
    data = files.read_json(manifest)
    data["sources"][1]["edition"] = "unpinned: record it"
    data.pop("applicability")
    files.publish(manifest, data)
    terms = evidence.contract(evidence.load_pack(manifest), used(pack), authored=procedures.authored())
    assert {"source zt-calibration: edition", "applicability: record the documented model/option/firmware applicability"} <= set(terms["missing"])
    assert not any("bump" in need for need in terms["missing"])
    files.publish(manifest, {**data, "version": "1.0"})
    with pytest.raises(ConfigError, match="non-skeleton pack needs"):
        evidence.contract(evidence.load_pack(manifest), used(pack), authored=procedures.authored())


def test_unused_abcd_reference_does_not_block_the_authored_support_pack():
    pack = evidence.load_pack(support.PACK)
    terms = evidence.contract(pack, used(pack), authored=support.authored())
    assert terms["sources"] == {} and terms["unused"]["abcd"].startswith("unused:") and terms["complete"]
    with pytest.raises(ConfigError, match="declared unused"):
        evidence.contract(pack, used(pack), slices=("abcd",), authored=support.authored())


def build(tmp_path, reviews=(), reference=procedures.reference):
    manifest = pinned(tmp_path / "source") if not (tmp_path / "source" / "pack.json").exists() else tmp_path / "source" / "pack.json"
    pack = evidence.load_pack(manifest)
    rows = procedures.scenarios(0, 1)
    built, applied = evidence.apply_reviews(procedures.frames(rows), reviews)
    golds = procedures.evaluate(built)
    assignment = evidence.assign(built, {"fit": 8, "model_select": 6}, seed=0, counted=evidence.situation_frames(built, golds, procedures.TRAINED))
    out = tmp_path / f"pack-{len(list(tmp_path.glob('pack-*')))}"
    manifest = evidence.publish(out, pack, scenarios=rows, frames=built, golds=golds, definition=files.read_yaml(procedures.DEFINITION),
                                name=procedures.DEFINITION.name, condition=procedures.TRAINED, assignment=assignment, reviews=reviews, applied=applied,
                                producer="bench.procedures.reference", producer_digest=evidence.code_digest(procedures, evidence), reference=reference,
                                fields=procedures.FIELDS, stratum=procedures.stratum, authored=procedures.authored())
    return manifest, evidence.read(out), built, golds, assignment


def confirmations(data, reviewer="rev"):
    return [Review(row["frame"], reviewer, "evidence", row["content_sha256"], "confirm", binding=row["binding"]) for row in data["review_worksheet"]]


def test_sampled_human_completion_plus_mechanical_success_is_reviewed(tmp_path):
    (tmp_path / "source").mkdir()
    manifest, data, built, golds, assignment = build(tmp_path)
    review = manifest["review"]
    assert manifest["mechanical"]["failed"] == 0 and manifest["mechanical"]["frames"] == len(built) and review["sources"]["complete"]
    assert review["status"].startswith("unreviewed: human:") and manifest["review_plan"]["human_frames"] == len(data["review_worksheet"])
    required = {row["frame"] for row in data["review_worksheet"]}
    assert len(required) < len(built) and manifest["review_plan"]["sampled"] == round(manifest["review_plan"]["pool"] * evidence.REVIEW_FRACTION)
    golds_ = {f.id: golds[f.id] for f in built}
    assert all(any(reason in ("branch", "stratum", "unresolved", "ambiguous_or_contradictory", "sample") for reason in row["reasons"])
               for row in data["review_worksheet"])
    assert all(row["frame"] in required for row in data["review_worksheet"] if not golds_[row["frame"]].complete)
    reviews = confirmations(data)
    manifest, again, *_ = build(tmp_path, reviews)
    assert manifest["review"]["status"] == "reviewed" and manifest["review"]["disclosure"].startswith("sampled human review:")
    assert manifest["review"]["reviewer_distinct"] and manifest["review"]["frames"]["confirmed"]["frames"] == len(required)
    for name in ("annotations.jsonl", "situations.jsonl", "scenarios.jsonl", "definition.yaml"):
        assert manifest["payloads"][name]["sha256"] == evidence.read(tmp_path / "pack-0")["manifest"]["payloads"][name]["sha256"]
    strip = lambda rows: [{k: v for k, v in row.items() if k != "label_sources"} for row in rows]
    assert strip(again["labels"]) == strip(data["labels"]) and again["labels"] != data["labels"]
    partial, *_ = build(tmp_path, reviews[1:])
    assert partial["review"]["status"] == "unreviewed: human: 1 of %d required frames lack a current confirmation" % len(required)
    stale, *_ = build(tmp_path, [replace(reviews[0], binding="0" * 64), *reviews[1:]])
    assert stale["review"]["status"].startswith("unreviewed: human: 1 of")
    own, *_ = build(tmp_path, confirmations(data, reviewer="unassigned"))
    assert own["review"]["status"] == "reviewed" and own["review"]["self_review"] and "not independent expert validation" in own["review"]["disclosure"]


def test_replacement_needs_confirmation_dispute_blocks_and_skip_removes_before_assignment(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, built, golds, _ = build(tmp_path)
    reviews = confirmations(data)
    target = next(f for f in built if f.id == data["review_worksheet"][0]["frame"])
    corrected = tuple(a for a in target.assertions if a.id != target.assertions[-1].id)
    replaced = Review(target.id, "rev", "evidence", target.content, "replace", replacement=corrected)
    manifest, after, *_ = build(tmp_path, [*[r for r in reviews if r.frame != target.id], replaced])
    assert manifest["review_application"]["replaced"] == [target.id] and target.id in manifest["review"]["unconfirmed_replacements"]
    assert manifest["review"]["status"].startswith("unreviewed")
    fixed = next(row for row in after["reference"] if row["frame"] == target.id)
    assert fixed["binding"] != data["review_worksheet"][0]["binding"]
    confirm = Review(target.id, "rev", "evidence", replace(target, assertions=corrected).content, "confirm", binding=fixed["binding"])
    rows = {row["frame"]: row for row in after["review_worksheet"]}
    rest = [Review(f, "rev", "evidence", row["content_sha256"], "confirm", binding=row["binding"]) for f, row in rows.items() if f != target.id]
    manifest, *_ = build(tmp_path, [replaced, confirm, *rest])
    assert manifest["review"]["status"] == "reviewed" and not manifest["review"]["unconfirmed_replacements"]
    disputed, *_ = build(tmp_path, [*reviews, Review(target.id, "other", "evidence", target.content, "skip")])
    assert disputed["review"]["disputes"] == [target.id] and disputed["review"]["status"].startswith("unreviewed")
    other = next(f for f in built if f.id not in rows and f.condition != procedures.TRAINED)
    skipped, kept, *_ = build(tmp_path, [Review(other.id, "rev", "evidence", other.content, "skip")])
    assert skipped["review_application"]["skipped"] == [other.id] and other.id not in {row["frame"] for row in kept["reference"]}
    with pytest.raises(ConfigError, match="names the frame binding"):
        evidence.apply_reviews(built, [Review(target.id, "rev", "evidence", target.content, "confirm")])


def test_mechanical_tier_rejects_a_reference_that_does_not_recompute(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, *_ = build(tmp_path)
    flipped = lambda frame: replace(procedures.reference(frame), prohibited=("resume_printing",)) if frame.id.endswith(":structured:en") else procedures.reference(frame)
    manifest, *_ = build(tmp_path, confirmations(data), reference=flipped)
    assert manifest["mechanical"]["failed"] and all(any(m.startswith("reference:") for m in ms) for ms in manifest["mechanical"]["failures"].values())
    assert manifest["review"]["status"].startswith("unreviewed: mechanical:")


def test_frame_situation_family_and_target_units_are_counted_independently(tmp_path):
    (tmp_path / "source").mkdir()
    manifest, data, built, golds, assignment = build(tmp_path)
    for role, counts in manifest["assignment"]["counts"].items():
        frames = [f for f in built if assignment["families"][assignment["components"][f.id]] == role]
        situations = [f for f in frames if f.condition == procedures.TRAINED and all(d.value is not None for d in golds[f.id].supervision.values())]
        assert counts == {"frames": len(frames), "situations": len(situations), "families": len({assignment["components"][f.id] for f in frames}),
                          "targets": len(frames) * len(procedures.SCOPE)}
    assert manifest["review"]["frames"]["frames"] == len(built) and manifest["review"]["schema"] == evidence.REVIEW_SCHEMA
    assert "situations" not in {k for k in manifest["review"] if k != "units"} and manifest["assignment"]["units"] == evidence.UNITS


def test_support_composition_traces_resolve_and_cover_decision_branches():
    rows = support.scenarios(0, 60)
    built = support.frames(rows, ("message",), ("en",))
    golds = support.build(built)
    composed = {f.id: support.response(f, golds[f.id]) for f in built}
    for frame in built:
        decision, ids = composed[frame.id], {a.id for a in frame.assertions}
        assert set(decision.evidence) <= ids and all(evidence.lookup(frame.input, p) is not None for p in decision.paths)
        assert decision.rules[0] == "S-RESPONSE" and decision.branch.split(":")[0] == decision.value and decision.scope
    by_variant = {}
    for frame in built:
        by_variant.setdefault(frame.id.split(":")[1], []).append(composed[frame.id])
    for variant, rule in (("negated_request", "S-NEGATED"), ("quoted_request", "S-QUOTED"), ("latest_request", "S-LATEST"),
                          ("other_order", "S-ENTITY"), ("ambiguous", "S-AMBIGUOUS")):
        assert any(rule in decision.rules for decision in by_variant.get(variant, ())), variant
    assert any("S-RESOLVED" in d.rules for d in by_variant["resolved_issue"] if any(t.startswith("issues/") for t in d.scope))
    branches = {d.branch for d in composed.values()}
    assert {"clarify", "no_action", "initiate_refund"} <= branches and any(b.startswith("refund_ineligible:") for b in branches)
    assert all(set(d.paths) <= set(support.ACCOUNT) for d in composed.values()) and any("amount" in d.paths for d in composed.values() if d.value == "refund_ineligible")
    plan = evidence.review_plan(built, golds, {f.id: "check" for f in built}, seed=0, composition=composed, stratum=support.stratum)
    chosen = {composed[f].branch for f in plan["human"]}
    assert chosen == branches and plan["compositions"] >= len(branches)
    assert json.dumps(asdict(composed[built[0].id]))
