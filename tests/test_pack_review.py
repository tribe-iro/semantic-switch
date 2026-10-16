from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
import json

import pytest

from semantic_switch import Actor, ConfigError, files
from bench import evidence, procedures, support

Review = partial(evidence.Review, actor=Actor("agent", "fixture-model", "fixture-session"), note="fixture decision")


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


def build(tmp_path, reviews=(), reference=None):
    manifest = pinned(tmp_path / "source") if not (tmp_path / "source" / "pack.json").exists() else tmp_path / "source" / "pack.json"
    pack = evidence.load_pack(manifest)
    rows = procedures.scenarios(0, 1)
    ref = procedures.standard()
    reviewed = evidence.review(pack, procedures.frames(rows), reviews, ref, authored=procedures.authored())
    assignment = evidence.assign(reviewed.frames, {"fit": 8, "model_select": 6}, seed=0,
                                 counted=evidence.situation_frames(reviewed.frames, reviewed.golds, procedures.TRAINED))
    out = tmp_path / f"pack-{len(list(tmp_path.glob('pack-*')))}"
    manifest = evidence.publish(out, pack, reviewed, reference or ref, scenarios=rows, definition=files.read_yaml(procedures.DEFINITION),
                                name=procedures.DEFINITION.name, condition=procedures.TRAINED, assignment=assignment,
                                producer="bench.procedures.reference", producer_digest=evidence.code_digest(procedures, evidence), stratum=procedures.stratum)
    return manifest, evidence.read(out), reviewed.frames, reviewed.golds, assignment


def confirmations(data, reviewer="rev", kind="evidence", target=()):
    return [Review(row["frame"], reviewer, kind, "confirm", row["binding"], target) for row in data["review_worksheet"]]


def bindings(data):
    return {row["frame"]: row["binding"] for row in data["reference"]}


def test_sampled_review_completion_plus_mechanical_success_is_reviewed(tmp_path):
    (tmp_path / "source").mkdir()
    manifest, data, built, golds, assignment = build(tmp_path)
    review = manifest["review"]
    assert manifest["mechanical"]["failed"] == 0 and manifest["mechanical"]["frames"] == len(built) and review["sources"]["complete"]
    assert review["status"].startswith("unreviewed: review:") and manifest["review_plan"]["review_frames"] == len(data["review_worksheet"])
    required = {row["frame"] for row in data["review_worksheet"]}
    assert len(required) < len(built) and manifest["review_plan"]["sampled"] == round(manifest["review_plan"]["pool"] * evidence.REVIEW_FRACTION)
    golds_ = {f.id: golds[f.id] for f in built}
    assert all(any(reason in ("branch", "stratum", "unresolved", "ambiguous_or_contradictory", "sample") for reason in row["reasons"])
               for row in data["review_worksheet"])
    assert all(row["frame"] in required for row in data["review_worksheet"] if not golds_[row["frame"]].complete)
    reviews = confirmations(data)
    manifest, again, *_ = build(tmp_path, reviews)
    assert manifest["review"]["status"] == "reviewed" and manifest["review"]["disclosure"].startswith("sampled review:")
    assert manifest["review"]["reviewer_distinct"] and manifest["review"]["frames"]["confirmed"]["frames"] == len(required)
    for name in ("annotations.jsonl", "situations.jsonl", "scenarios.jsonl", "definition.yaml"):
        assert manifest["payloads"][name]["sha256"] == evidence.read(tmp_path / "pack-0")["manifest"]["payloads"][name]["sha256"]
    strip = lambda rows: [{k: v for k, v in row.items() if k != "label_sources"} for row in rows]
    assert strip(again["labels"]) == strip(data["labels"]) and again["labels"] != data["labels"]
    partial, *_ = build(tmp_path, reviews[1:])
    assert partial["review"]["status"] == "unreviewed: review: 1 of %d required frames lack a current evidence confirmation" % len(required)
    stale, *_ = build(tmp_path, [replace(reviews[0], binding="0" * 64), *reviews[1:]])
    assert stale["review"]["status"].startswith("unreviewed: review: 1 of")
    own, *_ = build(tmp_path, confirmations(data, reviewer=evidence.author_of("bench.procedures.reference", procedures.REVISION)))
    assert own["review"]["status"] == "reviewed" and own["review"]["self_review"] and "includes author self-review" in own["review"]["disclosure"]


def test_replacement_needs_confirmation_dispute_blocks_and_skip_removes_before_assignment(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, built, golds, _ = build(tmp_path)
    reviews = confirmations(data)
    original = bindings(data)
    target = next(f for f in built if f.id == next(row["frame"] for row in data["review_worksheet"] if "prose" in row["frame"]))
    corrected = tuple(a for a in target.assertions if a.id != target.assertions[-1].id)
    replaced = Review(target.id, "rev", "evidence", "replace", original[target.id], replacement=corrected)
    manifest, after, *_ = build(tmp_path, [*[r for r in reviews if r.frame != target.id], replaced])
    assert manifest["review_application"]["replaced"] == [target.id] and target.id in manifest["review"]["unconfirmed_replacements"]
    assert manifest["review"]["status"].startswith("unreviewed")
    fixed = bindings(after)[target.id]
    assert fixed != original[target.id] and next(r for r in after["reviews"] if r["frame"] == target.id)["current"]
    confirm = Review(target.id, "rev", "evidence", "confirm", fixed)
    rows = {row["frame"]: row for row in after["review_worksheet"]}
    rest = [Review(f, "rev", "evidence", "confirm", row["binding"]) for f, row in rows.items() if f != target.id]
    manifest, *_ = build(tmp_path, [replaced, confirm, *rest])
    assert manifest["review"]["status"] == "reviewed" and not manifest["review"]["unconfirmed_replacements"]
    labelled, *_ = build(tmp_path, [replaced, Review(target.id, "rev", "label", "confirm", fixed), *rest])
    assert labelled["review"]["unconfirmed_replacements"] == [target.id] and labelled["review"]["status"].startswith("unreviewed")
    with pytest.raises(ConfigError, match="generated frame's worksheet binding"):
        build(tmp_path, [replaced, confirm, Review(target.id, "rev", "evidence", "skip", fixed), *rest])
    stale_confirm, *_ = build(tmp_path, [replaced, Review(target.id, "rev", "evidence", "confirm", original[target.id]), *rest])
    assert stale_confirm["review"]["disputes"] == [target.id]
    disputed, *_ = build(tmp_path, [*reviews, Review(target.id, "other", "evidence", "skip", original[target.id])])
    assert disputed["review"]["disputes"] == [target.id] and disputed["review"]["status"].startswith("unreviewed")
    other = next(f for f in built if f.id not in rows and f.condition != procedures.TRAINED)
    skipped, kept, *_ = build(tmp_path, [Review(other.id, "rev", "evidence", "skip", original[other.id])])
    assert skipped["review_application"]["skipped"] == [other.id] and other.id not in {row["frame"] for row in kept["reference"]}
    pack = evidence.load_pack(tmp_path / "source" / "pack.json")
    for bad, message in ((Review(target.id, "rev", "evidence", "confirm", None), "names the frame binding"),
                         (Review(target.id, "rev", "evidence", "confirm", fixed, ("tt_media",)), "frame-scoped"),
                         (Review(target.id, "rev", "label", "skip", fixed), "label review confirms"),
                         (Review(target.id, "rev", "evidence", "skip", fixed, replacement=corrected), "only a replacement"),
                         (Review(target.id, "rev", "label", "confirm", fixed, ("nope",)), "not targets of the task")):
        with pytest.raises(ConfigError, match=message):
            evidence.review(pack, built, [bad], procedures.standard(), authored=procedures.authored())


def test_mechanical_tier_rejects_a_reference_that_does_not_recompute(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, *_ = build(tmp_path)
    flipped = lambda frame: replace(procedures.reference(frame), prohibited=("resume_printing",)) if frame.id.endswith(":structured:en") else procedures.reference(frame)
    manifest, *_ = build(tmp_path, confirmations(data), reference=replace(procedures.standard(), evaluate=flipped))
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
    chosen = {composed[f].branch for f in plan["review"]}
    assert chosen == branches and plan["compositions"] >= len(branches)
    assert json.dumps(asdict(composed[built[0].id]))


def lineage(data, frame_id):
    row = next(r for r in data["labels"] if r["label_sources"]["tt_media"][0]["address"][0] == frame_id)
    return {fact: len(history[0].get("verifications", ())) for fact, history in row["label_sources"].items()}


def test_label_confirmations_support_only_their_targets_and_never_complete_frame_review(tmp_path):
    (tmp_path / "source").mkdir()
    manifest, data, built, golds, _ = build(tmp_path)
    current = [name for name in procedures.PRIMITIVES if name not in files.read_yaml(procedures.DEFINITION)["retired"]]
    required = [row["frame"] for row in data["review_worksheet"]]
    total = sum(len(evidence.targets(golds[frame.id])) for frame in built)
    assert manifest["review"]["frames"]["labels"] == {"targets": total, "confirmed": 0}
    assert all(set(row["targets"]) == {'/'.join(target) for target in evidence.targets(golds[row['frame']])} for row in data["review_worksheet"])
    action, *_ = build(tmp_path, confirmations(data, kind="label", target=("actions", "resume_printing")))
    review = action["review"]
    assert review["status"] == f"unreviewed: review: {len(required)} of {len(required)} required frames lack a current evidence confirmation"
    assert review["frames"]["confirmed"]["frames"] == 0 and review["frames"]["labels"]["confirmed"] == len(required) and not review["disputes"]
    labelled = next(f for f in required if f in {r["label_sources"]["tt_media"][0]["address"][0] for r in data["labels"]})
    whole = [Review(labelled, "rev", "label", "confirm", bindings(data)[labelled])]
    one = [Review(labelled, "ana", "label", "confirm", bindings(data)[labelled], ("tt_media",)),
           Review(labelled, "bo", "label", "confirm", bindings(data)[labelled], ("ribbon_loaded",))]
    partial, partial_data, *_ = build(tmp_path, one)
    assert partial["review"]["frames"]["labels"]["confirmed"] == 2 and not partial["review"]["disputes"] and partial["review"]["frames"]["confirmed"]["frames"] == 0
    assert lineage(partial_data, labelled) == {**dict.fromkeys(current, 0), "tt_media": 1, "ribbon_loaded": 1}
    sources = next(r for r in partial_data["labels"] if r["label_sources"]["tt_media"][0]["address"][0] == labelled)["label_sources"]
    assert {fact: [(v["verifier"], v["actor"]["kind"], v["method"], v["criterion"], v["outcome"]) for v in history[0].get("verifications", ())]
            for fact, history in sources.items() if fact in ("tt_media", "ribbon_loaded")} == {
        "tt_media": [("ana", "agent", "review", bindings(data)[labelled], "confirm")], "ribbon_loaded": [("bo", "agent", "review", bindings(data)[labelled], "confirm")]}
    assert all(source["derivation"] == "computed" and source["revision"] for history in sources.values() for source in history)
    full_labels, full_data, *_ = build(tmp_path, whole)
    assert full_labels["review"]["frames"]["labels"]["confirmed"] == len(evidence.targets(golds[labelled])) and full_labels["review"]["frames"]["confirmed"]["frames"] == 0
    assert lineage(full_data, labelled) == dict.fromkeys(current, 1)
    complete, complete_data, *_ = build(tmp_path, confirmations(data))
    assert complete["review"]["status"] == "reviewed" and complete["review"]["frames"]["labels"]["confirmed"] == sum(len(evidence.targets(golds[frame])) for frame in required)
    assert lineage(complete_data, labelled) == dict.fromkeys(current, 1)
    records = {(r["frame"], r["scope"]) for r in partial_data["reviews"]}
    assert records == {(labelled, "label tt_media"), (labelled, "label ribbon_loaded")} and all(r["current"] for r in partial_data["reviews"])
    from semantic_switch.compiler import compile_spec
    from semantic_switch.evidence import assessments
    plan = compile_spec(files.read_yaml(procedures.DEFINITION), procedures.DEFINITION.name, procedures.DEFINITION.parent)
    record = lambda rows: next(r for r in rows["labels"] if r["label_sources"]["tt_media"][0]["address"][0] == labelled)
    other = next(name for name in current if name not in ("tt_media", "ribbon_loaded"))
    targeted = assessments(plan, record(partial_data), partial["review"])
    assert (targeted["tt_media"].state, targeted["tt_media"].fidelity, targeted[other].state, targeted[other].fidelity) == ("verified", "unverified", "computed", "unverified")
    assert {(a.state, a.fidelity) for a in assessments(plan, record(complete_data), complete["review"]).values()} == {("verified", "verified")}
    drifted = {**complete["review"], "bindings": {key: ["0" * 64] for key in complete["review"]["bindings"]}}
    assert {(a.state, a.fidelity) for a in assessments(plan, record(complete_data), drifted).values()} == {("computed", "unverified")}
    assert {(a.state, a.fidelity) for a in assessments(plan, record(complete_data), None).values()} == {("computed", "undeclared")}


def test_source_rule_criterion_and_protocol_drift_make_every_decision_stale(tmp_path, monkeypatch):
    (tmp_path / "source").mkdir()
    _, data, built, *_ = build(tmp_path)
    original = bindings(data)
    skipped = next(f for f in built if f.id not in {row["frame"] for row in data["review_worksheet"]} and f.condition != procedures.TRAINED)
    reviews = [*confirmations(data), Review(skipped.id, "rev", "evidence", "skip", original[skipped.id])]
    manifest, *_ = build(tmp_path, reviews)
    assert manifest["review"]["status"] == "reviewed" and manifest["review"]["skipped"] == [skipped.id] and not manifest["review"]["stale"]
    required = len(data["review_worksheet"])
    expected = f"unreviewed: review: {required} of {required} required frames lack a current evidence confirmation"

    def drifted():
        manifest, after, *_ = build(tmp_path, reviews)
        assert manifest["review"]["status"] == expected and manifest["review"]["stale"] == sorted({r.frame for r in reviews})
        assert not manifest["review"]["skipped"] and skipped.id in {row["frame"] for row in after["reference"]}
        assert not any(r["current"] for r in after["reviews"]) and manifest["review"]["records"] == {"current": 0, "stale": len(reviews), "withdrawn": 0}
    with monkeypatch.context() as patch:
        patch.setattr(evidence, "PROTOCOL_DIGEST", "0" * 64)
        drifted()
    with monkeypatch.context() as patch:
        spec = procedures.definition("structured")
        edited = {**spec, "facts": {**spec["facts"], "needs_observation": {"check": "A decision depends on an observation that is absent."}}}
        patch.setattr(procedures, "definition", lambda condition: edited)
        drifted()
    source = tmp_path / "source"
    pack = files.read_json(source / "pack.json")
    rules = [{**r, "text": r["text"] + " Revised."} if r["id"] == "ZT-RESUME" else r for r in pack["rules"]]
    files.publish(source / "pack.json", {**pack, "rules": rules})
    evidence.pin_authored(source / "pack.json", procedures.authored())
    drifted()
    files.publish(source / "pack.json", pack)
    assert build(tmp_path, reviews)[0]["review"]["status"] == "reviewed"
    (source / "zt-calibration.txt").write_text("re-extracted excerpt")
    evidence.update_source(source / "pack.json", "zt-calibration", sha256=files.sha256_file(source / "zt-calibration.txt"))
    drifted()


def test_conflicting_corrections_are_disputes_and_identical_ones_agree(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, built, *_ = build(tmp_path)
    frame = next(f for f in built if f.condition == "prose" and len(f.assertions) > 2)
    binding = bindings(data)[frame.id]
    first, second = frame.assertions[:-1], frame.assertions[1:]
    manifest, *_ = build(tmp_path, [Review(frame.id, "ana", "evidence", "replace", binding, replacement=first),
                                    Review(frame.id, "bo", "evidence", "replace", binding, replacement=second)])
    assert manifest["review"]["disputes"] == [frame.id] and not manifest["review_application"]["replaced"]
    agreed, *_ = build(tmp_path, [Review(frame.id, "ana", "evidence", "replace", binding, replacement=first),
                                  Review(frame.id, "bo", "evidence", "replace", binding, replacement=first)])
    assert not agreed["review"]["disputes"] and agreed["review_application"]["replaced"] == [frame.id]


def test_a_correction_that_splits_one_observable_problem_is_rejected(tmp_path):
    (tmp_path / "source").mkdir()
    _, data, built, *_ = build(tmp_path)
    frame = next(f for f in built if f.id.endswith(":base:structured:en") and "ribbon_out" in f.id)
    reading = next(a for a in frame.assertions if a.indication == "reading")
    corrected = tuple(replace(a, status="stale") if a is reading else a for a in frame.assertions)
    with pytest.raises(ConfigError, match=rf"{frame.id.replace(':structured:', ':structured_permuted:')} and {frame.id}: one observable problem .*prohibited"):
        build(tmp_path, [Review(frame.id, "rev", "evidence", "replace", bindings(data)[frame.id], replacement=corrected)])
