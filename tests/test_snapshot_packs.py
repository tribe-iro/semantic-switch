from __future__ import annotations

from dataclasses import replace
import json

import pytest
import yaml

from semantic_switch import ConfigError, Runtime, files
from semantic_switch.presentation import identity as presentation_identity
from bench import diagnostics, evidence, procedures, snapshots, support
from bench.evidence import Decision, Gold
from bench.outcomes import journal_path
from semantic_switch.journal import replay
from semantic_switch.tools.dataset import EvidenceContext
from tests.conftest import FakeScorer



def printer(variant, template, condition="structured", seed=0, language="en"):
    rows = [row for row in procedures.scenarios(seed, 1) if row["template"] == template and row["variant"] == variant]
    (frame,) = procedures.frames(rows, (condition,), (language,))
    return rows[0], frame, procedures.reference(frame)


def staged(tmp_path):
    if not (tmp_path / "stage").exists():
        evidence.stage(tmp_path / "pack", tmp_path / "stage")
    return tmp_path / "stage" / procedures.DEFINITION.with_suffix(".dataset.json").name


def frontier(gold):
    return set(procedures.justified(gold)), set(gold.prohibited), set(procedures.unresolved(gold))


def published(tmp_path, quotas=None):
    pack = evidence.load_pack(procedures.PACK)
    rows = procedures.scenarios(0, 1)
    built, golds = procedures.build(rows)
    assignment = evidence.assign(built, quotas or {"fit": 8, "model_select": 6}, seed=0, counted=evidence.situation_frames(built, golds, procedures.TRAINED))
    manifest = evidence.publish(tmp_path / "pack", pack, scenarios=rows, frames=built, golds=golds, definition=files.read_yaml(procedures.DEFINITION),
                                name=procedures.DEFINITION.name, condition=procedures.TRAINED, assignment=assignment,
                                producer="bench.procedures.reference", producer_digest=evidence.code_digest(procedures, evidence),
                                reference=procedures.reference, fields=procedures.FIELDS, stratum=procedures.stratum, authored=procedures.authored())
    return manifest, built, golds, assignment


def test_hand_authored_printer_cases_check_the_reference():
    for case in files.read_jsonl(procedures.DATA / "expected.jsonl"):
        _, frame, gold = printer(case["variant"], case["template"], case.get("condition", "structured"))
        assert frontier(gold) == (set(case["justified"]), set(case["prohibited"]), set(case["unresolved"])), case


def test_hand_authored_support_cases_check_the_reference():
    for case in files.read_jsonl(support.DATA / "expected.jsonl"):
        frame = support.render("support-case", case["statements"], case.get("account", {}), frame_id="case", family="support-case", order="A10000")
        evidence.validate_frame(frame, support.FIELDS)
        gold = support.reference(frame)
        assert {t: v for t, v in gold.values().items() if v not in (False,) and t != "impact"} == case["members"], case
        assert support.response(frame, gold).value == case["response"], case


def test_missing_contradictory_stale_command_and_wrong_asset_evidence():
    _, _, missing = printer("missing_media", "method_mismatch")
    _, _, contradiction = printer("contradiction", "ribbon_out")
    _, _, stale = printer("stale_media", "method_mismatch")
    _, _, other = printer("other_asset", "method_mismatch")
    _, _, base = printer("base", "method_mismatch")
    assert "ZT-CONFLICT" not in missing.decisions["actions/set_method_tt"].rules and missing.decisions["actions/set_method_tt"].value is None
    assert "ZT-CONFLICT" in contradiction.decisions["actions/load_ribbon"].rules and contradiction.primitives["ribbon_loaded"].value is None
    assert "ZT-CURRENT" in stale.decisions["actions/set_method_tt"].rules and "ZT-ASSET" in other.decisions["actions/set_method_tt"].rules
    assert base.decisions["actions/set_method_tt"].value is True and frontier(stale) == frontier(other) == frontier(missing)
    _, _, commanded = printer("commanded_load", "ribbon_out")
    _, _, verified = printer("verified_load", "ribbon_out")
    assert frontier(commanded) == ({"verify_loading", "request_observation"}, set(), {"load_ribbon", "resume_printing"})
    assert frontier(verified)[0] == {"resume_printing"}


def test_snapshot_contrasts_cover_several_actions_enabled_unjustified_early_steps_and_cancel():
    _, detection, gold = printer("base", "detection")
    assert frontier(gold)[0] == {"start_calibration", "verify_loading"}
    assert "resume_printing" in detection.input["enabled"] and "resume_printing" in gold.prohibited
    _, _, reported = printer("reported_step", "calibration_step")
    assert {"calibration_next", "resume_printing"} <= frontier(reported)[1] and "verify_loading" in frontier(reported)[0]
    _, _, cancelled = printer("cancelled", "cancelled")
    _, _, completed = printer("completed", "cancelled")
    assert "resume_printing" in cancelled.prohibited and frontier(completed)[0] == {"resume_printing"}
    assert cancelled.primitives["calibration_verified"].value is False


def test_structured_and_prose_evidence_parity_and_hidden_state_invariance():
    rows = procedures.scenarios(1, 1)
    built, golds = procedures.build(rows)
    by = {}
    for frame in built:
        by.setdefault(evidence.counterpart(frame.id), []).append(golds[frame.id])
    assert all(len(group) == 6 and all(g.decisions == group[0].decisions and g.primitives == group[0].primitives for g in group) for group in by.values())
    row = next(r for r in rows if r["variant"] == "missing_media")
    other = {**row, "state": {**row["state"], "media": procedures.DT, "ribbon": False}}
    (a,), (b,) = procedures.frames([row], ("structured",), ("en",)), procedures.frames([other], ("structured",), ("en",))
    assert a.content == b.content and procedures.reference(a) == procedures.reference(b)
    evidence.invariant(procedures.reference, [a, b])
    with pytest.raises(ConfigError, match="identical public evidence"):
        evidence.invariant(lambda frame: frame.id, [a, replace(b, id="other")])


def test_spans_paths_leakage_and_added_text_are_validated():
    _, frame, _ = printer("base", "method_mismatch")
    note = next(a for a in frame.assertions if a.spans)
    evidence.validate_frame(frame, procedures.FIELDS)
    with pytest.raises(ConfigError, match="text changed"):
        evidence.validate_frame(replace(frame, input={**frame.input, "notes": frame.input["notes"].replace("I", "We", 1)}), procedures.FIELDS)
    with pytest.raises(ConfigError, match="outside the public text"):
        evidence.validate_frame(replace(frame, assertions=(replace(note, spans=(replace(note.spans[0], end=10_000),)),)), procedures.FIELDS)
    with pytest.raises(ConfigError, match="structured path"):
        evidence.validate_frame(replace(frame, assertions=(replace(note, spans=(), paths=("missing",)),)), procedures.FIELDS)
    with pytest.raises(ConfigError, match="stated value"):
        evidence.validate_frame(replace(frame, assertions=(replace(note, value=None),)), procedures.FIELDS)
    with pytest.raises(ConfigError, match="undeclared fields"):
        evidence.validate_frame(replace(frame, input={**frame.input, "gold": ["set_method_tt"]}), procedures.FIELDS)
    assert evidence.leakage(replace(frame, input={**frame.input, "notes": f"{frame.input['notes']} {note.id} {frame.scenario}"}))
    assert not evidence.leakage(replace(frame, input={**frame.input, "notes": "Operator says: just reset it and resume printing."}))
    review = evidence.review_draft(frame.assertions[:-1], frame.assertions)
    assert review["decision"] == "reject" and review["additions"] and not review["omissions"]


def test_reference_validity_requires_revision_scope_and_criteria():
    _, frame, gold = printer("base", "method_mismatch")
    evidence.validate_gold(frame, gold, procedures.SCOPE, procedures.REVISION)
    with pytest.raises(ConfigError, match="revision"):
        evidence.validate_gold(frame, gold, procedures.SCOPE, "other")
    with pytest.raises(ConfigError, match="target scope"):
        evidence.validate_gold(frame, replace(gold, decisions={k: v for k, v in gold.decisions.items() if "resume" not in k}), procedures.SCOPE, procedures.REVISION)
    with pytest.raises(ConfigError, match="no criterion"):
        evidence.validate_gold(frame, replace(gold, decisions={**gold.decisions, "actions/load_ribbon": Decision(False, ())}), procedures.SCOPE, procedures.REVISION)
    plan_gold = Gold(procedures.REVISION, ("actions/x",), {"actions/x": Decision(True, ("ZT-DEFAULT",))}, primitives={"tt_media": Decision("maybe", ("ZT-DEFAULT",))})
    from semantic_switch.compiler import compile_spec
    with pytest.raises(ConfigError, match="is not a"):
        evidence.label(compile_spec(files.read_yaml(procedures.DEFINITION)), frame, plan_gold, "fit", "p", "d" * 64, "t", "sel")


def test_orphan_rules_and_unknown_rule_sources(tmp_path):
    manifest = tmp_path / "pack.json"
    data = files.read_json(procedures.PACK)
    pack = evidence.load_pack(procedures.PACK)
    report = evidence.check_rules(pack, {"ZT-DEFAULT"})
    assert "ZT-RIBBON" in report["orphans"] and "ZT-DEFAULT" in report["conventions"]
    with pytest.raises(ConfigError, match="undefined rules"):
        evidence.check_rules(pack, {"ZT-NOPE"})
    files.publish(manifest, {**data, "rules": [*data["rules"], {"id": "X", "text": "t", "source": "nowhere"}]})
    with pytest.raises(ConfigError, match="name no source"):
        evidence.load_pack(manifest)
    files.publish(manifest, {**data, "sources": [{**data["sources"][0], "use": "maybe"}, *data["sources"][1:]]})
    with pytest.raises(ConfigError, match="declare neither"):
        evidence.load_pack(manifest)


def test_whole_family_assignment_is_exact_or_rejected():
    rows = procedures.scenarios(0, 2)
    built, golds = procedures.build(rows)
    counted = evidence.situation_frames(built, golds, procedures.TRAINED)
    assert counted and all(f.condition == "structured" for f in built if f.id in counted) and len(counted) < sum(f.condition == "structured" for f in built)
    assignment = evidence.assign(built, {"fit": 12, "model_select": 14}, seed=4, counted=counted)
    roles = {f.id: assignment["families"][assignment["components"][f.id]] for f in built}
    assert assignment["counts"]["fit"]["situations"] == 12 and assignment["counts"]["model_select"]["situations"] == 14
    assert sum(c["frames"] for c in assignment["counts"].values()) == len(built)
    assert sum(c["situations"] for c in assignment["counts"].values()) == len(counted)
    for frame in built:
        assert len({roles[other] for other in (frame.id, *frame.pairs)}) == 1
    with pytest.raises(ConfigError, match="exactly 7 fit situations"):
        evidence.assign(built, {"fit": 7}, seed=4, counted=counted)
    held = evidence.assign(built, {"fit": 48}, seed=4, check_only=lambda frame: "cancelled" in frame.tags)
    assert all(roles_ == "check" for f in built if "cancelled" in f.tags for roles_ in [held["families"][held["components"][f.id]]])


def test_deterministic_publication_labels_and_digest_validation(tmp_path):
    first, built, golds, _ = published(tmp_path)
    labels = files.read_jsonl(tmp_path / "pack" / first["payloads"]["labels.jsonl"]["file"])
    assert labels and all({s["origin"] for h in row["label_sources"].values() for s in (h if isinstance(h, list) else [x for v in h.values() for x in v])} == {"rule"}
                          and "actions" not in row["facts"] for row in labels)
    row = labels[0]
    assert set(row["label_sources"]) == set(row["facts"]) and row["label_sources"]["tt_media"][0]["origin"] == "rule"
    situations = files.read_jsonl(tmp_path / "pack" / first["payloads"]["situations.jsonl"]["file"])
    assert all(set(s) == {"schema", "input", "role", "groups", "pairs", "tags"} for s in situations)
    again, _, _, _ = published(tmp_path)
    assert {k: v["sha256"] for k, v in again["payloads"].items()} == {k: v["sha256"] for k, v in first["payloads"].items()}
    manifest, frames, meta = evidence.load_frames(tmp_path / "pack")
    assert [f.content for f in frames] == [f.content for f in built]
    path = tmp_path / "pack" / first["payloads"]["labels.jsonl"]["file"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ConfigError, match="payload digest"):
        evidence.read(tmp_path / "pack")


def test_support_contrasts_composition_and_abcd_prefixes(tmp_path):
    rows = support.scenarios(0, 2)
    built = support.frames(rows)
    golds = support.build(built)
    for frame in built:
        assert set(golds[frame.id].decisions) == set(support.SCOPE)
    ambiguous = next(f for f in built if f.id.endswith(":ambiguous:message:pt"))
    assert not golds[ambiguous.id].complete and support.response(ambiguous, golds[ambiguous.id]).value == "clarify"
    data = {"train": [{"convo_id": 7, "original": [["customer", "My shirt arrived with a stain."], ["agent", "Sorry!"],
                                                   ["customer", "I want a refund."]], "scenario": {"flow": "product_defect", "subflow": "refund_initiate"}}]}
    (tmp_path / "abcd.json").write_text(json.dumps(data))
    files.publish(tmp_path / "ann.jsonl", json.dumps({"convo_id": 7, "split": "train", "prefix_turns": 2, "order": "A10001",
                                                      "assertions": [{"property": "issue:stain_damage", "status": "current", "quote": "arrived with a stain"}]}) + "\n")
    (frame,) = support.abcd(tmp_path / "abcd.json", tmp_path / "ann.jsonl")
    gold = support.reference(frame)
    assert gold.values()["issues/stain_damage"] is True and gold.values()["requests/refund_initiate"] is False
    assert "refund" not in frame.input["message"] and frame.family == "abcd:7"
    files.publish(tmp_path / "ann.jsonl", json.dumps({"convo_id": 7, "split": "train", "prefix_turns": 2, "order": "A1",
                                                      "assertions": [{"property": "request:refund_initiate", "status": "current", "quote": "I want a refund"}]}) + "\n")
    with pytest.raises(ConfigError, match="not in the prefix"):
        support.abcd(tmp_path / "abcd.json", tmp_path / "ann.jsonl")


def test_unexamined_members_stay_unresolved_and_menu_permutation_keeps_ids(tmp_path):
    domain = snapshots.DOMAINS["zt411"]
    got = snapshots.targets(domain, {"actions": {"status": "uncertain", "states": {"set_method_tt": "unexamined", "load_ribbon": "applicable"}}})
    assert got["actions/set_method_tt"]["value"] is None and got["actions/load_ribbon"]["value"] is True and got["actions/resume_printing"]["value"] is None
    _, frame, _ = printer("base", "detection")
    oracle = lambda state, question, width: [.97, .03, 0.][:width] if "detection problem" in question or "attempted or reported" in question else [.03, .97, 0.][:width]
    spec = procedures.definition("structured")
    permuted = {**spec, "facts": {**spec["facts"], "actions": {**spec["facts"]["actions"], "cards": spec["facts"]["actions"]["cards"][::-1]}}}
    selected = []
    for definition in (spec, permuted):
        path = tmp_path / f"{len(selected)}.yaml"
        path.write_text(yaml.safe_dump(definition, sort_keys=False))
        with Runtime({"judge": FakeScorer(oracle)}) as rt:
            selected.append(rt.load(path, draft=True)(frame.input).facts["actions"].value)
    assert set(selected[0]) == set(selected[1]) == {"start_calibration", "verify_loading"}


def test_public_fake_runtime_snapshot_round_trip_records_identities_and_exposure(tmp_path):
    _, built, golds, assignment = published(tmp_path)
    scorers = {"base": FakeScorer(), "other": FakeScorer(name="other")}
    artifact = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), scorers, tmp_path / "out" / "eval.json", baseline="base", margin=.02)
    check = [f for f in built if assignment["families"][assignment["components"][f.id]] == "check"]
    assert artifact["fingerprint"]["presentation"] == presentation_identity() and set(artifact["fingerprint"]["arms"]) == {"base", "other"}
    for key, result in artifact["results"].items():
        condition = key.split("|")[2]
        assert len(result["rows"]) == sum(f.condition == condition for f in check)
        assert result["scored"]["expected"] == sum(f.condition == condition and golds[f.id].complete for f in check)
    assert {p["condition"] for p in artifact["paired"]} == {f"{k}:{c}" for k in ("workflow", "direct") for c in procedures.CONDITIONS}
    context = EvidenceContext.from_manifest(staged(tmp_path))
    audit = files.read_jsonl(context.audit)
    assert len(audit) == 1 and audit[0]["plan"]["id"] == artifact["plan"]["id"] and audit[0]["dataset_id"] == context.dataset_id
    assert {"baseline:base", "reference:other"} == set(artifact["plan"]["arms"])
    before = (tmp_path / "pack" / "benchmark.json").read_bytes()
    again = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), scorers, tmp_path / "out" / "eval2.json", baseline="base", margin=.02)
    assert len(files.read_jsonl(context.audit)) == 1 and (tmp_path / "pack" / "benchmark.json").read_bytes() == before
    assert again["exposure"]["state"] == "repeat" and again["plan"]["id"] == artifact["plan"]["id"]
    assert all(p["role"] == "reference" and "adoption" not in p for p in artifact["paired"])
    with pytest.raises(ConfigError, match="deployment receipt"):
        snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), scorers, tmp_path / "out" / "eval4.json", baseline="base", specialist="other", margin=.02)
    other = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), scorers, tmp_path / "out" / "eval3.json", baseline="other", margin=.02)
    assert other["exposure"]["state"] == "consulted" and len(files.read_jsonl(context.audit)) == 2
    row = artifact["results"]["base|workflow|structured"]["rows"][0]
    assert row["requests"] == 0 and row["logical_tokens"] > 0 and row["ms"] > 0 and row["native_work"] is not None and row["unavailable"] == {}
    assert {"en", "pt"} <= set(artifact["results"]["base|workflow|structured"]["slices"]) and artifact["counterparts"]["base|workflow"]["language"]["pairs"]
    baselines = artifact["baselines"]["actions"]
    assert baselines["always_empty"]["actions"]["recall"] == 0 and baselines["select_all"]["actions"]["prohibited_selected"] > 0 and "rule_only" in baselines
    assert all("exact" not in arm["actions"] and arm["exact"]["metric"] == "exact_sets" for arm in baselines.values())


def test_interrupted_check_retains_exposure(tmp_path):
    published(tmp_path)

    def broken(state, question, width):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(broken)}, tmp_path / "out" / "eval.json", baseline="base", margin=.02)
    assert files.read_jsonl(EvidenceContext.from_manifest(staged(tmp_path)).audit) and not (tmp_path / "out" / "eval.json").exists()


def perfect(frame):
    gold = procedures.reference(frame)
    return {"selected": set(procedures.justified(gold)), "unresolved": set()}


def test_episode_selection_transition_verification_and_replay():
    row, _, _ = printer("base", "ribbon_out")
    trace = procedures.episode(row, perfect, seed=0)
    actions = [step.get("scheduled") for step in trace["steps"]]
    assert actions[0] == "load_ribbon" and "verify_loading" in actions and trace["termination"] == "printing" and trace["goal_reached"]
    after = trace["steps"][1]
    assert "load_ribbon" in after["reference_unresolved"] and "verify_loading" in after["justified"]
    assert procedures.replay(row, trace)
    tampered = {**trace, "steps": [{**trace["steps"][0], "scheduled": "verify_loading"}, *trace["steps"][1:]]}
    assert not procedures.replay(row, tampered)


def test_episode_alternatives_divergence_termination_and_controls():
    row, _, _ = printer("base", "detection")
    traces = {seed: procedures.episode(row, perfect, seed=seed) for seed in range(6)}
    assert {t["steps"][0]["scheduled"] for t in traces.values()} == {"start_calibration", "verify_loading"}
    assert all(t["steps"][0]["choice"] == "seeded" for t in traces.values())
    prioritized = procedures.episode(row, perfect, seed=0, priority=("start_calibration",))
    assert prioritized["steps"][0]["choice"] == "document_priority"
    verifying = procedures.episode(row, lambda frame: {"selected": {"verify_loading"}, "unresolved": set()}, seed=0)
    assert verifying["termination"] == "nonprogress"
    capped = procedures.episode(row, perfect, seed=0, cap=1)
    assert capped["termination"] == "cap" and len(capped["steps"]) == 1
    prohibited = procedures.episode(row, lambda frame: {"selected": {"resume_printing", "start_calibration"}, "unresolved": set()}, seed=0)
    assert prohibited["termination"] == "prohibited_selected" and "scheduled" not in prohibited["steps"][0]
    controls = procedures.controls(row, seed=0)
    assert controls["always_abstain"]["termination"] == "empty_selection" and not controls["always_cancel"]["goal_reached"]
    calibrating, _, _ = printer("reported_step", "calibration_step")
    cancelled = procedures.episode(calibrating, lambda frame: {"selected": {"cancel_calibration"}, "unresolved": set()}, seed=0)
    assert not cancelled["goal_reached"] and cancelled["steps"][0]["scheduled"] == "cancel_calibration"


def test_interrupted_episode_restarts_from_initial_state():
    row, _, _ = printer("base", "ribbon_out")
    calls = []

    def flaky(frame):
        calls.append(frame.id)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return perfect(frame)
    with pytest.raises(KeyboardInterrupt):
        procedures.episode(row, flaky, seed=0)
    assert procedures.episode(row, flaky, seed=0) == procedures.episode(row, perfect, seed=0)


def test_public_snapshot_demos_run_through_runtime_only():
    from examples import printer_snapshot, support_snapshot
    with Runtime({"judge": FakeScorer()}) as rt:
        printed = printer_snapshot.run(rt.load(printer_snapshot.DEFINITION, draft=True))
        supported = support_snapshot.run(rt.load(support_snapshot.DEFINITION, draft=True))
    assert set(printed["menu"]) == set(procedures.MENU) and len(printed["decisions"]) == len(printer_snapshot.EVIDENCE)
    assert len(supported["decisions"]) == 3 and all("issues" in row for row in supported["decisions"])
    assert not any(key in json.dumps(printed["decisions"]) for key in ("private", "annotation", "scenario"))


def test_episode_runner_publishes_replayable_traces(tmp_path):
    published(tmp_path)
    artifact = snapshots.episodes(tmp_path / "pack", FakeScorer(), tmp_path / "out" / "episodes.json")
    assert artifact["traces"] and sum(artifact["terminations"].values()) == len(artifact["traces"]) == artifact["progress"]["completed"]
    assert files.read_json(tmp_path / "out" / "episodes.json")["controls"]["always_abstain"] == 0
    assert artifact["execution"]["completed"] == len(artifact["traces"])


def test_episode_runner_commits_complete_episodes_and_resumes(tmp_path, monkeypatch):
    published(tmp_path)
    output = tmp_path / "out" / "episodes.json"
    monkeypatch.setattr(diagnostics, "fingerprint", lambda: {"revision": None, "sources": "code"})
    original, calls = procedures.episode, []

    def interrupted(row, decide, **kwargs):
        calls.append(row["scenario"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return original(row, decide, **kwargs)
    monkeypatch.setattr(procedures, "episode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        snapshots.episodes(tmp_path / "pack", FakeScorer(), output)
    monkeypatch.setattr(procedures, "episode", original)
    artifact = snapshots.episodes(tmp_path / "pack", FakeScorer(), output)
    assert artifact["progress"]["resumed"] == 1 and artifact["execution"]["completed"] == len(artifact["traces"]) == artifact["progress"]["total"]
    with pytest.raises(ConfigError, match="different run"):
        snapshots.episodes(tmp_path / "pack", FakeScorer(), output, seed=1)
    monkeypatch.setattr(diagnostics, "fingerprint", lambda: {"revision": None, "sources": "changed"})
    with pytest.raises(ConfigError, match="different run"):
        snapshots.episodes(tmp_path / "pack", FakeScorer(), output)


def test_counterpart_slices_share_family_and_pairing_identity(tmp_path):
    rows = support.scenarios(3, 2)
    built = support.frames(rows)
    golds = support.build(built)
    assignment = evidence.assign(built, {"fit": 40}, seed=0)
    for frame in built:
        assert {assignment["components"][other] for other in frame.pairs} == {assignment["components"][frame.id]}
    pt = next(f for f in built if f.id.endswith(":base:message:pt"))
    en = next(f for f in built if f.id == pt.id.replace(":pt", ":en"))
    assert golds[pt.id] == golds[en.id] and pt.input["message"] != en.input["message"] and "pt" in pt.tags
    labels = support.definition("message_permuted")["facts"]["issues"]["labels"]
    assert list(labels) != list(support.ISSUES) and set(labels) == set(support.ISSUES)
    a = {"key": "s:v:message:en", "gold": {"values": {"x": True}}, "targets": {"x": {"value": True}}, "correct": True}
    report = snapshots.counterparts([a, {**a, "key": "s:v:message:pt"}, {**a, "key": "s:v:message_permuted:en", "targets": {"x": {"value": None}}, "correct": False},
                                     {**a, "key": "s:w:message:en", "gold": {"values": {"x": False}}}])
    assert report["language"] == {"pairs": 1, "same_prediction": 1, "both_correct": 1, "one_correct": 0}
    assert report["condition"]["pairs"] == 1 and report["condition"]["one_correct"] == 1 and report["condition_and_language"]["pairs"] == 1


def test_adoption_requires_lower_bound_strictly_above_required_improvement():
    reported = {"support": "reported"}
    even = {"forbidden_action_situations": 1, "failures": 0}

    def verdict(interval, margin=.02, specialist=even, exposure="fresh", review="reviewed", support="reported"):
        state = {"state": exposure, "independent": exposure in ("fresh", "repeat")}
        return snapshots.adoption({"support": support, "interval": interval}, margin, even, specialist, exposure=state, review=review)["verdict"]
    assert verdict([.03, .1]) == "adopt"
    assert verdict([.02, .1]) == "not_adopted" and verdict([0., .1], margin=0) == "not_adopted"
    assert verdict([-.01, .05]) == "not_adopted"
    assert verdict([.001, .1], margin=0) == "adopt"
    assert verdict([.03, .1], support="sparse") == "inconclusive"
    assert verdict([.03, .1], exposure="consulted") == verdict([.03, .1], exposure="unknown_history") == "reused_evidence"
    assert verdict([.03, .1], exposure="repeat") == "adopt"
    assert verdict([.03, .1], review="unreviewed") == "unreviewed_reference"
    assert verdict([.03, .1], specialist={"forbidden_action_situations": 2, "failures": 0}) == "constraint_violated"
    decision = snapshots.adoption({**reported, "interval": [.03, .1]}, .02, even, even, exposure={"state": "fresh", "independent": True}, review="reviewed")
    assert decision["schema"] == 2 and decision["required_improvement"] == .02 and decision["lower_bound"] == .03 and "tolerated_regression" not in decision


def test_snapshot_rows_commit_once_through_the_journal_and_resume_pending(tmp_path):
    published(tmp_path)
    output = tmp_path / "out" / "eval.json"
    asked = []

    def interrupted(state, question, width):
        asked.append(question)
        if len(asked) == 120:
            raise KeyboardInterrupt
        return [.03, .97, 0.][:width]
    with pytest.raises(KeyboardInterrupt):
        snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(interrupted)}, output, baseline="base", margin=0, kinds=("workflow",))
    _, committed, _, _ = replay(journal_path(output))
    assert committed and not output.exists()
    artifact = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(lambda s, q, w: [.03, .97, 0.][:w])}, output, baseline="base", margin=0,
                                  kinds=("workflow",))
    execution = artifact["execution"]
    assert execution["attempts"] == execution["completed"] == sum(len(r["rows"]) for r in artifact["results"].values()) and not execution["superseded"]
    assert sum(r["progress"]["resumed"] for r in artifact["results"].values()) == len(committed)


def test_failed_snapshot_rows_are_retained_unless_fresh(tmp_path):
    published(tmp_path)
    output = tmp_path / "out" / "eval.json"

    def broken(state, question, width):
        raise RuntimeError("backend down")
    first = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(broken)}, output, baseline="base", margin=0, kinds=("workflow",))
    rows = [row for result in first["results"].values() for row in result["rows"]]
    assert rows and all(row["outcome"] == "failed" and row["ms"] > 0 for row in rows)
    assert all(result["scored"]["failed"] == result["scored"]["submitted"] for result in first["results"].values())
    again = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(broken)}, output, baseline="base", margin=0, kinds=("workflow",))
    assert again["execution"]["attempts"] == first["execution"]["attempts"] and again["execution"]["run_id"] == first["execution"]["run_id"]
    fresh = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer(broken)}, output, baseline="base", margin=0, kinds=("workflow",), fresh=True)
    assert fresh["fresh"] and fresh["execution"]["run_id"] != first["execution"]["run_id"]


def test_published_report_set_aggregates_recompute_from_rows_or_are_unrecoverable(tmp_path):
    from bench import recompute
    published(tmp_path)
    artifact = snapshots.evaluate("zt411", tmp_path / "pack", staged(tmp_path), {"base": FakeScorer()}, tmp_path / "out" / "eval.json", baseline="base", margin=.02, kinds=("workflow",))
    states = recompute.recompute(tmp_path / "out" / "eval.json", tmp_path / "out" / "recomputed.json")
    again = files.read_json(tmp_path / "out" / "recomputed.json")
    assert set(states.values()) == {"recomputed"} and all(again["results"][k]["sets"] == v["sets"] for k, v in artifact["results"].items())
    assert again["baselines"] == artifact["baselines"] and again["recomputed"]["metrics"] == 3 and again["recomputed"]["source_sha256"]
    old = files.read_json(tmp_path / "out" / "eval.json")
    for result in old["results"].values():
        for row in result["rows"]:
            row["targets"] = {t: {k: v for k, v in p.items() if k != "state"} for t, p in row["targets"].items()}
        result["sets"]["actions"]["frontier"] = {"exact": 1, "exact_rate": 1.}
    for arms in old["baselines"].values():
        for arm in arms.values():
            arm.pop("rows")
    files.publish(tmp_path / "out" / "old.json", old)
    assert set(recompute.recompute(tmp_path / "out" / "old.json", tmp_path / "out" / "old-recomputed.json").values()) == {"unrecoverable"}
    repaired = files.read_json(tmp_path / "out" / "old-recomputed.json")
    sets = repaired["results"]["base|workflow|structured"]["sets"]["actions"]
    assert sets["exact"]["state"] == sets["actions"]["state"] == "unrecoverable" and "frontier" not in sets
    assert all(arm["exact"]["state"] == "unrecoverable" and "rows" not in arm for arm in repaired["baselines"]["actions"].values())
    with pytest.raises(FileExistsError):
        recompute.recompute(tmp_path / "out" / "eval.json", tmp_path / "out" / "recomputed.json")


def test_missing_media_unknown_reference_is_never_exact_through_every_set_adapter(tmp_path):
    from bench import recompute
    domain = snapshots.DOMAINS["zt411"]
    keys = domain.sets["actions"]
    row, frame, gold = printer("missing_media", "method_mismatch")
    _, base, complete = printer("base", "method_mismatch")
    unknown = {"set_method_tt", "resume_printing"}
    assert set(procedures.unresolved(gold)) == unknown and procedures.justified(gold) == ("request_observation",)

    def scored(reference, states, outcome="predicted"):
        return {"gold": snapshots.gold_of(reference), "outcome": outcome, "targets": snapshots.targets(domain, {"actions": {"states": states}})}
    matching = {k: "applicable" if k == "request_observation" else "rejected" for k in keys}
    (canonical,) = snapshots.set_rows(domain, [scored(gold, matching)], "actions")
    assert set(canonical["reference_unresolved"]) == unknown and canonical["gold"] == ["request_observation"]
    assert not canonical["complete_gold"] and canonical["prediction_complete"]
    report = snapshots.set_metrics([canonical])
    assert (report["exact"]["gold_complete"], report["exact"]["exact"], report["exact"]["incomplete_reference"]) == (0, 0, 1)
    assert report["actions"]["known_reference_support"] == len(keys) - 2 and report["actions"]["precision"] == report["actions"]["recall"] == 1
    guessing = snapshots.set_metrics(snapshots.set_rows(domain, [scored(gold, {**matching, "resume_printing": "applicable"})], "actions"))["actions"]
    assert guessing["unassessed_selected"] == 1 and guessing["precision"] == 1 and guessing["known_reference"]["false_positive"] == 0
    empty = {"values": dict.fromkeys(gold.values(), False), "prohibited": []}
    rejected = dict.fromkeys(keys, "rejected")
    cases = [{"gold": empty, "outcome": "predicted", "targets": snapshots.targets(domain, {"actions": {"states": rejected}})},
             {"gold": empty, "outcome": "failed", "targets": snapshots.targets(domain, {})},
             {"gold": empty, "outcome": "failed", "targets": snapshots.targets(domain, {"actions": {"states": rejected}})},
             {"gold": empty, "outcome": "predicted", "targets": snapshots.targets(domain, {"actions": {"states": {k: "unexamined" for k in keys}}})}]
    exact = snapshots.set_metrics(snapshots.set_rows(domain, cases, "actions"))["exact"]
    assert (exact["gold_complete"], exact["exact"], exact["failed"]) == (4, 1, 2)

    ruled = {frame.id: {"actions": {**dict.fromkeys(keys, False), "request_observation": True}},
             base.id: {"actions": {**dict.fromkeys(keys, False), "set_method_tt": True, "load_ribbon": None}}}
    baselines = snapshots.baselines(domain, [frame, base], {frame.id: gold, base.id: complete}, ruled)["actions"]
    assert all(arm["exact"]["gold_complete"] == 1 and arm["exact"]["incomplete_reference"] == 1 for arm in baselines.values())
    assert baselines["rule_only"]["exact"]["exact"] == 0 and baselines["rule_only"]["exact"]["prediction_complete"] == 0
    assert baselines["rule_only"]["actions"]["prediction_unresolved"] == 1 and baselines["select_all"]["actions"]["unassessed_selected"] == 2
    assert baselines["always_empty"]["actions"]["prohibited_selected"] == 0 and baselines["select_all"]["actions"]["frames_with_prohibited"] == 1

    trace = procedures.episode(row, perfect, seed=0)
    failed = procedures.episode(printer("base", "method_mismatch")[0], lambda f: {"selected": set(), "unresolved": set(keys), "outcome": "failed"}, seed=0)
    assert trace["goal_reached"] and failed["termination"] == "failed"
    steps = snapshots.step_rows([trace, failed])
    assert set(steps[0]["reference_unresolved"]) == unknown and not steps[0]["complete_gold"]
    episode = snapshots.set_metrics(steps)["exact"]
    assert (episode["situations"], episode["gold_complete"], episode["exact"], episode["failed"]) == (4, 3, 2, 1)
    files.publish(tmp_path / "episodes.json", {"traces": [trace, failed]})
    assert recompute.recompute(tmp_path / "episodes.json", tmp_path / "again.json") == {"snapshot_steps": "recomputed"}
    assert files.read_json(tmp_path / "again.json")["snapshot_steps"] == snapshots.set_metrics(steps)
    files.publish(tmp_path / "old-episodes.json", {"traces": [{**trace, "steps": [{k: v for k, v in s.items() if k != "outcome"} for s in trace["steps"]]}]})
    assert recompute.recompute(tmp_path / "old-episodes.json", tmp_path / "old-again.json") == {"snapshot_steps": "unrecoverable"}
