from __future__ import annotations

from dataclasses import asdict
import json
import threading
import urllib.error
import urllib.request

import pytest

from semantic_switch import Actor, ConfigError, files
from bench import evidence, procedures, review
from bench.evidence import Review
from tests.test_pack_review import build

AGENT = Actor("agent", "claude-test", "session-1")
HUMAN = Actor("human")


@pytest.fixture
def pack(tmp_path):
    (tmp_path / "source").mkdir()
    build(tmp_path)
    return tmp_path


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_decisions_bind_to_the_worksheet_and_complete_coverage(pack):
    data = evidence.read(pack / "pack-0")
    worksheet = {row["frame"]: row["binding"] for row in data["review_worksheet"]}
    session = review.Session(pack / "pack-0", pack / "reviews.jsonl", "ana", AGENT)
    first = data["review_worksheet"][0]
    target = first["targets"][0]
    session.decide({"frame": first["frame"], "decision": "confirm", "target": target})
    for frame in worksheet:
        session.decide({"frame": frame, "decision": "confirm"})
    recorded = evidence.load_reviews(pack / "reviews.jsonl")
    assert len(recorded) == len(worksheet) + 1 and {r.frame: r.binding for r in recorded if r.kind == "evidence"} == worksheet
    assert recorded[0] == Review(first["frame"], "ana", "label", "confirm", worksheet[first["frame"]], tuple(target.split("/")), None, AGENT)
    assert session.progress()["required"] == {"frames": len(worksheet), "done": len(worksheet), "remaining": 0} and session.progress()["next"] is None
    manifest, *_ = build(pack, recorded)
    coverage = manifest["review"]
    assert coverage["status"] == "reviewed" and coverage["frames"]["labels"]["confirmed"] == sum(len(row["targets"]) for row in data["review_worksheet"])
    assert coverage["actors"] == [{"reviewer": "ana", "actor": "agent claude-test session session-1"}] and not coverage["self_review"]
    assert "ana (agent claude-test session session-1)" in coverage["disclosure"] and coverage["author"] == "bench.procedures.reference@" + procedures.REVISION
    published = evidence.read(pack / "pack-1")["reviews"]
    assert {(row["actor"]["kind"], row["current"]) for row in published} == {("agent", True)}


def test_resume_starts_at_the_first_frame_without_an_evidence_decision(pack):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    session = review.Session(pack / "pack-0", pack / "reviews.jsonl", "tulio", HUMAN)
    assert session.progress()["next"] == 0
    session.decide({"frame": rows[0]["frame"], "decision": "confirm"})
    session.decide({"frame": rows[1]["frame"], "decision": "confirm", "target": rows[1]["targets"][0]})
    session.decide({"frame": rows[2]["frame"], "decision": "confirm"})
    resumed = review.Session(pack / "pack-0", pack / "reviews.jsonl", "other", AGENT)
    progress = resumed.progress()
    sampled = sum(row["reasons"] == ["sample"] for row in rows)
    assert progress["next"] == 1 and progress["required"]["done"] == 2 and progress["required"]["remaining"] == len(rows) - 2
    assert progress["mandatory"]["frames"] + progress["sampled"]["frames"] == len(rows) and progress["sampled"]["frames"] == sampled
    assert [f["decisions"] for f in resumed.state()["frames"][:3]] == [["confirm"], [], ["confirm"]]
    with open(pack / "reviews.jsonl", "a") as fh:
        fh.write(json.dumps({**json.loads(lines(pack / "reviews.jsonl")[0]), "frame": rows[1]["frame"], "binding": "0" * 64}) + "\n")
    assert review.Session(pack / "pack-0", pack / "reviews.jsonl").progress()["next"] == 1


def test_invalid_decisions_are_rejected_without_writing(pack):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    prose = next(row for row in rows if ":prose:" in row["frame"])
    path = pack / "reviews.jsonl"
    session = review.Session(pack / "pack-0", path, "ana", AGENT)
    session.decide({"frame": prose["frame"], "decision": "confirm"})
    before = lines(path)
    original = [asdict(a) for a in session.frames[prose["frame"]].assertions]
    text = prose["input"]["notes"]
    for payload, message in (({"frame": "nope", "decision": "confirm"}, "not a frame of the current review worksheet"),
                             ({"frame": prose["frame"], "decision": "confirm", "binding": "x"}, "a decision carries"),
                             ({"frame": prose["frame"], "decision": "accept"}, "confirm/replace/skip"),
                             ({"frame": prose["frame"], "decision": "confirm", "target": "nope"}, "not a target of this frame"),
                             ({"frame": prose["frame"], "decision": "skip", "target": prose["targets"][0]}, "label review confirms"),
                             ({"frame": prose["frame"], "decision": "confirm"}, "already decided this frame evidence"),
                             ({"frame": rows[0]["frame"], "decision": "skip"}, "records its reason"),
                             ({"frame": rows[0]["frame"], "decision": "skip", "note": "x", "replacement": original[:-1]}, "only a replacement"),
                             ({"frame": rows[0]["frame"], "decision": "replace"}, "only a replacement"),
                             ({"frame": prose["frame"], "decision": "replace", "replacement": original}, "equals the generated assertions"),
                             ({"frame": prose["frame"], "decision": "replace",
                               "replacement": [{**original[0], "spans": [{"field": "notes", "start": 0, "end": len(text) + 1}]}]}, "outside the public text"),
                             ({"frame": prose["frame"], "decision": "replace", "replacement": [{"id": "a0"}]}, "replacement assertions carry")):
        with pytest.raises(ConfigError, match=message):
            session.decide(payload)
    with pytest.raises(ConfigError, match="decided this frame evidence differently"):
        review.Session(pack / "pack-0", path, "bo", HUMAN).decide({"frame": prose["frame"], "decision": "skip", "note": "unclear"})
    assert lines(path) == before
    for reviewer, actor in (("ana", lambda: Actor("agent", "m")), ("ana", lambda: Actor("human", "m")), ("ana", lambda: Actor("unknown")),
                            ("ana", lambda: None), ("", lambda: HUMAN)):
        with pytest.raises(ConfigError):
            review.Session(pack / "pack-0", path, reviewer, actor()).decide({"frame": rows[0]["frame"], "decision": "confirm"})
    files.append(path, files.jsonl([{k: v for k, v in json.loads(before[0]).items() if k != "actor"}]))
    with pytest.raises(ConfigError, match="record their actor"):
        evidence.load_reviews(path)


def test_replacement_rebuilds_and_needs_a_confirmation_of_the_corrected_binding(pack):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    prose = next(row for row in rows if ":prose:" in row["frame"] and len(row["assertions"]) > 1)
    path = pack / "reviews.jsonl"
    session = review.Session(pack / "pack-0", path, "ana", HUMAN)
    corrected = prose["assertions"][:-1]
    record = session.decide({"frame": prose["frame"], "decision": "replace", "note": "last note is not stated",
                             "replacement": [{**a, "spans": [{k: s[k] for k in ("field", "start", "end")} for s in a["spans"]]} for a in corrected]})
    assert record["binding"] == prose["binding"] and record["replacement"] == corrected and session.progress()["corrections"] == [prose["frame"]]
    for row in rows:
        if row["frame"] != prose["frame"]:
            session.decide({"frame": row["frame"], "decision": "confirm"})
    manifest, after, *_ = build(pack, evidence.load_reviews(path))
    assert manifest["review"]["unconfirmed_replacements"] == [prose["frame"]] and manifest["review"]["status"].startswith("unreviewed")
    rebuilt = review.Session(pack / "pack-1", path, "ana", HUMAN)
    fixed = rebuilt.rows[rebuilt.index[prose["frame"]]]["binding"]
    assert fixed != prose["binding"] and rebuilt.state()["frames"][rebuilt.index[prose["frame"]]]["decisions"] == []
    assert rebuilt.decide({"frame": prose["frame"], "decision": "confirm"})["binding"] == fixed
    while (n := rebuilt.progress()["next"]) is not None:
        rebuilt.decide({"frame": rebuilt.rows[n]["frame"], "decision": "confirm"})
    manifest, *_ = build(pack, evidence.load_reviews(path))
    assert manifest["review"]["status"] == "reviewed" and not manifest["review"]["unconfirmed_replacements"]


def test_server_binds_localhost_and_records_through_the_same_operation(pack):
    session = review.Session(pack / "pack-0", pack / "reviews.jsonl", "ana", HUMAN)
    httpd = review.server(session, 0)
    host, port = httpd.server_address
    assert host == "127.0.0.1"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>Frame review</title>" in page and "http" not in page.split("<script>")[1].replace("/source/", "")
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        frame = json.loads(urllib.request.urlopen(f"{base}/api/frame?i={state['progress']['next']}").read())
        assert frame["frame"] == session.rows[0]["frame"] and frame["binding"] == session.rows[0]["binding"] and frame["language"] in procedures.LANGUAGES
        post = lambda body, kind="application/json": urllib.request.urlopen(urllib.request.Request(base + "/api/decide", json.dumps(body).encode(),
                                                                                                     {"Content-Type": kind}))
        done = json.loads(post({"frame": frame["frame"], "decision": "confirm"}).read())
        assert done["record"]["binding"] == frame["binding"] and done["state"]["progress"]["next"] == 1
        for body, kind, code in (({"frame": frame["frame"], "decision": "confirm"}, "application/json", 400),
                                 ({"frame": session.rows[1]["frame"], "decision": "confirm"}, "text/plain", 404)):
            with pytest.raises(urllib.error.HTTPError) as error:
                post(body, kind)
            assert error.value.code == code
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/api/state", headers={"Host": "attacker.example"}))
        assert error.value.code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert len(evidence.load_reviews(pack / "reviews.jsonl")) == 1


def test_record_and_status_commands(pack, monkeypatch, capsys):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    path = pack / "reviews.jsonl"
    monkeypatch.setattr("sys.argv", ["bench.review", "record", str(pack / "pack-0"), "--reviews", str(path), "--reviewer", "claude", "--actor", "agent",
                                     "--model", "claude-test", "--session", "s1", "--frame", rows[0]["frame"], "--decision", "skip", "--note", "duplicate"])
    review.main()
    assert json.loads(capsys.readouterr().out)["actor"] == {"kind": "agent", "model": "claude-test", "session": "s1"}
    monkeypatch.setattr(review.Worksheet, "fresh", lambda self: build(pack, evidence.load_reviews(path))[0])
    monkeypatch.setattr("sys.argv", ["bench.review", "status", str(pack / "pack-0"), "--reviews", str(path)])
    review.main()
    out = json.loads(capsys.readouterr().out)
    assert out["records"] == 1 and out["progress"]["corrections"] == [rows[0]["frame"]] and not out["worksheet"]["current"]
    assert out["coverage"]["skipped"] == [rows[0]["frame"]] and out["worksheet"]["rebuild"].endswith(f"--reviews {path}")


def test_withdrawal_is_append_only_scoped_to_its_reviewer_and_reopens_the_decision(pack):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    path, first, target = pack / "reviews.jsonl", rows[0]["frame"], rows[0]["targets"][0]
    ana, bo = review.Session(pack / "pack-0", path, "ana", HUMAN), review.Session(pack / "pack-0", path, "bo", AGENT)
    for row in rows:
        ana.decide({"frame": row["frame"], "decision": "confirm"})
    ana.decide({"frame": first, "decision": "confirm", "target": target})
    manifest, *_ = build(pack, evidence.load_reviews(path))
    assert manifest["review"]["status"] == "reviewed" and manifest["review"]["records"]["withdrawn"] == 0
    before = lines(path)
    for session, payload, message in ((bo, {"frame": first, "note": "mine"}, "only a decision's own reviewer withdraws it"),
                                      (bo, {"frame": first, "target": target, "note": "mine"}, "only a decision's own reviewer withdraws it"),
                                      (ana, {"frame": first}, "records its reason"),
                                      (ana, {"frame": first, "decision": "skip", "note": "x"}, "a decision carries"),
                                      (ana, {"frame": first, "target": "nope", "note": "x"}, "not a target of this frame")):
        with pytest.raises(ConfigError, match=message):
            session.withdraw(payload)
    with pytest.raises(ConfigError, match="already decided this frame evidence; withdraw it first"):
        ana.decide({"frame": first, "decision": "skip", "note": "duplicate"})
    assert lines(path) == before
    with pytest.raises(ConfigError, match="no standing frame evidence decision"):
        evidence.standing([*evidence.load_reviews(path), Review(first, "bo", "evidence", "withdraw", rows[0]["binding"], (), None, AGENT, "mine")])
    record = ana.withdraw({"frame": first, "note": "confirmed too quickly"})
    assert record["decision"] == "withdraw" and record["kind"] == "evidence" and record["binding"] == rows[0]["binding"] and record["target"] == []
    assert lines(path)[:len(before)] == before and len(lines(path)) == len(before) + 1
    assert ana.progress()["next"] == 0 and ana.progress()["required"]["done"] == len(rows) - 1
    assert [(d["kind"], d["target"]) for d in ana.view(0)["decisions"]] == [("label", target)]
    with pytest.raises(ConfigError, match="no current frame evidence decision to withdraw$"):
        ana.withdraw({"frame": first, "note": "again"})
    manifest, *_ = build(pack, evidence.load_reviews(path))
    coverage = manifest["review"]
    assert coverage["status"].startswith(f"unreviewed: review: 1 of {len(rows)} required frames lack a current evidence confirmation")
    assert coverage["records"] == {"current": len(rows), "stale": 0, "withdrawn": 1} and coverage["frames"]["remaining_frames"] == 1
    assert manifest["review_application"]["withdrawn"] == [first] and not coverage["disputes"]
    published = [row for row in evidence.read(pack / "pack-2")["reviews"] if row["withdrawn"]]
    assert published == [{"frame": first, "kind": "evidence", "target": [], "scope": "frame evidence", "author": coverage["author"], "reviewer": "ana",
                          "actor": {"kind": "human"}, "self_review": False, "reviewer_distinct": True, "binding": rows[0]["binding"], "note": None,
                          "current": False, "verdict": "confirm", "corrections": None,
                          "withdrawn": {"actor": {"kind": "human"}, "note": "confirmed too quickly"}}]
    ana.withdraw({"frame": first, "target": target, "note": "skipping the frame"})
    ana.decide({"frame": first, "decision": "skip", "note": "duplicate of another frame"})
    assert [r.decision for r in ana.current()[first]] == ["skip"] and ana.progress()["corrections"] == [first]
    log = evidence.load_reviews(path)
    assert [(r.kind, r.decision) for r in log if r.frame == first] == [("evidence", "confirm"), ("label", "confirm"), ("evidence", "withdraw"),
                                                                         ("label", "withdraw"), ("evidence", "skip")]
    manifest, *_ = build(pack, log)
    assert manifest["review"]["skipped"] == [first] and manifest["review"]["records"]["withdrawn"] == 2 and not manifest["review"]["disputes"]


def test_withdraw_command_and_status(pack, monkeypatch, capsys):
    rows = evidence.read(pack / "pack-0")["review_worksheet"]
    path = pack / "reviews.jsonl"
    review.Session(pack / "pack-0", path, "claude", Actor("agent", "claude-test", "s1")).decide({"frame": rows[0]["frame"], "decision": "confirm"})
    agent = ["--reviewer", "claude", "--actor", "agent", "--model", "claude-test", "--session", "s2"]
    monkeypatch.setattr("sys.argv", ["bench.review", "withdraw", str(pack / "pack-0"), "--reviews", str(path), *agent, "--frame", rows[0]["frame"],
                                     "--note", "wrong source page"])
    review.main()
    assert json.loads(capsys.readouterr().out)["actor"] == {"kind": "agent", "model": "claude-test", "session": "s2"}
    monkeypatch.setattr(review.Worksheet, "fresh", lambda self: build(pack, evidence.load_reviews(path))[0])
    monkeypatch.setattr("sys.argv", ["bench.review", "status", str(pack / "pack-0"), "--reviews", str(path)])
    review.main()
    out = json.loads(capsys.readouterr().out)
    assert out["records"] == 2 and out["progress"]["required"]["done"] == 0 and out["coverage"]["records"]["withdrawn"] == 1
    assert out["withdrawn"] == [{"frame": rows[0]["frame"], "reviewer": "claude", "scope": "frame evidence", "decision": "confirm", "note": "wrong source page"}]
    monkeypatch.setattr("sys.argv", ["bench.review", "withdraw", str(pack / "pack-0"), "--reviews", str(path), *agent, "--frame", rows[0]["frame"],
                                     "--note", "again"])
    with pytest.raises(SystemExit):
        review.main()
    assert len(evidence.load_reviews(path)) == 2
