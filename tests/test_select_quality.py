from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from semantic_switch import ConfigError, Runtime, files
from bench import select_quality as sq
from .conftest import FakeScorer

SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "s1mb-sample.json").read_text())["cases"]
CIVIL = next(row for row in SAMPLE if row["benchmark"] == "civil-comments-noul-test-v1")


def emotions(case_id, text, targets):
    def question(key):
        return {"id": key, "task": "noul", "instructions": f"Does the text express {key}?", "system_prompt": "",
                "options": [{"id": "false", "description": f"The text does not express {key}."}, {"id": "true", "description": f"The text expresses {key}."}]}
    return {"benchmark": "go-emotions-noul-test-v1", "task": "noul", "targets": targets,
            "case": {"case_id": case_id, "state": {"text": text}, "questions": [question(key) for key in targets]}}


def civil(case_id, text, targets):
    row = copy.deepcopy(CIVIL)
    row["case"].update(case_id=case_id, state={"text": text})
    row["targets"] = {**row["targets"], **targets}
    return row


HARD, SOFT = ({"false": 0.0, "true": 1.0}, {"false": 1.0, "true": 0.0}), {"false": .5, "true": .5}
ROWS = [emotions("e1", "Thanks so much, this made my day!", {"gratitude": HARD[0], "joy": SOFT, "anger": HARD[1]}),
        emotions("e2", "Why would anyone do this to me?", {"gratitude": HARD[1], "joy": HARD[1], "anger": HARD[0]}),
        civil("c1", "You are all wonderful people.", {"threat": HARD[1], "toxicity": HARD[1]}),
        civil("c2", "I will find you and make you pay.", {"threat": HARD[0]})]


def definitions(rows):
    return {(row["case"]["state"]["text"], q["options"][1]["description"]): row["targets"][q["id"]] for row in rows for q in row["case"]["questions"]}


def oracle(rows, failing=()):
    known = definitions(rows)

    def answer(state, question, width):
        found = re.search(r"^Candidate: \S+ — (.+)$", question, re.M) or re.search(r"^Claim about the input above: (.+)$", question, re.M)
        description = found.group(1)
        if any(word in description for word in failing):
            raise RuntimeError("provider down")
        target = next(value for (text, desc), value in known.items() if text in state and desc == description)
        p = .97 if target["true"] == 1 else .03 if target["false"] == 1 else .6
        return [p, 1 - p]
    return answer


def test_select_members_present_the_same_evidence_and_definitions_as_named_checks():
    row = ROWS[2]
    members = sq.labels(row["benchmark"], row["case"])
    fake = FakeScorer(oracle([row]))
    with Runtime({"judge": fake}) as rt:
        found = sq.predict(rt, None, row["case"], sq.TASKS[row["benchmark"]], members)
    selected, checked = fake.requests[:len(members)], fake.requests[len(members):]
    assert len(selected) == len(checked) == len(members) == 7
    for key, label in members.items():
        member = next(state for state, q in selected if f"Candidate: {key} — {label.description}\n" in q and sq.TASKS[row["benchmark"]] in q)
        check = next(state for state, q in checked if f"Claim about the input above: {label.description}\n" in q)
        assert member == check and row["case"]["state"]["text"] in member
        s, c = found["select"]["result"]["members"][key], found["checks"]["result"][key]
        assert {k: s[k] for k in ("status", "value", "prediction", "p_true", "mass")} == {k: c[k] for k in ("status", "value", "prediction", "p_true", "mass")}
    assert found["select"]["transport"] is None and found["select"]["result"]["complete"]
    with pytest.raises(ValueError, match="not an unprompted true/false criterion"):
        sq.labels("x", {"case_id": "1", "questions": [{**row["case"]["questions"][0], "system_prompt": "Be strict."}]})


class Interrupting(FakeScorer):
    def score(self, requests):
        if len(self.requests) >= 20:
            raise KeyboardInterrupt
        return super().score(requests)


def test_runner_scores_memberships_resumes_and_refuses_a_changed_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(sq, "CACHE", tmp_path)
    monkeypatch.setattr(sq, "install", lambda: {"evaluator": "pinned"})
    monkeypatch.setattr(sq, "evaluator", lambda command, limit, path: files.publish(path, "".join(json.dumps(row) + "\n" for row in ROWS)))
    output = tmp_path / "select.json"

    def run(judge, **changed):
        monkeypatch.setattr(sq, "judge_spec", lambda spec, **settings: (judge, None))
        return sq.run(SimpleNamespace(**{"judge": "fake", "provider": None, "key_file": None, "device": "gpu", "max_input_tokens": 4096,
                                         "output": str(output), "fresh": False, **changed}))
    with pytest.raises(KeyboardInterrupt):
        run(Interrupting(oracle(ROWS, failing=("threat",))))
    done = len(files.read_jsonl(f"{output}.journal.jsonl")) - 1
    assert 0 < done < len(ROWS)
    resumed = FakeScorer(oracle(ROWS, failing=("threat",)))
    artifact = run(resumed)
    sizes = [len(row["targets"]) for row in ROWS]
    assert len(resumed.requests) == 2 * sum(sizes[done:])
    hard = sum(max(t.values()) == 1 for row in ROWS for t in row["targets"].values())
    for arm in ("select", "checks"):
        members = artifact[arm]["members"]
        assert members["memberships"] == sum(sizes) and members["hard"] == hard and members["failed"] == 2
        assert members["predicted"] == members["raw_correct"] == hard - 2 and members["wrong_accepted"] == 0
        assert members["unresolved"] == 2 and members["soft_or_unlabelled"] == sum(sizes) - hard and members["statuses"]["failed"] == 2
        assert {k: members["raw"][k] for k in ("true_positive", "false_positive", "false_negative", "positive_precision", "positive_recall")} == \
            {"true_positive": 2, "false_positive": 0, "false_negative": 0, "positive_precision": 1.0, "positive_recall": 1.0}
        assert artifact[arm]["exact_set"]["fully_labelled_cases"] == 1 and artifact[arm]["exact_set"]["raw"] == {"eligible": 2, "agree": 2}
    assert artifact["select"]["statuses"] == {"failed": 2, "ok": 1, "uncertain": 1} and artifact["select"]["incomplete_cases"] == 0
    parity = artifact["parity"]
    assert parity["both_predicted"] == parity["raw_agreement"] == sum(sizes) - 2 and parity["regressions"] == parity["fixes"] == 0
    assert parity["p_true_abs_difference"]["max"] == pytest.approx(0, abs=1e-6)
    assert artifact["tasks"]["go-emotions-noul-test-v1"]["select"]["members"]["memberships"] == 6 and artifact["execution"]["completed"] == len(ROWS)
    idle = FakeScorer(oracle(ROWS))
    assert run(idle)["execution"]["completed"] == len(ROWS) and not idle.requests
    with pytest.raises(ConfigError, match="different run"):
        run(FakeScorer(oracle(ROWS)), max_input_tokens=2048)
