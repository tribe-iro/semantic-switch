from __future__ import annotations

import argparse
from bisect import bisect
from collections import Counter

from semantic_switch import files
from semantic_switch.judgment import MIN_MASS
from bench.external_primitives import CATEGORY, agreement, aligned, counts, member, paired, recorded, shared_inputs, top
from bench.latency import _summary

THRESHOLDS = (.5, .9, .95, .99, .999)
GROUPS = ("all", "noul", "choice", "score", "in_scope", "out_of_scope",
          *(f"{task}/{scope}" for task in ("noul", "choice", "score") for scope in ("in_scope", "out_of_scope")))
LENGTHS = (1000, 4000, 16000)
BOUNDARY = (f"cases recorded by every arm within the first 25 cases of every {CATEGORY} benchmark of S1MB, a community benchmark whose overlap with "
            "any arm's training data is unknown; pooled judgment counts, not benchmark means, leaderboard scores or independent samples. Jev is a "
            "focused-model reference of undisclosed size; one request per case does not establish its internal parallelism or serving hardware, "
            "and its outputs are kept only as benchmark artifacts")


def accepted_at(question, theta):
    probabilities = question["probabilities"]
    answer = top(probabilities)
    return answer if probabilities[answer] >= theta and (question["mass"] is None or question["mass"] >= MIN_MASS) else None


def curve(hard, column):
    rows = {}
    for theta in THRESHOLDS:
        answer = lambda q, theta=theta: accepted_at(q, theta)
        found = counts(hard, column, answer)
        rows[str(theta)] = {**found, "coverage": found["accepted"] / len(hard) if hard else None,
                            "benchmarks": len({i[0] for i in hard if answer(i[column]) is not None})}
    return rows


def worked(row):
    return bool(row["requests"]) if "requests" in row else any(p["status"] not in ("unsupported", "needs_input") for p in row["predictions"])


def timing(rows):
    requests = [request for row in rows for request in row.get("requests", ())]
    usage = Counter()
    for request in requests:
        usage.update({name: value for name, value in (request["usage"] or {}).items() if type(value) is int})
    return {"cases": len(rows), "questions": sum(len(row["predictions"]) for row in rows), "case_ms": _summary(rows),
            "non_failed_case_ms": _summary([row for row in rows if all(p["status"] != "failed" for p in row["predictions"])]),
            "model_work_case_ms": _summary([row for row in rows if worked(row)]),
            **({"requests": len(requests), "failed_requests": sum(r["completion"] == "failed" for r in requests),
                "request_ms": _summary([{"ms": r["elapsed_ms"]} for r in requests]), "usage": dict(usage),
                "returned_models": dict(Counter(r["returned_model"] for r in requests))} if any("requests" in row for row in rows) else {})}


def strata(rows, key):
    found = {}
    for row in rows:
        found.setdefault(key(row), []).append(row)
    return {str(name): timing(group) for name, group in sorted(found.items())}


def analyze(arms):
    names = list(arms)
    inputs = shared_inputs([arms[name][0] for name in names])
    keys, items = aligned(*(arms[name][1] for name in names))
    rows = {name: {key: arms[name][1][key] for key in keys} for name in names}
    columns = {name: 3 + index for index, name in enumerate(names)}
    joint = [item for item in items if all(item[column]["probabilities"] is not None for column in columns.values())]
    hard, soft = [item for item in joint if max(item[2].values()) == 1], [item for item in joint if max(item[2].values()) < 1]
    return {"schema": 2, "boundary": BOUNDARY, "inputs": inputs,
            "configuration_digests": {name: arms[name][0]["digest"] for name in names},
            "matched_cases": len(keys), "judgments": len(items), "jointly_scored": len(joint), "jointly_scored_hard": len(hard),
            "availability": {name: dict(sorted(Counter(item[column]["status"] for item in items).items())) for name, column in columns.items()},
            "unscored": {name: sum(item[column]["probabilities"] is None for item in items) for name, column in columns.items()},
            "acceptance_replay_mismatches_at_0.9": {name: sum(accepted_at(item[column], .9) != item[column]["accepted"] for item in joint)
                                                    for name, column in columns.items()},
            "hard_label": {group: {"items": len(values), "benchmarks": len({i[0] for i in values}),
                                   **{name: {"raw_correct": sum(i[2][top(i[column]["probabilities"])] == 1 for i in values), "curve": curve(values, column)}
                                      for name, column in columns.items()}}
                           for group in GROUPS for values in [[item for item in hard if member(item, group)]]},
            "soft_target": {name: agreement([(item[2], accepted_at(item[column], .9)) for item in soft]) for name, column in columns.items()},
            "paired": {f"{a}->{b}": paired(rows[a], rows[b]) for i, a in enumerate(names) for b in names[i + 1:]},
            "timing": {name: {"all": timing(list(rows[name].values())),
                              "question_count": strata(rows[name].values(), lambda row: len(row["predictions"])),
                              "input_characters_from": strata(rows[name].values(), lambda row: (0, *LENGTHS)[bisect(LENGTHS, row["input_characters"])])
                              if all("input_characters" in row for row in rows[name].values()) else None}
                       for name in names}}


def compare(paths, output):
    report = {**analyze({name: recorded(path) for name, path in paths.items()}), "artifacts": {name: str(path) for name, path in paths.items()}}
    files.publish(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model-free matched comparison of three saved S1MB arms")
    parser.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "ARTIFACT"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.arm) != 3:
        parser.error("three arms are required")
    compare(dict(args.arm), args.output)
