from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import sys
import time

from .errors import ConfigError

METRICS = 3
SPARSE_FAMILIES = 10
SPARSE_ACCEPTED = 20
INTERVAL = ("within-pack family-resampling interval: each draw resamples whole canonical families with all their rows; "
            "it does not certify production precision or representative sampling of real customer or equipment populations")


def ratio(k, n):
    return k / n if n else None


COUNTS = ("submitted", "predicted", "raw_correct", "raw_wrong", "accepted", "correct_accepted", "wrong_accepted",
          "correct_rejected", "wrong_rejected")


def correctness(rows) -> dict:
    rows = [(raw, bool(accepted)) for raw, accepted in rows]
    if any(raw is None and accepted for raw, accepted in rows):
        raise ConfigError("an accepted answer needs a prediction")
    predicted = [raw for raw, _ in rows if raw is not None]
    accepted = [raw for raw, chosen in rows if chosen]
    rejected = [raw for raw, chosen in rows if raw is not None and not chosen]
    return rates({"submitted": len(rows), "predicted": len(predicted), "raw_correct": sum(predicted),
                  "raw_wrong": len(predicted) - sum(predicted), "accepted": len(accepted), "correct_accepted": sum(accepted),
                  "wrong_accepted": len(accepted) - sum(accepted), "correct_rejected": sum(rejected),
                  "wrong_rejected": len(rejected) - sum(rejected)})


def rates(counts: dict) -> dict:
    c = counts
    return {"submitted": c["submitted"], "predicted": c["predicted"], "raw_correct": c["raw_correct"], "raw_wrong": c["raw_wrong"],
            "correct_per_submission": ratio(c["raw_correct"], c["submitted"]),
            "accuracy_given_prediction": ratio(c["raw_correct"], c["predicted"]),
            "accepted": c["accepted"], "correct_accepted": c["correct_accepted"], "wrong_accepted": c["wrong_accepted"],
            "coverage": ratio(c["accepted"], c["submitted"]), "precision": ratio(c["correct_accepted"], c["accepted"]),
            "correct_accepted_rate": ratio(c["correct_accepted"], c["submitted"]),
            "correct_rejected": c["correct_rejected"], "wrong_rejected": c["wrong_rejected"]}


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str).encode()).hexdigest()


def comparison_identity(*, task, projection, gold, answers, scope, condition) -> str:
    return identity({"metrics": METRICS, "task": task, "projection": projection, "gold": gold, "answers": answers, "scope": scope, "condition": condition})


def _unique(rows, name):
    keys = Counter(row["key"] for row in rows)
    repeated = sorted(str(k) for k, n in keys.items() if n > 1)
    if repeated:
        raise ConfigError(f"{name} repeats keys {', '.join(repeated[:5])}; each evaluated unit needs one row")
    return {row["key"]: row for row in rows}


def reconcile(expected, rows, name="arm") -> tuple[list[dict], dict]:
    scope, found = _unique(expected, "expected scope"), _unique(rows, name)
    outside = sorted(map(str, set(found) - set(scope)))
    if outside:
        raise ConfigError(f"{name} rows {', '.join(outside[:5])} are outside the frozen expected scope")
    for key, row in found.items():
        want = scope[key]
        if row.get("comparison") != want["comparison"]:
            raise ConfigError(f"{name} {key}: comparison identity differs from the expected scope; different task meaning, input, gold, answers, scope or condition cannot be paired")
        if row.get("gold") != want["gold"]:
            raise ConfigError(f"{name} {key}: gold differs from the expected scope")
    out = [{**found[key], "family": want["family"]} if key in found else
           {"key": key, "family": want["family"], "comparison": want["comparison"], "gold": want["gold"], "outcome": "unavailable",
            "correct": None, "accepted": False} for key, want in scope.items()]
    support = {"expected": len(scope), "submitted": len(found), "unavailable": len(scope) - len(found),
               "predicted": sum(row.get("correct") is not None for row in out), "families": len({row["family"] for row in out})}
    return out, support


def success(rows) -> dict:
    correct = sum(row.get("correct") is True for row in rows)
    predicted = sum(row.get("correct") is not None for row in rows)
    return {"expected": len(rows), "submitted": sum(row.get("outcome") != "unavailable" for row in rows), "predicted": predicted,
            "correct": correct, "correct_per_submission": ratio(correct, len(rows)), "accuracy_given_prediction": ratio(correct, predicted),
            "failed": sum(row.get("outcome") == "failed" for row in rows), "unavailable": sum(row.get("outcome") == "unavailable" for row in rows)}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def resample(rows, statistic, *, seed: int, count: int, level: float = .95) -> dict:
    if type(seed) is not int or type(count) is not int or count < 1:
        raise ConfigError("bootstrap needs an integer seed and a positive integer replicate count")
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    names = sorted(groups)
    point = statistic(rows)
    base = {"method": "family percentile bootstrap", "weighting": "row", "scope": INTERVAL, "seed": seed, "replicates": count, "level": level, "families": len(names),
            "rows": len(rows), "estimate": point}
    if len(names) < 2:
        return {**base, "interval": None, "valid_replicates": 0, "empty_replicates": 0, "support": "insufficient",
                "reason": "fewer than two families; no resampling interval"}
    rng = random.Random(seed)
    values, empty = [], 0
    for _ in range(count):
        value = statistic([row for _ in names for row in groups[names[rng.randrange(len(names))]]])
        if value is None:
            empty += 1
        else:
            values.append(value)
    tail = (1 - level) / 2
    interval = [percentile(values, tail), percentile(values, 1 - tail)] if values else None
    flags = [*(["sparse_families"] if len(names) < SPARSE_FAMILIES else []),
             *(["degenerate_interval"] if values and min(values) == max(values) else [])]
    return {**base, "interval": interval, "valid_replicates": len(values), "empty_replicates": empty,
            "support": "sparse" if flags else "reported", "flags": flags}


def paired(expected, left, right, *, seed: int, count: int, names=("left", "right")) -> dict:
    a, left_support = reconcile(expected, left, names[0])
    b, right_support = reconcile(expected, right, names[1])
    pairs = [{"key": x["key"], "family": x["family"], "left": x.get("correct") is True, "right": y.get("correct") is True,
              "matched": x.get("correct") is not None and y.get("correct") is not None} for x, y in zip(a, b)]
    def difference(rows):
        return ratio(sum(row["right"] for row in rows) - sum(row["left"] for row in rows), len(rows))
    return {"metric": "correct_per_submission difference (right - left)",
            "support": {names[0]: left_support, names[1]: right_support, "matched": sum(row["matched"] for row in pairs),
                        "unmatched": sum(not row["matched"] for row in pairs)},
            names[0]: success(a), names[1]: success(b),
            "fixes": sum(not row["left"] and row["right"] for row in pairs), "regressions": sum(row["left"] and not row["right"] for row in pairs),
            "both_correct": sum(row["left"] and row["right"] for row in pairs), "both_wrong": sum(not row["left"] and not row["right"] for row in pairs),
            "difference": resample(pairs, difference, seed=seed, count=count)}


def _positive(rows):
    tp = sum(gold and value for gold, value in rows)
    fp = sum(not gold and value for gold, value in rows)
    fn = sum(gold and not value for gold, value in rows)
    tn = sum(not gold and not value for gold, value in rows)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "positive_precision": ratio(tp, tp + fp), "positive_recall": ratio(tp, tp + fn), "negative_precision": ratio(tn, tn + fn)}


def membership(rows) -> dict:
    rows = list(rows)
    hard = [row for row in rows if row["gold"] is not None]
    raw = [(row["gold"], row["raw"]) for row in hard if row["raw"] is not None]
    accepted = [(row["gold"], row["value"]) for row in hard if row["accepted"]]
    positives = sum(row["gold"] for row in hard)
    return {"memberships": len(rows), "hard": len(hard), "soft_or_unlabelled": len(rows) - len(hard), "gold_positive": positives,
            "gold_negative": len(hard) - positives, "predicted": len(raw), "accepted": len(accepted),
            "coverage": ratio(len(accepted), len(hard)), "failed": sum(row.get("outcome") == "failed" for row in rows),
            "unresolved": sum(not row["accepted"] for row in hard),
            "raw": _positive(raw), "accepted_outcomes": {**_positive(accepted), "recall_of_gold_positive": ratio(sum(g and v for g, v in accepted), positives)},
            "micro_accepted_precision": ratio(sum(g == v for g, v in accepted), len(accepted))}


SET_FIELDS = ("scope", "gold", "reference_unresolved", "selected", "unresolved", "prohibited")
FAILED = ("failed", "unavailable")


def _set_row(row) -> dict:
    members = {name: list(row[name]) for name in SET_FIELDS}
    for name, ids in members.items():
        if len(set(ids)) != len(ids):
            raise ConfigError(f"set {name} repeats member IDs")
    outside = set().union(*(members[name] for name in SET_FIELDS[1:])) - set(members["scope"])
    if outside:
        raise ConfigError(f"set members {sorted(map(str, outside))} are outside the declared scope")
    if set(members["selected"]) & set(members["unresolved"]):
        raise ConfigError("a set member cannot be both selected and unresolved")
    if set(members["gold"]) & set(members["reference_unresolved"]):
        raise ConfigError("a reference member cannot be both a known positive and unresolved")
    if set(members["prohibited"]) & (set(members["gold"]) | set(members["reference_unresolved"])):
        raise ConfigError("a prohibited member must be a known reference negative")
    if type(row["prediction_complete"]) is not bool or type(row["complete_gold"]) is not bool:
        raise ConfigError("sets need explicit boolean prediction_complete and complete_gold")
    if row["complete_gold"] != (not members["reference_unresolved"]):
        raise ConfigError("complete_gold holds exactly when no scoped reference membership is unresolved")
    succeeded = row["outcome"] not in FAILED
    if row["prediction_complete"] and (members["unresolved"] or not succeeded):
        raise ConfigError("a complete prediction needs a successful operation and no unresolved members")
    return {**members, "complete_gold": row["complete_gold"], "prediction_complete": row["prediction_complete"], "outcome": row["outcome"],
            "succeeded": succeeded}


def exact_sets(rows) -> dict:
    rows = [_set_row(row) for row in rows]
    complete = [row for row in rows if row["complete_gold"]]
    exact = sum(row["succeeded"] and row["prediction_complete"] and set(row["selected"]) == set(row["gold"]) for row in complete)
    return {"metric": "exact_sets", "version": METRICS, "situations": len(rows), "gold_complete": len(complete),
            "incomplete_reference": len(rows) - len(complete), "prediction_complete": sum(row["prediction_complete"] for row in complete),
            "failed": sum(not row["succeeded"] for row in complete), "exact": exact, "exact_rate": ratio(exact, len(complete)),
            "denominator": "gold-complete situations; failed, partial, unresolved or absent predictions count as not exact; "
                           "incomplete-reference situations are reported, never scored"}


def action_sets(rows) -> dict:
    rows = [_set_row(row) for row in rows]
    known = [(k in row["gold"], k in row["selected"]) for row in rows for k in row["scope"] if k not in row["reference_unresolved"]]
    counts = _positive(known)
    return {"metric": "action_sets", "version": METRICS, "frames": len(rows), "known_reference_support": len(known),
            "precision": counts["positive_precision"], "recall": counts["positive_recall"], "known_reference": counts,
            "reference_unresolved": sum(len(row["reference_unresolved"]) for row in rows),
            "unassessed_selected": sum(len(set(row["selected"]) & set(row["reference_unresolved"])) for row in rows),
            "prohibited_selected": sum(len(set(row["selected"]) & set(row["prohibited"])) for row in rows),
            "frames_with_prohibited": sum(bool(set(row["selected"]) & set(row["prohibited"])) for row in rows),
            "required_omissions": counts["false_negative"], "prediction_unresolved": sum(len(row["unresolved"]) for row in rows),
            "failed": sum(not row["succeeded"] for row in rows),
            "denominator": "known-reference memberships; selections of unresolved reference memberships are unassessed, not false positives"}


def ordered(rows) -> dict:
    rows = list(rows)
    scored = [abs(row["predicted"] - row["gold"]) for row in rows if row["predicted"] is not None]
    return {"submitted": len(rows), "predicted": len(scored), "unresolved": len(rows) - len(scored),
            "exact": sum(d == 0 for d in scored), "distances": dict(sorted(Counter(scored).items())),
            "mean_absolute_distance": ratio(sum(scored), len(scored))}


def confusion(rows) -> dict:
    rows = list(rows)
    table = Counter((row["gold"], row["predicted"]) for row in rows if row["predicted"] is not None)
    return {"submitted": len(rows), "predicted": sum(table.values()), "correct": sum(n for (g, p), n in table.items() if g == p),
            "unresolved": sum(row["predicted"] is None and row.get("outcome") != "failed" for row in rows),
            "failed": sum(row.get("outcome") == "failed" for row in rows),
            "confusion": [{"gold": g, "predicted": p, "count": n} for (g, p), n in sorted(table.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))]}


def precision_report(rows, *, seed: int, count: int, target: float | None = None, sampling: str | None = None) -> dict:
    rows = [row for row in rows]
    accepted = [row for row in rows if row["accepted"]]
    def precision(sample):
        chosen = [row["correct"] for row in sample if row["accepted"]]
        return ratio(sum(chosen), len(chosen))
    estimate = precision(rows)
    interval = resample(rows, precision, seed=seed, count=count)
    flags = [*interval.get("flags", ()), *(["sparse_accepted"] if len(accepted) < SPARSE_ACCEPTED else []),
             *(["all_identical_outcomes"] if accepted and len({row["correct"] for row in accepted}) == 1 else []),
             *(["sampling_unidentified"] if sampling is None else [])]
    return {"accepted": len(accepted), "correct_accepted": sum(row["correct"] for row in accepted),
            "accepted_families": len({row["family"] for row in accepted}), "precision": estimate, "interval": interval,
            "flags": sorted(set(flags)), "sampling": sampling,
            "target": target, "observed_target": None if target is None or estimate is None else "met" if estimate >= target else "missed",
            "evidence": "insufficient" if estimate is None or flags else "reported",
            "scope": "empirical precision of a frozen threshold on this sample; not a population guarantee"}


def tally(rows) -> dict:
    rows = list(rows)
    return {"submitted": len(rows), "completed": len(rows), "raw_correct": sum(row.get("correct") is True for row in rows),
            "accepted": sum(bool(row.get("accepted")) for row in rows),
            "correct_accepted": sum(bool(row.get("accepted")) and row.get("correct") is True for row in rows),
            "failed": sum(row.get("outcome") == "failed" for row in rows)}


class Progress:
    FIELDS = ("submitted", "completed", "raw_correct", "accepted", "correct_accepted", "failed")

    def __init__(self, label, total, *, unit, resumed=(), every_rows=25, every_seconds=30., stream=None, clock=time.monotonic):
        self.counts = tally(resumed)
        self.label, self.total, self.unit, self.resumed, self.fresh = label, total, unit, self.counts["completed"], 0
        self.every_rows, self.every_seconds, self.stream, self.clock = every_rows, every_seconds, stream, clock
        self.started = self.last = clock()
        self.since = 0

    def line(self, final=False):
        c = self.counts
        return (f"{self.label} [{self.unit}] {'final' if final else 'running'}: completed {c['completed']}/{self.total} "
                f"(fresh {self.fresh}, resumed {self.resumed}) submitted {c['submitted']} raw_correct {c['raw_correct']} "
                f"accepted {c['accepted']} correct_accepted {c['correct_accepted']} failed {c['failed']} "
                f"elapsed {self.clock() - self.started:.1f}s")

    def emit(self, final=False):
        print(self.line(final), file=self.stream or sys.stderr, flush=True)
        self.last, self.since = self.clock(), 0

    def add(self, **counts):
        unknown = set(counts) - set(self.FIELDS)
        if unknown:
            raise ConfigError(f"unknown progress counters {sorted(unknown)}")
        for key, value in counts.items():
            self.counts[key] += int(value)
        self.fresh += counts.get("completed", 0)
        self.since += counts.get("completed", 0)
        if self.since >= self.every_rows or self.clock() - self.last >= self.every_seconds:
            self.emit()

    def record(self, rows):
        self.add(**tally(rows))

    def finish(self):
        self.emit(final=True)
        return {"unit": self.unit, "total": self.total, "resumed": self.resumed, "fresh": self.fresh, **self.counts, "elapsed_s": self.clock() - self.started}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if type(k) is not int or type(n) is not int or not 0 <= k <= n:
        raise ConfigError("Wilson interval requires integer counts 0 <= successes <= trials")
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [0.0 if k == 0 else max(0.0, centre - half), 1.0 if k == n else min(1.0, centre + half)]


def auc(scores, labels):
    if len(scores) != len(labels):
        raise ConfigError('AUC scores/labels length mismatch')
    ordered_ = sorted(zip(scores, labels), key=lambda row: row[0])
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    wins, before, start = 0., 0, 0
    while start < len(ordered_):
        end = start + 1
        while end < len(ordered_) and ordered_[end][0] == ordered_[start][0]:
            end += 1
        pos = sum(bool(label) for _, label in ordered_[start:end])
        neg = end - start - pos
        wins += pos * before + .5 * pos * neg
        before += neg
        start = end
    return round(wins / (positives * negatives), 3)
