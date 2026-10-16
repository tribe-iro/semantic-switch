from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping
import math
import numpy as np
import sys
import time

from .compiler import digest
from .errors import ConfigError
from .judgment import ESCAPE, TOLERANCE

METRICS = 5
BOOTSTRAP = "family-pcg64-int64-v1"
BOOTSTRAP_BYTES = 8 << 20
SPARSE_FAMILIES = 10
INTERVAL = ("within-pack family-resampling interval: each draw resamples whole canonical families with all their rows; "
            "it does not certify production precision or representative sampling of real customer or equipment populations")


def ratio(k, n):
    return k / n if n else None


COUNTS = ("submitted", "predicted", "raw_correct", "raw_wrong", "accepted", "correct_accepted", "wrong_accepted",
          "correct_rejected", "wrong_rejected")
ACCEPTED = ("ok", "none")
ESCAPES = ("none", "unstated")
UNAVAILABLE = ("failed", "unexamined", "unavailable")


def semantic_answer(prediction, keys=None):
    if prediction is None:
        return None
    kind, key = (prediction["kind"], prediction["key"]) if isinstance(prediction, Mapping) else (prediction.kind, prediction.key)
    if kind != "value" and kind not in ESCAPES:
        raise ConfigError(f"prediction kind {kind!r} is not value, none or unstated")
    found = key if kind == "value" else kind
    if keys is not None and found not in keys:
        raise ConfigError(f"predicted {found!r} is not one of the judgment's semantic keys {list(keys)}")
    return found


def scored_answer(prediction, gold, outcome, keys=None) -> dict:
    answer = None if outcome in UNAVAILABLE else semantic_answer(prediction, keys)
    return {"answer": answer, "correct": None if answer is None or gold is None else answer == gold,
            "accepted": answer is not None and outcome in ACCEPTED}


class Counts:
    FIELDS = (*COUNTS, "failed", "unavailable")

    def __init__(self, rows=(), *, strict=True):
        self.strict, self.counts = strict, dict.fromkeys(self.FIELDS, 0)
        for row in rows:
            self.add(*row)

    def add(self, correct, accepted, outcome=None):
        accepted, c = bool(accepted), self.counts
        if correct is None and accepted and self.strict:
            raise ConfigError("an accepted answer needs a prediction")
        c["submitted"] += 1
        c["failed"] += outcome == "failed"
        c["unavailable"] += outcome == "unavailable"
        if correct is not None:
            c["predicted"] += 1
            c["raw_correct" if correct else "raw_wrong"] += 1
            if not accepted:
                c["correct_rejected" if correct else "wrong_rejected"] += 1
        if accepted:
            c["accepted"] += 1
            if correct is not None:
                c["correct_accepted" if correct else "wrong_accepted"] += 1
        return self

    def rates(self) -> dict:
        return rates(self.counts)


def correctness(rows) -> dict:
    return Counts(rows).rates()


def rates(counts: dict) -> dict:
    c = counts
    return {"submitted": c["submitted"], "predicted": c["predicted"], "raw_correct": c["raw_correct"], "raw_wrong": c["raw_wrong"],
            "correct_per_submission": ratio(c["raw_correct"], c["submitted"]),
            "accuracy_given_prediction": ratio(c["raw_correct"], c["predicted"]),
            "accepted": c["accepted"], "correct_accepted": c["correct_accepted"], "wrong_accepted": c["wrong_accepted"],
            "coverage": ratio(c["accepted"], c["submitted"]), "precision": ratio(c["correct_accepted"], c["accepted"]),
            "correct_accepted_rate": ratio(c["correct_accepted"], c["submitted"]),
            "correct_rejected": c["correct_rejected"], "wrong_rejected": c["wrong_rejected"]}


def prediction_report(rows):
    rows = list(rows)
    def reduce(group):
        out = correctness((row.get('correct'), row.get('accepted', False) and row.get('gold') is not None, row.get('outcome')) for row in group)
        if group and all(row.get('acceptance') == 'unassessed' for row in group):
            out.update({key: None for key in ('accepted', 'correct_accepted', 'wrong_accepted', 'coverage', 'precision',
                                             'correct_accepted_rate', 'correct_rejected', 'wrong_rejected')})
            out['acceptance'] = 'unassessed'
        return out
    groups, pairs = {}, {}
    for row in rows:
        for tag in row.get('tags', ()):
            groups.setdefault(tag, []).append(row)
        for pair in row.get('pairs', ()):
            pairs.setdefault((row.get('target'), pair), []).append(row)
    return {**reduce(rows), 'unassessed_reference': sum(row.get('gold') is None for row in rows),
            'accepted_unassessed': sum(row.get('gold') is None and row.get('accepted', False) for row in rows), 'failed': sum(row.get('outcome') == 'failed' for row in rows),
            'unavailable': sum(row.get('outcome') in ('unavailable', 'unexamined') for row in rows),
            'wrong_accepted_per_submission': None if rows and all(row.get('acceptance') == 'unassessed' for row in rows) else
                ratio(sum(row.get('accepted', False) and row.get('correct') is False for row in rows), len(rows)),
            'per_class': per_class((row['target'], row['options'], row['gold'], row.get('answer'), row.get('correct')) for row in rows if row.get('gold') is not None),
            'behaviors': {tag: reduce(group) for tag, group in sorted(groups.items())},
            'pairs': {'assessed': sum(len(group) >= 2 for group in pairs.values()),
                      'all_correct': sum(len(group) >= 2 and all(row.get('correct') is True for row in group) for group in pairs.values())}}


def zero_failure_bound(families, *, level=.95):
    if type(families) is not int or families < 0 or not 0 < level < 1:
        raise ConfigError('zero-failure bound needs a nonnegative family count and a confidence level in (0, 1)')
    return {'level': level, 'families': families, 'upper': -math.expm1(math.log1p(-level) / families) if families else None,
            'scope': 'zero observed failures in independently sampled family-level binary events; not per-answer or targeted-audit risk'}


def comparison_identity(*, task, projection, gold, answers, scope, condition) -> str:
    return digest({"metrics": METRICS, "task": task, "projection": projection, "gold": gold, "answers": answers, "scope": scope, "condition": condition})


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
    c = Counts(((row.get("correct"), False, row.get("outcome")) for row in rows), strict=False).counts
    return {"expected": c["submitted"], "submitted": c["submitted"] - c["unavailable"], "predicted": c["predicted"], "correct": c["raw_correct"],
            "correct_per_submission": ratio(c["raw_correct"], c["submitted"]), "accuracy_given_prediction": ratio(c["raw_correct"], c["predicted"]),
            "failed": c["failed"], "unavailable": c["unavailable"]}


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_totals(numerators, denominators, draws):
    scratch = np.empty(draws.shape, dtype=np.result_type(numerators, denominators))
    np.take(numerators, draws, out=scratch, mode='clip')
    top = scratch.sum(axis=1)
    np.take(denominators, draws, out=scratch, mode='clip')
    return top, scratch.sum(axis=1)


def resample(rows, measure, *, seed: int, count: int, level: float = .95) -> dict:
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1 or not isinstance(level, (float, int)) or not math.isfinite(level) or not 0 < level < 1:
        raise ConfigError("bootstrap needs a nonnegative integer seed, positive replicate count and finite confidence level between zero and one")
    totals, size = {}, 0
    for row in rows:
        numerator, denominator = measure(row)
        found = totals.setdefault(row["family"], [0, 0])
        found[0] += numerator
        found[1] += denominator
        size += 1
    names = sorted(totals)
    numerators, denominators = [totals[name][0] for name in names], [totals[name][1] for name in names]
    point = ratio(sum(numerators), sum(denominators))
    base = {"method": "family percentile bootstrap", "implementation": BOOTSTRAP, "weighting": "row", "scope": INTERVAL, "seed": seed, "replicates": count, "level": level, "families": len(names),
            "rows": size, "estimate": point}
    if len(names) < 2:
        return {**base, "interval": None, "valid_replicates": 0, "empty_replicates": 0, "support": "insufficient",
                "reason": "fewer than two families; no resampling interval"}
    rng, width = np.random.Generator(np.random.PCG64(seed)), len(names)
    numerators, denominators = np.asarray(numerators), np.asarray(denominators)
    chunk = max(1, BOOTSTRAP_BYTES // (width * 16))
    values, empty = [], 0
    for start in range(0, count, chunk):
        draws = rng.integers(width, size=(min(chunk, count - start), width), dtype=np.int64)
        top, bottom = bootstrap_totals(numerators, denominators, draws)
        valid = bottom != 0
        empty += int((~valid).sum())
        values.extend((top[valid] / bottom[valid]).tolist())
        del draws, top, bottom, valid
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
    return {"metric": "correct_per_submission difference (right - left)",
            "support": {names[0]: left_support, names[1]: right_support, "matched": sum(row["matched"] for row in pairs),
                        "unmatched": sum(not row["matched"] for row in pairs)},
            names[0]: success(a), names[1]: success(b),
            "fixes": sum(not row["left"] and row["right"] for row in pairs), "regressions": sum(row["left"] and not row["right"] for row in pairs),
            "both_correct": sum(row["left"] and row["right"] for row in pairs), "both_wrong": sum(not row["left"] and not row["right"] for row in pairs),
            "difference": resample(pairs, lambda row: (row["right"] - row["left"], 1), seed=seed, count=count)}


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
    abstained = Counter(row["gold"] for row in rows if row["predicted"] is None and row.get("outcome") != "failed")
    return {"submitted": len(rows), "predicted": sum(table.values()), "correct": sum(n for (g, p), n in table.items() if g == p),
            "unresolved": sum(abstained.values()), "failed": sum(row.get("outcome") == "failed" for row in rows),
            "confusion": [{"gold": g, "predicted": p, "count": n} for (g, p), n in sorted(table.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))],
            "abstained": [{"gold": g, "count": n} for g, n in sorted(abstained.items(), key=lambda item: str(item[0]))]}


def per_class(rows) -> dict:
    cells = {}
    for key, classes, gold, predicted, correct in rows:
        found = cells.get(key)
        if found is None:
            found = cells[key] = {name: [0, 0, 0] for name in classes}
        found[gold][0] += 1
        found[gold][2] += bool(correct)
        if predicted in found:
            found[predicted][1] += 1
    return {key: {name: {"gold": g, "predicted": p, "correct": c, "recall": c / g if g else None} for name, (g, p, c) in found.items()}
            for key, found in cells.items()}


def recall_floor(correct, references, minimum) -> dict:
    value = correct / references if references else None
    return {"correct": correct, "recall": value, "required": minimum,
            "state": "unassessed" if value is None else "met" if value >= minimum else "unmet"}


def recall_counts(rows) -> dict:
    cells, families = {}, {}
    for scope, name, family, correct, accepted in rows:
        key = (*scope, name)
        cell = cells.setdefault(key, {"references": 0, "raw": 0, "operational": 0})
        cell["references"] += 1
        cell["raw"] += bool(correct)
        cell["operational"] += bool(correct) and bool(accepted)
        families.setdefault(key, set()).add(family)
    return {key: {**cell, "families": len(families[key])} for key, cell in cells.items()}


def regressions(current: dict, reference: dict) -> list:
    return sorted((scope, name) for scope, classes in current.items() for name, counts in classes.items()
                  if (before := reference.get(scope, {}).get(name)) is not None and before["gold"] == counts["gold"] and counts["correct"] < before["correct"])


def class_changes(current: dict, history: list) -> dict:
    def entry(scope, name, before, step):
        counts = current[scope][name]
        return {"scope": scope, "class": name, "gold": counts["gold"], "correct": counts["correct"],
                "reference_correct": before, "reference_step": step}
    out = {"base": [], "previous": [], "best": []}
    if not history:
        return out
    base, previous = history[0], history[-1]
    for side, point in (("base", base), ("previous", previous)):
        out[side] = [entry(scope, name, point["per_class"][scope][name]["correct"], point["step"]) for scope, name in regressions(current, point["per_class"])]
    best = {}
    for point in history:
        for scope, classes in point["per_class"].items():
            for name, counts in classes.items():
                if counts["gold"] == current.get(scope, {}).get(name, {}).get("gold") and counts["correct"] > best.get((scope, name), (-1, None))[0]:
                    best[(scope, name)] = (counts["correct"], point["step"])
    out["best"] = [entry(scope, name, correct, step) for (scope, name), (correct, step) in sorted(best.items(), key=lambda item: str(item[0]))
                   if current[scope][name]["correct"] < correct]
    return out


def nested(cells: dict) -> dict:
    out = {}
    for (task, target), counts in cells.items():
        out.setdefault(task, {})[target] = counts
    return out


class Progress:
    FIELDS = ("submitted", "raw_correct", "accepted", "correct_accepted", "failed")

    def __init__(self, label, total, *, unit, resumed=(), every_rows=25, every_seconds=30., stream=None, clock=time.monotonic):
        self.reducer = Counts(strict=False)
        for row in resumed:
            self._row(row)
        self.label, self.total, self.unit, self.resumed, self.fresh = label, total, unit, self.reducer.counts["submitted"], 0
        self.every_rows, self.every_seconds, self.stream, self.clock = every_rows, every_seconds, stream, clock
        self.started = self.last = clock()
        self.since = 0

    def _row(self, row):
        self.reducer.add(row.get("correct"), row.get("accepted"), row.get("outcome"))

    @property
    def counts(self) -> dict:
        c = self.reducer.counts
        return {"submitted": c["submitted"], "completed": c["submitted"], **{key: c[key] for key in self.FIELDS[1:]}}

    def line(self, final=False):
        c = self.counts
        return (f"{self.label} [{self.unit}] {'final' if final else 'running'}: completed {c['completed']}/{self.total} "
                f"(fresh {self.fresh}, resumed {self.resumed}) submitted {c['submitted']} raw_correct {c['raw_correct']} "
                f"accepted {c['accepted']} correct_accepted {c['correct_accepted']} failed {c['failed']} "
                f"elapsed {self.clock() - self.started:.1f}s")

    def emit(self, final=False):
        print(self.line(final), file=self.stream or sys.stderr, flush=True)
        self.last, self.since = self.clock(), 0

    def _advance(self, completed):
        self.fresh += completed
        self.since += completed
        if self.since >= self.every_rows or self.clock() - self.last >= self.every_seconds:
            self.emit()

    def add(self, **counts):
        unknown = set(counts) - set(self.FIELDS)
        if unknown:
            raise ConfigError(f"unknown progress counters {sorted(unknown)}")
        for key, value in counts.items():
            self.reducer.counts[key] += int(value)
        self._advance(counts.get("submitted", 0))

    def record(self, rows):
        before = self.reducer.counts["submitted"]
        for row in rows:
            self._row(row)
        self._advance(self.reducer.counts["submitted"] - before)

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


PROBABILITY = 1
BINS = 10
EDGES = tuple(k / BINS for k in range(BINS + 1))
SPARSE_BIN = 10
OPTION_SOFTMAX = "option-softmax@1"
COMPLETE = (OPTION_SOFTMAX, "renormalize-within-rounding-bound@1")
REASONS = ("failed", "missing", "unexamined", "not_judge", "staged", "no_distribution", "unknown_gold", "normalization", "invalid",
           "partial_options", "gold_outside_options", "incomplete")
CONVENTIONS = {"revision": PROBABILITY, "tolerance": TOLERANCE,
               "brier": "sum over every declared option of (p_k - 1[k = gold])**2; range [0, 2] at every width, never divided by the width",
               "nll": "-ln p(gold) over the declared options; zero gold probability is an infinite-loss event reported as nonfinite, never floored",
               "unconditioned_nll": "-ln(p(gold) * answer mass) for option-conditioned distributions whose computation supplies answer mass",
               "top": "first maximal option in declared order",
               "complete": f"every declared option has a finite probability in [0, 1], the total is within {TOLERANCE} of one and the normalization "
                           f"is one of {list(COMPLETE)}; nothing is renormalized or reconstructed from lower bounds, witnesses or composed output",
               "scope": "final: one full-stage presentation produced the deployed distribution; staged final distributions are unassessed and "
                        "their valid primitive stages are reported separately per stage",
               "reliability": "top-answer probability in ten equal-width bins [k/10, (k+1)/10), the last bin closed so probability 1 is included; "
                              "observed correctness is raw top-answer agreement with the declared reference",
               "macro": "unweighted mean over the fixed tasks that have scored rows; describes these tasks, no unseen-task interval",
               "micro": "every submitted target observation weighted equally",
               "accepted": "correctness of policy-accepted answers; a separate population from model probability reliability",
               "reference": "agreement with the declared reference; reviewed correctness only under a reviewed reference"}


def answer_key(value):
    return None if value is None else "true" if value is True else "false" if value is False else value


def observed(options, probabilities, *, stage, normalization, status, source, kind=None, mass=None, accepted=None, score=None, stages=(),
             presentations=(), models=(), execution=None) -> dict:
    return {"revision": PROBABILITY, "kind": kind, "options": list(options), "probabilities": None if probabilities is None else list(probabilities),
            "stage": stage, "normalization": normalization, "mass": mass, "status": status, "source": source, "accepted": accepted, "score": score,
            "stages": list(stages), "presentations": list(presentations), "models": list(models), "execution": execution}


def primitive(options, probabilities, *, stage, normalization, mass=None) -> dict:
    return {"stage": stage, "options": list(options), "probabilities": list(probabilities), "normalization": normalization, "mass": mass}


def observation(result, judgment, *, normalization, execution=None) -> dict:
    if result is None:
        return observed(judgment.keys, None, kind=judgment.kind, stage=None, normalization=normalization, status="unexamined", source=None, execution=execution)
    scores, shown = result.get("scores") or {}, result.get("presentations") or ()
    stages = [p["stage"] for p in shown]
    final = stages == ["full"]
    dist = [scores.get(k) for k in judgment.keys] if scores else None
    usable = dist is not None and bool(stages) and stages[-1] != "mentions" and not final
    return observed(judgment.keys, dist if final else None, kind=judgment.kind, stage="final" if final else "staged" if stages else None,
                    normalization=normalization,
                    status=result["status"], source=result.get("source", "judge"), mass=result.get("mass"),
                    accepted=answer_key(result.get("value")) if result["status"] == "ok" else None,
                    score=result["score"] if "score" in result else result.get("p_true"),
                    stages=[primitive(judgment.keys, dist, stage=stages[-1], normalization=normalization, mass=result.get("mass"))] if usable else (),
                    presentations=shown, models=result.get("observed_models", ()), execution=execution)


def _probability(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _distribution(o, gold):
    if gold is None:
        return "unknown_gold"
    if o["normalization"] not in COMPLETE:
        return "normalization"
    p, options, mass = o["probabilities"], o["options"], o["mass"]
    if any(v is not None and not _probability(v) for v in p) or mass is not None and not _probability(mass):
        return "invalid"
    if None in p:
        return "partial_options"
    if gold not in options:
        return "gold_outside_options"
    if abs(math.fsum(p) - 1) > TOLERANCE:
        return "incomplete"
    g = options.index(gold)
    top = max(range(len(p)), key=p.__getitem__)
    joint = None if mass is None or o["normalization"] != OPTION_SOFTMAX else p[g] * mass
    return {"brier": math.fsum((v - (k == g)) ** 2 for k, v in enumerate(p)), "nll": 0. - math.log(p[g]) if p[g] > 0 else math.inf,
            "unconditioned_nll": None if joint is None else 0. - math.log(joint) if joint > 0 else math.inf,
            "top": p[top], "correct": top == g, "zero_gold": p[g] == 0}


def _assess(entry):
    o = entry["observation"]
    if entry.get("outcome") == "failed" or o is not None and o["status"] == "failed":
        return "failed"
    if o is None:
        return "missing"
    if o["status"] == "unexamined":
        return "unexamined"
    if o["source"] not in (None, "judge"):
        return "not_judge"
    if o["stage"] == "staged":
        return "staged"
    if o["probabilities"] is None:
        return "no_distribution"
    return _distribution(o, entry["gold"])


def nonfinite_mean(values) -> dict:
    values = list(values)
    infinite = sum(v == math.inf for v in values)
    finite = [v for v in values if v != math.inf]
    mean = ratio(math.fsum(finite), len(finite))
    return {"value": None if infinite else mean, "nonfinite": "+inf" if infinite else None, "infinite": infinite, "finite_mean": mean, "rows": len(values)}


def reliability(points) -> dict:
    bins = [[] for _ in range(BINS)]
    for probability, correct, family in points:
        bins[min(bisect_right(EDGES, probability) - 1, BINS - 1)].append((probability, correct, family))
    total = sum(map(len, bins))
    table = [{"lower": EDGES[k], "upper": EDGES[k + 1], "upper_included": k == BINS - 1, "count": len(rows), "families": len({f for _, _, f in rows}),
              "mean_predicted": ratio(math.fsum(p for p, _, _ in rows), len(rows)), "observed_correct": ratio(sum(c for _, c, _ in rows), len(rows)),
              "support": "empty" if not rows else "sparse" if len(rows) < SPARSE_BIN else "reported"} for k, rows in enumerate(bins)]
    ece = math.fsum(row["count"] / total * abs(row["observed_correct"] - row["mean_predicted"]) for row in table if row["count"]) if total else None
    return {"bins": BINS, "edges": list(EDGES), "rows": total, "table": table,
            "ece": {"value": ece, "edges": "the ten equal-width bins above", "scope": "descriptive expected calibration error; not a gate"}}


def _population(pairs) -> dict:
    scored = [(entry, value) for entry, value in pairs if isinstance(value, dict)]
    reasons = Counter(value for _, value in pairs if isinstance(value, str))
    conditioned = [value["unconditioned_nll"] for _, value in scored if value["unconditioned_nll"] is not None]
    accepted = [entry for entry, _ in pairs if entry["observation"] is not None and entry["observation"]["source"] == "judge"
                and entry["observation"]["accepted"] is not None and entry["gold"] is not None]
    correct = sum(entry["observation"]["accepted"] == entry["gold"] for entry in accepted)
    return {"submitted": len(pairs), "scored": len(scored), "unscored": {reason: reasons[reason] for reason in REASONS},
            "targets": len({(entry["task"], entry["target"]) for entry, _ in pairs}), "families": len({entry["family"] for entry, _ in pairs}),
            "scored_families": len({entry["family"] for entry, _ in scored}),
            "brier": nonfinite_mean(value["brier"] for _, value in scored), "nll": nonfinite_mean(value["nll"] for _, value in scored),
            "unconditioned_nll": {**nonfinite_mean(conditioned), "unavailable": len(scored) - len(conditioned)},
            "zero_gold": sum(value["zero_gold"] for _, value in scored),
            "reliability": reliability([(value["top"], value["correct"], entry["family"]) for entry, value in scored]),
            "accepted": {"accepted": len(accepted), "correct": correct, "precision": ratio(correct, len(accepted))}}


def _grouped(pairs, field) -> dict:
    groups = {}
    for entry, value in pairs:
        groups.setdefault(str(entry[field]), []).append((entry, value))
    return {name: _population(group) for name, group in sorted(groups.items())}


def _macro(tasks) -> dict:
    contributing = [task for task in tasks.values() if task["scored"]]
    def mean(name):
        return nonfinite_mean(math.inf if task[name]["nonfinite"] else task[name]["value"] for task in contributing)
    return {"tasks": len(tasks), "contributing": len(contributing), "brier": mean("brier"), "nll": mean("nll")}


def _escaped(gold, options):
    return gold if gold in options or ESCAPE not in options else ESCAPE


def probabilities(entries, *, targets=True) -> dict:
    pairs = [(entry, _assess(entry)) for entry in entries]
    stages, tasks = {}, {}
    for entry, value in pairs:
        tasks.setdefault(str(entry["task"]), []).append((entry, value))
        for stage in entry["observation"]["stages"] if value == "staged" else ():
            stages.setdefault(stage["stage"], []).append((entry, _distribution(stage, _escaped(entry["gold"], stage["options"]))))
    summaries = {name: _population(group) for name, group in sorted(tasks.items())}
    return {"metric": "probabilities", "protocol": CONVENTIONS, "micro": _population(pairs), "macro": _macro(summaries), "tasks": summaries,
            **({"targets": {name: _grouped(group, "target") for name, group in sorted(tasks.items())}} if targets else {}),
            "kinds": _grouped(pairs, "kind"), "stages": {name: _population(group) for name, group in sorted(stages.items())}}


ROBUSTNESS = {"raw_flip": "present predictions on both sides whose semantic answers differ after mapping the sibling back to parent option keys",
              "accepted_flip": "pairs where neither side failed or went unexamined whose accepted decision (the accepted answer, or no acceptance) differs",
              "invariant": "consistent pairs answer identically; a consistent wrong pair is stable but not correct",
              "contrast": "passes only when both sides are correct; the reference gives them different answers, so a stable answer always fails",
              "probability": "total variation 0.5 * sum |p - q| over the declared options, only when both sides hold complete finite distributions",
              "denominators": "submitted pairs; decided excludes failed/unexamined sides; assessed needs a present prediction on both sides"}


def _share(k, n) -> dict:
    return {"count": k, "of": n, "rate": ratio(k, n)}


def _complete(scores, options) -> bool:
    return set(scores) == set(options) and all(_probability(v) for v in scores.values()) and abs(math.fsum(scores.values()) - 1) <= TOLERANCE


def _variation(pair):
    p, q, options = pair["parent"]["scores"], pair["sibling"]["scores"], pair["options"]
    return .5 * math.fsum(abs(p[k] - q[k]) for k in options) if _complete(p, options) and _complete(q, options) else None


def _robust(pairs) -> dict:
    sides = ("parent", "sibling")
    decided = [x for x in pairs if x["parent"]["outcome"] not in UNAVAILABLE and x["sibling"]["outcome"] not in UNAVAILABLE]
    assessed = [x for x in pairs if x["parent"]["answer"] is not None and x["sibling"]["answer"] is not None]
    same = [x for x in assessed if x["parent"]["answer"] == x["sibling"]["answer"]]
    kept = {id(x) for x in same}
    invariant = [x for x in assessed if x["expected"] == "invariant"]
    contrast = [x for x in pairs if x["expected"] == "changed"]
    distances = [d for d in map(_variation, assessed) if d is not None]
    decision = lambda side: side["answer"] if side["accepted"] else None
    return {"submitted": len(pairs), "decided": len(decided), "assessed": len(assessed),
            "families": len({x["family"] for x in pairs}), "transforms": len({x["transform"] for x in pairs}),
            "raw_flip": _share(len(assessed) - len(same), len(assessed)),
            "accepted_flip": _share(sum(decision(x["parent"]) != decision(x["sibling"]) for x in decided), len(decided)),
            "correct": {side: _share(sum(x[side]["correct"] is True for x in pairs), len(pairs)) for side in sides},
            "coverage": {side: _share(sum(x[side]["accepted"] for x in pairs), len(pairs)) for side in sides},
            "failed": {side: sum(x[side]["outcome"] in UNAVAILABLE for x in pairs) for side in sides},
            "unresolved": {side: sum(x[side]["outcome"] in ("uncertain", "needs_input") for x in pairs) for side in sides},
            "invariant": {"assessed": len(invariant), "consistent": sum(id(x) in kept for x in invariant),
                          "consistent_correct": sum(id(x) in kept and x["parent"]["correct"] is True for x in invariant),
                          "consistent_wrong": sum(id(x) in kept and x["parent"]["correct"] is False for x in invariant)},
            "contrast": {"submitted": len(contrast), "passed": _share(sum(x["parent"]["correct"] is True and x["sibling"]["correct"] is True for x in contrast),
                                                                      len(contrast)),
                         "stable": sum(id(x) in kept for x in contrast)},
            "probability": {"comparable": len(distances), "of": len(assessed), "mean_total_variation": ratio(math.fsum(distances), len(distances)),
                            "max_total_variation": max(distances) if distances else None}}


def robustness(pairs) -> dict:
    pairs = list(pairs)
    return {"metric": "robustness", "version": METRICS, "protocol": ROBUSTNESS, "all": _robust(pairs),
            "transformations": {kind: _robust([x for x in pairs if x["transformation"] == kind]) for kind in sorted({x["transformation"] for x in pairs})}}
