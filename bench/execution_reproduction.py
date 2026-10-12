from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, Check, Model, Pick, files
from semantic_switch.api import execution_identity
from bench import diagnostics
from bench.external_primitives import CACHE, primitive, state
from bench.outcomes import failure, measure, runtime
from semantic_switch.metrics import Progress

OUTPUT = Path("bench/results/execution-reproduction.json")
GEOMETRY = Path("bench/results/execution-geometry.json")
BENCHMARK = "go-emotions-noul-test-v1"
TARGETS = ((0, "annoyance"), (4, "surprise"))
FILLERS = (1000, 5000)
LENGTHS = tuple(range(0, 257, 4))
EMOTION = Pick("Which emotion dominates the text?", {
    "anger": "The writer is angry, hostile or insulting toward someone or something",
    "joy": "The writer is happy, amused, grateful or excited about something",
    "sadness": "The writer is sad, disappointed, grieving or remorseful about something",
    "fear": "The writer is afraid, nervous or worried about something that may happen",
    "neutral": "The text states information or asks something without a clear emotion"})


def cases(count):
    rows = [row for row in files.read_jsonl(CACHE / "cases-25.jsonl") if row["benchmark"] == BENCHMARK][:count]
    if len(rows) != count:
        raise SystemExit(f"{CACHE / 'cases-25.jsonl'}: run python -m bench.external_primitives --limit 25 first")
    return [(state(row["case"]), {q["id"]: primitive(q) for q in row["case"]["questions"]}) for row in rows]


FAILED = {"value": None, "status": "failed", "p": [], "mass": None}


def first(p):
    return p[0] if p else None


def observe(result):
    probs = [] if result.status == "failed" else [result.p_true] if hasattr(result, "p_true") else list(result.scores.values())
    return {"value": result.value, "status": result.status, "p": probs, "mass": result.mass}


def residency(scorer):
    return {"current_prefix_tokens": len(scorer._prefix or ()), "resident_prefix_tokens": sorted(map(len, scorer._resident)),
            "state_tokens": sorted(map(len, scorer._states.values()))}


class Session:
    def __init__(self, rt):
        self.rt, self.scorer, self.order = rt, rt.judge().scorer, []
        score = self.scorer.score

        def recorded(requests, **kwargs):
            self.order.append([(len(r.prefix), len(r.state), [len(s) for s in r.suffixes]) for r in requests])
            return score(requests, **kwargs)
        self.scorer.score = recorded

    def cold(self):
        self.scorer.invalidate()

    def ask(self, text, specs):
        before, self.order = residency(self.scorer), []
        measured = measure(lambda: self.rt.ask(text, specs), runtime=self.rt)
        known = {"questions": list(specs), "residency_before": before, "windows": [w for call in self.order for w in call], "ms": measured.ms, "work": measured.work}
        if measured.error is not None:
            return failure(measured, **known, results=dict.fromkeys(specs, FAILED))
        return {**known, "outcome": "predicted", "results": {key: observe(result) for key, result in measured.result.items()}}


def arms(session, data, index):
    text, specs = data[index]
    keys = list(specs)
    other = data[(index + 1) % len(data)]
    sub = lambda names: {key: specs[key] for key in names}
    out = {}
    session.cold()
    out["cold"] = session.ask(text, specs)
    session.cold()
    out["cold_repeat"] = session.ask(text, specs)
    out["warm"] = session.ask(text, specs)
    session.cold()
    session.ask(*other)
    out["evicted"] = session.ask(text, specs)
    session.cold()
    out["reversed"] = session.ask(text, sub(keys[::-1]))
    session.cold()
    out["shuffled"] = session.ask(text, sub(random.Random(index).sample(keys, len(keys))))
    for key in keys[:4]:
        session.cold()
        out[f"one:{key}"] = session.ask(text, sub([key]))
    for name, names in (("four", keys[:4]), ("seventeen", keys[:17]), ("seventeen_tail", keys[-17:])):
        session.cold()
        out[name] = session.ask(text, sub(names))
    session.cold()
    out["mixed_guidance"] = session.ask(text, {**sub(keys[:10]), "emotion": EMOTION})
    session.cold()
    out["unguided"] = session.ask(text, {key: Check(f"The text expresses {key}.") for key in keys[:5]})
    return out


def history(session, data, index, target):
    text, specs = data[index]
    others = [key for key in specs if key != target]
    one = lambda key: session.ask(text, {key: specs[key]})
    out = {}
    session.cold()
    out["alone"] = one(target)
    out["alone_again"] = one(target)
    for count in (1, 2, 3, 4, 5, 8):
        session.cold()
        for key in others[:count]:
            one(key)
        out[f"after_{count}_prefixes"] = one(target)
    for count in (1, 3, 4):
        session.cold()
        one(target)
        for key in others[:count]:
            one(key)
        out[f"return_after_{count}"] = one(target)
    session.cold()
    session.ask(text, {key: specs[key] for key in others[:3]})
    out["after_3_in_one_call"] = one(target)
    session.cold()
    for key in others[:3]:
        one(key)
    out["after_3_with_target_in_call"] = session.ask(text, {key: specs[key] for key in [*others[:3], target]})
    session.cold()
    session.ask(*data[(index + 1) % len(data)])
    out["after_other_case"] = one(target)
    session.cold()
    session.ask(text, {"emotion": EMOTION})
    out["after_unguided_pick"] = one(target)
    return {"target": target, "arms": out}


def offsets(session, data, index, target, lengths):
    text, specs = data[index]
    scorer, out = session.scorer, []
    spare = scorer.ctx.n_seq_max - 1
    for length in lengths:
        for content in FILLERS:
            session.cold()
            with scorer.operation():
                scorer.ctx.run([(content + k % 97, k, spare, None) for k in range(length)], None)
            found = session.ask(text, {target: specs[target]})
            out.append({"filler_tokens": length, "filler_token_base": content, **found["results"][target]})
    return out


def changes(reference, observed, keys=None):
    rows = []
    for key in keys or reference["results"]:
        if key not in observed["results"] or key not in reference["results"]:
            continue
        a, b = reference["results"][key], observed["results"][key]
        rows.append({"question": key, "accepted_change": (a["value"], a["status"]) != (b["value"], b["status"]),
                     "dp": max((abs(x - y) for x, y in zip(a["p"], b["p"])), default=0.), "identical": a == b})
    return {"questions": len(rows), "accepted_changes": sum(r["accepted_change"] for r in rows), "identical": sum(r["identical"] for r in rows),
            "max_dp": max((r["dp"] for r in rows), default=0.), "changed": [r["question"] for r in rows if r["accepted_change"]]}


def against_cold(found):
    return {name: changes(found["cold"], arm) for name, arm in found.items() if name not in ("cold", "unguided")}


def target_table(found, target):
    rows = {}
    for name, arm in found.items():
        if target in arm["results"] and name != "unguided":
            rows[name] = {**arm["results"][target], "work": arm["work"], "residency_before": arm["residency_before"]}
    return rows


def run(count, threads):
    model = Model(DEFAULT_JUDGE, device="cpu", repack=True, threads=threads, prefix_slots=4)
    data = cases(count)
    with runtime(model) as rt:
        session = Session(rt)
        session.ask("warm up", {"warm": Check("The text is about food.")})
        report = {"provenance": diagnostics.provenance(rt, [text for text, _ in data], boundary="public Runtime.ask; cache reset through LlamaScorer.invalidate"),
                  "execution": execution_identity(rt.judge().scorer.capacity()), "cases": {}, "histories": {}}
        progress = Progress("bench.execution_reproduction", count + len(TARGETS), unit="cases")
        for index in range(count):
            found = arms(session, data, index)
            report["cases"][str(index)] = {"arms": found}
            progress.record([{"outcome": "failed" if any(arm["outcome"] == "failed" for arm in found.values()) else "predicted"}])
            print(index, {name: value["accepted_changes"] for name, value in against_cold(found).items()}, flush=True)
        for index, target in TARGETS:
            found = history(session, data, index, target)
            report["histories"][f"{index}:{target}"] = found
            progress.record([{"outcome": "failed" if any(arm["outcome"] == "failed" for arm in found["arms"].values()) else "predicted"}])
            print(index, target, {name: (arm["results"][target]["status"], first(arm["results"][target]["p"])) for name, arm in found["arms"].items()}, flush=True)
        report["progress"] = progress.finish()
    return report


def scan(threads):
    model = Model(DEFAULT_JUDGE, device="cpu", repack=True, threads=threads, prefix_slots=4)
    data = cases(max(index for index, _ in TARGETS) + 1)
    with runtime(model) as rt:
        session = Session(rt)
        return {f"{index}:{target}": offsets(session, data, index, target, LENGTHS) for index, target in TARGETS}


def summarize(report):
    recorded = files.read_json(GEOMETRY)["devices"]["cpu"]["serial"]["repeatability"]
    targets = {}
    for index, target in TARGETS:
        found = report["cases"][str(index)]["arms"]
        rows = target_table(found, target) | {f"history:{name}": {**arm["results"][target], "work": arm["work"], "residency_before": arm["residency_before"]}
                                              for name, arm in report["histories"][f"{index}:{target}"]["arms"].items()}
        outcomes = {}
        for name, row in rows.items():
            outcomes.setdefault(f"{row['status']}:{row['value']}", []).append(name)
        distinct = {}
        for name, row in rows.items():
            distinct.setdefault((first(row["p"]), row["mass"]), []).append(name)
        targets[f"{index}:{target}"] = {"recorded": {name: changed for name, arm in recorded.items()
                                                     if (changed := [c for c in arm["changes"] if (c["case"], c["question"]) == (index, target)])},
                                        "outcomes": outcomes, "distinct_scores": len(distinct),
                                        "score_classes": [{"p": p, "mass": mass, "arms": names} for (p, mass), names in sorted(distinct.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0., item[0][1] or 0.))],
                                        "reproduced": len(outcomes) > 1}
    for key, rows in report.get("offsets", {}).items():
        by_length = {}
        for row in rows:
            by_length.setdefault(row["filler_tokens"], []).append((first(row["p"]), row["mass"], row["status"]))
        targets[key]["offset_scan"] = {"filler_tokens": sorted(by_length), "content_invariant": all(len(set(values)) == 1 for values in by_length.values()),
                                       "distinct_scores": len({values[0] for values in by_length.values()}),
                                       "outcomes": {status: [n for n, values in sorted(by_length.items()) if values[0][2] == status] for status in {v[0][2] for v in by_length.values()}},
                                       "p": {n: values[0][0] for n, values in sorted(by_length.items())}}
        targets[key]["reproduced"] = targets[key]["reproduced"] or len(targets[key]["offset_scan"]["outcomes"]) > 1
    flips = {case: {name: value for name, value in against_cold(data["arms"]).items() if value["accepted_changes"]} for case, data in report["cases"].items()}
    return {"schema": 1, "model": DEFAULT_JUDGE, "configuration": report["execution"],
            "varied": ["question count", "question order", "prefix/state cache history"], "targets": targets, "accepted_changes_against_cold": flips}


def main():
    parser = argparse.ArgumentParser(description="Rerun the recorded CPU accepted-outcome flips with one fixed numerical configuration")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--summarize", action="store_true", help="re-summarize the recorded raw runs without loading a model")
    parser.add_argument("--offsets", action="store_true", help="add the KV-offset scan of the target questions to the recorded runs")
    args = parser.parse_args()
    previous = files.read_json(OUTPUT) if args.summarize or args.offsets else None
    threads = previous["threads"] if previous else args.threads
    report = previous["raw"] if previous else run(args.cases, threads)
    if args.offsets:
        report["offsets"] = scan(threads)
    files.publish(OUTPUT, {**summarize(report), "threads": threads, "raw": report})
    print(OUTPUT)


if __name__ == "__main__":
    main()
