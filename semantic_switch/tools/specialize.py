from __future__ import annotations

import time
from dataclasses import replace

from ..errors import ConfigError
from ..judgment import OFF
from ..switch import Switch
from .common import auc, test_items
from .lock import FOLDS, fit_policy, folds_of, out_of_fold, rows_for, train_head


def _run(sw, snap, f: int, inputs: list, sources: tuple) -> tuple[list, list, float]:
    t0 = time.perf_counter()
    calls = sw.score_facts(inputs, snap, sources=sources, facts=[f])
    ms = (time.perf_counter() - t0) * 1000 / max(len(inputs), 1)
    return [sw.exec._own(snap, c, f)[0] if c.facts else None for c in calls], \
        [c.facts[f] if c.facts else None for c in calls], ms


def specialize(rt, path, fact_name: str) -> dict:
    sw = Switch(rt, path, draft=True)
    snap = sw.snapshot
    plan = snap.plan
    fact = plan.fact(fact_name)
    f = plan.fact_index[fact_name]
    items, _ = test_items(sw.paths, plan)
    rows = rows_for(sw, fact, snap.examples[fact.name], items)
    if fact.cards or len(rows) < FOLDS or len({r.label for r in rows}) < 2:
        raise ConfigError(f"{fact_name} needs at least {FOLDS} labelled inputs (examples or test labels) "
                          f"with two or more different values; it has {len(rows)}")
    fold = folds_of(rows)
    judge, cascade, sources, dists = [None] * len(rows), [None] * len(rows), [None] * len(rows), [None] * len(rows)
    ms = {"judge": [], "lexical": []}
    for k in range(FOLDS):
        held = [i for i, x in enumerate(fold) if x == k]
        kept = [rows[i] for i, x in enumerate(fold) if x != k]
        if not held or len({r.label for r in kept}) < 2:
            continue
        head = train_head(fact, kept)
        policy = fit_policy(fact, out_of_fold(fact, kept), OFF)
        examples = tuple(e for e in snap.examples[fact.name] if e.key in {r.group for r in kept})
        base = snap.with_examples(fact.name, examples)
        installed = replace(base, heads={**snap.heads, fact.name: head},
                            policies={**snap.policies, "classifier": {**snap.policies.get("classifier", {}),
                                                                      fact.name: policy}})
        inputs = [rows[i].data for i in held]
        j, states, jm = _run(sw, base, f, inputs, ("example", "judge"))
        c, cstates, cm = _run(sw, installed, f, inputs, ("example", "classifier", "judge"))
        ms["judge"].append(jm)
        ms["lexical"].append(cm)
        for i, a, b, s, d in zip(held, j, c, cstates, states):
            judge[i], cascade[i], sources[i], dists[i] = a, b, s.source if s else None, d.dist if d else None
    y = [r.label for r in rows]
    wrong = lambda g, t: g is not None and g != t
    right = lambda g, t: g is not None and g == t
    tally = lambda got: {"settled": sum(g is not None for g in got), "correct": sum(right(g, t) for g, t in zip(got, y)),
                         "wrong": sum(wrong(g, t) for g, t in zip(got, y))}
    report = {"fact": fact_name, "labelled": len(rows), "folds": FOLDS, "judge": tally(judge) | {
        "ms": round(sum(ms["judge"]) / max(len(ms["judge"]), 1), 1)}}
    if fact.kind == "check":
        scored = [(d[0], t == 0) for d, t in zip(dists, y) if d is not None and t != 2]
        report["judge"]["auc"] = auc([s for s, _ in scored], [t for _, t in scored])
    report["lexical"] = {
        **tally(cascade), "ms": round(sum(ms["lexical"]) / max(len(ms["lexical"]), 1), 1),
        "judge_calls_avoided": sum(s == "classifier" for s in sources),
        "gained": sum(right(c, t) and not right(j, t) for j, c, t in zip(judge, cascade, y)),
        "avoided": sum(wrong(j, t) and c is None for j, c, t in zip(judge, cascade, y)),
        "introduced": sum(wrong(c, t) and not wrong(j, t) for j, c, t in zip(judge, cascade, y))}
    lx, jd = report["lexical"], report["judge"]
    report["admit"] = lx["judge_calls_avoided"] > 0 and lx["wrong"] <= jd["wrong"]
    accuracy = jd["correct"] / len(rows)
    diagnosis = ("the judge ranks this judgment well but its boundary is off: reword the claim, add exact rules, "
                 "or rely on `ss lock` abstention" if jd.get("auc") and jd["auc"] >= 0.9 and accuracy < 0.85 else
                 "the judge barely separates this judgment: reword the claim or use a stronger judge (--precise)"
                 if jd.get("auc") is not None and jd["auc"] < 0.75 else "the judge separates this judgment")
    report["report"] = [
        f"{fact_name}: {len(rows)} labelled inputs, {FOLDS} folds by group, run through the executor.",
        f"  judge alone: {jd['correct']} correct, {jd['wrong']} wrong, {len(rows) - jd['settled']} abstained, "
        f"~{jd['ms']:.0f} ms per input" + (f"; ranking AUC {jd['auc']:.2f}: {diagnosis}" if "auc" in jd else "."),
        f"  with a lexical head: settled {lx['judge_calls_avoided']} without the judge; {lx['correct']} correct, "
        f"{lx['wrong']} wrong (gained {lx['gained']}, avoided {lx['avoided']}, introduced {lx['introduced']}); "
        f"~{lx['ms']:.0f} ms per input.",
        f"  verdict: {'admit' if report['admit'] else 'do not admit'}"
        + (f" — add `specialize: lexical` under {fact_name} in {sw.paths.yaml.name}, then `ss lock`."
           if report["admit"] else "."),
    ]
    return report
