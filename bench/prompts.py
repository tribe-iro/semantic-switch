"""Prompt framings and ensembles for check questions, chosen on dev facts and reported on held-out test facts.

Run from the repository root:
    python -m bench.prompts --judges Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from semantic_switch import Model, Runtime
from semantic_switch.protocols import ScoreRequest

from .suite import CHECKS, check_items

DEV = set(list(CHECKS)[0::2])
MEANING = "Judge by meaning, not exact wording: the claim can hold without using its words."
FRAMES = {
    "claim": (lambda c, ex: [f"Claim about the input above: {c}", *ex,
                             "Is the claim true or false? Answer with only true or false."], ("true", "false")),
    "meaning": (lambda c, ex: [f"Claim about the input above: {c}", *ex, MEANING,
                               "Is the claim true or false? Answer with only true or false."], ("true", "false")),
    "fits": (lambda c, ex: [f"Description: {c}", *ex,
                            "Does this description fit the input above? Answer with only yes or no."], ("yes", "no")),
    "fits_meaning": (lambda c, ex: [f"Description: {c}", *ex, MEANING.replace("claim", "description"),
                                    "Does this description fit the input above? Answer with only yes or no."],
                     ("yes", "no")),
}


def scores(rt, frame: str) -> list[float]:
    judge = rt.judge()
    a = judge.adapter
    build, (yes, no) = FRAMES[frame]
    slots = (a._variants(yes, yes.title()), a._variants(no, no.title()))
    out = []
    for it in check_items():
        ex = [f"Exception: {x} does not count." for x in it["excludes"]]
        suffix = a.content("\n".join(build(it["claim"], ex))) + a.tail
        state = a.state(f"[customer] message: {it['text']}")
        block = judge.scorer.score([ScoreRequest(a.prefix, state, (suffix,), (slots,))])
        out.append(float(np.exp(block.logp[0, 0])))
    return out


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def metrics(p, y) -> dict:
    p, y = np.asarray(p), np.asarray(y)
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    right = ((p >= .5) == y)[order]
    prec = np.cumsum(right) / np.arange(1, len(p) + 1)
    ok = np.where(prec >= .95)[0]
    s = np.where(p >= .9, 1, np.where(p <= .1, 0, -1))
    m = s >= 0
    pos, neg = p[y], p[~y]
    auc = ((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    return {"acc": round(float(right.mean()), 3), "auc": round(float(auc), 3),
            "oracle_cov@95": round((ok.max() + 1) / len(p), 3) if len(ok) else 0.0,
            "settled@0.9": round(float(m.mean()), 3),
            "prec@0.9": round(float(((s[m] == 1) == y[m]).mean()), 3) if m.any() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--output", type=Path, default=Path("bench/results/prompts.json"))
    args = ap.parse_args()
    items = list(check_items())
    y = np.array([it["label"] for it in items])
    dev = np.array([it["fact"] in DEV for it in items])
    report = json.loads(args.output.read_text()) if args.output.exists() else {}
    for source in args.judges.split(","):
        with Runtime({"judge": Model(source, adapter="chat")}) as rt:
            t0 = time.perf_counter()
            ps = {f: scores(rt, f) for f in FRAMES}
            ms = (time.perf_counter() - t0) / (len(FRAMES) * len(items)) * 1000
        ens = {"claim+fits": ("claim", "fits"), "meaning+fits_meaning": ("meaning", "fits_meaning"),
               "all4": tuple(FRAMES)}
        for name, parts in ens.items():
            z = np.mean([logit(ps[f]) for f in parts], axis=0)
            ps[name] = list(1 / (1 + np.exp(-z)))
        entry = {"ms_per_frame": round(ms, 1),
                 "dev": {f: metrics(np.array(p)[dev], y[dev]) for f, p in ps.items()},
                 "test": {f: metrics(np.array(p)[~dev], y[~dev]) for f, p in ps.items()},
                 "p": ps}
        report[Path(source.split(":")[-1]).stem] = entry
        print(source)
        for split in ("dev", "test"):
            for f, m in entry[split].items():
                print(f"  {split:4} {f:22} {m}")
    args.output.write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
