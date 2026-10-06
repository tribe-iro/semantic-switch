"""Compare judge models on held-out minimal pairs, with an optional hidden-state probe.

Run from the repository root:
    python -m bench.models --judges Qwen/Qwen3-1.7B-GGUF:Qwen3-1.7B-Q8_0.gguf,Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from semantic_switch import DEFAULT_JUDGE, Model, Runtime

QWEN3 = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"

from .learning_curve import QUESTION, dataset


def rows(seed: int, pairs: int):
    pool, held = dataset(seed, 100, pairs)
    text = lambda r: f"[customer] body: {r['input']['body']}"
    return [(text(r), r["value"]) for r in pool], [(text(r), r["value"]) for r in held]


def score(rt, items):
    t0 = time.perf_counter()
    ps = np.array([rt.check(t, QUESTION).p_true for t, _ in items])
    ys = np.array([y for _, y in items])
    settled = (ps >= .9) | (ps <= .1)
    return {"accuracy": round(float(((ps >= .5) == ys).mean()), 3), "settled": int(settled.sum()),
            "settled_correct": int((((ps >= .9) == ys) & settled).sum()), "items": len(ys),
            "ms_per_item": round((time.perf_counter() - t0) / len(ys) * 1000, 1)}


def probe(source: str, pool, held, sizes, threads: int):
    import llama_cpp as L
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from semantic_switch.adapters.qwen3 import Qwen3Adapter
    from semantic_switch.backend import Weights, _llama
    from semantic_switch.protocols import Question

    _llama()
    w = Weights(Model(source).path())
    p = L.llama_context_default_params()
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = 4096, 1024, 1024, 1
    p.n_threads = p.n_threads_batch = threads
    p.embeddings, p.pooling_type = True, 0
    ctx = L.llama_init_from_model(w.ptr, p)
    mem, batch = L.llama_get_memory(ctx), L.llama_batch_init(1024, 0, 1)
    adapter = Qwen3Adapter(w)

    def features(items):
        out = []
        for text, _ in items:
            req = adapter.request(adapter.state(text), [Question("check", QUESTION)])
            toks = req.prefix + req.state + req.suffixes[0]
            L.llama_memory_clear(mem, True)
            batch.n_tokens = len(toks)
            for j, t in enumerate(toks):
                batch.token[j], batch.pos[j], batch.n_seq_id[j] = t, j, 1
                batch.seq_id[j][0], batch.logits[j] = 0, j == len(toks) - 1
            L.llama_decode(ctx, batch)
            out.append(np.ctypeslib.as_array(L.llama_get_embeddings_ith(ctx, len(toks) - 1), shape=(w.n_embd,)).copy())
        return np.array(out)

    xp, xh = features(pool), features(held)
    yp, yh = np.array([y for _, y in pool]), np.array([y for _, y in held])
    out = {}
    for n in sizes:
        sc = StandardScaler().fit(xp[:2 * n])
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(xp[:2 * n]), yp[:2 * n])
        pm = clf.predict_proba(sc.transform(xh))[:, list(clf.classes_).index(True)]
        out[n] = {"auc": round(float(roc_auc_score(yh, pm)), 3), "accuracy": round(float(((pm >= .5) == yh).mean()), 3)}
    L.llama_batch_free(batch)
    L.llama_free(ctx)
    w.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", default=f"{DEFAULT_JUDGE},{QWEN3}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--probe", default="", help="probe sizes per class, e.g. 10,30,100 (Qwen-template judges)")
    ap.add_argument("--output", type=Path, default=Path("bench/results/models.json"))
    args = ap.parse_args()
    pool, held = rows(args.seed, args.pairs)
    report = {"task": QUESTION, "seed": args.seed, "items": len(held), "judges": {}}
    for source in args.judges.split(","):
        with Runtime({"judge": Model(source, threads=args.threads)}) as rt:
            entry = {"zero_shot": score(rt, held)}
        if args.probe:
            entry["probe"] = probe(source, pool, held, [int(s) for s in args.probe.split(",")], args.threads)
        report["judges"][source] = entry
        print(source, json.dumps(entry), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
