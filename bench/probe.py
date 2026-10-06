"""Linear probe on the judge's hidden state at the end of the question vs the answer-token readout.

python -m bench.probe [judge source]
A: one generic probe trained on the selection suite, tested on the fresh suite (unseen claims and domains).
B: per-claim probes, leave-one-pair-out, on the fresh suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from semantic_switch import Model, quality
from semantic_switch.adapters import adapter_for
from semantic_switch.api import DEFAULT_JUDGE
from semantic_switch.backend import Weights, _llama, reduce_row
from semantic_switch.judgment import Judgment

from .fresh import FRESH


def extract(source, items):
    L = _llama()
    w = Weights(Model(source).path())
    p = L.llama_context_default_params()
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = 4096, 1024, 1024, 1
    p.embeddings, p.pooling_type = True, 0
    ctx = L.llama_init_from_model(w.ptr, p)
    mem, batch = L.llama_get_memory(ctx), L.llama_batch_init(1024, 0, 1)
    adapter = adapter_for("chat", w)
    feats, judge = [], []
    for it in items:
        q = Judgment.check(it["claim"], tuple(it["excludes"])).question()
        req = adapter.request(adapter.state(quality.render(it["text"])), [q])
        toks = req.prefix + req.state + req.suffixes[0]
        L.llama_memory_clear(mem, True)
        batch.n_tokens = len(toks)
        for j, t in enumerate(toks):
            batch.token[j], batch.pos[j], batch.n_seq_id[j] = t, j, 1
            batch.seq_id[j][0], batch.logits[j] = 0, j == len(toks) - 1
        assert L.llama_decode(ctx, batch) == 0
        feats.append(np.ctypeslib.as_array(L.llama_get_embeddings_ith(ctx, len(toks) - 1), shape=(L.llama_model_n_embd(w.ptr),)).copy())
        logits = L.llama_get_logits_ith(ctx, len(toks) - 1)
        if logits:
            logp, _ = reduce_row(np.ctypeslib.as_array(logits, shape=(w.n_vocab,)).copy(), req.slots[0])
            judge.append(float(np.exp(logp[0])))
        else:
            judge.append(None)
    return np.array(feats), judge


def fit(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(x), y)
    return lambda z: clf.predict_proba(sc.transform(z))[:, 1]


def metrics(p, y):
    from sklearn.metrics import roc_auc_score

    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
    settled = (p >= .9) | (p <= .1)
    return {"accuracy": round(float(((p >= .5) == y).mean()), 3), "auc": round(float(roc_auc_score(y, p)), 3),
            "coverage_0.9": round(float(settled.mean()), 3),
            "precision_0.9": round(float(((p >= .5) == y)[settled].mean()), 3) if settled.any() else None}


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JUDGE
    sel, fresh = list(quality.check_items()), list(quality.check_items(FRESH))
    cache = Path("bench/results/probe_features.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        xs, js, xf, jf = z["xs"], list(z["js"]), z["xf"], list(z["jf"])
    else:
        xs, js = extract(source, sel)
        xf, jf = extract(source, fresh)
        np.savez(cache, xs=xs, js=np.array(js), xf=xf, jf=np.array(jf))
    ys, yf = np.array([i["label"] for i in sel]), np.array([i["label"] for i in fresh])
    report = {"A_generic_probe_fresh": metrics(fit(xs, ys)(xf), yf)}
    per_claim = np.zeros(len(fresh))
    for k, it in enumerate(fresh):
        train = [i for i, o in enumerate(fresh) if o["fact"] == it["fact"] and o["pair"] != it["pair"]]
        per_claim[k] = fit(xf[train], yf[train])(xf[k:k + 1])[0]
    report["B_per_claim_probe_fresh_loo"] = metrics(per_claim, yf)
    for name, items, x, y, j in (("fresh", fresh, xf, yf, jf), ("selection", sel, xs, ys, js)):
        rng = np.random.default_rng(0)
        for pairs in (2, 4, 6, 9):
            pred = np.zeros(len(items))
            for k, it in enumerate(items):
                others = sorted({o["pair"] for o in items if o["fact"] == it["fact"] and o["pair"] != it["pair"]})
                chosen = set(rng.choice(others, size=min(pairs, len(others)), replace=False))
                train = [i for i, o in enumerate(items) if o["pair"] in chosen]
                pred[k] = fit(x[train], y[train])(x[k:k + 1])[0]
            report[f"curve_{name}_{pairs}_pairs"] = metrics(pred, y)
        report[f"judge_{name}"] = metrics(j, y)
    report["n"] = {"selection": len(sel), "fresh": len(fresh), "dim": int(xs.shape[1])}
    print(json.dumps(report, indent=1))
    Path("bench/results/probe.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
