"""Judge ladder: accuracy, coverage at a precision target, calibration, quantisation shift, latency, memory.

Run from the repository root:
    python -m bench.judges --judges "ggml-org/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf,unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
    python -m bench.judges --analyze bench/results/judges
Each judge writes bench/results/judges/<file>.json with per-item probabilities; --analyze summarises all of them.
Acceptance follows execution: low answer mass is never settled.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from semantic_switch import Model, Runtime, quality

OUT = Path("bench/results/judges")
GRID = np.round(np.arange(0.50, 1.0, 0.01), 2)


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return float("nan")


def run(source: str, adapter: str, threads: int | None, shared: bool | None) -> dict:
    before = rss_mb()
    t0 = time.perf_counter()
    rt = Runtime({"judge": Model(source, adapter=adapter, threads=threads)})
    judge = rt.judge()
    if shared is not None:
        judge.scorer.shared = shared
    load_s = time.perf_counter() - t0
    res = quality.run(rt)
    out = {"source": source, "adapter": judge.adapter.name, "shared": judge.scorer.shared,
           "file_mb": round(os.path.getsize(Model(source).path()) / 2**20), "rss_mb": round(rss_mb() - before),
           "load_s": round(load_s, 1), "threads": threads or os.cpu_count(), **res}
    rt.close()
    return out


def _settle(p, t):
    return np.where(p >= t, 1, np.where(p <= 1 - t, 0, -1))


def _best_threshold(p, y, target: float, min_settled: int = 10):
    for t in GRID:
        s = _settle(p, t)
        m = s >= 0
        if m.sum() >= min_settled and ((s[m] == 1) == y[m]).mean() >= target:
            return float(t)
    return None


def coverage_at(p, y, facts, target: float = 0.95) -> dict:
    names = sorted(set(facts))
    halves = (set(names[0::2]), set(names[1::2]))
    settled = correct = 0
    for tune, test in (halves, halves[::-1]):
        a = np.array([f in tune for f in facts])
        b = np.array([f in test for f in facts])
        t = _best_threshold(p[a], y[a], target)
        if t is None:
            continue
        s = _settle(p[b], t)
        settled += int((s >= 0).sum())
        correct += int(((s == 1) == y[b])[s >= 0].sum())
    return {"coverage": round(settled / len(p), 3), "precision": round(correct / settled, 3) if settled else None}


def summarize(res: dict) -> dict:
    rows = [r for r in res["checks"] if r["p"] is not None and (r["mass"] is None or r["mass"] >= 0.5)]
    p = np.array([r["p"] for r in rows])
    y = np.array([r["label"] for r in rows])
    return {"model": Path(res["source"].split(":")[-1]).stem, "adapter": res["adapter"], "file_mb": res["file_mb"],
            "rss_mb": res["rss_mb"], **quality.summarize(res), "at_95": coverage_at(p, y, [r["fact"] for r in rows])}


def family(name: str) -> str:
    return re.sub(r"[-_.]?(q\d.*|bf16|f16|ud-.*)$", "", name.lower().replace("_q4_0-it", "-it-q4_0"))


def shift(a: dict, b: dict) -> dict:
    pa = np.array([r["p"] if r["p"] is not None else np.nan for r in a["checks"]])
    pb = np.array([r["p"] if r["p"] is not None else np.nan for r in b["checks"]])
    return {"mean_abs_dp": round(float(np.nanmean(np.abs(pa - pb))), 3),
            "p95_abs_dp": round(float(np.nanpercentile(np.abs(pa - pb), 95)), 3),
            "flips_0.5": int(((pa >= 0.5) != (pb >= 0.5)).sum()),
            "settle_changes_0.9": int((_settle(pa, 0.9) != _settle(pb, 0.9)).sum())}


def analyze(folder: Path) -> dict:
    results = {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if p.stem != "summary"}
    table = {k: summarize(v) for k, v in results.items()}
    shifts = {}
    for k, v in results.items():
        ref = [r for r in results if r != k and family(r) == family(k) and "q8" in r.lower()]
        if ref and "q8" not in k.lower():
            shifts[f"{k} vs {ref[0]}"] = shift(results[ref[0]], v)
    report = {"summary": table, "quantisation_shift": shifts}
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judges", default="")
    ap.add_argument("--adapter", default="chat")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--shared", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--analyze", type=Path, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shared = {"auto": None, "on": True, "off": False}[args.shared]
    for source in filter(None, args.judges.split(",")):
        res = run(source, args.adapter, args.threads, shared)
        name = Path(source.split(":")[-1]).stem + ("" if args.adapter == "chat" else f"@{args.adapter}")
        (OUT / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
        print(json.dumps(summarize(res)), flush=True)
    if args.analyze:
        print(json.dumps(analyze(args.analyze), indent=2))


if __name__ == "__main__":
    main()
