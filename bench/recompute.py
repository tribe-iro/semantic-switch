from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_switch import ConfigError, files
from semantic_switch.metrics import METRICS
from bench.snapshots import DOMAINS, set_metrics, set_rows, step_rows

ROWS = "stored rows lack the outcome or per-member reference and prediction states needed to establish completeness"
BASELINE = "stored baseline lacks canonical per-frame set rows"
STEPS = "stored episode steps lack the outcome or per-member reference and prediction states needed to establish completeness"
CANONICAL = ("scope", "gold", "reference_unresolved", "complete_gold", "selected", "unresolved", "prediction_complete", "outcome", "prohibited")
STEP = ("selected", "unresolved", "outcome", "justified", "prohibited", "reference_unresolved")


def recoverable(rows, domain, name) -> bool:
    return all("outcome" in row and all(f"{name}/{key}" in row["gold"]["values"] and "state" in row["targets"].get(f"{name}/{key}", {})
                                        for key in domain.sets[name]) for row in rows)


def rebuild(build, ok, reason) -> tuple[dict, str]:
    if ok:
        return set_metrics(build()), "recomputed"
    marker = {"version": METRICS, "state": "unrecoverable", "reason": reason}
    return {"actions": marker, "exact": marker}, "unrecoverable"


def snapshot(artifact) -> dict:
    domain = DOMAINS[artifact["domain"]]
    states = {}
    for key, result in artifact["results"].items():
        rows = result["rows"]
        for name in domain.sets:
            kept = {k: v for k, v in result["sets"][name].items() if k not in ("frontier", "actions", "exact")}
            metrics, states[f"results|{key}|{name}"] = rebuild(lambda: set_rows(domain, rows, name), recoverable(rows, domain, name), ROWS)
            result["sets"][name] = {**kept, **metrics}
    for name, arms in artifact["baselines"].items():
        for arm, entry in arms.items():
            rows = entry.get("rows")
            metrics, states[f"baselines|{name}|{arm}"] = rebuild(lambda: rows, rows is not None and all(set(CANONICAL) <= set(row) for row in rows), BASELINE)
            arms[arm] = {**metrics, **({"rows": rows} if rows is not None else {})}
    return states


def episodes(artifact) -> dict:
    steps = [step for trace in artifact["traces"] for step in trace["steps"]]
    artifact["snapshot_steps"], state = rebuild(lambda: step_rows(artifact["traces"]), all(set(STEP) <= set(step) for step in steps), STEPS)
    return {"snapshot_steps": state}


def recompute(report, to) -> dict:
    artifact = files.read_json(report)
    if artifact.get("domain") in DOMAINS and isinstance(artifact.get("results"), dict):
        states = snapshot(artifact)
    elif isinstance(artifact.get("traces"), list):
        states = episodes(artifact)
    else:
        raise ConfigError(f"{report}: not a bench.snapshots report")
    artifact["recomputed"] = {"metrics": METRICS, "source": str(report), "source_sha256": files.sha256_file(report), "aggregates": states}
    files.write_new(to, json.dumps(artifact, indent=1, ensure_ascii=False))
    return states


def main():
    parser = argparse.ArgumentParser(prog="bench.recompute", description="recompute set aggregates of a published bench.snapshots or episode report from its stored rows")
    parser.add_argument("report", type=Path)
    parser.add_argument("--to", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(recompute(args.report, args.to), indent=1))
    except ConfigError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
