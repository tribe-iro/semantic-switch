from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import semantic_switch as ss

DEFINITION = Path(__file__).resolve().parents[1] / "bench" / "data" / "zt411" / "printer.yaml"
EVIDENCE = [
    {"asset": "PR-07", "model": "ZT411", "print_method": "direct_thermal", "prompt": "paused", "enabled": ["set_method_tt", "set_method_dt", "load_ribbon",
     "start_calibration", "resume_printing"], "alerts": [], "verified": ["ribbon_loaded:true"], "attempted": [],
     "notes": "[08:02 Ana, PR-07] I loaded thermal-transfer labels."},
    {"asset": "PR-07", "model": "ZT411", "print_method": "thermal_transfer", "prompt": "calibration_1", "enabled": ["calibration_next", "cancel_calibration"],
     "alerts": [], "verified": ["ribbon_loaded:true"], "attempted": ["start_calibration"],
     "notes": "[08:20 Ana, PR-07] I did the first calibration step on the prompt."},
]


def judge(environ=os.environ):
    model = environ.get("OPENROUTER_MODEL")
    return ss.judge_from(f"openrouter:{model}", provider=environ.get("OPENROUTER_PROVIDER")) if model else ss.judge_from(ss.DEFAULT_JUDGE)


def run(switch):
    described = switch.describe()
    documents = [dict(document) for document in described["documents"]]
    menu = {card["id"]: card["title"] for card in described["facts"]["actions"]["definition"]["cards"]}
    out = []
    for evidence in EVIDENCE:
        result = switch(evidence, facts=["actions"])
        actions = result.facts["actions"]
        out.append({"evidence": evidence, "decision": result.answer, "status": result.status,
                    "justified": list(actions.value or ()), "unresolved": dict(actions.unresolved)})
    print(json.dumps({"documents": documents, "menu": menu}, indent=1))
    for row in out:
        print(json.dumps(row, indent=1))
    return {"documents": documents, "menu": menu, "decisions": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZT411 snapshot decisions through the public Runtime")
    parser.add_argument("--receipt", type=Path, help="specialization receipt: executes its frozen task; default: the authored draft on DEFAULT_JUDGE or OPENROUTER_MODEL")
    args = parser.parse_args()
    with (ss.Runtime.from_specialization(args.receipt) if args.receipt else ss.Runtime({"judge": judge()}, own_judges=True)) as runtime:
        run(runtime.load() if args.receipt else runtime.load(DEFINITION, draft=True))
