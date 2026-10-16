from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.environment import DATA, runtime

DEFINITION = DATA / "printer.yaml"
EVIDENCE = [
    {"asset": "PR-07", "model": "ZT411", "print_method": "direct_thermal", "prompt": "paused", "enabled": ["set_method_tt", "set_method_dt", "load_ribbon",
     "start_calibration", "resume_printing"], "alerts": "unknown", "verified": ["08:01 ribbon_loaded:true"], "attempted": [],
     "notes": "[08:02 Ana, PR-07] I loaded thermal-transfer labels."},
    {"asset": "PR-07", "model": "ZT411", "print_method": "thermal_transfer", "prompt": "calibration_1", "enabled": ["calibration_next", "cancel_calibration"],
     "alerts": "unknown", "verified": ["08:01 ribbon_loaded:true"], "attempted": ["08:15 start_calibration"],
     "notes": "[08:20 Ana, PR-07] I did the first calibration step on the prompt."},
]



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
    parser.add_argument("--lock", type=Path, help="frozen lock (X.lock.json, fitted lock bundle or current pointer); default: the authored draft on DEFAULT_JUDGE or OPENROUTER_MODEL")
    args = parser.parse_args()
    with runtime(args.lock) as bound:
        run(bound.load(args.lock) if args.lock else bound.load(DEFINITION, draft=True))
