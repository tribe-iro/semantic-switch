from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import semantic_switch as ss

DEFINITION = Path(__file__).resolve().parents[1] / "bench" / "data" / "support" / "support.yaml"
EVIDENCE = [
    {"order": "A10421", "delivered_days_ago": 6, "amount": 48.0, "refund_blocked": False,
     "message": "The shoes I got are two sizes too small. Please refund me."},
    {"order": "A10421", "delivered_days_ago": 6, "amount": 48.0, "refund_blocked": False,
     "message": "The shoes are too small. I want my money back. Actually, forget that. Please send me a replacement."},
    {"order": "A10421", "delivered_days_ago": 40, "amount": 48.0, "refund_blocked": False,
     "message": "On my other order A20077, the box was crushed. This one is fine, but can you tell me where my refund is?"},
]


def judge(environ=os.environ):
    model = environ.get("OPENROUTER_MODEL")
    return ss.judge_from(f"openrouter:{model}", provider=environ.get("OPENROUTER_PROVIDER")) if model else ss.judge_from(ss.DEFAULT_JUDGE)


def run(switch):
    described = switch.describe()
    documents = [dict(document) for document in described["documents"]]
    names = list(described["facts"])
    out = []
    for evidence in EVIDENCE:
        result = switch(evidence, facts=names)
        out.append({"evidence": evidence, "response": result.answer, "status": result.status,
                    **{name: {"value": fact.value, "status": fact.status} for name, fact in result.facts.items()}})
    print(json.dumps({"documents": documents}, indent=1))
    for row in out:
        print(json.dumps(row, indent=1, default=list))
    return {"documents": documents, "decisions": out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="support classification and response through the public Runtime")
    parser.add_argument("--receipt", type=Path, help="specialization receipt: executes its frozen task; default: the authored draft on DEFAULT_JUDGE or OPENROUTER_MODEL")
    args = parser.parse_args()
    with (ss.Runtime.from_specialization(args.receipt) if args.receipt else ss.Runtime({"judge": judge()}, own_judges=True)) as runtime:
        run(runtime.load() if args.receipt else runtime.load(DEFINITION, draft=True))
