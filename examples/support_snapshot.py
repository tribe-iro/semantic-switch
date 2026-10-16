from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.environment import DATA, runtime

DEFINITION = DATA / "support.yaml"
EVIDENCE = [
    {"order": "A10421", "delivered_days_ago": 6, "amount": 48.0, "refund_blocked": False,
     "message": "The shoes I got are two sizes too small. Please refund me."},
    {"order": "A10421", "delivered_days_ago": 6, "amount": 48.0, "refund_blocked": False,
     "message": "The shoes are too small. I want my money back. Actually, forget that. Please send me a replacement."},
    {"order": "A10421", "delivered_days_ago": 40, "amount": 48.0, "refund_blocked": False,
     "message": "On my other order A20077, the box was crushed. This one is fine, but can you tell me where my refund is?"},
]



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
    parser.add_argument("--lock", type=Path, help="frozen lock (X.lock.json, fitted lock bundle or current pointer); default: the authored draft on DEFAULT_JUDGE or OPENROUTER_MODEL")
    args = parser.parse_args()
    with runtime(args.lock) as bound:
        run(bound.load(args.lock) if args.lock else bound.load(DEFINITION, draft=True))
