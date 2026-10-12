from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import semantic_switch as ss

ACTION = ss.Pick(
    "What should support do next?",
    {"refund": "Refund the duplicate charge", "escalate": "Escalate to a supervisor",
     "ask_details": "Ask the customer for the missing order number"},
    allow_none=True,
)
ROUTE = {
    "name": "refund_route",
    "fields": {"message": {"type": "text", "required": True}, "refund_percent": {"type": "number", "required": True}},
    "reads": ["message"],
    "facts": {
        "approved": {"check": "A supervisor explicitly approves an exception for this refund."},
        "duplicate": {"check": "The customer reports being charged twice for the same order."},
    },
    "cases": [
        {"id": "approved", "when": "approved", "then": "pay_exception"},
        {"id": "small_duplicate", "when": "duplicate and refund_percent <= 50", "then": "refund"},
    ],
    "otherwise": "review",
}


def direct(message, runtime):
    r = ACTION(message, runtime)
    guess = r.prediction
    return {"status": r.status, "value": r.value,
            "prediction": None if guess is None else {"kind": guess.kind, "key": guess.key, "score": guess.score},
            "direct": None if guess is None or guess.kind != "value" else guess.key,
            "accepted": r.value if r.status == "ok" else None}


def routed(message, refund, paid, runtime):
    with TemporaryDirectory(prefix="semantic-switch-") as directory:
        path = Path(directory) / "refund_route.yaml"
        path.write_text(yaml.safe_dump(ROUTE), encoding="utf-8")
        r = runtime.load(path)({"message": message, "refund_percent": 100 * refund / paid})
    return {"status": r.status, "answer": r.answer, "case": r.case,
            "facts": {name: {"status": fact.status, "value": fact.value,
                             "prediction": None if fact.prediction is None else fact.prediction.value}
                      for name, fact in r.facts.items()}}


def run(runtime):
    message = "I was charged twice for order 7. Please fix it."
    return {"direct": direct(message, runtime), "routed": routed(message, 20.0, 80.0, runtime)}


if __name__ == "__main__":
    with ss.Runtime() as runtime:
        print(json.dumps(run(runtime), indent=2))
