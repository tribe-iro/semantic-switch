from __future__ import annotations

import semantic_switch as ss

INSTRUCTION = "What is the message mainly about?"
CATEGORIES = {"billing": "charges, invoices, payments or refunds",
              "delivery": "an order that is late, missing or damaged",
              "account": "login, profile or data access"}


def classify(message, propose, rt=None):
    result = ss.Pick(INSTRUCTION, CATEGORIES, allow_none=True)(message, rt)
    if result.status == "none" and result.complete:
        return result, propose(message, INSTRUCTION, dict(CATEGORIES))
    return result, None
