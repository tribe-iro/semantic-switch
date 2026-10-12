from __future__ import annotations

import argparse
import json

from semantic_switch import Guidance, Pick, Runtime


def main():
    parser = argparse.ArgumentParser(description="Interpret a finite list of reports under one maintenance policy.")
    parser.add_argument("reports", nargs="+", help="reports in result order")
    args = parser.parse_args()
    decision = Pick(
        "Which response does the policy require?",
        {"escalate": "Request technician attention", "continue": "Continue operation", "clarify": "Ask for evidence"},
        guidance=Guidance("Escalate unresolved overheating. Continue after a completed repair with normal operation. "
                          "If the report does not establish the current condition, ask for evidence."),
    )
    with Runtime() as runtime:
        for report, result in zip(args.reports, decision.run(args.reports, runtime)):
            print(json.dumps({"report": report, "result": result.as_dict()}))


if __name__ == "__main__":
    main()
