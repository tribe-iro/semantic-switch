"""Claim versus question wording of the same checks on the bundled and fresh suites. Run: python -m bench.question_form --help."""
from __future__ import annotations

import argparse
from pathlib import Path

from semantic_switch import ConfigError, DEFAULT_JUDGE, files
from bench import quality
from bench.fresh import FRESH
from bench.outcomes import judged

QUESTIONS = {
    "refund": "Is the customer asking to get money back for a charge?",
    "legal": "Is the customer threatening legal action against us?",
    "cancel": "Does the customer want to cancel their subscription?",
    "human": "Is the customer asking to talk to a human agent?",
    "broken": "Is the customer reporting that the product is not working?",
    "late": "Is the customer complaining that an order has not arrived?",
    "solved": "Does the customer say their problem is now solved?",
    "erase": "Is the customer asking us to delete their personal data?",
    "price": "Is the customer asking how much something costs?",
    "today": "Does the customer say they need a solution today?",
    "meeting": "Does the writer accept the proposed meeting time?",
    "leave": "Is the employee asking for time off?",
    "incident": "Is the sender reporting a security incident?",
    "heating": "Does the tenant say the heating is not working?",
    "pain": "Does the patient say the pain has gotten worse?",
    "shipped": "Does the supplier confirm the shipment has left the warehouse?",
    "decline": "Is the writer declining the job offer?",
    "address": "Does the customer want to change the delivery address?",
}
SUITES = {"bundled": quality.CHECKS, "fresh": FRESH}


def asked(spec: dict) -> dict:
    return {fact: (QUESTIONS[fact], excludes, pairs) for fact, (_, excludes, pairs) in spec.items()}


def run(rt) -> dict:
    out = {}
    for suite, spec in SUITES.items():
        for form, checks in (("claim", spec), ("question", asked(spec))):
            result = quality.run(rt, checks_spec=checks, picks_spec=[])
            out[f"{suite}:{form}"] = {"summary": quality.summarize(result), "checks": result["checks"]}
    return out


def paired(claim: list, question: list) -> dict:
    before, after = ({r["id"]: r for r in rows} for rows in (claim, question))
    if len(before) != len(claim) or len(after) != len(question) or before.keys() != after.keys() or \
            any((before[i]["label"], before[i]["input"]) != (after[i]["label"], after[i]["input"]) for i in before):
        raise ConfigError("paired wording arms need the same unique item IDs, gold and evidence")
    scored = lambda r: r["p"] is not None and not r.get("error")
    failed = lambda r: not scored(r)
    right = lambda r: scored(r) and (r["p"] >= 0.5) == r["label"]
    both = [i for i in before if scored(before[i]) and scored(after[i])]
    moved = lambda a, b, rows: {"correct": sum(right(rows[i]) for i in before if a(before[i]) and b(after[i])),
                                "incorrect": sum(not right(rows[i]) for i in before if a(before[i]) and b(after[i]))}
    return {"items": len(before), "claim_correct": sum(map(right, before.values())), "question_correct": sum(map(right, after.values())),
            "both_scored": len(both), "fixes": sum(right(after[i]) and not right(before[i]) for i in both),
            "regressions": sum(right(before[i]) and not right(after[i]) for i in both),
            "scored_to_failed": moved(scored, failed, before), "failed_to_scored": moved(failed, scored, after),
            "failed_to_failed": sum(failed(before[i]) and failed(after[i]) for i in before)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="local GGUF (path or repo:file) or openrouter:MODEL")
    parser.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL")
    parser.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    parser.add_argument("--device")
    parser.add_argument("--output", default="bench/results/question-form.json")
    args = parser.parse_args()
    with judged(args.judge, provider=args.provider, key_file=args.key_file, device=args.device) as (rt, recording):
        arms = run(rt)
        files.publish(args.output, {"schema": 3, "judge": args.judge, "model": rt.identity(), "execution": rt.execution(),
                                    "returned_models": sorted(getattr(rt.judge(), "observed_models", ())),
                                    "requests": recording.summary() if recording else None, "arms": arms,
                                    "paired": {suite: paired(arms[f"{suite}:claim"]["checks"], arms[f"{suite}:question"]["checks"]) for suite in SUITES}})
    for name, arm in arms.items():
        s = arm["summary"]
        print(f"{name:18} acc|predicted {s['accuracy_given_prediction']}  auc {s['auc']}  cov@.9 {s['coverage_0.9']}  prec@.9 {s['precision_0.9']}  "
              f"low_mass {s['low_mass']}  ms {s['ms_p50']}")


if __name__ == "__main__":
    main()
