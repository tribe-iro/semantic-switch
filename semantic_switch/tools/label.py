from __future__ import annotations

import getpass
import json
from pathlib import Path

from .. import files
from ..compiler import FactPlan
from ..errors import ConfigError
from ..evidence import Actor, Claim, LabelSource, Verification, decode, label_record, labelled, record_path, reviewed_content, sources_at, successor

REVIEW_SCHEMA = 2


def parse(fact: FactPlan, answer: str):
    if fact.kind == "check":
        return fact.value(fact.index(answer))
    value = fact.keys[int(answer)] if answer.isdigit() and int(answer) < len(fact.keys) else answer
    return fact.value(fact.index(value))


def _prompt(fact: FactPlan, supplied=None) -> str:
    choices = ("y/n/u" if fact.unstated else "y/n") if fact.kind == "check" else "  ".join(f"{k}={v}" for k, v in enumerate(fact.keys))
    shown = f"{fact.text}\n  Candidate: {fact.judgment.candidate}" if fact.member else fact.text
    if supplied is not None:
        return f"{shown}\n  Supplied: {supplied!r} [Enter confirms, s skips, {choices} replaces]: "
    return f"{shown}\n  [{choices}, Enter skips]: "


def _answer(fact: FactPlan, ask, show, prompt: str, supplied):
    while True:
        answer = ask(prompt).strip()
        if answer == "" and supplied is not None:
            return "confirm", fact.value(fact.index(supplied))
        if answer in ("", "s"):
            return "skip", None
        try:
            return "replace", parse(fact, answer)
        except ConfigError as e:
            show(f"  {e}; try again")


def _nest(labels: dict, fact: FactPlan, value):
    if fact.member:
        labels.setdefault(fact.path[0], {})[fact.path[1]] = value
    else:
        labels[fact.name] = value


def review(plan, record, reviewer, actor, ask, show) -> tuple[dict, dict, list]:
    labels, sources, skipped = {}, {}, []
    supplied = record.get("facts")
    pending = [pair for name, value in supplied.items() for pair in labelled(plan, name, value)] if supplied else [(plan.facts[f], None) for f in plan.observed]
    for fact, given in pending:
        outcome, value = _answer(fact, ask, show, _prompt(fact, given), given)
        if outcome == "replace" and given is not None and value == fact.value(fact.index(given)):
            outcome = "confirm"
        history = list(sources_at(record["label_sources"], fact.path)) if supplied else []
        if given is not None:
            entry = Verification("label", reviewer, actor, "review", fact.meaning_rev, reviewed_content("label", record["input"], fact.path, given), outcome)
            if outcome == "skip":
                skipped.append({"schema": REVIEW_SCHEMA, "input": record["input"], "address": list(fact.path), "value": given,
                                "verification": entry.record(), "label_sources": [LabelSource.parse(s).record() for s in history]})
                continue
            history[-1] = LabelSource.parse(history[-1]).verified(entry)
            history = list(successor([Claim.parse(given, history)], value, LabelSource("authored", reviewer, actor=actor)).history) if outcome == "replace" else history
        elif outcome == "skip":
            continue
        else:
            history = [LabelSource("authored", reviewer, actor=actor)]
        _nest(labels, fact, value)
        _nest(sources, fact, history)
    return labels, sources, skipped


def label(rt, path, inputs_file: str | None = None, to: str = "examples", stale: bool = False, ask=input, show=print, reviewer: str | None = None, *, actor: Actor) -> int:
    validate(inputs_file, stale)
    actor = Actor.parse(actor)
    if actor.kind == "unknown":
        raise ConfigError("ss label records who decides: a human, or an agent with its model and session")
    if to not in ("examples", "test") and not str(to).endswith(".jsonl"):
        raise ConfigError(f"label destination must be examples, test or a label file ending in .jsonl, not {to!r}")
    reviewer = reviewer or getpass.getuser()
    sw = rt.load(path, draft=True)
    plan = sw.plan
    if stale:
        return _stale(sw, ask, show, reviewer, actor)
    if to not in ("examples", "test") and Path(to).resolve() in {p.resolve() for p in (sw.paths.examples, sw.paths.test, sw.paths.yaml, Path(inputs_file))}:
        raise ConfigError("label file destination must differ from the inputs and the switch files; use examples or test for those")
    written = 0
    destination = sw.paths.test if to == "test" else sw.paths.examples if to == "examples" else Path(to)
    reviews = destination.with_suffix(".reviews.jsonl")
    for line, item in files.iter_records(Path(inputs_file), with_lines=True):
        try:
            record = decode(plan, item, labelled=isinstance(item, dict) and "facts" in item)
            show(json.dumps(record["input"], ensure_ascii=False))
            labels, sources, skipped = review(plan, record, reviewer, actor, ask, show)
            files.append(reviews, files.jsonl(skipped))
            if not labels:
                continue
            if to == "examples":
                if record.get("role") is not None:
                    raise ConfigError(f"role {record['role']!r} belongs to label files; label with --to test or --to FILE.jsonl to keep it")
                for name, value in labels.items():
                    for fact, member in labelled(plan, name, value):
                        decided = dict(sources=sources_at(sources, fact.path)) if record.get("facts") else dict(author=reviewer, actor=actor)
                        sw.correct(record["input"], fact=name, value={fact.path[1]: member} if fact.member else member, selections=record["selections"],
                                   groups=record["groups"], pairs=record["pairs"], **decided)
            else:
                files.append(destination, files.jsonl([label_record(plan, record["input"], labels, sources=sources, groups=record["groups"], pairs=record["pairs"],
                                                                    role=record.get("role"), tags=record.get("tags", ()), selections=record["selections"])]))
            written += 1
        except ConfigError as error:
            raise ConfigError(f"{inputs_file}:{line}: {error}") from None
    return written


def validate(inputs_file, stale: bool) -> None:
    given = inputs_file is not None and bool(str(inputs_file).strip())
    if stale and given:
        raise ConfigError("ss label --stale relabels the switch's stale corrections and takes no INPUTS")
    if not stale and not given:
        raise ConfigError("ss label needs INPUTS (a JSONL file of situations or labels) unless --stale is given")


def _stale(sw, ask, show, reviewer, actor) -> int:
    plan, written = sw.plan, 0
    for rec in files.read_jsonl(sw.paths.examples):
        try:
            fact = plan.at(*record_path(rec))
        except ConfigError:
            continue
        if rec["meaning_rev"] == fact.meaning_rev:
            continue
        show(json.dumps(rec["input"], ensure_ascii=False))
        outcome, value = _answer(fact, ask, show, _prompt(fact), None)
        if outcome == "skip":
            continue
        show(f"previous label: {rec['value']!r}")
        sw.correct(rec["input"], fact=fact.path[0], value={fact.path[1]: value} if fact.member else value, groups=rec.get("groups", ()), pairs=rec.get("pairs", ()),
                   selections=rec.get("selections", ()), author=reviewer, actor=actor, predecessors=[Claim.parse(rec["value"], rec["label_sources"])])
        written += 1
    return written
