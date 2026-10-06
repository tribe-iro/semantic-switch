from __future__ import annotations

import json
from pathlib import Path

from .. import files
from ..compiler import FactPlan
from ..errors import ConfigError
from ..examples import label_record
from ..switch import Switch


def parse(fact: FactPlan, answer: str):
    if fact.kind == "check":
        return fact.value(fact.index(answer))
    value = fact.keys[int(answer)] if answer.isdigit() and int(answer) < len(fact.keys) else answer
    return fact.value(fact.index(value))


def _prompt(fact: FactPlan) -> str:
    choices = ("y/n/u" if fact.unstated else "y/n") if fact.kind == "check" else "  ".join(f"{k}={v}" for k, v in enumerate(fact.keys))
    return f"{fact.text}\n  [{choices}, Enter skips]: "


def _ask_value(fact: FactPlan, ask, show, prompt: str):
    while True:
        answer = ask(prompt).strip()
        if answer in ("", "s"):
            return None
        try:
            return parse(fact, answer)
        except ConfigError as e:
            show(f"  {e}; try again")


def label(rt, path, inputs_file: str, to: str = "examples", stale: bool = False, ask=input, show=print) -> int:
    sw = Switch(rt, path, draft=True)
    plan = sw.plan
    written = 0
    if stale:
        current = {f.name: f.meaning_rev for f in plan.visible}
        for rec in files.read_jsonl(sw.paths.examples):
            if rec["fact"] in current and rec["meaning_rev"] != current[rec["fact"]]:
                fact = plan.fact(rec["fact"])
                show(json.dumps(rec["input"], ensure_ascii=False))
                value = _ask_value(fact, ask, show, f"{fact.text}\n  was {rec['value']!r}; type the current answer: ")
                if value is not None:
                    sw.correct(rec["input"], fact=fact.name, value=value)
                    written += 1
        return written
    for item in files.read_jsonl(Path(inputs_file)):
        data = item.get("input", item)
        show(json.dumps(data, ensure_ascii=False))
        labels = {f.name: v for f in plan.visible if not f.cards and (v := _ask_value(f, ask, show, _prompt(f))) is not None}
        if not labels:
            continue
        if to == "test":
            files.append_jsonl(sw.paths.test, [label_record(plan, data, labels)])
        else:
            for name, value in labels.items():
                sw.correct(data, fact=name, value=value)
        written += 1
    return written
