from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..context import canonical, project, render
from ..errors import ConfigError
from ..examples import partial
from ..switch import Switch
from .common import input_key, split_of, test_items


def labels(sw: Switch) -> dict:
    plan = sw.plan
    items, _ = test_items(sw.paths, plan)
    merged: dict = {}
    records = [(e.input, e.fact, e.value, None) for exs in sw.snapshot.examples.values() for e in exs]
    records += [(it["input"], n, v, it.get("group")) for it in items for n, v in it.get("facts", {}).items()]
    for data, name, value, group in records:
        fact = plan.fact(name)
        if fact.cards:
            continue
        values = project(plan.fields, sw.bind(data, sw.snapshot))
        key = (name, canonical(partial(values, fact.reads), fact.reads))
        label = fact.index(value)
        if key in merged and merged[key]["label"] != label:
            raise ConfigError(f"{name}: conflicting labels {fact.value(merged[key]['label'])!r} and {value!r} "
                              f"for the same input {key[1]}")
        entry = merged.setdefault(key, {"data": data, "values": values, "label": label, "group": None})
        entry["group"] = entry["group"] or group
    return merged


def export(rt, path, to: str | None = None) -> dict:
    sw = Switch(rt, path, draft=True)
    plan = sw.plan
    out = Path(to) if to else sw.paths.yaml.with_suffix(".export.jsonl")
    sources = {sw.paths.yaml, sw.paths.examples, sw.paths.test, sw.paths.lock}
    if out.resolve() in {s.resolve() for s in sources}:
        raise ConfigError(f"{out} is one of the switch's own files; export elsewhere")
    adapter = rt.judge().adapter
    identity = rt.identity("judge")
    rows = []
    for (name, _), entry in labels(sw).items():
        fact = plan.fact(name)
        request = adapter.request(adapter.state(render(plan.fields, entry["values"], fact.reads)),
                                  [fact.judgment.question()])
        data, group = entry["data"], entry["group"]
        rows.append({"judge": identity, "template": adapter.name, "prefix": list(request.prefix),
                     "input": list(request.state), "question": list(request.suffixes[0]),
                     "answer_slots": [list(s) for s in request.slots[0]], "label": entry["label"],
                     "fact": name, "meaning_rev": fact.meaning_rev, "decision_key": input_key(plan, data),
                     "group": group, "split": split_of(plan, {"input": data, "group": group})})
    fd, tmp = tempfile.mkstemp(dir=out.resolve().parent, prefix=f".{out.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, out)
    return {"rows": len(rows), "path": str(out)}
