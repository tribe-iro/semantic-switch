"""Equipment benchmark: fictional procedure selection through the public SDK.

python -m bench.equipment --device gpu [--direct]                        # default and precise local judges
python -m bench.equipment --judge openrouter:MODEL --provider PROVIDER --key-file KEY --direct
python -m bench.equipment --judge openrouter:typesafe/jev-1.13 --provider PROVIDER --jev-consent   # only with TypeSafe's consent
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from semantic_switch import DEFAULT_JUDGE, PRECISE_JUDGE, files
from bench.outcomes import JEV, destination, failure, judged, latency, measure, output_argument, provenance
from bench.outcomes import Progress
from semantic_switch.results import thaw

NONE, MISSING, REVIEW = "NO_PROCEDURE", "NEEDS_INPUT", "REVIEW"
MACHINES = ("AX-410", "AX-410B")
CLAIMS = {
    "hot": "The report establishes current overheating or a persistent temperature alarm on this unit.",
    "leak": "There is a current physical coolant leak on the current unit.",
    "vibration": "The report establishes current abnormal vibration or grinding noise on this unit.",
    "no_start": "The report establishes current failure to start or restart on this unit.",
}
KINDS = (("thermal", "hot", "firmware_major < 3"), ("coolant", "hot", "firmware_major >= 3 and pressure_bar < 1.0"),
         ("sensor", "hot", "firmware_major >= 3 and pressure_bar >= 1.0"), ("bearing", "vibration", None),
         ("startup", "no_start", 'mentions(active_alarms, "E17")'))
T, F, U = True, False, None

REPORTS = [
    ("hot_clear", "trap:none", (T, F, F, F),
     "Persistent temperature alarm on this unit right now. No coolant leak, no odd noise, it runs.",
     "Alarme de temperatura persistente nesta unidade agora. Sem vazamento, sem ruído estranho, está funcionando."),
    ("hot_other_unit_advice", "other_unit", (T, F, F, F),
     "This unit keeps overheating. No leak here. I think the reset for the AX-4100 next to it is the fix.",
     "Esta unidade continua superaquecendo. Sem vazamento aqui. Acho que o reset da AX-4100 ao lado resolve."),
    ("hot_leak_unstated", "missing_observation", (T, U, F, F),
     "The temperature alarm on this unit has been on all morning and it feels hot to the touch.",
     "O alarme de temperatura desta unidade está ligado a manhã toda e ela está quente ao toque."),
    ("leak_and_hot", "precedence", (T, T, F, F),
     "Coolant is dripping from this unit now and the temperature alarm is on. Maybe reset the sensor?",
     "Está pingando líquido de arrefecimento desta unidade agora e o alarme de temperatura está ligado. Resetar o sensor?"),
    ("leak_only", "trap:none", (F, T, F, F),
     "There is a puddle of coolant under this unit and it is still dripping. Temperature is normal.",
     "Há uma poça de líquido de arrefecimento sob esta unidade e continua pingando. A temperatura está normal."),
    ("other_unit_leak", "other_unit", (F, F, F, F),
     "The AX-401B leaked yesterday. This unit has no leak and runs normally at normal temperature.",
     "A AX-401B vazou ontem. Esta unidade não vaza e funciona normalmente com temperatura normal."),
    ("other_unit_hot", "other_unit", (F, F, F, F),
     "The machine beside this one overheats constantly. This unit is fine: normal temperature, no leak.",
     "A máquina ao lado desta superaquece direto. Esta unidade está bem: temperatura normal, sem vazamento."),
    ("resolved_hot", "past_resolved", (F, F, F, F),
     "The temperature alarm was fixed yesterday. This unit operates normally now. No coolant leak.",
     "O alarme de temperatura foi resolvido ontem. Esta unidade funciona normalmente agora. Sem vazamento."),
    ("past_leak", "past_resolved", (F, F, F, F),
     "Last month this unit had a coolant leak that was repaired. Today it is dry and running cool.",
     "No mês passado esta unidade teve um vazamento que foi consertado. Hoje está seca e funcionando fria."),
    ("hypothetical_hot", "hypothetical", (F, F, F, F),
     "This unit runs normally with no leak. What should we do if it starts to overheat later?",
     "Esta unidade funciona normalmente, sem vazamento. O que fazer se ela começar a superaquecer depois?"),
    ("hypothetical_vibration", "hypothetical", (F, F, F, F),
     "No problems on this unit, no leak. If it ever starts grinding, which procedure would apply?",
     "Nenhum problema nesta unidade, sem vazamento. Se algum dia começar a ranger, qual procedimento se aplica?"),
    ("vibration_clear", "trap:none", (F, F, T, F),
     "This unit started vibrating hard and makes a grinding noise. Temperature normal, no leak.",
     "Esta unidade começou a vibrar forte e faz um barulho de rangido. Temperatura normal, sem vazamento."),
    ("vibration_leak_unstated", "missing_observation", (F, U, T, F),
     "Loud grinding from the gearbox side of this unit since the shift started.",
     "Rangido alto do lado da caixa de engrenagens desta unidade desde o começo do turno."),
    ("no_start", "trap:none", (F, F, F, T),
     "This unit will not start after the restart. No leak, it is cold.",
     "Esta unidade não liga depois do reinício. Sem vazamento, está fria."),
    ("no_start_operator_e17", "contradiction", (F, F, F, T),
     "This unit will not start and the operator says the panel showed E17. No leak.",
     "Esta unidade não liga e o operador diz que o painel mostrou E17. Sem vazamento."),
    ("runs_but_operator_says_dead", "contradiction", (F, F, F, F),
     "The operator wrote that it would not start, but it started on the second try and runs fine now. No leak.",
     "O operador escreveu que não ligava, mas ligou na segunda tentativa e funciona bem agora. Sem vazamento."),
    ("all_normal", "no_procedure", (F, F, F, F),
     "Routine check: this unit is running normally, no leak, no alarms, no noise.",
     "Verificação de rotina: esta unidade funciona normalmente, sem vazamento, sem alarmes, sem ruído."),
    ("cosmetic", "no_procedure", (F, F, F, F),
     "The paint on this unit's door is scratched. Everything else works and there is no leak.",
     "A pintura da porta desta unidade está arranhada. O resto funciona e não há vazamento."),
    ("hot_and_vibration", "precedence", (T, F, T, F),
     "This unit overheats and also vibrates abnormally. No coolant leak anywhere.",
     "Esta unidade superaquece e também vibra de forma anormal. Nenhum vazamento de líquido."),
    ("sensor_says_hot_operator_denies", "contradiction", (T, F, F, F),
     "The temperature alarm stays on even though the operator thinks it is a false alarm. No leak.",
     "O alarme de temperatura continua ligado, embora o operador ache que é alarme falso. Sem vazamento."),
    ("intermittent_hot_now", "trap:none", (T, F, F, F),
     "It overheated twice last week and the persistent temperature alarm is back on right now. No leak.",
     "Superaqueceu duas vezes semana passada e o alarme de temperatura persistente voltou agora. Sem vazamento."),
    ("leak_other_fluid_unit", "other_unit", (F, F, F, F),
     "Oil on the floor came from the forklift, not from this unit. This unit is dry and normal.",
     "O óleo no chão veio da empilhadeira, não desta unidade. Esta unidade está seca e normal."),
    ("no_start_resolved", "past_resolved", (F, F, F, F),
     "It failed to start this morning but maintenance fixed it; it now starts every time. No leak.",
     "Não ligou hoje de manhã, mas a manutenção consertou; agora liga sempre. Sem vazamento."),
]

CONTEXTS = {
    "fw2": {"firmware_major": 2, "pressure_bar": 0.9, "active_alarms": []},
    "fw3_low": {"firmware_major": 3, "pressure_bar": 0.99, "active_alarms": []},
    "fw3_boundary": {"firmware_major": 3, "pressure_bar": 1.0, "active_alarms": []},
    "fw3_missing_pressure": {"firmware_major": 3, "active_alarms": []},
    "e17": {"firmware_major": 3, "pressure_bar": 1.2, "active_alarms": ["E17"]},
}
PLAN = {
    "hot_clear": ("fw2", "fw3_low", "fw3_boundary"), "hot_other_unit_advice": ("fw3_low",),
    "hot_leak_unstated": ("fw2",), "leak_and_hot": ("fw3_boundary",), "leak_only": ("fw2",),
    "other_unit_leak": ("fw3_low",), "other_unit_hot": ("fw2",), "resolved_hot": ("fw3_low",),
    "past_leak": ("fw2",), "hypothetical_hot": ("fw3_low",), "hypothetical_vibration": ("fw2",),
    "vibration_clear": ("fw3_low",), "vibration_leak_unstated": ("fw2",), "no_start": ("e17", "fw2"),
    "no_start_operator_e17": ("fw3_low",), "runs_but_operator_says_dead": ("e17",), "all_normal": ("e17",),
    "cosmetic": ("fw2",), "hot_and_vibration": ("fw3_missing_pressure", "fw2"),
    "sensor_says_hot_operator_denies": ("fw3_boundary",), "intermittent_hot_now": ("fw3_missing_pressure",),
    "leak_other_fluid_unit": ("fw3_low",), "no_start_resolved": ("e17",),
}


def cards(shape: str):
    out = []
    claim = (lambda f: {"fact": f}) if shape == "facts" else (lambda f: CLAIMS[f])
    for m in MACHINES:
        mid = m.replace("-", "")
        base = f'machine == "{m}"'
        for kind, fact, cond in KINDS:
            out.append({"id": f"{mid}_{kind}", "aliases": [m], "requires": f"{base} and {cond}" if cond else base,
                        "applies_when": [claim(fact)], "excludes": [claim("leak")]})
        out.append({"id": f"{mid}_leak", "aliases": [m], "requires": base, "applies_when": [claim("leak")],
                    "supersedes": [f"{mid}_{k}" for k, _, _ in KINDS]})
    return out


def order(ids):
    rank = {k: n for n, k in enumerate(("leak", "thermal", "coolant", "sensor", "bearing", "startup"))}
    return sorted(ids, key=lambda i: rank[i.split("_", 1)[1]])


FIELDS = {"machine": {"type": "text", "required": True}, "firmware_major": {"type": "number"},
          "pressure_bar": {"type": "number"}, "active_alarms": {"type": "json", "source": "controller"},
          "report": {"type": "text", "required": True, "author": "operator"}}


def catalogue_spec(shape: str) -> dict:
    cs = cards(shape)
    facts = {"leak": {"check": CLAIMS["leak"], "unstated": True}} | {
        k: {"check": v} for k, v in CLAIMS.items() if k != "leak"} if shape == "facts" else {}
    facts["procedures"] = {"select": "Which maintenance procedures apply to this situation?", "cards": cs}
    return {"name": f"equipment_{shape}", "fields": FIELDS, "reads": ["report"], "facts": facts,
            "cases": [{"id": i, "when": f'procedures has "{i}"', "then": i} for i in order([c["id"] for c in cs])],
            "otherwise": NONE}


def cases_spec() -> dict:
    facts = {"leak": {"check": CLAIMS["leak"], "unstated": True}} | {
        k: {"check": v} for k, v in CLAIMS.items() if k != "leak"}
    cases = []
    for c in cards("facts"):
        fact = c["applies_when"][0]["fact"]
        sem = fact if fact == "leak" else f"{fact} and not leak"
        cases.append({"id": c["id"], "when": f"({c['requires']}) and {sem}", "then": c["id"]})
    order_of = order([c["id"] for c in cases])
    return {"name": "equipment_cases", "fields": FIELDS, "reads": ["report"], "facts": facts,
            "cases": sorted(cases, key=lambda c: order_of.index(c["id"])), "otherwise": NONE}


def expected(machine: str, ctx: dict, gold) -> str:
    hot, leak, vib, nostart = gold
    if machine not in MACHINES:
        return NONE
    mid = machine.replace("-", "")
    if leak is True:
        return f"{mid}_leak"
    if hot:
        if leak is None:
            return MISSING
        if ctx["firmware_major"] < 3:
            return f"{mid}_thermal"
        if "pressure_bar" not in ctx:
            return MISSING
        return f"{mid}_coolant" if ctx["pressure_bar"] < 1.0 else f"{mid}_sensor"
    if vib:
        return MISSING if leak is None else f"{mid}_bearing"
    if nostart and "E17" in ctx["active_alarms"]:
        return f"{mid}_startup"
    return NONE


def dataset() -> list[dict]:
    rows = []
    for rid, trap, gold, en, pt in REPORTS:
        for n, ctx_name in enumerate(PLAN[rid]):
            machine = MACHINES[n % 2]
            for lang, text in (("en", en), ("pt", pt)):
                ctx = CONTEXTS[ctx_name]
                rows.append({"id": f"{rid}:{ctx_name}:{machine}:{lang}", "trap": trap, "lang": lang, "group": rid,
                             "input": {"machine": machine, **ctx, "report": text},
                             "expected": expected(machine, ctx, gold)})
    for lang, text in zip(("en", "pt"), REPORTS[0][3:]):
        rows.append({"id": f"near_identical:AX-410BX:{lang}", "trap": "near_identical_machine", "lang": lang,
                     "group": REPORTS[0][0], "input": {"machine": "AX-410BX", **CONTEXTS["fw2"], "report": text},
                     "expected": NONE})
    return rows


def outcome(r) -> str:
    return r.answer if r.status == "ok" else MISSING if r.status == "needs_input" else \
        REVIEW if r.status == "uncertain" else "FAILED"


def summarize(rows) -> dict:
    act = [r for r in rows if r["got"] not in (REVIEW, MISSING, "FAILED")]
    return {"items": len(rows), "correct": sum(r["got"] == r["expected"] for r in rows),
            "wrong_actions": sum(r["got"] != r["expected"] for r in act),
            "abstained": sum(r["got"] in (REVIEW, "FAILED") for r in rows), "failed": sum(r.get("outcome") == "failed" for r in rows),
            "needs_input_correct": sum(r["got"] == r["expected"] == MISSING for r in rows),
            "no_model_call": sum(r["judged"] == 0 for r in rows),
            "latency": latency(r["ms"] for r in rows),
            "by_trap": {t: f"{sum(r['got'] == r['expected'] for r in rows if r['trap'] == t)}/"
                           f"{sum(r['trap'] == t for r in rows)}" for t in sorted({r["trap"] for r in rows})}}


def with_theta(spec: dict, theta: float | None) -> dict:
    if theta is None:
        return spec
    return {**spec, "facts": {k: {**v, "min_score": theta} if "check" in v else v for k, v in spec["facts"].items()}}


def at_equal_error(sweep: dict, budget: int) -> dict:
    ok = [(s["correct"], th) for th, s in sweep.items() if s["wrong_actions"] <= budget]
    best = max(ok, key=lambda x: x[0], default=(None, None))
    return {"wrong_action_budget": budget, "correct": best[0], "theta": best[1]}


DIRECT = ("Choose the single applicable maintenance procedure for the current unit. Structured fields are "
          "authoritative; operator guesses and other units are not observations about the current unit. Respect every "
          "hard eligibility rule. A current leak takes precedence. Choose NO_PROCEDURE if none applies and NEEDS_INPUT "
          "if a required observation is missing.")


def direct_options() -> dict:
    out = {}
    for c in cards("facts"):
        when = " ".join(CLAIMS[x["fact"]] for x in c["applies_when"])
        unless = ", ".join(CLAIMS[x["fact"]] for x in c.get("excludes", ())) or "none"
        out[c["id"]] = f"Eligible only if {c['requires']}. Applies when: {when} Unless: {unless}"
    return out | {NONE: "No procedure applies.", MISSING: "A required observation is missing."}


def run_switch(rt, path: Path, rows: list) -> list:
    sw = rt.load(path, draft=True)
    sw(rows[0]["input"])
    sequential = [sw(row["input"]) for row in rows]
    batched = sw.run([row["input"] for row in rows])
    def public_outcome(result):
        values = result.as_dict(True)
        values.pop("cost", None)
        return values
    def equal(left, right):
        if isinstance(left, float) and isinstance(right, (float, int)):
            return abs(left - right) <= 1e-5
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right
    parity = [equal(public_outcome(a), public_outcome(b)) for a, b in zip(sequential, batched)]
    def differences(left, right):
        if isinstance(left, float) and isinstance(right, (float, int)):
            return abs(left - right), True
        if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            values = [differences(left[k], right[k]) for k in left]
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
            values = [differences(a, b) for a, b in zip(left, right)]
        else:
            return 0., left == right
        return max((value[0] for value in values), default=0.), all(value[1] for value in values)
    deltas = [differences(public_outcome(a), public_outcome(b)) for a, b in zip(sequential, batched)]
    return [{**row, "got": outcome(result), "judged": result.cost["judge"],
             "ms": result.cost["elapsed_ms"], "status": result.status,
             "criteria": CLAIMS, "effective_definition": thaw(sw.plan.spec),
             "timing": "runtime batch-amortized inference; not serialized end-to-end latency",
             "public": public_outcome(result), "batch_public": public_outcome(batch),
             "stream_batch_equal": same, "stream_batch_discrete_equal": discrete,
             "stream_batch_max_numeric_delta": delta}
            for row, result, batch, same, (delta, discrete) in zip(rows, sequential, batched, parity, deltas)]


def run_direct(rt, rows: list) -> tuple[list, dict]:
    options = direct_options()
    out, progress = [], Progress("bench.equipment direct", len(rows), unit="items")
    for row in rows:
        measured = measure(lambda: rt.pick(row["input"], DIRECT, options), runtime=rt)
        r = measured.result
        out.append({**row, **failure(measured, got="FAILED", judged=1, status="failed", result=None, criterion=DIRECT, options=options,
                                     presentations=[], slot_map=[], complete=False, scores=None)} if measured.error else
                   {**row, "got": r.index if r.status == "ok" else REVIEW, "judged": 1, "outcome": "predicted",
                    "ms": measured.ms, "native_work": measured.work, "status": r.status, "result": measured.output,
                    "criterion": DIRECT, "options": options,
                    "presentations": r.as_dict(True)["presentations"],
                    "slot_map": list(r.scores), "complete": r.complete, "scores": r.scores})
        progress.record([{"correct": out[-1]["got"] == row["expected"], "accepted": out[-1]["status"] == "ok", "outcome": out[-1]["outcome"]}])
    return out, progress.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="append", help=f"repeatable: local GGUF (path or repo:file) or openrouter:MODEL; {JEV} needs --jev-consent")
    ap.add_argument("--provider", help="OpenRouter provider route for openrouter:MODEL judges")
    ap.add_argument("--key-file", type=Path, help="OpenRouter API key file (default: OPENROUTER_API_KEY)")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--thetas", default="0.6,0.7,0.8,0.9,0.95,0.99")
    ap.add_argument("--direct", action="store_true", help="also let the judge choose from the whole catalogue")
    ap.add_argument("--jev-consent", action="store_true")
    output_argument(ap)
    args = ap.parse_args()
    destination(args.output)
    judges = args.judge or [DEFAULT_JUDGE, PRECISE_JUDGE]
    if JEV in judges and not args.jev_consent:
        ap.error("Jev runs need TypeSafe's consent: pass --jev-consent once it is given")
    rows = dataset()
    shapes = {"catalogue_facts": catalogue_spec("facts"), "catalogue_strings": catalogue_spec("strings"),
              "cases": cases_spec()}
    thetas = [float(x) for x in args.thetas.split(",") if x.strip()]
    report = {"boundary": "completed public Switch operations; fictional development cohort",
              "threshold_sweep": "exploratory; not held-out deployed-policy selection", "fictional": True, "reports": len({r["input"]["report"] for r in rows}), "items": len(rows),
              "schema": 2, "facts": list(CLAIMS), "cards": len(shapes["catalogue_facts"]["facts"]["procedures"]["cards"]),
              "requested_device": args.device, "results": {}}
    with tempfile.TemporaryDirectory(prefix="equipment-") as tmp:
        for model in judges:
            with judged(model, provider=args.provider, key_file=args.key_file, device=args.device) as (rt, receipts):
                results = report["results"].setdefault(model, {})
                for shape, spec in shapes.items():
                    sweep = {}
                    for theta in [None] + (thetas if shape != "catalogue_strings" else []):
                        path = Path(tmp) / f"{shape}-{theta}.yaml"
                        files.publish(path, with_theta(spec, theta))
                        got = run_switch(rt, path, rows)
                        key = "default" if theta is None else theta
                        sweep[key] = summarize(got)
                        if theta is None:
                            results[shape] = {"default": sweep[key], "rows": got}
                        print(model, shape, key, json.dumps({k: v for k, v in sweep[key].items() if k != "by_trap"}),
                              flush=True)
                    results[shape]["sweep"] = sweep
                    files.publish(args.output, report)
                if args.direct:
                    got, progress = run_direct(rt, rows)
                    results["direct"] = {"default": summarize(got), "rows": got, "sweep": {"default": summarize(got)}, "progress": progress}
                    print(model, "direct", json.dumps(results["direct"]["default"]), flush=True)
                report.setdefault("provenance", {})[model] = provenance(rt, rows)
                files.publish(args.output, report)
                report.setdefault("requests", {})[model] = receipts.summary()
                files.publish(args.output, report)
    budget = min(s["wrong_actions"] for m in report["results"].values() for sh in m.values()
                 for s in sh["sweep"].values())
    for m in report["results"].values():
        for sh in m.values():
            sh["at_equal_error"] = at_equal_error(sh["sweep"], budget)
    files.publish(args.output, report)
    print(json.dumps({m: {sh: {"default": {k: v for k, v in d["default"].items() if k != "by_trap"},
                               "at_equal_error": d["at_equal_error"]} for sh, d in r.items()}
                      for m, r in report["results"].items()}, indent=1))


if __name__ == "__main__":
    main()
