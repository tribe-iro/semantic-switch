from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import sys

from semantic_switch import ConfigError, files
from semantic_switch.metrics import identity
from bench import evidence
from bench.evidence import Assertion, Decision, Frame, Gold, Prose

DATA = Path(__file__).parent / "data" / "zt411"
PACK, DEFINITION = DATA / "pack.json", DATA / "printer.yaml"
REVISION = "zt411-reference-1"
ACTIONS = {"set_method_tt": "Set print method to thermal transfer", "set_method_dt": "Set print method to direct thermal",
           "load_ribbon": "Load a ribbon", "start_calibration": "Start manual sensor calibration",
           "verify_loading": "Inspect and verify media and ribbon loading", "calibration_next": "Confirm the calibration prompt and continue",
           "cancel_calibration": "Cancel calibration", "resume_printing": "Return to printing",
           "request_observation": "Ask the operator for the missing or conflicting observation"}
MENU = tuple(ACTIONS)
SCOPE = tuple(f"actions/{action}" for action in MENU)
PRIMITIVES = ("tt_media", "dt_media", "ribbon_loaded", "ribbon_absent", "detection_symptom", "step_verified", "unverified_step",
              "calibration_verified", "method_inconsistent", "ready_to_print", "needs_observation")
FIELDS = ("asset", "model", "print_method", "prompt", "enabled", "alerts", "verified", "attempted", "notes")
IDLE, CALIBRATION = ("ready", "paused"), ("calibration_1", "calibration_2")
TT, DT = "thermal_transfer", "direct_thermal"
ABSENT, CONFLICT = "absent", "conflict"
STRUCTURED = {"configuration": "print_method", "display": "prompt", "reading": "alerts", "verification": "verified", "command": "attempted"}
TRAINED = "structured"
CONDITIONS = ("structured", "prose", "structured_permuted")
LANGUAGES = ("en", "pt")
PHRASES = {
    "en": {("media_type", TT): ["loaded thermal-transfer labels", "put in a roll of thermal-transfer media"],
           ("media_type", DT): ["loaded direct-thermal labels", "put in a roll of direct-thermal media"],
           ("ribbon_loaded", True): ["a ribbon is loaded", "the ribbon is in"],
           ("ribbon_loaded", False): ["there is no ribbon loaded", "the ribbon spindle is empty"],
           ("detection_symptom", True): ["labels are skipping and it stops mid-roll", "it reports media out although labels are loaded"],
           ("step", "calibration_1"): ["did the first calibration step on the prompt", "finished what the first calibration prompt asked"],
           ("step", "calibration_2"): ["did the second calibration step on the prompt", "finished what the second calibration prompt asked"],
           ("calibration_complete", True): ["finished the manual sensor calibration", "completed the sensor calibration"]},
    "pt": {("media_type", TT): ["coloquei etiquetas de transferência térmica", "pus um rolo de mídia de transferência térmica"],
           ("media_type", DT): ["coloquei etiquetas térmicas diretas", "pus um rolo de mídia térmica direta"],
           ("ribbon_loaded", True): ["há um ribbon instalado", "o ribbon está colocado"],
           ("ribbon_loaded", False): ["não há ribbon instalado", "o eixo do ribbon está vazio"],
           ("detection_symptom", True): ["as etiquetas estão pulando e ela para no meio do rolo", "ela indica falta de mídia embora haja etiquetas"],
           ("step", "calibration_1"): ["fiz a primeira etapa da calibração indicada na tela", "terminei o que a primeira tela de calibração pediu"],
           ("step", "calibration_2"): ["fiz a segunda etapa da calibração indicada na tela", "terminei o que a segunda tela de calibração pediu"],
           ("calibration_complete", True): ["terminei a calibração manual dos sensores", "concluí a calibração dos sensores"]}}
WORDS = {"en": {TT: "thermal transfer", DT: "direct thermal", "ready": "ready", "paused": "paused", "calibration_1": "calibration 1",
                "calibration_2": "calibration 2", "stale": "Earlier shift: ", "method": "Display shows print method {}.", "prompt": "Display prompt: {}.",
                "alert": "Controller alert {}.", "command": "Pressed {}; not checked yet.", "step": "Checked: {} step is complete.",
                "complete": "Checked: calibration completed on the display.", "checked": "Checked: {}.", "I": "I {}.", "none": "No operator notes.", "controller": "controller"},
         "pt": {TT: "transferência térmica", DT: "térmica direta", "ready": "pronta", "paused": "pausada", "calibration_1": "calibração 1",
                "calibration_2": "calibração 2", "stale": "Turno anterior: ", "method": "O visor mostra o método de impressão {}.",
                "prompt": "Mensagem no visor: {}.", "alert": "Alerta do controlador {}.", "command": "Pressionei {}; ainda não conferi.",
                "step": "Conferido: a etapa {} está concluída.", "complete": "Conferido: calibração concluída no visor.", "checked": "Conferido: {}.",
                "I": "Eu {}.", "none": "Sem notas do operador.", "controller": "controlador"}}
ACTION_TEXT = {"pt": {"set_method_tt": "definir o método de impressão como transferência térmica", "set_method_dt": "definir o método de impressão como térmica direta",
                      "load_ribbon": "colocar um ribbon", "start_calibration": "iniciar a calibração manual dos sensores",
                      "verify_loading": "inspecionar e conferir a mídia e o ribbon", "calibration_next": "confirmar a tela de calibração e continuar",
                      "cancel_calibration": "cancelar a calibração", "resume_printing": "voltar a imprimir",
                      "request_observation": "pedir ao operador a observação ausente ou conflitante"}}
NAMES = ("Ana", "Bruno", "Carla", "Diego", "Elena", "Farid")


@dataclass(frozen=True)
class State:
    asset: str
    model: str
    media: str
    ribbon: bool
    method: str
    prompt: str
    symptom: bool = False
    step_done: bool = False
    calibrated: bool = False


@dataclass(frozen=True)
class Event:
    time: int
    indication: str
    actor: str
    asset: str
    property: str
    value: object
    status: str = "current"
    phrase: int = 0


def alert(value):
    return {("ribbon_loaded", False): "RIBBON OUT"}[value]


def render(scenario: str, state: State, events, *, condition: str, frame_id: str, family: str, tags=(), seeds=(), pairs=(), language="en") -> Frame:
    if condition not in CONDITIONS or language not in LANGUAGES:
        raise ConfigError(f"condition must be one of {CONDITIONS} and language one of {LANGUAGES}")
    structured = condition.removesuffix("_permuted") == "structured"
    data = {"asset": state.asset, "model": state.model}
    notes, assertions, lists = Prose("notes"), [], {"alerts": [], "verified": [], "attempted": []}
    for n, e in enumerate(sorted(events, key=lambda e: (e.time, e.indication, e.property))):
        key = f"a{n}"
        clock = f"{8 + e.time // 60:02d}:{e.time % 60:02d}"
        if structured and e.indication in STRUCTURED and e.asset == state.asset and e.status == "current":
            target = STRUCTURED[e.indication]
            if target in lists:
                lists[target].append(alert((e.property, e.value)) if e.indication == "reading" else e.value if e.indication == "command" else
                                     f"{e.property}:{json.dumps(e.value)}")
            else:
                data[target] = e.value
            assertions.append(Assertion(key, e.asset, e.property, e.value, f"e{n}", e.time, e.indication, e.status, paths=(target,)))
            continue
        notes.add(f"[{clock} {e.actor or WORDS[language]['controller']}, {e.asset}] ")
        notes.add(sentence(e, language), key)
        notes.add(" ")
        spans = tuple(notes.spans[key])
        assertions.append(Assertion(key, e.asset, e.property, e.value, f"e{n}", e.time, e.indication, e.status, spans=spans))
    if structured:
        data |= {"enabled": list(enabled(data.get("prompt"))), **lists}
    data["notes"] = notes.text.rstrip() or WORDS[language]["none"]
    return Frame(frame_id, family, scenario, condition, data, tuple(assertions), tuple(sorted(tags)), tuple(seeds), tuple(pairs))


def enabled(prompt) -> tuple[str, ...]:
    if prompt in CALIBRATION:
        return "calibration_next", "cancel_calibration"
    return ("set_method_tt", "set_method_dt", "load_ribbon", "start_calibration", *(("resume_printing",) if prompt == "paused" else ())) if prompt in IDLE else ()


def sentence(e: Event, language="en") -> str:
    w, phrases = WORDS[language], PHRASES[language]
    stale = w["stale"] if e.indication == "handover_note" else ""
    if e.indication == "configuration":
        return w["method"].format(w[e.value])
    if e.indication == "display":
        return w["prompt"].format(w[e.value])
    if e.indication == "reading":
        return w["alert"].format(alert((e.property, e.value)))
    if e.indication == "command":
        return w["command"].format(ACTION_TEXT[language][e.value] if language in ACTION_TEXT else ACTIONS[e.value].lower())
    if e.indication == "verification":
        if e.property == "step":
            return w["step"].format(w[e.value])
        if e.property == "calibration_complete":
            return w["complete"]
        return w["checked"].format(phrases[(e.property, e.value)][0])
    text = phrases[(e.property, e.value)][e.phrase % 2]
    return stale + (w["I"].format(text) if e.property in ("media_type", "step", "calibration_complete") else f"{text[0].upper()}{text[1:]}.")


class View:
    def __init__(self, frame: Frame):
        self.asset = frame.input["asset"]
        self.values, self.evidence, self.attempted, self.reported, self.steps, self.verified = {}, {}, set(), set(), set(), set()
        self.ignored, self.stale = [], []
        for a in frame.assertions:
            if a.subject != self.asset:
                self.ignored.append(a.id)
            elif a.status != "current":
                self.stale.append(a.id)
            elif a.indication == "command":
                self.attempted.add(a.value)
            elif a.property == "step":
                (self.steps if a.indication == "verification" else self.reported).add(a.value)
            elif a.property == "calibration_complete" and a.indication != "verification":
                self.reported.add("calibration_complete")
            else:
                self.values.setdefault(a.property, set()).add(json.dumps(a.value))
                self.evidence.setdefault(a.property, []).append(a.id)
                if a.indication == "verification":
                    self.verified.add(a.property)

    def get(self, prop):
        found = self.values.get(prop, set())
        return json.loads(next(iter(found))) if len(found) == 1 else CONFLICT if found else ABSENT


def consistency(method, media, ribbon):
    if media in (ABSENT, CONFLICT):
        return None, ("media_type",)
    if method != media:
        return False, ()
    if media == DT:
        return True, ()
    if ribbon in (ABSENT, CONFLICT):
        return None, ("ribbon_loaded",)
    return ribbon is True, ()


def reference(frame: Frame) -> Gold:
    v = View(frame)
    method, prompt, media, ribbon, symptom = (v.get(p) for p in ("print_method", "prompt", "media_type", "ribbon_loaded", "detection_symptom"))
    context = (*(("ZT-ASSET",) if v.ignored else ()), *(("ZT-CURRENT",) if v.stale else ()))
    out, prohibited = {}, []

    def decide(action, value, *rules, unresolved=()):
        conflict = ("ZT-CONFLICT",) if any(v.get(p) == CONFLICT for p in unresolved) else ()
        out[f"actions/{action}"] = Decision(value, (*context, *rules, *conflict), tuple(i for p in unresolved for i in v.evidence.get(p, ())), tuple(unresolved))
    def forbid(action, *rules):
        out[f"actions/{action}"] = Decision(False, (*context, *rules))
        prohibited.append(action)
    if prompt in (ABSENT, CONFLICT) or method in (ABSENT, CONFLICT):
        for action in MENU[:-1]:
            decide(action, None, "ZT-DEFAULT", unresolved=tuple(p for p, x in (("prompt", prompt), ("print_method", method)) if x in (ABSENT, CONFLICT)))
    else:
        idle, calibrating = prompt in IDLE, prompt in CALIBRATION
        for action, want, other, rule in (("set_method_tt", TT, DT, "ZT-METHOD-TT"), ("set_method_dt", DT, TT, "ZT-METHOD-DT")):
            if not idle or method == want:
                decide(action, False, "ZT-DEFAULT")
            elif media in (ABSENT, CONFLICT):
                decide(action, None, rule, "ZT-DIAGNOSE", unresolved=("media_type",))
            else:
                decide(action, media == want, rule)
        if not idle or method != TT or media == DT:
            decide("load_ribbon", False, "ZT-DEFAULT" if media != DT else "ZT-METHOD-DT")
        elif ribbon in (ABSENT, CONFLICT):
            decide("load_ribbon", None, "ZT-RIBBON", "ZT-DIAGNOSE", unresolved=("ribbon_loaded",))
        else:
            decide("load_ribbon", ribbon is False, "ZT-RIBBON")
        consistent, needs = consistency(method, media, ribbon)
        if not idle or symptom == ABSENT:
            decide("start_calibration", False, "ZT-DEFAULT")
        elif symptom == CONFLICT or consistent is None:
            decide("start_calibration", None, "ZT-CAL-START", unresolved=("detection_symptom",) if symptom == CONFLICT else needs)
        else:
            decide("start_calibration", bool(symptom) and consistent, "ZT-CAL-START")
        if unverified(v):
            decide("verify_loading", True, "ZT-VERIFY", "ZT-COMMAND")
        elif idle and symptom is True and consistent:
            decide("verify_loading", True, "ZT-CAL-START")
        else:
            decide("verify_loading", False, "ZT-DEFAULT")
        if not calibrating:
            decide("calibration_next", False, "ZT-DEFAULT")
        elif prompt in v.steps:
            decide("calibration_next", True, "ZT-CAL-NEXT")
        else:
            forbid("calibration_next", "ZT-CAL-NEXT", *(("ZT-COMMAND",) if prompt in v.reported else ()))
        if not calibrating:
            decide("cancel_calibration", False, "ZT-DEFAULT")
        elif consistent is None:
            decide("cancel_calibration", None, "ZT-CAL-CANCEL", unresolved=needs)
        else:
            decide("cancel_calibration", consistent is False, "ZT-CAL-CANCEL")
        calibrated = v.get("calibration_complete") is True
        if calibrating:
            forbid("resume_printing", "ZT-RESUME")
        elif prompt != "paused":
            decide("resume_printing", False, "ZT-DEFAULT")
        elif consistent is False:
            forbid("resume_printing", "ZT-RESUME", *(("ZT-RIBBON",) if ribbon is False else ()))
        elif consistent is None:
            decide("resume_printing", None, "ZT-RESUME", unresolved=needs)
        elif symptom == CONFLICT and not calibrated:
            decide("resume_printing", None, "ZT-RESUME", unresolved=("detection_symptom",))
        elif symptom is True and not calibrated:
            forbid("resume_printing", "ZT-RESUME", *(("ZT-CAL-CANCEL",) if "cancel_calibration" in v.attempted else ()))
        else:
            decide("resume_printing", True, "ZT-RESUME")
    pending = any(d.value is None for d in out.values())
    decide("request_observation", pending, "ZT-DIAGNOSE" if pending else "ZT-DEFAULT")
    return Gold(REVISION, SCOPE, {target: out[target] for target in SCOPE}, tuple(a for a in MENU if a in prohibited), primitives(v, pending))


def unverified(v: View, accepted=()) -> bool:
    loading = "load_ribbon" in v.attempted and "ribbon_loaded" not in v.verified and "commanded_load" not in accepted
    steps = set() if "reported_step" in accepted else v.reported - v.steps - {"calibration_complete"}
    completion = "calibration_complete" in v.reported and v.get("calibration_complete") is not True and "reported_completion" not in accepted
    return loading or bool(steps) or completion


def stated(value, rule, ids=()):
    return Decision(None if value == CONFLICT else "unstated" if value == ABSENT else value, (rule, *(("ZT-CONFLICT",) if value == CONFLICT else ())), tuple(ids))


def primitives(v: View, pending: bool) -> dict:
    method, prompt, media, ribbon, symptom, calibrated = (v.get(p) for p in ("print_method", "prompt", "media_type", "ribbon_loaded", "detection_symptom",
                                                                               "calibration_complete"))
    flip = {True: False, False: True}
    consistent, needs = consistency(method, media, ribbon) if method not in (ABSENT, CONFLICT) else (None, ("print_method",))
    unknown = CONFLICT if any(v.get(p) == CONFLICT for p in needs) else ABSENT
    inconsistent = method != media if media in (TT, DT) and method in (TT, DT) else CONFLICT if CONFLICT in (media, method) else ABSENT
    if consistent is None:
        ready = unknown
    else:
        ready = consistent and (calibrated is True or symptom is not True) if symptom != CONFLICT or calibrated is True else CONFLICT
    return {"tt_media": stated(media == TT if media in (TT, DT) else media, "ZT-METHOD-TT", v.evidence.get("media_type", ())),
            "dt_media": stated(media == DT if media in (TT, DT) else media, "ZT-METHOD-DT", v.evidence.get("media_type", ())),
            "ribbon_loaded": stated(ribbon, "ZT-RIBBON", v.evidence.get("ribbon_loaded", ())),
            "ribbon_absent": stated(flip.get(ribbon, ribbon), "ZT-RIBBON", v.evidence.get("ribbon_loaded", ())),
            "detection_symptom": Decision(None if symptom == CONFLICT else symptom is True, ("ZT-CAL-START", *(("ZT-CONFLICT",) if symptom == CONFLICT else ())),
                                          tuple(v.evidence.get("detection_symptom", ()))),
            "step_verified": Decision(prompt in CALIBRATION and prompt in v.steps, ("ZT-CAL-NEXT", "ZT-COMMAND")),
            "unverified_step": Decision(unverified(v), ("ZT-VERIFY", "ZT-COMMAND")),
            "calibration_verified": Decision(True if calibrated is True else False if "calibration_complete" in v.reported or "cancel_calibration" in v.attempted
                                             else "unstated", ("ZT-RESUME", "ZT-COMMAND")),
            "method_inconsistent": stated(inconsistent, "ZT-CAL-CANCEL"),
            "ready_to_print": stated(ready, "ZT-RESUME"),
            "needs_observation": Decision(pending, ("ZT-DIAGNOSE",))}


def definition(condition: str) -> dict:
    spec = files.read_yaml(DEFINITION)
    actions = spec["facts"]["actions"]
    if condition == "structured":
        return spec
    if condition == "structured_permuted":
        return {**spec, "facts": {**spec["facts"], "actions": {**actions, "cards": evidence.permuted(actions["cards"])}}}
    cards = [{key: value for key, value in card.items() if key != "requires"} for card in actions["cards"]]
    return {**spec, "fields": {name: spec["fields"][name] for name in ("asset", "model", "notes")}, "reads": ["asset", "model", "notes"],
            "facts": {**spec["facts"], "actions": {**actions, "cards": cards}}}


def justified(gold: Gold) -> tuple[str, ...]:
    return tuple(t.split("/")[1] for t, d in gold.decisions.items() if d.value is True)


def unresolved(gold: Gold) -> tuple[str, ...]:
    return tuple(t.split("/")[1] for t, d in gold.decisions.items() if d.value is None)


def ev(time, indication, asset, prop, value, actor="", status="current", phrase=0):
    return Event(time, indication, actor, asset, prop, value, status, phrase)


def scenarios(seed: int, per_template: int, variants: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for template in TEMPLATES:
        for k in range(per_template):
            asset, other = (f"PR-{n:02d}" for n in rng.sample(range(1, 60), 2))
            name, model, phrase = rng.choice(NAMES), rng.choice(("ZT411", "ZT421")), rng.randrange(2)
            sid = f"zt-{template.__name__}-{seed}-{k}"
            for variant, state, events, tags in evidence.contrast_quota(template(asset, other, name, model, phrase), variants, f"{seed}:{sid}"):
                out.append({"scenario": sid, "variant": variant, "template": template.__name__, "state": asdict(state),
                            "events": [asdict(e) for e in events], "tags": sorted(tags), "goal": "return_to_printing", "seed": seed})
    return out


def method_mismatch(asset, other, name, model, phrase, loaded=TT):
    configured = DT if loaded == TT else TT
    state = State(asset, model, loaded, loaded == TT, configured, "paused")
    base = [ev(0, "configuration", asset, "print_method", configured), ev(0, "display", asset, "prompt", "paused"),
            *([ev(1, "verification", asset, "ribbon_loaded", True, name)] if loaded == TT else [])]
    media = ev(2, "operator_report", asset, "media_type", loaded, name, phrase=phrase)
    return [("base", state, [*base, media], {"configured_mode"}),
            ("missing_media", state, base, {"configured_mode", "missing_observation"}),
            ("stale_media", state, [*base, replace(media, indication="handover_note", status="stale", time=-30)], {"configured_mode", "stale_handover"}),
            ("other_asset", state, [*base, replace(media, asset=other)], {"configured_mode", "wrong_machine"})]


def ribbon_out(asset, other, name, model, phrase):
    state = State(asset, model, TT, False, TT, "paused")
    base = [ev(0, "configuration", asset, "print_method", TT), ev(0, "display", asset, "prompt", "paused"),
            ev(1, "operator_report", asset, "media_type", TT, name, phrase=phrase)]
    reading = ev(2, "reading", asset, "ribbon_loaded", False)
    loaded = replace(state, ribbon=True)
    return [("base", state, [*base, reading], {"prohibited_resume"}),
            ("commanded_load", loaded, [*base, replace(reading, status="stale"), ev(3, "command", asset, "attempted", "load_ribbon", name)],
             {"commanded_not_verified"}),
            ("verified_load", loaded, [*base, replace(reading, status="stale"), ev(3, "command", asset, "attempted", "load_ribbon", name),
                                      ev(4, "verification", asset, "ribbon_loaded", True, name)], {"verified_completion"}),
            ("contradiction", state, [*base, reading, ev(3, "operator_report", asset, "ribbon_loaded", True, name, phrase=phrase)], {"contradiction"})]


def detection(asset, other, name, model, phrase):
    state = State(asset, model, TT, True, TT, "paused", symptom=True)
    base = [ev(0, "configuration", asset, "print_method", TT), ev(0, "display", asset, "prompt", "paused"),
            ev(1, "verification", asset, "ribbon_loaded", True, name), ev(1, "operator_report", asset, "media_type", TT, name, phrase=phrase)]
    symptom = ev(2, "operator_report", asset, "detection_symptom", True, name, phrase=phrase)
    return [("base", state, [*base, symptom], {"several_valid"}),
            ("no_symptom", replace(state, symptom=False), base, {"no_symptom"}),
            ("other_asset_symptom", replace(state, symptom=False), [*base, replace(symptom, asset=other)], {"wrong_machine"})]


def calibration_step(asset, other, name, model, phrase):
    state = State(asset, model, TT, True, TT, "calibration_1", symptom=True, step_done=True)
    base = [ev(0, "configuration", asset, "print_method", TT), ev(1, "operator_report", asset, "media_type", TT, name, phrase=phrase),
            ev(1, "verification", asset, "ribbon_loaded", True, name), ev(2, "command", asset, "attempted", "start_calibration", name),
            ev(3, "display", asset, "prompt", "calibration_1")]
    reported = ev(4, "operator_report", asset, "step", "calibration_1", name, phrase=phrase)
    return [("reported_step", state, [*base, reported], {"early_next", "early_resume", "commanded_not_verified"}),
            ("verified_step", state, [*base, ev(4, "verification", asset, "step", "calibration_1", name)], {"verified_completion", "early_resume"}),
            ("wrong_method", replace(state, method=DT), [*base[1:], ev(0, "configuration", asset, "print_method", DT), reported],
             {"configured_mode", "early_next"})]


def cancelled(asset, other, name, model, phrase):
    state = State(asset, model, TT, True, TT, "paused", symptom=True)
    base = [ev(0, "configuration", asset, "print_method", TT), ev(1, "operator_report", asset, "media_type", TT, name, phrase=phrase),
            ev(1, "verification", asset, "ribbon_loaded", True, name), ev(2, "operator_report", asset, "detection_symptom", True, name, phrase=phrase),
            ev(5, "display", asset, "prompt", "paused")]
    return [("cancelled", state, [*base, ev(4, "command", asset, "attempted", "cancel_calibration", name)], {"cancel_not_completion"}),
            ("completed", replace(state, symptom=False, calibrated=True),
             [*base[:3], replace(base[3], status="stale"), base[4], ev(4, "verification", asset, "calibration_complete", True)], {"verified_completion"}),
            ("reported_completion", replace(state, symptom=False, calibrated=True),
             [*base, ev(4, "operator_report", asset, "calibration_complete", True, name, phrase=phrase)], {"commanded_not_verified"})]


def direct_thermal(asset, other, name, model, phrase):
    return method_mismatch(asset, other, name, model, phrase, DT)


TEMPLATES = (method_mismatch, ribbon_out, detection, calibration_step, cancelled, direct_thermal)


def frames(planned, conditions=CONDITIONS, languages=LANGUAGES) -> list[Frame]:
    out = []
    for row in planned:
        state, events = State(**row["state"]), [Event(**e) for e in row["events"]]
        ids = {(c, l): f"{row['scenario']}:{row['variant']}:{c}:{l}" for c in conditions for l in languages}
        for (condition, language), frame_id in ids.items():
            out.append(render(row["scenario"], state, events, condition=condition, frame_id=frame_id, family=row["scenario"], language=language,
                              tags=(*row["tags"], condition, language, row["template"]), seeds=(f"scenario:{row['scenario']}",),
                              pairs=tuple(i for i in ids.values() if i != frame_id)))
    return out


def evaluate(built) -> dict:
    for frame in built:
        evidence.validate_frame(frame, FIELDS)
    golds = {frame.id: evidence.validate_gold(frame, reference(frame), SCOPE, REVISION) for frame in built}
    evidence.invariant(reference, built)
    return golds


def build(planned, *, conditions=CONDITIONS, languages=LANGUAGES):
    built = frames(planned, conditions, languages)
    return built, evaluate(built)


def authored() -> dict:
    return {"catalogue": identity({"actions": ACTIONS, "action_text": ACTION_TEXT, "words": WORDS, "names": NAMES,
                                   "phrases": {language: [[list(key), value] for key, value in table.items()] for language, table in PHRASES.items()}})}


def stratum(frame: Frame, gold: Gold, role: str) -> tuple:
    return (frame.condition, role, next(tag for tag in frame.tags if tag in LANGUAGES), "forbidden_action" if gold.prohibited else "no_forbidden_action")


CONVENTIONS = {
    "a": (None, ()),
    "b": ("Convention B, an authored benchmark variant and not a manufacturer revision: an operator's report that the current calibration prompt's step "
          "was completed establishes that step; a pressed ribbon-load command is not completion and still needs a verification.", ("reported_step",)),
    "c": ("Convention C, an authored benchmark variant and not a manufacturer revision: a pressed ribbon-load command establishes the loading step; "
          "a reported calibration step is not completion and still needs a verification.", ("commanded_load",)),
    "d": ("Convention D, an authored benchmark variant and not a manufacturer revision: an operator's report establishes the current calibration "
          "prompt's step and the completion of a manual sensor calibration; a pressed ribbon-load command is not completion and still needs a verification.",
          ("reported_step", "reported_completion"))}
TRANSFER_ROLES = {"fit": ("a", "b"), "model_select": ("c",), "check": ("d",)}
TRANSFER_FACT = ("Under the supplied procedure document, a loading or calibration step on this printer was attempted or reported "
                 "but is not established as completed.")


def transfer_definition(pack) -> dict:
    spec = files.read_yaml(DEFINITION)
    documents = {f"zt411_{key}": {"text": " ".join(f"{rule.id}: {text if rule.id == 'ZT-COMMAND' and text else rule.text}" for rule in pack.rules)}
                 for key, (text, _) in CONVENTIONS.items()}
    return {"name": "zt411_transfer", "fields": spec["fields"], "reads": spec["reads"], "guidance": documents,
            "facts": {f"unverified_step_{key}": {"check": TRANSFER_FACT, "guidance": [f"zt411_{key}"]} for key in CONVENTIONS}}


def transfer_gold(frame: Frame, role: str) -> Gold:
    v = View(frame)
    return Gold(REVISION, tuple(f"unverified_step_{key}" for key in CONVENTIONS),
                primitives={f"unverified_step_{key}": Decision(unverified(v, CONVENTIONS[key][1]), ("ZT-COMMAND", "ZT-VERIFY")) for key in TRANSFER_ROLES[role]})


def transfer_command(args):
    pack = evidence.load_pack(PACK)
    planned = scenarios(args.seed, args.per_template, args.variants)
    built, _ = build(planned)
    assignment = evidence.assign(built, {"fit": args.fit, "model_select": args.model_select, "check": args.check}, seed=args.seed,
                                 counted={f.id for f in built if f.condition == TRAINED},
                                 families_minimum={"fit": 30, "model_select": 40, "check": 60} if args.fit == 100 else None)
    role = {f.id: assignment["families"][assignment["components"][f.id]] for f in built}
    transfer = {f.id: transfer_gold(f, role[f.id]) for f in built}
    return evidence.publish(args.output, pack, scenarios=planned, frames=built, golds=transfer, definition=transfer_definition(pack), name="printer-transfer.yaml",
                            condition=TRAINED, assignment=assignment, producer="bench.procedures.transfer",
                            producer_digest=evidence.code_digest(sys.modules[__name__], evidence), reference=lambda frame: transfer_gold(frame, role[frame.id]),
                            fields=FIELDS, stratum=stratum, rules=[rule.id for rule in pack.rules], authored=authored(), seed=args.seed,
                            extra={"objective": "bounded guidance transfer across four printer procedure documents; not unseen-manual ingestion or cross-equipment competence",
                                   "documents": {f"zt411_{key}": {"source": "pinned manual rules" if text is None else "authored benchmark convention variant",
                                                                  "roles": [role_ for role_, keys in TRANSFER_ROLES.items() if key in keys]}
                                                 for key, (text, _) in CONVENTIONS.items()},
                                   "revision": REVISION})


def pack_command(args):
    pack = evidence.load_pack(PACK)
    planned = scenarios(args.seed, args.per_template, args.variants)
    reviews = evidence.load_reviews(args.reviews) if args.reviews else ()
    built, applied = evidence.apply_reviews(frames(planned), reviews)
    golds = evaluate(built)
    assignment = evidence.assign(built, {"fit": args.fit, "model_select": args.model_select, "check": args.check}, seed=args.seed,
                                 counted=evidence.situation_frames(built, golds, TRAINED), families_minimum={"fit": 30, "model_select": 40, "check": 60} if args.fit == 100 else None)
    return evidence.publish(args.output, pack, scenarios=planned, frames=built, golds=golds, definition=files.read_yaml(DEFINITION), name=DEFINITION.name, condition=TRAINED, assignment=assignment,
                            reviews=reviews, applied=applied, producer="bench.procedures.reference", producer_digest=evidence.code_digest(sys.modules[__name__], evidence),
                            reference=reference, fields=FIELDS, stratum=stratum, authored=authored(), seed=args.seed,
                            extra={"menu": ACTIONS, "revision": REVISION})




def transition(state: State, action: str, events, time: int, actor="operator"):
    asset = state.asset

    def stale(props):
        return [replace(e, status="stale") if e.asset == asset and e.property in props and e.status == "current" else e for e in events]
    if action in ("set_method_tt", "set_method_dt"):
        method = TT if action == "set_method_tt" else DT
        return replace(state, method=method, calibrated=False), [*stale({"print_method"}), ev(time, "configuration", asset, "print_method", method)]
    if action == "load_ribbon":
        return replace(state, ribbon=True), [*stale({"ribbon_loaded"}), ev(time, "command", asset, "attempted", "load_ribbon", actor)]
    if action == "verify_loading":
        out = [*stale({"ribbon_loaded", "media_type"}), ev(time, "verification", asset, "ribbon_loaded", state.ribbon, actor),
               ev(time, "verification", asset, "media_type", state.media, actor)]
        if state.prompt in CALIBRATION and state.step_done:
            out.append(ev(time, "verification", asset, "step", state.prompt, actor))
        return state, out
    if action == "request_observation":
        return state, [*stale({"ribbon_loaded", "media_type"}), ev(time, "operator_report", asset, "media_type", state.media, actor),
                       ev(time, "operator_report", asset, "ribbon_loaded", state.ribbon, actor)]
    if action == "start_calibration" and state.prompt in IDLE:
        return replace(state, prompt="calibration_1", step_done=True), [*stale({"prompt", "step", "calibration_complete"}),
                                                                        ev(time, "command", asset, "attempted", "start_calibration", actor),
                                                                        ev(time, "display", asset, "prompt", "calibration_1"),
                                                                        ev(time, "operator_report", asset, "step", "calibration_1", actor)]
    if action == "calibration_next" and state.prompt in CALIBRATION and state.step_done:
        if state.prompt == "calibration_1":
            return replace(state, prompt="calibration_2"), [*stale({"prompt"}), ev(time, "display", asset, "prompt", "calibration_2"),
                                                            ev(time, "operator_report", asset, "step", "calibration_2", actor)]
        done = consistency(state.method, state.media, state.ribbon)[0] is True
        return replace(state, prompt="paused", calibrated=done, symptom=state.symptom and not done, step_done=False), \
            [*stale({"prompt", "step", *(("detection_symptom",) if done else ())}), ev(time, "display", asset, "prompt", "paused"),
             *([ev(time, "verification", asset, "calibration_complete", True)] if done else [])]
    if action == "cancel_calibration" and state.prompt in CALIBRATION:
        return replace(state, prompt="paused", step_done=False), [*stale({"prompt", "step"}), ev(time, "command", asset, "attempted", "cancel_calibration", actor),
                                                                  ev(time, "display", asset, "prompt", "paused")]
    if action == "resume_printing" and state.prompt == "paused":
        return replace(state, prompt="ready"), [*stale({"prompt"}), ev(time, "display", asset, "prompt", "ready")]
    return state, events


def terminal(state: State) -> str | None:
    if state.prompt != "ready":
        return None
    working = consistency(state.method, state.media, state.ribbon)[0] is True and not state.symptom
    return "printing" if working else "printing_failed"


def digest_state(state: State) -> str:
    return identity(asdict(state))


def progress_key(state: State, gold: Gold):
    return digest_state(state), identity({t: d.value for t, d in gold.decisions.items()})


def episode(row, decide, *, seed: int, condition="structured", cap: int = 5, priority=()) -> dict:
    state, events = State(**row["state"]), [Event(**e) for e in row["events"]]
    rng, seen, steps = random.Random(f"{seed}:{row['scenario']}:{row['variant']}"), set(), []
    initial = {"scenario": row["scenario"], "variant": row["variant"], "state_sha256": digest_state(state), "seed": seed, "condition": condition}
    reason = "cap"
    for step in range(cap):
        frame = render(row["scenario"], state, events, condition=condition, frame_id=f"{row['scenario']}:{row['variant']}:step{step}", family=row["scenario"])
        gold = reference(frame)
        if progress_key(state, gold) in seen:
            reason = "nonprogress"
            break
        seen.add(progress_key(state, gold))
        decision = decide(frame)
        record = {"step": step, "before_state": digest_state(state), "frame_sha256": frame.content, "selected": sorted(decision["selected"]),
                  "unresolved": sorted(decision["unresolved"]), "outcome": decision.get("outcome", "predicted"),
                  "justified": list(justified(gold)), "prohibited": list(gold.prohibited), "reference_unresolved": list(unresolved(gold))}
        steps.append(record)
        if decision.get("outcome") == "failed":
            reason = "failed"
            break
        if set(decision["selected"]) & set(gold.prohibited):
            reason = "prohibited_selected"
            break
        if not decision["selected"]:
            reason = "unresolved" if decision["unresolved"] else "empty_selection"
            break
        chosen = next((a for a in priority if a in decision["selected"]), None) or rng.choice(sorted(decision["selected"]))
        state, events = transition(state, chosen, events, 10 * (step + 1) + 10, actor="operator")
        record |= {"scheduled": chosen, "choice": "document_priority" if chosen in priority else "seeded", "after_state": digest_state(state),
                   "verified_after": sorted({e.property for e in events if e.indication == "verification" and e.status == "current"})}
        done = terminal(state)
        if done:
            reason = done
            break
    return {"initial": initial, "steps": steps, "termination": reason, "goal_reached": reason == "printing",
            "final_state_sha256": digest_state(state)}


def replay(row, trace, *, condition="structured") -> bool:
    state, events = State(**row["state"]), [Event(**e) for e in row["events"]]
    if digest_state(state) != trace["initial"]["state_sha256"]:
        return False
    for step, record in enumerate(trace["steps"]):
        frame = render(row["scenario"], state, events, condition=condition, frame_id=f"{row['scenario']}:{row['variant']}:step{step}", family=row["scenario"])
        if frame.content != record["frame_sha256"] or digest_state(state) != record["before_state"]:
            return False
        if "scheduled" not in record:
            break
        state, events = transition(state, record["scheduled"], events, 10 * (step + 1) + 10, actor="operator")
        if digest_state(state) != record["after_state"]:
            return False
    return digest_state(state) == trace["final_state_sha256"]


def controls(row, *, seed, condition="structured"):
    return {"always_abstain": episode(row, lambda frame: {"selected": set(), "unresolved": set()}, seed=seed, condition=condition),
            "always_cancel": episode(row, lambda frame: {"selected": {"cancel_calibration"}, "unresolved": set()}, seed=seed, condition=condition)}


def main():
    parser = argparse.ArgumentParser(prog="bench.procedures", description="Zebra ZT411/ZT421 snapshot pack: explicit offline construction; no model or network use")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("pack", "plan scenarios, build frames/reference and publish the benchmark artifacts"),
                       ("transfer", "publish the bounded four-document guidance-transfer pack: fit A/B, model_select C, check D")):
        p = commands.add_parser(name, help=text)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--per-template", type=int, default=40)
        p.add_argument("--variants", type=int, default=3, help="per scenario: base plus a seeded sample of up to N-1 contrast variants, keeping families small")
        p.add_argument("--fit", type=int, default=100)
        p.add_argument("--model-select", type=int, default=150)
        p.add_argument("--check", type=int, default=200)
        p.add_argument("--output", type=Path, required=True)
    commands.choices["pack"].add_argument("--reviews", type=Path, help="reviewer decisions JSONL (confirm with a worksheet binding; replace/skip with content digests)")
    commands.add_parser("pin-authored", help="record the reviewed authored rule and catalogue digests in the pack manifest")
    args = parser.parse_args()
    try:
        if args.command == "pin-authored":
            print(json.dumps(evidence.pin_authored(PACK, authored()), indent=1))
            return
        manifest = (pack_command if args.command == "pack" else transfer_command)(args)
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({"assignment": manifest["assignment"], "review_plan": manifest["review_plan"], "review": manifest["review"]["status"]}, indent=1))


if __name__ == "__main__":
    main()
