from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
import random
import sys

from semantic_switch import ConfigError, files
from semantic_switch.compiler import digest
from bench import evidence
from bench.evidence import Assertion, Decision, Frame, Gold, Prose

DATA = Path(__file__).parent / "data" / "zt411"
PACK, DEFINITION = DATA / "pack.json", Path(__file__).parents[1] / "examples" / "data" / "printer.yaml"
REVISION = "zt411-reference-3"
ACTIONS = {"set_method_tt": "Set print method to thermal transfer", "set_method_dt": "Set print method to direct thermal",
           "load_ribbon": "Load a ribbon", "start_calibration": "Start manual sensor calibration",
           "verify_loading": "Inspect and verify media and ribbon loading", "calibration_next": "Confirm the calibration prompt and continue",
           "cancel_calibration": "Cancel calibration", "resume_printing": "Return to printing",
           "request_observation": "Ask the operator for the missing or conflicting observation"}
MENU = tuple(ACTIONS)
SCOPE = tuple(f"actions/{action}" for action in MENU)
PRIMITIVES = ("tt_media", "dt_media", "ribbon_loaded", "ribbon_absent", "detection_symptom", "verification_due", "step_verified", "unverified_step",
              "calibration_verified", "method_inconsistent", "ready_to_print", "needs_observation")
FIELDS = ("asset", "model", "print_method", "prompt", "enabled", "alerts", "verified", "attempted", "notes")
IDLE, CALIBRATION = ("ready", "paused"), ("calibration_1", "calibration_2")
TT, DT = "thermal_transfer", "direct_thermal"
ABSENT, CONFLICT, UNREAD = "absent", "conflict", "unknown"
PROPERTIES = ("prompt", "print_method", "media_type", "ribbon_loaded", "detection_symptom")
DOMAIN = {"prompt": (*IDLE, *CALIBRATION), "print_method": (TT, DT), "media_type": (TT, DT), "ribbon_loaded": (True, False), "detection_symptom": (True, False)}
UNSTATED = ("tt_media", "dt_media", "ribbon_loaded", "ribbon_absent", "calibration_verified", "method_inconsistent", "ready_to_print")
STRUCTURED = {"configuration": "print_method", "display": "prompt", "reading": "alerts", "verification": "verified", "command": "attempted"}
TRAINED = "structured"
CONDITIONS = ("structured", "prose", "structured_permuted")
LANGUAGES = ("en", "pt")
PHRASES = {
    "en": {("media_type", TT): ["loaded thermal-transfer labels", "put in a roll of thermal-transfer media"],
           ("media_type", DT): ["loaded direct-thermal labels", "put in a roll of direct-thermal media"],
           ("media_changed", True): ["swapped the media roll at handover", "changed the label roll at shift change"],
           ("ribbon_loaded", True): ["a ribbon is loaded", "the ribbon is in"],
           ("ribbon_loaded", False): ["there is no ribbon loaded", "the ribbon spindle is empty"],
           ("detection_symptom", True): ["labels are skipping and it stops mid-roll", "it reports media out although labels are loaded"],
           ("step", "calibration_1"): ["did the first calibration step on the prompt", "finished what the first calibration prompt asked"],
           ("step", "calibration_2"): ["did the second calibration step on the prompt", "finished what the second calibration prompt asked"],
           ("calibration_complete", True): ["finished the manual sensor calibration", "completed the sensor calibration"]},
    "pt": {("media_type", TT): ["coloquei etiquetas de transferência térmica", "pus um rolo de mídia de transferência térmica"],
           ("media_type", DT): ["coloquei etiquetas térmicas diretas", "pus um rolo de mídia térmica direta"],
           ("media_changed", True): ["troquei o rolo de mídia na passagem de turno", "mudei o rolo de etiquetas na troca de turno"],
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
    events = tuple(events)
    structured = condition.removesuffix("_permuted") == "structured"
    data = {"asset": state.asset, "model": state.model}
    notes, assertions, lists = Prose("notes"), [], {"alerts": [], "verified": [], "attempted": []}
    scalar = {prop: {json.dumps(event.value) for event in events if event.asset == state.asset and event.status == 'current' and event.property == prop}
              for prop in ('prompt', 'print_method')}
    ambiguous = {prop for prop, values in scalar.items() if len(values) > 1}
    for n, e in enumerate(sorted(events, key=lambda e: (e.time, e.indication, e.property))):
        key = f"a{n}"
        clock = f"{8 + e.time // 60:02d}:{e.time % 60:02d}"
        if structured and e.indication in STRUCTURED and e.asset == state.asset and e.status == "current" and e.property not in ambiguous:
            target = STRUCTURED[e.indication]
            if target in lists:
                lists[target].append(f"{clock} " + (alert((e.property, e.value)) if e.indication == "reading" else e.value if e.indication == "command" else
                                                    f"{e.property}:{json.dumps(e.value)}"))
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
        data |= {"enabled": list(enabled(data.get("prompt"))), **lists, "alerts": lists["alerts"] or UNREAD}
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
    return stale + (w["I"].format(text) if e.property in ("media_type", "media_changed", "step", "calibration_complete") else f"{text[0].upper()}{text[1:]}.")


class View:
    def __init__(self, frame: Frame):
        self.asset = frame.input["asset"]
        self.values, self.evidence, self.attempted, self.reported, self.steps, self.verified = {}, {}, set(), set(), set(), set()
        self.ignored, self.stale, self.readings = [], [], set()
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
                if a.indication == "reading":
                    self.readings.add((a.property, json.dumps(a.value)))
                self.values.setdefault(a.property, set()).add(json.dumps(a.value))
                self.evidence.setdefault(a.property, []).append(a.id)
                if a.indication == "verification":
                    self.verified.add(a.property)

    @property
    def context(self) -> tuple[str, ...]:
        return (*(("ZT-ASSET",) if self.ignored else ()), *(("ZT-CURRENT",) if self.stale else ()))

    def get(self, prop):
        found = self.values.get(prop, set())
        return json.loads(next(iter(found))) if len(found) == 1 else CONFLICT if found else ABSENT

    def options(self, prop) -> tuple:
        found = self.values.get(prop)
        return tuple(json.loads(x) for x in sorted(found)) if found else (False,) if prop == "detection_symptom" else DOMAIN[prop]


def consistent(method, media, ribbon) -> bool:
    return method == media and (media == DT or ribbon is True)


def unverified(v: View, accepted=()) -> bool:
    loading = "load_ribbon" in v.attempted and "ribbon_loaded" not in v.verified and "commanded_load" not in accepted
    steps = set() if "reported_step" in accepted else v.reported - v.steps - {"calibration_complete"}
    completion = "calibration_complete" in v.reported and v.get("calibration_complete") is not True and "reported_completion" not in accepted
    return loading or bool(steps) or completion


def judge(v: View, world: dict) -> dict:
    prompt, method, media, ribbon = (world[p] for p in PROPERTIES[:4])
    symptom = world["detection_symptom"] or (ribbon is True and ("ribbon_loaded", "false") in v.readings)
    calibrated, idle, calibrating, fits = v.get("calibration_complete") is True, prompt in IDLE, prompt in CALIBRATION, consistent(method, media, ribbon)
    out = {}

    def decide(action, value, *rules, forbidden=False):
        out[f"actions/{action}"] = (value, rules, forbidden)
    for action, want, rule in (("set_method_tt", TT, "ZT-METHOD-TT"), ("set_method_dt", DT, "ZT-METHOD-DT")):
        if method == want:
            decide(action, False, "ZT-DEFAULT")
        elif calibrating:
            decide(action, False, "ZT-CAL-HOLD")
        else:
            decide(action, media == want, rule)
    if not idle or method != TT or media == DT:
        decide("load_ribbon", False, "ZT-METHOD-DT" if media == DT else "ZT-DEFAULT")
    else:
        decide("load_ribbon", ribbon is False, "ZT-RIBBON")
    for action in ("start_calibration", "verify_loading"):
        if action == "verify_loading" and unverified(v):
            decide(action, True, "ZT-VERIFY", "ZT-COMMAND")
        elif not idle or not symptom:
            decide(action, False, "ZT-DEFAULT")
        else:
            decide(action, fits, "ZT-CAL-START")
    if not calibrating:
        decide("calibration_next", False, "ZT-DEFAULT")
        decide("cancel_calibration", False, "ZT-DEFAULT")
    else:
        if prompt in v.steps:
            decide("calibration_next", True, "ZT-CAL-NEXT")
        else:
            decide("calibration_next", False, "ZT-CAL-NEXT", *(("ZT-COMMAND",) if prompt in v.reported else ()), forbidden=True)
        decide("cancel_calibration", not fits, "ZT-CAL-CANCEL")
    if calibrating:
        decide("resume_printing", False, "ZT-RESUME", forbidden=True)
    elif prompt != "paused":
        decide("resume_printing", False, "ZT-DEFAULT")
    elif not fits:
        decide("resume_printing", False, "ZT-RESUME", *(("ZT-RIBBON",) if ribbon is False else ()), forbidden=True)
    elif symptom and not calibrated:
        decide("resume_printing", False, "ZT-RESUME", *(("ZT-CAL-CANCEL",) if "cancel_calibration" in v.attempted else ()), forbidden=True)
    else:
        decide("resume_printing", True, "ZT-RESUME")
    calibration = calibrating or bool(v.attempted & {"start_calibration", "cancel_calibration"}) or "calibration_complete" in v.reported
    return {**out, "tt_media": (media == TT, ("ZT-METHOD-TT",), ("media_type",)), "dt_media": (media == DT, ("ZT-METHOD-DT",), ("media_type",)),
            "ribbon_loaded": (ribbon, ("ZT-RIBBON",), ("ribbon_loaded",)), "ribbon_absent": (not ribbon, ("ZT-RIBBON",), ("ribbon_loaded",)),
            "detection_symptom": (symptom, ("ZT-CAL-START",), ("detection_symptom",)),
            "step_verified": (calibrating and prompt in v.steps, ("ZT-CAL-NEXT", "ZT-COMMAND"), ()),
            "unverified_step": (unverified(v), ("ZT-VERIFY", "ZT-COMMAND"), ()),
            "calibration_verified": (True if calibrated else False if calibration else "unstated", ("ZT-RESUME", "ZT-COMMAND"), ()),
            "method_inconsistent": (method != media, ("ZT-CAL-CANCEL",), ()),
            "ready_to_print": (not calibrating and fits and (calibrated or not symptom), ("ZT-RESUME",), ())}


def sensitive(table: dict, varying, key) -> tuple[str, ...]:
    found = []
    for prop in varying:
        i, groups = PROPERTIES.index(prop), {}
        for world, outcome in table.items():
            groups.setdefault(world[:i] + world[i + 1:], set()).add(key(outcome))
        if any(len(group) > 1 for group in groups.values()):
            found.append(prop)
    return tuple(found)


def settle(v: View, table: dict, target: str, undecided=None) -> Decision:
    varying = tuple(p for p in PROPERTIES if len(v.options(p)) > 1)
    outcomes = [outcome[target] for outcome in table.values()]
    values = {o[0] for o in outcomes}
    value = next(iter(values)) if len(values) == 1 else None
    relevant = sensitive(table, varying, lambda outcome: outcome[target][0])
    cited = relevant if relevant else sensitive(table, varying, lambda outcome: outcome[target][:2])
    conflict = ("ZT-CONFLICT",) if any(p in v.values for p in cited) else ()
    rules = (*dict.fromkeys(rule for o in outcomes for rule in o[1]), *conflict)
    if target.startswith("actions/"):
        ids = tuple(i for p in cited for i in v.evidence.get(p, ()))
        return Decision(value, (*v.context, *rules), ids, relevant)
    ids = tuple(i for p in dict.fromkeys((*outcomes[0][2], *cited)) for i in v.evidence.get(p, ()))
    return Decision("unstated" if relevant and target in UNSTATED and not conflict else value, rules, ids)


def worlds(v: View) -> dict:
    return {world: judge(v, dict(zip(PROPERTIES, world))) for world in product(*(v.options(p) for p in PROPERTIES))}


def constructed(values: dict, readings=()) -> View:
    v = View.__new__(View)
    v.asset, v.ignored, v.stale = "P", [], []
    v.values = {key: {json.dumps(value) for value in entries} for key, entries in values.items()}
    v.evidence = {key: [] for key in values}
    v.attempted, v.reported, v.steps, v.verified = set(), set(), set(), set()
    v.readings = set(readings)
    return v


EXTRACTED = ("tt_media", "ribbon_loaded", "detection_symptom", "step_verified", "unverified_step", "calibration_verified", "method_inconsistent")
EXACT = ("prompt", "print_method")


def sufficiency(views, target: str, extracted=EXTRACTED, exact=EXACT) -> list[dict]:
    groups = {}
    for v in views:
        table = worlds(v)
        key = (tuple((p, tuple(sorted(v.values.get(p, ())))) for p in exact), tuple(settle(v, table, f).value for f in extracted))
        groups.setdefault(key, {}).setdefault(settle(v, table, target).value, v)
    return [{"target": target, "exact": {p: list(values) for p, values in key[0]}, "extracted": dict(zip(extracted, key[1])),
             "outcomes": {json.dumps(outcome): {p: sorted(values) for p, values in v.values.items()} | {"readings": sorted(v.readings)}
                          for outcome, v in found.items()}}
            for key, found in groups.items() if len(found) > 1]


def readiness_witness() -> list[dict]:
    values = {"prompt": ["paused"], "print_method": [TT], "media_type": [TT], "ribbon_loaded": [False, True]}
    return sufficiency([constructed(values, {("ribbon_loaded", "false")}), constructed({**values, "detection_symptom": [False, True]})], "ready_to_print")


def reference(frame: Frame) -> Gold:
    v = View(frame)
    table = worlds(v)
    out = {f"actions/{action}": settle(v, table, f"actions/{action}") for action in MENU[:-1]}
    prohibited = tuple(action for action in MENU[:-1] if out[f"actions/{action}"].value is False and all(o[f"actions/{action}"][2] for o in table.values()))
    pending = any(d.value is None for d in out.values())
    out["actions/request_observation"] = Decision(pending, (*v.context, "ZT-DIAGNOSE" if pending else "ZT-DEFAULT"))
    verify = out["actions/verify_loading"]
    primitives = {fact: settle(v, table, fact) for fact in PRIMITIVES if fact not in ("verification_due", "needs_observation")}
    primitives |= {"verification_due": Decision(verify.value, verify.rules, verify.evidence), "needs_observation": Decision(pending, ("ZT-DIAGNOSE",))}
    gold = Gold(REVISION, SCOPE, {target: out[target] for target in SCOPE}, prohibited, {fact: primitives[fact] for fact in PRIMITIVES})
    return view_gold(frame, gold) if frame.condition == 'prose' else gold


def definition(condition: str) -> dict:
    spec = files.read_yaml(DEFINITION)
    actions = spec["facts"]["actions"]
    if condition == "structured":
        return spec
    if condition == "structured_permuted":
        return {**spec, "facts": {**spec["facts"], "actions": {**actions, "cards": evidence.permuted(actions["cards"])}}}
    guards = {'set_method_tt': (['idle_prompt'], ['tt_method']), 'set_method_dt': (['idle_prompt', 'tt_method'], []),
              'load_ribbon': (['idle_prompt', 'tt_method'], []), 'start_calibration': (['idle_prompt'], []),
              'calibration_next': ([], ['idle_prompt']), 'cancel_calibration': ([], ['idle_prompt']), 'resume_printing': (['paused_prompt'], [])}
    cards = []
    for card in actions['cards']:
        if card.get('requires') and card['id'] not in guards:
            raise ConfigError(f"prose view needs authored guard mapping for {card['id']}")
        positive, negative = guards[card['id']] if card.get('requires') else ([], [])
        source = {**card.get('source', {}), **({'method_universe': list(DOMAIN['print_method'])} if 'tt_method' in negative else {}),
                  **({'prompt_universe': list(DOMAIN['prompt'])} if 'idle_prompt' in negative else {})}
        cards.append({**{key: value for key, value in card.items() if key != 'requires'},
                      'applies_when': [*card['applies_when'], *({'fact': name} for name in positive)],
                      'excludes': [*card.get('excludes', []), *({'fact': name} for name in negative)],
                      'source': source})
    checks = {'idle_prompt': 'Current evidence establishes that this printer is ready or paused. False when calibration 1 or 2 is established; unstated when the display state is missing, unsupported or not determined.',
              'paused_prompt': 'Current evidence establishes that this printer is paused. False when ready or calibration 1 or 2 is established; unstated when the display state is missing, unsupported or not determined.',
              'tt_method': 'Current evidence establishes thermal-transfer configuration for this printer. False when direct thermal is established; unstated when the method is missing, unsupported or not determined.'}
    return {**spec, "fields": {name: spec["fields"][name] for name in ("asset", "model", "notes")}, "reads": ["asset", "model", "notes"],
            "facts": {**spec["facts"], **{name: {'check': text, 'unstated': True}
                                          for name, text in checks.items()}, "actions": {**actions, "cards": cards}}}


def view_gold(frame, gold):
    if frame.condition != 'prose':
        return gold
    view, extra = View(frame), {}
    for name, prop, values in (('idle_prompt', 'prompt', IDLE), ('paused_prompt', 'prompt', ('paused',)), ('tt_method', 'print_method', (TT,))):
        options = view.options(prop)
        outcomes = {value in values for value in options}
        value = 'unstated' if any(value not in DOMAIN[prop] for value in options) else next(iter(outcomes)) if len(outcomes) == 1 else 'unstated' if view.get(prop) == ABSENT else None
        extra[name] = Decision(value, ('ZT-CURRENT',), tuple(view.evidence.get(prop, ())))
    return replace(gold, primitives={**gold.primitives, **extra})


def behavior_tags(frame):
    view = View(frame)
    return tuple(f'{prop}:{view.get(prop)}' for prop in ('prompt', 'print_method', 'media_type'))


def materialize_command(args):
    return evidence.materialize(args.parent, args.output, condition=args.condition, definition=definition(args.condition),
                                derive=view_gold, tags=lambda frame: (*behavior_tags(frame), 'existing_corpus'), producer='bench.procedures.view_gold',
                                producer_digest=evidence.code_digest(sys.modules[__name__], evidence))


def view_arguments(parser):
    parser.add_argument('parent', type=Path)
    parser.add_argument('--condition', choices=CONDITIONS, default='prose')
    parser.add_argument('--output', type=Path, required=True)


def coverage_arguments(parser):
    parser.add_argument('parent', type=Path)
    parser.add_argument('request', type=Path)
    parser.add_argument('--condition', choices=('structured', 'prose'), default='prose')
    parser.add_argument('--per-template', type=int, default=40)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--role', choices=('fit', 'model_select'), default='fit')
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)


def coverage_command(args):
    with files.committing(args.output):
        if (Path(args.output) / 'benchmark.json').exists():
            raise ConfigError('coverage publication already exists; reuse it or choose a new destination')
        parent = evidence.read(args.parent)
        request = files.read_json(args.request)
        if not isinstance(request, dict) or set(request) != {'seed', 'requests'} or type(request['seed']) is not int or not isinstance(request['requests'], list):
            raise ConfigError('coverage request is {seed, requests: [{target, class, tags, families}]}')
        for item in request['requests']:
            if not isinstance(item, dict) or set(item) != {'target', 'class', 'tags', 'families'} or not isinstance(item['target'], str) or not isinstance(item['class'], str) or not isinstance(item['tags'], list) or not all(isinstance(tag, str) for tag in item['tags']) or type(item['families']) is not int or item['families'] < 1:
                raise ConfigError('coverage entries declare target, semantic class, metadata tags and positive family count')
        if not request['requests'] or args.per_template < 1:
            raise ConfigError('coverage production requires requests and a positive per-template bound')
        definition_ = definition(args.condition)
        parent_families = {row['family'] for row in parent['reference']}
        parent_groups = {group for row in parent['situations'] for group in row['groups']}
        parent_inputs = {digest(row['input']) for row in parent['situations']}
        excluded = []
        for path in args.exclude:
            known = evidence.read(path)
            parent_families.update(row['family'] for row in known['reference'])
            parent_groups.update(group for row in known['situations'] for group in row['groups'])
            parent_inputs.update(digest(row['input']) for row in known['situations'])
            excluded.append({'path': str(path.resolve()), 'sha256': known['manifest_sha256']})
        planned = scenarios(request['seed'], args.per_template, args.variants)
        for row in planned:
            if row['scenario'] in parent_groups or row['scenario'] in parent_families:
                raise ConfigError('coverage seed reuses parent scenario ancestry; choose a new production seed')
        built = frames(planned, conditions=(args.condition,))
        built = [replace(frame, tags=tuple(sorted({*frame.tags, *behavior_tags(frame), 'coverage', f'coverage:{args.role}'}))) for frame in built]
        ref = evidence.Reference(lambda frame: view_gold(frame, reference(frame)), REVISION, SCOPE, tuple(definition_['fields']),
                                 {args.condition: evidence.criterion(definition_)})
        golds = evidence.references(ref, built)
        groups = {}
        for frame in built:
            groups.setdefault(frame.family, []).append(frame)
        selected, achieved = set(), [set() for _ in request['requests']]
        def matches(frame, item):
            value = golds[frame.id].supervision.get(item['target'])
            label = str(value.value).lower() if value is not None else None
            return label == item['class'] and set(item['tags']) <= set(frame.tags)
        duplicate_families = {family for family, members in groups.items() if any(frame.public in parent_inputs for frame in members)}
        for family, members in sorted(groups.items()):
            if family in duplicate_families:
                continue
            hits = [n for n, item in enumerate(request['requests']) if any(matches(frame, item) for frame in members)]
            if not any(len(achieved[n]) < request['requests'][n]['families'] for n in hits):
                continue
            selected.add(family)
            for n in hits:
                achieved[n].add(family)
        built = [frame for frame in built if frame.family in selected]
        assignment = {'seed': request['seed'], 'quotas': {}, 'families': {family: args.role for family in selected},
                      'components': {frame.id: frame.family for frame in built}, 'counted': [frame.id for frame in built],
                      'counts': {role: {} for role in ('fit', 'model_select', 'check')}}
        pack = replace(evidence.load_pack(PACK), created=files.now())
        reviewed = evidence.review(pack, built, (), ref, authored=authored())
        coverage = [{**item, 'achieved_families': len(found), 'state': 'met' if len(found) >= item['families'] else 'unmet'}
                    for item, found in zip(request['requests'], achieved)]
        return evidence.publish(args.output, pack, reviewed, ref, scenarios=[row for row in planned if row['scenario'] in selected],
                                definition=definition_, name='definition.yaml', condition=args.condition, assignment=assignment,
                                producer='bench.procedures.coverage_command', producer_digest=evidence.code_digest(sys.modules[__name__], evidence), seed=request['seed'],
                                extra={'parent': {'path': str(args.parent.resolve()), 'sha256': parent['manifest_sha256']},
                                       'excluded': excluded, 'coverage': coverage, 'duplicate_input_families_skipped': len(duplicate_families)})

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
            ("stale_media", state, [*base, replace(media, indication="handover_note", status="stale", time=-30),
                                    ev(-15, "operator_report", asset, "media_changed", True, name, phrase=phrase)], {"configured_mode", "stale_handover"}),
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


def standard() -> evidence.Reference:
    return evidence.Reference(reference, REVISION, SCOPE, FIELDS, {condition: evidence.criterion(definition(condition)) for condition in CONDITIONS})


def evaluate(built) -> dict:
    return evidence.references(standard(), built)


def build(planned, *, conditions=CONDITIONS, languages=LANGUAGES):
    built = frames(planned, conditions, languages)
    return built, evaluate(built)


def authored() -> dict:
    return {"catalogue": digest({"actions": ACTIONS, "action_text": ACTION_TEXT, "words": WORDS, "names": NAMES,
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
TRANSFER_FACT = "Under the supplied procedure document, a ribbon-load command or calibration step or completion report on this printer is not established as completed."


def transfer_definition(pack) -> dict:
    spec = files.read_yaml(DEFINITION)
    documents = {f"zt411_{key}": {"text": " ".join(f"{rule.id}: {text if rule.id == 'ZT-COMMAND' and text else rule.text}" for rule in pack.rules)}
                 for key, (text, _) in CONVENTIONS.items()}
    return {"name": "zt411_transfer", "fields": spec["fields"], "reads": spec["reads"], "guidance": documents,
            "facts": {f"unverified_step_{key}": {"check": TRANSFER_FACT, "guidance": [f"zt411_{key}"]} for key in CONVENTIONS}}


TRANSFER_SCOPE = tuple(f"unverified_step_{key}" for key in CONVENTIONS)


def transfer_gold(frame: Frame, role: str) -> Gold:
    v = View(frame)
    return Gold(REVISION, TRANSFER_SCOPE,
                primitives={f"unverified_step_{key}": Decision(unverified(v, CONVENTIONS[key][1]), ("ZT-COMMAND", "ZT-VERIFY")) for key in TRANSFER_ROLES[role]})


def transfer_command(args):
    pack = evidence.load_pack(PACK)
    planned = scenarios(args.seed, args.per_template, args.variants)
    built, _ = build(planned)
    assignment = evidence.quota_assignment(built, args, counted={f.id for f in built if f.condition == TRAINED})
    role = {f.id: assignment["families"][assignment["components"][f.id]] for f in built}
    spec = transfer_definition(pack)
    ref = evidence.Reference(lambda frame: transfer_gold(frame, role[frame.id]), REVISION, TRANSFER_SCOPE, FIELDS, dict.fromkeys(CONDITIONS, evidence.criterion(spec)))
    reviewed = evidence.review(pack, built, (), ref, rules=[rule.id for rule in pack.rules], authored=authored())
    return evidence.publish_domain(args, sys.modules[__name__], pack, reviewed, ref, scenarios=planned, assignment=assignment, producer="bench.procedures.transfer",
                                   definition=spec, name="printer-transfer.yaml",
                                   extra={"objective": "bounded guidance transfer across four printer procedure documents; not unseen-manual ingestion or cross-equipment competence",
                                          "documents": {f"zt411_{key}": {"source": "pinned manual rules" if text is None else "authored benchmark convention variant",
                                                                         "roles": [role_ for role_, keys in TRANSFER_ROLES.items() if key in keys]}
                                                        for key, (text, _) in CONVENTIONS.items()},
                                          "revision": REVISION})


def pack_command(args):
    pack = evidence.load_pack(PACK)
    planned = scenarios(args.seed, args.per_template, args.variants)
    ref = standard()
    reviewed = evidence.reviewed_pack(args, sys.modules[__name__], pack, frames(planned), ref)
    assignment = evidence.quota_assignment(reviewed.frames, args, counted=evidence.situation_frames(reviewed.frames, reviewed.golds, TRAINED))
    return evidence.publish_domain(args, sys.modules[__name__], pack, reviewed, ref, scenarios=planned, assignment=assignment, producer="bench.procedures.reference",
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
        done = consistent(state.method, state.media, state.ribbon)
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
    working = consistent(state.method, state.media, state.ribbon) and not state.symptom
    return "printing" if working else "printing_failed"


def digest_state(state: State) -> str:
    return digest(asdict(state))


def progress_key(state: State, gold: Gold):
    return digest_state(state), digest({t: d.value for t, d in gold.decisions.items()})


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


def templates(parser):
    evidence.pack_arguments(parser)
    parser.add_argument("--per-template", type=int, default=40)


def reviewed_templates(parser):
    templates(parser)
    parser.add_argument("--reviews", type=Path, help="reviewer decisions JSONL; every decision names its worksheet binding")


def main():
    evidence.pack_main(sys.modules[__name__], "Zebra ZT411/ZT421 snapshot pack: explicit offline construction; no model or network use",
                       {"pack": ("plan scenarios, build frames/reference and publish the benchmark artifacts", pack_command, reviewed_templates),
                        "view": ("materialize existing evidence under an input view without generating scenarios", materialize_command, view_arguments),
                        "coverage": ("publish bounded additive coverage from an explicit request", coverage_command, coverage_arguments),
                        "transfer": ("publish the bounded four-document guidance-transfer pack: fit A/B, model_select C, check D", transfer_command, templates)})


if __name__ == "__main__":
    main()
