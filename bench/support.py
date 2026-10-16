from __future__ import annotations

import json
from pathlib import Path
import random
import sys

from semantic_switch import ConfigError, files
from semantic_switch.compiler import digest
from bench import evidence
from bench.evidence import Assertion, Decision, Frame, Gold, Prose

DATA = Path(__file__).parent / "data" / "support"
PACK, DEFINITION = DATA / "pack.json", Path(__file__).parents[1] / "examples" / "data" / "support.yaml"
REVISION, CATALOGUE = "support-reference-3", "support-catalogue-3"
TEXT = {
    "en": {"size_mismatch": ("the size", ["the shoes I got are two sizes too small", "the jacket does not fit, it is the wrong size"]),
           "color_mismatch": ("the color", ["the sweater came in green instead of the navy I ordered", "the color is not the one I picked"]),
           "stain_damage": ("damage", ["the shirt arrived with a big stain on it", "the box was crushed and the lamp is cracked"]),
           "missing_delivery": ("the delivery", ["my package never arrived", "half of the order is missing"]),
           "disputed_charge": ("the charges", ["I was charged twice", "there is a charge on my card I do not recognize"]),
           "refund_initiate": ("a refund", ["please refund me", "I want my money back"]),
           "refund_status": ("a refund status check", ["can you tell me where my refund is?", "has my refund gone through yet?"]),
           "refund_update": ("changes to the refund", ["please send the refund to my other card instead", "the refund amount is wrong, please correct it"]),
           "return_item": ("a return", ["I would like to send it back", "how do I return it?"]),
           "replacement": ("a replacement", ["please send me a replacement", "I would like a new one sent out"]),
           "delivery_change": ("a delivery change", ["please deliver it to my office instead", "can you move the delivery to Friday?"]),
           "order_change": ("an order change", ["please remove the hat from my order before it ships", "I want to cancel one item that has not shipped"])},
    "pt": {"size_mismatch": ("o tamanho", ["os sapatos que recebi são dois números menores", "a jaqueta não serve, o tamanho está errado"]),
           "color_mismatch": ("a cor", ["o suéter veio verde em vez do azul-marinho que pedi", "a cor não é a que eu escolhi"]),
           "stain_damage": ("o dano", ["a camisa chegou com uma mancha grande", "a caixa veio amassada e a luminária está rachada"]),
           "missing_delivery": ("a entrega", ["meu pacote nunca chegou", "metade do pedido está faltando"]),
           "disputed_charge": ("as cobranças", ["fui cobrado duas vezes", "há uma cobrança no meu cartão que não reconheço"]),
           "refund_initiate": ("um reembolso", ["por favor, me reembolsem", "quero meu dinheiro de volta"]),
           "refund_status": ("uma consulta do status do reembolso", ["podem me dizer onde está meu reembolso?", "meu reembolso já foi processado?"]),
           "refund_update": ("mudanças no reembolso", ["por favor, mandem o reembolso para meu outro cartão", "o valor do reembolso está errado, corrijam por favor"]),
           "return_item": ("uma devolução", ["quero mandar de volta", "como faço para devolver?"]),
           "replacement": ("uma troca", ["por favor, me enviem um substituto", "quero que mandem um novo"]),
           "delivery_change": ("uma mudança na entrega", ["por favor, entreguem no meu escritório", "podem mudar a entrega para sexta-feira?"]),
           "order_change": ("uma mudança no pedido", ["por favor, tirem o chapéu do meu pedido antes do envio", "quero cancelar um item que ainda não foi enviado"])}}
TEMPLATES = {
    "en": {"other": "On my other order {other}, {text}", "this": "Regarding this order {order}, {text}",
           "negated_issue": "There is no problem with {topic}.", "negated_request": "I do not want {topic}.",
           "resolved": "The earlier problem with {topic} was already sorted out.", "hypothetical": 'My friend said to write "{text}", but I am not asking for that.',
           "ambiguous": "Great, just great, {topic}, I suppose.", "retain": "Also keep {topic} in place.", "unsupported": "Also, please change my account password. ",
           "hazard": "The cleaner leaked and burned my hand.", "forget": "Actually, forget that. "},
    "pt": {"other": "No meu outro pedido {other}, {text}", "this": "Sobre este pedido {order}, {text}",
           "negated_issue": "Não há problema com {topic}.", "negated_request": "Não quero {topic}.",
           "resolved": "O problema anterior com {topic} já foi resolvido.", "hypothetical": 'Um amigo disse para escrever "{text}", mas não estou pedindo isso.',
           "ambiguous": "Ótimo, simplesmente ótimo, {topic}, imagino.", "retain": "Mantenha também {topic}.",
           "unsupported": "Além disso, troquem a senha da minha conta. ", "hazard": "O produto de limpeza vazou e queimou minha mão.",
           "forget": "Na verdade, esqueçam isso. "}}
ISSUES = ("size_mismatch", "color_mismatch", "stain_damage", "missing_delivery", "disputed_charge")
REQUESTS = ("refund_initiate", "refund_status", "refund_update", "return_item", "replacement", "delivery_change", "order_change")
UNDELIVERED = frozenset({"size_mismatch", "color_mismatch", "stain_damage", "return_item", "order_change"})
LANGUAGES = ("en", "pt")
TRAINED = "message"
CONDITIONS = ("message", "message_permuted")
SCOPE = (*(f"issues/{k}" for k in ISSUES), *(f"requests/{k}" for k in REQUESTS), "hazard", "impact")
LEVELS = ("low", "moderate", "high")
FIELDS = ("order", "delivered_days_ago", "amount", "refund_blocked", "message")
RULE_OF = {"current": "S-CURRENT", "negated": "S-NEGATED", "resolved": "S-RESOLVED", "hypothetical": "S-QUOTED", "superseded": "S-LATEST",
           "ambiguous": "S-AMBIGUOUS"}


def closed(text: str) -> str:
    return text if text.endswith("?") else f"{text}."


def statement(prop, status, phrase, other_order=None, language="en", order=None):
    kind, key = prop.split(":")
    topic, phrases = TEXT[language][key]
    text, t = phrases[phrase % 2], TEMPLATES[language]
    if other_order:
        return closed(t["other"].format(other=other_order, text=text))
    if order:
        return closed(t["this"].format(order=order, text=text))
    if status in ("current", "superseded"):
        return closed(f"{text[0].upper()}{text[1:]}")
    return t[f"negated_{kind}" if status == "negated" else status].format(topic=topic, text=text)


def render(scenario, statements, account, *, frame_id, family, order, tags=(), seeds=(), pairs=(), condition="message", language="en") -> Frame:
    prose, t = Prose("message"), TEMPLATES[language]
    retracted = [f"s{n}" for n, s in enumerate(statements) if s["status"] == "superseded"]
    for n, s in enumerate(statements):
        key = f"s{n}"
        if s.get("retain"):
            prose.add(t["retain"].format(topic=TEXT[language][s["property"].split(":")[1]][0]), key)
        elif s["property"] == "unsupported":
            prose.add(t["unsupported"])
            continue
        elif s["property"] == "hazard":
            prose.add(t["hazard"], key)
        else:
            if s["status"] == "superseded_by":
                prose.add(t["forget"], key)
                for earlier in retracted:
                    prose.spans.setdefault(earlier, []).append(prose.spans[key][-1])
            prose.add(statement(s["property"], s["status"] if s["status"] != "superseded_by" else "current", s.get("phrase", 0), s.get("other"), language,
                                order if s.get("explicit") else None), key)
        prose.add(" ")
    assertions = [Assertion(f"s{n}", s.get("other") or order, s["property"], True, f"m{n}", n, "quoted" if s["status"] == "hypothetical" else "customer_statement",
                            "current" if s["status"] == "superseded_by" else s["status"], spans=tuple(prose.spans[f"s{n}"]))
                  for n, s in enumerate(statements) if s["property"] != "unsupported"]
    data = {"order": order, **{k: v for k, v in account.items() if v is not None}, "message": prose.text.rstrip()}
    return Frame(frame_id, family, scenario, condition, data, tuple(assertions), tuple(sorted(tags)), tuple(seeds), tuple(pairs))


def definition(condition: str) -> dict:
    spec = files.read_yaml(DEFINITION)
    if condition == "message":
        return spec
    return {**spec, "facts": {**spec["facts"], **{name: {**spec["facts"][name], "labels": dict(evidence.permuted(spec["facts"][name]["labels"].items()))}
                                                  for name in ("issues", "requests")}}}


def memberships(frame: Frame) -> dict:
    found = {}
    for a in frame.assertions:
        if a.subject != frame.input["order"]:
            found.setdefault(a.property, []).append(("other", "S-ENTITY", a.id))
            continue
        found.setdefault(a.property, []).append((a.status, RULE_OF.get(a.status, "S-CURRENT"), a.id))
    return found


def member(entries):
    entries = [e for e in entries or () if e[0] != "other"]
    rules = tuple(sorted({rule for _, rule, _ in entries}))
    ids = tuple(i for _, _, i in entries)
    if not entries:
        return Decision(False, ("S-ABSENT",))
    if any(status == "ambiguous" for status, _, _ in entries):
        return Decision(None, rules, ids, ("ambiguous",))
    current = {status == "current" for status, _, _ in entries}
    if current == {True}:
        return Decision(True, rules, ids)
    if any(status == "negated" for status, _, _ in entries) and True in current:
        return Decision(None, (*rules, "S-CONFLICT"), ids, ("contradiction",))
    return Decision(True in current, rules, ids)


def reference(frame: Frame) -> Gold:
    found = memberships(frame)
    decisions = {}
    for kind, catalogue in (("issue", ISSUES), ("request", REQUESTS)):
        for key in catalogue:
            entries = found.get(f"{kind}:{key}")
            decision = member(entries)
            if entries and all(e[0] == "other" for e in entries):
                decision = Decision(False, ("S-ENTITY", "S-ABSENT"))
            decisions[f"{kind}s/{key}"] = decision
    hazard = [e for e in found.get("hazard", ()) if e[0] == "current"]
    decisions["hazard"] = Decision(bool(hazard), ("S-HAZARD",), tuple(i for _, _, i in hazard))
    issues = {k: decisions[f"issues/{k}"].value for k in ISSUES}
    if None in issues.values():
        decisions["impact"] = Decision(None, ("S-IMPACT",), (), tuple(k for k, v in issues.items() if v is None))
    else:
        level = 2 if issues["missing_delivery"] or issues["disputed_charge"] else 1 if issues["stain_damage"] else 0
        decisions["impact"] = Decision(LEVELS[level], ("S-IMPACT",))
    return Gold(REVISION, SCOPE, {t: decisions[t] for t in SCOPE})


ACCOUNT = ("refund_blocked", "delivered_days_ago", "amount")


def composed(value, branch, rules, gold: Gold, targets, *, paths=(), unresolved=()) -> Decision:
    cited = [rule for target in targets for rule in gold.decisions[target].rules]
    return Decision(value, tuple(dict.fromkeys((*rules, *cited))), tuple(dict.fromkeys(i for target in targets for i in gold.decisions[target].evidence)),
                    tuple(unresolved), tuple(paths), tuple(targets), branch)


def response(frame: Frame, gold: Gold) -> Decision:
    v = gold.values()
    memberships = tuple(t for t in SCOPE if t.startswith(("issues/", "requests/")))
    unresolved = tuple(t for t in memberships if v[t] is None)
    if unresolved:
        return composed("clarify", "clarify", ("S-RESPONSE",), gold, unresolved, unresolved=unresolved)
    account, assessed = frame.input, ["requests/refund_initiate"]
    if v["requests/refund_initiate"]:
        missing = v["issues/missing_delivery"]
        blocks = tuple(path for path, hit in (("refund_blocked", bool(account.get("refund_blocked"))),
                                              ("delivered_days_ago", not missing and account.get("delivered_days_ago", 0) > 30),
                                              ("amount", account.get("amount", 0) > 500)) if hit)
        if blocks:
            return composed("refund_ineligible", f"refund_ineligible:{'+'.join(blocks)}", ("S-RESPONSE", "S-REFUND"), gold,
                            (*assessed, "issues/missing_delivery"), paths=blocks)
        issues = tuple(f"issues/{k}" for k in ISSUES if v[f"issues/{k}"])
        if issues:
            return composed("initiate_refund", "initiate_refund", ("S-RESPONSE", "S-REFUND"), gold, tuple(dict.fromkeys((*assessed, "issues/missing_delivery", *issues))),
                            paths=tuple(path for path in ACCOUNT if path in account))
        assessed += [f"issues/{k}" for k in ISSUES]
    for value, keys in (("refund_desk", ("refund_status", "refund_update")), ("return_or_replace", ("return_item", "replacement")),
                        ("change_order", ("delivery_change", "order_change"))):
        group = tuple(f"requests/{k}" for k in keys)
        hit = tuple(t for t in group if v[t])
        if hit:
            return composed(value, f"{value}:{'+'.join(t.split('/')[1] for t in hit)}", ("S-RESPONSE",), gold, (*assessed, *group))
        assessed += group
    return composed("no_action", "no_action", ("S-RESPONSE",), gold, tuple(assessed))


def authored() -> dict:
    return {"catalogue": digest({"text": TEXT, "templates": TEMPLATES, "issues": ISSUES, "requests": REQUESTS, "levels": LEVELS, "id": CATALOGUE})}


def stratum(frame: Frame, gold: Gold, role: str) -> tuple:
    return (frame.condition, role, next((tag for tag in frame.tags if tag in LANGUAGES), "abcd"), "hazard" if gold.values()["hazard"] else "no_hazard")


def plausible(*members) -> bool:
    return "missing_delivery" not in members or not UNDELIVERED & set(members)


def contrasts(rng, issue, request, other_request, phrase, other_order):
    def s(prop, status="current", **kw):
        return {"property": prop, "status": status, "phrase": phrase, **kw}
    base = [s(f"issue:{issue}"), s(f"request:{request}")]
    return [("base", base, {"plain"}),
            ("negated_request", [base[0], s(f"request:{request}", "negated")], {"negation"}),
            ("resolved_issue", [s(f"issue:{issue}", "resolved"), base[1]], {"resolved_issue"}),
            ("quoted_request", [base[0], s(f"request:{request}", "hypothetical")], {"quoted_suggestion"}),
            ("latest_request", [base[0], s(f"request:{request}", "superseded"), s(f"request:{other_request}", "superseded_by")], {"latest_request"}),
            ("retained_request", [base[0], base[1], s(f"request:{other_request}"), s(f"request:{request}", retain=True)], {"retained_priority"}),
            ("other_order", [s(f"issue:{issue}", other=other_order), s(f"request:{request}", explicit=True)], {"entity_change"}),
            ("multiple_issues", [*base, s(f"issue:{rng.choice([k for k in ISSUES if k != issue and plausible(issue, request, k)])}")], {"multiple_issues"}),
            ("unsupported", [base[0], {"property": "unsupported", "status": "current"}], {"unsupported_request"}),
            ("ambiguous", [base[0], s(f"request:{request}", "ambiguous")], {"ambiguous"})]


def scenarios(seed: int, count: int, variants: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for k in range(count):
        issue = rng.choice(ISSUES)
        request = rng.choice([r for r in REQUESTS if plausible(issue, r)])
        other_request = rng.choice([r for r in REQUESTS if r != request and plausible(issue, r)])
        order, other = (f"A{n:05d}" for n in rng.sample(range(10000, 99999), 2))
        account = {"delivered_days_ago": rng.choice((5, 12, 45)), "amount": rng.choice((40.0, 120.0, 800.0)), "refund_blocked": rng.random() < .15}
        sid = f"support-{seed}-{k}"
        for variant, statements, tags in evidence.contrast_quota(contrasts(rng, issue, request, other_request, rng.randrange(2), other), variants, f"{seed}:{sid}"):
            if variant == "base" and issue == "stain_damage" and rng.random() < .5:
                statements = [*statements, {"property": "hazard", "status": "current"}]
                tags = {*tags, "hazard"}
            out.append({"scenario": sid, "variant": variant, "order": order, "account": account, "statements": statements, "tags": sorted(tags), "seed": seed})
    return out


def frames(planned, conditions=CONDITIONS, languages=LANGUAGES) -> list[Frame]:
    out = []
    for row in planned:
        ids = {(c, l): f"{row['scenario']}:{row['variant']}:{c}:{l}" for c in conditions for l in languages}
        out += [render(row["scenario"], row["statements"], row["account"], frame_id=frame_id, family=row["scenario"], order=row["order"],
                       tags=(*row["tags"], condition, language), seeds=(f"scenario:{row['scenario']}",), condition=condition, language=language,
                       pairs=tuple(i for i in ids.values() if i != frame_id)) for (condition, language), frame_id in ids.items()]
    return out


def standard() -> evidence.Reference:
    return evidence.Reference(reference, REVISION, SCOPE, FIELDS, {condition: evidence.criterion(definition(condition)) for condition in CONDITIONS}, response)


def build(built):
    return evidence.references(standard(), built)


def abcd(data_path, annotations_path) -> list[Frame]:
    data = json.loads(Path(data_path).read_text())
    conversations = {c["convo_id"]: (split, c) for split, rows in data.items() for c in rows}
    out = []
    for row in files.read_jsonl(annotations_path):
        split, convo = conversations.get(row["convo_id"], (None, None))
        if convo is None or split != row["split"]:
            raise ConfigError(f"ABCD conversation {row['convo_id']} is absent from {row['split']}; annotate conversations from the pinned data")
        turns = convo["original"][:row["prefix_turns"]]
        text = "\n".join(f"{speaker}: {utterance}" for speaker, utterance in turns)
        prose = Prose("message").add(text)
        assertions = []
        for n, a in enumerate(row["assertions"]):
            start = text.find(a["quote"])
            if start < 0:
                raise ConfigError(f"ABCD {row['convo_id']}: quote {a['quote']!r} is not in the prefix; annotate only what the prefix supports")
            end = start + len(a["quote"])
            assertions.append(Assertion(f"s{n}", a.get("subject", row["order"]), a["property"], True, f"turn{n}", n, a.get("indication", "customer_statement"),
                                        a["status"], spans=(evidence.Span("message", start, end, evidence.sha256_text(text[start:end])),)))
        scenario = convo.get("scenario", {})
        hidden = tuple(str(v) for v in (scenario.get("flow"), scenario.get("subflow")) if v and "_" in str(v))
        frame = Frame(f"abcd:{row['convo_id']}:{row['prefix_turns']}", f"abcd:{row['convo_id']}", f"abcd:{row['convo_id']}", "message",
                      {"order": row["order"], "message": prose.text}, tuple(assertions), ("abcd_realism", f"abcd_{split}"), (f"abcd:{row['convo_id']}",))
        evidence.validate_frame(frame, FIELDS, private=hidden)
        out.append(frame)
    return out


def pack_command(args):
    if bool(args.abcd) != bool(args.abcd_annotations):
        raise ConfigError("--abcd and --abcd-annotations go together")
    pack = evidence.load_pack(PACK)
    planned = scenarios(args.seed, args.scenarios, args.variants)
    built = frames(planned)
    extra = {"catalogue": CATALOGUE, "revision": REVISION}
    if args.abcd:
        realism = abcd(args.abcd, args.abcd_annotations)
        built += realism
        extra["abcd"] = {"data_sha256": files.sha256_file(args.abcd), "annotations_sha256": files.sha256_file(args.abcd_annotations),
                         "frames": len(realism), "slice": "check realism slice, reported separately"}
    ref = standard()
    reviewed = evidence.reviewed_pack(args, sys.modules[__name__], pack, built, ref, slices=("abcd",) if args.abcd else ())
    assignment = evidence.quota_assignment(reviewed.frames, args, counted=evidence.situation_frames(reviewed.frames, reviewed.golds, TRAINED), golds=reviewed.golds,
                                           check_only=lambda frame: "abcd_realism" in frame.tags)
    return evidence.publish_domain(args, sys.modules[__name__], pack, reviewed, ref, scenarios=planned, assignment=assignment, producer="bench.support.reference", extra=extra)


def options(parser):
    evidence.pack_arguments(parser)
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--abcd", type=Path, help="locally downloaded, pinned ABCD conversations JSON")
    parser.add_argument("--abcd-annotations", type=Path, help="hand-authored per-prefix evidence annotations JSONL")
    parser.add_argument("--reviews", type=Path, help="reviewer decisions JSONL; every decision names its worksheet binding")


def main():
    evidence.pack_main(sys.modules[__name__], "support classification pack: explicit offline construction; no model or network use",
                       {"pack": ("plan scenarios, build frames/reference and publish the benchmark artifacts", pack_command, options)})


if __name__ == "__main__":
    main()
