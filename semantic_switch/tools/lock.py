from __future__ import annotations

from dataclasses import replace

from .. import files
from ..errors import ConfigError
from ..executor import Reuse
from ..judgment import OFF, accept
from ..judge import scored
from ..kernel import Sem, T, U, atoms_of
from ..snapshot import LOCK_SCHEMA, Snapshot, lock_id
from ..observations import Observation
from ..switch import Switch
from .common import input_key, evidence_interval
from .policy import STATES, as_policy, fit_report, head_scores, observed, side_rows, defaults, validate
from ..artifacts import binding_pins, fact_records, family_overlap, load_bundle
from ..results import thaw
from .data import FOLDS, Example, Labels, fold_of


class LockError(ConfigError):
    pass


def judged(sw: Switch, snap: Snapshot, f: int, examples: list[Example], reuse: Reuse | None = None) -> list:
    name = snap.plan.facts[f].name
    out = []
    for start in range(0, len(examples), 128):
        chunk = examples[start:start + 128]
        rows = sw.score([e.data for e in chunk], [name], sources=("judge",), snap=snap, reuse=reuse)
        out.extend(observed(e, row, name) for e, row in zip(chunk, rows))
        if reuse is not None:
            reuse.clear()
    return out


def audit_population(sw, snap, labels):
    from ..compiler import digest
    from ..context import ContextError, bind_input
    from ..evidence import Frame
    from .evaluate import embedded_failures
    plan, units, held = snap.plan, {}, {}
    def unit(data):
        try:
            key = ('situation', Frame(plan, bind_input(plan.fields, data)).full)
        except ContextError:
            key = ('invalid', digest(data))
        if key not in units:
            units[key] = {'data': data, 'facts': {}, 'rules': [], 'tests': []}
        return units[key]
    for f in plan.observed:
        fact = plan.facts[f]
        held[fact.name] = labels.examples(fact.name, 'check', taught=False)
        for example in held[fact.name]:
            unit(example.data)['facts'].setdefault(fact.name, []).append(example)
    for rule in labels.rules('check'):
        unit(rule['input'])['rules'].append(rule)
    for test in plan.tests:
        unit(test.input)['tests'].append(test)
    rule_atoms = {i for rule in plan.rules for i in atoms_of(rule.expr)}
    atoms = {f: [i for i, value in plan.fact_atoms[f] if i in rule_atoms and isinstance(value, Sem) and not plan.facts[f].hidden]
             for f in plan.observed}
    counts = {i: [0, 0, 0, 0, 0] for values in atoms.values() for i in values}
    sides, failures = {}, []
    decision = {'labelled': 0, 'correct': 0, 'wrong': 0, 'abstained': 0, 'no_model_call': 0}
    entries = iter(units.values())
    from itertools import islice
    while chunk := list(islice(entries, 128)):
        reuse = Reuse()
        data = [entry['data'] for entry in chunk]
        try:
            deployed = sw.score(data, demand=[(list(entry['facts']), bool(entry['rules'] or entry['tests'])) for entry in chunk], snap=snap, reuse=reuse)
            combined = sw.score(data, demand=[(list(entry['facts']), False) for entry in chunk], sources=('classifier', 'judge'), snap=snap, reuse=reuse)
            classifier = sw.score(data, demand=[([name for name in entry['facts'] if name in snap.heads], False) for entry in chunk],
                                  sources=('classifier',), snap=snap, reuse=reuse) if snap.heads else None
            for index, (entry, row, baseline) in enumerate(zip(chunk, deployed, combined)):
                for name, examples in entry['facts'].items():
                    f, fact = plan.fact_index[name], plan.fact(name)
                    raw = classifier[index] if classifier is not None and name in snap.heads else None
                    for example in examples:
                        if raw is not None:
                            target = raw.targets[name]
                            sides.setdefault(('classifier', name), {})[example.id] = (observed(example, raw, name), target.settled)
                        if raw is None or not raw.targets[name].settled or raw.error is not None:
                            sides.setdefault(('judge', name), {})[example.id] = (observed(example, baseline, name), False)
                        for i in atoms[f]:
                            want = plan.atoms[i].truth[example.label]
                            if want == U:
                                continue
                            got, values = baseline.conditions.get(plan.atom_keys[i], U), counts[i]
                            values[0] += 1
                            if got == T:
                                values[1] += want == T
                                values[2] += 1
                            elif got != U:
                                values[3] += want != T
                                values[4] += 1
                for expected in entry['rules']:
                    result = row.decision
                    ok = result.status == 'ok'
                    decision['labelled'] += 1
                    decision['correct'] += ok and result.rule == expected['rule']
                    decision['wrong'] += ok and result.rule != expected['rule']
                    decision['abstained'] += not ok
                    decision['no_model_call'] += not any(fact.source == 'judge' for fact in result.facts.values())
                for test in entry['tests']:
                    failures.extend(embedded_failures(test, row.decision))
        finally:
            reuse.clear()
    conditions = {plan.atom_keys[i]: {'labelled': n, 'coverage': round((true + false) / n, 3) if n else None,
                  'true_precision': round(ct / true, 3) if true else None, 'false_precision': round(cf / false, 3) if false else None,
                  'accepted': true + false} for i, (n, ct, true, cf, false) in counts.items() if n}
    decision['coverage'] = round((decision['correct'] + decision['wrong']) / decision['labelled'], 3) if decision['labelled'] else None
    return sides, conditions, decision, failures


def specialization(rt, role='judge'):
    from ..models import Model
    model = rt.models.get(role)
    if not isinstance(model, Model):
        return None, ()
    if model.provenance is not None and model.provenance.gguf_sha256 != rt.identity(role).get('sha256'):
        raise LockError(f'{model.source}: weights differ from their verified provenance {model.provenance.gguf_sha256}; reload the deployment')
    return model.provenance, None if model.provenance is None else tuple(thaw(row) for row in model.provenance.rows)


def native_framing(rt, plan):
    framings = {name: rt.judge(name).adapter.framing for name in plan.bindings if hasattr(rt.judge(name), 'adapter')}
    if len(set(framings.values())) > 1:
        raise LockError(f'bindings {sorted(framings)} use different native answer framings; one switch pins one framing, so bind its targets to models sharing a tokenizer/template')
    return next(iter(framings.values()), None)


def policy_sides(fact, default, report=None):
    if report is None:
        return {side: {"theta": theta, "status": "uncalibrated_default", "calibration": "absent", "fit": None, "state": "absent"}
                for side, theta in defaults(fact, default).items()}
    return {side: {"theta": r["theta"], "status": r["status"], "calibration": "absent" if r["status"] == "uncalibrated_default" else "in_sample",
                   "fit": r, "state": "absent"} for side, r in report.items()}


def lock_payload(plan, examples, *, models, execution, bindings, returned, framing, dataset, sides, membership, lineage, family_seeds,
                 tuned_keys=(), heads=None, artifacts=None, predictor=None, evidence=None, tests):
    from ..artifacts import candidate_digest
    heads = heads or {}
    data = {"family_seeds": family_seeds, "schema": LOCK_SCHEMA, "framing": framing, "dataset": dataset, "artifacts": artifacts or {},
            "predictor_artifacts": predictor or {}, "lineage": lineage, "switch": plan.name, "switch_rev": plan.switch_rev, "spec": plan.spec,
            "models": models, "execution": execution, "bindings": bindings, "returned_models": returned,
            "facts": fact_records(plan, lambda f: {
                "meaning_rev": f.meaning_rev, "examples": [e.record() for e in examples[f.name]],
                "head": heads[f.name].to_json() if f.name in heads else None,
                "policies": {src: sides[src][f.name] for src in sides if f.name in sides[src]},
                "membership": {k: sorted(v) for k, v in membership.get(f.name, {}).items()}}),
            "tuned_keys": sorted(tuned_keys), "evidence": evidence or {}, "tests": tests}
    data["candidate_id"] = candidate_digest(data)
    data["id"] = lock_id(data)
    data["created"] = files.now()
    return data


def out_of_fold(fact, examples, bundle, sw, snap):
    out = []
    by_id = {e.id: e for e in examples}
    ids = set(by_id)
    declared = {r['id'] for r in bundle.manifest['rows']}
    if ids != declared or any((row["group"], row["fold"], row["projection"]) !=
                              (example.group, example.fold, example.projection)
                              for row in bundle.manifest["rows"] for example in (by_id.get(row["id"]),) if example is not None):
        raise LockError(f'{fact.name}: bundle fit observations differ from tune dataset')
    for number in range(FOLDS):
        held = [e for e in examples if fold_of(e) == number]
        head = bundle.fold(number)
        unsupported = set(range(len(fact.keys))) - {e.label for e in examples if fold_of(e) != number}
        if head is None or unsupported:
            reason = ('unobserved training classes: ' + ', '.join(fact.keys[i] for i in sorted(unsupported))) if unsupported else bundle.manifest['folds'][str(number)].get('skip', 'unusable supplied training fold')
            out.extend(Observation(e.id, e.data, e.label, skip=reason, group=e.group) for e in held)
        else:
            if head.classes != fact.keys:
                raise LockError(f'{fact.name}: fold classes differ')
            out.extend(head_scores(fact, head, held, sw, snap))
    return out


def admitted(rt, plan, head_bundles) -> dict:
    for name in plan.bindings:
        if rt.identity(name).get("protocol") and not (rt.identity(name).get("remote") or {}).get("route"):
            raise LockError("locking a hosted judge requires a pinned provider: pass provider=PROVIDER (CLI --provider PROVIDER)")
    specialized = {name: specialization(rt, name) for name in plan.bindings}
    if set(head_bundles) - set(plan.fact_index) or any(plan.fact(name).hidden or plan.fact(name).cards for name in head_bundles if name in plan.fact_index):
        raise LockError("unknown head fact binding")
    return specialized


def overlap_of(rows, labels, examples):
    return family_overlap(rows, labels, [(e.id, e.group, e.projection) for e in examples])


def judge_lineage(specialized, bound, labels, examples, fact):
    return overlap_of(specialized[bound[fact.name]][1], labels, examples)


def developed(sw: Switch, snap: Snapshot, labels: Labels, specialized, bound, head_bundles, reuse: Reuse | None = None):
    plan = snap.plan
    warnings, artifacts, lineage, fitted = [], {}, {}, {}
    heads, policies, membership, development_rows = {}, {}, {}, {}
    for f in plan.observed:
        fact = plan.facts[f]
        tune = labels.examples(fact.name, "tune")
        held = labels.examples(fact.name, "check", taught=False)
        use = membership[fact.name] = {"trained": set(), "tuned": {e.projection for e in tune if not e.taught},
                                       "validated": {e.projection for e in held},
                                       "judge": set()}
        development_scope = judge_lineage(specialized, bound, labels, tune, fact)
        influenced = set(development_scope.overlap)
        audit_scope = judge_lineage(specialized, bound, labels, held, fact)
        use['judge'] = {e.projection for e in (*tune, *held) if e.id in influenced or e.id in audit_scope.overlap}
        use['tuned'] -= use['judge']
        lineage[f'judge-development:{fact.shown}'] = {'state': development_scope.state,
            'overlap': sorted(influenced), 'excluded': len(influenced), 'reason': development_scope.reason}
        if audit_scope.overlap:
            warnings.append(f"{fact.name}: specialization influenced {len(audit_scope.overlap)} audit observations; validation is not independent")
        tune_residue, residue = [e for e in tune if not e.taught and e.id not in influenced], held
        if fact.name in head_bundles:
            bundle = load_bundle(head_bundles[fact.name], fact, labels.training_id)
            artifacts[fact.name] = thaw(bundle.manifest)
            heads[fact.name] = bundle.head
            use['trained'] = {e.projection for e in tune}
            development = out_of_fold(fact, tune, bundle, sw, snap)
            fitted[("classifier", fact.name)] = policy_sides(fact, OFF, None if len({e.label for e in tune}) < len(fact.keys) else fit_report(fact, development, OFF))
            policy = as_policy(fact, fitted[("classifier", fact.name)])
            development_rows[("classifier", fact.name)] = development
            policies.setdefault('classifier', {})[fact.name] = policy
            dev_by_id = {o.id: o for o in development}
            from ..judgment import settle
            def needs_judge(example):
                observation = dev_by_id[example.id]
                if observation.dist is None:
                    return True
                atoms = [a for _, a in plan.fact_atoms[f] if isinstance(a, Sem)]
                if atoms:
                    return any(settle(fact.judgment, observation.dist, a.truth, policy, observation.mass, observation.pending)[0] == U for a in atoms)
                return accept(fact.judgment, observation.dist, policy, observation.mass, observation.pending)[0] is None
            tune_residue = [e for e in tune_residue if needs_judge(e)]
            head_lineage = overlap_of(bundle.manifest['rows'], labels, held)
            lineage[f'classifier:{fact.shown}'] = {'state': head_lineage.state, 'overlap': list(head_lineage.overlap)}
        development_rows[("judge", fact.name)] = judged(sw, snap, f, tune_residue, reuse)
        fitted[("judge", fact.name)] = policy_sides(fact, fact.policy, fit_report(fact, development_rows[("judge", fact.name)], fact.policy))
        policies.setdefault('judge', {})[fact.name] = as_policy(fact, fitted[("judge", fact.name)])
        assessment = judge_lineage(specialized, bound, labels, residue, fact)
        lineage[f'judge:{fact.shown}'] = {'state': assessment.state, 'overlap': list(assessment.overlap), 'reason': assessment.reason}
    return development_rows, fitted, heads, policies, membership, lineage, artifacts, warnings


def returned_models(rt, plan, bound, development_rows) -> dict:
    returned = {name: set() for name in plan.bindings}
    for (source, name), rows in development_rows.items():
        if source == 'judge':
            returned[bound[name]].update(model for row in rows for model in row.observed_models)
    for name, models in returned.items():
        if not rt.identity(name).get('model'):
            continue
        if any(row.error != 'invalid/missing input' and not row.observed_models
               for (source, fact), rows in development_rows.items() if source == 'judge' and bound[fact] == name for row in rows):
            raise LockError('development response is missing required predictor identity')
        if not models:
            from ..protocols import Question
            probe = scored(rt.judge(name), [('Predictor identity probe.', [Question('check', 'The input contains text.')])])[0]
            models.update(model for model in (probe.model, *(score.model for score in probe.scores)) if model is not None)
            if not models or any(score.error for score in probe.scores):
                raise LockError('cannot establish returned predictor identity; no lock published')
        if len(models) > 1:
            raise LockError(f'development returned mixed predictor identities: {sorted(models)}')
    return returned


def audit_sides(sw: Switch, snap: Snapshot, labels: Labels, specialized, bound, returned, development_rows, fitted, lineage, audit_use, frozen_id,
                observations) -> dict:
    plan, sides = snap.plan, {}
    for f in plan.observed:
        fact = plan.facts[f]
        held = labels.examples(fact.name, 'check', taught=False)
        residue = held
        for source in ('classifier', 'judge'):
            if fact.name not in snap.policies.get(source, {}):
                continue
            if source == 'judge':
                assessment = judge_lineage(specialized, bound, labels, residue, fact)
                lineage[f'judge:{fact.shown}'] = {'state': assessment.state, 'overlap': list(assessment.overlap), 'reason': assessment.reason}
            if source == 'classifier':
                found = observations.get(('classifier', fact.name), {})
                observed_rows = [found[e.id][0] for e in held]
                unsettled = [e for e in held if not found[e.id][1]]
            else:
                found = observations.get(('judge', fact.name), {})
                observed_rows = [found[e.id][0] for e in residue]
            development = development_rows[(source, fact.name)]
            if source == 'judge' and returned[bound[fact.name]] and any(row.error != 'invalid/missing input' and
                (not row.observed_models or set(row.observed_models) != returned[bound[fact.name]]) for row in observed_rows):
                raise LockError('audit returned missing/changed predictor identity; frozen baseline cannot expand')
            audited = side_rows(fact, observed_rows)
            base = defaults(fact, OFF if source == 'classifier' else fact.policy)
            thresholds = defaults(fact, snap.policies[source][fact.name])
            record = {name: {**fitted[(source, fact.name)][name], **validate(audited[name], fact.target, thresholds[name], base[name])}
                      for name in audited}
            population = held if source == 'classifier' else residue
            for name, value in record.items():
                theta = thresholds[name]
                resolved = audited[name]
                interval = evidence_interval([(e.group, e.selections, theta is not None and q >= theta, correct) for e, (q, correct) in zip(population, resolved)])
                if lineage[f'{source}:{fact.shown}']['state'] != 'declared_disjoint' or any(e.group in audit_use['consulted_families'] for e in population):
                    interval['interval'] = None
                    interval['reason'] = 'artifact lineage does not establish independence'
                if source == 'judge' and not development:
                    value['state'] = 'insufficient'
                    value['fitting_evidence'] = 'no predictor-independent development observations; uncalibrated default'
                value['interval'] = interval
                value['interval_95'] = interval['interval'] if interval.get('metric') == 'accepted_precision' else None
                value['interval_reason'] = interval.get('reason')
                value['lineage'] = lineage[f'{source}:{fact.shown}']
                value['dataset_id'] = labels.dataset_id
                value['candidate_id'] = frozen_id
                value['audit_use'] = audit_use
            sides.setdefault(source, {})[fact.name] = record
            if source == 'classifier':
                residue = unsettled
    return sides


def family_seeds_of(plan, labels: Labels, membership, specialized, bound) -> dict:
    family_seeds = {}
    for fact, stages in membership.items():
        for stage, projections in stages.items():
            seeds = family_seeds.setdefault(plan.fact(fact).path[0], {}).setdefault(stage, set())
            seeds.update(node for example in labels.examples(fact) if example.projection in projections for node in labels.seeds(example.group))
    family_seeds = {fact: {stage: sorted(nodes) for stage, nodes in stages.items()} for fact, stages in family_seeds.items()}
    for name, (_, trained) in specialized.items():
        for row in trained or ():
            if row['fact'] in family_seeds and any(bound.get(fact.name) == name for fact in plan.facts if fact.path[0] == row['fact']):
                nodes = family_seeds[row['fact']].setdefault('judge', [])
                family_seeds[row['fact']]['judge'] = sorted(set(nodes) | {tuple(node) for node in row.get('family_seeds', ())})
    return family_seeds


def build_lock(rt, path, *, head_bundles=None) -> tuple[dict, Snapshot, Switch]:
    sw = rt.load(path, draft=True)
    labels = Labels(sw)
    snap = replace(sw.snapshot, warnings=sw.snapshot.warnings + tuple(labels.warnings))
    plan = snap.plan
    head_bundles = dict(head_bundles or {})
    specialized = admitted(rt, plan, head_bundles)
    bound = {plan.facts[f].name: plan.binding_of(f) for f in plan.observed}
    reuse = Reuse()
    development_rows, fitted, heads, policies, membership, lineage, artifacts, warnings = developed(sw, snap, labels, specialized, bound, head_bundles, reuse)
    returned = returned_models(rt, plan, bound, development_rows)
    framing = native_framing(rt, plan)
    snap = replace(snap, heads=heads, policies=policies, models={name: rt.identity(name) for name in plan.bindings}, locked=True,
                   returned_models={name: tuple(sorted(models)) for name, models in returned.items()}, framing=framing)
    from ..artifacts import snapshot_digest
    frozen_id = snapshot_digest(snap, rt)
    from .dataset import EvidenceContext, evaluation_plan
    audit_use = EvidenceContext.of(labels).consult(evaluation_plan({'candidate': frozen_id}, scope=[list(plan.facts[f].path) for f in plan.observed]),
                                                   {*(e.group for f in plan.observed for e in labels.examples(plan.facts[f].name, 'check', taught=False)),
                                                    *(labels.group(('key', input_key(plan, it['input']))) for it in labels.rules('check'))})
    reuse.clear()
    audit_rows, condition_report, decision_report, failures = audit_population(sw, snap, labels)
    sides = audit_sides(sw, snap, labels, specialized, bound, returned, development_rows, fitted, lineage, audit_use, frozen_id, audit_rows)
    del audit_rows
    snap = replace(snap, sides=sides, candidate_id=frozen_id, dataset_id=labels.dataset_id, lineage=lineage)
    if failures:
        raise LockError("tests failed against the frozen plan:\n" + "\n".join(failures))
    from .policy import development_curve, score_diagnostics
    diagnostics = {f"{source}:{plan.fact(name).shown}": {"development_curve": development_curve(plan.fact(name), rows),
                                       "scores": score_diagnostics(plan.fact(name), rows, snap.policy(source, plan.fact(name)))}
                   for (source, name), rows in development_rows.items()}
    report = {"diagnostics": diagnostics, "audit_use": audit_use, "conditions": condition_report, "decisions": decision_report}
    tuned_keys = sorted({input_key(plan, it["input"]) for it in labels.rules("tune")})
    family_seeds = family_seeds_of(plan, labels, membership, specialized, bound)
    data = lock_payload(plan, snap.examples, **binding_pins(plan, rt),
                        returned={name: sorted(models) for name, models in returned.items()}, framing=framing.as_dict() if framing else None,
                        dataset=labels.manifest, sides=sides, membership=membership, heads=heads, lineage=lineage, family_seeds=family_seeds,
                        tuned_keys=tuned_keys, artifacts=artifacts,
                        predictor={name: provenance.record() for name, (provenance, _) in specialized.items() if provenance is not None},
                        evidence=report, tests=f"{len(plan.tests)}/{len(plan.tests)} passed against this plan")
    if data["candidate_id"] != frozen_id:
        raise LockError("audit changed frozen candidate identity")
    snap = replace(snap, candidate_id=frozen_id, dataset_id=labels.dataset_id, lineage=lineage, lock_id=data["id"], tuned_keys=frozenset(tuned_keys), warnings=snap.warnings + tuple(warnings),
                   membership={k: {u: frozenset(v) for u, v in m.items()} for k, m in membership.items()},
                   family_seeds=family_seeds, returned_models={name: tuple(models) for name, models in data["returned_models"].items()})
    return data, snap, sw


def lock(rt, path, dry_run: bool = False, *, head_bundles=None) -> dict:
    data, snap, sw = build_lock(rt, path, head_bundles=head_bundles)
    if not dry_run:
        files.publish(sw.paths.lock, data)
    tables = {n: f["policies"] for n, f in data["facts"].items() if f["policies"]} | \
        {f'{n} has "{k}"': m["policies"] for n, f in data["facts"].items() for k, m in f.get("members", {}).items() if m["policies"]}
    policies = {n: f["policies"] | ({"members": {k: m["policies"] for k, m in f["members"].items()}} if "members" in f else {})
                for n, f in data["facts"].items() if f["policies"] or "members" in f}
    targets = {state: [f"{n}: {src} {side}" for n, table in tables.items() for src, sd in table.items()
                       for side, s in sd.items() if s["state"] == state] for state in STATES}
    notes = [f"{n}: {src} {side} side {STATES[s['state']]}"
             + (f" (held-out precision {s['held_out_precision']:.0%} on {s['held_out_accepted']})"
                if s.get("held_out_precision") is not None else "")
             + (f"; check-half coverage {s['coverage_default']:.0%} → {s['coverage_result']:.0%}"
                if s.get("coverage_default") is not None and s.get("coverage_result") is not None
                and s["coverage_default"] != s["coverage_result"] else "")
             for n, table in tables.items() for src, sd in table.items() for side, s in sd.items() if s["state"] != "met"]
    return {"lock": None if dry_run else str(sw.paths.lock), "id": data["id"], "tests": data["tests"],
            "written": [str(p) for p in (sw.paths.dataset, sw.paths.audit, *(() if dry_run else (sw.paths.lock,)))],
            "heads": [n for n, f in data["facts"].items() if f["head"]],
            "tuned_inputs": len(data["tuned_keys"]), "policies": policies, "targets": targets, "evidence": data["evidence"],
            "warnings": notes + list(snap.warnings)}
