from __future__ import annotations

from dataclasses import replace

from .. import files
from ..compiler import FactPlan, digest
from ..errors import ConfigError
from ..judgment import OFF, Policy, accept
from ..judge import scored
from ..kernel import Sem, T, U, atoms_of
from ..snapshot import LOCK_SCHEMA, Snapshot
from ..observations import Observation
from ..switch import Switch
from .common import input_key, evidence_interval
from .policy import STATES, as_policy, fit_report, head_scores, side_rows, defaults, validate
from ..artifacts import fact_records, load_bundle, assess_lineage
from ..results import thaw
from .data import FOLDS, Example, Labels, fold_of


class LockError(ConfigError):
    pass


def judged(sw: Switch, snap: Snapshot, f: int, examples: list[Example]) -> list:
    calls = sw._score([e.data for e in examples], snap, ("judge",), [f]) if examples else []
    return [Observation(e.id, e.data, e.label, c.facts[f].dist if c.facts else None,
                        c.facts[f].mass if c.facts else None, c.facts[f].pending if c.facts else frozenset(),
                        error="invalid/missing input" if c.error or c.facts[f].missing else c.facts[f].error, group=e.group,
                        observed_models=c.facts[f].observed_models if c.facts else ())
            for c, e in zip(calls, examples)]


def settled_by_head(fact: FactPlan, head, policy: Policy, example: Example, sw=None, snap=None) -> bool:
    if sw is None:
        return accept(fact.judgment, list(head.probs(example.text)), policy)[0] is not None
    f = sw.plan.fact_index[fact.name]
    installed = replace(snap or sw.snapshot, heads={fact.name: head}, policies={"classifier": {fact.name: policy}})
    call = sw._score([example.data], installed, ("classifier",), [f])[0]
    return call.error is None and not sw.exec._unsettled(installed, call, f)


def conditions(sw: Switch, snap: Snapshot, labels: Labels) -> dict:
    plan = snap.plan
    used = {i for case in plan.cases for i in atoms_of(case.expr) if isinstance(plan.atoms[i], Sem)
            and not plan.facts[plan.atoms[i].fact].hidden}
    out = {}
    for i in sorted(used):
        sem = plan.atoms[i]
        fact = plan.facts[sem.fact]
        held = labels.examples(fact.name, "check", taught=False)
        if not held:
            continue
        calls = sw._score([e.data for e in held], snap, ("classifier", "judge"), [sem.fact])
        rows = [(c.st[i] if c.st else U, sem.truth[e.label]) for c, e in zip(calls, held) if sem.truth[e.label] != U]
        true = [want == T for got, want in rows if got == T]
        false = [want != T for got, want in rows if got not in (T, U)]
        out[plan.atom_keys[i]] = {"labelled": len(rows), "coverage": round((len(true) + len(false)) / len(rows), 3)
                                  if rows else None,
                                  "true_precision": round(sum(true) / len(true), 3) if true else None,
                                  "false_precision": round(sum(false) / len(false), 3) if false else None,
                                  "accepted": len(true) + len(false)}
    return out


def decisions(sw: Switch, snap: Snapshot, labels: Labels) -> dict:
    cases = [it for it in labels.cases("check")]
    results = sw.run([it["input"] for it in cases], snap) if cases else []
    ok = [(r, it) for r, it in zip(results, cases) if r.status == "ok"]
    correct = sum(r.case == it["case"] for r, it in ok)
    return {"labelled": len(cases), "correct": correct, "wrong": len(ok) - correct, "abstained": len(cases) - len(ok),
            "coverage": round(len(ok) / len(cases), 3) if cases else None,
            "no_model_call": sum(r.cost.get("judge", 0) == 0 for r in results)}


def specialization(rt):
    from ..api import Model
    model = rt.models.get('judge')
    if not isinstance(model, Model):
        return None, ()
    if model.provenance is not None and model.provenance.gguf_sha256 != rt.identity().get('sha256'):
        raise LockError(f'{model.source}: weights differ from their verified provenance {model.provenance.gguf_sha256}; reload the deployment')
    return model.provenance, None if model.provenance is None else tuple(thaw(row) for row in model.provenance.rows)


def policy_sides(fact, default, report=None):
    if report is None:
        return {side: {"theta": theta, "status": "uncalibrated_default", "calibration": "absent", "fit": None, "state": "absent"}
                for side, theta in defaults(fact, default).items()}
    return {side: {"theta": r["theta"], "status": r["status"], "calibration": "absent" if r["status"] == "uncalibrated_default" else "in_sample",
                   "fit": r, "state": "absent"} for side, r in report.items()}


def lock_payload(plan, examples, *, models, execution, returned, framing, dataset, sides, membership, lineage, family_seeds,
                 tuned_keys=(), heads=None, artifacts=None, predictor=None, evidence=None, tests):
    from ..artifacts import candidate_digest
    heads = heads or {}
    data = {"family_seeds": family_seeds, "schema": LOCK_SCHEMA, "framing": framing, "dataset": dataset, "artifacts": artifacts or {},
            "predictor_artifacts": predictor or {}, "lineage": lineage, "switch": plan.name, "switch_rev": plan.switch_rev, "spec": plan.spec,
            "models": models, "execution": execution, "returned_models": returned,
            "facts": fact_records(plan, lambda f: {
                "meaning_rev": f.meaning_rev, "examples": [e.record() for e in examples[f.name]],
                "head": heads[f.name].to_json() if f.name in heads else None,
                "policies": {src: sides[src][f.name] for src in sides if f.name in sides[src]},
                "membership": {k: sorted(v) for k, v in membership.get(f.name, {}).items()}}),
            "tuned_keys": sorted(tuned_keys), "evidence": evidence or {}, "tests": tests}
    data["candidate_id"] = candidate_digest(data)
    data["id"] = digest(data)[:12]
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


def build_lock(rt, path, *, head_bundles=None) -> tuple[dict, Snapshot, Switch]:
    if rt.identity().get("protocol") and not rt.identity().get("provider"):
        raise LockError("locking a hosted judge requires a pinned provider: pass provider=PROVIDER (CLI --provider PROVIDER)")
    sw = Switch(rt, path, draft=True)
    labels = Labels(sw)
    snap = replace(sw.snapshot, warnings=sw.snapshot.warnings + tuple(labels.warnings))
    plan = snap.plan
    provenance, trained = specialization(rt)

    def judge_lineage(examples):
        return assess_lineage(trained, labels, examples)

    warnings = []
    head_bundles = dict(head_bundles or {})
    if set(head_bundles) - set(plan.fact_index) or any(plan.fact(name).hidden or plan.fact(name).cards for name in head_bundles if name in plan.fact_index):
        raise LockError("unknown head fact binding")
    artifacts, lineage, fitted = {}, {}, {}
    heads, policies, sides, membership = {}, {}, {}, {}
    development_rows = {}
    for f in plan.observed:
        fact = plan.facts[f]
        tune = labels.examples(fact.name, "tune")
        held = labels.examples(fact.name, "check", taught=False)
        use = membership[fact.name] = {"trained": set(), "tuned": {e.projection for e in tune if not e.taught},
                                       "validated": {e.projection for e in held},
                                       "judge": set()}
        development_scope = judge_lineage(tune)
        influenced = set(development_scope.overlap)
        audit_scope = judge_lineage(held)
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
            head = heads[fact.name] = bundle.head
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
            head_lineage = assess_lineage(bundle.manifest['rows'], labels, held)
            lineage[f'classifier:{fact.shown}'] = {'state': head_lineage.state, 'overlap': list(head_lineage.overlap)}
        development_rows[("judge", fact.name)] = judged(sw, snap, f, tune_residue)
        fitted[("judge", fact.name)] = policy_sides(fact, fact.policy, fit_report(fact, development_rows[("judge", fact.name)], fact.policy))
        policies.setdefault('judge', {})[fact.name] = as_policy(fact, fitted[("judge", fact.name)])
        assessment = judge_lineage(residue)
        lineage[f'judge:{fact.shown}'] = {'state': assessment.state, 'overlap': list(assessment.overlap), 'reason': assessment.reason}
    returned = {model for (source, _), rows in development_rows.items() if source == 'judge'
                for row in rows for model in row.observed_models}
    requires_receipt = bool(rt.identity().get('model'))
    if requires_receipt and any(row.error != 'invalid/missing input' and not row.observed_models
                               for (source, _), rows in development_rows.items() if source == 'judge' for row in rows):
        raise LockError('development response is missing required predictor identity')
    if requires_receipt and not returned:
        from ..protocols import Question
        probe = scored(rt.judge(), [('Predictor identity probe.', [Question('check', 'The input contains text.')])])[0]
        returned.update(model for model in (probe.model, *(score.model for score in probe.scores)) if model is not None)
        if not returned or any(score.error for score in probe.scores):
            raise LockError('cannot establish returned predictor identity; no lock published')
    if len(returned) > 1:
        raise LockError(f'development returned mixed predictor identities: {sorted(returned)}')
    snap = replace(snap, heads=heads, policies=policies, models={"judge": rt.identity()}, locked=True,
                   returned_models=tuple(sorted(returned)),
                   framing=rt.judge().adapter.framing if hasattr(rt.judge(), "adapter") else None)
    from ..artifacts import snapshot_digest
    frozen_id = snapshot_digest(snap, rt)
    from .dataset import EvidenceContext, evaluation_plan
    audit_use = EvidenceContext.of(labels).consult(evaluation_plan({'candidate': frozen_id}, scope=[list(plan.facts[f].path) for f in plan.observed]),
                                                   {*(e.group for f in plan.observed for e in labels.examples(plan.facts[f].name, 'check', taught=False)),
                                                    *(labels.group(('key', input_key(plan, it['input']))) for it in labels.cases('check'))})
    for f in plan.observed:
        fact = plan.facts[f]
        held = labels.examples(fact.name, 'check', taught=False)
        residue = held
        for source in ('classifier', 'judge'):
            if fact.name not in policies.get(source, {}):
                continue
            if source == 'judge':
                assessment = judge_lineage(residue)
                lineage[f'judge:{fact.shown}'] = {'state': assessment.state, 'overlap': list(assessment.overlap), 'reason': assessment.reason}
            observed = head_scores(fact, heads[fact.name], held, sw, snap) if source == 'classifier' else judged(sw, snap, f, residue)
            development = development_rows[(source, fact.name)]
            if source == 'judge' and returned and any(row.error != 'invalid/missing input' and
                (not row.observed_models or set(row.observed_models) != returned) for row in observed):
                raise LockError('audit returned missing/changed predictor identity; frozen baseline cannot expand')
            audited = side_rows(fact, observed)
            base = defaults(fact, OFF if source == 'classifier' else fact.policy)
            thresholds = defaults(fact, policies[source][fact.name])
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
                residue = [e for e in held if not settled_by_head(fact, heads[fact.name], policies[source][fact.name], e, sw, snap)]
    snap = replace(snap, sides=sides, candidate_id=frozen_id, dataset_id=labels.dataset_id, lineage=lineage)
    from .evaluate import embedded_failures
    failures = [message for test in plan.tests for result in sw.run([test.input], snap)
                for message in embedded_failures(test, result)]
    if failures:
        raise LockError("tests failed against the frozen plan:\n" + "\n".join(failures))
    from .policy import development_curve, score_diagnostics
    diagnostics = {f"{source}:{plan.fact(name).shown}": {"development_curve": development_curve(plan.fact(name), rows),
                                       "scores": score_diagnostics(plan.fact(name), rows, snap.policy(source, plan.fact(name)))}
                   for (source, name), rows in development_rows.items()}
    report = {"diagnostics": diagnostics, "audit_use": audit_use, "conditions": conditions(sw, snap, labels), "decisions": decisions(sw, snap, labels)}
    tuned_keys = sorted({input_key(plan, it["input"]) for it in labels.cases("tune")})
    family_seeds = {}
    for fact, stages in membership.items():
        for stage, projections in stages.items():
            seeds = family_seeds.setdefault(plan.fact(fact).path[0], {}).setdefault(stage, set())
            seeds.update(node for example in labels.examples(fact) if example.projection in projections for node in labels.seeds(example.group))
    family_seeds = {fact: {stage: sorted(nodes) for stage, nodes in stages.items()} for fact, stages in family_seeds.items()}
    for row in trained or ():
        if row['fact'] in family_seeds:
            nodes = family_seeds[row['fact']].setdefault('judge', [])
            family_seeds[row['fact']]['judge'] = sorted(set(nodes) | {tuple(node) for node in row.get('family_seeds', ())})
    data = lock_payload(plan, snap.examples, models={"judge": rt.identity("judge")}, execution={"judge": rt.execution("judge")},
                        returned={"judge": sorted(returned)}, framing=rt.judge().adapter.framing.as_dict() if hasattr(rt.judge(), "adapter") else None,
                        dataset=labels.manifest, sides=sides, membership=membership, heads=heads, lineage=lineage, family_seeds=family_seeds,
                        tuned_keys=tuned_keys, artifacts=artifacts, predictor={'judge': provenance.record()} if provenance is not None else {},
                        evidence=report, tests=f"{len(plan.tests)}/{len(plan.tests)} passed against this plan")
    if data["candidate_id"] != frozen_id:
        raise LockError("audit changed frozen candidate identity")
    snap = replace(snap, candidate_id=frozen_id, dataset_id=labels.dataset_id, lineage=lineage, lock_id=data["id"], tuned_keys=frozenset(tuned_keys), warnings=snap.warnings + tuple(warnings),
                   membership={k: {u: frozenset(v) for u, v in m.items()} for k, m in membership.items()},
                   family_seeds=family_seeds, returned_models=tuple(data["returned_models"]["judge"]))
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
