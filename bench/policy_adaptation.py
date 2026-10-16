from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import functools
import json
from pathlib import Path
import time

from semantic_switch import Check, ConfigError, Model, Pick, files, tools
from semantic_switch.compiler import compile_spec
from semantic_switch.evidence import Actor, LabelSource, Verification, each_target, label_record, reviewed_content, situation_record
from semantic_switch.tools.data import Labels, fold_of
from semantic_switch.artifacts import artifact_digest
from bench.outcomes import Progress
from semantic_switch.training import Engine, fitted
from bench.outcomes import classification_metrics, destination, failure, measure, report, runtime

COHORT = Path(__file__).parents[1] / 'tests/fixtures/policy-adaptation.json'


def pair_metrics(rows):
    pairs = defaultdict(list)
    for row in rows:
        for identifier in row.get('pairs', ([row['pair']] if row.get('pair') else [])):
            pairs[identifier].append(row)
    out = []
    for name, values in pairs.items():
        if len(values) != 2:
            raise ConfigError(f'{name}: policy pairs require exactly two observations')
        expected = [row['expected'].get('raw_value', row['expected']['value']) for row in values]
        predicted = [row.get('raw_value') for row in values]
        out.append({'pair': name, 'relationship': 'invariance' if expected[0] == expected[1] else 'change',
                    'correct': all(row.get('raw_usable') and row.get('raw_value') == row['expected'].get('raw_value', row['expected']['value']) for row in values),
                    'predicted': predicted, 'expected': expected})
    return {'available': bool(out), 'pairs': out, 'success': sum(pair['correct'] for pair in out), 'submitted': len(out)}


def shortcut_ceiling(data):
    counts = defaultdict(lambda: defaultdict(int))
    for policy in data['policies']:
        for case in policy['cases']:
            counts[(policy['task_family'], case['text'])][json.dumps(case['gold'])] += 1
    total = sum(sum(labels.values()) for labels in counts.values())
    return {'submitted': total, 'case_only_best_possible_correct': sum(max(labels.values()) for labels in counts.values()),
            'instruction_dependent_cases': sum(len(labels) > 1 for labels in counts.values()),
            'scope': 'authored pairs; diagnostic bound, no fitted predictor'}


def audit_window(bound, installed, cases, window_id, cache_state):
    frozen = installed.snapshot.candidate_id
    measured = measure(lambda: installed.run([case.data for case in cases]), runtime=bound)
    known = {'id': window_id, 'members': [case.id for case in cases], 'cache_state': cache_state, 'total_ms': measured.ms,
             'measurement_scope': 'full audit window including result serialization'}
    if measured.error:
        return None, failure(measured, **known)
    if len(measured.result) != len(cases):
        raise ConfigError('audit returned incomplete ordered results')
    if installed.snapshot.candidate_id != frozen:
        raise ConfigError('audit changed frozen acceptance')
    return measured.result, {**known, 'native_work': measured.work, 'host_peak_bytes': measured.peak_rss_bytes,
                             'memory_scope': 'process resident-set peak within the window; peak reset at window start', 'device_memory': 'unmeasured'}


def materialize(runtime, directory, cohort=COHORT):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(cohort).read_text())
    if data.get('population') != 'authored adaptation development; not independent domain evidence':
        raise ConfigError('unexpected training population; consulted regression data must not be reused')
    exports, definitions = [], []
    for policy in data['policies']:
        judgment = Check(policy['question'], unstated=policy.get('unstated', False), guidance=policy['guidance']) if policy.get('kind') == 'check' else Pick(policy['question'], policy['options'], guidance=policy['guidance'])
        path = directory / (policy['id'] + '.yaml')
        judgment.save(path, rt=runtime)
        name = path.stem
        plan = runtime.load(path, draft=True).plan
        revision = f"{policy['id']}@{artifact_digest({key: value for key, value in policy.items() if key != 'cases'})}"
        rows = [{'input': {'message': case['text']}, 'facts': {name: case['gold']}, 'group': case['family'],
                 'pairs': case.get('pairs', ([case['pair']] if case.get('pair') else []))} for case in policy['cases']]
        files.publish(path.with_suffix('.test.jsonl'), ''.join(json.dumps(authored(plan, row, fixture_sources(plan, row, cohort, f"{policy['id']}#{index}", revision))) + '\n'
                                                               for index, row in enumerate(rows)))
        switch = runtime.load(path, draft=True)
        labels = Labels(switch)
        if not labels.examples(name, 'tune') or not labels.examples(name, 'check'):
            raise ConfigError(f'{name}: unsplittable cohort; supply more independent cases')
        exported = tools.export(runtime, path, native=True, task_family=policy['task_family'])
        exports.append({'path': exported['path'], 'task_family': policy['task_family'],
                        'role': 'fit' if policy['role'] == 'fit' else 'acceptance_fit'})
        definitions.append({'path': str(path), 'fact': name, 'role': policy['role'], 'task_family': policy['task_family'],
                            'case_tags': {case['family']: case['tags'] for case in policy['cases'] if case.get('tags')}})
    return {'exports': exports, 'definitions': definitions, 'cohort_digest': artifact_digest(data), 'shortcut_diagnostic': shortcut_ceiling(data), 'population': data['population']}



ROLES = ('fit', 'model_select', 'acceptance_fit', 'check')


def family_roles(examples):
    roles = {}
    for example in examples:
        role = 'check' if example.split == 'check' else ('fit', 'fit', 'fit', 'model_select', 'acceptance_fit')[fold_of(example)] if example.split == 'tune' else None
        if role is None or roles.setdefault(example.group, role) != role:
            raise ConfigError(f'{example.group}: family has no single fit/select/accept/check role')
    return roles


def lora_job(definition, export, directory, *, assets, quant, device, objective, epochs, max_tokens, modules=None):
    from semantic_trainer.jobs import prepare
    return prepare(definition, [export], directory, assets_manifest=assets, quant=quant, device=device, objective=objective,
                   budget={'max_tokens': max_tokens, 'max_sequence_tokens': 1024, 'max_seconds': 5400},
                   recipe={'epochs': epochs, **({'target_modules': modules} if modules else {})})


def suites(runtime, directory, *, assets, quant='Q4_0', device='cpu', epochs=2):
    from bench.fresh import FRESH
    from bench.quality import CHECKS
    directory = Path(directory)
    claims = {**CHECKS, **FRESH}
    if len(claims) != len(CHECKS) + len(FRESH):
        raise ConfigError('regression suite fact names collide')
    spec = {'name': 'suites', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
            'facts': {fact: {'check': claim, **({'excludes': list(excludes)} if excludes else {})} for fact, (claim, excludes, _) in claims.items()}}
    rows = [{'input': {'message': text}, 'facts': {fact: label}, 'group': f'suite:{fact}:{tag}', 'pairs': [f'{fact}:{k}']}
            for fact, (_, _, pairs) in claims.items() for k, (positive, negative, tag) in enumerate(pairs) for text, label in ((positive, True), (negative, False))]

    def publish(stage, selected):
        path = directory / stage / 'suites.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        files.publish(path, spec)
        plan = compile_spec(spec, path.name, path.parent)
        files.publish(path.with_suffix('.test.jsonl'), ''.join(json.dumps(authored(plan, row, each_target(row['facts'], [SUITES.record()])), ensure_ascii=False) + '\n'
                                                               for row in selected))
        switch = runtime.load(path, draft=True)
        labels = Labels(switch)
        return path, [e for name in claims for e in labels.examples(name)]

    fitting, examples = publish('fit', rows)
    roles = family_roles(examples)
    family = {(e.fact, e.data['message']): e.group for e in examples}
    audit, held = publish('audit', [row for row in rows if roles[family[next(iter(row['facts'])), row['input']['message']]] in ('acceptance_fit', 'check')])
    if {(e.group, e.split) for e in held} != {(e.group, e.split) for e in examples if roles[e.group] in ('acceptance_fit', 'check')}:
        raise ConfigError('audit definition changed family partitions')
    exported = tools.export(runtime, fitting, native=True, task_family='regression:suites')
    groups = {role: sorted(group for group, value in roles.items() if value == role) for role in ROLES}
    job = lora_job(fitting, exported['path'], directory, assets=assets, quant=quant, device=device,
                   objective='specialization', epochs=epochs, max_tokens=200000)
    tags = {f'suite:{fact}:{tag}': [tag] for fact, (_, _, pairs) in claims.items() for *_, tag in pairs}
    pilot = {'population': 'consulted regression suites (bench.quality CHECKS, bench.fresh FRESH); development pilot only, not clean bench evidence',
             'families': {role: len(values) for role, values in groups.items()},
             'observations': {role: sum(roles[e.group] == role for e in examples) for role in groups},
             'groups': groups, 'job': job['job'], 'commands': job['commands'], 'support': job['support'],
             'definitions': [{'path': str(audit), 'fact': fact, 'role': 'target', 'task_family': f'suite:{fact}', 'case_tags': tags} for fact in claims]}
    files.publish(directory / 'pilot.json', pilot)
    return pilot


CANDIDATES = {'all-linear': None, 'query-value': ['q_proj', 'v_proj']}
SMALL_CONTEXT = Path(__file__).parents[1] / 'tests/fixtures/small-context-cohort-v3.json'


CODEX = LabelSource('authored', 'codex', actor=Actor('agent', 'gpt-6.1-sol', '01a117f0-2b2f-7f31-b6a0-8b4ef9465af0'))
SUITES = LabelSource('authored', 'bench.quality CHECKS and bench.fresh FRESH (author unrecorded)', actor=Actor('unknown'))
AUTHORS = {COHORT.name: CODEX}


def author(fixture) -> LabelSource:
    return AUTHORS.get(Path(fixture).name) or LabelSource('authored', f'{Path(fixture).name} (author unrecorded)', actor=Actor('unknown'))


@functools.cache
def recomputation(fixture):
    path = Path(fixture).with_suffix('.verification.json')
    found = files.read_json(path)
    if found is not None and found['fixture_sha256'] != files.sha256_file(fixture):
        raise ConfigError(f'{path} recomputed another revision of {fixture}; recompute the changed fixture again')
    return found


def fixture_sources(plan, row, fixture, key, criterion) -> dict:
    data, found, out = situation_record(plan, row['input'])['input'], recomputation(Path(fixture).resolve()), {}
    for name, value in row['facts'].items():
        source = author(fixture)
        if found is not None and key in found['answers'] and found['answers'][key] == value:
            source = source.verified(Verification('label', found['verifier'], Actor.parse(found['actor']), found['method'], criterion,
                                                  reviewed_content('label', data, (name,), value), 'confirm'))
        out[name] = [source.record()]
    return out


def criterion(data, policy) -> str:
    return f"{policy}@{artifact_digest({'policy': data['policies'][policy], 'judgment': data['judgments'][data['policies'][policy]['judgment']]})}"


def authored(plan, row, sources):
    return label_record(plan, row['input'], row['facts'], sources=sources, groups=[row['group']], pairs=row.get('pairs', []),
                        selections=[{'selected_by': 'manual'}], created='bench.policy_adaptation')


def cohort_spec(data, rows):
    facts = {row['fact']: (row['policy'], list(row['input'])) for row in rows}
    return {'name': 'cohort', 'fields': data['fields'], 'reads': list(data['fields']),
            'guidance': {policy: {'text': data['policies'][policy]['guidance']} for policy in sorted({policy for policy, _ in facts.values()})},
            'facts': {fact: {**data['judgments'][data['policies'][policy]['judgment']], 'guidance': [policy], 'reads': reads} for fact, (policy, reads) in facts.items()}}


def cohort_rows(data):
    return [{'case_id': f"{situation['id']}:{variant['id']}:{language}", 'situation': situation['id'], 'domain': situation['domain'],
             'variant': variant['id'], 'contrast': variant.get('contrast', 'base'), 'language': language, 'policy': variant['policy'],
             'fact': fact, 'tags': variant['tags'], 'gold': variant['gold'], 'input': variant[language],
             'record': {'input': variant[language], 'facts': {fact: variant['gold']}, 'group': 'small-context:' + situation['id'],
                        'pairs': [f"{situation['id']}:{variant['id']}"]}}
            for situation in data['situations'] for variant in situation['variants'] for language in ('en', 'pt')
            for fact in (variant['policy'] + ('_reference' if 'reference' in variant[language] else ''),)]


def publish_definition(path, spec, rows, cohort, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    files.publish(path, spec)
    plan = compile_spec(spec, path.name, path.parent)
    files.publish(path.with_suffix('.test.jsonl'), ''.join(json.dumps(authored(plan, row['record'], fixture_sources(plan, row['record'], cohort, row['case_id'], criterion(data, row['policy']))),
                                                                  ensure_ascii=False) + '\n' for row in rows))


def materialize_cohort(runtime, directory, cohort=None):
    directory = Path(directory)
    frozen = files.read_json(directory / 'assignment.json')
    cohort = cohort or (frozen['cohort'] if frozen else SMALL_CONTEXT)
    data = json.loads(Path(cohort).read_text())
    if frozen and frozen['source_sha256'] != artifact_digest(data):
        raise ConfigError(f'{directory}: cohort source {cohort} changed after its family assignment was frozen from {frozen["cohort"]}; materialize into a new directory')
    rows = cohort_rows(data)
    path = directory / 'cohort.yaml'
    publish_definition(path, cohort_spec(data, rows), rows, cohort, data)
    switch = runtime.load(path, draft=True)
    labels = Labels(switch)
    examples = [e for fact in labels.by_fact for e in labels.examples(fact)]
    roles = family_roles(examples)
    assignment = {'schema': 1, 'cohort': str(cohort), 'source_sha256': artifact_digest(data), 'dataset_id': labels.dataset_id, 'definition': str(path),
                  'population': data['population'], 'target_precision': data['target_precision'], 'minimum_coverage': data['minimum_coverage'],
                  'objective': 'specialization', 'objective_scope': 'held-out situations under the same authored policies; not unseen-policy instruction transfer',
                  'situations': {row['situation']: labels.group(('group', row['record']['group'])) for row in rows}, 'families': roles,
                  'counts': {role: {'families': sum(value == role for value in roles.values()), 'observations': sum(roles[e.group] == role for e in examples)} for role in ROLES}}
    assignment['assignment_sha256'] = artifact_digest([assignment['situations'], roles])
    if frozen and frozen['assignment_sha256'] != assignment['assignment_sha256']:
        raise ConfigError('cohort family assignment changed; materialize into a new directory')
    files.publish(directory / 'assignment.json', assignment)
    return assignment, switch, data, rows


def cohort_pilot(runtime, directory, cohort_directory, *, assets, quant='Q4_0', device='cpu', epochs=2):
    directory = Path(directory)
    assignment, _, data, rows = materialize_cohort(runtime, cohort_directory)
    role = {row['case_id']: assignment['families'][assignment['situations'][row['situation']]] for row in rows}
    exported = tools.export(runtime, assignment['definition'], to=directory / 'cohort.export.jsonl', native=True, task_family='cohort:small-context')
    groups = {name: sorted(family for family, value in assignment['families'].items() if value == name) for name in ROLES}
    if not all(groups.values()):
        raise ConfigError(f'unsplittable cohort: empty roles {[name for name, values in groups.items() if not values]}')
    jobs = {candidate: lora_job(assignment['definition'], exported['path'], directory / f'job-{candidate}', assets=assets, quant=quant, device=device,
                                objective=assignment['objective'], epochs=epochs, max_tokens=300000, modules=modules)
            for candidate, modules in CANDIDATES.items()}
    definitions = []
    held_out = [row for row in rows if role[row['case_id']] in ('acceptance_fit', 'check')]
    for fact in sorted({row['fact'] for row in held_out}):
        held = [row for row in held_out if row['fact'] == fact]
        path = directory / 'audit' / f'{fact}.yaml'
        publish_definition(path, cohort_spec(data, held), held, assignment['cohort'], data)
        audit = Labels(runtime.load(path, draft=True))
        by_input = {json.dumps(row['input'], sort_keys=True): row for row in held}
        examples = audit.examples(fact)
        if {(e.group, e.split) for e in examples} != {(e.group, 'tune' if assignment['families'][e.group] == 'acceptance_fit' else 'check') for e in examples}:
            raise ConfigError(f'{fact}: audit definition changed family partitions')
        definitions.append({'path': str(path), 'fact': fact, 'role': 'target', 'task_family': f'cohort:{fact}',
                            'case_tags': {e.id: by_input[json.dumps(e.data, sort_keys=True)]['tags'] for e in examples}})
    pilot = {'population': assignment['population'], 'objective': assignment['objective'], 'objective_scope': assignment['objective_scope'],
             'assignment': {key: assignment[key] for key in ('source_sha256', 'assignment_sha256', 'dataset_id', 'counts', 'target_precision', 'minimum_coverage')},
             'groups': groups, 'jobs': {candidate: {'job': job['job'], 'commands': job['commands']} for candidate, job in jobs.items()},
             'definitions': definitions}
    files.publish(directory / 'pilot.json', pilot)
    return pilot


def summarize(rows):
    groups, classes, mechanisms, families = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row['task_family']].append(row)
        families[row['family']].append(row)
        classes[json.dumps(row['expected']['raw_value'])].append(row)
        for tag in row.get('tags', ()):
            mechanisms[tag].append(row)
    return {'metrics': classification_metrics(rows), 'pairs': pair_metrics(rows),
            'policy_macro_raw_accuracy': sum(sum(row['raw_value'] == row['expected']['raw_value'] and row['raw_usable'] for row in values) / len(values) for values in groups.values()) / len(groups) if groups else None,
            'class_metrics': {key: classification_metrics(values) for key, values in classes.items()},
            'mechanism_metrics': {key: classification_metrics(values) for key, values in mechanisms.items()} or None,
            'policy_metrics': {key: classification_metrics(values) for key, values in groups.items()},
            'family_metrics': {key: classification_metrics(values) for key, values in sorted(families.items())}}


def accepted_outcome(row):
    return 'abstained' if row['result']['status'] != 'ok' else 'correct' if row['result']['value'] == row['expected']['value'] else 'wrong'


def paired_changes(arms):
    paired = []
    if arms:
        baseline = {(row['case_id'], row['task_family']): row for row in arms[0]['rows']}
        if len(baseline) != len(arms[0]['rows']):
            raise ConfigError('duplicate baseline comparison observations')
        for arm in arms[1:]:
            candidate = {(row['case_id'], row['task_family']) for row in arm['rows']}
            if candidate != set(baseline) or len(candidate) != len(arm['rows']):
                raise ConfigError('candidate comparison observations differ from the ordered cohort')
            changes = []
            for row in arm['rows']:
                before = baseline[(row['case_id'], row['task_family'])]
                correct_before = before['raw_usable'] and before['raw_value'] == before['expected']['raw_value']
                correct_after = row['raw_usable'] and row['raw_value'] == row['expected']['raw_value']
                changes.append({'case_id': row['case_id'], 'family': row['family'], 'before_correct': correct_before, 'after_correct': correct_after,
                                'before_accepted': accepted_outcome(before), 'after_accepted': accepted_outcome(row)})
            families = defaultdict(list)
            for row in changes:
                families[row['family']].append(row)
            family_changes = [{'family': family, 'before_all_correct': all(row['before_correct'] for row in values),
                               'after_all_correct': all(row['after_correct'] for row in values)} for family, values in families.items()]
            paired.append({'baseline': arms[0]['name'], 'candidate': arm['name'], 'rows': changes,
                           'families': family_changes,
                           'family_fixes': sum(not row['before_all_correct'] and row['after_all_correct'] for row in family_changes),
                           'family_regressions': sum(row['before_all_correct'] and not row['after_all_correct'] for row in family_changes),
                           'fixes': sum(not row['before_correct'] and row['after_correct'] for row in changes),
                           'regressions': sum(row['before_correct'] and not row['after_correct'] for row in changes),
                           'unchanged_mistakes': sum(not row['before_correct'] and not row['after_correct'] for row in changes),
                           'accepted_transitions': dict(sorted(Counter(f"{row['before_accepted']}->{row['after_accepted']}" for row in changes).items()))})
    return paired


def compare(definitions, models, output, selection=None):
    arms = []
    for name, model in models.items():
        started = time.perf_counter()
        with Engine(model, own_judges=True) as bound:
            bound.judge()
            load_ms = (time.perf_counter() - started) * 1000
            rows, windows, locks, progress = [], [], {}, []
            for definition in definitions:
                if definition['role'] != 'target':
                    continue
                path, fact, tags = Path(definition['path']), definition['fact'], definition.get('case_tags', {})
                switch = bound.load(path, draft=True)
                labels = Labels(switch)
                audit = labels.examples(fact, 'check', taught=False)
                lock = locks[path] if path in locks else locks.setdefault(path, tools.lock(bound, path))
                installed = bound.load(path)
                frozen = installed.snapshot.candidate_id
                bound.invalidate()
                window_id = artifact_digest([name, str(path), frozen, [case.id for case in audit]])
                decisions, window = audit_window(bound, installed, audit, window_id, 'invalidated prefix and input cache; model already loaded')
                windows.append(window)
                replay, window = audit_window(bound, installed, audit, window_id + ':replay', 'immediate replay; bounded cache retention follows the recorded native configuration')
                window['accepted_agreement'] = None if decisions is None or replay is None else sum((a.status, a.answer) == (b.status, b.answer) and
                                                  {key: (value.status, value.value) for key, value in a.facts.items()} ==
                                                  {key: (value.status, value.value) for key, value in b.facts.items()}
                                                  for a, b in zip(decisions, replay))
                windows.append(window)
                counter = Progress(f'bench.policy_adaptation {name} {fact}', len(audit), unit='held-out cases')
                for case, result in zip(audit, decisions or [None] * len(audit)):
                    answer = None if result is None else result.facts[fact]
                    dist = [] if answer is None else list(answer.scores.values())
                    index = max(range(len(dist)), key=dist.__getitem__) if dist else None
                    rows.append({'case_id': case.id, 'family': case.group, 'task_family': definition['task_family'],
                                 'pairs': list(case.pairs),
                                 'tags': tags.get(case.id) or tags.get(case.group.removeprefix('group:'), []),
                                 'expected': {'slot': case.label, 'raw_value': switch.plan.fact(fact).value(case.label),
                                              'value': None if case.label in switch.plan.fact(fact).judgment.silence else switch.plan.fact(fact).value(case.label),
                                              'status': 'needs_input' if case.label in switch.plan.fact(fact).judgment.silence else 'ok'},
                                 'raw_value': None if index is None else switch.plan.fact(fact).value(index),
                                 'raw_usable': index is not None, 'raw_mass': None if answer is None else answer.mass,
                                 'result': {'status': 'failed', 'value': None} if answer is None else answer.as_dict(True),
                                 'decision': None if result is None else result.as_dict(True), 'candidate_id': frozen,
                                 'lock': lock, 'window_id': window_id, 'outcome': 'failed' if result is None else 'predicted'})
                    counter.record([{'outcome': rows[-1]['outcome'], 'accepted': rows[-1]['result']['status'] == 'ok',
                                     'correct': rows[-1]['raw_usable'] and rows[-1]['raw_value'] == rows[-1]['expected']['raw_value']}])
                progress.append(counter.finish())
            arms.append({'name': name, 'identity': bound.identity(), 'configuration': bound.info(), 'model_load_ms': load_ms, 'rows': rows, 'windows': windows, 'progress': progress, **summarize(rows)})
    artifact = {'schema': 2, 'population': 'authored held-out situations; policy-overlap specialization and calibration-assisted deployment',
                'objective': 'specialization', 'paired': paired_changes(arms),
                'cohort_counts': {'observations': len(arms[0]['rows']) if arms else 0,
                                  'situation_components': len({row['family'] for row in arms[0]['rows']}) if arms else 0,
                                  'policy_variants': sum(definition['role'] == 'target' for definition in definitions)},
                'quality_promotion': False, 'selection': selection, 'arms': arms}
    return report(output, artifact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('definitions', help='definitions JSON to compare, or the pilot directory with --suites')
    parser.add_argument('--base', required=True)
    parser.add_argument('--suites', action='store_true', help='materialize the regression-suite LoRA pilot (fit/select/accept/check families, native export, prepared job)')
    parser.add_argument('--assets', help='trainer asset manifest digest; --base must be its pipeline-built quantized base')
    parser.add_argument('--quant', default='Q4_0')
    parser.add_argument('--device', default='cpu', help='fitting device: cpu, xpu:N or cuda:N')
    parser.add_argument('--cohort', help='materialized small-context cohort directory; with it, DEFINITIONS is the pilot directory to create')
    parser.add_argument('--receipt', action='append', default=[], help='deployment receipt from semantic-trainer select; repeat, each arm named by its run directory')
    parser.add_argument('--precise')
    parser.add_argument('--output', type=Path, help='comparison report path (required unless --suites/--cohort); refused when it exists')
    args = parser.parse_args()
    if args.suites or args.cohort:
        settings = {'assets': args.assets, 'quant': args.quant, 'device': args.device}
        with runtime(Model(args.base)) as bound:
            pilot = cohort_pilot(bound, args.definitions, args.cohort, **settings) if args.cohort else suites(bound, args.definitions, **settings)
        print(json.dumps({key: value for key, value in pilot.items() if key not in ('groups', 'definitions', 'support')}))
        return
    if args.output is None:
        parser.error('a comparison publishes to --output')
    destination(args.output)
    receipts = {Path(receipt).resolve().parents[1].name: receipt for receipt in args.receipt}
    compare(json.loads(Path(args.definitions).read_text())['definitions'],
            {'base': Model(args.base), **{name: fitted(receipt) for name, receipt in receipts.items()},
             **({'precise': Model(args.precise)} if args.precise else {})}, args.output,
            {name: {'rule': report['rule'], 'selected': report['selected'],
                    'candidates': [{key: candidate[key] for key in ('name', 'raw_correct', 'class_regressions', 'p50_ms')} for candidate in report['candidates']]}
             for name, receipt in receipts.items() for report in (files.read_json(Path(receipt))['selection'],)} or None)


if __name__ == '__main__':
    main()
