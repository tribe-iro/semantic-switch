from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from semantic_switch import ConfigError, Model, files
from bench.outcomes import failure, latency, measure, output_argument, report, runtime as model_runtime
from bench.outcomes import Progress
from semantic_switch.context import ContextError
from semantic_switch.families import FamilyIndex
from semantic_switch.examples import derivation_summary, label_record, metadata_values, target_histories
from semantic_switch.results import thaw
from semantic_switch.selection import occurrences
from semantic_switch.tools.common import test_items, input_key, taught, uses


def _counts(counts):
    submitted, accepted = counts['submitted'], counts['accepted']
    return {**{key: counts[key] for key in ('submitted', 'accepted', 'correct', 'wrong', 'uncertain',
                                          'needs_input', 'failed', 'none', 'unexamined', 'raw_scored', 'raw_correct')},
            'coverage': accepted / submitted if submitted else 0,
            'accepted_precision': counts['correct'] / accepted if accepted else None,
            'settled_agreement': counts['correct'] / submitted if submitted else None,
            'all_row_agreement': counts['raw_correct'] / submitted if submitted and counts['raw_scored'] else None,
            'raw_scored_agreement': counts['raw_correct'] / counts['raw_scored'] if counts['raw_scored'] else None}


def summarize(observations):
    populations = {}
    for row in observations:
        if row.get('case_labelled'):
            population = populations.setdefault('cases', Counter())
            population['submitted'] += 1
            if row['result']['status'] == 'ok':
                population['accepted'] += 1
                correct = thaw(row['result']['rule']) == thaw(row['expected_case'])
                population['correct'] += correct
                population['wrong'] += not correct
            else:
                population[row['result']['status']] += 1
        for name, label in row['labels'].items():
            fact = row['result']['facts'].get(name)
            population = populations.setdefault('fact:' + name, Counter())
            population['submitted'] += 1
            if fact and fact['settled']:
                population['accepted'] += 1
                population['correct'] += fact['value'] == label
                population['wrong'] += fact['value'] != label
            else:
                population[fact['status'] if fact else 'unexamined'] += 1
            raw = row.get('raw_answers', {}).get(name)
            if raw is not None:
                population['raw_scored'] += 1
                correct = raw['value'] == label
                if raw.get('correct') is not correct:
                    raise ConfigError('raw audit correctness disagrees with canonical outcome and label')
                population['raw_correct'] += correct
    return {name: _counts(counts) for name, counts in populations.items()}


def compare(left, right):
    if any(not isinstance(value, dict) or value.get('schema') != 1 or not isinstance(value.get('observations'), list)
           for value in (left, right)):
        raise ConfigError('paired audit comparison requires supported audit artifacts')
    for artifact in (left, right):
        try:
            if any(not isinstance(artifact.get(key), str) or not artifact[key] for key in ('audit_sha256', 'candidate_id')) or any(
                not isinstance(artifact.get(key), dict) for key in ('runtime', 'guidance', 'metrics')):
                raise ConfigError('audit artifact lacks required execution identities')
            if not isinstance(artifact.get('switch_revision'), str) or not isinstance(artifact.get('meanings'), dict):
                raise ConfigError('audit artifact lacks authored meaning identities')
            ids = set()
            for row in artifact['observations']:
                if not isinstance(row.get('id'), str) or not row['id'].strip() or row['id'] in ids or not isinstance(row.get('labels'), dict):
                    raise ConfigError('audit observations require unique IDs and labels')
                ids.add(row['id'])
                if not isinstance(row.get('case_labelled'), bool) or row['result']['status'] not in ('ok', 'uncertain', 'needs_input', 'failed', 'none'):
                    raise ConfigError('invalid audit outcome')
                if not isinstance(row['result'].get('facts'), dict):
                    raise ConfigError('audit outcome lacks fact records')
                for fact in row['result']['facts'].values():
                    if type(fact.get('settled')) is not bool or fact.get('status') not in ('ok', 'uncertain', 'needs_input', 'failed', 'none'):
                        raise ConfigError('invalid audit fact outcome')
                    if fact['settled'] != (fact['status'] == 'ok'):
                        raise ConfigError('audit fact status disagrees with settlement')
                for raw in row.get('raw_answers', {}).values():
                    if type(raw.get('correct')) is not bool:
                        raise ConfigError('invalid raw audit correctness')
            if artifact.get('metrics') != summarize(artifact['observations']):
                raise ConfigError('audit metrics do not match raw outcomes')
            for row in artifact['observations']:
                if isinstance(row['elapsed_ms'], bool) or not isinstance(row['elapsed_ms'], (int, float)) or not np.isfinite(row['elapsed_ms']) or row['elapsed_ms'] < 0:
                    raise ConfigError('audit latency must be finite and nonnegative')
        except (KeyError, TypeError, ValueError) as error:
            raise ConfigError(f'malformed audit observations: {error}') from None
    if any(left[key] != right[key] for key in ('switch_revision', 'meanings')):
        raise ConfigError('paired audit comparison requires the same authored meaning')
    if left.get('audit_sha256') != right.get('audit_sha256') or left.get('guidance') != right.get('guidance'):
        raise ConfigError('paired audit comparison requires the same cohort and bound guidance')
    rows = left['observations'], right['observations']
    if len(rows[0]) != len(rows[1]):
        raise ConfigError('paired audit comparison has different submitted counts')
    pairs = []
    for a, b in zip(*rows):
        if any(a.get(key) != b.get(key) for key in ('id', 'input', 'labels', 'expected_case', 'case_labelled')):
            raise ConfigError('paired audit comparison has different observation order or labels')
        pairs.append({'id': a['id'], 'left_ms': a['elapsed_ms'], 'right_ms': b['elapsed_ms'],
                      'delta_ms': b['elapsed_ms'] - a['elapsed_ms'],
                      'left_status': a['result']['status'], 'right_status': b['result']['status']})
    configurations = {}
    for key in ('requested_device', 'threads', 'build', 'batch_inputs', 'configuration', 'capacity', 'requested_capacity', 'capabilities'):
        a, b = left['runtime'].get(key), right['runtime'].get(key)
        if key == 'capabilities':
            a, b = {k: v for k, v in (a or {}).items() if k != 'counters'}, {k: v for k, v in (b or {}).items() if k != 'counters'}
        if a != b:
            configurations[key] = {'left': a, 'right': b}
    return {'schema': 1, 'audit_sha256': left['audit_sha256'], 'submitted': len(pairs),
            'boundary': 'frozen operating points; no audit-driven threshold or model selection',
            'configuration_differences': configurations, 'matched_error_budget': 'not established',
            'left': {key: left[key] for key in ('candidate_id', 'runtime', 'metrics')},
            'right': {key: right[key] for key in ('candidate_id', 'runtime', 'metrics')},
            'paired_latency': pairs}


def reject_known_overlap(switch, items, lock):
    snap, index, stages = switch.snapshot, FamilyIndex(), {}
    def join(nodes):
        for node in nodes:
            index.join(nodes[0], node)
    for family, record in lock.get('dataset', {}).get('families', {}).items():
        nodes = [tuple(node) for node in record['seeds']]
        nodes += [('group', alias) for alias in record.get('aliases', (family,))]
        join(nodes)
    for manifest in [*lock.get('artifacts', {}).values(), *lock.get('predictor_artifacts', {}).values()]:
        for row in manifest.get('rows', ()):
            nodes = [tuple(node) for node in row.get('family_seeds', ())]
            if row.get('group'):
                nodes.append(('group', row['group']))
            if nodes:
                join(nodes)
                stages.setdefault(nodes[0], set()).add('trained')
    for table in snap.family_seeds.values():
        for stage, nodes in table.items():
            for node in nodes:
                stages.setdefault(tuple(node), set()).add(stage)
    for examples in snap.examples.values():
        for example in examples:
            nodes = [('key', input_key(snap.plan, example.input)), *[('group', group) for group in example.groups],
                     *[('pair', pair) for pair in example.pairs]]
            join(nodes)
            stages.setdefault(nodes[0], set()).add('corrected')
    audit_nodes = []
    for item in items:
        nodes = [('group', group) for group in item['groups']] + [('pair', pair) for pair in item['pairs']]
        try:
            nodes.insert(0, ('key', input_key(snap.plan, item['input'])))
        except ContextError:
            pass
        join(nodes)
        audit_nodes.append(nodes[0])
    exposure = {}
    for node, known in stages.items():
        exposure.setdefault(index.add(node), set()).update(known)
    for n, (item, node) in enumerate(zip(items, audit_nodes)):
        known = set(exposure.get(index.add(node), set()))
        try:
            known.update(uses(snap, item['input']))
            if taught(snap, item['input']):
                known.add('corrected')
        except ContextError:
            pass
        if known:
            raise ConfigError(f'audit row {n + 1}: known overlap with {", ".join(sorted(known))}; use an untouched family')


def run(runtime, definition, audit, *, after_row=None):
    loading_started = time.perf_counter()
    switch = runtime.load(definition)
    snap = switch.snapshot
    audit_digest = files.sha256_file(audit)
    lock_digest = files.sha256_file(switch.paths.lock)
    if not snap.locked or not snap.candidate_id:
        raise ConfigError('policy audit requires a frozen lock with a candidate identity')
    lock = files.read_json(switch.paths.lock)
    items, stale = test_items(switch.paths, switch.plan, str(audit))
    if stale or not items:
        raise ConfigError('policy audit requires nonempty labels matching the frozen meaning')
    ids = set()
    for n, item in enumerate(items):
        try:
            if 'input' not in item:
                raise ConfigError('input is required')
            try:
                identifier = input_key(switch.plan, item['input'])
            except ContextError:
                identifier = f'row-{n}'
            if identifier in ids:
                raise ConfigError('audit IDs (canonical input keys) must be unique; a repeated input is one situation')
            ids.add(identifier)
            item['id'] = identifier
            labels = item.get('facts')
            groups = metadata_values(item.get('groups', ()), 'group')
            if not isinstance(labels, dict) or not (labels or 'rule' in item) or not groups:
                raise ConfigError('labels and case-family membership are required')
            item['groups'] = list(groups)
            item['pairs'] = list(metadata_values(item.get('pairs', ()), 'pair'))
            item['selections'] = [selection.record() for selection in occurrences(item.get('selections', ()))]
            item['label_sources'] = label_record(switch.plan, item['input'], labels, sources=item.get('label_sources', {}))['label_sources']
            if 'rule' in item:
                if item.get('switch_rev') != switch.plan.switch_rev:
                    raise ConfigError('rule label lacks matching switch revision')
                if switch.plan.first and isinstance(item['rule'], list):
                    raise ConfigError('first-resolution rule labels must be a single rule ID')
                values = item['rule'] if isinstance(item['rule'], list) else [item['rule']]
                allowed = {rule.id for rule in switch.plan.rules} | ({'otherwise'} if switch.plan.otherwise is not None else set())
                if not values or any(not isinstance(value, str) or value not in allowed for value in values):
                    raise ConfigError('rule label must name a declared rule or an existing otherwise outcome')
            for name, label in labels.items():
                fact = switch.plan.fact(name)
                labels[name] = fact.value(fact.index(label))
                if item.get('revs', {}).get(name) != fact.meaning_rev:
                    raise ConfigError(f'{name} lacks matching meaning revision')
        except (ConfigError, KeyError, TypeError) as error:
            raise ConfigError(f'{audit}:{n + 1}: {error}') from None
    reject_known_overlap(switch, items, lock)
    runtime.judge()
    cold_load_seconds = time.perf_counter() - loading_started
    observations = []
    progress = Progress('bench.policy_audit', len(items), unit='audit rows')
    for n, item in enumerate(items):
        measured = measure(lambda: switch(item['input']))
        known = {'id': item['id'], 'input': item['input'], 'labels': item['facts'], 'case_labelled': 'rule' in item, 'expected_case': item.get('rule'),
                 'groups': item['groups'], 'pairs': item.get('pairs', []), 'selections': item.get('selections', []),
                 'label_derivations': list(map(derivation_summary, target_histories(item))), 'elapsed_ms': measured.ms, 'peak_rss_bytes': measured.peak_rss_bytes, 'batch': n}
        if measured.error is not None:
            row = failure(measured, **known, raw_answers={}, batch_inputs=None, batch_native_work={},
                          result={'status': 'failed', 'rule': None, 'facts': {name: {'settled': False, 'status': 'failed', 'value': None} for name in item['facts']}})
        else:
            result = measured.result
            raw_answers = {}
            for name, label in item['facts'].items():
                fact = result.facts.get(name)
                if fact and fact.source in ('judge', 'classifier') and fact.scores:
                    distribution = dict(fact.scores)
                    key = max(distribution, key=distribution.get)
                    definition = switch.plan.fact(name)
                    value = definition.value(definition.keys.index(key))
                    raw_answers[name] = {'value': value, 'correct': definition.index(value) == definition.index(label),
                                         'distribution': distribution, 'source': fact.source}
            operation = result.cost.get('operation') or {}
            row = {**known, 'outcome': 'predicted', 'raw_answers': raw_answers, 'batch_inputs': operation.get('inputs'),
                   'batch_native_work': dict(operation.get('native_work', {})), 'result': measured.output}
        observations.append(row)
        progress.record([{'outcome': row['outcome'], 'accepted': row['result']['status'] == 'ok',
                          'correct': thaw(row['result']['rule']) == thaw(row['expected_case']) if row['case_labelled'] and row['result']['status'] == 'ok' else None}])
        if after_row is not None:
            after_row()
    if switch.snapshot.candidate_id != snap.candidate_id or switch.snapshot.policies != snap.policies:
        raise RuntimeError('policy audit changed the frozen candidate')
    if files.sha256_file(audit) != audit_digest or files.sha256_file(switch.paths.lock) != lock_digest:
        raise RuntimeError('policy audit inputs changed during execution')
    described = thaw(switch.describe())
    return {'schema': 1, 'population': 'caller-supplied policy audit; representativeness and independence are caller declarations',
            'audit_sha256': audit_digest, 'lock_sha256': lock_digest, 'lock_id': snap.lock_id, 'candidate_id': snap.candidate_id,
            'switch_revision': described['revision'], 'meanings': {name: fact['revision'] for name, fact in described['facts'].items()},
            'independence': 'no known overlap; external independence remains caller declared',
            'guidance': {name: fact['guidance'] for name, fact in described['facts'].items()}, 'framing': snap.framing.as_dict() if snap.framing else None,
            'runtime': runtime.info(), 'policies': {source: {name: {'true': p.t_true, 'false': p.t_false} for name, p in values.items()}
                                                  for source, values in snap.policies.items()},
            'progress': progress.finish(), 'failed': sum(row['outcome'] == 'failed' for row in observations),
            'memory': {'max_operation_peak_rss_bytes': max(row['peak_rss_bytes'] for row in observations),
                       'device_memory': 'unmeasured'},
            'cold_load_seconds': cold_load_seconds, 'native_work': native_work(observations),
            'latency': latency(row['elapsed_ms'] for row in observations),
            'metrics': summarize(observations), 'observations': observations}


def native_work(observations):
    batches = {row['batch']: row['batch_native_work'] for row in observations}
    return {key: sum(work.get(key, 0) for work in batches.values()) for key in sorted({k for work in batches.values() for k in work})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('definition', type=Path, nargs='?')
    parser.add_argument('audit', type=Path, nargs='?')
    parser.add_argument('--model')
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('LEFT', 'RIGHT'))
    output_argument(parser)
    args = parser.parse_args()
    if args.compare:
        if args.definition or args.audit or args.model:
            parser.error('--compare cannot be combined with a model or audit inputs')
        report(args.output, compare(*(files.read_json(path) for path in args.compare)))
        return
    if not args.definition or not args.audit or not args.model:
        parser.error('definition, audit and --model are required for execution')
    with model_runtime(Model(args.model)) as runtime:
        result = run(runtime, args.definition, args.audit)
    report(args.output, result)


if __name__ == '__main__':
    main()
