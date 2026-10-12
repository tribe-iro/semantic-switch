from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from semantic_switch import Model, DEFAULT_JUDGE, PRECISE_JUDGE, files
from bench.outcomes import runtime
from bench import diagnostics
from semantic_switch.artifacts import read_export
from semantic_switch.inline import Check, ask_many
from semantic_switch.judge import scored
from semantic_switch.metrics import correctness
from semantic_switch.protocols import Question


def metrics(predictions, rows):
    valid = [(p, r['label']) for p, r in zip(predictions, rows) if p is not None]
    choices = [None if p is None else int(np.argmax(p)) for p in predictions]
    counts = correctness((None if choice is None else choice == r['label'], choice is not None and max(p) >= .9)
                         for p, choice, r in zip(predictions, choices, rows))
    if not valid:
        return {**counts, 'ece': None}
    p = np.asarray([v[0] for v in valid])
    gold = np.asarray([v[1] for v in valid])
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == gold
    bins = np.minimum((confidence * 10).astype(int), 9)
    calibration = sum(abs(float(confidence[bins == i].mean() - correct[bins == i].mean())) * int((bins == i).sum()) for i in range(10) if (bins == i).any()) / len(valid)
    return {**counts, 'ece': calibration, 'ece_bins': [int((bins == i).sum()) for i in range(10)],
            'brier': float(((p - np.eye(2)[gold]) ** 2).sum(axis=1).mean())}


def main():
    parser = argparse.ArgumentParser(description='Fixed-policy paired experiments on an explicitly supplied schema-3 audit export')
    parser.add_argument('data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--cascade', action='store_true')
    args = parser.parse_args()
    manifest, all_rows = read_export(args.data, fitting=False)
    rows = [r for r in all_rows if r['split'] == 'check' and not r['taught']]
    if not rows or any(r['kind'] != 'check' or len(r['classes']) != 2 for r in rows):
        raise ValueError('provide untaught binary check observations with pinned check families')
    roles = {'judge': Model(DEFAULT_JUDGE, device=args.device)}
    if args.cascade:
        roles['precise'] = Model(PRECISE_JUDGE, device=args.device)
    outputs, timings, vectors = {name: [] for name in ('binary', 'lettered', 'counterbalanced')}, {}, []
    cascade_rows = []
    with runtime(**roles) as rt:
        for role in roles:
            rt.judge(role)
        for index, row in enumerate(rows):
            question = Question('check', row['claim'], excludes=tuple(row['excludes']))
            state = row['text']
            arms = list(outputs)
            arms = arms[index % len(arms):] + arms[:index % len(arms)]
            results = {}
            for name in arms:
                scored(rt.judge(), [(state, [question])])
                started = time.perf_counter()
                if name == 'counterbalanced':
                    answer = diagnostics.counterbalanced(rt.judge(), state, question)
                    probability = answer.get('probs')
                else:
                    prepared = question if name == 'binary' else Question('pick', question.text, ('true', 'false'), question.excludes)
                    answer = scored(rt.judge(), [(state, [prepared])])[0].scores[0]
                    probability = None if answer.error or answer.low_mass else answer.probs.tolist()
                timings.setdefault(name, []).append((time.perf_counter() - started) * 1000)
                results[name] = probability
            for name in outputs:
                outputs[name].append(results[name])
            if args.cascade:
                spec = {'check': Check(question.text, question.excludes, .9)}
                measurements = {}
                for arm in (('small', 'large', 'cascade') if index % 2 else ('cascade', 'large', 'small')):
                    started = time.perf_counter()
                    if arm == 'cascade':
                        result = diagnostics.cascade(rt, [state], spec)
                        answer = result['merged'][0]['check']
                        measurements['escalated'] = bool(result['residual_indices'])
                    else:
                        answer = ask_many(lambda: rt.judge('judge' if arm == 'small' else 'precise'), [state], spec)[0]['check']
                    timings.setdefault(arm, []).append((time.perf_counter() - started) * 1000)
                    measurements[arm] = answer.as_dict(True)
                cascade_rows.append(measurements)
        report = {'schema': 1, 'dataset_id': manifest['dataset_id'], 'policy': {'true': .9, 'false': .9},
                  'population': 'caller supplied check export; untouched provenance must be declared by the caller',
                  'families': sorted({r['group'] for r in rows}), 'production_promoted': False,
                  'readout': {name: metrics(values, rows) for name, values in outputs.items()},
                  'timings': {name: {'samples': len(values), 'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99))} for name, values in timings.items()},
                  'cascade': cascade_rows, 'provenance': diagnostics.provenance(rt, rows)}
        files.publish(args.output, report)


if __name__ == '__main__':
    main()
