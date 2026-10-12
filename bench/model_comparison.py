from __future__ import annotations

import time
ENTERED_NS = time.perf_counter_ns()

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from semantic_switch import Pick, Rate, Check, DEFAULT_JUDGE, PRECISE_JUDGE, Model, files
from bench import quality
from bench.outcomes import failure, measure, runtime as model_runtime, task
from bench.fresh import FRESH, FRESH_PICKS, FRESH_TOPICS, FRESH_QUESTION


def timings(operation, repeats=6):
    measured = [measure(lambda: operation(index)) for index in range(repeats)]
    done = [item for item in measured if item.error is None]
    samples = [item.ms for item in done]
    return {'samples_ms': samples, 'observations': [item.output for item in done], 'samples': repeats,
            'failed': [failure(item, index=index) for index, item in enumerate(measured) if item.error is not None],
            'peak_rss_bytes': max((item.peak_rss_bytes for item in done if item.peak_rss_bytes is not None), default=None),
            'p50_ms': float(np.percentile(samples, 50)) if samples else None, 'p95_ms': float(np.percentile(samples, 95)) if samples else None}


def heterogeneous_workload():
    from bench.openrouter import URGENCY_QUESTION, URGENCY_LEVELS
    guidance = ('Refund means money returned, not replacements. Requests concern only this customer. '
                'A current explicit request counts; quoted, denied or resolved requests do not. '
                'For topic classify billing for payments/refunds, technical for faults; if fixing a fault is requested choose technical. '
                'For benefit classify the explicit remedy: cash for money returned, replacement for another item.')
    case = 'Return my duplicate payment to my card by the end of today. Do not replace the item. The product works. I do not want to cancel or speak with a human.'
    questions = {'refund': Check('The customer currently requests money returned.'),
                 'replacement': Check('The customer currently requests a replacement.'),
                 'cancel': Check('The customer currently requests cancellation.'),
                 'human': Check('The customer currently requests a human agent.'),
                 'broken': Check('The customer currently reports a product fault.'),
                 'topic': Pick('Classify the requested help under the topic rules.', {'billing': 'payments/refunds', 'technical': 'product faults'}),
                 'benefit': Pick('Which remedy is explicitly requested?', {'cash': 'money returned', 'replacement': 'another item'}),
                 'deadline': Rate(URGENCY_QUESTION, URGENCY_LEVELS)}
    return case, guidance, questions, {'refund': True, 'replacement': False, 'cancel': False, 'human': False,
                                     'broken': False, 'topic': 'billing', 'benefit': 'cash', 'deadline': 'today'}

def run(source, *, device='auto', repeats=6, suites=True):
    started = time.perf_counter()
    with model_runtime(Model(source, device=device)) as runtime:
        runtime.judge()
        preparation_ms = (time.perf_counter() - started) * 1000
        item = {'refund': Check('The customer asks for a refund.')}
        case = 'Please refund the duplicate charge.'
        timing = {'first_judgment': timings(lambda _: runtime.ask(case, item), 1),
                  'warm_prefix_new_case': timings(lambda n: runtime.ask(case + f' Ticket {n}', item), repeats),
                  'repeated_case': timings(lambda _: runtime.ask(case, item), repeats)}
        for count in (1, 8):
            questions = {str(k): Check('The customer asks for a refund.') for k in range(count)}
            timing[f'identical_question_branch_packing_{count}'] = timings(lambda n: runtime.ask(case + f' Questions {n}', questions), repeats)
            row = timings(lambda n: runtime.ask_many([case + f' Batch {n} item {k}' for k in range(count)], item), repeats)
            row['inputs_per_second'] = count * 1000 / row['p50_ms'] if row['p50_ms'] else None
            timing[f'finite_batch_{count}'] = row
        workload_case, guidance, questions, gold = heterogeneous_workload()
        heterogeneous = timings(lambda n: runtime.ask(workload_case + f' Record {n}.', questions, guidance=guidance), repeats)
        heterogeneous['expected'] = gold
        heterogeneous['input'] = workload_case
        heterogeneous['tasks'] = {key: task(runtime, item, workload_case, guidance, heterogeneous['observations'][0][key]) for key, item in questions.items()} \
            if heterogeneous['observations'] else None
        timing['heterogeneous_shared_guidance'] = heterogeneous
        from bench.guidance_latency import alternation
        result = {'schema': 2, 'population': 'consulted regression cohorts; not independent domain evidence',
                'boundary': 'complete public operations including JSON materialization; process startup measured by parent',
                'runtime': runtime.info(), 'model_preparation_ms': preparation_ms,
                'device_memory': 'unmeasured', 'actual_offload': 'unmeasured',
                'timings': timing, 'alternating_guidance': alternation(runtime, repeats)}
        if suites:
            original = quality.run(runtime)
            fresh = quality.run(runtime, checks_spec=FRESH, picks_spec=FRESH_PICKS, topics=FRESH_TOPICS, question=FRESH_QUESTION)
            result.update(quality={'summary': quality.summarize(original), 'observations': original},
                          fresh={'summary': quality.summarize(fresh), 'observations': fresh})
        result['operation_peak_rss_bytes'] = max((row['peak_rss_bytes'] for row in timing.values() if row['peak_rss_bytes'] is not None), default=None)
        return result


def isolated(source, output, *, device='auto', repeats=6, suites=True):
    started = time.perf_counter()
    command = [sys.executable, '-m', 'bench.model_comparison', '--model', source, '--device', device,
               '--repeats', str(repeats), '--output', str(output), '--launched-ns', str(time.perf_counter_ns())]
    if not suites:
        command.append('--skip-suites')
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f'isolated native measurement failed: {completed.stderr[-4000:]}')
    result = files.read_json(output)
    result['process_total_ms'] = (time.perf_counter() - started) * 1000
    result['memory_boundary'] = 'fresh subprocess for this model/device arm'
    files.publish(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--skip-suites', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--launched-ns', type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.model:
        if not args.output:
            parser.error('--output is required for an isolated arm')
        imports_ms = (time.perf_counter_ns() - ENTERED_NS) / 1e6
        result = run(args.model, device=args.device, repeats=args.repeats, suites=not args.skip_suites)
        result['process_startup_ms'] = (ENTERED_NS - args.launched_ns) / 1e6 if args.launched_ns is not None else None
        result['module_imports_and_argument_parsing_ms'] = imports_ms
        files.publish(args.output, result)
    else:
        if args.output:
            parser.error('--output requires --model')
        for tag, source in (('e2b', DEFAULT_JUDGE), ('e4b', PRECISE_JUDGE)):
            result = isolated(source, Path(f'bench/results/model-comparison-{tag}.json'), device=args.device,
                              repeats=args.repeats, suites=not args.skip_suites)
            print(json.dumps({'model': tag, 'preparation_ms': result['model_preparation_ms'],
                              'operation_peak_rss_bytes': result['operation_peak_rss_bytes']}), flush=True)


if __name__ == '__main__':
    main()
