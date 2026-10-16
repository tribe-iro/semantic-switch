from __future__ import annotations

import argparse
import json

from bench.outcomes import destination, failure, latency, measure, output_argument, report, runtime as model_runtime

from semantic_switch import Check, Guidance, Model, DEFAULT_JUDGE


def alternation(runtime, repeats=6):
    policies = [Guidance(f"Policy {i}: Store credit {'does' if i % 2 else 'does not'} count as a refund. " +
                         'The customer account and payment details are evidence. ' * 35) for i in range(3)]
    item = {'refund': Check('The customer requests a refund under this policy.')}
    shapes = []
    for count in (1, 2, 3):
        measured = [measure(lambda: runtime.ask(f'Ticket {count}-{i}: Please give me store credit for the duplicate payment.', item, guidance=policies[i % count]), runtime=runtime)
                    for i in range(repeats * count)]
        done = [row for row in measured[count:] if row.error is None] or [row for row in measured if row.error is None]
        worked = [row.work for row in measured if row.work is not None]
        shapes.append({'policies': count, 'latency': latency(row.ms for row in done), 'samples_ms': [row.ms for row in measured if row.error is None],
                       'observations': [row.output for row in measured if row.error is None],
                       'failed': [failure(row, index=index) for index, row in enumerate(measured) if row.error is not None],
                       'native_work': {key: sum(work[key] for work in worked) for key in worked[0]} if worked else None,
                       'peak_rss_bytes': max((row.peak_rss_bytes for row in measured if row.peak_rss_bytes is not None), default=None)})
    return {'schema': 1, 'runtime': runtime.info(), 'case_shape': 'distinct ~20-token payment tickets',
            'guidance_shape': 'three bounded ~350-token policies',
            'target': 'alternating p50 within 1.5 times warm-prefix/new-case baseline; diagnostic, not a gate', 'shapes': shapes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge', default=DEFAULT_JUDGE)
    output_argument(parser)
    args = parser.parse_args()
    destination(args.output)
    arms = []
    for slots in (1, 4):
        with model_runtime(Model(args.judge, prefix_slots=slots)) as runtime:
            runtime.check('warm', 'This is an input.')
            result = alternation(runtime)
        arms.append({'prefix_slots': slots, **result})
        print(slots, json.dumps(result['shapes']), flush=True)
    report(args.output, {'schema': 2, 'boundary': 'completed public ask; matched distinct cases; serial comparison', 'arms': arms})
