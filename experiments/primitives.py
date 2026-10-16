from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import tracemalloc

import numpy as np

from semantic_switch import DEFAULT_JUDGE, Model, Runtime, files
from semantic_switch.heads import LexicalHead, features
from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS, scored
from bench import diagnostics
from bench.outcomes import destination, latency, provenance, runtime


def measurement(operation, n=500):
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1e6)
    tracemalloc.start()
    for _ in range(100):
        operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'timing': latency(samples, 'us'), 'python_peak_bytes': peak, 'timing_includes_tracemalloc': False, 'allocation_samples': 100}


def run():
    output = {}
    text = 'A customer requests a refund for a duplicate charge. ' * 10
    keys, rng = np.array(sorted(features(text)), dtype=np.int32), np.random.default_rng(42)
    head = LexicalHead(('true', 'false'), keys, rng.normal(0, .1, (len(keys), 2)).astype(np.float32), np.zeros(2, dtype=np.float32))
    output['lexical_complete'] = measurement(lambda: head.probs(text))
    found = features(text)
    output['lexical_reduction'] = measurement(lambda: head.probs_features(found))
    rows = {int(j): r for r, j in enumerate(head.idx)}

    def original_loop():
        z = head.b.astype(np.float64, copy=True)
        for key, value in found.items():
            row = rows.get(key)
            if row is not None:
                z += value * head.w[row]
        e = np.exp(z - z.max())
        return e / e.sum()
    output['lexical_original_loop'] = measurement(original_loop)
    output['probability_difference'] = float(np.abs(original_loop() - head.probs_features(found)).max())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'exact.yaml'
        files.publish(path, {'name': 'exact', 'fields': {'enabled': {'type': 'bool'}}, 'facts': {}, 'rules': [{'id': 'on', 'when': 'enabled', 'then': 'on'}],
                             'otherwise': 'off'})
        with Runtime() as rt:
            switch = rt.load(path)
            output['exact_decision'] = measurement(lambda: switch({'enabled': True}))
            assert not rt.info()['loaded']
    with runtime(Model(DEFAULT_JUDGE, device='gpu')) as rt:
        judge, items = rt.judge(), [(CONFORMANCE_PROBES[0], CONFORMANCE_QUESTIONS[:1])]
        expected = scored(judge, items)[0].scores[0].probs
        zero = diagnostics.activated_suffix(rt, items)
        after = scored(judge, items)[0].scores[0].probs
        output['zero_adapter_control'] = {'max_diff': float(np.abs(expected - np.exp(zero['logp'][0])).max()), 'after_diff': float(np.abs(expected - after).max()),
                                          'errors': zero['errors'], 'nonzero_adapter_validated': False}
        output['provenance'] = provenance(rt, items, boundary='explicit primitive/native controls')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='explicit experiment: primitive and native controls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination(args.output)
    result = run()
    files.publish(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'provenance'}), flush=True)
