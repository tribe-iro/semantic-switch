from __future__ import annotations

import argparse
import time
import tempfile
import tracemalloc
from pathlib import Path

from semantic_switch import Runtime, files
from bench.outcomes import destination, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination(args.output)
    report = {'schema': 1, 'boundary': 'public exact-only catalog execution; zero model work', 'buckets': []}
    with tempfile.TemporaryDirectory() as directory, Runtime() as rt:
        for count in (1, 8, 32, 256, 1000):
            spec = {'name': 'scaling', 'fields': {'unit': {'type': 'text', 'required': True}},
                    'facts': {'procedures': {'select': 'Choose applicable procedures', 'cards': [
                        {'id': f'card{i}', 'aliases': [f'UNIT-{i}'], 'requires': f'unit == "UNIT-{i}"',
                         'applies_when': []} for i in range(count)]}}}
            path = Path(directory) / 'catalog.yaml'
            files.publish(path, spec)
            started = time.perf_counter()
            sw = rt.load(path, draft=True)
            compilation = (time.perf_counter() - started) * 1000
            samples = []
            for _ in range(50):
                started = time.perf_counter()
                result = sw({'unit': f'UNIT-{count - 1}'})
                samples.append((time.perf_counter() - started) * 1000)
            tracemalloc.start()
            sw({'unit': f'UNIT-{count - 1}'})
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            report['buckets'].append({'cards': count, 'load_compile_ms': compilation,
                'timing': latency(samples),
                'python_peak_bytes': peak, 'model_work': result.cost.get('judge', 0), 'status': result.status})
    files.publish(args.output, report)


if __name__ == '__main__':
    main()
