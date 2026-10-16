from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bench.evidence import download
from bench.outcomes import destination, failure, measure, output_argument, report, runtime, task, work_argument, work_directory

from semantic_switch import DEFAULT_JUDGE, Model, Pick
from bench.outcomes import Progress

COMMIT = '6bdb5af372fbff70e770128639afd678caf79086'
BASE = f'https://raw.githubusercontent.com/oborchers/decision-model-audit/{COMMIT}/'
NAMES = ('s1_arxiv', 's2_fedreg')


def fetched(name, filename, cache):
    path = Path(cache) / name / filename
    return path, download(BASE + f'data/{name}/{filename}', path)


def load(name, cache):
    result = []
    for filename in ('task.json', 'main.jsonl'):
        path, found = fetched(name, filename, cache)
        raw = path.read_bytes()
        result.append((json.loads(raw) if filename.endswith('.json') else [json.loads(line) for line in raw.splitlines()], found))
    return result


def audit(rt, name, cache) -> dict:
    (document, task_digest), (rows, rows_digest) = load(name, cache)
    pick = Pick(document['instructions'], document['labels'])
    measured = measure(lambda: pick.run([row['text'] for row in rows], rt=rt), runtime=rt)
    progress, observations = Progress(f'bench.external_audit {name}', len(rows), unit='items'), []
    for row, result in zip(rows, measured.result or [None] * len(rows)):
        raw = max(result.scores, key=result.scores.get) if result is not None and result.scores else None
        observations.append({'id': row['item_id'], 'label': row['label'], 'raw_answer': raw, 'raw_correct': raw == row['label'], 'input': row['text'],
                             'outcome': 'failed' if result is None else 'predicted', 'task': task(rt, pick, row['text']),
                             'result': {'value': None, 'status': 'failed'} if result is None else result.as_dict(verbose=True)})
        progress.record([{'correct': observations[-1]['raw_correct'], 'accepted': observations[-1]['result']['status'] == 'ok', 'outcome': observations[-1]['outcome']}])
    settled = [row for row in observations if row['result']['status'] == 'ok']
    return {'name': name, 'task_sha256': task_digest, 'rows_sha256': rows_digest,
            'submitted': len(rows), 'failed': sum(row['outcome'] == 'failed' for row in observations),
            'raw_agreement': sum(row['raw_correct'] for row in observations) / len(rows),
            'settled': len(settled), 'coverage': len(settled) / len(rows),
            'accepted_precision': sum(row['result']['value'] == row['label'] for row in settled) / len(settled) if settled else None,
            'elapsed_seconds': measured.ms / 1000, 'native_work': measured.work, 'peak_rss_bytes': measured.peak_rss_bytes,
            **(failure(measured) if measured.error else {'outcome': 'predicted'}), 'progress': progress.finish(), 'observations': observations}


def run(source, cache, names=NAMES):
    started = time.perf_counter()
    with runtime(Model(source, max_input_tokens=8192, n_ctx=32768)) as rt:
        rt.check('initialization', 'This is an input.')
        load_seconds = time.perf_counter() - started
        tasks = [audit(rt, name, cache) for name in names]
        return {'schema': 2, 'cohort_commit': COMMIT, 'population': 'institutional category agreement on fixed external audit; not domain representativeness',
                'runtime': rt.info(), 'cold_load_seconds': load_seconds, 'tasks': tasks}


def main():
    parser = argparse.ArgumentParser(description='commit-pinned external decision audit through public Pick')
    parser.add_argument('--model', default=DEFAULT_JUDGE)
    output_argument(parser)
    work_argument(parser)
    args = parser.parse_args()
    report(destination(args.output), run(args.model, work_directory(args.output, args.work)))


if __name__ == '__main__':
    main()
