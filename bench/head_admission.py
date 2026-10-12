from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request

from semantic_switch.examples import LabelSource, each_target, label_record
from semantic_switch import Model, DEFAULT_JUDGE, files, tools
from bench.external_audit import BASE, load
from semantic_switch.metrics import Progress
from bench.outcomes import failure, measure, runtime


def admission_work(arms):
    baseline = arms[0]['native_work']
    for arm in arms:
        facts = [fact for row in arm['results'] for fact in row['result'].get('facts', {}).values()]
        arm['classifier_accepted'] = sum(fact.get('source') == 'classifier' and
                                         (fact.get('status') == 'ok' or fact.get('settled') is True) for fact in facts)
        arm['avoided_native_work'] = {key: baseline[key] - arm['native_work'][key] for key in baseline}
        arm['admitted_speedup'] = arm['classifier_accepted'] > 0 and arm['avoided_native_work'].get('submissions', 0) > 0
    return arms


def run():
    from semantic_trainer.lexical import train
    cache = Path('/tmp/decision-audit/data')
    (task, _), (audit, _) = load('s1_arxiv', cache)
    path = cache / 's1_arxiv/train_2024.jsonl'
    if not path.exists():
        with urllib.request.urlopen(BASE + 'data/s1_arxiv/train_2024.jsonl') as response:
            files.publish(path, response.read())
    training = [json.loads(line) for line in path.read_text().splitlines()][:400]
    with TemporaryDirectory(prefix='head-evidence-') as directory:
        switch = Path(directory) / 'category.yaml'
        files.publish(switch, {'name': 'category', 'fields': {'document': {'type': 'text', 'required': True}},
                               'reads': ['document'], 'facts': {'category': {'pick': task['instructions'], 'options': task['labels']}}})
        with runtime(Model(DEFAULT_JUDGE, max_input_tokens=8192, n_ctx=32768)) as rt:
            plan = rt.load(switch, draft=True).plan
            source = [LabelSource('unknown', 'S1MB train_2024').record()]
            rows = [label_record(plan, {'document': row['text']}, {'category': row['label']}, sources=each_target({'category': row['label']}, source),
                                 groups=[row['item_id']]) for row in training]
            files.publish(switch.with_suffix('.test.jsonl'), ''.join(json.dumps(row) + '\n' for row in rows))
            exported = tools.export(rt, switch)
            fitted = train(exported['path'], Path(directory) / 'head', max_iter=5000)
            arms, progress = [], Progress('bench.head_admission', 2 * len(audit), unit='documents')
            for name, heads in [('judge', None), ('lexical', {'category': fitted['path']})]:
                lock = tools.lock(rt, switch, head_bundles=heads)
                measured = measure(lambda: rt.load(switch).run([{'document': row['text']} for row in audit]), runtime=rt)
                if measured.error:
                    rows = [{'id': row['item_id'], 'label': row['label'], 'result': None, 'outcome': 'failed'} for row in audit]
                    arms.append(failure(measured, name=name, lock=lock, seconds=measured.ms / 1000, submitted=len(audit), accepted=0, wrong=0,
                                        classifier_accepted=0, peak_rss_bytes=measured.peak_rss_bytes, results=rows))
                    progress.record(rows)
                    continue
                results, elapsed = measured.result, measured.ms / 1000
                accepted = [(r, row) for r, row in zip(results, audit) if r.status == 'ok']
                classifier_accepted = sum(r.facts['category'].source == 'classifier' and r.facts['category'].status == 'ok' for r in results)
                arms.append({'classifier_accepted': classifier_accepted, 'name': name, 'lock': lock, 'seconds': elapsed, 'submitted': len(results),
                             'accepted': len(accepted), 'wrong': sum(r.facts['category'].value != row['label'] for r, row in accepted),
                             'native_work': measured.work, 'peak_rss_bytes': measured.peak_rss_bytes,
                             'results': [{'id': row['item_id'], 'label': row['label'], 'result': output} for row, output in zip(audit, measured.output)]})
                progress.record({'outcome': 'predicted', 'correct': r.facts['category'].value == row['label'], 'accepted': r.status == 'ok'}
                                for r, row in zip(results, audit))
            admission_work(arms)
            return {'schema': 1, 'population': 'older institutional arXiv labels; separate fixed 2026 category audit',
                    'fitting_rows': len(training), 'export': {k: v for k, v in exported.items() if k != 'path'},
                    'runtime': rt.info(), 'arms': arms, 'progress': progress.finish(),
                    'decision': 'No default or expansion justified by this experiment. Existing optional artifact inference remains explicit; inspect admitted sides and avoided native work.'}


if __name__ == '__main__':
    files.publish(Path('bench/results/head-admission.json'), run())
