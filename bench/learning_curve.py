"""Measure teaching and specialization on unseen synthetic refund requests.

Run from the repository root:
    python -m bench.learning_curve --output bench/results/learning-curve.json
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import tempfile
import time
from pathlib import Path

import yaml

from semantic_switch import Model, Runtime, tools
from semantic_switch.api import DEFAULT_JUDGE

QUESTION = "The customer requests money back for a charge already made."
TRAIN_POS = (
    "Please refund the payment for {item}; I was charged {amount}.",
    "I want my money back for {item}, which I already paid for.",
    "Cancel {item} and refund my previous payment of {amount}.",
    "Devolvam o dinheiro que paguei por {item}: {amount}.",
    "Quero o reembolso da cobrança de {amount} por {item}.",
    "Não quero uma troca de {item}; quero o dinheiro de volta.",
)
TRAIN_NEG = (
    "Do not refund my payment for {item}; please replace it instead.",
    "How much would {item} cost? Is the price {amount}?",
    "Cancel future renewals of {item}; keep the payment already made.",
    "Não quero reembolso de {item}, só uma troca.",
    "Qual é o preço de {item}? Ainda não paguei.",
    "Recebi o reembolso de {item}. Obrigado, não preciso de mais nada.",
)
# Different sentence families, not new order IDs pasted into training templates.
TEST_PAIRS = (
    ("The duplicate debit for {item} must be reversed to my bank account.",
     "The debit for {item} is correct; leave it in place and fix my access."),
    ("I paid {amount} for {item}. Please return that amount, rather than send another item.",
     "I paid {amount} for {item}. Send another item, rather than return that amount."),
    ("Someone told me not to request a refund for {item}. I disagree: return my payment.",
     "Someone told me to request a refund for {item}. I disagree: keep my payment."),
    ("Cobrou-se {amount} por {item} duas vezes. Estornem uma das cobranças.",
     "Cobrou-se {amount} por {item} uma vez. Não estornem; só enviem a fatura."),
    ("Já paguei por {item}, mas desisti. Quero recuperar o valor pago.",
     "Desisti de comprar {item}. Como ainda não paguei, não há valor a devolver."),
    ("Sobre {item}: prefiro a devolução do valor à substituição do produto.",
     "Sobre {item}: prefiro a substituição do produto à devolução do valor."),
)


def dataset(seed: int, per_class: int, test_pairs: int):
    rng = random.Random(seed)
    pool = []
    for i in range(per_class):
        # The same nuisance variables occur on both sides of each boundary.
        item = f"order {1000 + i}"
        amount = f"€{10 + i % 37}"
        pool.extend(({'input': {'body': templates[i % len(templates)].format(item=item, amount=amount)},
                      'value': value}) for templates, value in ((TRAIN_POS, True), (TRAIN_NEG, False)))
    # Shuffle pairs, preserving balanced, nested prefixes at every training size.
    pairs = [pool[i:i + 2] for i in range(0, len(pool), 2)]
    rng.shuffle(pairs)
    pool = [row for pair in pairs for row in pair]
    heldout = []
    for i in range(test_pairs):
        pair = TEST_PAIRS[i % len(TEST_PAIRS)]
        for text, value in zip(pair, (True, False)):
            heldout.append({'input': {'body': text.format(item=f"subscription {9000 + i}",
                                                        amount=f"€{42 + i}")}, 'value': value})
    return pool, heldout


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 3)


def evaluate(sw, heldout):
    sw(heldout[0]['input'])  # warmup excluded
    records, elapsed = [], []
    for item in heldout:
        start = time.perf_counter()
        result = sw(item['input'])
        elapsed.append((time.perf_counter() - start) * 1000)
        records.append((result, item['value']))
    accepted = [(r, y) for r, y in records if r.status == 'ok']
    n = len(records)
    return {
        'items': n, 'accepted': len(accepted), 'coverage': len(accepted) / n,
        'accepted_accuracy': sum(r.answer == y for r, y in accepted) / len(accepted) if accepted else None,
        'wrong_accepted': sum(r.answer != y for r, y in accepted),
        'failed': sum(r.status == 'failed' for r, _ in records),
        'silent_misses': sum(r.status == 'ok' and r.answer is False and y for r, y in records),
        'judge_calls': sum(r.cost.get('judge', 0) for r, _ in records),
        'head_settled': sum(r.cost.get('classifier', 0) for r, _ in records),
        'exact_matches': sum(r.cost.get('example', 0) for r, _ in records),
        'tokens': sum(r.cost.get('tokens', 0) for r, _ in records),
        'latency_ms': {'p50': percentile(elapsed, .5), 'p95': percentile(elapsed, .95),
                       'mean': round(statistics.mean(elapsed), 3)},
    }


def run(rt, sizes, seeds, test_pairs, progress=print):
    curves = []
    rt.judge()  # model startup is reported separately by callers, not as warm latency
    for seed in seeds:
        pool, heldout = dataset(seed, max(sizes), test_pairs)
        test_keys = {json.dumps(r['input'], sort_keys=True) for r in heldout}
        assert not test_keys & {json.dumps(r['input'], sort_keys=True) for r in pool}
        with tempfile.TemporaryDirectory(prefix='ss-learning-') as directory:
            path = Path(directory) / 'refund.yaml'
            for size in sizes:
                examples = pool[:2 * size]
                spec = {'name': 'refund', 'fields': {'body': {'type': 'text', 'required': True}},
                        'reads': ['body'], 'exact_override': False,
                        'facts': {'refund': {'check': QUESTION, 'examples': examples}},
                        'cases': [{'id': 'yes', 'when': 'refund', 'then': True},
                                  {'id': 'no', 'when': 'not refund', 'then': False}],
                        'otherwise': None}
                path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
                path.with_suffix('.lock.json').unlink(missing_ok=True)
                judge = evaluate(rt.load(path, draft=True), heldout)
                row = {'seed': seed, 'points_per_class': size, 'training_points': len(examples),
                       'judge': judge, 'cascade': None, 'train_ms': None}
                if size:
                    spec['facts']['refund']['specialize'] = 'lexical'
                    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
                    start = time.perf_counter()
                    tools.lock(rt, path)  # actual package training and artifact path
                    row['train_ms'] = round((time.perf_counter() - start) * 1000, 3)
                    row['cascade'] = evaluate(rt.load(path), heldout)
                    assert row['cascade']['exact_matches'] == 0
                curves.append(row)
                progress(f"seed={seed} points/class={size}: judge={judge['accepted']}/{judge['items']} accepted; "
                         f"cascade heads={row['cascade']['head_settled'] if size else '-'}", flush=True)
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='0,3,10,30,100', help='labelled points per class')
    parser.add_argument('--seeds', default='7,19,42')
    parser.add_argument('--test-pairs', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'auto'])
    parser.add_argument('--judge', default=DEFAULT_JUDGE)
    parser.add_argument('--output', type=Path, default=Path('bench/results/learning-curve.json'))
    args = parser.parse_args()
    sizes, seeds = [int(v) for v in args.sizes.split(',')], [int(v) for v in args.seeds.split(',')]
    if not sizes or min(sizes) < 0 or args.test_pairs < 1:
        parser.error('sizes must be nonnegative and test-pairs must be positive')
    start = time.perf_counter()
    with Runtime({'judge': Model(args.judge, device=args.device, threads=args.threads)}) as rt:
        rt.judge()
        startup_ms = round((time.perf_counter() - start) * 1000, 3)
        curves = run(rt, sizes, seeds, args.test_pairs)
    report = {'task': QUESTION, 'synthetic': True, 'python': platform.python_version(),
              'machine': platform.machine(), 'configuration': vars(args) | {'output': str(args.output)},
              'judge_startup_ms': startup_ms, 'thresholds': [.90, .10],
              'notes': ['Training and test sentence families are disjoint; test pairs include negation.',
                        'Exact overrides are disabled. Thresholds are fixed, not tuned on test labels.',
                        'Heads train on all included points.',
                        'Tokens and judge calls are compute proxies, not measured dollar costs.',
                        'Synthetic results do not establish real-world or Jev parity.'], 'curves': curves}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
