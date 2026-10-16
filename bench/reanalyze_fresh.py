from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from semantic_switch import files
from semantic_switch.compiler import digest
from bench.quality import SUITE_VERSION
from bench.outcomes import output_argument, report


REVERSED_PAIRS = {'decline:4', 'decline:9'}


def reanalyze(source):
    original = files.read_json(source)
    revised = deepcopy(original)
    arms = {}
    for name, rows in revised.items():
        if not isinstance(rows, list):
            continue
        previous = original[name]
        for row in rows:
            if row.get('fact') == 'decline' and row.get('pair') in REVERSED_PAIRS:
                row['label'] = not row['label']
        def counts(values):
            valid = [row for row in values if row.get('p') is not None]
            extreme = [row for row in valid if max(row['p'], 1 - row['p']) >= .999]
            return {'submitted': len(values), 'scored': len(valid),
                    'raw_correct': sum((row['p'] >= .5) == row['label'] for row in valid),
                    'extreme_scored': len(extreme),
                    'extreme_errors': sum((row['p'] >= .5) != row['label'] for row in extreme)}
        gold = lambda values: [{'fact': row['fact'], 'pair': row['pair'], 'label': row['label']} for row in values]
        arms[name] = {'old_gold_sha256': digest(gold(previous)), 'corrected_gold_sha256': digest(gold(rows)),
                      'original_metrics': counts(previous), 'corrected_metrics': counts(rows), 'observations': rows}
    return {'schema': 1, 'suite_version': SUITE_VERSION, 'source': str(source), 'source_sha256': files.sha256_file(source),
            'population': 'retrospective corrected-gold scoring; no new inference or independent evidence',
            'supersedes': 'accuracy and extreme-score error summaries derived from the original gold in this source',
            'presentation': 'original historical presentation and scores retained; neutral rendering applies only to new runs',
            'arms': arms}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    output_argument(parser)
    args = parser.parse_args()
    report(args.output, reanalyze(args.source))
