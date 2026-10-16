from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from semantic_switch import ConfigError, files
from semantic_switch.training import specialization_request, specialize

from bench import evidence

TASK_FAMILY, QUANT, FIT_SITUATIONS = 'bench:zt411', 'Q8_0', 100
DEVICES = {'fit': 'xpu:0', 'native': 'gpu'}
RECIPE = ('rank=16', 'alpha=32', 'dropout=0.0', 'batch_size=8', 'microbatch=1', 'learning_rate=0.0002', 'schedule=constant', 'epochs=4', 'min_epochs=2',
          'seed=0', 'selection_unit=complete_situation')
BUDGET = ('max_sequence_tokens=1024', 'max_seconds=18000')


def descriptor(staged, work, profile, cadence=None) -> dict:
    measured = files.read_json(profile)
    if not isinstance(measured, dict) or measured.get('kind') != 'fitting-profile':
        raise ConfigError(f'{profile}: not a fitting profile; give the measured profile of this workload')
    assets = measured['configuration']['assets']
    if assets['quant'] != QUANT:
        raise ConfigError(f"{profile}: measured a {assets['quant']} base; the early task experiment fits the {QUANT} base")
    return specialization_request(staged['definition'], staged['labels'], work, devices=DEVICES, assets=assets['manifest'], quant=QUANT, recipe=RECIPE,
                                  budget=BUDGET, fit_situations=FIT_SITUATIONS, task_family=TASK_FAMILY, profile=profile, cadence=cadence)


def main():
    parser = argparse.ArgumentParser(description='Section 21 early printer task experiment: one frozen specialization request over the staged printer pack, '
                                                 'bound to a measured fitting profile, run through the specialization controller; without --cadence the controller reports the measured choices. '
                                                 'Explicit model job; --until finalize loads only the native base for the tune export.')
    parser.add_argument('--pack', type=Path, required=True, help='published printer pack directory')
    parser.add_argument('--stages', type=Path, required=True, help='stage root receiving the verified staged pack')
    parser.add_argument('--profile', type=Path, required=True, help='measured fitting profile of this workload')
    parser.add_argument('--to', type=Path, required=True, help='specialization work directory; it also holds the run corpus binding stage.json')
    parser.add_argument('--cadence', type=int, help='monitoring cadence (eval_every_examples) chosen from the profile\'s measured choices')
    parser.add_argument('--trainer', default=os.environ.get('SEMANTIC_TRAINER'), help='absolute semantic-trainer executable of the fitting environment')
    parser.add_argument('--until', default='finalize', help='last pipeline stage (finalize precedes fitting); fit runs the declared attempt')
    parser.add_argument('--attempt', type=int, help='fit attempt number; a new number fits into a new attempt directory')
    args = parser.parse_args()
    staged = evidence.bind_corpus(args.pack, args.stages, args.to / 'stage.json')
    result = specialize(args.trainer, args.to, descriptor(staged, args.to, args.profile, args.cadence), until=args.until, attempt=args.attempt)
    print(json.dumps({'stage': staged, 'result': result}, default=str), flush=True)
    raise SystemExit(result['exit'])


if __name__ == '__main__':
    main()
