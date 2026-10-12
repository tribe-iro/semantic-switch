from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np

from semantic_switch import ConfigError, Model, files
from semantic_switch.artifacts import artifact_digest
from semantic_switch.backend import reduce_row
from bench.outcomes import runtime


def probe(fixture, model, output, indices=None):
    fixture = files.read_json(Path(fixture))
    presentations = fixture['presentations']
    selected = list(range(len(presentations))) if indices is None else indices
    if any(type(index) is not int or not 0 <= index < len(presentations) for index in selected):
        raise ConfigError('presentation index outside the native fixture')
    rows, captured = [], {}
    with runtime(model) as bound:
        scorer = bound.judge().scorer
        context = scorer.ctx
        for index in selected:
            entry = presentations[index]
            tokens = entry['prefix'] + entry['input'] + entry['question']
            shapes = [('whole', [tokens]), ('segments', [entry['prefix'], entry['input'], entry['question']]),
                      ('chunks37', [tokens[start:start + 37] for start in range(0, len(tokens), 37)])]
            outcomes, baseline = {}, None
            for name, chunks in shapes:
                scorer.invalidate()
                position, answers = 0, []
                for chunk in chunks:
                    entries = [(token, position + offset, 1, 'answer' if position + offset == len(tokens) - 1 else None)
                               for offset, token in enumerate(chunk)]
                    context.run(entries, lambda out, row, sequence: answers.append(context.logits(row).copy()))
                    position += len(chunk)
                if len(answers) != 1:
                    raise ConfigError('native segmentation returned incomplete answer positions')
                logits = answers[0]
                baseline = logits if baseline is None else baseline
                logp, log_mass = reduce_row(logits, tuple(tuple(group) for group in entry['answer_slots']))
                captured[f'{index}_{name}'] = logits
                outcomes[name] = {'chunk_lengths': list(map(len, chunks)), 'logp': logp.astype(float).tolist(),
                                  'log_mass': log_mass, 'argmax': int(logp.argmax()),
                                  'max_full_logit_difference': float(np.max(np.abs(logits - baseline)))}
            rows.append({'index': index, 'presentation': entry, 'tokens': tokens, 'shapes': outcomes})
        report = {'schema': 1, 'population': 'native numerical diagnostic; one sequence without sharing or cache copies',
                  'fixture_digest': artifact_digest(fixture), 'identity': bound.identity(), 'configuration': bound.info(),
                  'sequence': 1, 'rows': rows,
                  'argmax_changes': {shape: sum(row['shapes'][shape]['argmax'] != row['shapes']['whole']['argmax'] for row in rows)
                                    for shape in ('segments', 'chunks37')}}
    archive = io.BytesIO()
    np.savez_compressed(archive, **captured)
    return files.publish(Path(output), report, {Path(output).with_suffix('.npz').name: [archive.getvalue()]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Isolate token segmentation from KV ownership using a single sequence.')
    parser.add_argument('fixture')
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--index', type=int, action='append')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    probe(args.fixture, Model(args.model, device=args.device, threads=args.threads), args.output, args.index)
