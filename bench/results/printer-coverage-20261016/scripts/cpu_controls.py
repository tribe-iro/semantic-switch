import json
from collections import defaultdict
from pathlib import Path

from bench import evidence, procedures
from semantic_switch import files
from semantic_switch.compiler import compile_spec
from semantic_switch.context import render
from semantic_switch.examples import Frame
from semantic_switch.tools.data import observation_id
from semantic_switch.training import read_export
from semantic_trainer.lexical import compact, fit, matrix


root = Path(__file__).resolve().parent
cohorts = files.read_json(root / 'cohorts.json')
run = next(Path('work/zt411-t7/specializations').glob('*/fit/*/resolved-job.json'))
resolved = files.read_json(run)
roles = {row['id']: row['role'] for row in resolved['memberships']}
_, original = read_export(resolved['sources'][0]['path'])
original = [row for row in original if roles.get(row['id']) in ('fit', 'model_select')]
plan = compile_spec(files.read_yaml(procedures.DEFINITION), procedures.DEFINITION.name, procedures.DEFINITION.parent)
for row in original:
    fact = plan.fact(row['fact'])
    assert render(plan.fields, Frame(plan, row['bound_input']).values, fact.reads) == row['text']
report = {'scope': 'Fixed T7 model_select, fit-only lexical controls, unchanged features/C=10/seed=0; no action or independent-check claim', 'arms': {}}
for arm in ('original', 'random', 'targeted'):
    rows = {row['id']: dict(row, role=roles[row['id']]) for row in original}
    if arm != 'original':
        data = evidence.read(cohorts['arms'][arm]['pack'])
        for record in data['labels']:
            frame = Frame(plan, record['input'])
            for name, value in record['facts'].items():
                if name not in plan.fact_index:
                    continue
                fact = plan.fact(name)
                if fact.cards or fact.hidden:
                    continue
                projection = frame.read(fact)
                key = observation_id(fact, projection)
                row = {'id': key, 'fact': name, 'text': render(plan.fields, frame.values, fact.reads),
                       'classes': fact.keys, 'label': fact.index(value), 'role': 'fit'}
                if key in rows:
                    assert rows[key]['role'] == 'fit' and rows[key]['label'] == row['label']
                rows[key] = row
    grouped = defaultdict(list)
    for row in rows.values():
        grouped[row['fact']].append(row)
    targets, heads, predictions = [], {}, []
    for name, found in sorted(grouped.items()):
        training = [row for row in found if row['role'] == 'fit']
        held = [row for row in found if row['role'] == 'model_select']
        x, active = compact(matrix([row['text'] for row in training]))
        head = fit(x, active, [row['label'] for row in training], training[0]['classes'])
        heads[name] = head.to_json()
        counts = defaultdict(lambda: {'correct': 0, 'references': 0})
        for row in held:
            cell = counts[row['classes'][row['label']]]
            cell['references'] += 1
            predicted = int(head.probs(row['text']).argmax())
            cell['correct'] += predicted == row['label']
            predictions.append({'id': row['id'], 'target': name, 'gold': row['classes'][row['label']],
                                'prediction': row['classes'][predicted]})
        targets.append({'target': name, 'fit_observations': len(training), 'per_class': dict(counts)})
    total = sum(cell['references'] for target in targets for cell in target['per_class'].values())
    correct = sum(cell['correct'] for target in targets for cell in target['per_class'].values())
    report['arms'][arm] = {'correct': correct, 'references': total, 'accuracy': correct / total, 'targets': targets}
    files.publish(root / arm / 'lexical-heads.json', heads)
    files.publish(root / arm / 'lexical-predictions.jsonl', files.jsonl(predictions))
files.publish(root / 'cpu-controls.json', report)
print(json.dumps({name: {key: value for key, value in result.items() if key != 'targets'} for name, result in report['arms'].items()}))
