import json
from collections import Counter
from pathlib import Path

from bench import evidence, procedures
from semantic_switch import Runtime, files
from semantic_testkit.scoring import FakeScorer


root = Path(__file__).resolve().parent
published = evidence.read('packs/zt411')
_, frames, metadata = evidence.frames_of(published, 'packs/zt411')
frames = [frame for frame in frames if frame.condition == 'structured']
checks = {name: spec['check'] for name, spec in files.read_yaml(procedures.DEFINITION)['facts'].items() if 'check' in spec}
current = {}
distributions = {True: [.98, .01, .01], False: [.01, .98, .01], 'unstated': [.01, .01, .98]}


def score(state, question, width):
    name = next(name for name, text in checks.items() if text in question)
    value = current[name]
    return [1 / width] * width if value is None else distributions[value][:width]


mismatches, counts = [], Counter()
with Runtime(FakeScorer(score)) as runtime:
    switch = runtime.load(procedures.DEFINITION, draft=True)
    for frame in frames:
        gold = procedures.reference(frame)
        current = {name: gold.primitives[name].value for name in checks}
        result = switch(frame.input, facts=[*checks, 'actions'])
        states = dict(result.facts['actions'].states)
        values = {name: True if state == 'applicable' else False if state in ('ineligible', 'rejected') else None for name, state in states.items()}
        differences = [{'action': name, 'expected': gold.decisions['actions/' + name].value, 'actual': value, 'state': states[name]}
                       for name, value in values.items() if value != gold.decisions['actions/' + name].value]
        if differences:
            mismatches.append({'frame': frame.id, 'role': metadata[frame.id]['role'], 'differences': differences})
            counts.update(difference['action'] for difference in differences)

report = {'scope': 'Existing task/Executor with reference primitive answers; no language model inference. Membership states, not semantic action ranking.',
          'corpus_sha256': published['manifest_sha256'], 'definition_sha256': files.sha256_file(procedures.DEFINITION),
          'frames': len(frames), 'frames_with_differences': len(mismatches), 'by_action': dict(counts), 'mismatches': mismatches}
(root / 'contract.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'mismatches'}))
