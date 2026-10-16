import json
from pathlib import Path

from bench import evidence, procedures
from semantic_switch import Runtime, files
from semantic_switch.compiler import compile_spec
from semantic_switch.context import render
from semantic_switch.examples import Frame
from semantic_switch.heads import LexicalHead
from semantic_switch.metrics import action_sets, exact_sets
from semantic_testkit.scoring import FakeScorer


root = Path(__file__).resolve().parent
data = evidence.read('packs/zt411')
_, frames, metadata = evidence.frames_of(data, 'packs/zt411')
frames = [frame for frame in frames if frame.condition == 'structured' and metadata[frame.id]['role'] in ('model_select', 'check')]
spec = files.read_yaml(procedures.DEFINITION)
plan = compile_spec(spec, procedures.DEFINITION.name, procedures.DEFINITION.parent)
checks = {name: fact['check'] for name, fact in spec['facts'].items() if 'check' in fact}
report = {'scope': 'Raw lexical argmax projected through normal Executor/cards under an explicit always-answer policy. No calibrated confidence, LM vocabulary mass, latency or accepted-quality comparison with LoRA.',
          'corpus_sha256': data['manifest_sha256'], 'arms': {}}
for arm in ('original', 'random', 'targeted'):
    heads = {name: LexicalHead.from_json(record) for name, record in files.read_json(root / arm / 'lexical-heads.json').items()}
    predicted = {}
    def score(state, question, width):
        name = next(name for name, text in checks.items() if text in question)
        return [float(index == predicted[name]) for index in range(width)]
    rows = {'model_select': [], 'check': []}
    with Runtime(FakeScorer(score)) as runtime:
        switch = runtime.load(procedures.DEFINITION, draft=True)
        for frame in frames:
            given = Frame(plan, frame.input)
            predicted = {name: int(head.probs(render(plan.fields, given.values, plan.fact(name).reads)).argmax()) for name, head in heads.items()}
            result = switch(frame.input, facts=[*checks, 'actions'])
            states = dict(result.facts['actions'].states)
            gold = procedures.reference(frame)
            unresolved = [name for name, state in states.items() if state not in ('applicable', 'rejected', 'ineligible', 'superseded')]
            rows[metadata[frame.id]['role']].append({'frame': frame.id, 'scope': list(procedures.MENU),
                'selected': [name for name, state in states.items() if state == 'applicable'], 'unresolved': unresolved,
                'gold': [name for name in procedures.MENU if gold.decisions['actions/' + name].value is True],
                'reference_unresolved': [name for name in procedures.MENU if gold.decisions['actions/' + name].value is None],
                'prohibited': list(gold.prohibited), 'complete_gold': gold.complete, 'prediction_complete': not unresolved, 'outcome': 'predicted'})
    report['arms'][arm] = {role: {'actions': action_sets(found), 'exact': exact_sets(found)} for role, found in rows.items()}
    files.publish(root / arm / 'lexical-action-rows.jsonl', files.jsonl([dict(row, role=role) for role, found in rows.items() for row in found]))
files.publish(root / 'lexical-actions.json', report)
print(json.dumps(report, indent=2))
