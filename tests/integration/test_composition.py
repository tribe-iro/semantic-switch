import json
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip('torch')
import semantic_switch as ss
from semantic_switch import ConfigError, Runtime, api, backend, files
from semantic_switch.evidence import reviewed_content
from semantic_switch.tools.data import Labels
from semantic_switch.training import (Actor, LabelSource, Task, TrainingUse, Verification, compare_deployments, current, declaration, entry, point,
                                      recompute_comparison, rollback, specialization_request, specialize)
from semantic_testkit.assets import fake_assets
from semantic_testkit.candidates import Native, fake_base
from semantic_testkit.labels import TEACHER, declared
from semantic_testkit.pipeline import install, worker
from semantic_testkit.scoring import FakeScorer

VOCAB, CORRECTED = 4096, 11
OPTIONS = {'escalate': 'Request a technician for a current unresolved incident', 'continue': 'Continue operation when the incident is resolved',
           'clarify': 'Ask whether the incident remains unresolved'}
SPEC = {'name': 'ops', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
        'facts': {'action': {'pick': 'Which response does the policy require?', 'options': OPTIONS, 'allow_none': True},
                  'repaired': {'check': 'The message states that the repair succeeded.', 'unstated': True},
                  'issues': {'select': 'Which problems does the message report?', 'max_claims': 2, 'labels': {'heat': 'Overheating', 'jam': 'A paper jam'}}}}
ACTIONS, REPAIRS = ('escalate', 'continue', 'clarify'), (True, False, 'unstated')
HUMAN = Actor('human')
TEACHING = LabelSource('judged', TEACHER, training_use=TrainingUse.parse(declared()))


def role(i):
    return 'check' if i % 5 == 0 else 'model_select' if i % 5 == 3 else 'fit'


def gold(i):
    return {'action': 'none' if i % 9 == 7 else ACTIONS[i % 3], 'repaired': REPAIRS[i % 3], 'issues': {'heat': i % 2 == 0, **({'jam': i % 4 == 1} if i % 7 else {})}}


def raw_records(task):
    meaning = task.targets()['action']['meaning_rev']
    out = []
    for i in range(36):
        data, wanted = {'message': f'report {i}'}, gold(i)
        groups = ['thread:bridge'] if i in (2, 4) else [f'situation-{i}']
        sources = {'action': [TEACHING], 'repaired': [TEACHING], 'issues': {key: [TEACHING] for key in wanted['issues']}}
        if i in (15, 20):
            entry_ = Verification('label', 'ana', HUMAN, 'review', meaning if i == 15 else 'meaning-0', reviewed_content('label', data, ('action',), wanted['action']),
                                  'confirm')
            sources['action'] = [TEACHING.verified(entry_)]
        if i == CORRECTED:
            first = LabelSource('authored', 'ana', actor=HUMAN, replaces='escalate')
            second = LabelSource('authored', 'bo', actor=HUMAN, replaces='continue')
            out += [task.label(data, {'action': 'escalate'}, sources={'action': [TEACHING]}, groups=groups, role=role(i)),
                    task.label(data, {'action': 'continue', 'issues': wanted['issues']}, sources={'action': [TEACHING, first], 'issues': sources['issues']},
                               groups=groups, role=role(i))]
            sources['action'] = [TEACHING, first, second]
            wanted = {key: value for key, value in wanted.items() if key != 'issues'}
            sources.pop('issues')
        out.append(task.label(data, wanted, sources=sources, groups=groups, role=role(i)))
    return out


def authored(root, monkeypatch):
    root.mkdir(parents=True)
    definition = root / 'ops.yaml'
    files.publish(definition, SPEC)
    with Runtime(FakeScorer()) as runtime:
        task = Task(runtime.load(definition, draft=True))
        runtime.load(definition).correct({'message': 'report 999'}, 'action', 'continue', actor=HUMAN)
    records = raw_records(task)
    held = task.accumulator(max_situations=64, max_claims=1024, max_bytes=1 << 22)
    for record in reversed(records):
        held.add(task.unit(record))
    labels = root / 'labels.jsonl'
    files.publish(labels, ''.join(json.dumps(held.settle(key, strict=True)[0], sort_keys=True) + '\n' for key in sorted(held.keys())))
    return definition, labels


def base(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    manifest, gguf = fake_assets(tmp_path / 'assets')
    return fake_base(manifest, VOCAB), gguf


def described(definition, labels, work, manifest):
    return specialization_request(definition, labels, work, devices={'fit': 'cpu', 'native': 'cpu'}, assets=manifest, quant='Q4_0',
                                  recipe=['epochs=2', 'min_epochs=1', 'seed=0'], budget=['max_tokens=1000000', 'max_sequence_tokens=1000'], cadence=8,
                                  task_family='example:ops')


def evidence(root, frozen, labels):
    root.mkdir(parents=True)
    shutil.copy(frozen / 'task.yaml', root / 'ops.yaml')
    shutil.copy(labels, root / 'ops.test.jsonl')
    with Runtime(FakeScorer()) as runtime:
        Labels(runtime.load(root / 'ops.yaml', draft=True))
    paths = files.Paths.of(root / 'ops.yaml')
    return paths.test, paths.dataset, paths.audit


def baseline(root, frozen):
    shutil.copytree(frozen, root)
    with Runtime(FakeScorer(name='baseline')) as runtime:
        ss.tools.lock(runtime, root / 'task.yaml')
    return root / 'task.lock.json'


def test_controller_composes_accumulation_publication_comparison_and_promotion_with_scripted_deployment(tmp_path, monkeypatch):
    from semantic_trainer import jobs
    from semantic_trainer.pipeline import Controller, read_export
    manifest, gguf = base(tmp_path, monkeypatch)
    operations = install(monkeypatch, gguf=gguf, vocab=VOCAB)
    definition, labels = authored(tmp_path / 'sources', monkeypatch)
    rows = {json.loads(line)['input']['message']: json.loads(line) for line in labels.read_text().splitlines()}
    corrected = rows[f'report {CORRECTED}']
    assert corrected['facts'] == {'action': 'clarify', 'repaired': REPAIRS[CORRECTED % 3], 'issues': gold(CORRECTED)['issues']}
    assert [(s['producer'], s.get('replaces')) for s in corrected['label_sources']['action']] == [(TEACHER, None), ('ana', 'escalate'), ('bo', 'continue')]
    assert rows['report 14']['facts']['issues'] == {'heat': True} and rows['report 7']['facts']['action'] == 'none' and rows['report 2']['facts']['repaired'] == 'unstated'

    request = tmp_path / 'direct' / 'request.json'
    files.write_new(request, described(definition, labels, tmp_path / 'direct', manifest))
    pointer = tmp_path / 'current.json'
    direct = Controller(request, operations=operations, emit=lambda **event: None).run()
    assert direct['state'] == 'completed' and direct['representation'] == 'merged'
    assignment = files.read_json(Path(direct['stages']['freeze']['path']) / 'frozen-inputs.json')['assignment']
    held = set(assignment['check_components'])
    _, exported = read_export(direct['stages']['export']['path'], native=True)
    assert held and not {row['component'] for row in exported} & held
    assert max(token for row in exported for entry in row['presentations'] for token in entry.prefix + entry.input + entry.question) < VOCAB
    emitted = {(row['bound_input']['message'], row['fact'], row.get('member')): row['label'] for row in exported}
    assert emitted[(f'report {CORRECTED}', 'action', None)] == ACTIONS.index('clarify')
    bridged = {row['component'] for row in exported if row['bound_input']['message'] in ('report 2', 'report 4')}
    assert len(bridged) == 1
    with Runtime() as runtime:
        keys = list(Task(runtime.load(Path(direct['stages']['freeze']['path']) / 'task' / 'task.yaml', draft=True)).targets()['action']['answers'])
    assert emitted[('report 7', 'action', None)] == keys.index('none') and emitted[('report 2', 'repaired', None)] == 2
    assert ('report 14', 'issues', 'heat') in emitted and ('report 14', 'issues', 'jam') not in emitted and ('report 13', 'issues', 'jam') in emitted
    resolved, observations = jobs.resolve(direct['job'])
    assert not {row['component'] for row in observations} & held
    assert [item['projection'] for item in resolved['corrections']] == ['{"message":"report 999"}']

    frozen = Path(direct['stages']['freeze']['path'])
    trainer = worker(tmp_path, monkeypatch, gguf=gguf, vocab=VOCAB)
    adopted = baseline(tmp_path / 'baseline', frozen / 'task')
    first = point(pointer, adopted, reason='initial', expected=None)
    external = specialize(trainer, tmp_path / 'external', described(definition, labels, tmp_path / 'external', manifest))
    assert external['state'] == 'completed' and external['exit'] == 0 and files.read_json(pointer) == first
    assert [external['stages'][name]['id'] for name in ('freeze', 'export')] == [direct['stages'][name]['id'] for name in ('freeze', 'export')]
    elsewhere, _ = jobs.resolve(external['job'])
    assert (elsewhere['fit']['id'], elsewhere['monitor']['id']) == (resolved['fit']['id'], resolved['monitor']['id'])

    shutil.rmtree(tmp_path / 'sources')
    resumed = Controller(request, operations=operations, emit=lambda **event: None).run()
    assert {stage['state'] for stage in resumed['stages'].values()} == {'reused'} and resumed['lock'] == direct['lock']

    deployment = Path(direct['receipt']).parent
    moved = tmp_path / 'moved' / deployment.name
    shutil.copytree(deployment, moved)
    with Runtime() as runtime:
        loaded = runtime.load(moved)
        exact = loaded({'message': 'report 999'}, facts=['action']).facts['action']
        assert (exact.source, exact.prediction.key) == ('example', 'continue')

    labelled, dataset, audit = evidence(tmp_path / 'evidence', frozen / 'task', frozen / 'labels.jsonl')
    assert not audit.exists() or not audit.read_text()
    built = len(Native.built)
    compared = compare_deployments(adopted, direct['receipt'], labelled, dataset, tmp_path / 'compared', count=200,
                                   declaration=declaration(margin=.05), bindings={'baseline': {'judge': FakeScorer(name='baseline')}})
    assert len(files.read_jsonl(audit)) == 1 and compared['exposure']['state'] == 'fresh' and len(Native.built) > built
    assert compared['lineage']['candidate']['state'] == 'declared_disjoint' and compared['situations'] == 8
    with Runtime() as runtime:
        task = Task(runtime.load(frozen / 'task' / 'task.yaml', draft=True))
    targets = {row['input']: row['targets'] for row in compared['rows']['candidate']}
    at = lambda i: targets[task.input_key({'message': f'report {i}'})]
    assert (at(15)['action']['reference']['state'], at(20)['action']['reference']['state']) == ('verified', 'judged')
    assert at(20)['action']['reference']['verifications']['stale'] == 1
    assert at(15)['action']['reference']['binding'] == {'revision': 2, 'label': [task.targets()['action']['meaning_rev']], 'evidence': []}
    assert (at(25)['action']['gold'], at(20)['repaired']['gold'], at(35)['issues/heat']['gold']) == ('none', 'unstated', 'false')
    assert compared['reference']['state'] == 'partial' and compared['decision']['verdict'] == 'partial_reference'

    def forbidden(*args, **kwargs):
        raise AssertionError('report projection performs no inference')
    with monkeypatch.context() as patched:
        patched.setattr(api, 'Runtime', forbidden)
        patched.setattr(backend, 'LlamaScorer', forbidden)
        again = recompute_comparison(tmp_path / 'compared' / 'comparison.json', tmp_path / 'recomputed.json')
    assert {key: again[key] for key in ('arms', 'paired', 'reference', 'decision')} == {key: compared[key] for key in ('arms', 'paired', 'reference', 'decision')}

    candidate = entry(direct['receipt'])
    promoted = point(pointer, direct['receipt'], reason='reviewed comparison', expected=entry(adopted).id)
    assert promoted['id'] == candidate.id and current(pointer) == candidate.path
    with pytest.raises(ConfigError, match='not the expected'):
        point(pointer, external['receipt'], reason='stale decision', expected=entry(adopted).id)
    assert files.read_json(pointer) == promoted
    assert rollback(pointer, reason='regression', expected=candidate.id)['id'] == entry(adopted).id
