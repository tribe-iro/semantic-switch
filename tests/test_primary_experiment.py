from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from semantic_switch import Runtime, files
from semantic_switch.evidence import path_record
from semantic_switch.tools.common import input_key
from semantic_switch.tools.data import Labels
from semantic_switch.training import EvidenceContext, Task, assess_lineage
from semantic_testkit.scoring import FakeScorer

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = {'escalate': 'Request a technician', 'continue': 'Continue operation'}
REVIEWER = [{'derivation': 'authored', 'producer': 'reviewer', 'actor': {'kind': 'human'}}]


@pytest.mark.parametrize('kind', ['zt411', 'support'])
@pytest.mark.parametrize('published', [False, True])
def test_training_runner_consumes_existing_pack_without_building_it(tmp_path, kind, published):
    script = tmp_path / 'bench' / 'zt411_gpu.sh'
    script.parent.mkdir()
    script.write_bytes((ROOT / 'bench' / 'zt411_gpu.sh').read_bytes())
    python = tmp_path / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$COMMAND_LOG"\nexit 73\n')
    python.chmod(0o755)
    pack, work, log = tmp_path / 'corpus', tmp_path / 'work', tmp_path / 'commands'
    if published:
        pack.mkdir()
        (pack / 'benchmark.json').write_text('{}\n')
        (pack / 'reviews.jsonl').write_text('{"reviewer":"preserved"}\n')
    before = {p.name: p.read_bytes() for p in pack.glob('*')}
    env = {**os.environ, 'PACK_KIND': kind, 'PACK': str(pack), 'WORK': str(work), 'COMMAND_LOG': str(log),
           'REVIEWS': str(tmp_path / 'must-not-be-loaded.jsonl')}
    for _ in range(2):
        result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
        assert result.returncode == (73 if published else 2)
        assert {p.name: p.read_bytes() for p in pack.glob('*')} == before
    if published:
        assert log.read_text().splitlines() == [f'-m bench.evidence stage {pack} --under {work}/stages --binding {work}/stage.json'] * 2
    else:
        assert 'training requires an existing published pack' in result.stderr
        assert not log.exists() and not work.exists()


def definition(directory):
    path = directory / 'action.yaml'
    files.publish(path, {'name': 'action', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'], 'exact_override': True,
                         'facts': {'action': {'pick': 'Which response applies?', 'options': OPTIONS}}})
    return path


def deployment(directory, rows=()):
    task = definition(directory / 'task')
    with Runtime(FakeScorer()) as runtime:
        switch = runtime.load(task, draft=True)
        switch.correct({'message': 'report 500'}, 'action', 'escalate', sources=REVIEWER)
    lock = {'facts': {'action': {'examples': files.read_jsonl(files.Paths.of(task).examples)}}, 'family_seeds': {}}
    return SimpleNamespace(id='receipt', plan=switch.plan, data=lock, rows=list(rows), receipt=object())


def evidence(directory, messages):
    path = definition(directory)
    paths = files.Paths.of(path)
    with Runtime(FakeScorer()) as runtime:
        task = Task(runtime.load(path, draft=True))
        files.publish(paths.test, ''.join(json.dumps(task.label({'message': message}, {'action': 'continue'}, sources={'action': REVIEWER}, role='check',
                                                                groups=[f'family-{n}'])) + '\n' for n, message in enumerate(messages)))
        Labels(runtime.load(path, draft=True))
        plan = runtime.load(path, draft=True).plan
    context = EvidenceContext.from_manifest(paths.dataset)
    return context, plan, [({'message': message}, context.member(plan, {'message': message})) for message in messages]


def override(message, uses=('override',), seeds=()):
    key = json.dumps({'message': message}, separators=(',', ':'))
    return {'id': f'observation:{message}', **path_record(('action',)), 'projection': key, 'uses': list(uses), 'family_seeds': [list(seed) for seed in seeds]}


def test_lineage_assesses_overrides_projections_and_family_closure_against_the_same_evidence(tmp_path):
    context, plan, items = evidence(tmp_path / 'evidence', ['report 2000', 'report 2001'])
    undeclared = assess_lineage(deployment(tmp_path / 'undeclared'), context, items)
    assert undeclared['state'] == 'undeclared_overrides' and undeclared['situations'] == 2 and undeclared['dataset_id'] == context.dataset_id
    assert undeclared['targets'] == {'judge:action': {'state': 'undeclared_overrides', 'overlap': [], 'undeclared_overrides': 1}}

    declared = assess_lineage(deployment(tmp_path / 'declared', [override('report 500')]), context, items)
    assert declared['state'] == 'declared_disjoint' and declared['targets']['judge:action'] == {'state': 'declared_disjoint', 'overlap': [], 'undeclared_overrides': 0}

    corrected_context, corrected_plan, corrected_items = evidence(tmp_path / 'corrected', ['report 500', 'report 2002'])
    overlap = assess_lineage(deployment(tmp_path / 'overlap', [override('report 500')]), corrected_context, corrected_items)
    assert overlap['state'] == 'known_overlap' and overlap['targets']['judge:action']['overlap'] == [input_key(corrected_plan, {'message': 'report 500'})]

    trained = override('report 77', uses=('adaptation',), seeds=[('key', input_key(plan, {'message': 'report 2001'}))])
    family = assess_lineage(deployment(tmp_path / 'family', [override('report 500'), trained]), context, items)
    assert family['state'] == 'known_overlap' and family['targets']['judge:action']['overlap'] == [input_key(plan, {'message': 'report 2001'})]


def test_context_resolves_group_names_and_aliases_to_assigned_families(tmp_path):
    context, plan, items = evidence(tmp_path / 'evidence', ['report 2000', 'report 2001'])
    for (data, family), group in zip(items, ('family-0', 'family-1')):
        assert context.resolve(group) == context.resolve(family) == family
    assert context.resolve('unassigned') == 'unassigned'


def test_primary_script_specializes_through_the_controller_then_compares_before_its_derived_check():
    script = (ROOT / 'bench' / 'zt411_gpu.sh').read_text()
    stage, specialize, compare, check = (script.index(marker) for marker in ('stage "$PACK" --under "$WORK/stages"', 'specialize "$TASK"', 'if [[ -n "$REFERENCE" ]]',
                                                                             '--results "$RESULTS" --check "$WORK/checks"'))
    assert stage < specialize < compare < check and 'sha256sum' not in script and 'mv ' not in script and 'partial' not in script.lower()
    assert '--trainer "$XPU/semantic-trainer"' in script and 'selection_unit=complete_situation' in script
    assert 'step()' not in script and 'DONE' not in script and 'eval(' not in script and 'tail -n' not in script
    assert 'semantic-trainer" check' not in script and '/semantic-trainer check' not in script and 'semantic-trainer" prepare' not in script
    assert '--role model_select' in script[:compare] and '${UNTIL:+--until "$UNTIL"}' in script


def corpus(tmp_path):
    from argparse import Namespace
    from bench import procedures
    procedures.pack_command(Namespace(seed=0, per_template=2, variants=3, reviews=None, output=tmp_path / 'pack', fit=32, model_select=2, check=16))
    return tmp_path / 'pack'


def snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def test_binding_stages_once_resumes_and_leaves_a_read_only_corpus_untouched(tmp_path):
    from bench import evidence
    pack = corpus(tmp_path)
    before = snapshot(pack)
    for path in (pack, *pack.iterdir()):
        path.chmod(0o555 if path.is_dir() else 0o444)
    try:
        first = evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
        audit = Path(first['audit'])
        audit.write_text('{"consulted": "first"}\n')
        again = evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    finally:
        for path in (pack, *pack.iterdir()):
            path.chmod(0o755 if path.is_dir() else 0o644)
    assert first == again == files.read_json(tmp_path / 'run' / 'stage.json')
    assert first['manifest_sha256'] == evidence.manifest_sha256(pack) and Path(first['definition']).parent.name == first['manifest_sha256'][:16]
    assert audit.read_text() == '{"consulted": "first"}\n' and snapshot(pack) == before and len(evidence.read(pack)['manifest']['payloads']) == 8


def test_changed_publication_after_staging_is_refused_and_the_binding_is_kept(tmp_path):
    from bench import evidence
    pack = corpus(tmp_path)
    bound = evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    Path(bound['audit']).write_text('{"consulted": "kept"}\n')
    manifest = files.read_json(pack / 'benchmark.json')
    manifest['pack']['version'] = 'changed'
    (pack / 'benchmark.json').write_text(json.dumps(manifest))
    observed = evidence.manifest_sha256(pack)
    with pytest.raises(evidence.ConfigError, match=rf"corpus manifest {observed} differs from the bound corpus {bound['manifest_sha256']}"):
        evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    assert files.read_json(tmp_path / 'run' / 'stage.json') == bound and Path(bound['audit']).read_text() == '{"consulted": "kept"}\n'
    assert sorted(p.name for p in (tmp_path / 'run' / 'stages').iterdir()) == [bound['manifest_sha256'][:16]]


def test_corrupt_payload_names_its_path_and_generates_nothing(tmp_path, monkeypatch):
    from bench import evidence, procedures
    pack = corpus(tmp_path)
    monkeypatch.setattr(procedures, 'pack_command', lambda *a, **k: pytest.fail('training reached corpus generation'))
    labels = pack / files.read_json(pack / 'benchmark.json')['payloads']['labels.jsonl']['file']
    labels.write_bytes(labels.read_bytes()[:-2])
    with pytest.raises(evidence.ConfigError, match=rf"{labels}: payload digest differs"):
        evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    labels.unlink()
    with pytest.raises(evidence.ConfigError, match=rf"{labels}: payload named by .* is missing"):
        evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    assert not (tmp_path / 'run' / 'stage.json').exists()


def test_evaluation_refuses_a_publication_changed_after_fitting(tmp_path):
    from bench import evidence
    pack = corpus(tmp_path)
    bound = evidence.bind_corpus(pack, tmp_path / 'run' / 'stages', tmp_path / 'run' / 'stage.json')
    manifest = files.read_json(pack / 'benchmark.json')
    manifest['pack']['version'] = 'republished'
    (pack / 'benchmark.json').write_text(json.dumps(manifest))
    before = snapshot(pack)
    result = subprocess.run([str(ROOT / '.venv' / 'bin' / 'python'), '-m', 'bench.snapshots', 'zt411', '--pack', str(pack), '--evidence', bound['evidence'],
                             '--baseline', 'base=unused.gguf', '--margin', '0', '--output', str(tmp_path / 'out.json'), '--corpus', bound['manifest_sha256']],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and f"differs from the bound corpus {bound['manifest_sha256']}" in result.stderr
    assert evidence.manifest_sha256(pack) in result.stderr and snapshot(pack) == before and not (tmp_path / 'out.json').exists()


def test_majority_choices_use_only_fit_roles_and_declared_check_tie_order():
    from bench.snapshots import majority_baseline
    frames = [SimpleNamespace(id=f'{name}:base:structured:en', condition='structured', input={'label': label})
              for name, label in [('a', True), ('b', False), ('c', False)]]
    meta = {frame.id: {'role': 'fit' if index < 2 else 'model_select', 'family': frame.id} for index, frame in enumerate(frames)}
    module = SimpleNamespace(TRAINED='structured', reference=lambda frame: SimpleNamespace(supervision={'ready': SimpleNamespace(value=frame.input['label'])}))
    domain = SimpleNamespace(module=module)
    original = majority_baseline(domain, frames, meta, frames[2:])
    frames[2].input['label'] = True
    changed = majority_baseline(domain, frames, meta, frames[2:])
    assert original['choices'] == changed['choices'] == {'ready': True}
    assert original['correct'] == 0 and changed['correct'] == 1 and original['action_accuracy'] is None
