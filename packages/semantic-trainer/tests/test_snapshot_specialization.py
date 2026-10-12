from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from semantic_switch import ConfigError, Runtime, api, backend, files, tools
from semantic_switch.artifacts import load_receipt, read_export
from semantic_switch.compiler import compile_spec
from semantic_switch.tools.dataset import EvidenceContext
from tests.conftest import FakeScorer
from tests.test_adaptation_contracts import pipeline_base
from bench import evidence, procedures, snapshots
from test_lora import fake_fitting
from test_specialize import REALIZED, Native

NATIVE = {'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048}
QUOTAS = {'fit': 32, 'model_select': 2, 'check': 16}


def plan_of(staged):
    definition = Path(staged['definition'])
    return compile_spec(files.read_yaml(definition), definition.name, definition.parent)


def pack(tmp_path):
    return procedures.pack_command(Namespace(seed=0, per_template=2, variants=3, reviews=None, output=tmp_path / 'pack',
                                             fit=QUOTAS['fit'], model_select=QUOTAS['model_select'], check=QUOTAS['check']))


def exported(tmp_path, monkeypatch, staged):
    identity = api.Runtime.identity
    manifest = pipeline_base(tmp_path, monkeypatch)
    Native.tokens = FakeScorer()
    with Runtime({'judge': Native.tokens}) as runtime:
        result = tools.export(runtime, staged['definition'], native=True, task_family='bench:zt411', labels=staged['labels'])
    monkeypatch.setattr(api.Runtime, 'identity', identity)
    return manifest, result


def deployment_fakes(monkeypatch):
    from semantic_trainer import select
    def deploy(fitted, located, peft, destination, quant, handle):
        Path(destination).write_bytes(b'merged ' + Path(peft).name.encode())
        return files.sha256_file(destination)
    monkeypatch.setattr(select, 'deploy', deploy)
    monkeypatch.setattr(select, 'verify_zero', lambda *args: None)
    monkeypatch.setattr(select, 'conversion_bytes', lambda located: {'conversion': 1})
    monkeypatch.setattr(backend, 'LlamaScorer', Native)
    Native.built, Native.realized = [], REALIZED


def test_public_pack_round_trip_through_trainer_and_snapshot_evaluation(tmp_path, monkeypatch):
    from semantic_trainer import jobs, lora, select
    published = pack(tmp_path)
    counts = published['assignment']['counts']
    assert counts['fit']['situations'] == QUOTAS['fit'] and counts['model_select']['situations'] == QUOTAS['model_select'] and counts['check']['situations'] >= QUOTAS['check']
    assert all(counts[role]['frames'] > counts[role]['situations'] for role in ('fit', 'model_select'))
    staged = evidence.stage(tmp_path / 'pack', tmp_path / 'work')
    with pytest.raises(ConfigError, match='already exists'):
        evidence.stage(tmp_path / 'pack', tmp_path / 'work')
    staged_rows = files.read_jsonl(staged['labels'])
    labels, check = [row for row in staged_rows if row['role'] != 'check'], [row for row in staged_rows if row['role'] == 'check']
    assert {row['role'] for row in labels} == {'fit', 'model_select'} and check and staged['review'] == published['review']['status']
    context = EvidenceContext.from_manifest(staged['evidence'])
    assert context.review == published['review'] and context.audit == Path(staged['audit']) and not context.audit.read_text()
    assert {context.role(context.member(plan_of(staged), row['input'])) for row in check} == {'check'}
    assert all(source['origin'] == 'rule' and source['address'][0] for row in labels for history in row['label_sources'].values() for source in history)
    manifest, result = exported(tmp_path, monkeypatch, staged)
    export_manifest, rows = read_export(result['path'], native=True)
    assert len({row['group'] for row in rows}) == counts['fit']['families'] + counts['model_select']['families']
    assert result['situations'] == result['complete_situations'] == QUOTAS['fit'] + QUOTAS['model_select'] and len(rows) > result['situations']
    manifest, loaded, _ = fake_fitting(monkeypatch, manifest, rows)
    recipe = {'selection_unit': 'complete_situation', 'epochs': 1, 'min_epochs': 1, 'batch_size': 8, 'eval_every_examples': 8}
    with pytest.raises(ConfigError, match='exactly 7 situations'):
        jobs.prepare(staged['definition'], [result['path']], tmp_path / 'infeasible', assets_manifest=manifest, fit_situations=7, recipe=recipe)
    prepared = jobs.prepare(staged['definition'], [result['path']], tmp_path / 'job', assets_manifest=manifest, fit_situations=QUOTAS['fit'],
                            native=NATIVE, recipe=recipe)
    resolved, observations = jobs.resolve(prepared['job'])
    populations = resolved['populations']
    assert populations['fit']['situations'] == QUOTAS['fit'] and populations['model_select']['complete_situations'] == QUOTAS['model_select']
    assert populations['fit']['observations'] > QUOTAS['fit'] and populations['acceptance_fit']['observations'] == 0
    by_input = {evidence.identity(situation['input']): situation['role'] for situation in evidence.read(tmp_path / 'pack')['situations']}
    assert all(by_input[evidence.identity(row['bound_input'])] == row['role'] for row in observations)
    deployment_fakes(monkeypatch)
    fitted = lora.train(prepared['job'], tmp_path / 'run')
    assert loaded and fitted['candidates']
    candidates = files.read_json(Path(fitted['candidates']))['candidates']
    assert all(set(prediction) == {'answer', 'stages'} for candidate in candidates for prediction in candidate['predictions'].values())
    chosen = select.select(prepared['job'], tmp_path / 'run')
    assert chosen['unit'] == 'complete_situation'
    receipt = load_receipt(chosen['receipt'])
    assert receipt.data['acceptance']['calibration'] == 'absent' and receipt.data['selection']['unit'] == 'complete_situation'
    assert all(trial['hf_agreement']['gate'] is False and 'stages' in trial['hf_agreement']['specialist'] for trial in receipt.data['selection']['candidates'] if trial['name'] != 'base')
    lock = files.read_json(tmp_path / 'run' / 'deployment' / 'task.lock.json')
    assert {side['status'] for fact in lock['facts'].values() if fact['policies'] for side in fact['policies']['judge'].values()} == {'uncalibrated_default'}
    report = tools.check_deployment(chosen['receipt'], staged['labels'], staged['evidence'], tmp_path / 'check')
    assert report['situations'] == len(check) and report['exposure']['state'] == 'fresh' and report['review']['status'] == published['review']['status']
    assert report['evidence']['dataset_id'] == staged['dataset_id'] and {entry['state'] for entry in report['lineage'].values()} == {'declared_disjoint'}
    artifact = snapshots.evaluate('zt411', tmp_path / 'pack', staged['evidence'], {'base': resolved['model']['gguf'], 'specialist': chosen['receipt']},
                                  tmp_path / 'out' / 'snapshots.json', baseline='base', specialist='specialist', margin=.05, conditions=(procedures.TRAINED,), native=NATIVE)
    assert artifact['exposure']['state'] == 'consulted'
    assert receipt.data['base']['sha256'] == files.sha256_file(resolved['model']['gguf'])
    assert set(artifact['results']) == {f'{arm}|{kind}|structured' for arm in ('base', 'specialist') for kind in ('workflow', 'direct')}
    assert artifact['fingerprint']['arms']['specialist']['spec'] == chosen['receipt']
    assert artifact['fingerprint']['task'] == {procedures.TRAINED: files.sha256_file(receipt.verified_task())}


def test_local_arms_refuse_cpu_fallback_and_mixed_executions():
    snapshots.local_executions({'base': {'device': 'gpu', 'build': 'a'}, 'specialist': {'device': 'gpu', 'build': 'b'}, 'jev': None}, {'device': 'gpu'})
    with pytest.raises(ConfigError, match='no CPU fallback'):
        snapshots.local_executions({'base': {'device': 'cpu'}}, {'device': 'gpu'})
    with pytest.raises(ConfigError, match='different native executions'):
        snapshots.local_executions({'base': {'device': 'gpu', 'n_ctx': 2048}, 'specialist': {'device': 'gpu', 'n_ctx': 4096}}, {'device': 'gpu'})


def test_bounded_printer_transfer_pack_prepares_held_out_documents(tmp_path, monkeypatch):
    from semantic_switch.compiler import compile_spec
    from semantic_trainer import jobs
    published = procedures.transfer_command(Namespace(seed=0, per_template=2, variants=None, fit=8, model_select=12, check=6, output=tmp_path / 'pack'))
    assert published['documents']['zt411_a']['source'] == 'pinned manual rules' and published['documents']['zt411_d']['roles'] == ['check']
    staged = evidence.stage(tmp_path / 'pack', tmp_path / 'work')
    spec = files.read_yaml(staged['definition'])
    plan = compile_spec(spec, Path(staged['definition']).name)
    revisions = {key: plan.fact(f'unverified_step_{key}').meaning_rev for key in 'abcd'}
    assert len(set(revisions.values())) == 4 and len({fact['check'] for fact in spec['facts'].values()}) == 1
    assert all(('ZT-COMMAND: ' + text) in spec['guidance'][f'zt411_{key}']['text'] for key, (text, _) in procedures.CONVENTIONS.items() if text)
    staged_rows = files.read_jsonl(staged['labels'])
    labels, check = [row for row in staged_rows if row['role'] != 'check'], [row for row in staged_rows if row['role'] == 'check']
    assert {(row['role'], tuple(sorted(row['facts']))) for row in labels} == {('fit', ('unverified_step_a', 'unverified_step_b')), ('model_select', ('unverified_step_c',))}
    assert {tuple(row['facts']) for row in check} == {('unverified_step_d',)}
    built, _ = procedures.build(procedures.scenarios(0, 2))
    views = [procedures.View(frame) for frame in built if frame.condition == procedures.TRAINED]
    for left, right in (('a', 'b'), ('a', 'c'), ('a', 'd'), ('c', 'd'), ('b', 'd')):
        assert any(procedures.unverified(v, procedures.CONVENTIONS[left][1]) != procedures.unverified(v, procedures.CONVENTIONS[right][1]) for v in views)
    manifest = pipeline_base(tmp_path, monkeypatch)
    exports = {}
    with Runtime({'judge': FakeScorer()}) as runtime:
        for key in 'abc':
            exports[key] = tools.export(runtime, staged['definition'], to=tmp_path / 'work' / f'{key}.export.jsonl', native=True, labels=staged['labels'],
                                        fact=f'unverified_step_{key}', task_family=f'zt411:{key}')['path']
    rows = {key: read_export(path, native=True)[1] for key, path in exports.items()}
    assert all('Convention' not in row['text'] and 'ZT-COMMAND' not in row['text'] for group in rows.values() for row in group)
    assert not {row['group'] for row in rows['c']} & {row['group'] for key in 'ab' for row in rows[key]}
    prepared = jobs.prepare(staged['definition'], list(exports.values()), tmp_path / 'job', assets_manifest=manifest, objective='instruction_transfer', recipe={'eval_every_examples': 8})
    resolved, observations = jobs.resolve(prepared['job'])
    assert {(source['task_family'], role) for source in resolved['sources'] for role in source['roles']} == {('zt411:a', 'fit'), ('zt411:b', 'fit'), ('zt411:c', 'model_select')}
    assert prepared['support']['absent_fit_support'] == ['unverified_step_c'] and resolved['policy_families'] == ['zt411:a', 'zt411:b']
    assert resolved['populations']['acceptance_fit']['observations'] == 0
    with pytest.raises(ConfigError, match='without fit support: unverified_step_c'):
        jobs.prepare(staged['definition'], list(exports.values()), tmp_path / 'specialization', assets_manifest=manifest, recipe={'eval_every_examples': 8})
    spec['guidance']['zt411_c']['text'] += ' Revised.'
    files.publish(staged['definition'], spec)
    assert jobs.resolve(prepared['job'])[0]['id'] == resolved['id']
    with pytest.raises(ConfigError, match='unverified_step_c changed or left the task definition'):
        jobs.prepare(staged['definition'], list(exports.values()), tmp_path / 'revised', assets_manifest=manifest, objective='instruction_transfer',
                     recipe={'eval_every_examples': 8})
