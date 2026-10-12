import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from semantic_switch import ConfigError, Runtime, files, tools
from semantic_switch.artifacts import artifact_digest, load_receipt, read_export
from tests.conftest import FakeScorer
from test_specialize import REALIZED, Native, selection

ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / 'packages/semantic-trainer/src/semantic_trainer'
INTERNAL = ('semantic_switch.compiler', 'semantic_switch.snapshot', 'semantic_switch.tools.data', 'semantic_switch.tools.lock', 'semantic_switch.tools.policy')


def test_trainer_flow_uses_supported_records_without_plan_snapshot_or_examples():
    for name in ('jobs', 'lora', 'select', 'collect', 'migrate', 'deploy', 'cli'):
        tree = ast.parse((TRAINER / f'{name}.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(INTERNAL), f'{name}:{node.lineno}'
            assert not (isinstance(node, ast.Attribute) and node.attr in ('plan', 'snapshot', 'at', 'fact_index')), f'{name}:{node.lineno}'
            assert not (isinstance(node, ast.Name) and node.id == 'Example'), f'{name}:{node.lineno}'


def test_task_description_keys_targets_by_full_address(tmp_path):
    from semantic_switch.training import INTEGRATION, describe_task
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'Money back.'}, 'issues': {'select': 'Which issues?', 'labels': {'refund': 'Money back', 'billing': 'Invoice'}}}})
    described = describe_task(path)
    assert described['integration'] == INTEGRATION and described['fields'] == {'message': {'type': 'text', 'required': True, 'values': None}}
    assert {'refund', 'issues/refund', 'issues/billing'} <= set(described['targets']) and described['targets']['refund']['answers'] == {'true': True, 'false': False}
    assert described['targets']['refund']['reads'] == ['message'] and len(described['targets']['refund']['meaning_rev']) == 8


def test_receipt_loading_needs_neither_the_trainer_nor_the_fitting_stack(tmp_path, monkeypatch):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    receipt = module.select(job, run)['receipt']
    code = '''
import importlib.abc, json, sys
from types import SimpleNamespace
BLOCKED = ('semantic_trainer', 'torch', 'transformers', 'peft', 'safetensors', 'sklearn', 'scipy')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(name)
sys.meta_path.insert(0, Block())
from semantic_switch import Runtime, api, backend
api.Model._host_bytes = lambda self: {'weights': 1}
api.available_host_bytes = lambda: 1 << 50
from semantic_switch.artifacts import load_receipt
from tests.conftest import FakeScorer
realized, receipt = json.loads(sys.argv[1]), sys.argv[2]
class Native(FakeScorer):
    def __init__(self, path, **kwargs):
        super().__init__()
        self.path, self.shared, self.weights = path, True, SimpleNamespace(hybrid=False)
    def capacity(self):
        return dict(realized)
    def close(self):
        pass
backend.LlamaScorer = Native
import semantic_switch.tools
loaded = load_receipt(receipt)
assert not [name for name in sys.modules if name.split('.')[0] in (*BLOCKED, 'llama_cpp')]
with Runtime.from_specialization(receipt) as runtime:
    switch = runtime.load()
    status = switch({'message': 'report 4'}).facts['action'].status
assert not [name for name in sys.modules if name.split('.')[0] in BLOCKED]
print(json.dumps({'lock': switch.snapshot.lock_id, 'receipt': loaded.id, 'status': status}))
'''
    done = subprocess.run([sys.executable, '-c', code, json.dumps(REALIZED), receipt], cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.splitlines()[-1])
    assert out['receipt'] == load_receipt(receipt).id and out['lock'] == load_receipt(receipt).data['lock']['id'] and out['status']


def test_freeze_rejects_changed_adapter_configuration_framing_and_failed_policy_fit_without_publishing(tmp_path, monkeypatch):
    from semantic_switch.tools import deployment, policy
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch, acceptance_fit=True)
    config = run / 'candidates' / 'step-8' / 'adapter_config.json'
    original = config.read_text()
    config.write_text('{"r": 4}')
    with pytest.raises(ConfigError, match='adapter configuration drift'):
        module.select(job, run)
    config.write_text(original)
    realized = deployment.realized
    monkeypatch.setattr(deployment, 'realized', lambda runtime: {**realized(runtime), 'framing': {'changed': True}})
    with pytest.raises(ConfigError, match='differs from the selected trial'):
        module.select(job, run)
    monkeypatch.setattr(deployment, 'realized', realized)
    def failing(*args):
        raise ConfigError('policy fit failed')
    monkeypatch.setattr(policy, 'fit_report', failing)
    with pytest.raises(ConfigError, match='policy fit failed'):
        module.select(job, run)
    assert not (run / 'deployment').exists() and not list(run.glob('.deployment-*')) and not list(run.glob('.select-*'))


def test_freeze_validates_staging_task_identity_and_addresses_before_scoring(tmp_path, monkeypatch):
    from semantic_switch.tools import freeze_specialization
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch, acceptance_fit=True)
    _, observations = module.resolve(job)
    accepting = [row for row in observations if row['role'] == 'acceptance_fit']
    history = module.membership(accepting, ('acceptance_fit',))
    staging = tmp_path / 'staging'
    staging.mkdir()
    def freeze(**change):
        given = {'model': FakeScorer(), 'task': resolved['task'], 'acceptance': accepting,
                 'provenance': history, 'population': {}, 'expected': {}, 'staging': staging, 'job': None, **change}
        return freeze_specialization(**given)
    (staging / 'left.txt').write_text('x')
    with pytest.raises(ConfigError, match='empty caller-owned staging'):
        freeze()
    (staging / 'left.txt').unlink()
    with pytest.raises(ConfigError, match='frozen task changed since preparation'):
        freeze(task={**resolved['task'], 'behavior': 'f' * 64})
    with pytest.raises(ConfigError, match='override declarations differ from the frozen correction population'):
        freeze(provenance=[*history, {**history[0], 'id': 'a' * 64, 'sources': [], 'uses': ['override']}])
    with pytest.raises(ConfigError, match='unresolved address ghost'):
        freeze(acceptance=[{**accepting[0], 'fact': 'ghost'}])
    with pytest.raises(ConfigError, match='meaning revision or label classes'):
        freeze(acceptance=[{**accepting[0], 'meaning_rev': 'changed'}])
    with pytest.raises(ConfigError, match='provenance declares uses'):
        freeze(provenance=[{**row, 'uses': ['selection']} for row in history])
    with pytest.raises(ConfigError, match='specialization provenance'):
        freeze(provenance=[{**history[0], 'sources': []}])
    assert not list(staging.iterdir())


def pristine(run, name):
    kept = run.parent / f'pristine-{name}'
    if not kept.exists():
        files.publish(kept, files.read_json(run / name))
    return files.read_json(kept)


def legacy(run, resolved, *, sources=None, **change):
    from semantic_trainer.migrate import PRECEDING
    fitted = pristine(run, 'resolved-job.json')
    old = {key: value for key, value in fitted.items() if key != 'corrections'}
    old |= {'schema': PRECEDING['resolved-job.json'], 'recipe': resolved['recipe'], 'fit': {'fields': {}, 'id': 'a' * 64}, 'monitor': {'fields': {}, 'id': 'b' * 64},
            'model': {key: value for key, value in fitted['model'].items() if key != 'files'},
            'task': {'definition': '/moved/action.yaml', 'sha256': fitted['task']['sha256'], 'revisions': fitted['task']['revisions']},
            'sources': sources or [{**source, 'path': f"/moved/{Path(source['path']).name}"} for source in fitted['sources']],
            'memberships': [{**member, 'component': member['sources'][0]['component']} for member in fitted['memberships']], **change}
    files.publish(run / 'resolved-job.json', old)
    listed = pristine(run, 'candidates.json')
    files.publish(run / 'candidates.json', {**listed, 'schema': PRECEDING['candidates.json'], 'job_id': 'e' * 64, 'fit_id': 'a' * 64, 'monitor_id': 'b' * 64,
                                            'fitted_job_id': artifact_digest(old), 'rule': {'order': ['correct', 'regressions', 'base', 'step']}})
    files.publish(run / 'run.json', {'schema': PRECEDING['run.json'], 'state': 'ready', 'loss': listed['loss'], 'job_id': 'e' * 64})
    for name in ('progress.jsonl', 'updates.jsonl'):
        (run / name).write_text('{}\n')
    files.publish(run / 'metrics.json', {'stop': listed['stop']})
    return old


def test_migration_admits_path_and_family_renaming_through_normal_candidate_validation(tmp_path, monkeypatch):
    from semantic_trainer.migrate import migrate
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    legacy(run, resolved)
    before = {path.name: files.sha256_file(path) for path in run.iterdir() if path.is_file()}
    with pytest.raises(ConfigError, match='unsupported candidate list'):
        module.select(job, run)
    result = migrate(job, run, tmp_path / 'compatibility.json')
    assert result['outcome'] == 'eligible' and result['changes']['fit'] == result['changes']['monitor'] == [] and not result['changes']['acceptance_fit']
    assert result['changes']['families']['original'] == result['changes']['families']['resolved']
    assert {path.name: files.sha256_file(path) for path in run.iterdir() if path.is_file()} == before
    record = files.read_json(tmp_path / 'compatibility.json')
    assert record['original']['job_id'] == 'e' * 64 and record['job_id'] == resolved['id'] and all(c['config_sha256'] for c in record['candidates'])
    with pytest.raises(ConfigError, match='already exists'):
        migrate(job, run, tmp_path / 'compatibility.json')
    chosen = module.select(job, run, tmp_path / 'compatibility.json')
    receipt = load_receipt(chosen['receipt'])
    assert chosen['selected'] == 'step-8' and receipt.data['fit_id'] == resolved['fit']['id'] and receipt.data['job_id'] == resolved['id']


def test_migration_refuses_refits_bridges_and_missing_evidence_and_limits_changed_monitoring(tmp_path, monkeypatch):
    from semantic_trainer.migrate import migrate
    from test_contract import republish
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    old = legacy(run, resolved, objective='instruction_transfer')
    with pytest.raises(ConfigError, match=r"changed numerical fitting inputs \['objective'\].*refit"):
        migrate(job, run, tmp_path / 'objective.json')
    source = resolved['sources'][0]
    manifest, rows = read_export(source['path'], native=True)
    fit = next(i for i, row in enumerate(rows) if row['id'] in resolved['roles']['fit'])
    entry = rows[fit]['native']['presentations'][0]
    changed = {**entry, 'answer_slots': entry['answer_slots'][::-1], 'answer_text': entry['answer_text'][::-1], 'id': None}
    changed['id'] = artifact_digest([rows[fit]['id'], {k: v for k, v in changed.items() if k != 'id'}])
    rows[fit] = {**rows[fit], 'native': {**rows[fit]['native'], 'presentations': [changed, *rows[fit]['native']['presentations'][1:]]}}
    relabelled = republish(tmp_path / 'old' / 'labels.export.jsonl.manifest.json', files.read_published(source['path']), rows)
    legacy(run, resolved, sources=[{**s, 'path': str(tmp_path / 'old' / 'labels.export.jsonl.manifest.json'), 'export_id': relabelled['export_id']} for s in old['sources']])
    with pytest.raises(ConfigError, match=r"changed numerical fitting inputs \['observations'\]"):
        migrate(job, run, tmp_path / 'slots.json')
    legacy(run, resolved, sources=[{**s, 'path': '/gone.manifest.json', 'export_id': 'd' * 64} for s in old['sources']])
    with pytest.raises(ConfigError, match='is unavailable; migration needs its rows'):
        migrate(job, run, tmp_path / 'gone.json')
    legacy(run, resolved, recipe={**old['recipe'], 'eval_every_examples': 16}, work={**old['work'], 'evaluation_points': 3})
    result = migrate(job, run, tmp_path / 'monitor.json')
    assert result['outcome'] == 'selection_evidence_required' and not result['eligible'] and set(result['changes']['monitor']) == {'cadence', 'evaluation_points'}
    with pytest.raises(ConfigError, match='selection_evidence_required: monitoring or selection plan changed .*new fit'):
        module.select(job, run, tmp_path / 'monitor.json')
    fitted = legacy(run, resolved)
    files.publish(run / 'resolved-job.json', {key: value for key, value in fitted.items() if key != 'probe_ids'})
    listed = files.read_json(run / 'candidates.json')
    files.publish(run / 'candidates.json', {**listed, 'fitted_job_id': artifact_digest({key: value for key, value in fitted.items() if key != 'probe_ids'})})
    with pytest.raises(ConfigError, match=r"missing equivalence evidence \['resolved-job.json:probe_ids'\]"):
        migrate(job, run, tmp_path / 'missing.json')
    legacy(run, resolved)
    files.publish(run / 'candidates.json', {**files.read_json(run / 'candidates.json'), 'loss': {**listed['loss'], 'version': 1}})
    with pytest.raises(ConfigError, match='loss-v1 artifacts cannot be relabelled'):
        migrate(job, run, tmp_path / 'v1.json')
    legacy(run, resolved)
    bridge = tmp_path / 'bridge.jsonl'
    chosen = [row for row in files.read_jsonl(records) if any(f"group:{group}" in {r['group'] for r in read_export(source['path'], native=True)[1]
                                                                                   if r['id'] in resolved['roles'][role]}
                                                           for group in row['groups'] for role in ('fit', 'model_select'))]
    by_role = {role: next(row for row in chosen if any(r['group'] == f"group:{row['groups'][0]}" and r['id'] in resolved['roles'][role]
                                                       for r in read_export(source['path'], native=True)[1])) for role in ('fit', 'model_select')}
    files.publish(bridge, ''.join(json.dumps({**row, 'groups': ['bridge'], 'role': 'acceptance_fit'}) + '\n' for row in by_role.values()))
    from semantic_switch import api
    identity = api.Runtime.identity
    monkeypatch.setattr(api.Runtime, 'identity', lambda self, role='judge': {**identity(self, role), 'sha256': files.sha256_file(resolved['model']['gguf'])})
    with Runtime({'judge': Native.tokens}) as runtime:
        bridged = tools.export(runtime, definition, to=tmp_path / 'bridge.export.jsonl', native=True, task_family='maintenance:test', labels=bridge)['path']
    data = files.read_yaml(job)
    data['exports'].append({'path': bridged, 'export_id': read_export(bridged, native=True)[0]['export_id'], 'roles': {}})
    files.publish(job, data)
    with pytest.raises(ConfigError, match='crosses roles'):
        migrate(job, run, tmp_path / 'bridged.json')
