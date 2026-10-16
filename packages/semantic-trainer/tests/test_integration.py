import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.artifacts import load_receipt
import semantic_switch.training as integration
import semantic_trainer
from semantic_testkit.scoring import FakeScorer
from semantic_trainer.jobs import single_task
from test_specialize import REALIZED, selection

TRAINER = Path(semantic_trainer.__file__).parent
PUBLIC = {'Runtime', 'Model', 'ConfigError'}


def imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield node.lineno, None, [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            yield node.lineno, node.module, [alias.name for alias in node.names]


def test_trainer_imports_only_the_public_and_integration_surfaces_including_function_local_imports():
    sources = sorted(TRAINER.glob('*.py'))
    assert {'jobs.py', 'lora.py', 'select.py', 'cli.py'} <= {path.name for path in sources} and not {'collect.py', 'probe.py'} & {path.name for path in sources}
    consumed = set()
    for path in sources:
        tree = ast.parse(path.read_text())
        for line, module, names in imports(tree):
            where = f'{path.name}:{line}'
            assert 'semantic_learn' not in {(module or name).split('.')[0] for name in names}, where
            if module is None:
                assert not any(name.split('.')[0] == 'semantic_switch' for name in names), where
            elif module == 'semantic_switch':
                assert set(names) <= PUBLIC, where
            elif module.split('.')[0] == 'semantic_switch':
                assert module == 'semantic_switch.training' and all(hasattr(integration, name) for name in names), where
                consumed.update(names)
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.Attribute) and node.attr in ('plan', 'snapshot', 'at', 'fact_index')), f'{path.name}:{node.lineno}'
            assert not (isinstance(node, ast.Name) and node.id == 'Example'), f'{path.name}:{node.lineno}'
    assert {'describe_task', 'freeze_task', 'read_export', 'freeze_specialization', 'Selection', 'admit', 'gpu_available', 'exact_families'} <= consumed
    assert not {'CONTRACT', 'INTEGRATION', 'supported', '__all__'} & set(vars(integration)) and not hasattr(semantic_trainer, 'INTEGRATION')


def test_function_local_internal_imports_are_detected():
    tree = ast.parse('def stage():\n    from semantic_switch.compiler import compile_spec\n    import semantic_switch.tools\n')
    assert [(module, names) for _, module, names in imports(tree)] == [('semantic_switch.compiler', ['compile_spec']), (None, ['semantic_switch.tools'])]


def test_incompatible_request_schema_fails_at_decoding(tmp_path):
    from semantic_switch.training import REQUEST_SCHEMA, decode_request
    with pytest.raises(ConfigError, match=f'not a schema-{REQUEST_SCHEMA} specialization request'):
        decode_request({'schema': REQUEST_SCHEMA - 1, 'kind': 'specialization-request', 'integration': 7}, 'old.json')


def test_task_description_keys_targets_by_full_address(tmp_path):
    from semantic_switch.training import describe_task
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': 'Money back.'}, 'issues': {'select': 'Which issues?', 'labels': {'refund': 'Money back', 'billing': 'Invoice'}}}})
    described = describe_task(path)
    assert 'integration' not in described and described['fields'] == {'message': {'type': 'text', 'required': True, 'values': None}}
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
api.Model.resources = lambda self: {'weights': api.Resources(1)}
api.available_host_bytes = lambda: 1 << 50
from semantic_switch.artifacts import load_receipt
from semantic_testkit.scoring import FakeScorer
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
with Runtime() as runtime:
    switch = runtime.load(receipt)
    status = switch({'message': 'report 4'}).facts['action'].status
assert not [name for name in sys.modules if name.split('.')[0] in BLOCKED]
print(json.dumps({'lock': switch.snapshot.lock_id, 'receipt': loaded.id, 'status': status}))
'''
    done = subprocess.run([sys.executable, '-c', code, json.dumps(REALIZED), receipt], cwd=tmp_path,
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
    monkeypatch.setattr(deployment, 'realized', lambda runtime, role='judge': {**realized(runtime, role), 'framing': {'changed': True}})
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
        given = {'model': FakeScorer(), 'task': single_task(resolved)[1], 'acceptance': accepting,
                 'provenance': history, 'population': {}, 'expected': {}, 'staging': staging, 'job': None, **change}
        return freeze_specialization(**given)
    (staging / 'left.txt').write_text('x')
    with pytest.raises(ConfigError, match='empty caller-owned staging'):
        freeze()
    (staging / 'left.txt').unlink()
    with pytest.raises(ConfigError, match='frozen task changed since preparation'):
        freeze(task={**single_task(resolved)[1], 'behavior': 'f' * 64})
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


def test_examples_to_fitted_specialist_needs_no_usage_capture_teacher_or_semantic_learn(tmp_path, monkeypatch):
    from semantic_switch import Runtime
    for name in [name for name in sys.modules if name.split('.')[0] == 'semantic_learn']:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, 'semantic_learn', None)
    with pytest.raises(ImportError):
        __import__('semantic_learn')
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    result = module.select(job, run)
    receipt = load_receipt(result['receipt'])
    assert receipt.provenance.kind == 'specialization' and receipt.provenance.rows
    with Runtime() as runtime:
        assert runtime.load(result['receipt']).snapshot.lock_id == receipt.data['lock']['id']
    assert all(sys.modules[name] is None for name in sys.modules if name.split('.')[0] == 'semantic_learn')
