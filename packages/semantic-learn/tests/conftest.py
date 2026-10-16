import json
from pathlib import Path
import stat
import sys

import pytest

from semantic_switch import files
from semantic_switch.training import RECEIPT_SCHEMA, artifact_digest

REFUND = 'The customer asks to receive money back for a charge.'
SPEC = {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'], 'facts': {'refund': {'check': REFUND}}}
TRAINER = '''#!{python}
import json, pathlib, sys
here = pathlib.Path(__file__).parent
if sys.argv[1] == 'protocol':
    print(json.dumps({{'request': {request}, 'result': {result}, 'version': 'fake', 'stages': ['freeze', 'fit', 'lock']}}))
    raise SystemExit(0)
with (here / 'calls.jsonl').open('a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
request = json.loads(pathlib.Path(sys.argv[2]).read_text())
labels = request['sources']['labels']['path']
if (here / 'fail-once').exists():
    (here / 'fail-once').unlink()
    print(json.dumps({{'schema': {result}, 'kind': 'specialization-result', 'state': 'failed', 'stage': 'fit', 'error': 'interrupted'}}))
    raise SystemExit(1)
print(json.dumps({{'schema': {result}, 'kind': 'specialization-result', 'state': 'completed', 'stage': 'lock', 'labels': labels,
                  'lock': str(pathlib.Path(request['work']) / 'lock' / 'task.lock.json'), 'receipt': str(pathlib.Path(request['work']) / 'receipt.json')}}))
'''


def definition(directory, statement=REFUND):
    path = Path(directory) / 'support.yaml'
    files.publish(path, {**SPEC, 'facts': {'refund': {'check': statement}}})
    return path


def receipt(directory, name, keys=()):
    root = Path(directory) / name
    root.mkdir(parents=True)
    (root / 'model.gguf').write_bytes(name.encode())
    files.publish(root / 'task.yaml', SPEC)
    files.publish(root / 'task.lock.json', {'id': f'lock-{name}'})
    digest = artifact_digest(name)
    rows = [{'id': artifact_digest([name, key]), 'fact': 'refund', 'projection': key, 'group': 'g', 'component': digest, 'family_seeds': [['key', key]],
             'sources': [{'export_id': digest, 'path': 'export.jsonl', 'task_family': 'local:support', 'namespace': 'support', 'group': 'g', 'component': digest,
                          'aliases': [], 'family_seeds': [['key', key]]}], 'label_sources': [digest], 'uses': ['adaptation']} for key in keys]
    data = {'schema': RECEIPT_SCHEMA, 'kind': 'specialization-receipt', 'gguf': 'model.gguf', 'task': 'task.yaml',
            'lock': {'path': 'task.lock.json', 'id': f'lock-{name}', 'candidate_id': f'candidate-{name}'},
            'files': {item: files.sha256_file(root / item) for item in ('model.gguf', 'task.yaml', 'task.lock.json')}, 'native': {}, 'execution': {},
            'scope': {'targets': ['refund'], 'whole_task': True, 'binding': 'judge'}, 'publication': 'application',
            'provenance': {'kind': 'specialization' if keys else 'base', 'rows': rows}}
    data['id'] = artifact_digest(data)
    files.publish(root / 'receipt.json', data)
    return root / 'receipt.json'


@pytest.fixture
def trainer(tmp_path):
    from semantic_switch.training import REQUEST_SCHEMA, RESULT_SCHEMA
    root = tmp_path / 'trainer-env'
    root.mkdir()
    path = root / 'semantic-trainer'
    path.write_text(TRAINER.format(python=sys.executable, request=REQUEST_SCHEMA, result=RESULT_SCHEMA))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def calls(trainer):
    log = Path(trainer).parent / 'calls.jsonl'
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
