import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from typer.testing import CliRunner

from semantic_switch import ConfigError, files
from semantic_switch.cli import app
from semantic_switch.training import REQUEST_SCHEMA
from semantic_switch.tools.specialize import read_request, request, specialize

WORKER = '''#!{python}
import json, os, sys, time
log = os.environ["FAKE_TRAINER_LOG"]
with open(log, "a") as fh:
    fh.write(json.dumps(sys.argv) + "\\n")
if sys.argv[1] == "protocol":
    print(json.dumps({{"request": int(os.environ.get("FAKE_REQUEST", "{request}")), "result": 1, "version": "fake",
                      "stages": ["freeze", "assets", "export", "population", "profile", "cadence", "finalize", "fit", "select", "lock"]}}))
    sys.exit(0)
mode = os.environ["FAKE_TRAINER_MODE"]
print("stage progress", file=sys.stderr)
if mode == "hang":
    with open(os.environ["FAKE_TRAINER_PID"], "w") as fh:
        fh.write(str(os.getpid()))
    time.sleep(600)
if mode == "garbage":
    print("not a result")
    sys.exit(1)
state, code = {{"complete": ("completed", 0), "await": ("awaiting_configuration", 0), "fail": ("failed", 2), "lie": ("completed", 1)}}[mode]
print(json.dumps({{"schema": 1, "kind": "specialization-result", "state": state, "stage": "lock", "argv": sys.argv[1:]}}))
sys.exit(code)
'''


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    path = tmp_path / 'bin' / 'semantic-trainer'
    path.parent.mkdir()
    path.write_text(WORKER.format(python=sys.executable, request=REQUEST_SCHEMA))
    path.chmod(0o755)
    monkeypatch.setenv('FAKE_TRAINER_LOG', str(tmp_path / 'argv.jsonl'))
    monkeypatch.setenv('FAKE_TRAINER_MODE', 'complete')
    return path


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'argv.jsonl').read_text().splitlines()]


def sources(tmp_path):
    definition = tmp_path / 'src' / 'support.yaml'
    files.publish(definition, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                               'facts': {'refund': {'check': 'The customer asks for money back.'}}})
    labels = tmp_path / 'src' / 'labels.jsonl'
    files.publish(labels, '{}\n')
    return definition, labels


def built(tmp_path, work, **changes):
    definition, labels = sources(tmp_path)
    return request(definition, labels, work, **{'devices': {'fit': 'xpu:0', 'native': 'gpu'}, 'model': {'repo': 'google/gemma-4-E2B-it', 'revision': 'main'},
                                                'recipe': ['seed=0'], **changes})


def test_facade_writes_one_request_and_passes_an_argument_vector_without_a_shell(tmp_path, trainer):
    work = tmp_path / 'work; $(touch injected) `x`'
    result = specialize(str(trainer), work, built(tmp_path, work), until='profile', attempt=2, cadence=552)
    request_path = work / 'request.json'
    assert result['state'] == 'completed' and result['exit'] == 0
    assert calls(tmp_path) == [[str(trainer), 'protocol'], [str(trainer), 'pipeline', str(request_path), '--until', 'profile', '--attempt', '2', '--cadence', '552']]
    assert not (tmp_path / 'injected').exists()
    recorded = read_request(request_path)
    assert recorded['devices'] == {'fit': 'xpu:0', 'native': 'gpu'} and recorded['model']['quant'] == 'Q8_0' and recorded['cadence'] is None
    assert recorded['sources']['labels']['sha256'] == files.sha256_file(tmp_path / 'src' / 'labels.jsonl')
    again = specialize(str(trainer), work)
    assert again['state'] == 'completed' and calls(tmp_path)[-1] == [str(trainer), 'pipeline', str(request_path)]
    with pytest.raises(ConfigError, match='holds another specialization request'):
        specialize(str(trainer), work, built(tmp_path, work, representation='runtime-adapter'))


@pytest.mark.parametrize('configured, error', [(None, 'no trainer executable is configured'), ('semantic-trainer', 'absolute path'),
                                               ('/nonexistent/semantic-trainer', 'absolute path')])
def test_missing_or_relative_trainer_fails_before_any_request(tmp_path, configured, error):
    work = tmp_path / 'work'
    with pytest.raises(ConfigError, match=error) as raised:
        specialize(configured, work, built(tmp_path, work))
    assert 'pip install' in str(raised.value) and not work.exists()


def test_unknown_until_fails_against_the_trainer_stages_before_any_request(tmp_path, trainer):
    work = tmp_path / 'work'
    with pytest.raises(ConfigError, match="--until 'fitting' names no pipeline stage; stages: freeze, assets"):
        specialize(str(trainer), work, built(tmp_path, work), until='fitting')
    assert calls(tmp_path) == [[str(trainer), 'protocol']] and not work.exists()


def test_incompatible_trainer_protocol_fails_before_work(tmp_path, trainer, monkeypatch):
    monkeypatch.setenv('FAKE_REQUEST', '99')
    work = tmp_path / 'work'
    with pytest.raises(ConfigError, match='install the semantic-trainer release built for this semantic-switch'):
        specialize(str(trainer), work, built(tmp_path, work))
    assert calls(tmp_path) == [[str(trainer), 'protocol']] and not work.exists()


def test_failed_child_propagates_its_result_and_code_and_protocol_breaches_fail(tmp_path, trainer, monkeypatch):
    work = tmp_path / 'work'
    monkeypatch.setenv('FAKE_TRAINER_MODE', 'fail')
    failed = specialize(str(trainer), work, built(tmp_path, work))
    assert failed['state'] == 'failed' and failed['exit'] == 2
    for mode, error in (('garbage', 'without one schema-1 specialization result'), ('lie', 'broke the result protocol')):
        monkeypatch.setenv('FAKE_TRAINER_MODE', mode)
        with pytest.raises(ConfigError, match=error):
            specialize(str(trainer), work)


def test_cli_maps_awaiting_configuration_and_resumes_from_the_work_root(tmp_path, trainer, monkeypatch):
    definition, labels = sources(tmp_path)
    monkeypatch.setenv('SEMANTIC_TRAINER', str(trainer))
    monkeypatch.setenv('FAKE_TRAINER_MODE', 'await')
    runner = CliRunner()
    first = runner.invoke(app, ['--json', 'specialize', str(definition), '--examples', str(labels), '--to', str(tmp_path / 'work'), '--device', 'xpu:0',
                                '--native-device', 'gpu', '--recipe', 'seed=0', '--fit-situations', '100'])
    assert first.exit_code == 4 and json.loads(first.stdout)['state'] == 'awaiting_configuration'
    recorded = read_request(tmp_path / 'work' / 'request.json')
    assert recorded['selection'] == {'fit_situations': 100, 'acceptance_fit': False} and recorded['profile'] is None
    monkeypatch.setenv('FAKE_TRAINER_MODE', 'complete')
    resumed = runner.invoke(app, ['--json', 'specialize', '--to', str(tmp_path / 'work'), '--cadence', '552'])
    assert resumed.exit_code == 0 and json.loads(resumed.stdout)['argv'][-2:] == ['--cadence', '552']


def test_cancelling_the_facade_terminates_and_reaps_the_worker(tmp_path, trainer):
    definition, labels = sources(tmp_path)
    work = tmp_path / 'work'
    files.write_new(work / 'request.json', built(tmp_path, work))
    pid_file = tmp_path / 'worker.pid'
    script = ('import sys; from semantic_switch.tools.specialize import specialize; specialize(sys.argv[1], sys.argv[2], grace=5)')
    environment = {**os.environ, 'FAKE_TRAINER_MODE': 'hang', 'FAKE_TRAINER_PID': str(pid_file)}
    facade = subprocess.Popen([sys.executable, '-c', script, str(trainer), str(work)], env=environment, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline and facade.poll() is None
        time.sleep(.05)
    worker = int(pid_file.read_text())
    facade.send_signal(signal.SIGTERM)
    assert facade.wait(30) != 0
    with pytest.raises(ProcessLookupError):
        os.kill(worker, 0)


def test_request_rejects_cadence_in_the_recipe_and_unknown_representation(tmp_path):
    work = tmp_path / 'work'
    with pytest.raises(ConfigError, match='separate decision'):
        built(tmp_path, work, recipe=['eval_every_examples=550'])
    with pytest.raises(ConfigError, match='representation must be one of'):
        built(tmp_path, work, representation='blended')
    with pytest.raises(ConfigError, match='either a base model'):
        request(*sources(tmp_path), work, devices={'fit': 'cpu', 'native': 'cpu'})


def test_core_facade_imports_no_trainer_or_torch():
    code = ('import sys, semantic_switch.cli, semantic_switch.tools.specialize; '
            'assert not {"semantic_trainer", "torch"} & set(sys.modules), sorted(m for m in sys.modules if m.startswith(("semantic_trainer", "torch")))')
    subprocess.run([sys.executable, '-c', code], check=True)


def test_invoke_cancellation_stops_a_grandchild_ignoring_termination(tmp_path):
    import signal
    import time
    worker = tmp_path / 'tree.py'
    worker.write_text('import subprocess,sys,time\nfrom pathlib import Path\n'
                      'child=subprocess.Popen([sys.executable,"-c","import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"],stdout=subprocess.DEVNULL)\n'
                      'Path(sys.argv[1]).write_text(str(child.pid))\ntime.sleep(60)\n')
    pid_file = tmp_path / 'grandchild'
    code = 'from semantic_switch.tools.specialize import invoke;import sys;invoke(sys.argv[1:],grace=.15)'
    supervisor = subprocess.Popen([sys.executable, '-c', code, sys.executable, str(worker), str(pid_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        until = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert pid_file.exists()
        grandchild = int(pid_file.read_text())
        supervisor.send_signal(signal.SIGTERM)
        supervisor.communicate(timeout=5)
        assert supervisor.returncode != 0
        proc = Path(f'/proc/{grandchild}/stat')
        assert not proc.exists() or proc.read_text().split()[2] == 'Z'
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.communicate()
