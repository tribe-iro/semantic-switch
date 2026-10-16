import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from semantic_switch import Runtime, cli, files, judge, tools
from semantic_testkit.scoring import FakeScorer

SPEC = {'name': 'demo', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
        'facts': {'refund': {'check': 'The customer asks to receive money back for a charge.'}},
        'rules': [{'id': 'refund', 'when': 'refund'}], 'otherwise': 'other'}
CASE = {'message': 'refund my charge'}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files.publish(tmp_path / 'switches' / 'demo.yaml', SPEC)
    monkeypatch.setattr(cli, 'judge', lambda source, **settings: FakeScorer())
    return tmp_path


class Refusing:
    async def __call__(self, *args):
        raise AssertionError('probe request sent')


def invoke(*arguments, stdin=json.dumps(CASE)):
    result = CliRunner().invoke(cli.app, ['--judge', 'fake', *arguments], input=stdin)
    assert result.exception is None or result.exit_code in (0, 1, 2), result.exception
    return result.stdout


def test_runtime_load_owns_name_resolution_and_the_cli_passes_names_unchanged(workspace):
    with Runtime(FakeScorer()) as runtime:
        switch = runtime.load('demo')
        assert switch.paths.yaml == Path('switches/demo.yaml')
        expected = switch(CASE).as_dict(False)
        assert runtime.load('switches/demo.yaml').paths.yaml == Path('switches/demo.yaml')
    assert json.loads(invoke('--json', 'run', 'demo')) == expected
    files.publish(workspace / 'demo.yaml', {**SPEC, 'otherwise': 'local'})
    with Runtime(FakeScorer()) as runtime:
        assert runtime.load('demo').paths.yaml == Path('switches/demo.yaml') and runtime.load('demo.yaml').paths.yaml == Path('demo.yaml')


def test_run_verbose_covers_text_raw_and_json_explanations(workspace, capsys, monkeypatch):
    from semantic_switch.executor import Operation
    monkeypatch.setattr(Operation, "elapsed", lambda self, live: 0.)
    with Runtime(FakeScorer()) as runtime:
        result = runtime.load('demo')(CASE)
    token = cli._INVOCATION_OPTIONS.set({'json': False, 'verbose': True})
    try:
        cli.emit(result)
        cli.emit(result, raw=True)
    finally:
        cli._INVOCATION_OPTIONS.reset(token)
    assert capsys.readouterr().out == result.explain() + '\n' + result.explain(True) + '\n'
    record = json.loads(invoke('--json', '--verbose', 'run', 'demo', '--raw'))
    assert record['explain'] == result.explain(True).splitlines() and record['status'] == result.status
    assert 'explain' not in json.loads(invoke('--json', 'run', 'demo'))
    assert 'explain' not in {command.name or command.callback.__name__ for command in cli.app.registered_commands}


def test_doctor_reuses_complete_cached_conformance_and_reports_hosted_as_unsupported(monkeypatch):
    with Runtime(FakeScorer()) as runtime:
        first = tools.doctor(runtime)
        monkeypatch.setattr(type(runtime.judge()), 'selftest', lambda self: pytest.fail('native checks repeated'))
        second = tools.doctor(runtime)
    assert first['status'] == second['status'] == 'passed' and second['cached'] and second['checks'] == first['checks']
    with Runtime(judge('openrouter:m', codec='chat', provider='p', transport=Refusing()), own_judges=True) as runtime:
        hosted = tools.doctor(runtime)
    assert hosted['status'] == 'unsupported' and hosted['ok'] and hosted['checks'] == {}
