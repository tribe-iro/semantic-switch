import json
import subprocess

import pytest

from semantic_switch import Check, ConfigError, Runtime
from semantic_switch.compiler import compile_spec
from semantic_switch.results import Fact, thaw

from .conftest import FakeScorer

CLAIM = 'The customer asks to receive money back for a charge.'


def spec(cases):
    return {'name': 'route', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
            'facts': {'refund': {'check': CLAIM}}, 'cases': cases, 'otherwise': 'other'}


def test_case_then_null_round_trips(tmp_path):
    plan = compile_spec(spec([{'id': 'quiet', 'when': 'refund', 'then': None}, {'id': 'named', 'when': 'not refund'}]))
    assert [(c.id, c.then) for c in plan.cases] == [('quiet', None), ('named', 'named')]
    again = compile_spec(json.loads(json.dumps(thaw(plan.spec))))
    assert [(c.id, c.then) for c in again.cases] == [('quiet', None), ('named', 'named')] and again.switch_rev == plan.switch_rev
    with pytest.raises(ConfigError, match='needs an id'):
        compile_spec(spec([{'when': 'refund', 'then': None}]))
    assert compile_spec(spec([{'when': 'refund', 'then': 'pay'}])).cases[0].id == 'pay'
    path = tmp_path / 'route.yaml'
    path.write_text(json.dumps(spec([{'id': 'quiet', 'when': 'refund', 'then': None}])))
    result = Runtime({'judge': FakeScorer()}).load(path, draft=True)({'message': 'Please refund my last charge'})
    assert (result.status, result.case, json.loads(json.dumps(result.as_dict()))['answer']) == ('ok', 'quiet', None)


@pytest.mark.parametrize('state, inline, saved', [
    ('', 'needs_input', 'needs_input'), ('   ', 'needs_input', 'needs_input'),
    ({'message': ''}, 'needs_input', 'needs_input'), ({'message': '  '}, 'needs_input', 'needs_input'),
    ({}, 'needs_input', 'needs_input'), ({'message': False}, 'present', 'failed')])
def test_empty_input_parity(tmp_path, state, inline, saved):
    fake = FakeScorer()
    rt = Runtime({'judge': fake})
    status = rt.check(state, CLAIM).status
    assert (status if status == 'needs_input' else 'present') == inline
    assert bool(fake.requests) == (inline == 'present')
    switch = Check(CLAIM).save(tmp_path / 'refund.yaml', rt=rt)
    assert switch(state).status == saved


def test_score_facts_returns_public_facts(rt, tmp_path):
    switch = Check(CLAIM).save(tmp_path / 'refund.yaml', rt=rt)
    out = switch.score_facts(['Please refund my last charge', '', 'hello'], sources=('judge',))
    assert [set(row) for row in out] == [{'refund'}] * 3 and all(isinstance(f, Fact) for row in out for f in row.values())
    refund, missing, hello = (row['refund'] for row in out)
    assert (refund.status, refund.value, refund.source, refund.mass) == ('ok', True, 'judge', 1.0)
    assert list(refund.scores) == ['true', 'false'] and refund.scores['true'] > 0.9
    assert missing.status == 'needs_input' and not missing.scores
    assert (hello.status, hello.value) == ('ok', False)
    json.dumps([{name: fact.as_dict(True) for name, fact in row.items()} for row in out])


def test_source_fingerprint_ignores_reports(tmp_path, monkeypatch):
    from bench.diagnostics import fingerprint
    from tests.test_artifact_layer import loaded
    root = tmp_path / 'repo'
    (root / 'bench' / 'results').mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '--allow-empty', '-m', 'x'], check=True)
    (root / 'bench' / 'run.py').write_text('x = 1\n')
    loaded(monkeypatch, root / 'bench' / 'run.py')
    before = fingerprint(root)
    (root / 'bench' / 'results' / 'report.json').write_text('{}')
    (root / 'notes.txt').write_text('scratch')
    (root / 'bench' / 'unused.py').write_text('y = 1\n')
    (root / 'bench' / 'run.py.write-lock').write_text('')
    assert fingerprint(root) == before and before['revision']
    (root / 'bench' / 'run.py').write_text('x = 2\n')
    assert fingerprint(root)['sources'] != before['sources']