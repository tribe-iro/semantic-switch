import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_switch import Runtime, files, tools
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import compile_spec
from semantic_testkit.labels import TEACHER, declared
from semantic_testkit.scoring import FakeScorer, letters_of

HERE = Path(__file__).resolve().parents[1]
MESSAGES = {'please refund the double charge': 'Billing', 'the app crashes with error 502': 'Technical', 'cancel my plan at renewal': 'Retention'}


def load():
    spec = importlib.util.spec_from_file_location('routing_acquire', HERE.parents[1] / 'experiments' / 'routing' / 'acquire.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher(disputed):
    def oracle(state, question, width):
        options, wanted = letters_of(question), next((queue for message, queue in MESSAGES.items() if message in state), 'Ask')
        if disputed in state:
            wanted = 'Account'
        hit = next(letter for text, letter in options.items() if text.startswith(wanted))
        return [.96 if chr(65 + index) == hit else .04 / (width - 1) for index in range(width)]
    return oracle


@pytest.fixture
def acquire(tmp_path, monkeypatch):
    module = load()
    root = tmp_path / 'routing'
    root.mkdir()
    files.publish(root / 'routing.yaml', module.definition())
    monkeypatch.setattr(module, 'DEFINITION', root / 'routing.yaml')
    monkeypatch.setattr(module, 'DATA', root / 'data')
    monkeypatch.setattr(module, 'CHECK', root / 'check')
    monkeypatch.setattr(module, 'spent', lambda: 0.)
    module.task.cache_clear()
    return module


def test_generated_variants_differ_only_in_reads_and_keep_their_meanings():
    module = load()
    spec = module.definition()
    first, second = (spec['facts'][name] for name in ('routing', 'routing_reference'))
    assert {k: v for k, v in first.items() if k != 'reads'} == {k: v for k, v in second.items() if k != 'reads'} and first['reads'] != second['reads']
    assert files.read_yaml(HERE / 'routing.yaml') == spec
    plan = compile_spec(spec, 'routing.yaml', HERE)
    assert {fact.name: fact.meaning_rev for fact in plan.facts} == {'routing': '757ab6d6', 'routing_reference': 'b8277f2d'}


def test_fake_transport_collection_reaches_review_and_export(acquire, tmp_path, monkeypatch):
    from semantic_learn import teacher as learn
    rows = [acquire.situation(f'routing-{n:04d}', 'fit', [], ['plain'], message, None, translation=False, phenomenon='plain')
            for n, message in enumerate(MESSAGES)]
    rows.append(acquire.situation('routing-0003', 'fit', [], ['plain'], 'hello, see the note', 'Plan fees are listed monthly.', translation=False,
                                  phenomenon='plain'))
    acquire.envelopes(acquire.DATA, rows)
    files.publish(acquire.DATA / 'situations.jsonl', acquire.jsonl(rows))
    use = tmp_path / 'use.yaml'
    files.publish(use, declared())
    scorers = {'a': FakeScorer(teacher('nothing disputed')), 'b': FakeScorer(teacher('cancel my plan'))}
    for name, scorer in scorers.items():
        monkeypatch.setattr(learn, 'judge', lambda source, provider=None, scorer=scorer: scorer)
        counts = acquire.label(SimpleNamespace(pool='fit', teacher=TEACHER, provider=None, training_use=str(use), name=name, retry_failed=False))
        assert counts['routing']['settled'] == len(MESSAGES) and counts['routing_reference']['settled'] == 1
    assert [entry['stage'] for entry in files.read_jsonl(acquire.DATA / 'receipts.jsonl')] == ['label'] * 4
    exported = acquire.review(SimpleNamespace(pool='fit', teachers=['a', 'b'], name='first', size=3, audit=0, stratum=['conflict'], seed=0))
    queue = files.read_jsonl(exported['queue'])
    assert exported['conflicts'] == ['routing'] and len(queue) == 1 and 'facts' not in queue[0] and queue[0]['input']['message'] == 'cancel my plan at renewal'
    audit = acquire.review(SimpleNamespace(pool='fit', teachers=['a'], name='audit', size=4, audit=4, stratum=[], seed=0))
    labelled = files.read_jsonl(audit['queue'])
    assert len(labelled) == 4 and all(history[0]['training_use'] == declared() for row in labelled for history in row['label_sources'].values())
    assembled = acquire.assemble(SimpleNamespace(teacher='a', reviewed=[]))
    assert assembled['unresolved_situations'] == 0 and assembled['families_per_size'] == {}
    with Runtime(FakeScorer()) as runtime:
        result = tools.export(runtime, acquire.DEFINITION, split='all', labels=audit['queue'])
    exported_rows = read_export(result['path'], fitting=False)[1]
    assert {row['label_derivation'] for row in exported_rows} == {'judged'} and len(exported_rows) == 4
