from __future__ import annotations

import semantic_switch as ss
from semantic_switch import Check, Runtime, files, tools
from semantic_testkit.scoring import FakeScorer
from bench.outcomes import Receipts, runtime
from tests.test_openrouter_judge import Provider
from tests.test_remaining_contracts import definition


def hosted(provider, receipts):
    return runtime(ss.judge('openrouter:m', codec='chat', transport=provider), receipts=receipts)


def test_event_receipts_record_usage_operation_and_count_each_request_once():
    provider, receipts = Provider(), Receipts()
    with hosted(provider, receipts) as rt:
        with receipts.operation('warmup', warmup=True):
            rt.ask('Please refund me.', {'refund': Check('The customer asks for a refund.')})
        before = len(receipts)
        with receipts.operation('pair'):
            rt.ask('Please refund me.', {'a': Check('The customer asks for a refund.'), 'b': Check('The customer is angry.')})
        summary = receipts.summary()
    assert summary['requests'] == len(receipts) == len(provider.bodies) and len(receipts) > before
    assert summary['by_operation']['warmup']['warmup'] and not summary['by_operation']['pair']['warmup']
    assert summary['usage']['reported'] == summary['requests'] and summary['failed'] == 0
    assert all(row['operation']['id'] == 'pair' for row in receipts.since(before))
    assert summary['latency']['samples'] == summary['requests'] and summary['source'].startswith('request receipts')


def test_local_runtime_has_no_request_receipts():
    receipts = Receipts()
    with runtime(FakeScorer(), receipts=receipts) as rt:
        rt.ask('case', {'refund': Check('The customer asks for a refund.')})
    assert receipts.summary()['requests'] == 0


def test_regression_comparison_freezes_grouped_policy_without_retuning(tmp_path):
    from bench.openrouter import regression_binding, run_audit
    audit = tmp_path / 'audit.jsonl'
    with Runtime(FakeScorer(lambda *args: [.99, .005, .005])) as rt:
        path = regression_binding(rt, tmp_path / 'model', audit)
        before = files.read_json(path.with_suffix('.lock.json'))
        result = run_audit(rt, Receipts(), path, audit)
        assert files.read_json(path.with_suffix('.lock.json')) == before
        assert result['candidate_id'] == before['candidate_id']
        assert result['metrics']['fact:refund']['submitted'] == 4 and result['requests']['requests'] == 0
        assert all(row['label_derivations'] == ['authored'] for row in result['observations'])
        assert len({row['id'] for row in result['observations']}) == 4


def test_policy_audit_rejects_repeated_inputs_before_scoring(tmp_path):
    import json
    import pytest
    from semantic_switch import ConfigError
    from semantic_switch.evidence import label_record
    from bench.policy_audit import run
    scorer = FakeScorer()
    with Runtime(scorer) as rt:
        path = definition(tmp_path)
        tools.lock(rt, path)
        sw = rt.load(path)
        rows = [label_record(sw.plan, {'message': 'case'}, {'refund': True},
                             sources={'refund': [{'derivation': 'authored', 'producer': 'fixture', 'actor': {'kind': 'human'}}]}, groups=[f'family{n}'])
                for n in range(2)]
        audit = tmp_path / 'audit.jsonl'
        files.publish(audit, ''.join(json.dumps(row) + '\n' for row in rows))
        before = len(scorer.requests)
        with pytest.raises(ConfigError, match='IDs'):
            run(rt, path, audit)
        assert len(scorer.requests) == before
