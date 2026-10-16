import numpy as np
import pytest

from semantic_switch import Runtime, files
from semantic_switch.compiler import compile_spec, digest
from semantic_switch.context import json_value
from semantic_switch.examples import Claim, Frame, LabelSource
from semantic_switch.judge import Request
from semantic_switch.results import freeze
from semantic_testkit.scoring import FakeScorer
from bench.outcomes import fingerprint, provenance


def test_nested_ownership_and_returned_records_are_independent():
    plan = compile_spec({'name': 'json', 'fields': {'data': {'type': 'json'}}, 'facts': {'c': {'check': 'A value is present.'}}})
    original = {'data': {'rows': [{'n': 1}, None]}}
    frame = Frame(plan, original)
    key = frame.full
    original['data']['rows'][0]['n'] = 2
    assert frame.full == key and frame.values['data']['rows'][0]['n'] == 1
    with pytest.raises(TypeError):
        frame.values['data']['rows'][0]['n'] = 3
    record = frame.record()
    record['data']['rows'][0]['n'] = 4
    assert frame.values['data']['rows'][0]['n'] == 1
    assert freeze(frame.values['data']) is frame.values['data']
    assert json_value(frame.values['data']) is frame.values['data']
    assert files._json(frame.values) == files._json(frame.record())


def test_claim_owns_value_and_counts_encoded_bytes():
    source = LabelSource('unknown', '汉' * 1000)
    claim = Claim({'v': [1]}, [source])
    assert claim.size == len(files._json(claim.record()).encode('utf-8'))
    with pytest.raises(TypeError):
        claim.value['v'][0] = 2
    with pytest.raises(AttributeError):
        claim.value = {'v': [2]}
    record = claim.record()
    record['value']['v'][0] = 3
    assert claim.value['v'][0] == 1


def test_unobserved_switch_receipts_remain_typed(tmp_path, monkeypatch):
    path = tmp_path / 's.yaml'
    files.publish(path, {'name': 's', 'fields': {'message': {'type': 'text'}}, 'facts': {'c': {'check': 'The customer asks for a refund.'}}})
    calls = []
    original = Request.as_dict
    monkeypatch.setattr(Request, 'as_dict', lambda self: (calls.append(self), original(self))[1])
    with Runtime(FakeScorer()) as rt:
        result = rt.load(path)('refund me')
        assert calls == []
        request = Request('test', 1., 1, 200, {'cost': 2.})
        result.cost.requests = (request,)
        assert result.as_dict(True)['cost']['requests'] == [original(request)]
        assert calls == [request]


def test_content_identity_covers_middle_array_and_ignores_imports():
    a = np.arange(2000)
    b = a.copy()
    b[1000] += 1
    assert digest(a) != digest(b)
    with Runtime(FakeScorer()) as rt:
        before = provenance(rt, [a])
        import semantic_switch.testing.drivers
        assert fingerprint()['sources'] == before['sources']
        assert provenance(rt, [b])['dataset'] != before['dataset']


def test_scalar_array_identity_preserves_its_original_shape():
    assert digest(np.asarray(1)) != digest(np.asarray([1]))


def test_frozen_records_reuse_owned_maps_and_cost_obeys_mapping_operations():
    from semantic_switch.results import Cost
    record = freeze({'nested': {'rows': [1, 2, None]}})
    assert freeze(record) is record
    cost = Cost({'elapsed_ms': 1.}, (Request('r', 0., 1, 200, None),))
    assert len(cost) == 2
    assert len(cost.pop('requests')) == 1
    assert 'requests' not in cost and len(cost) == 1
