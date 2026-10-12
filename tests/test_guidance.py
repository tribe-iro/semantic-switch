import json

import pytest

from semantic_switch.artifacts import read_export
from semantic_switch import Check, ConfigError, Guidance, Pick, Rate, Runtime, files, judge_from, tools
from semantic_switch.compiler import compile_spec
from semantic_switch.hosted import Reply, TransportError
from semantic_switch.judge import Scored, Scores
from semantic_switch.protocols import Question
from tests.conftest import FakeScorer, transport


def structured(send):
    return judge_from("openrouter:typesafe/jev-1.13", transport=transport(lambda url, body, timeout: Reply(200, send(body))))


def definition(tmp_path, guidance=None):
    path = tmp_path / 'policy.yaml'
    files.publish(path, {'name': 'policy', 'guidance': guidance or {},
                           'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                           'facts': {'refund': {'check': 'Requests money back.', 'unstated': True},
                                     'plain': {'check': 'Requests money back.', 'guidance': []}},
                           'cases': [{'id': 'refund', 'when': 'refund', 'then': {'steps': [{'action': 'refund'}]}}]})
    return path


def test_guidance_normalization_and_identity(tmp_path):
    path = tmp_path / 'terms.txt'
    path.write_bytes(b'first\r\nsecond')
    a = Guidance.load(path)
    path.unlink()
    assert a.digest == Guidance('first\nsecond').digest
    assert a.sections[0].source_location
    with pytest.raises(ConfigError):
        Guidance(sections=(('a', 'one', None), ('a', 'two', None)))
    with pytest.raises(ConfigError):
        Guidance('\ud800')


def test_item_guidance_replaces_operation_default():
    captured = []
    class Judge:
        identity = {'source': 'test'}
        accelerated = False
        max_slots = 20
        batch_inputs = 8
        supports_guidance = True
        def fit(self, text, limit):
            return text, False
        def score_many(self, items):
            import numpy as np
            captured.extend(items)
            return [Scored(tuple(Scores(np.array([.99, .01]), None) for _ in qs), 0, 'test-model') for _, qs in items]
    with Runtime({'judge': Judge()}) as rt:
        answers = rt.ask_many(['one', 'two'], {'inherit': Check('claim'), 'empty': Check('claim', guidance=Guidance()),
                                             'replace': Check('claim', guidance='own')}, guidance='shared')
    assert [q.guidance.text for q in captured[0][1]] == ['[guidance]\nshared', '', '[guidance]\nown']
    assert answers[0]['inherit'].guidance == Guidance('shared').references
    assert answers[1]['replace'].observed_models == ('test-model',)


def test_empty_guidance_preserves_existing_prompt_tokens_and_lock(tmp_path):
    scorer = FakeScorer(lambda state, question, width: [.99, .01] if width == 2 else [.99, .005, .005])
    with Runtime({'judge': scorer}) as rt:
        judge = rt.judge()
        a = Check('claim').prepare('x', Guidance()).make(judge)[0].question
        b = Check('claim', guidance=Guidance()).prepare('x', Guidance()).make(judge)[0].question
        assert judge.adapter.request(judge.adapter.state('x'), [a]) == judge.adapter.request(judge.adapter.state('x'), [b])
        path = definition(tmp_path)
        tools.lock(rt, path)
        assert rt.load(path)('x').status == 'ok'


def test_pinned_guidance_lock_survives_source_edit_or_removal(tmp_path):
    source = tmp_path / 'terms.txt'
    source.write_text('Refund includes store credit.')
    path = definition(tmp_path, {'terms': {'file': 'terms.txt'}})
    with Runtime({'judge': FakeScorer(lambda state, question, width: [.99, .01] if width == 2 else [.99, .005, .005])}) as rt:
        old = rt.load(path).plan.fact('refund').meaning_rev
        plain = rt.load(path).plan.fact('plain').meaning_rev
        tools.lock(rt, path)
        source.write_text('Refund excludes store credit.')
        draft = rt.load(path, draft=True)
        assert draft.plan.fact('refund').meaning_rev != old
        assert draft.plan.fact('plain').meaning_rev == plain
        source.unlink()
        restored = rt.load(path)
        assert restored.plan.fact('refund').meaning_rev == old
        assert restored('x').answer == {'steps': [{'action': 'refund'}]}
        json.dumps(restored('x').as_dict())


def test_guidance_legacy_judge_refuses_nonempty_binding():
    judge = structured(lambda _: {})
    judge.supports_guidance = False
    with Runtime({'judge': judge}) as rt:
        with pytest.raises(ConfigError, match='guidance'):
            rt.check('x', 'claim', guidance='rules')
    judge.close()


def test_unknown_guidance_reference_fails_before_decode():
    with pytest.raises(ConfigError, match='unknown'):
        compile_spec({'name': 'x', 'facts': {'a': {'check': 'claim', 'guidance': ['missing']}}})


def test_accepted_unknown_needs_input_vs_low_confidence_uncertain(tmp_path):
    fake = FakeScorer(lambda state, question, width: [.01, .01, .98])
    path = definition(tmp_path, {'rules': {'text': 'Use only supplied evidence.'}})
    with Runtime({'judge': fake}) as rt:
        inline = rt.check('unknown', 'Requests money back.', unstated=True, guidance='Use only supplied evidence.')
        assert inline.status == 'needs_input' and inline.error == 'insufficient_evidence'
        result = rt.load(path)('unknown')
        assert result.status == 'needs_input' and result.missing == ()
        assert result.facts['refund'].why == 'insufficient_evidence'
    with Runtime({'judge': FakeScorer(lambda *args: [.34, .33, .33])}) as rt:
        assert rt.check('unknown', 'claim', unstated=True, guidance='rules').status == 'uncertain'


@pytest.mark.parametrize('invalid', [None, [], 1, {'A': True, 'B': 0}, {'A': 0, 'B': 0}, {'AA': .9, 'B': .1}, {'A': 1.0000001, 'B': -.0000001}])
def test_hosted_malformed_answer_isolated(invalid):
    judge = structured(lambda _: {'model': 'm', 'answers': {'q0': {'probabilities': invalid}, 'q1': {'noul': .99}}})
    try:
        scores = judge.score_many([('x', [Question('pick', 'choice', ('a', 'b')), Question('check', 'claim')])])[0]
        assert scores.scores[0].error and not scores.scores[1].error
    finally:
        judge.close()


def test_hosted_payload_programming_error_propagates():
    judge = structured(lambda _: {})
    judge.payload = lambda *args: (_ for _ in ()).throw(AttributeError('bug'))
    try:
        with pytest.raises(AttributeError, match='bug'):
            judge.score_many([('x', [Question('check', 'claim')])])
    finally:
        judge.close()


def test_hosted_transport_failure_isolated():
    def send(body):
        if body['state'] == 'bad':
            raise TransportError('offline')
        return {'model': 'm', 'answers': {'q0': {'noul': .99}}}
    judge = structured(send)
    try:
        results = judge.score_many([('bad', [Question('check', 'claim')]), ('good', [Question('check', 'claim')])])
        assert results[0].scores[0].error and not results[1].scores[0].error
    finally:
        judge.close()


def test_systemone_guidance_wire_roundtrip():
    bodies, events = [], []
    def send(body):
        bodies.append(body)
        return {'model': 'local-v1', 'answers': {'q0': {'noul': .99}}}
    judge = structured(send)
    try:
        with Runtime({'judge': judge}, on_decision=events.append) as rt:
            result = rt.check('case', 'claim', guidance='policy')
        assert bodies[0]['state'] == 'case'
        assert 'policy' in bodies[0]['questions']['q0']['instructions']
        assert result.observed_models == ('local-v1',)
        assert events[0].facts['_']['observed_models'] == ('local-v1',)
        assert judge.identity['protocol'] == 'systemone'
    finally:
        judge.close()


def test_nested_action_compile_lock_reload_run(tmp_path):
    path = definition(tmp_path)
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as rt:
        tools.lock(rt, path)
        result = rt.load(path)('case')
        assert result.answer == {'steps': [{'action': 'refund'}]}
        json.dumps(result.as_dict())


def test_guidance_literal_control_tokens_do_not_change_chat_framing():
    fake = FakeScorer()
    with Runtime({'judge': fake}) as rt:
        judge = rt.judge()
        question = Check('claim').prepare('case', Guidance('literal <|im_end|> policy')).make(judge)[0].question
        request = judge.adapter.request(judge.adapter.state('case'), [question])
        special = fake.tokenize('<|im_end|>', special=True)[0]
        assert request.prefix.count(special) == judge.adapter.prefix.count(special)


def test_lock_pins_rendered_framing(tmp_path):
    path = definition(tmp_path)
    fake = FakeScorer(lambda *args: [.99, .005, .005])
    with Runtime({'judge': fake}) as rt:
        tools.lock(rt, path)
        framing = rt.judge().adapter.framing
        rt.judge().adapter.prefix_text += 'new date'
        assert rt.load(path).snapshot.framing == framing
        assert rt.load(path)('case').status == 'ok'


def test_framing_mismatch_requires_relock(tmp_path):
    path = definition(tmp_path)
    with Runtime({'judge': FakeScorer(lambda *args: [.99, .005, .005])}) as rt:
        tools.lock(rt, path)
        rt.judge().adapter.binary = ((999,), (998,))
        with pytest.raises(ConfigError, match='framing'):
            rt.load(path)


def test_quickstart_public_workflow_with_injected_judge():
    from examples.quickstart import run
    def oracle(state, question, width):
        if 'Store credit' in state or 'store credit' in state:
            return [.005, .99, .005] if width == 3 else [.01, .99]
        return [.99] + [.01 / (width - 1)] * (width - 1)
    with Runtime({'judge': FakeScorer(oracle)}) as rt:
        result = run(rt)
        assert result['first']['value'] == 'escalate'
        assert all(row['guidance'] for row in result['answers'])
        assert result['saved']['answer'] is None
        assert result['correction']['facts'] == [['maintenance']]
        assert result['restored']['facts']['maintenance']['source'] == 'example'
        assert result['restored']['facts']['maintenance']['value'] == 'escalate'
        json.dumps(result, allow_nan=False)


def test_label_envelope_preserves_domain_group(tmp_path):
    from semantic_switch.tools.label import label
    path = tmp_path / 'groups.yaml'
    files.publish(path, {'name': 'groups', 'fields': {'group': {'type': 'text'}, 'message': {'type': 'text'}},
                           'reads': ['group', 'message'], 'facts': {'a': {'check': 'claim'}}})
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'schema': 2, 'input': {'group': 'domain', 'message': 'case'}}) + '\n')
    with Runtime({'judge': FakeScorer()}) as rt:
        label(rt, path, str(source), ask=lambda _: 'y', show=lambda _: None)
    record = files.read_jsonl(path.with_suffix('.examples.jsonl'))[0]
    assert record['input']['group'] == 'domain' and 'group' not in record


def test_pair_family_seeds_survive_multiple_corrections(tmp_path):
    from semantic_switch.tools.data import Labels
    path = definition(tmp_path)
    with Runtime({'judge': FakeScorer()}) as rt:
        sw = rt.load(path)
        sw.correct('a', fact='refund', value=True, pair='first')
        sw.correct('a', fact='refund', value=True, pair='second')
        sw.correct('b', fact='refund', value=True, pair='second')
        labels = Labels(rt.load(path))
        examples = labels.examples('refund')
        assert len({e.group for e in examples}) == 1
        assert {('pair', 'first'), ('pair', 'second')} <= set(map(tuple, labels.seeds(examples[0].group)))
        tools.lock(rt, path)
        assert set(rt.load(path).snapshot.examples['refund'][0].pairs) == {'first', 'second'}


def test_runtime_close_waits_for_active_borrowed_judge():
    import threading
    import numpy as np
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Judge:
        identity = {'source': 'test'}
        accelerated = False
        max_slots = 20
        batch_inputs = 1
        def fit(self, text, limit):
            return text, False
        def score_many(self, items):
            started.set()
            release.wait(5)
            return [Scored((Scores(np.array([.99, .01]), None),), 0)]
    rt = Runtime({'judge': Judge()})
    worker = threading.Thread(target=lambda: rt.check('case', 'claim'))
    worker.start()
    assert started.wait(2)
    closer = threading.Thread(target=lambda: (rt.close(), closed.set()))
    closer.start()
    assert not closed.wait(.02)
    release.set()
    worker.join(2)
    closer.join(2)
    assert closed.is_set()
    with pytest.raises(RuntimeError, match='closed'):
        rt.check('case', 'claim')


@pytest.mark.slow
def test_guidance_native_prefix_residency_preserves_scores():
    real = Runtime()
    import numpy as np
    policies = ['Store credit is a refund.', 'Policy: refund means cash returned to the original payment method. Store credit is not a refund. A request for store credit is not a refund request.', 'Refunds require cash back.']
    statements = {'refund': Check('The customer requests a refund.', unstated=True, min_score=.8)}
    outcomes = [real.ask('Please give me store credit.', statements, guidance=p)['refund'] for p in policies]
    before = real.info()['capabilities']['counters']['prefix_misses']
    repeated = [real.ask('Please give me store credit.', statements, guidance=p)['refund'] for p in policies]
    after = real.info()['capabilities']['counters']['prefix_misses']
    assert after == before
    for a, b in zip(outcomes, repeated):
        assert a.status == b.status and a.value == b.value
        assert np.isfinite([a.p_true, a.p_unstated, b.p_true, b.p_unstated]).all()
        assert (a.p_true >= .8) == (b.p_true >= .8)
        assert (a.p_unstated >= .8) == (b.p_unstated >= .8)
    assert outcomes[0].status == 'ok' and outcomes[0].value is True
    assert outcomes[1].status == 'ok' and outcomes[1].value is False
    real.close()


def test_cli_guidance_flag_matches_sdk(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from typer.testing import CliRunner
    from semantic_switch import cli
    source = tmp_path / 'policy.txt'
    source.write_text('Store credit is not a refund.')
    fake = FakeScorer(lambda state, question, width: [.01, .99])
    rt = Runtime({'judge': fake})
    @contextmanager
    def session():
        yield rt
    monkeypatch.setattr(cli, 'session', session)
    runner = CliRunner()
    try:
        expected = rt.check('store credit', 'Refund requested.', guidance=Guidance.load(source))
        result = runner.invoke(cli.app, ['check', 'Refund requested.', '--guidance', str(source)], input='store credit')
        assert result.exit_code == 1, result.output
        actual = json.loads(result.stdout)
        assert actual['value'] == expected.value and actual['guidance'] == expected.as_dict()['guidance']
        questions = tmp_path / 'questions.yaml'
        files.publish(questions, {'refund': {'check': 'Refund requested.'}})
        result = runner.invoke(cli.app, ['ask', str(questions), '--guidance', str(source)], input='store credit')
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)['refund']['guidance'] == expected.as_dict()['guidance']
    finally:
        rt.close()


def test_lock_candidate_digest_binds_actual_framing(tmp_path):
    path = definition(tmp_path)
    with Runtime({'judge': FakeScorer()}) as rt:
        tools.lock(rt, path)
        lock_path = path.with_suffix('.lock.json')
        data = files.read_json(lock_path)
        from semantic_switch.artifacts import candidate_digest
        prior = candidate_digest(data)
        data['framing']['prefix_text'] += 'different rules'
        assert candidate_digest(data) != prior


def test_guidance_quality_advisory_profile_and_capacity_separate():
    from types import SimpleNamespace
    from semantic_switch.api import behavior_identity
    from semantic_switch.guidance import REVISION
    from semantic_switch.judge import LlamaJudge
    import warnings
    scorer = FakeScorer()
    with Runtime({'judge': scorer}) as rt:
        adapter = rt.judge().adapter
        identity = rt.judge().identity
        profile = {'schema': 1, 'guidance_revision': REVISION, 'judge': behavior_identity(identity),
                   'configuration': {'max_input_tokens': 1024, 'n_ctx': 10000, 'n_batch': 512, 'prefix_slots': 4},
                   'max_guidance_tokens': 1, 'cohort': 'synthetic-contrast', 'evidence': 'measured.json#sha256'}
        native = SimpleNamespace(ctx=SimpleNamespace(n_ctx=10000, n_batch=512), prefix_slots=4)
        judge = LlamaJudge(native, adapter, identity, quality_profile=profile)
        binding = Guidance('many words beyond the measured range')
        with pytest.warns(UserWarning, match='unmeasured'):
            judge._advise(binding)
        with warnings.catch_warnings(record=True) as caught:
            judge._advise(binding)
        assert not caught
        profile['configuration']['n_ctx'] = 9999
        judge._advised.clear()
        with warnings.catch_warnings(record=True) as caught:
            judge._advise(binding)
        assert not caught


def test_hosted_mixed_version_warning_is_operation_local():
    def transport(body):
        return {'model': body['state'], 'answers': {'q0': {'noul': .99}}}
    with Runtime({'judge': structured(transport)}) as rt:
        with pytest.warns(UserWarning, match='mixed model versions'):
            results = rt.ask_many(['one', 'two'], {'claim': Check('Claim.')})
        assert [row['claim'].observed_models for row in results] == [('one',), ('two',)]


def test_guidance_case_label_staleness_follows_decision_dependencies(tmp_path):
    from semantic_switch.tools.common import test_items
    path = definition(tmp_path, {'terms': {'text': 'Store credit is a refund.'}})
    with Runtime({'judge': FakeScorer()}) as rt:
        sw = rt.load(path)
        sw.correct('case', case='refund')
        prior = sw.plan.switch_rev
        spec = files.read_yaml(path)
        spec['guidance']['terms']['text'] = 'Store credit is not a refund.'
        files.publish(path, spec)
        draft = rt.load(path, draft=True)
        assert draft.plan.switch_rev != prior
        rows, warnings = test_items(draft.paths, draft.plan)
        assert 'case' not in rows[0] and any('stale answer' in w for w in warnings)


def test_card_hidden_claim_guidance_not_interned_across_policies():
    plan = compile_spec({'name': 'cards', 'guidance': {'first': {'text': 'Store credit counts.'},
                                                    'second': {'text': 'Store credit does not count.'}},
                         'facts': {'procedure': {'select': 'Select a procedure.', 'cards': [
                             {'id': 'one', 'title': 'One', 'guidance': ['first'], 'applies_when': ['Refund requested.']},
                             {'id': 'two', 'title': 'Two', 'guidance': ['second'], 'applies_when': ['Refund requested.']}]}}})
    first, second = plan.fact('procedure').cards
    assert first.claims != second.claims
    assert plan.facts[first.claims[0]].judgment.guidance != plan.facts[second.claims[0]].judgment.guidance


@pytest.mark.parametrize('name', ['max_input_tokens', 'prefix_slots', 'n_batch'])
@pytest.mark.parametrize('value', [None, True, 0, -1, 1.5])
def test_input_budget_invalid_before_resource_loading(name, value):
    from semantic_switch import Model
    with pytest.raises(ConfigError, match='positive integer'):
        Model('unresolved:model.gguf', **{name: value})


@pytest.mark.parametrize('settings', [{'n_batch': 32}, {'n_ctx': 32}, {'prefix_slots': 230}, {'prefix_slots': 513}])
def test_native_topology_rejected_before_loading_weights(monkeypatch, settings):
    from semantic_switch import backend
    monkeypatch.setattr(backend, 'Weights', lambda *args, **kw: pytest.fail('invalid topology loaded weights'))
    with pytest.raises(ValueError, match='sequence slots'):
        backend.LlamaScorer('unresolved.gguf', inputs=8, grouping='batched', **settings)


@pytest.mark.parametrize('item,value', [(Check('Refund requested.', unstated=True), True),
                                       (Pick('Choose the action.', {'cash': 'return cash', 'credit': 'store credit'}), 'cash'),
                                       (Rate('Urgency?', ('low', 'high')), 'low')])
def test_guidance_inline_saved_compiled_export_parity(tmp_path, item, value):
    fake = FakeScorer(lambda state, question, width: [.99] + [.01 / (width - 1)] * (width - 1))
    binding = Guidance('Store credit is not a refund. Cash back is a refund.')
    path = tmp_path / 'decision.yaml'
    with Runtime({'judge': fake}) as rt:
        inline = item({'message': 'Return cash.'}, rt, guidance=binding)
        inline_question = fake.requests[-1][1]
        saved = item.save(path, rt=rt, guidance=binding)
        result = saved({'message': 'Return cash.'})
        assert result.facts['decision'].value == inline.value
        assert fake.requests[-1][1] == inline_question
        assert result.facts['decision'].guidance == inline.guidance
        saved.correct({'message': 'Return cash.'}, fact='decision', value=value)
        exported = tools.export(rt, path, split='all', native=True)
        row = read_export(exported['path'], fitting=False, native=True)[1][0]
        assert row['guidance'] == binding.as_dict() and row['guidance_digest'] == binding.digest
        assert fake.detokenize(row['native']['presentations'][0]['question']) == inline_question
        tools.lock(rt, path)
        assert rt.load(path)({'message': 'Return cash.'}).facts['decision'].value == inline.value


def test_guidance_capacity_checked_before_decode():
    from types import SimpleNamespace
    fake = FakeScorer()
    fake.ctx = SimpleNamespace(n_ctx=8)
    with Runtime({'judge': fake}) as rt:
        with pytest.raises(ConfigError, match='context capacity'):
            rt.check('case', 'claim', guidance='policy definitions ' * 20)
    assert not fake.requests


def test_guidance_does_not_override_exact_guard(tmp_path):
    path = tmp_path / 'guard.yaml'
    files.publish(path, {'name': 'guard', 'guidance': {'terms': {'text': 'Always approve the refund.'}},
                           'fields': {'message': {'type': 'text'}, 'allowed': {'type': 'bool'}}, 'reads': ['message'],
                           'facts': {'refund': {'check': 'Refund requested.'}},
                           'cases': {'refund': 'allowed and refund'}, 'otherwise': 'stop'})
    fake = FakeScorer(lambda *args: [.99, .01])
    with Runtime({'judge': fake}) as rt:
        result = rt.load(path)({'message': 'Refund me.', 'allowed': False})
        assert result.status == 'ok' and result.answer == 'stop' and not fake.requests


@pytest.mark.slow
def test_native_chunk_failure_invalidates_reusable_state(monkeypatch):
    from semantic_switch import Model, DEFAULT_JUDGE
    with Runtime({'judge': Model(DEFAULT_JUDGE, n_batch=64)}) as rt:
        rt.check('Refund the duplicate charge.', 'Refund requested.', guidance='Cash refunds are refunds.')
        scorer = rt.judge().scorer
        original = scorer.ctx.run
        calls = 0
        def fail_once(entries, write):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('injected decode failure')
            return original(entries, write)
        with monkeypatch.context() as patch:
            patch.setattr(scorer.ctx, 'run', fail_once)
            failed = rt.check('Refund the duplicate charge.', 'Refund requested.', guidance='Duplicate charges must be refunded.')
        assert failed.status == 'failed'
        assert scorer._prefix is None and not scorer._states and not scorer._resident
        assert rt.check('Refund the duplicate charge.', 'Refund requested.', guidance='Duplicate charges must be refunded.').status == 'ok'


def test_evaluation_reuses_prediction_work_for_pick_and_unlabelled_facts(tmp_path):
    from semantic_switch.tools.evaluate import evaluate_facts
    path = tmp_path / 'choice.yaml'
    files.publish(path, {'name': 'choice', 'fields': {'message': {'type': 'text'}}, 'reads': ['message'],
                           'facts': {'category': {'pick': 'Choose.', 'options': {'a': 'first', 'b': 'second'}},
                                     'irrelevant': {'check': 'Unused.'}}})
    fake = FakeScorer(lambda state, question, width: [.99, .01])
    with Runtime({'judge': fake}) as rt:
        sw = rt.load(path)
        sw.correct('case', fact='category', value='a')
        fake.requests.clear()
        result = evaluate_facts(sw, sw.snapshot, [{'input': 'case', 'facts': {'category': 'a'}}])
        assert result['category']['accepted'] == 1 and not fake.requests


@pytest.mark.parametrize('destination', ['test', 'examples'])
def test_imported_labels_preserve_unknown_acquisition_and_origin(tmp_path, destination):
    from semantic_switch.tools.label import label
    path = definition(tmp_path)
    source = tmp_path / 'labels.jsonl'
    with Runtime({'judge': FakeScorer()}) as rt:
        revision = rt.load(path).plan.fact('refund').meaning_rev
        source.write_text(json.dumps({'schema': 2, 'input': 'case', 'facts': {'refund': True}, 'revs': {'refund': revision},
                                      'label_sources': {'refund': [{'origin': 'unknown', 'producer': 'unknown'}]}}) + '\n')
        prompts = []
        assert label(rt, path, str(source), to=destination, ask=lambda prompt: prompts.append(prompt) or '', show=lambda _: None, reviewer='ana') == 1
        record = files.read_jsonl(path.with_suffix('.test.jsonl' if destination == 'test' else '.examples.jsonl'))[0]
        sources = record['label_sources']['refund'] if destination == 'test' else record['label_sources']
        assert [(s['origin'], [r['outcome'] for r in s['reviews']]) for s in sources] == [('unknown', ['confirm'])] and 'Supplied: True' in prompts[0]
        assert [row['selected_by'] for row in record['selections']] in ([], ['unknown'])


def test_invalid_label_envelope_reports_actual_source_line(tmp_path):
    from semantic_switch.tools.label import label
    source = tmp_path / 'bad.jsonl'
    source.write_text('\n' + json.dumps({'schema': 2, 'input': 'case', 'surprise': True}) + '\n')
    with Runtime({'judge': FakeScorer()}) as rt:
        with pytest.raises(ConfigError, match='bad.jsonl:2:.*unsupported situation fields'):
            label(rt, definition(tmp_path), str(source), show=lambda _: None)


@pytest.mark.parametrize('payload', [{1: 'bad key'}, {'amount': float('inf')}, {'steps': {'set'}}, object()])
def test_action_payload_invalid_at_owning_definition(payload):
    with pytest.raises(ConfigError, match='cases.*then'):
        compile_spec({'name': 'bad', 'facts': {}, 'cases': [{'id': 'bad', 'when': 'true', 'then': payload}]})


def test_guidance_groups_tokenize_state_once_and_keep_question_order():
    with Runtime({'judge': FakeScorer()}) as rt:
        judge = rt.judge()
        terms, other = Guidance('Store credit counts.'), Guidance('Store credit does not count.')
        questions = [Question('check', 'Asks for a refund.', guidance=terms), Question('check', 'Mentions a charge.', guidance=other),
                     Question('pick', 'Topic?', ('billing', 'outage'), guidance=terms)]
        alone = [judge.score('I was charged twice.', [q])[0].probs for q in questions]
        state, calls = judge.adapter.state, []
        judge.adapter.state = lambda text: calls.append(text) or state(text)
        grouped = judge.score('I was charged twice.', questions)
        assert calls == ['I was charged twice.']
        assert all(s.probs.tolist() == a.tolist() for s, a in zip(grouped, alone))


def test_answer_capacity_error_names_effective_limit():
    from semantic_switch.presentation import validate_question
    with Runtime({'judge': FakeScorer()}) as rt:
        judge = rt.judge()
        judge.adapter.max_slots = 3
        with pytest.raises(ConfigError, match='at most 3 answer slots, got 4'):
            validate_question(judge, Question('pick', 'Topic?', ('a', 'b', 'c', 'd')))
        judge.adapter.max_slots = 50
        with pytest.raises(ConfigError, match='at most 20 answer slots, got 21'):
            validate_question(judge, Question('pick', 'Topic?', tuple('abcdefghijklmnopqrstu')))
