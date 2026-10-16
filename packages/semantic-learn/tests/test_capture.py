import re

import pytest

from semantic_switch import Check, ConfigError, files
from semantic_learn.capture import Bounds, Capture, Consent, Redactor, Sampling, captured
from semantic_learn.review import review_export
from semantic_testkit.scoring import FakeScorer

REFUND = 'The customer asks to receive money back for a charge.'
KEY = b'k' * 32
CONSENT = Consent('support-terms', '2026-09')


def definition(tmp_path, statement=REFUND):
    path = tmp_path / 'support.yaml'
    files.publish(path, {'name': 'support', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
                         'facts': {'refund': {'check': statement}}})
    return path


def scrub(data):
    return {key: re.sub(r'\S+@\S+', '[email]', value) for key, value in data.items()} if isinstance(data, dict) else re.sub(r'\S+@\S+', '[email]', data)


def capture(tmp_path, judge, *, probability=1., records=100, size=1 << 20, name='support'):
    return Capture(tmp_path / f'{name}.capture.jsonl', redactor=Redactor('email-v1', scrub), sampling=Sampling(probability, 'seed-1'),
                   bounds=Bounds(records, size), link_key=KEY, judge=judge)


def test_opt_out_and_unwrapped_calls_infer_without_capture(tmp_path):
    judge = FakeScorer()
    with capture(tmp_path, judge) as learn:
        switch = learn.load(definition(tmp_path))
        assert learn.decide(switch, {'message': 'refund me at ana@example.org'}, consent=None).facts['refund'].value is True
        assert learn.runtime.check('please refund me', REFUND).status == 'ok'
        switch({'message': 'a direct call, never captured'})
        disclosed = learn.disclosure()
    assert disclosed['opted_out'] == 1 and disclosed['unwrapped'] == 2 and disclosed['captured'] == 0 and disclosed['records'] == 0
    assert list(captured(tmp_path / 'support.capture.jsonl')) == [] and 'ana@' not in judge.requests[0][0]
    assert files.read_published(tmp_path / 'support.capture.json')['disclosure']['opted_out'] == 1


def test_redaction_runs_before_any_request_and_capture_stores_the_same_redacted_input(tmp_path):
    judge = FakeScorer()
    with capture(tmp_path, judge) as learn:
        switch = learn.load(definition(tmp_path))
        result = learn.decide(switch, {'message': 'please refund ana@example.org'}, consent=CONSENT, correlation={'thread': 'crm:T-1', 'entity': 'crm:C-7'})
        learn.ask('refund to bob@example.org', {'c': Check(REFUND)}, consent=CONSENT)
        failing = Capture(tmp_path / 'other.capture.jsonl', redactor=Redactor('broken', lambda data: 1 / 0), sampling=Sampling(1., 's'),
                          bounds=Bounds(10, 1 << 20), link_key=KEY, judge=judge)
        before = len(judge.requests)
        with pytest.raises(ConfigError, match='failed before inference.*nothing was sent'):
            failing.decide(failing.load(definition(tmp_path)), {'message': 'refund ana@example.org'}, consent=CONSENT)
        assert len(judge.requests) == before
        failing.close()
    assert result.facts['refund'].value is True and all('@' not in state for state, _ in judge.requests)
    switched, inline = list(captured(tmp_path / 'support.capture.jsonl'))
    assert switched['situation']['input'] == {'message': 'please refund [email]'} and inline['input'] == 'refund to [email]'
    assert switched['redactor'] == 'email-v1' and switched['consent'] == CONSENT.record() and switched['decision']['facts']['refund']['prediction']['value'] is True
    assert switched['task']['targets']['refund']['meaning_rev'] and switched['predictor']['judge'] and switched['event'] and switched['call']
    assert switched['sampling'] == {'reason': 'random', 'probability': 1.} and switched['situation']['selections'][0]['inclusion_probability'] == 1.
    assert 'facts' not in switched['situation'] and 'label_sources' not in switched['situation']
    raw = (tmp_path / 'support.capture.jsonl').read_text()
    assert 'example.org' not in raw and 'T-1' not in raw and 'C-7' not in raw


def test_bounded_log_drops_with_disclosure_and_never_blocks_inference(tmp_path):
    with capture(tmp_path, FakeScorer(), records=2) as learn:
        switch = learn.load(definition(tmp_path))
        results = [learn.decide(switch, {'message': f'refund {i}'}, consent=CONSENT) for i in range(4)]
        disclosed = learn.disclosure()
    assert all(result.facts['refund'].value is True for result in results)
    assert disclosed['captured'] == 2 and disclosed['dropped'] == {'records': 2, 'bytes': 0, 'encoding': 0, 'storage': 0} and len(list(captured(tmp_path / 'support.capture.jsonl'))) == 2
    with capture(tmp_path, FakeScorer(), size=900, name='small') as learn:
        switch = learn.load(definition(tmp_path))
        learn.decide(switch, {'message': 'refund please'}, consent=CONSENT)
        assert learn.disclosure()['dropped']['bytes'] == 1 and (tmp_path / 'small.capture.jsonl').stat().st_size <= 900
    with pytest.raises(ConfigError, match='positive record and byte'):
        Bounds(0, 10)


def test_correlation_links_are_scoped_hashed_and_missing_links_disclosed(tmp_path):
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path))
        for message, correlation in (('refund a', {'thread': 'crm:T-1'}), ('refund b', {'thread': 'crm:T-1'}), ('refund c', {'thread': 'mail:T-1'}),
                                     ('refund d', None)):
            learn.decide(switch, {'message': message}, consent=CONSENT, correlation=correlation)
        with pytest.raises(ConfigError, match='source namespace'):
            learn.decide(switch, {'message': 'refund e'}, consent=CONSENT, correlation={'thread': 'T-1'})
    a, b, c, d = list(captured(tmp_path / 'support.capture.jsonl'))
    assert a['links'] == b['links'] and a['links']['thread']['id'] != c['links']['thread']['id'] and a['situation']['groups'] == b['situation']['groups']
    assert a['missing_links'] == ['entity'] and d['missing_links'] == ['thread', 'entity'] and a['independence'].startswith('thread and entity links are missing')
    exported = review_export(definition(tmp_path), tmp_path / 'queue.review.json', size=4, audit=4, captures=[tmp_path / 'support.capture.jsonl'])
    assert exported['population']['rows'] == 4 and exported['population']['families'] == 3


def test_sampling_is_a_deterministic_replayable_decision(tmp_path):
    messages = [f'refund number {i}' for i in range(40)]
    runs = []
    for name in ('first', 'second'):
        with capture(tmp_path, FakeScorer(), probability=.5, name=name) as learn:
            switch = learn.load(definition(tmp_path))
            for message in messages:
                learn.decide(switch, {'message': message}, consent=CONSENT)
            learn.flush()
            runs.append((learn.disclosure(), [record['situation']['input']['message'] for record in captured(tmp_path / f'{name}.capture.jsonl')]))
    (first, chosen), (second, again) = runs
    assert chosen == again and 0 < len(chosen) < 40 and first['not_sampled'] == second['not_sampled'] == 40 - len(chosen)
    keys = [f'key-{i}' for i in range(200)]
    assert [Sampling(.5, 'seed-1').chosen(key) for key in keys] == [Sampling(.5, 'seed-1').chosen(key) for key in keys]
    assert [Sampling(.5, 'seed-1').chosen(key) for key in keys] != [Sampling(.5, 'seed-2').chosen(key) for key in keys]
    assert all(Sampling(1, 'seed-1').chosen(key) for key in keys) and set(Sampling(.25, 's').chosen(key) for key in keys) == {True, False}


def test_weak_feedback_orders_review_but_never_becomes_a_label(tmp_path):
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path))
        for i in range(6):
            learn.decide(switch, {'message': f'refund case {i}'}, consent=CONSENT)
        learn.flush()
        events = [record['event'] for record in captured(tmp_path / 'support.capture.jsonl')]
        learn.feedback(events[3], 'refund', False, source='agent:router')
    exported = review_export(definition(tmp_path), tmp_path / 'queue.review.json', size=1, audit=0, captures=[tmp_path / 'support.capture.jsonl'],
                             strata=('feedback',))
    (row,) = files.read_jsonl(exported['queue'])
    assert row['input'] == {'message': 'refund case 3'} and 'facts' not in row and row['selections'][-1]['selected_by'] == 'targeted'
    assert exported['strata']['feedback']['rows'] == 1 and exported['review'][:2] == ['ss', 'label']


def test_stale_meanings_are_excluded_from_review(tmp_path):
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path))
        learn.decide(switch, {'message': 'refund old'}, consent=CONSENT)
    changed = tmp_path / 'changed'
    changed.mkdir()
    with pytest.raises(ConfigError, match='no current situations'):
        review_export(definition(changed, 'The customer asks for a refund of a duplicate charge.'), tmp_path / 'stale.review.json', size=1, audit=1,
                      captures=[tmp_path / 'support.capture.jsonl'])


def repeatable(redactor, samples):
    return all(redactor(sample) == redactor(sample) for sample in samples)


def test_redactor_conformance_checks_repeatability_outside_the_capture_path():
    unstable = iter(range(100))
    assert repeatable(Redactor('email-v1', scrub), ['refund ana@example.org', {'message': 'mail bob@example.org'}])
    assert not repeatable(Redactor('drift', lambda data: f'{data}{next(unstable)}'), ['refund ana@example.org'])


def test_redactor_runs_once_on_an_owned_snapshot_and_returns_owned_containers(tmp_path):
    shared, calls = {}, []
    def leaky(data):
        calls.append(1)
        shared['message'] = scrub(data['message'])
        return shared
    def mutating(data):
        data['message'] = scrub(data['message']) + '!'
        data.setdefault('seen', []).append(1)
        return data
    caller = {'message': 'refund ana@example.org', 'meta': {'tags': ['x']}}
    original = {'message': 'refund ana@example.org', 'meta': {'tags': ['x']}}
    accepted = Redactor('leaky', leaky)(caller)
    shared['message'] = 'ana@example.org leaked after validation'
    assert accepted == {'message': 'refund [email]'} and accepted is not shared and len(calls) == 1
    assert Redactor('mutating', mutating)(caller) == {'message': 'refund [email]!', 'meta': {'tags': ['x']}, 'seen': [1]}
    pure = lambda data: {**data, 'message': scrub(data['message'])}
    assert Redactor('pure', pure)(caller) == {'message': 'refund [email]', 'meta': {'tags': ['x']}} and caller == original
    for name, apply, data, match in (('nonfinite', lambda data: {'message': float('inf')}, caller, 'returned not supported finite JSON evidence'),
                                     ('tuple-keys', lambda data: {1: 'x'}, caller, 'returned not supported finite JSON evidence'),
                                     ('listed', lambda data: [data], caller, 'returned list, not text or a mapping'),
                                     ('pure', pure, {'message': float('nan')}, 'not supported finite JSON evidence'),
                                     ('raising', lambda data: data['missing'], caller, 'failed before inference \\(KeyError')):
        with pytest.raises(ConfigError, match=match):
            Redactor(name, apply)(data)
    assert caller == original
    judge, seen = FakeScorer(), []
    def counted(data):
        seen.append(1)
        return pure(data)
    with Capture(tmp_path / 'once.capture.jsonl', redactor=Redactor('counted', counted), sampling=Sampling(1., 's'), bounds=Bounds(10, 1 << 20),
                 link_key=KEY, judge=judge) as learn:
        learn.decide(learn.load(definition(tmp_path)), caller, consent=CONSENT)
        with pytest.raises(ConfigError, match='nothing was sent'):
            learn.decide(learn.load(definition(tmp_path)), {'message': float('nan')}, consent=CONSENT)
    assert len(seen) == 1 and len(judge.requests) == 1 and caller == original


def test_log_buffers_flushes_reopens_and_recovers_a_torn_tail(tmp_path, monkeypatch):
    from semantic_learn import capture as module
    appends = []
    real = files.append
    monkeypatch.setattr(module.files, 'append', lambda path, data: (appends.append(len(data)), real(path, data))[1])
    path = tmp_path / 'support.capture.jsonl'
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path))
        for i in range(5):
            learn.decide(switch, {'message': f'refund {i}'}, consent=CONSENT)
        buffered = learn.disclosure()
        assert buffered['buffered'] == 5 and buffered['persisted'] == 0 and len(appends) == 1
    assert len(appends) == 2 and len(list(captured(path))) == 5
    closed = files.read_published(tmp_path / 'support.capture.json')['disclosure']
    assert closed['persisted'] == 5 and closed['buffered'] == 0 and closed['records'] == 5 and closed['bytes'] == path.stat().st_size
    with path.open('ab') as fh:
        fh.write(b'{"kind": "captured-dec')
    with capture(tmp_path, FakeScorer()) as learn:
        reopened = learn.disclosure()['reopened']
        assert reopened == {'records': 5, 'bytes': path.stat().st_size, 'torn': True}
        learn.decide(learn.load(definition(tmp_path)), {'message': 'refund again'}, consent=CONSENT)
    assert len(list(captured(path))) == 6 and files.read_published(tmp_path / 'support.capture.json')['disclosure']['bytes'] == path.stat().st_size
    with pytest.raises(ConfigError, match='another capture configuration'):
        Capture(path, redactor=Redactor('other', scrub), sampling=Sampling(1., 'seed-1'), bounds=Bounds(100, 1 << 20), link_key=KEY, judge=FakeScorer())


def test_large_records_flush_pending_first_and_buffers_flush_at_the_threshold(tmp_path, monkeypatch):
    from semantic_learn import capture as module
    appends = []
    real = files.append
    monkeypatch.setattr(module.files, 'append', lambda path, data: (appends.append(data), real(path, data))[1])
    with capture(tmp_path, FakeScorer(), records=1000, size=8 << 20) as learn:
        switch = learn.load(definition(tmp_path))
        learn.decide(switch, {'message': 'refund small'}, consent=CONSENT)
        learn.feedback('event', 'refund', 'x' * (module.FLUSH + 1), source='agent:router')
        assert [data.count(b'\n') for data in appends[1:]] == [1, 1] and len(appends[2]) > module.FLUSH
        while len(appends) == 3:
            learn.feedback('event', 'refund', 'y' * 4096, source='agent:router')
        assert len(appends[3]) >= module.FLUSH and learn.disclosure()['buffered'] == 0
    assert len(list(captured(tmp_path / 'support.capture.jsonl'))) == 2 + appends[3].count(b'\n')


def test_storage_failure_preserves_the_decision_and_reconciles_drops(tmp_path, monkeypatch):
    from semantic_learn import capture as module
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path))
        learn.decide(switch, {'message': 'refund kept'}, consent=CONSENT)
        learn.flush()
        learn.decide(switch, {'message': 'refund lost'}, consent=CONSENT)
        def broken(path, data):
            raise OSError('disk full')
        monkeypatch.setattr(module.files, 'append', broken)
        with pytest.warns(RuntimeWarning, match='storage failed'):
            learn.flush()
        result = learn.decide(switch, {'message': 'refund later'}, consent=CONSENT)
        with pytest.warns(RuntimeWarning, match='storage failed'):
            learn.close()
        disclosed = learn.disclosure()
    assert result.facts['refund'].value is True
    assert disclosed['captured'] == 3 == disclosed['persisted'] + disclosed['buffered'] + disclosed['dropped']['storage'] and disclosed['dropped']['storage'] == 2
    assert [record['situation']['input']['message'] for record in captured(tmp_path / 'support.capture.jsonl')] == ['refund kept']


def test_task_is_cached_per_switch_snapshot_and_refreshed_on_change(tmp_path):
    with capture(tmp_path, FakeScorer()) as learn:
        switch = learn.load(definition(tmp_path), draft=True)
        first = learn.task(switch)
        learn.decide(switch, {'message': 'refund one'}, consent=CONSENT)
        assert learn.task(switch) is first
        definition(tmp_path, 'The customer asks for a refund of a duplicate charge.')
        switch.reload()
        assert learn.task(switch) is not first and learn.task(switch).targets()['refund']['meaning_rev'] != first.targets()['refund']['meaning_rev']


def test_close_drains_storage_and_concurrent_closers(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered, release = threading.Event(), threading.Event()
    learn = capture(tmp_path, FakeScorer())
    switch = learn.load(definition(tmp_path))
    store = learn._store

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return store(*args, **kwargs)
    monkeypatch.setattr(learn, '_store', delayed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        call = pool.submit(learn.decide, switch, {'message': 'refund me'}, consent=CONSENT)
        assert entered.wait(5)
        closing = [pool.submit(learn.close) for _ in range(2)]
        assert all(not future.done() for future in closing)
        release.set()
        assert call.result(5).status == 'ok'
        summaries = [future.result(5) for future in closing]
    assert summaries[0] == summaries[1] and summaries[0]['persisted'] == 1


def test_complete_unterminated_header_is_not_duplicated(tmp_path):
    learn = capture(tmp_path, FakeScorer())
    path = learn.path
    learn.close()
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    with capture(tmp_path, FakeScorer()) as reopened:
        assert reopened.log.reopened['records'] == 0
    assert len(path.read_bytes().splitlines()) == 1


def test_capture_pins_task_and_inference_across_reload(tmp_path, monkeypatch):
    fake = FakeScorer()
    with capture(tmp_path, fake) as learn:
        path = definition(tmp_path)
        switch = learn.load(path)
        run = switch.run

        def reloaded(inputs, snapshot, **options):
            definition_ = files.read_yaml(path)
            definition_['fields']['priority'] = {'type': 'text', 'required': True}
            definition_['reads'].append('priority')
            files.publish(path, definition_)
            switch.reload()
            return run(inputs, snapshot, **options)
        monkeypatch.setattr(switch, 'run', reloaded)
        result = learn.decide(switch, {'message': 'refund me', 'priority': 'urgent'}, consent=CONSENT)
        assert result.status == 'ok' and 'priority' not in fake.requests[-1][0]
    record = next(captured(learn.path))
    assert record['situation']['input'] == {'message': 'refund me'}


def test_capture_rejects_close_from_its_own_redactor(tmp_path):
    learn = Capture(tmp_path / 'self.capture.jsonl', redactor=Redactor('self', lambda data: learn.close()), sampling=Sampling(1., 'seed'),
                    bounds=Bounds(10, 1 << 20), link_key=KEY, judge=FakeScorer())
    try:
        with pytest.raises(ConfigError, match='cannot close inside an admitted call'):
            learn.ask('refund me', {}, consent=CONSENT)
    finally:
        learn.close()
