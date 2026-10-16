import numpy as np
import pytest

from semantic_switch import Check, Model, Runtime
from semantic_switch.context import OwnedJSON, OwnedMapping, JSONArray
from semantic_switch.executor import Reuse
from semantic_switch.journal import Journal
from semantic_switch.judge import Request, Scored, Scores
from semantic_switch.metrics import bootstrap_totals
from semantic_switch.protocols import Tokens
from semantic_testkit.scoring import FakeScorer


def test_packed_tokens_own_one_immutable_payload():
    raw = np.arange(2048, dtype=np.int32)
    tokens = Tokens.from_array(raw)
    raw[:] = 0
    assert tokens.nbytes == 8192 and list(tokens[:3]) == [0, 1, 2]
    assert tokens[-1] == 2047 and Tokens(tokens) is tokens
    assert hash(tokens) == hash(Tokens(range(2048)))
    assert isinstance(tokens.array.base, bytes) and not tokens.array.flags.writeable
    with pytest.raises(ValueError):
        tokens.array.setflags(write=True)
    with pytest.raises(AttributeError):
        tokens._data = b''


def test_all_owned_container_constructors_take_nested_ownership():
    for constructor in (OwnedJSON, OwnedMapping):
        raw = {'x': [{'n': 1}]}
        owned = constructor(raw)
        raw['x'][0]['n'] = 2
        assert owned['x'][0]['n'] == 1
    raw = {'n': 1}
    owned = JSONArray([raw])
    raw['n'] = 2
    assert owned[0]['n'] == 1


def test_reuse_owns_probability_and_receipt_buffers():
    vector, usage = np.array([.95, .05]), {'nested': {'tokens': 3}}
    request = Request('one', 1., 1, 200, usage)
    reuse = Reuse()
    reuse.put('key', Scored((Scores(vector, 1., model='custom', request=request),), 10))
    vector[:] = [.05, .95]
    usage['nested']['tokens'] = 90
    saved = reuse.get('key')
    assert saved.scores[0].probs == (.95, .05)
    assert saved.scores[0].request.usage['nested']['tokens'] == 3
    reuse.clear()
    assert reuse.get('key') is None


def test_same_pending_prepared_request_is_scored_once(tmp_path):
    fake = FakeScorer()
    fake.batch_inputs = 8
    with Runtime(fake) as rt:
        switch = Check('The customer asks for money back.').save(tmp_path / 'refund.yaml', rt=rt)
        rows = switch.score(['refund one'] * 8, ['refund'], sources=('judge',), reuse=Reuse())
    assert len(rows) == 8 and len(fake.requests) == 1
    assert all(row.targets['refund'].dist == rows[0].targets['refund'].dist for row in rows)


def test_windowed_primitive_stream_batches_and_stays_bounded():
    fake, admitted = FakeScorer(), []
    fake.batch_inputs = 8
    def inputs():
        for index in range(20):
            admitted.append(index)
            yield f'refund {index}'
    with Runtime(fake) as rt:
        stream = Check('The customer asks for money back.').many(inputs(), rt=rt, window=4)
        next(stream)
        assert len(admitted) == 4 and len(fake.requests) == 4
        stream.close()
        assert len(admitted) == 4


def test_journal_keeps_offsets_not_payloads_and_repairs_separator(tmp_path):
    from semantic_switch.compiler import digest
    path, first, second = tmp_path / 'journal.jsonl', digest('one'), digest('two')
    with Journal(path, {}) as journal:
        journal.commit([(first, {'large': 'x' * 65536})])
        assert 'payload' not in journal.current[first]
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    with Journal(path, {}) as journal:
        journal.commit([(second, {'value': 2})])
        assert journal.payload(first) == {'large': 'x' * 65536}
        assert journal.payload(second) == {'value': 2}
        assert len(journal.current) == 2


def test_bootstrap_reduces_the_same_draws_exactly():
    numerators, denominators = np.array([1, 8, 0, 4]), np.array([3, 9, 2, 4])
    draws = np.random.Generator(np.random.PCG64(9)).integers(4, size=(31, 4))
    top, bottom = bootstrap_totals(numerators, denominators, draws)
    assert top.tolist() == [sum(int(numerators[i]) for i in row) for row in draws]
    assert bottom.tolist() == [sum(int(denominators[i]) for i in row) for row in draws]


def test_default_capacity_tracks_limits_not_prefix_slot_count():
    small, large = Model('model.gguf', prefix_slots=1), Model('model.gguf', prefix_slots=4)
    assert small.n_ctx == large.n_ctx == 3584
    assert large.capacity().total_ctx == 21504
    assert small.capacity().n_ctx_seq == large.capacity().n_ctx_seq == 3584
    assert Model('model.gguf', max_prefix_tokens=256).n_ctx == 2816


def test_stage_import_closure_excludes_controller_and_includes_freeze_helpers():
    from semantic_trainer.pipeline import build, implementation
    built = build()
    freeze = implementation('freeze', built)
    assert 'jobs.py' in freeze['build']['semantic-trainer']['files']
    assert 'guidance.py' in freeze['build']['semantic-switch']['files']
    for stage in ('freeze', 'assets', 'profile', 'fit', 'select'):
        selected = implementation(stage, built)
        assert all(not {'cli.py', 'pipeline.py'} & record.get('files', {}).keys() for record in selected['build'].values())
    before = implementation('fit', built)
    built['native'] = {'libraries': {'different.so': 'changed'}}
    assert implementation('fit', built) != before


def test_unicode_mentions_match_independent_literal_reference():
    import re
    import unicodedata
    from semantic_switch.kernel import BEFORE, AFTER, Mentions
    aliases = ('i', 'İ', 'ı', 'ſ', 'K', 'ß', 'ss', 'Σ', 'ς', 'a+b', '++', 'C++', 'credit', 'credit card', 'card')
    engine = Mentions(aliases)
    for text in ('I İ ı i', 'ſ s', 'K K k', 'ß ẞ ss', 'Σ σ ς', 'a+b C++ ++++', 'CREDIT CARD', 'x.card', 'credit-card'):
        normalized = unicodedata.normalize('NFC', text)
        expected = {alias for alias in aliases if re.search(BEFORE + re.escape(alias) + AFTER, normalized, re.IGNORECASE)}
        assert engine.found(normalized) == expected


def test_benchmark_fingerprints_are_owned_and_fresh_between_runs(tmp_path):
    from bench.outcomes import fingerprint
    root = tmp_path / 'repo'
    (root / 'bench').mkdir(parents=True)
    path = root / 'bench' / 'run.py'
    path.write_text('one')
    first = fingerprint(root)
    with pytest.raises(TypeError):
        first['sources'] = 'bad'
    path.write_text('two')
    assert fingerprint(root) != first


def test_native_conformance_cannot_change_production_execution_mode():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from semantic_switch.backend import LlamaScorer
    scorer = object.__new__(LlamaScorer)
    scorer.ctx, scorer.shared = SimpleNamespace(n_batch=64), True
    scorer.operation = nullcontext
    scorer.invalidate = lambda: None
    def diagnostic(*args):
        scorer.shared, scorer.ctx.n_batch = False, 1
        return {'diagnostic': {'ok': False}}
    scorer._selftest = diagnostic
    assert scorer.selftest(None, (), ()) == {'diagnostic': {'ok': False}}
    assert scorer.shared is True and scorer.ctx.n_batch == 64


@pytest.mark.parametrize('changed', ['probability', 'mass'])
def test_public_conformance_rejects_small_probability_or_mass_drift(changed):
    from types import SimpleNamespace
    from semantic_switch.judge import LlamaJudge
    judge = object.__new__(LlamaJudge)
    judge.scorer = SimpleNamespace(selftest=lambda *args: {})
    judge.adapter = SimpleNamespace(letters=((1,), (2,)), binary=((3,), (4,)), polar=((5,), (6,)))
    calls = 0
    def score(text, questions):
        nonlocal calls
        calls += 1
        drift = calls == 3
        return [Scores(np.array([.91, .09]) + (np.array([1e-5, -1e-5]) if drift and changed == 'probability' else 0),
                       .9 + (1e-5 if drift and changed == 'mass' else 0)) for _ in questions]
    judge.score = score
    checks = judge.selftest()
    assert checks['D2 batching']['ok'] is False
    assert checks['D2 batching']['includes_mass'] is True
