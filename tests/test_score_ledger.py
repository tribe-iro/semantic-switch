import numpy as np
import pytest

from semantic_switch import ConfigError, Runtime
from semantic_switch.protocols import ScoreBlock, ScoreLedger, ScoreRequest
from semantic_switch.training import Selection
from semantic_testkit.scoring import FakeScorer
from tests.test_execution_identity import REALIZED


class Counting:
    def __init__(self, failing=()):
        self.calls, self.failing, self.invalidated = [], set(failing), 0

    def score(self, requests, **options):
        self.calls.append((list(requests), options))
        width = max(len(slots) for request in requests for slots in request.slots)
        block = ScoreBlock.empty(sum(len(request.suffixes) for request in requests), width)
        row = 0
        for request in requests:
            for suffix, slots in zip(request.suffixes, request.slots):
                if tuple(suffix) in self.failing:
                    block.error[row] = 'failed'
                else:
                    block.logp[row, :len(slots)] = np.log(np.full(len(slots), 1 / len(slots))) + suffix[0] * 1e-3
                    block.log_mass[row] = -suffix[0] * 1e-2
                row += 1
        return block

    def invalidate(self):
        self.invalidated += 1


class Native(FakeScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls, self.realized = [], dict(REALIZED)

    def score(self, requests, *, reference=False):
        self.calls.append((list(requests), reference))
        return super().score(requests)

    def capacity(self):
        return dict(self.realized)


def request(question, groups=((1,), (2,)), state=(7,)):
    return ScoreRequest((3,), state, (question,), (groups,))


def ledger(scorer, population, binding='a'):
    return ScoreLedger(scorer, lambda: binding, population)


def test_replay_reuses_only_identical_requests_and_charges_the_rest():
    scorer = Counting()
    exported = [request((10,)), request((11,)), request((10,), groups=((2,), (1,))), request((10,), state=(8,)), request((12,), groups=((1,), (2,), (4,)))]
    recording = ledger(scorer, exported)
    public = [request((10,)), ScoreRequest((3,), (7,), ((11,), (12,)), (((1,), (2,)), ((1,), (2,), (4,))))]
    recorded = recording.score(public)
    assert list(recording.blocks) == [('a', request((10,)))]
    block, reuse = recording.replay(exported, scorer.score)
    assert reuse == {'requests': 5, 'reused': 1, 'scored': 4, 'rows': 5, 'scored_rows': 4}
    assert scorer.calls[-1][0] == exported[1:]
    assert np.array_equal(block.logp[0, :2], recorded.logp[0, :2]) and block.log_mass[0] == recorded.log_mass[0]
    assert block.logp[0, :2].tolist() == pytest.approx([np.log(.5) + .01] * 2) and block.log_mass[0] == pytest.approx(-.1)
    fresh = Counting().score(exported)
    assert np.array_equal(block.logp, fresh.logp) and np.array_equal(block.log_mass, fresh.log_mass) and block.error == fresh.error
    assert block.logp.shape == (5, 3) and np.isneginf(block.logp[0, 2])


def test_captured_arrays_are_copies_that_survive_scratch_reuse():
    class Scratch(Counting):
        def score(self, requests, **options):
            self.block = super().score(requests, **options)
            return self.block
    scorer = Scratch()
    recording = ledger(scorer, [request((10,))])
    recording.score([request((10,))])
    scorer.block.logp[:], scorer.block.log_mass[:] = 0, 0
    block, reuse = recording.replay([request((10,))], pytest.fail)
    assert reuse['reused'] == 1 and block.logp[0].tolist() == pytest.approx([np.log(.5) + .01] * 2) and block.log_mass[0] == pytest.approx(-.1)


def test_failed_partial_and_reference_rows_are_never_reused():
    scorer = Counting(failing={(12,)})
    partial = ScoreRequest((3,), (7,), ((11,), (12,)), (((1,), (2,)), ((1,), (2,))))
    recording = ledger(scorer, [request((10,)), request((13,)), partial])
    recording.score([request((13,)), partial])
    recording.score([request((10,))], reference=True)
    assert scorer.calls[-1][1] == {'reference': True} and list(recording.blocks) == [('a', request((13,)))]
    block, reuse = recording.replay([request((10,)), partial, request((13,))], scorer.score)
    assert reuse == {'requests': 3, 'reused': 1, 'scored': 2, 'rows': 4, 'scored_rows': 3}
    assert scorer.calls[-1][0] == [request((10,)), partial] and block.error == [None, None, 'failed', None]


def test_replay_without_misses_scores_nothing_and_preserves_order():
    scorer = Counting()
    recording = ledger(scorer, [request((10,)), request((11,))])
    recording.score([request((10,)), request((11,))])
    calls = len(scorer.calls)
    block, reuse = recording.replay([request((11,)), request((10,))], pytest.fail)
    assert reuse['scored'] == 0 and len(scorer.calls) == calls and block.log_mass.tolist() == pytest.approx([-.11, -.10])


def test_storage_is_bounded_by_the_declared_population():
    scorer = Counting()
    recording = ledger(scorer, [request((10,))])
    recording.score([request((n,)) for n in range(10, 20)])
    assert list(recording.blocks) == [('a', request((10,)))]
    recording.release()
    recording.score([request((10,))])
    assert recording.blocks == {}


def test_same_tokens_under_a_different_binding_score_again():
    scorer, binding = Counting(), ['adapter-a']
    recording = ScoreLedger(scorer, lambda: binding[0], [request((10,))])
    recording.score([request((10,))])
    binding[0] = 'adapter-b'
    block, reuse = recording.replay([request((10,))], scorer.score)
    assert reuse['reused'] == 0 and scorer.calls[-1][0] == [request((10,))]
    recording.score([request((10,))])
    assert list(recording.blocks) == [('adapter-b', request((10,)))]
    binding[0] = 'adapter-a'
    assert recording.replay([request((10,))], scorer.score)[1]['reused'] == 0


def test_a_binding_change_during_scoring_captures_nothing():
    scorer, keys = Counting(), iter(['before', 'after'])
    recording = ScoreLedger(scorer, lambda: next(keys), [request((10,))])
    recording.score([request((10,))])
    assert recording.blocks == {}


def test_invalidation_prohibits_replay_and_reaches_the_scorer():
    scorer = Counting()
    recording = ledger(scorer, [request((10,))])
    recording.score([request((10,))])
    recording.invalidate()
    assert scorer.invalidated == 1 and recording.replay([request((10,))], scorer.score)[1]['reused'] == 0


def test_incomplete_ordered_scores_fail():
    class Short(Counting):
        def score(self, requests, **options):
            block = super().score(requests, **options)
            return ScoreBlock(block.logp[:-1], block.log_mass[:-1], block.error[:-1])
    with pytest.raises(ConfigError, match='incomplete ordered scores'):
        ledger(Short(), [request((10,))]).score([request((10,)), request((11,))])


def public_requests(scorer):
    return [request for requests, reference in scorer.calls if not reference for request in requests]


def test_selection_judge_keeps_one_scorer_and_replays_its_public_scores():
    native = Native()
    with Runtime(FakeScorer()) as ordinary:
        ordinary.check('a tiny report', 'The report is tiny.')
        assert isinstance(ordinary.judge().scorer, FakeScorer) and not hasattr(ordinary, 'recording')
    with Selection(native, [], job=None) as probe:
        probe.check('a tiny report', 'The report is tiny.')
        population = public_requests(native)
    native.calls.clear()
    with Selection(native, population) as selection:
        judge = selection.judge()
        scorer = judge.scorer
        assert isinstance(scorer, ScoreLedger) and scorer.scorer is native
        first = selection.check('a tiny report', 'The report is tiny.')
        assert len(native.calls) == 1 and len(scorer.blocks) == len(population)
        block, reuse = selection.replay(population)
        assert reuse == {'requests': len(population), 'reused': len(population), 'scored': 0, 'rows': len(population), 'scored_rows': 0}
        assert len(native.calls) == 1 and selection.judge() is judge and judge.scorer is scorer
        fresh = native.score(population)
        assert np.array_equal(block.logp, fresh.logp) and np.array_equal(block.log_mass, fresh.log_mass)
        assert selection.check('a tiny report', 'The report is tiny.').as_dict() == first.as_dict()
        selection.score_native(population, reference=True)
        assert native.calls[-1] == (population, True)
        other = request((10,))
        mixed, reuse = selection.replay([other, *population])
        assert reuse['reused'] == len(population) and reuse['scored'] == 1 and native.calls[-1] == ([other], False)
        assert np.array_equal(mixed.logp[1:], block.logp)
    assert scorer.blocks == {} and scorer.population == frozenset()


def test_independent_selection_handles_share_no_ledger():
    first, second = Native(), Native()
    with Selection(first, []) as probe:
        probe.check('a tiny report', 'The report is tiny.')
        population = public_requests(first)
    second.vocab, second.inv = dict(first.vocab), list(first.inv)
    with Selection(first, population) as a, Selection(second, population) as b:
        a.check('a tiny report', 'The report is tiny.')
        calls = len(second.calls)
        assert a.judge().scorer is not b.judge().scorer and b.judge().scorer.blocks == {}
        assert b.replay(population)[1]['scored'] == len(population) and len(second.calls) == calls + 1
        assert a.replay(population)[1]['reused'] == len(population)


def test_binding_keys_follow_content_and_execution_identity():
    def key(name, **realized):
        native = Native(name=name)
        native.realized.update(realized)
        with Selection(native, []) as selection:
            return selection.judge().scorer.binding()
    assert key('base') == key('base') != key('adapter') and key('base', shared=False) != key('base')
    native = Native()
    with Selection(native, []) as probe:
        probe.check('a tiny report', 'The report is tiny.')
        population = public_requests(native)
    with Selection(native, population) as selection:
        selection.check('a tiny report', 'The report is tiny.')
        native.realized['scheduling'] = 'other@1'
        assert selection.replay(population)[1]['reused'] == 0
