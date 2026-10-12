import numpy as np
import pytest

from semantic_switch import Runtime
from semantic_switch.protocols import ScoreBlock, ScoreLedger, ScoreRequest
from tests.conftest import FakeScorer


class Counting:
    def __init__(self, failing=()):
        self.calls, self.failing = [], set(failing)

    def score(self, requests, **options):
        self.calls.append((list(requests), options))
        width = max(len(slots) for request in requests for slots in request.slots)
        block = ScoreBlock.empty(sum(len(request.suffixes) for request in requests), width)
        row = 0
        for request in requests:
            for suffix, slots in zip(request.suffixes, request.slots):
                if suffix in self.failing:
                    block.error[row] = 'failed'
                else:
                    block.logp[row, :len(slots)] = np.log(np.full(len(slots), 1 / len(slots))) + suffix[0] * 1e-3
                    block.log_mass[row] = -suffix[0] * 1e-2
                row += 1
        return block


def request(question, groups=((1,), (2,)), state=(7,)):
    return ScoreRequest((3,), state, (question,), (groups,))


def test_replay_reuses_only_identical_token_requests_and_charges_the_rest():
    scorer = Counting()
    ledger = ScoreLedger(scorer)
    public = [request((10,)), ScoreRequest((3,), (7,), ((11,), (12,)), (((1,), (2,)), ((1,), (2,), (4,))))]
    recorded = ledger.score(public)
    exported = [request((10,)), request((11,)), request((10,), groups=((2,), (1,))), request((10,), state=(8,)), request((12,), groups=((1,), (2,), (4,)))]
    block, reuse = ledger.replay(exported, scorer.score)
    assert reuse == {'requests': 5, 'reused': 1, 'scored': 4, 'rows': 5, 'scored_rows': 4}
    assert scorer.calls[-1][0] == exported[1:]
    assert np.array_equal(block.logp[0, :2], recorded.logp[0, :2]) and block.log_mass[0] == recorded.log_mass[0]
    fresh = Counting().score(exported)
    assert np.array_equal(block.logp, fresh.logp) and np.array_equal(block.log_mass, fresh.log_mass) and block.error == fresh.error
    assert block.logp.shape == (5, 3) and np.isneginf(block.logp[0, 2])


def test_failed_or_reference_rows_are_never_reused():
    scorer = Counting(failing={(10,)})
    ledger = ScoreLedger(scorer)
    ledger.score([request((10,))])
    ledger.score([request((11,))], reference=True)
    block, reuse = ledger.replay([request((10,)), request((11,))], scorer.score)
    assert reuse['reused'] == 0 and reuse['scored'] == 2 and block.error == ['failed', None]


def test_replay_without_misses_scores_nothing():
    scorer = Counting()
    ledger = ScoreLedger(scorer)
    ledger.score([request((10,)), request((11,))])
    calls = len(scorer.calls)
    block, reuse = ledger.replay([request((11,)), request((10,))], pytest.fail)
    assert reuse['scored'] == 0 and len(scorer.calls) == calls and block.log_mass.tolist() == pytest.approx([-.11, -.10])


def test_runtime_recording_wraps_and_restores_the_judge_scorer():
    fake = FakeScorer()
    with Runtime({'judge': fake}) as runtime:
        runtime.check('a tiny report', 'The report is tiny.')
        before = runtime.judge().scorer
        with runtime.recording() as ledger:
            runtime.check('a tiny report', 'The report is tiny.')
            assert runtime.judge().scorer is ledger and ledger.blocks
            requests = list(ledger.blocks)
            block, reuse = ledger.replay(requests, runtime.score_native)
        assert runtime.judge().scorer is before
    assert reuse['reused'] == len(requests) and reuse['scored'] == 0 and not any(block.error)
