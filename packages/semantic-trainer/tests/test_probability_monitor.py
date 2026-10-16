import json
import math
from pathlib import Path

import pytest

from semantic_switch.judgment import ESCAPE
from semantic_switch.metrics import OPTION_SOFTMAX, probabilities
from semantic_testkit.labels import presentation
from semantic_trainer.monitor import probability_entries
from semantic_trainer.objective import distribution


def entry(index, stage, slot_map, variant='canonical'):
    return presentation(f'p{index}', stage, variant, [1], [2], [3], [[4], [5]], 0, slot_map)


def row(identifier, presentations, label=1, classes=('x', 'y')):
    return {'id': identifier, 'task': 'task', 'classes': list(classes), 'label': label, 'fact': 'action', 'kind': 'pick', 'component': f'c-{identifier}',
            'presentations': presentations}


def report(rows, observations):
    out = probabilities(probability_entries(rows, {r['id']: {'probability': o} for r, o in zip(rows, observations)}))
    json.dumps(out, allow_nan=False)
    return out


def test_option_permutations_map_through_slot_maps_to_the_same_semantic_scores():
    canonical, permuted = row('a', [entry(0, 'full', [0, 1])]), row('b', [entry(1, 'full', [1, 0])])
    first = distribution(canonical, [[math.log(.3), math.log(.7)]], [math.log(.9)])
    second = distribution(permuted, [[math.log(.7), math.log(.3)]], [math.log(.9)])
    assert first['options'] == ['x', 'y'] and second['options'] == ['y', 'x'] and first['normalization'] == OPTION_SOFTMAX
    a, b = report([canonical], [first])['micro'], report([permuted], [second])['micro']
    assert a['brier']['value'] == pytest.approx(b['brier']['value']) == pytest.approx(2 * .3 ** 2)
    assert a['nll']['value'] == pytest.approx(b['nll']['value']) == pytest.approx(-math.log(.7))
    assert a['unconditioned_nll']['value'] == pytest.approx(-math.log(.7 * .9))


def test_same_option_probabilities_with_different_answer_mass_change_only_unconditioned_nll():
    native, hf = row('n', [entry(0, 'full', [0, 1])]), row('h', [entry(1, 'full', [0, 1])])
    logp = [[math.log(.25), math.log(.75)]]
    a = report([native], [distribution(native, logp, [0.])])['micro']
    b = report([hf], [distribution(hf, logp, [math.log(.5)])])['micro']
    assert a['brier'] == b['brier'] and a['nll'] == b['nll'] and a['nll']['value'] == pytest.approx(-math.log(.75))
    assert a['unconditioned_nll']['value'] == pytest.approx(-math.log(.75)) and b['unconditioned_nll']['value'] == pytest.approx(-math.log(.375))


def test_staged_rows_report_canonical_primitive_stages_and_leave_the_final_distribution_unassessed():
    staged = row('s', [entry(0, 'mentions', [1, -1]), entry(1, 'mentions', [-1, 1], 'permuted'), entry(2, 'fallback', [0, 1])], label=0)
    observation = distribution(staged, [[math.log(.2), math.log(.8)], [0., -50.], [math.log(.6), math.log(.4)]], [math.log(.5), 0., math.log(.8)])
    assert observation['stage'] == 'staged' and observation['probabilities'] is None
    assert [stage['stage'] for stage in observation['stages']] == ['mentions', 'fallback'] and observation['stages'][0]['options'] == ['y', ESCAPE]
    out = report([staged], [observation])
    assert out['micro']['unscored']['staged'] == 1 and out['micro']['scored'] == 0
    assert out['stages']['mentions']['nll']['value'] == pytest.approx(-math.log(.8))
    assert out['stages']['mentions']['unconditioned_nll']['value'] == pytest.approx(-math.log(.4))
    assert out['stages']['fallback']['brier']['value'] == pytest.approx(2 * .4 ** 2)


def test_hf_evaluation_distributions_reproduce_the_loss_v2_reference():
    torch = pytest.importorskip('torch')
    from semantic_trainer.objective import answer_logits, bind, evaluate, prepared
    from test_lora import TOKENS, adapted_tiny, fit_rows, reference, sequences
    model = adapted_tiny()
    rows = [{**r, 'fact': 'action', 'kind': 'pick', 'component': r['id']} for r in fit_rows()]
    results = evaluate(bind(model), rows, prepared(rows, bind(model).vocabulary), 0, TOKENS, lambda: 1)
    single = rows[1]
    observation = results[single['id']]['probability']
    (shown,) = single['presentations']
    with torch.no_grad():
        logits = answer_logits(bind(model.eval()), sequences([shown]), 0)
    _, nll, conditional, mass = reference(logits, [shown])[0]
    gold = observation['options'].index(single['classes'][single['label']])
    assert sum(observation['probabilities']) == pytest.approx(1) and observation['mass'] == pytest.approx(mass.item(), rel=1e-5)
    assert -math.log(observation['probabilities'][gold]) == pytest.approx(conditional.item(), rel=1e-4, abs=1e-5)
    out = report([single], [observation])['micro']
    assert out['unconditioned_nll']['value'] == pytest.approx(nll.item(), rel=1e-4, abs=1e-5) == pytest.approx(results[single['id']]['nll'], rel=1e-4, abs=1e-5)
    assert all(results[r['id']]['probability']['stage'] == 'staged' for r in (rows[0], rows[2]))


def test_monitor_points_publish_probability_diagnostics_beside_unchanged_loss(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from semantic_switch import files
    from semantic_trainer import lora
    from semantic_trainer.jobs import resolve
    from test_lora import fake_native, monitored_job
    job, _, _ = monitored_job(tmp_path, monkeypatch, recipe={'epochs': 1, 'eval_every_examples': 8, 'min_epochs': 1, 'patience_examples': 1000})
    fake_native(monkeypatch)
    result = lora.train(job, tmp_path / 'out')
    progress = [record for record in files.read_jsonl(Path(result['path']) / 'progress.jsonl') if record['kind'] == 'evaluation']
    _, observations = resolve(job)
    selected = sum(r['role'] == 'model_select' for r in observations)
    for point in progress:
        diagnostics = point['model_select']['probabilities']
        assert point['loss']['version'] == 2 and diagnostics['micro']['submitted'] == selected and 'targets' not in diagnostics
        assert diagnostics['micro']['scored'] == selected and diagnostics['micro']['unconditioned_nll']['rows'] == selected
        assert diagnostics['micro']['nll']['value'] > 0 and diagnostics['micro']['reliability']['rows'] == selected
        assert diagnostics['protocol']['revision'] == 1 and list(diagnostics['kinds']) == ['pick']


def test_native_predictions_store_semantic_distributions_with_answer_mass():
    import numpy as np
    from types import SimpleNamespace
    from semantic_switch.protocols import ScoreBlock
    from semantic_trainer.lora import native_predictions
    rows = [row('a', [entry(0, 'full', [1, 0])], label=0), row('s', [entry(1, 'mentions', [0, -1]), entry(2, 'fallback', [0, 1])], label=1)]
    block = ScoreBlock.empty(3, 2)
    block.logp[:] = np.log([[.8, .2], [.1, .9], [.3, .7]])
    block.log_mass[:] = np.log([.5, 1., .25])
    runtime = SimpleNamespace(score_native=lambda requests, role: block)
    predictions, _ = native_predictions(runtime, rows)
    first = predictions['a']
    assert first['answer'] == 1 and first['probability']['options'] == ['y', 'x'] and first['probability']['probabilities'] == pytest.approx([.8, .2])
    out = report(rows, [predictions['a']['probability'], predictions['s']['probability']])
    assert out['micro']['scored'] == 1 and out['micro']['nll']['value'] == pytest.approx(-math.log(.2))
    assert out['micro']['unconditioned_nll']['value'] == pytest.approx(-math.log(.1)) and out['micro']['unscored']['staged'] == 1
    assert out['stages']['mentions']['nll']['value'] == pytest.approx(-math.log(.9)) and out['stages']['fallback']['unconditioned_nll']['value'] == pytest.approx(-math.log(.7 * .25))
