
from semantic_switch import model_jobs
from pathlib import Path

import pytest

from semantic_switch import ConfigError, Runtime, api, files
from semantic_switch.compiler import compile_spec
from semantic_testkit.scoring import FakeScorer
from bench import procedures

HISTORICAL = Path(__file__).parent / 'data' / 'printer-zt411-reference-3.yaml'
DISTRIBUTION = {True: [.98, .01, .01], False: [.01, .98, .01], 'unstated': [.01, .01, .98]}


class Gold:
    def __init__(self, spec):
        self.checks = {name: fact['check'] for name, fact in spec['facts'].items() if 'check' in fact}
        self.values, self.asked = {}, []

    def __call__(self, state, question, width):
        name = next(name for name, text in self.checks.items() if text in question)
        self.asked.append(name)
        value = self.values[name]
        return [1 / width] * width if value is None else DISTRIBUTION[value][:width] if width == 3 else DISTRIBUTION[value][:2]


def decisions(path, frames, golds):
    spec = files.read_yaml(path)
    gold = Gold(spec)
    out = {}
    with Runtime(FakeScorer(gold)) as rt:
        switch = rt.load(path, draft=True)
        for frame in frames:
            gold.values = {name: golds[frame.id].primitives[name].value for name in gold.checks}
            result = switch(frame.input, facts=[*gold.checks, 'actions'])
            out[frame.id] = dict(result.facts['actions'].states)
    return out, gold.asked


def test_retired_printer_complements_preserve_every_action_state_over_the_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    frames = [frame for frame in procedures.frames(procedures.scenarios(0, 1, 2)) if frame.condition == procedures.TRAINED]
    golds = {frame.id: procedures.reference(frame) for frame in frames}
    for gold in golds.values():
        primitives = gold.primitives
        assert primitives['ribbon_absent'].value == {True: False, False: True}.get(primitives['ribbon_loaded'].value, primitives['ribbon_loaded'].value)
        assert primitives['dt_media'].value == {True: False, False: True}.get(primitives['tt_media'].value, primitives['tt_media'].value)
    old, old_asked = decisions(HISTORICAL, frames, golds)
    new, new_asked = decisions(procedures.DEFINITION, frames, golds)
    assert old == new and len(frames) > 20
    assert {'ribbon_absent', 'dt_media'} <= set(old_asked) and not {'ribbon_absent', 'dt_media'} & set(new_asked)
    assert len(old_asked) / len(frames) == 12 and len(new_asked) / len(frames) == 10


def test_media_universe_is_declared_where_negation_relies_on_it():
    spec = files.read_yaml(procedures.DEFINITION)
    universe = {tuple(card['source']['media_universe']) for card in spec['facts']['actions']['cards'] if 'media_universe' in card.get('source', {})}
    assert universe == {procedures.DOMAIN['media_type']}
    assert set(spec['retired']) == {'dt_media', 'ribbon_absent'} and not set(spec['retired']) & set(spec['facts'])


COMPLEMENT = {'name': 'negation', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
              'facts': {'loaded': {'check': 'A ribbon is loaded.', 'unstated': True}}}


def complement(folded):
    facts = dict(COMPLEMENT['facts'])
    load = {'id': 'load', 'applies_when': [], 'excludes': [{'fact': 'loaded'}]} if folded else {'id': 'load', 'applies_when': [{'fact': 'absent'}]}
    if not folded:
        facts['absent'] = {'check': 'No ribbon is loaded.', 'unstated': True}
    return {**COMPLEMENT, 'facts': {**facts, 'actions': {'select': 'What should happen?', 'cards': [load, {'id': 'keep', 'applies_when': [{'fact': 'loaded'}]}]}}}


@pytest.mark.parametrize('truth', [True, False, 'unstated', None])
def test_card_exclusion_is_the_kleene_complement_of_an_independently_judged_negation(tmp_path, monkeypatch, truth):
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    negated = {True: False, False: True}.get(truth, truth)
    states = []
    for folded in (False, True):
        path = tmp_path / f'negation-{folded}.yaml'
        files.publish(path, complement(folded))
        values = {'A ribbon is loaded.': truth, 'No ribbon is loaded.': negated}
        asked = []

        def oracle(state, question, width):
            text = next(text for text in values if text in question)
            asked.append(text)
            return [1 / width] * width if values[text] is None else DISTRIBUTION[values[text]][:width]
        with Runtime(FakeScorer(oracle)) as rt:
            result = rt.load(path, draft=True)({'message': 'ribbon'}, facts=['actions'])
        states.append((result.facts['actions'].status, dict(result.facts['actions'].states), len(set(asked))))
    assert states[0][:2] == states[1][:2] and (states[0][2], states[1][2]) == (2, 1)


def test_retired_names_load_historical_labels_without_accepting_unknown_facts(tmp_path):
    spec = {**complement(True), 'retired': ['absent']}
    plan = compile_spec(spec)
    from semantic_switch.evidence import decode
    record = {'schema': 3, 'input': {'message': 'x'}, 'facts': {'loaded': True, 'absent': False}, 'revs': {}, 'label_sources': {}}
    with pytest.raises(ConfigError):
        decode(plan, {**record, 'facts': {'loaded': True, 'typo': False}}, labelled=True)
    with pytest.raises(ConfigError, match='only retired'):
        decode(plan, {**record, 'facts': {'absent': False}}, labelled=True)
    with pytest.raises(ConfigError, match='current facts'):
        compile_spec({**spec, 'retired': ['loaded']})
    assert compile_spec(complement(True)).spec == {key: value for key, value in plan.spec.items() if key != 'retired'}


def test_readiness_has_an_insufficiency_witness_for_independent_primitive_summaries():
    (witness,) = procedures.readiness_witness()
    assert set(witness['outcomes']) == {'false', 'null'} and witness['extracted']['ribbon_loaded'] is None
    frames = [frame for frame in procedures.frames(procedures.scenarios(0, 1, 1)) if frame.condition == procedures.TRAINED]
    assert all(entry['target'] == 'ready_to_print' for entry in procedures.sufficiency(map(procedures.View, frames), 'ready_to_print'))
