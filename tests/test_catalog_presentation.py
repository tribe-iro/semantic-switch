
from semantic_switch import model_jobs
import re

import pytest

from semantic_switch import ConfigError, Runtime, api, files
from semantic_switch.compiler import compile_spec
from semantic_testkit.scoring import FakeScorer
from bench import procedures

DISTRIBUTION = {True: [.98, .01, .01], False: [.01, .98, .01], 'unstated': [.01, .01, .98]}


class Recorder(FakeScorer):
    def __init__(self, oracle):
        super().__init__(oracle)
        self.prefixes = []

    def score(self, requests, **kwargs):
        self.prefixes.extend(len(r.prefix) for r in requests)
        return super().score(requests, **kwargs)


def run(path, frames, golds, catalog):
    spec = files.read_yaml(path)
    plan = compile_spec(spec)
    checks = {name: fact['check'] for name, fact in spec['facts'].items() if 'check' in fact}
    entries = {plan.catalog_ids[plan.fact_index[name]]: name for name in checks} if catalog else {}
    values = {}

    def oracle(state, question, width):
        found = re.search(r'criterion \[(C\d+)\]', question)
        name = entries[found.group(1)] if found else next(name for name, text in checks.items() if text in question)
        value = values[name]
        return [1 / width] * width if value is None else DISTRIBUTION[value][:width] if width == 3 else DISTRIBUTION[value][:2]
    scorer = Recorder(oracle)
    out = {}
    with Runtime(scorer) as rt:
        switch = rt.load(path, draft=True)
        for frame in frames:
            values.update({name: golds[frame.id].primitives[name].value for name in checks})
            out[frame.id] = dict(switch(frame.input, facts=[*checks, 'actions']).facts['actions'].states)
    return out, scorer


def test_catalog_presentation_preserves_actions_and_moves_criteria_into_the_cached_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    spec = files.read_yaml(procedures.DEFINITION)
    inline, catalogued = tmp_path / 'inline.yaml', tmp_path / 'catalog.yaml'
    files.publish(inline, spec)
    files.publish(catalogued, {**spec, 'presentation': 'catalog'})
    frames = [frame for frame in procedures.frames(procedures.scenarios(0, 1, 1)) if frame.condition == procedures.TRAINED]
    golds = {frame.id: procedures.reference(frame) for frame in frames}
    left, plain = run(inline, frames, golds, False)
    right, pointed = run(catalogued, frames, golds, True)
    assert left == right and len(plain.requests) == len(pointed.requests)
    suffix = lambda scorer: sum(len(scorer.tokenize(q)) for _, q in scorer.requests)
    assert suffix(pointed) < .6 * suffix(plain) and min(pointed.prefixes) > max(plain.prefixes)
    assert all('criterion [C' in q for _, q in pointed.requests) and not any('criterion [C' in q for _, q in plain.requests)


def test_catalog_identity_follows_content_and_order_and_inline_tasks_keep_their_identity():
    spec = files.read_yaml(procedures.DEFINITION)
    plain, catalog = compile_spec(spec), compile_spec({**spec, 'presentation': 'catalog'})
    assert plain.catalog is None and 'presentation' not in plain.spec and catalog.spec['presentation'] == 'catalog'
    facts = dict(spec['facts'])
    reordered = compile_spec({**spec, 'presentation': 'catalog', 'facts': {name: facts[name] for name in reversed(facts)}})
    edited = compile_spec({**spec, 'presentation': 'catalog', 'facts': {**facts, 'tt_media': {**facts['tt_media'], 'check': facts['tt_media']['check'] + ' Edited.'}}})
    assert len({catalog.catalog, reordered.catalog, edited.catalog}) == 3


def test_categorical_catalog_keeps_options_inline_and_the_prefix_cap_reports_overflow(tmp_path, monkeypatch):
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    spec = {'name': 'route', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'], 'presentation': 'catalog',
            'facts': {'topic': {'pick': 'Which topic does the message raise?', 'options': {'refund': 'Money back', 'billing': 'Invoice error'}},
                      'urgent': {'check': 'The customer needs an answer today.'}}}
    path = tmp_path / 'route.yaml'
    files.publish(path, spec)
    with Runtime(FakeScorer(lambda state, question, width: [.9] + [.1 / (width - 1)] * (width - 1))) as rt:
        result = rt.load(path, draft=True)({'message': 'refund please'})
        asked = rt.judge().scorer.requests
    assert result.facts['topic'].value == 'refund' and any('A) Money back' in q and 'criterion [C1]' in q for _, q in asked)
    huge = {**spec, 'facts': {**spec['facts'], **{f'f{i}': {'check': 'word ' * 60 + str(i)} for i in range(30)}}}
    files.publish(path, huge)
    with Runtime(FakeScorer()) as rt, pytest.raises(ConfigError, match='prefix'):
        rt.load(path, draft=True)({'message': 'x'})
