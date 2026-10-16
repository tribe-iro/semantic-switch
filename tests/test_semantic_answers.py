from __future__ import annotations

import json

import pytest

from semantic_switch import ConfigError, Runtime, api, files, tools
from semantic_switch.evidence import Actor, Criteria, LabelSource, Verification, assess, reviewed_content
from semantic_switch.metrics import scored_answer, semantic_answer
from semantic_switch.results import Prediction
from semantic_switch.tools import deployment
from semantic_switch.tools.data import Labels
from semantic_switch.training import (Task, compare_deployments, decide, declaration, recompute_comparison, reference_support, score_observations)
from semantic_testkit.scoring import FakeScorer, letters_of

CLAIM = 'The customer asks to receive money back for a charge.'
OPTIONS = {'charges': 'charges or invoices', 'login': 'login trouble'}


def spec(options=OPTIONS):
    return {'name': 'triage', 'fields': {'message': {'type': 'text', 'required': True}}, 'reads': ['message'],
            'facts': {'topic': {'pick': 'What is the message mainly about?', 'options': options, 'allow_none': True},
                      'refund': {'check': CLAIM, 'unstated': True}},
            'rules': [{'id': 'refund', 'when': 'refund', 'then': 'billing'}], 'otherwise': 'general'}


COMPUTED = [{'derivation': 'computed', 'producer': 'reference', 'revision': '1'}]
CHECK = [('hello silent', 'none', 'unstated', 'a'), ('Hello,  SILENT', 'none', 'unstated', 'a'), ('charge refund', 'charges', True, 'b'),
         ('charge only', 'charges', False, 'b'), ('login silent', 'login', False, 'c'), ('hello again', 'charges', False, 'd'), ('boom', 'none', False, 'e')]


def described(question, width, wanted):
    p = [.02 / (width - 1)] * width
    p[ord(next(letter for desc, letter in letters_of(question).items() if desc.startswith(wanted))) - 65] = .98
    total = sum(p)
    return [v / total for v in p]


def good(state, question, width):
    s = state.lower()
    if 'boom' in s:
        raise RuntimeError('backend exploded')
    if 'money back' in question:
        return described(question, width, 'the input does not say' if 'silent' in s else 'the input states it is true' if 'refund' in s
                         else 'the input states it is false')
    return described(question, width, 'charges' if 'charge' in s else 'login' if 'login' in s else 'none of the above')


def stubborn(state, question, width):
    return described(question, width, 'the input states it is false' if 'money back' in question else 'login')


def locked(directory, scorer, options=OPTIONS):
    directory.mkdir(parents=True)
    path = directory / 'triage.yaml'
    files.publish(path, spec(options))
    with Runtime(scorer) as runtime:
        tools.lock(runtime, path)
    return directory / 'triage.lock.json'


def evidence(directory, sources=None):
    directory.mkdir(parents=True)
    path = directory / 'triage.yaml'
    files.publish(path, spec())
    paths = files.Paths.of(path)
    with Runtime(FakeScorer()) as runtime:
        task = Task(runtime.load(path, draft=True))
        files.publish(paths.test, ''.join(json.dumps(task.label({'message': message}, {'topic': topic, 'refund': refund}, role='check', groups=[f'family-{family}'],
                                                                sources=(sources or {}).get(message, {'topic': COMPUTED, 'refund': COMPUTED}))) + '\n'
                                          for message, topic, refund, family in CHECK))
        Labels(runtime.load(path, draft=True))
    return paths.test, paths.dataset


def test_semantic_answers_map_escapes_by_judgment_keys_and_keep_unavailable_predictions_unscored():
    keys = ('charges', 'login', 'none')
    assert semantic_answer(Prediction('none', None, None, .98), keys) == 'none' and semantic_answer({'kind': 'unstated', 'key': None}) == 'unstated'
    assert scored_answer(Prediction('none', None, None, .98), 'none', 'none', keys) == {'answer': 'none', 'correct': True, 'accepted': True}
    assert scored_answer({'kind': 'unstated', 'key': None}, 'unstated', 'needs_input') == {'answer': 'unstated', 'correct': True, 'accepted': False}
    assert scored_answer(Prediction('none', None, None, .98), 'charges', 'none', keys) == {'answer': 'none', 'correct': False, 'accepted': True}
    assert scored_answer(Prediction('value', 'login', 'login', .6), 'charges', 'uncertain', keys) == {'answer': 'login', 'correct': False, 'accepted': False}
    assert scored_answer(Prediction('value', 'login', 'login', .9), 'login', 'failed', keys) == {'answer': None, 'correct': None, 'accepted': False}
    assert scored_answer(None, 'none', 'unexamined') == {'answer': None, 'correct': None, 'accepted': False}
    assert scored_answer({'kind': 'value', 'key': 'login'}, None, 'ok') == {'answer': 'login', 'correct': None, 'accepted': True}
    with pytest.raises(ConfigError, match='semantic keys'):
        semantic_answer(Prediction('value', 'other', 'other', .9), keys)
    with pytest.raises(ConfigError, match='not value, none or unstated'):
        semantic_answer({'kind': 'maybe', 'key': None})


def test_native_selection_scoring_accepts_a_confident_none_and_scores_a_correct_unstated_without_accepting_it(tmp_path):
    path = tmp_path / 'triage.yaml'
    files.publish(path, spec())
    with Runtime(FakeScorer(oracle=good)) as runtime:
        switch = runtime.load(path, draft=True)
        plan = switch.plan
        rows = [{'id': 'topic', 'fact': 'topic', 'bound_input': {'message': 'hello silent'}, 'meaning_rev': plan.fact('topic').meaning_rev,
                 'label': plan.fact('topic').keys.index('none')},
                {'id': 'refund', 'fact': 'refund', 'bound_input': {'message': 'hello silent'}, 'meaning_rev': plan.fact('refund').meaning_rev, 'label': 2}]
        topic, refund = score_observations(switch, rows)
    assert (topic['answer'], topic['correct'], topic['accepted'], topic['result']['status']) == ('none', True, True, 'none')
    assert (refund['answer'], refund['correct'], refund['accepted'], refund['result']['status']) == ('unstated', True, False, 'needs_input')


def history(*sources):
    return [LabelSource.parse(source) for source in sources]


def test_reference_support_keeps_strata_and_never_upgrades_a_mixed_population():
    data, value = {'message': 'charge refund'}, True
    confirm = Verification('label', 'ana', Actor('human'), 'review', 'meaning-1', reviewed_content('label', data, ('refund',), value), 'confirm')
    stale = Verification('label', 'ana', Actor('human'), 'review', 'meaning-1', reviewed_content('label', data, ('refund',), False), 'confirm')
    authored = {'derivation': 'authored', 'producer': 'ana', 'actor': {'kind': 'human'}}
    records = [assess(history(*sources), data=data, path=('refund',), value=value, criteria=Criteria(frozenset({'meaning-1'}))).record() for sources in (
        COMPUTED, [{**authored, 'verifications': [confirm.record()]}], [{**authored, 'verifications': [stale.record()]}],
        [{'derivation': 'judged', 'producer': 'openrouter:teacher'}])]
    assert [record['state'] for record in records] == ['computed', 'verified', 'authored', 'judged']
    found = reference_support(records, scope=('verified', 'computed'), review={'status': 'undisclosed'})
    assert (found['state'], found['targets'], found['within_scope'], found['established'], found['verified'], found['unverified'], found['stale']) == \
        ('partial', 4, 2, 2, 1, 3, 1)
    assert found['strata'] == {'authored': 1, 'computed': 1, 'judged': 1, 'verified': 1} and found['derivations'] == {'authored': 2, 'computed': 1, 'judged': 1}
    assert reference_support(records[2:], scope=('verified', 'computed'), review=None)['state'] == 'agreement'
    assert reference_support(records[2:3], scope=('authored',), review={'status': 'undisclosed'})['state'] == 'established'
    assert reference_support(records[:2], scope=('verified', 'computed'), review={'status': 'unreviewed'})['state'] == 'unreviewed'
    assert reference_support([records[0], None], scope=('computed',), review=None)['state'] == 'unavailable'
    legacy = {key: value for key, value in records[1].items() if key != 'binding'}
    assert reference_support([records[0], legacy], scope=('verified', 'computed'), review=None)['state'] == 'unavailable'
    assert reference_support([], scope=('computed',), review=None)['state'] == 'empty'
    with pytest.raises(ConfigError, match='reference scope'):
        declaration(margin=.1, reference=('judged',))


def test_decide_consumes_the_typed_reference_assessment():
    arm = {'failed': 0, 'unresolved': 0, 'forbidden_accepted': 0, 'targets': {'precision': .95}, 'latency_ms': None, 'provider_cost': None}
    report = {'declaration': declaration(margin=.05), 'exposure': {'state': 'fresh', 'independent': True}, 'lineage': {'candidate': {'state': 'declared_disjoint'}},
              'arms': {'baseline': arm, 'candidate': arm}, 'paired': {'difference': {'interval': [.1, .2], 'support': 'reported'}}}
    verdicts = {state: decide({**report, 'reference': {'state': state}})['verdict']
                for state in ('established', 'partial', 'agreement', 'unreviewed', 'unavailable')}
    assert verdicts == {'established': 'supported', 'partial': 'partial_reference', 'agreement': 'reference_agreement',
                        'unreviewed': 'unreviewed_reference', 'unavailable': 'unassessed_reference'}


def manifest(tmp_path, locks, transforms=None):
    path = tmp_path / 'robustness.json'
    files.write_new(path, {'kind': 'robustness-transforms', 'schema': 1, 'transforms': transforms or [
        {'id': 'perm', 'transformation': 'permutation', 'expected': 'invariant', 'parent': {'message': 'hello silent'}, 'locks': locks['perm'],
         'mapping': {'topic': {'login': 'login', 'charges': 'charges', 'none': 'none'}}},
        {'id': 'rename', 'transformation': 'rename', 'expected': 'invariant', 'parent': {'message': 'charge refund'}, 'locks': locks['rename'],
         'mapping': {'topic': {'invoices': 'charges', 'login': 'login', 'none': 'none'}}},
        {'id': 'format', 'transformation': 'format', 'expected': 'invariant', 'parent': {'message': 'hello silent'}, 'input': {'message': 'Hello,  SILENT'}},
        {'id': 'contrast', 'transformation': 'contrast', 'expected': 'changed', 'parent': {'message': 'charge refund'}, 'input': {'message': 'charge only'},
         'targets': ['refund']}]})
    return path


def arms(tmp_path):
    root = tmp_path / 'locks'
    made = {}
    for arm, oracle in (('baseline', stubborn), ('candidate', good)):
        made[arm] = locked(root / arm, FakeScorer(name=arm, oracle=oracle))
        made[f'{arm}-perm'] = locked(root / f'{arm}-perm', FakeScorer(name=arm, oracle=oracle), {'login': OPTIONS['login'], 'charges': OPTIONS['charges']})
        made[f'{arm}-rename'] = locked(root / f'{arm}-rename', FakeScorer(name=arm, oracle=oracle), {'invoices': OPTIONS['charges'], 'login': OPTIONS['login']})
    return made


def run(tmp_path, made, robustness, to='compared', **declared):
    labels, dataset = evidence(tmp_path / 'evidence') if not (tmp_path / 'evidence').exists() else \
        (tmp_path / 'evidence' / 'triage.test.jsonl', tmp_path / 'evidence' / 'triage.dataset.json')
    return compare_deployments(made['baseline'], made['candidate'], labels, dataset, tmp_path / to, count=200, robustness=robustness,
                               declaration=declaration(margin=.05, forbidden={'topic': ['none']}, **declared),
                               bindings={'baseline': {'judge': FakeScorer(name='baseline', oracle=stubborn)}, 'candidate': {'judge': FakeScorer(name='candidate', oracle=good)}})


def test_comparison_scores_escapes_semantically_reports_robustness_and_recomputes_stored_rows_without_inference(tmp_path, monkeypatch):
    made = arms(tmp_path)
    locks = {kind: {arm: str(made[f'{arm}-{kind}']) for arm in ('baseline', 'candidate')} for kind in ('perm', 'rename')}
    report = run(tmp_path, made, manifest(tmp_path, locks))
    candidate = {row['input']: row['targets'] for row in report['rows']['candidate']}
    hello = next(targets for key, targets in candidate.items() if targets['topic']['gold'] == 'none' and targets['refund']['correct'] is True and
                 targets['topic']['outcome'] == 'none')
    assert (hello['topic']['answer'], hello['topic']['correct'], hello['topic']['accepted']) == ('none', True, True)
    assert (hello['refund']['answer'], hello['refund']['correct'], hello['refund']['accepted'], hello['refund']['outcome']) == ('unstated', True, False, 'needs_input')
    counts = report['arms']['candidate']['targets']
    assert (counts['submitted'], counts['predicted'], counts['raw_correct'], counts['accepted'], counts['correct_accepted'], counts['wrong_accepted']) == \
        (14, 12, 10, 9, 8, 1)
    assert report['arms']['candidate']['forbidden_accepted'] == 3 and report['arms']['candidate']['failed'] == 1
    assert report['reference']['state'] == 'established' and report['reference']['strata'] == {'computed': 14}
    assert report['decision']['verdict'] == 'inconclusive' and report['decision']['reference'] == 'established'
    latency = report['arms']['candidate']['latency_ms']
    assert latency['cold']['calls'] == 1 and latency['steady']['calls'] == 6 and report['arms']['candidate']['cost']['submitted'] == 7
    good_arm, stubborn_arm = (report['robustness']['arms'][arm]['all'] for arm in ('candidate', 'baseline'))
    assert (good_arm['submitted'], good_arm['assessed'], good_arm['families'], good_arm['transforms']) == (7, 7, 2, 4)
    assert good_arm['raw_flip'] == {'count': 1, 'of': 7, 'rate': 1 / 7} and good_arm['accepted_flip']['count'] == 1
    assert good_arm['invariant'] == {'assessed': 6, 'consistent': 6, 'consistent_correct': 6, 'consistent_wrong': 0}
    assert good_arm['contrast'] == {'submitted': 1, 'passed': {'count': 1, 'of': 1, 'rate': 1.}, 'stable': 0}
    assert good_arm['probability']['comparable'] == 7 and good_arm['probability']['max_total_variation'] == pytest.approx(.97)
    assert stubborn_arm['raw_flip']['count'] == 0 and stubborn_arm['invariant']['consistent_wrong'] == 6
    assert stubborn_arm['contrast'] == {'submitted': 1, 'passed': {'count': 0, 'of': 1, 'rate': 0.}, 'stable': 1}
    assert report['robustness']['arms']['candidate']['transformations']['rename']['correct']['sibling']['count'] == 2
    assert report['robustness']['arms']['candidate']['transformed_calls']['submitted'] == 2
    assert report['plan']['metrics']['robustness'] == report['robustness']['id']

    def forbidden(*args, **kwargs):
        raise AssertionError('recomputation must not run inference')
    monkeypatch.setattr(api, 'Runtime', forbidden)
    again = recompute_comparison(tmp_path / 'compared' / 'comparison.json', tmp_path / 'recomputed.json')
    assert {key: again[key] for key in ('arms', 'paired', 'reference', 'robustness', 'decision', 'exposure')} == \
        {key: report[key] for key in ('arms', 'paired', 'reference', 'robustness', 'decision', 'exposure')}
    assert again['recomputed']['id'] == report['id']

    with files.open_published(tmp_path / 'compared' / 'comparison.json') as (stored, handles):
        rows = [json.loads(line) for line in handles['rows.jsonl']]
    for row in rows:
        for target in row['targets'].values():
            for key in ('answer', 'reference', 'options', 'scores'):
                target.pop(key)
            kind = (target['prediction'] or {}).get('kind')
            target['correct'] = None if kind != 'value' or target['outcome'] in ('failed', 'unexamined') else target['prediction']['key'] == target['gold']
            target['accepted'] = target['correct'] is not None and target['outcome'] == 'ok'
    old = tmp_path / 'old' / 'comparison.json'
    files.publish(old, {**{k: v for k, v in stored.items() if k not in ('payloads', 'robustness')}, 'schema': 2}, {'rows.jsonl': [json.dumps(row) + '\n' for row in rows]})
    repaired = recompute_comparison(old, tmp_path / 'repaired.json')
    assert repaired['arms']['candidate']['targets'] == counts and repaired['recomputed']['schema'] == 2
    assert repaired['reference']['state'] == 'unavailable' and repaired['decision']['verdict'] == 'unassessed_reference'
    with pytest.raises(ConfigError, match='already exists'):
        recompute_comparison(old, tmp_path / 'repaired.json')


def test_mixed_reference_support_is_reported_and_blocks_adoption(tmp_path):
    made = arms(tmp_path)
    judged = [{'derivation': 'judged', 'producer': 'openrouter:teacher'}]
    evidence(tmp_path / 'evidence', {'charge refund': {'topic': judged, 'refund': judged}})
    report = run(tmp_path, made, None)
    assert (report['reference']['state'], report['reference']['strata'], report['reference']['within_scope']) == ('partial', {'computed': 12, 'judged': 2}, 12)
    assert report['decision']['verdict'] == 'partial_reference' and 'robustness' not in report


@pytest.mark.parametrize('change, message', [
    (lambda t: t[0].update(mapping={'topic': {'login': 'charges', 'charges': 'charges', 'none': 'none'}}), 'bijection'),
    (lambda t: t[0].update(transformation='rename'), 'a permutation reorders'),
    (lambda t: t[0].update(expected='changed'), 'expected answer'),
    (lambda t: t[2].update(input={'message': 'charge only'}), 'distinct input of the parent family'),
    (lambda t: t[2].update(parent={'message': 'charge refund'}, input={'message': 'charge only'}), 'not meaning-preserving'),
    (lambda t: t[3].update(targets=['topic']), 'must change its declared targets'),
    (lambda t: t[3].update(input={'message': 'unlabelled'}), 'not a labelled check-role input'),
    (lambda t: t[1]['locks'].pop('baseline'), 'one transformed lock for every arm')])
def test_robustness_manifest_is_validated_before_any_scoring(tmp_path, change, message):
    made = arms(tmp_path)
    locks = {kind: {arm: str(made[f'{arm}-{kind}']) for arm in ('baseline', 'candidate')} for kind in ('perm', 'rename')}
    transforms = json.loads(json.dumps(files.read_json(manifest(tmp_path, locks))['transforms']))
    change(transforms)
    path = tmp_path / 'changed.json'
    files.write_new(path, {'kind': 'robustness-transforms', 'schema': 1, 'transforms': transforms})
    with pytest.raises(ConfigError, match=message):
        run(tmp_path, made, path)
    assert not (tmp_path / 'compared').exists()


def test_option_paraphrase_is_not_presumed_equivalent(tmp_path):
    made = arms(tmp_path)
    made['candidate-para'] = locked(tmp_path / 'para', FakeScorer(name='candidate', oracle=good), {'login': 'trouble signing in', 'charges': OPTIONS['charges']})
    made['baseline-para'] = locked(tmp_path / 'para-b', FakeScorer(name='baseline', oracle=stubborn), {'login': 'trouble signing in', 'charges': OPTIONS['charges']})
    path = manifest(tmp_path, {}, [{'id': 'para', 'transformation': 'permutation', 'expected': 'invariant', 'parent': {'message': 'hello silent'},
                                    'locks': {arm: str(made[f'{arm}-para']) for arm in ('baseline', 'candidate')},
                                    'mapping': {'topic': {'login': 'login', 'charges': 'charges', 'none': 'none'}}}])
    with pytest.raises(ConfigError, match='option paraphrases are not presumed equivalent'):
        run(tmp_path, made, path)


def test_comparison_accepts_only_confirmations_of_the_frozen_task_meaning_and_freezes_that_binding(tmp_path):
    from semantic_switch.compiler import compile_spec
    plan, data = compile_spec(spec(), 'triage.yaml'), {'message': 'charge refund'}

    def reviewed(criterion):
        entry = Verification('label', 'ana', Actor('agent', 'claude-opus-5-5', 's-1'), 'review', criterion, reviewed_content('label', data, ('topic',), 'charges'), 'confirm')
        return [{'derivation': 'judged', 'producer': 'openrouter:teacher', 'verifications': [entry.record()]}]
    found = {}
    for name, criterion in (('current', plan.fact('topic').meaning_rev), ('old', 'meaning-0')):
        made = arms(tmp_path / name)
        evidence(tmp_path / name / 'evidence', {'charge refund': {'topic': reviewed(criterion), 'refund': COMPUTED}})
        found[name] = run(tmp_path / name, made, None)
    for name, state, verdict in (('current', 'verified', 'established'), ('old', 'judged', 'partial')):
        report = found[name]
        target = next(row['targets']['topic'] for row in report['rows']['candidate'] if row['targets']['refund']['gold'] == 'true' and
                      row['targets']['topic']['gold'] == 'charges')
        assert target['reference']['state'] == state and target['reference']['binding'] == {'revision': 2, 'label': [plan.fact('topic').meaning_rev], 'evidence': []}
        assert report['reference']['state'] == verdict and report['plan']['metrics']['assessment'] == 2
        assert report['label_provenance']['assessments'].get('verified', 0) == (name == 'current')
    assert found['old']['decision']['verdict'] == 'partial_reference'
