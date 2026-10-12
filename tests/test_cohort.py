from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path

import pytest

from semantic_switch import Runtime, files, judge_from
from semantic_switch.artifacts import artifact_digest, read_export
from semantic_switch.errors import ConfigError
from semantic_switch.tools.data import Labels
from bench.policy_adaptation import ROLES, SMALL_CONTEXT, cohort_pilot, materialize_cohort
from bench import policy_workflow
from bench.policy_workflow import CHUNK, chunks, cohort_cases, cohort_summary, judge_cases, primitive, run_cohort
from bench.receipts import RecordingTransport
from .conftest import FakeScorer
from .test_openrouter_judge import Provider


@pytest.fixture
def cohort(tmp_path):
    fake = FakeScorer()
    with Runtime({'judge': fake}) as runtime:
        yield runtime, fake, tmp_path, *cohort_cases(runtime, tmp_path / 'cohort')


def test_cohort_maps_authored_gold_and_freezes_one_family_assignment(cohort):
    runtime, _, tmp_path, assignment, cases = cohort
    assert len(cases) == 256 and len(assignment['situations']) == 32 == len(set(assignment['situations'].values()))
    assert sum(count['observations'] for count in assignment['counts'].values()) == 256 and all(count['families'] for count in assignment['counts'].values())
    assert assignment['target_precision'] == .95 and assignment['minimum_coverage'] == .5 and assignment['objective'] == 'specialization'
    for situation, family in assignment['situations'].items():
        members = [case for case in cases if case['situation'] == situation]
        assert len(members) == 8 and {case['role'] for case in members} == {assignment['families'][family]}
    expected = {(case['policy'], json.dumps(case['gold'])): case['expected'] for case in cases}
    assert expected[('duplicate', '"unstated"')] == {'value': None, 'raw_value': None, 'status': 'needs_input'}
    assert expected[('duplicate', 'false')]['raw_value'] is False
    assert expected[('routing_v1', '"clarify"')] == {'value': 'clarify', 'raw_value': 'clarify', 'status': 'ok'}
    assert all(('[retrieved] reference:' in case['evidence']) == ('reference' in case['input']) for case in cases)
    assert materialize_cohort(runtime, tmp_path / 'cohort')[0] == assignment
    data = json.loads(SMALL_CONTEXT.read_text())
    data['situations'][0]['variants'][0]['gold'] = 'technical'
    files.publish(tmp_path / 'changed.json', data)
    with pytest.raises(ConfigError, match='changed after its family assignment was frozen'):
        materialize_cohort(runtime, tmp_path / 'cohort', tmp_path / 'changed.json')


def test_revised_cohort_records_adjudication_and_directories_keep_their_frozen_source(tmp_path):
    root, path, chain = Path(__file__).parents[1], SMALL_CONTEXT, []
    while 'adjudication' in (data := json.loads(path.read_text())):
        adjudication = data['adjudication']
        chain.append({item['case'] for item in adjudication['items']})
        path = root / adjudication['supersedes']['source']
        assert artifact_digest(json.loads(path.read_text())) == adjudication['supersedes']['source_sha256']
    assert path.name == 'small-context-cohort.json' and len(chain) == 2
    assert chain[0] >= {'security-09:contradictory', 'invoice-01:boundary', 'invoice-01:base', 'invoice-09:quoted', 'invoice-09:procedure', 'invoice-09:boundary'}
    assert chain[1] >= {'routing-10:unstated', 'security-09:contradictory', 'invoice-01:boundary', 'invoice-09:procedure'}
    adjudication = json.loads(SMALL_CONTEXT.read_text())['adjudication']
    superseded = root / adjudication['supersedes']['source']
    with Runtime({'judge': FakeScorer()}) as runtime:
        old = materialize_cohort(runtime, tmp_path / 'old', superseded)[0]
        assert materialize_cohort(runtime, tmp_path / 'old')[0] == old and old['source_sha256'] == adjudication['supersedes']['source_sha256']
        assert materialize_cohort(runtime, tmp_path / 'new')[0]['source_sha256'] == artifact_digest(json.loads(SMALL_CONTEXT.read_text()))


def test_inline_cases_present_the_saved_fact(cohort):
    runtime, fake, _, assignment, cases = cohort
    switch = runtime.load(assignment['definition'], draft=True)
    for case in {case['fact']: case for case in cases}.values():
        inline = primitive(case['authored'], case['guidance'])(case['evidence'], runtime)
        asked = fake.requests[-1]
        saved = switch.score_facts([case['input']], sources=('judge',), facts=[case['fact']])[0][case['fact']]
        assert fake.requests[-1] == asked and dict(saved.scores) == dict(inline.scores)


def test_raised_chunk_becomes_failed_submitted_rows_with_elapsed_time(cohort):
    from types import SimpleNamespace
    from bench.policy_workflow import evaluated
    runtime, _, _, _, cases = cohort
    _, item, members = next(chunks(cases, 'guided', CHUNK))

    def broken(evidence, runtime):
        raise RuntimeError('backend exploded')
    rows, record = judge_cases(runtime, SimpleNamespace(run=broken, guidance=item.guidance, describe=item.describe), members)
    assert len(rows) == len(members) and record['error'] == {'type': 'RuntimeError', 'stage': 'execute'} and record['requests'] == []
    assert all(row['outcome'] == 'failed' and row['ms'] >= 0 and row['error']['type'] == 'RuntimeError' and row['result']['status'] == 'failed' for row in rows)
    scored = [evaluated({**row, 'condition': 'guided'}) for row in rows]
    assert all(row['outcome'] == 'failed' and row['correct'] is None and not row['accepted'] for row in scored)
    assert cohort_summary(rows)['metrics']['statuses']['failed'] == len(members)


def test_guidance_blind_judge_fails_exactly_the_procedure_contrasts(cohort):
    runtime, fake, _, assignment, cases = cohort
    switch = runtime.load(assignment['definition'], draft=True)
    first = {}
    for case in cases:
        first.setdefault(case['evidence'], case)

    def oracle(state, question, width):
        case = first[max((evidence for evidence in first if evidence in state), key=len)]
        top = switch.plan.fact(case['fact']).index(case['gold'])
        return [.97 if k == top else .03 / (width - 1) for k in range(width)]
    fake.oracle = oracle
    rows = [row for _, item, members in chunks(cases, 'guided', CHUNK) for row in judge_cases(runtime, item, members)[0]]
    wrong = {case['case_id'] for case in cases if first[case['evidence']]['gold'] != case['gold']}
    assert len(wrong) == 14 and all('procedure' in case['tags'] for case in cases if case['case_id'] in wrong)
    summary = cohort_summary(rows)
    assert summary['metrics']['raw_correct'] == 242 and summary['metrics']['wrong_accepted'] == 14 and summary['metrics']['families'] == 32
    assert summary['metrics']['statuses']['needs_input'] == sum(case['expected']['status'] == 'needs_input' for case in cases)
    assert summary['phenomena']['procedure']['raw_correct'] == summary['phenomena']['procedure']['submitted'] - 14
    assert sum(metrics['submitted'] for metrics in summary['roles'].values()) == 256
    by_id = {case['case_id']: case for case in cases}
    failed = [pair for pair in summary['contrasts']['pairs'] if not pair['correct']]
    assert summary['contrasts']['submitted'] == 192 and {pair['left'] for pair in failed} | {pair['right'] for pair in failed} >= wrong
    assert all(pair['observed'] == 'preserve' for pair in failed if by_id[pair['left']]['evidence'] == by_id[pair['right']]['evidence'])


def test_pilot_consumes_the_assignment_with_disjoint_family_roles(cohort, monkeypatch):
    from .test_adaptation_contracts import pipeline_base
    runtime, _, tmp_path, assignment, _ = cohort
    import bench.policy_adaptation as module
    with pytest.raises(ConfigError, match='without fit support: routing_v1: clarify'):
        cohort_pilot(runtime, tmp_path / 'pilot', tmp_path / 'cohort', assets=pipeline_base(tmp_path, monkeypatch))
    prepared = []
    monkeypatch.setattr(module, 'lora_job', lambda definition, export, directory, **settings: prepared.append((export, settings)) or
                        {'job': str(directory / 'job.yaml'), 'commands': {'select': ['semantic-trainer', 'select']}})
    pilot = cohort_pilot(runtime, tmp_path / 'pilot', tmp_path / 'cohort', assets='a' * 64)
    assert pilot['assignment']['assignment_sha256'] == assignment['assignment_sha256']
    assert pilot['groups'] == {role: sorted(family for family, value in assignment['families'].items() if value == role) for role in ROLES}
    owners = [group for values in pilot['groups'].values() for group in values]
    assert len(owners) == len(set(owners)) == 32
    assert [settings.get('modules') for _, settings in prepared] == [None, ['q_proj', 'v_proj']] and prepared[0][0] == prepared[1][0]
    assert set(pilot['jobs']) == {'all-linear', 'query-value'}
    _, rows = read_export(prepared[0][0], fitting=False, native=True)
    assert {row['group'] for row in rows} == {group for role in ROLES[:3] for group in pilot['groups'][role]}
    folds = {'fit': (0, 1, 2), 'model_select': (3,), 'acceptance_fit': (4,)}
    assert all(row['group'] in pilot['groups'][role] for row in rows for role, values in folds.items() if row['fold'] in values)
    source = runtime.load(assignment['definition'], draft=True)
    held = 0
    for definition in pilot['definitions']:
        audit = Labels(runtime.load(definition['path'], draft=True))
        examples = audit.examples(definition['fact'])
        held += len(examples)
        assert audit.sw.plan.fact(definition['fact']).meaning_rev == source.plan.fact(definition['fact']).meaning_rev
        assert all(definition['case_tags'][e.id] for e in examples)
        assert all((e.split == 'check') == (assignment['families'][e.group] == 'check') and assignment['families'][e.group] != 'fit' for e in examples)
    assert held == assignment['counts']['acceptance_fit']['observations'] + assignment['counts']['check']['observations']


def fake_judged(monkeypatch, bindings):
    @contextmanager
    def judged(spec, *, provider=None, key_file=None, batch_inputs=None):
        judge, recording = bindings[spec]()
        with Runtime({'judge': judge}) as runtime:
            yield runtime, recording
    monkeypatch.setattr(policy_workflow, 'judged', judged)


def hosted(provider):
    recording = RecordingTransport(send=provider)
    return judge_from('openrouter:m', transport=recording), recording


def logged(output):
    return [entry['payload'] for entry in files.read_jsonl(f'{output}.journal.jsonl')[1:]]


def test_procedure_runner_groups_chunks_orders_and_selects_control(cohort, monkeypatch, capsys):
    _, _, tmp_path, _, cases = cohort
    local, provider = FakeScorer(), Provider()
    fake_judged(monkeypatch, {'local': lambda: (local, None), 'openrouter:m': lambda: hosted(provider)})
    output = tmp_path / 'procedure.json'
    artifact = run_cohort({'local': 'local', 'm': 'openrouter:m'}, tmp_path / 'cohort', output)
    order = [case['case_id'] for case in cases]
    assert artifact['conditions'] == ['guided'] and len(local.requests) == len(cases)
    by_id = {case['case_id']: case for case in cases}
    records = [row for row in logged(output) if 'members' in row]
    for arm in ('local', 'm'):
        own = [row for row in records if row['arm'] == arm]
        sizes = {}
        for row in own:
            groups = {(by_id[case]['judgment'], by_id[case]['guidance'].digest) for case in row['members']}
            assert len(row['members']) <= CHUNK and groups == {(row['group']['judgment'], row['group']['guidance'])}
            sizes.setdefault(groups.pop(), []).append(len(row['members']))
        assert sum(map(sum, sizes.values())) == len(cases)
        assert all(len(values) == math.ceil(sum(values) / CHUNK) for values in sizes.values())
    for arm in artifact['arms']:
        assert [row['case_id'] for row in arm['rows']] == order and 'control' not in arm
        assert arm['metrics']['submitted'] == len(cases) and arm['timing']['executed_cases'] == len(cases)
    local_arm, hosted_arm = artifact['arms']
    assert local_arm['timing']['http_attempts'] is None and hosted_arm['timing']['http_attempts']['attempts'] == len(provider.bodies) > 0
    assert hosted_arm['returned_models'] == ['m@p'] and artifact['paired'][0]['support']['matched'] + artifact['paired'][0]['support']['unmatched'] == len(cases)
    assert f'final: completed {2 * len(cases)}/{2 * len(cases)} (fresh {2 * len(cases)}, resumed 0)' in capsys.readouterr().err.splitlines()[-1]
    controlled = FakeScorer()
    fake_judged(monkeypatch, {'local': lambda: (controlled, None)})
    artifact = run_cohort({'local': 'local'}, tmp_path / 'cohort', tmp_path / 'control.json', control=True)
    arm, = artifact['arms']
    assert artifact['conditions'] == ['guided', 'omitted'] and len(controlled.requests) == 2 * len(cases)
    assert [row['case_id'] for row in arm['control']['rows']] == order
    assert all(row['task']['guidance'] for row in arm['rows']) and not any(row['task']['guidance'] for row in arm['control']['rows'])


class Interrupting(FakeScorer):
    def score(self, requests):
        if len(self.requests) >= 40:
            raise KeyboardInterrupt
        return super().score(requests)


def test_procedure_runner_resumes_retries_failures_and_refuses_changed_configuration(cohort, monkeypatch, capsys):
    _, _, tmp_path, _, cases = cohort
    output, judges = tmp_path / 'procedure.json', {'local': 'local'}
    def run(judge, **settings):
        fake_judged(monkeypatch, {'local': lambda: (judge, None)})
        return run_cohort(judges, tmp_path / 'cohort', output, chunk=8, **settings)
    with pytest.raises(KeyboardInterrupt):
        run(Interrupting())
    done = {row['case_id'] for row in logged(output) if 'case_id' in row}
    assert 0 < len(done) < len(cases)
    portuguese = [case['evidence'] for case in cases if case['language'] == 'pt']
    def oracle(state, question, width):
        if any(evidence in state for evidence in portuguese):
            raise RuntimeError('provider down')
        return [1 / width] * width
    resumed = FakeScorer(oracle)
    artifact = run(resumed)
    failing = sum(case['language'] == 'pt' for case in cases if case['case_id'] not in done)
    assert len(resumed.requests) == len(cases) - len(done) and failing
    arm, = artifact['arms']
    assert [row['case_id'] for row in arm['rows']] == [case['case_id'] for case in cases]
    assert arm['metrics']['protocol_failures'] == failing and arm['metrics']['failure_rate'] == failing / len(cases)
    assert f'final: completed {len(cases)}/{len(cases)} (fresh {len(cases) - len(done)}, resumed {len(done)})' in (last := capsys.readouterr().err.splitlines()[-1])
    assert f'failed {failing} ' in last
    idle = FakeScorer()
    assert run(idle)['arms'][0]['metrics']['protocol_failures'] == failing and not idle.requests
    retried = FakeScorer()
    artifact = run(retried, retry_failed=True)
    assert len(retried.requests) == failing and artifact['arms'][0]['metrics']['protocol_failures'] == 0
    superseded = {entry['attempt'] for entry in artifact['execution']['superseded']}
    replaced = [record['payload'] for record in files.read_jsonl(f'{output}.journal.jsonl')[1:] if record['attempt'] in superseded]
    assert len(replaced) == failing and all(row['result']['status'] == 'failed' for row in replaced)
    for changed in ({'chunk': 4}, {'control': True}):
        with pytest.raises(ConfigError, match='different run'):
            fake_judged(monkeypatch, {'local': lambda: (FakeScorer(), None)})
            run_cohort(judges, tmp_path / 'cohort', output, **{'chunk': 8, **changed})


def test_fingerprint_ignores_files_the_run_never_loaded(tmp_path):
    from bench.diagnostics import fingerprint
    before = fingerprint()
    stray = Path(__file__).parents[1] / "bench" / f"unloaded_{tmp_path.name}.py"
    try:
        stray.write_text("X = 1\n")
        assert fingerprint() == before
    finally:
        stray.unlink()
