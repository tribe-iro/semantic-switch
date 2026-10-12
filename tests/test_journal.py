from __future__ import annotations

import json
import threading

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.compiler import digest
from semantic_switch.journal import Journal, replay


def key(name):
    return digest(['work', name])


def test_independent_resumptions_contend_and_release_with_the_journal(tmp_path):
    path = tmp_path / 'run.journal.jsonl'
    with Journal(path, {'task': 'a'}) as journal:
        errors = []
        thread = threading.Thread(target=lambda: errors.append(pytest.raises(ConfigError, Journal, path, {'task': 'a'})))
        thread.start()
        thread.join()
        with pytest.raises(ConfigError, match='locked by .*"owner"'):
            Journal(path, {'task': 'a'})
        journal.commit([(key('one'), {'value': 1})])
    with Journal(path, {'task': 'a'}) as resumed:
        assert resumed.pending([key('one'), key('two')]) == [key('two')] and resumed.run_id == journal.run_id


def test_retry_appends_a_new_attempt_and_streams_superseded_ones(tmp_path):
    path = tmp_path / 'run.journal.jsonl'
    with Journal(path, {'task': 'a'}) as journal:
        journal.commit([(key('one'), {'outcome': 'failed'}), (key('two'), {'outcome': 'uncertain'})])
        failed = lambda payload: payload['outcome'] == 'failed'
        assert journal.pending([key('one'), key('two')]) == [] and journal.pending([key('one'), key('two')], failed) == [key('one')]
        first = journal.current[key('one')]['attempt']
        journal.commit([(key('one'), {'outcome': 'answered'})])
        summary = journal.summary()
    assert journal.payloads() == [{'outcome': 'uncertain'}, {'outcome': 'answered'}]
    assert summary['superseded'] == [{'key': key('one'), 'attempt': first, 'sequence': 1}] and summary['attempts'] == 3 and summary['completed'] == 2
    assert [record['sequence'] for record in files.read_jsonl(path)[1:]] == [1, 2, 3]


def test_complete_prefix_survives_a_torn_last_unit(tmp_path):
    path = tmp_path / 'run.journal.jsonl'
    with Journal(path, {'task': 'a'}) as journal:
        journal.commit([(key('one'), {'value': 1}), (key('two'), {'value': 2})])
    with open(path, 'a') as stream:
        stream.write('{"schema": 1, "kind": "completion", "sequence": 3, "key": "')
    with Journal(path, {'task': 'a'}) as resumed:
        assert resumed.pending([key(n) for n in ('one', 'two', 'three')]) == [key('three')]
        resumed.commit([(key('three'), {'value': 3})])
    header, current, sequence, attempts = replay(path)
    assert sequence == attempts == 3 and len(current) == 3 and header['configuration'] == {'task': 'a'}


@pytest.mark.parametrize('damage, match', [
    (lambda rows: rows[1].update(sequence=5), 'sequence 5 follows 0'),
    (lambda rows: rows[1].update(schema=0), 'malformed journal completion'),
    (lambda rows: rows[1].update(key='one'), 'malformed journal completion'),
    (lambda rows: rows[0].update(schema=2), 'not a schema-1 execution journal header'),
    (lambda rows: rows[0]['configuration'].update(task='b'), 'not a schema-1 execution journal header'),
])
def test_fingerprint_schema_and_interior_corruption_reject(tmp_path, damage, match):
    path = tmp_path / 'run.journal.jsonl'
    with Journal(path, {'task': 'a'}) as journal:
        journal.commit([(key('one'), {'value': 1}), (key('two'), {'value': 2})])
    rows = files.read_jsonl(path)
    damage(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ConfigError, match=match):
        Journal(path, {'task': 'a'})
    path.write_text(path.read_text().replace('"sequence": 2', '"sequence": 2,,'))
    with pytest.raises(ConfigError):
        Journal(path, {'task': 'a'})


def test_changed_configuration_is_another_run_and_keys_are_digests(tmp_path):
    path = tmp_path / 'run.journal.jsonl'
    with Journal(path, {'task': 'a'}) as journal:
        with pytest.raises(ConfigError, match='full sha256 digests'):
            journal.commit([('case-1', {})])
    with pytest.raises(ConfigError, match='this is a different run'):
        Journal(path, {'task': 'b'})


def test_later_publication_retains_referenced_generations_and_prune_is_explicit(tmp_path, monkeypatch):
    commit = tmp_path / 'labels.manifest.json'
    first = files.publish(commit, {'run': 1}, {'labels.jsonl': ['{"n": 1}\n']})
    second = files.publish(commit, {'run': 2}, {'labels.jsonl': ['{"n": 2}\n']})
    old = tmp_path / first['payloads']['labels.jsonl']['file']
    assert old.exists() and (tmp_path / second['payloads']['labels.jsonl']['file']).exists()
    import os
    real = os.replace
    def crash(source, target):
        if str(target) == str(commit):
            raise OSError('crash before the manifest rename')
        return real(source, target)
    monkeypatch.setattr(os, 'replace', crash)
    with pytest.raises(OSError):
        files.publish(commit, {'run': 3}, {'labels.jsonl': ['{"n": 3}\n']})
    monkeypatch.undo()
    with files.open_published(commit) as (manifest, handles):
        assert manifest['run'] == 2 and handles['labels.jsonl'].read() == b'{"n": 2}\n'
    removed = files.prune(tmp_path)
    assert old.name in removed and not old.exists() and (tmp_path / second['payloads']['labels.jsonl']['file']).exists()
    with files.open_published(commit) as (manifest, _):
        assert manifest['run'] == 2
    (tmp_path / second['payloads']['labels.jsonl']['file']).unlink()
    with pytest.raises(ConfigError, match='payload generation it names is missing'):
        with files.open_published(commit):
            pass
