from __future__ import annotations

from pathlib import Path
import json
import os
import uuid

from . import files
from .compiler import digest, is_digest
from .errors import ConfigError

SCHEMA = 1
FIELDS = {"schema", "kind", "sequence", "key", "attempt", "payload"}


def _key(key):
    if not is_digest(key):
        raise ConfigError(f"journal work keys are full sha256 digests of the work identity, not {key!r}")
    return key


def replay(path, superseded=None):
    path = Path(path)
    records = files.located_records(path)
    first = next(records, None)
    if first is None:
        return None, {}, 0, 0
    line, _, _, header = first
    if not isinstance(header, dict) or header.get("schema") != SCHEMA or header.get("kind") != "header" or not isinstance(header.get("run_id"), str) \
            or header.get("digest") != digest(header.get("configuration")):
        raise ConfigError(f"{path}:{line}: not a schema-{SCHEMA} execution journal header; start a new run at another path")
    current, sequence, attempts = {}, 0, 0
    for line, offset, length, record in records:
        if not isinstance(record, dict) or set(record) != FIELDS or record["schema"] != SCHEMA or record["kind"] != "completion" \
                or not is_digest(record["key"]) or not isinstance(record["attempt"], str) or not record["attempt"]:
            raise ConfigError(f"{path}:{line}: malformed journal completion record; the journal is damaged")
        if record["sequence"] != sequence + 1:
            raise ConfigError(f"{path}:{line}: journal sequence {record['sequence']} follows {sequence}; the journal is damaged")
        sequence, attempts = record["sequence"], attempts + 1
        if superseded is not None and record['key'] in current:
            old = current[record['key']]
            superseded.append({'key': record['key'], 'attempt': old['attempt'], 'sequence': old['sequence']})
        current[record["key"]] = {**{key: record[key] for key in ("sequence", "attempt")}, "offset": offset, "length": length}
    return header, current, sequence, attempts


class Journal:
    def __init__(self, path, configuration: dict):
        self.path = Path(path)
        self.lease = files.Lease(self.path.with_name(self.path.name + ".run"))
        try:
            self.configuration_digest = digest(configuration)
            self._superseded = []
            header, self.current, self.sequence, self.attempts = replay(self.path, self._superseded)
            if header is None:
                header = {"schema": SCHEMA, "kind": "header", "run_id": uuid.uuid4().hex, "configuration": configuration, "digest": self.configuration_digest}
                files.append(self.path, files.jsonl([header]))
            elif header["digest"] != self.configuration_digest:
                raise ConfigError(f"{self.path}: recorded configuration {header['digest']} differs from {self.configuration_digest}; "
                                  "this is a different run: resume with the recorded configuration or start a new run at another path")
            self.run_id = header["run_id"]
            self.fd = os.open(self.path, os.O_RDONLY)
        except BaseException:
            self.lease.release()
            raise

    def pending(self, keys, retry=None) -> list:
        return [key for key in map(_key, keys) if key not in self.current or retry is not None and retry(self.payload(key))]

    def commit(self, units) -> list:
        records = [{"schema": SCHEMA, "kind": "completion", "sequence": self.sequence + index, "key": _key(key), "attempt": uuid.uuid4().hex, "payload": payload}
                   for index, (key, payload) in enumerate(units, 1)]
        lines = [files.jsonl([record]) for record in records]
        if not lines:
            return records
        offset, _ = files.append(self.path, b''.join(lines))
        for record, line in zip(records, lines):
            self.sequence, self.attempts = record['sequence'], self.attempts + 1
            if record['key'] in self.current:
                old = self.current[record['key']]
                self._superseded.append({'key': record['key'], 'attempt': old['attempt'], 'sequence': old['sequence']})
            self.current[record['key']] = {'sequence': record['sequence'], 'attempt': record['attempt'], 'offset': offset, 'length': len(line)}
            offset += len(line)
        return records

    def payload(self, key):
        entry = self.current[key]
        record = json.loads(os.pread(self.fd, entry['length'], entry['offset']))
        if record['key'] != key or record['attempt'] != entry['attempt']:
            raise ConfigError(f'{self.path}: indexed journal record changed')
        return record['payload']

    def payloads(self):
        for key in sorted(self.current, key=lambda key: self.current[key]['sequence']):
            yield self.payload(key)

    def superseded(self):
        yield from sorted(self._superseded, key=lambda record: record['sequence'])

    def summary(self) -> dict:
        return {"journal": str(self.path), "run_id": self.run_id, "configuration_digest": self.configuration_digest, "sequence": self.sequence,
                "completed": len(self.current), "attempts": self.attempts, "superseded": list(self.superseded()),
                "delivery": "at least once: work interrupted before its record committed may repeat"}

    def publish(self, commit_path, manifest, payloads=None):
        return files.publish(commit_path, {**manifest, "execution": self.summary()}, payloads or {})

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.lease.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
