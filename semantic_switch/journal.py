from __future__ import annotations

from pathlib import Path
import uuid

from . import files
from .compiler import digest
from .errors import ConfigError

SCHEMA = 1
FIELDS = {"schema", "kind", "sequence", "key", "attempt", "payload"}


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _key(key):
    if not _digest(key):
        raise ConfigError(f"journal work keys are full sha256 digests of the work identity, not {key!r}")
    return key


def replay(path):
    path = Path(path)
    records = files.iter_records(path, with_lines=True)
    first = next(records, None)
    if first is None:
        return None, {}, 0, 0
    line, header = first
    if not isinstance(header, dict) or header.get("schema") != SCHEMA or header.get("kind") != "header" or not isinstance(header.get("run_id"), str) \
            or header.get("digest") != digest(header.get("configuration")):
        raise ConfigError(f"{path}:{line}: not a schema-{SCHEMA} execution journal header; start a new run at another path")
    current, sequence, attempts = {}, 0, 0
    for line, record in records:
        if not isinstance(record, dict) or set(record) != FIELDS or record["schema"] != SCHEMA or record["kind"] != "completion" \
                or not _digest(record["key"]) or not isinstance(record["attempt"], str) or not record["attempt"]:
            raise ConfigError(f"{path}:{line}: malformed journal completion record; the journal is damaged")
        if record["sequence"] != sequence + 1:
            raise ConfigError(f"{path}:{line}: journal sequence {record['sequence']} follows {sequence}; the journal is damaged")
        sequence, attempts = record["sequence"], attempts + 1
        current[record["key"]] = record
    return header, current, sequence, attempts


class Journal:
    def __init__(self, path, configuration: dict):
        self.path = Path(path)
        self.lease = files.Lease(self.path.with_name(self.path.name + ".run"))
        try:
            self.configuration_digest = digest(configuration)
            header, self.current, self.sequence, self.attempts = replay(self.path)
            if header is None:
                header = {"schema": SCHEMA, "kind": "header", "run_id": uuid.uuid4().hex, "configuration": configuration, "digest": self.configuration_digest}
                files.append(self.path, [header])
            elif header["digest"] != self.configuration_digest:
                raise ConfigError(f"{self.path}: recorded configuration {header['digest']} differs from {self.configuration_digest}; "
                                  "this is a different run: resume with the recorded configuration or start a new run at another path")
            self.run_id = header["run_id"]
        except BaseException:
            self.lease.release()
            raise

    def pending(self, keys, retry=None) -> list:
        return [key for key in map(_key, keys) if key not in self.current or retry is not None and retry(self.current[key]["payload"])]

    def commit(self, units) -> list:
        records = [{"schema": SCHEMA, "kind": "completion", "sequence": self.sequence + index, "key": _key(key), "attempt": uuid.uuid4().hex, "payload": payload}
                   for index, (key, payload) in enumerate(units, 1)]
        files.append(self.path, records)
        for record in records:
            self.sequence, self.attempts = record["sequence"], self.attempts + 1
            self.current[record["key"]] = record
        return records

    def payloads(self) -> list:
        return [record["payload"] for record in sorted(self.current.values(), key=lambda record: record["sequence"])]

    def superseded(self):
        for _, record in files.iter_records(self.path, with_lines=True):
            if record.get("kind") == "completion" and record["sequence"] <= self.sequence and self.current[record["key"]]["attempt"] != record["attempt"]:
                yield {"key": record["key"], "attempt": record["attempt"], "sequence": record["sequence"]}

    def summary(self) -> dict:
        return {"journal": str(self.path), "run_id": self.run_id, "configuration_digest": self.configuration_digest, "sequence": self.sequence,
                "completed": len(self.current), "attempts": self.attempts, "superseded": list(self.superseded()),
                "delivery": "at least once: work interrupted before its record committed may repeat"}

    def publish(self, commit_path, manifest, payloads=None):
        return files.publish(commit_path, {**manifest, "execution": self.summary()}, payloads or {})

    def close(self):
        self.lease.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
