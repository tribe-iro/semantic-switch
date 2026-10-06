from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .errors import ConfigError


@dataclass(frozen=True)
class Paths:
    yaml: Path
    examples: Path
    test: Path
    lock: Path

    @classmethod
    def of(cls, path) -> Paths:
        p = Path(path)
        stem = p.parent / (p.name.rsplit(".", 1)[0] if p.suffix in (".yaml", ".yml") else p.name)
        yml = p if p.suffix in (".yaml", ".yml") else Path(f"{stem}.yaml")
        return cls(yml, Path(f"{stem}.examples.jsonl"), Path(f"{stem}.test.jsonl"), Path(f"{stem}.lock.json"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"{path}: switch file not found")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cache_dir() -> Path:
    d = Path(os.environ.get("SEMANTIC_SWITCH_CACHE", Path.home() / ".cache" / "semantic_switch"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: str) -> str:
    import hashlib

    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    store_path = cache_dir() / "sha256.json"
    store = read_json(store_path) or {}
    if key not in store:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        store[key] = h.hexdigest()
        write_json_atomic(store_path, store)
    return store[key]


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
