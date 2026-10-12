from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import warnings
from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
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

    @property
    def dataset(self):
        return self.yaml.with_suffix(".dataset.json")

    @property
    def audit(self):
        return self.yaml.with_suffix(".audit.jsonl")

    @classmethod
    def of(cls, path) -> Paths:
        p = Path(path)
        stem = p.parent / (p.name.rsplit(".", 1)[0] if p.suffix in (".yaml", ".yml") else p.name)
        yml = p if p.suffix in (".yaml", ".yml") else Path(f"{stem}.yaml")
        return cls(yml, Path(f"{stem}.examples.jsonl"), Path(f"{stem}.test.jsonl"), Path(f"{stem}.lock.json"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{path}: switch file not found")
    class AuthoredLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            explicit = set()
            for key, value in node.value:
                if key.tag == "tag:yaml.org,2002:merge":
                    raise ConfigError(f"{path}:{key.start_mark.line + 1}: YAML merge keys are unsupported; write explicit definitions")
                name = self.construct_object(key, deep=deep)
                try:
                    if name in explicit:
                        raise ConfigError(f"{path}:{key.start_mark.line + 1}: duplicate key {name!r}")
                    explicit.add(name)
                except TypeError:
                    raise ConfigError(f"{path}:{key.start_mark.line + 1}: mapping keys must be scalar") from None
            return super().construct_mapping(node, deep=deep)
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=AuthoredLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from None


_mutex_guard = threading.Lock()
_mutexes: dict[Path, threading.RLock] = {}
_owned: set[Path] = set()


def _holder(sidecar: Path) -> str:
    return sidecar.read_text(encoding="utf-8").strip()


def _flock(path: Path, blocking: bool, owner: str | None = None) -> int:
    sidecar = path.with_name(path.name + ".write-lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(sidecar, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ConfigError(f"{path}: locked by {_holder(sidecar)}") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps({"pid": os.getpid(), "argv": sys.argv, "started": now(), **({"owner": owner} if owner else {})}).encode(), 0)
    return fd


def _unflock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def locked(path, *, blocking: bool = True):
    path = Path(path).resolve()
    sidecar = path.with_name(path.name + ".write-lock")
    with _mutex_guard:
        mutex = _mutexes.setdefault(path, threading.RLock())
    if not mutex.acquire(blocking=blocking):
        raise ConfigError(f"{path}: locked by another thread of this process ({_holder(sidecar)})")
    try:
        if path in _owned:
            yield
            return
        fd = _flock(path, blocking)
        _owned.add(path)
        try:
            yield
        finally:
            _owned.discard(path)
            _unflock(fd)
    finally:
        mutex.release()


class Lease:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.owner = secrets.token_hex(8)
        self.pid = os.getpid()
        self._guard = threading.Lock()
        self._fd = _flock(self.path, False, self.owner)

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        with self._guard:
            fd, self._fd = self._fd, None
        if fd is not None and self.pid == os.getpid():
            _unflock(fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _json(data, **kwargs):
    from .results import thaw
    return json.dumps(thaw(data), ensure_ascii=False, sort_keys=False, allow_nan=False, **kwargs)


def _encode(path: Path, data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if path.suffix in (".yaml", ".yml"):
        from .results import thaw
        return yaml.safe_dump(thaw(data), sort_keys=False, allow_unicode=True).encode("utf-8")
    return (_json(data, indent=1) + "\n").encode("utf-8")


def _sync_dir(path: Path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _staged(target: Path, chunks, temps: list) -> tuple[Path, str, int]:
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    temps.append(tmp)
    checksum, count = hashlib.sha256(), 0
    try:
        if target.exists():
            os.fchmod(fd, target.stat().st_mode & 0o777)
        for chunk in chunks:
            data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            write_all(fd, data)
            checksum.update(data)
            count += data.count(b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return tmp, checksum.hexdigest(), count


def _generation(name: str) -> re.Pattern:
    path = Path(name)
    return re.compile(re.escape(path.stem) + r"\.[0-9a-f]{16}" + re.escape(path.suffix))


def _stored(name: str, digest: str) -> str:
    path = Path(name)
    return f"{path.stem}.{digest[:16]}{path.suffix}"


def publish(commit_path, manifest, payloads: Mapping[str, Iterable[bytes | str]] = {}):
    commit_path = Path(commit_path)
    directory = commit_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temps: list = []
    with locked(commit_path):
        try:
            listed = {}
            for name, chunks in payloads.items():
                tmp, digest, count = _staged(directory / name, chunks, temps)
                listed[name] = {"file": _stored(name, digest), "sha256": digest, "count": count}
                os.replace(tmp, directory / listed[name]["file"])
            committed = {**manifest, "payloads": listed} if payloads else manifest
            tmp, _, _ = _staged(commit_path, [_encode(commit_path, committed)], temps)
            if payloads:
                _sync_dir(directory)
            os.replace(tmp, commit_path)
            _sync_dir(directory)
            return committed
        finally:
            for tmp in temps:
                if tmp.exists():
                    tmp.unlink()


def write_new(path, data) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encode(path, data)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(payload).hexdigest()


def publish_directory(staged, destination) -> Path:
    staged, destination = Path(staged), Path(destination)
    if staged.parent.resolve() != destination.parent.resolve():
        raise ConfigError(f"{staged}: stage beside {destination} so publication is one same-filesystem rename")
    for path in sorted(staged.rglob("*"), reverse=True):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        elif path.is_dir():
            _sync_dir(path)
    _sync_dir(staged)
    if destination.exists():
        raise ConfigError(f"{destination}: already published; publish into a new destination")
    os.rename(staged, destination)
    _sync_dir(destination.parent)
    return destination


def read_published(commit_path) -> dict | None:
    commit_path = Path(commit_path)
    manifest = read_json(commit_path)
    if any(not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or not _generation(name).fullmatch(entry["file"])
           for name, entry in (manifest or {}).get("payloads", {}).items()):
        raise ConfigError(f"{commit_path}: invalid publication payloads; re-run")
    return manifest


def _verify(commit_path: Path, fh, entry) -> None:
    checksum, count = hashlib.sha256(), 0
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        checksum.update(chunk)
        count += chunk.count(b"\n")
    if checksum.hexdigest() != entry["sha256"] or count != entry["count"]:
        raise ConfigError(f"{commit_path}: incomplete publication; re-run")
    fh.seek(0)


@contextmanager
def open_published(commit_path):
    commit_path = Path(commit_path)
    manifest = read_published(commit_path)
    with ExitStack() as stack:
        try:
            handles = {name: stack.enter_context((commit_path.parent / entry["file"]).open("rb"))
                       for name, entry in (manifest or {}).get("payloads", {}).items()}
        except FileNotFoundError:
            raise ConfigError(f"{commit_path}: a payload generation it names is missing; re-run the publication") from None
        for name, fh in handles.items():
            _verify(commit_path, fh, manifest["payloads"][name])
        yield manifest, handles


def prune(directory) -> list[str]:
    directory = Path(directory)
    referenced, generations = set(), []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix == ".json":
            try:
                manifest = read_json(path)
            except ConfigError:
                continue
            if isinstance(manifest, dict) and isinstance(manifest.get("payloads"), dict):
                referenced.update(entry.get("file") for entry in manifest["payloads"].values() if isinstance(entry, dict))
                generations += [(name, path) for name in manifest["payloads"]]
    removed = sorted({candidate.name for name, _ in generations for candidate in directory.iterdir()
                      if _generation(name).fullmatch(candidate.name) and candidate.name not in referenced})
    for name in removed:
        (directory / name).unlink()
    if removed:
        _sync_dir(directory)
    return removed


def _torn(fragment: bytes) -> bool:
    try:
        json.loads(fragment)
        return False
    except ValueError:
        return True


def _line_end(fd, size: int) -> int:
    end = size
    while end:
        start = max(0, end - (1 << 16))
        at = os.pread(fd, end - start, start).rfind(b"\n")
        if at >= 0:
            return start + at + 1
        end = start
    return 0


def append(path, records) -> None:
    payload = "".join(_json(r) + "\n" for r in records).encode("utf-8")
    if not payload:
        return
    path = Path(path)
    with locked(path):
        created = not path.exists()
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            size = os.fstat(fd).st_size
            cut = _line_end(fd, size)
            if cut < size:
                if _torn(os.pread(fd, size - cut, cut)):
                    os.ftruncate(fd, cut)
                else:
                    payload = b"\n" + payload
            write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            _sync_dir(path.parent)


def iter_records(path, *, with_lines: bool = False):
    path = Path(path)
    if not path.exists():
        return
    with path.open("rb") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, TypeError) as e:
                if not line.endswith(b"\n") and _torn(line):
                    return
                raise ConfigError(f"{path}:{n}: invalid JSON ({e})") from None
            yield (n, value) if with_lines else value


def read_jsonl(path) -> list[dict]:
    return list(iter_records(path))


def read_json(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as e:
        raise ConfigError(f"{path}: invalid JSON ({e})") from None


def cache_dir() -> Path:
    return Path(os.environ.get("SEMANTIC_SWITCH_CACHE", Path.home() / ".cache" / "semantic_switch"))


def _cache_entry(kind: str, key: str) -> Path:
    return cache_dir() / kind / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")


@contextmanager
def _optional_cache(action: str):
    try:
        yield
    except OSError as e:
        if not isinstance(e, PermissionError) and e.errno != errno.EROFS:
            raise
        warnings.warn(f"cache {action} unavailable ({e}); continuing without the cache", stacklevel=4)


def read_cached(kind: str, key: str):
    with _optional_cache("lookup"):
        try:
            entry = read_json(_cache_entry(kind, key))
        except ConfigError as e:
            warnings.warn(f"{e}; ignoring damaged cache entry", stacklevel=2)
            return None
        return entry["value"] if isinstance(entry, dict) and entry.get("key") == key else None
    return None


def write_cached(kind: str, key: str, value) -> None:
    with _optional_cache("publication"):
        publish(_cache_entry(kind, key), {"key": key, "value": value})


def sha256_file(path) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{st.st_ctime_ns}|{st.st_ino}"
    known = read_cached("sha256", key)
    if known is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        known = h.hexdigest()
        write_cached("sha256", key, known)
    return known
