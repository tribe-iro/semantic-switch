from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import warnings
import weakref
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
        stem = p.parent / (p.name.removesuffix(".lock.json") if p.name.endswith(".lock.json") else
                           p.name.rsplit(".", 1)[0] if p.suffix in (".yaml", ".yml") else p.name)
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
_mutexes: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_lock_roots: dict = {}


class _PathLock:
    __slots__ = ("rlock", "fd", "depth", "__weakref__")

    def __init__(self):
        self.rlock, self.fd, self.depth = threading.RLock(), None, 0


def lock_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = _lock_roots.get(base)
    if root is None:
        root = Path(base) / "semantic_switch" / "locks" if base else Path(tempfile.gettempdir()) / f"semantic_switch-{os.getuid()}" / "locks"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.getuid():
            raise ConfigError(f"{root}: lock directory belongs to another user; set XDG_RUNTIME_DIR to a private runtime directory")
        _lock_roots[base] = root
    return root


def lock_file(path) -> Path:
    return lock_dir() / (hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:32] + ".lock")


def _holder(path: Path) -> str:
    try:
        return lock_file(path).read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown holder"


def _flock(path: Path, blocking: bool, owner: str | None = None) -> int:
    fd = os.open(lock_file(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ConfigError(f"{path}: locked by {_holder(path)}") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps({"path": str(path), "pid": os.getpid(), "argv": sys.argv, "started": now(), **({"owner": owner} if owner else {})}).encode(), 0)
    return fd


def _unflock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def locked(path, *, blocking: bool = True):
    path = Path(path).resolve()
    with _mutex_guard:
        entry = _mutexes.get(path)
        if entry is None:
            entry = _mutexes[path] = _PathLock()
    if not entry.rlock.acquire(blocking=blocking):
        raise ConfigError(f"{path}: locked by another thread of this process ({_holder(path)})")
    try:
        if not entry.depth:
            entry.fd = _flock(path, blocking)
        entry.depth += 1
        try:
            yield
        finally:
            entry.depth -= 1
            if not entry.depth:
                fd, entry.fd = entry.fd, None
                _unflock(fd)
    finally:
        entry.rlock.release()


def committing(directory):
    return locked(Path(directory) / ".commit")


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
    from .context import json_default
    return json.dumps(data, ensure_ascii=False, sort_keys=False, allow_nan=False, default=json_default, **kwargs)


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


class _Payload:
    def __init__(self, target, temps):
        self.target = target
        self.tmp = target.with_name(f'.{target.name}.{os.getpid()}.{threading.get_ident()}.tmp')
        fd = os.open(self.tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        temps.append(self.tmp)
        if target.exists():
            os.fchmod(fd, target.stat().st_mode & 0o777)
        self.stream = os.fdopen(fd, 'wb')
        self.checksum, self.count = hashlib.sha256(), 0

    def write(self, chunk):
        data = chunk.encode('utf-8') if isinstance(chunk, str) else chunk
        self.stream.write(data)
        self.checksum.update(data)
        self.count += data.count(b'\n')

    def finish(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return self.tmp, self.checksum.hexdigest(), self.count


def _staged(target, chunks, temps):
    payload = _Payload(target, temps)
    try:
        for chunk in chunks:
            payload.write(chunk)
        return payload.finish()
    finally:
        payload.stream.close()


def _generation(name: str) -> re.Pattern:
    path = Path(name)
    return re.compile(re.escape(path.stem) + r"\.[0-9a-f]{16}" + re.escape(path.suffix))


def _stored(name: str, digest: str) -> str:
    path = Path(name)
    return f"{path.stem}.{digest[:16]}{path.suffix}"


class Publication:
    def __init__(self, commit_path):
        self.path = Path(commit_path)
        self.directory = self.path.parent
        self.temps, self.outputs = [], {}

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = locked(self.path)
        self.lock.__enter__()
        return self

    def payload(self, name):
        if name not in self.outputs:
            self.outputs[name] = _Payload(self.directory / name, self.temps)
        return self.outputs[name]

    def commit(self, manifest):
        staged, listed = [], {}
        for name, payload in self.outputs.items():
            tmp, checksum, count = payload.finish()
            listed[name] = {'file': _stored(name, checksum), 'sha256': checksum, 'count': count}
            staged.append((tmp, self.directory / listed[name]['file']))
        committed = {**manifest, 'payloads': listed} if listed else manifest
        tmp, _, _ = _staged(self.path, [_encode(self.path, committed)], self.temps)
        with committing(self.directory):
            for source, target in staged:
                os.replace(source, target)
            if staged:
                _sync_dir(self.directory)
            os.replace(tmp, self.path)
            _sync_dir(self.directory)
        return committed

    def __exit__(self, *error):
        try:
            for payload in self.outputs.values():
                payload.stream.close()
            for tmp in self.temps:
                if tmp.exists():
                    tmp.unlink()
        finally:
            self.lock.__exit__(*error)


def publish(commit_path, manifest, payloads: Mapping[str, Iterable[bytes | str]] = {}):
    with Publication(commit_path) as publication:
        for name, chunks in payloads.items():
            sink = publication.payload(name)
            for chunk in chunks:
                sink.write(chunk)
        return publication.commit(manifest)


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
    with ExitStack() as stack:
        with committing(commit_path.parent):
            manifest = read_published(commit_path)
            try:
                handles = {name: stack.enter_context((commit_path.parent / entry["file"]).open("rb"))
                           for name, entry in (manifest or {}).get("payloads", {}).items()}
            except FileNotFoundError:
                raise ConfigError(f"{commit_path}: a payload generation it names is missing; re-run the publication") from None
        for name, fh in handles.items():
            _verify(commit_path, fh, manifest["payloads"][name])
        yield manifest, handles


GENERATION = re.compile(r"(?P<stem>.+)\.[0-9a-f]{16}(?P<suffix>(?:\.[^.]*)?)")


def prune(directory, *, preserve=()) -> list[str]:
    directory = Path(directory)
    with committing(directory):
        entries = [path for path in directory.iterdir() if path.is_file()]
        referenced, names = set(preserve), set()
        for path in entries:
            if path.suffix != ".json":
                continue
            try:
                manifest = read_json(path)
            except ConfigError:
                continue
            if isinstance(manifest, dict) and isinstance(manifest.get("payloads"), dict):
                referenced.update(entry.get("file") for entry in manifest["payloads"].values() if isinstance(entry, dict))
                names.update(manifest["payloads"])
        removed = sorted(path.name for path in entries if path.name not in referenced and (path.name.endswith(".write-lock") or
                         (found := GENERATION.fullmatch(path.name)) is not None and found["stem"] + found["suffix"] in names))
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


def jsonl(records) -> bytes:
    return "".join(_json(record) + "\n" for record in records).encode("utf-8")


def append(path, data: bytes):
    if type(data) is not bytes or data and not data.endswith(b"\n"):
        raise TypeError("files.append takes complete encoded JSONL bytes; encode records with files.jsonl")
    if not data:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked(path):
        created = not path.exists()
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            size = os.fstat(fd).st_size
            cut = _line_end(fd, size)
            if cut < size:
                if _torn(os.pread(fd, size - cut, cut)):
                    os.ftruncate(fd, cut)
                    size = cut
                else:
                    write_all(fd, b"\n")
                    size += 1
            write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            _sync_dir(path.parent)
        return size, size + len(data)


def located_records(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open('rb') as fh:
        offset = 0
        for n, line in enumerate(fh, 1):
            start, offset = offset, offset + len(line)
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, TypeError) as error:
                if not line.endswith(b'\n') and _torn(line):
                    return
                raise ConfigError(f'{path}:{n}: invalid JSON ({error})') from None
            yield n, start, len(line), value


def iter_records(path, *, with_lines=False):
    for n, _, _, value in located_records(path):
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


def source_content(root, paths, cache=None) -> dict:
    root, cache = Path(root).resolve(), {} if cache is None else cache
    result = {}
    for path in sorted(set(Path(path).resolve() for path in paths)):
        if path not in cache:
            cache[path] = sha256_file(path)
        result[str(path.relative_to(root))] = cache[path]
    return result


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
