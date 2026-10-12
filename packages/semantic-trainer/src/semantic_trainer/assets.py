from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from semantic_switch import files, model_job
from semantic_switch.artifacts import artifact_digest
from semantic_switch.errors import ConfigError

SCHEMA = 1
REPOSITORY, TOOLCHAIN = 'https://github.com/ggml-org/llama.cpp', 'a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd'
TOOLS = ('convert_hf_to_gguf.py', 'convert_lora_to_gguf.py', 'build/bin/llama-export-lora', 'build/bin/llama-quantize')
HF_PATTERNS = ['*.json', '*.safetensors', '*.model', '*.jinja', '*.txt']
COMMIT, DIGEST, QUANT = re.compile('[0-9a-f]{40}'), re.compile('[0-9a-f]{64}'), re.compile('(?!F16$)[A-Z][A-Z0-9_]*')


def root():
    configured = os.environ.get('SEMANTIC_TRAINER_ASSETS')
    return Path(configured or Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'semantic-switch' / 'trainer').resolve()


def layout(repo, commit):
    model = Path('models', repo.replace('/', '--'), commit)
    return {'hf': model / 'hf', 'gguf': model / f'llama.cpp-{TOOLCHAIN}', 'toolchain': Path('llama.cpp', TOOLCHAIN)}


def run(command, **kwargs):
    print(f'assets: {shlex.join(map(str, command))}', file=sys.stderr, flush=True)
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise ConfigError(f'{shlex.join(map(str, command))} failed ({result.returncode}): {(result.stderr or result.stdout)[-3000:]}')
    return result.stdout


def revision_of(repo, revision):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo, revision=revision).sha
    except Exception as error:
        raise ConfigError(f'cannot resolve {repo}@{revision} ({error}); pass --revision with a 40-hex commit') from None


def download(repo, commit, destination):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo, revision=commit, allow_patterns=HF_PATTERNS))
    for source in sorted(snapshot.rglob('*')):
        if source.is_file():
            target = destination / source.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source.resolve(), target)
            except OSError:
                shutil.copyfile(source.resolve(), target)


def toolchain(destination, source):
    run(['git', 'init', '-q', destination])
    run(['git', '-C', destination, 'fetch', '-q', '--depth', '1', source, TOOLCHAIN])
    run(['git', '-C', destination, 'checkout', '-q', '--detach', 'FETCH_HEAD'])
    if run(['git', '-C', destination, 'rev-parse', 'HEAD']).strip() != TOOLCHAIN:
        raise ConfigError(f'{source} did not provide llama.cpp commit {TOOLCHAIN}')
    build = destination / 'build'
    run(['cmake', '-S', destination, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_CURL=OFF'])
    run(['cmake', '--build', build, '--target', 'llama-export-lora', 'llama-quantize', '-j', os.cpu_count() or 1])


def materialize(base, target, make):
    (base / 'tmp').mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.build-', dir=base / 'tmp'))
    try:
        staged = work / target.name
        make(staged)
        (base / target).parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, base / target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def tree(folder, names=None):
    paths = sorted(p for p in folder.rglob('*') if p.is_file()) if names is None else [folder / n for n in names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing or not paths:
        raise ConfigError(f'incomplete assets under {folder}: missing {missing or "all files"}; run semantic-trainer assets')
    return {str(p.relative_to(folder)): files.sha256_file(p) for p in paths}


def record(base, repo, commit, quants):
    paths = layout(repo, commit)
    tools, models = base / paths['toolchain'], base / paths['gguf']
    gguf_py = sorted(str(p.relative_to(tools)) for p in (tools / 'gguf-py' / 'gguf').rglob('*.py'))
    manifest = {'schema': SCHEMA, 'kind': 'trainer-assets',
                'hf': {'repo': repo, 'commit': commit, 'path': str(paths['hf']), 'files': tree(base / paths['hf'])},
                'toolchain': {'repository': REPOSITORY, 'commit': TOOLCHAIN, 'path': str(paths['toolchain']), 'files': tree(tools, [*TOOLS, *gguf_py])},
                'gguf': {kind: {'path': str(paths['gguf'] / f'{kind}.gguf'), 'sha256': tree(models, [f'{kind}.gguf'])[f'{kind}.gguf']} for kind in ('F16', *quants)}}
    digest = artifact_digest(manifest)
    files.publish(base / 'manifests' / f'{digest}.json', manifest)
    return digest


def located(base, digest, manifest):
    return {'manifest': digest, 'root': str(base), 'repo': manifest['hf']['repo'], 'commit': manifest['hf']['commit'],
            'hf': str(base / manifest['hf']['path']), 'files': dict(sorted(manifest['hf']['files'].items())), 'toolchain': {'commit': manifest['toolchain']['commit'], 'path': str(base / manifest['toolchain']['path'])},
            'gguf': {kind: {'path': str(base / entry['path']), 'sha256': entry['sha256']} for kind, entry in manifest['gguf'].items()}}


def load(digest):
    base = root()
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ConfigError('assets.manifest must be the 64-hex digest returned by semantic-trainer assets')
    manifest = files.read_json(base / 'manifests' / f'{digest}.json')
    if manifest is None:
        raise ConfigError(f'asset manifest {digest} not found under {base}; run semantic-trainer assets --model REPO --quant TYPE or set SEMANTIC_TRAINER_ASSETS')
    if manifest.get('schema') != SCHEMA or artifact_digest(manifest) != digest:
        raise ConfigError(f'asset manifest {digest} was modified; rebuild with semantic-trainer assets')
    rebuild = f"rebuild with: semantic-trainer assets --model {manifest['hf']['repo']} " + ' '.join(f'--quant {q}' for q in manifest['gguf'] if q != 'F16')
    entries = [(base / manifest[group]['path'] / name, sha) for group in ('hf', 'toolchain') for name, sha in manifest[group]['files'].items()]
    for path, sha in entries + [(base / entry['path'], entry['sha256']) for entry in manifest['gguf'].values()]:
        if not path.is_file():
            raise ConfigError(f'{path}: asset missing from manifest {digest}; {rebuild}')
        if files.sha256_file(path) != sha:
            raise ConfigError(f'{path}: asset changed since manifest {digest}; {rebuild}')
    return located(base, digest, manifest)


def build(repo, quants, *, revision=None, source=REPOSITORY):
    if not isinstance(repo, str) or repo.count('/') != 1 or not all(repo.split('/')):
        raise ConfigError('--model must be a Hugging Face repository such as google/gemma-4-E2B-it')
    quants = sorted(set(quants))
    if not quants or any(not QUANT.fullmatch(q) for q in quants):
        raise ConfigError('--quant requires llama-quantize types such as Q4_0 or Q8_0; F16 is always built')
    commit = revision if revision and COMMIT.fullmatch(revision) else revision_of(repo, revision or 'main')
    base, paths = root(), layout(repo, commit)
    tools = base / paths['toolchain']
    f16 = paths['gguf'] / 'F16.gguf'
    steps = [(paths['toolchain'], lambda out: toolchain(out, source)), (paths['hf'], lambda out: download(repo, commit, out)),
             (f16, lambda out: run([sys.executable, tools / 'convert_hf_to_gguf.py', base / paths['hf'], '--outtype', 'f16', '--outfile', out])),
             *((paths['gguf'] / f'{q}.gguf', lambda out, q=q: run([tools / 'build/bin/llama-quantize', base / f16, out, q])) for q in quants)]
    pending = [(target, make) for target, make in steps if not (base / target).exists()]
    fetched = (paths['toolchain'], paths['hf'])
    for target, make in pending:
        if target in fetched:
            materialize(base, target, make)
    converting = [(target, make) for target, make in pending if target not in fetched]
    if converting:
        with model_job(extra={'conversion': sum(path.stat().st_size for path in (base / paths['hf']).glob('*.safetensors'))}):
            for target, make in converting:
                materialize(base, target, make)
    digest = record(base, repo, commit, quants)
    assets = load(digest)
    return {'manifest': digest, 'path': str(base / 'manifests' / f'{digest}.json'), 'root': str(base), 'model': {'repo': repo, 'commit': commit},
            'toolchain': {'repository': REPOSITORY, 'commit': TOOLCHAIN}, 'gguf': {kind: entry['sha256'] for kind, entry in assets['gguf'].items()},
            'built': [str(target) for target, _ in pending], 'job': {'assets': {'manifest': digest, 'quant': quants[0]}}}
