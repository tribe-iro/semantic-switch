
from semantic_switch import model_jobs
from pathlib import Path

import pytest

from semantic_switch import ConfigError, api, files
from semantic_trainer import assets


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'assets'))
    monkeypatch.setenv('SEMANTIC_MODEL_LOCK', str(tmp_path / 'model-job.lock'))
    monkeypatch.setattr(model_jobs, 'available_host_bytes', lambda: 1 << 50)
    commands = []
    def run(command, **kwargs):
        command = [str(part) for part in command]
        commands.append(command)
        assert all(str(tmp_path / 'assets') in part for part in command if part.startswith('/') and 'python' not in part)
        if command[:2] == ['git', 'init']:
            Path(command[-1]).mkdir()
        elif command[-1] == 'HEAD':
            return assets.TOOLCHAIN + '\n'
        elif '--build' in command:
            source = Path(command[command.index('--build') + 1]).parent
            for name in assets.TOOLS:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_text(name)
            (source / 'gguf-py' / 'gguf').mkdir(parents=True)
            (source / 'gguf-py' / 'gguf' / 'gguf_writer.py').write_text('writer')
        elif any(part.endswith('convert_hf_to_gguf.py') for part in command):
            Path(command[command.index('--outfile') + 1]).write_bytes(b'f16')
        elif command[0].endswith('llama-quantize'):
            Path(command[2]).write_bytes(command[3].encode())
        return ''
    def download(repo, commit, destination):
        commands.append(['download', repo, commit])
        destination.mkdir()
        files.publish(destination / 'config.json', {'model_type': 'gemma4'})
        (destination / 'model.safetensors').write_bytes(b'weights')
    monkeypatch.setattr(assets, 'run', run)
    monkeypatch.setattr(assets, 'download', download)
    monkeypatch.setattr(assets, 'revision_of', lambda repo, revision: pytest.fail('network revision lookup'))
    return commands


def test_assets_build_once_under_root_and_verify(tmp_path, toolchain):
    receipt = assets.build('google/gemma-4-E2B-it', ['Q4_0'], revision='c' * 40)
    root = tmp_path / 'assets'
    assert receipt['root'] == str(root) and receipt['manifest'] == Path(receipt['path']).stem
    assert len(receipt['built']) == 4 and receipt['job'] == {'assets': {'manifest': receipt['manifest'], 'quant': 'Q4_0'}}
    assert Path(toolchain[0][-1]).is_relative_to(root / 'tmp') and 'fetch' in toolchain[1] and toolchain[1][-1] == assets.TOOLCHAIN
    assert any('--target' in command and 'llama-export-lora' in command and 'llama-quantize' in command for command in toolchain)
    assert not list((root / 'tmp').iterdir())
    located = assets.load(receipt['manifest'])
    assert set(located['gguf']) == {'F16', 'Q4_0'} and Path(located['hf'], 'model.safetensors').is_file()
    manifest = files.read_json(receipt['path'])
    assert set(manifest['toolchain']['files']) == {*assets.TOOLS, 'gguf-py/gguf/gguf_writer.py'}
    count = len(toolchain)
    again = assets.build('google/gemma-4-E2B-it', ['Q4_0'], revision='c' * 40)
    assert again['manifest'] == receipt['manifest'] and again['built'] == [] and len(toolchain) == count
    extended = assets.build('google/gemma-4-E2B-it', ['Q8_0', 'Q4_0'], revision='c' * 40)
    assert extended['built'] == [str(assets.layout('google/gemma-4-E2B-it', 'c' * 40)['gguf'] / 'Q8_0.gguf')] and extended['manifest'] != receipt['manifest']


@pytest.mark.parametrize('target', ['toolchain', 'hf', 'gguf', 'manifest', 'missing'])
def test_assets_refuse_changed_files(tmp_path, toolchain, target):
    receipt = assets.build('google/gemma-4-E2B-it', ['Q4_0'], revision='c' * 40)
    located = assets.load(receipt['manifest'])
    path = {'toolchain': Path(located['toolchain']['path'], 'build/bin/llama-export-lora'), 'hf': Path(located['hf'], 'config.json'),
            'gguf': Path(located['gguf']['Q4_0']['path']), 'manifest': Path(receipt['path']), 'missing': Path(located['gguf']['F16']['path'])}[target]
    if target == 'missing':
        path.unlink()
    elif target == 'manifest':
        files.publish(path, {**files.read_json(path), 'gguf': {}})
    else:
        path.write_bytes(b'changed')
    with pytest.raises(ConfigError, match={'manifest': 'modified', 'missing': 'missing'}.get(target, 'changed since manifest')):
        assets.load(receipt['manifest'])


def test_assets_root_is_xdg_cache_or_one_override(tmp_path, monkeypatch):
    monkeypatch.delenv('SEMANTIC_TRAINER_ASSETS', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    assert assets.root() == tmp_path / 'xdg' / 'semantic-switch' / 'trainer'
    monkeypatch.setenv('SEMANTIC_TRAINER_ASSETS', str(tmp_path / 'pinned'))
    assert assets.root() == tmp_path / 'pinned'


@pytest.mark.parametrize('repo,quants,match', [('gemma', ['Q4_0'], 'repository'), ('google/gemma-4-E2B-it', ['F16'], 'quant'), ('google/gemma-4-E2B-it', [], 'quant')])
def test_assets_reject_invalid_requests(toolchain, repo, quants, match):
    with pytest.raises(ConfigError, match=match):
        assets.build(repo, quants, revision='c' * 40)


def test_jobs_resolve_assets_only_through_manifest(tmp_path, monkeypatch):
    from semantic_switch.artifacts import native_model
    from semantic_trainer.jobs import resolve
    from semantic_testkit.assets import training_job
    job_path, job, _ = training_job(tmp_path, monkeypatch)
    resolved, _ = resolve(job_path)
    root = str(tmp_path / 'assets')
    assert resolved['assets'] == job['assets'] and resolved['model']['gguf'].startswith(root) and resolved['model']['hf'].startswith(root)
    assert resolved['toolchain']['path'].startswith(root) and 'source' not in resolved['native'] and native_model(resolved).source == resolved['model']['gguf']
    for change, match in (({'assets': {**job['assets'], 'manifest': 'f' * 64}}, 'not found'), ({'assets': {**job['assets'], 'quant': 'Q8_0'}}, 'no .Q8_0. base'),
                          ({'assets': {'manifest': job['assets']['manifest']}}, 'manifest and quant'), ({'model': {'hf': '/tmp/hf'}}, 'unsupported job fields'),
                          ({'schema': 7}, 'schema: 8'), ({'loss': {**job['loss'], 'version': 1}}, 'prepare a new job')):
        files.publish(job_path, {**job, **change})
        with pytest.raises(ConfigError, match=match):
            resolve(job_path)
    files.publish(job_path, job)
    Path(resolved['model']['gguf']).write_bytes(b'other base')
    with pytest.raises(ConfigError, match='changed since manifest'):
        resolve(job_path)


def test_fitting_settings_decode_without_torch_and_the_device_is_validated_before_model_construction(tmp_path, monkeypatch):
    import subprocess
    import sys
    from semantic_trainer.jobs import fitting_device, fitting_settings, physical_cores
    code = ("import sys; from semantic_trainer.jobs import fitting_settings; "
            "assert fitting_settings({'device': 'cuda:7'})['precision'] == 'bfloat16'; assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)
    for settings, match in (({}, 'cpu, xpu:N or cuda:N'), ({'device': 'gpu'}, 'cpu, xpu:N or cuda:N'), ({'device': 'cpu', 'precision': 'float16'}, 'float32 or bfloat16'),
                            ({'device': 'cpu', 'threads': 0}, 'threads'), ({'device': 'cpu', 'attention': 'flash'}, 'attention'),
                            ({'device': 'cpu', 'fused_optimizer': 1}, 'fused_optimizer'), ({'device': 'cpu', 'offload': True}, 'fitting accepts')):
        with pytest.raises(ConfigError, match=match):
            fitting_settings(settings)
    assert fitting_settings({'device': 'xpu:0', 'gradient_checkpointing': True, 'attention': 'sdpa', 'fused_optimizer': True}) == {
        'device': 'xpu:0', 'precision': 'bfloat16', 'threads': physical_cores(), 'attention': 'sdpa', 'gradient_checkpointing': True, 'fused_optimizer': True}
    with pytest.raises(ConfigError, match='bfloat16'):
        fitting_settings({'device': 'xpu:0', 'precision': 'float32'})
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ConfigError, match='unavailable.*README'):
        fitting_device(fitting_settings({'device': 'cuda:0'}))
    monkeypatch.setattr(torch.xpu, 'is_available', lambda: True)
    monkeypatch.setattr(torch.xpu, 'device_count', lambda: 1)
    assert fitting_device(fitting_settings({'device': 'xpu:0'})) == 'xpu:0' and fitting_device(fitting_settings({'device': 'cpu'})) == 'cpu'
    with pytest.raises(ConfigError, match='unavailable'):
        fitting_device(fitting_settings({'device': 'xpu:1'}))
