from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from semantic_switch import ConfigError
from semantic_switch.training import files

from . import assets
from .adapter import identify, zero_delta
from .jobs import MODULES

REPRESENTATIONS = ('merged', 'runtime-adapter')

INSPECT = ("import json,sys,numpy as np; from gguf import GGUFReader; r=GGUFReader(sys.argv[1]); "
           "print(json.dumps({t.name: {'shape': [int(n) for n in t.data.shape], 'finite': bool(np.isfinite(t.data).all())} for t in r.tensors}))")


def expected_tensors(fitted):
    expected = {}
    for name, shape in fitted['targets'].items():
        match = re.search(r'layers\.(\d+)\.(self_attn|mlp)\.(\w+)$', name)
        if not match or MODULES.get(match[3], ('',))[0] != match[2]:
            raise ConfigError(f'unsupported conversion target {name}; supported: {list(MODULES)} language projections')
        stem = f'blk.{match[1]}.{MODULES[match[3]][1]}.weight'
        expected[stem + '.lora_a'] = [fitted['recipe']['rank'], shape[1]]
        expected[stem + '.lora_b'] = [shape[0], fitted['recipe']['rank']]
    return expected


def adapter(fitted, located, peft, destination):
    identify(fitted, peft)
    tools = Path(located['toolchain']['path'])
    assets.run([sys.executable, tools / 'convert_lora_to_gguf.py', '--base', located['hf'], '--outtype', 'f16', '--outfile', destination, peft])
    tensors = json.loads(assets.run([sys.executable, '-c', INSPECT, destination], env={**os.environ, 'PYTHONPATH': str(tools / 'gguf-py')}))
    expected = expected_tensors(fitted)
    if set(expected) != set(tensors) or any(tensors[name]['shape'] != shape or not tensors[name]['finite'] for name, shape in expected.items()):
        raise ConfigError(f'{peft}: converted adapter targets/shapes or finite tensors do not match the fitted architecture; refit with this toolchain')


def representation_of(value):
    if value not in REPRESENTATIONS:
        raise ConfigError(f'deployment representation must be one of {", ".join(REPRESENTATIONS)}, not {value!r}')
    return value


def conversion_bytes(located):
    return {'conversion:F16': Path(located['gguf']['F16']['path']).stat().st_size}


def deploy(fitted, located, peft, destination, quant, job):
    with job.join(sum(conversion_bytes(located).values()), use='conversion'):
        return convert(fitted, located, peft, destination, quant)


def convert(fitted, located, peft, destination, quant):
    tools = Path(located['toolchain']['path'])
    scratch = Path(located['root']) / 'tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.deploy-', dir=scratch))
    try:
        adapter(fitted, located, peft, work / 'adapter.gguf')
        assets.run([tools / 'build/bin/llama-export-lora', '-m', located['gguf']['F16']['path'], '--lora', work / 'adapter.gguf', '-o', work / 'merged-F16.gguf'])
        assets.run([tools / 'build/bin/llama-quantize', work / 'merged-F16.gguf', work / 'merged.gguf', quant])
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(work / 'merged.gguf', destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return files.sha256_file(destination)


def verify_zero(fitted, located, peft, quant, work, job):
    produced = deploy(fitted, located, peft, Path(work) / 'zero.gguf', quant, job)
    Path(work, 'zero.gguf').unlink()
    expected = located['gguf'][quant]['sha256']
    if produced != expected:
        raise ConfigError(f'zero adapter merged into the F16 base and quantized to {quant} gives {produced}, not the pipeline-built base {expected}; '
                          f'the base or toolchain differs from manifest {located["manifest"]}: rebuild with semantic-trainer assets and refit')
    return produced


def verify_runtime_zero(fitted, located, peft, quant, base_sha256):
    identify(fitted, peft)
    if not zero_delta(peft):
        raise ConfigError(f'{peft}: the zero adapter has nonzero lora_B factors, so its runtime delta over the base is not zero; refit from a fresh initialization')
    expected = located['gguf'][quant]['sha256']
    if base_sha256 != expected:
        raise ConfigError(f'runtime-adapter base {base_sha256} is not the pipeline-built {quant} base {expected} of manifest {located["manifest"]}; '
                          'rebuild with semantic-trainer assets and refit')
    return expected
