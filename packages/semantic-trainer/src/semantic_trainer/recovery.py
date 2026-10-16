from __future__ import annotations

import random
import shutil
import uuid
from pathlib import Path

import numpy as np

from semantic_switch import ConfigError
from semantic_switch.training import files

from .monitor import Work
from .adapter import CONFIG, WEIGHTS


def latest(directory):
    pointer = files.read_json(Path(directory) / 'recovery.json')
    return None if pointer is None else Path(directory) / pointer['latest']


def load(directory, resolved):
    import torch
    path = latest(directory)
    if path is None:
        return None
    metadata = files.read_json(path / 'checkpoint.json')
    expected = {'fit': resolved['fit']['id'], 'monitor': resolved.get('monitor', {}).get('id')}
    if metadata is None or metadata['identity'] != expected or metadata['sha256'] != files.sha256_file(path / 'state.pt') or any(files.sha256_file(path / name) != checksum for name, checksum in metadata['adapter'].items()):
        raise ConfigError(f'{path}: recovery state does not match this fit and monitor')
    from safetensors.torch import load_file
    payload = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
    payload['adapter'] = load_file(str(path / WEIGHTS))
    pointer = files.read_json(Path(directory) / 'recovery.json')
    payload['checkpoint_seconds'] = pointer.get('checkpoint_seconds', 0.)
    return payload


def save(directory, model, trained, monitor, state):
    import torch
    began = monitor.deadline.elapsed()
    directory = Path(directory)
    root = directory / 'recovery'
    root.mkdir(exist_ok=True)
    name = f"step-{state['step']}-{uuid.uuid4().hex[:12]}"
    staged, destination = root / f'.{name}.staging', root / name
    staged.mkdir()
    backend = monitor.backend
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if backend is not None and str(monitor.device).split(':')[0] != 'cpu':
        rng['device'] = backend.get_rng_state(monitor.device)
    model.save_pretrained(staged, safe_serialization=True)
    payload = {'state': state,
               'optimizer': trained[1].state_dict(), 'scheduler': trained[2].state_dict(), 'rng': rng,
               'monitor': {'work': monitor.work.totals(), 'points': monitor.points, 'updates': monitor.updates, 'losses': monitor.losses,
                           'seen': monitor.seen, 'feasibility': monitor.feasibility}, 'elapsed': monitor.deadline.elapsed()}
    torch.save(payload, staged / 'state.pt')
    files.publish(staged / 'checkpoint.json', {'identity': {'fit': monitor.resolved['fit']['id'], 'monitor': monitor.resolved.get('monitor', {}).get('id')},
                                              'step': state['step'], 'adapter': {name: files.sha256_file(staged / name) for name in (CONFIG, WEIGHTS)}, 'retained': [point['name'] for point in state.get('retained', ())], 'sha256': files.sha256_file(staged / 'state.pt')})
    files.publish_directory(staged, destination)
    old = files.read_json(directory / 'recovery.json') or {}
    files.publish(directory / 'recovery.json', {'latest': str(destination.relative_to(directory)), 'previous': old.get('latest'),
                                              'elapsed': monitor.deadline.elapsed(), 'checkpoint_seconds': monitor.deadline.elapsed() - began, 'step': state['step']})
    keep = {str(destination.relative_to(directory)), old.get('latest')}
    for path in root.iterdir():
        if path.is_dir() and not path.name.startswith('.') and str(path.relative_to(directory)) not in keep:
            shutil.rmtree(path)
    previous = files.read_json(directory / old['latest'] / 'checkpoint.json') if old.get('latest') else {}
    candidates = directory / 'candidates'
    kept_candidates = {'zero', *(point['name'] for point in state.get('retained', ())), *((previous or {}).get('retained', ()))}
    if candidates.is_dir():
        for candidate in candidates.iterdir():
            if candidate.is_dir() and not candidate.name.startswith('.') and candidate.name not in kept_candidates:
                shutil.rmtree(candidate)
    monitor.checkpointed(monitor.deadline.elapsed() - began)
    return monitor.deadline.elapsed()


def restore(payload, model, trained, monitor):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    if set(payload['adapter']) != set(get_peft_model_state_dict(model)):
        raise ConfigError('recovery adapter parameter names differ from the fitted model')
    set_peft_model_state_dict(model, payload['adapter'])
    trained[2].load_state_dict(payload['scheduler'])
    trained[1].load_state_dict(payload['optimizer'])
    saved = payload['monitor']
    monitor.work = Work(**saved['work'])
    monitor.checkpointed(payload.get('checkpoint_seconds', 0.))
    for key in ('points', 'updates', 'losses', 'seen', 'feasibility'):
        setattr(monitor, key, saved[key])
    monitor.flushed = len(monitor.updates)
    random.setstate(payload['rng']['python'])
    np.random.set_state(payload['rng']['numpy'])
    torch.set_rng_state(payload['rng']['torch'])
    if monitor.backend is not None and 'device' in payload['rng']:
        monitor.backend.set_rng_state(payload['rng']['device'], monitor.device)
    return payload['state']
