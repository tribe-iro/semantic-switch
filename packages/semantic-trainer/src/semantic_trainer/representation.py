from __future__ import annotations

from pathlib import Path

from semantic_switch import ConfigError
from semantic_switch.training import Adapter, estimate, peak

from .deploy import adapter, deploy

FILES = {'merged': ('specialist.gguf', None), 'runtime-adapter': ('base.gguf', 'adapter.gguf')}


def residency(models, shared) -> dict:
    return estimate(*models) if shared else peak(*(estimate(model) for model in models))


def runtime_adapter(fitted, located, peft, destination, quant):
    adapter(fitted, located, peft, destination)
    converted = Adapter.verified(str(destination), located['gguf'][quant]['path'])
    if converted.rank != fitted['recipe']['rank'] or converted.alpha != float(fitted['recipe']['alpha']):
        raise ConfigError(f'{destination}: converted adapter rank/alpha {converted.rank}/{converted.alpha} differ from the fitted recipe '
                          f"{fitted['recipe']['rank']}/{fitted['recipe']['alpha']}; refit with this toolchain")
    return converted


def deployable(representation, fitted, located, peft, work, name, quant, job):
    base = located['gguf'][quant]
    if representation == 'merged':
        path = Path(work) / f'{name}.gguf'
        return {'source': str(path), 'sha256': deploy(fitted, located, peft, path, quant, job), 'lora': None, 'adapter': None}
    converted = runtime_adapter(fitted, located, peft, Path(work) / f'{name}.adapter.gguf', quant)
    return {'source': base['path'], 'sha256': base['sha256'], 'lora': converted.path, 'adapter': converted.record()}


def realized_binding(entry, realized):
    lora = realized['identity'].get('lora')
    if realized['identity']['sha256'] != entry['sha256'] or (lora or None) != (entry['adapter'] or None):
        raise ConfigError(f"{entry['source']}: scored binding differs from the deployed weights {entry['sha256']} and adapter {entry['adapter']}")
