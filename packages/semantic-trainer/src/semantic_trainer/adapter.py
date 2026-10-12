from __future__ import annotations

import json
from numbers import Real
from pathlib import Path

from semantic_switch import files
from semantic_switch.artifacts import artifact_digest
from semantic_switch.errors import ConfigError

WEIGHTS, CONFIG = 'adapter_model.safetensors', 'adapter_config.json'
EMPTY = (None, False, '', (), [], {})

IDENTITY = ('peft_type', 'task_type', 'r', 'lora_alpha', 'use_rslora', 'lora_dropout', 'target_modules', 'fan_in_fan_out', 'bias', 'modules_to_save',
            'rank_pattern', 'alpha_pattern', 'init_lora_weights')
VARIANTS = ('exclude_modules', 'layers_to_transform', 'layers_pattern', 'megatron_config', 'trainable_token_indices', 'loftq_config', 'eva_config', 'corda_config',
            'lora_ga_config', 'use_dora', 'velora_config', 'alora_invocation_tokens', 'use_qalora', 'monteclora_config', 'layer_replication', 'lora_bias',
            'target_parameters', 'use_bdlora', 'arrow_config', 'kasa_config', 'ensure_weight_tying')
PROVENANCE = ('auto_mapping', 'peft_version', 'base_model_name_or_path', 'revision', 'inference_mode', 'megatron_core', 'qalora_group_size')
INITIALIZERS = (True, False, 'gaussian')

PROFILES = {'0.21': {'task_type': None, 'peft_type': 'LORA', 'r': 8, 'lora_alpha': 8, 'use_rslora': False, 'lora_dropout': 0.0, 'target_modules': None,
                     'fan_in_fan_out': False, 'bias': 'none', 'modules_to_save': None, 'rank_pattern': {}, 'alpha_pattern': {}, 'init_lora_weights': True,
                     'exclude_modules': None, 'layers_to_transform': None, 'layers_pattern': None, 'megatron_config': None, 'trainable_token_indices': None,
                     'loftq_config': {}, 'eva_config': None, 'corda_config': None, 'lora_ga_config': None, 'use_dora': False, 'velora_config': None,
                     'alora_invocation_tokens': None, 'use_qalora': False, 'monteclora_config': None, 'layer_replication': None, 'lora_bias': False,
                     'target_parameters': None, 'use_bdlora': None, 'arrow_config': None, 'kasa_config': None, 'ensure_weight_tying': False}}


def tensor_shapes(path) -> dict:
    path = Path(path)
    try:
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            if not 0 < size <= min(path.stat().st_size - 8, 100 << 20):
                raise ValueError(f'header length {size}')
            header = json.loads(stream.read(size))
        return {name: list(tensor['shape']) for name, tensor in header.items() if name != '__metadata__'}
    except (ValueError, KeyError, TypeError, UnicodeError) as error:
        raise ConfigError(f'{path}: not a readable safetensors file ({error})') from None


def profile(recorded, declared):
    if recorded is not None and declared is not None and recorded != declared:
        raise ConfigError(f'adapter configuration drift: {CONFIG} declares PEFT {declared}, but the run recorded fitting with PEFT {recorded}')
    found = recorded or declared
    return found, PROFILES.get('.'.join(str(found).split('.')[:2])) if found else None


def completed(config, version, defaults):
    unknown = sorted(set(config) - {*IDENTITY, *VARIANTS, *PROVENANCE})
    if unknown:
        raise ConfigError(f'unsupported adapter settings {unknown}: their effect on the fitted adapter cannot be established; refit with vanilla LoRA')
    missing = [key for key in (*IDENTITY, *VARIANTS) if key not in config]
    if missing and defaults is None:
        raise ConfigError(f'insufficient historical evidence: {CONFIG} omits {missing} and PEFT {version or "(unrecorded)"} has no documented default profile '
                          f'(supported: {sorted(PROFILES)}); the installed library defaults are not evidence of the fitted configuration')
    return {**{key: defaults[key] for key in missing}, **config}


def patterned(pattern, value, key):
    if not isinstance(pattern, dict):
        raise ConfigError(f'adapter configuration drift: {key} must be an object')
    varied = sorted(name for name, given in pattern.items() if given != value)
    if varied:
        raise ConfigError(f'unsupported adapter setting {key} for {varied[:5]}: per-module ranks or scales differ from the vanilla recipe')


def normalized(config, fitted) -> dict:
    recipe = fitted['recipe']
    variants = sorted(key for key in VARIANTS if config[key] not in EMPTY)
    if variants:
        raise ConfigError(f'unsupported adapter variant or setting {variants}: only vanilla LoRA adapters are admitted')
    if not any(config['init_lora_weights'] is option or config['init_lora_weights'] == option == 'gaussian' for option in INITIALIZERS):
        raise ConfigError(f"unsupported adapter initialization {config['init_lora_weights']!r}: it can rewrite base weights; supported {list(INITIALIZERS)}")
    if not isinstance(config['r'], int) or isinstance(config['r'], bool) or not isinstance(config['lora_alpha'], Real) or isinstance(config['lora_alpha'], bool) \
            or not isinstance(config['lora_dropout'], Real) or not isinstance(config['target_modules'], list) \
            or not all(isinstance(name, str) for name in config['target_modules']):
        raise ConfigError('adapter configuration drift: r, lora_alpha, lora_dropout or target_modules has an unsupported type')
    patterned(config['rank_pattern'], config['r'], 'rank_pattern')
    patterned(config['alpha_pattern'], config['lora_alpha'], 'alpha_pattern')
    revision = config.get('revision')
    if revision is not None and revision != fitted['model']['revision']:
        raise ConfigError(f"adapter configuration drift: base revision {revision} differs from the pinned {fitted['model']['revision']}")
    found = {'peft_type': config['peft_type'], 'task_type': config['task_type'], 'r': config['r'], 'lora_alpha': config['lora_alpha'],
             'use_rslora': config['use_rslora'], 'lora_dropout': float(config['lora_dropout']), 'target_modules': sorted(config['target_modules']),
             'fan_in_fan_out': config['fan_in_fan_out'], 'bias': config['bias'], 'modules_to_save': sorted(config['modules_to_save'] or ())}
    expected = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': recipe['rank'], 'lora_alpha': recipe['alpha'], 'use_rslora': False,
                'lora_dropout': float(recipe['dropout']), 'target_modules': sorted(fitted['targets']), 'fan_in_fan_out': False, 'bias': 'none', 'modules_to_save': []}
    drift = sorted(key for key in expected if found[key] != expected[key])
    if drift:
        shown = [key for key in drift if key != 'target_modules']
        raise ConfigError(f'adapter configuration drift in {drift}: {CONFIG} does not describe the recorded fitting recipe and targets '
                          f'(found {dict((key, found[key]) for key in shown)}, fitted {dict((key, expected[key]) for key in shown)})')
    return {**found, 'scaling': found['lora_alpha'] / found['r']}


def weights(path, fitted, rank):
    expected = {}
    for name, (out, inner) in fitted['targets'].items():
        expected[f'base_model.model.{name}.lora_A.weight'] = [rank, inner]
        expected[f'base_model.model.{name}.lora_B.weight'] = [out, rank]
    found = tensor_shapes(path)
    if found != expected:
        wrong = sorted(set(found) ^ set(expected)) or sorted(name for name in expected if found[name] != expected[name])
        raise ConfigError(f'{path}: adapter weight drift: tensors {wrong[:4]} do not match the configured rank, targets and orientation')


def identify(fitted, peft) -> dict:
    peft = Path(peft)
    for name in (WEIGHTS, CONFIG):
        if not (peft / name).is_file():
            raise ConfigError(f'{peft}: missing {name}; the candidate needs its adapter weights and configuration')
    config = files.read_json(peft / CONFIG)
    if not isinstance(config, dict):
        raise ConfigError(f'{peft / CONFIG}: adapter configuration must be a JSON object')
    version, defaults = profile(fitted.get('dependencies', {}).get('peft'), config.get('peft_version'))
    adapter = normalized(completed(config, version, defaults), fitted)
    weights(peft / WEIGHTS, fitted, adapter['r'])
    return {'peft_sha256': files.sha256_file(peft / WEIGHTS), 'config_sha256': files.sha256_file(peft / CONFIG), 'config_digest': artifact_digest(adapter)}


def verified(fitted, peft, recorded, configuration=('config_sha256', 'config_digest')) -> dict:
    found = identify(fitted, peft)
    if found['peft_sha256'] != recorded['peft_sha256']:
        raise ConfigError(f'{peft}: adapter weight drift since fitting; the recorded candidate is unavailable')
    drifted = [key for key in configuration if found[key] != recorded[key]]
    if drifted:
        raise ConfigError(f'{peft}: adapter configuration drift since fitting ({", ".join(drifted)}); the recorded candidate is unavailable')
    return found
