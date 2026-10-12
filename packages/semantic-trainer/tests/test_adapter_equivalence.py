import json

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.artifacts import load_receipt
from test_integration import legacy
from test_specialize import peft_adapter, selection

FITTED = {'targets': {f'model.language_model.layers.{layer}.{block}.{name}': shape for layer in (0, 1)
                      for block, name, shape in (('self_attn', 'q_proj', [16, 8]), ('self_attn', 'o_proj', [8, 16]), ('mlp', 'down_proj', [8, 24]))},
          'recipe': {'rank': 4, 'alpha': 8, 'dropout': 0.05}, 'model': {'revision': 'c' * 40}, 'dependencies': {'peft': '0.21.2'}}
DEFAULTED = ('exclude_modules', 'use_dora', 'lora_bias', 'trainable_token_indices', 'loftq_config', 'arrow_config', 'kasa_config', 'ensure_weight_tying', 'use_rslora')


def identify(directory, fitted=FITTED):
    from semantic_trainer.adapter import identify
    return identify(fitted, directory)


def test_equivalent_supported_representations_share_one_normalized_identity(tmp_path):
    original = identify(peft_adapter(tmp_path / 'original', FITTED))
    first = next(iter(FITTED['targets']))
    variants = {'path': {'base_model_name_or_path': '/elsewhere/hf/gemma-4-e2b', 'inference_mode': False, 'auto_mapping': {'base_model_class': 'Gemma4'}},
                'order': {'target_modules': sorted(FITTED['targets'])},
                'defaults': {key: ... for key in DEFAULTED},
                'patterns': {'rank_pattern': {first: 4}, 'alpha_pattern': {first: 8}},
                'initializer': {'init_lora_weights': 'gaussian'},
                'revision': {'revision': 'c' * 40}}
    for name, change in variants.items():
        found = identify(peft_adapter(tmp_path / name, FITTED, **change))
        assert found['config_digest'] == original['config_digest'] and found['config_sha256'] != original['config_sha256'], name
    recorded = identify(peft_adapter(tmp_path / 'recorded', {**FITTED, 'dependencies': {}}, peft_version=..., **{key: ... for key in DEFAULTED}), FITTED)
    assert recorded['config_digest'] == original['config_digest']


@pytest.mark.parametrize('change, match', [
    ({'lora_alpha': 999}, r"drift in \['lora_alpha'\]"),
    ({'r': 8}, r"drift in \['r'\]"),
    ({'lora_dropout': 0.1}, r"drift in \['lora_dropout'\]"),
    ({'use_rslora': True}, r"drift in \['use_rslora'\]"),
    ({'target_modules': sorted(FITTED['targets'])[:-1]}, r"drift in \['target_modules'\]"),
    ({'target_modules': ['q_proj', 'o_proj', 'down_proj']}, r"drift in \['target_modules'\]"),
    ({'target_modules': 'q_proj|o_proj'}, 'unsupported type'),
    ({'fan_in_fan_out': True}, r"drift in \['fan_in_fan_out'\]"),
    ({'bias': 'lora_only'}, r"drift in \['bias'\]"),
    ({'modules_to_save': ['lm_head']}, r"drift in \['modules_to_save'\]"),
    ({'task_type': 'SEQ_CLS'}, r"drift in \['task_type'\]"),
    ({'peft_type': 'LOHA'}, r"drift in \['peft_type'\]"),
    ({'use_dora': True}, 'unsupported adapter variant'),
    ({'lora_bias': True}, 'unsupported adapter variant'),
    ({'layer_replication': [[0, 2], [1, 2]]}, 'unsupported adapter variant'),
    ({'alora_invocation_tokens': [5, 6]}, 'unsupported adapter variant'),
    ({'rank_pattern': {'model.language_model.layers.0.self_attn.q_proj': 8}}, 'unsupported adapter setting rank_pattern'),
    ({'alpha_pattern': {'model.language_model.layers.1.mlp.down_proj': 999}}, 'unsupported adapter setting alpha_pattern'),
    ({'init_lora_weights': 'pissa'}, 'unsupported adapter initialization'),
    ({'init_lora_weights': 1}, 'unsupported adapter initialization'),
    ({'use_future_variant': True}, r"unsupported adapter settings \['use_future_variant'\]"),
    ({'revision': 'd' * 40}, 'base revision'),
    ({'peft_version': '0.17.0'}, 'declares PEFT 0.17.0'),
])
def test_behavior_affecting_configuration_changes_are_rejected_with_unchanged_weights(tmp_path, change, match):
    with pytest.raises(ConfigError, match=match):
        identify(peft_adapter(tmp_path / 'changed', FITTED, **change))


@pytest.mark.parametrize('weights, match', [({'rank': 8}, 'adapter weight drift'), ({'transposed': True}, 'adapter weight drift')])
def test_weight_rank_and_orientation_must_match_the_configuration(tmp_path, weights, match):
    with pytest.raises(ConfigError, match=match):
        identify(peft_adapter(tmp_path / 'changed', FITTED, **weights))
    (tmp_path / 'broken').mkdir()
    peft_adapter(tmp_path / 'broken', FITTED)
    (tmp_path / 'broken' / 'adapter_model.safetensors').write_bytes(b'step-8')
    with pytest.raises(ConfigError, match='not a readable safetensors file'):
        identify(tmp_path / 'broken')


def test_omitted_fields_without_a_documented_recorded_profile_are_missing_evidence(tmp_path):
    unrecorded = {**FITTED, 'dependencies': {}}
    with pytest.raises(ConfigError, match=r"insufficient historical evidence: .* omits \['use_dora'\] and PEFT \(unrecorded\)"):
        identify(peft_adapter(tmp_path / 'unrecorded', unrecorded, peft_version=..., use_dora=...), unrecorded)
    with pytest.raises(ConfigError, match=r'insufficient historical evidence: .* PEFT 0.9.0 has no documented default profile'):
        identify(peft_adapter(tmp_path / 'old', unrecorded, peft_version='0.9.0', use_dora=...), unrecorded)
    assert identify(peft_adapter(tmp_path / 'complete', unrecorded, peft_version='0.9.0'), unrecorded)['config_digest']


def candidates_with(run, name, recorded):
    listed = files.read_json(run / 'candidates.json')
    listed['candidates'] = [{**candidate, **recorded} if candidate['name'] == name else candidate for candidate in listed['candidates']]
    files.publish(run / 'candidates.json', listed)


def test_migration_proves_configuration_equivalence_before_eligibility_and_select_rechecks_it(tmp_path, monkeypatch):
    from semantic_trainer.migrate import migrate
    module, job, run, resolved, *_ = selection(tmp_path, monkeypatch)
    legacy(run, resolved)
    config = run / 'candidates' / 'step-8' / 'adapter_config.json'
    original = files.read_json(config)
    files.publish(config, {**original, 'lora_alpha': 999})
    candidates_with(run, 'step-8', {'config_sha256': files.sha256_file(config)})
    with pytest.raises(ConfigError, match=r"adapter configuration drift in \['lora_alpha'\]"):
        migrate(job, run, tmp_path / 'alpha.json')
    assert not (tmp_path / 'alpha.json').exists()
    equivalent = {key: value for key, value in original.items() if key not in DEFAULTED} | {'base_model_name_or_path': '/relocated/hf'}
    files.publish(config, equivalent)
    candidates_with(run, 'step-8', {'config_sha256': files.sha256_file(config)})
    result = migrate(job, run, tmp_path / 'compatibility.json')
    assert result['eligible'] and result['outcome'] == 'eligible'
    record = files.read_json(tmp_path / 'compatibility.json')
    chosen = {candidate['name']: candidate for candidate in record['candidates']}['step-8']
    assert chosen['config_sha256'] == files.sha256_file(config) and chosen['peft_sha256'] and len(chosen['config_digest']) == 64
    zero = {candidate['name']: candidate for candidate in record['candidates']}['zero']
    assert zero['config_digest'] == chosen['config_digest']
    selected = module.select(job, run, tmp_path / 'compatibility.json')
    assert selected['selected'] == 'step-8' and load_receipt(selected['receipt']).data['job_id'] == resolved['id']


def test_post_migration_configuration_edit_invalidates_the_compatibility_record(tmp_path, monkeypatch):
    from semantic_trainer.migrate import migrate
    module, job, run, resolved, *_ = selection(tmp_path, monkeypatch)
    legacy(run, resolved)
    assert migrate(job, run, tmp_path / 'compatibility.json')['eligible']
    config = run / 'candidates' / 'step-16' / 'adapter_config.json'
    config.write_text(json.dumps({**files.read_json(config), 'inference_mode': False}))
    with pytest.raises(ConfigError, match='adapter configuration drift since fitting'):
        module.select(job, run, tmp_path / 'compatibility.json')
    weights = run / 'candidates' / 'step-16' / 'adapter_model.safetensors'
    config.write_text(json.dumps({**files.read_json(config), 'inference_mode': True}))
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(bytes(payload))
    with pytest.raises(ConfigError, match='adapter weight drift since fitting'):
        module.select(job, run, tmp_path / 'compatibility.json')


def test_current_schema_candidates_record_and_recheck_the_normalized_configuration(tmp_path, monkeypatch):
    module, job, run, resolved, *_ = selection(tmp_path, monkeypatch)
    listed = files.read_json(run / 'candidates.json')
    assert all(len(candidate['config_digest']) == 64 for candidate in listed['candidates'])
    config = run / 'candidates' / 'step-8' / 'adapter_config.json'
    files.publish(config, {**files.read_json(config), 'lora_alpha': 999})
    candidates_with(run, 'step-8', {'config_sha256': files.sha256_file(config)})
    with pytest.raises(ConfigError, match=r"adapter configuration drift in \['lora_alpha'\]"):
        module.select(job, run)
