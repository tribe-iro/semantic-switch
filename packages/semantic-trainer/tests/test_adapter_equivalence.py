import hashlib
import json
import sys

import pytest

from semantic_switch import ConfigError, files
from semantic_switch.artifacts import load_receipt
from semantic_testkit.candidates import Native
from test_specialize import PEFT_CONFIG, lora_config, peft_adapter, recorded, selection

FITTED = recorded({'targets': {f'model.language_model.layers.{layer}.{block}.{name}': shape for layer in (0, 1)
                               for block, name, shape in (('self_attn', 'q_proj', [16, 8]), ('self_attn', 'o_proj', [8, 16]), ('mlp', 'down_proj', [8, 24]))},
                   'recipe': {'rank': 4, 'alpha': 8, 'dropout': 0.05}, 'model': {'revision': 'c' * 40}})
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
                'revision': {'revision': 'c' * 40},
                'version': {'peft_version': ...}}
    for name, change in variants.items():
        found = identify(peft_adapter(tmp_path / name, FITTED, **change))
        assert found['config_digest'] == original['config_digest'] and found['config_sha256'] != original['config_sha256'], name


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


def test_publication_records_the_complete_effective_configuration_and_validation_never_reconstructs_it(tmp_path):
    from semantic_trainer.adapter import COMPLETE, effective
    serialized = {key: value for key, value in lora_config(FITTED).items() if key not in DEFAULTED}
    complete = effective({**serialized, 'target_modules': set(FITTED['targets'])}, '0.21.2')
    assert list(complete) == list(COMPLETE) and complete == {key: lora_config(FITTED)[key] for key in COMPLETE} | {'target_modules': sorted(FITTED['targets'])}
    assert complete['use_dora'] is False and complete['use_rslora'] is False and complete['loftq_config'] == {} and complete['kasa_config'] is None
    with pytest.raises(ConfigError, match=r"PEFT 0.9.0 omits adapter settings \['use_rslora', 'exclude_modules', 'trainable_token_indices', 'loftq_config', 'use_dora', 'lora_bias', "
                                          r"'arrow_config', 'kasa_config', 'ensure_weight_tying'\] and has no documented default profile"):
        effective(serialized, '0.9.0')
    assert effective(PEFT_CONFIG, '0.9.0')['r'] == 8
    with pytest.raises(ConfigError, match=r"unsupported adapter settings \['use_future_variant'\]"):
        effective({**PEFT_CONFIG, 'use_future_variant': False}, '0.21.2')
    adapter = peft_adapter(tmp_path / 'adapter', FITTED)
    partial = {**FITTED, 'adapter': {key: value for key, value in FITTED['adapter'].items() if key != 'use_dora'}}
    for fitted in ({key: value for key, value in FITTED.items() if key != 'adapter'}, partial, {**FITTED, 'dependencies': {}}):
        with pytest.raises(ConfigError, match='lacks the complete effective adapter configuration and PEFT version'):
            identify(adapter, fitted)
    with pytest.raises(ConfigError, match=r"drift in \['lora_alpha'\]"):
        identify(adapter, {**FITTED, 'adapter': {**FITTED['adapter'], 'lora_alpha': 999}})


def candidates_with(run, name, recorded_):
    listed = files.read_json(run / 'candidates.json')
    listed['candidates'] = [{**candidate, **recorded_} if candidate['name'] == name else candidate for candidate in listed['candidates']]
    files.publish(run / 'candidates.json', listed)


def refused(module, monkeypatch):
    def admission(*args, **kwargs):
        raise AssertionError('a rejected run reached model admission')
    monkeypatch.setattr(module, 'model_job', admission)


def test_complete_current_runs_publish_and_select_against_their_original_fitted_record(tmp_path, monkeypatch):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    listed, fitted = files.read_json(run / 'candidates.json'), files.read_json(run / 'resolved-job.json')
    canonical = json.dumps(fitted, sort_keys=True, ensure_ascii=False).encode()
    assert listed['fitted_job_id'] == hashlib.sha256(canonical).hexdigest() and fitted['id'] == listed['job_id'] == resolved['id']
    assert fitted['adapter']['lora_alpha'] == resolved['recipe']['alpha'] and fitted['dependencies']['peft'] == '0.21.2'
    assert all(len(candidate['config_digest']) == 64 for candidate in listed['candidates'])
    selected = module.select(job, run)
    assert selected['selected'] == 'step-8' and load_receipt(selected['receipt']).data['job_id'] == resolved['id']
    assert [name for name, _, _ in deployed] == ['zero', 'step-8', 'step-16']


@pytest.mark.parametrize('edit', [
    lambda fitted: {**fitted, 'recipe': {**fitted['recipe'], 'learning_rate': fitted['recipe']['learning_rate'] * 2}},
    lambda fitted: {**fitted, 'monitor': {**fitted['monitor'], 'fields': {**fitted['monitor']['fields'], 'cadence': '0' * 64}}},
    lambda fitted: {**fitted, 'assets': {**fitted['assets'], 'manifest': 'f' * 64}},
    lambda fitted: {**fitted, 'model': {**fitted['model'], 'files': {**fitted['model']['files'], 'config.json': 'e' * 64}}},
    lambda fitted: {**fitted, 'adapter': {**fitted['adapter'], 'lora_dropout': 0.5}},
], ids=['learning_rate', 'monitor', 'assets', 'base_files', 'adapter'])
def test_edited_fitted_metadata_fails_before_conversion_and_the_authentic_record_restores_eligibility(tmp_path, monkeypatch, edit):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    path = run / 'resolved-job.json'
    original = path.read_bytes()
    before = {name: files.sha256_file(run / name) for name in ('candidates.json', 'candidates/step-8/adapter_model.safetensors', 'candidates/step-8/adapter_config.json')}
    files.publish(path, edit(json.loads(original)))
    selecting = module.model_job
    refused(module, monkeypatch)
    with pytest.raises(ConfigError, match='resolved-job.json is not the record its candidates were fitted under .*restore the original fitted record or refit with '
                                          rf'semantic-trainer lora {job} --to NEW_RUN; edited metadata alone does not establish changed weights'):
        module.select(job, run)
    assert deployed == [] and Native.built == [] and not (run / 'deployment').exists()
    assert {name: files.sha256_file(run / name) for name in before} == before
    path.write_bytes(original)
    monkeypatch.setattr(module, 'model_job', selecting)
    assert module.select(job, run)['selected'] == 'step-8'


def test_a_redigested_record_cannot_replace_the_recorded_fitting(tmp_path, monkeypatch):
    from semantic_switch.artifacts import artifact_digest
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    fitted, listed = files.read_json(run / 'resolved-job.json'), files.read_json(run / 'candidates.json')
    refused(module, monkeypatch)
    def forged(record, match):
        files.publish(run / 'resolved-job.json', record)
        files.publish(run / 'candidates.json', {**listed, 'fitted_job_id': artifact_digest(record)})
        with pytest.raises(ConfigError, match=match):
            module.select(job, run)
    forged({**fitted, 'recipe': {**fitted['recipe'], 'learning_rate': 1.}}, f"resolved-job.json does not describe job {resolved['id']}")
    forged({**fitted, 'fit': {'fields': {**fitted['fit']['fields'], 'optimizer': '0' * 64}, 'id': fitted['fit']['id']}}, r"changed numerical fitting inputs \['optimizer'\]")
    forged({**fitted, 'id': 'a' * 64}, 'name different job, fit or monitor identities')
    files.publish(run / 'resolved-job.json', fitted)
    files.publish(run / 'candidates.json', {key: value for key, value in listed.items() if key != 'fitted_job_id'})
    with pytest.raises(ConfigError, match='candidates.json records no fitted_job_id, so the fitted record cannot be proven'):
        module.select(job, run)
    files.publish(run / 'candidates.json', listed)
    files.publish(run / 'run.json', {**files.read_json(run / 'run.json'), 'monitor_id': 'b' * 64})
    with pytest.raises(ConfigError, match='name different job, fit or monitor identities'):
        module.select(job, run)
    assert deployed == [] and Native.built == []


def test_changed_numerical_fitting_inputs_require_a_refit(tmp_path, monkeypatch):
    from semantic_trainer.jobs import prepare
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    export = resolved['sources'][0]['path']
    faster = prepare(definition, [export], tmp_path / 'faster', assets_manifest=resolved['assets']['manifest'], recipe={'eval_every_examples': 8, 'learning_rate': 4e-4},
                     native={'adapter': 'qwen3', 'device': 'cpu', 'n_ctx': 2048})['job']
    refused(module, monkeypatch)
    with pytest.raises(ConfigError, match=rf"changed numerical fitting inputs \['optimizer'\]; the candidates do not fit this job: refit with semantic-trainer lora {faster}"):
        module.select(faster, run)
    assert deployed == []


@pytest.mark.parametrize('name, schema', [('candidates.json', 6), ('resolved-job.json', 7), ('run.json', 3)])
def test_unsupported_fitting_schemas_require_a_refit_without_opening_a_model(tmp_path, monkeypatch, name, schema):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    files.publish(run / name, {**files.read_json(run / name), 'schema': schema})
    refused(module, monkeypatch)
    with pytest.raises(ConfigError, match=rf'unsupported fitting records \({name} schema {schema}; current schemas .*refit the current job with semantic-trainer lora {job} --to NEW_RUN'):
        module.select(job, run)
    assert deployed == [] and Native.built == []


@pytest.mark.parametrize('argv', [['migrate', 'job.yaml', '--run', 'run', '--to', 'record.json'], ['select', 'job.yaml', '--run', 'run', '--compatibility', 'record.json']])
def test_historical_compatibility_commands_are_rejected_before_any_model_work(monkeypatch, capsys, argv):
    from semantic_trainer import cli, select
    def selecting(*args):
        raise AssertionError('selection started')
    monkeypatch.setattr(select, 'select', selecting)
    monkeypatch.setattr(sys, 'argv', ['semantic-trainer', *argv])
    with pytest.raises(SystemExit) as exited:
        cli.main()
    assert exited.value.code == 2 and ('invalid choice' in capsys.readouterr().err if argv[0] == 'migrate' else 'unrecognized arguments: --compatibility' in capsys.readouterr().err)


@pytest.mark.parametrize('change, match', [
    ({'lora_alpha': 999}, r"adapter configuration drift in \['lora_alpha'\]"),
    ({'r': 8}, r"adapter configuration drift in \['r'\]"),
    ({'target_modules': ['q_proj']}, r"adapter configuration drift in \['target_modules'\]"),
    ({'revision': 'd' * 40}, 'base revision .* differs from the pinned'),
])
def test_changed_adapter_configuration_fails_selection_even_with_relisted_hashes(tmp_path, monkeypatch, change, match):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    config = run / 'candidates' / 'step-8' / 'adapter_config.json'
    files.publish(config, {**files.read_json(config), **change})
    candidates_with(run, 'step-8', {'config_sha256': files.sha256_file(config)})
    refused(module, monkeypatch)
    with pytest.raises(ConfigError, match=match):
        module.select(job, run)
    assert deployed == []


def test_changed_base_weight_or_configuration_bytes_fail_selection(tmp_path, monkeypatch):
    module, job, run, resolved, records, deployed, definition = selection(tmp_path, monkeypatch)
    refused(module, monkeypatch)
    listed = files.read_json(run / 'candidates.json')
    files.publish(run / 'candidates.json', {**listed, 'base_sha256': 'f' * 64})
    with pytest.raises(ConfigError, match='not the pipeline-built Q4_0 base'):
        module.select(job, run)
    files.publish(run / 'candidates.json', listed)
    config = run / 'candidates' / 'step-16' / 'adapter_config.json'
    original = config.read_bytes()
    config.write_text(json.dumps({**files.read_json(config), 'inference_mode': False}))
    with pytest.raises(ConfigError, match=r'adapter configuration drift since fitting \(config_sha256\)'):
        module.select(job, run)
    config.write_bytes(original)
    weights = run / 'candidates' / 'step-16' / 'adapter_model.safetensors'
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(bytes(payload))
    with pytest.raises(ConfigError, match='adapter weight drift since fitting'):
        module.select(job, run)
    assert deployed == [] and Native.built == []
