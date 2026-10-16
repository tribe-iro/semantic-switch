from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from semantic_switch import files
from semantic_switch.training import model_job
from bench.outcomes import destination, failure, measure


def project(job, shape, output):
    from safetensors.torch import save
    from semantic_trainer.jobs import resolve
    from semantic_trainer.lora import dependencies, hf_bytes, targets
    from semantic_trainer.objective import answer_logits, bind, collate, packed, prepared, slot_loss, slot_statistics
    output = destination(output)
    resolved, rows = resolve(job)
    torch, AutoConfig, AutoModel, AutoTokenizer, LoraConfig, get_peft_model = dependencies()
    recipe, precision = resolved['recipe'], resolved['fitting']['precision']
    torch.manual_seed(recipe['seed'])
    torch.set_num_threads(4)
    hf = resolved['model']['hf']
    if AutoConfig.from_pretrained(hf, local_files_only=True).model_type == 'gemma4':
        from transformers import AutoModelForImageTextToText
        AutoModel = AutoModelForImageTextToText
    with model_job(extra={'hf_weights': hf_bytes(resolved)}):
        base = AutoModel.from_pretrained(hf, local_files_only=True, dtype=getattr(torch, precision), attn_implementation='eager')
        base.requires_grad_(False)
        model = get_peft_model(base, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'], target_modules=list(targets(base, recipe['target_modules'])),
                                                task_type='CAUSAL_LM')).eval()
        binding = bind(model)
        tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        presentation = rows[0]['presentations'][0]
        entry = prepared(rows[:1], binding.vocabulary)[rows[0]].presentations[0]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokens, mask, lengths = collate([entry], pad)
        slots = packed([entry.answers]).to(binding.device)
        kept = []

        def step():
            logits = answer_logits(binding, [entry], pad) if shape == 'answer' else \
                model(input_ids=tokens.to(binding.device), attention_mask=mask.to(binding.device), use_cache=False).logits[:, lengths[0] - 1]
            loss = slot_loss(slot_statistics(logits, slots), slots.gold)
            loss.nll.sum().backward()
            kept.append(logits[0].detach().float().contiguous())
            return {'nll': float(loss.nll[0].detach()), 'conditional_nll': float(loss.conditional[0].detach()), 'mass': float(loss.mass[0].detach())}
        measured = measure(step)
        tensors = {name: parameter.grad.detach().float().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
    report = {'schema': 3, 'population': 'single matched fitting presentation; performance and projection equivalence only', 'job': resolved['id'],
              'presentation': presentation.id, 'shape': shape, 'precision': precision, 'tokens': int(lengths[0]), 'seconds': measured.ms / 1000,
              'window_peak_rss_bytes': measured.peak_rss_bytes, 'torch': version('torch'), 'transformers': version('transformers')}
    if measured.error:
        return files.publish(output, failure(measured, **report))
    report |= measured.output
    return files.publish(output, report, {output.with_suffix('.safetensors').name: [save({'answer_logits': kept[0], **tensors})]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='explicit experiment: answer-position projection versus full-forward logits for one fitting presentation')
    parser.add_argument('job')
    parser.add_argument('--shape', choices=('answer', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project(args.job, args.shape, args.output)
