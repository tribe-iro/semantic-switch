from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from semantic_switch import files, model_job
from bench.outcomes import failure, measure


def project(job, shape, output):
    from semantic_trainer.jobs import resolve
    from safetensors.torch import save
    from semantic_trainer.lora import answer_logits, collate, dependencies, group_loss, hf_bytes, targets
    resolved, rows = resolve(job)
    torch, AutoConfig, AutoModel, AutoTokenizer, LoraConfig, get_peft_model = dependencies()
    torch.manual_seed(resolved['recipe']['seed'])
    torch.set_num_threads(4)
    hf, dtype = resolved['model']['hf'], resolved['model']['dtype']
    if AutoConfig.from_pretrained(hf, local_files_only=True).model_type == 'gemma4':
        from transformers import AutoModelForImageTextToText
        AutoModel = AutoModelForImageTextToText
    with model_job(extra={'hf_weights': hf_bytes(resolved)}):
        base = AutoModel.from_pretrained(hf, local_files_only=True, dtype=getattr(torch, dtype), attn_implementation='eager')
        base.requires_grad_(False)
        recipe = resolved['recipe']
        model = get_peft_model(base, LoraConfig(r=recipe['rank'], lora_alpha=recipe['alpha'], target_modules=list(targets(base, recipe['target_modules'])), task_type='CAUSAL_LM')).eval()
        tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
        entry = rows[0]['native']['presentations'][0]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokens, mask, lengths = collate([entry], pad)
        kept = []
        def step():
            logits = answer_logits(model, [entry], pad)[0] if shape == 'answer' else model(input_ids=tokens, attention_mask=mask, use_cache=False).logits[0, lengths[0] - 1]
            loss, conditional, mass = group_loss(logits, entry['answer_slots'], entry['label'])
            loss.backward()
            kept.append(logits.detach().float().contiguous())
            return {'nll': float(loss.detach()), 'conditional_nll': float(conditional.detach()), 'mass': float(mass.detach())}
        measured = measure(step)
        tensors = {name: parameter.grad.detach().float().contiguous() for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
    report = {'schema': 2, 'population': 'single matched fitting presentation; performance and projection equivalence only', 'job': resolved['id'],
              'presentation': entry['id'], 'shape': shape, 'dtype': dtype, 'tokens': int(lengths[0]), 'seconds': measured.ms / 1000,
              'window_peak_rss_bytes': measured.peak_rss_bytes, 'torch': version('torch'), 'transformers': version('transformers')}
    if measured.error:
        return files.publish(Path(output), failure(measured, **report))
    report |= measured.output
    return files.publish(Path(output), report, {Path(output).with_suffix('.safetensors').name: [save({'answer_logits': kept[0], **tensors})]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--shape', choices=('answer', 'full'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project(args.job, args.shape, args.output)
