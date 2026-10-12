from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from .. import files
from ..artifacts import EXPORT_SCHEMA, ROW_SCHEMA, export_identity
from ..compiler import digest
from ..examples import path_record, read_envelopes
from ..errors import ConfigError
from ..judge import MAX_QUESTION_TOKENS
from ..kernel import F
from ..results import thaw
from ..switch import Switch
from .common import input_key
from .data import Labels


def export(rt, path, to=None, split='tune', *, native=False, task_family=None, seed=0, fact=None, labels=None):
    if split not in ('tune', 'check', 'all', None):
        raise ConfigError('export split must be tune, check or all')
    sw = Switch(rt, path, draft=True)
    plan = sw.plan
    if fact is not None:
        plan.fact(fact)
    out = Path(to) if to else sw.paths.yaml.with_suffix('.export.jsonl')
    manifest_path = Path(str(out) + '.manifest.json')
    sources = {sw.paths.yaml, sw.paths.examples, sw.paths.test, sw.paths.lock, sw.paths.dataset, *((Path(labels),) if labels else ())}
    if out.resolve() in {p.resolve() for p in sources} or manifest_path.resolve() in {p.resolve() for p in sources}:
        raise ConfigError('export destination collides with switch or label files')
    if labels is not None:
        if not Path(labels).is_file():
            raise ConfigError(f'{labels}: label file not found')
        read_envelopes(plan, labels, labelled=True)
    labels = Labels(sw, labels)
    adapter, identity, execution = None, None, None
    if native:
        judge = rt.judge()
        if not hasattr(judge, 'adapter'):
            raise ConfigError('native export requires a local tokenizer')
        adapter, identity, execution = judge.adapter, rt.identity(), rt.execution()
        tokenizer = judge.tokenizer_identity()
    skipped_rule, entries = 0, []

    def lines():
        nonlocal skipped_rule
        for name in labels.by_fact:
            if fact is not None and plan.fact(name).path[0] != fact:
                continue
            spec, f = plan.fact(name), plan.fact_index[name]
            for example in labels.examples(name, None if split in (None, 'all') else split):
                if example.split == 'conflict':
                    continue
                call = sw.exec.start(sw.snapshot, sw.bind(example.data, sw.snapshot))
                if call.error or call.facts[f].missing:
                    raise ConfigError(f'export {example.id}/{name}: invalid or missing input')
                eligible = call.facts[f].eligible
                if eligible is not None and all(value == F for value in eligible[:len(spec.keys) - spec.allow_none]):
                    skipped_rule += 1
                    continue
                if eligible is not None and eligible[example.label] == F:
                    raise ConfigError(f'export {example.id}/{name}: gold option is ineligible')
                row = {'schema': ROW_SCHEMA, 'id': example.id, **path_record(spec.path), 'kind': spec.kind,
                       'meaning_rev': spec.meaning_rev, 'claim': spec.text, 'excludes': list(spec.judgment.excludes), 'classes': list(spec.keys), 'label': example.label,
                       **({'candidate': spec.judgment.candidate} if spec.member else {}),
                       'value': spec.value(example.label), 'guidance': spec.judgment.guidance.as_dict(),
                       'guidance_digest': spec.judgment.guidance.digest, 'bound_input': thaw(sw.bind(example.data, sw.snapshot)),
                       'projection': example.projection, 'text': example.text, 'group': example.group, 'component': labels.component(example.group),
                       'aliases': labels.manifest['families'][example.group]['aliases'],
                       'fold': example.fold, 'split': example.split, 'taught': example.taught,
                       'pairs': list(example.pairs), 'family_seeds': labels.seeds(example.group), 'decision_key': input_key(plan, example.data),
                       'selections': [s.record() for s in example.selections], 'label_sources': [source.record() for source in example.sources],
                       'label_origin': example.label_origin, 'tags': list(example.tags), 'situations': list(example.situations)}
                if adapter is not None:
                    stages = [sw.exec.presentation(sw.snapshot, call, f)]
                    if stages[0].stage == 'mentions':
                        call.facts[f].focus_done = True
                        stages.append(sw.exec.presentation(sw.snapshot, call, f))
                    presentations = []
                    for stage, question, slots in ((p.stage, p.question, p.slots) for p in stages):
                        variants = [('canonical', question, list(slots))]
                        if question.kind == 'pick' and len(slots) > 1:
                            order = list(range(len(slots)))
                            random.Random(digest([seed, example.id, stage])).shuffle(order)
                            if order == list(range(len(slots))):
                                order = order[1:] + order[:1]
                            variants.append(('permuted', replace(question, options=tuple(question.options[i] for i in order)), [slots[i] for i in order]))
                        for variant, rendered, mapping in variants:
                            gold = mapping.index(example.label) if example.label in mapping else mapping.index(-1)
                            request = adapter.request(adapter.state(example.text), [rendered])
                            if len(request.state) > judge.max_input_tokens or len(request.suffixes[0]) > MAX_QUESTION_TOKENS:
                                raise ConfigError(f'export {example.id}/{stage}: token limits exceeded')
                            entry = {'stage': stage, 'variant': variant, 'slot_map': mapping, 'label': gold, 'global_label': example.label,
                                     'prefix': list(request.prefix), 'input': list(request.state),
                                     'question': list(request.suffixes[0]), 'answer_slots': [list(s) for s in request.slots[0]],
                                     'answer_text': [[rt.detokenize([t]) for t in group] for group in request.slots[0]],
                                     'claim': rendered.text, 'options': list(rendered.options), 'excludes': list(rendered.excludes),
                                     **({'candidate': rendered.candidate} if rendered.candidate else {}),
                                     'pending': sorted(call.facts[f].pending)}
                            entry['id'] = digest([example.id, entry])
                            presentations.append(entry)
                    row['native'] = {'judge': identity, 'execution': execution, 'template': adapter.name, 'presentations': presentations,
                                     'presentation': identity['presentation'], 'framing': adapter.framing.as_dict(),
                                     'tokenizer': tokenizer, 'authored_kind': spec.kind}
                    if task_family is not None:
                        if not isinstance(task_family, str) or not task_family.strip():
                            raise ConfigError('task_family must be nonempty')
                        row['task_family'] = task_family
                line = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'
                entries.append({**{k: row[k] for k in ('id', 'fact', 'member', 'projection', 'group', 'component', 'fold', 'split', 'meaning_rev', 'family_seeds') if k in row},
                                'sha256': hashlib.sha256(line.encode()).hexdigest()})
                yield line
    payload = list(lines())
    scope = [list(plan.facts[f].path) for f in plan.observed if fact is None or plan.facts[f].path[0] == fact]
    exported = {entry['id'] for entry in entries}
    index = []
    for identity, entry in sorted(labels.situations.items()):
        where = labels.split(entry['input'])
        if where == 'conflict' or split not in (None, 'all') and where != split:
            continue
        call = sw.exec.start(sw.snapshot, sw.bind(entry['input'], sw.snapshot))
        targets = []
        for path in scope:
            observation = entry['labels'].get(tuple(path))
            if observation in exported:
                targets.append({'address': path, 'observation': observation})
                continue
            f = plan.fact_index[plan.at(*path).name]
            state = None if call.error else call.facts[f]
            eligible = None if state is None else state.eligible
            exact = eligible is not None and all(value == F for value in eligible[:len(plan.facts[f].keys) - plan.facts[f].allow_none])
            targets.append({'address': path, 'outcome': 'missing' if state is None or state.missing else 'exact' if exact else 'unlabelled'})
        index.append({'id': identity, 'records': sorted(entry['records']), 'targets': targets,
                      'complete': all('observation' in t or t['outcome'] == 'exact' for t in targets)})
    manifest = {'schema': EXPORT_SCHEMA, 'native': native, 'dataset_id': labels.dataset_id, 'training_id': labels.training_id, 'namespace': labels.manifest['namespace'],
                'split': split or 'all', 'scope': scope, 'situations': index, 'rows': entries}
    manifest['export_id'] = export_identity(manifest)
    files.publish(manifest_path, manifest, {out.name: payload})
    return {'rows': len(entries), 'skipped_rule': skipped_rule, 'path': str(manifest_path), 'export_id': manifest['export_id'],
            'situations': len(index), 'complete_situations': sum(entry['complete'] for entry in index), 'dataset_id': labels.dataset_id, 'split': split or 'all'}
