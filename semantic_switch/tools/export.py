from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from ..artifacts import ExportEncoder
from ..compiler import digest
from ..context import bind_input
from ..examples import path_record, read_envelopes
from ..errors import ConfigError
from ..protocols import MAX_QUESTION_TOKENS
from ..kernel import F
from ..results import thaw
from .common import input_key
from .data import Labels


def destination(sw, to, labels):
    out = Path(to) if to else sw.paths.yaml.with_suffix('.export.jsonl')
    manifest_path = Path(str(out) + '.manifest.json')
    sources = {sw.paths.yaml, sw.paths.examples, sw.paths.test, sw.paths.lock, sw.paths.dataset, *((Path(labels),) if labels else ())}
    if out.resolve() in {p.resolve() for p in sources} or manifest_path.resolve() in {p.resolve() for p in sources}:
        raise ConfigError('export destination collides with switch or label files')
    if labels is not None:
        if not Path(labels).is_file():
            raise ConfigError(f'{labels}: label file not found')
        read_envelopes(sw.plan, labels, labelled=True)
    return out, manifest_path


def native_binding(rt, plan, scope):
    bound = sorted({plan.binding_of(plan.fact_index[plan.at(*path).name]) for path in scope})
    if len(bound) != 1:
        raise ConfigError(f'a native export scores one binding; scope {["/".join(path) for path in scope]} spans bindings {bound}: export each binding scope separately')
    judge = rt.judge(bound[0])
    if not hasattr(judge, 'adapter'):
        raise ConfigError('native export requires a local tokenizer')
    identity = rt.identity(bound[0])
    return judge, {'judge': identity, 'execution': rt.execution(bound[0]), 'template': judge.adapter.name, 'framing': judge.adapter.framing.as_dict(),
                   'tokenizer': judge.tokenizer_identity(), 'presentation': identity['presentation']}


def prerequisite(plan, spec, label):
    options = thaw(plan.spec)['facts'].get(spec.name, {}).get('options') or {}
    option = options.get(spec.keys[label]) if isinstance(options, dict) and label < len(spec.keys) else None
    return option.get('requires') if isinstance(option, dict) else None


def refuse_gold(plan, spec, example, stage, eligible):
    requires = prerequisite(plan, spec, example.label)
    raise ConfigError(f"export observation {example.id} target {'/'.join(spec.path)} stage {stage}: gold {spec.keys[example.label]!r} is excluded by "
                      f"full-stage eligibility ({'requires ' + repr(requires) if requires else 'its prerequisites'} evaluate "
                      f"{'false' if eligible is not None and eligible[example.label] == F else 'unshown'} for this input) and the stage declares no escape; "
                      'correct the label or its prerequisite; no escape gold is invented')


def presentations(sw, judge, spec, example, seed, encoder):
    (preview,) = sw.preview([example.data], [spec.name], sources=('judge',), requests=False)
    stages = [stage for stage in preview.stages if stage.target == spec.path]
    if preview.error is not None or not stages:
        raise ConfigError(f"export {example.id}/{spec.name}: no judge presentation for this input ({preview.error or 'decided without the judge'})")
    adapter = judge.adapter
    state, out = adapter.state(example.text), []
    text = lambda groups: [[judge.scorer.detokenize([t]) for t in group] for group in groups]
    for stage in stages:
        question, slots = stage.presentation.question, stage.presentation.slots
        if example.label not in slots and -1 not in slots:
            refuse_gold(sw.plan, spec, example, stage.stage, stage.eligible)
        variants = [('canonical', question, list(slots))]
        if question.kind == 'pick' and len(slots) > 1:
            order = list(range(len(slots)))
            random.Random(digest([seed, example.id, stage.stage])).shuffle(order)
            if order == list(range(len(slots))):
                order = order[1:] + order[:1]
            variants.append(('permuted', replace(question, options=tuple(question.options[i] for i in order)), [slots[i] for i in order]))
        for variant, rendered, mapping in variants:
            request = adapter.request(state, [rendered])
            if len(request.state) > judge.max_input_tokens or len(request.suffixes[0]) > MAX_QUESTION_TOKENS:
                raise ConfigError(f'export {example.id}/{stage.stage}: token limits exceeded')
            out.append(encoder.presentation(example.id, example.label, stage.stage, variant, request.prefix, request.state, request.suffixes[0], request.slots[0],
                                            (rendered.text, tuple(rendered.options), tuple(rendered.excludes), rendered.candidate), text, mapping))
    return out


def exact(spec, target) -> bool:
    return target.eligible is not None and all(value == F for value in target.eligible[:len(spec.keys) - spec.allow_none])


def target_descriptor(spec):
    return {**path_record(spec.path), 'kind': spec.kind, 'meaning_rev': spec.meaning_rev, 'claim': spec.text, 'excludes': list(spec.judgment.excludes),
            'classes': list(spec.keys), 'values': [spec.value(index) for index in range(len(spec.keys))],
            **({'candidate': spec.judgment.candidate} if spec.member else {}), 'guidance': spec.judgment.guidance.as_dict(),
            'guidance_digest': spec.judgment.guidance.digest}


def owned(example):
    return {'id': example.id, 'label': example.label, 'fold': example.fold, 'split': example.split, 'taught': example.taught, 'pairs': list(example.pairs),
            'tags': list(example.tags), 'situations': list(example.situations), 'selections': [s.record() for s in example.selections],
            'label_sources': [source.record() for source in example.sources], 'label_derivation': example.label_derivation}


def situation_index(sw, labels, scope, split, exported):
    plan = sw.plan
    names = [plan.at(*path).name for path in scope]
    chosen = [(identity, entry) for identity, entry in sorted(labels.situations.items())
              if (where := labels.split(entry['input'])) != 'conflict' and (split in (None, 'all') or where == split)]
    rows = sw.score([entry['input'] for _, entry in chosen], names, sources=()) if chosen else []
    for (identity, entry), row in zip(chosen, rows):
        targets = []
        for path, name in zip(scope, names):
            linked = entry['labels'].get(tuple(path))
            if linked in exported:
                targets.append({'address': path, 'observation': linked})
                continue
            target = None if row.error else row.targets[name]
            targets.append({'address': path, 'outcome': 'missing' if target is None or target.missing else 'exact' if exact(plan.at(*path), target) else 'unlabelled'})
        yield {'id': identity, 'records': sorted(entry['records']), 'targets': targets,
               'complete': all('observation' in t or t['outcome'] == 'exact' for t in targets)}


def export(rt, path, to=None, split='tune', *, native=False, task_family=None, seed=0, scope=None, labels=None):
    from ..training import target_scope
    if split not in ('tune', 'check', 'all', None):
        raise ConfigError('export split must be tune, check or all')
    if task_family is not None and (not isinstance(task_family, str) or not task_family.strip()):
        raise ConfigError('task_family must be nonempty')
    sw = rt.load(path, draft=True)
    plan = sw.plan
    scope = target_scope(plan, scope)
    out, manifest_path = destination(sw, to, labels)
    labels = Labels(sw, labels)
    judge, descriptor = native_binding(rt, plan, scope) if native else (None, None)
    encoder = ExportEncoder({'dataset_id': labels.dataset_id, 'training_id': labels.training_id, 'namespace': labels.manifest['namespace'],
                             'split': split or 'all', 'scope': scope, 'task_family': task_family, 'native': descriptor})
    exported, skipped_rule = set(), 0

    def observations():
        nonlocal skipped_rule
        for name in labels.by_fact:
            spec = plan.fact(name)
            if list(spec.path) not in scope:
                continue
            target = encoder.target(name, lambda: target_descriptor(spec))
            examples = [example for example in labels.examples(name, None if split in (None, 'all') else split) if example.split != 'conflict']
            rows = sw.score([example.data for example in examples], [name], sources=()) if examples else []
            for example, row in zip(examples, rows):
                if row.error or row.targets[name].missing:
                    raise ConfigError(f'export {example.id}/{name}: invalid or missing input')
                eligible = row.targets[name].eligible
                if exact(spec, row.targets[name]):
                    skipped_rule += 1
                    continue
                if judge is None and eligible is not None and eligible[example.label] == F:
                    refuse_gold(plan, spec, example, 'full', eligible)
                projection = {'bound_input': thaw(bind_input(plan.fields, example.data)), 'projection': example.projection, 'text': example.text,
                              'decision_key': input_key(plan, example.data)}
                family = {'group': example.group, 'component': labels.component(example.group),
                          'aliases': labels.manifest['families'][example.group]['aliases'], 'family_seeds': labels.seeds(example.group)}
                found = None if judge is None else presentations(sw, judge, spec, example, seed, encoder)
                exported.add(example.id)
                yield encoder.observation(owned(example), target, projection, family, found)

    manifest = encoder.publish(manifest_path, observations(), lambda: situation_index(sw, labels, scope, split, exported))
    situations = encoder.situations
    return {'rows': len(encoder.digests), 'skipped_rule': skipped_rule, 'path': str(manifest_path), 'export_id': manifest['export_id'],
            'situations': len(situations), 'complete_situations': sum(entry['complete'] for entry in situations), 'dataset_id': labels.dataset_id,
            'split': split or 'all'}
