from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
import unicodedata

from semantic_switch import ConfigError, Model, Runtime, files, judge
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import compile_spec
from semantic_switch.evidence import LABEL_SCHEMA
from semantic_switch.metrics import comparison_identity, paired, per_class as reduce_classes
from bench.outcomes import Progress
from semantic_switch.tools import score_observations
from semantic_switch.tools.evaluate import contrast_pairs
from semantic_switch.tools.export import export
from bench import outcomes
from bench.outcomes import fingerprint, finish, judge_spec, latency, recorded, resumable, resumed, runtime, work_key
from bench.policy_adaptation import criterion, fixture_sources

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'specialists' / 'routing'
DEFINITION = TASK / 'routing.yaml'
COHORT = ROOT / 'tests' / 'fixtures' / 'small-context-cohort-v2.json'
SETS = {'provisional_check': 'check-provisional', 'cohort_v2_routing': 'cohort-routing'}
FACT = {False: 'routing', True: 'routing_reference'}


def words(text):
    text = unicodedata.normalize('NFKD', text.lower())
    tokens = re.findall(r'[a-z0-9]+', ''.join(c for c in text if not unicodedata.combining(c)))
    return {' '.join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.


def provisional_check():
    draft, detail = files.read_jsonl(TASK / 'check' / 'check.draft.jsonl'), files.read_jsonl(TASK / 'check' / 'check.draft.detail.jsonl')
    out = []
    for record, info in zip(draft, detail, strict=True):
        if record['input'] != info['input']:
            raise ConfigError('check.draft.jsonl and check.draft.detail.jsonl are not aligned')
        agreed = info['supplied'] and all(info['settled'].values()) and set(info['teachers'].values()) == {info['intended']}
        if agreed != ('facts' in record) or agreed and record['facts'] != {info['fact']: info['intended']}:
            raise ConfigError(f"{info['group']}: draft value disagrees with the detail's teacher/author agreement")
        if agreed:
            out.append({**record, 'tags': sorted({*record.get('tags', ()), info['phenomenon'], *(('translation',) if info['translation'] else ())})})
    return out


def cohort_routing():
    cohort, out = json.loads(COHORT.read_text()), []
    plan = compile_spec(files.read_yaml(DEFINITION), DEFINITION.name, DEFINITION.parent)
    for situation in cohort['situations']:
        if situation['domain'] != 'customer_routing':
            continue
        family = f"cohort-{situation['id']}"
        for variant in situation['variants']:
            if variant['policy'] != 'routing_v1':
                continue
            for language in ('en', 'pt'):
                text = variant[language]
                suffix = '' if language == 'en' else '/pt'
                pairs = [] if variant['id'] == 'base' else [f"{family}/{variant['id']}{suffix}"]
                pairs += [f"{family}/{other['id']}{suffix}" for other in situation['variants'] if other.get('contrast', 'base') == variant['id'] and other['id'] != 'base'
                          and other['policy'] == 'routing_v1']
                facts = {FACT['reference' in text]: variant['gold']}
                out.append({'schema': LABEL_SCHEMA, 'input': dict(text), 'facts': facts, 'groups': [family], 'pairs': pairs,
                            'tags': sorted({*variant['tags'], *(('pt',) if language == 'pt' else ())}), 'role': 'check',
                            'label_sources': fixture_sources(plan, {'input': dict(text), 'facts': facts}, COHORT, f"{situation['id']}:{variant['id']}:{language}",
                                                             criterion(cohort, 'routing_v1'))})
    return out


def manifests(work):
    return {name: Path(work) / 'exports' / f'{stem}.export.jsonl.manifest.json' for name, stem in SETS.items()}


def scores(work, arm):
    return Path(work) / 'scores' / f'{arm}.json'


def exported(rows):
    return {row['id']: row for row in read_export(rows, fitting=False)[1]}


def build(args):
    labels, paths = args.work / 'labels', manifests(args.work)
    labels.mkdir(parents=True, exist_ok=True)
    made = {'provisional_check': provisional_check(), 'cohort_v2_routing': cohort_routing()}
    out = {}
    with Runtime(judge('openrouter:google/gemma-4-31b-it', key='export-only'), own_judges=True) as rt:
        for name, records in made.items():
            path = labels / f'{name}.jsonl'
            files.publish(path, ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in records))
            result = export(rt, DEFINITION, str(paths[name]).removesuffix('.manifest.json'), 'check', labels=path)
            out[name] = {'labels': len(records), 'rows': result['rows'], 'dataset_id': result['dataset_id']}
    held = [row for row in read_export(args.fit, fitting=True)[1]]
    seeds = {tuple(seed) for row in held for seed in row['family_seeds']} | {('group', row['group']) for row in held}
    shingles = [words(' '.join(row['bound_input'].values())) for row in held]
    for name, path in paths.items():
        rows = list(exported(path).values())
        crossing = sorted({row['group'] for row in rows if row['split'] != 'check' or seeds & ({tuple(s) for s in row['family_seeds']} | {('group', row['group'])})})
        near = [max(jaccard(words(' '.join(row['bound_input'].values())), other) for other in shingles) for row in rows]
        out[name] |= {'families': len({row['group'] for row in rows}), 'check_split': all(row['split'] == 'check' for row in rows),
                      'crossing_fit_select_acceptance_families': crossing, 'max_trigram_jaccard_vs_fit1000_tune': round(max(near), 3),
                      'rows_at_or_above_0.5': sum(n >= .5 for n in near)}
    out['tune_reference'] = {'export': str(args.fit), 'rows': len(held), 'families': len({row['group'] for row in held})}
    files.publish(args.work / 'eval-sets.json', out)
    return out


def score(args):
    paths = manifests(args.work)
    sets = {name: exported(path) for name, path in paths.items()}
    hosted = args.judge.startswith('openrouter:')
    if not hosted and args.native is None:
        raise ConfigError('a local arm names --native RESOLVED-JOB whose native execution settings every local arm shares')
    native = None if hosted else {**files.read_json(args.native)['native'], 'source': args.judge, 'threads': args.threads}
    config = {'native': native, 'script': 'bench.routing_specialist', 'code': fingerprint(),
              'definition': files.sha256_file(DEFINITION), 'arm': args.arm, 'judge': args.judge, 'provider': args.provider,
              'sets': {name: files.sha256_file(path) for name, path in paths.items()},
              'model_sha256': None if hosted else files.sha256_file(args.judge)}
    with resumable(scores(args.work, args.arm), config, fresh=args.fresh) as journal:
        keys = {work_key(name, key): (name, row) for name, rows in sets.items() for key, row in rows.items()}
        todo = journal.pending(keys, (lambda row: row['answer'] is None) if args.retry_failed else None)
        with runtime(judge_spec(args.judge, provider=args.provider, key_file=args.key_file, batch_inputs=args.batch_inputs) if hosted else Model(**native)) as bound:
            switch = bound.load(DEFINITION, draft=True)
            identity = bound.identity()
            progress = Progress(f'bench.routing_specialist {args.arm}', len(keys), unit='observations', resumed=resumed(journal, keys, todo))
            for key in todo:
                name, row = keys[key]
                started = time.perf_counter()
                try:
                    (result,) = score_observations(switch, [row])
                    error = None
                except Exception as failure:
                    result, error = {'result': {'prediction': None, 'status': 'failed'}, 'correct': None, 'accepted': False}, f'{type(failure).__name__}: {failure}'
                ms = (time.perf_counter() - started) * 1000
                committed = {'set': name, 'id': row['id'], 'answer': result['answer'], 'correct': result['correct'], 'accepted': result['accepted'],
                             'status': result['result']['status'], 'result': result['result'], 'error': error, 'ms': ms, 'outcome': 'failed' if error else 'predicted'}
                journal.commit([(key, committed)])
                progress.record([committed])
            progress.finish()
        return finish(journal, scores(args.work, args.arm), lambda rows: {'arm': args.arm, 'judge': args.judge, 'provider': args.provider, 'identity': identity,
                                               'rows': len(rows), 'failed': sum(row['answer'] is None for row in rows)})


def summarize(rows, scored, base=None):
    present = [i for i in rows if i in scored]
    classes = next(iter(rows.values()))['classes']
    per_class = reduce_classes((((), classes, classes[rows[i]['label']], scored[i]['answer'], scored[i]['correct']) for i in present)).get((), {})
    per_class = {key: per_class.get(key, {'gold': 0, 'predicted': 0, 'correct': 0, 'recall': None}) for key in classes}
    confusion = defaultdict(Counter)
    for i in present:
        confusion[classes[rows[i]['label']]][scored[i]['answer'] or 'failed'] += 1
    families = defaultdict(list)
    for i in present:
        families[rows[i]['group']].append(bool(scored[i]['correct']))
    pairs = [[bool(scored[i]['correct']) for i in ids] for ids in contrast_pairs([rows[i] for i in present]).values() if len(ids) >= 2]
    language = {lang: [i for i in present if ('pt' in rows[i]['tags']) == (lang == 'pt')] for lang in ('en', 'pt')}
    ms = [scored[i]['ms'] for i in present if scored[i]['answer'] is not None]
    out = {'raw_correct': sum(bool(scored[i]['correct']) for i in present), 'submitted': len(present),
           'predicted': sum(scored[i]['answer'] is not None for i in present), 'failures': sum(scored[i]['answer'] is None for i in present),
           'accepted_correct': sum(bool(scored[i]['correct']) and scored[i]['accepted'] for i in present),
           'accepted_wrong': sum(scored[i]['correct'] is False and scored[i]['accepted'] for i in present),
           'per_class': per_class, 'gold_vs_predicted': {gold: dict(counts) for gold, counts in confusion.items()},
           'contrast_pairs': {'pairs': len(pairs), 'both_correct': sum(all(p) for p in pairs)},
           'languages': {lang: {'raw_correct': sum(bool(scored[i]['correct']) for i in ids), 'submitted': len(ids)} for lang, ids in language.items()},
           'families_all_correct': sum(all(v) for v in families.values()), 'families': len(families),
           'latency': latency(ms),
           'failure_examples': [{'id': i, 'error': scored[i].get('error'), 'status': scored[i]['status']} for i in present if scored[i]['answer'] is None][:10]}
    if base is not None:
        before = {f: all(v) for f, v in base['families'].items()}
        expected = [{'key': i, 'family': rows[i]['group'], 'gold': rows[i]['label'], 'comparison': comparison(rows[i])} for i in rows]
        def evaluated(found):
            return [{'key': i, 'gold': rows[i]['label'], 'comparison': comparison(rows[i]), 'correct': None if found[i]['answer'] is None else bool(found[i]['correct']),
                     'accepted': found[i]['accepted'], 'outcome': 'failed' if found[i]['answer'] is None else 'predicted'} for i in present if i in found]
        compared = paired(expected, evaluated(base['scored']), evaluated(scored), seed=0, count=2000, names=('base', 'arm'))
        out['vs_base'] = {**compared, 'matched': compared['support']['matched'],
                          'family_fixes': sorted(f for f, v in families.items() if all(v) and not before.get(f, True)),
                          'family_regressions': sorted(f for f, v in families.items() if before.get(f) and not all(v)),
                          'class_regressions': {k: {'base': base['per_class'][k]['correct'], 'arm': v['correct']} for k, v in per_class.items()
                                                if v['correct'] < base['per_class'][k]['correct']},
                          'families_with_more_wrong_by_class': families_by_class(rows, scored, base['scored'], present)}
    return out, families


def comparison(row):
    return comparison_identity(task=[row['fact'], row.get('meaning_rev')], projection=row['id'], gold=row['label'], answers=row['classes'],
                               scope='observation', condition='routing')


def families_by_class(rows, scored, base, present):
    out = defaultdict(set)
    for i in present:
        if i in base and bool(base[i]['correct']) and not scored[i]['correct']:
            out[rows[i]['classes'][rows[i]['label']]].add(rows[i]['group'])
    return {k: sorted(v) for k, v in sorted(out.items())}


def curve(work, size):
    run = Path(work) / 'runs' / size
    metrics, resolved = files.read_json(run / 'metrics.json'), files.read_json(run / 'resolved-job.json')
    receipt = files.read_json(run / 'deployment' / 'specialist.gguf.receipt.json') if (run / 'deployment').exists() else None
    steps = [e['steps'] - (metrics['epochs'][i - 1]['steps'] if i else 0) for i, e in enumerate(metrics['epochs'])]
    groups = {row['id']: row['group'] for source in resolved['sources'] for row in read_export(source['path'], fitting=True)[1]}
    roles = {role: {'observations': len(resolved[key]), 'families': len({groups[i] for i in resolved[key]})}
             for role, key in (('fit', 'fit_ids'), ('model_select', 'selection_ids'), ('acceptance_fit', 'acceptance_ids'))}
    out = {'roles': roles, 'fitting': resolved['fitting'], 'placement': resolved['placement'],
           'trainable_parameters': resolved['trainable_parameters'], 'fit_wall_seconds': round(metrics['seconds'], 1),
           'epochs': [{'epoch': e['epoch'], 'loss': round(e['loss'], 4), 'seconds': round(e['seconds'], 1), 'steps': n, 'step_seconds': round(e['seconds'] / n, 2),
                       'peak_device_bytes': e['peak_device_bytes'], 'tokens': e['tokens'], 'fit_predicted': e['predicted']} for e, n in zip(metrics['epochs'], steps)],
           'peak_device_bytes': max(e['peak_device_bytes'] or 0 for e in metrics['epochs']), 'peak_host_bytes': metrics['peak_host_bytes'],
           'max_hf_native_base_logp_difference': round(metrics['max_hf_native_base_logp_difference'], 3),
           'fit_all_presentations_correct': {'before': sum(o['all_presentations_correct'] for o in metrics['learning_before']['observations']),
                                             'after': sum(o['all_presentations_correct'] for o in metrics['learning_after']['observations']),
                                             'of': len(metrics['learning_after']['observations'])}}
    if receipt is not None:
        selection = receipt['selection']
        out |= {'selected': receipt['selected'], 'specialist_sha256': receipt['gguf']['sha256'],
                'model_select': {'observations': selection['observations'], 'families': selection['families'],
                                 'candidates': [{k: c[k] for k in ('name', 'epoch', 'raw_correct', 'submitted', 'failures', 'class_regressions', 'family_fixes',
                                                                  'family_regressions', 'pairs', 'p50_ms')} | {'per_class_correct': {f'{fact}:{k}': v['correct'] for fact, cls in c['per_class'].items() for k, v in cls.items()}}
                                                for c in selection['candidates']]},
                'acceptance': {'policies': receipt['acceptance']['acceptance_fit']['policies'],
                               'provisional_check': {k: v for k, v in receipt['acceptance']['check'].items() if k not in ('rows', 'facts')} if receipt['acceptance']['check'] else None}}
    return out


def report(args):
    outcomes.destination(args.output)
    sets = {name: exported(path) for name, path in manifests(args.work).items()}
    arms = {}
    for name in args.arms:
        artifact = files.read_json(scores(args.work, name))
        header, rows = recorded(scores(args.work, name))
        arms[name] = {'meta': {**{k: artifact[k] for k in ('judge', 'provider', 'identity')}, 'native': header['configuration'].get('native')}, 'rows': list(rows.values())}
    out = {'arms': {}, 'matched': {}}
    for set_name, rows in sets.items():
        matched = [i for i in rows if all(any(r['set'] == set_name and r['id'] == i and r['answer'] is not None for r in arm['rows']) for arm in arms.values())]
        out['matched'][set_name] = {'rows': len(rows), 'scored_by_every_arm': len(matched)}
        base = None
        for name, arm in arms.items():
            scored = {r['id']: r for r in arm['rows'] if r['set'] == set_name}
            full, families = summarize(rows, scored, base)
            on_matched, _ = summarize({i: rows[i] for i in matched}, scored)
            if base is None:
                base = {'scored': scored, 'families': families, 'per_class': full['per_class']}
            out['arms'].setdefault(name, {'judge': arm['meta']['judge'], 'provider': arm['meta']['provider'],
                                          'identity': {k: arm['meta']['identity'].get(k) for k in ('source', 'sha256', 'model', 'provider', 'protocol') if k in arm['meta']['identity']},
                                          'native': arm['meta']['native'], 'sets': {}})['sets'][set_name] = {**full, 'matched_raw_correct': on_matched['raw_correct'], 'matched_submitted': on_matched['submitted']}
    out['curve'] = {size: curve(args.work, size) for size in args.runs}
    outcomes.report(args.output, out)
    return {name: {s: (v['raw_correct'], v['submitted'], v['failures']) for s, v in arm['sets'].items()} for name, arm in out['arms'].items()}


def markdown(args):
    d, lines = files.read_json(args.report), []
    names = {'provisional_check': 'Provisional check (200 rows, 62 families)', 'cohort_v2_routing': 'Cohort v2 routing_v1 (82 rows, 11 families)'}
    for set_name, title in names.items():
        lines += [f'**{title}**', '', '| Arm | Raw correct / submitted | Failed | vs base fixes / regressions | Families all correct | Families regressed vs base | Contrast pairs both correct | en | pt | p50 / p95 ms |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for name, arm in d['arms'].items():
            x = arm['sets'][set_name]
            vb = x.get('vs_base')
            lines.append(f"| {name} | {x['raw_correct']} / {x['submitted']} | {x['failures']} | {str(vb['fixes']) + ' / ' + str(vb['regressions']) if vb else '—'} | {x['families_all_correct']} / {x['families']} | "
                         f"{len(vb['family_regressions']) if vb else '—'} | {x['contrast_pairs']['both_correct']} / {x['contrast_pairs']['pairs']} | "
                         f"{x['languages']['en']['raw_correct']} / {x['languages']['en']['submitted']} | {x['languages']['pt']['raw_correct']} / {x['languages']['pt']['submitted']} | {x['latency']['p50']:.0f} / {x['latency']['p95']:.0f} |")
        lines += ['', 'Per class, correct / gold (predicted):', '', '| Arm | ' + ' | '.join(next(iter(d['arms'].values()))['sets'][set_name]['per_class']) + ' | Classes below base (families newly wrong) |', '|---|' + '---|' * 6]
        for name, arm in d['arms'].items():
            x = arm['sets'][set_name]
            regress = '; '.join(f"{k} {v['base']}→{v['arm']}" for k, v in x['vs_base']['class_regressions'].items()) if 'vs_base' in x else '—'
            newly = ', '.join(f"{k} {len(v)}" for k, v in x['vs_base']['families_with_more_wrong_by_class'].items()) if 'vs_base' in x else ''
            lines.append(f"| {name} | " + ' | '.join(f"{c['correct']} / {c['gold']} ({c['predicted']})" for c in x['per_class'].values()) + f" | {regress or 'none'}{f' ({newly})' if newly else ''} |")
        lines.append('')
    lines += ['**Per fit size**', '', '| Size | Fit families / obs | Selected | model_select raw (base / e1 / e2 / e3 of obs) | Fit wall s | Step s (e1/e2/e3) | Peak device GB | Peak host GB | Fit obs all-correct before → after |',
              '|---|---|---|---|---|---|---|---|---|']
    for size, c in d['curve'].items():
        ms = c.get('model_select')
        lines.append(f"| {size} | {c['roles']['fit']['families']} / {c['roles']['fit']['observations']} | {c.get('selected', 'pending')} | "
                     f"{' / '.join(str(t['raw_correct']) for t in ms['candidates']) + ' of ' + str(ms['observations']) if ms else 'pending'} | {c['fit_wall_seconds']:.0f} | "
                     f"{' / '.join(str(e['step_seconds']) for e in c['epochs'])} | {c['peak_device_bytes'] / 1e9:.2f} | {c['peak_host_bytes'] / 1e9:.1f} | "
                     f"{c['fit_all_presentations_correct']['before']} → {c['fit_all_presentations_correct']['after']} of {c['fit_all_presentations_correct']['of']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(prog='bench.routing_specialist')
    parser.add_argument('--work', type=Path, required=True, help='work root: labels/, exports/, scores/ and runs/ live under it')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('sets', help='provisional check (teachers and author agree) and cohort v2 routing_v1 items as check exports, with overlap verification')
    p.add_argument('--fit', type=Path, required=True, help='largest native tune export manifest (fit + model_select + acceptance_fit families)')
    p = commands.add_parser('score', help='checkpointed per-row scoring of one arm on both evaluation sets')
    p.add_argument('--arm', required=True)
    p.add_argument('--judge', required=True, help='local GGUF path or openrouter:MODEL')
    p.add_argument('--provider')
    p.add_argument('--key-file', type=Path)
    p.add_argument('--batch-inputs', type=int)
    p.add_argument('--native', type=Path, help='resolved job whose native execution settings local arms share (required for local arms)')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--fresh', action='store_true')
    p = commands.add_parser('report', help='per-arm summaries; the first arm is the base for regressions')
    p.add_argument('--arms', nargs='+', required=True)
    p.add_argument('--runs', nargs='*', default=[], help='fitted run names under the work directory, e.g. fit-100')
    outcomes.output_argument(p)
    p = commands.add_parser('markdown', help='README tables from a published report')
    p.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.command == 'markdown':
        print(markdown(args))
        return
    try:
        result = {'sets': build, 'score': score, 'report': report}[args.command](args)
    except ConfigError as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps(result, default=str)[:4000])


if __name__ == '__main__':
    main()
