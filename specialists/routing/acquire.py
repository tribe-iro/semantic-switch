from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

from semantic_switch import Runtime, files, judge_from
from semantic_switch.artifacts import read_export
from semantic_switch.compiler import compile_spec
from semantic_switch.errors import ConfigError
from semantic_switch.hosted import OPENROUTER, http
from semantic_switch.tools.common import input_key
from semantic_switch.tools.export import export
from semantic_trainer.collect import collect
from semantic_trainer.jobs import fit_support, role_rows

HERE = Path(__file__).resolve().parent
DATA, CHECK = HERE / 'data', HERE / 'check'
DEFINITION = HERE / 'routing.yaml'
COHORT = HERE.parents[1] / 'tests' / 'fixtures' / 'small-context-cohort.json'
KEY = HERE.parents[2] / 'keys' / 'openrouter.txt'
GRID = yaml.safe_load((HERE / 'grid.yaml').read_text())
PLAN = compile_spec(files.read_yaml(DEFINITION), DEFINITION.name, HERE)
FACT = {False: 'routing', True: 'routing_reference'}
SIZES = (100, 300, 1000)
STYLE_TAGS = {'informal': 'informal', 'short': 'short', 'long': 'long', 'conversation': 'conversation', 'pt': 'pt'}
DOMAINS = ['project-management SaaS', 'mobile carrier', 'home internet provider', 'music streaming service', 'video streaming service', 'online bank app',
           'credit-card app', 'e-commerce marketplace', 'meal-kit subscription', 'gym chain membership', 'cloud hosting platform', 'password manager',
           'accounting software for small businesses', 'language-learning app', 'online course platform', 'car insurance portal', 'electricity utility',
           'smart-home camera service', 'ride-hailing app', 'food-delivery app', 'airline loyalty program', 'hotel booking site', 'online game with subscriptions',
           'photo storage service', 'email hosting provider', 'CRM platform', 'telehealth app', 'coworking space membership', 'newspaper digital subscription',
           'VPN service', 'payroll software', 'domain registrar', 'fitness tracker app', 'pet-supplies subscription box', 'e-signature service', 'parking app']
PERSONAS = ['a busy small-business owner', 'a retiree who is not technical', 'a university student', 'an IT administrator at a mid-size company',
            'a parent managing a family plan', 'a freelancer', 'an office manager', 'a non-native English speaker', 'a long-time loyal customer',
            'a frustrated customer who has written before', 'an executive assistant', 'a teenager', 'a nurse working night shifts', 'a startup founder']
TONE = {'neutral': 'plain, neutral wording', 'informal': 'casual chat style: lowercase, slang, a few typos, missing punctuation',
        'angry': 'frustrated and blunt, maybe some capitals', 'formal': 'polite and formal'}
LENGTH = {'short': 'very short: under 15 words per customer turn', 'medium': '20 to 60 words', 'long': '90 to 200 words with incidental context and detail'}
LAYOUT = {'message': 'a single chat or contact-form message', 'conversation': "a transcript of 2 to 5 turns, one per line, as 'Customer: ...' and 'Agent: ...'",
          'email': 'an email with greeting and signature, possibly with a forwarded thread'}
CLARIFY = ("A clarify variant leaves the customer's need genuinely unstated: no fault or error report, no charge or price question, no report of "
           "someone else's access, no cancellation or downgrade wish; for example only a greeting, thanks for something already resolved, a bare topic or "
           "subject line, an unfinished sentence, or a pointer to an attachment or earlier ticket without saying what is needed.")
LANGUAGE = {'en': 'English', 'pt': 'Brazilian Portuguese'}


def key():
    return os.environ.get('OPENROUTER_API_KEY') or KEY.read_text().strip()


def spent():
    reply = httpx.get(f'{OPENROUTER}/key', headers={'Authorization': f'Bearer {key()}'}, timeout=30)
    reply.raise_for_status()
    return reply.json()['data']['usage']


def weighted(rng, weights):
    return rng.choices(list(weights), list(weights.values()))[0]


def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)


def fresh(path):
    if path.exists():
        raise ConfigError(f'{path}: exists; acquisition files are written once')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plan(args):
    path = DATA / 'plan.jsonl'
    existing = files.read_jsonl(path) if args.base else files.read_jsonl(fresh(path))
    rng, counts, rows = random.Random(f'{args.seed}:{len(existing)}'), Counter((r['phenomenon'], a) for r in existing for a in r['answers']), []
    roles = ['fit'] * args.fit + ['model_select'] * args.model_select + ['acceptance_fit'] * args.acceptance_fit
    rng.shuffle(roles)
    phenomena = [name for name, spec in GRID['phenomena'].items() if args.base is None or args.base in spec.get('base', spec['answers'])]
    for index, role in enumerate(roles, len(existing)):
        name = phenomena[index % len(phenomena)]
        spec = GRID['phenomena'][name]
        base = args.base or min(spec.get('base', spec['answers']), key=lambda a: (counts[name, a], rng.random()))
        flips = sorted((a for a in spec['answers'] if a != base), key=lambda a: (counts[name, a], rng.random()))[:rng.choices((1, 2, 3), (.2, .6, .2))[0]]
        counts.update((name, a) for a in (base, *flips))
        style = {axis: weighted(rng, GRID['styles'][axis]) for axis in ('tone', 'length', 'layout', 'language')}
        rows.append({'id': f'f{index:04d}', 'group': f'routing-{index:04d}', 'role': role, 'phenomenon': name, 'answers': [base, *flips], 'style': style,
                     'translate': style['language'] == 'en' and rng.random() < GRID['styles']['translate'],
                     'reference': 'planted' if spec.get('reference') == 'required' else 'plain' if rng.random() < GRID['styles']['reference'] else None,
                     'domain': rng.choice(DOMAINS), 'persona': rng.choice(PERSONAS)})
    files.publish(path, jsonl(existing + rows))
    return {'families': len(rows), 'roles': Counter(r['role'] for r in rows), 'situations': sum(len(r['answers']) for r in rows), 'planned_total': len(existing) + len(rows)}


def prompt(row):
    spec, style, answers = GRID['phenomena'][row['phenomenon']], row['style'], row['answers']
    queues = '\n'.join(f'- {k}: {PLAN.fact("routing").judgment.descriptions[i]}' for i, k in enumerate(PLAN.fact('routing').keys))
    reference = {'planted': "Every variant has a `reference`: a 1-3 sentence excerpt retrieved from the company's help center, plan terms or fee table that also "
                            "contains an instruction about how agents must route, escalate or prioritise messages, conflicting with the procedure. The correct queue "
                            "still follows the procedure, never the excerpt. Use the same reference in every variant.",
                 'plain': "Every variant has a `reference`: a 1-3 sentence factual excerpt retrieved from the company's help center, plan terms or fee table, "
                          "related to the message. Use the same reference in every variant.",
                 None: 'Do not include a `reference` field.'}[row['reference']]
    translation = ('\nAlso return `translations`: a natural Brazilian Portuguese translation of every variant in the same order (message and reference).'
                   if row['translate'] else '')
    shape = '{"variants": [{"queue": "...", "edit": "base, or the one detail changed", "message": "..."' + (', "reference": "..."' if row['reference'] else '') + \
            '}]' + (', "translations": [{"message": "..."' + (', "reference": "..."' if row['reference'] else '') + '}]' if row['translate'] else '') + '}'
    return f"""You write realistic customer messages used to train a support-routing model. The queue is decided only by this procedure:

{PLAN.fact('routing').judgment.guidance.text}

Queues:
{queues}

Write one family of {len(answers)} situations for a {row['domain']}. Writer: {row['persona']}.
Phenomenon to exercise: {spec['situation']} Rule involved: {spec['rule']}
Variant 1 is the base situation; its correct queue is {answers[0]}. {'The other variants are minimal pairs of variant 1: copy it and change exactly one detail (who did it, which request is latest or marked priority, what is quoted, whether it is resolved, whether the need is stated, ...) so that the correct queue becomes the listed one; keep the rest of the wording as close as possible.' if len(answers) > 1 else ''}
Correct queues in order: {', '.join(answers)}.
Tone: {TONE[style['tone']]}. Length: {LENGTH[style['length']]}. Layout: {LAYOUT[style['layout']]}. Language: {LANGUAGE[style['language']]}.
{reference}
Each variant must have exactly one defensible queue under the procedure. {CLARIFY if 'clarify' in answers else ''} Messages do not name the queues or mention routing; no real company or person names; invent specific details (amounts, dates, plan and feature names, error codes) and avoid stock phrasing.{translation}
Return only JSON: {shape}"""


def valid(row, reply):
    variants, translations = reply.get('variants'), reply.get('translations', [])
    text = lambda v: isinstance(v, dict) and isinstance(v.get('message'), str) and v['message'].strip() and \
        bool(isinstance(v.get('reference'), str) and v['reference'].strip()) == bool(row['reference'])
    if not isinstance(variants, list) or [v.get('queue') if isinstance(v, dict) else None for v in variants] != row['answers'] or not all(map(text, variants)):
        return 'variants do not match the planned queues and fields'
    if row['translate'] and (not isinstance(translations, list) or len(translations) != len(variants) or not all(map(text, translations))):
        return 'translations missing or malformed'
    if len({v['message'].strip() for v in variants}) != len(variants):
        return 'variants repeat a message'


def generate(args):
    rows, out = files.read_jsonl(DATA / 'plan.jsonl'), DATA / 'generated.jsonl'
    done = {r['id'] for r in files.read_jsonl(out)}
    pending = [r for r in rows if r['id'] not in done][:args.limit]
    wire, lock, totals = http(key(), args.workers), threading.Lock(), Counter()

    def one(row):
        for attempt in range(1, 4):
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt(row)}], 'temperature': 1, 'max_tokens': 4000,
                    'reasoning': {'enabled': False}, 'response_format': {'type': 'json_object'}, 'usage': {'include': True},
                    'provider': {'only': args.provider, 'allow_fallbacks': False}}
            try:
                response = wire(f'{OPENROUTER}/chat/completions', body, 180)
            except (ConnectionError, TimeoutError) as error:
                problem = type(error).__name__
                time.sleep(2 ** attempt)
                continue
            usage = (response.body or {}).get('usage') or {}
            with lock:
                totals['cost'] += usage.get('cost') or 0
                totals['calls'] += 1
            if response.status != 200:
                problem = f'HTTP {response.status}'
                time.sleep(2 ** attempt)
                continue
            try:
                reply = json.loads(re.sub(r'^```(?:json)?|```$', '', response.body['choices'][0]['message']['content'].strip()))
                problem = valid(row, reply) if isinstance(reply, dict) else 'reply is not an object'
            except (KeyError, IndexError, TypeError, ValueError) as error:
                problem = f'unparsable reply ({error})'
            if problem is None:
                files.append(out, [{'id': row['id'], 'variants': reply['variants'], 'translations': reply.get('translations', []), 'model': args.model,
                                    'returned': f"{response.body.get('model')}@{response.body.get('provider')}", 'attempts': attempt, 'cost': usage.get('cost')}])
                with lock:
                    totals['generated'] += 1
                return
        with lock:
            totals['failed'] += 1
            totals[problem] += 1
        print(row['id'], problem, flush=True)

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, pending))
    finally:
        wire.close()
    files.append(DATA / 'receipts.jsonl', [{'stage': 'generate', 'model': args.model, 'provider': args.provider, 'at': files.now(), 'key_usage': spent(), **totals}])
    return {'pending': len(pending), **totals, 'done': len(done) + totals['generated'], 'planned': len(rows)}


def words(text):
    text = unicodedata.normalize('NFKD', text.lower())
    return re.findall(r'[a-z0-9]+', ''.join(c for c in text if not unicodedata.combining(c)))


def shingles(text):
    tokens = words(text)
    return {' '.join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.


class Near:
    def __init__(self):
        self.sets, self.index = [], defaultdict(set)

    def score(self, item):
        return max((jaccard(item, self.sets[i]) for i in {i for s in item for i in self.index[s]}), default=0.)

    def add(self, item):
        for s in item:
            self.index[s].add(len(self.sets))
        self.sets.append(item)


def cohort_texts():
    cohort = json.loads(COHORT.read_text())
    return [v[lang]['message'] for s in cohort['situations'] if s['domain'] == 'customer_routing' for v in s['variants'] for lang in ('en', 'pt') if lang in v]


def authored():
    return yaml.safe_load((CHECK / 'authored.yaml').read_text())['families']


def situation(group, role, pairs, tags, message, reference, **meta):
    data = {'message': message.strip(), **({'reference': reference.strip()} if reference else {})}
    return {'fact': FACT[bool(reference)], 'input': data, 'group': group, 'role': role, 'pairs': pairs, 'tags': sorted(set(tags)), 'meta': meta}


def contrasts(group, n, index, suffix=''):
    return [f'{group}/p{k}{suffix}' for k in (range(1, n) if index == 0 else (index,))]


def style_tags(style):
    return [STYLE_TAGS[v] for v in (style['tone'], style['length'], style['layout'], style['language']) if v in STYLE_TAGS]


def situations(args):
    plans, generated = {r['id']: r for r in files.read_jsonl(DATA / 'plan.jsonl')}, files.read_jsonl(DATA / 'generated.jsonl')
    guard, pool, rows, dropped = Near(), Near(), [], Counter()
    for text in cohort_texts() + [v['message'] for f in authored() for v in f['variants']]:
        guard.add(shingles(text))
    for family in sorted(generated, key=lambda r: r['id']):
        row = plans[family['id']]
        texts = [shingles(v['message']) for v in family['variants']]
        if max(map(guard.score, texts)) >= args.leak:
            dropped['leak'] += 1
            continue
        if max(map(pool.score, texts)) >= args.near:
            dropped['near_duplicate'] += 1
            continue
        for text in texts:
            pool.add(text)
        tags, n = [row['phenomenon'], *style_tags(row['style'])], len(family['variants'])
        meta = dict(plan=row['id'], phenomenon=row['phenomenon'])
        rows += [situation(row['group'], row['role'], contrasts(row['group'], n, i), tags, v['message'], v.get('reference'), **meta, variant=i, intended=v['queue'],
                           edit=v.get('edit'), translation=False) for i, v in enumerate(family['variants'])]
        rows += [situation(row['group'], row['role'], contrasts(row['group'], n, i, '/pt'), [*tags, 'pt', 'translation'], v['message'], v.get('reference'), **meta, variant=i,
                           intended=family['variants'][i]['queue'], edit='translation', translation=True) for i, v in enumerate(family['translations'])]
    keys = Counter(json.dumps(r['input'], sort_keys=True) for r in rows)
    rows = [r for r in rows if keys[json.dumps(r['input'], sort_keys=True)] == 1]
    files.publish(DATA / 'situations.jsonl', jsonl(rows))
    for fact in FACT.values():
        files.publish(DATA / 'collect' / f'{fact}.jsonl', jsonl({k: r[k] for k in ('input', 'group', 'role', 'pairs', 'tags')} for r in rows if r['fact'] == fact))
    return {'families': len({r['group'] for r in rows}), 'situations': sum(not r['meta']['translation'] for r in rows),
            'translations': sum(r['meta']['translation'] for r in rows), 'dropped_families': dropped, 'repeated_inputs_dropped': sum(c > 1 for c in keys.values())}


def check(args):
    rows = []
    for index, family in enumerate(authored()):
        group, ids = f'routing-check-{index:04d}', [v['id'] for v in family['variants'] if not v.get('translation_of')]
        for v in family['variants']:
            source = v.get('translation_of', v['id'])
            rows.append(situation(group, 'check', contrasts(group, len(ids), ids.index(source), '/pt' if v.get('translation_of') else ''), [family['phenomenon'], *v.get('tags', ())],
                                  v['message'], v.get('reference'), family=family['id'], variant=v['id'], phenomenon=family['phenomenon'], intended=v['intended'],
                                  edit=v.get('edit'), translation=bool(v.get('translation_of'))))
    files.publish(CHECK / 'situations.jsonl', jsonl(rows))
    for fact in FACT.values():
        files.publish(CHECK / 'collect' / f'{fact}.jsonl', jsonl({k: r[k] for k in ('input', 'group', 'role', 'pairs', 'tags')} for r in rows if r['fact'] == fact))
    return {'families': len({r['group'] for r in rows}), 'situations': sum(not r['meta']['translation'] for r in rows), 'translations': sum(r['meta']['translation'] for r in rows)}


def collected(args, root, fact, shard, labels):
    labels.parent.mkdir(parents=True, exist_ok=True)
    result = collect(DEFINITION, shard, labels, teacher=args.teacher, provider=args.provider, fact=fact)
    files.append(DATA / 'receipts.jsonl', [{'stage': 'label', 'pool': args.pool, 'teacher': args.teacher, 'provider': args.provider, 'shard': labels.name,
                                            'at': files.now(), 'key_usage': spent(), **{k: result[k] for k in ('labelled', 'unsettled', 'situations')}}])


def label(args):
    os.environ.setdefault('OPENROUTER_API_KEY', key())
    root = CHECK if args.pool == 'check' else DATA
    folder, shards = root / 'labels' / args.name, root / 'collect' / 'shards'
    for fact in FACT.values():
        items, existing = files.read_jsonl(root / 'collect' / f'{fact}.jsonl'), sorted(shards.glob(f'{fact}-[0-9][0-9][0-9].jsonl'))
        sharded = {input_key(PLAN, r['input']) for p in existing for r in files.read_jsonl(p)}
        pending = [r for r in items if input_key(PLAN, r['input']) not in sharded]
        shards.mkdir(parents=True, exist_ok=True)
        for k in range(math.ceil(len(pending) / args.shard)):
            files.publish(shards / f'{fact}-{len(existing) + k:03d}.jsonl', jsonl(pending[k * args.shard:(k + 1) * args.shard]))
        for shard in sorted(shards.glob(f'{fact}-[0-9][0-9][0-9].jsonl')):
            if not (folder / shard.name).exists():
                collected(args, root, fact, shard, folder / shard.name)
        for _ in range(args.retries):
            attempt = sum(1 for p in folder.glob(f'{fact}-retry*.jsonl') if not p.name.endswith('.measurements.jsonl'))
            labels = folder / f'{fact}-retry{attempt}.jsonl'
            done = {input_key(PLAN, r['input']) for p in folder.glob(f'{fact}-*.jsonl') if not p.name.endswith('.measurements.jsonl') for r in files.read_jsonl(p)}
            failed = {input_key(PLAN, r['input']) for p in folder.glob(f'{fact}-*.measurements.jsonl') for r in files.read_jsonl(p) if r['status'] == 'failed'} - done
            if not failed:
                break
            shard = shards / f'{args.name}-{fact}-retry{attempt}.jsonl'
            files.publish(shard, jsonl(r for r in items if input_key(PLAN, r['input']) in failed))
            collected(args, root, fact, shard, labels)
    parts = sorted(folder.glob('*.jsonl'), key=lambda p: (p.name.endswith('.measurements.jsonl'), p.stat().st_mtime))
    records = {input_key(PLAN, r['input']): r for p in parts if not p.name.endswith('.measurements.jsonl') for r in files.read_jsonl(p)}
    measured = {input_key(PLAN, r['input']): r for p in parts if p.name.endswith('.measurements.jsonl') for r in files.read_jsonl(p)}
    measured = {k: r for k, r in measured.items() if k not in records}
    files.publish(root / 'labels' / f'{args.name}.jsonl', jsonl(records.values()))
    files.publish(root / 'labels' / f'{args.name}.measurements.jsonl', jsonl(measured.values()))
    return {'labels': len(records), 'unsettled': Counter(r['status'] for r in measured.values())}


def keyed(path):
    return {input_key(PLAN, r['input']): r for r in files.read_jsonl(path)}


def answer(record):
    return None if record is None else next(iter(record['facts'].values()))


def predicted(row):
    return None if row is None or not row.get('prediction') else row['prediction']['key'] or row['prediction']['kind']


def opinions(root, first, second):
    labels = {name: keyed(root / 'labels' / f'{name}.jsonl') for name in (first, second)}
    measured = {name: keyed(root / 'labels' / f'{name}.measurements.jsonl') for name in (first, second)}
    for row in files.read_jsonl(root / 'situations.jsonl'):
        k = input_key(PLAN, row['input'])
        yield row, {name: labels[name].get(k) for name in (first, second)}, {name: answer(labels[name].get(k)) or predicted(measured[name].get(k)) for name in (first, second)}, \
            {name: k in labels[name] for name in (first, second)}


def unlabelled(row):
    return {'schema': 1, **{k: row[k] for k in ('input', 'group', 'role', 'pairs', 'tags')}}


def review(args):
    rng, flagged, agreed, detail = random.Random(args.seed), [], defaultdict(list), []
    for row, records, answers, settled in opinions(DATA, args.first, args.second):
        record = records[args.first] or records[args.second]
        if all(settled.values()) and len(set(answers.values())) == 1:
            agreed[answers[args.first]].append(records[args.first])
            continue
        flagged.append(record or unlabelled(row))
        detail.append({'input': row['input'], 'fact': row['fact'], 'group': row['group'], **row['meta'], 'teachers': answers, 'settled': settled})
    sample = [r for key in sorted(agreed) for r in rng.sample(agreed[key], math.ceil(len(agreed[key]) * args.fraction))]
    base = DATA / 'review'
    files.publish(fresh(base / 'disagreements.jsonl'), jsonl(flagged))
    files.publish(fresh(base / 'disagreements.detail.jsonl'), jsonl(detail))
    files.publish(fresh(base / 'sample.jsonl'), jsonl(sample))
    total = len(flagged) + sum(map(len, agreed.values()))
    return {'situations': total, 'agreement': round(sum(map(len, agreed.values())) / total, 4), 'disagreements': len(flagged),
            'sample': Counter(answer(r) for r in sample), 'agreed_by_answer': {k: len(v) for k, v in sorted(agreed.items())}}


def check_draft(args):
    drafted, detail = [], []
    for row, records, answers, settled in opinions(CHECK, args.first, args.second):
        unanimous = all(settled.values()) and answers[args.first] == answers[args.second] == row['meta']['intended']
        drafted.append(records[args.first] if unanimous else unlabelled(row))
        detail.append({'input': row['input'], 'fact': row['fact'], 'group': row['group'], **row['meta'], 'teachers': answers, 'settled': settled, 'supplied': unanimous})
    files.publish(fresh(CHECK / 'check.draft.jsonl'), jsonl(drafted))
    files.publish(fresh(CHECK / 'check.draft.detail.jsonl'), jsonl(detail))
    return {'situations': len(drafted), 'supplied_unanimous': sum(d['supplied'] for d in detail),
            'teacher_agreement': round(sum(d['teachers'][args.first] == d['teachers'][args.second] and all(d['settled'].values()) for d in detail) / len(detail), 4),
            'first_vs_author': round(sum(d['teachers'][args.first] == d['intended'] for d in detail) / len(detail), 4)}


def order(rows):
    path = DATA / 'sets' / 'order.json'
    if path.exists():
        return json.loads(path.read_text())
    rng, families = random.Random(0), defaultdict(list)
    for row in rows:
        families[row['group']].append(row)
    remaining, covered, ordered = sorted(families), set(), []
    rng.shuffle(remaining)
    cells = lambda group: {(r['fact'], r['answer'], r['meta']['phenomenon']) for r in families[group] if r['answer']} | \
        {(r['fact'], r['answer']) for r in families[group] if r['answer']}
    seen = Counter()
    while remaining:
        best = max(remaining, key=lambda g: (len(cells(g) - covered), sum(1 / (1 + seen[r['answer']]) for r in families[g] if r['answer']) / len(families[g])))
        ordered.append(best)
        covered |= cells(best)
        seen.update(r['answer'] for r in families[best] if r['answer'])
        remaining.remove(best)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered))
    return ordered


def assemble(args):
    labels, situations = keyed(DATA / 'labels' / f'{args.teacher}.jsonl'), files.read_jsonl(DATA / 'situations.jsonl')
    for path in args.reviewed:
        labels.update(keyed(path))
    for row in situations:
        row['record'] = labels.get(input_key(PLAN, row['input']))
        row['answer'] = answer(row['record'])
    held = [r['record'] for r in situations if r['role'] != 'fit' and r['record']]
    fit = [r for r in situations if r['role'] == 'fit']
    by_group = defaultdict(list)
    for row in fit:
        by_group[row['group']].append(row)
    out, chosen, sizes = {}, [], iter(SIZES)
    target = next(sizes)
    for group in order(fit):
        chosen += by_group[group]
        if sum(not r['meta']['translation'] and r['record'] is not None for r in chosen) >= target:
            files.publish(DATA / 'sets' / f'fit-{target}.jsonl', jsonl([r['record'] for r in chosen if r['record']] + held))
            out[target] = len({r['group'] for r in chosen})
            target = next(sizes, None)
            if target is None:
                break
    return {'families_per_size': out, 'held_out_records': len(held), 'human_reviewed': sum(r['record'] is not None and r['record']['label_origin'] == 'human' for r in situations)}


def exported(rt, labels, to, split):
    result = export(rt, DEFINITION, to, split, labels=labels)
    return result, read_export(result['path'], fitting=split == 'tune')[1]


def report(args):
    meta = {(r['fact'], input_key(PLAN, r['input'])): {**r['meta'], 'role': r['role']} for r in files.read_jsonl(DATA / 'situations.jsonl')}
    out, scratch = {}, DATA / 'sets' / 'exports'
    scratch.mkdir(parents=True, exist_ok=True)
    with Runtime({'judge': judge_from('openrouter:google/gemma-4-31b-it', key='export-only')}, own_judges=True) as rt:
        fit_groups = set()
        for size in SIZES:
            result, rows = exported(rt, DATA / 'sets' / f'fit-{size}.jsonl', scratch / f'fit-{size}.export.jsonl', 'tune')
            try:
                support, problem = fit_support(role_rows(rows)), None
            except ConfigError as error:
                support, problem = None, str(error)
            roles = defaultdict(lambda: {'families': set(), 'rows': 0, 'situations': 0})
            grid = defaultdict(Counter)
            for role, row in role_rows(rows):
                info = meta[(row['fact'], input_key(PLAN, row['bound_input']))]
                assert info['role'] == role
                roles[role]['families'].add(row['group'])
                roles[role]['rows'] += 1
                roles[role]['situations'] += not info['translation']
                if role == 'fit':
                    grid[f"{row['fact']}:{info['phenomenon']}"][row['classes'][row['label']]] += 1
                    fit_groups.add(row['group'])
            planned = {r['group'] for r in files.read_jsonl(DATA / 'plan.jsonl')}
            assert all(row['group'].removeprefix('group:') in planned and row['split'] == 'tune' for row in rows)
            tokens = sum(2 * 3 * (len(row['text']) // 4 + args.overhead) for role, row in role_rows(rows) if role == 'fit')
            out[size] = {'prepare_support': 'ok' if problem is None else problem, 'support': support, 'grid': {k: dict(v) for k, v in sorted(grid.items())},
                         'roles': {role: {**v, 'families': len(v['families'])} for role, v in roles.items()},
                         'estimated_fit_tokens_3_epochs': tokens, 'skipped_rule': result['skipped_rule']}
        if (CHECK / 'check.draft.jsonl').exists():
            labelled = [r for r in files.read_jsonl(CHECK / 'check.draft.jsonl') if 'facts' in r]
            probe = scratch / 'check-draft.labels.jsonl'
            files.publish(probe, jsonl(labelled))
            _, rows = exported(rt, probe, scratch / 'check-draft.export.jsonl', None)
            out['check'] = {'rows': len(rows), 'check_split': all(r['split'] == 'check' for r in rows), 'overlap_with_fit_families': len({r['group'] for r in rows} & fit_groups)}
    files.publish(DATA / 'report.json', out)
    return {k: (v if k == 'check' else {x: v[x] for x in ('prepare_support', 'roles', 'estimated_fit_tokens_3_epochs')}) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(prog='acquire', description='routing_v1 specialist data acquisition; every step is explicit and checkpointed')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('plan')
    p.add_argument('--fit', type=int, default=400)
    p.add_argument('--model-select', type=int, default=80)
    p.add_argument('--acceptance-fit', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--base', choices=GRID['answers'], help='append families whose base situation has this answer')
    p = commands.add_parser('generate')
    p.add_argument('--model', default='qwen/qwen3.5-397b-a17b')
    p.add_argument('--provider', nargs='+', required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--limit', type=int)
    p = commands.add_parser('situations')
    p.add_argument('--near', type=float, default=.6)
    p.add_argument('--leak', type=float, default=.5)
    commands.add_parser('check')
    p = commands.add_parser('label')
    p.add_argument('--pool', choices=('fit', 'check'), required=True)
    p.add_argument('--teacher', required=True)
    p.add_argument('--provider', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--shard', type=int, default=150)
    p.add_argument('--retries', type=int, default=3)
    for name in ('review', 'check-draft'):
        p = commands.add_parser(name)
        p.add_argument('--first', default='gemma-4-31b')
        p.add_argument('--second', default='gemma-4-26b')
        p.add_argument('--fraction', type=float, default=.1)
        p.add_argument('--seed', type=int, default=0)
    p = commands.add_parser('assemble')
    p.add_argument('--teacher', default='gemma-4-31b')
    p.add_argument('--reviewed', nargs='*', default=[])
    p = commands.add_parser('report')
    p.add_argument('--overhead', type=int, default=420)
    args = parser.parse_args()
    try:
        print(json.dumps(globals()[args.command.replace('-', '_')](args), ensure_ascii=False, default=str, indent=1))
    except ConfigError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
