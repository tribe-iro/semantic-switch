from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import resource
import shutil
import sys
import time

from semantic_switch import ConfigError
from semantic_switch.training import Journal, class_changes, files, nested, per_class as reduce_classes, probabilities, record_path, regressions

from .jobs import LOSS, development_recall, ranking, unit_correct
from .objective import UNTIMED, bind, evaluate, prepared

RETAINED = 2


class Exhausted(ConfigError):
    def __init__(self, reason):
        super().__init__(f'{reason} reached')
        self.reason = reason


class Infeasible(ConfigError):
    def __init__(self, record):
        super().__init__(f"projected plan exceeds the declared budgets: {json.dumps(record['projection'])}; no further fitting")
        self.record = record


class Deadline:
    def __init__(self, seconds, clock=time.monotonic, consumed=0.):
        self.seconds, self.clock, self.started, self.crossed, self.consumed = seconds, clock, clock(), None, consumed

    def elapsed(self):
        return self.consumed + self.clock() - self.started

    def expired(self):
        return self.elapsed() >= self.seconds

    def check(self, operation):
        if self.expired():
            self.crossed = self.crossed or operation
            raise Exhausted('time_budget')

    def record(self, last):
        elapsed = self.elapsed()
        return {'limit_seconds': self.seconds, 'elapsed_seconds': elapsed, 'overrun_seconds': max(0., elapsed - self.seconds), 'crossed': self.crossed,
                'last_complete_point': last}


@dataclass
class Work:
    updates: int = 0
    points: int = 0
    checkpoints: int = 0
    training_tokens: int = 0
    monitor_tokens: int = 0
    tasks: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=lambda: {'update': [], 'point': [], 'checkpoint': []})

    def totals(self):
        return asdict(self)


def per_class(rows, results):
    def observed(row):
        predicted = results[row['id']]['predicted']
        return ((row['task'], '/'.join(record_path(row))), row['classes'], row['classes'][row['label']],
                None if predicted is None else row['classes'][predicted], results[row['id']]['correct'])
    return nested(reduce_classes(map(observed, rows)))


def probability_entries(rows, results):
    return [{'task': row['task'], 'target': '/'.join(record_path(row)), 'kind': 'membership' if 'member' in row else row['kind'], 'family': row['component'],
             'gold': row['classes'][row['label']], 'outcome': 'predicted', 'observation': results[row['id']]['probability']} for row in rows]



def permutation_summary(rows, results):
    out = {}
    for row in rows:
        target = '/'.join(record_path(row))
        for item in results[row['id']].get('permutations', ()):
            group = out.setdefault(target, {}).setdefault(item['stage'], {'comparable': 0, 'disagreements': 0, 'mean_probability_delta': 0.,
                                                                         'max_probability_delta': 0., 'mean_mass_delta': 0., 'unassessed': {}})
            if not item['comparable']:
                reason = item['reason']
                group['unassessed'][reason] = group['unassessed'].get(reason, 0) + 1
                continue
            group['comparable'] += 1
            group['disagreements'] += item['disagreement']
            group['mean_probability_delta'] += item['mean_probability_delta']
            group['max_probability_delta'] = max(group['max_probability_delta'], item['max_probability_delta'])
            group['mean_mass_delta'] += item['mass_delta']
    for stages in out.values():
        for group in stages.values():
            for key in ('mean_probability_delta', 'mean_mass_delta'):
                group[key] = group[key] / group['comparable'] if group['comparable'] else None
    return out

def peak_host_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Monitor:
    def __init__(self, to, resolved, model, pad, backend, device, deadline):
        self.to, self.resolved, self.pad, self.backend, self.device, self.deadline = to, resolved, pad, backend, device, deadline
        roles = {}
        for row in resolved['observations']:
            roles.setdefault(row['role'], []).append(row)
        self.fit, self.select = roles['fit'], roles['model_select']
        fitting = {row['id']: row for row in self.fit}
        self.probe = [fitting[identifier] for identifier in resolved['probe_ids']]
        self.binding = bind(model)
        self.inputs = prepared([*self.fit, *self.select], self.binding.vocabulary)
        self.window = resolved['recipe']['max_padded_tokens']
        self.unit = resolved['recipe']['selection_unit']
        self.point_tokens = sum(self.inputs[row].tokens for row in [*self.select, *self.probe])
        self.tasks = {row['id']: row['task'] for row in self.select}
        self.points, self.updates, self.losses, self.flushed, self.work = [], [], [], 0, Work()
        self.committed_updates = {row['step'] for row in files.read_jsonl(self.to / 'updates.jsonl')} if (self.to / 'updates.jsonl').exists() else set()
        self.committed_points = {row['step'] for row in files.read_jsonl(self.to / 'progress.jsonl') if row['kind'] == 'evaluation'} if (self.to / 'progress.jsonl').exists() else set()
        self.phases, self.feasibility = UNTIMED, None
        self.seen = {'situations': set(), 'targets': set(), 'presentations': set(), 'families': set()}

    def consumed(self, batch, update, seconds):
        for row in batch:
            self.seen['situations'].update(row['situations'])
            self.seen['targets'].add(row['id'])
            self.seen['families'].add(row['component'])
            self.seen['presentations'].update(entry.id for entry in row['presentations'])
            consumed = self.work.tasks.setdefault(row['task'], {'observations': 0, 'tokens': 0})
            consumed['observations'] += 1
            consumed['tokens'] += self.inputs[row].tokens
        self.work.updates += 1
        self.work.training_tokens += update['tokens']
        self.work.seconds['update'].append(seconds)
        self.updates.append(update)
        self.losses.append(update['loss'])

    def checkpointed(self, seconds):
        self.work.checkpoints += 1
        self.work.seconds['checkpoint'].append(seconds)

    def flush(self):
        pending = [row for row in self.updates[self.flushed:] if row['step'] not in self.committed_updates]
        files.append(self.to / 'updates.jsonl', files.jsonl(pending))
        self.committed_updates.update(row['step'] for row in pending)
        self.flushed = len(self.updates)

    def record(self, record):
        if record['kind'] == 'evaluation' and record['step'] in self.committed_points:
            return
        files.append(self.to / 'progress.jsonl', files.jsonl([record]))
        if record['kind'] == 'evaluation':
            self.committed_points.add(record['step'])
        print(json.dumps(record), file=sys.stderr, flush=True)

    def charge(self, tokens):
        self.work.monitor_tokens += tokens
        files.publish(self.to / 'elapsed.json', {'elapsed': self.deadline.elapsed()})

    def poll(self):
        self.deadline.check('evaluation_row')

    def token_work(self):
        return {'training': self.work.training_tokens, 'monitoring': self.work.monitor_tokens}

    def point(self, step, examples):
        if self.resolved['budget']['max_monitor_tokens'] is not None and self.work.monitor_tokens + self.point_tokens > self.resolved['budget']['max_monitor_tokens']:
            raise Exhausted('monitor_budget')
        before, began = self.work.monitor_tokens, self.deadline.elapsed()
        telemetry = {}
        try:
            def evaluated(role, rows):
                config = {'fit': self.resolved['fit']['id'], 'monitor': self.resolved.get('monitor', {}).get('id'), 'step': step,
                          'role': role, 'rows': [row['id'] for row in rows], 'window': self.window,
                          'batch': self.resolved['recipe']['evaluation_batch_size']}
                with Journal(self.to / f'evaluation-{step}-{role}.jsonl', config) as journal:
                    return evaluate(self.binding, rows, self.inputs, self.pad, self.window, self.poll, self.charge, self.phases,
                                    presentations=self.resolved['recipe']['evaluation_batch_size'], telemetry=telemetry, journal=journal)
            with self.phases('monitor:model_select'):
                selected = evaluated('model_select', self.select)
            with self.phases('monitor:probe'):
                probed = evaluated('probe', self.probe) if self.probe else {}
        except BaseException as error:
            try:
                self.flush()
                self.record({'kind': 'interrupted_evaluation', 'step': step, 'examples_seen': examples, 'reserved_tokens': self.point_tokens,
                         'charged_tokens': self.work.monitor_tokens - before, 'error': f'{type(error).__name__}: {error}', 'published': False,
                         'token_work': self.token_work()})
            except Exception:
                pass
            raise
        seconds, late = self.deadline.elapsed() - began, self.deadline.expired()
        self.work.points += 1
        self.work.seconds['point'].append(seconds)
        record = {'kind': 'evaluation', 'loss': LOSS, 'step': step, 'examples_seen': examples, 'epoch': examples / len(self.fit),
                  'seen': {key: len(value) for key, value in self.seen.items()},
                  'train_loss': sum(self.losses) / len(self.losses) if self.losses else None,
                  'model_select': {'nll': sum(self.inputs[row].objective * selected[row['id']]['nll'] for row in self.select) / len(self.select),
                                   'correct': unit_correct({i: r['correct'] for i, r in selected.items()}, self.resolved['situations'], 'model_select', self.tasks),
                                   'per_class': per_class(self.select, selected), 'population': 'selection proxy; not independent quality',
                                   'probabilities': probabilities(probability_entries(self.select, selected), targets=False)},
                  'probe': {'enabled': bool(self.probe), 'correct': sum(r['correct'] for r in probed.values()), 'submitted': len(probed),
                            'per_class': per_class(self.probe, probed), 'probabilities': probabilities(probability_entries(self.probe, probed), targets=False) if self.probe else None,
                            'population': 'in-sample fit probe; never independent quality'},
                  'seconds': self.deadline.elapsed(), 'point_seconds': seconds, 'execution_work': telemetry, 'permutations': permutation_summary(self.select, selected), 'completed_after_deadline': late, 'peak_host_bytes': peak_host_bytes(),
                  'peak_device_bytes': self.backend.max_memory_allocated(self.device) if self.backend else None,
                  'peak_reserved_device_bytes': self.backend.max_memory_reserved(self.device) if self.backend else None,
                  'token_work': {**self.token_work(), 'point': self.work.monitor_tokens - before}}
        flat = {f'{task}:{target}': counts for task, targets in record['model_select']['per_class'].items() for target, counts in targets.items()}
        record['model_select']['class_changes'] = class_changes(flat, self.points)
        point = {'name': 'zero' if step == 0 else f'step-{step}', 'step': step, 'examples_seen': examples, 'correct': record['model_select']['correct'],
                 'nll': record['model_select']['nll'], 'completed_after_deadline': late, 'per_class': flat,
                 'regressions': [f'{scope}:{name}' for scope, name in regressions(flat, self.points[0]['per_class'])] if self.points else [],
                 'predictions': {i: {'answer': r['predicted'], 'stages': r['stages']} for i, r in selected.items()}}
        point['qualification'] = record['model_select']['qualification'] = development_recall(self.select, selected, self.resolved['recipe'].get('development_recall', []))
        self.points.append(point)
        ranked = ranking([p for p in self.points if p['step']], self.unit)
        record['best_loss_step'] = min(self.points, key=lambda p: (p['nll'], p['step']))['step']
        record['best_correct_step'] = ranked[0]['step'] if ranked else None
        with self.phases('progress_io'):
            self.flush()
            self.record(record)
        self.losses = []
        return point


def save(model, directory):
    model.save_pretrained(directory, safe_serialization=True)


def checkpoint(model, directory, monitor):
    began = monitor.deadline.elapsed()
    from .adapter import CONFIG, WEIGHTS
    from .recovery import latest
    source = latest(monitor.to)
    metadata = files.read_json(source / 'checkpoint.json') if source is not None else None
    if metadata is not None and monitor.points and metadata['step'] == monitor.points[-1]['step']:
        directory.mkdir(parents=True)
        for name in (CONFIG, WEIGHTS):
            os.link(source / name, directory / name)
    else:
        save(model, directory)
    monitor.checkpointed(monitor.deadline.elapsed() - began)


def retain(to, model, unit, retained, point, monitor):
    ranked = ranking([*retained, point], unit)[:RETAINED]
    if point not in ranked:
        return retained
    candidates = to / 'candidates'
    staged = candidates / f".{point['name']}.staging"
    if not (candidates / point['name']).exists():
        if staged.exists():
            shutil.rmtree(staged)
        checkpoint(model, staged, monitor)
        files.publish_directory(staged, candidates / point['name'])
    files.publish(to / 'retained.json', {'unit': unit, 'retained': [p['name'] for p in ranked]})
    return ranked


def stopped(reason, monitor, retained, step, examples, total, late, control):
    deadline = monitor.deadline
    if reason in ('epoch_cap', 'score_plateau') and deadline.expired():
        reason, deadline.crossed = 'time_budget', deadline.crossed or 'final_boundary'
    monitor.flush()
    if not retained:
        raise ConfigError(f'{reason}: no fully evaluated optimizer step to select; no candidate published')
    last, best = monitor.points[-1], retained[0]
    boundary = best['step'] == last['step']
    return {'reason': reason, 'convergence': 'score_plateau' if reason == 'score_plateau' and not boundary else 'not_demonstrated', 'control': control,
            'steps_trained': step, 'examples_trained': examples, 'max_optimizer_steps': total, 'evaluation_points': len(monitor.points),
            'last_evaluation': {'step': last['step'], 'examples_seen': last['examples_seen']},
            'unevaluated': {'steps': step - last['step'], 'examples': examples - last['examples_seen']},
            'best_monitored': {'name': best['name'], 'step': best['step'], 'examples_seen': best['examples_seen']}, 'best_at_boundary': boundary,
            'deadline': {**deadline.record(last['name']), 'completed_after_deadline': late} if reason == 'time_budget' else None,
            'work': monitor.work.totals(), 'inputs': monitor.inputs.record()}


def feasibility(resolved, work, elapsed):
    plan, budget, planned = resolved['schedule'], resolved['budget'], resolved['work']
    measured = {f'{stage}_seconds': sum(values) / len(values) if values else 0. for stage, values in work.seconds.items()}
    remaining = {'updates': plan['max_optimizer_steps'] - work.updates, 'points': planned['evaluation_points'] - work.points}
    remaining['checkpoints'] = remaining['points']
    projected = {'training_seconds': remaining['updates'] * measured['update_seconds'], 'monitoring_seconds': remaining['points'] * measured['point_seconds'],
                 'checkpoint_seconds': remaining['checkpoints'] * measured['checkpoint_seconds']}
    seconds = elapsed + sum(projected.values())
    projection = {'seconds': seconds, **projected, 'max_seconds': budget['max_seconds'], 'training_tokens': planned['training_tokens'], 'max_tokens': budget['max_tokens'],
                  'monitor_tokens': planned['monitor_tokens'], 'max_monitor_tokens': budget['max_monitor_tokens'],
                  'evaluation_points': planned['evaluation_points'], 'optimizer_steps': plan['max_optimizer_steps']}
    return {'kind': 'feasibility', 'feasible': seconds <= budget['max_seconds'] and (budget['max_tokens'] is None or planned['training_tokens'] <= budget['max_tokens'])
            and (budget['max_monitor_tokens'] is None or planned['monitor_tokens'] <= budget['max_monitor_tokens']),
            'elapsed_seconds': elapsed, 'completed': {key: value for key, value in work.totals().items() if key != 'seconds'}, 'remaining': remaining,
            'measured': {**measured, 'point_tokens': planned['point_tokens'], 'updates_measured': len(work.seconds['update']),
                         'points_measured': len(work.seconds['point']), 'checkpoints_measured': len(work.seconds['checkpoint'])},
            'projection': projection}
