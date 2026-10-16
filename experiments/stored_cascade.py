from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from semantic_switch import ConfigError
from semantic_switch.training import arm_summary, artifact_digest, compare_deployments, files

SCHEMA = 1
ARMS = {'teacher': 'baseline', 'student': 'candidate', 'fallback': 'fallback'}
FAILURES = ('fail', 'defer')
SCOPE = ('explicitly invoked experiment over stored whole-call rows of complete frozen student and teacher switches with identical task semantics; '
         'it is evidence for or against a future two-model serving mode, not a public cascade API or scheduler')


def _total(*values):
    return None if any(value is None for value in values) else sum(values)


def _keyed(rows, name) -> dict:
    keyed = {row['input']: row for row in rows}
    if len(keyed) != len(rows):
        raise ConfigError(f'{name} repeats an input; each arm holds one whole-call row per input')
    return keyed


def cascade(student, teacher, *, failed='fail', forbidden=None, fallback=None) -> dict:
    if failed not in FAILURES:
        raise ConfigError(f"declare the failed-student policy: one of {', '.join(FAILURES)}")
    first, second = _keyed(student, 'student'), _keyed(teacher, 'teacher')
    if set(first) != set(second) or any(first[key]['targets'].keys() != second[key]['targets'].keys() for key in first):
        raise ConfigError('student and teacher rows must cover the same inputs and targets; compare complete switches with identical task semantics')
    rows = []
    for key in sorted(first):
        settled, reference = first[key], second[key]
        if settled['status'] == 'uncertain' or settled['status'] == 'failed' and failed == 'defer':
            rows.append({**reference, 'stage': 'teacher', 'deferred_from': settled['status'], 'elapsed_ms': _total(settled['elapsed_ms'], reference['elapsed_ms']),
                         'provider_cost': _total(settled['provider_cost'], reference['provider_cost']), 'requests': settled['requests'] + reference['requests']})
        else:
            rows.append({**settled, 'stage': 'student'})
    forbidden = dict(forbidden or {})
    arms = {'student': student, 'teacher': teacher, 'cascade': rows, **({'fallback': fallback} if fallback is not None else {})}
    return {'policy': {'defer': ['uncertain'], 'failed': failed, 'kept': ['ok', 'none', 'needs_input'],
                       'rule': 'only a declared uncertain whole call defers; none is a valid answer, needs_input cannot be completed by another model, '
                               'and each call keeps one source: scores and partially settled facts are never averaged or merged'},
            'deferred': sum(row['stage'] == 'teacher' for row in rows), 'deferral_rate': sum(row['stage'] == 'teacher' for row in rows) / len(rows) if rows else None,
            'stages': dict(Counter(row['stage'] for row in rows)), 'arms': {name: arm_summary(value, forbidden) for name, value in arms.items()}, 'scope': SCOPE}


def experiment(student, teacher, labels, evidence, to, *, declared, failed='fail', fallback=None, seed=0, count=2000, bindings=None, job=None) -> dict:
    if failed not in FAILURES:
        raise ConfigError(f"declare the failed-student policy before scoring: one of {', '.join(FAILURES)}")
    to = Path(to)
    if (to / 'cascade.json').exists():
        raise ConfigError(f'{to}: a cascade experiment is already published there; run it into a new directory')
    bindings = dict(bindings or {})
    if set(bindings) - set(ARMS):
        raise ConfigError(f"bindings name the arms {', '.join(ARMS)}, not {sorted(set(bindings) - set(ARMS))}")
    report = compare_deployments(teacher, student, labels, evidence, to / 'arms', declaration=declared, seed=seed, count=count,
                                 arms={'fallback': fallback} if fallback is not None else None,
                                 bindings={ARMS[name]: value for name, value in bindings.items()}, job=job)
    rows = report['rows']
    found = cascade(rows['candidate'], rows['baseline'], failed=failed, forbidden=declared['forbidden'], fallback=rows.get('fallback'))
    out = {'schema': SCHEMA, 'kind': 'cascade-experiment', 'comparison': report['id'], 'plan': report['plan']['id'],
           'student': report['arms']['candidate']['entry'], 'teacher': report['arms']['baseline']['entry'],
           **({'fallback': report['arms']['fallback']['entry']} if fallback is not None else {}),
           'exposure': report['exposure'], 'lineage': {'student': report['lineage']['candidate'], 'teacher': report['lineage']['baseline']},
           'reference': report['reference'], 'review': report['review'], 'situations': report['situations'], 'families': report['families'],
           'paired': report['paired'], 'decision': report['decision'], **found,
           'population': report['population']}
    out['id'] = artifact_digest(out)
    files.write_new(to / 'cascade.json', out)
    return out


def main():
    from semantic_switch import judge
    from semantic_switch.training import declaration
    parser = argparse.ArgumentParser(prog='stored_cascade', description='Explicit experiment: matched student/teacher/cascade over complete frozen locks on '
                                                                      'fresh check families; not a serving API')
    parser.add_argument('student', help='student frozen lock (e.g. a fitted lock bundle)')
    parser.add_argument('teacher', help='teacher frozen lock; an ordinary lock binds --teacher-judge')
    parser.add_argument('--labels', required=True)
    parser.add_argument('--evidence', required=True, help='NAME.dataset.json evidence assignment; its audit records exposure before scoring')
    parser.add_argument('--to', required=True)
    parser.add_argument('--teacher-judge', help='judge source bound to an ordinary teacher lock, e.g. openrouter:MODEL')
    parser.add_argument('--provider', help='pinned OpenRouter provider route of --teacher-judge')
    parser.add_argument('--fallback', help='lexical-head frozen lock reported as a further arm')
    parser.add_argument('--failed', required=True, choices=FAILURES, help='declared policy for failed or cancelled student calls')
    parser.add_argument('--claim', default='non_inferiority', choices=('improvement', 'non_inferiority'))
    parser.add_argument('--margin', type=float, required=True, help='student versus teacher margin of the paired claim')
    parser.add_argument('--benefit', choices=('latency', 'cost'), help='operational benefit a non-inferiority claim must show')
    parser.add_argument('--forbidden', action='append', default=[], metavar='ADDRESS=ANSWER_KEY')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    forbidden = {}
    for value in args.forbidden:
        address, separator, key = value.partition('=')
        if not separator or not address or not key:
            parser.exit(2, f'--forbidden expects ADDRESS=ANSWER_KEY, got {value!r}\n')
        forbidden.setdefault(address, []).append(key)
    teacher = judge(args.teacher_judge, **({'provider': args.provider} if args.provider else {})) if args.teacher_judge else None
    try:
        declared = declaration(claim=args.claim, margin=args.margin, benefit=args.benefit, forbidden=forbidden)
        print(json.dumps(experiment(args.student, args.teacher, args.labels, args.evidence, args.to, declared=declared, failed=args.failed,
                                    fallback=args.fallback, seed=args.seed, bindings={'teacher': {'judge': teacher}} if teacher is not None else None),
                         default=str))
    except ConfigError as error:
        parser.exit(2, f'{error}\n')
    finally:
        close = getattr(teacher, 'close', None)
        if close is not None:
            close()


if __name__ == '__main__':
    main()
