import argparse
import json
import os

from semantic_switch import ConfigError


def pairs(values, kind, cast=str) -> dict:
    out = {}
    for value in values:
        key, separator, given = value.partition('=')
        if not separator or not key or key in out:
            raise ConfigError(f'--{kind} expects one KEY=VALUE per key, got {value!r}')
        try:
            out[key] = cast(given)
        except ValueError:
            raise ConfigError(f'--{kind} {key} expects {cast.__name__}, got {given!r}') from None
    return out


def parser():
    parser = argparse.ArgumentParser(prog='semantic-learn', description='Explicit capture, teacher acquisition, review export and refitting rounds; '
                                                                        'nothing runs in the background. Compare, promote and roll back deployments with ss.')
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect', help='label situations with a teacher judge through a resumable journal; publishes labels and unresolved targets')
    collect.add_argument('definition')
    collect.add_argument('inputs', help='JSONL schema-3 situation envelopes {schema, input, groups, pairs, role, tags, selections}')
    collect.add_argument('--teacher', required=True, help='judge source whose terms permit training use, e.g. openrouter:google/gemma-4-31b-it')
    collect.add_argument('--provider', help='pinned OpenRouter provider route')
    collect.add_argument('--training-use', required=True, metavar='DECLARATION',
                         help='YAML/JSON training use declaration {permission, model, provider, contract, revision, use} naming the observed teacher route')
    collect.add_argument('--fact', help='label only this visible fact and its members')
    collect.add_argument('--to', required=True, help='collection manifest NAME.collection.json; NAME.journal.jsonl resumes the run')
    collect.add_argument('--window', type=int, default=32, help='situations rendered and scored per committed window')
    collect.add_argument('--retry-failed', action='store_true', help='repeat situations whose targets failed in transport')
    review = commands.add_parser('review', help='bounded review queue for ss label: a seeded random audit plus optional declared triage strata')
    review.add_argument('definition')
    review.add_argument('--to', required=True, help='NAME.review.json; the queue is published beside it')
    review.add_argument('--size', type=int, required=True, help='total rows to review')
    review.add_argument('--audit', type=int, required=True, help='rows drawn uniformly at random from the eligible population')
    review.add_argument('--capture', action='append', default=[], help='NAME.capture.jsonl log, repeatable')
    review.add_argument('--collection', action='append', default=[], help='NAME.collection.json from semantic-learn collect, repeatable')
    review.add_argument('--stratum', action='append', default=[], help='triage stratum in priority order: teacher_uncertain, disagreement, inconsistency, feedback, conflict')
    review.add_argument('--seed', type=int, default=0)
    diagnose = commands.add_parser('diagnose', help='record an evidence-backed review reason without creating a training label')
    diagnose.add_argument('review')
    diagnose.add_argument('unit')
    diagnose.add_argument('--reason', choices=('input_view', 'coverage', 'label', 'relation', 'optimization', 'acceptance'), required=True)
    diagnose.add_argument('--reviewer', required=True)
    diagnose.add_argument('--evidence', required=True)
    diagnose.add_argument('--note', required=True)
    diagnose.add_argument('--to', required=True)
    situations = commands.add_parser('situations', help='write the current captured, redacted situations for teacher collection')
    situations.add_argument('definition')
    situations.add_argument('captures', nargs='+', help='NAME.capture.jsonl logs')
    situations.add_argument('--to', required=True, help='new situation JSONL')
    round_ = commands.add_parser('round', help='freeze eligible examples and fit a candidate lock from the pinned base through the configured trainer')
    round_.add_argument('definition')
    round_.add_argument('--labels', action='append', required=True, help='reviewed or collected label JSONL, repeatable')
    round_.add_argument('--to', required=True, help='round directory; rerun unchanged to resume an interrupted round')
    round_.add_argument('--request', required=True, help='specialization request/recipe JSON (the ss specialize request schema); the round supplies its '
                                                        'frozen labels, task sources and work root')
    round_.add_argument('--trainer', default=os.environ.get('SEMANTIC_TRAINER'), help='absolute semantic-trainer executable (default: SEMANTIC_TRAINER)')
    round_.add_argument('--cutoff', required=True, help='ISO time with UTC offset; later labels are excluded')
    round_.add_argument('--serving', required=True, choices=('separate_host', 'interrupt', 'fallback'), help='serving policy while this round fits')
    round_.add_argument('--min-reviewed', type=int, required=True, help='reviewed targets required per target address')
    round_.add_argument('--min-per-class', type=int, required=True, help='examples required per answer of every target')
    round_.add_argument('--incumbent', help='receipt.json of the active deployment, recorded for the later comparison')
    round_.add_argument('--tombstones', help='deletion tombstone store; tombstoned inputs are excluded')
    round_.add_argument('--cap', action='append', default=[], metavar='ORIGIN:PRODUCER=N', help='maximum examples from one label producer')
    round_.add_argument('--cadence', type=int)
    round_.add_argument('--until')
    round_.add_argument('--attempt', type=int)
    tombstone = commands.add_parser('tombstone', help='mark an input deleted for future exports and rounds')
    tombstone.add_argument('store')
    tombstone.add_argument('definition')
    tombstone.add_argument('input', help='the input as JSON')
    tombstone.add_argument('--reason', required=True)
    tombstone.add_argument('--scope', default='situation', choices=('situation', 'projection'),
                           help='delete this exact situation, or explicitly every situation sharing its read projection')
    affected = commands.add_parser('affected', help='published deployments whose recorded memberships include tombstoned inputs')
    affected.add_argument('store')
    affected.add_argument('receipts', nargs='+')
    affected.add_argument('--pointer')
    return parser


def run(args):
    if args.command == 'diagnose':
        from .review import diagnose
        return diagnose(args.review, args.unit, reason=args.reason, reviewer=args.reviewer, evidence=args.evidence, note=args.note, to=args.to)
    if args.command == 'collect':
        from semantic_switch.training import files
        from . import collect
        return collect(args.definition, args.inputs, args.to, teacher=args.teacher, training_use=files.read_yaml(args.training_use), provider=args.provider,
                       fact=args.fact, window=args.window, retry_failed=args.retry_failed)
    if args.command == 'review':
        from .review import review_export
        return review_export(args.definition, args.to, size=args.size, audit=args.audit, captures=args.capture, collected=args.collection,
                             strata=tuple(args.stratum), seed=args.seed)
    if args.command == 'situations':
        from .review import situations
        return situations(args.definition, args.captures, args.to)
    if args.command == 'round':
        from .rounds import run_round
        return run_round(args.definition, args.labels, args.to, trainer=args.trainer, request=args.request, cutoff=args.cutoff, serving=args.serving,
                         min_reviewed=args.min_reviewed, min_per_class=args.min_per_class, incumbent=args.incumbent, deleted=args.tombstones,
                         caps=pairs(args.cap, 'cap', int), until=args.until, attempt=args.attempt, cadence=args.cadence)
    if args.command == 'tombstone':
        from .rounds import tombstone
        return tombstone(args.store, args.definition, json.loads(args.input), reason=args.reason, scope=args.scope)
    from .rounds import affected
    return affected(args.store, args.receipts, args.pointer)


def main():
    arguments = parser()
    args = arguments.parse_args()
    try:
        print(json.dumps(run(args), default=str))
    except ConfigError as error:
        arguments.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
