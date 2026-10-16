import argparse
import json

from semantic_switch import ConfigError


def bound(pairs) -> dict:
    out = {}
    for pair in pairs:
        role, separator, judge = pair.partition('=')
        if not separator or not role or not judge or role in out:
            raise ConfigError(f'--binding expects one ROLE=JUDGE per remaining model role, got {pair!r}')
        out[role] = judge
    return out


def settings(pairs, defaults, kind) -> dict:
    import yaml
    parsed = {}
    for pair in pairs:
        key, separator, value = pair.partition('=')
        if not separator or not key:
            raise ConfigError(f'--{kind} expects KEY=VALUE, got {pair!r}')
        if key not in defaults:
            raise ConfigError(f'unknown --{kind} key {key!r}; supported: {", ".join(sorted(defaults))}')
        if key in parsed:
            raise ConfigError(f'--{kind} {key} given more than once')
        expected = int if defaults[key] is None else type(defaults[key])
        try:
            parsed[key] = yaml.safe_load(value) if expected is list else expected(value)
        except ValueError:
            raise ConfigError(f'--{kind} {key} expects {expected.__name__}, got {value!r}') from None
    return parsed


def main():
    from .deploy import REPRESENTATIONS
    parser = argparse.ArgumentParser(prog='semantic-trainer')
    commands = parser.add_subparsers(dest='command', required=True)
    lexical = commands.add_parser('lexical')
    lexical.add_argument('export')
    lexical.add_argument('--to', required=True)
    lexical.add_argument('--fact')
    lexical.add_argument('--c', type=float, default=10.)
    assets = commands.add_parser('assets', help='build or verify the pinned HF snapshot, F16/quantized GGUF bases and llama.cpp toolchain')
    assets.add_argument('--model', required=True, help='Hugging Face repository, e.g. google/gemma-4-E2B-it')
    assets.add_argument('--quant', action='append', required=True, help='llama-quantize type of the deployed base, e.g. Q4_0; repeatable')
    assets.add_argument('--revision', help='HF commit or ref; default resolves main to its commit')
    assets.add_argument('--toolchain-source', help='git remote providing the pinned llama.cpp commit')
    prepare = commands.add_parser('prepare', help='create one LoRA job from the task, native exports, asset manifest and family roles')
    prepare.add_argument('definition', nargs='?')
    prepare.add_argument('exports', nargs='*', help='native tune export manifests made with the pipeline-built base')
    prepare.add_argument('--mixture', help='frozen task-mixture manifest instead of DEFINITION EXPORTS: one multi-task instruction_transfer job')
    prepare.add_argument('--init-run', help='warm start from a retained candidate of this ready fitted run (default: fresh initialization)')
    prepare.add_argument('--init-candidate', help='retained candidate name of --init-run, e.g. step-64; disclosed as experimental without selection evidence')
    prepare.add_argument('--init-selection', help='warm start from the candidate chosen by this initializer-selection record (semantic-trainer select --purpose initializer)')
    prepare.add_argument('--assets', required=True, help='manifest digest from semantic-trainer assets')
    prepare.add_argument('--quant', default='Q4_0')
    prepare.add_argument('--device', default='cpu', help='fitting device: cpu, xpu:N or cuda:N')
    prepare.add_argument('--fit-situations', type=int, help='choose whole fit families totalling exactly this many situations')
    prepare.add_argument('--objective', choices=('specialization', 'instruction_transfer'), default='specialization')
    prepare.add_argument('--recipe', action='append', default=[], metavar='KEY=VALUE',
                         help='recipe setting, once per key, e.g. eval_every_examples=550 selection_unit=complete_situation; without eval_every_examples the job '
                              'is a population to profile before choosing the cadence')
    prepare.add_argument('--budget', action='append', default=[], metavar='KEY=VALUE',
                         help='budget setting, once per key: max_tokens, max_monitor_tokens, max_sequence_tokens, max_seconds, feasibility_updates')
    prepare.add_argument('--native-device', choices=('auto', 'cpu', 'gpu'), default='auto', help='native selection device; benchmarks request gpu, which refuses CPU fallback')
    prepare.add_argument('--acceptance-fit', action='store_true', help='reserve acceptance_fit families to fit thresholds after selection (default: authored, uncalibrated acceptance)')
    prepare.add_argument('--profile', help='fitting profile from semantic-trainer profile backing the chosen eval_every_examples; copied into the job')
    prepare.add_argument('--scope', action='append', metavar='FACT[/MEMBER]', help='adapt only these targets (repeatable); default: every exported target')
    prepare.add_argument('--to', required=True, help='new job directory')
    profile = commands.add_parser('profile', help='complete-operation timing of optimizer updates and a bounded monitoring sample; '
                                                  'writes a fitting profile with projected cadences')
    profile.add_argument('job')
    profile.add_argument('--to', required=True, help='new profile JSON path')
    profile.add_argument('--cadence', type=int, action='append', default=[], help='eval_every_examples to project; repeatable (default: quarter to two epochs)')
    profile.add_argument('--steps', type=int, default=3)
    profile.add_argument('--warmup', type=int, default=1)
    lora = commands.add_parser('lora', help='fit under model_select monitoring; retains the zero adapter and the two best evaluated points')
    lora.add_argument('job')
    lora.add_argument('--to', required=True)
    lora.add_argument('--dry-run', action='store_true')
    lora.add_argument('--checkpoint-seconds', type=float, default=300., help='operational recovery interval; does not change the numerical fit identity')
    select = commands.add_parser('select', help='deploy every retained candidate in the declared representation and select on model_select families')
    select.add_argument('--representation', choices=REPRESENTATIONS, default='merged',
                        help='merged: quantized base+delta GGUF per candidate; runtime-adapter: pinned quantized base plus converted adapter GGUF')
    select.add_argument('job')
    select.add_argument('--run', required=True, help='fitted run directory from semantic-trainer lora')
    select.add_argument('--to', help='new deployment directory (default: RUN/deployment)')
    select.add_argument('--purpose', choices=('application', 'initializer'), default='application',
                        help='application: publish a single-task deployment; initializer: rank base and shortlist per held-out selection task and publish '
                             'RUN/initializer-selection/record.json, never a lock')
    select.add_argument('--binding', action='append', default=[], metavar='ROLE=JUDGE',
                        help='publish a complete application for a scoped job: one judge per remaining target role, pinned in its lock; omitted: scoped support only')
    check = commands.add_parser('check', help='evaluate a published deployment on check-role families; records the frozen plan in the evidence audit before loading the model')
    check.add_argument('receipt', help='deployment receipt.json from semantic-trainer select')
    check.add_argument('labels', help='label JSONL bound by the evidence assignment; only its check-role families are evaluated')
    check.add_argument('--evidence', required=True, help='DATASET_MANIFEST (NAME.dataset.json) assigning the labels; its audit ledger records exposure')
    check.add_argument('--to', required=True, help='new evaluation directory')
    check.add_argument('--binding', action='append', default=[], metavar='ROLE=JUDGE', help='remaining role of a complete application, as published')
    pipeline = commands.add_parser('pipeline', help='run the fixed resumable specialization stages of a request written by ss specialize; stage progress goes to '
                                                    'stderr and one versioned result to stdout')
    pipeline.add_argument('request', help='WORK/request.json')
    pipeline.add_argument('--until', metavar='STAGE', help='stop after this stage, whether reused or computed')
    pipeline.add_argument('--attempt', type=int, help='fit attempt number; a new number fits a new run from the pinned base')
    pipeline.add_argument('--cadence', type=int, help='commit this monitoring cadence (eval_every_examples) for the measured profile')
    commands.add_parser('protocol', help='print the specialization request/result protocol this trainer speaks')
    qualify = commands.add_parser('qualify', help='declare or resume five seed/LR perturbations through existing fit and selection')
    qualify.add_argument('job')
    qualify.add_argument('--to', required=True)
    qualify.add_argument('--run', action='store_true', help='execute incomplete runs; otherwise materialize jobs and summarize available results')
    qualify.add_argument('--representation', choices=REPRESENTATIONS, default='merged')
    qualify.add_argument('--nominal-run', help='reuse a ready matching seed-0 run without copying weights')
    qualify.add_argument('--nominal-receipt', help='native receipt of --nominal-run when published outside RUN/deployment')
    diagnose = commands.add_parser('diagnose', help='bounded fit-only cohort and slot/label audit; no deployment or check calls')
    diagnose.add_argument('job')
    diagnose.add_argument('--to', required=True)
    diagnose.add_argument('--target', required=True)
    diagnose.add_argument('--class', dest='answer', required=True)
    diagnose.add_argument('--families', type=int, default=4)
    diagnose.add_argument('--epochs', type=int, default=4)
    diagnose.add_argument('--rank', type=int)
    diagnose.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.command == 'pipeline':
        from .pipeline import command
        raise SystemExit(command(args.request, until=args.until, attempt=args.attempt, cadence=args.cadence))
    try:
        if args.command == 'qualify':
            from .qualification import qualify
            result = qualify(args.job, args.to, execute=args.run, representation=args.representation,
                             nominal_run=args.nominal_run, nominal_receipt=args.nominal_receipt)
        elif args.command == 'diagnose':
            from .qualification import diagnostic
            result = diagnostic(args.job, args.to, target=args.target, answer=args.answer, families=args.families, epochs=args.epochs, rank=args.rank, execute=args.run)
        elif args.command == 'lexical':
            try:
                from . import train_lexical
            except ImportError as error:
                raise ConfigError(str(error)) from error
            result = train_lexical(args.export, args.to, fact=args.fact, c=args.c)
        elif args.command == 'assets':
            from .assets import REPOSITORY, build
            result = build(args.model, args.quant, revision=args.revision, source=args.toolchain_source or REPOSITORY)
        elif args.command == 'prepare':
            from .jobs import BUDGET, CADENCE, DEFAULTS, prepare
            if (args.init_run is None) != (args.init_candidate is None) or args.init_run and args.init_selection:
                raise ConfigError('warm start names both --init-run and --init-candidate, or only --init-selection')
            if bool(args.mixture) == bool(args.definition):
                raise ConfigError('prepare takes either DEFINITION EXPORTS... or --mixture MANIFEST')
            common = dict(assets_manifest=args.assets, quant=args.quant, device=args.device, budget=settings(args.budget, BUDGET, 'budget'),
                          recipe=settings(args.recipe, {**DEFAULTS, CADENCE: None}, 'recipe'), native={'device': args.native_device}, acceptance_fit=args.acceptance_fit,
                          fit_situations=args.fit_situations, profile=args.profile, scope=args.scope or None,
                          initialization={'run': args.init_run, 'candidate': args.init_candidate} if args.init_run else
                          {'selection': args.init_selection} if args.init_selection else 'fresh')
            if args.mixture:
                from .mixture import prepare_mixture
                result = prepare_mixture(args.mixture, args.to, **common)
            else:
                result = prepare(args.definition, args.exports, args.to, objective=args.objective, **common)
        elif args.command == 'select':
            from .select import select, select_initializer
            if args.purpose == 'initializer' and (args.to or args.binding):
                raise ConfigError('initializer selection publishes RUN/initializer-selection; --to and --binding name an application deployment')
            result = select_initializer(args.job, args.run, args.representation) if args.purpose == 'initializer' else \
                select(args.job, args.run, args.representation, to=args.to, bindings=bound(args.binding) or None)
        elif args.command == 'protocol':
            from .pipeline import protocol
            result = protocol()
        elif args.command == 'check':
            from semantic_switch.training import check_deployment
            result = check_deployment(args.receipt, args.labels, args.evidence, args.to, bindings=bound(args.binding) or None)
        elif args.command == 'profile':
            from .lora import profile
            result = profile(args.job, args.to, steps=args.steps, warmup=args.warmup, cadences=args.cadence)
        else:
            from .lora import train
            result = train(args.job, args.to, dry_run=args.dry_run, checkpoint_seconds=args.checkpoint_seconds)
        print(json.dumps(result))
        if args.command == 'qualify' and args.run and result['summary']['completed'] != result['summary']['declared']:
            raise SystemExit(1)
    except ConfigError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
