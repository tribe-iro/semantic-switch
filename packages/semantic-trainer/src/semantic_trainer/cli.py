import argparse
import json

from semantic_switch.errors import ConfigError


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
    collect = commands.add_parser('collect', help='label situations with a teacher judge through a resumable journal; publishes labels and unresolved targets')
    collect.add_argument('definition')
    collect.add_argument('inputs', help='JSONL schema-2 situation envelopes {schema, input, groups, pairs, role, tags, selections}')
    collect.add_argument('--teacher', required=True, help='judge source whose terms permit training use, e.g. openrouter:google/gemma-4-31b-it')
    collect.add_argument('--provider', help='pinned OpenRouter provider route')
    collect.add_argument('--fact', help='label only this visible fact and its members')
    collect.add_argument('--to', required=True, help='collection manifest NAME.collection.json; NAME.journal.jsonl resumes the run')
    collect.add_argument('--window', type=int, default=32, help='situations rendered and scored per committed window')
    collect.add_argument('--retry-failed', action='store_true', help='repeat situations whose targets failed in transport')
    prepare = commands.add_parser('prepare', help='create one LoRA job from the task, native exports, asset manifest and family roles')
    prepare.add_argument('definition')
    prepare.add_argument('exports', nargs='+', help='native tune export manifests made with the pipeline-built base')
    prepare.add_argument('--assets', required=True, help='manifest digest from semantic-trainer assets')
    prepare.add_argument('--quant', default='Q4_0')
    prepare.add_argument('--device', default='cpu', help='fitting device: cpu, xpu:N or cuda:N')
    prepare.add_argument('--fit-situations', type=int, help='choose whole fit families totalling exactly this many situations')
    prepare.add_argument('--objective', choices=('specialization', 'instruction_transfer'), default='specialization')
    prepare.add_argument('--recipe', action='append', default=[], metavar='KEY=VALUE',
                         help='recipe setting, once per key; eval_every_examples is required, e.g. eval_every_examples=550 selection_unit=complete_situation')
    prepare.add_argument('--budget', action='append', default=[], metavar='KEY=VALUE',
                         help='budget setting, once per key: max_tokens, max_monitor_tokens, max_sequence_tokens, max_seconds, feasibility_updates')
    prepare.add_argument('--native-device', choices=('auto', 'cpu', 'gpu'), default='auto', help='native selection device; benchmarks request gpu, which refuses CPU fallback')
    prepare.add_argument('--acceptance-fit', action='store_true', help='reserve acceptance_fit families to fit thresholds after selection (default: authored, uncalibrated acceptance)')
    prepare.add_argument('--profile', help='fitting profile from semantic-trainer profile backing the chosen eval_every_examples; copied into the job')
    prepare.add_argument('--to', required=True, help='new job directory')
    profile = commands.add_parser('profile', help='per-phase timing of full optimizer updates and complete cold/steady monitor points of a prepared job; '
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
    select = commands.add_parser('select', help='deploy every retained candidate as a merged, quantized GGUF and select on model_select families')
    select.add_argument('job')
    select.add_argument('--run', required=True, help='fitted run directory from semantic-trainer lora')
    select.add_argument('--compatibility', help='record from semantic-trainer migrate admitting a run of the preceding candidate schema')
    migrate = commands.add_parser('migrate', help='compare a ready run of the preceding loss-v2 job/candidate schemas with a re-prepared job; writes a compatibility record')
    migrate.add_argument('job', help='the re-prepared job.yaml')
    migrate.add_argument('--run', required=True, help='original fitted run directory; never modified')
    migrate.add_argument('--to', required=True, help='new compatibility record path')
    check = commands.add_parser('check', help='evaluate a published deployment on check-role families; records the frozen plan in the evidence audit before loading the model')
    check.add_argument('receipt', help='deployment receipt.json from semantic-trainer select')
    check.add_argument('labels', help='label JSONL bound by the evidence assignment; only its check-role families are evaluated')
    check.add_argument('--evidence', required=True, help='DATASET_MANIFEST (NAME.dataset.json) assigning the labels; its audit ledger records exposure')
    check.add_argument('--to', required=True, help='new evaluation directory')
    args = parser.parse_args()
    try:
        if args.command == 'lexical':
            try:
                from .lexical import train
            except ImportError as error:
                raise ConfigError('install fitting dependencies: pip install "semantic-trainer[lexical]"') from error
            result = train(args.export, args.to, fact=args.fact, c=args.c)
        elif args.command == 'assets':
            from .assets import REPOSITORY, build
            result = build(args.model, args.quant, revision=args.revision, source=args.toolchain_source or REPOSITORY)
        elif args.command == 'collect':
            from .collect import collect
            result = collect(args.definition, args.inputs, args.to, teacher=args.teacher, provider=args.provider, fact=args.fact, window=args.window,
                             retry_failed=args.retry_failed)
        elif args.command == 'prepare':
            from .jobs import BUDGET, DEFAULTS, prepare
            result = prepare(args.definition, args.exports, args.to, assets_manifest=args.assets, quant=args.quant, device=args.device, objective=args.objective,
                             budget=settings(args.budget, BUDGET, 'budget'), recipe=settings(args.recipe, DEFAULTS, 'recipe'), native={'device': args.native_device},
                             acceptance_fit=args.acceptance_fit, fit_situations=args.fit_situations, profile=args.profile)
        elif args.command == 'select':
            from .select import select
            result = select(args.job, args.run, args.compatibility)
        elif args.command == 'migrate':
            from .migrate import migrate
            result = migrate(args.job, args.run, args.to)
        elif args.command == 'check':
            from semantic_switch.tools import check_deployment
            result = check_deployment(args.receipt, args.labels, args.evidence, args.to)
        elif args.command == 'profile':
            from .lora import profile
            result = profile(args.job, args.to, steps=args.steps, warmup=args.warmup, cadences=args.cadence)
        else:
            from .lora import train
            result = train(args.job, args.to, dry_run=args.dry_run)
        print(json.dumps(result))
    except ConfigError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
