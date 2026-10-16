from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import typer
import yaml

from . import judgment, tools
from .api import DEFAULT_JUDGE, PRECISE_JUDGE, Runtime, judge
from .drivers import REGISTRY, pairs
from .errors import ConfigError
from .evidence import Actor
from .inline import Check, Pick, Rate, Select

EXIT = {"ok": 0, "none": 0, "uncertain": 2, "needs_input": 2, "failed": 3}
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Interpret input evidence under bounded instructions. Start with pick, check or rate; use run for saved switches and frozen locks. "
                       "Exit codes (worst status over all results, also with --each): " + ", ".join(f"{s} {c}" for s, c in EXIT.items()) +
                       "; check without --each exits 1 for an accepted false; 5 configuration error.")
_INVOCATION_OPTIONS = ContextVar("semantic_switch_cli_options", default=None)
NATIVE = ("device", "threads", "max_input_tokens", "max_prefix_tokens", "n_ctx", "n_batch", "prefix_slots", "repack")


def invocation_options():
    options = _INVOCATION_OPTIONS.get()
    if options is None:
        return {"json": not sys.stdout.isatty(), "verbose": False, "judge": None, "provider": None}
    return options


def worst(statuses) -> int:
    return max((EXIT[s] for s in statuses), default=0)


@app.callback()
def options(ctx: typer.Context, judge: str = typer.Option(None, help=f"judge: GGUF path, repo:file, fitted lock or SCHEME:MODEL (e.g. openrouter:MODEL); "
                                                                     f"default {DEFAULT_JUDGE.split(':')[1]}, or the model a fitted lock carries"),
            precise: bool = typer.Option(False, help=f"use the precise judge ({PRECISE_JUDGE.split(':')[1]})"),
            provider: str = typer.Option(None, help="OpenRouter provider route; required to lock a hosted judge"),
            device: str = typer.Option(None, help="local: auto (default), cpu or gpu"),
            threads: int = typer.Option(None, help="local: CPU threads"),
            max_input_tokens: int = typer.Option(None, help="local: input evidence token cap (default 1024)"),
            max_prefix_tokens: int = typer.Option(None, help="local: prefix token cap, chat template plus guidance (default 1024)"),
            n_ctx: int = typer.Option(None, help="local, advanced: smaller per-stream request window for tight memory; default and maximum = input cap + 1536 question-suffix cap (framing included) + prefix cap"),
            n_batch: int = typer.Option(None, help="local: native decode batch capacity (default 512)"),
            prefix_slots: int = typer.Option(None, help="local: resident guidance prefix slots (default 4)"),
            repack: bool = typer.Option(None, "--repack/--no-repack", help="local: CPU weight repacking (default on); participates in execution identity"),
            setting: list[str] = typer.Option(None, "--setting", help=f"judge driver setting KEY=VALUE, repeatable ({REGISTRY.help()})"),
            json_out: bool = typer.Option(False, "--json", help="machine-readable JSON lines"),
            verbose: bool = typer.Option(False, help="include scores, cost and detail")):
    source = judge or os.environ.get("SEMANTIC_SWITCH_JUDGE") or (PRECISE_JUDGE if precise else None)
    ctx.obj = dict(json=json_out or not sys.stdout.isatty(), verbose=verbose, judge=source, provider=provider, device=device,
                   threads=threads, max_input_tokens=max_input_tokens, max_prefix_tokens=max_prefix_tokens, n_ctx=n_ctx, n_batch=n_batch, prefix_slots=prefix_slots,
                   repack=repack, settings=pairs(setting))
    token = _INVOCATION_OPTIONS.set(ctx.obj)
    ctx.call_on_close(lambda: _INVOCATION_OPTIONS.reset(token))


@contextmanager
def session(options=None):
    opts = options or invocation_options()
    settings = driver_settings(opts)
    with Runtime(judge(opts["judge"] or DEFAULT_JUDGE, **settings) if opts["judge"] or settings else None, own_judges=True) as rt:
        yield rt


def driver_settings(opts) -> dict:
    given = {key: opts[key] for key in ("provider", *NATIVE) if opts.get(key) is not None}
    extra = opts.get("settings") or {}
    if set(given) & set(extra):
        raise ConfigError(f"{', '.join(sorted(set(given) & set(extra)))} given both as an option and as --setting")
    return {**given, **extra}


def emit(obj, raw: bool = False):
    opts = invocation_options()
    if hasattr(obj, "as_dict"):
        explained = opts["verbose"] and hasattr(obj, "explain")
        if opts["json"]:
            record = obj.as_dict(opts["verbose"])
            text = json.dumps({**record, "explain": obj.explain(raw).splitlines()} if explained else record, ensure_ascii=False, default=str)
        else:
            text = obj.explain(raw) if explained else str(obj)
    elif isinstance(obj, dict) and obj and all(hasattr(v, "as_dict") for v in obj.values()):
        text = json.dumps({k: v.as_dict(opts["verbose"]) for k, v in obj.items()}, ensure_ascii=False, default=str) \
            if opts["json"] else "\n".join(f"{k}: {v}" for k, v in obj.items())
    else:
        text = json.dumps(obj, ensure_ascii=False, default=str, indent=None if opts["json"] else 2)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def states():
    for n, line in enumerate(sys.stdin, 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ConfigError(f"stdin line {n}: invalid JSON ({e.msg})") from None


def prompt(text: str) -> str:
    sys.stderr.write(text)
    sys.stderr.flush()
    return sys.stdin.readline().rstrip("\n")


def stdin_state():
    text = sys.stdin.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text.strip()


def statuses(result) -> list[str]:
    return [v.status for v in result.values()] if isinstance(result, dict) else [result.status]


def chunked(items, size: int):
    chunk = []
    try:
        for item in items:
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
    except BaseException:
        if chunk:
            yield chunk
        raise
    if chunk:
        yield chunk


def run_stream(score, size: int, raw: bool = False):
    seen = set()
    for chunk in chunked(states(), 1 if sys.stdin.isatty() else max(1, size)):
        for r in score(chunk):
            seen.update(statuses(r))
            emit(r, raw)
    raise typer.Exit(worst(seen))


def load_guidance(path=None):
    from .guidance import Guidance
    return Guidance.load(path) if path is not None else Guidance()


def inline(spec, each: bool, bool_exit: bool = False, guidance_file=None):
    guidance = load_guidance(guidance_file)
    with session() as rt:
        if each:
            run_stream(lambda chunk: [row["_"] for row in rt.ask_many(chunk, {"_": spec}, guidance=guidance)], 2 * rt.batch_inputs())
        r = rt.ask(stdin_state(), {"_": spec}, guidance=guidance)["_"]
    emit(r)
    raise typer.Exit((0 if r.value else 1) if bool_exit and r.status == "ok" else EXIT[r.status])


@app.command()
def check(statement: str, min_score: float = typer.Option(None), each: bool = False,
          excludes: list[str] = typer.Option([], "--excludes"),
          unstated: bool = typer.Option(False, help="three-way: supported, contradicted or insufficient evidence"),
          guidance: Path = typer.Option(None, "--guidance", help="bounded instruction file")):
    inline(Check(statement, tuple(excludes), min_score, unstated), each, bool_exit=True, guidance_file=guidance)


def files_choice_options(path):
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        from .files import read_yaml
        value = read_yaml(path)
        if not isinstance(value, (dict, list)) or not value:
            raise ConfigError(f"{path}: expected named choice descriptions or a candidate list")
        return value
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.command()
def pick(question: str, options: Path = typer.Option(..., help="YAML/JSON named choice descriptions, or one candidate per text line"), allow_none: bool = False,
         shortlist: str = typer.Option(None), focus: str = typer.Option(None), min_score: float = typer.Option(None),
         each: bool = False, guidance: Path = typer.Option(None, "--guidance", help="bounded instruction file")):
    try:
        opts = files_choice_options(options)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{options}: {error}") from None
    inline(Pick(question, opts, allow_none=allow_none, shortlist=shortlist, min_score=min_score,
                ids=(lambda o: o) if focus else None, focus=focus), each, guidance_file=guidance)


@app.command()
def rate(question: str, levels: str = typer.Option(...), min_score: float = typer.Option(None), each: bool = False,
         guidance: Path = typer.Option(None, "--guidance", help="bounded instruction file")):
    inline(Rate(question, tuple(x.strip() for x in levels.split(",")), min_score), each, guidance_file=guidance)


@app.command()
def select(question: str, labels: Path = typer.Option(..., help="YAML mapping of label keys to definitions"),
           excludes: list[str] = typer.Option([], "--excludes"), unstated: bool = False, min_score: float = typer.Option(None),
           max_claims: int = typer.Option(None, help="explicit budget of judged memberships (1-256); default examines every label"),
           each: bool = False, guidance: Path = typer.Option(None, "--guidance", help="bounded instruction file")):
    inline(Select(question, judgment.labels(labels), excludes=tuple(excludes), unstated=unstated, min_score=min_score,
                  max_claims=max_claims), each, guidance_file=guidance)


def question_specs(path: Path) -> dict:
    from .guidance import Guidance
    from .files import read_yaml

    def fail(field, message):
        raise ConfigError(f"{path}:{field}: {message}")

    try:
        definitions = read_yaml(path)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: {error}") from None
    if not isinstance(definitions, dict) or not definitions:
        fail("questions", "expected a nonempty mapping")
    common = {"min_score", "guidance", "excludes"}
    allowed = {"check": common | {"check", "unstated"}, "pick": common | {"pick", "options", "allow_none", "shortlist", "focus"},
               "rate": common | {"rate", "levels"}, "select": common | {"select", "labels", "unstated", "max_claims"}}
    constructors = {"check": Check, "pick": Pick, "rate": Rate, "select": Select}
    out = {}
    for name, q in definitions.items():
        if not isinstance(name, str) or not name.strip():
            fail("questions", "question names must be nonempty strings")
        if not isinstance(q, dict):
            fail(name, "expected a question mapping")
        kinds = set(q) & set(allowed)
        if len(kinds) != 1:
            fail(name, "specify exactly one of check, pick, rate or select")
        kind = next(iter(kinds))
        unknown = set(q) - allowed[kind]
        if unknown:
            fail(name, f"unsupported fields: {', '.join(sorted(map(str, unknown)))}")
        required = {"pick": "options", "rate": "levels", "select": "labels"}.get(kind)
        if required is not None and required not in q:
            fail(name, f"{kind} needs {required}")
        args = {key: value for key, value in q.items() if key != kind}
        if args.get("guidance") == []:
            args["guidance"] = Guidance()
        try:
            if isinstance(args.get("labels"), str):
                args["labels"] = judgment.labels(path.parent / args["labels"])
            out[name] = constructors[kind](q[kind], **args)
        except ConfigError as error:
            fail(name, str(error))
    return out


@app.command()
def ask(questions: Path, each: bool = False, guidance: Path = typer.Option(None, "--guidance", help="bounded instruction file")):
    specs = question_specs(questions)
    binding = load_guidance(guidance)
    with session() as rt:
        if each:
            run_stream(lambda chunk: rt.ask_many(chunk, specs, guidance=binding), 2 * rt.batch_inputs())
        result = rt.ask(stdin_state(), specs, guidance=binding)
    emit(result)
    raise typer.Exit(worst(statuses(result)))


@app.command(help="Run a saved switch or frozen lock on stdin; --verbose adds the explanation (text, or an explain field with --json), --raw shows "
                  "the raw scores in it.")
def run(switch: str, each: bool = False,
        batch: int = typer.Option(0, help="with --each: read ahead at most N lines (default: twice the participating judges' batch size); results stay in input order"),
        raw: bool = typer.Option(False, help="with --verbose: raw scores in the explanation")):
    with session() as rt:
        sw = rt.load(switch)
        if each:
            run_stream(sw.run, batch if batch > 0 else 2 * max((rt.batch_inputs(name) for name in sw.plan.bindings), default=1), raw)
        r = sw(stdin_state())
    emit(r, raw)
    raise typer.Exit(EXIT[r.status])


@app.command()
def correct(switch: str, fact: str = typer.Option(None), value: str = typer.Option(None),
            rule: str = typer.Option(None, help="record the rule this input should reach; its fact labels follow"),
            author: str = typer.Option(None, help="author identity recorded on the label (default: the login name)"),
            actor: str = typer.Option(..., help="who writes the label: human, or agent with --model and --session"),
            model: str = typer.Option(None, help="agent model identity (agent actors)"), session_id: str = typer.Option(None, "--session", help="agent session identity (agent actors)")):
    by = {"author": author, "actor": Actor(actor, model, session_id)}
    with session() as rt:
        sw = rt.load(switch)
        data = stdin_state()
        emit(sw.correct(data, rule=rule, **by) if rule else sw.correct(data, fact=fact, value=value, **by))


@app.command()
def label(switch: str, inputs: str | None = typer.Argument(None), to: str = typer.Option("examples", help="examples, test or a label FILE.jsonl; supplied values are reviewed (Enter confirms, s skips, an answer replaces)"),
          stale: bool = False, reviewer: str = typer.Option(None, help="audit identity recorded on labels and verifications (default: the login name)"),
          actor: str = typer.Option(..., help="who decides: human, or agent with --model and --session"),
          model: str = typer.Option(None, help="agent model identity (agent actors)"), session_id: str = typer.Option(None, "--session", help="agent session identity (agent actors)")):
    from .tools.label import validate
    validate(inputs, stale)
    with session() as rt:
        n = tools.label(rt, switch, inputs, to, stale, ask=prompt, show=lambda s: typer.echo(s, err=True), reviewer=reviewer,
                        actor=Actor(actor, model, session_id))
        typer.echo(f"labelled {n} inputs")


@app.command()
def prune(directory: Path):
    from . import files
    emit({"removed": files.prune(directory), "scope": "generation files no current manifest in this directory names"})


@app.command("analyze-head")
def analyze_head(switch: str, fact: str, head: str = typer.Option(..., help="supplied head bundle")):
    with session() as rt:
        report = tools.analyze_head(rt, switch, fact, head_bundle=head)
    if invocation_options()["json"]:
        emit(report)
    else:
        typer.echo("\n".join(report["report"]))


SPECIALIZED = {"completed": 0, "stopped": 0, "awaiting_configuration": 4, "cancelled": 130}


@app.command(help="Fit a verified specialist lock through the configured semantic-trainer pipeline; rerun with --to WORK and no SWITCH to resume. "
                  "Exit 0 when completed or stopped by --until, 4 when awaiting --cadence, otherwise the trainer's failure code.")
def specialize(switch: str = typer.Argument(None, help="switch definition; omit to resume the request recorded in --to"),
               examples: Path = typer.Option(None, "--examples", help="label JSONL; its families keep their assigned roles and check families never enter fitting"),
               to: Path = typer.Option(..., "--to", help="work root holding request.json, the pipeline journal and every stage output"),
               trainer: str = typer.Option(None, envvar="SEMANTIC_TRAINER", help="absolute semantic-trainer executable of the fitting environment"),
               fit_device: str = typer.Option("cpu", "--device", help="fitting device: cpu, xpu:N or cuda:N"),
               native_device: str = typer.Option("auto", "--native-device", help="native export/selection device: auto, cpu or gpu (gpu refuses CPU fallback)"),
               model: str = typer.Option("google/gemma-4-E2B-it", help="Hugging Face base repository"),
               revision: str = typer.Option("main", help="base commit or ref, resolved once to a commit and pinned in the work root"),
               assets: str = typer.Option(None, help="verified trainer asset manifest digest instead of --model/--revision"),
               quant: str = typer.Option("Q8_0", help="quantized native base type"),
               recipe: list[str] = typer.Option(None, "--recipe", help="recipe KEY=VALUE, repeatable (semantic-trainer prepare --help)"),
               budget: list[str] = typer.Option(None, "--budget", help="budget KEY=VALUE, repeatable"),
               fit_situations: int = typer.Option(None, help="choose whole fit families totalling exactly this many situations"),
               acceptance_fit: bool = typer.Option(False, help="reserve acceptance_fit families to fit thresholds"),
               task_family: str = typer.Option(None, help="namespaced task family NAMESPACE:NAME (default local:<switch name>)"),
               scope: list[str] = typer.Option(None, "--scope", help="adapt only these targets (FACT or FACT/MEMBER, repeatable)"),
               profile: Path = typer.Option(None, help="existing fitting profile of this workload instead of measuring one"),
               profile_steps: int = typer.Option(None, help="measured optimizer updates when profiling"),
               profile_warmup: int = typer.Option(1, help="unmeasured warmup updates when profiling"),
               profile_cadence: list[int] = typer.Option(None, "--profile-cadence", help="eval_every_examples to project, repeatable"),
               cadence: int = typer.Option(None, help="monitoring cadence (eval_every_examples) chosen from the measured profile"),
               representation: str = typer.Option("merged", help="deployment representation: merged or runtime-adapter"),
               until: str = typer.Option(None, help="stop after this pipeline stage"),
               attempt: int = typer.Option(None, help="fit attempt number; a new number fits a new run")):
    from .tools.specialize import request, specialize as run
    specification = None
    if switch is not None:
        if examples is None:
            raise ConfigError("specialize needs --examples LABELS")
        specification = request(switch, examples, to, devices={"fit": fit_device, "native": native_device},
                                model=None if assets else {"repo": model, "revision": revision}, assets=assets, quant=quant, recipe=recipe or (),
                                budget=budget or (), fit_situations=fit_situations, acceptance_fit=acceptance_fit, task_family=task_family, scope=scope,
                                profile=profile, measure={"steps": profile_steps or 3, "warmup": profile_warmup, "cadences": sorted(profile_cadence or ())} if profile_steps is not None or profile_cadence else None,
                                representation=representation)
    result = run(trainer, to, specification, until=until, attempt=attempt, cadence=cadence)
    emit(result)
    raise typer.Exit(SPECIALIZED.get(result["state"], result["exit"]))


@app.command(help="Compare two frozen locks on the check-role families of assigned evidence: one bounded scoring operation per arm, journaled and "
                  "resumable. Fitted locks run the model they carry; ordinary locks bind --judge (default model otherwise) and verify its pinned identity. "
                  "Exposure, lineage and review decide whether the paired result is an adoption claim or descriptive only; the promotion decision is "
                  "published beside the report as decision.json for ss promote.")
def compare(baseline: str = typer.Option(..., "--baseline", help="incumbent frozen lock, fitted lock bundle or current pointer"),
            candidate: str = typer.Option(..., "--candidate", help="candidate frozen lock, fitted lock bundle or current pointer"),
            examples: Path = typer.Option(..., "--examples", help="label JSONL; only its check-role families are consulted"),
            evidence: Path = typer.Option(..., "--evidence", help="evidence manifest (X.dataset.json) assigning families to roles"),
            margin: float = typer.Option(..., "--margin", help="required improvement, or the tolerated non-inferiority loss, as a fraction of targets"),
            to: Path = typer.Option(..., "--to", help="new directory for comparison.json, decision.json, rows and the resumable journal"),
            claim: str = typer.Option("improvement", help="improvement or non_inferiority"),
            benefit: str = typer.Option(None, help="operational benefit a non-inferiority claim must show: latency or cost"),
            required_precision: float = typer.Option(None, help="accepted precision the candidate must reach"),
            tolerate: list[str] = typer.Option(None, "--tolerate", help="tolerated regression count failed|unresolved|forbidden=N, repeatable"),
            forbidden: list[str] = typer.Option(None, "--forbidden", help="ADDRESS=ANSWER_KEY never to be accepted, repeatable"),
            recall: list[str] = typer.Option(None, "--recall", help="protected class recall ADDRESS=CLASS:raw|operational=MIN, repeatable; withholding "
                                                                    "is an operational miss and absent references leave it unassessed"),
            seed: int = typer.Option(0, help="family-resampling seed"),
            replicates: int = typer.Option(2000, help="family-resampling replicates"),
            reference: list[str] = typer.Option(None, "--reference", help="label assessment state accepted as an established reference (repeatable): "
                                                                          "verified, computed or authored; default verified and computed"),
            robustness: Path = typer.Option(None, "--robustness", help="authored robustness-transforms manifest for the descriptive per-arm diagnostic")):
    from .tools.deployment import ESTABLISHED, compare_deployments, declaration, entry
    declared = declaration(claim=claim, margin=margin, required_precision=required_precision, benefit=benefit, reference=tuple(reference or ESTABLISHED),
                           tolerated=counts(tolerate or ()), forbidden=answers(forbidden or ()), recall=recalls(recall or ()))
    opts = invocation_options()
    settings = driver_settings(opts)
    explicit = judge(opts["judge"] or DEFAULT_JUDGE, **settings) if opts["judge"] or settings else None
    try:
        bindings = {name: {"judge": explicit} for name, path in (("baseline", baseline), ("candidate", candidate))
                    if explicit is not None and entry(path).receipt is None}
        report = compare_deployments(baseline, candidate, examples, evidence, to, declaration=declared, seed=seed, count=replicates, bindings=bindings,
                                     robustness=robustness)
    finally:
        close = getattr(explicit, "close", None)
        if close is not None:
            close()
    emit({"comparison": str(to / "comparison.json"), "decision": str(to / "decision.json"), "id": report["id"], "promotion": report["promotion"],
          "population": report["population"], "arms": report["arms"], "paired": report["paired"], "reference": report["reference"],
          **({"robustness": report["robustness"]} if "robustness" in report else {})})


def counts(values) -> dict:
    out = pairs(values)
    for key, value in out.items():
        if not value.isdigit():
            raise ConfigError(f"--tolerate {key} expects a nonnegative count, got {value!r}")
    return {key: int(value) for key, value in out.items()}


def recalls(values) -> dict:
    out = {}
    for value in values:
        address, _, rest = value.partition("=")
        name, _, floor = rest.partition(":")
        kind, _, minimum = floor.partition("=")
        try:
            out.setdefault(address, {}).setdefault(name, {})[kind] = float(minimum)
        except ValueError:
            raise ConfigError(f"--recall expects ADDRESS=CLASS:raw|operational=MIN, got {value!r}") from None
    return out


def answers(values) -> dict:
    out = {}
    for value in values:
        address, separator, key = value.partition("=")
        if not separator or not address or not key:
            raise ConfigError(f"--forbidden expects ADDRESS=ANSWER_KEY, got {value!r}")
        out.setdefault(address, []).append(key)
    return out


@app.command(help="Atomically point the current deployment at a verified lock: a first adoption without --decision, otherwise only the candidate of a "
                  "supported decision.json and only while the pointer still serves the incumbent that decision compared against.")
def promote(pointer: Path, lock: str = typer.Argument(..., help="verified fitted lock bundle or frozen lock"),
            reason: str = typer.Option(..., help="recorded in the pointer history"),
            decision: Path = typer.Option(None, help="decision.json of a supported ss compare against the active deployment")):
    from .tools.deployment import promote as point
    emit(point(pointer, lock, reason=reason, decision=decision))


@app.command(help="Point the current deployment back at a previously verified lock from its history; a different current entry fails unchanged.")
def rollback(pointer: Path, reason: str = typer.Option(..., help="recorded in the pointer history"),
             expected: str = typer.Option(..., help="entry id of the incumbent being rolled back"),
             to: str = typer.Option(None, "--to", help="lock entry id from the pointer history (default: the previous one)")):
    from .tools.deployment import rollback as back
    emit(back(pointer, reason=reason, expected=expected, to=to))


@app.command()
def test(switch: str, data: str = typer.Option(None, help="alternate label file; its manifest is published beside the switch dataset and its "
                                                          "consultation is recorded in the switch audit ledger")):
    with session() as rt:
        report = tools.run_test(rt, switch, data)
    if invocation_options()["json"]:
        emit(report)
    else:
        typer.echo(tools.explain_test(report))


@app.command()
def diff(switch: str):
    with session() as rt:
        emit(tools.diff(rt, switch))


@app.command()
def lock(switch: str, dry_run: bool = typer.Option(False, help="evaluate without publishing the lock; dataset assignments and audit exposure are recorded"),
         head: list[str] = typer.Option(None, help="FACT=BUNDLE, repeat for distinct facts")):
    with session() as rt:
        bindings = {}
        for entry in head or ():
            fact, separator, bundle = entry.partition("=")
            if not separator or not fact or not bundle or fact in bindings:
                raise ConfigError("--head requires unique FACT=BUNDLE bindings")
            bindings[fact] = bundle
        emit(tools.lock(rt, switch, dry_run, head_bundles=bindings))


@app.command()
def export(switch: str, to: str = typer.Option(None, "--to", help="output JSONL (default: <switch>.export.jsonl)"),
           split: str = typer.Option("tune", help="tune, check or all; fitting accepts tune only"),
           native: bool = typer.Option(False, help="include local native training presentations"),
           task_family: str = typer.Option(None, help="owner-defined policy/task family for native training"),
           scope: list[str] = typer.Option(None, "--scope", help="export only these targets (FACT or FACT/MEMBER, repeatable); family assignments still use the complete dataset"),
           labels: str = typer.Option(None, help="reviewed or synthetic label file used instead of the switch test labels; never copied")):
    with session() as rt:
        emit(tools.export(rt, switch, to, None if split == "all" else split, native=native, task_family=task_family, scope=scope or None, labels=labels))


@app.command()
def doctor(model: str = "judge"):
    with session() as rt:
        report = tools.doctor(rt, model)
    emit(report)
    raise typer.Exit(0 if report["ok"] else 1)


def main():
    try:
        app()
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        sys.exit(5)
