from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import batched
from pathlib import Path

import typer
import yaml

from . import judgment, tools
from .api import DEFAULT_JUDGE, PRECISE_JUDGE, Runtime, judge_from
from .errors import ConfigError
from .inline import Check, Pick, Rate, Select

EXIT = {"ok": 0, "none": 0, "uncertain": 2, "needs_input": 2, "failed": 3}
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Interpret case evidence under bounded instructions. Start with pick, check or rate; use run for saved switches. "
                       "Exit codes (worst status over all results, also with --each): " + ", ".join(f"{s} {c}" for s, c in EXIT.items()) +
                       "; check without --each exits 1 for an accepted false; 5 configuration error.")
_INVOCATION_OPTIONS = ContextVar("semantic_switch_cli_options", default=None)
NATIVE = ("device", "threads", "max_input_tokens", "n_ctx", "n_batch", "prefix_slots", "repack")


def invocation_options():
    options = _INVOCATION_OPTIONS.get()
    if options is None:
        return {"json": not sys.stdout.isatty(), "verbose": False, "judge": DEFAULT_JUDGE, "provider": None}
    return options


def worst(statuses) -> int:
    return max((EXIT[s] for s in statuses), default=0)


@app.callback()
def options(ctx: typer.Context, judge: str = typer.Option(None, help="judge: GGUF path, repo:file or openrouter:MODEL"),
            precise: bool = typer.Option(False, help=f"use the precise judge ({PRECISE_JUDGE.split(':')[1]})"),
            provider: str = typer.Option(None, help="OpenRouter provider route; required to lock a hosted judge"),
            device: str = typer.Option(None, help="local: auto (default), cpu or gpu"),
            threads: int = typer.Option(None, help="local: CPU threads"),
            max_input_tokens: int = typer.Option(None, help="local: native case token budget (default 1024)"),
            n_ctx: int = typer.Option(None, help="local: native resident token capacity"),
            n_batch: int = typer.Option(None, help="local: native decode batch capacity (default 512)"),
            prefix_slots: int = typer.Option(None, help="local: resident guidance prefix slots (default 4)"),
            repack: bool = typer.Option(None, "--repack/--no-repack", help="local: CPU weight repacking (default on); participates in execution identity"),
            json_out: bool = typer.Option(False, "--json", help="machine-readable JSON lines"),
            verbose: bool = typer.Option(False, help="include scores, cost and detail")):
    source = judge or os.environ.get("SEMANTIC_SWITCH_JUDGE") or (PRECISE_JUDGE if precise else DEFAULT_JUDGE)
    ctx.obj = dict(json=json_out or not sys.stdout.isatty(), verbose=verbose, judge=source, provider=provider, device=device,
                   threads=threads, max_input_tokens=max_input_tokens, n_ctx=n_ctx, n_batch=n_batch, prefix_slots=prefix_slots,
                   repack=repack)
    token = _INVOCATION_OPTIONS.set(ctx.obj)
    ctx.call_on_close(lambda: _INVOCATION_OPTIONS.reset(token))


@contextmanager
def session(options=None):
    opts = options or invocation_options()
    judge = judge_from(opts["judge"], provider=opts.get("provider"), **{key: opts[key] for key in NATIVE if opts.get(key) is not None})
    with Runtime({"judge": judge}, own_judges=True) as rt:
        yield rt


def switch_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    local = Path("switches") / f"{name}.yaml"
    return local if local.exists() else Path(f"{name}.yaml")


def emit(obj):
    if hasattr(obj, "as_dict"):
        text = json.dumps(obj.as_dict(invocation_options()["verbose"]), ensure_ascii=False, default=str) if invocation_options()["json"] else \
            (obj.explain() if invocation_options()["verbose"] and hasattr(obj, "explain") else str(obj))
    elif isinstance(obj, dict) and obj and all(hasattr(v, "as_dict") for v in obj.values()):
        text = json.dumps({k: v.as_dict(invocation_options()["verbose"]) for k, v in obj.items()}, ensure_ascii=False, default=str) \
            if invocation_options()["json"] else "\n".join(f"{k}: {v}" for k, v in obj.items())
    else:
        text = json.dumps(obj, ensure_ascii=False, default=str, indent=None if invocation_options()["json"] else 2)
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


def run_stream(results):
    seen = set()
    for r in results:
        seen.update(statuses(r))
        emit(r)
    raise typer.Exit(worst(seen))


def load_guidance(path=None):
    from .guidance import Guidance
    return Guidance.load(path) if path is not None else Guidance()


def inline(spec, each: bool, bool_exit: bool = False, guidance_file=None):
    guidance = load_guidance(guidance_file)
    with session() as rt:
        if each:
            run_stream(rt.ask(s, {"_": spec}, guidance=guidance)["_"] for s in states())
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
            run_stream(rt.ask(s, specs, guidance=binding) for s in states())
        result = rt.ask(stdin_state(), specs, guidance=binding)
    emit(result)
    raise typer.Exit(worst(statuses(result)))


@app.command()
def run(switch: str, each: bool = False,
        batch: int = typer.Option(0, help="with --each: run N lines as one finite batch, results in order")):
    with session() as rt:
        sw = rt.load(switch_path(switch))
        if each and batch > 1:
            run_stream(r for chunk in batched(states(), batch) for r in sw.run(list(chunk)))
        if each:
            run_stream(sw.many(states()))
        r = sw(stdin_state())
    emit(r)
    raise typer.Exit(EXIT[r.status])


@app.command()
def explain(switch: str, raw: bool = False):
    with session() as rt:
        r = rt.load(switch_path(switch))(stdin_state())
    if invocation_options()["json"]:
        sys.stdout.write(json.dumps({**r.as_dict(True), "explain": r.explain(raw).splitlines()}, ensure_ascii=False,
                                    default=str) + "\n")
    else:
        typer.echo(r.explain(raw))


@app.command()
def correct(switch: str, fact: str = typer.Option(None), value: str = typer.Option(None),
            case: str = typer.Option(None)):
    with session() as rt:
        sw = rt.load(switch_path(switch))
        data = stdin_state()
        emit(sw.correct(data, case=case) if case else sw.correct(data, fact=fact, value=value))


@app.command()
def label(switch: str, inputs: str = typer.Argument(""), to: str = typer.Option("examples", help="examples, test or a label FILE.jsonl; supplied values are reviewed (Enter confirms, s skips, an answer replaces)"),
          stale: bool = False, reviewer: str = typer.Option(None, help="audit identity recorded on review decisions (default: the login name)")):
    with session() as rt:
        n = tools.label(rt, switch_path(switch), inputs, to, stale, ask=prompt, show=lambda s: typer.echo(s, err=True), reviewer=reviewer)
        typer.echo(f"labelled {n} inputs")


@app.command()
def prune(directory: Path):
    from . import files
    emit({"removed": files.prune(directory), "scope": "generation files no current manifest in this directory names"})


@app.command()
def specialize(switch: str, fact: str, head: str = typer.Option(..., help="supplied head bundle")):
    with session() as rt:
        report = tools.specialize(rt, switch_path(switch), fact, head_bundle=head)
    if invocation_options()["json"]:
        emit(report)
    else:
        typer.echo("\n".join(report["report"]))


@app.command()
def test(switch: str, data: str = typer.Option(None), intervals: bool = False):
    with session() as rt:
        report = tools.run_test(rt, switch_path(switch), data, intervals)
    if invocation_options()["json"]:
        emit(report)
    else:
        typer.echo(tools.explain_test(report))


@app.command()
def diff(switch: str):
    with session() as rt:
        emit(tools.diff(rt, switch_path(switch)))


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
        emit(tools.lock(rt, switch_path(switch), dry_run, head_bundles=bindings))


@app.command()
def export(switch: str, to: str = typer.Option(None, "--to", help="output JSONL (default: <switch>.export.jsonl)"),
           split: str = typer.Option("tune", help="tune, check or all; fitting accepts tune only"),
           native: bool = typer.Option(False, help="include local native training presentations"),
           task_family: str = typer.Option(None, help="owner-defined policy/task family for native training"),
           fact: str = typer.Option(None, help="export only this fact; family assignments still use the complete dataset"),
           labels: str = typer.Option(None, help="reviewed or synthetic label file used instead of the switch test labels; never copied")):
    with session() as rt:
        emit(tools.export(rt, switch_path(switch), to, None if split == "all" else split, native=native, task_family=task_family, fact=fact, labels=labels))


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
