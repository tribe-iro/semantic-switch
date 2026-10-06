from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import typer
import yaml

from . import tools
from .api import DEFAULT_JUDGE, PRECISE_JUDGE, Model, Runtime
from .errors import ConfigError
from .inline import Check, Pick, Rate

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
EXIT = {"ok": 0, "none": 0, "uncertain": 2, "needs_input": 2, "failed": 3}


OPTS: dict = {}


def worst(statuses) -> int:
    return max((EXIT[s] for s in statuses), default=0)


@app.callback()
def options(judge: str = typer.Option(None, help="judge model: GGUF path or repo:file"),
            precise: bool = typer.Option(False, help=f"use the precise judge ({PRECISE_JUDGE.split(':')[1]})"),
            device: str = typer.Option("auto", help="auto, cpu or gpu"),
            threads: int = typer.Option(None, help="CPU threads"),
            json_out: bool = typer.Option(False, "--json", help="machine-readable JSON lines"),
            verbose: bool = typer.Option(False, help="include scores, cost and detail")):
    source = judge or os.environ.get("SEMANTIC_SWITCH_JUDGE") or (PRECISE_JUDGE if precise else DEFAULT_JUDGE)
    OPTS.update(json=json_out or not sys.stdout.isatty(), verbose=verbose, judge=source, device=device,
                threads=threads)


@contextmanager
def session():
    rt = Runtime({"judge": Model(OPTS["judge"], device=OPTS["device"], threads=OPTS["threads"])})
    try:
        yield rt
    finally:
        rt.close()


def switch_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    local = Path("switches") / f"{name}.yaml"
    return local if local.exists() else Path(f"{name}.yaml")


def emit(obj):
    if hasattr(obj, "as_dict"):
        text = json.dumps(obj.as_dict(OPTS["verbose"]), ensure_ascii=False, default=str) if OPTS["json"] else \
            (obj.explain() if OPTS["verbose"] and hasattr(obj, "explain") else str(obj))
    elif isinstance(obj, dict) and obj and all(hasattr(v, "as_dict") for v in obj.values()):
        text = json.dumps({k: v.as_dict(OPTS["verbose"]) for k, v in obj.items()}, ensure_ascii=False, default=str) \
            if OPTS["json"] else "\n".join(f"{k}: {v}" for k, v in obj.items())
    else:
        text = json.dumps(obj, ensure_ascii=False, default=str, indent=None if OPTS["json"] else 2)
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
    raise typer.Exit(4 if "failed" in seen else worst(seen))


def inline(spec, each: bool, bool_exit: bool = False):
    with session() as rt:
        if each:
            run_stream(rt.ask(s, {"_": spec})["_"] for s in states())
        r = rt.ask(stdin_state(), {"_": spec})["_"]
    emit(r)
    raise typer.Exit((0 if r.value else 1) if bool_exit and r.status == "ok" else EXIT[r.status])


@app.command()
def check(statement: str, min_score: float = typer.Option(None), each: bool = False,
          excludes: list[str] = typer.Option([], "--excludes"),
          unstated: bool = typer.Option(False, help="three-way: true, false, or the input does not say")):
    inline(Check(statement, tuple(excludes), min_score, unstated), each, bool_exit=True)


@app.command()
def pick(question: str, options: Path = typer.Option(...), allow_none: bool = False,
         shortlist: str = typer.Option(None), focus: str = typer.Option(None), min_score: float = typer.Option(None),
         each: bool = False):
    opts = [line.strip() for line in options.read_text(encoding="utf-8").splitlines() if line.strip()]
    inline(Pick(question, opts, allow_none=allow_none, shortlist=shortlist, min_score=min_score,
                ids=(lambda o: o) if focus else None, focus=focus), each)


@app.command()
def rate(question: str, levels: str = typer.Option(...), min_score: float = typer.Option(None), each: bool = False):
    inline(Rate(question, tuple(x.strip() for x in levels.split(",")), min_score), each)


def question_specs(path: Path) -> dict:
    out = {}
    for name, q in yaml.safe_load(path.read_text(encoding="utf-8")).items():
        if "check" in q:
            out[name] = Check(q["check"], tuple(q.get("excludes", ())), q.get("min_score"), q.get("unstated", False))
        elif "pick" in q:
            out[name] = Pick(q["pick"], q["options"], allow_none=q.get("allow_none", False),
                             min_score=q.get("min_score"))
        else:
            out[name] = Rate(q["rate"], tuple(q["levels"]), q.get("min_score"))
    return out


@app.command()
def ask(questions: Path, each: bool = False):
    specs = question_specs(questions)
    with session() as rt:
        if each:
            run_stream(rt.ask(s, specs) for s in states())
        result = rt.ask(stdin_state(), specs)
    emit(result)
    raise typer.Exit(worst(statuses(result)))


@app.command()
def run(switch: str, each: bool = False):
    with session() as rt:
        sw = rt.load(switch_path(switch))
        if each:
            run_stream(sw.many(states()))
        r = sw(stdin_state())
    emit(r)
    raise typer.Exit(EXIT[r.status])


@app.command()
def explain(switch: str, raw: bool = False):
    with session() as rt:
        r = rt.load(switch_path(switch))(stdin_state())
    if OPTS["json"]:
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
def label(switch: str, inputs: str = typer.Argument(""), to: str = "examples", stale: bool = False):
    with session() as rt:
        n = tools.label(rt, switch_path(switch), inputs, to, stale, ask=prompt, show=lambda s: typer.echo(s, err=True))
        typer.echo(f"labelled {n} inputs")


@app.command()
def specialize(switch: str, fact: str):
    with session() as rt:
        report = tools.specialize(rt, switch_path(switch), fact)
    if OPTS["json"]:
        emit(report)
    else:
        typer.echo("\n".join(report["report"]))


@app.command()
def test(switch: str, data: str = typer.Option(None), intervals: bool = False):
    with session() as rt:
        report = tools.run_test(rt, switch_path(switch), data, intervals)
    if OPTS["json"]:
        emit(report)
    else:
        typer.echo(tools.explain_test(report))


@app.command()
def diff(switch: str):
    with session() as rt:
        emit(tools.diff(rt, switch_path(switch)))


@app.command()
def lock(switch: str, dry_run: bool = typer.Option(False, help="show what would be locked without writing")):
    with session() as rt:
        emit(tools.lock(rt, switch_path(switch), dry_run))


@app.command()
def export(switch: str, to: str = typer.Option(None, "--to", help="output JSONL (default: <switch>.export.jsonl)")):
    with session() as rt:
        emit(tools.export(rt, switch_path(switch), to))


@app.command()
def doctor(model: str = "judge", quality: bool = typer.Option(False, help="also run the judge quality probe")):
    with session() as rt:
        report = tools.doctor(rt, model, quality)
    emit(report)
    raise typer.Exit(0 if report["ok"] else 1)


@app.command()
def bench(switch: str = typer.Argument(None), n: int = typer.Option(20, "-n", "--n")):
    with session() as rt:
        emit(tools.bench(rt, switch_path(switch) if switch else None, n))


def main():
    try:
        app()
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        sys.exit(5)
