from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import parse_qs, urlsplit

from semantic_switch import ConfigError, files
from bench import evidence
from bench.outcomes import process
from semantic_switch import Actor
from bench.evidence import Review

HOST = "127.0.0.1"
ROOT = Path(__file__).resolve().parents[1]
FIELDS = frozenset({"frame", "decision", "target", "replacement", "note"})
WITHDRAWAL = frozenset({"frame", "target", "note"})
COVERAGE = ("status", "disclosure", "author", "reviewers", "actors", "self_review", "frames", "roles", "records", "stale", "disputes",
            "unconfirmed_replacements", "skipped")


def builder(manifest):
    module, _, kind = manifest["reference"]["producer"].rpartition(".")
    if kind != "reference":
        raise ConfigError(f"{manifest['pack']['name']}: built by {manifest['reference']['producer']}, which takes no review decisions")
    return importlib.import_module(module)


class Worksheet:
    def __init__(self, directory, reviews=None):
        self.directory = Path(directory)
        data = evidence.read(self.directory)
        self.manifest, self.rows = data["manifest"], data["review_worksheet"]
        self.module = builder(self.manifest)
        self.reviews = Path(reviews) if reviews else self.directory / "reviews.jsonl"
        self.index = {row["frame"]: n for n, row in enumerate(self.rows)}
        self.author = evidence.author_of(self.manifest["reference"]["producer"], self.manifest["reference"]["revision"])

    def rebuild(self) -> list[str]:
        return ["python", "-m", self.module.__name__, "pack", "--output", str(self.directory), "--reviews", str(self.reviews)]

    def standing(self) -> tuple[tuple, tuple]:
        return evidence.standing(evidence.load_reviews(self.reviews))

    def current(self) -> dict:
        out = {}
        for review in self.standing()[0]:
            n = self.index.get(review.frame)
            if n is not None and review.binding == self.rows[n]["binding"]:
                out.setdefault(review.frame, []).append(review)
        return out

    def progress(self) -> dict:
        current = self.current()
        done = {frame for frame, reviews in current.items() if any(r.kind == "evidence" for r in reviews)}

        def tally(rows):
            reviewed = sum(row["frame"] in done for row in rows)
            return {"frames": len(rows), "done": reviewed, "remaining": len(rows) - reviewed}
        return {"required": tally(self.rows), "mandatory": tally([row for row in self.rows if row["reasons"] != ["sample"]]),
                "sampled": tally([row for row in self.rows if row["reasons"] == ["sample"]]),
                "corrections": sorted(frame for frame, reviews in current.items() if any(r.decision != "confirm" for r in reviews)),
                "next": next((n for n, row in enumerate(self.rows) if row["frame"] not in done), None)}

    def fresh(self) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, "-m", self.module.__name__, "pack", "--output", tmp,
                       *(["--reviews", str(self.reviews.resolve())] if self.reviews.is_file() else [])]
            done = process(command, cwd=ROOT)
            if done["exit_code"]:
                raise ConfigError(f"{' '.join(command[1:])}: {done['stderr'].strip()}")
            return evidence.read(tmp)["manifest"]

    def is_current(self, built) -> bool:
        return built["payloads"]["review_worksheet.jsonl"]["sha256"] == self.manifest["payloads"]["review_worksheet.jsonl"]["sha256"]

    def status(self) -> dict:
        built, (_, withdrawn) = self.fresh(), self.standing()
        return {"pack": self.manifest["pack"]["name"], "version": self.manifest["pack"]["version"], "reviews": str(self.reviews),
                "records": len(evidence.load_reviews(self.reviews)),
                "withdrawn": [{"frame": d.frame, "reviewer": d.reviewer, "scope": evidence.scope_of(d), "decision": d.decision, "note": w.note}
                              for d, w in withdrawn],
                "worksheet": {"current": self.is_current(built), "rebuild": " ".join(self.rebuild())},
                "progress": self.progress(), "coverage": {key: built["review"][key] for key in COVERAGE}}


class Session(Worksheet):
    def __init__(self, directory, reviews=None, reviewer: str | None = None, actor: Actor | None = None):
        super().__init__(directory, reviews)
        if reviewer is not None:
            evidence.check_actor(actor, reviewer)
        self.reviewer, self.actor = reviewer, actor
        _, frames, _ = evidence.load_frames(self.directory)
        self.frames = {frame.id: frame for frame in frames}
        self.ref = self.module.standard()
        self.golds, self.cited = None, None

    def sources(self) -> dict:
        if self.cited is None:
            root = evidence.load_pack(self.module.PACK).root
            self.cited = {}
            for source in self.manifest["pack"]["sources"]:
                if source["use"] != evidence.CITED:
                    continue
                excerpt, document = root / source["excerpt"], root / source["document"]["path"]
                self.cited[source["id"]] = {
                    "section": source["section"], "edition": source["edition"], "url": source["url"], "locator": source["locator"],
                    "excerpt": excerpt.read_text() if excerpt.is_file() and files.sha256_file(excerpt) == source["sha256"] else None,
                    "document": document if document.is_file() and files.sha256_file(document) == source["document"]["sha256"] else None}
        return self.cited

    def state(self) -> dict:
        current = self.current()
        return {"pack": self.manifest["pack"]["name"], "version": self.manifest["pack"]["version"], "reviewer": self.reviewer, "actor": str(self.actor or ""),
                "author": self.author, "self_review": self.reviewer == self.author, "reviews": str(self.reviews), "rebuild": " ".join(self.rebuild()),
                "status": self.manifest["review"]["status"], "progress": self.progress(),
                "frames": [{"frame": row["frame"], "sampled": row["reasons"] == ["sample"],
                            "decisions": sorted({r.decision for r in current.get(row["frame"], ()) if r.kind == "evidence"})} for row in self.rows]}

    def view(self, n: int) -> dict:
        if not 0 <= n < len(self.rows):
            raise ConfigError(f"frame index {n} is outside the worksheet's {len(self.rows)} frames")
        row, sources = self.rows[n], self.sources()
        frame = self.frames[row["frame"]]
        cited = sorted({rule["source"] for rule in row["rules"].values()} & set(sources))
        return {**{key: row[key] for key in ("frame", "reasons", "role", "family", "condition", "binding", "input", "assertions", "gold", "composition",
                                             "targets", "rules")},
                "index": n, "count": len(self.rows), "tags": list(frame.tags), "language": next((t for t in frame.tags if t in self.module.LANGUAGES), None),
                "sources": {sid: {**{k: v for k, v in sources[sid].items() if k != "document"}, "document": sources[sid]["document"] is not None}
                            for sid in cited},
                "decisions": [{"reviewer": r.reviewer, "actor": str(r.actor), "kind": r.kind, "decision": r.decision, "target": "/".join(r.target),
                               "note": r.note} for r in self.current().get(row["frame"], ())]}

    def corrected(self, frame, assertions) -> tuple:
        if not isinstance(assertions, list) or not assertions:
            raise ConfigError(f"{frame.id}: a replacement lists the corrected assertions")

        def span(s):
            text = frame.input.get(s.get("field"))
            if not isinstance(text, str) or not isinstance(s.get("start"), int) or not isinstance(s.get("end"), int) or not 0 <= s["start"] < s["end"] <= len(text):
                raise ConfigError(f"{frame.id}: span {s} is outside the public text")
            return {"sha256": evidence.sha256_text(text[s["start"]:s["end"]]), **s}
        try:
            fixed = evidence.assertions_of([{**a, "spans": [span(s) for s in a.get("spans", ())]} for a in assertions])
        except (TypeError, AttributeError) as error:
            raise ConfigError(f"{frame.id}: replacement assertions carry id, subject, property, value, event, time, indication, status, spans and paths ({error})") from None
        if fixed == frame.assertions:
            raise ConfigError(f"{frame.id}: the replacement equals the generated assertions; confirm the frame instead")
        candidate = replace(frame, assertions=fixed)
        gold = evidence.references(self.ref, [candidate])
        if self.golds is None:
            self.golds = {f.id: self.ref.evaluate(f) for f in self.frames.values()}
        found = [c for c in evidence.conflicts(self.ref, [candidate, *(f for f in self.frames.values() if f.id != frame.id)], {**self.golds, **gold})
                 if frame.id in c[:2]]
        if found:
            raise ConfigError(f"{frame.id}: the replacement gives one observable problem of {found[0][1]} different answers at {found[0][2]}")
        return fixed

    def row(self, payload, fields) -> tuple[dict, tuple]:
        if not isinstance(payload, dict) or set(payload) - fields:
            raise ConfigError(f"a decision carries {sorted(fields)}")
        n = self.index.get(payload.get("frame"))
        if n is None:
            raise ConfigError(f"{payload.get('frame')}: not a frame of the current review worksheet")
        row = self.rows[n]
        target = tuple(payload["target"].split("/")) if payload.get("target") else ()
        if target and payload["target"] not in row["targets"]:
            raise ConfigError(f"{row['frame']}: {payload['target']} is not a target of this frame")
        return row, target

    def withdraw(self, payload) -> dict:
        row, target = self.row(payload, WITHDRAWAL)
        review = evidence.admit(Review(row["frame"], self.reviewer, "label" if target else "evidence", evidence.WITHDRAW, row["binding"], target,
                                       None, self.actor, (payload.get("note") or "").strip() or None))
        scope = [r for r in self.current().get(row["frame"], ()) if r.kind == review.kind and r.target == review.target]
        if not any(evidence.withdraws(review, r) for r in scope):
            raise ConfigError(f"{row['frame']}: {self.reviewer} has no current {evidence.scope_of(review)} decision to withdraw"
                              + ("; only a decision's own reviewer withdraws it" if scope else ""))
        return evidence.append_review(self.reviews, review)

    def decide(self, payload) -> dict:
        row, target = self.row(payload, FIELDS)
        if payload.get("replacement") is not None and payload.get("decision") != "replace":
            raise ConfigError(f"{row['frame']}: a replacement, and only a replacement, carries the corrected assertions")
        replacement = None if payload.get("replacement") is None else self.corrected(self.frames[row["frame"]], payload["replacement"])
        review = evidence.admit(Review(row["frame"], self.reviewer, "label" if target else "evidence", payload.get("decision"), row["binding"], target,
                                       replacement, self.actor, (payload.get("note") or "").strip() or None))
        mine = [r for r in self.current().get(row["frame"], ()) if r.kind == review.kind and r.target == review.target]
        if any(r.reviewer == self.reviewer for r in mine):
            raise ConfigError(f"{row['frame']}: {self.reviewer} already decided this {evidence.scope_of(review)}; withdraw it first")
        if any(evidence.act(r) != evidence.act(review) for r in mine):
            raise ConfigError(f"{row['frame']}: another reviewer decided this {evidence.scope_of(review)} differently; resolve the dispute in {self.reviews}")
        return evidence.append_review(self.reviews, review)


def server(session: Session, port: int = 8765) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def local(self) -> bool:
            port_ = self.server.server_address[1]
            if self.headers.get("Host") in (f"{HOST}:{port_}", f"localhost:{port_}"):
                return True
            self.reply(403, {"error": "serve only to localhost"})
            return False

        def do_GET(self):
            if not self.local():
                return
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self.reply(200, PAGE.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self.reply(200, session.state())
                elif url.path == "/api/frame":
                    self.reply(200, session.view(int(parse_qs(url.query).get("i", ["0"])[0])))
                elif url.path.startswith("/source/") and (source := session.sources().get(url.path.removeprefix("/source/"))) and source["document"]:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header("Content-Length", str(source["document"].stat().st_size))
                    self.end_headers()
                    with source["document"].open("rb") as fh:
                        shutil.copyfileobj(fh, self.wfile)
                else:
                    self.reply(404, {"error": "not found"})
            except (ConfigError, ValueError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            if not self.local():
                return
            operation = {"/api/decide": session.decide, "/api/withdraw": session.withdraw}.get(urlsplit(self.path).path)
            if operation is None or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(404, {"error": "POST application/json to /api/decide or /api/withdraw"})
                return
            try:
                record = operation(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            except (ConfigError, ValueError) as error:
                self.reply(400, {"error": str(error)})
                return
            self.reply(200, {"record": record, "state": session.state()})
    return HTTPServer((HOST, port), Handler)


def actor_of(args) -> Actor:
    return Actor(args.actor, args.model, args.session)


def main():
    parser = argparse.ArgumentParser(prog="bench.review", description="frame review of a published reference pack; local only, no model or network use")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("serve", "serve the review page on 127.0.0.1"), ("status", "rebuild offline with the recorded decisions and print coverage"),
                       ("show", "print one worksheet frame as JSON (default: the first without an evidence decision)"),
                       ("record", "append one decision bound to the current worksheet"),
                       ("withdraw", "append a withdrawal of your own current decision; the history is kept and you may decide again")):
        p = commands.add_parser(name, help=text)
        p.add_argument("pack", type=Path)
        p.add_argument("--reviews", type=Path, help="decisions JSONL (default PACK/reviews.jsonl)")
        if name in ("serve", "record", "withdraw"):
            p.add_argument("--reviewer", required=True, help="reviewer identity recorded on every decision")
            p.add_argument("--actor", choices=sorted(evidence.ACTORS), required=True, help="who decides: human, or agent with --model and --session")
            p.add_argument("--model", help="agent model identity (agent actors)")
            p.add_argument("--session", help="agent session identity (agent actors)")
    commands.choices["serve"].add_argument("--port", type=int, default=8765)
    commands.choices["show"].add_argument("--frame")
    p = commands.choices["record"]
    p.add_argument("--frame", required=True)
    p.add_argument("--decision", choices=sorted(evidence.DECISIONS), required=True)
    p.add_argument("--target", help="confirm one target's label instead of the frame evidence")
    p.add_argument("--note", help="reason; required for skip")
    p.add_argument("--replacement", type=Path, help="JSON list of corrected assertions for replace")
    p = commands.choices["withdraw"]
    p.add_argument("--frame", required=True)
    p.add_argument("--target", help="withdraw a target's label confirmation instead of the frame evidence decision")
    p.add_argument("--note", required=True, help="reason for the withdrawal")
    args = parser.parse_args()
    try:
        if args.command == "status":
            json.dump(Worksheet(args.pack, args.reviews).status(), sys.stdout, indent=1)
            print()
            return
        session = Session(args.pack, args.reviews) if args.command == "show" else Session(args.pack, args.reviews, args.reviewer, actor_of(args))
        if args.command == "show":
            n = session.index[args.frame] if args.frame in session.index else session.progress()["next"] if args.frame is None else None
            if n is None:
                raise ConfigError(f"{args.frame or 'no frame'}: {'not in the worksheet' if args.frame else 'every frame has an evidence decision'}")
            out = session.view(n)
        elif args.command == "record":
            out = session.decide({"frame": args.frame, "decision": args.decision, "target": args.target, "note": args.note,
                                  "replacement": files.read_json(args.replacement) if args.replacement else None})
        elif args.command == "withdraw":
            out = session.withdraw({"frame": args.frame, "target": args.target, "note": args.note})
        else:
            built = session.fresh()
            if not session.is_current(built):
                raise ConfigError(f"{args.pack}: its review worksheet is not what the current builder produces; decisions bound to it would be stale. "
                                  f"Rebuild first: {' '.join(session.rebuild())}")
            httpd = server(session, args.port)
            print(f"reviewing {session.manifest['pack']['name']} as {session.reviewer} ({session.actor}); author {session.author}"
                  f"{' (self-review)' if session.reviewer == session.author else ''}; decisions -> {session.reviews}\n"
                  f"http://{HOST}:{httpd.server_address[1]}/  (Ctrl-C to stop)", flush=True)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                httpd.server_close()
            return
    except ConfigError as error:
        parser.exit(2, f"{error}\n")
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    print()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frame review</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#dcdcde;--card:#fff;--mark:#ffe58a;--hot:#ff9f43;--ok:#1f7a3d;--bad:#b3261e;--accent:#2457c5}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--muted:#9a9aa2;--line:#34343a;--card:#1f1f23;--mark:#6b5a12;--hot:#a5541c;--ok:#5cc27f;--bad:#f2867d;--accent:#8fb0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center}
header b{font-size:15px}.muted{color:var(--muted)}.bar{display:inline-block;width:120px;height:8px;background:var(--line);border-radius:4px;vertical-align:middle;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}main{max-width:1180px;margin:0 auto;padding:12px 16px 96px;display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:14px}
@media (max-width:900px){main{grid-template-columns:1fr}}section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:0;overflow-x:auto}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 6px}.chip{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin:0 4px 2px 0;font-size:12px}
pre.text{white-space:pre-wrap;word-break:break-word;margin:4px 0 8px;font:13px/1.5 ui-monospace,monospace}mark{background:var(--mark);color:inherit;border-radius:2px;cursor:help}mark.hot,tr.hot td{background:var(--hot)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:3px 5px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.v-true{color:var(--ok);font-weight:600}.v-false{color:var(--muted)}.v-null{color:var(--bad);font-weight:600}code{font:12px ui-monospace,monospace}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--card);border-top:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:5px 11px;cursor:pointer}button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.small{padding:0 6px;font-size:12px}kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-radius:3px;padding:0 3px;margin-left:4px}
#msg{flex:1 1 200px}#msg.err{color:var(--bad)}.done{color:var(--ok);font-weight:600}details{margin:4px 0}details pre{max-height:320px;overflow:auto;white-space:pre-wrap;font:12px ui-monospace,monospace}
#replace input,#replace select{width:100%;font:12px ui-monospace,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);padding:2px}
</style></head><body>
<header><b id="pack"></b><span id="who" class="muted"></span><span id="prog"></span><span id="status" class="muted"></span></header>
<main><div>
<section><h2>Frame</h2><div id="head"></div></section>
<section style="margin-top:12px"><h2>Observable input</h2><div id="input"></div></section>
<section style="margin-top:12px"><h2>Asserted evidence</h2><div id="assertions"></div></section>
<section id="replace" style="margin-top:12px" hidden><h2>Replacement assertions</h2><p class="muted">Spans are field:start-end (several separated by ;). Values are JSON. The corrected frame needs a confirmation after the pack is rebuilt.</p>
<table><thead><tr><th>id</th><th>subject</th><th>property</th><th>value</th><th>event</th><th>time</th><th>indication</th><th>status</th><th>spans</th><th>paths</th><th></th></tr></thead><tbody id="rows"></tbody></table>
<p><button id="addrow">Add assertion</button> <input id="rnote" placeholder="reason (optional)" style="width:50%"> <button class="primary" id="sendrep">Record replacement</button> <button id="cancelrep">Cancel</button></p></section>
</div><div>
<section><h2>Reference answer</h2><div id="gold"></div></section>
<section style="margin-top:12px"><h2>Cited rules and sources</h2><div id="rules"></div></section>
<section style="margin-top:12px"><h2>Decisions on this binding</h2><div id="decisions"></div></section>
</div></main>
<footer><button class="primary" id="confirm">Confirm evidence<kbd>c</kbd></button><button id="skip">Skip<kbd>s</kbd></button><button id="rep">Replace<kbd>r</kbd></button>
<button id="withdraw" hidden>Withdraw mine<kbd>w</kbd></button><button id="prev">Prev<kbd>p</kbd></button><button id="next">Next<kbd>n</kbd></button><button id="todo">Next open<kbd>u</kbd></button><span id="msg" class="muted"></span></footer>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const INDICATIONS=["reading","display","configuration","operator_report","handover_note","command","verification","customer_statement","agent_statement","quoted","account_field"];
const STATUSES=["current","stale","resolved","negated","hypothetical","ambiguous","superseded"];
let S=null,F=null,at=0;
async function api(path,body){const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});const j=await r.json();if(!r.ok)throw new Error(j.error);return j}
function say(t,err){$("msg").textContent=t;$("msg").className=err?"err":"muted"}
function bar(p){const pct=p.frames?100*p.done/p.frames:100;return `<span class="bar"><i style="width:${pct}%"></i></span> ${p.done}/${p.frames}`}
function header(){const p=S.progress;$("pack").textContent=`${S.pack} ${S.version}`;
$("who").textContent=`reviewer ${S.reviewer} (${S.actor}) · author ${S.author}${S.self_review?" · self-review, disclosed":""}`;
$("prog").innerHTML=`mandatory ${bar(p.mandatory)} · sampled ${bar(p.sampled)} · remaining ${p.required.remaining}`+(p.corrections.length?` · <b>${p.corrections.length} correction(s): rebuild before confirming them</b>`:"");
$("status").textContent=S.status}
function val(v){const s=JSON.stringify(v);return `<span class="v-${v===true?"true":v===false?"false":v===null?"null":"x"}">${esc(s)}</span>`}
function highlight(text,field){const spans=[];F.assertions.forEach(a=>a.spans.filter(s=>s.field===field).forEach(s=>spans.push([s.start,s.end,a.id])));
const cut=[...new Set([0,text.length,...spans.flatMap(s=>[s[0],s[1]])])].sort((a,b)=>a-b);let out="";
for(let i=0;i<cut.length-1;i++){const a=cut[i],b=cut[i+1],ids=spans.filter(s=>s[0]<=a&&s[1]>=b).map(s=>s[2]);const t=esc(text.slice(a,b));
out+=ids.length?`<mark data-a="${ids.join(" ")}" title="${ids.map(id=>{const x=F.assertions.find(y=>y.id===id);return `${id}: ${x.subject} ${x.property}=${JSON.stringify(x.value)} (${x.status}, ${x.indication})`}).join("\n")}">${t}</mark>`:t}return out}
function heat(ids){document.querySelectorAll("mark,tr[data-a]").forEach(m=>m.classList.toggle("hot",ids.length>0&&m.dataset.a.split(" ").some(i=>ids.includes(i))))}
function bindHeat(root){root.querySelectorAll("[data-ev]").forEach(el=>{el.onmouseenter=()=>heat(el.dataset.ev.split(" ").filter(Boolean));el.onmouseleave=()=>heat([])})}
function render(){header();const sampled=F.reasons.join()==="sample";
$("head").innerHTML=`<div><b>${esc(F.frame)}</b> <span class="muted">${F.index+1} of ${F.count}</span></div><div>${F.reasons.map(r=>`<span class="chip">${esc(r)}</span>`).join("")}<span class="chip">${sampled?"sampled":"mandatory"}</span></div>
<div class="muted">condition <b>${esc(F.condition)}</b> · language <b>${esc(F.language||"?")}</b> · role ${esc(F.role)} · family ${esc(F.family)}</div><div class="muted">tags ${F.tags.map(esc).join(", ")}</div><div class="muted">binding <code>${F.binding.slice(0,16)}</code></div>`;
const paths=new Map();F.assertions.forEach(a=>a.paths.forEach(p=>paths.set(p,[...(paths.get(p)||[]),a.id])));
$("input").innerHTML=Object.entries(F.input).map(([k,v])=>{const ids=[...paths.entries()].filter(([p])=>p===k||p.startsWith(k+".")).flatMap(e=>e[1]);
const body=typeof v==="string"?`<pre class="text">${highlight(v,k)}</pre>`:`<pre class="text">${ids.length?`<mark data-a="${ids.join(" ")}">${esc(JSON.stringify(v,null,1))}</mark>`:esc(JSON.stringify(v,null,1))}</pre>`;
return `<div><code>${esc(k)}</code>${body}</div>`}).join("");
$("assertions").innerHTML=F.assertions.length?`<table><tr><th>id</th><th>subject</th><th>property</th><th>value</th><th>status</th><th>indication</th><th>t</th><th>at</th></tr>${F.assertions.map(a=>`<tr data-a="${a.id}" data-ev="${a.id}"><td>${a.id}</td><td>${esc(a.subject)}</td><td>${esc(a.property)}</td><td>${val(a.value)}</td><td>${a.status}</td><td>${a.indication}</td><td>${a.time}</td><td>${[...a.spans.map(s=>`${s.field}[${s.start}:${s.end}]`),...a.paths].map(esc).join(" ")}</td></tr>`).join("")}</table>`:`<p class="muted">no assertions: every answer rests on absence/default rules</p>`;
const labelled=new Set(F.decisions.filter(d=>d.kind==="label"&&d.reviewer===S.reviewer).map(d=>d.target));
const row=(t,d,btn)=>`<tr data-ev="${d.evidence.join(" ")}"><td><code>${esc(t)}</code></td><td>${val(d.value)}</td><td>${d.rules.map(esc).join(" ")}</td><td>${d.evidence.map(esc).join(" ")}${d.paths.map(esc).join(" ")}${d.unresolved.length?` <span class="v-null">unresolved ${d.unresolved.map(esc).join(" ")}</span>`:""}${d.branch?` branch <code>${esc(d.branch)}</code>`:""}</td><td>${btn?(labelled.has(t)?`<span class="done">ok</span>`:`<button class="small" data-label="${esc(t)}" title="confirm this target label">label</button>`):""}</td></tr>`;
const g=F.gold,targets=new Set(F.targets);let html=`<table><tr><th>target</th><th>value</th><th>rules</th><th>evidence</th><th></th></tr>`;
for(const [t,d] of Object.entries({...g.decisions,...g.primitives}))html+=row(t,d,targets.has(t));html+=`</table>`;
html+=`<p>prohibited actions: ${g.prohibited.length?g.prohibited.map(p=>`<span class="chip">${esc(p)}</span>`).join(""):"<span class='muted'>none</span>"} · complete ${val(g.complete)} · revision <code>${esc(g.revision)}</code></p>`;
if(F.composition){const c=F.composition;html+=`<h2>Composed response</h2><table><tr data-ev="${c.evidence.join(" ")}"><td>${val(c.value)}</td><td>branch <code>${esc(c.branch)}</code></td><td>${c.rules.map(esc).join(" ")}</td><td>${c.evidence.map(esc).join(" ")} ${c.paths.map(esc).join(" ")}</td></tr></table><p class="muted">assessed ${c.scope.map(esc).join(", ")}</p>`}
$("gold").innerHTML=html;
$("rules").innerHTML=Object.values(F.rules).map(r=>{const s=F.sources[r.source];return `<p><b>${esc(r.id)}</b> ${esc(r.text)}<br><span class="muted">${s?`${esc(r.source)}: ${esc(s.edition)}<br>`+s.locator.map(l=>`${esc(l.section)}, pages ${l.pages[0]}–${l.pages[1]}${s.document?` <a href="/source/${encodeURIComponent(r.source)}#page=${l.pages[0]}" target="_blank">open page</a>`:""}`).join("<br>"):esc(r.source)}</span></p>`}).join("")
+Object.entries(F.sources).map(([id,s])=>`<details><summary>pinned excerpt: ${esc(id)}</summary>${s.excerpt?`<pre>${esc(s.excerpt)}</pre>`:"<p class='muted'>local excerpt absent or changed; fetch and verify the source</p>"}</details>`).join("");
$("decisions").innerHTML=F.decisions.length?F.decisions.map(d=>`<div>${esc(d.decision)} ${d.kind==="label"?`label <code>${esc(d.target)}</code>`:"evidence"} by ${esc(d.reviewer)} (${esc(d.actor)})${d.note?`: ${esc(d.note)}`:""}</div>`).join(""):`<span class="muted">none yet</span>`;
const mine=F.decisions.some(d=>d.kind==="evidence"&&d.reviewer===S.reviewer);["confirm","skip","rep"].forEach(b=>$(b).disabled=mine);$("withdraw").hidden=!mine;
document.querySelectorAll("[data-label]").forEach(b=>b.onclick=()=>decide({decision:"confirm",target:b.dataset.label},false));
bindHeat(document);$("replace").hidden=true;say(mine?"you already decided this frame":"")}
async function go(i){if(!S)return;at=Math.max(0,Math.min(S.frames.length-1,i));F=await api(`/api/frame?i=${at}`);render();history.replaceState(null,"",`#${at}`)}
async function decide(body,advance=true){try{const r=await api("/api/decide",{frame:F.frame,...body});S=r.state;say(`recorded ${r.record.decision} ${r.record.kind}${r.record.target.length?" "+r.record.target.join("/"):""}`);
if(advance&&S.progress.next!==null){const n=S.frames.findIndex((f,j)=>j>at&&!f.decisions.length);await go(n>=0?n:S.progress.next)}else await go(at)}catch(e){say(e.message,true)}}
async function withdraw(){const note=prompt("Reason for withdrawing your evidence decision (required):");if(!note||!note.trim())return;
try{const r=await api("/api/withdraw",{frame:F.frame,note});S=r.state;say("withdrew your evidence decision; the history is kept and you may decide again");await go(at)}catch(e){say(e.message,true)}}
function skip(){const note=prompt("Reason for skipping this frame (required):");if(note&&note.trim())decide({decision:"skip",note})}
function cell(v){return `<td><input value="${esc(v)}"></td>`}
function sel(opts,v){return `<td><select>${opts.map(o=>`<option${o===v?" selected":""}>${o}</option>`).join("")}</select></td>`}
function addRow(a){const tr=document.createElement("tr");tr.innerHTML=cell(a.id)+cell(a.subject)+cell(a.property)+cell(JSON.stringify(a.value))+cell(a.event)+cell(a.time)+sel(INDICATIONS,a.indication)+sel(STATUSES,a.status)
+cell(a.spans.map(s=>`${s.field}:${s.start}-${s.end}`).join("; "))+cell(a.paths.join(", "))+`<td><button class="small">x</button></td>`;tr.querySelector("button").onclick=()=>tr.remove();$("rows").appendChild(tr)}
function openReplace(){$("rows").innerHTML="";F.assertions.forEach(addRow);$("replace").hidden=false;$("replace").scrollIntoView()}
function sendReplace(){try{const rows=[...$("rows").children].map(tr=>{const v=[...tr.querySelectorAll("input,select")].map(e=>e.value.trim());
return {id:v[0],subject:v[1],property:v[2],value:JSON.parse(v[3]),event:v[4],time:Number(v[5]),indication:v[6],status:v[7],
spans:v[8]?v[8].split(";").map(s=>{const m=s.trim().match(/^([\w.]+):(\d+)-(\d+)$/);if(!m)throw new Error(`span ${s}: use field:start-end`);return {field:m[1],start:+m[2],end:+m[3]}}):[],
paths:v[9]?v[9].split(",").map(p=>p.trim()):[]}});decide({decision:"replace",replacement:rows,note:$("rnote").value})}catch(e){say(e.message,true)}}
$("confirm").onclick=()=>decide({decision:"confirm"});$("skip").onclick=skip;$("rep").onclick=openReplace;$("addrow").onclick=()=>addRow({id:"",subject:"",property:"",value:true,event:"",time:0,indication:INDICATIONS[0],status:"current",spans:[],paths:[]});
$("withdraw").onclick=withdraw;$("sendrep").onclick=sendReplace;$("cancelrep").onclick=()=>$("replace").hidden=true;$("prev").onclick=()=>go(at-1);$("next").onclick=()=>go(at+1);$("todo").onclick=()=>S.progress.next!==null&&go(S.progress.next);
document.addEventListener("keydown",e=>{if(e.ctrlKey||e.metaKey||e.altKey||/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;
const k={c:()=>!$("confirm").disabled&&decide({decision:"confirm"}),s:()=>!$("skip").disabled&&skip(),r:()=>!$("rep").disabled&&openReplace(),n:()=>go(at+1),p:()=>go(at-1),u:()=>S.progress.next!==null&&go(S.progress.next),w:()=>!$("withdraw").hidden&&withdraw()}[e.key];if(k){e.preventDefault();k()}});
(async()=>{try{S=await api("/api/state");const h=parseInt(location.hash.slice(1));await go(Number.isInteger(h)?h:S.progress.next??0)}catch(e){say(e.message,true)}})();
</script></body></html>
"""


if __name__ == "__main__":
    main()
