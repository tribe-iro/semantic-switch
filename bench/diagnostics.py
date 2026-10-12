"""Explicit backend experiments; these functions are outside normal SDK execution."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from semantic_switch import files
from semantic_switch.judge import scored
from semantic_switch.protocols import Question


CODE_ROOT = Path(__file__).resolve().parents[1]


def fingerprint(root=CODE_ROOT):
    root = Path(root).resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    loaded = sorted({Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module, "__file__", None)} |
                    {root / "pyproject.toml"})
    listing = [[str(path.relative_to(root)), files.sha256_file(path)] for path in loaded if path.is_relative_to(root) and path.is_file()
               and "site-packages" not in path.parts]
    return {"revision": revision, "sources": hashlib.sha256(json.dumps(listing, separators=(",", ":")).encode()).hexdigest()}


def provenance(rt, inputs=(), *, boundary="explicit backend diagnostic"):
    return {"schema": 2, **fingerprint(),
            "dataset": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "execution": rt.info(), "boundary": boundary}


def timing(rt, items, repeats=10):
    judge = rt.judge()
    scorer = getattr(judge, "scorer", None)
    if scorer is None:
        raise ValueError("native timing requires a local judge")
    context = scorer.ctx
    samples, totals = [], []
    with scorer.operation():
        original_run = context.run
        def run(entries, writer):
            for start in range(0, len(entries), context.n_batch):
                chunk = entries[start:start + context.n_batch]
                acquired = []
                def output(row, j, sequence):
                    begin = time.perf_counter()
                    context.L.llama_synchronize(context.ptr)
                    completed = time.perf_counter()
                    context.logits(j)
                    logits = time.perf_counter()
                    writer(row, j, sequence)
                    reduced = time.perf_counter()
                    acquired.append({"completion_ms": (completed - begin) * 1000,
                                     "logits_ms": (logits - completed) * 1000,
                                     "reduction_ms": (reduced - logits) * 1000})
                begin = time.perf_counter()
                original_run(chunk, output if writer else None)
                if not acquired:
                    enqueued = time.perf_counter()
                    context.L.llama_synchronize(context.ptr)
                    completed = time.perf_counter()
                    acquired.append({"completion_ms": (completed - enqueued) * 1000,
                                     "logits_ms": 0., "reduction_ms": 0.})
                elapsed = (time.perf_counter() - begin) * 1000
                samples.append({"tokens": len(chunk), "outputs": sum(e[3] is not None for e in chunk),
                                "enqueue_ms": max(0., elapsed - sum(sum(v for k, v in row.items()) for row in acquired)),
                                "outputs_timing": acquired, "total_ms": elapsed})
        try:
            context.run = run
            begin = time.perf_counter()
            requests = [judge.adapter.request(judge.adapter.state(state), qs) for state, qs in items]
            preparation = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            block = scorer.score(requests)
            instrumented = (time.perf_counter() - begin) * 1000
        finally:
            context.run = original_run
        for _ in range(repeats):
            begin = time.perf_counter()
            scored(judge, items)
            totals.append((time.perf_counter() - begin) * 1000)
    return {**provenance(rt, items), "preparation_ms": preparation, "instrumented_ms": instrumented,
            "submissions": samples, "failures": block.error, "samples": repeats,
            "uninstrumented_ms": {"p50": float(np.percentile(totals, 50)), "p95": float(np.percentile(totals, 95))}}


def counterbalanced(judge, state, question):
    if question.kind == "rate":
        raise ValueError("rate keeps its ordered rubric; it is outside this experiment")
    categorical = Question("pick", question.text, ("true", "false"), question.excludes) if question.kind == "check" else question
    reverse = Question("pick", categorical.text, tuple(reversed(categorical.options)), categorical.excludes)
    results = scored(judge, [(state, [categorical, reverse])])[0]
    a, b = results.scores
    if a.error or b.error or a.low_mass or b.low_mass:
        return {"error": a.error or b.error or "low answer mass", "branches": results.scores}
    p = (a.probs + b.probs[::-1]) / 2
    masses = [score.mass for score in (a, b) if score.mass is not None]
    return {"probs": p.tolist(), "mass": min(masses) if masses else None,
            "disagreement": float(np.abs(a.probs - b.probs[::-1]).max()), "tokens": results.tokens}


def cascade(rt, states, items, *, small="judge", precise="precise"):
    from semantic_switch.inline import ask_many
    first = ask_many(lambda: rt.judge(small), states, items)
    residual = [i for i, row in enumerate(first) if any(result.status == "uncertain" for result in row.values())]
    second = ask_many(lambda: rt.judge(precise), [states[i] for i in residual], items)
    merged = [dict(row) for row in first]
    for index, row in zip(residual, second):
        for key, result in row.items():
            if first[index][key].status == "uncertain":
                merged[index][key] = result
    return {"small": first, "precise_residue": second, "residual_indices": residual, "merged": merged,
            "identities": {small: rt.identity(small), precise: rt.identity(precise)}}


def vectors(rt, items):
    judge = rt.judge()
    scorer, vectors_ = judge.scorer, []
    with scorer.operation():
        context = scorer.ctx
        original_writer = scorer._writer
        context.L.llama_set_embeddings(context.ptr, True)
        dimension = context.L.llama_model_n_embd(scorer.weights.ptr)
        def writer(slots, block, done):
            consume = original_writer(slots, block, done)
            def output(row, j, sequence):
                pointer = context.L.llama_get_embeddings_ith(context.ptr, j)
                if not pointer:
                    raise RuntimeError("native embedding output is unavailable")
                vectors_.append((row, np.ctypeslib.as_array(pointer, shape=(dimension,)).copy()))
                consume(row, j, sequence)
            return output
        try:
            scorer._writer = writer
            results = scored(judge, items)
        finally:
            scorer._writer = original_writer
            context.L.llama_set_embeddings(context.ptr, False)
            scorer.invalidate()
    return {"representation": "answer-position", "dimension": dimension, "dtype": "float32",
            "identity": rt.identity(), "vectors": [vector for _, vector in sorted(vectors_, key=lambda item: item[0])],
            "results": results}


def activated_suffix(rt, items, *, adapter_path=None, manifest=None, scale=1.0):
    import ctypes
    from semantic_switch.api import behavior_identity
    from semantic_switch.protocols import ScoreBlock
    judge = rt.judge()
    scorer, context = judge.scorer, judge.scorer.ctx
    if scorer.weights.hybrid:
        raise ValueError("activated suffix cache interchangeability is unproven for recurrent/hybrid models")
    if adapter_path:
        if not isinstance(manifest, dict) or manifest.get("boundary") != "base-input/task-suffix":
            raise ValueError("adapter manifest must bind training to the base-input/task-suffix boundary")
        if manifest.get("base") != behavior_identity(rt.identity()):
            raise ValueError("adapter manifest base/prompt identity differs from this runtime")
        if not manifest.get("family_seeds") or not manifest.get("meaning_revisions"):
            raise ValueError("adapter manifest must record training families and meaning revisions")
    requests = [judge.adapter.request(judge.adapter.state(state), questions) for state, questions in items]
    width = max((len(slots) for request in requests for slots in request.slots), default=1)
    block = ScoreBlock.empty(sum(len(request.slots) for request in requests), width)
    with scorer.operation():
        adapter = None
        try:
            if adapter_path:
                adapter = context.L.llama_adapter_lora_init(scorer.weights.ptr, str(adapter_path).encode())
                if not adapter:
                    raise RuntimeError("cannot load experiment adapter")
            row = 0
            for request in requests:
                scorer._ensure_prefix(request.prefix)
                sequence = scorer._ensure_states(request.prefix, [request.state])[request.state]
                for suffix, slots in zip(request.suffixes, request.slots):
                    branch = 1 + scorer.inputs
                    context.cp(sequence, branch)
                    try:
                        if adapter:
                            pointers = (context.L.llama_adapter_lora_p_ctypes * 1)(adapter)
                            scales = (ctypes.c_float * 1)(scale)
                            if context.L.llama_set_adapters_lora(context.ptr, pointers, 1, scales):
                                raise RuntimeError("cannot activate experiment adapter")
                        entries = [(token, len(request.prefix) + len(request.state) + k, branch,
                                    row if k == len(suffix) - 1 else None) for k, token in enumerate(suffix)]
                        context.run(entries, scorer._writer({row: slots}, block, set()))
                    finally:
                        context.rm(branch)
                        if adapter:
                            context.L.llama_set_adapters_lora(context.ptr, None, 0, None)
                    row += 1
        finally:
            context.L.llama_set_adapters_lora(context.ptr, None, 0, None)
            if adapter:
                context.L.llama_adapter_lora_free(adapter)
    return {"logp": block.logp.tolist(), "log_mass": block.log_mass.tolist(), "errors": block.error,
            "adapter": adapter_path, "boundary": "base-input/task-suffix", "identity": rt.identity()}


def frozen_replay(rt, switch, inputs, samples=2):
    from semantic_switch import ConfigError
    if not switch.snapshot.locked or not switch.snapshot.candidate_id:
        raise ConfigError('native replay requires a frozen candidate')
    scorer = rt.judge().scorer
    snapshot = switch.snapshot
    arms, differences = [], []
    with scorer.operation():
        original = scorer.fused
        try:
            for iteration in range(samples):
                baseline = None
                for fused in ((True, False) if iteration % 2 == 0 else (False, True)):
                    for shape in ('serial', 'finite_batch'):
                        scorer.fused = fused
                        scorer.invalidate()
                        before = rt.info()['capabilities']['counters']
                        started = time.perf_counter()
                        results = [switch(value) for value in inputs] if shape == 'serial' else switch.run(list(inputs))
                        rows = [result.as_dict(True) for result in results]
                        json.dumps(rows, allow_nan=False)
                        arm = {'iteration': iteration, 'fused': fused, 'shape': shape,
                               'elapsed_ms': (time.perf_counter() - started) * 1000, 'observations': rows}
                        after = rt.info()['capabilities']['counters']
                        arm['native_work'] = {key: after[key] - before.get(key, 0) for key in after}
                        arms.append(arm)
                        if baseline is None:
                            baseline = arm
                            continue
                        for index, (a, b) in enumerate(zip(baseline['observations'], rows)):
                            facts = set(a['facts']) | set(b['facts'])
                            differences.append({'iteration': iteration, 'input': index,
                                'baseline': {'fused': baseline['fused'], 'shape': baseline['shape']},
                                'arm': {'fused': fused, 'shape': shape},
                                'status_changed': a['status'] != b['status'],
                                'case_changed': a['case'] != b['case'], 'action_changed': a['answer'] != b['answer'],
                                'fact_outcomes_changed': [name for name in sorted(facts)
                                    if tuple(a['facts'].get(name, {}).get(k) for k in ('status', 'value')) !=
                                       tuple(b['facts'].get(name, {}).get(k) for k in ('status', 'value'))],
                                'scores_changed': [name for name in sorted(facts)
                                    if a['facts'].get(name, {}).get('scores') != b['facts'].get(name, {}).get('scores')]})
        finally:
            scorer.fused = original
            scorer.invalidate()
    if switch.snapshot is not snapshot:
        raise RuntimeError('native replay changed the frozen snapshot')
    return {'schema': 1, 'candidate_id': snapshot.candidate_id, 'runtime': rt.info(),
            'boundary': 'actual frozen per-fact policies; complete SDK and JSON materialization',
            'policies': {source: {name: {'true': p.t_true, 'false': p.t_false} for name, p in table.items()}
                         for source, table in snapshot.policies.items()}, 'arms': arms, 'differences': differences}


def fusion_comparison(rt, items_factory, samples=10):
    judge, results = rt.judge(), {"fused": [], "separate": []}
    scorer = judge.scorer
    differences, outcomes, native_work = [], [], {"fused": [], "separate": []}
    from semantic_switch.judgment import Judgment, Policy, accept
    with scorer.operation():
        original = scorer.fused
        try:
            for iteration in range(samples):
                items = items_factory(iteration)
                predictions = {}
                for mode in ("fused", "separate") if iteration % 2 == 0 else ("separate", "fused"):
                    scorer.fused = mode == "fused"
                    scorer.invalidate()
                    request = judge.adapter.request(judge.adapter.state(items[0][0]), items[0][1])
                    scorer._ensure_prefix(request.prefix)
                    begin = time.perf_counter()
                    output = scored(judge, items)
                    elapsed = (time.perf_counter() - begin) * 1000
                    results[mode].append(elapsed)
                    predictions[mode] = [score for result in output for score in result.scores]
                    native_work[mode].append(dict(output[0].native_work) if output else {})
                differences.append(max(float(np.abs(a.probs - b.probs).max()) for a, b in
                                       zip(predictions["fused"], predictions["separate"]) if a.probs is not None and b.probs is not None))
                questions = [question for _, requested in items for question in requested]
                for question, fused, separate in zip(questions, predictions["fused"], predictions["separate"]):
                    judgment = Judgment.check(question.text) if question.kind == "check" else Judgment(
                        question.kind, question.text, question.options, question.options)
                    boundary = max(fused.probs) if fused.probs is not None else .9
                    thresholds = sorted({.9, .95, float(boundary), float(np.nextafter(boundary, 0.)), float(np.nextafter(boundary, 1.))})
                    for threshold in thresholds:
                        if not .5 < threshold <= 1:
                            continue
                        policy = Policy(threshold, threshold)
                        def outcome(score):
                            return (None, score.error) if score.error else accept(judgment, score.probs, policy, score.mass)
                        a, b = outcome(fused), outcome(separate)
                        outcomes.append({"iteration": iteration, "kind": question.kind, "threshold": threshold,
                                         "fused": a, "separate": b, "same_decision": a == b,
                                         "fused_mass": fused.mass, "separate_mass": separate.mass})
        finally:
            scorer.fused = original
            scorer.invalidate()
    return {"samples": samples, "max_probability_difference": max(differences),
            "policy_outcomes": outcomes, "native_work": native_work,
            "boundary": "shared warmed prefix; complete scoring and materialization; threshold-neighbor probes are descriptive",
            "arms": {name: {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}
                     for name, values in results.items()}}


def cost_curve(rt, *, repeats=3, lengths=(32, 127, 511, 513, 1024), branches=(1, 8, 32)):
    from semantic_switch.compiler import digest
    from semantic_switch.protocols import ScoreRequest
    from semantic_switch.judge import CONFORMANCE_PROBES, CONFORMANCE_QUESTIONS
    judge = rt.judge()
    scorer = judge.scorer
    reference = judge.adapter.request(judge.adapter.state(CONFORMANCE_PROBES[0]), CONFORMANCE_QUESTIONS[:1])
    pad = scorer.tokenize(' evidence')
    if not pad:
        raise ValueError('cost diagnostic requires a padding token')
    buckets = []
    with scorer.operation():
        for condition in ('new', 'cached'):
            for count in branches:
                for length in lengths:
                    suffix = (pad[0],) * max(0, length - len(reference.suffixes[0])) + reference.suffixes[0]
                    request = ScoreRequest(reference.prefix, reference.state, (suffix,) * count, reference.slots * count)
                    samples, submissions, failures = [], [], 0
                    for _ in range(repeats):
                        scorer.invalidate()
                        if condition == 'cached':
                            scorer.score([reference])
                        before = scorer.ctx.submissions
                        started = time.perf_counter()
                        block = scorer.score([request])
                        scorer.ctx.L.llama_synchronize(scorer.ctx.ptr)
                        elapsed = (time.perf_counter() - started) * 1000
                        if len(samples) < repeats:
                            samples.append(elapsed)
                            submissions.append(scorer.ctx.submissions - before)
                            failures += sum(error is not None for error in block.error)
                    fit = samples[:-1] if len(samples) > 1 else samples
                    held = samples[-1]
                    median = float(np.median(fit))
                    buckets.append({'cache': condition, 'branches': count, 'suffix_tokens': len(suffix),
                                    'total_suffix_tokens': count * len(suffix), 'samples_ms': samples,
                                    'p50_ms': median, 'p95_ms': float(np.percentile(samples, 95)),
                                    'held_error_ms': abs(held - median), 'submissions': submissions, 'failures': failures})
    fingerprint = digest(rt.cache_key())
    return {'schema': 1, 'fingerprint': fingerprint, 'buckets': buckets,
            'fit': 'measured bucket medians; no out-of-range extrapolation',
            'provenance': provenance(rt), 'production_installed': False}
