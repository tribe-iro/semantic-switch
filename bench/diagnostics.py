from __future__ import annotations

import json
import time

import numpy as np

from semantic_switch.judge import scored
from semantic_switch.protocols import Question
from bench.outcomes import latency, measure, provenance


def timing(rt, items, repeats=10):
    judge = rt.judge()
    scorer = getattr(judge, "scorer", None)
    if scorer is None:
        raise ValueError("native timing requires a local judge")
    context = scorer.ctx
    samples, totals = [], []
    with scorer.operation():
        original_decode = context.decode

        def decode(spans, on_output=None):
            spans = list(spans)
            acquired = []

            def output(key, j):
                begin = time.perf_counter()
                context.L.llama_synchronize(context.ptr)
                completed = time.perf_counter()
                context.logits(j)
                logits = time.perf_counter()
                on_output(key, j)
                reduced = time.perf_counter()
                acquired.append({"completion_ms": (completed - begin) * 1000, "logits_ms": (logits - completed) * 1000,
                                 "reduction_ms": (reduced - logits) * 1000})
            before, begin = context.submissions, time.perf_counter()
            original_decode(spans, output if on_output else None)
            if not acquired:
                enqueued = time.perf_counter()
                context.L.llama_synchronize(context.ptr)
                acquired.append({"completion_ms": (time.perf_counter() - enqueued) * 1000, "logits_ms": 0., "reduction_ms": 0.})
            elapsed = (time.perf_counter() - begin) * 1000
            samples.append({"tokens": sum(len(tokens) for tokens, *_ in spans), "outputs": sum(span[3] is not None for span in spans),
                            "decodes": context.submissions - before,
                            "enqueue_ms": max(0., elapsed - sum(sum(row.values()) for row in acquired)),
                            "outputs_timing": acquired, "total_ms": elapsed})
        try:
            context.decode = decode
            begin = time.perf_counter()
            requests = [judge.adapter.request(judge.adapter.state(state), qs) for state, qs in items]
            preparation = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            block = scorer.score(requests)
            instrumented = (time.perf_counter() - begin) * 1000
        finally:
            del context.decode
        for _ in range(repeats):
            begin = time.perf_counter()
            scored(judge, items)
            totals.append((time.perf_counter() - begin) * 1000)
    return {**provenance(rt, items, boundary="explicit backend diagnostic"), "preparation_ms": preparation, "instrumented_ms": instrumented,
            "submissions": samples, "failures": block.error, "samples": repeats,
            "uninstrumented": latency(totals)}


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


def vectors(rt, items):
    judge = rt.judge()
    scorer, vectors_ = judge.scorer, []
    with scorer.operation():
        context = scorer.ctx
        original = scorer._emit
        context.L.llama_set_embeddings(context.ptr, True)
        dimension = context.L.llama_model_n_embd(scorer.weights.ptr)

        def emit(block, done):
            consume = original(block, done)

            def output(key, j):
                pointer = context.L.llama_get_embeddings_ith(context.ptr, j)
                if not pointer:
                    raise RuntimeError("native embedding output is unavailable")
                vectors_.append((key[0], np.ctypeslib.as_array(pointer, shape=(dimension,)).copy()))
                consume(key, j)
            return output
        try:
            scorer._emit = emit
            results = scored(judge, items)
        finally:
            del scorer._emit
            context.L.llama_set_embeddings(context.ptr, False)
            scorer.invalidate()
    return {"representation": "answer-position", "dimension": dimension, "dtype": "float32",
            "identity": rt.identity(), "vectors": [vector for _, vector in sorted(vectors_, key=lambda item: item[0])],
            "results": results}


def activated_suffix(rt, items, *, adapter_path=None, manifest=None):
    from semantic_switch.api import behavior_identity
    from semantic_switch.backend import Lora
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
                adapter = Lora(scorer.weights, str(adapter_path), "experiment")
            row = 0
            for request in requests:
                scorer._ensure_prefix(request.prefix)
                sequence = scorer._ensure_state(request.prefix, request.state)
                for suffix, slots in zip(request.suffixes, request.slots):
                    branch = scorer.layout.branch
                    context.cp(sequence, branch)
                    try:
                        scorer._activate(adapter)
                        context.decode(((suffix, len(request.prefix) + len(request.state), branch, (row, slots)),), scorer._emit(block, set()))
                    finally:
                        context.rm(branch)
                        scorer._activate(None)
                    row += 1
        finally:
            scorer._activate(None)
            if adapter:
                adapter.close()
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
        try:
            for iteration in range(samples):
                baseline = None
                for shape in ('serial', 'finite_batch') if iteration % 2 == 0 else ('finite_batch', 'serial'):
                    scorer.invalidate()
                    before = rt.info()['capabilities']['counters']
                    started = time.perf_counter()
                    results = [switch(value) for value in inputs] if shape == 'serial' else switch.run(list(inputs))
                    rows = [result.as_dict(True) for result in results]
                    json.dumps(rows, allow_nan=False)
                    arm = {'iteration': iteration, 'shape': shape, 'elapsed_ms': (time.perf_counter() - started) * 1000, 'observations': rows}
                    after = rt.info()['capabilities']['counters']
                    arm['native_work'] = {key: after[key] - before.get(key, 0) for key in after}
                    arms.append(arm)
                    if baseline is None:
                        baseline = arm
                        continue
                    for index, (a, b) in enumerate(zip(baseline['observations'], rows)):
                        facts = set(a['facts']) | set(b['facts'])
                        differences.append({'iteration': iteration, 'input': index, 'baseline': {'shape': baseline['shape']}, 'arm': {'shape': shape},
                            'status_changed': a['status'] != b['status'],
                            'rule_changed': a['rule'] != b['rule'], 'action_changed': a['answer'] != b['answer'],
                            'fact_outcomes_changed': [name for name in sorted(facts)
                                if tuple(a['facts'].get(name, {}).get(k) for k in ('status', 'value')) !=
                                   tuple(b['facts'].get(name, {}).get(k) for k in ('status', 'value'))],
                            'scores_changed': [name for name in sorted(facts)
                                if a['facts'].get(name, {}).get('scores') != b['facts'].get(name, {}).get('scores')]})
        finally:
            scorer.invalidate()
    if switch.snapshot is not snapshot:
        raise RuntimeError('native replay changed the frozen snapshot')
    return {'schema': 1, 'candidate_id': snapshot.candidate_id, 'runtime': rt.info(),
            'boundary': 'actual frozen per-fact policies; complete SDK and JSON materialization',
            'policies': {source: {name: {'true': p.t_true, 'false': p.t_false} for name, p in table.items()}
                         for source, table in snapshot.policies.items()}, 'arms': arms, 'differences': differences}


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
                    median = latency(fit)['p50']
                    buckets.append({'cache': condition, 'branches': count, 'suffix_tokens': len(suffix),
                                    'total_suffix_tokens': count * len(suffix), 'samples_ms': samples,
                                    'timing': latency(samples),
                                    'held_error_ms': abs(held - median), 'submissions': submissions, 'failures': failures})
    fingerprint = digest(rt.cache_key())
    return {'schema': 1, 'fingerprint': fingerprint, 'buckets': buckets,
            'fit': 'measured bucket medians; no out-of-range extrapolation',
            'provenance': provenance(rt, boundary="explicit backend diagnostic"), 'production_installed': False}


class Session:
    def __init__(self, rt):
        self.rt, self.scorer, self.order = rt, rt.judge().scorer, []
        score = self.scorer.score

        def recorded(requests, **kwargs):
            self.order.append([(len(r.prefix), len(r.state), [len(s) for s in r.suffixes]) for r in requests])
            return score(requests, **kwargs)
        self.scorer.score = recorded

    def residency(self):
        scorer = self.scorer
        return {"current_prefix_tokens": len(scorer._prefix or ()), "resident_prefix_tokens": sorted(map(len, scorer._resident)),
                "state_tokens": [len(scorer._state)] if scorer._state is not None else []}

    def cold(self):
        self.scorer.invalidate()

    def occupy(self, tokens):
        with self.scorer.operation():
            self.scorer.ctx.decode(((tuple(tokens), 0, self.scorer.ctx.n_seq_max - 1, None),))

    def ask(self, text, specs):
        before, self.order = self.residency(), []
        measured = measure(lambda: self.rt.ask(text, specs), runtime=self.rt)
        return measured, {"residency_before": before, "windows": [w for call in self.order for w in call]}


def native_build():
    from semantic_switch.api import _build
    build = _build()
    return None if build == "none" else json.loads(build)


def lora_support() -> dict:
    from semantic_switch.backend import _llama, lora_abi
    try:
        return {"status": "supported", **lora_abi(_llama())}
    except Exception as error:
        return {"status": "unsupported", "error": str(error)}


def row_scores(block, row, width):
    from semantic_switch.judge import _scores
    return _scores(block, row, width)


def switch_adapter(scorer, lora) -> tuple[bool, float]:
    switched = lora is not scorer._active
    started = time.perf_counter()
    scorer._activate(lora)
    return switched, (time.perf_counter() - started) * 1000


def grouped_requests(rt, requests):
    from semantic_switch.protocols import ScoreBlock
    judge = rt.judge()
    with judge.scorer._bound() as (scorer, binding):
        context = scorer.ctx
        if any(len(request.prefix) + len(request.state) + len(suffix) > context.sequence_limit for request in requests for suffix in request.suffixes):
            raise ValueError('grouped diagnostic request exceeds sequence capacity')
        block = ScoreBlock.empty(sum(len(request.suffixes) for request in requests), max(len(slots) for request in requests for slots in request.slots))
        previous = context.outputs_per_decode
        try:
            scorer.invalidate()
            scorer._bind(binding.get('lora'), binding.get('binding'))
            branches = list(range(2, context.n_seq_max))
            context.outputs_per_decode = min(len(branches), context.n_outputs_max)
            row = 0
            for request in requests:
                scorer.invalidate()
                scorer._ensure_prefix(request.prefix)
                state = scorer._ensure_state(request.prefix, request.state)
                for start in range(0, len(request.suffixes), len(branches)):
                    spans = []
                    for index, branch in zip(range(start, len(request.suffixes)), branches):
                        context.cp(state, branch)
                        spans.append((request.suffixes[index], len(request.prefix) + len(request.state), branch, (row + index, request.slots[index])))
                    try:
                        context.decode(spans, scorer._emit(block, set()))
                    finally:
                        for branch in branches:
                            context.rm(branch)
                row += len(request.suffixes)
            return block
        finally:
            context.outputs_per_decode = previous
            scorer.invalidate()
