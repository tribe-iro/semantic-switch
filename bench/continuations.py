from __future__ import annotations

import numpy as np


def joint_log_probability(logits, tokens):
    if len(logits) != len(tokens) or not tokens:
        raise ValueError('one conditional distribution is required per continuation token')
    total = 0.
    for row, token in zip(logits, tokens):
        row = np.asarray(row, dtype=np.float64)
        maximum = row.max()
        total += row[token] - maximum - np.log(np.exp(row - maximum).sum())
    return float(total)


def numeric(judge, state, question):
    adapter, scorer = judge.adapter, judge.scorer
    options = ('true', 'false') if question.kind == 'check' else question.options
    names = [f'[{i + 1}]' for i in range(len(options))]
    text = '\n'.join([f'Question: {question.text}', *[f'{name}) {option}' for name, option in zip(names, options)],
                      *[f'Except: {value}.' for value in question.excludes],
                      'Answer with only the bracketed identifier of the best option.'])
    prefix = adapter.prefix_for(question.guidance, question.framing)
    framing = question.framing or adapter.framing
    tokens = prefix + adapter.state(state) + adapter.content(text) + adapter.template(framing.tail_text)
    candidates = [adapter.content(name) for name in names]
    likelihoods, submissions = [], []
    with scorer.operation():
        context = scorer.ctx
        context.clear()
        scorer._prefix = None
        scorer._states = {}
        scorer._resident = {}
        try:
            for candidate in candidates:
                context.clear()
                rows = []
                sequence = tokens + candidate[:-1]
                before = context.submissions
                first = len(tokens) - 1
                context.run([(token, position, 1, position - first if position >= first else None)
                             for position, token in enumerate(sequence)],
                            lambda row, j, sequence: rows.append(context.logits(j).copy()))
                likelihoods.append(joint_log_probability(rows, candidate))
                submissions.append(context.submissions - before)
        finally:
            context.clear()
    values = np.asarray(likelihoods)
    maximum = values.max()
    log_support = maximum + np.log(np.exp(values - maximum).sum())
    probs = np.exp(values - log_support)
    return {'probs': probs.tolist(), 'joint_logp': likelihoods, 'continuations': [list(value) for value in candidates],
            'valid_answer_support': float(np.exp(log_support)), 'teacher_forced_submissions': submissions,
            'authored_indices': list(range(len(options)))}
