from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from semantic_switch.artifacts import BUNDLE_SCHEMA, read_export, publish_bundle
from semantic_switch.errors import ConfigError
from semantic_switch.heads import FEATURE_IDENTITY, N_FEATURES, LexicalHead, features


def matrix(texts):
    rows, cols, values = [], [], []
    for row, text in enumerate(texts):
        for column, value in features(text).items():
            rows.append(row)
            cols.append(column)
            values.append(value)
    return csr_matrix((values, (rows, cols)), shape=(len(texts), N_FEATURES), dtype=np.float32)


def fit(texts, y, classes, *, c=10.0, seed=0, max_iter=5000, tolerance=1e-4):
    observed = sorted(set(y))
    if len(observed) < 2:
        raise ConfigError('lexical fit requires at least two observed classes')
    clf = LogisticRegression(C=c, random_state=seed, max_iter=max_iter, tol=tolerance, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            clf.fit(matrix(texts), y)
        except ConvergenceWarning:
            raise ConfigError('lexical fit did not converge; no bundle published') from None
    w = np.zeros((N_FEATURES, len(classes)), dtype=np.float32)
    b = np.full(len(classes), -30., dtype=np.float32)
    if len(observed) == 2:
        lo, hi = clf.classes_
        w[:, hi], b[lo], b[hi] = clf.coef_[0], 0., clf.intercept_[0]
    else:
        for row, cls in enumerate(clf.classes_):
            w[:, cls], b[cls] = clf.coef_[row], clf.intercept_[row]
    idx = np.flatnonzero(np.abs(w).sum(axis=1) > 0).astype(np.int32)
    return LexicalHead(tuple(classes), idx, w[idx], b)


def train(export, to, *, fact=None, c=10.0, seed=0, max_iter=5000, tolerance=1e-4):
    if isinstance(c, bool) or isinstance(tolerance, bool) or type(seed) is not int or not np.isfinite(c) or c <= 0 or not np.isfinite(tolerance) or tolerance <= 0 or type(max_iter) is not int or max_iter < 1:
        raise ConfigError('invalid lexical fitting parameters')
    source, rows = read_export(export)
    rows = [r for r in rows if 'member' not in r]
    names = {r['fact'] for r in rows}
    if fact is None:
        if len(names) != 1:
            raise ConfigError('select one --fact from the exported dataset')
        fact = next(iter(names))
    rows = [r for r in rows if r['fact'] == fact]
    if not rows:
        raise ConfigError(f'{fact}: no tune observations')
    classes, revision = rows[0]['classes'], rows[0]['meaning_rev']
    if any(r['classes'] != classes or r['meaning_rev'] != revision for r in rows):
        raise ConfigError('inconsistent meaning/classes')
    params = dict(c=c, seed=seed, max_iter=max_iter, tolerance=tolerance)
    final = fit([r['text'] for r in rows], [r['label'] for r in rows], classes, **params)
    heads, folds = {'final.head.json': final}, {}
    for number in range(5):
        kept = [r for r in rows if r['fold'] != number]
        held = [r for r in rows if r['fold'] == number]
        entry = {'fit_ids': sorted(r['id'] for r in kept), 'held_ids': sorted(r['id'] for r in held)}
        if not held or len({r['label'] for r in kept}) < 2:
            entry['skip'] = 'empty held fold' if not held else 'training fold has fewer than two classes'
        else:
            entry['head'] = f'fold-{number}.head.json'
            heads[entry['head']] = fit([r['text'] for r in kept], [r['label'] for r in kept], classes, **params)
            entry['unobserved_classes'] = [classes[i] for i in range(len(classes)) if i not in {r['label'] for r in kept}]
        folds[str(number)] = entry
    manifest = {'schema': BUNDLE_SCHEMA, 'kind': 'lexical-bundle', 'feature_identity': FEATURE_IDENTITY,
                'fact': fact, 'meaning_rev': revision, 'classes': classes, 'dataset_id': source['dataset_id'],
                'training_id': source['training_id'], 'export_id': source['export_id'], 'trainer': {'name': 'semantic-trainer', 'version': '0.1.0', 'solver': 'lbfgs', **params},
                'final': {'head': 'final.head.json'}, 'folds': folds,
                'unobserved_classes': [classes[i] for i in range(len(classes)) if i not in {r['label'] for r in rows}],
                'rows': [{k: r[k] for k in ('id', 'fact', 'projection', 'group', 'fold', 'family_seeds')} for r in rows]}
    return {'path': str(to), 'id': publish_bundle(to, manifest, heads), 'rows': len(rows), 'folds': len(folds)}
