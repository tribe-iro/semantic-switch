from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from semantic_switch import ConfigError
from semantic_switch.training import BUNDLE_SCHEMA, FEATURE_IDENTITY, N_FEATURES, LexicalHead, admit, features, publish_bundle, read_export

from . import __version__


def matrix(texts):
    indptr, indices, values = [0], [], []
    for text in texts:
        found = features(text)
        indices.append(np.fromiter(found.keys(), dtype=np.int32, count=len(found)))
        values.append(np.fromiter(found.values(), dtype=np.float32, count=len(found)))
        indptr.append(indptr[-1] + len(found))
    empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))
    rows = csr_matrix((np.concatenate(values or [empty[1]]), np.concatenate(indices or [empty[0]]), np.asarray(indptr, dtype=np.int64)),
                      shape=(len(texts), N_FEATURES), dtype=np.float32)
    rows.sort_indices()
    return rows


def compact(rows):
    active = np.unique(rows.indices).astype(np.int32)
    return csr_matrix((rows.data, np.searchsorted(active, rows.indices).astype(np.int32), rows.indptr), shape=(rows.shape[0], len(active))), active


def fit(x, active, y, classes, *, c=10.0, seed=0, max_iter=5000, tolerance=1e-4):
    observed = sorted(set(y))
    if len(observed) < 2:
        raise ConfigError('lexical fit requires at least two observed classes')
    clf = LogisticRegression(C=c, random_state=seed, max_iter=max_iter, tol=tolerance, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        try:
            clf.fit(x, y)
        except ConvergenceWarning:
            raise ConfigError('lexical fit did not converge; no bundle published') from None
    w = np.zeros((len(active), len(classes)), dtype=np.float32)
    b = np.full(len(classes), -30., dtype=np.float32)
    if len(observed) == 2:
        lo, hi = clf.classes_
        w[:, hi], b[lo], b[hi] = clf.coef_[0], 0., clf.intercept_[0]
    else:
        for row, cls in enumerate(clf.classes_):
            w[:, cls], b[cls] = clf.coef_[row], clf.intercept_[row]
    kept = np.flatnonzero(np.abs(w).sum(axis=1) > 0)
    return LexicalHead(tuple(classes), active[kept], w[kept], b)


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
    admit((f"observation {r['id']} ({fact}, fold {r['fold']})", r['label_sources']) for r in rows)
    params = dict(c=c, seed=seed, max_iter=max_iter, tolerance=tolerance)
    x, active = compact(matrix([r['text'] for r in rows]))
    labels = np.asarray([r['label'] for r in rows])
    final = fit(x, active, labels, classes, **params)
    heads, folds = {'final.head.json': final}, {}
    folds_of = np.asarray([r['fold'] for r in rows])
    for number in range(5):
        kept, held = np.flatnonzero(folds_of != number), np.flatnonzero(folds_of == number)
        entry = {'fit_ids': sorted(rows[i]['id'] for i in kept), 'held_ids': sorted(rows[i]['id'] for i in held)}
        observed = set(labels[kept].tolist())
        if not len(held) or len(observed) < 2:
            entry['skip'] = 'empty held fold' if not len(held) else 'training fold has fewer than two classes'
        else:
            entry['head'] = f'fold-{number}.head.json'
            heads[entry['head']] = fit(x[kept], active, labels[kept], classes, **params)
            entry['unobserved_classes'] = [classes[i] for i in range(len(classes)) if i not in observed]
        folds[str(number)] = entry
    manifest = {'schema': BUNDLE_SCHEMA, 'kind': 'lexical-bundle', 'feature_identity': FEATURE_IDENTITY,
                'fact': fact, 'meaning_rev': revision, 'classes': classes, 'dataset_id': source['dataset_id'],
                'training_id': source['training_id'], 'export_id': source['export_id'], 'trainer': {'name': 'semantic-trainer', 'version': __version__, 'solver': 'lbfgs', **params},
                'final': {'head': 'final.head.json'}, 'folds': folds,
                'unobserved_classes': [classes[i] for i in range(len(classes)) if i not in {r['label'] for r in rows}],
                'rows': [{k: r[k] for k in ('id', 'fact', 'projection', 'group', 'fold', 'family_seeds')} for r in rows]}
    return {'path': str(to), 'id': publish_bundle(to, manifest, heads), 'rows': len(rows), 'folds': len(folds)}
