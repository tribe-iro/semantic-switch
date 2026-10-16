import numpy as np


def _fit(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000))
    model.fit(x, y)
    return model


def probabilities(model, x, classes):
    result = np.zeros((len(x), classes))
    result[:, model.classes_] = model.predict_proba(x)
    return result


def train_probe(vectors_, labels, groups, *, check_groups):
    x = np.asarray(vectors_, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) != len(y) or len(y) != len(groups) or y.dtype.kind not in 'iu' or (y < 0).any():
        raise ValueError('invalid probe vectors, labels or groups')
    tune = np.array([group not in check_groups for group in groups])
    if len(set(y[tune])) < 2 or not (~tune).any():
        raise ValueError('probe requires at least two tune classes and disjoint check families')
    model = _fit(x[tune], y[tune])
    return {'model': model, 'check_indices': np.flatnonzero(~tune).tolist(),
            'probs': probabilities(model, x[~tune], int(y.max()) + 1).tolist(),
            'tune': int(tune.sum()), 'check': int((~tune).sum())}


def grouped_probe(vectors, labels, groups, check_groups, *, folds=None):
    from semantic_switch.training import artifact_digest as digest
    fitted = train_probe(vectors, labels, groups, check_groups=check_groups)
    x, y = np.asarray(vectors, dtype=np.float32), np.asarray(labels)
    tune_indices = [i for i, g in enumerate(groups) if g not in check_groups]
    tune_groups = set(groups) - set(check_groups)
    folds = {g: int(digest(['probe-fold', g])[:8], 16) % 5 for g in tune_groups} if folds is None else {g: folds[g] for g in tune_groups}
    if any(type(fold) is not int or not 0 <= fold < 5 for fold in folds.values()):
        raise ValueError('invalid supplied probe folds')
    classes = int(y.max()) + 1
    oof, skipped = {}, []
    for number in range(5):
        kept = [i for i in tune_indices if folds[groups[i]] != number]
        held = [i for i in tune_indices if folds[groups[i]] == number]
        if not held:
            continue
        if len(set(y[kept])) < classes:
            skipped.extend(held)
            continue
        model = _fit(x[kept], y[kept])
        oof.update(zip(held, probabilities(model, x[held], classes).tolist()))
    sizes = sorted(set(min(len(tune_indices), size) for size in (8, 16, 32, 64, 128, len(tune_indices))))
    curves = []
    for size in sizes:
        selected_groups, indices = set(), []
        for group in sorted(folds):
            selected_groups.add(group)
            indices = [i for i in tune_indices if groups[i] in selected_groups]
            if len(indices) >= size:
                break
        if len(set(y[indices])) < 2:
            curves.append({'rows': len(indices), 'state': 'insufficient classes'})
            continue
        model = _fit(x[indices], y[indices])
        predictions = model.classes_[model.predict_proba(x[fitted['check_indices']]).argmax(axis=1)]
        curves.append({'rows': len(indices), 'families': len(selected_groups),
                       'correct': int((predictions == y[fitted['check_indices']]).sum()), 'check': fitted['check']})
    return {**fitted, 'oof': oof, 'skipped': skipped, 'sample_curves': curves,
            'preprocessing': 'standardization fitted within each supplied tune fold; no check fitting'}
