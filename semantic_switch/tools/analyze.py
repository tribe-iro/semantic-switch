from __future__ import annotations

from dataclasses import replace
import time

from ..artifacts import load_bundle, snapshot_digest
from ..errors import ConfigError
from ..judgment import OFF
from .data import Labels
from .lock import out_of_fold
from .policy import fit_policy
from .dataset import EvidenceContext, evaluation_plan


def analyze_head(rt, path, fact_name, *, head_bundle=None):
    if head_bundle is None:
        raise ConfigError('analyze-head analyzes a supplied head: use --head BUNDLE; fit with semantic-trainer lexical EXPORT --to BUNDLE')
    sw = rt.load(path, draft=True)
    fact, labels = sw.plan.fact(fact_name), Labels(sw)
    bundle = load_bundle(head_bundle, fact, labels.training_id)
    tune = labels.examples(fact_name, 'tune')
    observations = out_of_fold(fact, tune, bundle, sw, sw.snapshot)
    policy = fit_policy(fact, observations, OFF)
    snap = replace(sw.snapshot, heads={fact_name: bundle.head}, policies={'classifier': {fact_name: policy}})
    held = labels.examples(fact_name, 'check', taught=False)
    candidates = {'judge': (sw.snapshot, ('judge',)), 'lexical': (snap, ('classifier', 'judge'))}
    identities = {name: snapshot_digest(candidate, rt) for name, (candidate, _) in candidates.items()}
    audit_use = EvidenceContext.of(labels).consult(evaluation_plan(identities, scope=[list(fact.path)]), {e.group for e in held})
    arms = {}
    for name, (candidate, sources) in candidates.items():
        candidate_id = identities[name]
        started = time.perf_counter()
        rows = sw.score([e.data for e in held], [fact_name], sources=sources, snap=candidate)
        targets = [row.targets[fact_name] for row in rows]
        own = [None if row.error is not None else target.accepted for row, target in zip(rows, targets)]
        arms[name] = {'settled': sum(v is not None for v in own),
                      'correct': sum(v == e.label for v, e in zip(own, held) if v is not None),
                      'wrong': sum(v != e.label for v, e in zip(own, held) if v is not None),
                      'ms': (time.perf_counter() - started) * 1000 / len(held) if held else None,
                      'judge_calls_avoided': sum(target.source == 'classifier' for target in targets),
                      'candidate_id': candidate_id, 'audit_use': audit_use}
    skipped = [{'id': o.id, 'group': o.group, 'reason': o.skip} for o in observations if o.skip]
    return {'fact': fact_name, 'bundle': bundle.identity.sha256, 'dataset_id': labels.dataset_id,
            'labelled': len(held), 'evaluated': len(held), 'skipped': skipped,
            'development': {'submitted': len(observations), 'evaluated': len(observations) - len(skipped), 'skipped': skipped},
            'population': 'identified check sample; comparison does not admit an artifact', **arms,
            'report': [f'{fact_name}: supplied lexical bundle; {len(held)} check observations.',
                       f'{len(skipped)} unusable development folds; no fitting in semantic-switch.']}
