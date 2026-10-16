import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from bench import evidence, procedures
from semantic_switch import files
from semantic_switch.compiler import compile_spec, digest
from semantic_switch.examples import Frame as InputFrame
from semantic_switch.families import exact_families


root = Path(__file__).resolve().parent
if (root / 'cohorts.json').exists():
    print((root / 'cohorts.json').read_text())
    raise SystemExit(0)
parent = evidence.read('packs/zt411')
_, old, _ = evidence.frames_of(parent, 'packs/zt411')
spec = procedures.definition('structured')
plan = compile_spec(spec)
facts = [fact for fact in plan.facts if not fact.cards and not fact.hidden]
known = {digest(frame.input) for frame in old}
projections = {fact.name: {InputFrame(plan, frame.input).read(fact) for frame in old if frame.condition == 'structured'} for fact in facts}
planned = procedures.scenarios(20261015, 80, None)
built = procedures.frames(planned, conditions=('structured',))
built = [replace(frame, tags=tuple(sorted({*frame.tags, *procedures.behavior_tags(frame), 'coverage_experiment'}))) for frame in built]
ref = evidence.Reference(procedures.reference, procedures.REVISION, procedures.SCOPE, procedures.FIELDS,
                         {'structured': evidence.criterion(spec)})
golds = evidence.references(ref, built)
groups = defaultdict(list)
for frame in built:
    groups[frame.family].append(frame)
blocked = {family for family, members in groups.items() if any(frame.public in known or any(
    InputFrame(plan, frame.input).read(fact) in projections[fact.name] for fact in facts) for frame in members)}
groups = {family: members for family, members in groups.items() if family not in blocked}
counted = evidence.situation_frames(built, golds, 'structured')
sizes = {family: sum(frame.id in counted for frame in members) for family, members in groups.items()}
groups = {family: members for family, members in groups.items() if sizes[family]}
cells = [
    {'name': 'readiness_detection', 'target': 'ready_to_print', 'class': True, 'tags': ['detection']},
    {'name': 'readiness_after_calibration', 'target': 'ready_to_print', 'class': True, 'tags': ['cancelled']},
    {'name': 'mismatch_during_calibration', 'target': 'method_inconsistent', 'class': True, 'tags': ['calibration_step']},
    {'name': 'missing_ribbon', 'target': 'ribbon_loaded', 'class': 'unstated', 'tags': ['ribbon_out']},
]


def membership(frame, cell):
    return golds[frame.id].primitives[cell['target']].value == cell['class'] and set(cell['tags']) <= set(frame.tags)


memberships = {family: {cell['name'] for cell in cells if any(membership(frame, cell) for frame in members)}
               for family, members in groups.items()}
targeted = set()
for cell in sorted(cells, key=lambda cell: sum(cell['name'] in found for found in memberships.values())):
    available = sorted((family for family, found in memberships.items() if cell['name'] in found), key=lambda family: digest([431, family]))
    have = sum(cell['name'] in memberships[family] for family in targeted)
    for family in available:
        if have >= 4:
            break
        if family not in targeted:
            targeted.add(family)
            have += 1
    if sum(cell['name'] in memberships[family] for family in targeted) < 4:
        raise ValueError(f'insufficient independent candidate families for {cell["name"]}')
histogram = Counter(sizes[family] for family in targeted)
rng = random.Random(431)
random_control = set()
for size, count in sorted(histogram.items()):
    random_control.update(rng.sample(sorted(family for family in groups if sizes[family] == size), count))
report = {'corpus_sha256': parent['manifest_sha256'], 'definition_sha256': files.sha256_file(procedures.DEFINITION),
          'seed': 20261015, 'candidate_families': len(groups), 'excluded_overlap_families': len(blocked), 'cells': cells,
          'scope': 'Synthetic structured-input development experiment. No new wording or independent confirmation claim.', 'arms': {}}
for name, selected in [('random', random_control), ('targeted', targeted)]:
    frames = [frame for family in sorted(selected) for frame in groups[family]]
    assignment = {'seed': 431, 'quotas': {'fit': sum(sizes[family] for family in selected)}, 'families': dict.fromkeys(selected, 'fit'),
                  'components': {frame.id: frame.family for frame in frames}, 'counted': sorted(counted & {frame.id for frame in frames}),
                  'counts': {role: {} for role in ('fit', 'model_select', 'check')}}
    pack = replace(evidence.load_pack(procedures.PACK), created=files.now())
    reviewed = evidence.review(pack, frames, (), ref, authored=procedures.authored())
    destination = root / name
    evidence.publish(destination, pack, reviewed, ref, scenarios=[row for row in planned if row['scenario'] in selected],
                     definition=spec, name='definition.yaml', condition='structured', assignment=assignment,
                     producer='bench.procedures.reference', producer_digest=evidence.code_digest(procedures, evidence), seed=431,
                     extra={'experiment': {'parent': parent['manifest_sha256'], 'arm': name, 'cells': cells}})
    published = evidence.read(destination)
    report['arms'][name] = {'pack': str(destination), 'sha256': published['manifest_sha256'], 'families': len(selected),
                            'family_size_histogram': dict(histogram), 'fit_situations': sum(sizes[family] for family in selected),
                            'labels': len(published['labels']), 'coverage': {cell['name']: sum(cell['name'] in memberships[family] for family in selected) for cell in cells},
                            'overlap_between_arms': len(targeted & random_control)}
files.publish(root / 'cohorts.json', report)
print(json.dumps(report, indent=2))
