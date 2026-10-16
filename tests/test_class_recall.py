import pytest

from semantic_switch import ConfigError
from semantic_switch.tools.deployment import class_recall, declaration


def row(gold, answer, accepted, family):
    return {'targets': {'ready': {'gold': gold, 'correct': answer == gold, 'accepted': accepted, 'family': family}}}


def test_operational_recall_counts_withholding_as_a_miss_and_absent_references_stay_unassessed():
    declared = declaration(margin=0, recall={'ready': {'true': {'raw': .5, 'operational': .5}, 'unstated': {'raw': .1}}})
    rows = [row('true', 'true', False, 'a'), row('true', 'true', False, 'b'), row('false', 'false', True, 'c')]
    found = class_recall(rows, declared['recall'])['ready']
    assert found['true']['references'] == 2 and found['true']['families'] == 2
    assert found['true']['raw']['state'] == 'met' and found['true']['operational'] == {'correct': 0, 'recall': 0., 'required': .5, 'state': 'unmet'}
    assert found['unstated']['raw']['state'] == 'unassessed' and found['unstated']['raw']['recall'] is None
    assert 'recall' not in declaration(margin=0)
    with pytest.raises(ConfigError):
        declaration(margin=0, recall={'ready': {'true': {'precision': .5}}})
