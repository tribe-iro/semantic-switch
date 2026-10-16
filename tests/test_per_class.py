from semantic_switch.training import nested, per_class


def test_one_reducer_counts_declared_classes_with_explicit_denominators():
    rows = [(('t', 'ready'), ('true', 'false', 'unstated'), 'false', 'false', True), (('t', 'ready'), ('true', 'false', 'unstated'), 'true', 'false', False),
            (('t', 'ready'), ('true', 'false', 'unstated'), 'true', None, False), (('u', 'ready'), ('a', 'b'), 'a', 'escape', False)]
    cells = per_class(iter(rows))
    assert cells[('t', 'ready')]['true'] == {'gold': 2, 'predicted': 0, 'correct': 0, 'recall': 0.}
    assert cells[('t', 'ready')]['unstated'] == {'gold': 0, 'predicted': 0, 'correct': 0, 'recall': None}
    assert cells[('t', 'ready')]['false'] == {'gold': 1, 'predicted': 2, 'correct': 1, 'recall': 1.}
    assert cells[('u', 'ready')] == {'a': {'gold': 1, 'predicted': 0, 'correct': 0, 'recall': 0.}, 'b': {'gold': 0, 'predicted': 0, 'correct': 0, 'recall': None}}
    assert set(nested(cells)) == {'t', 'u'} and nested(cells)['t']['ready'] is cells[('t', 'ready')]
