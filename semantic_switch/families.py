from __future__ import annotations

from array import array

from .errors import ConfigError

SEEDS = ('group', 'key', 'observation', 'pair', 'rendered', 'situation')


def seed(node) -> tuple:
    if not isinstance(node, (list, tuple)) or len(node) != 2 or node[0] not in SEEDS or not isinstance(node[1], str) or not node[1]:
        raise ConfigError(f'family seed must be [kind, name] with kind in {SEEDS} and a nonempty name, not {node!r}')
    return tuple(node)


def valid_seeds(value) -> bool:
    try:
        return isinstance(value, list) and all(seed(node) for node in value)
    except ConfigError:
        return False


class FamilyIndex:
    def __init__(self):
        self._parent, self._members, self._groups, self._aliases, self._canonical = {}, {}, {}, {}, {}

    def __contains__(self, node) -> bool:
        return tuple(node) in self._parent

    def __len__(self) -> int:
        return len(self._parent)

    def add(self, node, *aliases) -> tuple:
        node = seed(node)
        if node not in self._parent:
            self._parent[node], self._members[node], self._canonical[node] = node, [node], node
            self._groups[node] = {node[1]} if node[0] == 'group' else set()
        root = self._root(node)
        if aliases:
            self._aliases.setdefault(root, set()).update(aliases)
        return self._canonical[root]

    def _root(self, node) -> tuple:
        node = tuple(node)
        if node not in self._parent:
            raise ConfigError(f'{node!r} is not a family seed of this index')
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def find(self, node) -> tuple:
        return self._canonical[self._root(node)]

    def join(self, a, b) -> tuple:
        self.add(a), self.add(b)
        ra, rb = self._root(a), self._root(b)
        if ra == rb:
            return self._canonical[ra]
        small, large = (ra, rb) if len(self._members[ra]) < len(self._members[rb]) else (rb, ra)
        self._parent[small] = large
        self._members[large].extend(self._members.pop(small))
        self._groups[large] |= self._groups.pop(small)
        self._canonical[large] = min(self._canonical[large], self._canonical.pop(small))
        if small in self._aliases:
            self._aliases.setdefault(large, set()).update(self._aliases.pop(small))
        return self._canonical[large]

    def groups(self, node) -> frozenset:
        return frozenset(self._groups[self._root(node)])

    def seeds(self, node) -> tuple:
        return tuple(sorted(self._members[self._root(node)]))

    def roots(self) -> list:
        return sorted(self._canonical[root] for root in self._members)

    def name(self, node) -> str:
        root = self._root(node)
        aliases = self._aliases.get(root)
        return min(aliases) if aliases else '{}:{}'.format(*self._canonical[root])

    def history(self, node) -> tuple:
        return tuple(sorted(self._aliases.get(self._root(node), ())))


def exact_families(sizes, count, seed) -> set:
    from .compiler import digest
    order = sorted(sizes, key=lambda family: digest([seed, 'fit-family', family]))
    limit = (1 << count + 1) - 1
    reached, via, before = 1, array('l', [-1]) * (count + 1), array('l', [-1]) * (count + 1)
    for position, family in enumerate(order):
        size = sizes[family]
        fresh = (reached << size) & limit & ~reached
        reached |= fresh
        while fresh:
            low = fresh & -fresh
            total = low.bit_length() - 1
            via[total], before[total] = position, total - size
            fresh ^= low
    if not reached >> count & 1:
        near = [total for total in range(count, -1, -1) if reached >> total & 1][:5]
        raise ConfigError(f'cannot choose whole fit families totalling exactly {count} situations; family sizes {dict(sorted(sizes.items()))} '
                          f'reach {sorted(near)} near the target; adjust the scenario quotas')
    chosen, total = set(), count
    while total:
        chosen.add(order[via[total]])
        total = before[total]
    return chosen
