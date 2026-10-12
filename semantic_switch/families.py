from __future__ import annotations

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
        self.parent, self.members, self.aliases = {}, {}, {}

    def __contains__(self, node) -> bool:
        return tuple(node) in self.parent

    def add(self, node, *aliases) -> tuple:
        node = seed(node)
        if node not in self.parent:
            self.parent[node], self.members[node] = node, [node]
        root = self.find(node)
        if aliases:
            self.aliases.setdefault(root, set()).update(aliases)
        return root

    def find(self, node) -> tuple:
        node = tuple(node)
        if node not in self.parent:
            return self.add(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def join(self, a, b) -> tuple:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        self.members[lo] += self.members.pop(hi)
        if hi in self.aliases:
            self.aliases.setdefault(lo, set()).update(self.aliases.pop(hi))
        return lo

    def seeds(self, node) -> tuple:
        return tuple(sorted(self.members[self.find(node)]))

    def roots(self) -> list:
        return sorted(self.members)

    def name(self, node) -> str:
        root = self.find(node)
        return min(self.aliases[root]) if self.aliases.get(root) else f'{root[0]}:{root[1]}'

    def history(self, node) -> tuple:
        return tuple(sorted(self.aliases.get(self.find(node), ())))
