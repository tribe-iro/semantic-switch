from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from semantic_switch import ConfigError, Runtime
from semantic_switch.training import FamilyIndex, Task, artifact_digest

BUDGET = {'max_situations': 100000, 'max_claims': 1000000, 'max_bytes': 64 << 20}


def task_of(definition) -> Task:
    with Runtime() as runtime:
        return Task(runtime.load(Path(definition), draft=True))


def available(value, where) -> datetime:
    try:
        moment = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise ConfigError(f'{where}: availability time {value!r} is not an ISO time with its UTC offset')
    return moment.astimezone(timezone.utc)


def addresses(facts) -> dict:
    out = {}
    for name, value in facts.items():
        if isinstance(value, dict):
            out.update({f'{name}/{key}': member for key, member in value.items()})
        else:
            out[name] = value
    return out


def predictions(facts) -> dict:
    out = {}
    for name, fact in facts.items():
        for address, found in ((f'{name}/{key}', member) for key, member in fact['members'].items()) if fact['members'] else ((name, fact),):
            prediction = found['prediction']
            if prediction is not None and prediction['kind'] == 'value':
                out[address] = prediction['value']
    return out


class Families:
    def __init__(self, task, *, max_situations, max_bytes, retained=lambda: 0, **_):
        self.scope, self.index, self.of, self.limits, self.retained = 'projection', FamilyIndex(), task.identity('projection'), (max_situations, max_bytes), retained
        self.keys, self.bytes, self.removed, self.roots, self.names = 0, 0, set(), None, {}

    def node(self, data) -> tuple:
        return ('key', self.of(data))

    def add(self, envelope, *, removed=False) -> tuple:
        anchor = self.node(envelope['input'])
        linked = [*(('group', str(name)) for name in envelope.get('groups', ())), *(('pair', str(name)) for name in envelope.get('pairs', ()))]
        fresh = [node for node in dict.fromkeys((anchor, *linked)) if node not in self.index]
        keys, size = self.keys + (anchor in fresh), self.bytes + sum(len(node[1].encode("utf-8")) + 16 for node in fresh)
        if keys > self.limits[0] or size + self.retained() > self.limits[1]:
            raise ConfigError(f'retained situations and family edges exceed the budget of {self.limits[0]} situations and {self.limits[1]} bytes; split the '
                              'source or raise the budget explicitly')
        self.keys, self.bytes = keys, size
        self.index.add(anchor)
        for node in linked:
            self.index.join(anchor, node)
        if removed:
            self.removed.add(anchor)
        self.roots = None
        return anchor

    def remove(self, data):
        self.removed.add(self.node(data))
        self.roots = None

    def name(self, node) -> str:
        root = self.index.find(node)
        if root not in self.names:
            self.names[root] = artifact_digest(list(min(self.index.seeds(root))))
        return self.names[root]

    def touched(self, data) -> bool:
        if self.roots is None:
            self.roots = {self.index.find(node) for node in self.removed}
        return self.index.find(self.node(data)) in self.roots

    def shared(self, data) -> bool:
        return sum(node[0] == 'key' for node in self.index.seeds(self.node(data))) > 1

    def bridge(self, data, prefix) -> str:
        return f'{prefix}-bridge:{self.name(self.node(data))}'

    def component(self, data) -> tuple:
        return self.index.find(self.node(data))
