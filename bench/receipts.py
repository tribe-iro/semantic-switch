from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import threading
import time

from semantic_switch.hosted import http


class RecordingTransport:
    def __init__(self, *, send=None, key=None):
        self.send = send if send is not None else http(key)
        self.own_send = send is None
        self.records = []
        self.lock = threading.RLock()
        self.operation = ContextVar('benchmark_operation', default=None)
        self.current_operation = None

    @contextmanager
    def recording(self, operation, *, warmup=False):
        token = self.operation.set({'id': operation, 'warmup': warmup})
        previous = self.current_operation
        self.current_operation = self.operation.get()
        try:
            yield
        finally:
            self.operation.reset(token)
            self.current_operation = previous

    async def __call__(self, url, body, timeout):
        with self.lock:
            row = {'attempt': len(self.records) + 1, 'operation': deepcopy(self.operation.get() or self.current_operation), 'endpoint': url,
                   'requested_model': body.get('model'), 'requested_provider': next(iter((body.get('provider') or {}).get('only') or ()), None),
                   'completion': 'pending', 'status': None, 'usage': None, 'returned_model': None, 'returned_provider': None,
                   'response_id': None, 'error': None}
            self.records.append(row)
        started = time.perf_counter()
        try:
            reply = await self.send(url, body, timeout)
            response = reply.body if isinstance(reply.body, dict) else {}
            row.update(status=reply.status, returned_model=response.get('model'), returned_provider=response.get('provider'),
                       response_id=response.get('id'), usage=deepcopy(response.get('usage')))
            row['completion'] = 'response' if 200 <= reply.status < 300 else 'rejected'
            return reply
        except Exception as error:
            row['completion'] = 'failed'
            row['error'] = type(error).__name__
            raise
        finally:
            row['elapsed_ms'] = (time.perf_counter() - started) * 1000

    def summary(self):
        with self.lock:
            rows = deepcopy(self.records)
        operations = {}
        for row in rows:
            operations.setdefault((row['operation'] or {}).get('id', 'unidentified'), []).append(row)
        return {'requests': len(rows), 'failed': sum(row['completion'] != 'response' for row in rows), 'usage': usage(rows),
                'by_operation': {operation: {'requests': len(group), 'usage': usage(group),
                                             'warmup': all((row['operation'] or {}).get('warmup', False) for row in group)}
                                 for operation, group in operations.items()},
                'receipts': rows}

    async def close(self):
        if self.own_send:
            await self.send.close()


def usage(rows):
    totals = {}
    for row in rows:
        for name, value in (row['usage'] if isinstance(row['usage'], dict) else {}).items():
            if type(value) is int:
                totals[name] = totals.get(name, 0) + value
    return {'reported': sum(isinstance(row['usage'], dict) for row in rows), **totals}
