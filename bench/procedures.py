"""Procedure selection over a card catalog: exact links, exclusions, unstated details, supersedes, latency.

Run from the repository root:
    python -m bench.procedures [judge source]
"""
from __future__ import annotations

import sys
import time

from semantic_switch import Model, Runtime

CASES = [
    ({"machine": "MX-120A", "active_alarms": ["E17"], "report": "It won't start after a restart and shows E17. No leak."},
     "follow_procedure", {"P02"}),
    ({"machine": "MX-120", "active_alarms": ["E17"], "report": "Fails to start again, E17 on the display. No leak."},
     "follow_procedure", {"P01"}),
    ({"machine": "MX-120A", "active_alarms": ["E17"], "report": "Won't restart, E17."}, "needs_input", None),
    ({"machine": "MX-120A", "active_alarms": ["E17"], "report": "Oil is dripping under the pump. It still runs."},
     "follow_procedure", {"P03"}),
    ({"machine": "MX-120A", "active_alarms": [], "report": "Smoke coming out of the motor! No leak."},
     "emergency_shutdown", {"P09"}),
    ({"machine": "MX-120A", "active_alarms": ["E71"], "report": "Sensor fault on the display, no leak, otherwise fine."},
     "follow_procedure", {"P04"}),
    ({"machine": "MX-120", "active_alarms": [], "report": "Routine check: everything works, floor dry, no leaks."},
     "no_procedure", set()),
    ({"machine": "MX-120A", "active_alarms": ["E17"],
      "report": "Mixer 120 Advanced won't restart (E17). No leaks. It also smells like something is burning."},
     "emergency_shutdown", {"P09"}),
    ({"machine": "MX-120A", "active_alarms": ["E17"], "report": "A MX-120A não liga depois de religar, mostra E17. Sem vazamento."},
     "follow_procedure", {"P02"}),
    ({"machine": "MX-120A", "active_alarms": ["E17"], "report": "Há óleo escorrendo da bomba, mas a máquina funciona."},
     "follow_procedure", {"P03"}),
]


def main():
    models = {"judge": Model(sys.argv[1])} if len(sys.argv) > 1 else None
    with Runtime(models) as rt:
        sw = rt.load("switches/procedures.yaml", draft=True)
        sw(CASES[0][0])
        right = 0
        for data, answer, cards in CASES:
            t0 = time.perf_counter()
            r = sw(data)
            ms = (time.perf_counter() - t0) * 1000
            got = r.facts["procedures"].value if "procedures" in r.facts else None
            ok = (r.answer if r.status == "ok" else r.status) == answer and (cards is None or set(got or ()) == cards)
            right += ok
            print(f"{'✓' if ok else '✗'} {ms:5.0f} ms  {str(r):40}  cards={got}  | {data['report'][:50]}")
        print(f"{right}/{len(CASES)} correct")


if __name__ == "__main__":
    main()
