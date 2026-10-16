from __future__ import annotations

import argparse
import json

from semantic_switch import DEFAULT_JUDGE, Model
from bench.outcomes import destination, output_argument, report, runtime
from bench import quality

SLOTS = (0, 4, 9, 14, 19)


def main():
    parser = argparse.ArgumentParser(description="option-position robustness: does the answer survive moving the correct option to slot 1, 5, 10, 15 or 20?")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    output_argument(parser)
    args = parser.parse_args()
    destination(args.output)
    with runtime(Model(args.judge)) as rt:
        found = quality.positions(rt, SLOTS)
    print(json.dumps({key: value for key, value in found.items() if key != "rows"}, indent=1))
    report(args.output, found)


if __name__ == "__main__":
    main()
