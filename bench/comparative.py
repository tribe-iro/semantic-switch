from __future__ import annotations

import argparse
import json

from semantic_switch import DEFAULT_JUDGE, Model
from bench.outcomes import destination, output_argument, report, runtime
from bench import quality


def main():
    parser = argparse.ArgumentParser(description="comparative choice: does the pick follow the stated goal, and survive option reordering?")
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    output_argument(parser)
    args = parser.parse_args()
    destination(args.output)
    with runtime(Model(args.judge)) as rt:
        found = quality.comparative(rt)
    print(json.dumps({key: value for key, value in found.items() if key != "rows"}, indent=1))
    report(args.output, {"judge": args.judge, **found})


if __name__ == "__main__":
    main()
