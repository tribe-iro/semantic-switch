from __future__ import annotations

import argparse
from pathlib import Path

from semantic_switch import Runtime, tools
from semantic_trainer.lexical import train


def run(runtime, switch, fact, bundle):
    exported = tools.export(runtime, switch)
    trained = train(exported['path'], bundle, fact=fact)
    report = tools.lock(runtime, switch, head_bundles={fact: trained['path']})
    return report, tools.run_test(runtime, switch)


def main():
    parser = argparse.ArgumentParser(description='Fit a lexical classifier head for one saved check or pick fact and lock it.')
    parser.add_argument('switch', type=Path)
    parser.add_argument('fact')
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    with Runtime() as rt:
        report, tested = run(rt, args.switch, args.fact, args.bundle)
        print(report['id'])
        print(tools.explain_test(tested))


if __name__ == '__main__':
    main()
