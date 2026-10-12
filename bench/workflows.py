from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from semantic_switch import DEFAULT_JUDGE, Model, files, tools
from semantic_switch.cli import question_specs
from bench.diagnostics import frozen_replay
from bench.openrouter import scenarios
from bench.outcomes import runtime as model_runtime


def cli(arguments, value):
    started = time.perf_counter()
    process = subprocess.run([sys.executable, '-m', 'semantic_switch', *arguments],
                             input=json.dumps(value), capture_output=True, text=True)
    try:
        result = json.loads(process.stdout)
    except ValueError:
        raise RuntimeError(f'CLI emitted invalid JSON: {process.stdout!r}; {process.stderr[-1000:]}') from None
    return {'arguments': arguments, 'result': result, 'exit_code': process.returncode,
            'elapsed_ms': (time.perf_counter() - started) * 1000, 'stderr': process.stderr}


def run(source=DEFAULT_JUDGE):
    with TemporaryDirectory(prefix='semantic-workflow-') as directory:
        root = Path(directory)
        policy, questions, definition = root / 'policy.txt', root / 'questions.yaml', root / 'support.yaml'
        action_options, action_policy = root / 'actions.yaml', root / 'maintenance.txt'
        from examples.quickstart import ACTION, POLICY, REPORTS, run as quickstart
        files.publish(policy, 'Refund means money returned. Store credit and replacements are not refunds.')
        files.publish(questions, {'refund': {'check': 'The customer requests a refund.', 'unstated': True}})
        files.publish(definition, {'name': 'support', 'guidance': {'policy': {'text': policy.read_text()}},
            'fields': {'message': {'type': 'text', 'required': True}, 'allowed': {'type': 'bool', 'required': True}}, 'reads': ['message'],
            'facts': {'refund': {'check': 'The customer requests a refund.', 'unstated': True},
                      'topic': {'pick': 'What is the topic?', 'options': {'billing': {'description': 'charges or payments', 'aliases': ['refund']},
                                                                       'other': 'other subjects'}, 'focus': 'mentions'}},
            'cases': [{'id': 'refund', 'when': 'allowed == true and refund and topic == "billing"', 'then': {'action': 'return_money'}}],
            'otherwise': {'action': 'support'}})
        files.publish(action_options, ACTION.fact()['options'])
        files.publish(action_policy, POLICY.sections[0].text)
        value = 'Please refund the duplicate charge.'
        with model_runtime(Model(source)) as runtime:
            contrasts = []
            for identifier, case, guidance, expected, status in scenarios():
                result = runtime.check(case, 'The customer requests a refund under the supplied definition.',
                                       unstated=True, guidance=guidance)
                contrasts.append({'id': identifier, 'expected': {'value': expected, 'status': status},
                                  'actual': result.as_dict(True), 'correct': (result.value, result.status) == (expected, status)})
            tools.lock(runtime, definition)
            switch = runtime.load(definition)
            inputs = [{'message': case, 'allowed': True} for _, case, *_ in scenarios()] + [
                {'message': 'Please refund the charge.', 'allowed': False},
                {'message': 'Please refund the charge.'}, {'message': 7, 'allowed': True}]
            replay = frozen_replay(runtime, switch, inputs)
            sdk = [runtime.check(value, 'The customer requests a refund.', unstated=True, guidance=policy.read_text()).as_dict(True),
                   {key: result.as_dict(True) for key, result in runtime.ask(value, question_specs(questions), guidance=policy.read_text()).items()},
                   switch({'message': value, 'allowed': True}).as_dict(True), switch({'message': value}).as_dict(True),
                   switch({'message': 7, 'allowed': True}).as_dict(True), ACTION(REPORTS[0], runtime).as_dict(True)]
            demonstration = quickstart(runtime)
            info = runtime.info()
        prefix = ['--judge', source, '--json']
        commands = [([*prefix, 'check', 'The customer requests a refund.', '--unstated', '--guidance', str(policy)], value),
                    ([*prefix, 'ask', str(questions), '--guidance', str(policy)], value),
                    ([*prefix, 'run', str(definition)], {'message': value, 'allowed': True}),
                    ([*prefix, 'run', str(definition)], {'message': value}),
                    ([*prefix, 'run', str(definition)], {'message': 7, 'allowed': True}),
                    ([*prefix, 'pick', ACTION.question, '--options', str(action_options), '--guidance', str(action_policy)], REPORTS[0])]
        def outcomes(value):
            if 'status' in value:
                return {key: value.get(key) for key in ('status', 'value', 'case', 'answer', 'guidance')}
            return {key: outcomes(result) for key, result in value.items()}
        workflows = []
        for (arguments, payload), expected in zip(commands, sdk):
            row = {**cli(arguments, payload), 'sdk': expected}
            row['same_public_outcome'] = outcomes(row['result']) == outcomes(expected)
            workflows.append(row)
        return {'schema': 1, 'population': 'authored executable regression examples; not independent domain evidence',
                'runtime': info, 'contrasts': contrasts, 'cli': workflows[:5], 'direct_cli': workflows[5], 'native_replay': replay,
                'quickstart': demonstration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_JUDGE)
    parser.add_argument('--output', type=Path, default=Path('bench/results/public-workflows.json'))
    args = parser.parse_args()
    result = run(args.model)
    files.publish(args.output, result)
    print(json.dumps({'contrasts_correct': sum(row['correct'] for row in result['contrasts']),
                      'contrasts_submitted': len(result['contrasts']),
                      'cli_outcomes_match': all(row['same_public_outcome'] for row in result['cli'])}))


if __name__ == '__main__':
    main()
