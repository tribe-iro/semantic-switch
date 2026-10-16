import json
import os
import subprocess
import signal
import time
from pathlib import Path

from semantic_switch import Model, Runtime, files, tools
from semantic_switch.training import freeze_task, read_export
from semantic_trainer import jobs
from semantic_switch.tools.specialize import terminate


root = Path(__file__).resolve().parent
project = Path.cwd()
source = project / 'work/zt411-t7/specializations/66d3ae00a9ac53d9/finalize/c529d6723d394035-33c99a90/job.yaml'
source_job = files.read_yaml(source)
historical_run = next((project / 'work/zt411-t7/specializations').glob('*/fit/*/resolved-job.json'))
historical = files.read_json(historical_run)
cohorts = files.read_json(root / 'cohorts.json')
status_path = root / 'status.json'


def status(stage, **details):
    record = {'stage': stage, 'updated': files.now(), **details}
    files.publish(status_path, record)
    print(json.dumps(record), flush=True)


def command(arguments, logfile, environment=None):
    with logfile.open('a') as output:
        with subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True) as child:
            try:
                result = child.wait()
            except BaseException:
                terminate(child, 30.)
                raise
            if result:
                raise subprocess.CalledProcessError(result, arguments)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


signal.signal(signal.SIGTERM, interrupted)


try:
    declaration = {'corpus_sha256': cohorts['corpus_sha256'], 'source_job': str(source), 'original_fit_id': historical['fit']['id'],
                   'criterion': files.sha256_file(source_job['tasks'][next(iter(source_job['tasks']))]['source']),
                   'recipe': source_job['recipe'], 'arms': cohorts['arms'],
                   'evaluation': 'unchanged T7 model_select IDs; original exposed check for operational regression only',
                   'comparison': 'Random and targeted add 96 situations/16 size-matched families to the same original fit. Original historical T7 has fewer updates; random versus targeted is the matched intervention.',
                   'paid_calls': False, 'promotion': False}
    if (previous := files.read_json(root / 'declaration.json')) is not None and previous != declaration:
        raise ValueError('experiment declaration changed')
    if previous is None:
        files.publish(root / 'declaration.json', declaration)
    bundle = root / 'task'
    contract = next(iter(source_job['tasks']))
    if not (bundle / 'frozen-task.json').exists():
        freeze_task(source_job['tasks'][contract]['source'], bundle)
    from bench import evidence
    parent = evidence.read('packs/zt411', expected=cohorts['corpus_sha256'])
    original = [row for row in parent['labels'] if row['role'] in ('fit', 'model_select')]
    additions = {arm: evidence.read(record['pack'], expected=record['sha256'])['labels'] for arm, record in cohorts['arms'].items()}
    labels = root / 'export-labels.jsonl'
    if not labels.exists():
        files.publish(labels, files.jsonl([*original, *additions['random'], *additions['targeted']]))
    catalog_pid = 2400064
    status('waiting_for_existing_catalog', pid=catalog_pid)
    while Path(f'/proc/{catalog_pid}').exists():
        time.sleep(30)
    manifest_path = root / 'native.export.jsonl.manifest.json'
    if not manifest_path.exists():
        status('exporting_existing_and_published_additions')
        with Runtime(Model(historical['model']['gguf'], **historical['native'])) as runtime:
            tools.export(runtime, bundle / 'task.yaml', root / 'native.export.jsonl', 'tune', native=True,
                         task_family='bench:zt411', seed=0, labels=labels)
    manifest, exported = read_export(manifest_path)
    original_select = {row['id'] for row in historical['memberships'] if row['role'] == 'model_select'}
    prepared = {}
    for arm in ('random', 'targeted'):
        directory = root / arm
        job_path = directory / 'job.yaml'
        selected = [row for row in original if row['role'] == 'fit'] + additions[arm]
        roles = {'fit': sorted({group for row in selected for group in row['groups']}),
                 'model_select': sorted({group for row in original if row['role'] == 'model_select' for group in row['groups']})}
        job = jobs.population_job(source_job, source.parent)
        job['recipe'] = dict(source_job['recipe'])
        job.pop('selection', None)
        job['exports'] = [{'path': str(manifest_path), 'export_id': manifest['export_id'], 'task': contract, 'roles': roles}]
        files.publish(job_path, job)
        resolved, rows = jobs.resolve(job_path)
        selecting = {row['id'] for row in rows if row['role'] == 'model_select'}
        if selecting != original_select:
            raise ValueError(f'{arm}: fixed selection membership changed')
        if resolved['populations']['fit']['situations'] != 196:
            raise ValueError(f'{arm}: merged fit situation count differs: {resolved["populations"]["fit"]}')
        files.publish(directory / 'population.json', {key: resolved[key] for key in ('populations', 'support', 'recipe', 'fit', 'monitor')})
        prepared[arm] = job_path
    for arm, job_path in prepared.items():
        directory = root / arm
        run = directory / 'run'
        receipt = run / 'deployment' / 'receipt.json'
        if not receipt.exists():
            status('fitting', arm=arm, job=str(job_path), run=str(run))
            command(['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=22G', '-p', 'MemorySwapMax=0',
                     str(project / '.venv-xpu/bin/semantic-trainer'), 'lora', str(job_path), '--to', str(run)], directory / 'fit.log')
            status('selecting', arm=arm)
            command([str(project / '.venv-xpu/bin/semantic-trainer'), 'select', str(job_path), '--run', str(run)], directory / 'select.log')
    status('operational_regression_comparison')
    command([str(project / '.venv/bin/python'), '-m', 'bench.snapshots', 'zt411', '--pack', 'packs/zt411',
             '--corpus', cohorts['corpus_sha256'], '--evidence', 'work/zt411-t7/stages/66d3ae00a9ac53d9/printer.dataset.json',
             '--baseline', f'random={root / "random/run/deployment/receipt.json"}',
             '--specialist', f'targeted={root / "targeted/run/deployment/receipt.json"}', '--role', 'check',
             '--condition', 'structured', '--kind', 'workflow', '--native', 'device=gpu', '--native', 'threads=4',
             '--margin', '0', '--results', str(root / 'results')], root / 'comparison.log')
    evidence.read('packs/zt411', expected=cohorts['corpus_sha256'])
    status('completed', result_directory=str(root / 'results'))
    if (root / 'catalog-paused.json').exists() and not (project / 'work/robust-implementation-20261015/catalog-result.json').exists():
        environment = {**os.environ, 'PYTHONPATH': str(project / 'work/robust-implementation-20261015/catalog-code')}
        command(['systemd-run', '--user', '--scope', '-q', '-p', 'MemoryMax=22G', '-p', 'MemorySwapMax=0',
                 str(project / '.venv-xpu/bin/python'), str(project / 'work/robust-implementation-20261015/recover_catalog.py')],
                root / 'catalog-resume.log', environment)
except BaseException as error:
    status('failed', error=f'{type(error).__name__}: {error}')
    raise
