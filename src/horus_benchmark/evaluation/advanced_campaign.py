"""Build bounded, auditable Slurm work lists for the approved Abaporu campaign.

Stages declare dependencies by requiring the preceding manifests/files. A job
executes one command, or a bounded bundle, via argv (never eval or shell text).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

from horus_benchmark.evaluation.advanced_data import config
from horus_benchmark.evaluation.research_io import read_json, sha256, write_json


PREFIX = 'horus_benchmark.'
PAIRS = ('falcon', 'gptj_neo', 'qwen', 'ghost_weak')


def _command(module: str, *args) -> list[str]:
    return [PREFIX + module, *map(str, args)]


def build_jobs(config_path: Path, stage: str, output: Path,
               resources: Path, summary_plan: Path | None = None) -> Path:
    cfg = config(config_path)
    root = Path(cfg['output_root'])
    inputs = root / 'inputs'
    size = int(cfg['shard_size'])
    lock = Path(cfg['lm_lock'])
    common = ('--config', cfg['lm_config'], '--lock', lock)
    commands = []
    if stage == 'prepare':
        commands.append(_command('evaluation.advanced_data', 'prepare', '--config', config_path,
                                  '--output', inputs))
    elif stage == 'cpu-pilot':
        commands.append(_command('evaluation.advanced_pilot', '--config', config_path,
                                  '--output', root / 'cpu_pilot.json'))
    elif stage == 'common-prefix':
        commands.append(_command('evaluation.advanced_data', 'common-prefix',
                                  '--texts', inputs / 'texts.parquet', '--lock', lock,
                                  '--output', inputs / 'common_prefix.parquet'))
    elif stage == 'attacks':
        commands.append(_command('robustness.prepare', 'prepare',
                                  '--config', 'configs/experiments/robustness_n2_v1.yaml',
                                  '--release-dir', cfg['release_dir'],
                                  '--texts', inputs / 'texts.parquet', '--output', root / 'attacks'))
    elif stage in {'iid-plan', 'ood-plan', 'compound-plan', 'common-plan'}:
        name = stage.removesuffix('-plan')
        protocols = {'iid': ['iid'], 'ood': ['logo', 'lodo', 'loso'],
                     'compound': ['logo_lodo'], 'common': ['iid']}[name]
        target = root / 'plans' / f'{name}.csv'
        commands.append(_command('evaluation.advanced_runner', 'plan', '--config', config_path,
                                  '--output', target, '--protocols', *protocols,
                                  *(['--text-view', str(inputs / 'common_prefix.parquet')]
                                    if name == 'common' else [])))
    elif stage.startswith(('lm-', 'pos-')):
        family, population = stage.split('-', 1)
        merge = population.startswith('merge-')
        if merge:
            population = population.removeprefix('merge-')
        if population == 'attacks':
            attack_manifest = read_json(root / 'attacks' / 'attack_manifest.json')
            populations = []
            for key, artifact in attack_manifest['artifacts'].items():
                if artifact.get('kind') == 'attack_texts':
                    path = Path(artifact['path'])
                    if sha256(path) != artifact['sha256']:
                        raise ValueError('Attack text artifact changed')
                    populations.append((f'attacks/{key}', path))
        elif population in {'native', 'pilot', 'common'}:
            path = inputs / {'native': 'texts.parquet', 'pilot': 'pilot_texts.parquet',
                             'common': 'common_prefix.parquet'}[population]
            populations = [('common_prefix' if population == 'common' else population, path)]
        else:
            raise ValueError('Unknown feature population')
        for name, path in populations:
            rows = pq.ParquetFile(path).metadata.num_rows
            if family == 'lm':
                for pair in PAIRS:
                    args = [*common, '--texts', path, '--pair', pair,
                            '--output', root / 'cache' / name / pair, '--shard-size', size]
                    if merge:
                        commands.append(_command('evaluation.lm_cache', 'merge', *args))
                    else:
                        for index in range(math.ceil(rows / size)):
                            commands.append(_command('evaluation.lm_cache', 'score', *args,
                                                      '--shard-index', index))
            else:
                args = ['--texts', path, '--resources', resources,
                        '--output', root / 'linguistic' / name, '--shard-size', size]
                if merge:
                    commands.append(_command('evaluation.linguistic_cache', 'merge', *args))
                else:
                    for index in range(math.ceil(rows / size)):
                        commands.append(_command('evaluation.linguistic_cache', 'annotate',
                                                  *args, '--shard-index', index))
    elif stage in {'iid', 'ood', 'compound', 'common'}:
        path = root / 'plans' / f'{stage}.csv'
        marker = read_json(path.with_suffix('.json'))
        if marker['sha256'] != sha256(path):
            raise ValueError('Changed detector plan')
        runs = root / ('common_prefix/runs' if stage == 'common' else 'runs')
        count = len(pd.read_csv(path))
        for index in range(0, count, 4):
            commands.append(_command('evaluation.advanced_runner', 'run-bundle',
                                      '--config', config_path, '--plan', path,
                                      '--indices', *range(index, min(index + 4, count)), '--output', runs))
    elif stage == 'robustness-plan':
        commands.append(_command('evaluation.robustness_runner', 'plan', '--config', config_path,
                                  '--advanced-plan', root / 'plans' / 'iid.csv',
                                  '--output-dir', root / 'plans' / 'robustness'))
    elif stage in {'robustness-cpu', 'robustness-bert'}:
        kind = stage.split('-')[1]
        path = root / 'plans' / 'robustness' / f'{kind}.csv'
        for index in range(len(pd.read_csv(path))):
            commands.append(_command('evaluation.robustness_runner', 'run-task',
                                      '--config', config_path, '--plan', path,
                                      '--index', index, '--output', root / 'robustness' / 'runs'))
    elif stage == 'summary':
        if summary_plan is None:
            raise ValueError('--summary-plan is required')
        for index in range(len(pd.read_csv(summary_plan))):
            commands.append(_command('evaluation.summarize_advanced', 'shard',
                                      '--plan', summary_plan, '--index', index,
                                      '--runs', root / 'runs', '--output', root / 'summary_shards'))
    elif stage == 'summary-merge':
        path = summary_plan or root / 'plans' / 'summary.csv'
        commands.append(_command('evaluation.summarize_advanced', 'merge', '--plan', path,
                                  '--shards', root / 'summary_shards', '--output', root / 'summary'))
    elif stage == 'contrast':
        for model in cfg['models_iid']:
            commands.append(_command('evaluation.summarize_advanced', 'contrast',
                                      '--native-plan', root / 'plans' / 'iid.csv',
                                      '--native-runs', root / 'runs',
                                      '--view-plan', root / 'plans' / 'common.csv',
                                      '--view-runs', root / 'common_prefix' / 'runs',
                                      '--models', model, '--output', root / 'contrast' / model))
    elif stage == 'errors':
        plans = [root / 'plans' / f'{name}.csv' for name in ('iid', 'ood', 'compound')]
        if any(not path.is_file() for path in plans):
            raise ValueError('Final error report requires IID, OOD and compound plans')
        old = [Path(cfg['baseline_plans']) / f'{name}_{kind}.csv'
               for name in ('iid', 'logo', 'lodo', 'loso') for kind in ('cpu', 'bert')]
        commands.append(_command('evaluation.summarize_advanced', 'errors', '--plans', *plans,
                                  '--runs', root / 'runs', '--baseline-plans', *old,
                                  '--baseline-runs', cfg['baseline_runs'], '--output', root / 'errors'))
    elif stage == 'robustness-summary':
        directory = root / 'plans' / 'robustness'
        from horus_benchmark.evaluation.robustness_runner import check
        checked = check(directory, root / 'robustness' / 'runs')
        if not checked['complete']:
            raise ValueError('Robustness runs are incomplete; inspect robustness_runner check')
        for kind in ('cpu', 'bert'):
            for row in pd.read_csv(directory / f'{kind}.csv').itertuples(index=False):
                commands.append(_command('robustness.summarize', '--inputs',
                                          root / 'robustness' / 'runs' / row.task_id / 'predictions.parquet',
                                          '--output', root / 'robustness' / 'summary_shards' / row.task_id,
                                          '--replicates', cfg['bootstrap_replicates']))
    elif stage == 'robustness-merge':
        commands.append(_command('robustness.merge', '--plan-dir', root / 'plans' / 'robustness',
                                  '--runs', root / 'robustness' / 'runs',
                                  '--shards', root / 'robustness' / 'summary_shards',
                                  '--output', root / 'robustness' / 'summary'))
    else:
        raise ValueError(f'Unknown campaign stage: {stage}')
    if not commands:
        raise ValueError('No jobs in this stage')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'index': i, 'module': cmd[0], 'argv_json': json.dumps(cmd[1:])}
                  for i, cmd in enumerate(commands)]).to_csv(output, index=False)
    write_json(output.with_suffix('.json'), {'kind': 'horus_abaporu_job_list',
                                            'stage': stage, 'rows': len(commands),
                                            'config_sha256': sha256(config_path),
                                            'sha256': sha256(output)})
    return output


def worker(plan: Path, index: int) -> int:
    marker = read_json(plan.with_suffix('.json'))
    if marker['kind'] != 'horus_abaporu_job_list' or marker['sha256'] != sha256(plan):
        raise ValueError('Changed Slurm command plan')
    rows = pd.read_csv(plan, keep_default_na=False)
    if not 0 <= index < len(rows):
        raise IndexError('Slurm work index outside plan')
    row = rows.iloc[index]
    completed = plan.parent / 'completed' / plan.stem / f'{index:05d}.json'
    if completed.exists():
        prior = read_json(completed)
        if prior.get('plan_sha256') == sha256(plan) and prior.get('returncode') == 0:
            print('Already completed', plan.stem, index)
            return 0
        raise ValueError('Completion marker references another job plan')
    allowed = {'evaluation.advanced_data', 'evaluation.advanced_runner', 'evaluation.advanced_pilot',
               'evaluation.linguistic_cache', 'evaluation.lm_cache',
               'robustness.prepare', 'evaluation.robustness_runner',
               'evaluation.summarize_advanced', 'robustness.summarize', 'robustness.merge'}
    if row.module.removeprefix(PREFIX) not in allowed or not row.module.startswith(PREFIX):
        raise ValueError('Unknown campaign worker module')
    args = json.loads(row.argv_json)
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError('Worker arguments must be strings')
    print('Running', row.module, *args, flush=True)
    code = subprocess.run([sys.executable, '-m', row.module, *args], check=False).returncode
    if code == 0:
        write_json(completed, {'plan_sha256': sha256(plan), 'index': index,
                               'returncode': 0, 'module': row.module})
    return code


def pending(plan: Path) -> list[tuple[int, list[int]]]:
    marker = read_json(plan.with_suffix('.json'))
    if marker['sha256'] != sha256(plan):
        raise ValueError('Changed job plan')
    count = len(pd.read_csv(plan))
    groups = []
    for offset in range(0, count, 400):
        indices = []
        for index in range(offset, min(count, offset + 400)):
            path = plan.parent / 'completed' / plan.stem / f'{index:05d}.json'
            if not path.exists():
                indices.append(index - offset)
            else:
                prior = read_json(path)
                if prior.get('plan_sha256') != marker['sha256']:
                    raise ValueError('Completion marker references another job plan')
                if prior.get('returncode') != 0:
                    raise ValueError('Completion marker is not a successful command')
        if indices:
            groups.append((offset, indices))
    return groups


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('jobs')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--stage', required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--resources', type=Path, default=Path(os.environ.get(
        'STANZA_RESOURCES_DIR', '/mnt/recod/homes/victor.roquetto/.cache/stanza_horus')))
    a.add_argument('--summary-plan', type=Path)
    a = sub.add_parser('worker')
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--index', type=int, required=True)
    a = sub.add_parser('pending')
    a.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'jobs':
        print(build_jobs(args.config, args.stage, args.output, args.resources, args.summary_plan))
        return 0
    if args.command == 'pending':
        for offset, indexes in pending(args.plan):
            print(str(offset) + '|' + ','.join(map(str, indexes)))
        return 0
    return worker(args.plan, args.index)


if __name__ == '__main__':
    raise SystemExit(main())
