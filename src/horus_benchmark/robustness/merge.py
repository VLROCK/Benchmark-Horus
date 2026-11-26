"""Merge every planned robustness shard and compare attacked errors by ID."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from horus_benchmark.evaluation.research_io import read_json, sha256, write_json
from horus_benchmark.evaluation.robustness_runner import check
from horus_benchmark.robustness.summarize import error_agreement, REQUIRED


def merge(plan_dir: Path, runs: Path, shards: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    if not check(plan_dir, runs)['complete']:
        raise ValueError('Robustness scoring incomplete')
    plan = pd.concat([pd.read_csv(plan_dir / f'{kind}.csv') for kind in ('cpu', 'bert')])
    tables = defaultdict(list)
    records = {}
    conditions = None
    for task in plan.itertuples(index=False):
        directory = shards / task.task_id
        marker = read_json(directory / 'summary.json')
        score_path = runs / task.task_id / 'predictions.parquet'
        if (marker['bootstrap_replicates'] != 2000 or len(marker['inputs']) != 1 or
                marker['inputs'][0]['sha256'] != sha256(score_path)):
            raise ValueError('Robustness summary used different score inputs')
        records[task.task_id] = sha256(directory / 'summary.json')
        for name in ('paired_metrics.csv', 'paired_effects.csv', 'coverage.csv'):
            path = directory / name
            if sha256(path) != marker['artifacts'][name]['sha256']:
                raise ValueError('Robustness summary table changed')
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if not frame.empty and (set(frame.model) != {task.model} or
                                     set(frame.seed) != {task.seed}):
                raise ValueError('Robustness summary model/seed identity changed')
            tables[name].append(frame)
        found = pd.read_parquet(score_path, columns=['attack', 'intensity']).drop_duplicates()
        observed = set(zip(found.attack, found.intensity))
        if conditions is None:
            conditions = observed
        elif conditions != observed:
            raise ValueError('Robustness tasks have different attack conditions')
    output.mkdir(parents=True, exist_ok=False)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / name, index=False)
    agreements = []
    for attack, intensity in sorted(conditions or set()):
        parts = []
        for task in plan.itertuples(index=False):
            path = runs / task.task_id / 'predictions.parquet'
            table = pq.read_table(path, columns=sorted(REQUIRED),
                                  filters=[('attack', '=', attack), ('intensity', '=', intensity)])
            parts.append(table.to_pandas())
        agreements.append(error_agreement(pd.concat(parts, ignore_index=True)))
    (pd.concat(agreements, ignore_index=True) if agreements else pd.DataFrame()).to_csv(
        output / 'detector_error_agreement.csv', index=False)
    # The mean across seeds is descriptive, not a confidence interval.
    values = pd.concat(tables['paired_metrics.csv'], ignore_index=True)
    chosen = (values.loc[values.population.eq('all_valid') & values.condition.eq('both') &
                         values.metric.eq('roc_auc')] if not values.empty else pd.DataFrame())
    if not chosen.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matrix = chosen.groupby(['model', 'attack', 'intensity']).delta_attacked_minus_clean.mean().unstack(['attack', 'intensity'])
        fig, ax = plt.subplots(figsize=(13, max(5, len(matrix) * .35)))
        limit = max(float(matrix.abs().max().max()), .01)
        plot = ax.imshow(matrix.to_numpy(), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_yticks(range(len(matrix)), matrix.index)
        ax.set_xticks(range(len(matrix.columns)), [f'{a}\n{d:g}' for a, d in matrix.columns], rotation=35, ha='right')
        ax.set_title('Delta de ROC-AUC após ataque; média descritiva das sementes')
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(output / 'robustness_auc_delta.png', dpi=180)
        plt.close(fig)
    target = output / 'summary.json'
    write_json(target, {'kind': 'horus_robustness_complete_summary', 'complete': True,
                        'tasks': len(plan), 'shards': records,
                        'plan_manifest_sha256': sha256(plan_dir / 'manifest.json'),
                        'artifacts': {path.name: sha256(path) for path in output.iterdir()
                                      if path.is_file()},
                        'note': 'Paired clean/attacked IDs; fixed clean thresholds; conditional CIs; descriptive seeds.'})
    return target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan-dir', 'runs', 'shards', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args(argv)
    print(merge(args.plan_dir, args.runs, args.shards, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
