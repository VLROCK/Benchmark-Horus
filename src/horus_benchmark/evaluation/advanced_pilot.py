"""Resource-only development pilot; no detector accuracy is used for tuning."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from horus_benchmark.evaluation.advanced_data import config, load_release
from horus_benchmark.evaluation.advanced_runner import enrich, fit_model, ZERO_SHOT
from horus_benchmark.evaluation.research_io import checked_texts, sha256, write_json


def run(config_path: Path, output: Path) -> Path:
    cfg = config(config_path)
    root = Path(cfg['output_root'])
    texts, _ = load_release(cfg)
    pilot = checked_texts(root / 'inputs' / 'pilot_texts.parquet')
    pilot = texts.loc[texts.record_id.isin(set(pilot.record_id))]
    split = pd.read_parquet(Path(cfg['release_dir']) / f"split_seed{cfg['seeds'][0]}.parquet")
    frame = pilot.merge(split[['record_id', 'fold', 'split']], on='record_id',
                         validate='one_to_one')
    if not frame.split.eq('development').all():
        raise ValueError('Resource pilot contains a frozen test document')
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for model in cfg['models_iid']:
        if model in ZERO_SHOT:
            continue
        data, inputs = enrich(frame, model, cfg,
                              lm_root=root / 'cache' / 'pilot',
                              pos_root=root / 'linguistic' / 'pilot')
        data = data.loc[data.feature_status.eq('ok')].sort_values('record_id')
        train = data.loc[data.fold.ne(0)]
        held = data.loc[data.fold.eq(0)]
        # Two disjoint halves of development fold0 exercise the full API;
        # this is a timing fixture, never a scientific validation/test split.
        val_parts, test_parts = [], []
        for _, part in held.groupby('label_name', sort=True):
            groups = sorted(part.probe_group_id.unique())
            if len(groups) < 2:
                raise ValueError('Resource pilot needs two held-out groups per class')
            val_parts.append(part.loc[part.probe_group_id.isin(groups[::2])])
            test_parts.append(part.loc[part.probe_group_id.isin(groups[1::2])])
        validation, test = pd.concat(val_parts), pd.concat(test_parts)
        started = time.monotonic()
        fit_model(model, train, validation, test, cfg, int(cfg['pilot_seed']))
        rows.append({'model': model, 'train_rows': len(train),
                     'validation_rows': len(validation), 'scored_rows': len(test),
                     'seconds': time.monotonic() - started, 'inputs': inputs})
        print(model, rows[-1]['seconds'], flush=True)
    write_json(output, {'kind': 'horus_development_resource_pilot',
                        'config_sha256': sha256(config_path), 'models': rows,
                        'note': 'No accuracy recorded or used for choices; CPU search cost need not scale linearly.'})
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    print(run(args.config, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
