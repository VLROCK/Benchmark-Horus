"""Portable, hash-verified inputs and development-only pilots for Abaporu."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from horus_benchmark.evaluation import benchmark_runner as original
from horus_benchmark.evaluation.research_io import (
    checked_parquet, checked_texts, read_json, sha256, text_hash, write_json,
)


def config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if (cfg.get('schema_version') != 1 or
            cfg.get('experiment_id') != 'horus-n2-advanced-detectors-v1' or
            cfg.get('seeds') != [20261105, 20261106, 20261107] or
            cfg.get('iid_folds') != list(range(5)) or cfg.get('ood_folds') != [0] or
            cfg.get('fpr_targets') != [.01, .05] or
            cfg.get('bootstrap_replicates') != 2000):
        raise ValueError('Unreviewed advanced campaign configuration')
    return cfg


def load_release(cfg: dict) -> tuple[pd.DataFrame, dict]:
    directory = Path(cfg['release_dir'])
    release = read_json(directory / 'release_manifest.json')
    if (release.get('verified') is not True or
            release.get('release_id') != 'horus-n2-iid-v1' or
            release.get('rows') != 107566 or release.get('split_seeds') != cfg['seeds']):
        raise ValueError('Expected the frozen selected N2 release')
    prepared = read_json(cfg['prepared_manifest'])
    if prepared['release_manifest_sha256'] != sha256(directory / 'release_manifest.json'):
        raise ValueError('Prepared N2 cache references a different release')
    info = prepared['selected_n2_texts']
    texts = checked_parquet(info['path'], info['sha256'])
    texts = checked_texts(Path(info['path']))
    index = checked_parquet(directory / 'selected_index.parquet',
                            release['artifacts']['selected_index']['sha256'])
    if len(texts) != 107566 or set(texts.record_id) != set(index.record_id):
        raise ValueError('Selected text identities differ from frozen index')
    checked = texts.merge(index[['record_id', 'content_hash', 'probe_group_id',
                                 'label_name', 'broad_area']], on='record_id',
                          validate='one_to_one', suffixes=('', '_index'))
    if any(not checked[col].eq(checked[col + '_index']).all()
           for col in ('content_hash', 'probe_group_id', 'label_name', 'broad_area')):
        raise ValueError('Selected text metadata differs from release')
    for seed in cfg['seeds']:
        path = directory / f'split_seed{seed}.parquet'
        checked_parquet(path, release['artifacts'][f'split_seed{seed}']['sha256'])
    return texts, release


def partition(cfg: dict, texts: pd.DataFrame, task: dict) -> pd.DataFrame:
    original._check_pinned_file(Path(task['manifest_path']), task['manifest_sha256'])
    split = original.partition_task(pd.read_parquet(task['manifest_path']),
                                    int(task['fold']), task['protocol'])
    frame = texts.merge(split[['record_id', 'probe_group_id', 'label_name',
                                'role', 'fold', 'paired_iid_test']], on='record_id',
                         validate='one_to_one', suffixes=('', '_split'))
    if (len(frame) != len(texts) or
            not frame.probe_group_id.eq(frame.probe_group_id_split).all() or
            not frame.label_name.eq(frame.label_name_split).all()):
        raise ValueError('Protocol and text identities disagree')
    return frame.drop(columns=['probe_group_id_split', 'label_name_split'])


def prepare(config_path: Path, output: Path) -> Path:
    cfg = config(config_path)
    texts, release = load_release(cfg)
    if (output / 'manifest.json').exists():
        prior = read_json(output / 'manifest.json')
        if (prior['config_sha256'] != sha256(config_path) or
                prior['texts_sha256'] != sha256(output / 'texts.parquet') or
                prior['pilot_sha256'] != sha256(output / 'pilot_texts.parquet')):
            raise ValueError('Existing campaign inputs changed')
        return output / 'manifest.json'
    if output.exists() and set(path.name for path in output.iterdir()) - {
            'texts.parquet', 'pilot_texts.parquet', 'pilot_ids.csv'}:
        raise FileExistsError('Incomplete input folder contains unknown files')
    split = pd.read_parquet(Path(cfg['release_dir']) / f"split_seed{cfg['seeds'][0]}.parquet")
    development = texts.loc[texts.record_id.isin(
        split.loc[split.split.eq('development'), 'record_id'])].copy()
    development['sample_key'] = development.record_id.map(
        lambda value: text_hash(f"{cfg['pilot_seed']}|{value}"))
    # Resource pilot samples length ranks throughout each class/area, then
    # chooses deterministic nearby records; no detector outcome is consulted.
    pieces = []
    n = int(cfg['pilot_per_class_area'])
    for _, group in development.groupby(['label_name', 'broad_area'], sort=True):
        ordered = group.sort_values(['word_count_audit', 'sample_key', 'record_id'])
        if len(ordered) < n:
            raise ValueError('Insufficient development support for resource pilot')
        positions = [(2 * i + 1) * len(ordered) // (2 * n) for i in range(n)]
        pieces.append(ordered.iloc[positions])
    pilot = pd.concat(pieces).drop(columns='sample_key').sort_values('record_id')
    if len(pilot) != n * 12:
        raise ValueError('Resource pilot does not cover both labels and all areas')
    output.mkdir(parents=True, exist_ok=True)
    native_path = output / 'texts.parquet'
    pilot_path = output / 'pilot_texts.parquet'
    texts.to_parquet(native_path, index=False, compression='zstd')
    pilot.to_parquet(pilot_path, index=False, compression='zstd')
    pilot[['record_id']].to_csv(output / 'pilot_ids.csv', index=False)
    target = output / 'manifest.json'
    write_json(target, {'kind': 'horus_advanced_inputs', 'rows': len(texts),
                        'pilot_rows': len(pilot), 'release_id': release['release_id'],
                        'release_sha256': sha256(Path(cfg['release_dir']) / 'release_manifest.json'),
                        'config_sha256': sha256(config_path),
                        'texts_sha256': sha256(native_path),
                        'pilot_sha256': sha256(pilot_path),
                        'pilot_note': 'Development only; length-stratified resource pilot; no accuracy tuning.'})
    return target


def common_prefix(texts_path: Path, lock_path: Path, output: Path) -> Path:
    """Same raw prefix for all models, fitting each locked tokenizer's 512 cap."""
    texts = checked_texts(texts_path)
    lock = read_json(lock_path)
    identity = {'kind': 'same_raw_prefix_512', 'source_sha256': sha256(texts_path),
                'lock_sha256': sha256(lock_path), 'implementation_sha256': sha256(__file__)}
    if output.exists():
        prior = read_json(output.with_suffix('.json'))
        if any(prior.get(key) != value for key, value in identity.items()) or \
                prior.get('sha256') != sha256(output):
            raise ValueError('Existing common prefix has changed input or artifact identity')
        derived = checked_texts(output)
        if derived.record_id.tolist() != texts.record_id.tolist() or \
                derived.parent_content_hash.tolist() != texts.content_hash.tolist():
            raise ValueError('Common prefix source lineage changed')
        return output
    from transformers import AutoTokenizer
    # Support the model-lock schema maintained by lm_cache; no network here.
    models = lock['models']
    tokenizers = []
    for info in models.values():
        path = info.get('snapshot_path', info.get('path'))
        if path is None:
            raise ValueError('LM lock must contain local snapshot paths')
        tokenizers.append(AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                        trust_remote_code=False))
    # Include the already pinned documentary BERT tokenizer as well.
    tokenizers.append(AutoTokenizer.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        revision='94d69c95f98f7d5b2a8700c420230ae10def0baa',
        local_files_only=True, trust_remote_code=False))
    rows = []
    for row in texts.itertuples(index=False):
        content = row.content
        # Truncation offsets supply a safe upper bound; binary search refines
        # the actual raw prefix without decoding or rewriting any character.
        end = len(content)
        for tokenizer in tokenizers:
            encoded = tokenizer(content[:end], return_offsets_mapping=True,
                                truncation=True, max_length=512)
            offsets = [stop for start, stop in encoded['offset_mapping'] if stop > start]
            if offsets:
                end = min(end, max(offsets))
        prefix = content[:end]
        # Retokenization can merge/split final tokens; verify every cap.
        while prefix and any(len(t(prefix)['input_ids']) > 512 for t in tokenizers):
            prefix = prefix[:-1]
        rows.append({'record_id': row.record_id, 'content': prefix,
                     'content_hash': text_hash(prefix), 'parent_content_hash': row.content_hash,
                     'retained_characters': len(prefix), 'original_characters': len(content)})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False, compression='zstd')
    write_json(output.with_suffix('.json'), {**identity, 'sha256': sha256(output),
                                           'rows': len(rows)})
    return output


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--config', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('common-prefix')
    a.add_argument('--texts', type=Path, required=True)
    a.add_argument('--lock', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    result = (prepare(args.config, args.output) if args.command == 'prepare' else
              common_prefix(args.texts, args.lock, args.output))
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
