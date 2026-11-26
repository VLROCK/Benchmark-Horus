"""Frozen Portuguese UPOS annotations, cached by text identity in bounded shards.

Downloading and annotation are separate commands. Stanza never learns from
Horus. Corpus-dependent feature fits remain inside the detector's training fold.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import time
from pathlib import Path

import pandas as pd

from horus_benchmark.evaluation.research_io import (
    checked_texts, read_json, sha256, write_json,
)


def _inventory(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob('*')) if path.is_file()
            and path.suffix in {'.pt', '.json'}
            and ('pt/' in path.as_posix() or path == directory / 'resources.json')}


def download(directory: Path) -> Path:
    target = directory / 'horus_pt_resources.lock.json'
    if target.exists():
        _locked(directory)
        return target
    import stanza
    # Explicit no-charlm packages prevent optional, silently changing models.
    stanza.download('pt', model_dir=str(directory), processors={
        'tokenize': 'bosque', 'mwt': 'bosque', 'pos': 'bosque_nocharlm'},
        package=None, verbose=True)
    artifacts = _inventory(directory)
    if not artifacts:
        raise ValueError('No Portuguese Stanza resources downloaded')
    value = {'kind': 'horus_portuguese_stanza_lock', 'language': 'pt',
             'stanza': importlib.metadata.version('stanza'),
             'processors': {'tokenize': 'bosque', 'mwt': 'bosque',
                            'pos': 'bosque_nocharlm'}, 'artifacts': artifacts}
    if target.exists() and read_json(target) != value:
        raise ValueError('Existing Stanza lock changed; use a new resource directory')
    write_json(target, value)
    return target


def _locked(directory: Path) -> dict:
    lock = read_json(directory / 'horus_pt_resources.lock.json')
    if lock['stanza'] != importlib.metadata.version('stanza'):
        raise ValueError('Stanza library version differs from resource lock')
    for name, digest in lock['artifacts'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Stanza resource changed: {name}')
    return lock


def annotate(texts_path: Path, resources: Path, output: Path,
             shard_index: int, shard_size: int) -> Path:
    import stanza
    if shard_index < 0 or shard_size < 1:
        raise ValueError('Invalid annotation shard')
    lock = _locked(resources)
    frame = checked_texts(texts_path)
    subset = frame.iloc[shard_index * shard_size:(shard_index + 1) * shard_size]
    if subset.empty:
        raise IndexError('Annotation shard outside text population')
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'shard-{shard_index:05d}.parquet'
    identity = {'texts_sha256': sha256(texts_path),
                'resource_lock_sha256': sha256(resources / 'horus_pt_resources.lock.json'),
                'shard_index': shard_index, 'shard_size': shard_size,
                'rows': len(subset)}
    marker = target.with_suffix('.json')
    if marker.exists():
        prior = read_json(marker)
        if all(prior.get(k) == v for k, v in identity.items()) and \
                sha256(target) == prior['sha256']:
            return marker
        raise ValueError('Existing annotation shard has different input identity')
    nlp = stanza.Pipeline('pt', dir=str(resources), package=None,
                          processors=lock['processors'], use_gpu=False,
                          download_method=None, verbose=False)
    started = time.monotonic()
    rows = []
    for row in subset.itertuples(index=False):
        # Failures remain explicit and can never become an all-zero POS vector.
        try:
            doc = nlp(row.content)
            tags = [word.upos for sentence in doc.sentences for word in sentence.words]
            if not tags:
                raise ValueError('no_pos_tokens')
            rows.append({'record_id': row.record_id, 'content_hash': row.content_hash,
                         'upos': tags, 'status': 'ok', 'reason': ''})
        except (ValueError, RuntimeError, IndexError) as error:
            rows.append({'record_id': row.record_id, 'content_hash': row.content_hash,
                         'upos': [], 'status': 'invalid',
                         'reason': type(error).__name__ + ':' + str(error)[:250]})
    pd.DataFrame(rows).to_parquet(target, index=False, compression='zstd')
    write_json(marker, {**identity, 'sha256': sha256(target),
                        'seconds': time.monotonic() - started})
    return marker


def merge(texts_path: Path, resources: Path, output: Path, shard_size: int) -> Path:
    _locked(resources)
    texts = checked_texts(texts_path)
    frames = []
    lock_sha = sha256(resources / 'horus_pt_resources.lock.json')
    for index in range(math.ceil(len(texts) / shard_size)):
        path = output / f'shard-{index:05d}.parquet'
        record = read_json(path.with_suffix('.json'))
        if (record['texts_sha256'] != sha256(texts_path) or
                record['resource_lock_sha256'] != lock_sha or
                record['shard_size'] != shard_size or record['shard_index'] != index or
                record['sha256'] != sha256(path)):
            raise ValueError('Changed or mismatched annotation shard')
        frame = pd.read_parquet(path)
        expected = texts.iloc[index * shard_size:(index + 1) * shard_size]
        if (frame.record_id.tolist() != expected.record_id.tolist() or
                frame.content_hash.tolist() != expected.content_hash.tolist()):
            raise ValueError('Annotation shard IDs or text hashes differ')
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    path = output / 'annotations.parquet'
    manifest = output / 'manifest.json'
    if manifest.exists():
        prior = read_json(manifest)
        if (prior['texts_sha256'] != sha256(texts_path) or
                prior['resources_sha256'] != lock_sha or
                prior['annotations_sha256'] != sha256(path)):
            raise ValueError('Existing annotation merge changed')
        return manifest
    combined.to_parquet(path, index=False, compression='zstd')
    write_json(manifest, {'kind': 'horus_upos_cache', 'rows': len(combined),
                         'texts_sha256': sha256(texts_path),
                         'resources_sha256': lock_sha,
                         'annotations_sha256': sha256(path),
                         'status_counts': {str(k): int(v) for k, v in
                                           combined.status.value_counts().items()}})
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download')
    p.add_argument('--resources', type=Path, required=True)
    for command in ('annotate', 'merge'):
        p = sub.add_parser(command)
        p.add_argument('--resources', type=Path, required=True)
        p.add_argument('--texts', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--shard-size', type=int, default=2048)
        if command == 'annotate':
            p.add_argument('--shard-index', type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == 'download':
        print(download(args.resources))
    elif args.command == 'annotate':
        print(annotate(args.texts, args.resources, args.output,
                       args.shard_index, args.shard_size))
    else:
        print(merge(args.texts, args.resources, args.output, args.shard_size))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
