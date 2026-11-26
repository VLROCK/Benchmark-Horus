"""Small integrity helpers shared by derived research campaigns."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    def serializable(item):
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                return None
            # Exact infinite thresholds must remain infinite, even for scores
            # outside validation's observed range. float('+inf') restores it.
            return '+inf' if item > 0 else '-inf'
        if isinstance(item, dict):
            return {key: serializable(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [serializable(val) for val in item]
        return item
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def checked_parquet(path: str | Path, expected: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if expected is not None and sha256(path) != expected:
        raise ValueError(f'Changed parquet: {path}')
    frame = pd.read_parquet(path)
    if 'record_id' not in frame or frame.record_id.duplicated().any():
        raise ValueError(f'Missing or duplicate record identities: {path}')
    return frame


def checked_texts(path: str | Path) -> pd.DataFrame:
    frame = checked_parquet(path)
    if 'content' not in frame or frame.content.isna().any() or not frame.content.map(
            lambda value: isinstance(value, str)).all():
        raise ValueError('Text input requires a string content column')
    hashes = frame.content.map(text_hash)
    if 'content_hash' in frame and not hashes.eq(frame.content_hash).all():
        raise ValueError('Text content hash mismatch')
    frame['content_hash'] = hashes
    return frame.sort_values('record_id', kind='stable').reset_index(drop=True)
