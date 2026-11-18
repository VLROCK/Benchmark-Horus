"""Small integrity tests for derived-view scoring; no corpus or GPU required."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.evaluation.secondary_views import (
    _load_text, _predictions, _reference, _task_id,
)


def _test_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"record_id": "h", "label_name": "human", "probe_group_id": "gh",
         "broad_area": "social", "source_name": "source", "generator": None},
        {"record_id": "s", "label_name": "synthetic", "probe_group_id": "gs",
         "broad_area": "social", "source_name": None, "generator": "model"},
    ])


def test_reference_requires_identical_parent_identity() -> None:
    rows = _test_rows()
    baseline = rows[["record_id", "label_name", "probe_group_id"]].copy()
    baseline["score_n2"] = [0.2, 0.8]
    reference = _reference(rows, baseline)
    result = _predictions(rows, reference, np.asarray([0.1, 0.9]),
                          "document_to_views", "sentence")
    assert result.record_id.tolist() == ["h", "s"]
    assert result.score_document.tolist() == [0.2, 0.8]
    assert result.score_view.tolist() == [0.1, 0.9]
    changed = baseline.copy()
    changed.loc[0, "probe_group_id"] = "wrong"
    with pytest.raises(ValueError, match="mismatched"):
        _reference(rows, changed)


def test_derived_view_rejects_wrong_ids_or_content_hash(tmp_path) -> None:
    selected = pd.DataFrame({"record_id": ["h", "s"]})
    good = pd.DataFrame({"record_id": ["h"],
                         "content": ["Uma frase completa."],
                         "content_hash": [hashlib.sha256(
                             "Uma frase completa.".encode()).hexdigest()]})
    path = tmp_path / "view.parquet"
    good.to_parquet(path, index=False)
    assert _load_text(path, selected, subset=True).record_id.tolist() == ["h"]
    with pytest.raises(ValueError, match="do not match"):
        _load_text(path, selected, subset=False)
    bad = good.copy()
    bad.loc[0, "content_hash"] = "0" * 64
    bad.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="content hash"):
        _load_text(path, selected, subset=True)


def test_task_identity_changes_with_view_bytes() -> None:
    row = {"mode": "unit_trained", "view": "sentence", "model": "word_lr",
           "seed": 20261105, "fold": 0, "baseline_task_id": "old"}
    assert _task_id(row, "config", "view-a") != _task_id(row, "config", "view-b")
