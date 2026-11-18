"""Small integrity tests for derived-view scoring; no corpus or GPU required."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from horus_benchmark.evaluation import secondary_views
from horus_benchmark.evaluation.secondary_views import (
    _load_text, _predictions, _reference, _task_id, create_plan,
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


def test_plan_writes_cpu_and_bert_modes(tmp_path, monkeypatch) -> None:
    segment_dir = tmp_path / "segments"
    layout_dir = tmp_path / "layout"
    segment_dir.mkdir()
    layout_dir.mkdir()
    (segment_dir / "segment_manifest.json").write_text("{}", encoding="utf-8")
    (layout_dir / "layout_manifest.json").write_text("{}", encoding="utf-8")
    views = {}
    for name in (*secondary_views.UNITS, *secondary_views.LAYOUTS):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        views[name] = path
    config = {
        "schema_version": 1,
        "experiment_id": "horus-n2-secondary-views-v1",
        "baseline_config": "baseline.yaml",
        "baseline_prepared": "prepared",
        "baseline_cpu_plan": "cpu.csv",
        "baseline_bert_plan": "bert.csv",
        "baseline_runs": "runs",
        "segments_dir": str(segment_dir),
        "layout_dir": str(layout_dir),
        "seeds": [20261105, 20261106, 20261107],
        "folds": [0, 1, 2, 3, 4],
        "fpr_targets": [0.01, 0.05],
        "bootstrap_replicates": 2000,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(secondary_views, "_inputs", lambda _cfg: ({}, {}, {}, {}))
    monkeypatch.setattr(secondary_views, "_view_paths", lambda *_args: views)
    monkeypatch.setattr(secondary_views, "_baseline_plan", lambda _cfg: pd.DataFrame([
        {"model": "word_lr", "seed": 20261105, "fold": 0, "task_id": "cpu-task"},
        {"model": "bertimbau", "seed": 20261105, "fold": 0, "task_id": "bert-task"},
    ]))
    output = tmp_path / "plans"
    paths = create_plan(config_path, output)
    assert len(paths) == 6
    assert {path.name: len(pd.read_csv(path)) for path in paths} == {
        "document_to_views_cpu.csv": 1,
        "document_to_views_bert.csv": 1,
        "unit_trained_cpu.csv": 2,
        "unit_trained_bert.csv": 2,
        "layout_retrained_cpu.csv": 2,
        "layout_retrained_bert.csv": 2,
    }
