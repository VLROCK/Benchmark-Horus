"""Exact-ID OOD comparison checks using only synthetic predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from horus_benchmark.evaluation.summarize_benchmarks import Round
from horus_benchmark.evaluation import summarize_compound_ood as module
from horus_benchmark.evaluation.summarize_compound_ood import (
    _paired_frames, _read_shard_plan, merge_shards, paired_delta, plan_shards,
)


THRESHOLDS = {"youden": .5, "fpr_0.01": .75, "fpr_0.05": .65}


def _round(protocol: str, fold: int = 0, *, mismatched_group: bool = False,
           iid_subset: bool = False) -> Round:
    ids = [f"id-{index}" for index in range(8)]
    labels = ["human"] * 4 + ["synthetic"] * 4
    scores = ([.1, .2, .7, .8, .3, .4, .9, 1.0] if protocol == "logo_lodo" else
              [.1, .2, .3, .4, .6, .7, .8, .9])
    frame = pd.DataFrame({
        "record_id": ids,
        "label_name": labels,
        "probe_group_id": [f"group-{index}" for index in range(8)],
        "role": (["test_target"] * 8 if protocol == "iid" else
                 ["test_control"] * 4 + ["test_target"] * 4),
        "paired_iid_test": [index in {0, 1, 4, 5} for index in range(8)],
        "score_n2": np.asarray(scores) + fold * .001,
    })
    if iid_subset:
        frame = frame.loc[frame.record_id.isin(
            {"id-0", "id-1", "id-4", "id-5"})].reset_index(drop=True)
    if mismatched_group:
        frame.loc[frame.record_id.eq("id-0"), "probe_group_id"] = "wrong"
    return Round(f"{protocol}-{fold}", "char4_svm", protocol,
                 "all" if protocol == "iid" else "g1::social", 11, fold,
                 THRESHOLDS, frame, {})


def test_compound_contrast_is_on_exact_ids_with_frozen_thresholds() -> None:
    compound = [_round("logo_lodo", fold) for fold in range(2)]
    logo = [_round("logo", fold) for fold in range(2)]
    result = paired_delta(compound, logo, "logo", replicates=16,
                          confidence=.95, seed=7)
    assert result["status"] == "eligible"
    assert result["paired_rows"] == 8
    assert result["paired_human_rows"] == result["paired_synthetic_rows"] == 4
    assert result["delta_roc_auc"] < 0
    assert result["reference_roc_auc"] == pytest.approx(1.0)
    assert result["delta_roc_auc_ci_low"] <= result["delta_roc_auc_ci_high"]
    assert result["compound_observed_fpr_at_fpr_0.01"] > 0


def test_iid_contrast_uses_only_frozen_iid_test_intersection() -> None:
    compound = [_round("logo_lodo")]
    iid = [_round("iid", iid_subset=True)]
    result = paired_delta(compound, iid, "iid", replicates=8,
                          confidence=.95, seed=3)
    assert result["paired_rows"] == 4
    assert result["paired_human_rows"] == result["paired_synthetic_rows"] == 2
    assert result["status"] == "eligible"


def test_iid_contrast_reports_missing_binary_support() -> None:
    compound = [_round("logo_lodo")]
    iid = [_round("iid", iid_subset=True)]
    iid[0].predictions = iid[0].predictions.loc[
        iid[0].predictions.label_name.eq("human")].reset_index(drop=True)
    compound[0].predictions.loc[
        compound[0].predictions.label_name.eq("synthetic"), "paired_iid_test"] = False
    result = paired_delta(compound, iid, "iid", replicates=8,
                          confidence=.95, seed=3)
    assert result["status"] == "unsupported"
    assert result["reason"] == "paired_binary_class_missing"
    assert result["paired_human_rows"] == 2
    assert result["paired_synthetic_rows"] == 0


def test_contrast_rejects_mismatched_provenance_group() -> None:
    compound = _round("logo_lodo")
    reference = _round("logo", mismatched_group=True)
    with pytest.raises(ValueError, match="labels, groups or scores"):
        _paired_frames(compound, reference, "logo")


def test_contrast_rejects_missing_logo_test_id() -> None:
    compound = _round("logo_lodo")
    logo = _round("logo")
    logo.predictions = logo.predictions.iloc[:-1].copy()
    with pytest.raises(ValueError, match="absent from logo"):
        _paired_frames(compound, logo, "logo")


def test_shard_plan_hash_and_missing_shard_gate(tmp_path, monkeypatch) -> None:
    task_path = tmp_path / "tasks.csv"
    tasks = pd.DataFrame([{
        "seed": 20261105, "protocol": "logo_lodo", "target": "g1::social",
        "heldout_generator": "g1", "heldout_area": "social",
        "status": "eligible", "manifest_sha256": "pinned",
    }])
    tasks.to_csv(task_path, index=False)
    monkeypatch.setattr(module, "_validated_grid",
                        lambda _cfg, *, full_check: (task_path, tasks))
    plan_path = tmp_path / "shards.csv"
    config_path = tmp_path / "summary.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "experiment_id": "horus-n2-compound-ood-summary-v1",
        "compound_tasks": str(task_path), "shard_plan": str(plan_path),
        "compound_runs": str(tmp_path / "compound-runs"),
        "reference_runs": str(tmp_path / "reference-runs"),
        "compound_plans": [str(tmp_path / "compound.csv")],
        "reference_plans": [str(tmp_path / "reference.csv")],
        "seeds": [20261105, 20261106, 20261107],
        "models_cpu": ["char4_svm"], "models_bert": ["bertimbau"],
        "folds_cpu": [0, 1, 2, 3, 4], "folds_bert": [0],
        "bootstrap_replicates": 2, "bootstrap_confidence": .95,
    }), encoding="utf-8")
    plan_shards(config_path, plan_path)
    cfg = module._config(config_path)
    plan = _read_shard_plan(cfg, config_path, task_path)
    assert plan.shard_index.tolist() == [0]
    (tmp_path / "compound.csv").write_text("task_id\n", encoding="utf-8")
    (tmp_path / "reference.csv").write_text("task_id\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        merge_shards(config_path, tmp_path / "shards", tmp_path / "summary")
    plan_path.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Shard plan/config/task grid changed"):
        _read_shard_plan(cfg, config_path, task_path)
