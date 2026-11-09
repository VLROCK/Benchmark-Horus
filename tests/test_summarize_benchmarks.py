"""Small score files exercise aggregation without loading the Hórus corpus."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.evaluation.summarize_benchmarks import (
    Round, _iid_subgroups, _paired_summary, summarize,
)
from horus_benchmark.metrics.binary_detection import metric_values


THRESHOLDS = {"youden": 0.5, "fpr_0.01": 0.75, "fpr_0.05": 0.65}


def _predictions(protocol: str, fold: int) -> pd.DataFrame:
    labels = ["human"] * 4 + ["synthetic"] * 4
    if protocol == "iid":
        scores = [.1, .2, .3, .4, .6, .7, .8, .9]
    else:
        scores = [.1, .2, .3, .8, .4, .7, .8, .9]
    frame = pd.DataFrame({
        "record_id": [f"id-{index}" for index in range(8)],
        "label_name": labels,
        "broad_area": ["social"] * 8,
        "source_name": ["source"] * 8,
        "generator": ["generator"] * 8,
        "word_count_audit": [120] * 8,
        "probe_group_id": [f"group-{index}" for index in range(8)],
        "role": (["test_target"] * 8 if protocol == "iid" else
                 ["test_control"] * 4 + ["test_target"] * 4),
        "paired_iid_test": [True] * 8,
        "score_n2": np.asarray(scores) + fold * .002,
    })
    if protocol != "iid":
        frame.loc[len(frame)] = ["unmatched", "synthetic", "social", "source",
                                 "generator", 140, "unmatched-group",
                                 "test_unmatched_target", True, .99]
    return frame


def _write_task(runs, protocol: str, fold: int, model: str = "char4_svm",
                n3s: bool = False) -> dict:
    task_id = f"{protocol}-fold-{fold}"
    task = {"task_id": task_id, "protocol": protocol,
            "target": "all" if protocol == "iid" else "generator",
            "seed": 11, "fold": fold, "model": model,
            "manifest_sha256": "fixture", "counterfactual_n3s": n3s}
    directory = runs / task_id
    directory.mkdir(parents=True)
    predictions = _predictions(protocol, fold)
    if n3s:
        predictions["score_n3s"] = predictions.score_n2.copy()
        predictions.loc[predictions.record_id.eq("id-4"), "score_n3s"] = .05
    path = directory / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    test = predictions.loc[predictions.role.ne("test_unmatched_target")]
    metrics = metric_values(test.label_name.eq("synthetic").to_numpy(dtype=int),
                            test.score_n2.to_numpy(), THRESHOLDS)
    payload = {"task": task, "counts": {"binary_test": len(test)},
               "evaluation": {"validation_thresholds": THRESHOLDS,
                              "metrics": metrics},
               "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "n3s_inference_only_metrics": (
                   metric_values(test.label_name.eq("synthetic").to_numpy(dtype=int),
                                 test.score_n3s.to_numpy(), THRESHOLDS) if n3s else None)}
    (directory / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return task


def test_multiple_plans_and_paired_iid_ood_summary(tmp_path) -> None:
    runs = tmp_path / "runs"
    iid = [_write_task(runs, "iid", fold) for fold in range(5)]
    ood = [_write_task(runs, "logo", fold) for fold in range(5)]
    iid_plan, ood_plan = tmp_path / "iid.csv", tmp_path / "logo.csv"
    pd.DataFrame(iid).to_csv(iid_plan, index=False)
    pd.DataFrame(ood).to_csv(ood_plan, index=False)
    output = tmp_path / "summary"
    summarize([iid_plan, ood_plan], runs, output, replicates=12)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["planned_tasks"] == summary["completed_tasks"] == 10
    assert summary["paired_seed_targets"] == 1
    seeds = pd.read_csv(output / "seed_metrics.csv")
    assert len(seeds) == 2
    assert seeds.test_rows.tolist() == [8, 8] or sorted(seeds.test_rows.tolist()) == [8, 8]
    paired = pd.read_csv(output / "paired_iid_ood_deltas.csv")
    assert paired.loc[0, "paired_rows"] == 8
    assert paired.loc[0, "delta_roc_auc"] < 0
    assert paired.loc[0, "iid_roc_auc"] == pytest.approx(1.0)
    assert len(pd.read_csv(output / "ood_macro_by_seed.csv")) == 1


def test_rejects_modified_prediction_file(tmp_path) -> None:
    runs = tmp_path / "runs"
    tasks = [_write_task(runs, "iid", fold) for fold in range(5)]
    plan = tmp_path / "plan.csv"
    pd.DataFrame(tasks).to_csv(plan, index=False)
    changed = _predictions("iid", 0)
    changed.loc[0, "score_n2"] = .99
    changed.to_parquet(runs / "iid-fold-0" / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        summarize(plan, runs, tmp_path / "out", replicates=2)


def test_requires_all_five_folds(tmp_path) -> None:
    runs = tmp_path / "runs"
    tasks = [_write_task(runs, "iid", fold) for fold in range(4)]
    plan = tmp_path / "plan.csv"
    pd.DataFrame(tasks).to_csv(plan, index=False)
    with pytest.raises(ValueError, match="No complete seed"):
        summarize(plan, runs, tmp_path / "out", replicates=2)


def test_single_planned_bertimbau_ood_fold_uses_matching_iid_fold(tmp_path) -> None:
    runs = tmp_path / "runs"
    iid = [_write_task(runs, "iid", fold, "bertimbau") for fold in range(5)]
    logo = [_write_task(runs, "logo", 0, "bertimbau")]
    iid_plan, logo_plan = tmp_path / "iid.csv", tmp_path / "logo.csv"
    pd.DataFrame(iid).to_csv(iid_plan, index=False)
    pd.DataFrame(logo).to_csv(logo_plan, index=False)
    output = tmp_path / "summary"
    summarize([iid_plan, logo_plan], runs, output, replicates=4)
    seed_metrics = pd.read_csv(output / "seed_metrics.csv")
    assert seed_metrics.loc[seed_metrics.protocol.eq("logo"), "folds"].item() == 1
    paired = pd.read_csv(output / "paired_iid_ood_deltas.csv")
    assert paired.folds.item() == 1


def test_n3s_is_paired_inference_only_across_iid_folds(tmp_path) -> None:
    runs = tmp_path / "runs"
    tasks = [_write_task(runs, "iid", fold, n3s=True) for fold in range(5)]
    plan = tmp_path / "iid_n3s.csv"
    pd.DataFrame(tasks).to_csv(plan, index=False)
    output = tmp_path / "summary"
    summarize(plan, runs, output, replicates=12)
    rounds = pd.read_csv(output / "n3s_inference_only_rounds.csv")
    seed = pd.read_csv(output / "n3s_inference_only_iid_by_seed.csv")
    assert len(rounds) == 5 and len(seed) == 1
    assert seed.folds.item() == 5
    assert seed.test_rows.item() == 8
    assert seed.delta_roc_auc.item() < 0
    assert "inference only" in seed.interpretation.item()


def test_iid_subgroups_are_content_only_and_support_gated() -> None:
    frame = pd.DataFrame({
        "record_id": [f"r-{index}" for index in range(200)],
        "label_name": ["human"] * 100 + ["synthetic"] * 100,
        "probe_group_id": [f"g-{index}" for index in range(200)],
        "role": ["test_target"] * 200,
        "paired_iid_test": [True] * 200,
        "score_n2": [.2] * 100 + [.8] * 100,
        "broad_area": ["academic"] * 200,
        "word_count_audit": [120] * 200,
    })
    rounds = [Round("fixture", "char4_svm", "iid", "all", 11, 0,
                    THRESHOLDS, frame, {})]
    result = _iid_subgroups(rounds)
    assert {item["factor"] for item in result} == {"broad_area", "length_bin"}
    assert all(item["roc_auc"] == 1 for item in result)


def test_paired_summary_does_not_use_unmatched_targets() -> None:
    iid = [Round("iid", "char4_svm", "iid", "all", 11, fold,
                 THRESHOLDS, _predictions("iid", fold), {}) for fold in range(5)]
    ood = [Round("ood", "char4_svm", "logo", "generator", 11, fold,
                 THRESHOLDS, _predictions("logo", fold), {}) for fold in range(5)]
    result = _paired_summary(ood, iid, replicates=8, seed=9)
    assert result is not None and result["paired_rows"] == 8
