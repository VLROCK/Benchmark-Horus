"""Tiny paired-score/coverage gates for advanced detector summaries."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.evaluation.summarize_benchmarks import Round
from horus_benchmark.evaluation.summarize_advanced import (
    _thresholds, agreement, common_valid, contrast, load_result, paired_delta, subgroup_metrics,
)
from horus_benchmark.evaluation import summarize_advanced as module


THRESHOLDS = {"youden": .5, "fpr_0.01": .8, "fpr_0.05": .7}


def _round(fold=0, *, invalid=(), model="detector"):
    frame = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(8)],
        "label_name": ["human"] * 4 + ["synthetic"] * 4,
        "probe_group_id": [f"g{i}" for i in range(8)],
        "role": ["test_control"] * 4 + ["test_target"] * 4,
        "paired_iid_test": [True] * 8,
        "score_n2": [.1, .2, .3, .4, .6, .7, .8, .9],
        "status": ["ok"] * 8,
        "broad_area": ["social"] * 8,
        "source_name": ["source"] * 4 + [None] * 4,
        "generator": [None] * 4 + ["generator"] * 4,
        "word_count_audit": [30, 100, 300, 1000] * 2,
    })
    frame.loc[list(invalid), "status"] = "invalid"
    frame.loc[list(invalid), "score_n2"] = np.nan
    return Round(f"task-{fold}", model, "logo", "generator", 19, fold,
                 THRESHOLDS, frame, {})


def test_common_valid_intersection_retains_full_denominator():
    rounds = [_round(0, invalid=[0]), _round(1, invalid=[7])]
    aligned, coverage = common_valid(rounds)
    assert coverage["binary_total"] == 8
    assert coverage["binary_common_valid"] == 6
    assert coverage["human_common_valid"] == coverage["synthetic_common_valid"] == 3
    assert coverage["retention"] == .75
    assert all(set(item.predictions.record_id) == {f"r{i}" for i in range(1, 7)}
               for item in aligned)
    assert len(rounds[0].predictions) == 8
    bad = _round(1)
    bad.predictions.loc[0, "probe_group_id"] = "changed"
    with pytest.raises(ValueError, match="identities differ"):
        common_valid([_round(0), bad])


def test_identical_paired_protocol_scores_produce_zero_delta():
    new = [_round(0), _round(1)]
    old = [replace(item, protocol="iid", target="all") for item in new]
    result = paired_delta(new, old, replicates=8, seed=9)
    assert result["status"] == "eligible"
    assert result["paired_rows"] == 8
    assert result["delta_roc_auc"] == 0
    assert result["delta_roc_auc_ci_low"] == result["delta_roc_auc_ci_high"] == 0


def test_paired_validity_failure_is_explicit_and_thresholds_can_be_infinite():
    result = paired_delta([_round(0, invalid=range(4, 8))], [_round(0)],
                          replicates=8, seed=9)
    assert result["status"] == "unsupported"
    assert result["reason"] == "common_valid_binary_class_missing"
    assert result["paired_human_rows"] == 4
    assert result["paired_synthetic_rows"] == 0
    thresholds = _thresholds({**THRESHOLDS, "youden": "+inf"})
    assert thresholds["youden"] == np.inf


def test_error_agreement_uses_same_ids_and_conditioned_score_correlations():
    one, two = _round(model="first"), _round(model="second", invalid=[0])
    two.predictions.loc[1, "score_n2"] = .95
    rows = agreement(one, two)
    all_youden = next(row for row in rows if row["population"] == "all" and
                      row["threshold"] == "youden")
    assert all_youden["common_valid_rows"] == 7
    assert all_youden["both_correct"] == 6
    assert all_youden["only_b_wrong"] == 1
    assert all_youden["only_a_wrong"] == all_youden["both_wrong"] == 0
    assert {row["population"] for row in rows} == {"all", "human", "synthetic"}
    two.predictions.loc[1, "label_name"] = "synthetic"
    with pytest.raises(ValueError, match="inconsistent labels"):
        agreement(one, two)


def test_unsupported_task_needs_no_predictions_but_requires_reason(tmp_path):
    task = {"task_id": "tiny", "model": "alta_pos", "protocol": "iid", "fold": 0,
            "seed": 20261105, "target": "all"}
    directory = tmp_path / "tiny"
    directory.mkdir()
    path = directory / "metrics.json"
    path.write_text(json.dumps({"kind": "horus_advanced_result", "task": task,
                               "status": "unsupported", "reason": "missing_class"}),
                    encoding="utf-8")
    result, record, hashes = load_result(task, tmp_path)
    assert result is None
    assert record["reason"] == "missing_class"
    assert str(path) in hashes
    path.write_text(json.dumps({"kind": "horus_advanced_result", "task": task,
                               "status": "unsupported", "reason": ""}), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a reason"):
        load_result(task, tmp_path)


def test_subgroup_counts_and_rates_remain_visible_for_single_label_sources():
    rounds = [_round(0), _round(1)]
    rounds[1].predictions.loc[0, "score_n2"] = .95
    identity = {"model": "detector", "protocol": "iid", "target": "all",
                "source_area": "", "seed": 19, "text_view": "native"}
    rows = subgroup_metrics(rounds, identity, minimum=2)
    area = next(row for row in rows if row["dimension"] == "broad_area")
    source = next(row for row in rows if row["dimension"] == "human_source")
    generator = next(row for row in rows if row["dimension"] == "synthetic_generator")
    short = next(row for row in rows if row["dimension"] == "word_count" and
                 row["subgroup"] == "<100")
    assert area["binary_supported"] is True
    assert area["mean_fp_youden"] == .5
    assert area["fpr_youden"] == .125
    assert source["human_rows"] == 4 and source["synthetic_rows"] == 0
    assert source["binary_supported"] is False
    assert "roc_auc" not in source
    assert source["fpr_youden"] == .125
    assert "tpr_youden" not in source
    assert generator["tpr_youden"] == 1
    assert short["binary_supported"] is False
    assert "roc_auc" not in short
    assert short["human_rows"] == short["synthetic_rows"] == 1
    assert len(json.loads(source["fold_error_counts_json"])) == 6


def test_native_common_prefix_contrast_pairs_all_folds_and_records_policy_limits(tmp_path, monkeypatch):
    native_path, view_path = tmp_path / "native.csv", tmp_path / "view.csv"
    plans = {}
    for path, policy in ((native_path, "native"), (view_path, "common_prefix.parquet")):
        plans[path] = pd.DataFrame([{
            "model": "detector", "protocol": "iid", "target": "all", "source_area": "",
            "seed": 19, "fold": fold, "text_view": policy,
            "task_id": f"{policy}-{fold}", "config_sha256": "config",
            "classical_config_sha256": "classifier", "lm_lock_sha256": "lock",
        } for fold in range(5)])
    monkeypatch.setattr(module, "_plans", lambda paths: (plans[paths[0]], {}))
    monkeypatch.setattr(module, "load_result", lambda task, _runs: (
        replace(_round(task["fold"]), protocol="iid", target="all"),
        {"status": "complete", "coverage": {"test_total": 8}}, {}))
    original = paired_delta

    def small_bootstrap(new, reference, *, replicates, seed):
        assert replicates == 2000
        return original(new, reference, replicates=8, seed=seed)

    monkeypatch.setattr(module, "paired_delta", small_bootstrap)
    target = contrast(native_path, tmp_path / "native-runs", view_path,
                      tmp_path / "view-runs", tmp_path / "summary")
    summary = json.loads(target.read_text(encoding="utf-8"))
    result = pd.read_csv(target.parent / "native_common_prefix_deltas.csv")
    assert summary["complete"] is True
    assert summary["model_seed_groups"] == 1
    assert summary["bootstrap_replicates"] == 2000
    assert result.iloc[0].delta_roc_auc == 0
    assert result.iloc[0].paired_rows == 8
    assert result.iloc[0].direction == "common_prefix_minus_native"
    assert "thresholds differ" in result.iloc[0].interpretation
    plans[view_path] = plans[view_path].iloc[:-1]
    with pytest.raises(ValueError, match="all five"):
        contrast(native_path, tmp_path / "native-runs", view_path,
                 tmp_path / "view-runs", tmp_path / "missing-fold-summary")
