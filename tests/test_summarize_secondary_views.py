"""Paired, group-aware secondary-view summary behavior on small fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.evaluation.summarize_secondary_views import (
    Fold, _aligned, _auc_batch, _metric_batch, _score_plan, paired_summary,
)
from horus_benchmark.metrics.binary_detection import metric_values


THRESHOLDS = {"youden": .5, "fpr_0.01": .9, "fpr_0.05": .75}


def _folds(*, change_last_id: bool = False) -> list[Fold]:
    ids = [f"r{i}" for i in range(8)]
    labels = ["human"] * 4 + ["synthetic"] * 4
    scores = [.05, .15, .25, .35, .65, .75, .85, .95]
    output = []
    for fold in range(5):
        row_ids = ids.copy()
        if change_last_id and fold == 4:
            row_ids[-1] = "another"
        frame = pd.DataFrame({
            "record_id": row_ids, "label_name": labels,
            "probe_group_id": [f"g{i}" for i in range(8)],
            "score_view": scores, "score_document": scores,
        })
        output.append(Fold("word_lr", 20261105, fold, "document_to_views",
                           "sentence", f"task-{fold}", frame,
                           THRESHOLDS, THRESHOLDS))
    return output


def test_weighted_auc_handles_ties_and_group_multiplicities() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([.1, .5, .5, .9])
    plan = _score_plan(scores, THRESHOLDS)
    groups = np.arange(4)
    weights = np.asarray([[1, 1, 1, 1], [2, 0, 1, 3]])
    auc = _auc_batch(labels, plan, groups, weights)
    assert auc[0] == pytest.approx(.875)
    assert auc[1] == pytest.approx(1.0)
    batch = _metric_batch(labels, plan, groups, weights, THRESHOLDS)
    baseline = metric_values(labels, scores, THRESHOLDS)
    assert {key: values[0] for key, values in batch.items()} == pytest.approx(baseline)


def test_identical_document_and_sentence_have_zero_paired_delta() -> None:
    folds = _folds()
    full, delta = paired_summary(folds, replicates=30, batch_size=7)
    assert full["test_rows"] == 8
    assert full["folds"] == 5
    assert len(delta) == 8
    assert all(abs(row["delta_view_minus_document"]) < 1e-12 for row in delta)
    assert all(abs(row["delta_ci_low"]) < 1e-12 and
               abs(row["delta_ci_high"]) < 1e-12 for row in delta)


def test_nuisance_subset_keeps_parent_fold_frames_unchanged() -> None:
    folds = _folds()
    original_ids = folds[0].prediction.record_id.tolist()
    ids = {"r0", "r1", "r4", "r5"}
    matched, _ = paired_summary(folds, replicates=10, membership_ids=ids)
    assert matched["test_rows"] == 4
    assert folds[0].prediction.record_id.tolist() == original_ids
    with pytest.raises(ValueError, match="outside this view"):
        paired_summary(folds, replicates=10, membership_ids={"missing"})


def test_fold_alignment_rejects_changed_parent_ids() -> None:
    with pytest.raises(ValueError, match="Fold test IDs"):
        _aligned(_folds(change_last_id=True))
