"""Small deterministic checks; no corpus or cluster experiment is read."""

import numpy as np
import pytest

from horus_benchmark.metrics.binary_detection import (
    evaluate_binary, fpr_threshold, grouped_bootstrap, youden_threshold,
)


def test_validation_thresholds_never_use_test_labels() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.7, 0.8])
    threshold = fpr_threshold(y, score, 0.25)
    assert np.mean(score[y == 0] >= threshold) <= 0.25
    assert np.mean(score[y == 0] >= 0.3) > 0.25
    assert np.isfinite(youden_threshold(y, score))


def test_tied_human_scores_are_not_split_at_fpr_boundary() -> None:
    y = np.array([0] * 5 + [1] * 5)
    score = np.array([0.8, 0.8, 0.1, 0.1, 0.1, 0.9, 0.7, 0.6, 0.5, 0.4])
    threshold = fpr_threshold(y, score, 0.2)
    assert threshold > 0.8


def test_grouped_intervals_keep_linked_documents_together() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.8, 0.85, 0.9])
    groups = np.array(["h1", "h1", "h2", "h2", "s1", "s1", "s2", "s2"])
    result = evaluate_binary(y, score, y, score, groups,
                             bootstrap_replicates=20, bootstrap_seed=4)
    assert result["metrics"]["roc_auc"] == 1.0
    assert result["metrics"]["balanced_accuracy"] == 1.0
    assert result["metrics"]["observed_fpr_at_fpr_0.01"] == 0.0
    assert result["confidence_intervals_95"]["roc_auc"] == (1.0, 1.0)


def test_mixed_label_group_rejected() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    with pytest.raises(ValueError, match="spans labels"):
        grouped_bootstrap(y, s, np.array(["x", "h", "x", "s"]),
                          {"youden": 0.5}, replicates=2)
