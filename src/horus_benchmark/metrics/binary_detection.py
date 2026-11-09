"""Validation-calibrated binary detection metrics and group-aware uncertainty.

Scores increase toward synthetic. Thresholds are frozen on validation data;
bootstrap intervals condition on the fitted detector and chosen thresholds.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


def _arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Labels and scores must be nonempty aligned vectors")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("Both labels and finite scores are required")
    return y, s


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold with maximal validation TPR-FPR, conservative on ties."""
    y, s = _arrays(labels, scores)
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    best = np.flatnonzero(np.isclose(tpr - fpr, np.max(tpr - fpr)))
    return float(np.max(thresholds[best]))


def fpr_threshold(labels: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Lowest threshold with empirical human validation FPR <= target.

    Prediction uses score >= threshold. Equal scores are never split across a
    threshold; the next-highest representable float gives zero FPR if needed.
    """
    if not 0 < target < 1:
        raise ValueError("FPR target must be in (0, 1)")
    y, s = _arrays(labels, scores)
    human = s[y == 0]
    candidates = np.r_[np.unique(human), np.nextafter(np.max(human), np.inf)]
    allowed = candidates[np.asarray([(human >= t).mean() <= target for t in candidates])]
    return float(np.min(allowed))


def metric_values(labels: np.ndarray, scores: np.ndarray,
                  thresholds: dict[str, float]) -> dict[str, float]:
    y, s = _arrays(labels, scores)
    result = {"roc_auc": float(roc_auc_score(y, s))}
    for key, threshold in thresholds.items():
        prediction = s >= threshold
        if key == "youden":
            result["balanced_accuracy"] = float(balanced_accuracy_score(y, prediction))
        else:
            result[f"tpr_at_{key}"] = float(prediction[y == 1].mean())
            result[f"observed_fpr_at_{key}"] = float(prediction[y == 0].mean())
            result[f"balanced_accuracy_at_{key}"] = float(
                balanced_accuracy_score(y, prediction))
    return result


def _group_indices(labels: np.ndarray, groups: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Require a single label per linked group; preserve group sizes on draw."""
    indexed: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indexed[str(group)].append(index)
    by_label: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for indexes in indexed.values():
        values = np.unique(labels[indexes])
        if len(values) != 1:
            raise ValueError("A bootstrap group spans labels")
        by_label[int(values[0])].append(np.asarray(indexes, dtype=np.int64))
    if any(not values for values in by_label.values()):
        raise ValueError("Both classes need at least one bootstrap group")
    return by_label


def grouped_bootstrap(labels: np.ndarray, scores: np.ndarray,
                      groups: np.ndarray, thresholds: dict[str, float],
                      *, replicates: int = 2000, confidence: float = 0.95,
                      seed: int = 20261109) -> dict[str, tuple[float, float]]:
    """Percentile CI, stratified by label and resampling whole probe groups."""
    y, s = _arrays(labels, scores)
    g = np.asarray(groups)
    if g.shape != y.shape or replicates < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap groups, repetitions or confidence")
    by_label = _group_indices(y, g)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        draw = []
        for label in (0, 1):
            group_rows = by_label[label]
            chosen = rng.integers(0, len(group_rows), size=len(group_rows))
            draw.extend(group_rows[int(index)] for index in chosen)
        positions = np.concatenate(draw)
        for key, value in metric_values(y[positions], s[positions], thresholds).items():
            collected[key].append(value)
    tail = (1 - confidence) / 2
    return {key: (float(np.quantile(values, tail)),
                  float(np.quantile(values, 1 - tail)))
            for key, values in collected.items()}


def evaluate_binary(validation_labels: np.ndarray, validation_scores: np.ndarray,
                    test_labels: np.ndarray, test_scores: np.ndarray,
                    test_groups: np.ndarray, *,
                    fpr_targets: tuple[float, ...] = (0.01, 0.05),
                    bootstrap_replicates: int = 2000,
                    bootstrap_seed: int = 20261109) -> dict:
    yv, sv = _arrays(validation_labels, validation_scores)
    yt, st = _arrays(test_labels, test_scores)
    thresholds = {"youden": youden_threshold(yv, sv)}
    for target in fpr_targets:
        thresholds[f"fpr_{target:g}"] = fpr_threshold(yv, sv, target)
    metrics = metric_values(yt, st, thresholds)
    intervals = grouped_bootstrap(
        yt, st, test_groups, thresholds, replicates=bootstrap_replicates,
        seed=bootstrap_seed)
    return {"validation_thresholds": thresholds, "metrics": metrics,
            "confidence_intervals_95": intervals,
            "interval_note": "Conditional on fitted model and validation thresholds; "
                             "whole probe_group_id resampled within label."}
