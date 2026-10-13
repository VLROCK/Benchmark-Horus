"""Train-only-fitted diagnostic shortcut probes on official splits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from horus_benchmark.audit.text_features import BOOLEAN_FEATURES, NUMERIC_FEATURES
from horus_benchmark.data import DatasetLoader


LENGTH_FEATURES = (
    "char_count_audit", "word_count_audit", "lexical_token_count",
    "letters_per_word", "chars_per_word",
)
PUNCTUATION_FEATURES = tuple(
    name for name in NUMERIC_FEATURES
    if name.endswith("_per_100_words") and name != "newline_per_100_words"
)
PRESENTATION_FEATURES = (
    "newline_per_100_words", "paragraph_count", "nonempty_line_count", *BOOLEAN_FEATURES
)
SURFACE_GROUPS = {
    "length": LENGTH_FEATURES,
    "punctuation": PUNCTUATION_FEATURES,
    "presentation": PRESENTATION_FEATURES,
    "all_surface": (*NUMERIC_FEATURES, *BOOLEAN_FEATURES),
}


def validation_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose Youden's J threshold using validation labels only."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    positions = np.flatnonzero(finite)
    best = positions[np.argmax((true_positive_rate - false_positive_rate)[finite])]
    return float(thresholds[best])


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Bootstrap test observations within each label for diagnostic CIs."""

    if replicates < 1 or not 0 < confidence < 1:
        raise ValueError("replicates and confidence must be positive and valid")
    rng = np.random.default_rng(seed)
    label_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in label_indices):
        raise ValueError("Both labels are required for bootstrap intervals")
    aucs = np.empty(replicates)
    balanced = np.empty(replicates)
    for i in range(replicates):
        sample = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in label_indices]
        )
        aucs[i] = roc_auc_score(y_true[sample], scores[sample])
        balanced[i] = balanced_accuracy_score(y_true[sample], scores[sample] >= threshold)
    alpha = (1 - confidence) / 2
    return {
        "roc_auc_ci_low": float(np.quantile(aucs, alpha)),
        "roc_auc_ci_high": float(np.quantile(aucs, 1 - alpha)),
        "balanced_accuracy_ci_low": float(np.quantile(balanced, alpha)),
        "balanced_accuracy_ci_high": float(np.quantile(balanced, 1 - alpha)),
    }


def _metrics(
    *,
    probe: str,
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    train_rows: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    seed: int,
) -> dict[str, Any]:
    threshold = validation_threshold(y_validation, validation_scores)
    return {
        "probe": probe,
        "train_rows": train_rows,
        "validation_rows": len(y_validation),
        "test_rows": len(y_test),
        "threshold_selected_on_validation": threshold,
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_scores >= threshold)),
        **stratified_bootstrap_intervals(
            y_test, test_scores, threshold,
            replicates=bootstrap_replicates,
            confidence=bootstrap_confidence,
            seed=seed,
        ),
    }


def _surface_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, list(columns)].astype(float).to_numpy()


def run_surface_probes(
    features: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Fit simple classifiers on train and evaluate untouched official test."""

    train = features.loc[features["split"] == "train"]
    validation = features.loc[features["split"] == "validation"]
    test = features.loc[features["split"] == "test"]
    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    results = []
    for probe_name, columns in SURFACE_GROUPS.items():
        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        pipeline.fit(_surface_matrix(train, columns), y_train)
        validation_scores = pipeline.predict_proba(_surface_matrix(validation, columns))[:, 1]
        test_scores = pipeline.predict_proba(_surface_matrix(test, columns))[:, 1]
        results.append(
            _metrics(
                probe=probe_name,
                y_validation=y_validation,
                validation_scores=validation_scores,
                y_test=y_test,
                test_scores=test_scores,
                train_rows=len(train),
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_confidence=bootstrap_confidence,
                seed=seed,
            )
        )
    return pd.DataFrame(results)


def _stable_sample(frame: pd.DataFrame, per_label: int, seed: int) -> pd.DataFrame:
    selected = []
    for _, group in frame.groupby("label", sort=True):
        scores = group["record_id"].map(
            lambda identifier: hashlib.sha256(
                f"{seed}|{identifier}".encode("utf-8")
            ).hexdigest()
        )
        selected.append(group.loc[scores.sort_values(kind="stable").index[:per_label]])
    return pd.concat(selected, ignore_index=True)


def _read_selected_texts(loader: DatasetLoader, identifiers: set[str]) -> Mapping[str, str]:
    selected: dict[str, str] = {}
    for batch in loader.iter_batches(columns=["record_id", "content"], batch_size=2048):
        for record_id, content in zip(
            batch.column("record_id").to_pylist(), batch.column("content").to_pylist()
        ):
            if record_id in identifiers:
                selected[record_id] = content
    if len(selected) != len(identifiers):
        raise ValueError("Some sampled record IDs were not found in the corpus")
    return selected


def run_char4gram_probe(
    features: pd.DataFrame,
    loader: DatasetLoader,
    *,
    max_train_per_label: int,
    max_validation_per_label: int,
    max_test_per_label: int,
    max_features: int,
    min_df: int,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sampled text-only diagnostic, with vectorizer fit on train only."""

    sample = {
        "train": _stable_sample(features.loc[features["split"] == "train"], max_train_per_label, seed),
        "validation": _stable_sample(features.loc[features["split"] == "validation"], max_validation_per_label, seed),
        "test": _stable_sample(features.loc[features["split"] == "test"], max_test_per_label, seed),
    }
    identifiers = set(pd.concat(sample.values())["record_id"])
    texts_by_id = _read_selected_texts(loader, identifiers)

    def texts(split: str) -> list[str]:
        return [texts_by_id[identifier] for identifier in sample[split]["record_id"]]

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 4), max_features=max_features,
        min_df=min_df, sublinear_tf=True, dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(texts("train"))
    validation_matrix = vectorizer.transform(texts("validation"))
    test_matrix = vectorizer.transform(texts("test"))
    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
    model.fit(train_matrix, sample["train"]["label"].to_numpy(dtype=int))
    validation_scores = model.predict_proba(validation_matrix)[:, 1]
    test_scores = model.predict_proba(test_matrix)[:, 1]
    result = _metrics(
        probe="char_4gram_sampled",
        y_validation=sample["validation"]["label"].to_numpy(dtype=int),
        validation_scores=validation_scores,
        y_test=sample["test"]["label"].to_numpy(dtype=int),
        test_scores=test_scores,
        train_rows=len(sample["train"]),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        seed=seed,
    )
    result["vocabulary_size"] = len(vectorizer.vocabulary_)
    coefficients = model.coef_[0]
    feature_names = vectorizer.get_feature_names_out()
    indices = np.concatenate((np.argsort(coefficients)[:20], np.argsort(coefficients)[-20:]))
    top_features = pd.DataFrame(
        {
            "char_4gram": feature_names[indices],
            "coefficient": coefficients[indices],
            "association": np.where(coefficients[indices] >= 0, "synthetic", "human"),
        }
    ).sort_values("coefficient")
    return pd.DataFrame([result]), top_features
