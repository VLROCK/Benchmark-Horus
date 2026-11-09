"""Text-only shortcut probes and classical detector baselines.

Every vectorizer, imputer, scaler and dimensionality reducer is fitted inside
the supplied training partition. `inner_folds` may be used for C selection,
but outer validation/test labels are never used for model selection here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from horus_benchmark.features.shortcut_features import SURFACE_GROUPS, surface_matrix


MODEL_NAMES = (
    "surface_length", "surface_presentation", "surface_punctuation",
    "surface_all", "word_lr", "char4_lr", "char4_svm", "human_iforest",
)


def _linear_classifier(model_name: str, c: float, seed: int, cfg: dict[str, Any]) -> Any:
    if model_name == "char4_svm":
        return LinearSVC(
            C=c, class_weight="balanced", dual="auto",
            max_iter=int(cfg["linear"]["svm_max_iter"]), random_state=seed,
        )
    return LogisticRegression(
        C=c, class_weight="balanced", solver="liblinear",
        max_iter=int(cfg["linear"]["lr_max_iter"]), random_state=seed,
    )


def _tfidf_vectorizer(model_name: str, cfg: dict[str, Any]) -> TfidfVectorizer:
    tfidf = cfg["tfidf"]
    return TfidfVectorizer(
        analyzer="char" if model_name.startswith("char4") else "word",
        ngram_range=(4, 4) if model_name.startswith("char4") else (1, 2),
        min_df=tfidf["min_df"], max_df=tfidf["max_df"],
        max_features=tfidf["max_features"], sublinear_tf=tfidf["sublinear_tf"],
        dtype=np.float32,
    )
DEFAULT_CONFIG: dict[str, Any] = {
    "c_grid": [0.1, 1.0, 10.0],
    "tfidf": {
        "min_df": 5, "max_df": 0.95, "max_features": 200_000,
        "sublinear_tf": True,
    },
    "linear": {"lr_max_iter": 2_000, "svm_max_iter": 5_000},
    "iforest": {"svd_components": 256, "n_estimators": 300, "max_samples": 4_096},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the declarative model specification, rejecting schema drift."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Expected text baseline configuration schema_version=1")
    if raw.get("experiment_id") != "horus-n2-text-baselines-v1":
        raise ValueError("Unexpected text baseline experiment_id")
    config = {name: deepcopy(raw[name]) for name in DEFAULT_CONFIG}
    models = raw.get("models", [])
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("Model list must be nonempty and unique")
    if set(models) - set(MODEL_NAMES):
        raise ValueError("Unknown model in baseline configuration")
    config["models"] = list(models)
    c_grid = config["c_grid"]
    if not c_grid or any(float(value) <= 0 for value in c_grid):
        raise ValueError("C grid must contain positive values")
    return config


class SurfaceTransformer(BaseEstimator, TransformerMixin):
    """Pure text-to-feature transform; no metadata and no fitted vocabulary."""

    def __init__(self, group: str):
        self.group = group

    def fit(self, texts: list[str], y: np.ndarray | None = None) -> "SurfaceTransformer":
        if self.group not in SURFACE_GROUPS:
            raise ValueError(f"Unknown surface group: {self.group}")
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        return surface_matrix(texts, self.group)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(SURFACE_GROUPS[self.group], dtype=object)


class HumanIsolationForest:
    """One-class human reference, with every preprocessing fit on humans only."""

    def __init__(self, config: dict[str, Any], *, seed: int):
        self.config = config
        self.seed = seed

    def fit(self, texts: list[str], labels: np.ndarray) -> "HumanIsolationForest":
        y = np.asarray(labels, dtype=int)
        if len(texts) != len(y):
            raise ValueError("Text/label length mismatch")
        human_texts = [text for text, label in zip(texts, y) if label == 0]
        if len(human_texts) < 3:
            raise ValueError("At least three human training texts are required")
        tfidf = self.config["tfidf"]
        self.vectorizer_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=tfidf["min_df"],
            max_df=tfidf["max_df"], max_features=tfidf["max_features"],
            sublinear_tf=tfidf["sublinear_tf"], dtype=np.float32,
        )
        matrix = self.vectorizer_.fit_transform(human_texts)
        dimensions = min(
            int(self.config["iforest"]["svd_components"]),
            matrix.shape[0] - 1, matrix.shape[1] - 1,
        )
        if dimensions < 1:
            raise ValueError("Too few human vocabulary features for SVD")
        self.svd_ = TruncatedSVD(n_components=dimensions, random_state=self.seed)
        reduced = self.svd_.fit_transform(matrix)
        self.forest_ = IsolationForest(
            n_estimators=int(self.config["iforest"]["n_estimators"]),
            max_samples=min(int(self.config["iforest"]["max_samples"]), len(human_texts)),
            contamination="auto", random_state=self.seed, n_jobs=1,
        )
        self.forest_.fit(reduced)
        self.human_train_rows_ = len(human_texts)
        return self

    def decision_function(self, texts: list[str]) -> np.ndarray:
        # Higher score must consistently mean more likely synthetic.
        matrix = self.vectorizer_.transform(texts)
        return -self.forest_.decision_function(self.svd_.transform(matrix))


def build_estimator(
    model_name: str, *, c: float = 1.0, seed: int = 0,
    config: dict[str, Any] | None = None,
) -> Any:
    """Construct an unfitted model whose input is a sequence of text strings."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown text model: {model_name}")
    cfg = deepcopy(DEFAULT_CONFIG if config is None else config)
    if model_name == "human_iforest":
        return HumanIsolationForest(cfg, seed=seed)
    if c <= 0:
        raise ValueError("C must be positive")
    if model_name in SURFACE_GROUPS:
        return Pipeline([
            ("surface", SurfaceTransformer(model_name)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", _linear_classifier(model_name, c, seed, cfg)),
        ])
    return Pipeline([
        ("tfidf", _tfidf_vectorizer(model_name, cfg)),
        ("classifier", _linear_classifier(model_name, c, seed, cfg)),
    ])


def synthetic_scores(estimator: Any, texts: list[str]) -> np.ndarray:
    """Return a one-dimensional score, always larger for synthetic evidence."""

    if isinstance(estimator, HumanIsolationForest):
        scores = estimator.decision_function(texts)
    elif hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(texts)[:, 1]
    else:
        scores = estimator.decision_function(texts)
    result = np.asarray(scores, dtype=np.float64).reshape(-1)
    if result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise ValueError("Invalid model score vector")
    return result


def score_texts(estimator: Any, model_name: str, texts: list[str]) -> np.ndarray:
    """Score a paired text view with a model already fitted on the N2 train.

    This is intended for counterfactual evaluation on the same document IDs;
    no fitting or threshold recalibration occurs here.
    """

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown text model: {model_name}")
    return synthetic_scores(estimator, list(texts))


def select_c_inner_cv(
    model_name: str, texts: list[str], labels: np.ndarray,
    inner_folds: np.ndarray, *, seed: int, config: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Select C on group-safe inner folds supplied by the protocol layer.

    The external caller must supply fold IDs that keep `probe_group_id` intact;
    this helper never sees outer validation or test examples.
    """

    if model_name == "human_iforest":
        raise ValueError("The one-class model has no C parameter")
    cfg = DEFAULT_CONFIG if config is None else config
    y = np.asarray(labels, dtype=int)
    folds = np.asarray(inner_folds)
    if len(texts) != len(y) or len(y) != len(folds):
        raise ValueError("Text/label/fold length mismatch")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("At least two inner folds are required")
    c_values = sorted({float(value) for value in cfg["c_grid"]})
    aucs_by_c: dict[float, list[float]] = {c: [] for c in c_values}
    # Expensive text vectorization and surface extraction occur once per inner
    # fold, not once per (fold, C) combination.
    all_surface = surface_matrix(texts, model_name) if model_name in SURFACE_GROUPS else None
    for fold in unique_folds:
        is_validation = folds == fold
        y_fit, y_validation = y[~is_validation], y[is_validation]
        if set(y_fit) != {0, 1} or set(y_validation) != {0, 1}:
            raise ValueError("Every inner train/validation partition needs both labels")
        if all_surface is not None:
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(imputer.fit_transform(all_surface[~is_validation]))
            x_validation = scaler.transform(imputer.transform(all_surface[is_validation]))
        else:
            vectorizer = _tfidf_vectorizer(model_name, cfg)
            fit_texts = [text for text, keep in zip(texts, ~is_validation) if keep]
            validation_texts = [text for text, keep in zip(texts, is_validation) if keep]
            x_fit = vectorizer.fit_transform(fit_texts)
            x_validation = vectorizer.transform(validation_texts)
        for c in c_values:
            classifier = _linear_classifier(model_name, c, seed, cfg)
            classifier.fit(x_fit, y_fit)
            scores = (classifier.predict_proba(x_validation)[:, 1]
                      if hasattr(classifier, "predict_proba")
                      else classifier.decision_function(x_validation))
            aucs_by_c[c].append(float(roc_auc_score(y_validation, scores)))
    outcomes = {str(c): float(np.mean(aucs_by_c[c])) for c in c_values}
    selected = min(
        (float(value) for value in outcomes),
        key=lambda value: (-outcomes[str(value)], value),
    )
    return selected, outcomes


def fit_score(
    model_name: str,
    train_texts: list[str], y_train: np.ndarray,
    val_texts: list[str], test_texts: list[str],
    *, seed: int, inner_folds: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one baseline and score outer validation/test without looking at labels.

    The caller owns threshold calibration and metrics. `inner_folds` contains
    the existing development-fold ID for each outer-training row.
    """

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown text model: {model_name}")
    training = list(train_texts)
    validation = list(val_texts)
    test = list(test_texts)
    y = np.asarray(y_train, dtype=int)
    if len(training) != len(y) or set(y) != {0, 1}:
        raise ValueError("Training data must have aligned texts and both binary labels")
    cfg = DEFAULT_CONFIG if config is None else config
    if model_name == "human_iforest":
        selected_c = None
        inner_auc: dict[str, float] = {}
    elif inner_folds is not None:
        selected_c, inner_auc = select_c_inner_cv(
            model_name, training, y, inner_folds, seed=seed, config=cfg,
        )
    else:
        selected_c, inner_auc = 1.0, {}
    model = build_estimator(
        model_name, c=selected_c or 1.0, seed=seed, config=cfg,
    )
    model.fit(training, y)
    return {
        "validation_scores": synthetic_scores(model, validation),
        "test_scores": synthetic_scores(model, test),
        "selected_c": selected_c,
        "inner_auc_by_c": inner_auc,
        "estimator": model,
    }


def linear_coefficients(estimator: Any) -> list[tuple[str, float]]:
    """Return signed weights for interpretation; no claim of causal features."""

    if not isinstance(estimator, Pipeline):
        return []
    names = (
        estimator.named_steps["surface"].get_feature_names_out()
        if "surface" in estimator.named_steps
        else estimator.named_steps["tfidf"].get_feature_names_out()
    )
    weights = estimator.named_steps["classifier"].coef_[0]
    if len(names) != len(weights):
        raise AssertionError("Coefficient/feature mismatch")
    return [(str(name), float(weight)) for name, weight in zip(names, weights)]
