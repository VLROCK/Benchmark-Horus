"""Classical PT-BR detectors and the explicitly adapted ALTA 2024 ensemble.

Reference: Thomas, Hoang and Mitchell (2024), https://aclanthology.org/2024.alta-1.19/.
The original task was sentence attribution in English hybrid documents. This
module supplies document/sentence adapters, Portuguese readability and UPOS,
training-only references and group-safe inner selection. It is not an exact
reproduction of the unpublished original implementation.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from horus_benchmark.features.stylometry import (
    FEATURE_VERSION, FUNCTION_WORD_VERSION, READABILITY_NAMES, STYLE_NAMES,
    STRUCTURE_NAMES, UPOS, pos_frequencies, readability, stylometry, words,
)


MODEL_NAMES = (
    "stylo_lr", "stylo_svm", "stylo_lr_no_structure", "stylo_svm_no_structure",
    "alta_nb", "alta_delta", "alta_readability", "alta_pos", "alta_entropy",
    "alta_vote",
)
ALTA_COMPONENTS = ("alta_nb", "alta_delta", "alta_readability", "alta_pos", "alta_entropy")
DEFAULT_CONFIG = {
    "c_grid": [0.1, 1.0, 10.0], "delta_words": [100, 500, 1000],
    "tfidf": {"min_df": 5, "max_df": .95, "max_features": 200_000,
              "sublinear_tf": True, "ngram_range": [1, 2]},
    "nb_alpha": 1.0, "readability_pca": 7, "readability_neighbors": 5,
    "pos_neighbors": 3, "entropy_history_chars": 2_000_000,
    "lr_max_iter": 2000, "svm_max_iter": 5000,
    "without_structure": False,
}


def _config(value: dict | None) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if value is not None:
        for name in cfg:
            if name in value:
                cfg[name] = deepcopy(value[name])
    if (not cfg["c_grid"] or any(float(c) <= 0 for c in cfg["c_grid"]) or
            not cfg["delta_words"] or any(int(n) < 1 for n in cfg["delta_words"]) or
            int(cfg["entropy_history_chars"]) < 2 or float(cfg["nb_alpha"]) <= 0 or
            min(int(cfg[name]) for name in ("readability_pca", "readability_neighbors",
                                            "pos_neighbors")) < 1):
        raise ValueError("Invalid classical detector configuration")
    return cfg


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if "label_name" not in frame or set(frame.label_name) - {"human", "synthetic"}:
        raise ValueError("Expected human/synthetic training labels")
    return frame.label_name.eq("synthetic").to_numpy(dtype=np.int8)


def _frame(frame: pd.DataFrame, *, training: bool = False) -> pd.DataFrame:
    if ({"record_id", "content"} - set(frame) or frame.record_id.duplicated().any() or
            frame.content.isna().any() or not frame.content.map(
                lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("Detector frame requires unique IDs and nonempty text")
    if training:
        if {"fold", "label_name"} - set(frame) or set(_labels(frame)) != {0, 1}:
            raise ValueError("Training frame needs folds and both labels")
        if frame.fold.isna().any() or frame.fold.lt(0).any():
            raise ValueError("Training fold IDs must be nonnegative")
        if "probe_group_id" in frame:
            if frame.probe_group_id.isna().any() or frame.groupby(
                    "probe_group_id").fold.nunique().gt(1).any():
                raise ValueError("A linked training group crosses inner folds")
    return frame.reset_index(drop=True)


def _inner(frame: pd.DataFrame):
    folds = sorted(frame.fold.unique())
    if len(folds) < 2:
        raise ValueError("At least two group-safe inner folds are required")
    for fold in folds:
        fit = frame.loc[frame.fold.ne(fold)].reset_index(drop=True)
        held = frame.loc[frame.fold.eq(fold)].reset_index(drop=True)
        if set(_labels(fit)) != {0, 1} or set(_labels(held)) != {0, 1}:
            raise ValueError("An inner training/validation fold lacks a class")
        yield fit, held


def _numeric(frame: pd.DataFrame, kind: str, *, without_structure: bool = False
             ) -> np.ndarray:
    if kind in {"pos", "style"} and "upos" not in frame:
        raise ValueError("Pinned Portuguese UPOS annotations are required")
    if kind == "readability":
        rows = [readability(text) for text in frame.content]
    elif kind == "pos":
        rows = [pos_frequencies(tags) for tags in frame.upos]
    elif kind == "style":
        rows = [stylometry(text, tags, without_structure=without_structure)
                for text, tags in zip(frame.content, frame.upos)]
    else:
        raise ValueError("Unknown numeric feature family")
    if not rows:
        dimension = (len(READABILITY_NAMES) if kind == "readability" else len(UPOS)
                     if kind == "pos" else len(STYLE_NAMES) -
                     (len(STRUCTURE_NAMES) if without_structure else 0))
        return np.empty((0, dimension), dtype=float)
    return np.asarray(rows, dtype=float)


class FrameEstimator:
    """An estimator which inspects only content/UPOS when scoring a frame."""

    def __init__(self, model: Any, family: str, *, without_structure: bool = False):
        self.model = model
        self.family = family
        self.without_structure = without_structure

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _frame(frame)
        if frame.empty:
            return np.asarray([], dtype=float)
        values = (frame.content.tolist() if self.family == "text" else
                  _numeric(frame, self.family, without_structure=self.without_structure))
        if hasattr(self.model, "predict_proba"):
            classes = self.model.classes_
            scores = self.model.predict_proba(values)[:, int(np.flatnonzero(classes == 1)[0])]
        else:
            scores = self.model.decision_function(values)
        result = np.asarray(scores, dtype=float).reshape(-1)
        if len(result) != len(frame) or not np.isfinite(result).all():
            raise ValueError("Invalid classical detector scores")
        return result

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        cut = .5 if hasattr(self.model, "predict_proba") else 0.0
        return (self.score_frame(frame) > cut).astype(np.int8)


class BurrowsDelta:
    """Training-only frequent words, z-scores and class stylistic profiles."""

    def __init__(self, max_words: int):
        self.max_words = max_words

    def _matrix(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=float)
        for index, text in enumerate(texts):
            tokens = words(text)
            counts = Counter(tokens)
            matrix[index] = [counts[word] / max(len(tokens), 1) for word in self.vocabulary]
        return matrix

    def fit(self, texts: list[str], labels: np.ndarray):
        counts = Counter(word for text in texts for word in words(text))
        self.vocabulary = [item[0] for item in sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[:self.max_words]]
        if not self.vocabulary:
            raise ValueError("Burrows Delta has no training vocabulary")
        matrix = self._matrix(texts)
        self.mean = matrix.mean(axis=0)
        self.scale = matrix.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        z = (matrix - self.mean) / self.scale
        self.profiles = np.asarray([z[labels == label].mean(axis=0) for label in (0, 1)])
        return self

    def decision_function(self, texts: list[str]) -> np.ndarray:
        z = (self._matrix(texts) - self.mean) / self.scale
        # Positive means nearer the synthetic profile; the mean is equivalent
        # to Burrows' sum for a fixed vocabulary size and is easier to inspect.
        human = np.abs(z - self.profiles[0]).mean(axis=1)
        synthetic = np.abs(z - self.profiles[1]).mean(axis=1)
        return human - synthetic


class SuffixAutomaton:
    """Exact substring matching, O(history + query) expected time.

    A reversed history allows a left-to-right scan of the reversed query to
    recover the longest matching prefix at every original query position.
    The integer -1 separates documents and can never appear in a Unicode query.
    """

    def __init__(self, documents: list[str]):
        self.next: list[dict[int, int]] = [{}]
        self.link = [-1]
        self.length = [0]
        self.last = 0
        for doc in reversed(documents):
            for char in reversed(doc):
                self._extend(ord(char))
            self._extend(-1)

    def _extend(self, char: int) -> None:
        current = len(self.next)
        self.next.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)
        previous = self.last
        while previous >= 0 and char not in self.next[previous]:
            self.next[previous][char] = current
            previous = self.link[previous]
        if previous >= 0:
            target = self.next[previous][char]
            if self.length[previous] + 1 == self.length[target]:
                self.link[current] = target
            else:
                clone = len(self.next)
                self.next.append(self.next[target].copy())
                self.length.append(self.length[previous] + 1)
                self.link.append(self.link[target])
                while previous >= 0 and self.next[previous].get(char) == target:
                    self.next[previous][char] = clone
                    previous = self.link[previous]
                self.link[target] = self.link[current] = clone
        self.last = current

    def longest_matches(self, text: str) -> np.ndarray:
        output = np.zeros(len(text), dtype=np.int64)
        state = matched = 0
        for index in range(len(text) - 1, -1, -1):
            char = ord(text[index])
            while state and char not in self.next[state]:
                state = self.link[state]
                matched = min(matched, self.length[state])
            target = self.next[state].get(char)
            if target is None:
                state = matched = 0
            else:
                state = target
                matched += 1
            output[index] = matched
        return output


class CrossParsedEntropy:
    def __init__(self, history_chars: int, seed: int):
        self.history_chars = history_chars
        self.seed = seed

    def fit_frame(self, frame: pd.DataFrame):
        self.automata = []
        self.histories = []
        self.history_ids = []
        for label in ("human", "synthetic"):
            subset = frame.loc[frame.label_name.eq(label)].copy()
            subset["history_key"] = subset.record_id.map(lambda value: hashlib.sha256(
                f"{self.seed}|entropy|{value}".encode()).hexdigest())
            subset.sort_values(["history_key", "record_id"], kind="stable", inplace=True)
            documents, ids = [], []
            remaining = self.history_chars
            for row in subset.itertuples(index=False):
                if remaining == 0:
                    break
                part = row.content[:remaining]
                documents.append(part)
                ids.append(str(row.record_id))
                remaining -= len(part)
            if not documents:
                raise ValueError("Entropy history lacks a class")
            self.histories.append(self.history_chars - remaining)
            self.history_ids.append(ids)
            self.automata.append(SuffixAutomaton(documents))
        return self

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _frame(frame)
        result = []
        for text in frame.content:
            rates = []
            for model in self.automata:
                matches = model.longest_matches(text)
                # Eq. (1), Thomas et al.: sum over the first n-1 start
                # positions. An unmatched position receives unit length to
                # make finite-sample estimates defined for unseen characters.
                n = len(text)
                rate = (n * math.log2(max(n - 1, 2)) /
                        max(float(np.maximum(matches[:-1], 1).sum()), 1.0))
                rates.append(rate)
            result.append(rates[0] - rates[1])
        return np.asarray(result, dtype=float)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score_frame(frame) > 0).astype(np.int8)


def _fit_fixed(name: str, frame: pd.DataFrame, cfg: dict, seed: int,
               parameter: float | int | None = None):
    labels = _labels(frame)
    if name == "alta_entropy":
        return CrossParsedEntropy(int(cfg["entropy_history_chars"]), seed).fit_frame(frame)
    if name == "alta_delta":
        model = BurrowsDelta(int(parameter)).fit(frame.content.tolist(), labels)
        return FrameEstimator(model, "text")
    if name == "alta_nb":
        opts = cfg["tfidf"]
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=tuple(opts.get("ngram_range", [1, 2])),
                                     min_df=opts["min_df"], max_df=opts["max_df"],
                                     max_features=opts["max_features"],
                                     sublinear_tf=opts["sublinear_tf"], dtype=np.float32)),
            ("classifier", MultinomialNB(alpha=float(cfg["nb_alpha"]))),
        ])
        model.fit(frame.content.tolist(), labels)
        return FrameEstimator(model, "text")
    if name in {"alta_readability", "alta_pos"}:
        family = "readability" if name == "alta_readability" else "pos"
        matrix = _numeric(frame, family)
        neighbors = int(cfg["readability_neighbors"] if family == "readability"
                        else cfg["pos_neighbors"])
        if len(frame) < neighbors:
            raise ValueError("Too few training rows for the declared KNN neighbors")
        steps = []
        if family == "readability":
            dimensions = min(int(cfg["readability_pca"]), matrix.shape[1], len(frame) - 1)
            if dimensions < 1:
                raise ValueError("Too few training rows for readability PCA")
            steps = [("scaler", StandardScaler()),
                     ("pca", PCA(n_components=dimensions, svd_solver="full"))]
        steps.append(("classifier", KNeighborsClassifier(n_neighbors=neighbors,
                                                         weights="uniform", n_jobs=1)))
        model = Pipeline(steps).fit(matrix, labels)
        return FrameEstimator(model, family)
    if name.startswith("stylo_"):
        without_structure = bool(cfg["without_structure"] or name.endswith("_no_structure"))
        matrix = _numeric(frame, "style", without_structure=without_structure)
        classifier = (LinearSVC(C=float(parameter), class_weight="balanced", dual="auto",
                                max_iter=int(cfg["svm_max_iter"]), random_state=seed)
                      if "svm" in name else
                      LogisticRegression(C=float(parameter), class_weight="balanced",
                                         solver="liblinear", max_iter=int(cfg["lr_max_iter"]),
                                         random_state=seed))
        model = Pipeline([("scaler", StandardScaler()),
                          ("classifier", classifier)]).fit(matrix, labels)
        return FrameEstimator(model, "style", without_structure=without_structure)
    raise ValueError(f"Unknown individual classical model: {name}")


def _fit_tuned(name: str, frame: pd.DataFrame, cfg: dict, seed: int):
    candidates = (sorted({float(c) for c in cfg["c_grid"]}) if name.startswith("stylo_")
                  else sorted({int(n) for n in cfg["delta_words"]})
                  if name == "alta_delta" else [None])
    outcomes = {}
    if len(candidates) > 1:
        collected = {str(candidate): [] for candidate in candidates}
        for fit, held in _inner(frame):
            # Extract the deterministic dense measurements once per split,
            # not once per C. Scaling is still fitted anew on inner train.
            without_structure = bool(cfg["without_structure"] or name.endswith("_no_structure"))
            if name.startswith("stylo_"):
                fit_values = _numeric(fit, "style", without_structure=without_structure)
                held_values = _numeric(held, "style", without_structure=without_structure)
                scaler = StandardScaler().fit(fit_values)
                fit_values, held_values = scaler.transform(fit_values), scaler.transform(held_values)
            for candidate in candidates:
                if name.startswith("stylo_"):
                    classifier = (LinearSVC(C=float(candidate), class_weight="balanced",
                                            dual="auto", max_iter=int(cfg["svm_max_iter"]),
                                            random_state=seed) if "svm" in name else
                                  LogisticRegression(C=float(candidate), class_weight="balanced",
                                                     solver="liblinear", max_iter=int(cfg["lr_max_iter"]),
                                                     random_state=seed))
                    classifier.fit(fit_values, _labels(fit))
                    scores = (classifier.predict_proba(held_values)[:, 1]
                              if hasattr(classifier, "predict_proba") else
                              classifier.decision_function(held_values))
                else:
                    fitted = _fit_fixed(name, fit, cfg, seed, candidate)
                    scores = fitted.score_frame(held)
                collected[str(candidate)].append(float(roc_auc_score(_labels(held), scores)))
        outcomes = {key: float(np.mean(values)) for key, values in collected.items()}
        chosen = min(candidates, key=lambda candidate: (-outcomes[str(candidate)], candidate))
    else:
        chosen = candidates[0]
    fitted = _fit_fixed(name, frame, cfg, seed, chosen)
    details = {"selected_parameter": chosen, "inner_auc": outcomes}
    if name == "alta_entropy":
        details.update({
            "history_chars_by_label": dict(zip(("human", "synthetic"), fitted.histories)),
            "history_ids_sha256_by_label": {
                label: hashlib.sha256("".join(f"{rid}\n" for rid in ids).encode()).hexdigest()
                for label, ids in zip(("human", "synthetic"), fitted.history_ids)},
        })
    if name == "alta_delta":
        details.update({"vocabulary_size": len(fitted.model.vocabulary),
                        "vocabulary_sha256": hashlib.sha256("\n".join(
                            fitted.model.vocabulary).encode()).hexdigest()})
    return fitted, details


class WeightedVote:
    def __init__(self, models: dict[str, Any], weights: dict[str, float]):
        self.models, self.weights = models, weights

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _frame(frame)
        scores = np.zeros(len(frame), dtype=float)
        for name, model in self.models.items():
            scores += self.weights[name] * model.predict_frame(frame)
        return scores

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score_frame(frame) > .5).astype(np.int8)


def _fit_vote(frame: pd.DataFrame, cfg: dict, seed: int):
    # Nested tuning is repeated within each OOF training subset. The held OOF
    # fold cannot affect vocabulary, C/delta selection, references or weights.
    if len(frame.fold.unique()) < 3:
        raise ValueError("ALTA vote requires at least three folds for nested OOF weights")
    predictions = {name: np.zeros(len(frame), dtype=np.int8) for name in ALTA_COMPONENTS}
    for fold in sorted(frame.fold.unique()):
        is_held = frame.fold.eq(fold).to_numpy()
        fit = frame.loc[~is_held].reset_index(drop=True)
        held = frame.loc[is_held].reset_index(drop=True)
        if set(_labels(fit)) != {0, 1} or set(_labels(held)) != {0, 1}:
            raise ValueError("OOF vote fold lacks a class")
        for name in ALTA_COMPONENTS:
            model, _ = _fit_tuned(name, fit, cfg, seed)
            predictions[name][is_held] = model.predict_frame(held)
    kappas = {name: float(cohen_kappa_score(_labels(frame), values))
              for name, values in predictions.items()}
    weights = {name: max(kappa, 0.0) if np.isfinite(kappa) else 0.0
               for name, kappa in kappas.items()}
    total = sum(weights.values())
    weights = ({name: value / total for name, value in weights.items()} if total else
               {name: 1 / len(ALTA_COMPONENTS) for name in ALTA_COMPONENTS})
    models, selections = {}, {}
    for name in ALTA_COMPONENTS:
        models[name], selections[name] = _fit_tuned(name, frame, cfg, seed)
    return WeightedVote(models, weights), {
        "weights": weights, "oof_kappa": kappas,
        "component_selections": selections,
        "weight_note": "Nested group-safe OOF hard predictions; no outer validation labels used",
    }


def fit_classical(name: str, train_frame: pd.DataFrame, validation_frame: pd.DataFrame,
                   test_frame: pd.DataFrame, cfg: dict | None = None,
                   seed: int = 20261105) -> dict[str, Any]:
    """Fit within the supplied train and score caller-ordered validation/test.

    External validation and test label columns are never inspected. Every
    vocabulary, reference, scaling/PCA and hyperparameter selection uses train.
    Caller supplies frozen ``fold`` IDs which keep linked documents together.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown classical adapted detector: {name}")
    train = _frame(train_frame, training=True)
    validation, test = _frame(validation_frame), _frame(test_frame)
    if set(train.record_id) & (set(validation.record_id) | set(test.record_id)) or \
            set(validation.record_id) & set(test.record_id):
        raise ValueError("Train, validation and test IDs overlap")
    for column in ("probe_group_id",):
        if all(column in frame for frame in (train, validation, test)):
            groups = [set(frame[column]) for frame in (train, validation, test)]
            if groups[0] & (groups[1] | groups[2]) or groups[1] & groups[2]:
                raise ValueError("Linked groups cross train/validation/test")
    config = _config(cfg)
    estimator, selection = (_fit_vote(train, config, seed) if name == "alta_vote" else
                             _fit_tuned(name, train, config, seed))
    metadata = {
        "name": name, "adaptation": "ptbr_document_alta2024_v1",
        "seed": int(seed), "feature_version": FEATURE_VERSION,
        "function_word_version": FUNCTION_WORD_VERSION,
        "syllable_note": "Pyphen pt_BR hyphenation proxy, not gold phonological syllables",
        "entropy_note": "Eq1 cross-parsed character matching; unseen lengths floored at one; histories training-only",
        "config": config, "selection": selection,
        "training_rows": len(train), "training_id_sha256": hashlib.sha256(
            "".join(f"{rid}\n" for rid in sorted(train.record_id)).encode()).hexdigest(),
    }
    return {"estimator": estimator,
            "validation_scores": estimator.score_frame(validation),
            "test_scores": estimator.score_frame(test), "metadata": metadata}
