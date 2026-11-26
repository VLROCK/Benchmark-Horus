"""Ghostbuster PT-BR adaptation with nested, train-only symbolic selection.

Reference: Verma et al., NAACL 2024; vivek3141/ghostbuster, utils/symbolic.py,
utils/n_gram.py and utils/featurize.py. Frozen Qwen 0.5B/1.5B probabilities
replace ada/davinci; human training documents replace the English Brown corpus.
The grammar, depth-three search, discount .9 and C=10 are retained. Inner-fold
ROC-AUC replaces the upstream random CV accuracy. No outer labels are read.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VECTOR_NAMES = ("strong", "weak", "trigram", "unigram")
VECTOR_OPERATIONS = ("add", "sub", "mul", "div", "gt", "lt")
SCALARS = ("max", "min", "avg", "avg_top25", "len", "var", "l2")
REQUIRED = {"record_id", "token_ids", "weak_logprobs_0", "weak_logprobs_1", "weak_token_strings"}


def _config(value: dict[str, Any]) -> dict[str, Any]:
    cfg = {"max_depth": 3, "discount": .9, "c": 10., "max_iter": 1000,
           "vocab_size": 151936, **value.get("ghostbuster", value)}
    if (int(cfg["max_depth"]) != 3 or float(cfg["discount"]) != .9 or
            float(cfg["c"]) != 10 or int(cfg["max_iter"]) < 1 or int(cfg["vocab_size"]) < 2):
        raise ValueError("Unreviewed Ghostbuster settings")
    return cfg


def _sequences(frame: pd.DataFrame) -> list[list[int]]:
    if REQUIRED - set(frame) or frame.record_id.duplicated().any():
        raise ValueError("Unique cached probability rows required for Ghostbuster")
    sequences = []
    for row in frame.itertuples(index=False):
        ids = np.asarray(row.token_ids, dtype=np.int64)
        first, second = np.asarray(row.weak_logprobs_0), np.asarray(row.weak_logprobs_1)
        if (ids.ndim != 1 or len(ids) < 2 or first.shape != (len(ids) - 1,) or
                second.shape != first.shape or len(row.weak_token_strings) != len(first) or
                not np.isfinite(first).all() or
                not np.isfinite(second).all() or np.any(first > 1e-6) or np.any(second > 1e-6)):
            raise ValueError(f"Misaligned or invalid weak-model probabilities: {row.record_id}")
        sequences.append(ids.tolist())
    return sequences


class HumanNgrams:
    """Interpolated discount/backoff with continuation unigram probabilities.

    Document boundaries have independent BOS contexts. Token probabilities are
    aligned to the same targets as the frozen causal models (all but token 0).
    The unseen unigram fallback matches the original implementation (1/V).
    """

    def __init__(self, discount: float = .9, vocab_size: int = 151936):
        self.discount, self.vocab_size = discount, vocab_size

    def fit(self, sequences: Iterable[list[int]]) -> "HumanNgrams":
        self.bigram: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        self.trigram: dict[tuple[int, ...], Counter] = defaultdict(Counter)
        predecessors: dict[int, set[int]] = defaultdict(set)
        count = 0
        for sequence in sequences:
            count += 1
            previous2, previous1 = -1, -1
            for token in sequence:
                if token < 0 or token >= self.vocab_size:
                    raise ValueError("Token ID is outside the frozen tokenizer vocabulary")
                self.bigram[(previous1,)][token] += 1
                self.trigram[(previous2, previous1)][token] += 1
                predecessors[token].add(previous1)
                previous2, previous1 = previous1, token
        if not count:
            raise ValueError("Human training documents required for n-grams")
        self.unigram_counts = {token: len(values) for token, values in predecessors.items()}
        self.continuation_total = sum(self.unigram_counts.values())
        self.totals_bigram = {context: sum(values.values()) for context, values in self.bigram.items()}
        self.totals_trigram = {context: sum(values.values()) for context, values in self.trigram.items()}
        self.document_count = count
        return self

    def unigram_probability(self, token: int) -> float:
        value = self.unigram_counts.get(token, 0)
        return value / self.continuation_total if value else 1 / self.vocab_size

    def _backoff(self, context: tuple[int, ...], token: int, lower: float,
                 table: dict, totals: dict) -> float:
        following = table.get(context)
        if not following:
            return lower
        total = totals[context]
        return (max(following.get(token, 0) - self.discount, 0) / total +
                self.discount * len(following) / total * lower)

    def vectors(self, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        unigrams, trigrams = [], []
        for index in range(1, len(tokens)):
            token = tokens[index]
            unigram = self.unigram_probability(token)
            bigram = self._backoff((tokens[index - 1],), token, unigram,
                                   self.bigram, self.totals_bigram)
            context = (tokens[index - 2] if index >= 2 else -1, tokens[index - 1])
            trigram = self._backoff(context, token, bigram, self.trigram, self.totals_trigram)
            unigrams.append(unigram)
            trigrams.append(trigram)
        return np.asarray(trigrams, dtype=np.float64), np.asarray(unigrams, dtype=np.float64)


def expressions(depth: int = 3) -> list[tuple]:
    """Port the upstream commutative pruning and maximum-depth convention.

    A depth-three expression combines at most three probability vectors and
    reduces them with a scalar operation. Subtraction/comparison orientation
    follows the published repository's ordered grammar, not a new full grammar.
    """
    if depth != 3:
        raise ValueError("Approved symbolic depth is three")
    choices = defaultdict(list)
    for first in range(len(VECTOR_NAMES)):
        for second in range(first):
            for operation in VECTOR_OPERATIONS:
                if operation != "div":
                    choices[first].append((operation, second))
    for first in range(len(VECTOR_NAMES)):
        for second in range(len(VECTOR_NAMES)):
            if first != second:
                choices[first].append(("div", second))
    result = []
    def descend(start: int, operations: tuple, last: int, current_depth: int) -> None:
        if current_depth >= depth:
            return
        result.extend((start, operations, scalar) for scalar in SCALARS)
        for operation, other in choices[last]:
            descend(start, (*operations, (operation, other)), other, current_depth + 1)
    for start in range(len(VECTOR_NAMES)):
        descend(start, (), start, 0)
    return result


def _operate(left: np.ndarray, operation: str, right: np.ndarray) -> np.ndarray:
    if operation == "add":
        return left + right
    if operation == "sub":
        return left - right
    if operation == "mul":
        return left * right
    if operation == "div":
        return np.divide(left, right, out=np.zeros_like(left, dtype=float), where=right != 0)
    if operation == "gt":
        return (left > right).astype(float)
    if operation == "lt":
        return (left < right).astype(float)
    raise ValueError("Unknown vector operation")


def _scalar(values: np.ndarray, name: str) -> float:
    if name == "max":
        return float(np.max(values))
    if name == "min":
        return float(np.min(values))
    if name == "avg":
        return float(np.mean(values))
    if name == "avg_top25":
        return float(np.sort(values)[-25:].mean())
    if name == "len":
        return float(len(values))
    if name == "var":
        return float(np.var(values))
    if name == "l2":
        return float(np.linalg.norm(values))
    raise ValueError("Unknown scalar operation")


def _handcrafted(strong: np.ndarray, weak: np.ndarray, pieces: list[str]) -> np.ndarray:
    """Retain the seven upstream handcrafted summaries, including its padding.

    The source code applies these to probability vectors and recognizes GPT
    byte-BPE's Ġ word marker. Qwen uses the same marker; feature metadata records
    this adaptation. The upstream >3 probability outlier feature is retained
    literally rather than silently reinterpreted as negative log-probability.
    """
    outliers = list(strong[strong > 3])
    outlier_count = len(outliers)
    outliers += [0.] * max(0, 50 - len(outliers))
    differences = sorted(strong - weak, reverse=True)
    differences += [0.] * max(0, 50 - len(differences))
    lengths, current = [], 0
    for piece in pieces:
        if piece.startswith("Ġ"):
            lengths.append(current)
            current = 1
        else:
            current += 1
    # Upstream omits the final incomplete count; retain that convention.
    lengths = sorted(lengths, reverse=True)
    lengths += [0.] * max(0, 50 - len(lengths))
    return np.asarray([outlier_count, np.mean(outliers[:25]), np.mean(outliers[25:50]),
                       np.mean(differences[:25]), np.mean(differences[25:]),
                       np.mean(lengths[:25]), np.mean(lengths[25:])])


def feature_matrix(frame: pd.DataFrame, ngrams: HumanNgrams,
                   selected: list[tuple] | None = None) -> np.ndarray:
    sequences = _sequences(frame)
    grammar = expressions() if selected is None else selected
    result = np.empty((len(frame), 7 + len(grammar)), dtype=np.float32)
    for position, (row, tokens) in enumerate(zip(frame.itertuples(index=False), sequences)):
        strong, weak = np.exp(np.asarray(row.weak_logprobs_1, dtype=np.float64)), np.exp(
            np.asarray(row.weak_logprobs_0, dtype=np.float64))
        trigram, unigram = ngrams.vectors(tokens)
        vectors = (strong, weak, trigram, unigram)
        pieces = list(getattr(row, "weak_token_strings", []))
        result[position, :7] = _handcrafted(strong, weak, pieces)
        cache = {}
        for j, (start, operations, scalar) in enumerate(grammar):
            key = (start, operations)
            if key not in cache:
                values = vectors[start]
                for operation, other in operations:
                    values = _operate(values, operation, vectors[other])
                cache[key] = values
            result[position, j + 7] = _scalar(cache[key], scalar)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Ghostbuster symbolic feature; no silent clipping")
    return result


def _pipeline(cfg: dict, seed: int) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(C=float(cfg["c"]), penalty="l2",
                                                        max_iter=int(cfg["max_iter"]),
                                                        solver="liblinear", random_state=seed))])


def _ngram_fit(frame: pd.DataFrame, cfg: dict) -> HumanNgrams:
    humans = frame.loc[frame.label_name.eq("human")].sort_values("record_id")
    return HumanNgrams(float(cfg["discount"]), int(cfg["vocab_size"])).fit(_sequences(humans))


@dataclass
class GhostbusterEstimator:
    ngrams: HumanNgrams
    selected_expressions: list[tuple]
    classifier: Pipeline

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(feature_matrix(
            frame, self.ngrams, self.selected_expressions))[:, 1]


def fit_ghostbuster(train: pd.DataFrame, validation: pd.DataFrame,
                    test: pd.DataFrame, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Select grammar features with nested train-only n-grams and scaling.

    Outer validation/test frames are not inspected until the final model has
    been fitted. Their labels are never accessed in this function.
    """
    config = _config(cfg)
    _sequences(train)
    _sequences(validation)
    _sequences(test)
    partitions = (train, validation, test)
    identities = [set(frame.record_id) for frame in partitions]
    if identities[0] & (identities[1] | identities[2]) or identities[1] & identities[2]:
        raise ValueError("Train, validation and test IDs overlap")
    if all("probe_group_id" in frame for frame in partitions):
        if any(frame.probe_group_id.isna().any() for frame in partitions):
            raise ValueError("Missing linked group identity")
        groups = [set(frame.probe_group_id) for frame in partitions]
        if groups[0] & (groups[1] | groups[2]) or groups[1] & groups[2]:
            raise ValueError("Linked groups cross train/validation/test")
    if {"fold", "label_name"} - set(train) or set(train.label_name) != {"human", "synthetic"}:
        raise ValueError("Training labels and frozen inner fold IDs required")
    if "probe_group_id" in train:
        grouped = train.groupby("probe_group_id")
        if train.probe_group_id.isna().any() or grouped.fold.nunique().gt(1).any():
            raise ValueError("Linked training group crosses inner folds")
        if grouped.label_name.nunique().gt(1).any():
            raise ValueError("Linked training group spans labels")
    folds = sorted(train.fold.unique())
    if len(folds) < 2:
        raise ValueError("At least two group-safe internal folds required")
    grammar = expressions(int(config["max_depth"]))
    inner = []
    # Each fold uses an independent human reference and fitted scaler. In
    # particular, even unsupervised n-gram counts do not see held-out texts.
    for fold in folds:
        fitted = train.loc[train.fold.ne(fold)].reset_index(drop=True)
        held = train.loc[train.fold.eq(fold)].reset_index(drop=True)
        if any(set(part.label_name) != {"human", "synthetic"} for part in (fitted, held)):
            raise ValueError("Each inner fold requires both classes")
        ngrams = _ngram_fit(fitted, config)
        inner.append((feature_matrix(fitted, ngrams, grammar),
                      fitted.label_name.eq("synthetic").to_numpy(dtype=int),
                      feature_matrix(held, ngrams, grammar),
                      held.label_name.eq("synthetic").to_numpy(dtype=int)))
    selected, remaining, history = [], list(range(len(grammar))), []
    def evaluate(columns: list[int], *, handcrafted: bool = False) -> float:
        # Upstream selects symbolic expressions first, then appends its seven
        # handcrafted summaries to the final classifier. Keep that order.
        included = ([*range(7)] if handcrafted else []) + [7 + index for index in columns]
        values = []
        for xfit, yfit, xheld, yheld in inner:
            model = _pipeline(config, seed)
            model.fit(xfit[:, included], yfit)
            values.append(roc_auc_score(yheld, model.predict_proba(xheld[:, included])[:, 1]))
        return float(np.mean(values))
    baseline = evaluate([], handcrafted=True)
    current = 0.0
    while remaining:
        best_index, best_value = None, current
        for index in remaining:
            value = evaluate([*selected, index])
            # Deterministic strict comparison; equal candidates retain the
            # first published grammar order. No outer-score-driven budget cap.
            if value > best_value + 1e-12:
                best_index, best_value = index, value
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        history.append({"expression_index": best_index, "inner_roc_auc": best_value})
        current = best_value
    ngrams = _ngram_fit(train, config)
    chosen = [grammar[index] for index in selected]
    classifier = _pipeline(config, seed)
    classifier.fit(feature_matrix(train, ngrams, chosen), train.label_name.eq("synthetic"))
    estimator = GhostbusterEstimator(ngrams, chosen, classifier)
    ids = "".join(f"{value}\n" for value in sorted(train.record_id.astype(str)))
    return {"estimator": estimator, "validation_scores": estimator.score_frame(validation),
            "test_scores": estimator.score_frame(test),
            "metadata": {"method": "ghostbuster_ptbr_adapted", "config": config,
                         "candidate_features": len(grammar), "selected_features": selected,
                         "selected_expressions": chosen, "forward_selection": history,
                         "handcrafted_inner_auc": baseline, "selected_inner_auc": current,
                         "human_ngram_train_documents": ngrams.document_count,
                         "train_ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
                         "adaptations": ["Qwen 0.5B/1.5B replace ada/davinci",
                                         "human fold training data replace Brown Corpus",
                                         "document boundaries reset BOS context",
                                         "nested frozen inner folds and ROC-AUC selection",
                                         "train-only scaling replaces global normalization"]}}
