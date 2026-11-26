"""Ghostbuster grammar, conditional contexts and train-only selection fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.detectors import ghostbuster_adapted as module


def _frame(prefix="train", labels=True):
    rows = []
    for fold in range(2):
        for label in ("human", "synthetic"):
            for index in range(3):
                probability = .2 if label == "human" else .8
                rows.append({"record_id": f"{prefix}-{fold}-{label}-{index}",
                             "fold": fold, "label_name": label,
                             "token_ids": [1, 2, 3, 2],
                             "weak_logprobs_0": np.log([.3, .4, .3]),
                             "weak_logprobs_1": np.log([probability] * 3),
                             "weak_token_strings": ["Ġuma", "Ġfrase", "."]})
    result = pd.DataFrame(rows)
    return result if labels else result.drop(columns="label_name")


def test_published_symbolic_grammar_has_2534_depth_three_features() -> None:
    grammar = module.expressions(3)
    assert len(grammar) == 2534
    assert len(set(grammar)) == len(grammar)
    assert max(len(operations) for _, operations, _ in grammar) == 2


def test_ngram_reference_does_not_match_across_document_boundaries() -> None:
    reference = module.HumanNgrams(vocab_size=10).fit([[1, 2], [3, 4]])
    assert (2, 3) not in reference.trigram
    assert reference.bigram.get((2,), {}).get(3, 0) == 0
    assert reference.document_count == 2
    tri, uni = reference.vectors([1, 2, 3])
    assert tri.shape == uni.shape == (2,)
    assert np.isfinite(tri).all() and (tri > 0).all()


def test_feature_matrix_requires_full_token_alignment() -> None:
    frame = _frame()
    reference = module.HumanNgrams(vocab_size=10).fit([[1, 2, 3]])
    values = module.feature_matrix(frame, reference, [(0, (), "avg")])
    assert values.shape == (12, 8)
    frame.at[0, "token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="Misaligned"):
        module.feature_matrix(frame, reference, [])


def test_inner_ngram_fit_excludes_held_fold_and_outer_labels(monkeypatch) -> None:
    seen = []
    actual = module._ngram_fit

    def fit_reference(frame, cfg):
        seen.append(set(frame.record_id))
        return actual(frame, cfg)

    monkeypatch.setattr(module, "_ngram_fit", fit_reference)
    monkeypatch.setattr(module, "expressions", lambda depth=3: [(0, (), "avg")])
    train = _frame()
    validation = _frame("validation", labels=False)
    test = _frame("test", labels=False)
    result = module.fit_ghostbuster(train, validation, test, {"vocab_size": 10}, 7)
    assert len(result["validation_scores"]) == len(validation)
    assert len(result["test_scores"]) == len(test)
    assert seen[0] == set(train.loc[train.fold.eq(1), "record_id"])
    assert seen[1] == set(train.loc[train.fold.eq(0), "record_id"])
    assert seen[2] == set(train.record_id)
    assert not any(any(value.startswith("test") or value.startswith("validation") for value in ids) for ids in seen)


def test_ghostbuster_rejects_overlapping_ids_before_model_selection() -> None:
    train, validation, test = _frame(), _frame("validation", labels=False), _frame("test", labels=False)
    validation.loc[0, "record_id"] = train.loc[0, "record_id"]
    with pytest.raises(ValueError, match="IDs overlap"):
        module.fit_ghostbuster(train, validation, test, {"vocab_size": 10}, 7)


def test_ghostbuster_rejects_group_crossing_partitions_or_internal_folds() -> None:
    train, validation, test = _frame(), _frame("validation", labels=False), _frame("test", labels=False)
    for frame in (train, validation, test):
        frame["probe_group_id"] = frame.record_id
    validation.loc[0, "probe_group_id"] = train.loc[0, "probe_group_id"]
    with pytest.raises(ValueError, match="groups cross"):
        module.fit_ghostbuster(train, validation, test, {"vocab_size": 10}, 7)
    validation["probe_group_id"] = validation.record_id
    one = train.index[train.fold.eq(0)][0]
    two = train.index[train.fold.eq(1)][0]
    train.loc[two, "probe_group_id"] = train.loc[one, "probe_group_id"]
    with pytest.raises(ValueError, match="crosses inner folds"):
        module.fit_ghostbuster(train, validation, test, {"vocab_size": 10}, 7)
