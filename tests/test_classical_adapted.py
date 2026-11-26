"""Small scientific invariants for PT-BR classical adapters, no corpus or GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.detectors.classical_adapted import (
    BurrowsDelta, CrossParsedEntropy, SuffixAutomaton, fit_classical,
)
from horus_benchmark.features.stylometry import (
    READABILITY_NAMES, STYLE_NAMES, STRUCTURE_NAMES, mattr, pos_frequencies,
    readability, stylometry,
)


def _frame(prefix: str, n: int = 24) -> pd.DataFrame:
    rows = []
    for index in range(n):
        label = "human" if index % 2 == 0 else "synthetic"
        text = ("Eu escrevi uma mensagem, com ideias e dúvidas. Outra frase simples!"
                if label == "human" else
                "A análise apresenta resultados claros. Este estudo propõe métodos sistemáticos.")
        rows.append({"record_id": f"{prefix}-{index}", "content": text,
                     "label_name": label, "fold": (index // 2) % 4,
                     "probe_group_id": f"{prefix}-g-{index}",
                     "upos": ["PRON", "VERB", "DET", "NOUN", "PUNCT"]})
    return pd.DataFrame(rows)


SMALL = {"c_grid": [1.0], "delta_words": [8], "entropy_history_chars": 500,
         "tfidf": {"min_df": 1, "max_df": 1.0, "max_features": 100,
                   "sublinear_tf": True, "ngram_range": [1, 2]}}


def _brute_match(text: str, histories: list[str]) -> list[int]:
    result = []
    for start in range(len(text)):
        longest = 0
        for end in range(start + 1, len(text) + 1):
            if any(text[start:end] in history for history in histories):
                longest = end - start
            else:
                break
        result.append(longest)
    return result


@pytest.mark.parametrize("histories,text", [
    (["abracadabra", "uma frase"], "cadabra uma frase!"),
    (["ab", "cd"], "abcd"),
    (["ação", "coração"], "ação e coração"),
    (["aaaa", "abab"], "aaababa"),
    (["xy"], "zzz"),
])
def test_suffix_automaton_matches_brute_force_without_cross_document_matches(histories, text):
    assert SuffixAutomaton(histories).longest_matches(text).tolist() == _brute_match(text, histories)


def test_burrows_training_vocabulary_and_positive_score_direction():
    model = BurrowsDelta(4).fit(["humano humano real", "natural humano",
                                "artificial sistema", "sistema artificial"],
                               np.asarray([0, 0, 1, 1]))
    vocabulary = model.vocabulary.copy()
    score = model.decision_function(["humano humano", "sistema artificial", "inédito"])
    assert score[0] < 0 < score[1]
    assert model.vocabulary == vocabulary
    assert "inédito" not in model.vocabulary


def test_style_features_are_finite_and_structure_ablation_has_named_columns():
    text = "Eu escrevo uma frase curta.\n\nOutra mensagem contém palavras!"
    tags = ["PRON", "VERB", "DET", "NOUN", "ADJ", "PUNCT"]
    assert readability(text).shape == (len(READABILITY_NAMES),)
    assert np.isfinite(stylometry(text, tags)).all()
    assert stylometry(text, tags).shape == (len(STYLE_NAMES),)
    assert stylometry(text, tags, without_structure=True).shape == (
        len(STYLE_NAMES) - len(STRUCTURE_NAMES),)
    assert mattr(["a", "b", "a", "b"], 2) == 1.0
    with pytest.raises(ValueError, match="Unknown UPOS"):
        pos_frequencies(["NOT_A_TAG"])


@pytest.mark.parametrize("name", ["alta_nb", "alta_delta", "alta_pos", "alta_entropy"])
def test_external_labels_do_not_affect_fitted_scores(name):
    train, val, test = _frame("train"), _frame("val", 8), _frame("test", 8)
    first = fit_classical(name, train, val, test, SMALL, seed=19)
    val["label_name"] = "unknown"
    test["label_name"] = "also unknown"
    second = fit_classical(name, train, val, test, SMALL, seed=19)
    np.testing.assert_allclose(first["test_scores"], second["test_scores"])
    np.testing.assert_allclose(first["validation_scores"], second["validation_scores"])
    assert first["metadata"]["training_id_sha256"] == second["metadata"]["training_id_sha256"]


def test_entropy_histories_stay_inside_training_and_respect_budget():
    train = _frame("train")
    model = CrossParsedEntropy(120, 19).fit_frame(train)
    assert all(value <= 120 for value in model.histories)
    assert all(set(ids) <= set(train.record_id) for ids in model.history_ids)
    assert np.isfinite(model.score_frame(_frame("held", 4))).all()


def test_linked_group_cannot_cross_inner_or_outer_partitions():
    train, val, test = _frame("train"), _frame("val", 8), _frame("test", 8)
    train.loc[2, "probe_group_id"] = train.loc[0, "probe_group_id"]
    with pytest.raises(ValueError, match="crosses inner folds"):
        fit_classical("alta_nb", train, val, test, SMALL)
    train = _frame("train")
    val.loc[0, "probe_group_id"] = train.loc[0, "probe_group_id"]
    with pytest.raises(ValueError, match="Linked groups cross"):
        fit_classical("alta_nb", train, val, test, SMALL)


def test_vote_weights_come_from_internal_predictions_and_sum_to_one():
    train, val, test = _frame("train"), _frame("val", 8), _frame("test", 8)
    fit = fit_classical("alta_vote", train, val.drop(columns="label_name"),
                         test.drop(columns="label_name"), SMALL, seed=19)
    weights = fit["metadata"]["selection"]["weights"]
    assert set(weights) == {"alta_nb", "alta_delta", "alta_readability", "alta_pos", "alta_entropy"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in weights.values())
    assert np.all((fit["test_scores"] >= 0) & (fit["test_scores"] <= 1))
