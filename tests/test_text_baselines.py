"""Small synthetic checks for text-only baseline behavior and leakage gates."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from horus_benchmark.detectors.text_baselines import (
    DEFAULT_CONFIG, MODEL_NAMES, build_estimator, fit_score, load_config, score_texts,
    select_c_inner_cv,
)
from horus_benchmark.features.shortcut_features import (
    SURFACE_GROUPS, extract_shortcut_features, surface_matrix,
)


def _tiny_config() -> dict:
    cfg = {
        **DEFAULT_CONFIG,
        "c_grid": [0.1, 1.0],
        "tfidf": {**DEFAULT_CONFIG["tfidf"], "min_df": 1, "max_df": 1.0,
                  "max_features": 1000},
        "iforest": {**DEFAULT_CONFIG["iforest"], "svd_components": 2,
                    "n_estimators": 10, "max_samples": 4},
    }
    return cfg


def test_declarative_config_lists_only_supported_text_models() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/detectors/text_baselines_v1.yaml"
    loaded = load_config(path)
    assert tuple(loaded["models"]) == MODEL_NAMES
    assert loaded["tfidf"]["min_df"] == 5


def _examples() -> tuple[list[str], np.ndarray, np.ndarray]:
    human = [
        "Meu relato sobre o bairro, as pessoas e a chuva de ontem. " + word
        for word in ("casa", "rua", "praça", "janela", "livro", "amigo")
    ]
    synthetic = [
        "Em primeiro lugar, apresentamos um resumo estruturado da análise. " + word
        for word in ("sistema", "modelo", "método", "tabela", "etapa", "processo")
    ]
    texts = [item for pair in zip(human, synthetic) for item in pair]
    labels = np.asarray([0, 1] * 6)
    folds = np.asarray([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    return texts, labels, folds


def test_surface_groups_use_only_text_and_are_finite() -> None:
    plain = "Primeira frase.\n\n- Segundo ponto, com URL e EMAIL."
    features = extract_shortcut_features(plain)
    assert features["paragraph_count"] == 2
    assert features["unordered_list_marker"] == 1
    assert features["url_marker"] == 1
    assert features["email_marker"] == 1
    assert set(features) == set(SURFACE_GROUPS["surface_all"])
    assert surface_matrix([plain], "surface_length").shape == (1, 3)
    assert np.isfinite(surface_matrix([plain], "surface_all")).all()


def test_inner_c_uses_only_supplied_training_folds() -> None:
    texts, y, folds = _examples()
    c, evidence = select_c_inner_cv(
        "word_lr", texts, y, folds, seed=2, config=_tiny_config(),
    )
    assert c in {0.1, 1.0}
    assert set(evidence) == {"0.1", "1.0"}
    assert all(0 <= auc <= 1 for auc in evidence.values())
    with pytest.raises(ValueError, match="both labels"):
        select_c_inner_cv("word_lr", texts, y,
                          np.where(y == 0, 0, 1), seed=2,
                          config=_tiny_config())


def test_fit_score_preserves_train_only_vocabulary_and_paired_scoring() -> None:
    texts, y, folds = _examples()
    outcome = fit_score(
        "word_lr", texts, y,
        ["palavra_inedita de validação"], ["palavra_inedita no teste"],
        seed=3, inner_folds=folds, config=_tiny_config(),
    )
    assert "palavra_inedita" not in outcome["estimator"].named_steps["tfidf"].vocabulary_
    assert outcome["validation_scores"].shape == (1,)
    assert np.array_equal(
        score_texts(outcome["estimator"], "word_lr", ["palavra_inedita no teste"]),
        outcome["test_scores"],
    )
    assert outcome["selected_c"] in {0.1, 1.0}


def test_human_only_model_does_not_fit_synthetic_vocabulary() -> None:
    texts, y, _ = _examples()
    texts[1] = "exclusivamente_sintetico uma palavra só"
    model = build_estimator("human_iforest", seed=4, config=_tiny_config())
    model.fit(texts, y)
    assert model.human_train_rows_ == 6
    assert "exclusivamente_sintetico" not in model.vectorizer_.vocabulary_
    scores = score_texts(model, "human_iforest", texts[:2])
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
