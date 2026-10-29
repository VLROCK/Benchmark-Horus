from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horus_benchmark.audit.compare_normalization_views import (
    _fpr_threshold, _load_selected, _sample_ids, paired_intervals,
)
from horus_benchmark.audit.middle_segments import align_segment, select_middle


def test_middle_paragraph_not_replaced_by_front_or_footer() -> None:
    text = "Abstract English words.\n\n" + ("Miolo confiável permanece aqui. " * 12) + "\n\nReferências bibliográficas."
    selected = select_middle(text, "paragraph", "rid", min_words=10, max_words=100,
                             middle_start=0.35, middle_end=0.65,
                             max_fraction_of_document=0.95, seed=7)
    assert selected is not None
    assert selected.text.startswith("Miolo")
    changed = "Resumo inicial.\n\n" + selected.text + "\n\nRodapé"
    matched, status, similarity = align_segment(selected, changed, "paragraph",
                                                 minimum_similarity=0.92, minimum_margin=0.03)
    assert matched is not None and matched.text == selected.text
    assert status == "exact" and similarity == 1.0


def test_removed_or_repeated_segment_is_not_substituted() -> None:
    original_text = "Começo.\n\n" + ("Trecho central original. " * 12) + "\n\nFim."
    selected = select_middle(original_text, "paragraph", "rid", min_words=10,
                             max_words=100, middle_start=0.35, middle_end=0.65,
                             max_fraction_of_document=0.95, seed=7)
    assert selected is not None
    absent, status, _ = align_segment(selected, "Começo.\n\nFim.", "paragraph",
                                      minimum_similarity=0.92, minimum_margin=0.03)
    assert absent is None and status == "missing_or_changed_boundary"
    repeated, status, _ = align_segment(selected, selected.text + "\n\n" + selected.text,
                                        "paragraph", minimum_similarity=0.92,
                                        minimum_margin=0.03)
    assert repeated is None and status == "ambiguous"


def test_middle_sentence_is_distinct_from_paragraph() -> None:
    text = "Começo curto. " + ("Esta frase central tem conteúdo suficiente para o teste. " * 4) + "Fim curto."
    selected = select_middle(text, "sentence", "rid", min_words=8, max_words=20,
                             middle_start=0.35, middle_end=0.65,
                             max_fraction_of_document=0.25, seed=7)
    assert selected is not None
    assert selected.text.startswith("Esta frase")
    assert len(selected.text.split()) < len(text.split())


def test_paired_bootstrap_zero_delta_for_identical_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = paired_intervals(y, {view: scores for view in ("raw", "n1", "n2")},
                              {view: (0.5, 0.3) for view in ("raw", "n1", "n2")},
                              replicates=20, confidence=0.95, seed=7)
    assert (result["delta_ci_low"] == 0).all()
    assert (result["delta_ci_high"] == 0).all()


def test_fpr_threshold_is_selected_on_validation_negatives_only() -> None:
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate((np.arange(100) / 100, np.ones(100)))
    threshold = _fpr_threshold(y, scores, 0.01)
    assert np.mean(scores[y == 0] > threshold) <= 0.01


def test_loading_selected_rejects_metadata_change(tmp_path) -> None:
    path = tmp_path / "view.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "record_id": "r1", "label": 1, "split": "test",
        "broad_area": "academic", "content": "Texto de teste.",
    }]), path)
    expected = pd.DataFrame([{
        "record_id": "r1", "label": 0, "split": "test", "broad_area": "academic",
    }])
    with pytest.raises(ValueError, match="Protected metadata"):
        _load_selected(path, expected, 128)


def test_sample_ids_sorts_string_hashes_deterministically(tmp_path) -> None:
    rows = [
        {"record_id": f"{split}-{label}-{index}", "label": label,
         "split": split, "broad_area": "academic"}
        for split in ("train", "validation", "test")
        for label in (0, 1)
        for index in range(3)
    ]
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.Table.from_pylist(rows), first)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), second)
    limits = {"train": 1, "validation": 1, "test": 1}
    selected_first = _sample_ids(first, limits, seed=7)
    selected_second = _sample_ids(second, limits, seed=7)
    assert selected_first["record_id"].tolist() == selected_second["record_id"].tolist()
    assert len(selected_first) == 6
