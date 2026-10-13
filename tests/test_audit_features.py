from __future__ import annotations

import pandas as pd

from horus_benchmark.audit.coverage import assign_length_bin, near_pair_review_tables
from horus_benchmark.audit.text_features import (
    conservative_comparison_text, extract_text_features, sha256_utf8,
)


def test_independent_hashes_and_presentation_features() -> None:
    content = "# Título\n\n- Item, dois!\n- Mais um?\n"
    features = extract_text_features(content)

    assert features["content_sha256_audit"] == sha256_utf8(content)
    assert features["word_count_audit"] == len(content.split())
    assert features["paragraph_count"] == 2
    assert features["markdown_heading"]
    assert features["markdown_bullet"]
    assert features["comma_per_100_words"] > 0
    assert features["question_per_100_words"] > 0


def test_comparison_normalization_is_explicit_and_non_mutating() -> None:
    source = "  Texto\u00a0\u00a0 COM  espaços  "
    normalized = conservative_comparison_text(source)

    assert normalized == "texto com espaços"
    assert source.startswith("  Texto")


def test_length_bins_are_half_open() -> None:
    bins = [0, 150, 400]
    assert assign_length_bin(149, bins) == "0-149"
    assert assign_length_bin(150, bins) == "150-399"
    assert assign_length_bin(400, bins) == "400+"


def test_near_pair_review_attaches_provenance_without_equating_url_and_text() -> None:
    features = pd.DataFrame([
        {"record_id": "a", "source_name": "source", "source_url": "https://example/a",
         "source_record_id": "one", "split_group_id": "g1", "word_count_audit": 100},
        {"record_id": "b", "source_name": "source", "source_url": "https://example/b",
         "source_record_id": "two", "split_group_id": "g2", "word_count_audit": 101},
    ])
    pairs = pd.DataFrame([{
        "left_record_id": "a", "right_record_id": "b", "cross_split": True,
        "jaccard_5gram": 0.91, "containment_5gram": 0.96,
    }])

    tables = near_pair_review_tables(features, pairs)

    assert len(tables["near_pair_review"]) == 1
    assert not tables["near_pair_review"].iloc[0]["same_source_url"]
    assert not tables["near_pair_review"].iloc[0]["same_split_group"]
    assert tables["cross_split_near_by_source"].iloc[0]["pairs"] == 1

