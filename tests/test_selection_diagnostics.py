"""Focused checks for exploratory selection arithmetic and view identity."""

import numpy as np
import pandas as pd
import pytest

from horus_benchmark.audit.selection_diagnostics import (
    NUMBERED_LIST, UNORDERED_LIST, _candidate_ids, _check_paired_views,
    _coarse_bins, _ks, _probe_split, _smd,
)


def test_coarse_bins_include_boundaries_and_long_tail() -> None:
    bins = _coarse_bins(pd.Series([0, 149, 150, 399, 400, 5000]),
                        [0, 150, 400, 800, 1800, 5000])
    assert bins.tolist() == ["0-149", "0-149", "150-399", "150-399", "400-799", "5000+"]


def test_selection_scenarios_keep_common_area_cells_and_balance_within_bins() -> None:
    frame = pd.DataFrame([
        ("h1", "human", "academic", "0-149", 1, "a"),
        ("h2", "human", "academic", "0-149", 1, "b"),
        ("s1", "synthetic", "academic", "0-149", 1, "c"),
        ("h3", "human", "academic", "150-399", 2, "d"),
        ("s2", "synthetic", "social", "0-149", 1, "e"),
    ], columns=["record_id", "label_name", "broad_area", "coarse_bin", "fine_bin", "sample_key"])
    assert len(_candidate_ids(frame, "full")) == 5
    assert _candidate_ids(frame, "common_support") == {"h1", "h2", "s1"}
    assert _candidate_ids(frame, "balanced_coarse") == {"h1", "s1"}
    assert _candidate_ids(frame, "balanced_fine") == {"h1", "s1"}
    assert _candidate_ids(frame.sample(frac=1, random_state=3), "balanced_coarse") == {"h1", "s1"}


def test_new_n3_keeps_human_text_and_checks_metadata() -> None:
    n2 = pd.DataFrame({
        "record_id": ["h", "s"], "label_name": ["human", "synthetic"],
        "broad_area": ["social", "social"], "source_name": ["blog", None],
        "generator": [None, "g"], "split_group_id": ["a", "b"],
        "word_count_audit": [20, 40], "log_words": [1.0, 2.0],
        "content_sha256_audit": ["human_n2", "synthetic_n2"],
        "normalization_status": ["ok", "ok"],
    })
    n3 = n2.copy()
    n3.loc[1, "content_sha256_audit"] = "synthetic_n3"
    n3.loc[1, "word_count_audit"] = 38
    n3.loc[1, "log_words"] = 3.0
    # The real loader supplies every observable feature in both frames.
    from horus_benchmark.audit.selection_diagnostics import FEATURES
    for feature in FEATURES:
        if feature not in n2:
            n2[feature] = 0
            n3[feature] = 0
    _check_paired_views(n2, n3)
    n3.loc[0, "content_sha256_audit"] = "human_n3"
    with pytest.raises(ValueError, match="byte for byte"):
        _check_paired_views(n2, n3)
    n3.loc[0, "content_sha256_audit"] = "human_n2"
    n3.loc[1, "generator"] = "changed"
    with pytest.raises(ValueError, match="Protected metadata"):
        _check_paired_views(n2, n3)


def test_balance_diagnostics_have_expected_direction() -> None:
    human = np.array([1., 2., 3.])
    synthetic = np.array([2., 3., 4.])
    assert _smd(human, synthetic) > 0
    assert _ks(human, human) == 0
    assert 0 < _ks(human, synthetic) < 1


def test_probe_split_keeps_existing_groups_together() -> None:
    frame = pd.DataFrame({"record_id": ["a", "b", "c", "d"],
                          "split_group_id": ["parent", "parent", None, None]})
    config = {"seed": 7, "probe_train_fraction": .7,
              "probe_validation_fraction": .15, "probe_test_fraction": .15}
    parts = _probe_split(frame, config)
    assert parts[0] == parts[1]
    assert np.array_equal(parts, _probe_split(frame, config))


def test_unordered_and_numbered_markers_are_distinct() -> None:
    assert UNORDERED_LIST.search("- item sem número")
    assert not NUMBERED_LIST.search("- item sem número")
    assert NUMBERED_LIST.search("1. seção numerada")
    assert not UNORDERED_LIST.search("1. seção numerada")
    assert not UNORDERED_LIST.search("**ênfase**")
