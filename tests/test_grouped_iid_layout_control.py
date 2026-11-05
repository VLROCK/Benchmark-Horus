"""Integrity checks for provisional group-disjoint IID and layout control."""

import pandas as pd
import pytest

from horus_benchmark.audit.prepare_selection_protocol import layout_control_text
from horus_benchmark.protocols.grouped_iid import (
    add_strata, assign_groups, check_observed_pairs, split_diagnostics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for label in ("human", "synthetic"):
        for area in ("academic", "journalistic"):
            for number in range(60):
                rows.append({"record_id": f"{label}-{area}-{number}",
                             "probe_group_id": f"{label}-{area}-{number // 2}",
                             "label_name": label, "broad_area": area,
                             "word_count_audit": 100 + 20 * (number % 3)})
    return add_strata(pd.DataFrame(rows), [0, 100, 250])


def test_grouped_splits_cover_all_ids_without_observed_pair_leakage() -> None:
    frame = _frame()
    fractions = {"train": .7, "validation": .15, "test": .15}
    split = assign_groups(frame, fractions, seed=11)
    assert len(split) == len(frame)
    assert set(split.record_id) == set(frame.record_id)
    assert split.groupby("probe_group_id").split.nunique().max() == 1
    assert set(split.split) == {"train", "validation", "test"}
    first = frame.iloc[0:2].record_id.tolist()
    pairs = pd.DataFrame({"left_record_id": [first[0]], "right_record_id": [first[1]]})
    assert check_observed_pairs(split, pairs)["observed_pairs_cross_split"] == 0
    strata, by_area = split_diagnostics(split, fractions)
    assert strata.rows.sum() == len(frame)
    assert by_area.rows.sum() == len(frame)
    assert split.set_index("record_id").split.equals(
        assign_groups(frame.sample(frac=1, random_state=2), fractions, seed=11
                      ).set_index("record_id").split.reindex(split.record_id))


def test_observed_pair_checker_rejects_cross_partition_pair() -> None:
    split = pd.DataFrame({"record_id": ["a", "b"], "split": ["train", "test"]})
    pairs = pd.DataFrame({"left_record_id": ["a"], "right_record_id": ["b"]})
    with pytest.raises(ValueError, match="cross splits"):
        check_observed_pairs(split, pairs)


def test_layout_control_preserves_tokens_and_punctuation() -> None:
    content = "Olá, mundo!\n\n- Uma linha.\nOutra linha?"
    result = layout_control_text(content)
    assert "\n" not in result
    assert result == "Olá, mundo! - Uma linha. Outra linha?"
    assert result.split() == content.split()


def test_length_stratum_boundaries() -> None:
    frame = pd.DataFrame({"record_id": ["a", "b", "c"], "label_name": ["human"] * 3,
                          "broad_area": ["academic"] * 3,
                          "word_count_audit": [99, 100, 250]})
    assert add_strata(frame, [0, 100, 250]).length_stratum.tolist() == [0, 1, 2]
