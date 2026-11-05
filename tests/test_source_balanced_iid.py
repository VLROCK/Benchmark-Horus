"""Source-aware IID candidate arithmetic and no-crossing invariants."""

import pandas as pd

from horus_benchmark.audit.refine_grouped_iid import provenance_shift
from horus_benchmark.protocols.grouped_iid import (
    _provenance_vectors, add_strata, assign_heldout_cv, check_observed_pairs,
)


def _sample() -> pd.DataFrame:
    rows = []
    for label in ("human", "synthetic"):
        for area in ("academic", "social"):
            for number in range(80):
                rows.append({"record_id": f"{label}-{area}-{number}",
                             "probe_group_id": f"{label}-{area}-{number // 2}",
                             "label_name": label, "broad_area": area,
                             "source_name": "one" if area == "academic" else
                             ("source-a" if number < 40 else "source-b"),
                             "generator": "model-a" if number < 40 else "model-b",
                             "word_count_audit": 100 + number % 5})
    return add_strata(pd.DataFrame(rows), [0, 100, 250])


def test_provenance_vectors_exclude_single_source_area() -> None:
    frame = _sample().loc[lambda value: value.label_name.eq("human")]
    groups = sorted(frame.probe_group_id.unique())
    vectors = _provenance_vectors(frame, groups, minimum=10)
    assert vectors.shape == (len(groups), 2)  # social sources; academic has one
    assert vectors.sum() == 80


def test_source_aware_holdout_folds_keep_ids_and_observed_pairs_together() -> None:
    frame = _sample()
    result = assign_heldout_cv(frame, test_fraction=.15, folds=5, seed=9,
                                provenance_weight=1.0, material_min_rows=10)
    assert set(result.record_id) == set(frame.record_id)
    assert result.groupby("probe_group_id").split.nunique().max() == 1
    assert result.groupby("probe_group_id").fold.nunique().max() == 1
    assert result.loc[result.split.eq("test"), "fold"].eq(-1).all()
    assert result.loc[result.split.eq("development"), "fold"].between(0, 4).all()
    shuffled = assign_heldout_cv(frame.sample(frac=1, random_state=21),
                                 test_fraction=.15, folds=5, seed=9,
                                 provenance_weight=1.0, material_min_rows=10)
    assert (result.set_index("record_id")[["split", "fold"]].sort_index()
            .equals(shuffled.set_index("record_id")[["split", "fold"]].sort_index()))
    pair = pd.DataFrame({"left_record_id": [frame.iloc[0].record_id],
                         "right_record_id": [frame.iloc[1].record_id]})
    assert check_observed_pairs(result, pair)["observed_pairs_cross_split"] == 0


def test_provenance_shift_is_relative_to_area_class_and_marks_single_source() -> None:
    report = pd.DataFrame([
        ("development", "informational", "human", "wiki", 60),
        ("development", "informational", "human", "other", 20),
        ("test", "informational", "human", "wiki", 10),
        ("test", "informational", "human", "other", 10),
        ("development", "academic", "human", "scielo", 80),
        ("test", "academic", "human", "scielo", 20),
    ], columns=["split", "broad_area", "label_name", "provenance", "rows"])
    shifts = provenance_shift(report, minimum=10)
    wiki = shifts.loc[shifts.split.eq("test") & shifts.provenance.eq("wiki")].iloc[0]
    assert abs(wiki.shift_pp + 20) < 1e-9  # 50% in test versus 70% overall
    assert bool(wiki.material_diverse)
    assert not shifts.loc[shifts.provenance.eq("scielo"), "material_diverse"].any()
