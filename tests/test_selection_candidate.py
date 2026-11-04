"""Checks for conservative eligibility, lineage and deterministic matching."""

import pandas as pd

from horus_benchmark.audit.selection_candidate import (
    _article_url, _flags, build_lineage, quality_table, select_cell_balanced,
)


def _records() -> pd.DataFrame:
    rows = [
        ("h1", "human", "academic", "source-a", None, "same", "parent-a", "src-1", "https://a.org/article/1", "hash-1", 100),
        ("h2", "human", "academic", "source-a", None, "same", "parent-a", "src-1", "https://a.org/article/1", "hash-1", 100),
        ("h3", "human", "academic", "source-a", None, "other", "parent-b", "src-2", "https://a.org/article/2", "hash-3", 100),
        ("s1", "synthetic", "academic", None, "generator-a", "", "", "", "", "hash-4", 101),
        ("s2", "synthetic", "academic", None, "generator-b", "", "", "", "", "hash-5", 102),
    ]
    return pd.DataFrame(rows, columns=["record_id", "label_name", "broad_area", "source_name",
                                       "generator", "duplicate_group_id", "split_group_id",
                                       "source_record_id", "source_url", "content_sha256_audit",
                                       "word_count_audit"])


def _pairs() -> pd.DataFrame:
    return pd.DataFrame([
        ("h2", "h3", .62, 1.0),
        ("s1", "s2", .96, .97),
    ], columns=["left_record_id", "right_record_id", "jaccard_5gram", "containment_5gram"])


def test_lineage_groups_only_strong_near_pair_and_records_containment() -> None:
    lineage, decisions, counts = build_lineage(
        _records(), _pairs(), {"max_metadata_group_size": 20, "min_global_jaccard": .9})
    groups = lineage.set_index("record_id").selection_group_id
    assert groups["h1"] == groups["h2"]
    assert groups["h2"] != groups["h3"]
    assert groups["s1"] == groups["s2"]
    probe_groups = lineage.set_index("record_id").probe_group_id
    assert probe_groups["h2"] == probe_groups["h3"]
    assert decisions.set_index("left_record_id").loc["h2", "decision"] == "unresolved_containment_only"
    assert counts["unresolved_containment_only"] == 1


def test_quality_holds_severe_flags_but_keeps_metadata_warning_and_deduplicates() -> None:
    records = _records()
    lineage, _, _ = build_lineage(records, _pairs().iloc[0:0],
                                   {"max_metadata_group_size": 20, "min_global_jaccard": .9})
    trace = pd.DataFrame({"record_id": records.record_id,
                          "normalization_flags": ["SOURCE_HASH_MISMATCH", "", "ENCODING_REMAINS", "", ""],
                          "normalization_review_reasons": ["", "", "", "", ""]})
    quality = quality_table(records, trace, lineage, {"hold_flags": ["ENCODING_REMAINS"]})
    state = quality.set_index("record_id").quality_state
    assert state["h1"] == "eligible"
    assert state["h2"] == "redundant_exact"
    assert state["h3"] == "hold_for_review"
    assert _flags("SOURCE_HASH_MISMATCH;") == {"SOURCE_HASH_MISMATCH"}


def test_exact_representative_is_chosen_among_eligible_rows() -> None:
    records = _records()
    lineage, _, _ = build_lineage(records, _pairs().iloc[0:0],
                                   {"max_metadata_group_size": 20, "min_global_jaccard": .9})
    trace = pd.DataFrame({"record_id": records.record_id,
                          "normalization_flags": ["ENCODING_REMAINS", "", "", "", ""],
                          "normalization_review_reasons": ["", "", "", "", ""]})
    quality = quality_table(records, trace, lineage, {"hold_flags": ["ENCODING_REMAINS"]})
    state = quality.set_index("record_id").quality_state
    assert state["h1"] == "hold_for_review"
    assert state["h2"] == "eligible"


def test_selection_balances_area_length_cells_and_is_order_independent() -> None:
    records = _records()
    records["quality_state"] = "eligible"
    first = select_cell_balanced(records, .5, 73)
    second = select_cell_balanced(records.sample(frac=1, random_state=5), .5, 73)
    assert set(first.record_id) == set(second.record_id)
    assert first.label_name.value_counts().to_dict() == {"human": 2, "synthetic": 2}
    assert set(first.broad_area) == {"academic"}


def test_url_does_not_treat_section_or_query_as_document_id() -> None:
    assert not _article_url("https://site.org/news")
    assert not _article_url("https://site.org/article/view?id=2")
    assert _article_url("https://site.org/article/2") == "site.org/article/2"
