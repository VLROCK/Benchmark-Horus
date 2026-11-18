"""Small, text-only tests for derived layout and nuisance controls."""

import pandas as pd
import pytest

from horus_benchmark.audit.secondary_controls import (
    _cell_columns, _features, layout_all_text, layout_blank_text,
    match_nuisance,
)


def _test_frame() -> pd.DataFrame:
    rows = []
    for label, values in (("human", ["h1", "h2", "h3"]),
                          ("synthetic", ["s1", "s2", "s3"])):
        for record_id in values:
            rows.append({
                "record_id": record_id, "parent_record_id": record_id,
                "probe_group_id": record_id, "label_name": label,
                "broad_area": "journalistic", "source_name": "source" if label == "human" else None,
                "generator": "model" if label == "synthetic" else None,
                "content": "Uma frase simples.\n\nOutra frase completa.",
                "content_hash": "", "word_count_audit": 7,
                "parent_word_count_audit": 7,
            })
    return pd.DataFrame(rows)


def test_layout_ablations_preserve_tokens_but_target_different_breaks() -> None:
    value = "Linha um.\nLinha dois.\n\n- Linha três.\nLinha quatro."
    blank = layout_blank_text(value)
    all_lines = layout_all_text(value)
    assert blank == "Linha um.\nLinha dois. - Linha três.\nLinha quatro."
    assert all_lines == "Linha um. Linha dois. - Linha três. Linha quatro."
    assert value.split() == blank.split() == all_lines.split()
    assert layout_blank_text("a\r\n \r\nb") == "a b"


def test_length_matching_is_deterministic_and_balanced() -> None:
    frame = _test_frame()
    frame["unit_length_bin"] = 4
    frame["parent_length_bin"] = 4
    frame.loc[frame.record_id.eq("s3"), "unit_length_bin"] = 5
    selected, cells = match_nuisance(frame, unit="document", tier="length", seed=19)
    shuffled, _ = match_nuisance(frame.sample(frac=1, random_state=11),
                                 unit="document", tier="length", seed=19)
    assert selected.sort_values("record_id").reset_index(drop=True).equals(
        shuffled.sort_values("record_id").reset_index(drop=True))
    assert selected.label_name.value_counts().to_dict() == {"human": 2, "synthetic": 2}
    assert selected.groupby("pair_id").label_name.nunique().eq(2).all()
    assert cells.matched_per_class.sum() == 2


def test_presentation_tier_uses_layout_and_list_overlap_only_on_documents() -> None:
    frame = _test_frame()
    frame["unit_length_bin"] = 4
    frame["paragraph_bin"] = "1"
    frame["markdown_bullet"] = False
    frame.loc[frame.record_id.eq("s3"), "paragraph_bin"] = "5+"
    frame.loc[frame.record_id.eq("s2"), "markdown_bullet"] = True
    selected, _ = match_nuisance(frame, unit="document", tier="presentation", seed=19)
    assert selected.label_name.value_counts().to_dict() == {"human": 1, "synthetic": 1}
    assert _cell_columns("paragraph", "length") == [
        "broad_area", "unit_length_bin", "parent_length_bin"]
    with pytest.raises(ValueError, match="only for documents"):
        _cell_columns("paragraph", "presentation")


def test_segment_length_control_uses_parent_length_and_rejects_outcomes() -> None:
    frame = _test_frame()
    frame["unit_length_bin"] = 4
    frame["parent_length_bin"] = 7
    frame.loc[frame.record_id.eq("s3"), "parent_length_bin"] = 8
    selected, cells = match_nuisance(frame, unit="sentence", tier="length", seed=19)
    assert selected.label_name.value_counts().to_dict() == {"human": 2, "synthetic": 2}
    assert cells.matched_per_class.sum() == 2
    frame["score_n2"] = 0.5
    with pytest.raises(ValueError, match="must not contain model outcomes"):
        match_nuisance(frame, unit="sentence", tier="length", seed=19)


def test_observed_unit_word_count_must_match_content() -> None:
    import hashlib

    frame = _test_frame()
    frame["word_count_audit"] = frame.content.map(lambda value: len(value.split()))
    frame["parent_word_count_audit"] = frame.word_count_audit
    frame["content_hash"] = frame.content.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    features = _features(frame, "document", .25)
    assert features.paragraph_count.eq(2).all()
    assert features.markdown_bullet.eq(False).all()
    frame.loc[0, "word_count_audit"] = 99
    with pytest.raises(ValueError, match="word counts disagree"):
        _features(frame, "document", .25)
