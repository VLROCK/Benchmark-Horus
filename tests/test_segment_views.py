"""Small, deterministic checks for derived middle-unit views."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from horus_benchmark.protocols.segment_views import (
    _coverage, _derive_split, build_units,
)


CFG = {
    "selection_seed": 13,
    "middle_start": 0.35,
    "middle_end": 0.65,
    "length_bin_width": 0.25,
    "units": {
        "paragraph": {"min_words": 5, "max_words": 30,
                      "max_fraction_of_document": 0.35},
        "sentence": {"min_words": 5, "max_words": 30,
                     "max_fraction_of_document": 0.25},
    },
}


def _parents() -> pd.DataFrame:
    blocks = [
        "A abertura apresenta um contexto breve para esta discussão.",
        "O primeiro trecho explica a situação com algumas palavras simples.",
        "Este trecho central reúne informações suficientes para a análise.",
        "O quarto trecho oferece uma continuação igualmente compreensível.",
        "O fim encerra a exposição sem acrescentar outro assunto.",
    ]
    content = "\n\n".join(blocks)
    single = "Este documento tem apenas um parágrafo longo e completo."
    return pd.DataFrame([
        {"record_id": "r1", "content": content,
         "content_hash": hashlib.sha256(content.encode()).hexdigest(),
         "probe_group_id": "g1", "label_name": "human",
         "broad_area": "academic", "source_name": "scielo.br",
         "generator": None, "word_count_audit": len(content.split())},
        {"record_id": "r2", "content": single,
         "content_hash": hashlib.sha256(single.encode()).hexdigest(),
         "probe_group_id": "g2", "label_name": "synthetic",
         "broad_area": "academic", "source_name": None,
         "generator": "model-x", "word_count_audit": len(single.split())},
    ])


def test_middle_units_keep_parent_identity_and_exact_offsets() -> None:
    parents = _parents()
    for unit in ("paragraph", "sentence"):
        texts, eligibility = build_units(parents, CFG, unit)
        assert len(eligibility) == len(parents)
        assert texts.record_id.tolist() == ["r1"]
        row = texts.iloc[0]
        assert row.parent_record_id == row.record_id == "r1"
        assert row.probe_group_id == "g1"
        assert row.label_name == "human"
        assert row.source_name == "scielo.br"
        assert parents.iloc[0].content[row.segment_start:row.segment_end] == row.content
        assert row.content_hash == hashlib.sha256(row.content.encode()).hexdigest()
        assert eligibility.loc[eligibility.record_id.eq("r2"), "reason"].iloc[0] \
            == "middle_unit_too_large_fraction"
        reverse_texts, _ = build_units(parents.iloc[::-1].reset_index(drop=True), CFG, unit)
        assert reverse_texts.segment_id.tolist() == texts.segment_id.tolist()


def test_split_is_an_exact_group_safe_subset() -> None:
    split = pd.DataFrame([
        {"record_id": "r1", "probe_group_id": "g1", "split": "development", "fold": 2},
        {"record_id": "r2", "probe_group_id": "g2", "split": "test", "fold": -1},
    ])
    texts, _ = build_units(_parents(), CFG, "paragraph")
    derived = _derive_split(split, texts)
    assert derived.record_id.tolist() == ["r1"]
    assert derived.fold.tolist() == [2]
    with pytest.raises(ValueError, match="unrelated"):
        _derive_split(split, pd.DataFrame({"record_id": ["absent"]}))
    bad = split.copy()
    bad.loc[1, "probe_group_id"] = "g1"
    with pytest.raises(ValueError, match="group-safe"):
        _derive_split(bad, pd.DataFrame({"record_id": ["r1", "r2"]}))


def test_coverage_reports_all_parent_denominators() -> None:
    _, eligibility = build_units(_parents(), CFG, "paragraph")
    split = pd.DataFrame([
        {"record_id": "r1", "split": "development", "fold": 0},
        {"record_id": "r2", "split": "test", "fold": -1},
    ])
    table = _coverage(eligibility, "paragraph", {20261105: split})
    human = table.loc[table.dimension.eq("class") & table.label_name.eq("human")].iloc[0]
    synthetic = table.loc[table.dimension.eq("class") &
                          table.label_name.eq("synthetic")].iloc[0]
    assert (human.parent_rows, human.eligible_rows) == (1, 1)
    assert (synthetic.parent_rows, synthetic.eligible_rows) == (1, 0)
    assert table.dimension.eq("seed_split_class_area").sum() == 2
