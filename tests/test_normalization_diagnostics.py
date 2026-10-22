"""Small fixtures for provenance hypotheses and deterministic sampling."""

from __future__ import annotations

import hashlib

from horus_benchmark.data.normalization_diagnostics import (
    SOURCE_HEADING, _choose_academic_sample, _hash_explanation,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_hash_diagnostic_does_not_treat_any_variant_as_source_repair() -> None:
    assert _hash_explanation("ação\r\nfinal", _sha("ação\nfinal")) == "matches_newline_lf"
    assert _hash_explanation(" ação ", _sha("ação")) == "matches_trimmed"
    assert _hash_explanation("texto", "bad") == "invalid_sha256_format"
    assert _hash_explanation("texto", _sha("outro")) == "unexplained_by_tested_variants"


def test_academic_sample_balances_generators_before_repeating() -> None:
    cells = {
        "model_a": [("a", {"record_id": "a1"}), ("b", {"record_id": "a2"})],
        "model_b": [("a", {"record_id": "b1"})],
    }
    assert [row["record_id"] for row in _choose_academic_sample(cells, 3)] == [
        "a1", "b1", "a2",
    ]


def test_source_heading_detects_markdown_and_plain_references() -> None:
    assert SOURCE_HEADING.search("Texto\n## Referências\nAutor, 2001.")
    assert SOURCE_HEADING.search("Texto\nFontes consultadas:\nDocumento.")
    assert SOURCE_HEADING.search("Texto\n**Referências** *(selecionadas)*\n* Autor, 2001.")
    assert not SOURCE_HEADING.search("As referências conceituais aparecem no texto.")
