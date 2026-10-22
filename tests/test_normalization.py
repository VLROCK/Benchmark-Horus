"""Behavior and invariants for full-corpus, class-blind normalization."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from horus_benchmark.data.normalization import (
    RuleConfig, apply_n2_edits, n2_loss_guard, normalize_text, propose_n2_edits,
)
from horus_benchmark.data.normalize_corpus import run
from horus_benchmark.data.normalization_diagnostics import run as run_diagnostics
from horus_benchmark.data.normalization_report import build_report
from horus_benchmark.data.normalization_review import export_review
from horus_benchmark.data.prepare_normalized_audit import prepare
from horus_benchmark.data.preview_n2 import run as preview_n2
from horus_benchmark.data.verify_normalization import verify
from horus_benchmark.data.verify_hash_repair import run as verify_hash_repair


def test_preserves_natural_hypothetical_and_colloquial_writing() -> None:
    text = "Gente, imagina um boletim hipotético sobre a cidade? rs É só um cenário."
    result = normalize_text(text, RuleConfig())
    assert result.text == text
    assert result.status == "ok"


def test_recounts_and_repairs_only_clear_mojibake() -> None:
    result = normalize_text("Monitore a posiĂ§Ă£o do site", RuleConfig())
    assert result.text == "Monitore a posição do site"
    assert result.operations["MOJIBAKE_REPAIR"] == 1
    assert result.original_chars != result.normalized_chars


def test_mixed_mojibake_is_repaired_with_other_unicode_present() -> None:
    result = normalize_text("A posiĂ§Ă£o é esta 🙂", RuleConfig())
    assert result.text == "A posição é esta 🙂"
    assert result.operations["MOJIBAKE_REPAIR"] == 1


def test_uppercase_portuguese_is_not_mojibake_or_review() -> None:
    result = normalize_text("NÃO HÁ CONCLUSÃO SOBRE A ÁREA.", RuleConfig())
    assert result.text == "NÃO HÁ CONCLUSÃO SOBRE A ÁREA."
    assert "ENCODING_CANDIDATE" not in result.flags
    assert result.status == "ok"


def test_math_inequalities_are_not_html() -> None:
    result = normalize_text("Se T < 0,5s e depois T > 2,5s, comparar.", RuleConfig())
    assert "HTML_REMAINS" not in result.flags
    assert result.status == "ok"


def test_code_declaration_is_not_english_abstract() -> None:
    result = normalize_text("abstract public class Exemplo { }", RuleConfig())
    assert "LANG_SECTION" not in result.flags
    assert result.status == "ok"


def test_ambiguous_bibliography_and_placeholder_are_diagnostic_only() -> None:
    result = normalize_text("Referências\n[Seu Nome]", RuleConfig())
    assert {"REFERENCES", "PLACEHOLDER"}.issubset(result.flags)
    assert result.status == "ok"


def test_dense_residual_html_requires_three_real_tags() -> None:
    result = normalize_text("<span>A</span><i>B</i>", RuleConfig())
    assert "HTML_DENSE_REMAINS" in result.flags
    assert result.status == "review"


def test_code_example_is_protected_from_html_and_contact_rewrites() -> None:
    text = "Exemplo:\n```html\n<a href=\"https://example.org\">teste</a>\n```"
    result = normalize_text(text, RuleConfig())
    assert result.text == text
    assert "CODE_EXAMPLE" in result.flags
    assert "URL_MASK" not in result.operations
    assert result.status == "ok"


def test_contact_mask_keeps_sentence_punctuation() -> None:
    result = normalize_text("Leia https://example.org. Escreva a@b.com!", RuleConfig())
    assert result.text == "Leia URL. Escreva EMAIL!"


def test_n2_only_removes_a_strict_terminal_footer() -> None:
    text = "Texto autoral.\nCopyright © 2011 Editora. Todos os direitos reservados."
    assert normalize_text(text, RuleConfig(stage="n1")).text == text
    n2 = normalize_text(text, RuleConfig(stage="n2"))
    assert n2.text == "Texto autoral."
    assert n2.operations["TERMINAL_FOOTER"] == 1


def test_n2_parallel_abstract_requires_both_portuguese_boundaries() -> None:
    text = (
        "Resumo\n" + "Este estudo examina a escrita. " * 8 + "\n"
        "Abstract\n" + "This study examines the writing of texts and methods in our research. " * 5 + "\n"
        "Keywords: writing\nIntrodução\n" + "O corpo permanece em português. " * 180
    )
    n1 = normalize_text(text, RuleConfig(stage="n1"))
    n2 = normalize_text(text, RuleConfig(stage="n2"))
    assert "Abstract" in n1.text
    assert "Abstract" not in n2.text
    assert "Introdução" in n2.text
    assert n2.operations["PARALLEL_ABSTRACT_REMOVE"] == 1


def test_n2_keeps_portuguese_abstract_with_english_terminology() -> None:
    text = (
        "Resumo\n" + "Pesquisa de linguagem em português. " * 10 + "\n"
        "Abstract\n" + "Este estudo de machine learning apresenta resultados no Brasil. " * 8 + "\n"
        "Introdução\n" + "O texto continua em português. " * 20
    )
    result = normalize_text(text, RuleConfig(stage="n2"))
    assert result.text == normalize_text(text, RuleConfig(stage="n1")).text
    assert "PARALLEL_ABSTRACT_REMOVE" not in result.operations


def test_n2_keeps_bibliography_by_default() -> None:
    body = "Pesquisa empírica em língua portuguesa. " * 210
    bibliography = "Referências\nAutor A. Livro, 2001.\nAutor B. Artigo, 2010.\nAutor C. Estudo, 2017.\n" + "Outro estudo citado, 2017. " * 12
    result = normalize_text(body + "\n" + bibliography, RuleConfig(stage="n2"))
    assert result.text.endswith(bibliography.strip())
    assert "TERMINAL_REFERENCES_CANDIDATE" in result.flags
    assert "TERMINAL_REFERENCES_REMOVE" not in result.operations


def test_n2_terminal_references_requires_substantial_bibliography() -> None:
    body = "Pesquisa empírica em língua portuguesa. " * 210
    bibliography = "Referências\nAutor A. Livro, 2001.\nAutor B. Artigo, 2010.\nAutor C. Estudo, 2017.\n" + "Outro estudo citado, 2017. " * 12
    result = normalize_text(body + "\n" + bibliography, RuleConfig(
        stage="n2", remove_strict_terminal_references=True,
    ))
    assert "Referências" not in result.text
    assert result.operations["TERMINAL_REFERENCES_REMOVE"] == 1
    assert result.status == "ok"


def test_n2_loss_guard_retains_original_n1_text() -> None:
    text = (
        "Resumo\n" + "Pesquisa em português. " * 10 + "\n"
        "Abstract\n" + "This study examines the writing of texts and methods in our research. " * 6
        + "\nIntrodução\n" + "O corpo permanece em português. " * 25
    )
    result = normalize_text(text, RuleConfig(stage="n2"))
    assert result.text == normalize_text(text, RuleConfig(stage="n1")).text
    assert "N2_GUARD_RETAINED" in result.flags
    assert "PARALLEL_ABSTRACT_REMOVE" not in result.operations
    assert result.status == "review"


def test_n2_leading_english_abstract_requires_portuguese_resumo_and_intro() -> None:
    text = (
        "Abstract\n" + "This study examines the writing of texts and methods in our research. " * 6
        + "\nResumo\n" + "Este estudo apresenta resultados. " * 25
        + "\nIntrodução\n" + "O corpo permanece em português. " * 200
    )
    result = normalize_text(text, RuleConfig(stage="n2"))
    assert result.text.startswith("Resumo")
    assert "PARALLEL_ABSTRACT_REMOVE" in result.operations


def test_n2_preserves_content_between_abstract_and_introduction() -> None:
    abstract = "This study examines the writing of texts and methods in our research. " * 6
    text = (
        "Resumo\n" + "Pesquisa em português. " * 20 + "\n"
        "Abstract\n" + abstract + "\n"
        "Key words: writing and methods\n"
        "Detetive 2\nTive tive\nMeu pai é detetive\n"
        "Introdução\n" + "O corpo permanece em português. " * 100
    )
    result = normalize_text(text, RuleConfig(stage="n2"))
    assert result.text == normalize_text(text, RuleConfig(stage="n1")).text
    assert "PARALLEL_ABSTRACT_REMOVE" not in result.operations


def test_n2_stops_at_other_language_section_not_introduction() -> None:
    abstract = "This study examines the writing of texts and methods in our research. " * 6
    text = (
        "Resumo\n" + "Pesquisa em português. " * 20 + "\n"
        "Abstract\n" + abstract + "\n"
        "Resumen\nEste texto apresenta resultados.\n"
        "Introdução\n" + "O corpo permanece em português. " * 100
    )
    result = normalize_text(text, RuleConfig(stage="n2"))
    assert "Abstract" not in result.text
    assert "Resumen" in result.text
    assert "Introdução" in result.text


def test_n2_proposals_have_stable_spans_and_no_class_input() -> None:
    text = "Pesquisa extensa em português. " * 210
    text += "\nReferências\nA. Livro, 2001.\nB. Artigo, 2010.\nC. Estudo, 2017.\n" + "Outra obra. " * 20
    rules = RuleConfig(stage="n2", remove_strict_terminal_references=True)
    edits = propose_n2_edits(text, rules)
    assert len(edits) == 1
    assert text[edits[0].start:edits[0].end].startswith("Referências")
    assert not n2_loss_guard(text, apply_n2_edits(text, edits), rules)
    assert apply_n2_edits(text, edits).endswith("português.")


def test_n2_does_not_cut_reference_heading_without_reference_lines() -> None:
    text = "Texto em português. " * 210 + "\nReferências\nEsta seção discute referências conceituais no corpo da análise. " * 8
    edits = propose_n2_edits(text, RuleConfig(stage="n2", remove_strict_terminal_references=True))
    assert not edits


def test_n2_includes_empty_preceding_reference_heading_in_cut() -> None:
    text = "Texto em português. " * 210
    text += "\nReferências\n\nBibliografia\nA. Livro, 2001.\nB. Artigo, 2010.\nC. Estudo, 2017.\n" + "Outra obra. " * 20
    rules = RuleConfig(stage="n2", remove_strict_terminal_references=True)
    edits = propose_n2_edits(text, rules)
    assert len(edits) == 1
    assert text[edits[0].start:edits[0].end].startswith("Referências")


def test_large_loss_is_flagged_without_exclusion() -> None:
    text = "texto " * 40 + "<p> " * 25
    result = normalize_text(text, RuleConfig())
    assert "SUBSTANTIAL_LOSS" in result.flags
    assert result.status == "review"
    assert result.text


def test_full_stream_preserves_rows_and_protected_metadata(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    schema = pa.schema([
        ("record_id", pa.large_string()), ("content", pa.large_string()),
        ("char_count", pa.int64()), ("word_count", pa.int64()),
        ("content_hash", pa.large_string()), ("normalized_hash", pa.large_string()),
        ("label_name", pa.large_string()), ("broad_area", pa.large_string()),
        ("split", pa.large_string()), ("generator", pa.large_string()),
        ("source_name", pa.large_string()),
    ])
    original = [
        {"record_id": "h1", "content": "Olá, mundo.", "char_count": 999,
         "word_count": 999, "content_hash": "legacy", "normalized_hash": "legacy",
         "label_name": "human", "broad_area": "social", "split": "test",
         "generator": "human", "source_name": "fixture_human"},
        {"record_id": "s1", "content": "Aqui há um link: https://example.org", "char_count": 0,
         "word_count": 0, "content_hash": "legacy", "normalized_hash": "legacy",
         "label_name": "synthetic", "broad_area": "social", "split": "train",
         "generator": "model", "source_name": "fixture_synthetic"},
    ]
    pq.write_table(pa.Table.from_pylist(original, schema=schema), source_dir / "all.parquet")
    (tmp_path / "taxonomy.csv").write_text("domain,theme,subtheme\nsocial,T,S\n", encoding="utf-8")
    dataset_cfg = {
        "dataset_id": "fixture", "dataset_version": "fixture-v1", "workspace_root": ".",
        "resources": {"corpus": {"path": "source/all.parquet", "format": "parquet",
                                 "role": "processed_canonical_corpus", "expected_rows": 2,
                                 "immutable": True},
                      "taxonomy": {"path": "taxonomy.csv", "format": "csv",
                                   "role": "synthetic_generation_taxonomy", "expected_rows": 1,
                                   "immutable": True}},
    }
    (tmp_path / "dataset.yaml").write_text(yaml.safe_dump(dataset_cfg), encoding="utf-8")
    normal_cfg = {"schema_version": 1, "workspace_root": ".",
                  "dataset_config": "dataset.yaml", "corpus_resource": "corpus",
                  "batch_size": 1, "rules": {"stage": "n1"}}
    config = tmp_path / "normal.yaml"
    config.write_text(yaml.safe_dump(normal_cfg), encoding="utf-8")
    run_dir = tmp_path / "results" / "n1"
    run(config, run_dir)
    assert verify(config, run_dir).is_file()
    derived = pq.read_table(run_dir / "normalized_corpus.parquet").to_pylist()
    trace = pq.read_table(run_dir / "normalization_trace.parquet").to_pylist()
    assert [row["record_id"] for row in derived] == ["h1", "s1"]
    assert [row["split"] for row in derived] == ["test", "train"]
    assert derived[0]["char_count"] == len(original[0]["content"])
    assert derived[1]["content"] == "Aqui há um link: URL"
    assert trace[0]["source_char_count_declared"] == 999
    assert trace[0]["normalization_review_reasons"] == "SOURCE_HASH_MISMATCH"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["output_rows"] == 2
    assert (run_dir / "flag_prevalence_by_label_area.csv").is_file()
    build_report(run_dir)
    assert (run_dir / "tables" / "class_gap_delta.csv").is_file()
    blind = export_review(config, run_dir, per_cell=1, maximum=2)
    assert blind.is_file()
    assert len(blind.read_text(encoding="utf-8").splitlines()) >= 1
    base_audit = tmp_path / "audit.yaml"
    base_audit.write_text(yaml.safe_dump({"workspace_root": ".", "output_root": "results/audits",
                                          "split_resources": ["train", "validation", "test"]}), encoding="utf-8")
    prepared = prepare(config, run_dir, base_audit)
    audit_config = yaml.safe_load(prepared.read_text(encoding="utf-8"))
    assert audit_config["split_resources"] == []
    assert audit_config["input_is_normalized_view"] is True
    diagnostic_dir = tmp_path / "results" / "diagnostics"
    run_diagnostics(config, diagnostic_dir, academic_sample=1, hash_sample=1)
    diagnostic = json.loads((diagnostic_dir / "summary.json").read_text(encoding="utf-8"))
    assert diagnostic["counts"]["content_hash_mismatch"] == 2
    assert diagnostic["counts"]["word_count_mismatch"] == 2
    repair_dir = tmp_path / "results" / "hash-repair"
    verify_hash_repair(config, diagnostic_dir, run_dir, repair_dir)
    repair = json.loads((repair_dir / "summary.json").read_text(encoding="utf-8"))
    assert repair["mismatched_source_hashes"] == 2
    assert repair["different_text_same_sha256_pairs"] == 0
    n2_config = tmp_path / "n2.yaml"
    n2_raw = {**normal_cfg, "rules": {"stage": "n2", "remove_strict_terminal_references": True}}
    n2_config.write_text(yaml.safe_dump(n2_raw), encoding="utf-8")
    preview_dir = tmp_path / "results" / "preview"
    preview_n2(n2_config, run_dir, preview_dir, examples_per_kind=1)
    preview = json.loads((preview_dir / "summary.json").read_text(encoding="utf-8"))
    assert preview["counts"]["rows"] == 2
    assert preview["counts"].get("candidate_rows", 0) == 0
