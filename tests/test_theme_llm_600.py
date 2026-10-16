"""Checks for sampling and agreement that do not load an LLM."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from horus_benchmark.audit.theme_common import ThemeContext, sha256_file
from horus_benchmark.audit.theme_llm_600 import (
    create_sample, evaluate_agreement, pair_metrics, prepare_inference,
    read_llm_labels, render_prompt,
)
from horus_benchmark.audit.theme_llm_worker import parse_theme, read_requests
from horus_benchmark.data import DatasetCatalog


AREAS = ("academic", "informational", "journalistic", "juridical", "literary", "social")


def test_parser_accepts_only_exact_json_taxonomy_label() -> None:
    allowed = {"Antropologia", "Biologia", "incerto", "fora_taxonomia"}
    assert parse_theme('{"theme":"Antropologia"}', allowed) == "Antropologia"
    assert parse_theme('pensei nisso {"theme":"Biologia"}', allowed) == "Biologia"
    assert parse_theme('{"theme":"Astrologia"}', allowed) is None
    assert parse_theme('O tema é Biologia.', allowed) is None


def test_area_prompt_lists_every_theme_and_examples() -> None:
    class Context:
        taxonomy = {"academic": ("Antropologia", "Biologia")}

    prompts = {
        "common": "Escolha um tema.",
        "areas": {"academic": {"name": "Acadêmica", "guidance": "Tema central.",
                               "examples": [{"text": "Etnografia", "theme": "Antropologia"}]}},
    }
    prompt = render_prompt(Context(), prompts, "academic", "Texto de teste")
    assert "- Antropologia" in prompt and "- Biologia" in prompt
    assert "Etnografia -> Antropologia" in prompt
    assert "Texto de teste" in prompt


def test_kappa_is_conditional_on_known_area() -> None:
    frame = pd.DataFrame([
        {"broad_area": "academic", "left": "A", "right": "A"},
        {"broad_area": "academic", "left": "B", "right": "B"},
        {"broad_area": "social", "left": "C", "right": "D"},
        {"broad_area": "social", "left": "D", "right": "D"},
    ])
    result = pair_metrics(frame, "left", "right")
    assert result["raw_agreement"] == pytest.approx(0.75)
    assert result["kappa_within_area"] == pytest.approx(0.5)
    single = pd.DataFrame([{"broad_area": "academic", "left": "A", "right": "A"}])
    assert pair_metrics(single, "left", "right")["kappa_global"] is None


def test_sample_reuses_120_ids_and_never_exposes_raw_theme(tmp_path: Path) -> None:
    corpus = tmp_path / "all.parquet"
    records = [
        {"record_id": f"{area}-{i:03}", "label_name": "human",
         "broad_area": area, "source_name": "source", "split": "train",
         "word_count": 4, "theme": "private metadata", "content": f"Texto {area} {i}"}
        for area in AREAS for i in range(120)
    ]
    pd.DataFrame(records).to_parquet(corpus, index=False)
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        "dataset_id: toy\ndataset_version: v1\nresources:\n"
        f"  corpus:\n    path: '{corpus.as_posix()}'\n"
        "    format: parquet\n    role: corpus\n", encoding="utf-8",
    )
    taxonomy_file = tmp_path / "taxonomy.csv"
    taxonomy_file.write_text(
        "domain,theme\n" + "".join(f"{area},A\n" for area in AREAS), encoding="utf-8",
    )
    mapping_file = tmp_path / "map.csv"
    mapping_file.write_text("broad_area,raw_theme,canonical_theme\n", encoding="utf-8")
    prompts_file = tmp_path / "prompts.yaml"
    prompts_file.write_text(
        "version: 1\ncommon: Classifique.\nareas:\n" + "".join(
            f"  {area}:\n    name: {area}\n    guidance: tema\n    examples: []\n"
            for area in AREAS
        ), encoding="utf-8",
    )
    context = ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={"experiment_id": "toy-llm", "seed": 7, "corpus_resource": "corpus",
                "sampling": {"method": "pilot_plus_uniform_without_replacement",
                             "pilot_per_area": 20, "total_per_area": 100},
                "prompts": "prompts.yaml",
                "llm": {"max_input_characters": 6000, "models": {}}},
        catalog=DatasetCatalog.from_yaml(catalog_file),
        taxonomy={area: ("A",) for area in AREAS}, aliases={},
        taxonomy_path=taxonomy_file, mapping_path=mapping_file,
        output_root=tmp_path / "results",
    )
    pilot_run = context.new_run("pilot", extra={"corpus_sha256": sha256_file(corpus)})
    pilot = pilot_run.directory / "pilot_annotation.csv"
    with pilot.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "broad_area", "content", "theme_human"])
        writer.writeheader()
        for area in AREAS:
            for i in range(20):
                writer.writerow({"record_id": f"{area}-{i:03}", "broad_area": area,
                                 "content": f"Texto {area} {i}", "theme_human": "A"})

    sample = create_sample(context, pilot)
    requests = prepare_inference(context, sample)

    frame = pd.read_parquet(sample)
    assert len(frame) == 600
    assert frame["record_id"].nunique() == 600
    assert frame.groupby("broad_area").size().to_dict() == {area: 100 for area in AREAS}
    assert (frame["sample_origin"] == "existing_pilot").sum() == 120
    assert "theme" not in frame and "theme_human" not in frame
    assert json.loads((sample.parent / "sample_summary.json").read_text(encoding="utf-8"))["human_labels_used"] is False
    prepared, metadata = read_requests(requests)
    assert len(prepared) == 600 and metadata["manifest_sha256"] == sha256_file(sample)
    assert all("private metadata" not in row["prompt"] for row in prepared)
    assert all("source_name" not in row and "theme_human" not in row for row in prepared)

    model_dir = tmp_path / "gemma"
    model_dir.mkdir()
    model_metadata = metadata | {"model_key": "gemma4_31b"}
    (model_dir / "metadata.json").write_text(
        json.dumps(model_metadata), encoding="utf-8",
    )
    annotations = [
        {"record_id": row["record_id"], "broad_area": row["broad_area"],
         "theme": "A", "status": "valid", "input_truncated": False}
        for row in prepared
    ]
    annotations_path = model_dir / "annotations.jsonl"
    annotations_path.write_text(
        "\n".join(json.dumps(row) for row in annotations) + "\n",
        encoding="utf-8",
    )
    labels = read_llm_labels(context, annotations_path, sample, "gemma4_31b")
    assert len(labels) == 600
    assert set(labels["gemma4_31b"]) == {"A"}

    requests.write_bytes(requests.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        read_requests(requests)


def test_evaluation_uses_same_600_ids_and_reports_agreement_only(
    tmp_path: Path, monkeypatch,
) -> None:
    corpus = tmp_path / "all.parquet"
    corpus.write_bytes(b"toy corpus")
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        "dataset_id: toy\ndataset_version: v1\nresources:\n"
        f"  corpus:\n    path: '{corpus.as_posix()}'\n"
        "    format: parquet\n    role: corpus\n", encoding="utf-8",
    )
    taxonomy_file = tmp_path / "taxonomy.csv"
    taxonomy_file.write_text(
        "domain,theme\n" + "".join(f"{area},A\n{area},B\n" for area in AREAS),
        encoding="utf-8",
    )
    mapping_file = tmp_path / "map.csv"
    mapping_file.write_text("broad_area,raw_theme,canonical_theme\n", encoding="utf-8")
    context = ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={"experiment_id": "toy-eval", "seed": 3, "corpus_resource": "corpus",
                "threshold": {"fixed_probability": 0.95},
                "analysis": {"bootstrap_replicates": 5, "min_source_rows": 20}},
        catalog=DatasetCatalog.from_yaml(catalog_file),
        taxonomy={area: ("A", "B") for area in AREAS}, aliases={},
        taxonomy_path=taxonomy_file, mapping_path=mapping_file,
        output_root=tmp_path / "results",
    )
    frame = pd.DataFrame([
        {"record_id": f"{area}-{index:03}", "broad_area": area,
         "source_name": "source", "split": "train", "word_count": index + 1,
         "sample_origin": "new_uniform", "content": "texto"}
        for area in AREAS for index in range(100)
    ])
    bert = frame[["record_id", "broad_area"]].assign(
        theme_predicted="A", top_probability=0.96, mapped_theme=None,
        mapping_method="unmapped", model_run_id="toy",
    )
    sample = tmp_path / "sample.parquet"
    gemma_file = tmp_path / "gemma.jsonl"
    gpt_file = tmp_path / "gpt.jsonl"
    bert_file = tmp_path / "bert.parquet"
    for path in (sample, gemma_file, gpt_file, bert_file):
        path.write_bytes(b"fingerprint")
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_600.checked_manifest",
        lambda _context, _path: frame,
    )
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_600.read_llm_labels",
        lambda _context, _path, _sample, key: frame[["record_id", "broad_area"]].assign(
            **{key: "A", f"{key}_input_truncated": False},
        ),
    )
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_workflow._predictions",
        lambda _context, _path: bert,
    )

    report_file = evaluate_agreement(context, sample, gemma_file, gpt_file, bert_file)

    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["sample_rows"] == 600
    assert report["human_reference"] is False
    assert report["three_way_forced_agreement"] == 1.0
    assert report["pairs"]["gemma4_31b__gpt_oss_120b"]["rows"] == 600
    threshold = json.loads((report_file.parent / "fixed_bert_threshold.json").read_text(encoding="utf-8"))
    assert threshold["threshold"] == 0.95
    assert (report_file.parent / "agreement_rows.csv").is_file()
