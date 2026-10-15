from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from horus_benchmark.audit.theme_common import ThemeContext, normalized_key, sha256_file
from horus_benchmark.audit.theme_annotation_ui import AnnotationSheet, _page
from horus_benchmark.audit.theme_workflow import (
    _area_stratified_kappa, _assigned, choose_threshold, export_review_sample, materialize_themes,
    proportional_allocation,
)
from horus_benchmark.data import DatasetCatalog


def test_manual_mapping_is_area_conditional() -> None:
    context = ThemeContext(
        config_path=Path("unused.yaml"), config={}, catalog=None,  # type: ignore[arg-type]
        taxonomy={
            "academic": ("Biologia", "Matemática"),
            "informational": ("Saúde",),
        },
        aliases={("academic", normalized_key("Biological Sciencie")): "Biologia"},
        taxonomy_path=Path("unused.csv"), mapping_path=Path("unused-map.csv"),
        output_root=Path("unused"),
    )
    assert context.canonical_existing("academic", " Biological  Sciencie ") == (
        "Biologia", "manual_mapping",
    )
    assert context.canonical_existing("academic", "Biologia") == (
        "Biologia", "exact_taxonomy",
    )
    assert context.canonical_existing("informational", "Biological Sciencie") == (
        None, "unmapped",
    )


def test_blind_annotation_form_saves_only_valid_area_theme(tmp_path: Path) -> None:
    sheet_path = tmp_path / "pilot_annotation.csv"
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "record_id", "broad_area", "content", "theme_human", "annotation_note",
        ])
        writer.writeheader()
        writer.writerow({"record_id": "h1", "broad_area": "academic",
                         "content": "<texto humano>", "theme_human": "", "annotation_note": ""})
    sheet = AnnotationSheet(sheet_path, {"academic": ("Biologia", "Química")})
    page = _page(sheet, 0, "test-token")
    assert "&lt;texto humano&gt;" in page
    assert "<texto humano>" not in page
    assert "Biologia" in page and "fora taxonomia" in page
    with pytest.raises(ValueError):
        sheet.save(0, "h1", "Direito civil", "")
    sheet.save(0, "h1", "Biologia", "anotado")
    restored = AnnotationSheet(sheet_path, {"academic": ("Biologia", "Química")})
    assert restored.completed() == 1
    assert restored.rows[0]["theme_human"] == "Biologia"
    assert restored.rows[0]["annotation_note"] == "anotado"


def test_threshold_uses_lower_confidence_bound_and_can_abstain() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    right = np.array([True, True, True, True, False])
    selected = choose_threshold(
        scores, right, target_precision=0.5, min_accepted=2, confidence=0.95,
    )
    assert selected["threshold"] == 0.6
    assert selected["accepted"] == 4
    impossible = choose_threshold(
        scores, right, target_precision=0.99, min_accepted=2, confidence=0.95,
    )
    assert impossible["threshold"] > 1


def test_mapping_precedes_model_and_uncertain_is_retained() -> None:
    frame = pd.DataFrame([
        {"mapped_theme": "Biologia", "theme_predicted": "Matemática", "top_probability": 0.99},
        {"mapped_theme": None, "theme_predicted": "Matemática", "top_probability": 0.95},
        {"mapped_theme": None, "theme_predicted": "Biologia", "top_probability": 0.60},
    ])
    assert _assigned(frame, 0.9).tolist() == ["Biologia", "Matemática", "incerto"]


def test_source_strata_allocation_preserves_sample_size() -> None:
    counts = pd.Series({"small": 10, "medium": 20, "large": 70})
    allocated = proportional_allocation(counts, 10)
    assert allocated.to_dict() == {"small": 1, "medium": 2, "large": 7}
    assert (allocated <= counts).all()
    rare = proportional_allocation(pd.Series({"a": 1, "b": 1, "c": 98}), 10)
    assert rare.to_dict() == {"a": 1, "b": 1, "c": 8}


def test_area_stratified_kappa_uses_within_area_chance() -> None:
    paired = pd.DataFrame([
        {"broad_area": "academic", "theme_human_primary": "A", "theme_human_second": "A"},
        {"broad_area": "academic", "theme_human_primary": "B", "theme_human_second": "B"},
        {"broad_area": "social", "theme_human_primary": "C", "theme_human_second": "D"},
        {"broad_area": "social", "theme_human_primary": "D", "theme_human_second": "D"},
    ])
    assert _area_stratified_kappa(paired) == pytest.approx(0.5)


def test_review_sample_has_disjoint_roles_and_excludes_pilot(
    tmp_path: Path, monkeypatch,
) -> None:
    areas = ("academic", "informational", "journalistic", "juridical", "literary", "social")
    source = tmp_path / "all.parquet"
    source.write_bytes(b"toy source fingerprint")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        f"dataset_id: toy\ndataset_version: v1\nresources:\n"
        f"  corpus:\n    path: '{source.as_posix()}'\n"
        f"    format: parquet\n    role: corpus\n", encoding="utf-8",
    )
    taxonomy_path = tmp_path / "taxonomy.csv"
    taxonomy_path.write_text(
        "domain,theme\n" + "".join(f"{area},A\n" for area in areas),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text("broad_area,raw_theme,canonical_theme\n", encoding="utf-8")
    context = ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={
            "experiment_id": "toy-sample", "seed": 7, "corpus_resource": "corpus",
            "sampling": {"pilot_per_area": 1, "random_per_area": 100,
                         "second_annotator_per_area": 20,
                         "diagnostic_total": 200},
        },
        catalog=DatasetCatalog.from_yaml(catalog_path),
        taxonomy={area: ("A",) for area in areas}, aliases={},
        taxonomy_path=taxonomy_path, mapping_path=mapping_path,
        output_root=tmp_path / "results",
    )
    prediction_rows = [
        {"record_id": f"{area}_{i}", "broad_area": area,
         "source_name": "s1" if i % 2 else "s2", "theme_predicted": "A",
         "top_probability": 0.5 + i / 1000, "mapped_theme": None}
        for area in areas for i in range(160)
    ]
    predictions = pd.DataFrame(prediction_rows)
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_workflow._predictions",
        lambda _context, _path: predictions,
    )
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_workflow._blind_sheet",
        lambda _context, selected: selected[["record_id", "broad_area"]],
    )
    prediction_file = tmp_path / "predictions.parquet"
    prediction_file.write_bytes(b"prediction fingerprint")
    pilot_run = context.new_run("pilot", extra={"corpus_sha256": sha256_file(source)})
    pilot = pilot_run.directory / "pilot_annotation.csv"
    pilot_ids = {f"{area}_0" for area in areas}
    pd.DataFrame({"record_id": sorted(pilot_ids)}).to_csv(pilot, index=False)
    manifest_path = export_review_sample(context, prediction_file, pilot)
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 800
    assert manifest["record_id"].is_unique
    assert not pilot_ids.intersection(manifest["record_id"])
    assert manifest["sample_role"].value_counts().to_dict() == {
        "calibration": 300, "evaluation": 300, "diagnostic": 200,
    }
    assert int(manifest["second_annotator"].sum()) == 120
    assert manifest.loc[manifest["second_annotator"]].groupby("broad_area").size().to_dict() == {
        area: 20 for area in areas
    }
    random_sample = manifest.loc[manifest["sample_role"] != "diagnostic"]
    assert random_sample.groupby("broad_area").size().to_dict() == {
        area: 100 for area in areas
    }
    context.config["sampling"]["protocol"] = "fixed_120"
    with pytest.raises(ValueError, match="piloto como amostra final"):
        export_review_sample(context, prediction_file, pilot)


def test_materialization_preserves_source_and_all_record_ids(tmp_path: Path) -> None:
    source = tmp_path / "all.parquet"
    rows = [
        {"record_id": "h1", "label_name": "human", "broad_area": "academic",
         "theme": "Biological Sciencie", "split": "train", "benchmark_eligible": True,
         "source_name": "example", "generator": None, "content": "Texto humano."},
        {"record_id": "h2", "label_name": "human", "broad_area": "academic",
         "theme": None, "split": "test", "benchmark_eligible": False,
         "source_name": "example", "generator": None, "content": "Outro texto."},
        {"record_id": "s1", "label_name": "synthetic", "broad_area": "academic",
         "theme": "Matemática", "split": "train", "benchmark_eligible": True,
         "source_name": None, "generator": "g", "content": "Texto sintético."},
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        f"dataset_id: toy\ndataset_version: v1\nresources:\n"
        f"  corpus:\n    path: '{source.as_posix()}'\n    format: parquet\n"
        f"    role: processed_canonical_corpus\n    expected_rows: 3\n",
        encoding="utf-8",
    )
    taxonomy_path = tmp_path / "taxonomy.csv"
    taxonomy_path.write_text("domain,theme\nacademic,Biologia\nacademic,Matemática\n", encoding="utf-8")
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text(
        "broad_area,raw_theme,canonical_theme\n"
        "academic,Biological Sciencie,Biologia\n", encoding="utf-8",
    )
    context = ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={"experiment_id": "toy-theme", "seed": 1, "corpus_resource": "corpus",
                "batch_size": 2},
        catalog=DatasetCatalog.from_yaml(catalog_path),
        taxonomy={"academic": ("Biologia", "Matemática")},
        aliases={("academic", normalized_key("Biological Sciencie")): "Biologia"},
        taxonomy_path=taxonomy_path, mapping_path=mapping_path,
        output_root=tmp_path / "results",
    )
    prediction_run = context.new_run("predict", extra={"corpus_sha256": sha256_file(source)})
    predictions = prediction_run.directory / "human_predictions.parquet"
    pd.DataFrame([
        {"record_id": "h1", "broad_area": "academic", "theme_original": "Biological Sciencie",
         "mapped_theme": "Biologia", "mapping_method": "manual_mapping",
         "theme_predicted": "Matemática", "top_probability": 0.99, "model_run_id": "model-v1"},
        {"record_id": "h2", "broad_area": "academic", "theme_original": None,
         "mapped_theme": None, "mapping_method": "unmapped",
         "theme_predicted": "Biologia", "top_probability": 0.60, "model_run_id": "model-v1"},
    ]).to_parquet(predictions, index=False)
    threshold_run = context.new_run("evaluate")
    threshold = threshold_run.directory / "threshold.json"
    threshold.write_text(json.dumps({
        "threshold": 0.9, "predictions_sha256": sha256_file(predictions),
    }), encoding="utf-8")
    original_hash = sha256_file(source)
    derived = materialize_themes(context, predictions, threshold)
    assert sha256_file(source) == original_hash
    output = pq.read_table(derived).to_pandas().set_index("record_id")
    assert set(output.index) == {"h1", "h2", "s1"}
    assert output.loc["h1", "theme"] == "Biologia"
    assert output.loc["h1", "theme_original"] == "Biological Sciencie"
    assert output.loc["h2", "theme"] == "incerto"
    assert output.loc["s1", "theme_assignment_method"] == "generation_prompt"
    comparison = pd.read_csv(derived.parent / "theme_comparison_full.csv")
    biology = comparison.loc[comparison["theme_canonical"] == "Biologia"].iloc[0]
    assert biology["human_rows"] == 1
    assert biology["synthetic_rows"] == 0
