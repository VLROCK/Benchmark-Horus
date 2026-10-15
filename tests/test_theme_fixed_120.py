from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from horus_benchmark.audit.theme_common import ThemeContext, sha256_file
from horus_benchmark.audit.theme_workflow import evaluate_fixed_pilot
from horus_benchmark.data import DatasetCatalog


AREAS = ("academic", "informational", "journalistic", "juridical", "literary", "social")


def test_fixed_120_uses_all_shared_ids_without_human_threshold_search(
    tmp_path: Path, monkeypatch,
) -> None:
    corpus = tmp_path / "all.parquet"
    corpus.write_bytes(b"fixed corpus fingerprint")
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        "dataset_id: toy\ndataset_version: v1\nresources:\n"
        f"  corpus:\n    path: '{corpus.as_posix()}'\n"
        "    format: parquet\n    role: corpus\n",
        encoding="utf-8",
    )
    taxonomy_file = tmp_path / "taxonomy.csv"
    taxonomy_file.write_text(
        "domain,theme\n" + "".join(f"{area},A\n{area},B\n" for area in AREAS),
        encoding="utf-8",
    )
    mapping_file = tmp_path / "mapping.csv"
    mapping_file.write_text("broad_area,raw_theme,canonical_theme\n", encoding="utf-8")
    context = ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={
            "experiment_id": "toy-fixed", "seed": 42, "corpus_resource": "corpus",
            "sampling": {"protocol": "fixed_120", "pilot_per_area": 20},
            "threshold": {"policy": "fixed", "fixed_probability": 0.95,
                          "wilson_confidence": 0.95},
        },
        catalog=DatasetCatalog.from_yaml(catalog_file),
        taxonomy={area: ("A", "B") for area in AREAS}, aliases={},
        taxonomy_path=taxonomy_file, mapping_path=mapping_file,
        output_root=tmp_path / "results",
    )
    records = [
        {"record_id": f"{area}-{index}", "broad_area": area,
         "source_name": "source", "mapped_theme": None,
         "theme_predicted": "A" if index % 2 == 0 else "B",
         "top_probability": 0.96 if index % 2 == 0 else 0.50}
        for area in AREAS for index in range(20)
    ]
    predictions = pd.DataFrame.from_records(records)
    prediction_file = tmp_path / "predictions.parquet"
    prediction_file.write_bytes(b"prediction fingerprint")
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_workflow._predictions",
        lambda _context, _path: predictions,
    )
    pilot_run = context.new_run("pilot", extra={"corpus_sha256": sha256_file(corpus)})
    pilot = pilot_run.directory / "pilot_annotation.csv"
    primary = tmp_path / "primary_annotation.csv"
    second = tmp_path / "second_annotation.csv"
    columns = ["record_id", "broad_area", "content", "theme_human", "annotation_note"]
    for destination in (pilot, primary, second):
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in records:
                writer.writerow({
                    "record_id": row["record_id"], "broad_area": row["broad_area"],
                    "content": "texto", "theme_human": (
                        "" if destination == pilot else row["theme_predicted"]
                    ),
                    "annotation_note": "",
                })

    threshold_path = evaluate_fixed_pilot(
        context, prediction_file, pilot, primary, second,
    )

    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    report = json.loads((threshold_path.parent / "evaluation.json").read_text(encoding="utf-8"))
    assert threshold["threshold"] == 0.95
    assert threshold["policy"] == "fixed_before_human_evaluation"
    assert report["sample"]["unique_texts"] == 120
    assert report["model_vs_primary"]["accepted"] == 60
    assert report["model_vs_second"]["accepted"] == 60
    assert report["human_human_agreement"]["cohen_kappa_global"] == 1.0
    assert report["human_human_agreement"]["cohen_kappa_area_stratified"] == 1.0
