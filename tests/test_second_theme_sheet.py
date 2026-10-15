from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_second_theme_sheet import prepare  # noqa: E402


def test_second_sheet_keeps_ids_and_texts_but_never_first_answers(tmp_path: Path) -> None:
    source = tmp_path / "pilot_annotation.csv"
    destination = tmp_path / "second_annotation.csv"
    columns = ["record_id", "broad_area", "content", "theme_human", "annotation_note", "model_prediction"]
    areas = ("academic", "informational", "journalistic", "juridical", "literary", "social")
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for area in areas:
            for index in range(20):
                writer.writerow({
                    "record_id": f"{area}-{index}", "broad_area": area,
                    "content": f"Texto {area} {index}",
                    "theme_human": "tema privado" if index == 0 else "",
                    "annotation_note": "nota privada" if index == 0 else "",
                    "model_prediction": "tema privado",
                })
    original = source.read_bytes()

    prepare(source, destination)

    with destination.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120
    assert len({row["record_id"] for row in rows}) == 120
    assert all(row["content"] and not row["theme_human"] and not row["annotation_note"] for row in rows)
    assert "model_prediction" not in rows[0]
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare(source, destination)
    assert source.read_bytes() == original
