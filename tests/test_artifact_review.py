from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from horus_benchmark.audit.artifact_review import (
    AREAS, choose_sample, length_bin, read_pilot, select_diagnostic, stable_priority,
)


def _record(rid: str, area: str, label: str, priority: str, **extra: str) -> dict:
    return {
        "record_id": rid,
        "broad_area": area,
        "label_name": label,
        "selection_priority": priority,
        "source_name": extra.get("source_name", "source-a"),
        "generator": extra.get("generator", "model-a"),
        "length_bin": extra.get("length_bin", "150-399"),
        "split": extra.get("split", "train"),
        "theme": extra.get("theme", "theme-a"),
        "genre": extra.get("genre", "genre-a"),
    }


def test_stable_priority_ignores_corpus_order_and_separates_seeds() -> None:
    assert stable_priority(7, "r1") == stable_priority(7, "r1")
    assert stable_priority(7, "r1") != stable_priority(8, "r1")
    assert stable_priority(7, "r1") != stable_priority(7, "r2")


@pytest.mark.parametrize(
    ("words", "expected"),
    [(0, "0-149"), (149, "0-149"), (150, "150-399"),
     (399, "150-399"), (400, "400-799"), (799, "400-799"), (800, "800+")],
)
def test_length_bin_boundaries(words: int, expected: str) -> None:
    assert length_bin(words) == expected


def test_diagnostic_rows_are_distinct_from_core_and_cover_length() -> None:
    core = [_record("core", "academic", "synthetic", "0", length_bin="0-149")]
    candidates = core + [
        _record("same", "academic", "synthetic", "1", length_bin="0-149"),
        _record("other", "academic", "synthetic", "2", length_bin="800+"),
    ]
    selected = select_diagnostic(candidates, core, 1, ("length_bin",))
    assert [row["record_id"] for row in selected] == ["other"]


def test_sample_preserves_pilot_and_separates_core_from_diagnostic() -> None:
    pilot = {}
    humans = {}
    synthetics = {}
    for area in AREAS:
        rows = [
            _record(f"h-{area}-{i}", area, "human", f"{i:03d}",
                    source_name=f"source-{i % 3}",
                    length_bin=("0-149", "150-399", "400-799", "800+")[i % 4])
            for i in range(42)
        ]
        humans[area] = rows
        pilot.update((row["record_id"], row) for row in rows[:20])
        for generator in ("model-a", "model-b"):
            synthetics[(area, generator)] = [
                _record(f"s-{area}-{generator}-{i}", area, "synthetic", f"{i:03d}",
                        generator=generator,
                        length_bin=("0-149", "150-399", "400-799", "800+")[i % 4])
                for i in range(8)
            ]
    selected = choose_sample(pilot, humans, synthetics)
    counts = Counter(row["sample_role"] for row in selected)
    assert counts == {
        "core_human_pilot": 120,
        "diagnostic_human": 120,
        "core_synthetic": 24,
        "diagnostic_synthetic": 24,
    }
    assert {row["record_id"] for row in selected if row["sample_role"] == "core_human_pilot"} == set(pilot)
    assert len({row["record_id"] for row in selected}) == len(selected)


def test_pilot_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "pilot.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "broad_area", "content"])
        writer.writeheader()
        for area in AREAS:
            for i in range(20):
                writer.writerow({"record_id": "duplicate" if i == 0 else f"{area}-{i}",
                                 "broad_area": area, "content": "example"})
    with pytest.raises(ValueError, match="120 distinct"):
        read_pilot(path)
