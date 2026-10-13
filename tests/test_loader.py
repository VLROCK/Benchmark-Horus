from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horus_benchmark.data import DatasetLoader, ResourceSpec
from horus_benchmark.data.loader import compare_schema


def _resource(path: Path, resource_format: str) -> ResourceSpec:
    return ResourceSpec(
        name="fixture",
        path=path,
        format=resource_format,
        role="test_fixture",
        expected_rows=3,
    )


def test_parquet_loader_projects_filters_and_batches(tmp_path: Path) -> None:
    path = tmp_path / "fixture.parquet"
    table = pa.table(
        {
            "record_id": ["a", "b", "c"],
            "label": [0, 1, 1],
            "content": ["human", "synthetic one", "synthetic two"],
        }
    )
    pq.write_table(table, path, compression="zstd")
    loader = DatasetLoader(_resource(path, "parquet"))

    projected = loader.read_table(
        columns=["record_id", "label"],
        filter_expression=ds.field("label") == 1,
    )
    batches = list(loader.iter_batches(columns=("record_id",), batch_size=2))

    assert loader.count_rows() == 3
    assert projected.column_names == ["record_id", "label"]
    assert projected["record_id"].to_pylist() == ["b", "c"]
    assert [batch.num_rows for batch in batches] == [2, 1]
    assert loader.physical_metadata()["row_groups"] == 1


def test_csv_loader_and_limited_read(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    table = pa.table({"id": [1, 2, 3], "theme": ["a", "b", "c"]})
    pacsv.write_csv(table, path)
    loader = DatasetLoader(_resource(path, "csv"))

    limited = loader.read_table(columns=["theme"], limit=2)

    assert loader.count_rows() == 3
    assert limited["theme"].to_pylist() == ["a", "b"]


def test_schema_comparison_reports_missing_extra_and_type_mismatch() -> None:
    schema = pa.schema([("record_id", pa.string()), ("label", pa.int64())])
    differences = compare_schema(
        schema,
        {"record_id": "large_string", "content": "large_string"},
    )

    assert "missing column: content" in differences
    assert "undeclared column: label" in differences
    assert any(item.startswith("type mismatch for record_id") for item in differences)

