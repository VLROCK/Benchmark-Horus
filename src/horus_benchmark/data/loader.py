"""Centralized PyArrow loader for the formats present in the workspace."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from horus_benchmark.data.catalog import ResourceSpec


class DatasetLoader:
    """Read one catalog resource with projection, filtering and batching."""

    def __init__(self, resource: ResourceSpec):
        self.resource = resource

    @cached_property
    def dataset(self) -> ds.Dataset:
        if not self.resource.path.is_file():
            raise FileNotFoundError(f"Dataset resource not found: {self.resource.path}")
        return ds.dataset(str(self.resource.path), format=self.resource.format)

    @property
    def schema(self) -> pa.Schema:
        return self.dataset.schema

    def count_rows(self, filter_expression: ds.Expression | None = None) -> int:
        return self.dataset.count_rows(filter=filter_expression)

    def read_table(
        self,
        *,
        columns: Sequence[str] | None = None,
        filter_expression: ds.Expression | None = None,
        limit: int | None = None,
        batch_size: int = 65_536,
        use_threads: bool = True,
    ) -> pa.Table:
        """Read a projected table, optionally stopping after ``limit`` rows."""

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        projected_columns = list(columns) if columns is not None else None
        if limit is not None:
            return self.dataset.head(
                limit,
                columns=projected_columns,
                filter=filter_expression,
                batch_size=batch_size,
                use_threads=use_threads,
            )
        scanner = self.dataset.scanner(
            columns=projected_columns,
            filter=filter_expression,
            batch_size=batch_size,
            use_threads=use_threads,
        )
        return scanner.to_table()

    def iter_batches(
        self,
        *,
        columns: Sequence[str] | None = None,
        filter_expression: ds.Expression | None = None,
        batch_size: int = 65_536,
        use_threads: bool = True,
    ) -> Iterator[pa.RecordBatch]:
        """Yield bounded batches without materializing the full resource."""

        scanner = self.dataset.scanner(
            columns=list(columns) if columns is not None else None,
            filter=filter_expression,
            batch_size=batch_size,
            use_threads=use_threads,
        )
        yield from scanner.to_batches()

    def physical_metadata(self) -> dict[str, Any]:
        """Return non-content file metadata useful for inventory checks."""

        result: dict[str, Any] = {
            "path": str(self.resource.path),
            "format": self.resource.format,
            "bytes": self.resource.path.stat().st_size,
        }
        if self.resource.format != "parquet":
            return result

        parquet = pq.ParquetFile(self.resource.path)
        compressions = {
            parquet.metadata.row_group(row_group).column(column).compression
            for row_group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
        result.update(
            {
                "rows": parquet.metadata.num_rows,
                "columns": parquet.metadata.num_columns,
                "row_groups": parquet.metadata.num_row_groups,
                "compression": sorted(compressions),
                "created_by": parquet.metadata.created_by,
            }
        )
        return result


def compare_schema(
    actual: pa.Schema, declared_columns: dict[str, str]
) -> list[str]:
    """Return human-readable differences against a declared Arrow schema."""

    actual_columns = {field.name: str(field.type) for field in actual}
    issues: list[str] = []
    for name, declared_type in declared_columns.items():
        if name not in actual_columns:
            issues.append(f"missing column: {name}")
        elif actual_columns[name] != declared_type:
            issues.append(
                f"type mismatch for {name}: "
                f"expected {declared_type}, found {actual_columns[name]}"
            )
    for name in actual_columns.keys() - declared_columns.keys():
        issues.append(f"undeclared column: {name}")
    return sorted(issues)

