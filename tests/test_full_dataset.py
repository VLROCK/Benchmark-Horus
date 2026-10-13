"""Structural integration checks for the complete, unnormalized corpus."""

from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds
import pytest

from horus_benchmark.config import load_yaml
from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.data.loader import compare_schema


CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "datasets" / "horus_full.yaml"
)


@pytest.fixture(scope="module")
def full_catalog() -> DatasetCatalog:
    catalog = DatasetCatalog.from_yaml(CATALOG_PATH)
    if catalog.missing_files():
        pytest.skip("Complete Horus corpus is not available locally")
    return catalog


@pytest.mark.integration
def test_full_resource_counts_and_schemas(full_catalog: DatasetCatalog) -> None:
    assert full_catalog.schema_manifest_path is not None
    declared = load_yaml(full_catalog.schema_manifest_path)["columns"]
    for name in ("corpus", "train", "validation", "test"):
        resource = full_catalog.resource(name)
        loader = DatasetLoader(resource)
        assert loader.count_rows() == resource.expected_rows
        assert compare_schema(loader.schema, declared) == []


@pytest.mark.integration
def test_full_labels_and_split_coverage(full_catalog: DatasetCatalog) -> None:
    corpus = DatasetLoader(full_catalog.resource("corpus"))
    assert corpus.count_rows(ds.field("label_name") == "human") == 97_659
    assert corpus.count_rows(ds.field("label_name") == "synthetic") == 97_495

    corpus_ids = set(corpus.read_table(columns=["record_id"])["record_id"].to_pylist())
    split_ids = []
    for name in ("train", "validation", "test"):
        identifiers = (
            DatasetLoader(full_catalog.resource(name))
            .read_table(columns=["record_id"])["record_id"]
            .to_pylist()
        )
        assert len(identifiers) == len(set(identifiers))
        split_ids.append(set(identifiers))
    assert sum(map(len, split_ids)) == len(set().union(*split_ids))
    assert set().union(*split_ids) == corpus_ids
