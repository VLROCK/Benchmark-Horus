from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds
import pytest

from horus_benchmark.config import load_yaml
from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.data.loader import compare_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "configs" / "datasets" / "horus_rc.yaml"


@pytest.fixture(scope="module")
def catalog() -> DatasetCatalog:
    loaded = DatasetCatalog.from_yaml(CATALOG_PATH)
    if loaded.missing_files():
        pytest.skip("Local Horus release snapshot is not available")
    return loaded


@pytest.mark.integration
def test_registered_counts_and_parquet_schemas(catalog: DatasetCatalog) -> None:
    assert catalog.schema_manifest_path is not None
    declared = load_yaml(catalog.schema_manifest_path)["columns"]
    for name in ("corpus", "train", "validation", "test", "benchmark_balanced"):
        resource = catalog.resource(name)
        loader = DatasetLoader(resource)
        assert loader.count_rows() == resource.expected_rows
        assert compare_schema(loader.schema, declared) == []


@pytest.mark.integration
def test_labels_are_consistent(catalog: DatasetCatalog) -> None:
    loader = DatasetLoader(catalog.resource("corpus"))
    assert loader.count_rows(ds.field("label_name") == "human") == 97_659
    assert loader.count_rows(ds.field("label_name") == "synthetic") == 86_252
    inconsistent = (
        ((ds.field("label") == 0) & (ds.field("label_name") != "human"))
        | ((ds.field("label") == 1) & (ds.field("label_name") != "synthetic"))
        | ~ds.field("label").isin([0, 1])
    )
    assert loader.count_rows(inconsistent) == 0


@pytest.mark.integration
def test_split_ids_are_disjoint_and_cover_corpus(catalog: DatasetCatalog) -> None:
    corpus_ids = set(
        DatasetLoader(catalog.resource("corpus"))
        .read_table(columns=["record_id"])["record_id"]
        .to_pylist()
    )
    split_ids = {
        name: set(
            DatasetLoader(catalog.resource(name))
            .read_table(columns=["record_id"])["record_id"]
            .to_pylist()
        )
        for name in ("train", "validation", "test")
    }

    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert set.union(*split_ids.values()) == corpus_ids


@pytest.mark.integration
def test_synthetic_rows_follow_declared_taxonomy(catalog: DatasetCatalog) -> None:
    taxonomy_rows = DatasetLoader(catalog.resource("taxonomy")).read_table(
        columns=["domain", "theme", "subtheme"]
    )
    taxonomy = {
        (row["domain"], row["theme"], row["subtheme"])
        for row in taxonomy_rows.to_pylist()
    }
    synthetic_rows = DatasetLoader(catalog.resource("corpus")).read_table(
        columns=["broad_area", "theme", "subtheme"],
        filter_expression=ds.field("label_name") == "synthetic",
    )
    observed = {
        (row["broad_area"], row["theme"], row["subtheme"])
        for row in synthetic_rows.to_pylist()
    }

    assert len(taxonomy) == 665
    assert observed == taxonomy
