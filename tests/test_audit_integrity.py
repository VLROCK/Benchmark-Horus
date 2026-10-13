from __future__ import annotations

import pandas as pd

from horus_benchmark.audit.integrity import (
    _physical_partition_checks, duplicate_and_leakage_groups,
)
from horus_benchmark.audit.text_features import sha256_utf8
from horus_benchmark.data.catalog import DatasetCatalog, ResourceSpec


def test_duplicate_groups_identify_cross_split_text_and_source() -> None:
    frame = pd.DataFrame(
        [
            {"record_id": "a", "source_record_id": "doc1", "label_name": "human",
             "source_url": "https://example.org/a", "split": "train",
             "content_sha256_audit": "hash1", "comparison_sha256_audit": "norm1",
             "content_hash": "hash1", "normalized_hash": "norm1",
             "duplicate_group_id": "group1", "split_group_id": "group1"},
            {"record_id": "b", "source_record_id": "doc1", "label_name": "human",
             "source_url": "https://example.org/a", "split": "test",
             "content_sha256_audit": "hash1", "comparison_sha256_audit": "norm1",
             "content_hash": "hash1", "normalized_hash": "norm1",
             "duplicate_group_id": "group1", "split_group_id": "group1"},
        ]
    )

    summary, details = duplicate_and_leakage_groups(frame)
    exact = summary.loc[summary["key_type"] == "content_sha256_audit"].iloc[0]
    source = summary.loc[summary["key_type"] == "human_source_url"].iloc[0]

    assert exact["cross_split_groups"] == 1
    assert source["cross_split_groups"] == 1
    assert details["cross_split"].any()


def test_physical_splits_can_be_checked_without_materialized_benchmark(tmp_path) -> None:
    records = []
    resources = {}
    for split in ("train", "validation", "test"):
        content = f"Texto de {split}"
        record = {
            "record_id": split, "content_hash": sha256_utf8(content),
            "content_sha256_audit": sha256_utf8(content),
            "split": split, "content": content,
        }
        records.append(record)
        path = tmp_path / f"{split}.parquet"
        pd.DataFrame([record]).drop(columns="content_sha256_audit").to_parquet(path)
        resources[split] = ResourceSpec(
            name=split, path=path, format="parquet", role="derived_split",
            expected_rows=1,
        )

    catalog = DatasetCatalog(
        config_path=tmp_path / "catalog.yaml", workspace_root=tmp_path,
        dataset_id="small", dataset_version="v1", status="provisional",
        resources=resources,
    )
    checks = _physical_partition_checks(
        catalog, pd.DataFrame(records), ("train", "validation", "test"), None
    )

    assert checks
    assert all(check["count"] == 0 for check in checks)
    assert not any(check["check"].startswith("benchmark_") for check in checks)

