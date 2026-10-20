"""Reusable, non-mutating structural and split-leakage checks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.data.loader import compare_schema
from horus_benchmark.config import load_yaml
from horus_benchmark.audit.text_features import sha256_utf8


GROUP_FIELDS = (
    "record_id",
    "source_record_id",
    "content_sha256_audit",
    "comparison_sha256_audit",
    "content_hash",
    "normalized_hash",
    "duplicate_group_id",
    "split_group_id",
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def duplicate_and_leakage_groups(
    features: pd.DataFrame,
    extra_group_fields: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize repeated keys and retain their split/label relationships."""

    summaries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for field in (*GROUP_FIELDS, *extra_group_fields, "human_source_url"):
        if field == "human_source_url":
            view = features.loc[features["label_name"] == "human"]
            source = "source_url"
        else:
            view = features
            source = field
        nonempty = view.loc[view[source].notna() & (view[source].astype(str) != "")]
        counts = nonempty[source].value_counts(dropna=True)
        repeated_keys = counts.index[counts > 1]
        repeated = nonempty.loc[nonempty[source].isin(repeated_keys)].groupby(
            source, sort=False, dropna=True
        )
        duplicate_groups = 0
        duplicate_rows = 0
        cross_split_groups = 0
        cross_label_groups = 0
        for key, frame in repeated:
            if len(frame) < 2:
                continue
            duplicate_groups += 1
            duplicate_rows += len(frame)
            splits = sorted(frame["split"].dropna().astype(str).unique())
            labels = sorted(frame["label_name"].dropna().astype(str).unique())
            cross_split_groups += len(splits) > 1
            cross_label_groups += len(labels) > 1
            groups.append(
                {
                    "key_type": field,
                    "key_sha256_prefix": _fingerprint(str(key)),
                    "rows": len(frame),
                    "splits": "|".join(splits),
                    "labels": "|".join(labels),
                    "cross_split": len(splits) > 1,
                    "cross_label": len(labels) > 1,
                    "example_record_ids": "|".join(frame["record_id"].astype(str).head(3)),
                }
            )
        summaries.append(
            {
                "key_type": field,
                "duplicate_groups": duplicate_groups,
                "rows_in_duplicate_groups": duplicate_rows,
                "cross_split_groups": cross_split_groups,
                "cross_label_groups": cross_label_groups,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(groups)


def _physical_partition_checks(
    catalog: DatasetCatalog,
    corpus: pd.DataFrame,
    split_resources: Sequence[str],
    benchmark_resource: str | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    corpus_by_id = corpus.set_index("record_id")
    observed_ids: set[str] = set()
    for name in split_resources:
        table = DatasetLoader(catalog.resource(name)).read_table(
            columns=["record_id", "content_hash", "split", "content"]
        )
        partition = table.to_pandas()
        identifiers = set(partition["record_id"])
        checks.append(
            {
                "check": f"physical_split_{name}_duplicate_ids",
                "count": len(partition) - len(identifiers),
                "severity": "error",
            }
        )
        checks.append(
            {
                "check": f"physical_split_{name}_unexpected_split_value",
                "count": int((partition["split"] != name).sum()),
                "severity": "error",
            }
        )
        checks.append(
            {
                "check": f"physical_split_{name}_overlap_previous_splits",
                "count": len(identifiers & observed_ids),
                "severity": "error",
            }
        )
        observed_ids.update(identifiers)
        missing_from_corpus = identifiers - set(corpus_by_id.index)
        checks.append(
            {
                "check": f"physical_split_{name}_missing_from_corpus",
                "count": len(missing_from_corpus),
                "severity": "error",
            }
        )
        shared = partition[partition["record_id"].isin(corpus_by_id.index)]
        expected_hash = corpus_by_id.loc[shared["record_id"], "content_hash"].to_numpy()
        checks.append(
            {
                "check": f"physical_split_{name}_content_hash_mismatch",
                "count": int((shared["content_hash"].to_numpy() != expected_hash).sum()),
                "severity": "error",
            }
        )
        actual_hash = shared["content"].fillna("").map(sha256_utf8).to_numpy()
        expected_actual_hash = corpus_by_id.loc[
            shared["record_id"], "content_sha256_audit"
        ].to_numpy()
        checks.append(
            {
                "check": f"physical_split_{name}_actual_content_mismatch",
                "count": int((actual_hash != expected_actual_hash).sum()),
                "severity": "error",
            }
        )
    checks.append(
        {
            "check": "physical_splits_symmetric_difference_with_corpus",
            "count": len(observed_ids ^ set(corpus_by_id.index)),
            "severity": "error",
        }
    )

    if benchmark_resource is None:
        return checks

    benchmark = DatasetLoader(catalog.resource(benchmark_resource)).read_table(
        columns=["record_id", "content_hash", "content"]
    ).to_pandas()
    benchmark_ids = set(benchmark["record_id"])
    checks.append(
        {
            "check": "benchmark_duplicate_ids",
            "count": len(benchmark) - len(benchmark_ids),
            "severity": "error",
        }
    )
    checks.append(
        {
            "check": "benchmark_ids_missing_from_corpus",
            "count": len(benchmark_ids - set(corpus_by_id.index)),
            "severity": "error",
        }
    )
    shared = benchmark[benchmark["record_id"].isin(corpus_by_id.index)]
    expected_hash = corpus_by_id.loc[shared["record_id"], "content_hash"].to_numpy()
    checks.append(
        {
            "check": "benchmark_content_hash_mismatch",
            "count": int((shared["content_hash"].to_numpy() != expected_hash).sum()),
            "severity": "error",
        }
    )
    actual_hash = shared["content"].fillna("").map(sha256_utf8).to_numpy()
    expected_actual_hash = corpus_by_id.loc[
        shared["record_id"], "content_sha256_audit"
    ].to_numpy()
    checks.append(
        {
            "check": "benchmark_actual_content_mismatch",
            "count": int((actual_hash != expected_actual_hash).sum()),
            "severity": "error",
        }
    )
    return checks


def run_integrity_checks(
    catalog: DatasetCatalog,
    features: pd.DataFrame,
    *,
    corpus_resource: str,
    split_resources: Sequence[str],
    benchmark_resource: str | None,
    taxonomy_resource: str,
    lineage_fields: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full-corpus checks and return checks, summaries and group details."""

    checks: list[dict[str, Any]] = []

    def add(name: str, count: int, severity: str = "error") -> None:
        checks.append({"check": name, "count": int(count), "severity": severity})

    expected_rows = catalog.resource(corpus_resource).expected_rows
    if expected_rows is not None:
        add("corpus_row_count_mismatch", abs(len(features) - expected_rows))
    add("missing_record_id", features["record_id"].isna().sum() + (features["record_id"] == "").sum())
    add("duplicate_record_id", features["record_id"].duplicated().sum())
    add("missing_or_blank_content", features["content_blank"].sum())
    add("content_hash_mismatch", (features["content_hash"] != features["content_sha256_audit"]).sum())
    add("label_name_mismatch", (
        ((features["label"] == 0) & (features["label_name"] != "human"))
        | ((features["label"] == 1) & (features["label_name"] != "synthetic"))
        | ~features["label"].isin([0, 1])
    ).sum())
    if split_resources:
        add("invalid_split", (~features["split"].isin(split_resources)).sum())
    add("missing_domain", features["broad_area"].isna().sum() + (features["broad_area"] == "").sum())
    add("declared_char_count_mismatch", (features["char_count"] != features["char_count_audit"]).sum(), "warning")
    add("declared_word_count_mismatch", (features["word_count"] != features["word_count_audit"]).sum(), "warning")

    if catalog.schema_manifest_path is not None:
        declared = load_yaml(catalog.schema_manifest_path)["columns"]
        differences = compare_schema(DatasetLoader(catalog.resource(corpus_resource)).schema, declared)
        add("corpus_schema_differences", len(differences))

    taxonomy = DatasetLoader(catalog.resource(taxonomy_resource)).read_table(
        columns=["domain", "theme", "subtheme"]
    ).to_pandas()
    taxonomy_set = set(map(tuple, taxonomy[["domain", "theme", "subtheme"]].itertuples(index=False, name=None)))
    synthetic = features.loc[features["label_name"] == "synthetic"]
    synthetic_keys = list(
        synthetic[["broad_area", "theme", "subtheme"]].itertuples(index=False, name=None)
    )
    add("synthetic_taxonomy_rows_unmatched", sum(key not in taxonomy_set for key in synthetic_keys), "warning")

    group_summary, group_details = duplicate_and_leakage_groups(features, lineage_fields)
    for row in group_summary.itertuples(index=False):
        severity = "warning" if row.key_type in {"source_record_id", "human_source_url"} else "error"
        add(f"cross_split_{row.key_type}_groups", row.cross_split_groups, severity)
    if split_resources:
        checks.extend(_physical_partition_checks(catalog, features, split_resources, benchmark_resource))
    checks_frame = pd.DataFrame(checks)
    checks_frame["passed"] = checks_frame["count"] == 0
    return checks_frame, group_summary, group_details
