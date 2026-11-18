"""Synthetic protocol checks; never consume corpus text or benchmark results."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from horus_benchmark.protocols.compound_ood import build, build_one, check
from horus_benchmark.protocols.ood_manifests import (
    _prepare_base, _sha256, build as build_parent,
)


SEEDS = [20261105, 20261106, 20261107]


def _sample() -> pd.DataFrame:
    rows = []
    for area in ("academic", "social"):
        for fold in range(5):
            # LOGO matches controls in length cells. One human per fold lies
            # outside the synthetic target cell and stays in development.
            for j in range(4):
                rid = f"{area}-h-{fold}-{j}"
                rows.append((rid, rid, "human", area, "source-a" if j < 2
                             else "source-b", "human", 200 if j == 0 else 100,
                             "development", fold))
            for generator in ("g1", "g2"):
                for j in range(3):
                    rid = f"{area}-{generator}-{fold}-{j}"
                    group = ("cross-generator-area-pair" if fold == 0 and j == 0 and
                             (area, generator) in {("academic", "g1"),
                                                   ("social", "g2")} else rid)
                    rows.append((rid, group, "synthetic", area, "pipeline",
                                 generator, 100, "development", fold))
        for label, source, generator, suffix in (
                ("human", "source-a", "human", "h1"),
                ("human", "source-b", "human", "h2"),
                ("synthetic", "pipeline", "g1", "m1"),
                ("synthetic", "pipeline", "g2", "m2")):
            rid = f"{area}-test-{suffix}"
            rows.append((rid, rid, label, area, source, generator, 100,
                         "test", -1))
    return pd.DataFrame(rows, columns=[
        "record_id", "probe_group_id", "label_name", "broad_area",
        "source_name", "generator", "word_count_audit", "split", "fold",
    ])


def _base_and_logo() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _sample()
    base = _prepare_base(raw.drop(columns=["split", "fold"]),
                         raw[["record_id", "split", "fold"]], {}, .25, SEEDS[0])
    from horus_benchmark.protocols.ood_manifests import build_one as build_parent_one
    parent, stats = build_parent_one(
        base, "logo", "g1", None,
        {"min_binary_test_per_class": 1, "min_other_human_development": 1},
        SEEDS[0])
    assert stats["eligible"]
    return base, parent[["record_id", "role"]]


def test_joint_ood_uses_exact_area_slice_of_logo_and_withholds_both_factors() -> None:
    base, logo = _base_and_logo()
    result, stats = build_one(base, logo, "g1", "academic",
                              {"min_binary_test_per_class": 1,
                               "development_folds": 5}, SEEDS[0])
    assert stats["eligible"]
    assert len(result) == len(base)
    assert set(result.record_id) == set(base.record_id)
    assert (set(result.loc[result.role.isin(
        {"test_target", "test_unmatched_target", "test_control"}), "record_id"]) ==
            set(base.loc[logo.role.isin(
                {"test_target", "test_unmatched_target", "test_control"}) &
                base.broad_area.eq("academic"), "record_id"]))
    assert set(result.loc[result.role.eq("development"), "broad_area"]) == {"social"}
    assert set(base.loc[result.role.eq("development"), "generator"]) <= {"human", "g2"}
    assert result.set_index("record_id").at["social-g2-0-0", "role"] == "embargo"
    assert result.loc[result.role.eq("development")].groupby("fold").label_name.nunique().eq(2).all()
    assert result.loc[result.role.eq("development"), "paired_iid_test"].eq(False).all()
    assert stats["paired_iid_synthetic_rows"] > 0
    assert stats["paired_iid_human_rows"] > 0


def test_joint_ood_reports_unsupported_cell_without_changing_roles() -> None:
    base, logo = _base_and_logo()
    result, stats = build_one(base, logo, "g1", "academic",
                              {"min_binary_test_per_class": 10_000,
                               "development_folds": 5}, SEEDS[0])
    assert stats["status"] == "unsupported"
    assert "binary_test_below_minimum" in stats["reason"]
    assert len(result) == len(base)
    assert result.role.eq("test_target").any()


def test_joint_ood_rejects_parent_logo_that_drops_a_target() -> None:
    base, logo = _base_and_logo()
    tampered = logo.copy()
    victim = base.index[base.label_name.eq("synthetic") &
                        base.generator.eq("g1") &
                        base.broad_area.eq("academic")][0]
    tampered.loc[victim, "role"] = "development"
    try:
        build_one(base, tampered, "g1", "academic",
                  {"min_binary_test_per_class": 1,
                   "development_folds": 5}, SEEDS[0])
    except ValueError as error:
        assert "every generator-area target" in str(error)
    else:
        raise AssertionError("A parent LOGO target was silently returned to development")


def test_compound_build_and_check_tiny_frozen_release(tmp_path) -> None:
    raw = _sample()
    release_dir = tmp_path / "horus-n2-iid-v1"
    release_dir.mkdir()
    index = raw.drop(columns=["split", "fold"])
    index_path = release_dir / "selected_index.parquet"
    index.to_parquet(index_path, index=False)
    artifacts = {"selected_index": {"sha256": _sha256(index_path)}}
    for seed in SEEDS:
        name = f"split_seed{seed}"
        path = release_dir / f"{name}.parquet"
        raw[["record_id", "probe_group_id", "split", "fold"]].to_parquet(
            path, index=False)
        artifacts[name] = {"sha256": _sha256(path)}
    (release_dir / "release_manifest.json").write_text(json.dumps({
        "verified": True, "kind": "frozen_n2_source_balanced_iid_release",
        "release_id": "horus-n2-iid-v1", "text_view": "n2",
        "development_folds": 5, "split_seeds": SEEDS, "rows": len(index),
        "artifacts": artifacts,
    }), encoding="utf-8")
    parent_config = tmp_path / "parent.yaml"
    parent_config.write_text(yaml.safe_dump({
        "schema_version": 1, "release_id": "horus-n2-iid-v1",
        "release_dir": str(release_dir), "seeds": SEEDS,
        "length_bin_width": .25, "min_binary_test_per_class": 1,
        "min_other_human_development": 1, "source_aliases": {},
    }), encoding="utf-8")
    parent_dir = tmp_path / "parent"
    build_parent(parent_config, parent_dir)
    config_path = tmp_path / "joint.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "experiment_id": "horus-n2-compound-generator-domain-v1",
        "release_id": "horus-n2-iid-v1", "release_dir": str(release_dir),
        "parent_ood_dir": str(parent_dir), "seeds": SEEDS,
        "development_folds": 5, "length_bin_width": .25,
        "min_binary_test_per_class": 1,
    }), encoding="utf-8")
    output = tmp_path / "compound"
    build(config_path, output)
    assert check(output) == output / "summary.json"
    tasks = pd.read_csv(output / "tasks.csv", keep_default_na=False)
    assert len(tasks) == 3 * 2 * 2
    assert set(tasks.protocol) == {"logo_lodo"}
    assert tasks.status.eq("eligible").all()
    assert (tasks.manifest_path != "").all()
    assert (output / "test_provenance.csv").is_file()
