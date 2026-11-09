"""Small synthetic checks for OOD partitions; never use corpus records."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from horus_benchmark.protocols.ood_manifests import (
    _prepare_base, _sha256, _source_family, build, build_one, check,
)


def _sample() -> pd.DataFrame:
    rows = []
    for area in ("academic", "social"):
        for fold in range(5):
            for j in range(10):
                rid = f"{area}-h-{fold}-{j}"
                rows.append((rid, rid, "human", area, "s1" if j < 2 else "s2",
                             "human", 100, "development", fold))
            for j in range(6):
                rid = f"{area}-m-{fold}-{j}"
                # One non-target generator shares a known near-pair group
                # with the target generator. It must be embargoed in LOGO.
                group = f"{area}-m-0-pair" if area == "academic" and fold == 0 and j in (0, 3) else rid
                rows.append((rid, group, "synthetic", area, "pipeline",
                             "g1" if j < 3 else "g2", 100, "development", fold))
        for label, source, generator, suffix in (
                ("human", "s1", "human", "h1"),
                ("human", "s2", "human", "h2"),
                ("synthetic", "pipeline", "g1", "m1"),
                ("synthetic", "pipeline", "g2", "m2")):
            rid = f"{area}-test-{suffix}"
            rows.append((rid, rid, label, area, source, generator, 100, "test", -1))
    columns = ["record_id", "probe_group_id", "label_name", "broad_area",
               "source_name", "generator", "word_count_audit", "split", "fold"]
    raw = pd.DataFrame(rows, columns=columns)
    return _prepare_base(raw.drop(columns=["split", "fold"]),
                         raw[["record_id", "split", "fold"]], {}, .25, 9)


def _config() -> dict:
    return {"min_binary_test_per_class": 1, "min_other_human_development": 1}


def test_logo_withholds_all_generator_rows_and_near_group() -> None:
    base = _sample()
    result, stats = build_one(base, "logo", "g2", None, _config(), 9)
    assert stats["eligible"]
    assert stats["unmatched_target_rows"] == 0
    by_id = result.set_index("record_id")
    assert (result.loc[base.generator.eq("g2"), "role"] == "test_target").all()
    assert by_id.at["academic-m-0-0", "role"] == "embargo"
    assert result.loc[result.role.eq("development"), "probe_group_id"].isin(
        result.loc[result.role.isin({"test_target", "test_control"}), "probe_group_id"]
    ).sum() == 0
    assert stats["paired_iid_synthetic_rows"] >= 1
    assert stats["paired_iid_human_rows"] >= 1


def test_lodo_withholds_both_labels_in_area() -> None:
    base = _sample()
    result, stats = build_one(base, "lodo", "social", None, _config(), 9)
    assert stats["eligible"]
    assert set(result.loc[base.broad_area.eq("social"), "role"]) == {"test_target"}
    assert set(result.loc[result.role.eq("development"), "broad_area"]) == {"academic"}
    assert stats["binary_human_rows"] > 0
    assert stats["binary_synthetic_rows"] > 0


def test_loso_keeps_other_human_sources_and_reports_single_source_failure() -> None:
    base = _sample()
    result, stats = build_one(base, "loso", "s1", "academic", _config(), 9)
    assert stats["eligible"]
    assert stats["other_human_development"] > 0
    assert (result.loc[base.label_name.eq("human") & base.broad_area.eq("academic") &
                       base.source_family.eq("s1"), "role"] == "test_target").all()
    assert base.loc[result.role.eq("development") & base.broad_area.eq("academic") &
                    base.label_name.eq("human"), "source_family"].eq("s2").all()
    single = base.copy()
    single.loc[single.broad_area.eq("academic") & single.label_name.eq("human"),
               "source_family"] = "s1"
    _, unsupported = build_one(single, "loso", "s1", "academic", _config(), 9)
    assert unsupported["status"] == "unsupported"
    assert "other_human_source_development_below_minimum" in unsupported["reason"]


def test_source_aliases_are_explicit_only() -> None:
    names = pd.Series(["Folha de S.Paulo", "www1.folha.uol.com.br", "Other"])
    result = _source_family(names, {"Folha de S.Paulo": "folha.uol.com.br",
                                   "www1.folha.uol.com.br": "folha.uol.com.br"})
    assert result.tolist() == ["folha.uol.com.br", "folha.uol.com.br", "Other"]


def test_build_and_check_tiny_frozen_release(tmp_path) -> None:
    base = _sample()
    release_dir = tmp_path / "horus-n2-iid-v1"
    release_dir.mkdir()
    index = base[["record_id", "probe_group_id", "label_name", "broad_area",
                  "source_name", "generator", "word_count_audit"]]
    index_path = release_dir / "selected_index.parquet"
    index.to_parquet(index_path, index=False)
    seeds = [20261105, 20261106, 20261107]
    artifacts = {"selected_index": {"sha256": _sha256(index_path)}}
    for seed in seeds:
        name = f"split_seed{seed}"
        path = release_dir / f"{name}.parquet"
        base[["record_id", "probe_group_id", "iid_split", "iid_fold"]].rename(
            columns={"iid_split": "split", "iid_fold": "fold"}).to_parquet(path, index=False)
        artifacts[name] = {"sha256": _sha256(path)}
    (release_dir / "release_manifest.json").write_text(json.dumps({
        "verified": True, "kind": "frozen_n2_source_balanced_iid_release",
        "release_id": "horus-n2-iid-v1", "text_view": "n2",
        "development_folds": 5, "split_seeds": seeds, "rows": len(index),
        "artifacts": artifacts,
    }), encoding="utf-8")
    config_path = tmp_path / "ood.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 1, "release_id": "horus-n2-iid-v1",
        "release_dir": str(release_dir), "seeds": seeds,
        "length_bin_width": .25, "min_binary_test_per_class": 1,
        "min_other_human_development": 1, "source_aliases": {},
    }), encoding="utf-8")
    output = tmp_path / "ood"
    build(config_path, output)
    assert check(output) == output / "summary.json"
    tasks = pd.read_csv(output / "tasks.csv")
    assert set(tasks.protocol) == {"logo", "lodo", "loso"}
    assert (tasks.loc[tasks.status.eq("eligible"), "manifest_path"] != "").all()
    assert tasks.loc[tasks.status.eq("eligible")].shape[0] > 0
