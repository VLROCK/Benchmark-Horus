"""Release freeze checks for ID coverage, grouped folds and immutable inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from horus_benchmark.protocols import freeze_iid


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    selected_rows = []
    split_rows = []
    for label in ("human", "synthetic"):
        for provenance in ("a", "b"):
            for number in range(120):
                rid = f"{label}-{provenance}-{number}"
                selected_rows.append({
                    "record_id": rid, "label_name": label, "broad_area": "academic",
                    "source_name": provenance if label == "human" else "generated",
                    "generator": provenance if label == "synthetic" else "human",
                    "selection_group_id": rid, "probe_group_id": rid,
                    "length_stratum": 1, "stratum": f"academic|{label}|1",
                })
                split_rows.append({
                    "record_id": rid, "label_name": label, "broad_area": "academic",
                    "probe_group_id": rid, "length_stratum": 1,
                    "stratum": f"academic|{label}|1",
                    "split": "test" if number < 30 else "development",
                    "fold": -1 if number < 30 else (number - 30) // 30,
                })
    selected = pd.DataFrame(selected_rows)
    split = pd.DataFrame(split_rows)
    pairs = pd.DataFrame({"left_record_id": ["human-a-30"],
                          "right_record_id": ["human-a-31"]})
    expected = {
        "seed": 7, "rows": 480, "test_rows": 120,
        "development_fold_sizes": {"0": 120, "1": 120, "2": 120},
        "observed_pairs_both_selected": 1,
        "observed_pairs_cross_split": 0,
        "max_test_stratum_n100_pp": 0.0,
        "max_fold_stratum_n100_pp": 0.0,
        "max_test_material_shift_pp": 0.0,
        "max_fold_material_shift_pp": 0.0,
        "material_provenances_missing_test": 0,
        "material_provenances_missing_folds": 0,
    }
    return split, selected, pairs, expected


def test_freeze_split_validates_ids_groups_pairs_and_balance() -> None:
    split, selected, pairs, expected = _fixture()
    report = freeze_iid.validate_split_manifest(split, selected, pairs, expected, 3, .25, 1)
    assert report["rows"] == 480
    assert report["observed_pairs_cross_split"] == 0
    assert report["max_test_material_shift_pp"] == 0


def test_freeze_split_rejects_missing_ids_or_metadata_drift() -> None:
    split, selected, pairs, expected = _fixture()
    with pytest.raises(ValueError, match="Split IDs"):
        freeze_iid.validate_split_manifest(split.iloc[1:], selected, pairs, expected, 3, .25, 1)
    changed = split.copy()
    changed.loc[0, "label_name"] = "synthetic"
    with pytest.raises(ValueError, match="label_name disagrees"):
        freeze_iid.validate_split_manifest(changed, selected, pairs, expected, 3, .25, 1)


def test_freeze_split_rejects_observed_pair_crossing_folds() -> None:
    split, selected, pairs, expected = _fixture()
    changed = split.copy()
    left = changed.record_id.eq("human-a-31")
    right = changed.record_id.eq("human-a-60")
    changed.loc[left, "fold"] = 1
    changed.loc[right, "fold"] = 0
    with pytest.raises(ValueError, match="near-duplicate pairs cross splits"):
        freeze_iid.validate_split_manifest(changed, selected, pairs, expected, 3, .25, 1)


def test_freeze_split_rejects_selection_group_crossing_folds() -> None:
    split, selected, pairs, expected = _fixture()
    selected.loc[selected.record_id.eq("human-a-60"), "selection_group_id"] = "human-a-30"
    with pytest.raises(ValueError, match="selection_group_id crosses"):
        freeze_iid.validate_split_manifest(split, selected, pairs, expected, 3, .25, 1)


def test_freeze_manifest_rejects_tampered_index(tmp_path: Path, monkeypatch) -> None:
    source_split = tmp_path / "source.parquet"
    source_split.write_bytes(b"split fixture")
    payload = {
        "schema_version": 1, "kind": "frozen_n2_source_balanced_iid_release",
        "verified": True,
        "release_id": "horus-n2-iid-v1", "config_path": "fixture.yaml",
        "split_seeds": [7], "inputs": {"split_seed7": {
            "path": str(source_split), "sha256": freeze_iid.sha256_file(source_split)}},
    }
    index = pd.DataFrame({"record_id": ["a"], "label_name": ["human"]})
    monkeypatch.setattr(freeze_iid, "_build", lambda _path: (payload.copy(), index))
    output = tmp_path / "horus-n2-iid-v1"
    freeze_iid.freeze(Path("fixture.yaml"), output)
    assert freeze_iid.check(output) == output / freeze_iid.MANIFEST
    with pytest.raises(FileExistsError, match="already exists"):
        freeze_iid.freeze(Path("fixture.yaml"), output)
    original_index = (output / freeze_iid.INDEX).read_bytes()
    (output / freeze_iid.INDEX).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        freeze_iid.check(output)
    (output / freeze_iid.INDEX).write_bytes(original_index)
    (output / "split_seed7.parquet").write_bytes(b"different split")
    with pytest.raises(ValueError, match="Frozen split differs from source"):
        freeze_iid.check(output)
