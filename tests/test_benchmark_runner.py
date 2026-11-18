"""Protocol wiring checks with synthetic IDs only, no real dataset run."""

import json

import pandas as pd
import pytest

from horus_benchmark.evaluation import benchmark_runner as runner
from horus_benchmark.evaluation.benchmark_runner import _task_id, partition_task


def _iid_rows() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for label in ("human", "synthetic"):
            rid = f"{label}-{fold}"
            rows.append((rid, rid, label, "development", fold))
    for label in ("human", "synthetic"):
        rid = f"{label}-test"
        rows.append((rid, rid, label, "test", -1))
    return pd.DataFrame(rows, columns=["record_id", "probe_group_id",
                                       "label_name", "split", "fold"])


def test_iid_round_rotates_one_fold_without_touching_test() -> None:
    frame = partition_task(_iid_rows(), 2, "iid")
    assert frame.role.value_counts().to_dict() == {
        "train": 8, "validation": 2, "test_target": 2}
    assert frame.loc[frame.role.eq("test_target"), "paired_iid_test"].all()
    assert set(frame.loc[frame.role.eq("validation"), "fold"]) == {2}


def test_ood_manifest_keeps_target_and_embargo_outside_fit() -> None:
    frame = _iid_rows().rename(columns={"split": "role"})
    frame.loc[frame.role.eq("development"), "paired_iid_test"] = False
    frame.loc[frame.role.eq("test"), "role"] = "test_target"
    frame.loc[frame.role.eq("test_target"), "paired_iid_test"] = True
    frame.loc[frame.record_id.eq("human-0"), "role"] = "embargo"
    round_ = partition_task(frame, 1, "logo")
    assert round_.loc[round_.record_id.eq("human-0"), "role"].item() == "embargo"
    assert set(round_.loc[round_.role.eq("validation"), "fold"]) == {1}
    assert round_.loc[round_.role.eq("test_target"), "paired_iid_test"].all()


def test_compound_ood_uses_existing_development_folds() -> None:
    frame = _iid_rows().rename(columns={"split": "role"})
    frame["paired_iid_test"] = False
    frame.loc[frame.role.eq("test"), "role"] = "test_target"
    frame.loc[frame.role.eq("test_target"), "paired_iid_test"] = True
    frame.loc[frame.record_id.eq("synthetic-0"), "role"] = "embargo"
    round_ = partition_task(frame, 1, "logo_lodo")
    assert round_.loc[round_.record_id.eq("synthetic-0"), "role"].item() == "embargo"
    assert set(round_.loc[round_.role.eq("validation"), "fold"]) == {1}
    assert not set(round_.loc[round_.role.eq("train"), "record_id"]) & set(
        round_.loc[round_.role.eq("test_target"), "record_id"])


def test_crossing_probe_group_is_rejected() -> None:
    frame = _iid_rows()
    frame.loc[frame.record_id.eq("human-1"), "probe_group_id"] = "human-0"
    with pytest.raises(ValueError, match="crosses"):
        partition_task(frame, 1, "iid")


def test_task_identity_covers_manifest_and_counterfactual() -> None:
    row = {"protocol": "logo", "target": "g1", "seed": 20261105,
           "fold": 0, "model": "word_lr", "manifest_path": "a.parquet",
           "manifest_sha256": "a" * 64, "counterfactual_n3s": False}
    first = _task_id(row)
    assert first == _task_id({**row, "seed": "20261105"})
    assert first != _task_id({**row, "manifest_sha256": "b" * 64})
    assert first != _task_id({**row, "counterfactual_n3s": True})


def test_one_synthetic_iid_task_writes_aligned_scores(tmp_path, monkeypatch) -> None:
    """Exercise the orchestration boundary without accessing Hórus text."""
    split = pd.concat([_iid_rows(), _iid_rows().assign(
        record_id=lambda frame: frame.record_id + "-copy",
        probe_group_id=lambda frame: frame.probe_group_id + "-copy")],
        ignore_index=True)
    split_path = tmp_path / "split.parquet"
    split.to_parquet(split_path, index=False)
    selected = split[["record_id", "label_name", "probe_group_id"]].copy()
    selected["broad_area"] = "synthetic_area"
    selected["source_name"] = "synthetic_source"
    selected["generator"] = "synthetic_generator"
    selected["word_count_audit"] = selected.record_id.str.len()
    selected["content"] = selected.record_id.map(
        lambda value: "texto exemplo " * (3 if value.startswith("synthetic") else 2) + value)
    text_path = tmp_path / "selected.parquet"
    selected.to_parquet(text_path, index=False)
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "release_manifest.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    cfg = {
        "text_model_config": "configs/detectors/text_baselines_v1.yaml",
        "bertimbau_config": "configs/detectors/bertimbau_n2_v1.yaml",
        "fpr_targets": [0.01, 0.05], "bootstrap_replicates": 4,
    }
    monkeypatch.setattr(runner, "_config", lambda _: cfg)
    monkeypatch.setattr(runner, "_release", lambda _: (release_dir, {}))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = {
        "kind": "selected_n2_text_cache",
        "config_sha256": runner.sha256_file(config_path),
        "release_manifest_sha256": runner.sha256_file(release_dir / "release_manifest.json"),
        "selected_n2_texts": {"path": str(text_path),
                              "sha256": runner.sha256_file(text_path)},
    }
    (prepared_dir / "prepared_manifest.json").write_text(
        json.dumps(prepared), encoding="utf-8")
    task = {"protocol": "iid", "target": "all", "seed": 20261105, "fold": 0,
            "model": "surface_length", "manifest_path": str(split_path),
            "manifest_sha256": runner.sha256_file(split_path),
            "counterfactual_n3s": False}
    task["task_id"] = _task_id(task)
    plan_path = tmp_path / "plan.csv"
    pd.DataFrame([task]).to_csv(plan_path, index=False)
    output = runner.run_task(config_path, prepared_dir, plan_path, 0, tmp_path / "runs")
    assert output.is_file()
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["counts"] == {"train": 16, "validation": 4,
                                  "test_scored": 4, "binary_test": 4,
                                  "paired_binary_test": 4, "unmatched_target": 0}
    predictions = pd.read_parquet(output.parent / "predictions.parquet")
    assert set(predictions.label_name) == {"human", "synthetic"}
    assert predictions.score_n2.notna().all()
