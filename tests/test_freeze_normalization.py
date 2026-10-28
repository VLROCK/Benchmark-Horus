from __future__ import annotations

import json
from pathlib import Path

import pytest

from horus_benchmark.data import freeze_normalization


def test_freeze_records_artifacts_and_rejects_tampering(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "n2"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "dataset_id": "fixture", "dataset_version": "v1", "stage": "n2",
        "source_sha256": "source", "config_sha256": "config",
        "code_sha256": {"normalization.py": "code"}, "output_rows": 2,
    }), encoding="utf-8")
    for name in freeze_normalization.ARTIFACTS:
        if name != "summary.json":
            (run_dir / name).write_text(name, encoding="utf-8")

    def fake_verify(_config_path: Path, _run_dir: Path) -> Path:
        target = run_dir / "verification.json"
        target.write_text(json.dumps({
            "verified": True, "stage": "n2", "rows": 2,
            "source_sha256": "source",
        }), encoding="utf-8")
        return target

    monkeypatch.setattr(freeze_normalization, "verify", fake_verify)
    target = freeze_normalization.freeze(tmp_path / "config.yaml", run_dir)
    assert target.is_file()
    assert freeze_normalization.check(run_dir) == target
    with pytest.raises(FileExistsError):
        freeze_normalization.freeze(tmp_path / "config.yaml", run_dir)

    (run_dir / "normalized_corpus.parquet").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        freeze_normalization.check(run_dir)
