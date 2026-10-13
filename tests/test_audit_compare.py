from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from horus_benchmark.audit.compare import compare_runs


def test_comparison_rejects_changed_label_before_writing_output(tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    pd.DataFrame(
        {"record_id": ["a"], "label": [0], "label_name": ["human"],
         "split": ["train"], "broad_area": ["academic"]}
    ).to_parquet(before / "features.parquet")
    pd.DataFrame(
        {"record_id": ["a"], "label": [1], "label_name": ["synthetic"],
         "split": ["train"], "broad_area": ["academic"]}
    ).to_parquet(after / "features.parquet")

    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="protected metadata: label"):
        compare_runs(before, after, output)
    assert not output.exists()

