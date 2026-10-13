from __future__ import annotations

from pathlib import Path

import pytest

from horus_benchmark.config import ConfigError
from horus_benchmark.data import DatasetCatalog


def test_catalog_resolves_paths_from_workspace_root(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs" / "datasets"
    config_dir.mkdir(parents=True)
    data_file = tmp_path / "sample.csv"
    data_file.write_text("id,text\n1,example\n", encoding="utf-8")
    catalog_file = config_dir / "catalog.yaml"
    catalog_file.write_text(
        """
dataset_id: example
dataset_version: v1
workspace_root: ../..
resources:
  sample:
    path: sample.csv
    format: csv
    role: test_fixture
    expected_rows: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    catalog = DatasetCatalog.from_yaml(catalog_file)

    assert catalog.dataset_id == "example"
    assert catalog.resource("sample").path == data_file.resolve()
    assert catalog.missing_files() == []


def test_catalog_rejects_unknown_format(tmp_path: Path) -> None:
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(
        """
dataset_id: example
dataset_version: v1
resources:
  sample:
    path: sample.txt
    format: txt
    role: test_fixture
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unsupported format"):
        DatasetCatalog.from_yaml(catalog_file)

