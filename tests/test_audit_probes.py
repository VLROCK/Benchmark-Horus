from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from horus_benchmark.audit.probes import (
    run_char4gram_probe, stratified_bootstrap_intervals, validation_threshold,
)
from horus_benchmark.data import DatasetLoader, ResourceSpec


def test_validation_threshold_and_bootstrap_are_deterministic() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = validation_threshold(labels, scores)
    first = stratified_bootstrap_intervals(
        labels, scores, threshold, replicates=20, confidence=0.95, seed=123
    )
    second = stratified_bootstrap_intervals(
        labels, scores, threshold, replicates=20, confidence=0.95, seed=123
    )

    assert 0.3 < threshold <= 0.7
    assert first == second
    assert first["roc_auc_ci_low"] == 1.0


def test_char4gram_probe_runs_on_deterministic_split_fixture(tmp_path) -> None:
    records = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            for index in range(2):
                records.append(
                    {"record_id": f"{split}-{label}-{index}", "label": label,
                     "split": split,
                     "content": ("ensaio humano com variação " if label == 0 else
                                 "texto sintético com estrutura ") + str(index)}
                )
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    loader = DatasetLoader(ResourceSpec(
        name="fixture", path=path, format="parquet", role="test_fixture"
    ))
    features = pd.DataFrame(records).drop(columns="content")

    scores, top_features = run_char4gram_probe(
        features, loader, max_train_per_label=2, max_validation_per_label=2,
        max_test_per_label=2, max_features=100, min_df=1,
        bootstrap_replicates=10, bootstrap_confidence=0.95, seed=123,
    )

    assert scores.iloc[0]["probe"] == "char_4gram_sampled"
    assert scores.iloc[0]["train_rows"] == 4
    assert not top_features.empty
