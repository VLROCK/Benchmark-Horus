from __future__ import annotations

import json
from pathlib import Path

from horus_benchmark.runtime import create_run, seed_everything


def test_seed_everything_is_repeatable() -> None:
    import random

    seeded = seed_everything(2026)
    first = [random.random() for _ in range(3)]
    seed_everything(2026)
    second = [random.random() for _ in range(3)]

    assert first == second
    assert seeded == {"python_random": 2026, "numpy": 2026}


def test_create_run_persists_config_metadata_and_result(tmp_path: Path) -> None:
    context = create_run(
        run_root=tmp_path,
        experiment_id="structural validation",
        resolved_config={"seed": 2026, "dataset": "provisional"},
        seed=2026,
        dataset_version="provisional-v0",
    )
    result_path = context.write_result("summary", {"valid": True})
    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))

    assert context.config_path.is_file()
    assert metadata["seed"] == 2026
    assert metadata["dataset_version"] == "provisional-v0"
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"valid": True}

