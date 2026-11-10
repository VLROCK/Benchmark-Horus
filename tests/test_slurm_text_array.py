"""Exercise bounded Slurm array submission without contacting a scheduler."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Requires POSIX bash")
ROOT = Path(__file__).resolve().parents[1]


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _plan(workspace: Path, count: int) -> Path:
    prepared = workspace / "results/evaluation/n2-v1/prepared"
    prepared.mkdir(parents=True)
    (prepared / "prepared_manifest.json").write_text("{}\n", encoding="utf-8")
    path = workspace / "plan.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "model"])
        writer.writeheader()
        writer.writerows({"task_id": str(index), "model": "surface_length"}
                         for index in range(count))
    return path


def test_submit_uses_local_array_indices_and_global_offsets(tmp_path: Path) -> None:
    plan = _plan(tmp_path, 801)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    capture = tmp_path / "sbatch-args.txt"
    _executable(binary_dir / "sbatch",
                "printf '%s|' \"$@\" >> \"$HORUS_CAPTURE\"\n"
                "printf '\\n' >> \"$HORUS_CAPTURE\"\n"
                "echo 999\n")
    env = {**os.environ, "PATH": f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
           "HORUS_PYTHON_BIN": sys.executable, "HORUS_CAPTURE": str(capture),
           "HORUS_QOS": "normal"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_text_evaluation_plan.sh"),
         str(plan), "cpu", "4"], cwd=tmp_path, env=env,
        text=True, capture_output=True, check=True)
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line, last, offset in zip(lines, (399, 399, 0), (0, 400, 800)):
        assert f"--array=0-{last}%4|" in line
        assert f"HORUS_ARRAY_OFFSET={offset}|" in line
        assert "--qos=normal|" in line
    assert "800-800" in result.stdout


@pytest.mark.parametrize("script_name", ["slurm_text_evaluation_cpu.sh",
                                          "slurm_text_evaluation_bert.sh"])
def test_task_script_adds_offset_to_slurm_index(tmp_path: Path,
                                                 script_name: str) -> None:
    capture = tmp_path / "python-args.txt"
    fake_python = _executable(tmp_path / "python-stub",
                              "printf '%s|' \"$@\" > \"$HORUS_CAPTURE\"\n")
    env = {**os.environ, "SLURM_SUBMIT_DIR": str(tmp_path),
           "SLURM_ARRAY_TASK_ID": "2", "HORUS_ARRAY_OFFSET": "800",
           "HORUS_PLAN": "plan.csv", "HORUS_PYTHON_BIN": str(fake_python),
           "HORUS_CAPTURE": str(capture)}
    subprocess.run(["bash", str(ROOT / "scripts" / script_name)],
                   env=env, text=True, capture_output=True, check=True)
    assert "--task-index|802|" in capture.read_text(encoding="utf-8")
