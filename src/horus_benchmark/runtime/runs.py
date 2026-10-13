"""Creation of traceable run directories and persisted configurations."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError("experiment_id must contain at least one safe character")
    return cleaned.lower()


def _config_digest(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    directory: Path
    metadata_path: Path
    config_path: Path

    def write_result(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Persist one small JSON result inside this run."""

        safe_name = _slug(name)
        target = self.directory / f"{safe_name}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return target


def create_run(
    *,
    run_root: str | Path,
    experiment_id: str,
    resolved_config: Mapping[str, Any],
    seed: int,
    dataset_version: str,
    extra_metadata: Mapping[str, Any] | None = None,
) -> RunContext:
    """Create a unique run directory with immutable-at-creation metadata."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    now = datetime.now(timezone.utc)
    digest = _config_digest(resolved_config)
    run_id = f"{now:%Y%m%dT%H%M%S%fZ}_{_slug(experiment_id)}_{digest[:10]}"
    directory = Path(run_root).expanduser().resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)

    config_path = directory / "config.resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(dict(resolved_config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "seed": seed,
        "config_sha256": digest,
        "python": sys.version,
        "platform": platform.platform(),
    }
    if extra_metadata:
        metadata["extra"] = dict(extra_metadata)

    metadata_path = directory / "run.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return RunContext(
        run_id=run_id,
        directory=directory,
        metadata_path=metadata_path,
        config_path=config_path,
    )

