"""Small, explicit helpers for declarative YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing required structure."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without applying implicit project defaults."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return payload


def resolve_from(base: str | Path, value: str | Path) -> Path:
    """Resolve ``value`` against ``base`` while preserving absolute paths."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(base).expanduser().resolve() / candidate).resolve()

