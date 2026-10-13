"""Declarative dataset catalog with paths relative to the workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from horus_benchmark.config import ConfigError, load_yaml, resolve_from


SUPPORTED_FORMATS = frozenset({"csv", "parquet"})


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """Resolved description of one physical dataset resource."""

    name: str
    path: Path
    format: str
    role: str
    expected_rows: int | None = None
    immutable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetCatalog:
    """A versioned collection of data resources."""

    config_path: Path
    workspace_root: Path
    dataset_id: str
    dataset_version: str
    status: str
    resources: Mapping[str, ResourceSpec]
    manifest_path: Path | None = None
    schema_manifest_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetCatalog":
        config_path = Path(path).expanduser().resolve()
        payload = load_yaml(config_path)

        for key in ("dataset_id", "dataset_version", "resources"):
            if key not in payload:
                raise ConfigError(f"Missing required catalog key: {key}")

        resource_payload = payload["resources"]
        if not isinstance(resource_payload, dict) or not resource_payload:
            raise ConfigError("Catalog 'resources' must be a non-empty mapping")

        workspace_root = resolve_from(
            config_path.parent, payload.get("workspace_root", ".")
        )
        resources: dict[str, ResourceSpec] = {}
        for name, raw_spec in resource_payload.items():
            if not isinstance(raw_spec, dict):
                raise ConfigError(f"Resource '{name}' must be a mapping")
            missing = {"path", "format", "role"} - raw_spec.keys()
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ConfigError(f"Resource '{name}' is missing: {missing_text}")

            resource_format = str(raw_spec["format"]).lower()
            if resource_format not in SUPPORTED_FORMATS:
                raise ConfigError(
                    f"Unsupported format '{resource_format}' for resource '{name}'"
                )

            expected_rows = raw_spec.get("expected_rows")
            if expected_rows is not None and (
                not isinstance(expected_rows, int) or expected_rows < 0
            ):
                raise ConfigError(
                    f"Resource '{name}' expected_rows must be a non-negative integer"
                )

            standard_keys = {
                "path",
                "format",
                "role",
                "expected_rows",
                "immutable",
            }
            resources[str(name)] = ResourceSpec(
                name=str(name),
                path=resolve_from(workspace_root, raw_spec["path"]),
                format=resource_format,
                role=str(raw_spec["role"]),
                expected_rows=expected_rows,
                immutable=bool(raw_spec.get("immutable", True)),
                metadata={
                    key: value
                    for key, value in raw_spec.items()
                    if key not in standard_keys
                },
            )

        manifest = payload.get("manifest")
        schema_manifest = payload.get("schema_manifest")
        return cls(
            config_path=config_path,
            workspace_root=workspace_root,
            dataset_id=str(payload["dataset_id"]),
            dataset_version=str(payload["dataset_version"]),
            status=str(payload.get("status", "unspecified")),
            resources=resources,
            manifest_path=(resolve_from(workspace_root, manifest) if manifest else None),
            schema_manifest_path=(
                resolve_from(workspace_root, schema_manifest)
                if schema_manifest
                else None
            ),
        )

    def resource(self, name: str) -> ResourceSpec:
        """Return one named resource with a useful error message."""

        try:
            return self.resources[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self.resources))
            raise KeyError(f"Unknown resource '{name}'. Available: {choices}") from exc

    def missing_files(self) -> list[Path]:
        """List registered resources that are absent from disk."""

        return [spec.path for spec in self.resources.values() if not spec.path.is_file()]

