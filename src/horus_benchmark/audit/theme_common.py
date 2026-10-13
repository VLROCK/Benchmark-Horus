"""Taxonomy, configuration and provenance shared by the theme audit stages."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horus_benchmark.config import ConfigError, load_yaml, resolve_from
from horus_benchmark.data import DatasetCatalog
from horus_benchmark.runtime import create_run


def normalized_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ThemeContext:
    config_path: Path
    config: dict[str, Any]
    catalog: DatasetCatalog
    taxonomy: dict[str, tuple[str, ...]]
    aliases: dict[tuple[str, str], str]
    taxonomy_path: Path
    mapping_path: Path
    output_root: Path

    def canonical_existing(self, area: str, raw_theme: str | None) -> tuple[str | None, str]:
        if not raw_theme or not raw_theme.strip():
            return None, "unmapped"
        key = normalized_key(raw_theme)
        canonical = {
            normalized_key(theme): theme for theme in self.taxonomy[area]
        }.get(key)
        if canonical is not None:
            return canonical, "exact_taxonomy"
        alias = self.aliases.get((area, key))
        return (alias, "manual_mapping") if alias else (None, "unmapped")

    def new_run(self, stage: str, *, extra: dict[str, Any] | None = None):
        provenance = {
            "config_path": str(self.config_path),
            "catalog": str(self.catalog.config_path),
            "taxonomy_sha256": sha256_file(self.taxonomy_path),
            "manual_mapping_sha256": sha256_file(self.mapping_path),
            **(extra or {}),
        }
        return create_run(
            run_root=self.output_root,
            experiment_id=f"{self.config['experiment_id']}-{stage}",
            resolved_config=self.config,
            seed=int(self.config["seed"]),
            dataset_version=self.catalog.dataset_version,
            extra_metadata=provenance,
        )


def load_context(path: str | Path) -> ThemeContext:
    config_path = Path(path).expanduser().resolve()
    config = load_yaml(config_path)
    required = {
        "experiment_id", "workspace_root", "dataset_config", "corpus_resource",
        "taxonomy_resource", "manual_mapping", "output_root", "seed",
        "sampling", "model", "threshold",
    }
    missing = required - config.keys()
    if missing:
        raise ConfigError(f"Theme config is missing: {sorted(missing)}")
    root = resolve_from(config_path.parent, config["workspace_root"])
    catalog = DatasetCatalog.from_yaml(resolve_from(root, config["dataset_config"]))
    taxonomy_path = catalog.resource(config["taxonomy_resource"]).path
    mapping_path = resolve_from(root, config["manual_mapping"])
    taxonomy: dict[str, set[str]] = {}
    with taxonomy_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            area, theme = row["domain"].strip(), row["theme"].strip()
            if not area or not theme:
                raise ConfigError("Taxonomy has an empty domain or theme")
            taxonomy.setdefault(area, set()).add(theme)
    if len(taxonomy) != 6:
        raise ConfigError(f"Expected six broad areas, found {len(taxonomy)}")
    ordered = {area: tuple(sorted(themes)) for area, themes in sorted(taxonomy.items())}
    aliases: dict[tuple[str, str], str] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            area = row["broad_area"].strip()
            raw = row["raw_theme"].strip()
            canonical = row["canonical_theme"].strip()
            if area not in ordered or canonical not in ordered[area] or not raw:
                raise ConfigError(f"Invalid manual theme mapping: {row}")
            key = (area, normalized_key(raw))
            if key in aliases and aliases[key] != canonical:
                raise ConfigError(f"Conflicting manual theme mapping: {row}")
            canonical_names = {normalized_key(theme): theme for theme in ordered[area]}
            if key[1] in canonical_names and canonical_names[key[1]] != canonical:
                raise ConfigError(f"Mapping conflicts with exact taxonomy theme: {row}")
            aliases[key] = canonical
    return ThemeContext(
        config_path=config_path,
        config=config,
        catalog=catalog,
        taxonomy=ordered,
        aliases=aliases,
        taxonomy_path=taxonomy_path,
        mapping_path=mapping_path,
        output_root=resolve_from(root, config["output_root"]),
    )


def read_run_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.parent / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata beside {path}: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_artifact_context(context: ThemeContext, artifact: Path) -> None:
    meta = read_run_metadata(artifact)
    if meta["dataset_version"] != context.catalog.dataset_version:
        raise ValueError(f"Artifact dataset version differs: {artifact}")
    extra = meta.get("extra", {})
    if extra.get("taxonomy_sha256") != sha256_file(context.taxonomy_path):
        raise ValueError(f"Artifact taxonomy differs: {artifact}")
    if extra.get("manual_mapping_sha256") != sha256_file(context.mapping_path):
        raise ValueError(f"Artifact manual mapping differs: {artifact}")
