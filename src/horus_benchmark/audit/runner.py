"""End-to-end audit of one immutable corpus resource."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from horus_benchmark.audit.coverage import (
    coverage_tables, create_figures, feature_summaries, near_pair_review_tables,
)
from horus_benchmark.audit.integrity import run_integrity_checks
from horus_benchmark.audit.near_duplicates import (
    NearDuplicateConfig, bottom_k_signature, candidate_pairs,
    exact_shingle_similarity, verified_pairs,
)
from horus_benchmark.audit.probes import run_char4gram_probe, run_surface_probes
from horus_benchmark.audit.text_features import extract_text_features
from horus_benchmark.config import load_yaml, resolve_from
from horus_benchmark.data import DatasetCatalog, DatasetLoader
from horus_benchmark.runtime import configure_logging, create_run, seed_everything


READ_COLUMNS = (
    "record_id", "source_record_id", "label", "label_name", "broad_area",
    "theme", "subtheme", "source_name", "source_url", "generator", "genre",
    "size_category", "char_count", "word_count", "split", "content_hash",
    "normalized_hash", "duplicate_group_id", "split_group_id",
    "benchmark_eligible", "creation_date", "license", "prompt_version", "content",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_features(
    loader: DatasetLoader,
    *,
    batch_size: int,
    near_config: NearDuplicateConfig | None,
    lineage_fields: tuple[str, ...],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[tuple[int, ...]]]:
    rows: list[dict[str, Any]] = []
    signatures: list[tuple[int, ...]] = []
    available = set(loader.schema.names)
    missing_lineage = set(lineage_fields) - available
    if missing_lineage:
        raise ValueError(f"Configured lineage fields absent from corpus: {sorted(missing_lineage)}")
    for batch in loader.iter_batches(
        columns=[*READ_COLUMNS, *lineage_fields], batch_size=batch_size
    ):
        for record in batch.to_pylist():
            content = record.pop("content")
            text = content if isinstance(content, str) else ""
            measured = extract_text_features(text)
            rows.append({**record, **measured, "content_blank": not bool(text.strip())})
            if near_config is not None:
                signatures.append(bottom_k_signature(text, near_config))
        if len(rows) % 20_000 < batch_size:
            logger.info("Measured %s records", len(rows))
    return pd.DataFrame.from_records(rows), signatures


def _read_signatures_only(
    loader: DatasetLoader,
    expected_ids: list[str],
    config: NearDuplicateConfig,
    batch_size: int,
    logger: logging.Logger,
) -> list[tuple[int, ...]]:
    signatures: list[tuple[int, ...]] = []
    for batch in loader.iter_batches(columns=["record_id", "content"], batch_size=batch_size):
        for record_id, content in zip(
            batch.column("record_id").to_pylist(), batch.column("content").to_pylist()
        ):
            if record_id != expected_ids[len(signatures)]:
                raise ValueError("Corpus row order changed since feature extraction")
            signatures.append(bottom_k_signature(content or "", config))
        if len(signatures) % 20_000 < batch_size:
            logger.info("Indexed %s records for near duplicates", len(signatures))
    if len(signatures) != len(expected_ids):
        raise ValueError("Reused feature rows differ from current corpus row count")
    return signatures


def _candidate_texts(
    loader: DatasetLoader,
    candidates: list[tuple[int, int, int]],
    batch_size: int,
    expected_ids: list[str],
) -> dict[int, str]:
    needed = {index for left, right, _ in candidates for index in (left, right)}
    selected: dict[int, str] = {}
    row_index = 0
    for batch in loader.iter_batches(columns=["record_id", "content"], batch_size=batch_size):
        for record_id, content in zip(
            batch.column("record_id").to_pylist(), batch.column("content").to_pylist()
        ):
            if row_index in needed:
                if record_id != expected_ids[row_index]:
                    raise ValueError("Corpus row order changed between audit passes")
                selected[row_index] = content if isinstance(content, str) else ""
            row_index += 1
    if len(selected) != len(needed):
        raise ValueError("Candidate indices did not match the corpus row count")
    return selected


def _near_duplicate_table(
    loader: DatasetLoader,
    features: pd.DataFrame,
    signatures: list[tuple[int, ...]],
    config: NearDuplicateConfig,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates, metadata = candidate_pairs(signatures, config)
    if not candidates:
        return pd.DataFrame(), {**metadata, "verified_pairs": 0, "cross_split_pairs": 0}
    selected = _candidate_texts(
        loader, candidates, batch_size, features["record_id"].tolist()
    )
    accepted = verified_pairs(candidates, selected, config)
    del selected
    output: list[dict[str, Any]] = []
    for pair in accepted:
        left = features.iloc[int(pair["left_index"])]
        right = features.iloc[int(pair["right_index"])]
        output.append(
            {"left_record_id": left["record_id"], "right_record_id": right["record_id"],
             "left_split": left["split"], "right_split": right["split"],
             "left_label": left["label_name"], "right_label": right["label_name"],
             "left_domain": left["broad_area"], "right_domain": right["broad_area"],
             "cross_split": left["split"] != right["split"],
             "cross_label": left["label_name"] != right["label_name"],
             "shared_bottom_k": pair["shared_bottom_k"],
             "jaccard_5gram": pair["jaccard"],
             "containment_5gram": pair["containment"],
             "jaccard_match": pair["jaccard_match"],
             "containment_match": pair["containment_match"]}
        )
    frame = pd.DataFrame(output)
    metadata.update(
        {
            "verified_pairs": len(frame),
            "cross_split_pairs": int(frame["cross_split"].sum()) if not frame.empty else 0,
            "cross_label_pairs": int(frame["cross_label"].sum()) if not frame.empty else 0,
        }
    )
    return frame, metadata


def _save_csv_tables(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")


def _report(
    directory: Path,
    catalog: DatasetCatalog,
    features: pd.DataFrame,
    checks: pd.DataFrame,
    groups: pd.DataFrame,
    near_summary: dict[str, Any] | None,
    probes: pd.DataFrame | None,
    benchmark_resource: str | None,
) -> None:
    errors = checks.loc[(checks["severity"] == "error") & (~checks["passed"])]
    warnings = checks.loc[(checks["severity"] == "warning") & (~checks["passed"])]
    lines = [
        f"# Auditoria exploratória — {catalog.dataset_version}", "",
        f"Gerada em {datetime.now(timezone.utc).isoformat()} (UTC).",
        f"Corpus analisado: **{len(features):,}** textos. Esta versão é **{catalog.status}**.", "",
        "## Integridade e leakage", "",
        f"- Checagens estruturais com contagem positiva: {len(errors)} erros e {len(warnings)} avisos.",
        "- Veja `tables/integrity_checks.csv` para todas as contagens e `tables/repeated_groups.csv` para IDs de revisão.",
        "- Colisões de `content_sha256_audit` são igualdade exata de bytes UTF-8; `comparison_sha256_audit` usa NFKC, casefold e espaços colapsados apenas para comparar.",
        "- Grupos de URL humana ou `source_record_id` em mais de um split são indícios de possível parentesco documental; precisam de revisão da proveniência.",
    ]
    if benchmark_resource is None:
        lines.append(
            "- Este catálogo não inclui um benchmark materializado; as checagens "
            "específicas do benchmark não foram executadas."
        )
    if not groups.empty:
        for field in ("content_sha256_audit", "comparison_sha256_audit", "split_group_id", "human_source_url"):
            match = groups.loc[groups["key_type"] == field]
            if not match.empty:
                lines.append(f"- {field}: {int(match['cross_split_groups'].iloc[0])} grupos atravessam splits.")
    if near_summary is not None:
        lines += [
            f"- Busca aproximada: {near_summary['verified_pairs']} pares verificados, "
            f"{near_summary['cross_split_pairs']} entre splits; "
            f"{near_summary['short_documents_not_indexed']} textos curtos sem assinatura.",
            "- A busca por assinaturas é uma triagem com limites de recuperação. Zero pares encontrados não comprova ausência de near-duplicates.",
        ]
        if near_summary["candidate_limit_reached"]:
            lines.append("- O limite de candidatos foi atingido; a cobertura da busca aproximada ficou truncada.")
    lines += [
        "", "## Cobertura e possíveis atalhos", "",
        "As tabelas de cobertura incluem classe × domínio × comprimento, fonte humana × domínio, gerador × domínio e completude de metadados.",
        "`shortcut_gaps.csv` registra diferenças padronizadas de médias e KS para medidas numéricas, além de diferenças de prevalência para marcas de apresentação.",
        "Essas diferenças são descritivas: misturam autoria, domínio, fonte, gênero e condições de coleta/geração; não identificam causa isoladamente.",
    ]
    if probes is not None:
        lines += [
            "", "## Sondas diagnósticas", "",
            "Sondas lineares usam apenas atributos superficiais ou TF-IDF de 4-gramas de caracteres. O vocabulário, a imputação e a escala são ajustados apenas no treino; o limiar é escolhido na validação e a métrica é calculada no teste.",
            "Os intervalos são bootstrap estratificado por rótulo sobre previsões do teste; não incluem variação de treino, fonte ou gerador.",
            "Alta separabilidade indica sinal explorável e exige controle adicional; não demonstra que o sinal é espúrio nem valida o benchmark.",
        ]
    lines += [
        "", "## Limites da inferência", "",
        "Esta auditoria não testa contaminação dos dados de pré-treino de LLMs externos. IDs de documento-pai/trecho acadêmico não aparecem no schema atual, logo parentesco entre introdução e conclusão não pode ser descartado por completo.",
        "Não houve limpeza, seleção, normalização ou alteração de rótulos nesta execução.",
    ]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    config_path: str | Path, *, reuse_features: str | Path | None = None
) -> Path:
    """Run the configured audit and return its unique output directory."""

    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    workspace_root = resolve_from(config_file.parent, config.get("workspace_root", "."))
    catalog = DatasetCatalog.from_yaml(resolve_from(workspace_root, config["dataset_config"]))
    seed = int(config["seed"])
    seeded_libraries = seed_everything(seed)
    run = create_run(
        run_root=resolve_from(workspace_root, config["output_root"]),
        experiment_id=str(config["experiment_id"]),
        resolved_config=config,
        seed=seed,
        dataset_version=catalog.dataset_version,
        extra_metadata={"dataset_id": catalog.dataset_id, "seeded_libraries": seeded_libraries},
    )
    logger = configure_logging(log_file=run.directory / "audit.log", force=True)
    logger.info("Starting audit %s", run.run_id)
    tables_dir = run.directory / "tables"
    figures_dir = run.directory / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    loader = DatasetLoader(catalog.resource(config["corpus_resource"]))

    file_inventory = {}
    for name, spec in catalog.resources.items():
        file_inventory[name] = {"path": str(spec.path), "bytes": spec.path.stat().st_size,
                                "sha256": _hash_file(spec.path)}
    run.write_result("input_files", file_inventory)

    near_config = (
        NearDuplicateConfig(**{key: value for key, value in config["near_duplicates"].items() if key != "enabled"})
        if config["near_duplicates"].get("enabled", True) else None
    )
    lineage_fields = tuple(config.get("lineage_fields", []))
    if reuse_features is None:
        features, signatures = _read_features(
            loader, batch_size=int(config["batch_size"]), near_config=near_config,
            lineage_fields=lineage_fields, logger=logger,
        )
        features.to_parquet(run.directory / "features.parquet", index=False)
    else:
        source_features = Path(reuse_features).expanduser().resolve()
        previous_inventory = json.loads(
            (source_features.parent / "input_files.json").read_text(encoding="utf-8")
        )
        if previous_inventory != file_inventory:
            raise ValueError("Cannot reuse features: input file hashes changed")
        features = pd.read_parquet(source_features)
        if lineage_fields and any(field not in features for field in lineage_fields):
            raise ValueError("Reused features do not contain configured lineage columns")
        shutil.copy2(source_features, run.directory / "features.parquet")
        signatures = (
            _read_signatures_only(
                loader, features["record_id"].tolist(), near_config,
                int(config["batch_size"]), logger,
            ) if near_config is not None else []
        )
    logger.info("Features persisted for %s texts", len(features))

    checks, group_summary, group_details = run_integrity_checks(
        catalog, features, corpus_resource=config["corpus_resource"],
        split_resources=config["split_resources"],
        benchmark_resource=config.get("benchmark_resource"),
        taxonomy_resource=config["taxonomy_resource"],
        lineage_fields=lineage_fields,
    )
    tables = {"integrity_checks": checks, "repeated_group_summary": group_summary,
              "repeated_groups": group_details}
    logger.info("Integrity checks complete")
    _save_csv_tables(tables, tables_dir)

    near_summary = None
    if near_config is not None:
        near_pairs, near_summary = _near_duplicate_table(
            loader, features, signatures, near_config, int(config["batch_size"])
        )
        tables["near_duplicate_pairs"] = near_pairs
        tables.update(near_pair_review_tables(features, near_pairs))
        run.write_result("near_duplicate_search", near_summary)
        _save_csv_tables(
            {name: tables[name] for name in (
                "near_duplicate_pairs", "near_pair_review", "cross_split_near_by_source"
            )}, tables_dir
        )
        logger.info("Near-duplicate search verified %s pairs", near_summary["verified_pairs"])
    del signatures

    tables.update(coverage_tables(features, list(config["length_bins_words"])))
    tables.update(feature_summaries(features))
    _save_csv_tables(tables, tables_dir)
    logger.info("Coverage and feature summaries complete")

    probes = None
    probe_config = config.get("probes", {})
    if probe_config.get("enabled", False):
        common = {
            "bootstrap_replicates": int(probe_config["bootstrap_replicates"]),
            "bootstrap_confidence": float(probe_config["bootstrap_confidence"]),
            "seed": seed,
        }
        probes = run_surface_probes(features, **common)
        text_config = probe_config.get("text_probe", {})
        if text_config.get("enabled", False):
            char_probe, top_features = run_char4gram_probe(
                features, loader,
                max_train_per_label=int(text_config["max_train_per_label"]),
                max_validation_per_label=int(text_config["max_validation_per_label"]),
                max_test_per_label=int(text_config["max_test_per_label"]),
                max_features=int(text_config["max_features"]),
                min_df=int(text_config["min_df"]), **common,
            )
            probes = pd.concat([probes, char_probe], ignore_index=True)
            tables["top_char4grams"] = top_features
        tables["shortcut_probes"] = probes
        logger.info("Shortcut probes complete")

    _save_csv_tables(tables, tables_dir)
    create_figures(features, tables, figures_dir)
    _report(
        run.directory, catalog, features, checks, group_summary, near_summary,
        probes, config.get("benchmark_resource"),
    )
    run.write_result("audit_summary", {
        "rows": len(features), "integrity_error_checks_positive": int(
            ((checks["severity"] == "error") & (checks["count"] > 0)).sum()
        ), "integrity_warning_checks_positive": int(
            ((checks["severity"] == "warning") & (checks["count"] > 0)).sum()
        ), "near_duplicate_search": near_summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Audit complete: %s", run.directory)
    return run.directory


def refresh_summaries(run_dir: str | Path, config_path: str | Path) -> Path:
    """Regenerate coverage tables and figures from saved per-record measurements.

    This is useful when adding descriptive views without re-reading or changing
    the source corpus. It does not rerun integrity, near-duplicate or probes.
    """

    directory = Path(run_dir).expanduser().resolve()
    config = load_yaml(config_path)
    features = pd.read_parquet(directory / "features.parquet")
    tables = coverage_tables(features, list(config["length_bins_words"]))
    tables.update(feature_summaries(features))
    pairs_path = directory / "tables" / "near_duplicate_pairs.csv"
    if pairs_path.exists():
        tables.update(near_pair_review_tables(features, pd.read_csv(pairs_path)))
    probes_path = directory / "tables" / "shortcut_probes.csv"
    if probes_path.exists():
        tables["shortcut_probes"] = pd.read_csv(probes_path)
    _save_csv_tables(tables, directory / "tables")
    create_figures(features, tables, directory / "figures")
    return directory


def recheck_integrity(run_dir: str | Path, config_path: str | Path) -> Path:
    """Recheck structural and physical partitions using saved measurements."""

    directory = Path(run_dir).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    root = resolve_from(config_file.parent, config.get("workspace_root", "."))
    catalog = DatasetCatalog.from_yaml(resolve_from(root, config["dataset_config"]))
    previous_inventory = json.loads(
        (directory / "input_files.json").read_text(encoding="utf-8")
    )
    current_inventory = {
        name: {"path": str(spec.path), "bytes": spec.path.stat().st_size,
               "sha256": _hash_file(spec.path)}
        for name, spec in catalog.resources.items()
    }
    if current_inventory != previous_inventory:
        raise ValueError("Input files changed since this audit run")
    features = pd.read_parquet(directory / "features.parquet")
    checks, group_summary, group_details = run_integrity_checks(
        catalog, features, corpus_resource=config["corpus_resource"],
        split_resources=config["split_resources"],
        benchmark_resource=config.get("benchmark_resource"),
        taxonomy_resource=config["taxonomy_resource"],
        lineage_fields=tuple(config.get("lineage_fields", [])),
    )
    _save_csv_tables(
        {"integrity_checks": checks, "repeated_group_summary": group_summary,
         "repeated_groups": group_details}, directory / "tables"
    )
    near_path = directory / "near_duplicate_search.json"
    near_summary = json.loads(near_path.read_text(encoding="utf-8")) if near_path.exists() else None
    probe_path = directory / "tables" / "shortcut_probes.csv"
    probes = pd.read_csv(probe_path) if probe_path.exists() else None
    _report(
        directory, catalog, features, checks, group_summary, near_summary,
        probes, config.get("benchmark_resource"),
    )
    summary_path = directory / "audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["integrity_error_checks_positive"] = int(
        ((checks["severity"] == "error") & (checks["count"] > 0)).sum()
    )
    summary["integrity_warning_checks_positive"] = int(
        ((checks["severity"] == "warning") & (checks["count"] > 0)).sum()
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return directory


def verify_near_pairs(run_dir: str | Path, config_path: str | Path) -> Path:
    """Recheck saved near pairs against literal word-shingle sets."""

    directory = Path(run_dir).expanduser().resolve()
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    root = resolve_from(config_file.parent, config.get("workspace_root", "."))
    catalog = DatasetCatalog.from_yaml(resolve_from(root, config["dataset_config"]))
    pairs = pd.read_csv(directory / "tables" / "near_duplicate_pairs.csv")
    identifiers = set(pairs["left_record_id"]) | set(pairs["right_record_id"])
    loader = DatasetLoader(catalog.resource(config["corpus_resource"]))
    content_by_id: dict[str, str] = {}
    for batch in loader.iter_batches(columns=["record_id", "content"], batch_size=int(config["batch_size"])):
        for identifier, content in zip(
            batch.column("record_id").to_pylist(), batch.column("content").to_pylist()
        ):
            if identifier in identifiers:
                content_by_id[identifier] = content or ""
    if len(content_by_id) != len(identifiers):
        raise ValueError("Some near-pair record IDs are missing from the corpus")
    shingle_words = int(config["near_duplicates"]["shingle_words"])
    max_difference = 0.0
    threshold_failures = 0
    for pair in pairs.itertuples(index=False):
        jaccard, containment = exact_shingle_similarity(
            content_by_id[pair.left_record_id], content_by_id[pair.right_record_id],
            shingle_words,
        )
        max_difference = max(
            max_difference, abs(jaccard - pair.jaccard_5gram),
            abs(containment - pair.containment_5gram),
        )
        threshold_failures += not (
            jaccard >= float(config["near_duplicates"]["jaccard_threshold"])
            or containment >= float(config["near_duplicates"]["containment_threshold"])
        )
    result = {
        "pairs_checked": len(pairs),
        "maximum_metric_difference": max_difference,
        "pairs_below_threshold_after_literal_verification": int(threshold_failures),
        "passed": max_difference < 1e-12 and threshold_failures == 0,
    }
    target = directory / "near_duplicate_literal_verification.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise ValueError(f"Literal verification differs from saved pair table: {target}")
    return target
