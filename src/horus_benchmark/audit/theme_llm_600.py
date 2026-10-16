"""A blind, shared 600-text sample and agreement-only theme audit."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from horus_benchmark.audit.theme_common import (
    ThemeContext, read_run_metadata, sha256_file, validate_artifact_context,
)
from horus_benchmark.data import DatasetLoader


SPECIAL = {"incerto", "fora_taxonomia"}
MODEL_KEYS = ("gemma4_31b", "gpt_oss_120b")


def prompt_config(context: ThemeContext) -> tuple[Path, dict[str, Any]]:
    root = context.catalog.workspace_root
    path = (root / context.config["prompts"]).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or set(data.get("areas", {})) != set(context.taxonomy):
        raise ValueError("Prompt areas/version differ from the current taxonomy")
    for area, definition in data["areas"].items():
        for example in definition.get("examples", []):
            if example["theme"] not in context.taxonomy[area]:
                raise ValueError(f"Example outside taxonomy: {area} / {example['theme']}")
    return path, data


def render_prompt(context: ThemeContext, prompts: dict[str, Any], area: str, content: str) -> str:
    definition = prompts["areas"][area]
    themes = "\n".join(f"- {theme}" for theme in context.taxonomy[area])
    examples = "\n".join(
        f"Exemplo: {example['text']} -> {example['theme']}"
        for example in definition.get("examples", [])
    )
    return (
        f"{prompts['common']}\n\nÁrea: {definition['name']}\n"
        f"Instruções da área: {definition['guidance']}\n\n"
        f"Temas permitidos nesta área:\n{themes}\n\n"
        f"Exemplos ilustrativos, não retirados da amostra:\n{examples}\n\n"
        f"<texto_para_classificar>\n{content}\n</texto_para_classificar>"
    )


def create_sample(context: ThemeContext, pilot_path: Path) -> Path:
    """Reuse the original 20/area blind pilot; add 80/area uniformly."""
    sampling = context.config["sampling"]
    if (sampling.get("method") != "pilot_plus_uniform_without_replacement"
            or sampling.get("pilot_per_area") != 20
            or sampling.get("total_per_area") != 100):
        raise ValueError("This audit requires 20 pilot + 80 new texts per area")
    prompt_path, _ = prompt_config(context)
    corpus_path = context.catalog.resource(context.config["corpus_resource"]).path
    pilot_meta = read_run_metadata(pilot_path)
    if (pilot_meta.get("dataset_version") != context.catalog.dataset_version
            or pilot_meta.get("extra", {}).get("corpus_sha256") != sha256_file(corpus_path)):
        raise ValueError("Pilot and current complete corpus differ")
    pilot = pd.read_csv(pilot_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if (len(pilot) != 120 or pilot["record_id"].duplicated().any()
            or pilot.groupby("broad_area").size().to_dict() != {
                area: 20 for area in context.taxonomy
            }):
        raise ValueError("Pilot must contain 120 unique human IDs, 20 per area")
    loader = DatasetLoader(context.catalog.resource(context.config["corpus_resource"]))
    metadata = loader.read_table(
        columns=["record_id", "broad_area", "source_name", "split", "word_count"],
        filter_expression=ds.field("label_name") == "human",
    ).to_pandas()
    if metadata["record_id"].isna().any() or metadata["record_id"].duplicated().any():
        raise ValueError("Human IDs are missing or duplicated in the corpus")
    pilot_ids = set(pilot["record_id"])
    if not pilot_ids.issubset(set(metadata["record_id"])):
        raise ValueError("Pilot contains IDs absent from human corpus")
    rng = np.random.default_rng(int(context.config["seed"]) + 600)
    parts: list[pd.DataFrame] = []
    for area in context.taxonomy:
        candidates = metadata.loc[
            (metadata["broad_area"] == area)
            & (~metadata["record_id"].isin(pilot_ids))
        ].sort_values("record_id")
        if len(candidates) < 80:
            raise ValueError(f"Too few remaining human texts in {area}")
        added = candidates.iloc[rng.choice(len(candidates), 80, replace=False)].copy()
        added["sample_origin"] = "new_uniform"
        kept = metadata.loc[
            (metadata["broad_area"] == area) & metadata["record_id"].isin(pilot_ids)
        ].copy()
        if len(kept) != 20:
            raise ValueError(f"Pilot area differs from corpus: {area}")
        kept["sample_origin"] = "existing_pilot"
        parts.extend((kept, added))
    selected = pd.concat(parts, ignore_index=True)
    if len(selected) != 600 or selected["record_id"].duplicated().any():
        raise ValueError("Sample must have 600 distinct human texts")
    text = loader.read_table(
        columns=["record_id", "broad_area", "content"],
        filter_expression=ds.field("record_id").isin(selected["record_id"].tolist()),
    ).to_pandas()
    if len(text) != 600 or text["record_id"].duplicated().any():
        raise ValueError("Could not retrieve every selected text exactly once")
    pilot_text = pilot[["record_id", "broad_area", "content"]].merge(
        text, on=["record_id", "broad_area"], suffixes=("_pilot", "_corpus"),
        validate="one_to_one",
    )
    if len(pilot_text) != 120 or not pilot_text["content_pilot"].equals(pilot_text["content_corpus"]):
        raise ValueError("Pilot IDs, areas or texts differ from the complete corpus")
    manifest = selected.merge(text, on=["record_id", "broad_area"], validate="one_to_one")
    if len(manifest) != 600 or manifest["content"].isna().any() or (manifest["content"].str.strip() == "").any():
        raise ValueError("Sample text is missing or empty")
    manifest = manifest.sort_values(["broad_area", "record_id"]).reset_index(drop=True)
    run = context.new_run("sample-600", extra={
        "corpus_sha256": sha256_file(corpus_path),
        "pilot_sha256": sha256_file(pilot_path),
        "prompts_sha256": sha256_file(prompt_path),
    })
    path = run.directory / "sample_manifest.parquet"
    manifest.to_parquet(path, index=False)
    run.write_result("sample_summary", {
        "rows": 600, "per_area": 100, "reused_pilot": 120,
        "new_uniform": 480, "human_labels_used": False,
        "source_distribution": manifest.groupby(["broad_area", "source_name"], dropna=False)
        .size().reset_index(name="rows").to_dict(orient="records"),
    })
    return path


def checked_manifest(context: ThemeContext, path: Path) -> pd.DataFrame:
    validate_artifact_context(context, path)
    meta = read_run_metadata(path)
    corpus_path = context.catalog.resource(context.config["corpus_resource"]).path
    if meta.get("extra", {}).get("corpus_sha256") != sha256_file(corpus_path):
        raise ValueError("Manifest comes from a different corpus")
    frame = pd.read_parquet(path)
    if (len(frame) != 600 or frame["record_id"].duplicated().any()
            or frame.groupby("broad_area").size().to_dict() != {
                area: 100 for area in context.taxonomy
            }):
        raise ValueError("Manifest must contain 600 distinct IDs, 100 per area")
    return frame


def prepared_requests(context: ThemeContext, manifest_path: Path) -> tuple[bytes, dict[str, Any]]:
    """Build deterministic, blind prompts in the audit environment."""
    manifest = checked_manifest(context, manifest_path)
    prompt_path, prompts = prompt_config(context)
    max_chars = int(context.config["llm"]["max_input_characters"])
    if max_chars < 1:
        raise ValueError("max_input_characters must be positive")
    lines: list[str] = []
    for row in manifest.sort_values(["broad_area", "record_id"]).to_dict(orient="records"):
        content = row["content"]
        area = row["broad_area"]
        request = {
            "record_id": row["record_id"],
            "broad_area": area,
            "prompt": render_prompt(context, prompts, area, content[:max_chars]),
            "allowed_themes": sorted(set(context.taxonomy[area]) | SPECIAL),
            "input_truncated": len(content) > max_chars,
            "input_characters_used": min(len(content), max_chars),
        }
        lines.append(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {
        "schema_version": 1,
        "rows": 600,
        "manifest_sha256": sha256_file(manifest_path),
        "prompts_sha256": sha256_file(prompt_path),
        "taxonomy_sha256": sha256_file(context.taxonomy_path),
        "requests_sha256": hashlib.sha256(payload).hexdigest(),
        "llm_config": context.config["llm"],
    }
    return payload, metadata


def prepare_inference(context: ThemeContext, manifest_path: Path) -> Path:
    payload, metadata = prepared_requests(context, manifest_path)
    run = context.new_run("prepare-llm", extra={
        "sample_sha256": metadata["manifest_sha256"],
        "requests_sha256": metadata["requests_sha256"],
    })
    path = run.directory / "requests.jsonl"
    path.write_bytes(payload)
    run.write_result("requests_metadata", metadata)
    return path


def read_llm_labels(
    context: ThemeContext, path: Path, manifest_path: Path, model_key: str,
) -> pd.DataFrame:
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    prompt_path, _ = prompt_config(context)
    requests_payload, _ = prepared_requests(context, manifest_path)
    if (metadata.get("model_key") != model_key
            or metadata.get("manifest_sha256") != sha256_file(manifest_path)
            or metadata.get("prompts_sha256") != sha256_file(prompt_path)
            or metadata.get("taxonomy_sha256") != sha256_file(context.taxonomy_path)
            or metadata.get("requests_sha256") != hashlib.sha256(requests_payload).hexdigest()
            or metadata.get("llm_config") != context.config["llm"]):
        raise ValueError(f"LLM provenance differs: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {record["record_id"]: record for record in records}
    manifest = checked_manifest(context, manifest_path)
    expected = set(manifest["record_id"])
    if set(latest) != expected:
        raise ValueError(f"{model_key}: expected all 600 manifest IDs, got {len(latest)}")
    for record in latest.values():
        if record.get("status") != "valid":
            raise ValueError(f"{model_key}: invalid response for {record['record_id']}; retry it")
        area = record["broad_area"]
        if record["theme"] not in context.taxonomy[area] and record["theme"] not in SPECIAL:
            raise ValueError(f"{model_key}: theme outside taxonomy: {record['record_id']}")
    frame = pd.DataFrame.from_records(list(latest.values()))[
        ["record_id", "broad_area", "theme", "input_truncated"]
    ]
    return frame.rename(columns={
        "theme": model_key,
        "input_truncated": f"{model_key}_input_truncated",
    })


def _conditional_kappa(frame: pd.DataFrame, left: str, right: str) -> float | None:
    observed = 0.0
    expected = 0.0
    for _, group in frame.groupby("broad_area"):
        weight = len(group) / len(frame)
        observed += weight * float((group[left] == group[right]).mean())
        p_left = group[left].value_counts(normalize=True)
        p_right = group[right].value_counts(normalize=True)
        expected += weight * sum(
            p_left.get(label, 0.0) * p_right.get(label, 0.0)
            for label in set(p_left.index) | set(p_right.index)
        )
    return float((observed - expected) / (1 - expected)) if expected < 1 else None


def pair_metrics(frame: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    if frame.empty:
        return {"rows": 0, "raw_agreement": None, "kappa_global": None,
                "kappa_within_area": None}
    first = frame["broad_area"].astype(str) + "::" + frame[left].astype(str)
    second = frame["broad_area"].astype(str) + "::" + frame[right].astype(str)
    # Cohen's kappa is undefined if both raters use only one identical label.
    global_kappa = (
        cohen_kappa_score(first, second)
        if len(set(first) | set(second)) > 1 else None
    )
    joint = frame.loc[~frame[left].isin(SPECIAL) & ~frame[right].isin(SPECIAL)]
    return {
        "rows": len(frame),
        "raw_agreement": float((frame[left] == frame[right]).mean()),
        "kappa_global": (
            float(global_kappa)
            if global_kappa is not None and np.isfinite(global_kappa) else None
        ),
        "kappa_within_area": _conditional_kappa(frame, left, right),
        "left_special": int(frame[left].isin(SPECIAL).sum()),
        "right_special": int(frame[right].isin(SPECIAL).sum()),
        "joint_assigned_rows": len(joint),
        "joint_assigned_agreement_diagnostic": (
            float((joint[left] == joint[right]).mean()) if len(joint) else None
        ),
    }


def evaluate_agreement(
    context: ThemeContext, manifest_path: Path, gemma_path: Path,
    gpt_path: Path, bert_path: Path,
) -> Path:
    from horus_benchmark.audit.theme_workflow import _predictions

    manifest = checked_manifest(context, manifest_path)
    gemma = read_llm_labels(context, gemma_path, manifest_path, "gemma4_31b")
    gpt = read_llm_labels(context, gpt_path, manifest_path, "gpt_oss_120b")
    bert = _predictions(context, bert_path)
    selected_bert = bert.loc[bert["record_id"].isin(manifest["record_id"])].copy()
    if len(selected_bert) != 600:
        raise ValueError("BERT predictions do not cover the same 600 IDs")
    rows = manifest.drop(columns="content").merge(
        gemma, on=["record_id", "broad_area"], validate="one_to_one",
    ).merge(gpt, on=["record_id", "broad_area"], validate="one_to_one").merge(
        selected_bert[["record_id", "broad_area", "theme_predicted", "top_probability",
                       "mapped_theme", "mapping_method", "model_run_id"]],
        on=["record_id", "broad_area"], validate="one_to_one",
    )
    if len(rows) != 600:
        raise ValueError("Areas/IDs differ among sample, LLMs and BERT")
    rows = rows.rename(columns={"theme_predicted": "bert_forced"}).reset_index(drop=True)
    threshold = float(context.config["threshold"]["fixed_probability"])
    rows["bert_thresholded"] = np.where(
        rows["top_probability"] >= threshold, rows["bert_forced"], "incerto",
    )
    rows["length_quartile"] = pd.qcut(
        rows["word_count"].fillna(0).astype(float).rank(method="first"),
        4, labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    rows["bert_score_quartile"] = pd.qcut(
        rows["top_probability"].rank(method="first"),
        4, labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    rows["mapping_group"] = np.where(rows["mapped_theme"].notna(), "manual_map", "model_only")
    pairs = [
        ("gemma4_31b", "gpt_oss_120b"),
        ("gemma4_31b", "bert_forced"),
        ("gpt_oss_120b", "bert_forced"),
        ("gemma4_31b", "bert_thresholded"),
        ("gpt_oss_120b", "bert_thresholded"),
    ]
    rng = np.random.default_rng(int(context.config["seed"]) + 601)
    strata = [group.index.to_numpy() for _, group in rows.groupby("broad_area")]
    replicates = int(context.config["analysis"]["bootstrap_replicates"])
    metrics: dict[str, Any] = {}
    confusion_parts: list[pd.DataFrame] = []
    subgroup_rows: list[dict[str, Any]] = []
    for left, right in pairs:
        name = f"{left}__{right}"
        result = pair_metrics(rows, left, right)
        draws: list[float] = []
        for _ in range(replicates):
            indices = np.concatenate([
                rng.choice(group, len(group), replace=True) for group in strata
            ])
            value = _conditional_kappa(rows.iloc[indices], left, right)
            if value is not None and np.isfinite(value):
                draws.append(value)
        result["kappa_within_area_bootstrap_95"] = (
            [float(x) for x in np.quantile(draws, [0.025, 0.975])] if draws else None
        )
        result["by_area"] = {
            area: pair_metrics(group, left, right)
            for area, group in rows.groupby("broad_area")
        }
        result["by_length_quartile"] = {
            str(quartile): pair_metrics(group, left, right)
            for quartile, group in rows.groupby("length_quartile")
        }
        result["by_bert_score_quartile"] = {
            str(quartile): pair_metrics(group, left, right)
            for quartile, group in rows.groupby("bert_score_quartile")
        }
        result["by_mapping_group"] = {
            group_name: pair_metrics(group, left, right)
            for group_name, group in rows.groupby("mapping_group")
        }
        result["by_split_exploratory"] = {
            str(split): pair_metrics(group, left, right)
            for split, group in rows.groupby("split")
        }
        min_source = int(context.config["analysis"]["min_source_rows"])
        result["by_source_exploratory"] = {
            str(source): pair_metrics(group, left, right)
            for source, group in rows.groupby("source_name", dropna=False)
            if len(group) >= min_source
        }
        for factor in ("by_area", "by_length_quartile", "by_bert_score_quartile",
                       "by_mapping_group", "by_split_exploratory", "by_source_exploratory"):
            for level, value in result[factor].items():
                subgroup_rows.append({
                    "pair": name, "factor": factor, "level": level,
                    **{key: value[key] for key in ("rows", "raw_agreement", "kappa_global",
                                                    "kappa_within_area", "left_special", "right_special",
                                                    "joint_assigned_rows",
                                                    "joint_assigned_agreement_diagnostic")},
                })
        confusion = rows.groupby(["broad_area", left, right], dropna=False).size().reset_index(name="rows")
        confusion_parts.append(confusion.rename(columns={left: "left_theme", right: "right_theme"})
                               .assign(pair=name))
        metrics[name] = result
    run = context.new_run("agreement-600", extra={
        "sample_sha256": sha256_file(manifest_path),
        "gemma_sha256": sha256_file(gemma_path),
        "gpt_oss_sha256": sha256_file(gpt_path),
        "bert_sha256": sha256_file(bert_path),
    })
    rows.to_csv(run.directory / "agreement_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(subgroup_rows).to_csv(
        run.directory / "agreement_by_factor.csv", index=False, encoding="utf-8-sig",
    )
    pd.concat(confusion_parts, ignore_index=True).to_csv(
        run.directory / "pairwise_confusion.csv", index=False, encoding="utf-8-sig",
    )
    counts = pd.concat([
        rows.groupby(["broad_area", column], dropna=False).size().reset_index(name="rows")
        .rename(columns={column: "theme"}).assign(model=column)
        for column in ("gemma4_31b", "gpt_oss_120b", "bert_forced", "bert_thresholded")
    ], ignore_index=True)
    counts.to_csv(run.directory / "theme_counts_by_model.csv", index=False, encoding="utf-8-sig")
    disagreements = rows.loc[~(
        (rows["gemma4_31b"] == rows["gpt_oss_120b"])
        & (rows["gemma4_31b"] == rows["bert_forced"])
    )]
    disagreements.to_csv(run.directory / "disagreements.csv", index=False, encoding="utf-8-sig")
    run.write_result("fixed_bert_threshold", {
        "policy": "fixed_without_human_calibration",
        "threshold": threshold,
        "predictions_sha256": sha256_file(bert_path),
        "agreement_run_id": run.run_id,
        "note": "This is an operational score cutoff, not a precision guarantee",
    })
    return run.write_result("agreement", {
        "sample_rows": 600, "per_area": 100,
        "human_reference": False,
        "interpretation": "Agreement is not accuracy; no human or expert gold labels were used",
        "bert_threshold": threshold,
        "three_way_forced_agreement": float(1 - len(disagreements) / 600),
        "bert_operational_coverage": float((rows["bert_thresholded"] != "incerto").mean()),
        "llm_input_truncated": {
            key: int(rows[f"{key}_input_truncated"].sum())
            for key in MODEL_KEYS
        },
        "pairs": metrics,
    })
