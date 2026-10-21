"""A blind, shared 600-text sample and agreement-only theme audit."""

from __future__ import annotations

import json
import hashlib
import os
from collections import Counter
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
    if data.get("version") not in (1, 2) or set(data.get("areas", {})) != set(context.taxonomy):
        raise ValueError("Prompt areas/version differ from the current taxonomy")
    for area, definition in data["areas"].items():
        for example in definition.get("examples", []):
            if example["theme"] not in context.taxonomy[area] and example["theme"] not in SPECIAL:
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


def _shared_tokenizers(context: ThemeContext) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the two pinned tokenizer.json files without loading either LLM."""
    from tokenizers import Tokenizer

    hub = Path(os.environ.get("HORUS_HF_HUB") or os.environ.get("HF_HUB_CACHE")
               or Path.home() / ".cache" / "huggingface" / "hub")
    tokenizers: dict[str, Any] = {}
    fingerprints: dict[str, str] = {}
    for key in MODEL_KEYS:
        model = context.config["llm"]["models"][key]
        revision = model.get("revision")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"{key} requires a pinned 40-character revision")
        repository = "models--" + model["hf_repo"].replace("/", "--")
        snapshot = hub / repository / "snapshots" / revision
        path = snapshot / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned tokenizer: {path}")
        config_path = snapshot / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing pinned model config: {config_path}")
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        native_limit = model_config.get("text_config", model_config).get(
            "max_position_embeddings"
        )
        if native_limit is not None and int(context.config["llm"]["max_model_len"]) > int(native_limit):
            raise ValueError(f"Configured context exceeds {key} native limit: {native_limit}")
        tokenizers[key] = Tokenizer.from_file(str(path))
        fingerprints[key] = sha256_file(path)
    return tokenizers, fingerprints


def _shared_prefix(
    context: ThemeContext, prompts: dict[str, Any], area: str, content: str,
    tokenizers: dict[str, Any], token_limit: int,
) -> tuple[str, dict[str, int]]:
    """Longest common text prefix fitting both tokenizers' prompt budgets."""
    def lengths(characters: int) -> dict[str, int]:
        prompt = render_prompt(context, prompts, area, content[:characters])
        return {
            key: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            for key, tokenizer in tokenizers.items()
        }

    whole = lengths(len(content))
    if all(count <= token_limit for count in whole.values()):
        return content, whole
    if any(count > token_limit for count in lengths(0).values()):
        raise ValueError(f"Prompt instructions alone exceed token budget: {area}")
    low, high = 0, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if all(count <= token_limit for count in lengths(middle).values()):
            low = middle
        else:
            high = middle - 1
    return content[:low], lengths(low)


def prepared_requests(context: ThemeContext, manifest_path: Path) -> tuple[bytes, dict[str, Any]]:
    """Build deterministic, blind prompts in the audit environment."""
    manifest = checked_manifest(context, manifest_path)
    prompt_path, prompts = prompt_config(context)
    llm = context.config["llm"]
    policy = llm.get("input_policy", "character_prefix")
    tokenizers: dict[str, Any] = {}
    tokenizer_sha256: dict[str, str] = {}
    if policy == "shared_token_budget":
        tokenizers, tokenizer_sha256 = _shared_tokenizers(context)
        token_limit = (int(llm["max_model_len"]) - int(llm["max_output_tokens"])
                       - int(llm["chat_template_reserve_tokens"]))
        if token_limit < 1:
            raise ValueError("No input tokens remain after output and chat-template reserves")
    elif policy == "character_prefix":
        max_chars = int(llm["max_input_characters"])
        if max_chars < 1:
            raise ValueError("max_input_characters must be positive")
    else:
        raise ValueError(f"Unknown LLM input policy: {policy}")
    lines: list[str] = []
    truncated_by_area: Counter[str] = Counter()
    max_tokens_by_model: Counter[str] = Counter()
    for row in manifest.sort_values(["broad_area", "record_id"]).to_dict(orient="records"):
        content = row["content"]
        area = row["broad_area"]
        if policy == "shared_token_budget":
            selected, token_counts = _shared_prefix(
                context, prompts, area, content, tokenizers, token_limit,
            )
            for key, count in token_counts.items():
                max_tokens_by_model[key] = max(max_tokens_by_model[key], count)
        else:
            selected, token_counts = content[:max_chars], None
        truncated = len(selected) < len(content)
        if truncated:
            truncated_by_area[area] += 1
        request = {
            "record_id": row["record_id"],
            "broad_area": area,
            "prompt": render_prompt(context, prompts, area, selected),
            "allowed_themes": sorted(set(context.taxonomy[area]) | SPECIAL),
            "input_truncated": truncated,
            "input_characters_used": len(selected),
        }
        if token_counts is not None:
            request["input_tokens_by_model"] = token_counts
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
        "llm_config": llm,
        "input_policy": policy,
        "truncated_rows": sum(truncated_by_area.values()),
        "truncated_by_area": dict(truncated_by_area),
    }
    if policy == "shared_token_budget":
        metadata.update({
            "tokenizer_sha256": tokenizer_sha256,
            "model_revisions": {
                key: llm["models"][key]["revision"] for key in MODEL_KEYS
            },
            "prompt_token_limit": token_limit,
            "max_input_tokens_by_model": dict(max_tokens_by_model),
        })
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
    from horus_benchmark.audit.theme_llm_worker import read_jsonl

    records = read_jsonl(path)
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


def _consensus(frame: pd.DataFrame, left: str, right: str) -> tuple[pd.Series, pd.Series]:
    same = frame[left] == frame[right]
    named = same & ~frame[left].isin(SPECIAL)
    outside = same & frame[left].eq("fora_taxonomia")
    status = pd.Series(
        np.select([named, outside], ["named", "outside"], default="uncertain"),
        index=frame.index,
    )
    theme = pd.Series(
        np.where(named, frame[left], np.where(outside, "fora_taxonomia", "incerto")),
        index=frame.index,
    )
    return status, theme


def evaluate_llm_context(
    context: ThemeContext, baseline_context: ThemeContext, manifest_path: Path,
    baseline_gemma_path: Path, baseline_gpt_path: Path,
    gemma_path: Path, gpt_path: Path,
) -> Path:
    """Compare two LLM-only runs on the same 600 IDs and report consensus coverage."""
    manifest = checked_manifest(context, manifest_path)
    checked_manifest(baseline_context, manifest_path)
    baseline_gemma = read_llm_labels(
        baseline_context, baseline_gemma_path, manifest_path, "gemma4_31b",
    ).rename(columns={
        "gemma4_31b": "baseline_gemma4_31b",
        "gemma4_31b_input_truncated": "baseline_input_truncated",
    })
    baseline_gpt = read_llm_labels(
        baseline_context, baseline_gpt_path, manifest_path, "gpt_oss_120b",
    ).rename(columns={
        "gpt_oss_120b": "baseline_gpt_oss_120b",
        "gpt_oss_120b_input_truncated": "baseline_gpt_input_truncated",
    })
    gemma = read_llm_labels(context, gemma_path, manifest_path, "gemma4_31b")
    gpt = read_llm_labels(context, gpt_path, manifest_path, "gpt_oss_120b")
    rows = manifest.drop(columns="content").merge(
        baseline_gemma, on=["record_id", "broad_area"], validate="one_to_one",
    ).merge(
        baseline_gpt, on=["record_id", "broad_area"], validate="one_to_one",
    ).merge(
        gemma, on=["record_id", "broad_area"], validate="one_to_one",
    ).merge(
        gpt, on=["record_id", "broad_area"], validate="one_to_one",
    ).reset_index(drop=True)
    if len(rows) != 600:
        raise ValueError("Baseline and long-context LLM outputs do not share 600 IDs")
    if (not rows["baseline_input_truncated"].equals(rows["baseline_gpt_input_truncated"])
            or not rows["gemma4_31b_input_truncated"].equals(
                rows["gpt_oss_120b_input_truncated"]
            )):
        raise ValueError("The two LLMs did not receive identically truncated inputs")
    rows["baseline_consensus_status"], rows["baseline_consensus_theme"] = _consensus(
        rows, "baseline_gemma4_31b", "baseline_gpt_oss_120b",
    )
    rows["consensus_status"], rows["consensus_theme"] = _consensus(
        rows, "gemma4_31b", "gpt_oss_120b",
    )
    rows["gemma_changed"] = rows["baseline_gemma4_31b"] != rows["gemma4_31b"]
    rows["gpt_changed"] = rows["baseline_gpt_oss_120b"] != rows["gpt_oss_120b"]

    def coverage(group: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": len(group),
            "named": int(group["consensus_status"].eq("named").sum()),
            "outside": int(group["consensus_status"].eq("outside").sum()),
            "uncertain": int(group["consensus_status"].eq("uncertain").sum()),
            "baseline_truncated": int(group["baseline_input_truncated"].sum()),
            "long_truncated": int(group["gemma4_31b_input_truncated"].sum()),
            "gemma_label_changed": int(group["gemma_changed"].sum()),
            "gpt_label_changed": int(group["gpt_changed"].sum()),
            "pair": pair_metrics(group, "gemma4_31b", "gpt_oss_120b"),
        }

    rng = np.random.default_rng(int(context.config["seed"]) + 602)
    strata = [group.index.to_numpy() for _, group in rows.groupby("broad_area")]
    draws: list[float] = []
    for _ in range(int(context.config["analysis"]["bootstrap_replicates"])):
        indices = np.concatenate([
            rng.choice(group, len(group), replace=True) for group in strata
        ])
        value = _conditional_kappa(
            rows.iloc[indices], "gemma4_31b", "gpt_oss_120b",
        )
        if value is not None and np.isfinite(value):
            draws.append(value)
    overall = coverage(rows)
    overall["pair"]["kappa_within_area_bootstrap_95"] = (
        [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        if draws else None
    )
    by_area = {
        area: coverage(group) for area, group in rows.groupby("broad_area")
    }
    minimum = int(context.config["analysis"]["min_source_rows"])
    by_source = {
        str(source): coverage(group)
        for source, group in rows.groupby("source_name", dropna=False)
        if len(group) >= minimum
    }
    baseline_prompt_hash = json.loads(
        (baseline_gemma_path.parent / "metadata.json").read_text(encoding="utf-8")
    )["prompts_sha256"]
    new_prompt_hash = json.loads(
        (gemma_path.parent / "metadata.json").read_text(encoding="utf-8")
    )["prompts_sha256"]
    _, request_meta = prepared_requests(context, manifest_path)
    transitions = (
        rows.groupby(["baseline_consensus_status", "consensus_status"])
        .size().reset_index(name="rows").to_dict(orient="records")
    )
    run = context.new_run("llm-context-comparison", extra={
        "sample_sha256": sha256_file(manifest_path),
        "baseline_gemma_sha256": sha256_file(baseline_gemma_path),
        "baseline_gpt_sha256": sha256_file(baseline_gpt_path),
        "gemma_sha256": sha256_file(gemma_path),
        "gpt_sha256": sha256_file(gpt_path),
    })
    rows.to_csv(run.directory / "comparison_rows.csv", index=False, encoding="utf-8-sig")
    factors = [
        {"factor": "area", "level": area, **{k: v for k, v in result.items() if k != "pair"},
         "raw_agreement": result["pair"]["raw_agreement"],
         "kappa_within_area": result["pair"]["kappa_within_area"]}
        for area, result in by_area.items()
    ] + [
        {"factor": "source", "level": source, **{k: v for k, v in result.items() if k != "pair"},
         "raw_agreement": result["pair"]["raw_agreement"],
         "kappa_within_area": result["pair"]["kappa_within_area"]}
        for source, result in by_source.items()
    ]
    pd.DataFrame.from_records(factors).to_csv(
        run.directory / "coverage_by_factor.csv", index=False, encoding="utf-8-sig",
    )
    return run.write_result("comparison", {
        "sample_rows": 600,
        "human_reference": False,
        "interpretation": "Agreement and coverage are not classification accuracy",
        "shared_long_requests_sha256": request_meta["requests_sha256"],
        "baseline_prompts_sha256": baseline_prompt_hash,
        "long_prompts_sha256": new_prompt_hash,
        "isolated_context_change": baseline_prompt_hash == new_prompt_hash,
        "long_preparation": {
            key: request_meta[key] for key in (
                "input_policy", "prompt_token_limit", "truncated_rows",
                "truncated_by_area", "max_input_tokens_by_model",
                "tokenizer_sha256", "model_revisions",
            )
        },
        "overall": overall,
        "baseline_pair": pair_metrics(
            rows, "baseline_gemma4_31b", "baseline_gpt_oss_120b",
        ),
        "by_area": by_area,
        "by_source_exploratory": by_source,
        "consensus_status_transitions": transitions,
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
