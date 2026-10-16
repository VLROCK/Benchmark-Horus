"""Targeted, provenance-checked second pass for unresolved human themes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from horus_benchmark.audit.theme_common import ThemeContext, sha256_file
from horus_benchmark.audit.theme_llm_600 import (
    MODEL_KEYS, SPECIAL, _consensus, _shared_tokenizers, checked_manifest,
    prompt_config, read_llm_labels,
)
from horus_benchmark.audit.theme_llm_worker import parse_second_pass


def _settings(context: ThemeContext) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    settings = context.config["second_pass"]
    path = (context.catalog.workspace_root / settings["prompt_file"]).resolve()
    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (prompt.get("version") != 1 or not isinstance(prompt.get("common"), str)
            or not prompt["common"].strip()):
        raise ValueError("Invalid second-pass prompt version or instruction")
    if settings["wikipedia_theme"] in SPECIAL or any(
        settings["wikipedia_theme"] in themes for themes in context.taxonomy.values()
    ):
        raise ValueError("The source bucket must be distinct from topical themes")
    return settings, path, prompt


def _initial_rows(
    context: ThemeContext, sample_path: Path, gemma_path: Path, gpt_path: Path,
) -> pd.DataFrame:
    manifest = checked_manifest(context, sample_path)
    gemma = read_llm_labels(context, gemma_path, sample_path, "gemma4_31b")
    gpt = read_llm_labels(context, gpt_path, sample_path, "gpt_oss_120b")
    rows = manifest.merge(gemma, on=["record_id", "broad_area"], validate="one_to_one")
    rows = rows.merge(gpt, on=["record_id", "broad_area"], validate="one_to_one")
    if len(rows) != 600:
        raise ValueError("The first-pass annotations do not cover the 600 sample IDs")
    if rows["gemma4_31b_input_truncated"].any() or rows["gpt_oss_120b_input_truncated"].any():
        raise ValueError("Second pass requires complete first-pass inputs")
    rows["initial_status"], rows["initial_theme"] = _consensus(
        rows, "gemma4_31b", "gpt_oss_120b",
    )
    return rows.sort_values(["broad_area", "record_id"]).reset_index(drop=True)


def _targets(context: ThemeContext, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    settings, _, _ = _settings(context)
    wikipedia = rows["source_name"].eq(settings["wikipedia_source_name"])
    if int(wikipedia.sum()) != int(settings["expected_wikipedia_rows"]):
        raise ValueError("Wikipedia source count differs from the frozen pilot")
    if not rows.loc[wikipedia, "broad_area"].eq("informational").all():
        raise ValueError("Wikipedia source appears outside the informational area")
    targets = rows.loc[~wikipedia & rows["initial_status"].eq("uncertain")].copy()
    if len(targets) != int(settings["expected_target_rows"]):
        raise ValueError("Second-pass target count differs from the frozen pilot")
    return targets, wikipedia


def render_second_prompt(
    context: ThemeContext, template: dict[str, Any], area: str,
    content: str, candidate_labels: list[str],
    first_prompts: dict[str, Any] | None = None,
) -> str:
    if first_prompts is None:
        _, first_prompts = prompt_config(context)
    definition = first_prompts["areas"][area]
    themes = "\n".join(f"- {theme}" for theme in context.taxonomy[area])
    candidates = "\n".join(f"- {label}" for label in candidate_labels)
    return (
        f"{template['common']}\n\n"
        f"Área: {definition['name']}\n"
        f"Critério da área: {definition['guidance']}\n\n"
        f"Temas existentes nesta área:\n{themes}\n\n"
        f"Hipóteses iniciais anônimas (podem estar erradas):\n{candidates}\n\n"
        f"<texto_para_classificar>\n{content}\n</texto_para_classificar>"
    )


def prepared_second_requests(
    context: ThemeContext, sample_path: Path, gemma_path: Path, gpt_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Use only the unresolved non-Wikipedia IDs; expose no source or original theme."""
    rows = _initial_rows(context, sample_path, gemma_path, gpt_path)
    targets, _ = _targets(context, rows)
    _, prompt_path, prompt_template = _settings(context)
    tokenizers, fingerprints = _shared_tokenizers(context)
    llm = context.config["llm"]
    token_limit = (
        int(llm["max_model_len"]) - int(llm["max_output_tokens"])
        - int(llm["chat_template_reserve_tokens"])
    )
    if token_limit < 1:
        raise ValueError("No prompt tokens remain after context reserves")
    max_tokens: Counter[str] = Counter()
    lines: list[str] = []
    for row in targets.to_dict(orient="records"):
        area = row["broad_area"]
        candidates = sorted({row["gemma4_31b"], row["gpt_oss_120b"]})
        rendered = render_second_prompt(
            context, prompt_template, area, row["content"], candidates,
        )
        counts = {
            key: len(tokenizer.encode(rendered, add_special_tokens=False).ids)
            for key, tokenizer in tokenizers.items()
        }
        if any(count > token_limit for count in counts.values()):
            raise ValueError(f"Full second-pass prompt exceeds shared context: {row['record_id']}")
        for key, count in counts.items():
            max_tokens[key] = max(max_tokens[key], count)
        request = {
            "record_id": row["record_id"],
            "broad_area": area,
            "prompt": rendered,
            "candidate_labels": candidates,
            "allowed_themes": sorted(set(context.taxonomy[area]) | SPECIAL),
            "input_truncated": False,
            "input_characters_used": len(row["content"]),
            "input_tokens_by_model": counts,
        }
        lines.append(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    metadata = {
        "schema_version": 2,
        "task_kind": "theme_second_pass",
        "rows": len(targets),
        "manifest_sha256": sha256_file(sample_path),
        "upstream_gemma_sha256": sha256_file(gemma_path),
        "upstream_gpt_sha256": sha256_file(gpt_path),
        "prompts_sha256": sha256_file(prompt_path),
        "first_prompts_sha256": sha256_file(prompt_config(context)[0]),
        "taxonomy_sha256": sha256_file(context.taxonomy_path),
        "requests_sha256": hashlib.sha256(payload).hexdigest(),
        "llm_config": llm,
        "input_policy": "shared_token_budget",
        "truncated_rows": 0,
        "prompt_token_limit": token_limit,
        "max_input_tokens_by_model": dict(max_tokens),
        "tokenizer_sha256": fingerprints,
        "model_revisions": {key: llm["models"][key]["revision"] for key in MODEL_KEYS},
        "wikipedia_source_name": context.config["second_pass"]["wikipedia_source_name"],
        "wikipedia_theme": context.config["second_pass"]["wikipedia_theme"],
    }
    return payload, metadata


def prepare_second_pass(
    context: ThemeContext, sample_path: Path, gemma_path: Path, gpt_path: Path,
) -> Path:
    payload, metadata = prepared_second_requests(context, sample_path, gemma_path, gpt_path)
    run = context.new_run("prepare-second-pass", extra={
        "manifest_sha256": metadata["manifest_sha256"],
        "upstream_gemma_sha256": metadata["upstream_gemma_sha256"],
        "upstream_gpt_sha256": metadata["upstream_gpt_sha256"],
        "requests_sha256": metadata["requests_sha256"],
    })
    path = run.directory / "requests.jsonl"
    path.write_bytes(payload)
    run.write_result("requests_metadata", metadata)
    return path


def _second_labels(
    context: ThemeContext, path: Path, metadata: dict[str, Any],
    target_ids: set[str], model_key: str,
) -> pd.DataFrame:
    worker_meta = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "model_key": model_key,
        "task_kind": "theme_second_pass",
        "requests_sha256": metadata["requests_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "prompts_sha256": metadata["prompts_sha256"],
        "taxonomy_sha256": metadata["taxonomy_sha256"],
        "upstream_gemma_sha256": metadata["upstream_gemma_sha256"],
        "upstream_gpt_sha256": metadata["upstream_gpt_sha256"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "llm_config": metadata["llm_config"],
        "checkpoint_revision": metadata["model_revisions"][model_key],
    }
    if any(worker_meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Second-pass provenance differs: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(records) != len(target_ids) or {item["record_id"] for item in records} != target_ids:
        raise ValueError(f"Second-pass IDs differ for {model_key}")
    for record in records:
        area = record["broad_area"]
        allowed = set(context.taxonomy[area]) | SPECIAL
        parsed = parse_second_pass(record.get("raw_output", ""), allowed)
        if (record.get("status") != "valid" or record.get("input_truncated") is not False
                or parsed is None or any(record.get(key) != parsed[key] for key in parsed)):
            raise ValueError(f"Invalid second-pass response: {record['record_id']}")
    return pd.DataFrame.from_records(records)[
        ["record_id", "broad_area", "theme", "alternative_themes", "rationale"]
    ].rename(columns={
        "theme": model_key,
        "alternative_themes": f"{model_key}_alternatives",
        "rationale": f"{model_key}_rationale",
    })


def evaluate_second_pass(
    context: ThemeContext, sample_path: Path, first_gemma_path: Path,
    first_gpt_path: Path, second_gemma_path: Path, second_gpt_path: Path,
) -> Path:
    """Report coverage changes without treating LLM agreement as accuracy."""
    rows = _initial_rows(context, sample_path, first_gemma_path, first_gpt_path)
    targets, wikipedia = _targets(context, rows)
    _, metadata = prepared_second_requests(
        context, sample_path, first_gemma_path, first_gpt_path,
    )
    target_ids = set(targets["record_id"])
    for key, path in (("gemma4_31b", second_gemma_path),
                      ("gpt_oss_120b", second_gpt_path)):
        labels = _second_labels(context, path, metadata, target_ids, key)
        labels = labels.rename(columns={key: f"second_{key}"})
        rows = rows.merge(labels, on=["record_id", "broad_area"], how="left",
                          validate="one_to_one")
        if rows.loc[rows["record_id"].isin(target_ids), f"second_{key}"].isna().any():
            raise ValueError(f"Second-pass area/ID alignment differs for {key}")
    rows["final_theme"] = rows["initial_theme"]
    rows["assignment_method"] = rows["initial_status"].map({
        "named": "first_pass_consensus",
        "outside": "first_pass_outside",
        "uncertain": "unresolved",
    })
    rows.loc[wikipedia, "final_theme"] = context.config["second_pass"]["wikipedia_theme"]
    rows.loc[wikipedia, "assignment_method"] = "source_rule"
    second_gemma = rows["second_gemma4_31b"]
    second_gpt = rows["second_gpt_oss_120b"]
    matching = rows["record_id"].isin(target_ids) & second_gemma.eq(second_gpt)
    named = matching & ~second_gemma.isin(SPECIAL)
    outside = matching & second_gemma.eq("fora_taxonomia")
    rows.loc[named, "final_theme"] = second_gemma.loc[named]
    rows.loc[named, "assignment_method"] = "second_pass_consensus"
    rows.loc[outside, "final_theme"] = "fora_taxonomia"
    rows.loc[outside, "assignment_method"] = "second_pass_outside"
    rows["final_status"] = rows["assignment_method"].map({
        "first_pass_consensus": "named",
        "second_pass_consensus": "named",
        "first_pass_outside": "outside",
        "second_pass_outside": "outside",
        "source_rule": "source_bucket",
        "unresolved": "uncertain",
    })
    report_rows = rows.drop(columns="content")
    counts = report_rows["assignment_method"].value_counts().to_dict()
    summary = {
        "sample_rows": len(rows),
        "wikipedia_source_bucket": int(wikipedia.sum()),
        "second_pass_targets": len(targets),
        "initial_non_wikipedia_named": int((~wikipedia & rows["initial_status"].eq("named")).sum()),
        "initial_non_wikipedia_outside": int((~wikipedia & rows["initial_status"].eq("outside")).sum()),
        "second_pass_new_named": counts.get("second_pass_consensus", 0),
        "second_pass_new_outside": counts.get("second_pass_outside", 0),
        "remaining_unresolved": counts.get("unresolved", 0),
        "assignment_methods": counts,
        "by_area": {
            area: group["assignment_method"].value_counts().to_dict()
            for area, group in report_rows.groupby("broad_area")
        },
        "human_reference": False,
        "interpretation": "Second-pass consensus is operational coverage, not thematic accuracy",
        "manifest_sha256": metadata["manifest_sha256"],
        "upstream_gemma_sha256": metadata["upstream_gemma_sha256"],
        "upstream_gpt_sha256": metadata["upstream_gpt_sha256"],
        "second_requests_sha256": metadata["requests_sha256"],
        "second_gemma_sha256": sha256_file(second_gemma_path),
        "second_gpt_sha256": sha256_file(second_gpt_path),
    }
    run = context.new_run("evaluate-second-pass", extra={
        key: summary[key] for key in (
            "manifest_sha256", "upstream_gemma_sha256", "upstream_gpt_sha256",
            "second_requests_sha256", "second_gemma_sha256", "second_gpt_sha256",
        )
    })
    report_rows.to_csv(run.directory / "second_pass_rows.csv", index=False,
                       encoding="utf-8-sig")
    return run.write_result("second_pass_summary", summary)
