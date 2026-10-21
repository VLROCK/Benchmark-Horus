"""Sharded two-LLM theme annotation of the immutable complete Hórus corpus."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.dataset as ds
import yaml

from horus_benchmark.audit.theme_common import ThemeContext, sha256_file
from horus_benchmark.audit.theme_llm_600 import (
    MODEL_KEYS, SPECIAL, _shared_prefix, _shared_tokenizers, prompt_config, render_prompt,
)
from horus_benchmark.audit.theme_llm_second_pass import render_second_prompt
from horus_benchmark.audit.theme_llm_worker import (
    parse_second_pass, parse_theme, read_jsonl, read_requests,
)


def _corpus_path(context: ThemeContext) -> Path:
    return context.catalog.resource(context.config["corpus_resource"]).path


def _human_rows(context: ThemeContext) -> pd.DataFrame:
    columns = ["record_id", "label_name", "broad_area", "source_name", "theme",
               "split", "word_count", "content"]
    dataset = ds.dataset(_corpus_path(context), format="parquet")
    rows = dataset.to_table(
        columns=columns, filter=ds.field("label_name") == "human",
    ).to_pandas()
    expected = int(context.config["full"]["expected_human_rows"])
    if (len(rows) != expected or rows["record_id"].isna().any()
            or rows["record_id"].duplicated().any()
            or rows["content"].isna().any()
            or not set(rows["broad_area"]).issubset(context.taxonomy)):
        raise ValueError("Complete human corpus differs from the declared population")
    wikipedia = rows["source_name"].eq(context.config["full"]["wikipedia_source_name"])
    if (int(wikipedia.sum()) != int(context.config["full"]["expected_wikipedia_rows"])
            or not rows.loc[wikipedia, "broad_area"].eq("informational").all()):
        raise ValueError("Wikipedia source differs from the declared source rule")
    return rows.sort_values(["broad_area", "record_id"]).reset_index(drop=True)


def _token_limit(context: ThemeContext) -> int:
    llm = context.config["llm"]
    value = (int(llm["max_model_len"]) - int(llm["max_output_tokens"])
             - int(llm["chat_template_reserve_tokens"]))
    if llm.get("input_policy") != "shared_token_budget" or value < 1:
        raise ValueError("Full annotation requires a positive shared token budget")
    return value


def _fit_prompt(renderer, content: str, tokenizers: dict[str, Any], limit: int):
    """Use the same longest content prefix for both checkpoints."""
    def counts(length: int) -> dict[str, int]:
        prompt = renderer(content[:length])
        return {key: len(tok.encode(prompt, add_special_tokens=False).ids)
                for key, tok in tokenizers.items()}

    complete = counts(len(content))
    if all(value <= limit for value in complete.values()):
        return content, complete
    if any(value > limit for value in counts(0).values()):
        raise ValueError("Instructions exceed the shared model context")
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if all(value <= limit for value in counts(mid).values()):
            lo = mid
        else:
            hi = mid - 1
    return content[:lo], counts(lo)


def _write_shards(
    context: ThemeContext, run, manifest_path: Path, requests: Iterable[dict[str, Any]],
    *, task_kind: str, prompt_path: Path, token_fingerprints: dict[str, str],
    corpus_sha256: str, upstream_index_sha256: str | None = None,
) -> Path:
    shard_size = int(context.config["full"]["shard_size"])
    if not 1 <= shard_size <= 2000:
        raise ValueError("shard_size must be between 1 and 2000")
    folder = run.directory / "requests"
    folder.mkdir()
    metadata_base = {
        "schema_version": 3,
        "task_kind": task_kind,
        "manifest_sha256": sha256_file(manifest_path),
        "prompts_sha256": sha256_file(prompt_path),
        "taxonomy_sha256": sha256_file(context.taxonomy_path),
        "llm_config": context.config["llm"],
        "input_policy": "shared_token_budget",
        "prompt_token_limit": _token_limit(context),
        "tokenizer_sha256": token_fingerprints,
        "model_revisions": {
            key: context.config["llm"]["models"][key]["revision"] for key in MODEL_KEYS
        },
    }
    if upstream_index_sha256 is not None:
        metadata_base["upstream_index_sha256"] = upstream_index_sha256
    shards: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    total = 0
    truncated = 0

    def flush() -> None:
        nonlocal total, truncated
        if not buffer:
            return
        number = len(shards)
        name = f"shard_{number:04d}"
        directory = folder / name
        directory.mkdir()
        payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")) for row in buffer) + "\n").encode("utf-8")
        request_path = directory / "requests.jsonl"
        request_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        counts = {key: max(row["input_tokens_by_model"][key] for row in buffer)
                  for key in MODEL_KEYS}
        meta = {
            **metadata_base,
            "rows": len(buffer),
            "requests_sha256": digest,
            "truncated_rows": sum(row["input_truncated"] for row in buffer),
            "max_input_tokens_by_model": counts,
        }
        (directory / "requests_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        shards.append({
            "shard_id": number, "requests": f"requests/{name}/requests.jsonl",
            "rows": len(buffer), "requests_sha256": digest,
            "max_input_tokens_by_model": counts,
        })
        total += len(buffer)
        truncated += meta["truncated_rows"]
        buffer.clear()

    for request in requests:
        buffer.append(request)
        if len(buffer) == shard_size:
            flush()
    flush()
    index = {
        "schema_version": 1,
        "task_kind": task_kind,
        "dataset_version": context.catalog.dataset_version,
        "corpus_sha256": corpus_sha256,
        "manifest_file": manifest_path.name,
        "manifest_sha256": metadata_base["manifest_sha256"],
        "taxonomy_sha256": metadata_base["taxonomy_sha256"],
        "prompts_sha256": metadata_base["prompts_sha256"],
        "llm_config": context.config["llm"],
        "upstream_index_sha256": upstream_index_sha256,
        "rows": total,
        "truncated_rows": truncated,
        "shards": shards,
    }
    return run.write_result("requests_index", index)


def prepare_full_first(context: ThemeContext) -> Path:
    corpus = _corpus_path(context)
    corpus_hash = sha256_file(corpus)
    rows = _human_rows(context)
    run = context.new_run("prepare-full-first", extra={"corpus_sha256": corpus_hash})
    manifest_path = run.directory / "human_manifest.parquet"
    rows.drop(columns=["label_name", "content"]).rename(
        columns={"theme": "theme_original"},
    ).to_parquet(manifest_path, index=False)
    tokenizers, fingerprints = _shared_tokenizers(context)
    prompt_path, prompts = prompt_config(context)
    limit = _token_limit(context)
    wikipedia = context.config["full"]["wikipedia_source_name"]

    def requests():
        for row in rows.itertuples(index=False):
            if row.source_name == wikipedia:
                continue
            selected, counts = _shared_prefix(
                context, prompts, row.broad_area, row.content, tokenizers, limit,
            )
            yield {
                "record_id": row.record_id,
                "broad_area": row.broad_area,
                "prompt": render_prompt(context, prompts, row.broad_area, selected),
                "allowed_themes": sorted(set(context.taxonomy[row.broad_area]) | SPECIAL),
                "input_truncated": len(selected) < len(row.content),
                "input_characters_used": len(selected),
                "input_tokens_by_model": counts,
            }

    index_path = _write_shards(
        context, run, manifest_path, requests(), task_kind="theme_full_first",
        prompt_path=prompt_path, token_fingerprints=fingerprints,
        corpus_sha256=corpus_hash,
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = len(rows) - int(context.config["full"]["expected_wikipedia_rows"])
    if index["rows"] != expected:
        raise ValueError("Full first-pass request count differs from non-Wikipedia humans")
    return index_path


def _load_index(context: ThemeContext, path: Path, task_kind: str) -> dict[str, Any]:
    index = json.loads(path.read_text(encoding="utf-8"))
    prompt_path = (
        prompt_config(context)[0] if task_kind == "theme_full_first" else
        (context.catalog.workspace_root / context.config["full"]["second_prompt_file"]).resolve()
    )
    if (index.get("schema_version") != 1 or index.get("task_kind") != task_kind
            or index.get("dataset_version") != context.catalog.dataset_version
            or index.get("corpus_sha256") != sha256_file(_corpus_path(context))
            or index.get("taxonomy_sha256") != sha256_file(context.taxonomy_path)
            or index.get("prompts_sha256") != sha256_file(prompt_path)
            or index.get("llm_config") != context.config["llm"]):
        raise ValueError(f"Full-run index provenance differs: {path}")
    manifest = path.parent / index["manifest_file"]
    if sha256_file(manifest) != index["manifest_sha256"]:
        raise ValueError("Full-run human manifest differs from index")
    return index


def _collect(
    context: ThemeContext, index_path: Path, index: dict[str, Any], model_key: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for shard in index["shards"]:
        request_path = index_path.parent / shard["requests"]
        requests, request_meta = read_requests(request_path)
        if (request_meta["requests_sha256"] != shard["requests_sha256"]
                or request_meta["manifest_sha256"] != index["manifest_sha256"]
                or request_meta["prompts_sha256"] != index["prompts_sha256"]
                or request_meta["task_kind"] != index["task_kind"]):
            raise ValueError(f"Shard metadata differs: {request_path}")
        folder = index_path.parent / "inference" / model_key / f"shard_{shard['shard_id']:04d}"
        meta_path = folder / "metadata.json"
        output_path = folder / "annotations.jsonl"
        if not meta_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"Missing LLM output: {folder}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("model_key") != model_key
                or meta.get("task_kind") != index["task_kind"]
                or meta.get("requests_sha256") != shard["requests_sha256"]
                or meta.get("manifest_sha256") != index["manifest_sha256"]
                or meta.get("prompts_sha256") != index["prompts_sha256"]
                or meta.get("taxonomy_sha256") != index["taxonomy_sha256"]
                or meta.get("llm_config") != index["llm_config"]
                or meta.get("checkpoint_revision") != context.config["llm"]["models"][model_key]["revision"]
                or (index["upstream_index_sha256"] is not None and
                    meta.get("upstream_index_sha256") != index["upstream_index_sha256"])):
            raise ValueError(f"LLM output provenance differs: {folder}")
        expected = {row["record_id"]: row for row in requests}
        latest = {record["record_id"]: record for record in read_jsonl(output_path)}
        if set(latest) != set(expected):
            raise ValueError(f"Incomplete LLM shard: {folder} ({len(latest)}/{len(expected)})")
        for record_id, record in latest.items():
            request = expected[record_id]
            if (record.get("status") not in {"valid", "invalid"}
                    or record.get("broad_area") != request["broad_area"]
                    or record.get("input_truncated") != request["input_truncated"]):
                raise ValueError(f"Invalid LLM output for {record_id}: {folder}")
            allowed = set(request["allowed_themes"])
            theme = record.get("theme")
            parse_status = "valid"
            if record["status"] == "valid":
                if theme not in allowed:
                    raise ValueError(f"LLM label outside allowed themes for {record_id}: {folder}")
            elif theme is not None:
                raise ValueError(f"Invalid-status LLM output has a label for {record_id}: {folder}")
            elif index["task_kind"] == "theme_full_first":
                theme = parse_theme(record.get("raw_output", ""), allowed)
                parse_status = "normalized_label" if theme is not None else "parse_failure"
            else:
                parsed = parse_second_pass(
                    record.get("raw_output", ""), allowed,
                )
                theme = parsed["theme"] if parsed else None
                parse_status = "reparsed" if parsed else "parse_failure"
            if index["task_kind"] == "theme_full_second" and record["status"] == "valid":
                parsed = parse_second_pass(record.get("raw_output", ""), allowed)
                if parsed is None or any(record.get(key) != value for key, value in parsed.items()):
                    raise ValueError(f"Invalid second-pass JSON for {record_id}: {folder}")
            records.append({
                "record_id": record_id, "broad_area": record["broad_area"],
                model_key: theme if theme is not None else "incerto",
                f"{model_key}_truncated": record["input_truncated"],
                f"{model_key}_parse_status": parse_status,
            })
    frame = pd.DataFrame.from_records(records, columns=[
        "record_id", "broad_area", model_key, f"{model_key}_truncated",
        f"{model_key}_parse_status",
    ])
    if len(frame) != index["rows"] or frame["record_id"].duplicated().any():
        raise ValueError(f"Collected {model_key} outputs have duplicate or missing IDs")
    return frame


def _first_frame(context: ThemeContext, index_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = _load_index(context, index_path, "theme_full_first")
    manifest = pd.read_parquet(index_path.parent / index["manifest_file"])
    if len(manifest) != int(context.config["full"]["expected_human_rows"]):
        raise ValueError("Human manifest row count differs")
    rows = manifest.copy()
    for key in MODEL_KEYS:
        rows = rows.merge(_collect(context, index_path, index, key),
                          on=["record_id", "broad_area"], how="left", validate="one_to_one")
    wikipedia = rows["source_name"].eq(context.config["full"]["wikipedia_source_name"])
    for key in MODEL_KEYS:
        if rows.loc[~wikipedia, key].isna().any() or rows.loc[wikipedia, key].notna().any():
            raise ValueError("First-pass outputs do not match non-Wikipedia humans")
    same = rows[MODEL_KEYS[0]].eq(rows[MODEL_KEYS[1]])
    rows["first_status"] = "uncertain"
    rows.loc[same & ~rows[MODEL_KEYS[0]].isin(SPECIAL), "first_status"] = "named"
    rows.loc[same & rows[MODEL_KEYS[0]].eq("fora_taxonomia"), "first_status"] = "outside"
    rows.loc[wikipedia, "first_status"] = "source_bucket"
    rows["first_truncated"] = rows[[f"{key}_truncated" for key in MODEL_KEYS]].fillna(False).any(axis=1)
    return rows, index


def prepare_full_second(context: ThemeContext, first_index_path: Path) -> Path:
    first_rows, first_index = _first_frame(context, first_index_path)
    targets = first_rows.loc[
        first_rows["first_status"].eq("uncertain") & ~first_rows["first_truncated"]
    ].copy()
    source_rows = _human_rows(context).set_index("record_id")
    if not set(targets["record_id"]).issubset(source_rows.index):
        raise ValueError("Second-pass target IDs are absent from the corpus")
    first_hash = sha256_file(first_index_path)
    run = context.new_run("prepare-full-second", extra={
        "corpus_sha256": first_index["corpus_sha256"],
        "first_index_sha256": first_hash,
    })
    manifest_path = run.directory / "human_manifest.parquet"
    shutil.copyfile(first_index_path.parent / first_index["manifest_file"], manifest_path)
    tokenizers, fingerprints = _shared_tokenizers(context)
    prompt_path = (context.catalog.workspace_root /
                   context.config["full"]["second_prompt_file"]).resolve()
    template = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if template.get("version") != 1 or not template.get("common"):
        raise ValueError("Invalid second-pass prompt template")
    _, first_prompts = prompt_config(context)
    limit = _token_limit(context)

    def requests():
        for row in targets.itertuples(index=False):
            content = source_rows.at[row.record_id, "content"]
            candidates = sorted({row.gemma4_31b, row.gpt_oss_120b})
            renderer = lambda value: render_second_prompt(
                context, template, row.broad_area, value, candidates, first_prompts,
            )
            selected, counts = _fit_prompt(renderer, content, tokenizers, limit)
            yield {
                "record_id": row.record_id, "broad_area": row.broad_area,
                "prompt": renderer(selected), "candidate_labels": candidates,
                "allowed_themes": sorted(set(context.taxonomy[row.broad_area]) | SPECIAL),
                "input_truncated": len(selected) < len(content),
                "input_characters_used": len(selected),
                "input_tokens_by_model": counts,
            }

    index_path = _write_shards(
        context, run, manifest_path, requests(), task_kind="theme_full_second",
        prompt_path=prompt_path, token_fingerprints=fingerprints,
        corpus_sha256=first_index["corpus_sha256"],
        upstream_index_sha256=first_hash,
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index["rows"] != len(targets):
        raise ValueError("Second-pass index omitted target IDs")
    return index_path


def finalize_full(
    context: ThemeContext, first_index_path: Path, second_index_path: Path,
) -> Path:
    rows, first_index = _first_frame(context, first_index_path)
    second_index = _load_index(context, second_index_path, "theme_full_second")
    if (second_index["upstream_index_sha256"] != sha256_file(first_index_path)
            or second_index["manifest_sha256"] != first_index["manifest_sha256"]):
        raise ValueError("Second-pass index is not derived from this first pass")
    expected_targets = set(rows.loc[
        rows["first_status"].eq("uncertain") & ~rows["first_truncated"], "record_id",
    ])
    if second_index["rows"] != len(expected_targets):
        raise ValueError("Second-pass target count differs from first-pass uncertainty")
    for key in MODEL_KEYS:
        values = _collect(context, second_index_path, second_index, key)
        if set(values["record_id"]) != expected_targets:
            raise ValueError(f"Second-pass {key} IDs differ from unresolved first pass")
        values = values.rename(columns={
            key: f"second_{key}", f"{key}_truncated": f"second_{key}_truncated",
            f"{key}_parse_status": f"second_{key}_parse_status",
        })
        rows = rows.merge(values, on=["record_id", "broad_area"], how="left",
                          validate="one_to_one")
    wikipedia = rows["first_status"].eq("source_bucket")
    rows["theme_final"] = "incerto"
    rows["assignment_method"] = "unresolved"
    rows.loc[wikipedia, "theme_final"] = context.config["full"]["wikipedia_theme"]
    rows.loc[wikipedia, "assignment_method"] = "source_rule"
    first_named = rows["first_status"].eq("named") & ~rows["first_truncated"]
    first_outside = rows["first_status"].eq("outside") & ~rows["first_truncated"]
    rows.loc[first_named, "theme_final"] = rows.loc[first_named, "gemma4_31b"]
    rows.loc[first_named, "assignment_method"] = "first_pass_consensus"
    rows.loc[first_outside, "theme_final"] = "fora_taxonomia"
    rows.loc[first_outside, "assignment_method"] = "first_pass_outside"
    second_complete = (
        rows["record_id"].isin(expected_targets)
        & ~rows[[f"second_{key}_truncated" for key in MODEL_KEYS]].fillna(False).any(axis=1)
    )
    second_agree = second_complete & rows["second_gemma4_31b"].eq(rows["second_gpt_oss_120b"])
    second_named = second_agree & ~rows["second_gemma4_31b"].isin(SPECIAL)
    second_outside = second_agree & rows["second_gemma4_31b"].eq("fora_taxonomia")
    rows.loc[second_named, "theme_final"] = rows.loc[second_named, "second_gemma4_31b"]
    rows.loc[second_named, "assignment_method"] = "second_pass_consensus"
    rows.loc[second_outside, "theme_final"] = "fora_taxonomia"
    rows.loc[second_outside, "assignment_method"] = "second_pass_outside"
    parse_failed = rows[[
        f"{key}_parse_status" for key in MODEL_KEYS
    ] + [
        f"second_{key}_parse_status" for key in MODEL_KEYS
    ]].eq("parse_failure").any(axis=1)
    rows.loc[rows["assignment_method"].eq("unresolved") & parse_failed,
             "assignment_method"] = "parse_failure"
    truncated = rows["first_truncated"] | rows[[
        f"second_{key}_truncated" for key in MODEL_KEYS
    ]].fillna(False).any(axis=1)
    rows.loc[truncated, "theme_final"] = "incerto"
    rows.loc[truncated, "assignment_method"] = "context_truncated"
    rows["theme_status"] = rows["assignment_method"].map({
        "source_rule": "source_bucket", "first_pass_consensus": "named",
        "second_pass_consensus": "named", "first_pass_outside": "outside",
        "second_pass_outside": "outside", "unresolved": "uncertain",
        "parse_failure": "uncertain", "context_truncated": "context_truncated",
    })
    if rows["theme_status"].isna().any() or rows["record_id"].duplicated().any():
        raise ValueError("Final human annotations are incomplete or duplicated")
    assignments = rows[[
        "record_id", "broad_area", "source_name", "split", "theme_original",
        "theme_final", "theme_status", "assignment_method", "gemma4_31b",
        "gpt_oss_120b", "second_gemma4_31b", "second_gpt_oss_120b",
        "gemma4_31b_parse_status", "gpt_oss_120b_parse_status",
        "second_gemma4_31b_parse_status", "second_gpt_oss_120b_parse_status",
    ]].copy()
    run = context.new_run("materialize-full", extra={
        "corpus_sha256": first_index["corpus_sha256"],
        "first_index_sha256": sha256_file(first_index_path),
        "second_index_sha256": sha256_file(second_index_path),
    })
    assignments.to_parquet(run.directory / "human_theme_assignments.parquet", index=False)
    corpus = pd.read_parquet(_corpus_path(context))
    if (len(corpus) != int(context.config["full"]["expected_corpus_rows"])
            or corpus["record_id"].duplicated().any()):
        raise ValueError("Complete corpus row count or IDs differ")
    corpus["theme_original"] = corpus["theme"]
    new_columns = assignments[[
        "record_id", "theme_final", "theme_status", "assignment_method",
        "gemma4_31b", "gpt_oss_120b", "second_gemma4_31b", "second_gpt_oss_120b",
        "gemma4_31b_parse_status", "gpt_oss_120b_parse_status",
        "second_gemma4_31b_parse_status", "second_gpt_oss_120b_parse_status",
    ]]
    corpus = corpus.merge(new_columns, on="record_id", how="left", validate="one_to_one")
    human = corpus["label_name"].eq("human")
    if int(human.sum()) != len(assignments) or corpus.loc[human, "theme_final"].isna().any():
        raise ValueError("Derived corpus lacks some human annotations")
    corpus.loc[human, "theme"] = corpus.loc[human, "theme_final"]
    corpus.loc[~human, "theme_final"] = corpus.loc[~human, "theme_original"]
    corpus.loc[~human, "theme_status"] = "synthetic_original"
    corpus.loc[~human, "assignment_method"] = "synthetic_original"
    corpus["theme_assignment_run"] = run.run_id
    corpus.to_parquet(run.directory / "all_with_themes.parquet", index=False)
    for split in ("train", "validation", "test"):
        subset = corpus.loc[corpus["split"].eq(split)]
        spec = context.catalog.resource(context.config[f"{split}_resource"])
        if spec.expected_rows is not None and len(subset) != spec.expected_rows:
            raise ValueError(f"Derived {split} row count differs from immutable split")
        subset.to_parquet(run.directory / f"{split}_with_themes.parquet", index=False)
    grouped = corpus.groupby(
        ["label_name", "broad_area", "theme_status", "theme"], dropna=False,
    ).size().reset_index(name="rows")
    grouped.to_csv(run.directory / "theme_distribution.csv", index=False, encoding="utf-8-sig")
    return run.write_result("theme_full_summary", {
        "corpus_rows": len(corpus), "human_rows": int(human.sum()),
        "synthetic_rows": int((~human).sum()),
        "assignment_methods": rows["assignment_method"].value_counts().to_dict(),
        "parse_status_by_model": {
            key: rows[key].value_counts().to_dict() for key in (
                "gemma4_31b_parse_status", "gpt_oss_120b_parse_status",
                "second_gemma4_31b_parse_status", "second_gpt_oss_120b_parse_status",
            )
        },
        "by_area_status": rows.groupby(["broad_area", "theme_status"]).size()
        .reset_index(name="rows").to_dict(orient="records"),
        "first_requests": first_index["rows"],
        "second_requests": second_index["rows"],
        "first_truncated_requests": first_index["truncated_rows"],
        "second_truncated_requests": second_index["truncated_rows"],
        "corpus_sha256": first_index["corpus_sha256"],
        "first_index_sha256": sha256_file(first_index_path),
        "second_index_sha256": sha256_file(second_index_path),
        "interpretation": "LLM consensus is organizational metadata, not verified thematic accuracy",
    })
