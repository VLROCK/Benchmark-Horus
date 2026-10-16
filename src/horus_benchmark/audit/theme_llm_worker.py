"""Dependency-light, resumable vLLM worker for prepared theme prompts.

The BERT/audit environment prepares requests.jsonl. This module deliberately
uses only the standard library until it loads the existing vLLM stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_KEYS = ("gemma4_31b", "gpt_oss_120b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_theme(response: str, allowed: set[str]) -> str | None:
    """Read the final JSON theme; never infer a label from free-form prose."""
    response = re.split(r"<\|(?:im_sep|meta_sep)\|>final", response)[-1]
    decoder = json.JSONDecoder()
    last_theme: str | None = None
    for match in re.finditer(r"\{", response):
        try:
            value, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("theme"), str):
            last_theme = value["theme"].strip()
    return last_theme if last_theme in allowed else None


def parse_second_pass(response: str, allowed: set[str]) -> dict[str, Any] | None:
    """Accept only a complete second-pass JSON decision with valid labels."""
    response = re.split(r"<\|(?:im_sep|meta_sep)\|>final", response)[-1]
    decoder = json.JSONDecoder()
    decision = None
    for match in re.finditer(r"\{", response):
        try:
            value, _ = decoder.raw_decode(response[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "theme" in value:
            decision = value
    if not isinstance(decision, dict) or set(decision) != {
        "theme", "alternative_themes", "rationale",
    }:
        return None
    theme = decision["theme"]
    alternatives = decision["alternative_themes"]
    rationale = decision["rationale"]
    named = allowed - {"incerto", "fora_taxonomia"}
    if (not isinstance(theme, str) or theme not in allowed
            or not isinstance(alternatives, list) or len(alternatives) > 2
            or len(set(map(str, alternatives))) != len(alternatives)
            or any(not isinstance(item, str) or item not in named for item in alternatives)
            or not isinstance(rationale, str) or not 1 <= len(rationale.strip()) <= 700):
        return None
    if theme in alternatives:
        return None
    return {
        "theme": theme,
        "alternative_themes": alternatives,
        "rationale": rationale.strip(),
    }


def read_requests(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    metadata = json.loads((path.parent / "requests_metadata.json").read_text(encoding="utf-8"))
    task_kind = metadata.get("task_kind", "theme_first_pass")
    second = task_kind == "theme_second_pass" and metadata.get("schema_version") == 2
    first = task_kind == "theme_first_pass" and metadata.get("schema_version") == 1
    full = task_kind in {"theme_full_first", "theme_full_second"} and metadata.get("schema_version") == 3
    if not (first or second or full) or not isinstance(metadata.get("rows"), int):
        raise ValueError("Prepared request metadata has an unexpected schema or row count")
    if (first and metadata["rows"] != 600) or (second and not 1 <= metadata["rows"] <= 600) or (full and not 1 <= metadata["rows"] <= 2000):
        raise ValueError("Prepared request row count is invalid")
    if metadata.get("requests_sha256") != sha256_file(path):
        raise ValueError("Prepared requests differ from their recorded SHA-256")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != metadata["rows"] or len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Prepared requests have missing or duplicate IDs")
    areas = Counter(row["broad_area"] for row in rows)
    if first and (len(areas) != 6 or set(areas.values()) != {100}):
        raise ValueError("Prepared requests require 100 texts in each of six areas")
    if (second or full) and (not set(areas).issubset({
        "academic", "informational", "journalistic", "juridical", "literary", "social",
    }) or not areas):
        raise ValueError("Prepared requests have invalid areas")
    for row in rows:
        if (not isinstance(row.get("prompt"), str) or not row["prompt"]
                or not isinstance(row.get("allowed_themes"), list)
                or not set(row["allowed_themes"]).issuperset({"incerto", "fora_taxonomia"})):
            raise ValueError(f"Malformed prepared request: {row.get('record_id')}")
        if (second or task_kind == "theme_full_second") and (not isinstance(row.get("candidate_labels"), list)
                       or not 1 <= len(row["candidate_labels"]) <= 2
                       or any(not isinstance(label, str) for label in row["candidate_labels"])
                       or len(set(row["candidate_labels"])) != len(row["candidate_labels"])
                       or any(label not in row["allowed_themes"]
                              for label in row["candidate_labels"])):
            raise ValueError(f"Invalid second-pass candidates: {row.get('record_id')}")
    if metadata.get("input_policy") == "shared_token_budget":
        limit = metadata.get("prompt_token_limit")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Shared prompt token limit is missing")
        if not isinstance(metadata.get("tokenizer_sha256"), dict):
            raise ValueError("Shared tokenizer fingerprints are missing")
        for row in rows:
            counts = row.get("input_tokens_by_model")
            if (not isinstance(counts, dict) or set(counts) != set(MODEL_KEYS)
                    or any(not isinstance(value, int) or value > limit or value < 1
                           for value in counts.values())):
                raise ValueError(f"Prompt exceeds shared token budget: {row['record_id']}")
    return rows, metadata


def _save_jsonl(path: Path, records: dict[str, dict[str, Any]], order: list[str]) -> None:
    temp = path.with_name(path.name + ".partial")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for record_id in order:
            if record_id in records:
                stream.write(json.dumps(records[record_id], ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _read_existing(path: Path, *, append_mode: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    if not append_mode:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["record_id"]] = record
        return records
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.readline():
                    stream.truncate(start)
                    break
                raise ValueError(f"Corrupt nonterminal annotation line in {path}")
            records[record["record_id"]] = record
    return records


def infer_local(
    requests_path: Path, model_key: str, checkpoint: Path,
    output_dir: Path, limit: int | None = None,
    limit_strategy: str = "first",
) -> Path:
    rows, request_meta = read_requests(requests_path)
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown local model key: {model_key}")
    llm_cfg = request_meta["llm_config"]
    model_cfg = llm_cfg["models"][model_key]
    checkpoint = checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file() or not re.fullmatch(r"[0-9a-f]{40}", checkpoint.name):
        raise ValueError("Use a complete local Hugging Face snapshot with a 40-character revision")
    if model_cfg.get("revision") and checkpoint.name != model_cfg["revision"]:
        raise ValueError(f"Checkpoint revision differs from prepared request: {model_key}")
    expected_tokenizer = request_meta.get("tokenizer_sha256", {}).get(model_key)
    if expected_tokenizer:
        tokenizer_path = checkpoint / "tokenizer.json"
        if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != expected_tokenizer:
            raise ValueError(f"Checkpoint tokenizer differs from prepared request: {model_key}")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise ValueError("--limit must be between 1 and the prepared row count")
        if limit_strategy not in {"first", "longest"}:
            raise ValueError("--limit-strategy must be first or longest")
        ordered = rows
        if limit_strategy == "longest":
            ordered = sorted(
                rows,
                key=lambda row: (
                    -max(row.get("input_tokens_by_model", {}).values(), default=0),
                    row["record_id"],
                ),
            )
        if request_meta.get("task_kind", "").startswith("theme_full_"):
            chosen = ordered[:limit]
        else:
            counts: Counter[str] = Counter()
            per_area = max(1, limit // 6)
            chosen = []
            for row in ordered:
                area = row["broad_area"]
                if counts[area] < per_area and len(chosen) < limit:
                    chosen.append(row)
                    counts[area] += 1
        selected_ids = {row["record_id"] for row in chosen}
        rows_to_run = [row for row in rows if row["record_id"] in selected_ids]
    else:
        rows_to_run = rows
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"
    output_path = output_dir / "annotations.jsonl"
    identity = {
        "model_key": model_key,
        "model_repo": model_cfg["hf_repo"],
        "checkpoint_revision": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "requests_sha256": request_meta["requests_sha256"],
        "manifest_sha256": request_meta["manifest_sha256"],
        "prompts_sha256": request_meta["prompts_sha256"],
        "taxonomy_sha256": request_meta["taxonomy_sha256"],
        "llm_config": llm_cfg,
    }
    if expected_tokenizer:
        identity["tokenizer_sha256"] = request_meta["tokenizer_sha256"]
    if request_meta.get("task_kind") != "theme_first_pass":
        identity["task_kind"] = request_meta["task_kind"]
        if "upstream_gemma_sha256" in request_meta:
            identity["upstream_gemma_sha256"] = request_meta["upstream_gemma_sha256"]
            identity["upstream_gpt_sha256"] = request_meta["upstream_gpt_sha256"]
        if "upstream_index_sha256" in request_meta:
            identity["upstream_index_sha256"] = request_meta["upstream_index_sha256"]
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Output directory belongs to another model, sample or prompt")
    elif output_path.exists():
        raise ValueError("Annotations exist without metadata; refusing unsafe resume")
    else:
        meta_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    append_mode = request_meta.get("task_kind", "").startswith("theme_full_")
    records = _read_existing(output_path, append_mode=append_mode)
    all_order = [row["record_id"] for row in rows]
    candidates = [row for row in rows_to_run
                  if records.get(row["record_id"], {}).get("status") != "valid"]
    if not candidates:
        print(f"Already complete for requested rows: {model_key}", file=sys.stderr)
        return output_path

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Inference requires exactly one allocated CUDA GPU")
    print(
        f"{model_key}: {len(candidates)} pending, GPU={torch.cuda.get_device_name(0)}, "
        f"vLLM={vllm.__version__}, transformers={transformers.__version__}",
        file=sys.stderr, flush=True,
    )
    (output_dir / "runtime.json").write_text(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    kwargs = {
        "model": str(checkpoint),
        "tensor_parallel_size": 1,
        "max_model_len": int(llm_cfg["max_model_len"]),
        "gpu_memory_utilization": float(llm_cfg["gpu_memory_utilization"]),
        "enforce_eager": bool(model_cfg["enforce_eager"]),
        "trust_remote_code": True,
    }
    if "limit_mm_per_prompt" in model_cfg:
        kwargs["limit_mm_per_prompt"] = model_cfg["limit_mm_per_prompt"]
    engine = LLM(**kwargs)
    sampling = SamplingParams(
        temperature=float(llm_cfg["temperature"]),
        max_tokens=int(llm_cfg["max_output_tokens"]),
    )
    batch_size = int(llm_cfg["batch_size"])
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        conversations = [[{"role": "user", "content": row["prompt"]}] for row in batch]
        output = engine.chat(
            messages=conversations,
            sampling_params=sampling,
            chat_template_kwargs=model_cfg.get("chat_template_kwargs"),
            use_tqdm=False,
        )
        if len(output) != len(batch):
            raise RuntimeError("vLLM returned a different number of responses")
        for row, response in zip(batch, output):
            generated = response.outputs[0]
            raw = generated.text
            second_pass = request_meta.get("task_kind") in {
                "theme_second_pass", "theme_full_second",
            }
            decision = (
                parse_second_pass(raw, set(row["allowed_themes"]))
                if second_pass else None
            )
            theme = decision["theme"] if second_pass and decision else (
                None if second_pass else parse_theme(raw, set(row["allowed_themes"]))
            )
            actual_input_tokens = (
                len(response.prompt_token_ids)
                if response.prompt_token_ids is not None else None
            )
            if (actual_input_tokens is not None
                    and actual_input_tokens + int(llm_cfg["max_output_tokens"])
                    > int(llm_cfg["max_model_len"])):
                raise RuntimeError(
                    f"{model_key}: chat-template token count exceeds reserved context "
                    f"for {row['record_id']}"
                )
            records[row["record_id"]] = {
                "record_id": row["record_id"],
                "broad_area": row["broad_area"],
                "theme": theme,
                "status": "valid" if theme is not None else "invalid",
                "raw_output": raw,
                "alternative_themes": decision["alternative_themes"] if decision else None,
                "rationale": decision["rationale"] if decision else None,
                "input_truncated": row["input_truncated"],
                "input_characters_used": row["input_characters_used"],
                "input_tokens_estimated": row.get("input_tokens_by_model", {}).get(model_key),
                "input_tokens_actual": actual_input_tokens,
                "output_tokens": len(generated.token_ids),
                "finish_reason": generated.finish_reason,
            }
        if append_mode:
            with output_path.open("a", encoding="utf-8", newline="\n") as stream:
                for record in batch:
                    stream.write(json.dumps(records[record["record_id"]], ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            _save_jsonl(output_path, records, all_order)
        valid = sum(record["status"] == "valid" for record in records.values())
        print(f"{model_key}: {len(records)}/{len(rows)} responses, {valid} valid", file=sys.stderr, flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm-worker")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-strategy", choices=("first", "longest"), default="first")
    args = parser.parse_args(argv)
    result = infer_local(args.requests, args.model_key, args.checkpoint,
                         args.output_dir, args.limit, args.limit_strategy)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
