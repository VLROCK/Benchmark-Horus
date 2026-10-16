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


def read_requests(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    metadata = json.loads((path.parent / "requests_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("rows") != 600:
        raise ValueError("Prepared request metadata has an unexpected schema or row count")
    if metadata.get("requests_sha256") != sha256_file(path):
        raise ValueError("Prepared requests differ from their recorded SHA-256")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 600 or len({row["record_id"] for row in rows}) != 600:
        raise ValueError("Prepared requests require 600 unique IDs")
    areas = Counter(row["broad_area"] for row in rows)
    if len(areas) != 6 or set(areas.values()) != {100}:
        raise ValueError("Prepared requests require 100 texts in each of six areas")
    for row in rows:
        if (not isinstance(row.get("prompt"), str) or not row["prompt"]
                or not isinstance(row.get("allowed_themes"), list)
                or not set(row["allowed_themes"]).issuperset({"incerto", "fora_taxonomia"})):
            raise ValueError(f"Malformed prepared request: {row.get('record_id')}")
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


def infer_local(
    requests_path: Path, model_key: str, checkpoint: Path,
    output_dir: Path, limit: int | None = None,
) -> Path:
    rows, request_meta = read_requests(requests_path)
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown local model key: {model_key}")
    llm_cfg = request_meta["llm_config"]
    model_cfg = llm_cfg["models"][model_key]
    checkpoint = checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file() or not re.fullmatch(r"[0-9a-f]{40}", checkpoint.name):
        raise ValueError("Use a complete local Hugging Face snapshot with a 40-character revision")
    if limit is not None:
        if not 1 <= limit <= 600:
            raise ValueError("--limit must be between 1 and 600")
        counts: Counter[str] = Counter()
        per_area = max(1, limit // 6)
        chosen = []
        for row in rows:
            area = row["broad_area"]
            if counts[area] < per_area and len(chosen) < limit:
                chosen.append(row)
                counts[area] += 1
        rows_to_run = chosen
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
    records: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["record_id"]] = record
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
            theme = parse_theme(raw, set(row["allowed_themes"]))
            records[row["record_id"]] = {
                "record_id": row["record_id"],
                "broad_area": row["broad_area"],
                "theme": theme,
                "status": "valid" if theme is not None else "invalid",
                "raw_output": raw,
                "input_truncated": row["input_truncated"],
                "input_characters_used": row["input_characters_used"],
                "output_tokens": len(generated.token_ids),
                "finish_reason": generated.finish_reason,
            }
        _save_jsonl(output_path, records, all_order)
        valid = sum(record["status"] == "valid" for record in records.values())
        print(f"{model_key}: {len(records)}/600 responses, {valid} valid", file=sys.stderr, flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm-worker")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = infer_local(args.requests, args.model_key, args.checkpoint,
                         args.output_dir, args.limit)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
