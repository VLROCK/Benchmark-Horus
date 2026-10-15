"""Resumable local vLLM inference for the shared theme sample."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from horus_benchmark.audit.theme_common import ThemeContext, sha256_file
from horus_benchmark.audit.theme_llm_600 import (
    SPECIAL, checked_manifest, prompt_config, render_prompt,
)


def parse_theme(response: str, allowed: set[str]) -> str | None:
    """Accept a JSON object only; never infer a label from free-form prose."""
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
    context: ThemeContext, sample_path: Path, model_key: str,
    checkpoint: Path, output_dir: Path, limit: int | None = None,
) -> Path:
    """Run one model in its own validated environment and resume valid IDs."""
    model_cfg = context.config["llm"]["models"].get(model_key)
    if model_cfg is None:
        raise ValueError(f"Unknown local model key: {model_key}")
    checkpoint = checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file() or not re.fullmatch(r"[0-9a-f]{40}", checkpoint.name):
        raise ValueError("Use a complete local Hugging Face snapshot with a 40-character revision")
    sample = checked_manifest(context, sample_path)
    prompt_path, prompts = prompt_config(context)
    sample = sample.sort_values(["broad_area", "record_id"]).reset_index(drop=True)
    if limit is not None:
        if not 1 <= limit <= 600:
            raise ValueError("--limit must be between 1 and 600")
        # A smoke touches every broad area before limiting the total.
        sample = sample.groupby("broad_area", sort=True, group_keys=False).head(
            max(1, limit // 6)
        ).head(limit)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"
    output_path = output_dir / "annotations.jsonl"
    identity = {
        "model_key": model_key,
        "model_repo": model_cfg["hf_repo"],
        "checkpoint_revision": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "manifest_sha256": sha256_file(sample_path),
        "prompts_sha256": sha256_file(prompt_path),
        "taxonomy_sha256": sha256_file(context.taxonomy_path),
        "llm_config": context.config["llm"],
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
    records = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["record_id"]] = record
    all_order = checked_manifest(context, sample_path).sort_values(
        ["broad_area", "record_id"]
    )["record_id"].tolist()
    candidates = [
        row for row in sample.to_dict(orient="records")
        if records.get(row["record_id"], {}).get("status") != "valid"
    ]
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
    runtime_path = output_dir / "runtime.json"
    runtime_path.write_text(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    llm_cfg = context.config["llm"]
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
    max_chars = int(llm_cfg["max_input_characters"])
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        conversations = []
        for row in batch:
            content = row["content"][:max_chars]
            conversations.append([{
                "role": "user", "content": render_prompt(
                    context, prompts, row["broad_area"], content,
                ),
            }])
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
            allowed = set(context.taxonomy[row["broad_area"]]) | SPECIAL
            theme = parse_theme(raw, allowed)
            records[row["record_id"]] = {
                "record_id": row["record_id"],
                "broad_area": row["broad_area"],
                "theme": theme,
                "status": "valid" if theme is not None else "invalid",
                "raw_output": raw,
                "input_truncated": len(row["content"]) > max_chars,
                "input_characters_used": min(len(row["content"]), max_chars),
                "output_tokens": len(generated.token_ids),
                "finish_reason": generated.finish_reason,
            }
        _save_jsonl(output_path, records, all_order)
        valid = sum(record["status"] == "valid" for record in records.values())
        print(f"{model_key}: {len(records)}/600 responses, {valid} valid", file=sys.stderr, flush=True)
    return output_path
