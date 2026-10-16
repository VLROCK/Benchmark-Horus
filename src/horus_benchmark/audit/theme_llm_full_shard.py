"""Stdlib-only resolver for an indexed LLM shard on a Slurm array task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve(index_path: Path, shard_id: int, model_key: str) -> tuple[Path, Path]:
    index_path = index_path.expanduser().resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("task_kind") not in {"theme_full_first", "theme_full_second"}:
        raise ValueError("Not a full-corpus theme shard index")
    shards = index["shards"]
    if not 0 <= shard_id < len(shards) or shards[shard_id]["shard_id"] != shard_id:
        raise ValueError("Invalid Slurm array shard ID")
    if model_key not in {"gemma4_31b", "gpt_oss_120b"}:
        raise ValueError("Invalid model key")
    request = (index_path.parent / shards[shard_id]["requests"]).resolve()
    root = index_path.parent.resolve()
    if not request.is_relative_to(root / "requests") or not request.is_file():
        raise ValueError("Shard request path escapes the run directory or is absent")
    output = root / "inference" / model_key / f"shard_{shard_id:04d}"
    return request, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horus-theme-llm-full-shard")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args(argv)
    request, output = resolve(args.index, args.shard_id, args.model_key)
    print(request)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
