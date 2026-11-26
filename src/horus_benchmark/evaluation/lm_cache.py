"""Download pinned LM snapshots and materialize resumable, label-free scores.

The researcher executes this CLI on Abaporu. A shard is a JSONL journal plus a
completion manifest; a killed journal can resume individual IDs without making
hundreds of thousands of tiny files. Source text and labels are never written
to score journals. Full vocabulary logits remain in GPU memory only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd

from horus_benchmark.detectors.lm_probability import FrozenPair, load_config, sha256_file


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(temporary, path)


def download(config_path: Path, lock_path: Path, cache_dir: Path) -> Path:
    """Resolve all revisions before download; published locks never float."""
    from huggingface_hub import HfApi, snapshot_download
    cfg = load_config(config_path)
    if lock_path.exists():
        lock = _lock(config_path, lock_path)
    else:
        api = HfApi()
        entries = {}
        for key, entry in cfg["models"].items():
            commit = api.model_info(entry["repo_id"], revision=entry["revision"]).sha
            if not commit or len(commit) != 40:
                raise ValueError(f"Model has no resolved commit: {key}")
            entries[key] = {"repo_id": entry["repo_id"], "revision": commit}
        lock = {"schema_version": 1, "kind": "horus_frozen_lm_lock",
                "config_sha256": sha256_file(config_path), "models": entries}
        # Write the resolved commits first, so an interrupted download restarts
        # from exactly these revisions rather than newly resolved main branches.
        _write_json(lock_path, lock)
    for key, entry in lock["models"].items():
        repository_files = HfApi().list_repo_files(entry["repo_id"], revision=entry["revision"])
        has_safetensors = any(name.endswith(".safetensors") for name in repository_files)
        snapshot = Path(snapshot_download(
            repo_id=entry["repo_id"], revision=entry["revision"], cache_dir=str(cache_dir),
            allow_patterns=["*.json", "*.model", "*.txt", "*.safetensors", "*.bin", "merges.txt"],
            ignore_patterns=["optimizer*", "training*", "*.h5", "*.msgpack", "tf_model*", "flax_model*",
                             *(["*.bin"] if has_safetensors else [])]))
        # Prefer safetensors if present, but some historical GPT-J revisions
        # expose only PyTorch weights. Snapshot hashes record the actual choice.
        weights = sorted(snapshot.glob("*.safetensors")) or sorted(snapshot.glob("pytorch_model*.bin"))
        if not weights:
            raise ValueError(f"No model weights in pinned snapshot: {key}")
        entry["snapshot_path"] = str(snapshot)
        files = sorted(set(weights + list(snapshot.glob("*.json")) + list(snapshot.glob("*.model")) +
                           list(snapshot.glob("*.txt"))))
        entry["files"] = {path.name: sha256_file(path) for path in files}
        _write_json(lock_path, lock)
    lock["complete"] = True
    _write_json(lock_path, lock)
    return lock_path


def _lock(config_path: Path, lock_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("kind") != "horus_frozen_lm_lock" or
            lock.get("config_sha256") != sha256_file(config_path) or
            set(lock.get("models", {})) != set(cfg["models"])):
        raise ValueError("LM lock does not match the experiment configuration")
    for key, entry in lock["models"].items():
        revision = entry.get("revision", "")
        if (entry.get("repo_id") != cfg["models"][key]["repo_id"] or
                len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
            raise ValueError("Invalid pinned model identity")
    return lock


def _texts(path: Path, pilot_ids: Path | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["record_id", "content"])
    if frame.record_id.duplicated().any() or frame.record_id.isna().any() or frame.content.isna().any():
        raise ValueError("Text cache has duplicate IDs or missing content")
    if not frame.content.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Non-string text in input")
    frame = frame.sort_values("record_id", kind="stable").reset_index(drop=True)
    if pilot_ids is not None:
        selected = pd.read_csv(pilot_ids, keep_default_na=False)
        if "record_id" not in selected or selected.record_id.duplicated().any():
            raise ValueError("Pilot CSV requires unique record_id values")
        if not set(selected.record_id) <= set(frame.record_id):
            raise ValueError("Pilot references unknown IDs")
        frame = frame.loc[frame.record_id.isin(set(selected.record_id))].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No documents selected for probability extraction")
    return frame


def _identity(config_path: Path, lock_path: Path, texts_path: Path,
               pair: str, shard_size: int, pilot_ids: Path | None) -> dict[str, Any]:
    from horus_benchmark.detectors import lm_probability
    return {"implementation_sha256": hashlib.sha256((sha256_file(Path(__file__)) +
                                                     sha256_file(Path(lm_probability.__file__))).encode()).hexdigest(),
            "config_sha256": sha256_file(config_path), "lock_sha256": sha256_file(lock_path),
            "texts_sha256": sha256_file(texts_path), "pair": pair, "shard_size": shard_size,
            "pilot_ids_sha256": sha256_file(pilot_ids) if pilot_ids else None,
            "cache_code_sha256": sha256_file(Path(__file__)),
            "criterion_code_sha256": sha256_file(Path(__file__).parents[1] / "detectors" / "lm_probability.py")}


def _journal(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or not repair:
                raise ValueError(f"Corrupt LM journal: {path}")
            break
        if not line.endswith(b"\n") and repair:
            # A parsed but unterminated final record is kept and normalized.
            rows.append(row)
            break
        rows.append(row)
    if repair and (not raw.endswith(b"\n") or len(rows) != len(lines)):
        temporary = path.with_suffix(".repair.tmp")
        temporary.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
        os.replace(temporary, path)
    return rows


def score(config_path: Path, lock_path: Path, texts_path: Path, pair: str,
          output: Path, shard_index: int, shard_size: int,
          cache_dir: Path | None = None, pilot_ids: Path | None = None) -> Path:
    cfg, lock = load_config(config_path), _lock(config_path, lock_path)
    if lock.get("complete") is not True or pair not in cfg["pairs"]:
        raise ValueError("Complete download lock and known pair required")
    if shard_size < 1 or shard_index < 0:
        raise ValueError("Invalid LM shard bounds")
    frame = _texts(texts_path, pilot_ids)
    subset = frame.iloc[shard_index * shard_size:(shard_index + 1) * shard_size]
    if subset.empty:
        raise IndexError("LM shard index exceeds selected documents")
    identity = _identity(config_path, lock_path, texts_path, pair, shard_size, pilot_ids)
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "cache_identity.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Output cache belongs to another immutable scoring input")
    else:
        # Atomic creation also tolerates identical concurrent shard starts.
        _write_json(meta_path, identity)
    directory = output / "shards"
    directory.mkdir(exist_ok=True)
    journal = directory / f"{shard_index:05d}.jsonl"
    manifest = directory / f"{shard_index:05d}.json"
    observed = _journal(journal, repair=True)
    expected = {str(row.record_id): hashlib.sha256(row.content.encode("utf-8")).hexdigest()
                for row in subset.itertuples(index=False)}
    if (len({row["record_id"] for row in observed}) != len(observed) or
            any(row["record_id"] not in expected or row.get("content_hash") != expected[row["record_id"]]
                for row in observed)):
        raise ValueError("Resumed score journal has changed or duplicate document identities")
    completed = {row["record_id"] for row in observed}
    if manifest.exists():
        item = json.loads(manifest.read_text(encoding="utf-8"))
        if (item.get("identity") != identity or item.get("journal_sha256") != sha256_file(journal)
                or set(expected) != completed):
            raise ValueError("Completed LM shard changed")
        return manifest
    # Verify the files for this pair once per allocation, not per document.
    from huggingface_hub import snapshot_download
    for key in {cfg["pairs"][pair]["reference"], cfg["pairs"][pair]["scoring"]}:
        entry = lock["models"][key]
        snapshot = Path(snapshot_download(entry["repo_id"], revision=entry["revision"],
                                          cache_dir=str(cache_dir) if cache_dir else None,
                                          local_files_only=True))
        for name, digest in entry["files"].items():
            if sha256_file(snapshot / name) != digest:
                raise ValueError(f"Pinned model file changed: {key}/{name}")
    model = FrozenPair(cfg, lock, pair, cache_dir) if completed != set(expected) else None
    start = time.perf_counter()
    if model is not None and model.device == "cuda":
        model.torch.cuda.reset_peak_memory_stats()
    new_rows = []
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        for row in subset.itertuples(index=False):
            record_id = str(row.record_id)
            if record_id in completed:
                continue
            tick = time.perf_counter()
            result = model.score_text(row.content)
            item = {"record_id": record_id, "content_hash": expected[record_id],
                    "seconds": time.perf_counter() - tick, **result}
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            new_rows.append(item)
    combined = observed + new_rows
    if {row["record_id"] for row in combined} != set(expected):
        raise AssertionError("LM shard did not materialize every selected ID")
    payload = {"schema_version": 1, "kind": "horus_frozen_lm_score_shard",
               "identity": identity, "shard_index": shard_index, "rows": len(combined),
               "valid_rows": sum(row["status"] == "ok" for row in combined),
               "truncated_rows": sum(row["truncated"] for row in combined),
               "journal_sha256": sha256_file(journal),
               "elapsed_new_seconds": time.perf_counter() - start,
               "projected_107566_seconds": float(np.mean([row["seconds"] for row in combined]) * 107566),
               "new_rows": len(new_rows),
               "gpu_peak_allocated_bytes": int(model.torch.cuda.max_memory_allocated())
               if model is not None and model.device == "cuda" else 0,
               "gpu_name": model.torch.cuda.get_device_name()
               if model is not None and model.device == "cuda" else None,
               "gpu_total_memory_bytes": int(model.torch.cuda.get_device_properties(0).total_memory)
               if model is not None and model.device == "cuda" else None,
               "tokenizer": model.tokenizer_info if model is not None else {},
               "numpy": np.__version__,
               "software": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "huggingface-hub")}}
    _write_json(manifest, payload)
    return manifest


def check(config_path: Path, lock_path: Path, texts_path: Path, pair: str,
          output: Path, shard_size: int, pilot_ids: Path | None = None,
          merge: bool = False) -> dict[str, Any]:
    _lock(config_path, lock_path)
    frame = _texts(texts_path, pilot_ids)
    identity = _identity(config_path, lock_path, texts_path, pair, shard_size, pilot_ids)
    if json.loads((output / "cache_identity.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("LM cache identity changed")
    required = math.ceil(len(frame) / shard_size)
    missing, parts, manifests = [], [], {}
    expected = {str(row.record_id): hashlib.sha256(row.content.encode()).hexdigest()
                for row in frame.itertuples(index=False)}
    for index in range(required):
        manifest = output / "shards" / f"{index:05d}.json"
        journal = manifest.with_suffix(".jsonl")
        if not manifest.exists() or not journal.exists():
            missing.append(index)
            continue
        item = json.loads(manifest.read_text(encoding="utf-8"))
        rows = _journal(journal, repair=False)
        assigned = set(frame.iloc[index * shard_size:(index + 1) * shard_size].record_id.astype(str))
        if (item.get("identity") != identity or item.get("shard_index") != index or
                item.get("journal_sha256") != sha256_file(journal) or len(rows) != item.get("rows") or
                len({row["record_id"] for row in rows}) != len(rows) or
                {row["record_id"] for row in rows} != assigned or
                any(row.get("content_hash") != expected[row["record_id"]] for row in rows)):
            raise ValueError(f"Invalid LM cache shard: {index}")
        parts.extend(rows)
        manifests[str(index)] = sha256_file(manifest)
    result = {"complete": not missing, "planned_shards": required,
              "completed_shards": required - len(missing), "missing_shards": missing,
              "rows": len(parts), "valid_rows": sum(row["status"] == "ok" for row in parts)}
    if merge:
        if missing:
            raise ValueError(f"Cannot merge incomplete LM cache: {missing[:20]}")
        table = pd.DataFrame(parts).sort_values("record_id", kind="stable")
        path = output / "scores.parquet"
        if (output / 'manifest.json').exists():
            prior = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            if prior.get('identity') != identity or prior.get('scores_sha256') != sha256_file(path):
                raise ValueError('Existing merged LM cache changed')
            return prior
        temporary = output / 'scores.parquet.part'
        table.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        result.update(schema_version=1, kind="horus_frozen_lm_score_cache", identity=identity,
                      scores={"path": str(path), "sha256": sha256_file(path)}, shards=manifests,
                      scores_path=str(path), scores_sha256=sha256_file(path),
                      coverage_by_reason=table.groupby(["status", "reason"], dropna=False).size().reset_index(name="rows").to_dict("records"))
        _write_json(output / "manifest.json", result)
    return result


def load_scores(output: Path) -> pd.DataFrame:
    """Load a completed score table only after checking its published hash."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "horus_frozen_lm_score_cache" or manifest.get("complete") is not True:
        raise ValueError("A complete frozen LM cache is required")
    info = manifest["scores"]
    path = Path(info["path"])
    if not path.is_file() or sha256_file(path) != info["sha256"]:
        raise ValueError("Frozen LM score table changed")
    frame = pd.read_parquet(path)
    if (frame.record_id.duplicated().any() or len(frame) != manifest["rows"] or
            not set(frame.status) <= {"ok", "invalid"}):
        raise ValueError("Frozen LM score row identities or statuses are invalid")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download_cmd = sub.add_parser("download")
    download_cmd.add_argument("--config", type=Path, required=True)
    download_cmd.add_argument("--lock", type=Path, required=True)
    download_cmd.add_argument("--cache-dir", type=Path, required=True)
    for command in ("score", "check", "merge"):
        child = sub.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--lock", type=Path, required=True)
        child.add_argument("--texts", type=Path, required=True)
        child.add_argument("--pair", required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--shard-size", type=int, default=2048)
        child.add_argument("--pilot-ids", type=Path)
        if command == "score":
            child.add_argument("--shard-index", type=int, required=True)
            child.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.config, args.lock, args.cache_dir))
    elif args.command == "score":
        print(score(args.config, args.lock, args.texts, args.pair, args.output,
                    args.shard_index, args.shard_size, args.cache_dir, args.pilot_ids))
    else:
        result = check(args.config, args.lock, args.texts, args.pair, args.output,
                       args.shard_size, args.pilot_ids, merge=args.command == "merge")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
