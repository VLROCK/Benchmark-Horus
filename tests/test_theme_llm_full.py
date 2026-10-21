"""Small end-to-end sharding and materialization test without GPU inference."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from horus_benchmark.audit.theme_common import ThemeContext
from horus_benchmark.audit.theme_llm_full import (
    finalize_full, prepare_full_first, prepare_full_second,
)
from horus_benchmark.audit.theme_llm_full_shard import resolve
from horus_benchmark.audit.theme_llm_worker import _read_existing, read_jsonl, read_requests
from horus_benchmark.data import DatasetCatalog


AREAS = ("academic", "informational", "journalistic", "juridical", "literary", "social")
MODELS = ("gemma4_31b", "gpt_oss_120b")


class ToyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return SimpleNamespace(ids=[0] * len(text.split()))


def _context(tmp_path: Path) -> ThemeContext:
    corpus = tmp_path / "all.parquet"
    records = []
    for area in AREAS:
        for index in range(2):
            records.append({
                "record_id": f"{area}-{index}", "label_name": "human",
                "broad_area": area,
                "source_name": "Corpus Carolina - Wikipédia" if area == "informational" else "source",
                "theme": "raw human theme", "subtheme": "unchanged", "generator": None,
                "split": ("train", "validation", "test")[len(records) % 3],
                "word_count": 4, "content": (
                    f"Texto humano {area} {index}.\u2028Continuação."
                    if area == "academic" and index == 0
                    else f"Texto humano {area} {index}."
                ),
            })
    for index in range(3):
        records.append({
            "record_id": f"synthetic-{index}", "label_name": "synthetic",
            "broad_area": "academic", "source_name": "generator",
            "theme": "Tema A", "subtheme": "unchanged", "generator": "test-model",
            "split": ("train", "validation", "test")[index],
            "word_count": 3, "content": "Texto sintético.",
        })
    pd.DataFrame(records).to_parquet(corpus, index=False)
    catalog = tmp_path / "catalog.yaml"
    resources = {"corpus": {"path": "all.parquet", "format": "parquet", "role": "corpus",
                             "expected_rows": 15}}
    for split in ("train", "validation", "test"):
        resources[split] = {"path": "all.parquet", "format": "parquet", "role": split,
                            "expected_rows": 5}
    resources["taxonomy"] = {"path": "taxonomy.csv", "format": "csv", "role": "taxonomy"}
    catalog.write_text(yaml.safe_dump({
        "dataset_id": "toy", "dataset_version": "v1", "workspace_root": ".",
        "resources": resources,
    }), encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.csv"
    taxonomy.write_text(
        "domain,theme\n" + "".join(f"{area},Tema A\n{area},Tema B\n" for area in AREAS),
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("broad_area,raw_theme,canonical_theme\n", encoding="utf-8")
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(yaml.safe_dump({
        "version": 2, "common": "Classifique o tema.",
        "areas": {area: {"name": area, "guidance": "Tema central.", "examples": []}
                  for area in AREAS},
    }, allow_unicode=True), encoding="utf-8")
    second = tmp_path / "second.yaml"
    second.write_text("version: 1\ncommon: Reavalie as hipóteses.\n", encoding="utf-8")
    return ThemeContext(
        config_path=tmp_path / "config.yaml",
        config={
            "experiment_id": "toy-full", "seed": 2026, "corpus_resource": "corpus",
            "train_resource": "train", "validation_resource": "validation",
            "test_resource": "test", "prompts": "prompts.yaml",
            "full": {
                "expected_corpus_rows": 15, "expected_human_rows": 12,
                "wikipedia_source_name": "Corpus Carolina - Wikipédia",
                "wikipedia_theme": "Wikipédia", "expected_wikipedia_rows": 2,
                "shard_size": 3, "second_prompt_file": "second.yaml",
            },
            "llm": {
                "input_policy": "shared_token_budget", "max_model_len": 1024,
                "max_output_tokens": 100, "chat_template_reserve_tokens": 20,
                "models": {"gemma4_31b": {"revision": "a" * 40},
                           "gpt_oss_120b": {"revision": "b" * 40}},
            },
        },
        catalog=DatasetCatalog.from_yaml(catalog),
        taxonomy={area: ("Tema A", "Tema B") for area in AREAS}, aliases={},
        taxonomy_path=taxonomy, mapping_path=mapping, output_root=tmp_path / "results",
    )


def _fake_outputs(index_path: Path, stage: str, context: ThemeContext) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for shard in index["shards"]:
        request_path, _ = resolve(index_path, shard["shard_id"], MODELS[0])
        requests, request_meta = read_requests(request_path)
        for model in MODELS:
            _, output_dir = resolve(index_path, shard["shard_id"], model)
            output_dir.mkdir(parents=True)
            (output_dir / "metadata.json").write_text(json.dumps({
                "model_key": model, "task_kind": index["task_kind"],
                "requests_sha256": request_meta["requests_sha256"],
                "manifest_sha256": index["manifest_sha256"],
                "prompts_sha256": index["prompts_sha256"],
                "taxonomy_sha256": index["taxonomy_sha256"],
                "llm_config": index["llm_config"],
                "checkpoint_revision": context.config["llm"]["models"][model]["revision"],
                "upstream_index_sha256": index["upstream_index_sha256"],
            }), encoding="utf-8")
            with (output_dir / "annotations.jsonl").open("w", encoding="utf-8") as stream:
                for request in requests:
                    record_id = request["record_id"]
                    if stage == "first":
                        if record_id.endswith("-0"):
                            theme = "Tema A"
                        elif record_id.startswith("academic"):
                            theme = "fora_taxonomia"
                        else:
                            theme = "Tema A" if model == MODELS[0] else "Tema B"
                    else:
                        if record_id.startswith("journalistic"):
                            theme = "Tema A"
                        elif record_id.startswith("juridical"):
                            theme = "fora_taxonomia"
                        else:
                            theme = "Tema A" if model == MODELS[0] else "Tema B"
                    stream.write(json.dumps({
                        "record_id": record_id, "broad_area": request["broad_area"],
                        "theme": theme, "status": "valid",
                        "input_truncated": request["input_truncated"],
                        "raw_output": ('{"theme":"Tema A"}\u2028'
                                       if stage == "first" and record_id == "academic-0"
                                       else json.dumps({"theme": theme})),
                        **({"alternative_themes": [], "rationale": "Tema central.",
                            "raw_output": json.dumps({
                                "theme": theme, "alternative_themes": [],
                                "rationale": "Tema central.",
                            })} if stage == "second" else {}),
                    }, ensure_ascii=False) + "\n")


def _replace_response(index_path: Path, model: str, record_id: str, raw: str) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for shard in index["shards"]:
        _, folder = resolve(index_path, shard["shard_id"], model)
        path = folder / "annotations.jsonl"
        records = read_jsonl(path)
        for record in records:
            if record["record_id"] == record_id:
                record.update(status="invalid", theme=None, raw_output=raw)
                path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                                encoding="utf-8")
                return
    raise AssertionError(f"Missing fake response for {record_id}")


def test_full_corpus_shards_second_pass_and_derived_parquet(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_full._shared_tokenizers",
        lambda _context: ({key: ToyTokenizer() for key in MODELS},
                          {key: f"fingerprint-{key}" for key in MODELS}),
    )
    first_index = prepare_full_first(context)
    first = json.loads(first_index.read_text(encoding="utf-8"))
    assert first["rows"] == 10 and len(first["shards"]) == 4
    assert first["truncated_rows"] == 0
    _fake_outputs(first_index, "first", context)
    second_index = prepare_full_second(context, first_index)
    second = json.loads(second_index.read_text(encoding="utf-8"))
    assert second["rows"] == 4 and len(second["shards"]) == 2
    _fake_outputs(second_index, "second", context)
    summary_path = finalize_full(context, first_index, second_index)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["corpus_rows"] == 15
    assert summary["human_rows"] == 12
    assert summary["second_requests"] == 4
    derived = pd.read_parquet(summary_path.parent / "all_with_themes.parquet")
    assert len(derived) == 15
    assert set(derived.loc[derived["source_name"] == "Corpus Carolina - Wikipédia", "theme"]) == {"Wikipédia"}
    assert set(derived.loc[derived["label_name"] == "synthetic", "theme"]) == {"Tema A"}
    assert set(derived.loc[derived["label_name"] == "human", "theme_original"]) == {"raw human theme"}
    assert derived.loc[derived["record_id"] == "journalistic-1", "theme"].item() == "Tema A"
    assert derived.loc[derived["record_id"] == "juridical-1", "theme"].item() == "fora_taxonomia"
    assert derived.loc[derived["record_id"] == "literary-1", "theme"].item() == "incerto"


def test_append_only_resume_removes_only_incomplete_last_line(tmp_path: Path) -> None:
    path = tmp_path / "annotations.jsonl"
    path.write_bytes(b'{"record_id":"a","status":"valid"}\n{"record_id":"b"')
    records = _read_existing(path, append_mode=True)
    assert set(records) == {"a"}
    assert path.read_bytes() == b'{"record_id":"a","status":"valid"}\n'


def test_invalid_json_is_audited_and_revisited_without_inventing_a_label(
    tmp_path: Path, monkeypatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_full._shared_tokenizers",
        lambda _context: ({key: ToyTokenizer() for key in MODELS},
                          {key: f"fingerprint-{key}" for key in MODELS}),
    )
    first_index = prepare_full_first(context)
    _fake_outputs(first_index, "first", context)
    _replace_response(first_index, MODELS[0], "academic-0", '{"theme":"tema a"}')
    _replace_response(first_index, MODELS[0], "academic-1", '{"theme":"Tema inventado"}')
    second_index = prepare_full_second(context, first_index)
    assert json.loads(second_index.read_text(encoding="utf-8"))["rows"] == 5
    _fake_outputs(second_index, "second", context)
    _replace_response(second_index, MODELS[0], "journalistic-1", "resposta sem JSON")
    summary_path = finalize_full(context, first_index, second_index)
    rows = pd.read_parquet(summary_path.parent / "human_theme_assignments.parquet")
    indexed = rows.set_index("record_id")
    assert indexed.loc["academic-0", "theme_final"] == "Tema A"
    assert indexed.loc["academic-0", "gemma4_31b_parse_status"] == "normalized_label"
    assert indexed.loc["academic-1", "theme_final"] == "incerto"
    assert indexed.loc["academic-1", "assignment_method"] == "parse_failure"
    assert indexed.loc["journalistic-1", "theme_final"] == "incerto"
    assert indexed.loc["journalistic-1", "second_gemma4_31b_parse_status"] == "parse_failure"
