"""Focused checks for targeted LLM theme adjudication without model inference."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from horus_benchmark.audit.theme_common import sha256_file
from horus_benchmark.audit.theme_llm_second_pass import (
    evaluate_second_pass, prepared_second_requests,
)
from horus_benchmark.audit.theme_llm_worker import parse_second_pass, read_requests


AREAS = ("academic", "informational", "journalistic", "juridical", "literary", "social")


def test_second_pass_reuses_first_pass_model_settings() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    configs = root / "experiments/01_quality_audit/theme_agreement_600/configs"
    first = yaml.safe_load((configs / "theme_llm_600_long.yaml").read_text(encoding="utf-8"))
    second = yaml.safe_load((configs / "theme_llm_second_pass.yaml").read_text(encoding="utf-8"))
    assert first["llm"] == second["llm"]
    assert first["prompts"] == second["prompts"]


def test_second_pass_parser_checks_taxonomy_and_schema() -> None:
    allowed = {"Tema A", "Tema B", "incerto", "fora_taxonomia"}
    good = '{"theme":"incerto","alternative_themes":["Tema A","Tema B"],"rationale":"Ambos são plausíveis."}'
    assert parse_second_pass(good, allowed) == {
        "theme": "incerto", "alternative_themes": ["Tema A", "Tema B"],
        "rationale": "Ambos são plausíveis.",
    }
    assert parse_second_pass(good.replace("Tema A", "tema a"), allowed) == {
        "theme": "incerto", "alternative_themes": ["Tema A", "Tema B"],
        "rationale": "Ambos são plausíveis.",
    }
    assert parse_second_pass(good.replace("Tema B", "Tema inventado"), allowed) is None
    assert parse_second_pass(good.replace('"rationale":', '"confidence":0.9,"rationale":'), allowed) is None
    assert parse_second_pass("Acho que é Tema A", allowed) is None


def test_second_pass_only_targets_unresolved_non_wikipedia_and_preserves_others(
    tmp_path: Path, monkeypatch,
) -> None:
    class FakeTokenizer:
        def encode(self, value: str, add_special_tokens: bool = False):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[0] * len(value.split()))

    class FakeRun:
        def __init__(self, directory: Path):
            self.directory = directory
            directory.mkdir(parents=True, exist_ok=True)

        def write_result(self, name: str, payload: dict):
            path = self.directory / f"{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return path

    class FakeContext:
        def __init__(self):
            self.catalog = SimpleNamespace(workspace_root=tmp_path)
            self.taxonomy = {area: ("Tema A", "Tema B") for area in AREAS}
            self.taxonomy_path = tmp_path / "taxonomy.csv"
            self.taxonomy_path.write_text("toy taxonomy", encoding="utf-8")
            self.config = {
                "prompts": "first.yaml",
                "seed": 2026,
                "second_pass": {
                    "prompt_file": "second.yaml",
                    "wikipedia_source_name": "Corpus Carolina - Wikipédia",
                    "wikipedia_theme": "Wikipédia",
                    "expected_wikipedia_rows": 51,
                    "expected_target_rows": 170,
                },
                "llm": {
                    "input_policy": "shared_token_budget",
                    "max_model_len": 1024,
                    "max_output_tokens": 100,
                    "chat_template_reserve_tokens": 20,
                    "models": {
                        key: {"revision": char * 40}
                        for key, char in (("gemma4_31b", "a"), ("gpt_oss_120b", "b"))
                    },
                },
            }

        def new_run(self, stage: str, *, extra=None):
            return FakeRun(tmp_path / stage)

    first = {
        "version": 2,
        "common": "Classifique.",
        "areas": {
            area: {"name": area, "guidance": "Leia o tema central.", "examples": []}
            for area in AREAS
        },
    }
    import yaml

    (tmp_path / "first.yaml").write_text(yaml.safe_dump(first, allow_unicode=True), encoding="utf-8")
    (tmp_path / "second.yaml").write_text(
        "version: 1\ncommon: Faça uma segunda classificação.\n", encoding="utf-8",
    )
    context = FakeContext()
    records = []
    non_wikipedia_seen = 0
    for area in AREAS:
        for index in range(100):
            wikipedia = area == "informational" and index < 51
            if not wikipedia:
                non_wikipedia_seen += 1
            uncertain = not wikipedia and non_wikipedia_seen <= 170
            both_outside = not wikipedia and 171 <= non_wikipedia_seen <= 175
            left = "Tema A" if uncertain else ("fora_taxonomia" if both_outside else "Tema A")
            right = "Tema B" if uncertain else left
            if uncertain and non_wikipedia_seen == 170:
                left = right = "incerto"
            records.append({
                "record_id": f"{area}-{index:03}", "broad_area": area,
                "source_name": "Corpus Carolina - Wikipédia" if wikipedia else "outra fonte",
                "content": f"Texto humano {area} {index}",
                "gemma4_31b": "fora_taxonomia" if wikipedia else left,
                "gpt_oss_120b": "fora_taxonomia" if wikipedia else right,
                "initial_status": "outside" if wikipedia or both_outside else (
                    "uncertain" if uncertain else "named"
                ),
                "initial_theme": "fora_taxonomia" if wikipedia or both_outside else (
                    "incerto" if uncertain else "Tema A"
                ),
            })
    frame = pd.DataFrame(records)
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_second_pass._initial_rows",
        lambda *_args: frame.copy(),
    )
    monkeypatch.setattr(
        "horus_benchmark.audit.theme_llm_second_pass._shared_tokenizers",
        lambda *_args: (
            {key: FakeTokenizer() for key in ("gemma4_31b", "gpt_oss_120b")},
            {"gemma4_31b": "fingerprint-a", "gpt_oss_120b": "fingerprint-b"},
        ),
    )
    sample = tmp_path / "sample.parquet"
    first_gemma = tmp_path / "first_gemma.jsonl"
    first_gpt = tmp_path / "first_gpt.jsonl"
    for path in (sample, first_gemma, first_gpt):
        path.write_bytes(b"provenance")

    payload, metadata = prepared_second_requests(context, sample, first_gemma, first_gpt)
    assert metadata["rows"] == 170
    assert metadata["truncated_rows"] == 0
    requests_dir = tmp_path / "requests"
    requests_dir.mkdir()
    request_path = requests_dir / "requests.jsonl"
    request_path.write_bytes(payload)
    (requests_dir / "requests_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8",
    )
    requests, _ = read_requests(request_path)
    assert len(requests) == 170
    assert any(row["candidate_labels"] == ["incerto"] for row in requests)
    assert all("outra fonte" not in row["prompt"] for row in requests)
    assert all("Wikipédia" not in row["prompt"] for row in requests)

    outputs = []
    for key in ("gemma4_31b", "gpt_oss_120b"):
        directory = tmp_path / key
        directory.mkdir()
        worker_meta = {
            "model_key": key, "task_kind": "theme_second_pass",
            "requests_sha256": metadata["requests_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "prompts_sha256": metadata["prompts_sha256"],
            "taxonomy_sha256": metadata["taxonomy_sha256"],
            "upstream_gemma_sha256": metadata["upstream_gemma_sha256"],
            "upstream_gpt_sha256": metadata["upstream_gpt_sha256"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "llm_config": metadata["llm_config"],
            "checkpoint_revision": metadata["model_revisions"][key],
        }
        (directory / "metadata.json").write_text(json.dumps(worker_meta), encoding="utf-8")
        output = directory / "annotations.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(requests):
                theme = "Tema A" if index < 70 or key == "gemma4_31b" else "Tema B"
                decision = {"theme": theme, "alternative_themes": [], "rationale": "Tema central."}
                record = {
                    "record_id": row["record_id"], "broad_area": row["broad_area"],
                    "theme": theme, "alternative_themes": [], "rationale": "Tema central.",
                    "raw_output": json.dumps(decision), "status": "valid",
                    "input_truncated": False,
                }
                stream.write(json.dumps(record) + "\n")
        outputs.append(output)

    report_path = evaluate_second_pass(
        context, sample, first_gemma, first_gpt, *outputs,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["wikipedia_source_bucket"] == 51
    assert report["second_pass_targets"] == 170
    assert report["second_pass_new_named"] == 70
    assert report["remaining_unresolved"] == 100
    result_rows = pd.read_csv(report_path.parent / "second_pass_rows.csv")
    wikipedia_rows = result_rows.loc[result_rows["source_name"] == "Corpus Carolina - Wikipédia"]
    assert set(wikipedia_rows["final_theme"]) == {"Wikipédia"}
    assert set(wikipedia_rows["assignment_method"]) == {"source_rule"}
    assert (result_rows["assignment_method"] == "first_pass_outside").sum() == 5
