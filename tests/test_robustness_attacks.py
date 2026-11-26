"""Small attack identity/protection checks; no corpus or GPU is consumed."""

import json

import pytest

from horus_benchmark.robustness.attacks import Edit, perturb, replay, sha256_text


def test_attacks_are_deterministic_replayable_and_class_independent() -> None:
    text = "Não também informação pessoas processo. " * 40
    for attack in ("upper_lower", "whitespace", "misspelling", "zero_width_space", "homoglyph"):
        first = perturb(text, "parent", attack, .05, seed=7)
        second = perturb(text, "parent", attack, .05, seed=7)
        assert first == second
        assert first["record_id"] == first["parent_record_id"] == "parent"
        assert first["num_edits"] == int(first["eligible_positions"] * .05)
        edits = [Edit(**item) for item in json.loads(first["edits_json"])]
        assert replay(text, edits) == first["content"]
        assert first["content_hash"] == sha256_text(first["content"])
        assert first["parent_content_hash"] == sha256_text(text)


def test_code_urls_emails_and_numbers_are_protected() -> None:
    protected = "`processo informação` https://example.com/processo a@processo.com 2026 "
    code = "```python\nprocesso = 'informação'\n```"
    text = protected + code + "\nprocesso informação também pessoas " * 30
    for attack in ("upper_lower", "whitespace", "misspelling", "zero_width_space", "homoglyph"):
        result = perturb(text, "parent", attack, 1.0)
        assert protected.strip() in result["content"]
        assert code in result["content"]
        assert "2026" in result["content"]


def test_sparse_doses_do_not_force_an_edit_on_short_texts() -> None:
    result = perturb("Uma frase curta.", "parent", "upper_lower", .01)
    assert result["status"] == "no_op"
    assert result["reason"] == "dose_rounds_to_zero"
    assert result["content"] == "Uma frase curta."
    assert result["effective_fraction"] == 0
    assert result["num_edits"] == 0


def test_paragraph_insertion_changes_one_central_boundary_only() -> None:
    text = "Primeira frase completa. Segunda frase completa. Terceira frase completa. Quarta frase completa."
    result = perturb(text, "parent", "insert_paragraphs", 1.0)
    assert result["status"] == "changed"
    assert result["num_edits"] == 1
    assert result["content"].count("\n\n") == 1
    assert result["content"].split() == text.split()
    assert result["dose_unit"] == "one_central_boundary"
    with pytest.raises(ValueError, match="exactly one"):
        perturb(text, "parent", "insert_paragraphs", .05)


def test_empty_or_ineligible_sources_remain_traceable() -> None:
    empty = perturb("", "parent", "homoglyph", .05)
    assert empty["status"] == "invalid"
    assert empty["reason"] == "empty_source"
    numbers = perturb("12345", "parent", "homoglyph", .05)
    assert numbers["status"] == "no_op"
    assert numbers["reason"] == "no_eligible_position"
    with pytest.raises(ValueError, match="mismatched"):
        replay("abc", [Edit(0, 1, "wrong", "x")])


def test_presentation_only_changes_preserve_token_sequence() -> None:
    text = "Uma frase clara. Outra frase curta. Terceira frase clara. Quarta frase curta."
    spaced = perturb(text, "id", "whitespace", 1.0)
    assert spaced["content"].split() == text.split()
    assert spaced["num_edits"] > 0


def test_prepare_links_only_frozen_test_union_and_preserves_source(tmp_path) -> None:
    import pandas as pd
    import yaml

    from horus_benchmark.evaluation.benchmark_runner import sha256_file
    from horus_benchmark.robustness.attacks import ATTACKS
    from horus_benchmark.robustness.prepare import check, prepare

    rows = []
    for index, label in enumerate(("human", "human", "synthetic", "synthetic")):
        content = "Uma frase completa. Outra frase completa. Terceira frase completa. Quarta frase completa. " * 3
        rows.append({"record_id": f"id-{index}", "label_name": label,
                     "probe_group_id": f"group-{index}", "broad_area": "social",
                     "source_name": "source" if label == "human" else None,
                     "generator": "generator" if label == "synthetic" else None,
                     "content": content, "content_hash": sha256_text(content)})
    texts = pd.DataFrame(rows)
    texts_path = tmp_path / "texts.parquet"
    texts.to_parquet(texts_path, index=False)
    original_hash = sha256_file(texts_path)
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    index_path = release_dir / "selected_index.parquet"
    texts.drop(columns="content").to_parquet(index_path, index=False)
    artifacts = {"selected_index": {"sha256": sha256_file(index_path)}}
    seeds = [20261105, 20261106, 20261107]
    for seed in seeds:
        frame = texts[["record_id", "probe_group_id"]].copy()
        frame["split"] = ["test", "development", "test", "development"]
        frame["fold"] = [-1, 0, -1, 0]
        path = release_dir / f"split_seed{seed}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[f"split_seed{seed}"] = {"sha256": sha256_file(path)}
    (release_dir / "release_manifest.json").write_text(json.dumps({
        "verified": True, "release_id": "horus-n2-iid-v1", "text_view": "n2",
        "split_seeds": seeds, "rows": 4, "artifacts": artifacts}), encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "schema_version": 1, "experiment_id": "horus-n2-query-free-robustness-v1",
        "release_id": "horus-n2-iid-v1", "split_seeds": seeds, "attack_seed": 20261125,
        "attacks": {name: [1.0] if name == "insert_paragraphs" else [.01, .05]
                    for name in ATTACKS}, "bootstrap_replicates": 2000,
        "bootstrap_confidence": .95, "fpr_targets": [.01, .05]}), encoding="utf-8")
    output = tmp_path / "attacks"
    prepare(cfg, release_dir, texts_path, output)
    report = check(output)
    assert report == {"verified": True, "test_union_rows": 2, "conditions": 11}
    assert sha256_file(texts_path) == original_hash
    variant = pd.read_parquet(output / "upper_lower__0p01.parquet")
    assert set(variant.record_id) == {"id-0", "id-2"}
    assert variant.record_id.eq(variant.parent_record_id).all()
