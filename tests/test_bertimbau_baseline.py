from __future__ import annotations

from dataclasses import replace

import pytest

from horus_benchmark.detectors.bertimbau_baseline import (
    BertimbauConfig, _schedule_steps, _token_coverage, _validate_inputs,
)


def test_bertimbau_config_requires_pinned_revision() -> None:
    config = BertimbauConfig.from_mapping({"model": {"max_length": 512}})
    assert len(config.revision) == 40
    with pytest.raises(ValueError, match="pinned"):
        replace(config, revision="main").validate()


def test_bertimbau_gradient_accumulation_counts_partial_batches() -> None:
    config = BertimbauConfig(train_batch_size=16, gradient_accumulation_steps=2,
                              epochs=3, warmup_ratio=.1)
    # 65 rows = 5 microbatches = 3 optimizer steps per epoch.
    assert _schedule_steps(65, config) == (9, 1)


def test_bertimbau_rejects_missing_class_and_empty_text() -> None:
    with pytest.raises(ValueError, match="Both human"):
        _validate_inputs(["a", "b"], [0, 0], ["c", "d"], [0, 1], ["e"])
    with pytest.raises(ValueError, match="Empty"):
        _validate_inputs(["a", "b"], [0, 1], ["c", "d"], [0, 1], [" "])


class _FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool) -> int:
        assert not pair
        return 2

    def __call__(self, texts: list[str], **kwargs):
        assert kwargs["truncation"] is False
        return {"length": [len(text.split()) for text in texts]}


def test_bertimbau_token_coverage_counts_pretruncation_tokens() -> None:
    coverage = _token_coverage(["a b", "a b c d e f", "a"], _FakeTokenizer(), 6)
    assert coverage["max_content_tokens"] == 4
    assert coverage["truncated_rows"] == 1
    assert coverage["truncated_rate"] == pytest.approx(1 / 3)
    assert coverage["retained_token_fraction"] == pytest.approx(7 / 9)
