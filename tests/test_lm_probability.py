"""Probability identities and tokenizer alignment; no checkpoints required."""

from __future__ import annotations

import math

import numpy as np
import pytest

from horus_benchmark.detectors.lm_probability import (
    FrozenPair, analytic_discrepancy, assert_tokenizer_compatibility, binoculars_ratio,
)
from horus_benchmark.evaluation.lm_cache import _journal


def test_analytic_discrepancy_matches_closed_form() -> None:
    reference = np.log([[.5, .5], [.5, .5]])
    scoring = np.log([[.25, .75], [.25, .75]])
    # Each target contributes half log(3), and its conditional variance is
    # one quarter log(3)^2. Summation gives sqrt(2).
    assert analytic_discrepancy(reference, scoring, np.array([1, 1])) == pytest.approx(math.sqrt(2))
    with pytest.raises(ValueError, match="variance"):
        analytic_discrepancy(np.zeros((2, 2)), np.zeros((2, 2)), np.array([0, 1]))


def test_binoculars_uses_shifted_nll_and_unshifted_cross_entropy() -> None:
    observer = np.log([[.5, .5], [.5, .5], [.1, .9]])
    performer = np.log([[.25, .75], [.25, .75], [.8, .2]])
    expected_nll = -math.log(.75)
    expected_cross = -(math.log(.25) + math.log(.75) + .1 * math.log(.8) + .9 * math.log(.2)) / 3
    assert binoculars_ratio(observer, performer, np.array([0, 1, 1])) == pytest.approx(expected_nll / expected_cross)
    # Excluding EOS/padding applies only to the upstream denominator here.
    assert binoculars_ratio(observer, performer, np.array([0, 0, 1]), pad_token_id=1) == pytest.approx(1.)


class _Tokenizer:
    pad_token_id = None
    eos_token_id = None

    def __init__(self, swapped=False):
        self.swapped = swapped

    def get_vocab(self):
        return {"a": 1, "b": 0} if self.swapped else {"a": 0, "b": 1}

    def encode(self, text, add_special_tokens=True):
        return [0, 1, 1]


def test_equal_vocabulary_size_is_not_tokenizer_compatibility() -> None:
    with pytest.raises(ValueError, match="token-to-ID"):
        assert_tokenizer_compatibility(_Tokenizer(), _Tokenizer(swapped=True))
    assert assert_tokenizer_compatibility(_Tokenizer(), _Tokenizer())["vocabulary_entries"] == 2


def test_padded_head_exception_requires_exact_shared_token_mapping() -> None:
    class Padded(_Tokenizer):
        def get_vocab(self):
            return {"a": 0, "b": 1, "<extra_padding>": 2}
    result = assert_tokenizer_compatibility(Padded(), _Tokenizer(), common_vocab_size=2)
    assert result["excluded_padded_tokenizer_entries"] == [1, 0]
    with pytest.raises(ValueError, match="shared prefix"):
        assert_tokenizer_compatibility(Padded(), _Tokenizer(swapped=True), common_vocab_size=2)


def test_actual_torch_scoring_matches_numpy_oracles() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Model:
        def __init__(self, probabilities):
            self.logits = torch.tensor(np.log(probabilities), dtype=torch.float32)[None]

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=self.logits)

    model = FrozenPair.__new__(FrozenPair)
    model.torch, model.device = torch, "cpu"
    model.cfg = {"max_tokens": 512}
    model.pair = {"methods": ["fastdetect", "binoculars"]}
    model.tokenizer = model.second_tokenizer = _Tokenizer()
    model.pad_token_id = None
    model.models = [Model([[.5, .5]] * 3), Model([[.25, .75]] * 3)]
    result = model.score_text("Texto")
    assert result["status"] == "ok"
    assert result["fastdetect_score"] == pytest.approx(math.sqrt(2), rel=1e-5)
    expected = -math.log(.75) / (-.5 * (math.log(.25) + math.log(.75)))
    assert result["binoculars_raw"] == pytest.approx(expected, rel=1e-5)
    assert result["binoculars_score"] == -result["binoculars_raw"]


def test_interrupted_journal_repairs_only_last_partial_record(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"record_id":"ok"}\n{"record_id":')
    assert _journal(path, repair=True) == [{"record_id": "ok"}]
    assert path.read_bytes() == b'{"record_id": "ok"}\n'
    path.write_bytes(b'bad\n{"record_id":"ok"}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        _journal(path, repair=True)
