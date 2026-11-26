"""Frozen probability criteria with explicit upstream numerical conventions.

Binoculars follows ahans30/Binoculars (BSD-3-Clause): mean shifted performer
NLL divided by unshifted observer-to-performer cross entropy. Fast-DetectGPT
follows baoguangsheng/fast-detect-gpt (MIT), analytic discrepancy. These are
method reimplementations; no threshold or corpus label enters these functions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BINOCULARS_PUBLISHED_THRESHOLDS = {"accuracy": 0.9015310749276843,
                                   "low_fpr": 0.8536432310785527}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(cfg, dict) or cfg.get("schema_version") != 1 or
            cfg.get("experiment_id") != "horus-frozen-lm-pairs-v1"):
        raise ValueError("Unexpected frozen LM configuration")
    if int(cfg.get("max_tokens", 0)) != 512 or cfg.get("statistics_dtype") != "float32":
        raise ValueError("Unreviewed probability context or statistics precision")
    if cfg.get("device") not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    for pair in cfg["pairs"].values():
        if pair["reference"] not in cfg["models"] or pair["scoring"] not in cfg["models"]:
            raise ValueError("Unknown model in LM pair")
    return cfg


def _log_softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    largest = np.max(x, axis=-1, keepdims=True)
    return x - largest - np.log(np.exp(x - largest).sum(axis=-1, keepdims=True))


def analytic_discrepancy(reference_logits: np.ndarray, scoring_logits: np.ndarray,
                         labels: np.ndarray) -> float:
    """Pure NumPy oracle for target-aligned [positions, vocabulary] logits."""
    reference = np.asarray(reference_logits)
    scoring = np.asarray(scoring_logits)
    targets = np.asarray(labels, dtype=int)
    if (reference.shape != scoring.shape or reference.ndim != 2 or
            len(targets) != len(scoring) or len(targets) < 1 or
            targets.min() < 0 or targets.max() >= scoring.shape[1]):
        raise ValueError("Aligned reference/scoring logits and target IDs required")
    logp = _log_softmax(scoring)
    q = np.exp(_log_softmax(reference))
    expected = (q * logp).sum(axis=-1)
    variance = (q * logp ** 2).sum(axis=-1) - expected ** 2
    total = float(variance.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Zero or invalid conditional variance")
    return float((logp[np.arange(len(targets)), targets] - expected).sum() / np.sqrt(total))


def binoculars_ratio(observer_logits: np.ndarray, performer_logits: np.ndarray,
                      input_ids: np.ndarray, pad_token_id: int | None = None) -> float:
    """Single unpadded sequence oracle, matching the upstream unshifted denominator."""
    observer, performer = np.asarray(observer_logits), np.asarray(performer_logits)
    ids = np.asarray(input_ids, dtype=int)
    if observer.shape != performer.shape or observer.ndim != 2 or len(ids) != len(observer) or len(ids) < 2:
        raise ValueError("Two aligned full-sequence logits and at least two tokens required")
    logp = _log_softmax(performer)
    nll = -logp[:-1][np.arange(len(ids) - 1), ids[1:]].mean()
    cross = -(np.exp(_log_softmax(observer)) * logp).sum(axis=-1)
    mask = np.ones(len(ids), dtype=bool) if pad_token_id is None else ids != pad_token_id
    if not mask.any() or not np.isfinite(cross[mask]).all() or cross[mask].mean() <= 0:
        raise ValueError("Invalid Binoculars cross entropy")
    return float(nll / cross[mask].mean())


def assert_tokenizer_compatibility(left: Any, right: Any,
                                   common_vocab_size: int | None = None) -> dict[str, Any]:
    """Reject equal-size but differently indexed vocabularies or segmenters."""
    left_vocab, right_vocab = left.get_vocab(), right.get_vocab()
    excluded = [0, 0]
    if left_vocab != right_vocab:
        if common_vocab_size is None:
            raise ValueError("Pair tokenizers have different token-to-ID mappings")
        # GPT-J explicitly adds extra tokens for its padded head. Validate the
        # complete shared prefix rather than silently truncating arbitrary
        # incompatible vocabularies. Each scored document is checked again.
        shared_left = {token: index for token, index in left_vocab.items() if index < common_vocab_size}
        shared_right = {token: index for token, index in right_vocab.items() if index < common_vocab_size}
        if shared_left != shared_right:
            raise ValueError("Pair tokenizers have different token-to-ID mappings in their shared prefix")
        excluded = [len(left_vocab) - len(shared_left), len(right_vocab) - len(shared_right)]
    # Model-specific chat templates differ intentionally. The raw-text encoding
    # backend, vocabulary, special token addition and normalization must agree.
    probes = ["", "Olá, ação e coração!", "Texto\n\ncom 123 e código <p>.",
              "  espaços\tduplos  ", "😀 café 𝛼", "<|endoftext|>"]
    for probe in probes:
        for special in (False, True):
            if left.encode(probe, add_special_tokens=special) != right.encode(probe, add_special_tokens=special):
                raise ValueError("Pair tokenizers encode raw text differently")
    vocabulary = json.dumps(left.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
    return {"vocabulary_sha256": hashlib.sha256(vocabulary).hexdigest(),
            "vocabulary_entries": len(left.get_vocab()), "probes": len(probes),
            "excluded_padded_tokenizer_entries": excluded}


class FrozenPair:
    """Load a pair once and score texts, preserving native per-model token IDs.

    Models are local, revision-pinned, unquantized, eval mode, and the caller
    supplies no labels. Batch one avoids upstream EOS-padding ambiguities.
    """

    def __init__(self, cfg: dict[str, Any], lock: dict[str, Any], pair_name: str,
                 cache_dir: str | Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.cfg, self.pair_name = cfg, pair_name
        self.pair = cfg["pairs"][pair_name]
        self.device = cfg["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA allocation required for frozen LM scoring")
        self.models, tokenizers = [], []
        for key in (self.pair["reference"], self.pair["scoring"]):
            entry = lock["models"][key]
            options = {"revision": entry["revision"], "local_files_only": True,
                       "trust_remote_code": False, "cache_dir": str(cache_dir) if cache_dir else None}
            tokenizers.append(AutoTokenizer.from_pretrained(entry["repo_id"], **options))
            model = AutoModelForCausalLM.from_pretrained(
                entry["repo_id"], torch_dtype=getattr(torch, cfg["models"][key]["dtype"]),
                **options).to(self.device).eval()
            self.models.append(model)
        widths = [int(model.config.vocab_size) for model in self.models]
        self.output_vocab_size = min(widths)
        padded_fastdetect = widths[0] != widths[1] and "binoculars" not in self.pair["methods"]
        self.tokenizer_info = assert_tokenizer_compatibility(
            *tokenizers, common_vocab_size=self.output_vocab_size if padded_fastdetect else None)
        self.tokenizer, self.second_tokenizer = tokenizers
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        # GPT-J uses a padded output head (50400) whereas GPT-Neo uses 50257.
        # The official Fast-DetectGPT implementation restricts both heads to
        # their common prefix. We permit this only AFTER exact shared-prefix
        # tokenizer-map equality and check each document's IDs lie in it.
        # Binoculars retains identical full output dimensions as upstream.
        if widths[0] != widths[1] and "binoculars" in self.pair["methods"]:
            raise ValueError("Binoculars requires identical output vocabulary widths")
        self.tokenizer_info["output_vocab_widths"] = widths
        self.tokenizer_info["common_output_vocab_width"] = self.output_vocab_size

    def score_text(self, text: str) -> dict[str, Any]:
        torch = self.torch
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        observed = ids[:int(self.cfg["max_tokens"])]
        result: dict[str, Any] = {"status": "ok", "reason": "", "full_tokens": len(ids),
                                  "observed_tokens": len(observed), "truncated": len(ids) > len(observed),
                                  "full_chars": len(text), "observed_chars": None,
                                  "observed_words": None,
                                  "token_ids": observed, "weak_logprobs_0": [], "weak_logprobs_1": [],
                                  "weak_token_strings": [],
                                  "fastdetect_score": None, "binoculars_raw": None,
                                  "binoculars_score": None}
        if getattr(self.tokenizer, "is_fast", False):
            offsets = self.tokenizer(text, add_special_tokens=True,
                                     return_offsets_mapping=True)["offset_mapping"][:len(observed)]
            retained_end = max((int(end) for start, end in offsets if end > start), default=0)
            result["observed_chars"] = retained_end
            result["observed_words"] = len(text[:retained_end].split())
        if len(observed) < 2:
            return {**result, "status": "invalid", "reason": "fewer_than_two_tokens"}
        other = self.second_tokenizer.encode(text, add_special_tokens=True)[:int(self.cfg["max_tokens"])]
        if other != observed:
            return {**result, "status": "invalid", "reason": "per_document_tokenizer_alignment"}
        if max(observed) >= getattr(self, "output_vocab_size", len(self.tokenizer.get_vocab())):
            return {**result, "status": "invalid", "reason": "token_outside_shared_output_vocabulary"}
        inputs = torch.tensor([observed], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            first = self.models[0](input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0].float()
            second = self.models[1](input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[0].float()
            if first.shape[-1] != second.shape[-1]:
                first = first[:, :self.output_vocab_size]
                second = second[:, :self.output_vocab_size]
            if not bool(torch.isfinite(first).all() & torch.isfinite(second).all()):
                return {**result, "status": "invalid", "reason": "nonfinite_model_logits"}
            lp0 = torch.log_softmax(first, dim=-1)
            lp1 = torch.log_softmax(second, dim=-1)
            target = inputs[0, 1:]
            positions = torch.arange(len(target), device=self.device)
            actual0, actual1 = lp0[:-1][positions, target], lp1[:-1][positions, target]
            if "weak_logprobs" in self.pair["methods"]:
                result["weak_logprobs_0"] = actual0.cpu().tolist()
                result["weak_logprobs_1"] = actual1.cpu().tolist()
                result["weak_token_strings"] = self.tokenizer.convert_ids_to_tokens(observed[1:])
            if "fastdetect" in self.pair["methods"]:
                q = lp0[:-1].exp()
                mean = (q * lp1[:-1]).sum(dim=-1)
                variance = (q * lp1[:-1].square()).sum(dim=-1) - mean.square()
                total = variance.sum()
                if not bool(torch.isfinite(total)) or float(total) <= 0:
                    result.update(status="invalid", reason="invalid_conditional_variance")
                else:
                    value = float(((actual1 - mean).sum() / total.sqrt()).cpu())
                    if not np.isfinite(value):
                        result.update(status="invalid", reason="nonfinite_fastdetect")
                    else:
                        result["fastdetect_score"] = value
            if "binoculars" in self.pair["methods"]:
                mask = inputs[0] != self.pad_token_id if self.pad_token_id is not None else torch.ones_like(inputs[0], dtype=torch.bool)
                if not bool(mask.any()):
                    result.update(status="invalid", reason="empty_binoculars_denominator")
                else:
                    denominator = -(lp0.exp() * lp1).sum(dim=-1)[mask].mean()
                    raw = float((-actual1.mean() / denominator).cpu())
                    if not np.isfinite(raw):
                        result.update(status="invalid", reason="nonfinite_binoculars")
                    else:
                        result.update(binoculars_raw=raw, binoculars_score=-raw)
        return result
