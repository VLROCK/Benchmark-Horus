"""Binary BERTimbau baseline for frozen Hórus text-only evaluations.

The caller supplies already partitioned *N2 text*, never provenance metadata. All
tokenizer/model downloads are disabled: the pinned revision must be cached before
the Slurm job. No detector threshold is selected here; the evaluation layer owns
validation-only threshold calibration and uncertainty estimates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np


_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BertimbauConfig:
    name: str = "neuralmind/bert-base-portuguese-cased"
    revision: str = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
    max_length: int = 512
    learning_rate: float = 2e-5
    epochs: int = 3
    train_batch_size: int = 16
    inference_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    device: str = "cuda"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BertimbauConfig":
        """Accept either a flattened mapping or a YAML mapping with ``model``."""
        model = value.get("model", value)
        if not isinstance(model, Mapping):
            raise ValueError("BERTimbau model configuration must be a mapping")
        unknown = set(model) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown BERTimbau options: {sorted(unknown)}")
        config = cls(**{key: model[key] for key in cls.__dataclass_fields__ if key in model})
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name or not _COMMIT.fullmatch(self.revision):
            raise ValueError("BERTimbau requires a name and pinned 40-character commit revision")
        if self.max_length < 4 or self.max_length > 512:
            raise ValueError("BERTimbau max_length must lie in [4, 512]")
        if min(self.epochs, self.train_batch_size, self.inference_batch_size,
               self.gradient_accumulation_steps) < 1:
            raise ValueError("Epochs, batch sizes and accumulation must be positive")
        if (self.learning_rate <= 0 or self.weight_decay < 0 or
                not 0 <= self.warmup_ratio < 1 or self.max_grad_norm <= 0):
            raise ValueError("Invalid optimizer or warmup setting")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("BERTimbau device must be cuda or cpu")


def _validate_inputs(
    train_texts: Sequence[str], train_labels: Sequence[int],
    validation_texts: Sequence[str], validation_labels: Sequence[int],
    test_texts: Sequence[str],
) -> None:
    if not train_texts or not validation_texts or not test_texts:
        raise ValueError("Train, validation and test must all be nonempty")
    if len(train_texts) != len(train_labels) or len(validation_texts) != len(validation_labels):
        raise ValueError("Text and label lengths differ")
    if set(int(value) for value in train_labels) != {0, 1}:
        raise ValueError("Both human=0 and synthetic=1 must appear in training")
    if set(int(value) for value in validation_labels) != {0, 1}:
        raise ValueError("Both classes must appear in validation")
    if any(value not in (0, 1) for value in [*train_labels, *validation_labels]):
        raise ValueError("Only binary labels 0/1 are supported")
    if any(not isinstance(text, str) or not text.strip()
           for text in [*train_texts, *validation_texts, *test_texts]):
        raise ValueError("Empty or non-string text in BERTimbau input")


def _schedule_steps(n_rows: int, config: BertimbauConfig) -> tuple[int, int]:
    """Optimizer steps and linearly warmed steps, accounting for partial batches."""
    microbatches = math.ceil(n_rows / config.train_batch_size)
    steps = math.ceil(microbatches / config.gradient_accumulation_steps) * config.epochs
    warmup = math.ceil(steps * config.warmup_ratio)
    return steps, warmup


def _token_coverage(texts: Sequence[str], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Measure full token length in bounded batches, before model truncation."""
    special = int(tokenizer.num_special_tokens_to_add(pair=False))
    available = max_length - special
    if available < 1:
        raise ValueError("max_length leaves no room for content tokens")
    lengths: list[int] = []
    for offset in range(0, len(texts), 64):
        encoded = tokenizer(
            list(texts[offset:offset + 64]), add_special_tokens=False,
            truncation=False, padding=False, return_length=True,
        )
        lengths.extend(int(length) for length in encoded["length"])
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "rows": len(texts),
        "max_content_tokens": available,
        "truncated_rows": int((values > available).sum()),
        "truncated_rate": float(np.mean(values > available)),
        "retained_token_fraction": float(np.minimum(values, available).sum() / values.sum())
        if values.sum() else 1.0,
        "median_content_tokens": float(np.median(values)),
        "p95_content_tokens": float(np.quantile(values, .95)),
        "max_content_tokens_observed": int(values.max()),
    }


def _seed_all(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loader(
    texts: Sequence[str], labels: Sequence[int] | None, tokenizer: Any,
    batch_size: int, max_length: int, torch: Any, *, seed: int | None,
) -> Any:
    from torch.utils.data import DataLoader

    rows = [(text, None if labels is None else int(labels[index]))
            for index, text in enumerate(texts)]

    def collate(items: list[tuple[str, int | None]]) -> dict[str, Any]:
        encoded = tokenizer(
            [text for text, _ in items], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        if labels is not None:
            encoded["labels"] = torch.tensor([label for _, label in items], dtype=torch.long)
        return encoded

    generator = torch.Generator()
    generator.manual_seed(seed if seed is not None else 0)
    return DataLoader(
        rows, batch_size=batch_size, shuffle=seed is not None,
        collate_fn=collate, generator=generator, num_workers=0,
    )


def _predict(model: Any, batches: Any, device: Any, torch: Any,
             *, class_weights: Any | None = None) -> tuple[np.ndarray, float | None]:
    import torch.nn.functional as nnf

    model.eval()
    scores: list[float] = []
    total_loss = 0.0
    labeled = 0
    with torch.no_grad():
        for batch in batches:
            labels = batch.pop("labels", None)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            if labels is not None:
                loss = nnf.cross_entropy(
                    logits, labels.to(device), weight=class_weights, reduction="sum",
                )
                total_loss += float(loss.detach().cpu())
                labeled += len(labels)
    return np.asarray(scores, dtype=np.float64), total_loss / labeled if labeled else None


def train_and_score(
    train_texts: Sequence[str], train_labels: Sequence[int],
    validation_texts: Sequence[str], validation_labels: Sequence[int],
    test_texts: Sequence[str], config: BertimbauConfig | Mapping[str, Any],
    output_dir: str | Path, seed: int,
) -> dict[str, Any]:
    """Fit on train, choose epoch on validation loss, and score validation/test.

    Returns ``validation_scores`` and ``test_scores`` in the caller's input
    order, plus ``metadata``. The model is saved to ``output_dir/model``. This
    function intentionally does not inspect test labels, tune thresholds or
    write any source text to disk.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as nnf
    import transformers
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = config if isinstance(config, BertimbauConfig) else BertimbauConfig.from_mapping(config)
    cfg.validate()
    # Convert pandas Series/NumPy arrays to stable positional sequences.
    train_texts, validation_texts, test_texts = (
        list(train_texts), list(validation_texts), list(test_texts))
    train_labels, validation_labels = list(train_labels), list(validation_labels)
    _validate_inputs(train_texts, train_labels, validation_texts, validation_labels, test_texts)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BERTimbau baseline requires the requested CUDA allocation")
    _seed_all(int(seed), torch)
    device = torch.device(cfg.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model"
    if checkpoint.exists() or (output / "training_metadata.json").exists():
        raise FileExistsError(f"BERTimbau checkpoint already exists: {output}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name, revision=cfg.revision, local_files_only=True,
        trust_remote_code=False,
    )
    if cfg.max_length > tokenizer.model_max_length:
        raise ValueError("Configured max_length exceeds tokenizer model limit")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.name, revision=cfg.revision, local_files_only=True,
        trust_remote_code=False, num_labels=2,
        id2label={0: "human", 1: "synthetic"},
        label2id={"human": 0, "synthetic": 1},
    ).to(device)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision and resolved_revision != cfg.revision:
        raise ValueError("Cached BERTimbau revision differs from requested pinned commit")

    coverage = {
        "train": _token_coverage(train_texts, tokenizer, cfg.max_length),
        "validation": _token_coverage(validation_texts, tokenizer, cfg.max_length),
        "test": _token_coverage(test_texts, tokenizer, cfg.max_length),
    }
    train_batches = _loader(train_texts, train_labels, tokenizer, cfg.train_batch_size,
                            cfg.max_length, torch, seed=int(seed))
    validation_batches = _loader(validation_texts, validation_labels, tokenizer,
                                 cfg.inference_batch_size, cfg.max_length, torch, seed=None)
    test_batches = _loader(test_texts, None, tokenizer, cfg.inference_batch_size,
                           cfg.max_length, torch, seed=None)

    counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    # Mean weight is one for a balanced class; inverse frequencies are based
    # only on training labels and do not require source or generator metadata.
    weights = torch.tensor(len(train_labels) / (2 * counts),
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    total_steps, warmup_steps = _schedule_steps(len(train_texts), cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        n_batches = len(train_batches)
        for index, batch in enumerate(train_batches):
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            loss_sum = nnf.cross_entropy(
                logits, labels, weight=weights, reduction="sum")
            group_start = (index // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
            group_size = min(cfg.gradient_accumulation_steps, n_batches - group_start)
            group_rows = min(len(train_texts) - group_start * cfg.train_batch_size,
                             group_size * cfg.train_batch_size)
            (loss_sum / group_rows).backward()
            running_loss += float(loss_sum.detach().cpu())
            seen += len(labels)
            if (index + 1) % cfg.gradient_accumulation_steps == 0 or index + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        _, validation_loss = _predict(
            model, validation_batches, device, torch, class_weights=weights)
        if validation_loss is None:
            raise AssertionError("Validation loss missing")
        history.append({"epoch": epoch, "train_loss": running_loss / seen,
                        "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False).to(device)
    validation_scores, _ = _predict(best_model, validation_batches, device, torch)
    test_scores, _ = _predict(best_model, test_batches, device, torch)
    if len(validation_scores) != len(validation_texts) or len(test_scores) != len(test_texts):
        raise AssertionError("BERTimbau prediction count mismatch")
    metadata = {
        "model": asdict(cfg), "resolved_revision": resolved_revision or cfg.revision,
        "seed": int(seed), "positive_class": "synthetic",
        "train_rows": len(train_texts), "validation_rows": len(validation_texts),
        "test_rows": len(test_texts), "train_class_rows": counts.tolist(),
        "total_optimizer_steps": total_steps, "warmup_steps": warmup_steps,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "history": history, "token_coverage": coverage,
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint_sha256": _sha256_file(checkpoint / "model.safetensors"),
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del best_model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"validation_scores": validation_scores,
            "test_scores": test_scores, "metadata": metadata}
