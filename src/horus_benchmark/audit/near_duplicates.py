"""Deterministic candidate search for near duplicates across the full corpus.

The bottom-k fingerprint index is a recall-oriented screen. Only candidates
are verified with exact shingle Jaccard and containment. No negative result
from this screen proves that every possible near duplicate is absent.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class NearDuplicateConfig:
    shingle_words: int = 5
    bottom_k: int = 32
    minimum_shared_fingerprints: int = 6
    max_postings_per_fingerprint: int = 80
    max_pair_comparisons: int = 20_000_000
    jaccard_threshold: float = 0.80
    containment_threshold: float = 0.90
    min_tokens: int = 30

    def __post_init__(self) -> None:
        if self.shingle_words < 1 or self.bottom_k < 1 or self.min_tokens < 1:
            raise ValueError("Shingle size, bottom_k and min_tokens must be positive")
        if self.minimum_shared_fingerprints < 1:
            raise ValueError("minimum_shared_fingerprints must be positive")
        if self.max_postings_per_fingerprint < 2 or self.max_pair_comparisons < 1:
            raise ValueError("Candidate limits must be positive")
        for threshold in (self.jaccard_threshold, self.containment_threshold):
            if not 0 <= threshold <= 1:
                raise ValueError("Similarity thresholds must be between 0 and 1")


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold())


def shingle_hashes(value: str, words_per_shingle: int) -> set[int]:
    """Return stable 64-bit hashes of word shingles in a document."""

    tokens = _tokens(value)
    if len(tokens) < words_per_shingle:
        return set()
    hashes: set[int] = set()
    for position in range(len(tokens) - words_per_shingle + 1):
        shingle = "\x1f".join(tokens[position : position + words_per_shingle])
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        hashes.add(int.from_bytes(digest, "big"))
    return hashes


def literal_shingles(value: str, words_per_shingle: int) -> set[tuple[str, ...]]:
    """Return token tuples for collision-free final pair verification."""

    tokens = _tokens(value)
    return {
        tuple(tokens[position : position + words_per_shingle])
        for position in range(max(0, len(tokens) - words_per_shingle + 1))
    }


def bottom_k_signature(
    value: str, config: NearDuplicateConfig
) -> tuple[int, ...]:
    """Return a KMV/bottom-k sketch; short texts have no signature."""

    if len(_tokens(value)) < config.min_tokens:
        return ()
    return tuple(sorted(shingle_hashes(value, config.shingle_words))[: config.bottom_k])


def candidate_pairs(
    signatures: Sequence[tuple[int, ...]], config: NearDuplicateConfig
) -> tuple[list[tuple[int, int, int]], dict[str, int | bool]]:
    """Find candidate pairs sharing enough rare bottom-k fingerprints."""

    postings: dict[int, list[int]] = defaultdict(list)
    for row_index, signature in enumerate(signatures):
        for fingerprint in signature:
            postings[fingerprint].append(row_index)

    signature_sets = [set(signature) for signature in signatures]
    accepted: dict[tuple[int, int], int] = {}
    pair_comparisons = 0
    skipped_postings = 0
    truncated = False
    for fingerprint in sorted(postings):
        rows = postings[fingerprint]
        if len(rows) > config.max_postings_per_fingerprint:
            skipped_postings += 1
            continue
        for left, right in itertools.combinations(rows, 2):
            if pair_comparisons >= config.max_pair_comparisons:
                truncated = True
                break
            pair_comparisons += 1
            pair = (left, right)
            if pair in accepted:
                continue
            shared = len(signature_sets[left] & signature_sets[right])
            if shared >= config.minimum_shared_fingerprints:
                accepted[pair] = shared
        if truncated:
            break

    candidates = [
        (left, right, shared)
        for (left, right), shared in accepted.items()
    ]
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    return candidates, {
        "indexed_documents": sum(bool(signature) for signature in signatures),
        "short_documents_not_indexed": sum(not signature for signature in signatures),
        "fingerprints": len(postings),
        "frequent_fingerprints_skipped": skipped_postings,
        "pair_comparisons": pair_comparisons,
        "candidate_pairs_after_shared_threshold": len(candidates),
        "candidate_limit_reached": truncated,
    }


def exact_shingle_similarity(
    left: str, right: str, words_per_shingle: int
) -> tuple[float, float]:
    """Return exact Jaccard and containment on the configured shingle sets."""

    left_set = literal_shingles(left, words_per_shingle)
    right_set = literal_shingles(right, words_per_shingle)
    if not left_set or not right_set:
        return 0.0, 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union, intersection / min(len(left_set), len(right_set))


def verified_pairs(
    candidates: Iterable[tuple[int, int, int]],
    texts_by_index: dict[int, str],
    config: NearDuplicateConfig,
) -> list[dict[str, int | float | bool]]:
    """Verify candidate similarities and retain pairs meeting either threshold."""

    accepted = []
    for left, right, shared in candidates:
        jaccard, containment = exact_shingle_similarity(
            texts_by_index[left], texts_by_index[right], config.shingle_words
        )
        if jaccard >= config.jaccard_threshold or containment >= config.containment_threshold:
            accepted.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "shared_bottom_k": shared,
                    "jaccard": jaccard,
                    "containment": containment,
                    "jaccard_match": jaccard >= config.jaccard_threshold,
                    "containment_match": containment >= config.containment_threshold,
                }
            )
    return accepted

