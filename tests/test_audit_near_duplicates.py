from __future__ import annotations

from horus_benchmark.audit.near_duplicates import (
    NearDuplicateConfig, bottom_k_signature, candidate_pairs,
    exact_shingle_similarity, verified_pairs,
)


def test_near_duplicate_candidates_are_verified_with_exact_shingles() -> None:
    first = " ".join(f"palavra{i}" for i in range(100))
    edited = first.replace("palavra50", "termoalterado")
    unrelated = " ".join(f"outro{i}" for i in range(100))
    config = NearDuplicateConfig(minimum_shared_fingerprints=4)
    texts = [first, edited, unrelated]
    signatures = [bottom_k_signature(text, config) for text in texts]

    candidates, metadata = candidate_pairs(signatures, config)
    accepted = verified_pairs(candidates, dict(enumerate(texts)), config)

    assert metadata["indexed_documents"] == 3
    assert any(pair[:2] == (0, 1) for pair in candidates)
    assert len(accepted) == 1
    assert accepted[0]["jaccard"] >= config.jaccard_threshold


def test_containment_detects_a_short_excerpt_inside_long_document() -> None:
    long_text = " ".join(f"token{i}" for i in range(150))
    excerpt = " ".join(f"token{i}" for i in range(30, 80))

    jaccard, containment = exact_shingle_similarity(long_text, excerpt, 5)

    assert jaccard < 0.8
    assert containment == 1.0

