"""Seed management without importing heavyweight ML frameworks."""

from __future__ import annotations

import random
from typing import Any


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python and NumPy when available, returning what was seeded."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    random.seed(seed)
    seeded = {"python_random": seed}
    try:
        import numpy as np
    except ImportError:
        return seeded

    np.random.seed(seed)
    seeded["numpy"] = seed
    return seeded

