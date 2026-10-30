"""Random gate matched to another gate's rewrite rate."""

import numpy as np


def select_random(rewrite_count, total_count, seed=42):
    """Select exactly rewrite_count indices without replacement."""
    if not 0 <= rewrite_count <= total_count:
        raise ValueError("Invalid rewrite_count.")

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        total_count,
        size=rewrite_count,
        replace=False
    )

    return set(selected.tolist())