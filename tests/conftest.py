"""Shared fixtures: schema-conformant synthetic tables (test data only, never displayed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def make_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a random table that satisfies every rule in ``config.SCHEMA``."""
    rng = np.random.default_rng(seed)
    data = {}
    for col, spec in config.SCHEMA.items():
        if spec.kind == "numeric":
            data[col] = rng.integers(spec.min_value, spec.max_value + 1, size=n)
        else:
            data[col] = rng.choice(spec.allowed, size=n)
    return pd.DataFrame(data)


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
