import numpy as np
import pytest


@pytest.fixture
def seasonal_series():
    """Random walk + yearly seasonality, 120 points (deterministic)."""
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(size=120)) + np.sin(np.arange(120) * 2 * np.pi / 12)
