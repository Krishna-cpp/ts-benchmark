"""Granite TTM tests. Skipped automatically when torch / granite-tsfm are missing."""
import pickle

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tsfm_public")

from tsbench.foundation import GraniteTTM  # noqa: E402

pytestmark = pytest.mark.foundation


@pytest.fixture(scope="module")
def ttm():
    m = GraniteTTM()
    try:
        m.forecast(np.sin(np.arange(600) / 3), 12)  # triggers the (cached) download
    except OSError as e:  # offline and not cached
        pytest.skip(f"TTM weights unavailable: {e}")
    return m


def test_ttm_shape_and_no_lookahead(ttm, seasonal_series):
    a = ttm.forecast(seasonal_series[:100], 6)
    shifted = seasonal_series.copy()
    shifted[100:] += 1000
    b = ttm.forecast(shifted[:100], 6)
    assert a.shape == (6,) and np.isfinite(a).all()
    assert np.allclose(a, b)


def test_short_history_padding_is_level_invariant(ttm):
    """Regression test: padding short histories with raw zeros made forecasts
    depend on the series level. Shifting the input by a constant must shift
    the forecast by the same constant."""
    y = np.sin(np.arange(200) * 2 * np.pi / 12)  # shorter than the 512 context
    a = ttm.forecast(y, 12)
    b = ttm.forecast(y + 1000, 12)
    assert np.allclose(b - 1000, a, atol=0.05)


def test_ttm_pickles_without_weights(ttm):
    clone = pickle.loads(pickle.dumps(ttm))
    assert clone._models == {}
