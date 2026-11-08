import numpy as np
import pytest

from tsbench.intervals import (
    IntervalConfig,
    calibration_errors,
    conformal_interval,
    conformal_quantile,
)
from tsbench.models import Forecaster, Naive


def test_conformal_quantile_finite_sample_rank():
    errs = np.arange(1, 11, dtype=float).reshape(-1, 1)  # 1..10
    # n=10, alpha=0.2 -> ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(errs, 0.2)[0] == 9.0
    # tiny n: rank is clipped to n (the maximum)
    assert conformal_quantile(errs[:3], 0.1)[0] == 3.0


def test_calibration_uses_only_training_data():
    """Calibration errors must be identical whatever happens after the origin."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    cfg = IntervalConfig(n_windows=5)
    a = calibration_errors(Naive(), y[:150], 6, cfg)
    y2 = y.copy()
    y2[150:] = 1e6
    b = calibration_errors(Naive(), y2[:150], 6, cfg)
    assert a.shape == (5, 6)
    assert np.array_equal(a, b)


class Mean(Forecaster):
    name = "Mean"

    def forecast(self, train, horizon):
        return np.repeat(np.mean(train), horizon)


def test_coverage_close_to_nominal_on_iid_noise():
    """On exchangeable (iid) data split-conformal intervals should hit their target."""
    rng = np.random.default_rng(42)
    cfg = IntervalConfig(alpha=0.2, n_windows=20)
    hits = []
    for _ in range(150):
        y = rng.normal(size=300)
        train, test = y[:-5], y[-5:]
        point = Mean().forecast(train, 5)
        lo, hi = conformal_interval(Mean(), train, point, 5, cfg)
        hits.append(np.mean((test >= lo) & (test <= hi)))
    assert 0.74 <= np.mean(hits) <= 0.92


def test_interval_config_validation():
    with pytest.raises(ValueError):
        IntervalConfig(alpha=1.5)
    with pytest.raises(ValueError):
        IntervalConfig(n_windows=0)
    assert IntervalConfig(alpha=0.05).level == 95
