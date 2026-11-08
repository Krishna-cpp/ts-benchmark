import numpy as np
import pytest

from tsbench.metrics import coverage, interval_width, mae, mase, msis, naive_scale, rmse


def test_perfect_forecast_has_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mase(y, y, np.arange(10.0)) == 0.0


def test_mase_of_naive_in_sample_is_one():
    train = np.array([1.0, 3.0, 2.0, 5.0])
    # in-sample naive MAE = mean(|2|,|1|,|3|) = 2 ; forecast error of 2 -> MASE 1
    assert mase([0.0], [2.0], train, lag=1) == 1.0


def test_seasonal_scale_uses_lag():
    train = np.tile([0.0, 10.0], 5)  # perfectly periodic with period 2
    assert naive_scale(train, lag=1) == 10.0
    assert np.isnan(naive_scale(train, lag=2))  # seasonal naive is perfect -> undefined


def test_mase_constant_train_is_nan_not_inf():
    assert np.isnan(mase([1.0], [2.0], np.ones(10)))


def test_rmse_penalises_large_errors_more_than_mae():
    y, p = np.zeros(4), np.array([0.0, 0.0, 0.0, 4.0])
    assert mae(y, p) == 1.0
    assert rmse(y, p) == 2.0


def test_coverage_and_width():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo, hi = np.array([0.0, 2.5, 2.0, 3.0]), np.array([2.0, 3.0, 4.0, 5.0])
    assert coverage(y, lo, hi) == 0.75
    assert interval_width(lo, hi) == pytest.approx(1.625)


def test_msis_is_width_when_covered_and_penalises_misses():
    train = np.arange(10.0)  # naive scale = 1
    y = np.array([5.0, 5.0])
    assert msis(y, [4.0, 4.0], [6.0, 6.0], train, alpha=0.2) == pytest.approx(2.0)
    # second point: interval [1, 4] has width 3, misses by 1 -> penalty (2/0.2)*1 = 10
    assert msis(y, [4.0, 1.0], [6.0, 4.0], train, alpha=0.2) == pytest.approx((2 + 3 + 10) / 2)
