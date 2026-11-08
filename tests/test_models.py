import numpy as np
import pytest

from tsbench.data import load_datasets
from tsbench.models import (
    CLASSICAL_MODELS,
    ETS,
    SARIMA,
    GBMLags,
    Naive,
    SeasonalNaive,
    Theta,
    build_models,
)

ALL_CLASSICAL = [Naive(), SeasonalNaive(12), ETS(12), Theta(12), SARIMA(12),
                 GBMLags(12, difference=False), GBMLags(12, difference=True)]


def test_naive_and_seasonal_naive():
    train = np.arange(1, 25, dtype=float)
    assert (Naive().forecast(train, 3) == 24).all()
    assert (SeasonalNaive(12).forecast(train, 3) == np.array([13, 14, 15])).all()
    # a horizon longer than one season wraps around
    assert (SeasonalNaive(12).forecast(train, 14)[12:] == np.array([13, 14])).all()


@pytest.mark.parametrize("model", ALL_CLASSICAL, ids=lambda m: m.name)
def test_no_lookahead_models_only_use_train(model, seasonal_series):
    """Changing data AFTER the origin must not change a forecast made at the origin."""
    origin = 100
    future_b = seasonal_series.copy()
    future_b[origin:] += 1000
    a = model.forecast(seasonal_series[:origin], 6)
    b = model.forecast(future_b[:origin], 6)
    assert np.allclose(a, b)


@pytest.mark.parametrize("model", ALL_CLASSICAL, ids=lambda m: m.name)
def test_forecast_shape_and_finite(model, seasonal_series):
    pred = model.forecast(seasonal_series, 7)
    assert pred.shape == (7,)
    assert np.isfinite(pred).all()


def test_gbm_level_cannot_extrapolate_but_gbm_diff_can():
    """The headline finding as a unit test: on a pure linear trend a tree model
    on levels is stuck at the training range, the differenced one is not."""
    y = np.arange(200, dtype=float)
    level = GBMLags(12, difference=False).forecast(y, 10)
    diff = GBMLags(12, difference=True).forecast(y, 10)
    assert level.max() <= y.max() + 1e-6
    assert np.allclose(diff, np.arange(200, 210), atol=0.5)


def test_gbm_supervised_windows():
    X, t = GBMLags._supervised(np.arange(6.0), 3)
    assert X.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert t.tolist() == [3, 4, 5]


def test_build_models_per_dataset():
    co2, _, sun = load_datasets()
    assert [m.name for m in build_models(co2)] == CLASSICAL_MODELS
    sun_names = [m.name for m in build_models(sun)]
    assert "SeasonalNaive" not in sun_names and "ARIMA" in sun_names
    assert [m.name for m in build_models(sun, ["Naive", "SARIMA"])] == ["Naive", "ARIMA"]
    with pytest.raises(ValueError):
        build_models(co2, ["NotAModel"])


def test_build_models_legacy_int_api():
    assert [m.name for m in build_models(12)] == CLASSICAL_MODELS
