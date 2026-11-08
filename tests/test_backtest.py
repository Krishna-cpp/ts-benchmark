import numpy as np
import pandas as pd
import pytest

from tsbench.backtest import make_origins, rolling_origin, rolling_origin_backtest
from tsbench.data import Dataset, load_datasets
from tsbench.intervals import IntervalConfig
from tsbench.models import Forecaster, Naive, SeasonalNaive


def toy_dataset(n=120, n_folds=3):
    y = pd.Series(np.sin(np.arange(n) * 2 * np.pi / 12) + 0.01 * np.arange(n))
    return Dataset("toy", y, "toy", season=12, horizon=6, n_folds=n_folds, step=6,
                   scale_lag=12, n_lags=4)


def test_origins_end_at_series_end():
    origins = make_origins(n=100, horizon=10, step=5, n_folds=4)
    assert origins == [75, 80, 85, 90]
    assert origins[-1] + 10 == 100


def test_origins_reject_too_many_folds():
    with pytest.raises(ValueError):
        make_origins(n=50, horizon=10, step=10, n_folds=5, min_train=20)


def test_dataset_first_origin_matches_make_origins():
    for ds in load_datasets():
        assert make_origins(len(ds), ds.horizon, ds.step, ds.n_folds)[0] == ds.first_origin


def test_backtest_shapes_and_columns():
    ds = toy_dataset()
    res = rolling_origin_backtest(ds, [Naive(), SeasonalNaive(12)],
                                  intervals=IntervalConfig(n_windows=3), n_jobs=1)
    assert len(res.folds) == 2 * 3
    assert len(res.forecasts) == 2 * 3 * 6
    assert {"mae", "rmse", "mase", "coverage", "width", "msis", "fit_seconds"} <= set(res.folds)
    assert res.folds["error"].isna().all()
    # the actuals stored next to each forecast must be the true future values
    y = ds.values
    for r in res.forecasts.itertuples():
        assert r.actual == y[r.origin + r.step - 1]


def test_parallel_equals_sequential():
    ds = toy_dataset()
    a = rolling_origin_backtest(ds, [Naive(), SeasonalNaive(12)], intervals=None, n_jobs=1)
    b = rolling_origin_backtest(ds, [Naive(), SeasonalNaive(12)], intervals=None, n_jobs=2)
    cols = ["model", "fold", "mae", "mase"]
    pd.testing.assert_frame_equal(a.folds[cols], b.folds[cols])


class Broken(Forecaster):
    name = "Broken"

    def forecast(self, train, horizon):
        raise RuntimeError("boom")


def test_one_failing_model_does_not_kill_the_run():
    res = rolling_origin_backtest(toy_dataset(), [Naive(), Broken()], intervals=None)
    broken = res.folds[res.folds["model"] == "Broken"]
    assert broken["error"].str.contains("boom").all()
    assert res.folds.loc[res.folds["model"] == "Naive", "error"].isna().all()


def test_legacy_rolling_origin_api():
    ds = load_datasets(["sunspots_yearly"])[0]
    folds, last = rolling_origin(ds, [Naive()])
    train, actual, pred = last["Naive"]
    assert len(folds) == ds.n_folds
    assert len(train) + len(actual) == len(ds)
    assert (pred == train[-1]).all()
