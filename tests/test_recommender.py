import numpy as np
import pandas as pd
import pytest

from tsbench.data import Dataset
from tsbench.features import FEATURES, series_features, spectral_entropy
from tsbench.recommender import (
    cross_validate,
    error_table,
    fit_recommender,
    load_model,
    recommend,
    run_study,
    save_model,
    summarize,
)


def test_trend_and_seasonal_strength_pick_up_the_right_structure():
    t = np.arange(240, dtype=float)
    rng = np.random.default_rng(0)
    trend = series_features(t + rng.normal(0, 1, 240), season=12)
    season = series_features(10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 240), season=12)
    assert trend["trend_strength"] > 0.9 and trend["seasonal_strength"] < 0.3
    assert season["seasonal_strength"] > 0.9
    assert set(FEATURES) <= set(trend)


def test_spectral_entropy_separates_noise_from_signal():
    rng = np.random.default_rng(1)
    noise = spectral_entropy(rng.normal(size=500))
    wave = spectral_entropy(np.sin(np.arange(500) * 2 * np.pi / 20))
    assert noise > 0.85 > wave


def test_features_work_without_season_and_on_short_series():
    f = series_features(np.linspace(1, 2, 25) + np.random.default_rng(2).normal(0, 0.01, 25), 0)
    assert all(np.isfinite(v) for v in f.values())
    assert f["seasonal_strength"] == 0.0


def _synthetic_collection(n=24):
    """Half trending series (statistical models should win), half noisy and flat."""
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        t = np.arange(80, dtype=float)
        y = 0.5 * t + rng.normal(0, 1, 80) if i % 2 == 0 else rng.normal(10, 1, 80)
        out.append(Dataset(f"s{i}", pd.Series(y), f"synthetic {i} (other)", season=0, horizon=4,
                           n_folds=2, step=4, scale_lag=1, n_lags=6, arima_order=(1, 1, 1),
                           seasonal_order=(0, 0, 0)))
    return out


@pytest.mark.slow
def test_recommender_end_to_end(tmp_path):
    datasets = _synthetic_collection()
    folds, forecasts = run_study(datasets, with_ttm=False, n_jobs=2)
    Y = error_table(folds)
    assert {"Naive", "ETS", "ARIMA", "GBM-diff"} <= set(Y.columns)
    assert "SeasonalNaive" not in Y.columns  # no season -> model not built
    X = pd.DataFrame({ds.name: series_features(ds.values[:ds.first_origin], 0)
                      for ds in datasets}).T
    cv = cross_validate(X, Y.loc[X.index], n_splits=3, forecasts=forecasts)
    assert {"recommended_top_mase", "fixed_combo_mase"} <= set(cv.columns)
    s = summarize(cv, Y.loc[X.index])
    # the oracle is a lower bound for every strategy
    assert s["strategies"]["oracle (knows the winner in advance)"]["mean_mase"] <= \
        s["strategies"]["recommender, single pick"]["mean_mase"] + 1e-12
    assert (cv["oracle_mase"] <= cv["recommended_mase"] + 1e-12).all()

    rf = fit_recommender(X, Y.loc[X.index])
    save_model(rf, list(Y.columns), tmp_path / "rec.joblib")
    ranking = recommend(np.arange(60.0), season=0, bundle=load_model(tmp_path / "rec.joblib"))
    assert list(ranking.index[:1])[0] in Y.columns
    assert ranking.is_monotonic_increasing

    # the command-line tool on a user's CSV
    from tsbench.suggest import main as suggest

    csv = tmp_path / "sales.csv"
    pd.DataFrame({"month": range(60), "revenue": np.arange(60.0)}).to_csv(csv, index=False)
    out = suggest([str(csv), "--column", "revenue", "--model", str(tmp_path / "rec.joblib")])
    pd.testing.assert_series_equal(out, ranking)
