"""Datasets used in the benchmark.

All three ship with statsmodels, so the benchmark runs fully offline. They were
chosen to cover three different *regimes*, because the interesting question is
not "which model is best" but "which model is best for which kind of series":

=================  =========  ===========================================
dataset            frequency  regime
=================  =========  ===========================================
co2_monthly        monthly    strong trend + stable yearly seasonality
elnino_monthly     monthly    seasonality, no trend, irregular ENSO swings
sunspots_yearly    yearly     noisy quasi-cycle (~11y, varying), no trend
=================  =========  ===========================================
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class Dataset:
    """A univariate series plus the evaluation protocol that goes with it."""

    name: str
    series: pd.Series
    description: str
    season: int  # seasonal period (0 = none)
    horizon: int  # forecast horizon per fold
    n_folds: int  # number of rolling-origin folds
    step: int  # how far the forecast origin moves between folds
    scale_lag: int  # lag of the naive forecast used as MASE / MSIS denominator
    n_lags: int = 24  # input window for the lag-based ML models
    # Fixed (S)ARIMA orders. Chosen once from domain knowledge, NOT tuned per fold.
    arima_order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int] = field(default=(0, 1, 1))

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.series.values, dtype=float)

    @property
    def first_origin(self) -> int:
        """Training length of the earliest rolling-origin fold."""
        return len(self) - self.horizon - self.step * (self.n_folds - 1)

    def __len__(self) -> int:
        return len(self.series)

    def with_folds(self, n_folds: int) -> Dataset:
        """Copy with fewer/more folds (used by --quick)."""
        return replace(self, n_folds=n_folds)


def load_co2() -> Dataset:
    """Mauna Loa atmospheric CO2 (ppm), weekly resampled to monthly means."""
    raw = sm.datasets.co2.load_pandas().data["co2"]
    monthly = raw.resample("MS").mean().interpolate()
    monthly.name = "co2_ppm"
    return Dataset(
        "co2_monthly", monthly,
        description="Mauna Loa CO2: strong trend + stable yearly seasonality",
        season=12, horizon=12, n_folds=10, step=12, scale_lag=12, n_lags=24,
        arima_order=(1, 1, 1), seasonal_order=(0, 1, 1),
    )


def load_elnino() -> Dataset:
    """Monthly Pacific sea-surface temperature (deg C), 1950-2010.

    Clear yearly seasonality but no trend, plus irregular multi-year El Nino /
    La Nina swings that no model can see coming from the series alone.
    """
    df = sm.datasets.elnino.load_pandas().data
    values = df.drop(columns="YEAR").to_numpy(dtype=float).ravel()
    start = f"{int(df['YEAR'].iloc[0])}-01"
    s = pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"),
                  name="sea_temp_c")
    return Dataset(
        "elnino_monthly", s,
        description="Pacific sea temperature: seasonality, no trend, irregular ENSO swings",
        season=12, horizon=12, n_folds=10, step=12, scale_lag=12, n_lags=24,
        arima_order=(1, 0, 1), seasonal_order=(0, 1, 1),
    )


def load_sunspots() -> Dataset:
    """Yearly sunspot activity, 1700-2008. Noisy ~11 year cycle, no trend."""
    df = sm.datasets.sunspots.load_pandas().data
    s = pd.Series(df["SUNACTIVITY"].values,
                  index=pd.PeriodIndex(df["YEAR"].astype(int).astype(str), freq="Y"),
                  name="sunspots")
    return Dataset(
        "sunspots_yearly", s,
        description="Sunspots: noisy quasi-cycle of varying length, no trend",
        season=0, horizon=5, n_folds=12, step=5, scale_lag=1, n_lags=22,
        arima_order=(2, 0, 1), seasonal_order=(0, 0, 0),
    )


DATASETS: dict[str, Callable[[], Dataset]] = {
    "co2_monthly": load_co2,
    "elnino_monthly": load_elnino,
    "sunspots_yearly": load_sunspots,
}


def load_datasets(names: list[str] | None = None) -> list[Dataset]:
    """Load the named datasets (all of them if ``names`` is None)."""
    names = list(DATASETS) if names is None else names
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    return [DATASETS[n]() for n in names]


def load_all() -> list[Dataset]:
    """Backwards-compatible alias for :func:`load_datasets`."""
    return load_datasets()
