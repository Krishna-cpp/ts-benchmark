"""Forecasters.

Every model implements ``forecast(train, horizon) -> np.ndarray`` and may only
look at ``train`` - it has no access to the future. The leakage test in
``tests/test_models.py`` enforces this for every model in the registry.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import Dataset

# Model families, used for colouring plots and grouping in the report.
BASELINE, STATISTICAL, ML, FOUNDATION = "baseline", "statistical", "ml", "foundation"


class Forecaster(ABC):
    """Common interface for every model in the benchmark."""

    name: str = "Forecaster"
    family: str = BASELINE
    #: False for models that hold large in-memory weights (e.g. torch) and should
    #: run in the main process instead of being shipped to worker processes.
    parallel_safe: bool = True

    @abstractmethod
    def forecast(self, train: np.ndarray, horizon: int) -> np.ndarray:
        """Return a length-``horizon`` point forecast that uses only ``train``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# --------------------------------------------------------------------------- baselines
class Naive(Forecaster):
    """Repeat the last observed value."""

    name, family = "Naive", BASELINE

    def forecast(self, train, horizon):
        return np.repeat(np.asarray(train, dtype=float)[-1], horizon)


class SeasonalNaive(Forecaster):
    """Repeat the last full seasonal cycle."""

    family = BASELINE

    def __init__(self, season: int):
        if season < 1:
            raise ValueError("SeasonalNaive needs season >= 1")
        self.season = season
        self.name = "SeasonalNaive"

    def forecast(self, train, horizon):
        train = np.asarray(train, dtype=float)
        reps = int(np.ceil(horizon / self.season))
        return np.tile(train[-self.season:], reps)[:horizon]


# --------------------------------------------------------------------------- statistical
class ETS(Forecaster):
    """Holt-Winters exponential smoothing with a damped additive trend."""

    family = STATISTICAL

    def __init__(self, season: int = 0):
        self.season = season
        self.name = "ETS"

    def forecast(self, train, horizon):
        kw = dict(trend="add", damped_trend=True)
        if self.season:
            kw.update(seasonal="add", seasonal_periods=self.season)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(np.asarray(train, dtype=float), **kw).fit()
        return np.asarray(fit.forecast(horizon))


class Theta(Forecaster):
    """Theta method (Assimakopoulos & Nikolopoulos 2000), winner of the M3 competition.

    Equivalent to simple exponential smoothing with drift; seasonality is removed
    first by classical additive decomposition.
    """

    family = STATISTICAL

    def __init__(self, season: int = 0):
        self.season = season
        self.name = "Theta"

    def forecast(self, train, horizon):
        y = np.asarray(train, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ThetaModel(y, period=self.season or 1, deseasonalize=bool(self.season),
                             method="additive").fit()
            return np.asarray(fit.forecast(horizon))


class SARIMA(Forecaster):
    """Fixed-order (S)ARIMA. Orders are set per dataset, not tuned per fold
    (a deliberate simplification, listed under limitations)."""

    family = STATISTICAL

    def __init__(self, season: int = 0, order=(1, 1, 1), seasonal_order=(0, 1, 1)):
        self.season = season
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.name = "SARIMA" if season else "ARIMA"

    def forecast(self, train, horizon):
        y = np.asarray(train, dtype=float)
        if self.season:
            model = SARIMAX(y, order=self.order,
                            seasonal_order=(*self.seasonal_order, self.season))
        else:
            # a constant only makes sense if the series is not differenced
            trend = "c" if self.order[1] == 0 else None
            model = SARIMAX(y, order=self.order, trend=trend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp=False)
        return np.asarray(fit.forecast(horizon))


# --------------------------------------------------------------------------- machine learning
class GBMLags(Forecaster):
    """Gradient boosting on lagged values, forecasting recursively.

    difference=False : learns the level directly. Trees cannot extrapolate beyond
                       the range seen in training, so this fails on trending data.
    difference=True  : learns the change y_t - y_{t-1}, then integrates.

    The two variants differ in exactly this one line, which makes the comparison
    a clean ablation.
    """

    family = ML

    def __init__(self, n_lags: int = 24, difference: bool = False, random_state: int = 0):
        self.n_lags = n_lags
        self.difference = difference
        self.random_state = random_state
        self.name = "GBM-diff" if difference else "GBM-level"

    @staticmethod
    def _supervised(values: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
        """Turn a series into (X = previous n_lags values, y = next value) pairs."""
        X = np.lib.stride_tricks.sliding_window_view(values[:-1], n_lags)
        return X, values[n_lags:]

    def forecast(self, train, horizon):
        y = np.asarray(train, dtype=float)
        work = np.diff(y) if self.difference else y.copy()
        n_lags = max(1, min(self.n_lags, len(work) // 2))
        X, target = self._supervised(work, n_lags)
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                              random_state=self.random_state).fit(X, target)
        window = list(work[-n_lags:])
        preds = np.empty(horizon)
        for i in range(horizon):  # recursive: feed each prediction back as an input
            preds[i] = model.predict(np.asarray(window[-n_lags:]).reshape(1, -1))[0]
            window.append(preds[i])
        return y[-1] + np.cumsum(preds) if self.difference else preds


# --------------------------------------------------------------------------- registry
CLASSICAL_MODELS = ["Naive", "SeasonalNaive", "ETS", "Theta", "SARIMA", "GBM-level", "GBM-diff"]
FOUNDATION_MODELS = ["TTM", "TTM-FT"]
ALL_MODELS = CLASSICAL_MODELS + FOUNDATION_MODELS


def build_models(dataset: Dataset | int, include: list[str] | None = None) -> list[Forecaster]:
    """Instantiate the models configured for ``dataset``.

    ``include`` filters by name (``"SARIMA"`` also matches the non-seasonal
    ``"ARIMA"``). Foundation models are only built when named explicitly,
    because they need the optional ``foundation`` dependencies.

    Passing an ``int`` (a seasonal period) is supported for backwards compatibility.
    """
    if isinstance(dataset, int):
        season, n_lags, order, sorder = dataset, 24 if dataset else 22, None, None
    else:
        season, n_lags = dataset.season, dataset.n_lags
        order, sorder = dataset.arima_order, dataset.seasonal_order
    wanted = set(CLASSICAL_MODELS if include is None else include)
    unknown = wanted - set(ALL_MODELS) - {"ARIMA"}
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; choose from {ALL_MODELS}")

    sarima_kw = {}
    if order is not None:
        sarima_kw = dict(order=order, seasonal_order=sorder)
    elif not season:
        sarima_kw = dict(order=(2, 0, 1))

    candidates: list[tuple[str, Callable[[], Forecaster | None]]] = [
        ("Naive", lambda: Naive()),
        ("SeasonalNaive", lambda: SeasonalNaive(season) if season else None),
        ("ETS", lambda: ETS(season)),
        ("Theta", lambda: Theta(season)),
        ("SARIMA", lambda: SARIMA(season, **sarima_kw)),
        ("GBM-level", lambda: GBMLags(n_lags, difference=False)),
        ("GBM-diff", lambda: GBMLags(n_lags, difference=True)),
        ("TTM", lambda: _build_ttm(dataset)),
        ("TTM-FT", lambda: _build_ttm(dataset, fine_tune=True)),
    ]
    models = []
    for key, make in candidates:
        if key in wanted or (key == "SARIMA" and "ARIMA" in wanted):
            m = make()
            if m is not None:
                models.append(m)
    return models


def _build_ttm(dataset, fine_tune: bool = False):
    from .foundation import GraniteTTM, GraniteTTMFineTuned  # optional heavy import (torch)

    return GraniteTTMFineTuned() if fine_tune else GraniteTTM()
