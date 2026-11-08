"""Describe a time series with a few interpretable numbers.

These are the inputs to the model recommender. Each one has a plain meaning, so
the recommender's choices can be explained ("strong trend, weak seasonality ->
use a statistical model"). Definitions follow Wang, Smith & Hyndman (2006) and
Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, ch. 4.

All features are computed on the series *up to the first forecast origin*, so
the recommender never sees the data it is asked to predict.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import periodogram
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

FEATURES = [
    "log_length", "season", "trend_strength", "seasonal_strength", "spectral_entropy",
    "diff_acf1", "remainder_acf1", "lumpiness", "cv", "recent_change",
]


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else 0.0


def decompose(y: np.ndarray, season: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trend, seasonal and remainder parts (STL if seasonal, LOWESS trend if not)."""
    if season >= 2 and len(y) >= 2 * season + 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(y, period=season, robust=True).fit()
        return res.trend, res.seasonal, res.resid
    trend = lowess(y, np.arange(len(y)), frac=0.3, return_sorted=False)
    return trend, np.zeros_like(y), y - trend


def _strength(remainder: np.ndarray, component: np.ndarray) -> float:
    """1 - Var(remainder) / Var(component + remainder), clipped to [0, 1]."""
    v = np.var(component + remainder)
    return float(np.clip(1 - np.var(remainder) / v, 0, 1)) if v > 0 else 0.0


def spectral_entropy(y: np.ndarray) -> float:
    """Normalised Shannon entropy of the spectrum: ~0 = very regular, 1 = white noise."""
    x = np.diff(y) if len(y) > 2 else y
    x = (x - x.mean()) / (x.std() or 1.0)
    _, p = periodogram(x)
    p = p[1:]
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1 / len(p))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def lumpiness(y: np.ndarray, width: int) -> float:
    """Variance of the variances of non-overlapping windows (scaled series)."""
    x = (y - y.mean()) / (y.std() or 1.0)
    n = len(x) // width
    if n < 2:
        return 0.0
    return float(np.var([x[i * width:(i + 1) * width].var() for i in range(n)]))


def series_features(y: np.ndarray, season: int) -> dict[str, float]:
    """All features for one series (pass only the training part!)."""
    y = np.asarray(y, dtype=float)
    trend, seasonal, rem = decompose(y, season)
    width = season if season >= 2 else 10
    tail = y[-max(width, 4):]
    scale = np.mean(np.abs(y)) or 1.0
    return {
        "log_length": float(np.log(len(y))),
        "season": float(season),
        "trend_strength": _strength(rem, trend),
        "seasonal_strength": _strength(rem, seasonal) if season >= 2 else 0.0,
        "spectral_entropy": spectral_entropy(y),
        "diff_acf1": _acf1(np.diff(y)),
        "remainder_acf1": _acf1(rem),
        "lumpiness": lumpiness(y, width),
        "cv": float(np.std(y) / scale),
        # how much the level moved over the last season (or 10 points), relative to the
        # series' typical size: large values = a recent break or a steep trend
        "recent_change": float((tail[-1] - tail[0]) / scale),
    }
