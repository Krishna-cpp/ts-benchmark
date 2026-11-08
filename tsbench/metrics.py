"""Point and interval forecast metrics.

Point:    MAE, RMSE, MASE
Interval: coverage, mean width, MSIS (the interval metric used in the M4 competition)

MASE and MSIS are *scaled* by the in-sample error of a (seasonal) naive forecast,
which makes them comparable across series with different units.
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2)))


def naive_scale(y_train: ArrayLike, lag: int = 1) -> float:
    """In-sample MAE of the lag-``lag`` naive forecast. NaN if undefined or zero."""
    y_train = _arr(y_train)
    if lag < 1 or len(y_train) <= lag:
        return float("nan")
    scale = float(np.mean(np.abs(y_train[lag:] - y_train[:-lag])))
    return scale if scale > 0 else float("nan")


def mase(y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike, lag: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    MAE divided by the in-sample MAE of a (seasonal) naive forecast. MASE < 1
    means better than the *one-step* naive forecast in-sample; for multi-step
    horizons values above 1 are normal. Returns NaN when the training series is
    constant (the scale would be zero).
    """
    return mae(y_true, y_pred) / naive_scale(y_train, lag)


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of actual values that fall inside [lower, upper]."""
    y = _arr(y_true)
    return float(np.mean((y >= _arr(lower)) & (y <= _arr(upper))))


def interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    return float(np.mean(_arr(upper) - _arr(lower)))


def msis(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, y_train: ArrayLike,
         lag: int = 1, alpha: float = 0.2) -> float:
    """Mean Scaled Interval Score (Gneiting & Raftery 2007; used in M4).

    Rewards narrow intervals and penalises every miss by 2/alpha times its
    distance outside the interval, so a model cannot win by just making
    intervals wide (or narrow). Lower is better.
    """
    y, lo, hi = _arr(y_true), _arr(lower), _arr(upper)
    penalty = (2.0 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(hi - lo + penalty)) / naive_scale(y_train, lag)
