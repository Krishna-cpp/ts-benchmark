"""Model-agnostic prediction intervals via (split) conformal calibration.

Classical models such as ETS and SARIMA come with analytic intervals, but GBM
or foundation models do not, and analytic intervals rest on assumptions that
are often wrong. To compare *every* model on the same footing we build
intervals the same way for all of them:

1. Inside the training data only, run ``n_windows`` extra backtests ending at
   the forecast origin (non-overlapping, each ``horizon`` long).
2. Collect the absolute error at each horizon step ``h`` -> ``|e_h|``.
3. The interval half-width for step ``h`` is the conformal quantile of those
   errors, i.e. the ``ceil((n+1)(1-alpha))``-th smallest of ``n`` values.

Because calibration uses only data before the origin, the intervals never peek
at the future. Errors are estimated *per step*, so intervals widen naturally
with the horizon. The usual conformal guarantee assumes exchangeable errors,
which time series do not strictly satisfy, so empirical coverage is reported
rather than assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntervalConfig:
    alpha: float = 0.2  # 0.2 -> 80% intervals
    n_windows: int = 10  # calibration backtests per fold
    min_train: int = 24  # never calibrate on a history shorter than this

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.n_windows < 1:
            raise ValueError("n_windows must be >= 1")

    @property
    def level(self) -> int:
        return round(100 * (1 - self.alpha))


def conformal_quantile(abs_errors: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-sample conformal quantile of absolute errors along axis 0."""
    n = abs_errors.shape[0]
    k = min(n, math.ceil((n + 1) * (1 - alpha)))  # k-th smallest value
    return np.sort(abs_errors, axis=0)[k - 1]


def calibration_errors(model, train: np.ndarray, horizon: int,
                       cfg: IntervalConfig) -> np.ndarray:
    """Absolute errors of ``model`` on backtests inside ``train``. Shape (n, horizon)."""
    train = np.asarray(train, dtype=float)
    errors = []
    for j in range(1, cfg.n_windows + 1):
        origin = len(train) - j * horizon
        if origin < cfg.min_train:
            break
        pred = model.forecast(train[:origin], horizon)
        errors.append(np.abs(train[origin:origin + horizon] - pred))
    if not errors:
        raise ValueError("training series too short to calibrate intervals")
    return np.vstack(errors)


def conformal_interval(model, train: np.ndarray, point: np.ndarray, horizon: int,
                       cfg: IntervalConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) around ``point`` using conformal calibration on ``train``."""
    half = conformal_quantile(calibration_errors(model, train, horizon, cfg), cfg.alpha)
    return point - half, point + half
