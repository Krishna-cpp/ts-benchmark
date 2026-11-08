"""Rolling-origin evaluation.

The forecast origin moves forward through time; at every origin each model is
refit on the data available at that moment only and asked for ``horizon``
steps. This is the time-series equivalent of cross-validation (random K-fold
would leak the future into training).

    fold 0: [train ............]|test|
    fold 1: [train .................]|test|
    fold 2: [train ......................]|test|   <- last fold ends at series end
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from .data import Dataset
from .intervals import IntervalConfig, conformal_interval
from .metrics import coverage, interval_width, mae, mase, msis, naive_scale, rmse

log = logging.getLogger(__name__)


def make_origins(n: int, horizon: int, step: int, n_folds: int, min_train: int = 1) -> list[int]:
    """Forecast origins (= train lengths), oldest first. The last fold ends at ``n``."""
    if horizon < 1 or step < 1 or n_folds < 1:
        raise ValueError("horizon, step and n_folds must be >= 1")
    last = n - horizon
    origins = [last - step * i for i in range(n_folds)][::-1]
    if origins[0] < min_train:
        need = min_train + horizon + step * (n_folds - 1)
        raise ValueError(f"{n_folds} folds with step {step} need at least {need} "
                         f"observations, series has {n}")
    return origins


@dataclass
class BacktestResult:
    folds: pd.DataFrame  # one row per (dataset, model, fold) with all metrics
    forecasts: pd.DataFrame  # one row per (dataset, model, fold, step): actual, pred, lo, hi


def _evaluate(dataset: Dataset, model, fold: int, origin: int,
              intervals: IntervalConfig | None) -> tuple[dict, list[dict]]:
    """Fit/forecast one model at one origin and score it. Never raises."""
    y = dataset.values
    h = dataset.horizon
    train, actual = y[:origin], y[origin:origin + h]
    row = dict(dataset=dataset.name, model=model.name, family=model.family,
               fold=fold, origin=origin, error=None)
    pred = lo = hi = np.full(h, np.nan)
    try:
        t0 = time.perf_counter()
        pred = np.asarray(model.forecast(train, h), dtype=float)
        row["fit_seconds"] = time.perf_counter() - t0
        if pred.shape != (h,) or not np.all(np.isfinite(pred)):
            raise ValueError(f"bad forecast: shape {pred.shape}, finite={np.isfinite(pred).all()}")
        row.update(mae=mae(actual, pred), rmse=rmse(actual, pred),
                   mase=mase(actual, pred, train, dataset.scale_lag))
        if intervals is not None:
            lo, hi = conformal_interval(model, train, pred, h, intervals)
            row.update(coverage=coverage(actual, lo, hi), width=interval_width(lo, hi),
                       msis=msis(actual, lo, hi, train, dataset.scale_lag, intervals.alpha))
    except Exception as exc:  # one broken model/fold must not kill a long run
        row["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("%s / %s / fold %d failed: %s", dataset.name, model.name, fold, row["error"])
        log.debug(traceback.format_exc())
    scale = naive_scale(train, dataset.scale_lag)  # lets errors be compared across folds
    records = [dict(dataset=dataset.name, model=model.name, fold=fold, origin=origin,
                    step=i + 1, actual=actual[i], pred=pred[i], lo=lo[i], hi=hi[i],
                    scale=scale)
               for i in range(h)]
    return row, records


def rolling_origin_backtest(dataset: Dataset, models: list, *,
                            intervals: IntervalConfig | None = IntervalConfig(),
                            n_jobs: int = 1) -> BacktestResult:
    """Evaluate ``models`` on ``dataset`` with rolling-origin backtesting.

    Every (model, fold) pair is independent, so they run in parallel with
    ``n_jobs`` worker processes. Models flagged ``parallel_safe = False``
    (large torch models) run in the main process.
    """
    min_train = max(2 * dataset.n_lags + 2, dataset.scale_lag + 1)
    if intervals is not None:
        min_train += intervals.n_windows * dataset.horizon
    origins = make_origins(len(dataset), dataset.horizon, dataset.step,
                           dataset.n_folds, min_train)
    tasks = [(m, f, o) for m in models for f, o in enumerate(origins)]
    par = [t for t in tasks if getattr(t[0], "parallel_safe", True)]
    seq = [t for t in tasks if not getattr(t[0], "parallel_safe", True)]

    out = []
    if par:
        # one BLAS/OpenMP thread per worker process, otherwise 12 workers x 12
        # threads fight over 12 cores and everything gets ~3x slower
        with parallel_config(backend="loky", inner_max_num_threads=1):
            out = Parallel(n_jobs=n_jobs)(delayed(_evaluate)(dataset, m, f, o, intervals)
                                          for m, f, o in par)
    out += [_evaluate(dataset, m, f, o, intervals) for m, f, o in seq]

    folds = pd.DataFrame([r for r, _ in out])
    forecasts = pd.DataFrame([rec for _, recs in out for rec in recs])
    order = {m.name: i for i, m in enumerate(models)}
    folds = (folds.assign(_o=folds["model"].map(order))
                  .sort_values(["_o", "fold"]).drop(columns="_o").reset_index(drop=True))
    return BacktestResult(folds, forecasts)


def rolling_origin(dataset: Dataset, models: list):
    """Backwards-compatible API: point forecasts only, sequential.

    Returns (fold_results DataFrame, {model_name: (train, actual, pred)} for the last fold).
    """
    res = rolling_origin_backtest(dataset, models, intervals=None, n_jobs=1)
    y = dataset.values
    last_fold = res.folds["fold"].max()
    last = {}
    for name, g in res.forecasts[res.forecasts["fold"] == last_fold].groupby("model", sort=False):
        origin = int(g["origin"].iloc[0])
        last[name] = (y[:origin], g["actual"].to_numpy(), g["pred"].to_numpy())
    return res.folds, last
