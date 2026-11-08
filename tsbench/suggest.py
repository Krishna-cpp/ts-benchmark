"""Ask the trained recommender which forecasting models to use for your own series.

    python -m tsbench.suggest sales.csv --season 12
    python -m tsbench.suggest sales.csv --column revenue --season 4

The CSV needs one numeric column (pick it with --column if there are several),
oldest value first. The recommender was trained by ``python -m tsbench.recommender``
on M4 series; its file lives in models/recommender.joblib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import series_features
from .recommender import MODEL_PATH, TOP_K, load_model, recommend

DESCRIPTIONS = {
    "Naive": "repeat the last value", "SeasonalNaive": "repeat the last season",
    "ETS": "exponential smoothing", "Theta": "Theta method", "ARIMA": "(S)ARIMA",
    "GBM-level": "gradient boosting on levels", "GBM-diff": "gradient boosting on changes",
    "TTM": "IBM Granite TTM, zero-shot",
}


def read_series(path: Path, column: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    numeric = df.select_dtypes("number")
    if column:
        y = df[column]
    elif numeric.shape[1] == 1:
        y = numeric.iloc[:, 0]
    else:
        raise SystemExit(f"{path} has {numeric.shape[1]} numeric columns; choose one with "
                         f"--column ({', '.join(numeric.columns)})")
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 12:
        raise SystemExit(f"need at least 12 values, got {len(y)}")
    return y


def main(argv=None) -> pd.Series:
    p = argparse.ArgumentParser(prog="tsbench.suggest", description=__doc__.split("\n\n")[0])
    p.add_argument("csv", type=Path)
    p.add_argument("--column", help="numeric column to use (if the file has several)")
    p.add_argument("--season", type=int, default=0,
                   help="seasonal period: 12 monthly, 4 quarterly, 0 none/yearly (default 0)")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="trained recommender file")
    args = p.parse_args(argv)

    y = read_series(args.csv, args.column)
    ranking = recommend(y, args.season, load_model(args.model))
    f = series_features(y, args.season)
    print(f"{len(y)} values. Trend strength {f['trend_strength']:.2f}, seasonal strength "
          f"{f['seasonal_strength']:.2f}, noisiness {f['spectral_entropy']:.2f}.\n")
    print("Predicted error (MASE) per model, best first:")
    for i, (m, v) in enumerate(ranking.items(), 1):
        mark = "  <- use" if i <= TOP_K else ""
        print(f"  {i}. {m:<14} {v:5.2f}   {DESCRIPTIONS.get(m, '')}{mark}")
    print(f"\nSuggestion: average the forecasts of {', '.join(ranking.index[:TOP_K])}.")
    return ranking


if __name__ == "__main__":
    main()
