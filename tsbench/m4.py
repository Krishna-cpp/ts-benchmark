"""A reproducible sample of the M4 competition data (Makridakis et al., 2020).

M4 has 100,000 real series from business, finance, demography, industry and
macro-economics. The model recommender needs many *different* series to learn
"which model works on which kind of series", so this module draws a fixed,
seeded sample of monthly, quarterly and yearly series and turns each one into
the same :class:`~tsbench.data.Dataset` the rest of the benchmark uses.

The raw CSVs (~170 MB) are downloaded once into ``data/m4/`` (git-ignored):

    python -m tsbench.m4 --download
"""
from __future__ import annotations

import argparse
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset

log = logging.getLogger(__name__)

M4_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
DATA_DIR = Path("data/m4")


@dataclass(frozen=True)
class Frequency:
    name: str  # M4 file prefix
    season: int
    horizon: int  # the official M4 forecast horizon
    n_lags: int  # input window for the lag-based ML models
    min_length: int  # skip series shorter than this (train + test)
    max_length: int  # keep only the most recent observations (keeps fits fast)
    seasonal_order: tuple[int, int, int]


FREQUENCIES = {
    "monthly": Frequency("Monthly", 12, 18, 24, 144, 360, (0, 1, 1)),
    "quarterly": Frequency("Quarterly", 4, 8, 8, 64, 160, (0, 1, 1)),
    "yearly": Frequency("Yearly", 0, 6, 6, 40, 100, (0, 0, 0)),
}


def download(data_dir: Path = DATA_DIR) -> None:
    """Fetch the M4 train/test CSVs and the info file (skips files already present)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = ["M4-info.csv"] + [f"{f.name}-{split}.csv" for f in FREQUENCIES.values()
                               for split in ("train", "test")]
    for name in files:
        target = data_dir / name
        if target.exists() and target.stat().st_size > 0:
            continue
        sub = "" if name == "M4-info.csv" else ("Train/" if "train" in name else "Test/")
        log.info("downloading %s", name)
        urllib.request.urlretrieve(f"{M4_URL}/{sub}{name}", target)


def _read_wide(path: Path) -> dict[str, np.ndarray]:
    """M4 CSVs are wide: one row per series, trailing NaNs where a series ends."""
    df = pd.read_csv(path, index_col=0)
    return {sid: row.dropna().to_numpy(dtype=float) for sid, row in df.iterrows()}


def load_sample(n_per_frequency: dict[str, int] | None = None, seed: int = 0,
                data_dir: Path = DATA_DIR, n_folds: int = 3) -> list[Dataset]:
    """Seeded random sample of M4 series, stratified by frequency.

    Each series is the M4 training part followed by its official test part, so
    the last rolling-origin fold is exactly the M4 holdout.
    """
    n_per_frequency = n_per_frequency or {"monthly": 120, "quarterly": 90, "yearly": 90}
    if not (data_dir / "M4-info.csv").exists():
        raise FileNotFoundError(
            f"M4 data not found in {data_dir}; run: python -m tsbench.m4 --download")
    info = pd.read_csv(data_dir / "M4-info.csv", index_col="M4id")
    rng = np.random.default_rng(seed)
    out: list[Dataset] = []
    for key, n in n_per_frequency.items():
        f = FREQUENCIES[key]
        # enough history for every fold, for the ML lags and for the features (>= 24 points)
        min_len = max(f.min_length, f.horizon * n_folds + max(24, 2 * f.n_lags + 2))
        train = _read_wide(data_dir / f"{f.name}-train.csv")
        test = _read_wide(data_dir / f"{f.name}-test.csv")
        ids = sorted(sid for sid in train
                     if len(train[sid]) + len(test[sid]) >= min_len
                     and np.ptp(train[sid]) > 0)
        for sid in rng.choice(ids, size=min(n, len(ids)), replace=False):
            y = np.concatenate([train[sid], test[sid]])[-f.max_length:]
            category = str(info.loc[sid, "category"]).strip().lower()
            out.append(Dataset(
                name=f"m4_{sid}", series=pd.Series(y, name="value"),
                description=f"M4 {key} series {sid} ({category})",
                season=f.season, horizon=f.horizon, n_folds=n_folds, step=f.horizon,
                scale_lag=f.season or 1, n_lags=f.n_lags,
                arima_order=(1, 1, 1), seasonal_order=f.seasonal_order,
            ))
    log.info("sampled %d M4 series", len(out))
    return out


def frequency_of(ds: Dataset) -> str:
    return {12: "monthly", 4: "quarterly", 0: "yearly"}.get(ds.season, "other")


def category_of(ds: Dataset) -> str:
    return ds.description.rsplit("(", 1)[-1].rstrip(")") if "(" in ds.description else "other"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tsbench.m4", description="Download the M4 data.")
    p.add_argument("--download", action="store_true")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if args.download:
        download(args.data_dir)


if __name__ == "__main__":
    main()
