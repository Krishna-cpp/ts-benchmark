"""Turning per-fold scores into conclusions we can defend.

A difference in mean MASE is only interesting if it is larger than the noise
between folds. For every dataset we therefore report:

* mean and std of the metric across folds,
* a 95% bootstrap confidence interval for the mean,
* the mean rank of each model (rank 1 = best in that fold),
* a paired Wilcoxon signed-rank test of each model against the best model,
  Holm-corrected for the number of comparisons.

Folds are paired (all models see the exact same train/test split), which is
why a paired test is the right tool and is much more sensitive than comparing
two independent means.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm-Bonferroni step-down adjustment (NaNs are left untouched)."""
    p = pvalues.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(p.to_numpy())])
    out = pd.Series(np.minimum(adj, 1.0), index=p.index)
    return out.reindex(pvalues.index)


def paired_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples; NaN if undefined."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(wilcoxon(d).pvalue)


def summarize(folds: pd.DataFrame, metric: str = "mase", alpha: float = 0.05,
              seed: int = 0) -> pd.DataFrame:
    """One row per (dataset, model), best model first within each dataset."""
    ok = folds[folds["error"].isna()] if "error" in folds else folds
    extra = [c for c in ["mae", "rmse", "coverage", "width", "msis", "fit_seconds"] if c in ok]
    rows = []
    for ds, g in ok.groupby("dataset", sort=False):
        wide = g.pivot(index="fold", columns="model", values=metric)
        ranks = wide.rank(axis=1).mean()
        means = wide.mean()
        best = means.idxmin()
        pvals = pd.Series({m: paired_test(wide[m], wide[best]) for m in wide.columns})
        padj = holm(pvals.drop(best)).reindex(wide.columns)
        agg = g.groupby("model")[extra].mean() if extra else None
        if agg is not None and "fit_seconds" in agg:
            # median: robust to one-off costs such as loading model weights on the first call
            agg["fit_seconds"] = g.groupby("model")["fit_seconds"].median()
        for m in wide.columns:
            lo, hi = bootstrap_ci(wide[m].to_numpy(), seed=seed)
            row = {"dataset": ds, "model": m,
                   "family": g.loc[g["model"] == m, "family"].iloc[0],
                   "n_folds": int(wide[m].notna().sum()), metric: means[m],
                   f"{metric}_std": wide[m].std(), "ci_low": lo, "ci_high": hi,
                   "mean_rank": ranks[m], "p_vs_best": padj[m], "is_best": m == best}
            row["significantly_worse"] = bool(padj[m] < alpha) if np.isfinite(padj[m]) else False
            if agg is not None:
                row.update(agg.loc[m].to_dict())
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {ds: i for i, ds in enumerate(ok["dataset"].unique())}
    return (out.assign(_d=out["dataset"].map(order))
               .sort_values(["_d", metric]).drop(columns="_d").reset_index(drop=True))
