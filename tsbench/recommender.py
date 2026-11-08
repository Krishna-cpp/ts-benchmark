"""Which forecasting model should I use for *this* series?

The three-dataset benchmark shows that the best model depends on the data. This
module turns that observation into a tool:

1. Run the rolling-origin benchmark on a few hundred real M4 series
   (business, finance, demographic, industry, macro).
2. Describe every series with interpretable features (``features.py``),
   computed only on data before the first forecast origin.
3. Learn a model that predicts each forecaster's error from those features,
   and recommend the forecaster with the lowest predicted error.
4. Evaluate it with cross-validation *over series*: on series the recommender
   never saw, is its pick better than always using the single best model?

    python -m tsbench.recommender                 # 300 series, classical models + TTM
    python -m tsbench.recommender --n-series 60   # quick run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_text

from .backtest import rolling_origin_backtest
from .data import Dataset
from .features import FEATURES, series_features
from .models import BASELINE, FOUNDATION, ML, STATISTICAL, build_models

log = logging.getLogger("tsbench.recommender")

CANDIDATES = ["Naive", "SeasonalNaive", "ETS", "Theta", "SARIMA", "GBM-level", "GBM-diff"]
FAMILY = {"Naive": BASELINE, "SeasonalNaive": BASELINE, "ETS": STATISTICAL, "Theta": STATISTICAL,
          "ARIMA": STATISTICAL, "GBM-level": ML, "GBM-diff": ML, "TTM": FOUNDATION}
MODEL_PATH = Path("models/recommender.joblib")


# --------------------------------------------------------------------------- study
def _backtest_one(ds: Dataset, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        res = rolling_origin_backtest(ds, build_models(ds, names), intervals=None, n_jobs=1)
    except ValueError as e:  # e.g. series too short for the requested folds: skip it
        logging.getLogger("tsbench.recommender").warning("skipping %s: %s", ds.name, e)
        return pd.DataFrame(), pd.DataFrame()
    return res.folds, res.forecasts


COMBO = ["ETS", "ARIMA", "Theta", "GBM-diff"]  # fixed averaging baseline, chosen up front
TOP_K = 3  # how many of the recommender's picks to average, chosen up front


def run_study(datasets: list[Dataset], with_ttm: bool = True,
              n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold scores and per-step forecasts for every candidate model on every series."""
    t0 = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        parts = Parallel(n_jobs=n_jobs)(delayed(_backtest_one)(ds, CANDIDATES) for ds in datasets)
    log.info("classical models: %d series in %.0fs", len(datasets), time.perf_counter() - t0)
    if with_ttm:
        from .foundation import GraniteTTM

        ttm = GraniteTTM()  # one instance, weights loaded once, reused for every series
        t1 = time.perf_counter()
        for ds in datasets:
            res = rolling_origin_backtest(ds, [ttm], intervals=None, n_jobs=1)
            parts.append((res.folds, res.forecasts))
        log.info("TTM zero-shot: %d series in %.0fs", len(datasets), time.perf_counter() - t1)
    folds = pd.concat([f for f, _ in parts], ignore_index=True)
    forecasts = pd.concat([fc for _, fc in parts], ignore_index=True)
    # one name for the ARIMA family (seasonal or not), so every series has the same columns
    for df in (folds, forecasts):
        df["model"] = df["model"].replace({"SARIMA": "ARIMA"})
    return folds, forecasts


class Combiner:
    """MASE of the plain average of several models' forecasts, per series."""

    def __init__(self, forecasts: pd.DataFrame):
        key = ["dataset", "fold", "step"]
        self.pred = forecasts.pivot_table(index=key, columns="model", values="pred")
        self.truth = forecasts.groupby(key)[["actual", "scale"]].first()

    def mase(self, series: str, models: list[str]) -> float:
        p = self.pred.loc[series, models].mean(axis=1)
        t = self.truth.loc[series]
        err = (t["actual"] - p).abs() / t["scale"]
        return float(err.groupby(level="fold").mean().mean())


def feature_table(datasets: list[Dataset]) -> pd.DataFrame:
    """Features per series, from data before the first forecast origin only."""
    from .m4 import category_of, frequency_of

    rows = []
    for ds in datasets:
        f = series_features(ds.values[:ds.first_origin], ds.season)
        rows.append({"dataset": ds.name, "frequency": frequency_of(ds),
                     "category": category_of(ds), **f})
    return pd.DataFrame(rows).set_index("dataset")


def error_table(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE per (series, model); NaN where a model does not apply or failed."""
    ok = folds[folds["error"].isna()]
    return ok.pivot_table(index="dataset", columns="model", values="mase", aggfunc="mean")


# --------------------------------------------------------------------------- learning
def _fill_unavailable(y: pd.DataFrame) -> pd.DataFrame:
    """Models that do not apply (e.g. SeasonalNaive on yearly data) get an error
    well above the worst real one, so they are never the predicted best."""
    worst = y.max(axis=1)
    return y.apply(lambda col: col.fillna(worst * 2))


def fit_recommender(X: pd.DataFrame, Y: pd.DataFrame, seed: int = 0) -> RandomForestRegressor:
    """Random forest predicting log-MASE of every model at once from the features."""
    target = np.log(_fill_unavailable(Y).clip(lower=1e-3))
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.6,
                               random_state=seed, n_jobs=-1)
    return rf.fit(X[FEATURES].to_numpy(), target.to_numpy())


def recommend_from_features(rf, features: pd.DataFrame, models: list[str],
                            available: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predicted MASE for every model, per series (lower = better)."""
    pred = pd.DataFrame(np.exp(rf.predict(features[FEATURES].to_numpy())),
                        index=features.index, columns=models)
    if available is not None:  # never recommend a model that cannot run on the series
        pred = pred.where(available.reindex(index=pred.index, columns=models).notna())
    return pred


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5,
                   seed: int = 0, forecasts: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each series, held out in turn: what each strategy would have achieved.

    With ``forecasts`` it also scores averaging strategies: the recommender's top
    ``TOP_K`` models, and a fixed average of ``COMBO`` for every series.
    """
    models = list(Y.columns)
    combiner = Combiner(forecasts) if forecasts is not None else None
    rows = []
    for train_idx, test_idx in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        tr, te = X.index[train_idx], X.index[test_idx]
        rf = fit_recommender(X.loc[tr], Y.loc[tr], seed)
        pred = recommend_from_features(rf, X.loc[te], models, available=Y.loc[te])
        # what you'd pick without a recommender: the best model among those that run on
        # every series (SeasonalNaive can't do yearly data, so it can't be the default)
        complete = [m for m in models if Y[m].notna().all()]
        best_fixed = Y.loc[tr, complete].mean().idxmin()
        for s in te:
            actual = Y.loc[s]
            pick = pred.loc[s].idxmin()
            fixed = best_fixed if pd.notna(actual.get(best_fixed)) else actual.idxmin()
            row = {"dataset": s, "recommended": pick, "recommended_mase": actual[pick],
                   "best_fixed_model": fixed, "best_fixed_mase": actual[fixed],
                   "oracle_model": actual.idxmin(), "oracle_mase": actual.min()}
            if combiner is not None:
                top = list(pred.loc[s].dropna().nsmallest(TOP_K).index)
                row.update(recommended_top=" + ".join(top),
                           recommended_top_mase=combiner.mase(s, top),
                           fixed_combo_mase=combiner.mase(s, [m for m in COMBO if m in models]))
            rows.append(row)
    return pd.DataFrame(rows).set_index("dataset").loc[X.index]


def explain_rules(X: pd.DataFrame, Y: pd.DataFrame, max_depth: int = 3) -> tuple[str, float]:
    """A small decision tree on 'which *family* wins' - the human-readable version."""
    labels = Y.idxmin(axis=1).map(FAMILY)
    # balanced classes: otherwise "statistical" (the most common winner) fills every leaf
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=12, random_state=0,
                                  class_weight="balanced")
    tree.fit(X[FEATURES], labels)
    rules = export_text(tree, feature_names=FEATURES, decimals=2)
    return rules, float(balanced_accuracy_score(labels, tree.predict(X[FEATURES])))


def summarize(cv: pd.DataFrame, Y: pd.DataFrame) -> dict:
    strategies = {"oracle (knows the winner in advance)": cv["oracle_mase"],
                  "recommender, single pick": cv["recommended_mase"],
                  "best single model, always": cv["best_fixed_mase"]}
    if "recommended_top_mase" in cv:
        strategies[f"recommender, average of its top {TOP_K}"] = cv["recommended_top_mase"]
        strategies["average of " + ", ".join(COMBO) + ", always"] = cv["fixed_combo_mase"]
    for m in Y.columns:
        if Y[m].notna().all():
            strategies[f"always {m}"] = Y[m]
    table = {k: {"mean_mase": float(v.mean()), "median_mase": float(v.median()),
                 "geomean_mase": float(np.exp(np.log(v.clip(lower=1e-6)).mean()))}
             for k, v in strategies.items()}
    near = (cv["recommended_mase"] <= 1.10 * cv["oracle_mase"]).mean()
    near_fixed = (cv["best_fixed_mase"] <= 1.10 * cv["oracle_mase"]).mean()
    beats = (cv["recommended_mase"] < cv["best_fixed_mase"]).mean()
    loses = (cv["recommended_mase"] > cv["best_fixed_mase"]).mean()
    gap_closed = ((cv["best_fixed_mase"].mean() - cv["recommended_mase"].mean())
                  / (cv["best_fixed_mase"].mean() - cv["oracle_mase"].mean()))
    extra = {}
    if "recommended_top_mase" in cv:
        extra = {"top_k_better_than_fixed_combo": float(
                     (cv["recommended_top_mase"] < cv["fixed_combo_mase"]).mean()),
                 "top_k_better_than_best_fixed": float(
                     (cv["recommended_top_mase"] < cv["best_fixed_mase"]).mean())}
    return {"n_series": int(len(cv)), "strategies": table, **extra,
            "recommender_within_10pct_of_oracle": float(near),
            "best_fixed_within_10pct_of_oracle": float(near_fixed),
            "recommender_better_than_best_fixed": float(beats),
            "recommender_worse_than_best_fixed": float(loses),
            "share_of_gap_to_oracle_closed": float(gap_closed),
            "best_fixed_model": cv["best_fixed_model"].mode().iloc[0],
            "winner_counts": Y.idxmin(axis=1).value_counts().to_dict()}


# --------------------------------------------------------------------------- inference
def save_model(rf, models: list[str], path: Path = MODEL_PATH, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "models": models, "features": FEATURES, "meta": meta or {}}, path,
                compress=3)


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load a recommender this project trained itself.

    joblib files are pickles, so never load one from an untrusted source.
    """
    return joblib.load(path)


def recommend(y, season: int, bundle: dict | None = None) -> pd.Series:
    """Ranked predicted MASE per model for a new series (best first)."""
    bundle = bundle or load_model()
    f = pd.DataFrame([series_features(np.asarray(y, dtype=float), season)])
    pred = recommend_from_features(bundle["model"], f, bundle["models"]).iloc[0]
    if not season and "SeasonalNaive" in pred:
        pred = pred.drop("SeasonalNaive")
    return pred.sort_values()


# --------------------------------------------------------------------------- cli
def main(argv=None) -> dict:
    from .m4 import load_sample
    from .recommender_report import paired_tests, write_report

    p = argparse.ArgumentParser(prog="tsbench.recommender", description=__doc__.split("\n\n")[0])
    p.add_argument("--n-series", type=int, default=600, help="total M4 series (split 40/30/30 "
                   "monthly/quarterly/yearly)")
    p.add_argument("--folds", type=int, default=5, help="rolling-origin folds per series")
    p.add_argument("--no-ttm", action="store_true", help="skip IBM Granite TTM")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", type=Path, default=Path("results/recommender"))
    p.add_argument("--model-out", type=Path, default=MODEL_PATH)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("tsbench").setLevel(logging.INFO)
    t0 = time.perf_counter()
    n = args.n_series
    split = {"monthly": round(0.4 * n), "quarterly": round(0.3 * n)}
    split["yearly"] = n - sum(split.values())
    datasets = load_sample(split, seed=args.seed, n_folds=args.folds)

    folds, forecasts = run_study(datasets, with_ttm=not args.no_ttm, n_jobs=args.jobs)
    X = feature_table(datasets)
    Y = error_table(folds).loc[X.index]
    keep = Y.notna().sum(axis=1) >= 3  # drop series where most models failed
    X, Y = X[keep], Y[keep]

    cv = cross_validate(X, Y, seed=args.seed, forecasts=forecasts)
    rules, rule_acc = explain_rules(X, Y)
    summary = summarize(cv, Y)
    summary.update(rules_accuracy=rule_acc, runtime_seconds=time.perf_counter() - t0,
                   paired_tests=paired_tests(cv),
                   folds=args.folds, seed=args.seed, with_ttm=not args.no_ttm)

    args.out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out / "study_folds.csv", index=False)
    X.join(Y.add_prefix("mase_")).to_csv(args.out / "series_table.csv")
    cv.to_csv(args.out / "cross_validation.csv")
    (args.out / "rules.txt").write_text(rules, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rf = fit_recommender(X, Y, args.seed)
    save_model(rf, list(Y.columns), args.model_out,
               meta={"n_series": len(X), "seed": args.seed, "with_ttm": not args.no_ttm})
    write_report(X, Y, cv, summary, rules, rf, args.out)

    for name, v in sorted(summary["strategies"].items(), key=lambda kv: kv[1]["mean_mase"]):
        log.info("%-55s mean %.3f  median %.3f  geomean %.3f", name, v["mean_mase"],
                 v["median_mase"], v["geomean_mase"])
    log.info("done in %.0fs, report in %s/REPORT.md", summary["runtime_seconds"], args.out)
    return summary


if __name__ == "__main__":
    main()
