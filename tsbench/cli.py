"""tsbench: rolling-origin forecasting benchmark with conformal intervals and significance tests.

    python -m tsbench                       # full benchmark, classical models
    python -m tsbench --with-ttm            # + IBM Granite TinyTimeMixer (zero-shot)
    python -m tsbench --quick               # 3 folds, fast smoke run
    python -m tsbench --datasets co2_monthly --models Naive ETS SARIMA
    python -m tsbench --list                # show datasets and models
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import rolling_origin_backtest
from .data import DATASETS, load_datasets
from .intervals import IntervalConfig
from .models import ALL_MODELS, CLASSICAL_MODELS, build_models
from .report import (
    plot_accuracy_vs_cost,
    plot_error_by_horizon,
    plot_final_fold,
    plot_mase_comparison,
    results_markdown,
    run_metadata,
    write_metadata,
)
from .stats import summarize

log = logging.getLogger("tsbench")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tsbench", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), metavar="NAME",
                   help=f"subset of datasets (default: all of {list(DATASETS)})")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS + ["ARIMA"], metavar="NAME",
                   help=f"subset of models (default: {CLASSICAL_MODELS})")
    p.add_argument("--with-ttm", action="store_true",
                   help="add IBM Granite TinyTimeMixer, zero-shot (TTM) and fine-tuned "
                        "per fold (TTM-FT); needs: pip install -e .[foundation]")
    p.add_argument("--quick", action="store_true", help="3 folds per dataset, 5 calibration "
                                                        "windows: a fast smoke run")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="1 - interval level (default 0.2 -> 80%% intervals)")
    p.add_argument("--calib-windows", type=int, default=10,
                   help="conformal calibration backtests per fold (default 10)")
    p.add_argument("--no-intervals", action="store_true",
                   help="point forecasts only (about 10x faster)")
    p.add_argument("--jobs", type=int, default=-1, help="worker processes (-1 = all cores)")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--list", action="store_true", help="list datasets and models, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def main(argv=None) -> pd.DataFrame | None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.list:
        for ds in load_datasets():
            print(f"{ds.name:<16} n={len(ds):<4} season={ds.season:<3} "
                  f"horizon={ds.horizon:<3} folds={ds.n_folds:<3} {ds.description}")
        print("\nmodels:", ", ".join(ALL_MODELS), "(TTM needs the [foundation] extra)")
        return None

    names = list(args.models or CLASSICAL_MODELS)
    if args.with_ttm:
        names += [m for m in ("TTM", "TTM-FT") if m not in names]
    intervals = None if args.no_intervals else IntervalConfig(
        alpha=args.alpha, n_windows=5 if args.quick else args.calib_windows)

    datasets = load_datasets(args.datasets)
    if args.quick:
        datasets = [d.with_folds(3) for d in datasets]

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    all_folds, all_fc = [], []
    for ds in datasets:
        models = build_models(ds, names)
        calib = f" (+{intervals.n_windows} calibration fits each)" if intervals else ""
        log.info("%s: %d obs, %d models x %d folds%s", ds.name, len(ds), len(models),
                 ds.n_folds, calib)
        t = time.perf_counter()
        res = rolling_origin_backtest(ds, models, intervals=intervals, n_jobs=args.jobs)
        log.info("%s done in %.1fs", ds.name, time.perf_counter() - t)
        all_folds.append(res.folds)
        all_fc.append(res.forecasts)

    folds = pd.concat(all_folds, ignore_index=True)
    forecasts = pd.concat(all_fc, ignore_index=True)
    summary = summarize(folds)
    runtime = time.perf_counter() - t0

    folds.to_csv(args.out / "fold_results.csv", index=False)
    forecasts.to_csv(args.out / "forecasts.csv", index=False)
    summary.round(4).to_csv(args.out / "summary.csv", index=False)
    config = dict(datasets=[d.name for d in datasets], models=names, quick=args.quick,
                  alpha=None if intervals is None else intervals.alpha,
                  calib_windows=None if intervals is None else intervals.n_windows)
    meta = run_metadata(config, runtime)
    write_metadata(meta, args.out / "run_metadata.json")

    alpha = None if intervals is None else intervals.alpha
    if not args.no_plots:
        plot_mase_comparison(summary, folds, args.out / "mase_comparison.png")
        for ds in datasets:
            plot_final_fold(ds, forecasts, summary, args.out / f"forecasts_{ds.name}.png", alpha)
        plot_accuracy_vs_cost(summary, args.out / "accuracy_vs_cost.png")
        plot_error_by_horizon(forecasts, summary, args.out / "error_by_horizon.png")
    (args.out / "RESULTS.md").write_text(
        results_markdown(summary, datasets, folds, alpha, meta), encoding="utf-8")

    cols = [c for c in ["dataset", "model", "mase", "ci_low", "ci_high", "mean_rank",
                        "p_vs_best", "coverage", "msis", "fit_seconds"] if c in summary]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n" + summary[cols].round(3).to_string(index=False))
    n_err = folds["error"].notna().sum()
    if n_err:
        log.warning("%d model/fold evaluations failed - see %s", n_err, args.out / "RESULTS.md")
    log.info("finished in %.1fs, results in %s/", runtime, args.out)
    return summary


if __name__ == "__main__":
    main()
