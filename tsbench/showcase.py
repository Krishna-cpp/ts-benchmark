"""Build the visual showcase for the GitHub page from an existing results/ folder.

    python -m tsbench.showcase            # reads results/, writes docs/

Outputs (all regenerated from the benchmark output, never edited by hand):

    docs/hero_trees.png        the headline finding in one picture
    docs/rolling_origin.gif    animation explaining rolling-origin backtesting
    docs/social_preview.png    1280x640 card for GitHub / LinkedIn link previews
    docs/index.html            interactive results explorer (GitHub Pages ready)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_datasets
from .report import FAMILY_COLORS, GRID, INK, MUTED, plt

log = logging.getLogger("tsbench.showcase")


def _dates(series: pd.Series) -> list[str]:
    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    return [d.strftime("%Y-%m-%d") for d in idx]


# --------------------------------------------------------------------------- hero
def hero_trees(forecasts: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """GBM on levels hits an invisible ceiling; the same model on differences doesn't."""
    ds = load_datasets(["co2_monthly"])[0]
    y, dates = ds.values, ds.series.index
    fc = forecasts[(forecasts["dataset"] == ds.name)]
    fold = fc["fold"].max()
    fc = fc[fc["fold"] == fold]
    origin = int(fc["origin"].iloc[0])
    ctx = 60
    hist_x = dates[origin - ctx:origin]
    fut_x = dates[origin:origin + ds.horizon]
    train_max = y[:origin].max()
    s = summary[summary["dataset"] == ds.name].set_index("model")["mase"]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(hist_x, y[origin - ctx:origin], color=MUTED, lw=1.5)
    ax.plot(fut_x, y[origin:origin + ds.horizon], color=INK, lw=2.6, label="what happened")
    lvl = fc[fc["model"] == "GBM-level"].sort_values("step")["pred"]
    dif = fc[fc["model"] == "GBM-diff"].sort_values("step")["pred"]
    ax.plot(fut_x, lvl, color="#e34948", lw=2.4,
            label=f"GBM on levels   (mean MASE {s['GBM-level']:.2f})")
    ax.plot(fut_x, dif, color=FAMILY_COLORS["ml"], lw=2.4,
            label=f"GBM on changes (mean MASE {s['GBM-diff']:.2f})")
    ax.axhline(train_max, color="#e34948", ls=":", lw=1.3)
    ax.annotate("highest value the model ever saw in training", (hist_x[-30], train_max),
                xytext=(0, 5), textcoords="offset points", color="#e34948", fontsize=9)
    ax.axvline(dates[origin], color=GRID, lw=1.2)
    ax.annotate("forecast starts", (dates[origin], ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_ylabel("CO$_2$ at Mauna Loa (ppm)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, frameon=True, edgecolor=GRID)
    ax.set_title("A tree model can't predict a number it has never seen. "
                 "Teaching it changes instead of levels fixes that.",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- gif
def rolling_origin_gif(forecasts: pd.DataFrame, path: Path, model: str = "SARIMA") -> None:
    """One frame per fold: training window grows, the model forecasts the next year."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    ds = load_datasets(["co2_monthly"])[0]
    y, dates = ds.values, ds.series.index
    fc = forecasts[(forecasts["dataset"] == ds.name) & (forecasts["model"] == model)]
    folds = sorted(fc["fold"].unique())
    start = int(fc["origin"].min()) - 120
    end = len(y)

    fig, ax = plt.subplots(figsize=(8, 3.6))

    def draw(k):
        ax.clear()
        g = fc[fc["fold"] == folds[k]].sort_values("step")
        o = int(g["origin"].iloc[0])
        fx = dates[o:o + ds.horizon]
        ax.axvspan(dates[start], dates[o - 1], color="#2a78d6", alpha=0.07, lw=0)
        ax.plot(dates[start:o], y[start:o], color=MUTED, lw=1.2)
        ax.plot(dates[o + ds.horizon:end], y[o + ds.horizon:end], color=GRID, lw=1.2)
        ax.plot(fx, y[o:o + ds.horizon], color=INK, lw=2.2, label="actual (hidden from model)")
        ax.fill_between(fx, g["lo"], g["hi"], color="#2a78d6", alpha=0.22, lw=0,
                        label="80% interval")
        ax.plot(fx, g["pred"], color="#2a78d6", lw=2.2, label=f"{model} forecast")
        err = np.mean(np.abs(g["actual"] - g["pred"]) / g["scale"])
        ax.set_title(f"Fold {k + 1}/{len(folds)}: train on everything before "
                     f"{dates[o].strftime('%b %Y')}, forecast the next 12 months  "
                     f"(MASE {err:.2f})", loc="left", fontsize=9)
        ax.text(dates[start + 4], y[start:end].max(), "training data", color="#2a78d6",
                fontsize=8, va="top")
        ax.set_ylim(y[start:end].min() - 1, y[start:end].max() + 1.5)
        ax.set_ylabel("CO$_2$ (ppm)")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(folds))
    anim.save(path, writer=PillowWriter(fps=1.2), dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- social card
def social_preview(summary: pd.DataFrame, path: Path) -> None:
    """1280x640 link-preview card: the question, then one tile per dataset with its winner."""
    bg = "#f6f7f9"
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(bg)
    fig.text(0.05, 0.80, "tsbench", fontsize=44, weight="bold", color=INK)
    fig.text(0.05, 0.69, "Which forecasting model wins, and can you trust the answer?",
             fontsize=21, color=INK)
    fig.text(0.05, 0.62, "Statistics vs. gradient boosting vs. IBM Granite TinyTimeMixer  ·  "
             "rolling-origin backtests  ·  conformal intervals  ·  significance tests",
             fontsize=12, color=MUTED)
    fig.text(0.95, 0.83, "by Krishna  ·  github.com/Krishna-cpp", fontsize=13, color=MUTED,
             ha="right")
    datasets = list(summary["dataset"].unique())
    tile_w = 0.9 / len(datasets)
    for i, ds in enumerate(datasets):
        s = summary[summary["dataset"] == ds].sort_values("mase")
        ax = fig.add_axes([0.05 + i * tile_w, 0.08, tile_w - 0.03, 0.42])
        ax.set_facecolor(bg)
        top = s.head(5).iloc[::-1]
        ax.barh(range(len(top)), top["mase"], color=[FAMILY_COLORS[f] for f in top["family"]],
                height=0.7)
        for y, (_, r) in enumerate(top.iterrows()):
            ax.text(r["mase"] + s["mase"].head(5).max() * 0.02, y, f"{r['model']}  {r['mase']:.2f}",
                    va="center", fontsize=11, color=INK)
        ax.set_xlim(0, s["mase"].head(5).max() * 1.7)
        ax.set_title(ds.replace("_", " ") + "  ·  top 5 by MASE", loc="left", fontsize=12,
                     color=INK, weight="bold")
        ax.axis("off")
        ax.title.set_visible(True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


# --------------------------------------------------------------------------- explorer
PAGE_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
             '<meta name="viewport" content="width=device-width, initial-scale=1, '
             'viewport-fit=cover">\n')


def explorer_html(forecasts: pd.DataFrame, summary: pd.DataFrame, path: Path,
                  standalone: bool = True) -> None:
    """Self-contained interactive page: data is embedded, Plotly comes from a CDN.

    ``standalone`` wraps the fragment in a full HTML document (for GitHub Pages).
    """
    data = {"datasets": {}, "summary": json.loads(summary.round(4).to_json(orient="records"))}
    for ds in load_datasets():
        fc = forecasts[forecasts["dataset"] == ds.name]
        if fc.empty:
            continue
        models = {}
        for (m, f), g in fc.sort_values("step").groupby(["model", "fold"]):
            models.setdefault(m, {})[int(f)] = {
                "origin": int(g["origin"].iloc[0]),
                "pred": [None if not np.isfinite(v) else round(float(v), 4) for v in g["pred"]],
                "lo": [None if not np.isfinite(v) else round(float(v), 4) for v in g["lo"]],
                "hi": [None if not np.isfinite(v) else round(float(v), 4) for v in g["hi"]],
            }
        data["datasets"][ds.name] = {
            "description": ds.description, "horizon": ds.horizon, "unit": ds.series.name,
            "scale_lag": ds.scale_lag,
            "dates": _dates(ds.series), "values": [round(float(v), 4) for v in ds.values],
            "folds": sorted(int(f) for f in fc["fold"].unique()), "models": models,
        }
    template = (Path(__file__).parent / "explorer_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    if standalone:
        head, body = html.split("</style>", 1)
        html = PAGE_HEAD + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    path.write_text(html, encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="tsbench.showcase", description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("docs"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    forecasts = pd.read_csv(args.results / "forecasts.csv")
    summary = pd.read_csv(args.results / "summary.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    hero_trees(forecasts, summary, args.out / "hero_trees.png")
    rolling_origin_gif(forecasts, args.out / "rolling_origin.gif")
    social_preview(summary, args.out / "social_preview.png")
    explorer_html(forecasts, summary, args.out / "index.html")
    log.info("showcase written to %s/", args.out)


if __name__ == "__main__":
    main()
