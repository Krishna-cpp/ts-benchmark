"""Charts and REPORT.md for the model-recommender study (all generated, never hand-edited)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES
from .report import FAMILY_COLORS, FAMILY_LABELS, GRID, INK, MUTED, plt

FAMILY = {"Naive": "baseline", "SeasonalNaive": "baseline", "ETS": "statistical",
          "Theta": "statistical", "ARIMA": "statistical", "GBM-level": "ml", "GBM-diff": "ml",
          "TTM": "foundation"}
FEATURE_LABELS = {
    "log_length": "series length (log)", "season": "seasonal period",
    "trend_strength": "trend strength", "seasonal_strength": "seasonal strength",
    "spectral_entropy": "noisiness (spectral entropy)", "diff_acf1": "autocorrelation of changes",
    "remainder_acf1": "autocorrelation of remainder", "lumpiness": "lumpiness (changing variance)",
    "cv": "coefficient of variation", "recent_change": "recent change in level",
}


def _families(Y: pd.DataFrame) -> list[str]:
    return [f for f in FAMILY_COLORS if f in set(Y.columns.map(FAMILY))]


def plot_strategies(summary: dict, path: Path) -> None:
    s = summary["strategies"]
    names = list(s)
    vals = [s[k]["mean_mase"] for k in names]
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.2))
    for y, i in enumerate(order):
        n = names[i]
        model = n.replace("always ", "")
        if n.startswith("oracle"):
            color, hatch = "#d8d7d2", "///"
        elif n.startswith("recommender"):
            color, hatch = INK, None
        elif n.startswith("average"):
            color, hatch = "#6b6a66", None
        elif n.startswith("best single"):
            color, hatch = "#b9b8b3", None
        else:
            color, hatch = FAMILY_COLORS[FAMILY.get(model, "baseline")], None
        ax.barh(y, vals[i], color=color, hatch=hatch, edgecolor="white", height=0.66)
        ax.text(vals[i], y, f"  {vals[i]:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)), [names[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean MASE on held-out series (lower is better)")
    ax.set_title("Picking a model per series vs. always using the same one", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_who_wins(X: pd.DataFrame, Y: pd.DataFrame, path: Path) -> None:
    """Share of series won by each model family, by frequency and by domain."""
    fam = Y.idxmin(axis=1).map(FAMILY)
    fams = _families(Y)
    groups = [("all series", fam)]
    groups += [(f"{v} ({(X['frequency'] == v).sum()})", fam[X["frequency"] == v])
               for v in ["monthly", "quarterly", "yearly"] if (X["frequency"] == v).any()]
    cats = X["category"].value_counts()
    groups += [(f"{c} ({n})", fam[X["category"] == c]) for c, n in cats.items() if n >= 15]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(groups) + 1.4))
    for y, (_, g) in enumerate(groups[::-1]):
        share = g.value_counts(normalize=True)
        left = 0.0
        for f in fams:
            w = float(share.get(f, 0))
            ax.barh(y, w, left=left, color=FAMILY_COLORS[f], edgecolor="white", linewidth=2,
                    height=0.7)
            if w >= 0.08:
                ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white")
            left += w
    ax.set_yticks(range(len(groups)), [g[0] for g in groups[::-1]])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Which kind of model wins, by frequency and by domain", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in fams]
    fig.legend(handles, [FAMILY_LABELS[f] for f in fams], loc="lower center", ncol=len(fams),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_when_wins(X: pd.DataFrame, Y: pd.DataFrame, path: Path,
                   features=("trend_strength", "seasonal_strength", "spectral_entropy")) -> None:
    """Win share of each family across quartiles of a feature."""
    fam = Y.idxmin(axis=1).map(FAMILY)
    fams = _families(Y)
    fig, axes = plt.subplots(1, len(features), figsize=(4.1 * len(features), 3.2), sharey=True)
    for ax, feat in zip(np.atleast_1d(axes), features):
        x = X[feat]
        if x.nunique() < 4:
            continue
        bins = pd.qcut(x.rank(method="first"), 4, labels=["lowest", "low", "high", "highest"])
        share = pd.crosstab(bins, fam, normalize="index")
        for f in fams:
            if f in share:
                ax.plot(range(4), share[f], marker="o", ms=5, lw=2, color=FAMILY_COLORS[f],
                        label=FAMILY_LABELS[f])
        ax.set_xticks(range(4), ["lowest\nquarter", "", "", "highest\nquarter"])
        ax.set_title(FEATURE_LABELS[feat], loc="left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel("share of series won")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(fams), fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("When does each kind of model win?", x=0.01, ha="left", fontsize=10,
                 weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_importance(rf, path: Path) -> None:
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh([FEATURE_LABELS[f] for f in imp.index], imp, color=MUTED, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=GRID)
    ax.set_xlabel("importance in the recommender (impurity decrease)")
    ax.set_title("What the recommender looks at", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def paired_tests(cv: pd.DataFrame) -> list[dict]:
    """Paired comparisons of strategies across held-out series (log MASE ratios)."""
    from scipy.stats import wilcoxon

    single, top = "recommender (single pick)", "recommender (top 3 averaged)"
    pairs = [("recommended_mase", "best_fixed_mase", f"{single} vs. best single model"),
             ("recommended_top_mase", "best_fixed_mase", f"{top} vs. best single model"),
             ("recommended_top_mase", "fixed_combo_mase", f"{top} vs. fixed average"),
             ("fixed_combo_mase", "best_fixed_mase", "fixed average vs. best single model")]
    out = []
    for a, b, label in pairs:
        if a not in cv or b not in cv:
            continue
        d = np.log(cv[a].clip(lower=1e-6)) - np.log(cv[b].clip(lower=1e-6))
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
        out.append({"label": label, "ratio": float(np.exp(d.mean())), "p": p,
                    "better": float((d < 0).mean()), "worse": float((d > 0).mean())})
    return out


def rebuild(out: Path = Path("results/recommender"), model_path: Path | None = None) -> None:
    """Regenerate charts and REPORT.md from a finished study, without rerunning it."""
    import json

    from .recommender import MODEL_PATH, explain_rules, load_model

    table = pd.read_csv(out / "series_table.csv", index_col=0)
    X = table[[c for c in table.columns if not c.startswith("mase_")]]
    Y = table[[c for c in table.columns if c.startswith("mase_")]]
    Y.columns = [c.removeprefix("mase_") for c in Y.columns]
    cv = pd.read_csv(out / "cross_validation.csv", index_col=0)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    rules, summary["rules_accuracy"] = explain_rules(X, Y)
    summary["paired_tests"] = paired_tests(cv)
    (out / "rules.txt").write_text(rules, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rf = load_model(model_path or MODEL_PATH)["model"]
    write_report(X, Y, cv, summary, rules, rf, out)


def write_report(X, Y, cv, summary, rules, rf, out: Path) -> None:
    plot_strategies(summary, out / "strategies.png")
    plot_who_wins(X, Y, out / "who_wins.png")
    plot_when_wins(X, Y, out / "when_wins.png")
    plot_importance(rf, out / "feature_importance.png")

    s = summary["strategies"]
    rows = sorted(s.items(), key=lambda kv: kv[1]["mean_mase"])
    lines = ["# Model recommender: results", "",
             f"_Generated by `python -m tsbench.recommender` on {summary['n_series']} M4 series, "
             f"{summary['folds']} rolling-origin forecasts each, "
             f"in {summary['runtime_seconds'] / 60:.0f} minutes. Do not edit by hand._", "",
             "Every number below is measured on series the recommender did not see during "
             "training (5-fold cross-validation over series).", "",
             "Mean MASE is the usual competition score but a few disastrous series can dominate "
             "it, so the median and the geometric mean are shown too.", "",
             "| Strategy | Mean MASE | Median | Geometric mean |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean_mase']:.3f} | {v['median_mase']:.3f} | {v['geomean_mase']:.3f} |"
              for k, v in rows]
    lines += ["",
              f"* The recommender's pick was within 10% of the best possible model on "
              f"{summary['recommender_within_10pct_of_oracle']:.0%} of series "
              f"(always using {summary['best_fixed_model']}: "
              f"{summary['best_fixed_within_10pct_of_oracle']:.0%}).",
              f"* It beat always-{summary['best_fixed_model']} on "
              f"{summary['recommender_better_than_best_fixed']:.0%} of series and lost on "
              f"{summary['recommender_worse_than_best_fixed']:.0%} (the rest were the same pick).",
              f"* It closed {summary['share_of_gap_to_oracle_closed']:.0%} of the gap between "
              f"always-{summary['best_fixed_model']} and the oracle."]
    if "top_k_better_than_fixed_combo" in summary:
        lines += [f"* Averaging the recommender's top 3 beat the fixed average on "
                  f"{summary['top_k_better_than_fixed_combo']:.0%} of series and beat "
                  f"always-{summary['best_fixed_model']} on "
                  f"{summary['top_k_better_than_best_fixed']:.0%}."]
    tests = paired_tests(cv)
    if tests:
        lines += ["", "## Are the differences real?", "",
                  "Paired Wilcoxon signed-rank test on the log of each series' MASE ratio. "
                  "\"Typical change\" is the geometric mean ratio: 0.97 means 3% lower error.", "",
                  "| Comparison | Typical change | Better on | Worse on | p-value |",
                  "|---|---|---|---|---|"]
        for t in tests:
            p = "<0.001" if t["p"] < 0.001 else f"{t['p']:.3f}"
            lines.append(f"| {t['label']} | {t['ratio']:.3f} | {t['better']:.0%} | "
                         f"{t['worse']:.0%} | {p} |")
    by_freq = Y.groupby(X["frequency"]).mean().T.round(2)
    lines += ["", "## Mean MASE by frequency", "",
              "| Model | " + " | ".join(by_freq.columns) + " |",
              "|---|" + "---|" * len(by_freq.columns)]
    lines += [f"| {m} | " + " | ".join("–" if pd.isna(v) else f"{v:.2f}" for v in row) + " |"
              for m, row in by_freq.sort_values(by_freq.columns[0]).iterrows()]
    lines += ["",
              "![Strategies](strategies.png)", "", "## Who wins where", "",
              "Number of series each model won: "
              + ", ".join(f"{m} {n}" for m, n in summary["winner_counts"].items()) + ".", "",
              "![Who wins](who_wins.png)", "", "![When each family wins](when_wins.png)", "",
              "## Rules of thumb", "",
              "A depth-3 decision tree over which *family* wins, with the families weighted "
              "equally so it shows where each kind of model is most likely to win rather than "
              "just naming the most common winner everywhere (balanced accuracy "
              f"{summary['rules_accuracy']:.0%}; a readable summary, not the recommender itself):",
              "", "```", rules.rstrip(), "```", "",
              "![Feature importance](feature_importance.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
