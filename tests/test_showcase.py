import json

import pytest

from tsbench.cli import main as run_benchmark
from tsbench.showcase import main as build_showcase


@pytest.mark.slow
def test_showcase_builds_from_results(tmp_path):
    results, docs = tmp_path / "results", tmp_path / "docs"
    run_benchmark(["--quick", "--datasets", "co2_monthly", "--models", "Naive", "SARIMA",
                   "GBM-level", "GBM-diff", "--calib-windows", "3", "--no-plots",
                   "--out", str(results)])
    build_showcase(["--results", str(results), "--out", str(docs)])
    for f in ["hero_trees.png", "rolling_origin.gif", "social_preview.png", "index.html"]:
        assert (docs / f).stat().st_size > 1000, f

    html = (docs / "index.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "/*__DATA__*/null" not in html
    # the embedded data is valid JSON and contains every model's forecasts
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    data = json.loads(blob)
    assert set(data["datasets"]["co2_monthly"]["models"]) == {"Naive", "SARIMA", "GBM-level",
                                                              "GBM-diff"}
