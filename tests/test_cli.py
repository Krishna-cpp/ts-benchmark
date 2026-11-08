import json

import pandas as pd
import pytest

from tsbench.cli import main


@pytest.mark.slow
def test_end_to_end_quick_run(tmp_path):
    summary = main(["--quick", "--datasets", "sunspots_yearly", "--models", "Naive", "ARIMA",
                    "--calib-windows", "3", "--jobs", "1", "--out", str(tmp_path)])
    for f in ["fold_results.csv", "forecasts.csv", "summary.csv", "RESULTS.md",
              "run_metadata.json", "mase_comparison.png", "forecasts_sunspots_yearly.png"]:
        assert (tmp_path / f).exists(), f
    assert set(summary["model"]) == {"Naive", "ARIMA"}
    folds = pd.read_csv(tmp_path / "fold_results.csv")
    assert len(folds) == 2 * 3
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["config"]["quick"] is True


def test_list_does_not_run(capsys):
    assert main(["--list"]) is None
    assert "co2_monthly" in capsys.readouterr().out
