import numpy as np
import pandas as pd
import pytest

from tsbench.stats import bootstrap_ci, holm, paired_test, summarize


def test_holm_matches_hand_computation():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = holm(p)
    # sorted: a 0.01*3 = 0.03; c 0.03*2 = 0.06; b max(0.06, 0.04*1) = 0.06
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(5, 1, 50)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert np.isnan(bootstrap_ci(np.array([1.0]))[0])


def test_paired_test_detects_consistent_difference():
    a = np.arange(12, dtype=float)
    assert paired_test(a + 1, a) < 0.01
    assert np.isnan(paired_test(a, a))


def test_summarize_ranks_and_flags():
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(12):
        for model, base in [("good", 0.5), ("bad", 1.5)]:
            rows.append(dict(dataset="d", model=model, family="ml", fold=fold,
                             mase=base + rng.uniform(0, 0.1), mae=base, error=None))
    s = summarize(pd.DataFrame(rows)).set_index("model")
    assert s.loc["good", "is_best"] and not s.loc["bad", "is_best"]
    assert s.loc["bad", "significantly_worse"]
    assert s.loc["good", "mean_rank"] == 1.0 and s.loc["bad", "mean_rank"] == 2.0
    assert list(s.index) == ["good", "bad"]  # best first
