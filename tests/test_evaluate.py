"""Metrics and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import src.evaluate as ev


def test_metrics_on_a_known_case():
    actual = np.array([100.0, 200.0])
    pred = np.array([110.0, 180.0])
    assert ev.mape(actual, pred) == pytest.approx(10.0)
    assert ev.mae(actual, pred) == pytest.approx(15.0)
    assert ev.rmse(actual, pred) == pytest.approx(np.sqrt(250.0))


def test_dm_at_horizon_one_is_the_paired_t_test():
    rng = np.random.default_rng(0)
    e1, e2 = rng.normal(0, 1.0, 300), rng.normal(0, 1.3, 300)
    res = ev.diebold_mariano(e1, e2, h=1)
    ref = stats.ttest_1samp(e1**2 - e2**2, 0.0)
    assert res["statistic"] == pytest.approx(ref.statistic)
    assert res["p_value"] == pytest.approx(ref.pvalue)


def test_dm_variance_uses_h_minus_1_autocovariances_with_equal_weights():
    rng = np.random.default_rng(1)
    e1, e2 = rng.normal(0, 1.0, 400), rng.normal(0, 1.2, 400)
    h = 3
    d = e1**2 - e2**2
    n, dc = len(d), d - d.mean()
    gamma = [np.mean(dc[: n - k] * dc[k:]) for k in range(h)]
    var = (gamma[0] + 2 * (gamma[1] + gamma[2])) / n
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    expected = d.mean() / np.sqrt(var) * hln
    assert ev.diebold_mariano(e1, e2, h=h)["statistic"] == pytest.approx(expected)


def test_dm_bartlett_weights_shrink_the_autocovariances_linearly():
    rng = np.random.default_rng(4)
    d = rng.normal(0.1, 1.0, 400)
    h = 4
    n, dc = len(d), d - d.mean()
    gamma = [np.mean(dc[: n - k] * dc[k:]) for k in range(h)]
    var = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    expected = d.mean() / np.sqrt(var) * hln
    assert ev.dm_from_differential(d, h=h, bartlett=True)["statistic"] == pytest.approx(expected)


def test_headline_dm_is_the_squared_error_differential():
    rng = np.random.default_rng(5)
    e1, e2 = rng.normal(0, 1.0, 300), rng.normal(0, 1.5, 300)
    assert ev.diebold_mariano(e1, e2) == ev.dm_from_differential(e1**2 - e2**2)


def test_dm_prefers_the_model_with_smaller_errors():
    rng = np.random.default_rng(2)
    good, bad = rng.normal(0, 1.0, 500), rng.normal(0, 2.0, 500)
    forward = ev.diebold_mariano(good, bad)
    backward = ev.diebold_mariano(bad, good)
    assert forward["statistic"] < 0 < backward["statistic"]
    assert forward["statistic"] == pytest.approx(-backward["statistic"])
    assert forward["p_value"] < 0.001


def test_dm_does_not_reject_for_equally_accurate_models():
    rng = np.random.default_rng(3)
    res = ev.diebold_mariano(rng.normal(0, 1.0, 500), rng.normal(0, 1.0, 500))
    assert res["p_value"] > 0.05
