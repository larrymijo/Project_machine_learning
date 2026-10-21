"""Tests for the evaluation helpers (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (MeanBaseline, VenueBaseline, choose_draw_band, outcome_scores, outcomes,
                            regression_metrics, time_series_cv)
from src.linear_regression import LinearRegressionScratch


def test_metrics_by_hand():
    y_true = np.array([[1, 0], [2, 1], [3, 2]])
    y_pred = np.array([[1, 1], [2, 1], [5, 2]])
    table = regression_metrics(y_true, y_pred)
    assert table.loc["home goals", "MAE"] == pytest.approx(2 / 3)
    assert table.loc["home goals", "MSE"] == pytest.approx(4 / 3)
    assert table.loc["home goals", "RMSE"] == pytest.approx(np.sqrt(4 / 3))
    assert table.loc["home goals", "R²"] == pytest.approx(1 - 4 / 2)          # SS_res 4, SS_tot 2
    assert table.loc["away goals", "MAE"] == pytest.approx(1 / 3)
    assert table.loc["both (average)", "MAE"] == pytest.approx(0.5)


def test_perfect_predictions():
    y = np.array([[0, 1], [2, 2], [4, 0]])
    table = regression_metrics(y, y)
    assert (table["MAE"] == 0).all() and (table["R²"] == 1).all()


def test_outcomes_and_draw_band():
    assert list(outcomes([2, 1, 0], [1, 1, 3])) == ["H", "D", "A"]
    assert list(outcomes([1.2, 1.6], [1.0, 1.0], draw_band=0.3)) == ["D", "H"]


def test_choose_draw_band_finds_the_best_band():
    y_true = np.array([[1, 1], [2, 2], [3, 0], [0, 2]])
    y_pred = np.array([[1.3, 1.0], [1.4, 1.5], [2.5, 0.5], [0.5, 1.9]])   # gaps 0.3 / 0.1 are the draws
    band = choose_draw_band(y_true, y_pred)
    assert 0.3 < band <= 1.4                     # must cover the 0.3 gap but not the 1.4 gap
    assert outcome_scores(y_true, y_pred, band)["W/D/L accuracy"] == 1.0


def test_outcome_scores_rounding_and_negatives():
    y_true = np.array([[2, 0], [1, 1]])
    y_pred = np.array([[1.6, -0.2], [0.4, 1.2]])
    scores = outcome_scores(y_true, y_pred, draw_band=0.0)
    assert scores["exact score"] == 0.5          # (2, 0) is hit, (0, 1) is not
    assert scores["negative predictions"] == 1


def test_baselines():
    X = pd.DataFrame({"neutral": [0, 0, 1, 1]})
    Y = np.array([[2, 0], [2, 2], [1, 1], [3, 1]])
    np.testing.assert_allclose(MeanBaseline().fit(X, Y).predict(X[:1]), [[2, 1]])
    venue = VenueBaseline().fit(X, Y).predict(X)
    np.testing.assert_allclose(venue[0], [2, 1])     # home venues: means of rows 0, 1
    np.testing.assert_allclose(venue[2], [2, 1])     # neutral venues: means of rows 2, 3


def test_time_series_cv_returns_one_score_per_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    Y = X @ np.array([[1.0, 0.5], [0.0, 1.0], [2.0, 0.0]]) + rng.normal(scale=0.1, size=(300, 2))
    scores = time_series_cv(lambda: LinearRegressionScratch(), X, Y, n_splits=5)
    assert scores.shape == (5,) and (scores < 0.2).all()


# ---------- added after Gunnar's review of PR #4 ----------

def test_outcomes_exact_equality_and_band_edge():
    # exactly equal predicted goals are a draw even with draw_band = 0
    assert list(outcomes([1.5, 0.0], [1.5, 0.0], draw_band=0.0)) == ["D", "D"]
    # a difference exactly equal to the band is NOT a draw (strict <)
    assert list(outcomes([1.5], [1.0], draw_band=0.5)) == ["H"]
    assert list(outcomes([1.0], [1.5], draw_band=0.5)) == ["A"]


def test_venue_baseline_with_plain_arrays():
    X = np.array([[5.0, 0], [6.0, 0], [7.0, 1], [8.0, 1]])     # column 1 = neutral
    Y = np.array([[2, 0], [2, 2], [1, 1], [3, 1]])
    with pytest.raises(TypeError, match="neutral"):
        VenueBaseline().fit(X, Y)
    pred = VenueBaseline(neutral_index=1).fit(X, Y).predict(X)
    np.testing.assert_allclose(pred[[0, 2]], [[2, 1], [2, 1]])


def test_venue_baseline_empty_group_falls_back_to_the_overall_mean():
    X = pd.DataFrame({"neutral": [0, 0, 0]})
    Y = np.array([[1, 0], [2, 1], [3, 2]])
    model = VenueBaseline().fit(X, Y)
    np.testing.assert_allclose(model.predict(pd.DataFrame({"neutral": [1]})), [[2, 1]])
    assert not np.isnan(model.predict(pd.DataFrame({"neutral": [0, 1]}))).any()


def test_time_series_cv_rejects_unsorted_dates():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))
    dates = pd.date_range("2000-01-01", periods=50)
    assert len(time_series_cv(lambda: LinearRegressionScratch(), X, Y, dates=dates)) == 5
    with pytest.raises(ValueError, match="sorted by date"):
        time_series_cv(lambda: LinearRegressionScratch(), X, Y, dates=dates[::-1])


# ---------- Phase 6: bootstrap and permutation importance ----------
from src.evaluation import bootstrap_difference, permutation_importance


def test_bootstrap_identical_models_give_zero():
    rng = np.random.default_rng(3)
    y = rng.poisson(1.3, size=(200, 2))
    p = y + rng.normal(scale=0.8, size=y.shape)
    res = bootstrap_difference(y, p, p, n_boot=300)
    assert res["observed"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0 and res["share_a_better"] == 0


def test_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(4)
    y = rng.poisson(1.3, size=(300, 2)).astype(float)
    good = y + rng.normal(scale=0.3, size=y.shape)
    bad = y + rng.normal(scale=1.5, size=y.shape)
    res = bootstrap_difference(y, good, bad, n_boot=500)
    assert res["observed"] < 0 and res["ci_high"] < 0 and res["share_a_better"] == 1.0
    mae = bootstrap_difference(y, good, bad, metric="MAE", n_boot=200)
    assert mae["ci_high"] < 0
    with pytest.raises(ValueError):
        bootstrap_difference(y, good, bad, metric="R2")


def test_bootstrap_observed_matches_the_metric_table():
    rng = np.random.default_rng(5)
    y = rng.poisson(1.3, size=(100, 2)).astype(float)
    a, b = y + rng.normal(size=y.shape), y + rng.normal(scale=2, size=y.shape)
    res = bootstrap_difference(y, a, b, n_boot=50)
    expected = (regression_metrics(y, a).loc["both (average)", "RMSE"]
                - regression_metrics(y, b).loc["both (average)", "RMSE"])
    assert res["observed"] == pytest.approx(expected)


def test_permutation_importance_finds_the_relevant_feature():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["signal", "noise1", "noise2"])
    Y = np.column_stack([2 * X["signal"], -X["signal"]]) + rng.normal(scale=0.1, size=(400, 2))
    model = LinearRegressionScratch().fit(X, Y)
    imp = permutation_importance(model, X, Y, n_repeats=5)
    assert imp.index[0] == "signal"
    assert imp.loc["signal", "RMSE increase"] > 1.0
    assert abs(imp.loc["noise1", "RMSE increase"]) < 0.05 and abs(imp.loc["noise2", "RMSE increase"]) < 0.05
