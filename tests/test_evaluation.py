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
