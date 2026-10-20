"""Tests for the random-forest helpers (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import time_series_cv
from src.random_forest import PARAM_GRID, cv_results_table, grid_search_forest, make_forest

SMALL_GRID = {"n_estimators": [10], "max_depth": [None, 3], "min_samples_leaf": [1], "max_features": [1.0]}


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    Y = np.column_stack([1.5 + X[:, 0] + rng.normal(scale=0.3, size=n),
                         1.0 - X[:, 1] + rng.normal(scale=0.3, size=n)])
    return X, Y, pd.date_range("2000-01-01", periods=n)


def test_grid_is_the_one_in_the_plan():
    assert PARAM_GRID == {"n_estimators": [200, 500], "max_depth": [None, 10, 20],
                          "min_samples_leaf": [1, 5, 20], "max_features": [1.0, 0.5, "sqrt"]}


def test_grid_search_rmse_equals_our_time_series_cv():
    # The forest is tuned with scikit-learn's scorer, our scratch model with time_series_cv.
    # Both must compute the same number on the same folds, otherwise the two can't be compared.
    X, Y, dates = _data()
    search = grid_search_forest(X, Y, dates, param_grid=SMALL_GRID, n_jobs=1)
    table = cv_results_table(search)
    for _, row in table.iterrows():
        ours = time_series_cv(lambda: make_forest(n_jobs=1, n_estimators=10, max_depth=row["max_depth"],
                                                  min_samples_leaf=1, max_features=1.0), X, Y)
        np.testing.assert_allclose(row[[f"fold {k}" for k in range(1, 6)]].to_numpy(float), ours)


def test_cv_results_table_is_sorted_and_complete():
    X, Y, dates = _data()
    table = cv_results_table(grid_search_forest(X, Y, dates, param_grid=SMALL_GRID, n_jobs=1))
    assert len(table) == 2
    assert table["mean"].is_monotonic_increasing                 # best (lowest RMSE) first
    assert (table[[f"fold {k}" for k in range(1, 6)]] > 0).all().all()   # RMSE, not the negated score
    assert table["max_depth"].tolist().count(None) == 1          # None stays None (no NaN)


def test_grid_search_rejects_unsorted_dates():
    X, Y, dates = _data(n=60)
    with pytest.raises(ValueError, match="sorted by date"):
        grid_search_forest(X, Y, dates[::-1], param_grid=SMALL_GRID, n_jobs=1)
    with pytest.raises(ValueError, match="one value per row"):
        grid_search_forest(X, Y, dates[:-1], param_grid=SMALL_GRID, n_jobs=1)


def test_forest_is_reproducible_and_predicts_both_targets():
    X, Y, _ = _data()
    p1 = make_forest(n_estimators=20).fit(X, Y).predict(X)
    p2 = make_forest(n_estimators=20).fit(X, Y).predict(X)
    assert p1.shape == (len(X), 2)
    # random_state=42 -> the same trees. Not bit-identical: with n_jobs=-1 the trees' predictions are
    # added up in the order the threads finish, and floating-point addition is not associative.
    np.testing.assert_allclose(p1, p2, rtol=1e-12, atol=1e-12)


def test_forest_predictions_stay_within_the_training_range():
    # A leaf predicts a mean of training targets, so the forest can never predict negative goals
    # (unlike the linear model) - a property we rely on in the discussion.
    X, Y, _ = _data()
    Y = np.clip(Y, 0, None)
    pred = make_forest(n_estimators=20).fit(X, Y).predict(X * 10)   # far outside the training data
    assert (pred >= Y.min(axis=0)).all() and (pred <= Y.max(axis=0)).all()
