"""Algorithm 2: Random Forest Regressor (scikit-learn) and its tuning (Phase 5).

Model        An average of many decision trees. Each tree is grown on a bootstrap sample of the
             training matches (drawn with replacement), and at every split it only looks at a
             random subset of the features (`max_features`). Averaging many different, noisy trees
             lowers the variance that a single deep tree has.
Two targets  scikit-learn's forest supports several targets natively: every leaf stores the mean
             home AND away goals of its training matches, and a split is chosen to reduce the
             squared error summed over both targets. One fit, two predictions - like our scratch
             model, which fits one theta column per target.
Scaling      Not needed: a split only compares a feature with a threshold, so the scale of a
             feature doesn't matter.
Tuning       Grid search with the SAME 5 TimeSeriesSplit folds that were used to tune lambda of the
             scratch model (src/evaluation.py, time_series_cv), scored with the same RMSE (per
             target, then averaged over home and away goals). So the cross-validation RMSE of both
             algorithms can be compared directly. tests/test_random_forest.py checks that the two
             scorers give the same number.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.data import RANDOM_STATE
from src.evaluation import check_sorted_by_date

# The grid from docs/PLAN.md section 8: 2 x 3 x 3 x 3 = 54 combinations, x 5 folds = 270 fits.
PARAM_GRID = {
    "n_estimators": [200, 500],          # more trees = smoother average (never overfits, only slower)
    "max_depth": [None, 10, 20],         # how deep a tree may grow (None = until the leaves are pure / too small)
    "min_samples_leaf": [1, 5, 20],      # fewest training matches allowed in a leaf - larger = smoother predictions
    "max_features": [1.0, 0.5, "sqrt"],  # share of the features tried at each split (1.0 = all, "sqrt" = 4 of 17)
}

# Used to fill in parameters that are not tuned when describing a model ("before tuning" = these).
DEFAULTS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0}

SCORING = "neg_root_mean_squared_error"  # scikit-learn maximises scores, so the RMSE comes back negated


def make_forest(n_jobs=-1, **params):
    """A RandomForestRegressor with our fixed random_state; any parameter can be overridden."""
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, **params)


def grid_search_forest(X, Y, dates, param_grid=PARAM_GRID, n_splits=5, n_jobs=-1):
    """Try every combination in `param_grid` on the TimeSeriesSplit folds of the training data.

    The rows must be in date order (checked with `dates`), because TimeSeriesSplit builds the folds
    from row positions. The parallel work is spread over the grid (one fit per CPU core); the forests
    themselves run on one core each, so the cores are not over-subscribed.

    refit=False: the search only chooses the parameters. The final model is fitted explicitly
    afterwards, on the whole training window, so it is visible in the notebook what is fitted on what.
    Returns the fitted GridSearchCV object; use `cv_results_table` to read it.
    """
    check_sorted_by_date(dates, len(X))
    search = GridSearchCV(make_forest(n_jobs=1), param_grid, scoring=SCORING,
                          cv=TimeSeriesSplit(n_splits=n_splits), n_jobs=n_jobs,
                          refit=False, return_train_score=True)
    return search.fit(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))


def cv_results_table(search):
    """One row per parameter combination, best first: the validation RMSE of every fold, its mean and
    standard deviation, the mean training RMSE (to see overfitting) and the mean fit time per fold."""
    res = search.cv_results_
    n_splits = search.n_splits_
    params = list(res["params"])
    # dtype=object keeps the values exactly as given: otherwise pandas turns [None, 10, 20] into
    # [NaN, 10.0, 20.0], and a float max_depth is rejected by RandomForestRegressor.
    table = pd.DataFrame({k: pd.Series([p[k] for p in params], dtype=object) for k in params[0]})
    for k in range(n_splits):
        table[f"fold {k + 1}"] = -res[f"split{k}_test_score"]
    folds = [f"fold {k + 1}" for k in range(n_splits)]
    table["mean"] = table[folds].mean(axis=1)
    table["std"] = table[folds].std(axis=1, ddof=1)
    table["train RMSE"] = -res["mean_train_score"]
    table["fit time (s)"] = res["mean_fit_time"]
    return table.sort_values("mean", kind="stable").reset_index(drop=True)
