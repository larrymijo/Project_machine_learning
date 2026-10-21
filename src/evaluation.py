"""Evaluation helpers: metrics, baselines, match outcomes and time-series cross-validation (Phase 4).

Everything works on two targets at once (home goals, away goals). Metrics are written with
NumPy so it is clear exactly what is computed.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data import TARGETS


# ---------- metrics ----------

def regression_metrics(y_true, y_pred):
    """MAE, MSE, RMSE and R² per target, plus their average over the targets ("both")."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    table = pd.DataFrame({
        "MAE": np.abs(err).mean(axis=0),
        "MSE": (err ** 2).mean(axis=0),
        "RMSE": np.sqrt((err ** 2).mean(axis=0)),
        "R²": 1 - (err ** 2).sum(axis=0) / ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0),
    }, index=[t.replace("_score", " goals") for t in TARGETS])
    table.loc["both (average)"] = table.mean()
    return table


def outcomes(home, away, draw_band=0.0):
    """'H' (home win) / 'D' (draw) / 'A' (away win) from goals.

    A match is a draw when |home - away| < max(draw_band, 1e-9) - a strict inequality.
    The 1e-9 floor matters only for draw_band = 0 (used for real scores): "< 0" would never be
    true, so without the floor even 1 - 1 would not count as a draw. A difference exactly equal
    to a positive draw_band is NOT a draw."""
    diff = np.asarray(home, dtype=float) - np.asarray(away, dtype=float)
    result = np.where(diff > 0, "H", "A").astype(object)
    result[np.abs(diff) < max(draw_band, 1e-9)] = "D"
    return result


def choose_draw_band(y_true, y_pred, bands=np.arange(0, 1.01, 0.05)):
    """Draw band that maximises win/draw/loss accuracy on the given (training) predictions.

    A regression model almost never predicts exactly equal goals, so without a band it would
    never predict a draw. The band is chosen on the TRAINING data only.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    actual = outcomes(y_true[:, 0], y_true[:, 1])
    acc = [(outcomes(y_pred[:, 0], y_pred[:, 1], b) == actual).mean() for b in bands]
    return float(bands[int(np.argmax(acc))])


def outcome_scores(y_true, y_pred, draw_band):
    """Win/draw/loss accuracy and exact-score hit rate (predictions rounded, negatives set to 0)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    actual = outcomes(y_true[:, 0], y_true[:, 1])
    predicted = outcomes(y_pred[:, 0], y_pred[:, 1], draw_band)
    rounded = np.clip(np.rint(y_pred), 0, None)
    return {
        "W/D/L accuracy": (predicted == actual).mean(),
        "exact score": (rounded == y_true).all(axis=1).mean(),
        "predicted draws": (predicted == "D").mean(),
        "negative predictions": int((y_pred < 0).sum()),
    }


# ---------- baselines ----------

class MeanBaseline:
    """Always predicts the training mean of home goals and of away goals."""

    def fit(self, X, Y):
        self.means_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict(self, X):
        return np.tile(self.means_, (len(X), 1))


class VenueBaseline:
    """Predicts the training mean home / away goals separately for home and neutral venues.

    X is a DataFrame with a `neutral` column (0/1), or - for plain arrays, e.g. inside
    `time_series_cv` - pass `neutral_index`, the position of that column. If one venue group has
    no training rows, it falls back to the overall training mean instead of returning NaN.
    """

    def __init__(self, neutral_column="neutral", neutral_index=None):
        self.neutral_column = neutral_column
        self.neutral_index = neutral_index

    def _is_neutral(self, X):
        if hasattr(X, "columns"):
            return np.asarray(X[self.neutral_column]) == 1
        if self.neutral_index is None:
            raise TypeError("VenueBaseline needs a DataFrame with a 'neutral' column, "
                            "or neutral_index=<column position> for plain arrays")
        return np.asarray(X)[:, self.neutral_index] == 1

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        neutral = self._is_neutral(X)
        overall = Y.mean(axis=0)
        self.means_ = {flag: Y[neutral == flag].mean(axis=0) if (neutral == flag).any() else overall
                       for flag in (False, True)}
        return self

    def predict(self, X):
        neutral = self._is_neutral(X)
        return np.where(neutral[:, None], self.means_[True], self.means_[False])


# ---------- time-series cross-validation (for our own model) ----------

def check_sorted_by_date(dates, n_rows):
    """Raise a ValueError unless there is one date per row and the dates never go backwards.

    Used before every TimeSeriesSplit (our own loop below and the random-forest grid search):
    the folds are made from row positions, so unsorted rows would mean training on the future.
    """
    dates = pd.Series(np.asarray(dates))
    if len(dates) != n_rows:
        raise ValueError("dates must have one value per row of X")
    if not dates.is_monotonic_increasing:
        raise ValueError("rows must be sorted by date for time-series cross-validation")


def time_series_cv(make_model, X, Y, n_splits=5, dates=None):
    """Fit a fresh model on each TimeSeriesSplit fold; return the validation RMSE (average of both targets)
    per fold. `make_model` is a function that returns a new, unfitted model.

    TimeSeriesSplit only looks at row positions: it trains on the first rows and validates on the
    following ones. The rows of X and Y MUST therefore be in date order, otherwise the folds would
    train on the future. Pass `dates` (one per row) to have this checked.
    """
    if dates is not None:
        check_sorted_by_date(dates, len(X))
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    scores = []
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = make_model().fit(X[train_idx], Y[train_idx])
        err = model.predict(X[valid_idx]) - Y[valid_idx]
        scores.append(np.sqrt((err ** 2).mean(axis=0)).mean())
    return np.array(scores)


# ---------- comparing two models (Phase 6) ----------

def _per_target_error(y_true, y_pred, metric):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    if metric == "RMSE":
        return np.sqrt((err ** 2).mean(axis=-2)).mean(axis=-1)
    if metric == "MAE":
        return np.abs(err).mean(axis=-2).mean(axis=-1)
    raise ValueError("metric must be 'RMSE' or 'MAE'")


def bootstrap_difference(y_true, pred_a, pred_b, metric="RMSE", n_boot=2000, seed=42):
    """Paired bootstrap of  metric(model A) - metric(model B)  (averaged over home and away goals).

    The test matches are resampled with replacement `n_boot` times; both models are scored on the SAME
    resampled matches each time (paired), so the difficulty of the sample cancels out. Negative = A better.
    Returns the observed difference, the 95% interval of the bootstrap differences and the share of
    samples in which A is better. If the interval contains 0, the data can't tell the models apart.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = _per_target_error(y[idx], a[idx], metric) - _per_target_error(y[idx], b[idx], metric)
    return {"observed": float(_per_target_error(y, a, metric) - _per_target_error(y, b, metric)),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "share_a_better": float((diffs < 0).mean())}


def permutation_importance(model, X, Y, n_repeats=10, seed=42):
    """How much the RMSE (average of both targets) grows when one feature column is shuffled.

    Shuffling breaks the link between that feature and the goals while keeping its distribution, so a
    large increase means the model relies on the feature. Works the same for any model with predict().
    Returns a DataFrame (one row per feature) with the mean and standard deviation over the repeats.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    Y = np.asarray(Y, dtype=float)
    rng = np.random.default_rng(seed)
    base = _per_target_error(Y, model.predict(X), "RMSE")
    rows = {}
    for col in X.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            increases.append(_per_target_error(Y, model.predict(shuffled), "RMSE") - base)
        rows[col] = {"RMSE increase": np.mean(increases), "std": np.std(increases)}
    return pd.DataFrame(rows).T.sort_values("RMSE increase", ascending=False)
