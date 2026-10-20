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

def time_series_cv(make_model, X, Y, n_splits=5, dates=None):
    """Fit a fresh model on each TimeSeriesSplit fold; return the validation RMSE (average of both targets)
    per fold. `make_model` is a function that returns a new, unfitted model.

    TimeSeriesSplit only looks at row positions: it trains on the first rows and validates on the
    following ones. The rows of X and Y MUST therefore be in date order, otherwise the folds would
    train on the future. Pass `dates` (one per row) to have this checked.
    """
    if dates is not None:
        dates = pd.Series(np.asarray(dates))
        if len(dates) != len(X):
            raise ValueError("dates must have one value per row of X")
        if not dates.is_monotonic_increasing:
            raise ValueError("rows must be sorted by date for time-series cross-validation")
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    scores = []
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = make_model().fit(X[train_idx], Y[train_idx])
        err = model.predict(X[valid_idx]) - Y[valid_idx]
        scores.append(np.sqrt((err ** 2).mean(axis=0)).mean())
    return np.array(scores)
