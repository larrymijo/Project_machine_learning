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

    A draw is |home - away| < draw_band. With draw_band=0 (real scores) only equal scores
    count as a draw - a tiny tolerance is used so that a difference of exactly 0 is included."""
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
    """Predicts the training mean home / away goals separately for home and neutral venues."""

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        neutral = np.asarray(X["neutral"]) == 1
        self.means_ = {False: Y[~neutral].mean(axis=0), True: Y[neutral].mean(axis=0)}
        return self

    def predict(self, X):
        neutral = np.asarray(X["neutral"]) == 1
        return np.where(neutral[:, None], self.means_[True], self.means_[False])


# ---------- time-series cross-validation (for our own model) ----------

def time_series_cv(make_model, X, Y, n_splits=5):
    """Fit a fresh model on each TimeSeriesSplit fold; return the validation RMSE (average of both targets)
    per fold. `make_model` is a function that returns a new, unfitted model."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    scores = []
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        model = make_model().fit(X[train_idx], Y[train_idx])
        err = model.predict(X[valid_idx]) - Y[valid_idx]
        scores.append(np.sqrt((err ** 2).mean(axis=0)).mean())
    return np.array(scores)
