"""Correctness tests for the from-scratch linear regression (run with: pytest).

The reference solution is NumPy's least-squares solver (np.linalg.lstsq) - plain linear algebra,
not a machine-learning model - so the from-scratch class is never compared with a library regressor.
"""
import numpy as np
import pytest

from src.linear_regression import LinearRegressionScratch

RNG = np.random.default_rng(42)


def synthetic(m=500, noise=0.0):
    """Features on very different scales (like Elo vs 0/1 dummies) and two targets with known theta."""
    X = np.column_stack([RNG.normal(1500, 200, m), RNG.normal(1.3, 0.5, m), RNG.integers(0, 2, m)])
    true_theta = np.array([[0.5, 1.0],                        # intercepts
                           [0.002, -0.001],
                           [0.8, 0.3],
                           [-0.4, 0.2]])
    Y = np.column_stack([np.ones(m), X]) @ true_theta + noise * RNG.normal(size=(m, 2))
    return X, Y, true_theta


def lstsq_predictions(X, Y):
    Xb = np.column_stack([np.ones(len(X)), X])
    theta, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return Xb @ theta


def test_normal_equation_recovers_the_true_theta():
    X, Y, true_theta = synthetic()
    model = LinearRegressionScratch(method="normal", standardize=False).fit(X, Y)
    np.testing.assert_allclose(model.theta_, true_theta, atol=1e-8)


def test_standardising_does_not_change_the_predictions():
    X, Y, _ = synthetic(noise=0.3)
    raw = LinearRegressionScratch(method="normal", standardize=False).fit(X, Y)
    std = LinearRegressionScratch(method="normal", standardize=True).fit(X, Y)
    np.testing.assert_allclose(raw.predict(X), std.predict(X), atol=1e-8)


def test_matches_the_numpy_least_squares_solution():
    X, Y, _ = synthetic(noise=0.5)
    model = LinearRegressionScratch(method="normal").fit(X, Y)
    np.testing.assert_allclose(model.predict(X), lstsq_predictions(X, Y), atol=1e-8)


def test_gradient_descent_converges_to_the_normal_equation():
    X, Y, _ = synthetic(noise=0.5)
    exact = LinearRegressionScratch(method="normal").fit(X, Y)
    gd = LinearRegressionScratch(method="gradient", learning_rate=0.1, n_iters=20_000, tol=1e-14).fit(X, Y)
    np.testing.assert_allclose(gd.theta_, exact.theta_, atol=1e-5)


def test_gradient_descent_loss_never_increases():
    X, Y, _ = synthetic(noise=0.5)
    gd = LinearRegressionScratch(method="gradient", learning_rate=0.05, n_iters=500, tol=0).fit(X, Y)
    assert len(gd.loss_history_) == 501
    assert np.all(np.diff(gd.loss_history_) <= 1e-12)


def test_too_large_learning_rate_is_detected():
    X, Y, _ = synthetic()
    with pytest.raises(RuntimeError, match="diverged"):
        LinearRegressionScratch(method="gradient", learning_rate=5.0, n_iters=200).fit(X, Y)


def test_two_targets_equal_two_separate_fits():
    X, Y, _ = synthetic(noise=0.5)
    both = LinearRegressionScratch().fit(X, Y)
    home = LinearRegressionScratch().fit(X, Y[:, 0])
    away = LinearRegressionScratch().fit(X, Y[:, 1])
    np.testing.assert_allclose(both.predict(X)[:, 0], home.predict(X), atol=1e-10)
    np.testing.assert_allclose(both.predict(X)[:, 1], away.predict(X), atol=1e-10)
    assert home.predict(X).shape == (len(X),)               # 1-D in, 1-D out


def test_l2_shrinks_the_coefficients_but_not_the_intercept():
    X, Y, _ = synthetic(noise=0.5)
    norms = [np.linalg.norm(LinearRegressionScratch(l2=l2).fit(X, Y).coef_) for l2 in (0, 0.1, 1, 10)]
    assert norms == sorted(norms, reverse=True)
    huge = LinearRegressionScratch(l2=1e9).fit(X, Y)
    np.testing.assert_allclose(huge.intercept_, Y.mean(axis=0), atol=1e-6)   # all coefficients ~0 -> mean
    np.testing.assert_allclose(huge.coef_, 0, atol=1e-6)


def test_l2_gradient_descent_matches_l2_normal_equation():
    X, Y, _ = synthetic(noise=0.5)
    exact = LinearRegressionScratch(method="normal", l2=0.5).fit(X, Y)
    gd = LinearRegressionScratch(method="gradient", l2=0.5, learning_rate=0.1, n_iters=20_000, tol=1e-14).fit(X, Y)
    np.testing.assert_allclose(gd.theta_, exact.theta_, atol=1e-6)


def test_bad_input_is_rejected():
    X, Y, _ = synthetic()
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        LinearRegressionScratch().fit(X_nan, Y)
    with pytest.raises(RuntimeError, match="fit"):
        LinearRegressionScratch().predict(X)
    with pytest.raises(ValueError):
        LinearRegressionScratch(method="sgd")
