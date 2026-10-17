"""Algorithm 1: Linear Regression implemented from scratch with NumPy (no ready-made ML model).

Model        Y_hat = X_b @ theta, where X_b is X with a column of 1s added (the bias / intercept).
             Y can have several columns (here: home goals and away goals). theta then has one
             column per target - the same as fitting one regression per target on the same X.
Cost         J(theta) = MSE + l2 * sum(theta_j^2 for j >= 1)        (the bias is not penalised)
             MSE = (1/m) * sum of squared errors, averaged over the targets.
Fitting      "normal":   closed form, (X_b^T X_b + m * l2 * I*) theta = X_b^T Y, solved with
                         np.linalg.solve (I* = identity with a 0 for the bias).
             "gradient": batch gradient descent,
                         theta <- theta - learning_rate * [(2/m) X_b^T (X_b theta - Y) + 2 * l2 * theta*]
                         (theta* = theta with the bias row set to 0).
             Both minimise the same cost, so they must give the same theta (checked in the tests).
Scaling      With standardize=True the features are standardised with the TRAINING mean and
             standard deviation (stored in fit, reused in predict). Gradient descent needs this:
             with features on very different scales (Elo ~1500, dummies 0/1) the cost surface is a
             long narrow valley and gradient descent would barely move. It also makes the
             coefficients comparable with each other.
"""
import numpy as np


class LinearRegressionScratch:
    """Multi-output linear regression fitted with the normal equation or batch gradient descent.

    Parameters
    ----------
    method : "normal" or "gradient"
    learning_rate : step size for gradient descent
    n_iters : maximum number of gradient-descent iterations
    l2 : strength of the L2 (ridge) penalty; 0 = ordinary least squares
    tol : gradient descent stops when the cost improves by less than this
    standardize : standardise the features with the training mean / std

    Attributes after fit
    --------------------
    theta_ : (n_features + 1, n_targets) - row 0 is the intercept
    intercept_, coef_ : the same split into intercept (n_targets,) and coefficients (n_features, n_targets)
    loss_history_ : cost after every gradient-descent iteration (empty for the normal equation)
    n_iter_ : number of iterations used
    """

    def __init__(self, method="normal", learning_rate=0.05, n_iters=5000, l2=0.0, tol=1e-10, standardize=True):
        if method not in ("normal", "gradient"):
            raise ValueError("method must be 'normal' or 'gradient'")
        self.method = method
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.standardize = standardize

    # ---------- helpers ----------
    @staticmethod
    def _as_2d(a, name):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains missing or infinite values - fill them before fitting")
        return a

    def _scale(self, X):
        return (X - self.mean_) / self.std_ if self.standardize else X

    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _cost(self, Xb, Y, theta):
        residuals = Xb @ theta - Y
        mse = np.mean(residuals ** 2)                       # averaged over rows and targets
        return mse + self.l2 * np.sum(theta[1:] ** 2) / Y.shape[1]

    # ---------- fit / predict ----------
    def fit(self, X, Y):
        X = self._as_2d(X, "X")
        Y = self._as_2d(Y, "Y")
        self._single_target = np.ndim(Y) == 2 and Y.shape[1] == 1
        m, n = X.shape

        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)            # a constant column stays as it is
        Xb = self._add_bias(self._scale(X))

        penalty = np.eye(n + 1)
        penalty[0, 0] = 0.0                                  # never penalise the intercept

        if self.method == "normal":
            A = Xb.T @ Xb + m * self.l2 * penalty
            self.theta_ = np.linalg.solve(A, Xb.T @ Y)       # solve A theta = X_b^T Y (no explicit inverse)
            self.loss_history_ = []
            self.n_iter_ = 0
        else:
            theta = np.zeros((n + 1, Y.shape[1]))
            history = [self._cost(Xb, Y, theta)]
            for i in range(1, self.n_iters + 1):
                gradient = (2 / m) * Xb.T @ (Xb @ theta - Y) + 2 * self.l2 * (penalty @ theta)
                theta = theta - self.learning_rate * gradient
                cost = self._cost(Xb, Y, theta)
                if not np.isfinite(cost) or cost > 1e6 * history[0]:
                    raise RuntimeError(f"gradient descent diverged at iteration {i} - lower the learning_rate")
                history.append(cost)
                if abs(history[-2] - cost) < self.tol:
                    break
            self.theta_ = theta
            self.loss_history_ = history
            self.n_iter_ = len(history) - 1

        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        return self

    def predict(self, X):
        if not hasattr(self, "theta_"):
            raise RuntimeError("call fit() before predict()")
        X = self._as_2d(X, "X")
        Y_hat = self._add_bias(self._scale(X)) @ self.theta_
        return Y_hat.ravel() if self._single_target else Y_hat

    def get_params(self):
        return {k: getattr(self, k) for k in ("method", "learning_rate", "n_iters", "l2", "tol", "standardize")}

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"LinearRegressionScratch({args})"
