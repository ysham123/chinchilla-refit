"""Fitting the Chinchilla parametric loss L(N, D) = E + A / N**alpha + B / D**beta.

Parameters are optimized in the log-sum-exp form used by Hoffmann et al. (2022):

    log L_hat = LSE(a - alpha * log N, b - beta * log D, e),  A = exp(a), B = exp(b), E = exp(e)

and the fit minimizes a Huber loss on log-loss residuals. ``delta=None`` gives plain
least squares in log space.
"""

from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

PARAM_NAMES = ("a", "b", "e", "alpha", "beta")

# Values reported in Hoffmann et al. (2022), Approach 3.
HOFFMANN_REPORTED = {"A": 406.4, "B": 410.7, "E": 1.69, "alpha": 0.34, "beta": 0.28}

# Chinchilla's training compute, used to report a concrete allocation.
CHINCHILLA_FLOP = 5.76e23


def tokens_from_flops(flops, params):
    """Training tokens implied by the C = 6ND approximation."""
    return np.asarray(flops, dtype=float) / (6.0 * np.asarray(params, dtype=float))


def to_natural(theta):
    """(a, b, e, alpha, beta) -> dict with A, B, E, alpha, beta."""
    a, b, e, alpha, beta = theta
    return {"A": float(np.exp(a)), "B": float(np.exp(b)), "E": float(np.exp(e)),
            "alpha": float(alpha), "beta": float(beta)}


def to_theta(params):
    """dict with A, B, E, alpha, beta -> (a, b, e, alpha, beta)."""
    return np.array([np.log(params["A"]), np.log(params["B"]), np.log(params["E"]),
                     params["alpha"], params["beta"]])


def _terms(theta, log_n, log_d):
    a, b, e, alpha, beta = theta
    return np.stack([a - alpha * log_n, b - beta * log_d, np.full_like(log_n, e)])


def predict_log_loss(theta, log_n, log_d):
    return logsumexp(_terms(theta, log_n, log_d), axis=0)


def predict_loss(params, n, d):
    """Direct evaluation of E + A / N**alpha + B / D**beta."""
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    return params["E"] + params["A"] / n ** params["alpha"] + params["B"] / d ** params["beta"]


def objective(theta, log_n, log_d, log_l, delta=1e-3):
    """Summed Huber (or squared) loss on log residuals, with its analytic gradient."""
    terms = _terms(theta, log_n, log_d)
    pred = logsumexp(terms, axis=0)
    w = softmax(terms, axis=0)
    r = pred - log_l
    if delta is None:
        value = 0.5 * np.sum(r ** 2)
        g = r
    else:
        small = np.abs(r) <= delta
        value = np.sum(np.where(small, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta)))
        g = np.clip(r, -delta, delta)
    grad = np.array([
        np.sum(g * w[0]),
        np.sum(g * w[1]),
        np.sum(g * w[2]),
        -np.sum(g * w[0] * log_n),
        -np.sum(g * w[1] * log_d),
    ])
    return value, grad


def hoffmann_grid():
    """Initialization grid from Hoffmann et al., as used in Epoch AI's replication."""
    grid = []
    for alpha, beta, e, a, b in product(np.arange(0, 2.5, 0.5), np.arange(0, 2.5, 0.5),
                                        np.arange(-1, 1.5, 0.5), np.arange(0, 30, 5),
                                        np.arange(0, 30, 5)):
        grid.append(np.array([a, b, e, alpha, beta], dtype=float))
    return grid


TIGHT = {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20000}
SCIPY_DEFAULT = {}


def fit(log_n, log_d, log_l, delta=1e-3, inits=None, options=TIGHT):
    """L-BFGS-B from every initialization; returns the best result and all runs."""
    if inits is None:
        inits = hoffmann_grid()
    runs = []
    for x0 in inits:
        res = minimize(objective, x0, args=(log_n, log_d, log_l, delta), jac=True,
                       method="L-BFGS-B", options=options)
        runs.append({"x0": np.asarray(x0), "theta": res.x, "value": float(res.fun),
                     "success": bool(res.success)})
    finite = [r for r in runs if np.isfinite(r["value"])]
    best = min(finite, key=lambda r: r["value"])
    return best, runs


def compute_optimal_exponents(alpha, beta):
    """N_opt ~ C**a and D_opt ~ C**b under C = 6ND."""
    return beta / (alpha + beta), alpha / (alpha + beta)


def compute_optimal_allocation(params, flops):
    """Closed-form loss-minimizing (N, D) for a compute budget, from Hoffmann et al. eq. 4."""
    alpha, beta = params["alpha"], params["beta"]
    a, b = compute_optimal_exponents(alpha, beta)
    g = (alpha * params["A"] / (beta * params["B"])) ** (1.0 / (alpha + beta))
    n_opt = g * (flops / 6.0) ** a
    d_opt = (flops / 6.0) ** b / g
    return n_opt, d_opt
