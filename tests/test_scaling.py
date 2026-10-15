import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scaling as s  # noqa: E402

TRUE = {"A": 450.0, "B": 1800.0, "E": 1.8, "alpha": 0.34, "beta": 0.37}


def synthetic(n_runs=300, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    n = 10 ** rng.uniform(7.5, 10.5, n_runs)
    d = 10 ** rng.uniform(9.0, 11.5, n_runs)
    loss = s.predict_loss(TRUE, n, d) * np.exp(rng.normal(0, noise, n_runs))
    return np.log(n), np.log(d), np.log(loss)


def test_tokens_from_flops():
    assert s.tokens_from_flops(6e20, 1e9) == pytest.approx(1e11)


def test_log_sum_exp_form_matches_direct_formula():
    n, d = np.array([1e8, 3e9]), np.array([2e9, 4e11])
    direct = s.predict_loss(TRUE, n, d)
    via_lse = np.exp(s.predict_log_loss(s.to_theta(TRUE), np.log(n), np.log(d)))
    np.testing.assert_allclose(via_lse, direct, rtol=1e-12)


@pytest.mark.parametrize("delta", [1e-3, None])
def test_analytic_gradient_matches_finite_differences(delta):
    log_n, log_d, log_l = synthetic(50)
    theta = np.array([6.0, 7.0, 0.5, 0.3, 0.4])
    _, grad = s.objective(theta, log_n, log_d, log_l, delta)
    eps = 1e-6
    fd = [(s.objective(theta + eps * e, log_n, log_d, log_l, delta)[0]
           - s.objective(theta - eps * e, log_n, log_d, log_l, delta)[0]) / (2 * eps) for e in np.eye(5)]
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("delta", [1e-3, None])
def test_recovers_known_parameters(delta):
    log_n, log_d, log_l = synthetic()
    inits = [np.array([a, b, 0.5, 0.5, 0.5]) for a in (0.0, 5.0, 10.0) for b in (0.0, 5.0, 10.0)]
    best, _ = s.fit(log_n, log_d, log_l, delta=delta, inits=inits)
    p = s.to_natural(best["theta"])
    assert p["alpha"] == pytest.approx(TRUE["alpha"], abs=0.02)
    assert p["beta"] == pytest.approx(TRUE["beta"], abs=0.02)
    assert p["E"] == pytest.approx(TRUE["E"], abs=0.03)


def test_compute_optimal_allocation_spends_the_budget():
    flops = 1e23
    n_opt, d_opt = s.compute_optimal_allocation(TRUE, flops)
    assert 6 * n_opt * d_opt == pytest.approx(flops, rel=1e-9)


def test_compute_optimal_allocation_is_a_minimum():
    flops = 1e23
    n_opt, _ = s.compute_optimal_allocation(TRUE, flops)
    loss_at = lambda n: s.predict_loss(TRUE, n, flops / (6 * n))
    assert loss_at(n_opt) < loss_at(n_opt * 1.1)
    assert loss_at(n_opt) < loss_at(n_opt / 1.1)
