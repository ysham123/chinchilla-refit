"""Refit the Chinchilla parametric loss and compare estimators.

Usage:
    python refit.py                      # full run, writes results/ and figures/
    python refit.py --bootstrap 100      # quicker run

The data are Epoch AI's extraction of Hoffmann et al.'s Figure 4 (Besiroglu et al., 2024).
They are downloaded on first run and cached in data/, which is not committed.
"""

import os

# One BLAS thread per worker process; multithreaded BLAS inside a process pool oversubscribes
# the CPU and made the grid fit roughly 10x slower on a 4-core machine.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import urllib.request
from multiprocessing import Pool

import numpy as np

import scaling as s

DATA_URL = ("https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/"
            "main/data/svg_extracted_data.csv")
DATA_PATH = os.path.join("data", "svg_extracted_data.csv")

# Epoch AI's Huber grid fit on the same 240 runs (data_analysis.ipynb), used to validate this code.
EPOCH_GRID_FIT = {"A": 477.84171, "B": 2143.86379, "E": 1.81724, "alpha": 0.34731, "beta": 0.36718,
                  "objective": 0.0010182740346}

DELTAS = [1e-4, 1e-3, 1e-2, 1e-1, None]
HOLDOUT_FLOP = 1e21
N_EXCLUDED = 5


def load_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    n = np.array([float(r["Model Size"]) for r in rows])
    c = np.array([float(r["Training FLOP"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    return n, c, loss


def _fit_chunk(job):
    log_n, log_d, log_l, delta, inits, options = job
    best, runs = s.fit(log_n, log_d, log_l, delta=delta, inits=inits, options=options)
    return best, [(r["value"], r["theta"]) for r in runs]


def fit_grid(pool, log_n, log_d, log_l, delta, options=s.TIGHT, inits=None, chunks=16):
    """Full-grid fit, split across worker processes."""
    inits = s.hoffmann_grid() if inits is None else inits
    parts = [inits[i::chunks] for i in range(chunks)]
    out = pool.map(_fit_chunk, [(log_n, log_d, log_l, delta, p, options) for p in parts])
    best = min((b for b, _ in out), key=lambda r: r["value"])
    runs = [r for _, rs in out for r in rs]
    return best, runs


def _bootstrap_one(job):
    seed, log_n, log_d, log_l, warm = job
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(log_l), len(log_l))
    grid = s.hoffmann_grid()
    inits = list(warm) + [grid[i] for i in rng.choice(len(grid), 8, replace=False)]
    best, _ = s.fit(log_n[idx], log_d[idx], log_l[idx], delta=1e-3, inits=inits)
    return best["theta"]


def summarize(theta, objective_value=None):
    p = s.to_natural(theta)
    a, b = s.compute_optimal_exponents(p["alpha"], p["beta"])
    n_opt, d_opt = s.compute_optimal_allocation(p, s.CHINCHILLA_FLOP)
    out = dict(p, a=a, b=b, n_opt=n_opt, d_opt=d_opt, tokens_per_param=d_opt / n_opt)
    if objective_value is not None:
        out["objective"] = objective_value
    return out


def basins(runs, best_value, top=4):
    """Group grid end points by objective value to show where L-BFGS stops."""
    groups = {}
    for value, theta in runs:
        if not np.isfinite(value):
            continue
        key = float(f"{value:.6e}")
        g = groups.setdefault(key, {"count": 0, "theta": theta})
        g["count"] += 1
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["count"])[:top]
    total = sum(1 for v, _ in runs if np.isfinite(v))
    return [{"objective": k, "share": g["count"] / len(runs),
             "relative_to_best": k / best_value, **s.to_natural(g["theta"])} for k, g in ranked], total


def held_out_errors(params, n, d, loss):
    pred = s.predict_loss(params, n, d)
    log_err = np.log(pred) - np.log(loss)
    return {"rmse_log_loss": float(np.sqrt(np.mean(log_err ** 2))),
            "mean_abs_error": float(np.mean(np.abs(pred - loss))),
            "mean_signed_error": float(np.mean(pred - loss)),
            "pred": pred.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    t0 = time.time()

    n, c, loss = load_data()
    d = s.tokens_from_flops(c, n)
    keep = loss < np.sort(loss)[-N_EXCLUDED]
    n, c, d, loss = n[keep], c[keep], d[keep], loss[keep]
    log_n, log_d, log_l = np.log(n), np.log(d), np.log(loss)
    results = {"data": {"runs_total": int(keep.size), "runs_used": int(keep.sum()),
                        "excluded": f"{N_EXCLUDED} highest-loss runs, as in Epoch AI's analysis",
                        "flop_range": [float(c.min()), float(c.max())]}}

    with Pool(args.workers) as pool:
        # 1. Main fit and validation against Epoch AI.
        best, runs = fit_grid(pool, log_n, log_d, log_l, 1e-3)
        main_fit = summarize(best["theta"], best["value"])
        results["main_fit"] = main_fit
        results["epoch_grid_fit"] = EPOCH_GRID_FIT
        hoff_theta = s.to_theta(s.HOFFMANN_REPORTED)
        results["hoffmann_reported"] = summarize(hoff_theta, s.objective(hoff_theta, log_n, log_d, log_l)[0])
        print(f"main fit done ({time.time() - t0:.0f}s)")

        # 2. Where does L-BFGS stop from the standard grid?
        tight_basins, _ = basins(runs, best["value"])
        best_default, runs_default = fit_grid(pool, log_n, log_d, log_l, 1e-3, options=s.SCIPY_DEFAULT)
        default_basins, _ = basins(runs_default, best["value"])
        share_at_best = lambda rs: float(np.mean([v <= best["value"] * (1 + 1e-6) for v, _ in rs]))
        ratios = lambda rs: [float(v / best["value"]) for v, _ in rs if np.isfinite(v)]
        results["optimizer"] = {
            "grid_size": len(runs),
            "tight": {"share_reaching_best": share_at_best(runs), "top_end_points": tight_basins,
                      "objective_ratios": ratios(runs)},
            "scipy_default": {"share_reaching_best": share_at_best(runs_default),
                              "best": summarize(best_default["theta"], best_default["value"]),
                              "top_end_points": default_basins, "objective_ratios": ratios(runs_default)},
        }
        print(f"optimizer study done ({time.time() - t0:.0f}s)")

        # 3. Sensitivity to the Huber threshold (None = least squares in log space).
        sweep = []
        for delta in DELTAS:
            fit_d = (best, None) if delta == 1e-3 else fit_grid(pool, log_n, log_d, log_l, delta)
            sweep.append({"delta": delta, **summarize(fit_d[0]["theta"], fit_d[0]["value"])})
        results["delta_sweep"] = sweep
        print(f"delta sweep done ({time.time() - t0:.0f}s)")

        # 4. Extrapolation: fit below HOLDOUT_FLOP, predict the largest runs.
        train, test = c < HOLDOUT_FLOP, c >= HOLDOUT_FLOP
        extrap = {"train_runs": int(train.sum()), "test_runs": int(test.sum()),
                  "holdout_flop": HOLDOUT_FLOP, "test_loss": loss[test].tolist(), "estimators": {}}
        for name, delta in [("huber_1e-3", 1e-3), ("huber_1e-1", 1e-1), ("least_squares", None)]:
            b_tr, _ = fit_grid(pool, log_n[train], log_d[train], log_l[train], delta)
            p = s.to_natural(b_tr["theta"])
            extrap["estimators"][name] = {"params": p, **held_out_errors(p, n[test], d[test], loss[test])}
        extrap["reference_in_sample"] = {
            "hoffmann_reported": held_out_errors(s.HOFFMANN_REPORTED, n[test], d[test], loss[test]),
            "full_data_fit": held_out_errors(s.to_natural(best["theta"]), n[test], d[test], loss[test]),
        }
        results["extrapolation"] = extrap
        print(f"extrapolation done ({time.time() - t0:.0f}s)")

        # 5. Bootstrap uncertainty for the main estimator.
        warm = [best["theta"], hoff_theta]
        jobs = [(args.seed * 100000 + i, log_n, log_d, log_l, warm) for i in range(args.bootstrap)]
        thetas = pool.map(_bootstrap_one, jobs, chunksize=8)
    boot = [summarize(t) for t in thetas]
    keys = ["E", "A", "B", "alpha", "beta", "a", "b", "tokens_per_param"]
    ci = {k: [float(np.percentile([r[k] for r in boot], q)) for q in (2.5, 97.5)] for k in keys}
    hoff = results["hoffmann_reported"]
    results["bootstrap"] = {
        "resamples": args.bootstrap, "seed": args.seed, "ci95": ci,
        "hoffmann_inside_ci": {k: bool(ci[k][0] <= hoff[k] <= ci[k][1]) for k in keys},
        "samples": {k: [r[k] for r in boot] for k in ["alpha", "beta", "a"]},
    }
    results["runtime_seconds"] = round(time.time() - t0, 1)
    print(f"bootstrap done ({time.time() - t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    import figures
    figures.make_all(results)
    print("wrote results/results.json and figures/")


if __name__ == "__main__":
    main()
