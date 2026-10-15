"""Figures for the README, written to figures/."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", name), dpi=160, bbox_inches="tight")
    plt.close(fig)


def bootstrap_exponents(res):
    boot = res["bootstrap"]["samples"]
    main, hoff = res["main_fit"], res["hoffmann_reported"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(boot["alpha"], boot["beta"], s=9, color=SERIES[0], alpha=0.35, linewidths=0,
               label=f"Bootstrap refits (n={len(boot['alpha'])})")
    ax.scatter([main["alpha"]], [main["beta"]], s=70, color=SERIES[2], edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Huber refit, all 240 runs")
    ax.scatter([hoff["alpha"]], [hoff["beta"]], s=70, marker="D", color=SERIES[1], edgecolor=SURFACE,
               linewidth=1.5, zorder=3, label="Hoffmann et al. reported")
    ax.set_xlabel("alpha (model-size exponent)")
    ax.set_ylabel("beta (data exponent)")
    ax.set_title("Exponent uncertainty from 1,000 bootstrap refits")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "bootstrap_exponents.png")


def optimizer_end_points(res):
    opt = res["optimizer"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for i, (key, label) in enumerate([("tight", "Tight tolerance"), ("scipy_default", "SciPy defaults")]):
        r = np.sort(np.maximum(np.array(opt[key]["objective_ratios"]) - 1.0, 1e-12))
        y = np.arange(1, len(r) + 1) / len(r)
        ax.step(r, y, where="post", color=SERIES[i], linewidth=1.75, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Final objective relative to best, minus 1 (floored at 1e-12)")
    ax.set_ylabel(f"Share of the {opt['grid_size']:,} grid starts")
    ax.set_title("Where L-BFGS-B stops from the standard init grid")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "optimizer_end_points.png")


def delta_sweep(res):
    rows = res["delta_sweep"]
    labels = ["1e-4", "1e-3", "1e-2", "1e-1", "Least\nsquares"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for i, key in enumerate(["alpha", "beta"]):
        y = [r[key] for r in rows]
        ax.plot(x, y, color=SERIES[i], linewidth=1.75, marker="o", markersize=6,
                markeredgecolor=SURFACE, label=key)
        ax.annotate(key, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Huber threshold delta")
    ax.set_ylabel("Fitted exponent")
    ax.set_title("Fitted exponents across loss functions")
    ax.legend(loc="best", fontsize=9)
    _save(fig, "delta_sweep.png")


def extrapolation(res):
    ex = res["extrapolation"]
    actual = np.array(ex["test_loss"])
    names = [("huber_1e-3", "Huber, delta=1e-3"), ("least_squares", "Least squares")]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    lo, hi = actual.min() - 0.03, actual.max() + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--", label="Perfect prediction")
    for i, (key, label) in enumerate(names):
        pred = np.array(ex["estimators"][key]["pred"])
        ax.scatter(actual, pred, s=36, color=SERIES[i], edgecolor=SURFACE, linewidth=1,
                   label=f"{label} (MAE {ex['estimators'][key]['mean_abs_error']:.3f})", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Actual loss, held-out runs")
    ax.set_ylabel("Predicted loss")
    flop = f"{ex['holdout_flop']:.0e}".replace("e+", "e")
    ax.set_title(f"Fit below {flop} FLOP, predict the {ex['test_runs']} largest runs")
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "extrapolation.png")


def _describe(end_point):
    if end_point["relative_to_best"] < 1.0 + 1e-6:
        return "Best fit"
    if end_point["alpha"] > 5 and end_point["beta"] > 5:
        return "Both terms vanish"
    if end_point["beta"] > end_point["alpha"]:
        return "Data term vanishes"
    return "Model-size term vanishes"


def endpoint_shares(res):
    opt = res["optimizer"]["tight"]
    rows = [(f"{_describe(e)}  ({e['relative_to_best']:.1f}x loss)", e["share"], _describe(e) == "Best fit")
            for e in opt["top_end_points"]]
    rows.append(("Other end points", 1.0 - sum(r[1] for r in rows), False))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    y = np.arange(len(rows))
    colors = [SERIES[0] if best else "#b9b8b2" for _, _, best in rows]
    ax.barh(y, [share for _, share, _ in rows], color=colors, height=0.62)
    for yi, (_, share, _) in zip(y, rows):
        ax.annotate(f"{share:.1%}", (share, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(share for _, share, _ in rows) * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Share of the {res['optimizer']['grid_size']:,} standard grid starts (L-BFGS-B, tight tolerance)")
    ax.set_title("Only about a third of the standard starts reach the best fit", loc="left")
    _save(fig, "endpoint_shares.png")


def make_all(res):
    endpoint_shares(res)
    bootstrap_exponents(res)
    optimizer_end_points(res)
    delta_sweep(res)
    extrapolation(res)
