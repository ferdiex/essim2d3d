"""
figura3_forest_truncated_eval.py

Builds Figure 3 of the paper: forest plot of the Spearman correlation
between fitness evaluated with fewer steps (truncated_fitness) and
full-length fitness (full_fitness), for the three truncation levels
tested (750, 1000, 1500 steps), seed by seed and pooled.

Real data source: the output CSVs from truncated_eval_vs_full.py for
the v15 base policy (6 seeds at 1000 and 1500 steps, 3 seeds at 750
steps), session documented in README_world_model_addition.md.

Plot labels/titles are in Spanish (the paper's language). Code,
comments and console output are in English.

Usage:
    python3 figura3_forest_truncated_eval.py --data_dir . --out_dir .
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def spearman_ci(rho, n):
    if n < 4 or abs(rho) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def pooled(rhos, n):
    zs = np.array([np.arctanh(r) for r in rhos])
    z = np.mean(zs)
    se = 1 / np.sqrt((n - 3) * len(zs))
    rho = np.tanh(z)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    p = 2 * (1 - stats.norm.cdf(abs(z / se)))
    return rho, lo, hi, p


def build_figure(data_dir, out_dir):
    entries = []
    n_per_point = None
    pooled_results = {}

    for steps, seeds in [(750, [1, 2, 3]), (1000, [1, 2, 3, 4, 5, 6]), (1500, [1, 2, 3, 4, 5, 6])]:
        rhos = []
        for seed in seeds:
            path = os.path.join(data_dir, f"truncated_v15_{steps}pasos_seed{seed}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            n_per_point = len(df)
            rho, _ = stats.spearmanr(df["full_fitness"], df["truncated_fitness"])
            lo, hi = spearman_ci(rho, len(df))
            # English label -- this text renders inside the figure
            entries.append((f"{steps} steps, seed {seed}", rho, lo, hi, False, steps))
            rhos.append(rho)
        if rhos:
            rho_p, lo_p, hi_p, p_p = pooled(rhos, n_per_point)
            pooled_results[steps] = (rho_p, lo_p, hi_p, p_p, len(rhos))
            entries.append((f"{steps} steps (pooled, n={len(rhos)} seeds)",
                             rho_p, lo_p, hi_p, True, steps))

    color_by_steps = {750: "#4C72B0", 1000: "#55A868", 1500: "#C44E52"}

    fig, ax = plt.subplots(figsize=(9, 7))
    ypos = np.arange(len(entries))[::-1]
    for y, (label, rho, lo, hi, is_pooled, steps) in zip(ypos, entries):
        color = color_by_steps[steps]
        marker = "D" if is_pooled else "o"
        size = 90 if is_pooled else 40
        alpha = 1.0 if is_pooled else 0.6
        if not np.isnan(lo):
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5 if is_pooled else 1.0, alpha=alpha)
        ax.scatter([rho], [y], color=color, marker=marker, s=size, zorder=3, alpha=alpha)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([e[0] for e in entries])
    # --- English plot text ---
    ax.set_xlabel("Spearman correlation (truncated fitness vs. full fitness)")
    ax.set_title("Correlation between truncated evaluation and full evaluation,\nby number of steps evaluated", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(out_dir, "figura3_forest_truncated_eval.png")
    fig.savefig(out_png, dpi=200)
    print(f"Figure written to {out_png}\n")
    for steps, (rho_p, lo_p, hi_p, p_p, n_seeds) in pooled_results.items():
        approx_savings = {750: "≈2.7x", 1000: "≈1.9x", 1500: "≈1.3x"}[steps]
        print(f"{steps} steps ({n_seeds} seeds): rho={rho_p:.3f}  p={p_p:.4f}  "
              f"CI=({lo_p:.3f},{hi_p:.3f})  time savings {approx_savings}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_figure(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
