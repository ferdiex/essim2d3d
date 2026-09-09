"""
figura2_forest_dream_screen.py

Builds Figure 2 of the paper: forest plot of the Spearman correlation
between the candidate ranking produced by the world model
(dreamed_score) and the real ranking (real_fitness), seed by seed, plus
the pooled value for discovery, confirmation, and both combined.

Real data source: the 12 output CSVs from dream_screen_vs_real.py
(3 discovery seeds x 2 sigmas, 3 confirmation seeds x 2 sigmas),
session documented in README_world_model_addition.md.

Plot labels/titles are in Spanish (the paper's language). Code,
comments and console output are in English.

Usage:
    python3 figura2_forest_dream_screen.py --data_dir . --out_dir .
"""

import argparse
import glob
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
    n_per_point = None
    entries = []  # (label, rho, lo, hi, is_pooled)

    disc_rhos, conf_rhos = [], []
    for phase, phase_label, seeds in [("descubrimiento", "discovery", [1, 2, 3]),
                                        ("confirmacion", "confirmation", [4, 5, 6])]:
        for seed in seeds:
            for sigma_label, sigma_file in [("0.03", "0_03"), ("0.06", "0_06")]:
                pattern = os.path.join(data_dir, f"dream_screen_seed{seed}_sigma{sigma_file}_{phase}.csv")
                files = glob.glob(pattern)
                if not files:
                    continue
                df = pd.read_csv(files[0])
                n_per_point = len(df)
                rho, _ = stats.spearmanr(df["real_fitness"], df["dreamed_score"])
                lo, hi = spearman_ci(rho, len(df))
                # English label -- this text renders inside the figure
                label = f"{phase_label[:4]}. seed {seed}, σ={sigma_label}"
                entries.append((label, rho, lo, hi, False))
                (disc_rhos if phase == "descubrimiento" else conf_rhos).append(rho)

    rho_d, lo_d, hi_d, p_d = pooled(disc_rhos, n_per_point)
    rho_c, lo_c, hi_c, p_c = pooled(conf_rhos, n_per_point)
    rho_all, lo_all, hi_all, p_all = pooled(disc_rhos + conf_rhos, n_per_point)

    entries.append(("DISCOVERY (pooled)", rho_d, lo_d, hi_d, True))
    entries.append(("CONFIRMATION (pooled)", rho_c, lo_c, hi_c, True))
    entries.append(("ALL COMBINED (pooled)", rho_all, lo_all, hi_all, True))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ypos = np.arange(len(entries))[::-1]
    for y, (label, rho, lo, hi, is_pooled) in zip(ypos, entries):
        color = "#C44E52" if is_pooled else "#4C72B0"
        marker = "D" if is_pooled else "o"
        size = 70 if is_pooled else 40
        if not np.isnan(lo):
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5, alpha=0.7)
        ax.scatter([rho], [y], color=color, marker=marker, s=size, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([e[0] for e in entries])
    # --- English plot text ---
    ax.set_xlabel(r"Spearman correlation (dreamed ranking vs. real ranking)")
    ax.set_title("Correlation between candidate ranking from the world model\nand ranking from real simulation",
                  fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(out_dir, "figura2_forest_dream_screen.png")
    fig.savefig(out_png, dpi=200)
    print(f"Figure written to {out_png}")
    print(f"\nPooled discovery: rho={rho_d:.3f}  p={p_d:.4f}  CI=({lo_d:.3f},{hi_d:.3f})")
    print(f"Pooled confirmation: rho={rho_c:.3f}  p={p_c:.4f}  CI=({lo_c:.3f},{hi_c:.3f})")
    print(f"All combined: rho={rho_all:.3f}  p={p_all:.4f}  CI=({lo_all:.3f},{hi_all:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_figure(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
