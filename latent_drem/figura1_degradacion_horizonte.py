"""
figura1_degradacion_horizonte.py

Builds Figure 1 of the paper: how the world-model predictor's accuracy
(Ridge, weighted Ridge, MLP) degrades as the prediction horizon grows,
and how the success-event hit rate changes along with it.

Real data source: rollout_report_n60ep_horizons15102050100150200_ridge_vs_weighted_vs_mlp.csv
(or the shorter 1-50 version), direct output of validate_rollout_fidelity.py,
session documented in README_world_model_addition.md.

Plot labels/titles are in Spanish (the paper's language). Code,
comments and console output are in English.

Usage:
    python3 figura1_degradacion_horizonte.py \
        --csv rollout_report_n60ep_horizons15102050100150200_ridge_vs_weighted_vs_mlp.csv \
        --out_dir .
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

VARIANT_LABELS = {"unweighted": "Ridge", "weighted": "Ridge weighted", "mlp": "MLP"}
VARIANT_COLORS = {"unweighted": "#4C72B0", "weighted": "#DD8452", "mlp": "#C44E52"}


def build_figure(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax_x, ax_dist, ax_hit = axes

    for variant in ["unweighted", "weighted", "mlp"]:
        sub = df[df["variant"] == variant].sort_values("horizon")
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]

        ax_x.plot(sub["horizon"], sub["A_x_rmse"], marker="o", color=color, label=label)
        ax_dist.plot(sub["horizon"], sub["B_dist_to_food_mae"], marker="o", color=color, label=label)

        hit = sub.dropna(subset=["C_success_hit_rate"])
        if len(hit):
            ax_hit.plot(hit["horizon"], hit["C_success_hit_rate"], marker="o", color=color, label=label)

    # --- Plot text below is in English (matches final paper language) ---
    ax_x.set_xlabel("Horizon (decisions)")
    ax_x.set_ylabel(r"RMSE of $x(t{+}h)$")
    ax_x.set_title("(a) Raw state error")
    ax_x.grid(alpha=0.3)
    ax_x.legend(fontsize=8)

    ax_dist.axhline(0.08, color="gray", linestyle="--", linewidth=1,
                     label="Success threshold (0.08 m)")
    ax_dist.set_xlabel("Horizon (decisions)")
    ax_dist.set_ylabel("Mean absolute error, distance to goal (m)")
    ax_dist.set_title("(b) Error in distance to goal")
    ax_dist.grid(alpha=0.3)
    ax_dist.legend(fontsize=8)

    ax_hit.set_xlabel("Horizon (decisions)")
    ax_hit.set_ylabel("Success-event hit rate")
    ax_hit.set_title("(c) Success-event detection")
    ax_hit.set_ylim(-0.05, 1.05)
    ax_hit.grid(alpha=0.3)
    ax_hit.legend(fontsize=8)

    fig.suptitle("Degradation of world-model fidelity with the prediction horizon",
                  fontsize=11)
    # --- end of English plot text ---

    fig.tight_layout()
    out_png = os.path.join(out_dir, "figura1_degradacion_horizonte.png")
    fig.savefig(out_png, dpi=200)
    print(f"Figure written to {out_png}")

    # Note on panel (c): at horizon 1-20 there were no real success cases
    # within the window (C_success_n_real_within_horizon = 0), so those
    # points have no defined hit rate and don't appear on the curve.
    # This is itself part of the finding and is worth a note in the
    # figure caption or the main text.
    n_real = df[["variant", "horizon", "C_success_n_real_within_horizon"]]
    print("\nReal success cases within the window, by horizon "
          "(0 means the hit-rate curve is undefined there):")
    print(n_real[n_real["variant"] == "unweighted"].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rollout_report_n60ep_horizons15102050100150200_ridge_vs_weighted_vs_mlp.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_figure(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
