"""
figura_tabla1_comparacion_modelos.py

Builds Table 1 and Figure 1a/1b of the paper: comparison of Ridge, MLP,
Random Forest, and Gradient Boosting for predicting x(t+1) and chained
h(t+1) via the exact F().

Real data sources (not simulated for this figure):
  - comparison_report_full.csv       (300 episodes, ridge and mlp)
  - comparison_report__subset30.csv  (30 episodes, all 4 models)

Both files are the direct output of compare_world_model_candidates.py
run on h_x_actraw_300.csv, research session documented in
README_world_model_addition.md.

Plot labels/titles are in English (matches the paper's final target
language). Code, comments and console output are in English.

Usage:
    python3 figura_tabla1_comparacion_modelos.py \
        --full comparison_report_full.csv \
        --subset comparison_report__subset30.csv \
        --out_dir .
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

TARGET_CHAINED = "h_next (chained via predicted x, NEW)"
TARGET_X = "x_next"

MODEL_ORDER = ["ridge", "mlp", "random_forest", "gradient_boosting"]
MODEL_LABELS = {"ridge": "Ridge", "mlp": "MLP",
                 "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting"}


def build_table(full_path, subset_path, out_dir):
    full = pd.read_csv(full_path)
    subset = pd.read_csv(subset_path)

    rows = []
    for model in MODEL_ORDER:
        r2_full = full[(full["target"] == TARGET_CHAINED) & (full["model"] == model)]["r2_overall"]
        r2_subset = subset[(subset["target"] == TARGET_CHAINED) & (subset["model"] == model)]["r2_overall"]
        fit_subset = subset[(subset["target"] == TARGET_X) & (subset["model"] == model)]["fit_seconds"]
        rows.append({
            "modelo": MODEL_LABELS[model],
            "R2_h_chained_n300ep": float(r2_full.iloc[0]) if len(r2_full) else None,
            "R2_h_chained_n30ep": float(r2_subset.iloc[0]) if len(r2_subset) else None,
            "tiempo_ajuste_seg_n30ep": float(fit_subset.iloc[0]) if len(fit_subset) else None,
        })
    table = pd.DataFrame(rows)

    out_csv = os.path.join(out_dir, "tabla1_comparacion_modelos.csv")
    table.to_csv(out_csv, index=False)
    print(f"Table written to {out_csv}")
    print(table.to_string(index=False))
    return table


def build_figure(table, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    modelos = table["modelo"]
    r2_300 = table["R2_h_chained_n300ep"]
    r2_30 = table["R2_h_chained_n30ep"]

    x = range(len(modelos))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], r2_300, width, label="300 episodes", color="#4C72B0")
    ax1.bar([i + width / 2 for i in x], r2_30, width, label="30 episodes", color="#DD8452")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(modelos, rotation=20, ha="right")
    ax1.set_ylabel(r"$R^2$ of chained $h(t{+}1)$")
    ax1.set_ylim(0.85, 1.0)
    ax1.set_title("(a) Accuracy by model")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(modelos, table["tiempo_ajuste_seg_n30ep"], color="#55A868")
    ax2.set_yscale("log")
    ax2.set_ylabel("Fit time (s, log scale)")
    ax2.set_title("(b) Training cost (30 episodes)")
    ax2.set_xticks(range(len(modelos)))
    ax2.set_xticklabels(modelos, rotation=20, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Comparison of candidate models for the physics predictor",
                  fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "figura1_comparacion_modelos.png")
    fig.savefig(out_png, dpi=200)
    print(f"Figure written to {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="comparison_report_full.csv")
    ap.add_argument("--subset", default="comparison_report__subset30.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    table = build_table(args.full, args.subset, args.out_dir)
    build_figure(table, args.out_dir)


if __name__ == "__main__":
    main()
