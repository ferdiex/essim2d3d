"""
tabla4_figura4_arquitectura.py

Builds Table 4 and Figure 4 of the paper: compares the residual
pathway (direct shortcut from perception to decision) against the
memory-mediated pathway, with and without social signal, and the AUC
of two simple models (memory alone vs. perception alone) predicting
whether the episode ends in success.

Real data source: social_signal_diagnosis_report.csv, output of
diagnose_social_signal_effect.py on h_x_actraw_300.csv.

Plot labels/titles and table column names are in Spanish (the paper's
language). Code, comments and console output are in English.

Usage:
    python3 tabla4_figura4_arquitectura.py --out_dir .
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def build(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    residual = df[df["diagnostico"] == "residual_vs_heard_signal"].copy()
    auc = df[df["diagnostico"] == "h_vs_x_predice_exito"].copy()

    print("=== Table 4a: residual pathway vs. memory, with/without social signal ===")
    print(residual[["grupo", "n", "mediana_ratio_residual_sobre_h", "pct_residual_dominante"]]
          .to_string(index=False))
    print("\n=== Table 4b: AUC predicting episode success ===")
    print(auc[["grupo", "n", "auc"]].to_string(index=False))

    out_csv = os.path.join(out_dir, "tabla4_arquitectura.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFull table written to {out_csv}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # --- English plot text below ---
    labels1 = ["With social\nsignal", "Without social\nsignal"]
    values1 = residual["mediana_ratio_residual_sobre_h"].values
    ax1.bar(labels1, values1, color=["#C44E52", "#4C72B0"])
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1,
                label="Both pathways weigh equally")
    ax1.set_ylabel("Median (residual pathway / memory pathway)")
    ax1.set_title("(a) Residual pathway dominance")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    labels2 = ["Memory\nalone", "Perception\nalone"]
    values2 = auc["auc"].values
    ax2.bar(labels2, values2, color=["#55A868", "#DD8452"])
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance level")
    ax2.set_ylabel("AUC predicting episode success")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Which predicts the outcome better?")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Controller architecture: residual pathway and memory", fontsize=11)
    # --- end of English plot text ---

    fig.tight_layout(rect=[0.02, 0, 1, 1])
    out_png = os.path.join(out_dir, "figura4_arquitectura.png")
    fig.savefig(out_png, dpi=200)
    print(f"Figure written to {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="social_signal_diagnosis_report.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
