"""
spearman_significance.py

Reads one or more dream_screen_vs_real_report.csv files (columns:
candidate, real_fitness, real_time_s, dreamed_score, dream_time_s) and
reports the Spearman rank correlation between real_fitness and
dreamed_score for each, with a proper p-value and 95% CI (Fisher z
transform) -- not just the raw rho, which alone doesn't say whether a
result like 0.134 is meaningfully different from zero.

Usage:
    python3 spearman_significance.py \
        --csvs results_robustness/seed1_sigma0.03/dream_screen_vs_real_report.csv,results_robustness/seed1_sigma0.06/dream_screen_vs_real_report.csv,... \
        --labels seed1_s0.03,seed1_s0.06,...
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def spearman_with_ci(real_fitness, dreamed_score):
    rho, p = stats.spearmanr(real_fitness, dreamed_score)
    n = len(real_fitness)
    if n < 4 or abs(rho) >= 1.0:
        return rho, p, (float("nan"), float("nan")), n
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return rho, p, (lo, hi), n


def analyze(csv_path, col_x, col_y):
    df = pd.read_csv(csv_path)
    return spearman_with_ci(df[col_x].values, df[col_y].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", required=True, help="comma-separated report CSV paths")
    ap.add_argument("--labels", help="comma-separated labels, same order as --csvs")
    ap.add_argument("--col_x", default="real_fitness",
                     help="column name for the 'ground truth' score (default matches "
                          "dream_screen_vs_real.py; use full_fitness for truncated_eval_vs_full.py)")
    ap.add_argument("--col_y", default="dreamed_score",
                     help="column name for the cheap proxy score (default matches "
                          "dream_screen_vs_real.py; use truncated_fitness for truncated_eval_vs_full.py)")
    args = ap.parse_args()

    csv_paths = [c.strip() for c in args.csvs.split(",")]
    labels = ([l.strip() for l in args.labels.split(",")] if args.labels
              else [f"run{i}" for i in range(len(csv_paths))])
    if len(labels) != len(csv_paths):
        raise SystemExit("--csvs and --labels must have the same length, same order.")

    rows = []
    print(f"{'label':20s} {'n':>4s} {'rho':>8s} {'p':>8s} {'95% CI':>18s}  significant?")
    for label, path in zip(labels, csv_paths):
        rho, p, (lo, hi), n = analyze(path, args.col_x, args.col_y)
        sig = "YES" if p < 0.05 else "no"
        print(f"{label:20s} {n:4d} {rho:8.3f} {p:8.4f}  ({lo:6.3f}, {hi:6.3f})    {sig}")
        rows.append({"label": label, "n": n, "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                      "significant_p<0.05": p < 0.05})

    out = pd.DataFrame(rows)
    n_sig = out["significant_p<0.05"].sum()
    print(f"\n{n_sig} of {len(out)} runs reached p<0.05.")

    # Pooled fixed-effect meta-analysis across all runs (Fisher z) --
    # the number that actually matters, not the per-run count.
    ns = out["n"].values
    zs = np.arctanh(out["rho"].values)
    z_pooled = np.average(zs, weights=(ns - 3))
    se_pooled = 1 / np.sqrt(np.sum(ns - 3))
    rho_pooled = np.tanh(z_pooled)
    p_pooled = 2 * (1 - stats.norm.cdf(abs(z_pooled / se_pooled)))
    lo_pooled, hi_pooled = np.tanh(z_pooled - 1.96 * se_pooled), np.tanh(z_pooled + 1.96 * se_pooled)
    print(f"\nPOOLED (fixed-effect, all {len(out)} runs): rho={rho_pooled:.3f}  "
          f"p={p_pooled:.6f}  95% CI=({lo_pooled:.3f}, {hi_pooled:.3f})")
    if p_pooled < 0.05 and lo_pooled * hi_pooled > 0:
        print("Significant AND the CI doesn't cross zero -- this is the number to trust, "
              "not the per-run sign pattern.")
    else:
        print("Not a clean significant result once pooled -- treat individual significant "
              "runs above with caution (this is exactly the pattern that didn't replicate "
              "for A2 earlier this session).")


if __name__ == "__main__":
    main()
