"""
tabla3_duracion_vs_cantidad.py

Builds Table 3 of the paper: compares two ways of cutting evaluation
cost by a similar factor (~1.3x) -- shortening each episode's LENGTH
(1500 of 2000 steps) versus evaluating fewer full-length episodes
(7 of 9). Both are already-confirmed results from the same session
(discovery + confirmation, 6 seeds each), so this table pools each
side the same way and reports them side by side.

Real data sources:
  - truncated_v15_1500pasos_seed{1..6}.csv   (length-truncation side)
  - episode_baseline_report_..._seed{1..6}.csv (episode-count side)

Table/figure text is written in Spanish (paper language). Code,
comments and console output are in English.

Usage:
    python3 tabla3_duracion_vs_cantidad.py --data_dir . --out_dir .
"""

import argparse
import glob
import os

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


def collect_rhos(paths, col_a, col_b):
    rhos, n = [], None
    for p in paths:
        df = pd.read_csv(p)
        n = len(df)
        rho, _ = stats.spearmanr(df[col_a], df[col_b])
        rhos.append(rho)
    return rhos, n


def format_p(p):
    """Round(p, 6) silently collapses very small p-values (e.g. 1.8e-11)
    to exactly 0.0, losing the information that it's tiny-but-nonzero.
    Keep a floor label instead of a false zero."""
    if p < 1e-6:
        return "<0.000001"
    return round(p, 6)


def build_table(data_dir, out_dir):
    length_paths = sorted(glob.glob(os.path.join(data_dir, "truncated_v15_1500pasos_seed*.csv")))
    count_paths = sorted(glob.glob(os.path.join(
        data_dir, "episode_baseline_report_*_ep9_reduced7_*_seed*.csv")))

    length_rhos, n_length = collect_rhos(length_paths, "full_fitness", "truncated_fitness")
    count_rhos, n_count = collect_rhos(count_paths, "full_fitness", "reduced_fitness")

    rho_len, lo_len, hi_len, p_len = pooled(length_rhos, n_length)
    rho_cnt, lo_cnt, hi_cnt, p_cnt = pooled(count_rhos, n_count)

    table = pd.DataFrame([
        {"metodo": "Duración reducida (1500 de 2000 pasos)",
         "n_semillas": len(length_rhos), "rho_agrupado": round(rho_len, 3),
         "p": format_p(p_len), "ic_95_inf": round(lo_len, 3), "ic_95_sup": round(hi_len, 3),
         "ahorro_aprox": "1.3x"},
        {"metodo": "Cantidad reducida (7 de 9 episodios)",
         "n_semillas": len(count_rhos), "rho_agrupado": round(rho_cnt, 3),
         "p": format_p(p_cnt), "ic_95_inf": round(lo_cnt, 3), "ic_95_sup": round(hi_cnt, 3),
         "ahorro_aprox": "1.29x"},
    ])

    out_csv = os.path.join(out_dir, "tabla3_duracion_vs_cantidad.csv")
    table.to_csv(out_csv, index=False)
    print(f"Table written to {out_csv}\n")
    print(table.to_string(index=False))

    print(f"\nPer-seed rhos, length truncation: {[round(r, 3) for r in length_rhos]}")
    print(f"Per-seed rhos, count reduction:    {[round(r, 3) for r in count_rhos]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_table(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
