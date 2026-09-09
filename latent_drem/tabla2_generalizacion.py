"""
tabla2_generalizacion.py

Builds Table 2 of the paper: compares predictor fidelity when trained
and evaluated on the SAME evolutionary policy (v15) versus trained on
v15 and evaluated on a DIFFERENT policy (v14), never seen in training.

Real data sources:
  - rollout_report_n60ep_horizons15102050.csv (same policy, v15)
  - rollout_report_crosspolicy_trainh_x_actraw_300_evalh_x_actraw_v14pilot_30.csv
    (cross-policy, "unweighted" variant = unweighted Ridge)

Column names in the output table are in Spanish (the paper's language).
Code, comments and console output are in English.

Usage:
    python3 tabla2_generalizacion.py --out_dir .
"""

import argparse
import os

import pandas as pd


def build_table(same_policy_path, cross_policy_path, out_dir):
    same = pd.read_csv(same_policy_path)
    cross = pd.read_csv(cross_policy_path)
    cross = cross[cross["variant"] == "unweighted"]

    rows = []
    for h in [1, 5, 10, 20, 50]:
        s = same[same["horizon"] == h].iloc[0]
        c = cross[cross["horizon"] == h].iloc[0]
        rows.append({
            "horizonte": h,
            "x_rmse_misma_politica": round(s["A_x_rmse"], 4),
            "x_rmse_entre_politicas": round(c["A_x_rmse"], 4),
            "dist_mae_misma_politica": round(s["B_dist_to_food_mae"], 4),
            "dist_mae_entre_politicas": round(c["B_dist_to_food_mae"], 4),
        })
    table = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "tabla2_generalizacion.csv")
    table.to_csv(out_csv, index=False)
    print(f"Table written to {out_csv}")
    print(table.to_string(index=False))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same_policy", default="rollout_report_n60ep_horizons15102050.csv")
    ap.add_argument("--cross_policy",
                     default="rollout_report_crosspolicy_trainh_x_actraw_300_evalh_x_actraw_v14pilot_30.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_table(args.same_policy, args.cross_policy, args.out_dir)


if __name__ == "__main__":
    main()
