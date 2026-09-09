"""
tabla_pooling_y_suavidad.py

Builds two complementary tables for the paper:

Table 2b: leave-one-out result -- training on four pooled policies and
evaluating on the fifth, for each of the five policies, at horizon 50.

Table 2c: approach-speed metrics per policy (used to test, and reject,
the hypothesis that policy sesgo_check1_150 approaches the goal more
smoothly than the others).

Real data sources:
  - rollout_report_leave_one_out_pooled.csv
  - approach_speed_report.csv

Column names in the output tables are in Spanish (the paper's
language). Code, comments and console output are in English.

Usage:
    python3 tabla_pooling_y_suavidad.py --out_dir .
"""

import argparse
import os

import pandas as pd


def build_tables(leave_one_out_path, approach_speed_path, out_dir):
    loo = pd.read_csv(leave_one_out_path)
    loo50 = loo[loo["horizon"] == 50].copy()
    loo50["politica_evaluada"] = loo50["variant"].str.replace("pooled_holdout_", "", regex=False)
    name_map = {
        "h_x_actraw_300": "v15",
        "h_x_actraw_v14pilot_30": "v14",
        "h_x_actraw_social3d_v10_nn_annealing_gen143_30": "v10_gen143",
        "h_x_actraw_social3d_indep2_final_30": "indep2",
        "h_x_actraw_social3d_sesgo_check1_150_final_30": "sesgo_check1_150",
    }
    loo50["politica_evaluada"] = loo50["politica_evaluada"].replace(name_map)

    tabla_loo = loo50[["politica_evaluada", "n_trajectories", "A_x_rmse", "A_h_rmse",
                        "C_success_hit_rate", "C_success_n_real_within_horizon"]]
    tabla_loo = tabla_loo.rename(columns={
        "n_trajectories": "n_trayectorias",
        "C_success_hit_rate": "tasa_acierto_exito",
        "C_success_n_real_within_horizon": "n_exitos_reales",
    })
    out_csv1 = os.path.join(out_dir, "tabla_leave_one_out.csv")
    tabla_loo.to_csv(out_csv1, index=False)
    print(f"Table written to {out_csv1}")
    print(tabla_loo.to_string(index=False))

    print()
    speed = pd.read_csv(approach_speed_path)
    speed = speed.rename(columns={
        "policy": "politica",
        "n_success_trajectories": "n_trayectorias_exitosas",
        "mean_abs_step_delta": "cambio_promedio_por_paso",
        "std_step_delta": "variabilidad_por_paso",
        "median_decision_of_success": "mediana_decision_de_exito",
    })
    speed = speed[["politica", "n_trayectorias_exitosas", "cambio_promedio_por_paso",
                    "variabilidad_por_paso", "mediana_decision_de_exito"]]
    out_csv2 = os.path.join(out_dir, "tabla_suavidad_aproximacion.csv")
    speed.to_csv(out_csv2, index=False)
    print(f"Table written to {out_csv2}")
    print(speed.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leave_one_out", default="rollout_report_leave_one_out_pooled.csv")
    ap.add_argument("--approach_speed", default="approach_speed_report.csv")
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    build_tables(args.leave_one_out, args.approach_speed, args.out_dir)


if __name__ == "__main__":
    main()
