"""
analyze_approach_speed.py

Follow-up to the leave-one-out pooling check
(rollout_report_leave_one_out_pooled.csv): sesgo_check1_150 was both the
hardest policy to predict (worst raw error of the five) and the only
one where the chained rollout ever detected a real success (2/9).
v15, with far more real-success opportunities (23 within 50 steps),
scored 0. If it were just sample-size luck, v15 -- with more chances --
should have been at least as likely to hit one, not less. That argues
against pure chance and for something structurally different in how
sesgo_check1 approaches the goal.

Hypothesis tested here (offline, using ONLY the CSVs already collected
-- no simulation, no PyBullet): sesgo_check1 closes the final distance
to the goal more SMOOTHLY (smaller, more regular per-step changes) than
the other policies, giving a noisy regressor a wider window to catch
the threshold crossing even with worse raw accuracy elsewhere. A policy
that closes the gap in large, jagged steps would be much harder to
"catch" exactly, regardless of how good the model is on average.

Method: for every real trajectory that actually succeeds (own_success
column), reconstruct distance-to-food from x[9] (same inversion used
throughout: dist = -hunger_decay * ln(x9)) over the last `--window`
decisions leading up to the first real success step. Compute, per
trajectory, the step-to-step change in that distance, then aggregate
across all successful trajectories of a policy:
  - mean_step_delta: average signed change per step (should be
    negative -- distance shrinking -- sanity check).
  - mean_abs_step_delta: average size of the per-step change,
    regardless of direction. Smaller = smoother approach.
  - std_step_delta: how much the per-step change itself varies.
    Smaller = more regular/predictable approach, not just slow.
  - max_abs_step_delta: the single roughest jump seen in that window,
    across all successful trajectories -- a smooth-on-average approach
    that still has occasional big jumps would still be hard to catch.
  - median_decision_of_success: how many decisions into the episode
    the success happens (context, not part of the hypothesis).

Usage:
    python3 analyze_approach_speed.py \
        --csvs ../h_x_actraw_300.csv,../h_x_actraw_v14pilot_30.csv,../h_x_actraw_social3d_v10_nn_annealing_gen143_30.csv,../h_x_actraw_social3d_indep2_final_30.csv,../h_x_actraw_social3d_sesgo_check1_150_final_30.csv \
        --labels v15,v14,v10_gen143,indep2,sesgo_check1_150 \
        --window 10 \
        --out_dir results_approach_speed/

Usage (self-test: hand-built synthetic data with a KNOWN smooth policy
and a KNOWN jagged policy, checks the metrics actually tell them apart
-- a correctness check of the math, not just "doesn't crash"):
    python3 analyze_approach_speed.py --selftest
"""

import argparse
import os

import numpy as np
import pandas as pd

import compare_world_model_candidates as wm  # noqa: F401 -- kept imported so this
                                              # script fails loudly/early if the
                                              # sys.path setup for gru_dynamics
                                              # (see that module's docstring) is
                                              # broken in this environment, same
                                              # as the other two scripts
import validate_rollout_fidelity as vr


def analyze_policy(csv_path, window):
    df = pd.read_csv(csv_path)
    required = ["episode", "agent", "decision", "x_9", "own_success"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns needed for this analysis: {missing}")
    grouped = df.sort_values(["episode", "agent", "decision"]).groupby(["episode", "agent"])

    step_deltas_all = []       # every per-step delta, across all trajectories/windows
    per_traj_mean_delta = []   # one number per successful trajectory
    per_traj_std_delta = []
    success_decisions = []

    for (ep, agent), g in grouped:
        succ_idx = np.where(g["own_success"].values)[0]
        if len(succ_idx) == 0:
            continue
        first_succ = int(succ_idx[0])
        if first_succ < 1:
            continue  # no room for a window before it

        decay = vr.HUNGER_DECAY[agent]
        x9 = g["x_9"].values
        dist = vr.dist_from_hunger(x9, decay)

        start = max(0, first_succ - window)
        seg = dist[start:first_succ + 1]
        if len(seg) < 2:
            continue
        deltas = np.diff(seg)

        step_deltas_all.extend(deltas.tolist())
        per_traj_mean_delta.append(float(np.mean(deltas)))
        per_traj_std_delta.append(float(np.std(deltas)))
        success_decisions.append(first_succ)

    n = len(per_traj_mean_delta)
    if n == 0:
        return {"n_success_trajectories": 0}

    step_deltas_all = np.array(step_deltas_all)
    return {
        "n_success_trajectories": n,
        "mean_step_delta": float(np.mean(step_deltas_all)),
        "mean_abs_step_delta": float(np.mean(np.abs(step_deltas_all))),
        "std_step_delta": float(np.mean(per_traj_std_delta)),  # avg of per-traj std
        "max_abs_step_delta": float(np.max(np.abs(step_deltas_all))),
        "median_decision_of_success": float(np.median(success_decisions)),
    }


def run(csv_paths, labels, window, out_dir):
    rows = []
    for path, label in zip(csv_paths, labels):
        print(f"\n=== {label} ===")
        result = analyze_policy(path, window)
        result["policy"] = label
        if result["n_success_trajectories"] == 0:
            print("  No real successes found in this CSV -- skipping.")
            rows.append(result)
            continue
        print(f"  n_success_trajectories = {result['n_success_trajectories']}")
        print(f"  mean_step_delta        = {result['mean_step_delta']:.5f} "
              f"(negative = approaching, as expected)")
        print(f"  mean_abs_step_delta    = {result['mean_abs_step_delta']:.5f} "
              f"(smaller = smoother approach)")
        print(f"  std_step_delta         = {result['std_step_delta']:.5f} "
              f"(smaller = more regular, not just slow)")
        print(f"  max_abs_step_delta     = {result['max_abs_step_delta']:.5f} "
              f"(single roughest jump seen)")
        print(f"  median_decision_of_success = {result['median_decision_of_success']:.1f}")
        rows.append(result)

    df_out = pd.DataFrame(rows)
    print("\n=== Summary, sorted by mean_abs_step_delta (smoothest first) ===")
    valid = df_out[df_out["n_success_trajectories"] > 0].sort_values("mean_abs_step_delta")
    print(valid[["policy", "n_success_trajectories", "mean_abs_step_delta",
                  "std_step_delta", "max_abs_step_delta"]].to_string(index=False))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "approach_speed_report.csv")
        df_out.to_csv(report_path, index=False)
        print(f"\nFull report written to {report_path}")
    return df_out


def run_selftest():
    print("=== SELF-TEST: hand-built smooth vs jagged synthetic policies ===\n")
    rng = np.random.default_rng(0)

    def make_policy_csv(path, jagged, n_traj=20, n_steps=30):
        rows = []
        for ep in range(n_traj):
            dist = 1.0
            success_step = None
            for dec in range(n_steps):
                if jagged:
                    step = rng.uniform(0.02, 0.25)  # rough, irregular steps
                else:
                    step = 0.03 + rng.normal(0, 0.003)  # smooth, regular steps
                dist = max(0.0, dist - step)
                x9 = float(np.exp(-dist / 1.0))  # agent 'a', decay=1.0
                success = dist < vr.SUCCESS_THRESHOLD
                rows.append({
                    "episode": ep, "agent": "a", "decision": dec,
                    "x_9": x9, "own_success": success,
                })
                if success and success_step is None:
                    success_step = dec
                if success_step is not None and dec > success_step + 2:
                    break
        pd.DataFrame(rows).to_csv(path, index=False)

    smooth_path, jagged_path = "/tmp/_smooth_policy.csv", "/tmp/_jagged_policy.csv"
    make_policy_csv(smooth_path, jagged=False)
    make_policy_csv(jagged_path, jagged=True)

    df_out = run([smooth_path, jagged_path], ["smooth", "jagged"], window=10, out_dir=None)
    smooth_row = df_out[df_out["policy"] == "smooth"].iloc[0]
    jagged_row = df_out[df_out["policy"] == "jagged"].iloc[0]

    assert smooth_row["mean_abs_step_delta"] < jagged_row["mean_abs_step_delta"], (
        "FAILED: smooth policy should have smaller mean_abs_step_delta than jagged")
    assert smooth_row["std_step_delta"] < jagged_row["std_step_delta"], (
        "FAILED: smooth policy should have smaller std_step_delta than jagged")
    print("\nSelf-test finished without errors -- metrics correctly separate "
          "the smooth and jagged synthetic policies as designed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", help="comma-separated list of h_x_actraw CSVs")
    ap.add_argument("--labels", help="comma-separated labels, same order as --csvs")
    ap.add_argument("--window", type=int, default=10,
                     help="how many decisions before the real success step to look at")
    ap.add_argument("--out_dir", default="results_approach_speed")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.csvs or not args.labels:
        raise SystemExit("Need --csvs and --labels (or use --selftest).")

    csv_paths = [c.strip() for c in args.csvs.split(",")]
    labels = [l.strip() for l in args.labels.split(",")]
    if len(csv_paths) != len(labels):
        raise SystemExit("--csvs and --labels must have the same length, same order.")

    run(csv_paths, labels, args.window, args.out_dir)


if __name__ == "__main__":
    main()
