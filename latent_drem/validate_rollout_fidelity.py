"""
validate_rollout_fidelity.py

Groundwork for Paso 3 of the world-model plan ("validar fidelidad de
rollouts largos, midiendo degradacion del error acumulado"), scoped to
A+B+C as discussed (D -- Spearman rank correlation across trajectories
-- deliberately left out for now, saved for when the fitness substitute
is actually wired into the genetic algorithm, Paso 5).

Trains the Ridge x(t+1) predictor (chosen over MLP/RF/GBM: statistically
indistinguishable on chained h(t+1) R^2 in the full-300-episode
comparison, at a fraction of the cost -- see comparison_report_n300ep_
ridge_mlp.csv) on TRAIN episodes, then on held-out TEST episodes runs a
full chained rollout:

    h_0, x_0 = real starting state of the trajectory
    for t in range(horizon):
        x_pred(t+1) = ridge.predict([h_t, x_t, one-hot(action_t REAL)])
        h_pred(t+1) = F(h_t, x_pred(t+1))     <- exact, known recurrence
        h_t, x_t = h_pred(t+1), x_pred(t+1)   <- chain on OWN predictions

The REAL recorded action at each step is used (not re-derived from the
imagined state) on purpose: this isolates the world model's own
predictive error from a different, separate question (whether a policy
would make the same decisions given imagined state) -- consistent with
what we agreed before building this.

Metrics:

  A) Raw error by horizon: mean RMSE across x's 11 dims and h's 16 dims,
     real vs imagined, at horizons 1/5/10/20/50 (each rollout run once,
     per-horizon values read off the running chain, not re-run per
     horizon).

  B) Quantities that actually drive the fitness function (checked
     against controllers.py / evaluate3d.py, not assumed):
       - x[9] = exp(-dist_to_food / hunger_decay)   (hunger_decay = 1.25
         for brain_b, 1.0 for brain_a -- Unified3DController.__init__).
         Inverted to reconstruct dist_to_food: this is the quantity
         DIST_MULT (the dominant continuous shaping term in
         evaluate3d.py) is proportional to the change of. Reported as
         mean absolute error in reconstructed distance, real vs
         imagined, by horizon.
       - near_wall = max(x[:8]) > 0.82 (same threshold as
         collect_h_x_actraw_vectors.py). Reported as accuracy/precision/
         recall of the imagined rollout's near_wall flag against the
         REAL near_wall column already in the CSV, by horizon.

  C) Success-timing accuracy: SUCCESS_THRESHOLD = 0.08 (controllers.py).
     Ground truth = the real `own_success` column (from true physics,
     not reconstructed). For the imagined side, success is declared the
     first step where reconstructed dist_to_food < 0.08. Reports: hit
     rate (imagined rollout ever predicts success within the horizon,
     among trajectories that really do succeed within it), mean
     absolute step error when both agree success happens, and false-
     positive rate (imagined predicts success, real trajectory does
     not, within the same horizon).

Usage:
    python3 validate_rollout_fidelity.py \
        --csv ../h_x_actraw_300.csv \
        --weights ../models/social3d_v15_full_final.json \
        --horizons 1,5,10,20,50 \
        --out_dir results_rollout/

Usage (self-test, synthetic data, no real CSV/weights needed):
    python3 validate_rollout_fidelity.py --selftest
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

import compare_world_model_candidates as wm  # reuse load_data/build_pairs/F/etc.

SUCCESS_THRESHOLD = 0.08        # controllers.py, SUCCESS_THRESHOLD
NEAR_WALL_THRESHOLD = 0.82      # collect_h_x_actraw_vectors.py, near_wall computation
HUNGER_DECAY = {"a": 1.0, "b": 1.25}  # Unified3DController.__init__, self.hunger_decay


def dist_from_hunger(x9, decay, eps=1e-6):
    """Invert x[9] = exp(-dist/decay) -> dist = -decay * ln(x9).
    Real x[9] was computed by this exact formula from the true distance
    (controllers.py), so inverting it recovers the true distance exactly
    (up to floating point) -- not an approximation for the real side.
    For imagined x[9] this gives the model's implied distance estimate.
    Clipped because a regressor's output isn't guaranteed to stay in
    (0, 1]."""
    x9c = np.clip(x9, eps, 1.0)
    return -decay * np.log(x9c)


def load_trajectories(df, h_cols, x_cols):
    """Full ordered per-(episode, agent) trajectories, needed for
    chaining -- unlike wm.build_pairs, which flattens into single-step
    pairs and loses the ordering."""
    trajs = []
    grouped = df.sort_values(["episode", "agent", "decision"]).groupby(["episode", "agent"])
    for (ep, agent), group in grouped:
        trajs.append({
            "episode": ep,
            "agent": agent,
            "h": group[h_cols].values,
            "x": group[x_cols].values,
            "action": group["action"].values,
            "own_success": group["own_success"].values.astype(bool),
            "near_wall": group["near_wall"].values.astype(bool),
        })
    return trajs


def train_predictor(df, h_cols, x_cols, has_actraw, test_frac, seed, model_type="ridge",
                     boost=9.0, near_goal_range=5 * SUCCESS_THRESHOLD, train_all=False):
    """model_type: 'ridge' | 'ridge_weighted' | 'mlp'.
    'ridge_weighted' up-weights rows whose target x(t+1) implies a
    distance-to-food close to SUCCESS_THRESHOLD (see module docstring
    for the ramp definition) -- already tested, did not move
    C_success_hit_rate. 'mlp' is the capacity-hypothesis check: same
    architecture already validated in compare_world_model_candidates.py
    (64,64 hidden, alpha=1e-3), just applied here to the rollout.

    train_all=True: use every episode in `df` for training (no
    held-out split). Use this for the cross-policy check, where
    evaluation happens on an entirely separate CSV/model -- there's no
    leakage risk from using 100% of the training-policy data, since
    none of it overlaps with the eval set by construction."""
    pairs = wm.build_pairs(df, h_cols, x_cols, has_actraw)
    if train_all:
        train_mask = np.ones(len(pairs["episode"]), dtype=bool)
        test_mask = train_mask  # unused for reporting in cross-policy mode
    else:
        train_mask, test_mask = wm.episode_split(pairs["episode"], test_frac, seed)
    X_train = wm.feats_xpred(pairs, train_mask, add_actraw=False)
    y_train = pairs["x_next"][train_mask]

    if model_type == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                              alpha=1e-3, max_iter=2000, random_state=seed)
        model.fit(X_train, y_train)
    else:
        model = Ridge(alpha=1.0)
        if model_type == "ridge_weighted":
            agent_train = pairs["agent_t"][train_mask]
            decay = np.array([HUNGER_DECAY[a] for a in agent_train])
            dist_next = dist_from_hunger(y_train[:, 9], decay)
            closeness = np.clip((near_goal_range - dist_next) / near_goal_range, 0.0, 1.0)
            sample_weight = 1.0 + boost * closeness
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

    test_episodes = set(np.unique(pairs["episode"][test_mask]).tolist())
    return model, test_episodes


def run_rollout(traj, model, weights_by_agent, max_horizon):
    """Chain predictions for up to max_horizon steps (or trajectory
    length - 1, whichever is smaller). Returns per-step imagined h, x."""
    n = min(max_horizon, len(traj["h"]) - 1)
    if n <= 0:
        return None
    w = wm._as_array_weights(weights_by_agent["brain_a"] if traj["agent"] == "a"
                              else weights_by_agent["brain_b"])
    h_t = traj["h"][0].copy()
    x_t = traj["x"][0].copy()
    imagined_h = np.zeros((n, wm.H_DIM))
    imagined_x = np.zeros((n, wm.X_DIM))
    for t in range(n):
        action_t = traj["action"][t]  # REAL recorded action, not re-derived
        feat = np.concatenate([h_t, x_t, wm.one_hot(action_t)])[None, :]
        x_pred = model.predict(feat)[0]
        h_pred = wm.gru_F(h_t, x_pred, w)
        imagined_h[t] = h_pred
        imagined_x[t] = x_pred
        h_t, x_t = h_pred, x_pred
    return imagined_h, imagined_x


def evaluate_trajectories(trajs, model, weights_by_agent, horizons, variant="unweighted"):
    max_h = max(horizons)
    # A: raw error accumulators, per horizon step index (1-indexed steps)
    x_sq_err = {h: [] for h in horizons}
    h_sq_err = {h: [] for h in horizons}
    # B: dist-to-food error, near_wall confusion counts, per horizon
    dist_err = {h: [] for h in horizons}
    near_wall_tp = {h: 0 for h in horizons}
    near_wall_fp = {h: 0 for h in horizons}
    near_wall_fn = {h: 0 for h in horizons}
    near_wall_tn = {h: 0 for h in horizons}
    # C: success timing
    real_success_within = {h: 0 for h in horizons}     # denom for hit rate
    imagined_hit = {h: 0 for h in horizons}
    step_errors = {h: [] for h in horizons}
    false_positives = {h: 0 for h in horizons}
    n_traj_used = 0

    for traj in trajs:
        result = run_rollout(traj, model, weights_by_agent, max_h)
        if result is None:
            continue
        imagined_h, imagined_x = result
        n_avail = imagined_h.shape[0]
        n_traj_used += 1
        decay = HUNGER_DECAY[traj["agent"]]

        real_success_steps = np.where(traj["own_success"][1:n_avail + 1])[0]
        real_success_step = real_success_steps[0] if len(real_success_steps) else None

        real_dist = dist_from_hunger(traj["x"][1:n_avail + 1, 9], decay)
        imagined_dist = dist_from_hunger(imagined_x[:, 9], decay)
        imagined_success_steps = np.where(imagined_dist < SUCCESS_THRESHOLD)[0]
        imagined_success_step = imagined_success_steps[0] if len(imagined_success_steps) else None

        for hh in horizons:
            if hh > n_avail:
                continue
            idx = hh - 1  # 0-indexed step corresponding to horizon hh

            # --- A ---
            x_sq_err[hh].append(np.mean((imagined_x[idx] - traj["x"][hh]) ** 2))
            h_sq_err[hh].append(np.mean((imagined_h[idx] - traj["h"][hh]) ** 2))

            # --- B: distance-to-food ---
            dist_err[hh].append(abs(imagined_dist[idx] - real_dist[idx]))

            # --- B: near_wall confusion, at this single horizon step ---
            pred_near_wall = bool(np.max(imagined_x[idx, :8]) > NEAR_WALL_THRESHOLD)
            real_near_wall = bool(traj["near_wall"][hh])
            if pred_near_wall and real_near_wall:
                near_wall_tp[hh] += 1
            elif pred_near_wall and not real_near_wall:
                near_wall_fp[hh] += 1
            elif not pred_near_wall and real_near_wall:
                near_wall_fn[hh] += 1
            else:
                near_wall_tn[hh] += 1

            # --- C: success timing, evaluated within horizon hh ---
            real_succeeds_within = real_success_step is not None and real_success_step < hh
            imagined_succeeds_within = imagined_success_step is not None and imagined_success_step < hh
            if real_succeeds_within:
                real_success_within[hh] += 1
                if imagined_succeeds_within:
                    imagined_hit[hh] += 1
                    step_errors[hh].append(abs(imagined_success_step - real_success_step))
            elif imagined_succeeds_within:
                false_positives[hh] += 1

    rows = []
    for hh in horizons:
        n_x = len(x_sq_err[hh])
        if n_x == 0:
            continue
        row = {
            "variant": variant,
            "horizon": hh,
            "n_trajectories": n_x,
            "A_x_rmse": float(np.sqrt(np.mean(x_sq_err[hh]))),
            "A_h_rmse": float(np.sqrt(np.mean(h_sq_err[hh]))),
            "B_dist_to_food_mae": float(np.mean(dist_err[hh])),
        }
        tp, fp, fn, tn = near_wall_tp[hh], near_wall_fp[hh], near_wall_fn[hh], near_wall_tn[hh]
        total = tp + fp + fn + tn
        row["B_near_wall_accuracy"] = (tp + tn) / total if total else float("nan")
        row["B_near_wall_precision"] = tp / (tp + fp) if (tp + fp) else float("nan")
        row["B_near_wall_recall"] = tp / (tp + fn) if (tp + fn) else float("nan")

        denom = real_success_within[hh]
        row["C_success_hit_rate"] = imagined_hit[hh] / denom if denom else float("nan")
        row["C_success_n_real_within_horizon"] = denom
        row["C_success_mean_abs_step_error"] = (
            float(np.mean(step_errors[hh])) if step_errors[hh] else float("nan")
        )
        row["C_success_false_positives"] = false_positives[hh]
        rows.append(row)

    print(f"\nRollout evaluation used {n_traj_used} test trajectories (trajectories shorter "
          f"than a given horizon are simply excluded from that horizon's row, not padded).")
    return rows


def print_report(rows):
    print("\n=== A) Raw error by horizon (RMSE, real vs imagined) ===")
    for r in rows:
        print(f"  [{r['variant']:10s}] h={r['horizon']:3d}  x_rmse={r['A_x_rmse']:.4f}  "
              f"h_rmse={r['A_h_rmse']:.4f}  (n={r['n_trajectories']})")

    print("\n=== B) Distance-to-food (drives DIST_MULT shaping) + near_wall accuracy ===")
    for r in rows:
        print(f"  [{r['variant']:10s}] h={r['horizon']:3d}  dist_to_food MAE={r['B_dist_to_food_mae']:.4f}  "
              f"near_wall acc={r['B_near_wall_accuracy']:.3f} "
              f"prec={r['B_near_wall_precision']:.3f} rec={r['B_near_wall_recall']:.3f}")

    print("\n=== C) Success-timing accuracy (drives the dominant terminal bonus) ===")
    for r in rows:
        print(f"  [{r['variant']:10s}] h={r['horizon']:3d}  hit_rate={r['C_success_hit_rate']:.3f} "
              f"(n_real_success={r['C_success_n_real_within_horizon']})  "
              f"mean_abs_step_error={r['C_success_mean_abs_step_error']:.2f}  "
              f"false_positives={r['C_success_false_positives']}")


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

def run_selftest():
    print("=== SELF-TEST: synthetic weights + synthetic trajectories ===\n")
    weights = wm.make_synthetic_weights()
    df = wm.make_synthetic_csv(weights, n_episodes=10, n_decisions=60)
    h_cols = [f"h_{k}" for k in range(wm.H_DIM)]
    x_cols = [f"x_{k}" for k in range(wm.X_DIM)]
    df["near_wall"] = df[[f"x_{k}" for k in range(8)]].max(axis=1) > NEAR_WALL_THRESHOLD
    df["own_success"] = False

    all_rows = []
    for variant, model_type in [("unweighted", "ridge"), ("weighted", "ridge_weighted"),
                                 ("mlp", "mlp")]:
        model, test_episodes = train_predictor(df, h_cols, x_cols, has_actraw=True,
                                                 test_frac=0.4, seed=42, model_type=model_type)
        df_test = df[df["episode"].isin(test_episodes)]
        trajs = load_trajectories(df_test, h_cols, x_cols)
        all_rows += evaluate_trajectories(trajs, model, weights, horizons=[1, 5, 10, 20],
                                           variant=variant)
    print_report(all_rows)
    print("\nSelf-test finished without errors.")


def load_and_offset(csv_path, src_idx):
    """Loads one CSV and makes its `episode` ids collision-proof against
    every other pooled CSV (two different collection runs can both have
    an 'episode 0' -- without this, pooling would silently merge
    unrelated trajectories from different policies into one group)."""
    df, h_cols, x_cols, has_actraw = wm.load_data(csv_path)
    df = df.copy()
    df["episode"] = df["episode"].astype(str) + f"__src{src_idx}"
    return df, h_cols, x_cols, has_actraw


def pool_dataframes(csv_paths):
    dfs, h_cols, x_cols, has_actraw = [], None, None, True
    for i, p in enumerate(csv_paths):
        df, h_cols, x_cols, this_has_actraw = load_and_offset(p, i)
        has_actraw = has_actraw and this_has_actraw
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), h_cols, x_cols, has_actraw


def run_leave_one_out(csv_paths, weight_paths, horizons, seed, out_dir):
    """For each policy in the pool: train Ridge on the OTHER policies'
    data combined, evaluate the chained rollout on the held-out policy
    (using its own weights for the exact F()). Ridge only, on purpose --
    already established MLP is worse and 'weighted' didn't help, no
    reason to pay for either here. Directly comparable to the earlier
    single-policy cross-check (rollout_report_crosspolicy_...csv): same
    metrics, same held-out policy (v14), but now trained on a POOL
    instead of v15 alone."""
    if len(csv_paths) != len(weight_paths):
        raise SystemExit("--pool_csvs and --pool_weights must have the same length, same order.")

    all_rows = []
    for i in range(len(csv_paths)):
        held_csv, held_weights_path = csv_paths[i], weight_paths[i]
        train_paths = [c for j, c in enumerate(csv_paths) if j != i]
        held_name = os.path.splitext(os.path.basename(held_csv))[0]

        print(f"\n=== Leave-one-out fold: held out {held_name} "
              f"(trained on the other {len(train_paths)} policies pooled) ===")
        pooled_df, h_cols, x_cols, has_actraw = pool_dataframes(train_paths)
        print(f"Pooled training set: {pooled_df['episode'].nunique()} episodes "
              f"across {len(train_paths)} policies.")

        model, _ = train_predictor(pooled_df, h_cols, x_cols, has_actraw,
                                    test_frac=0.0, seed=seed, model_type="ridge", train_all=True)

        eval_df, _, _, _ = wm.load_data(held_csv)
        with open(held_weights_path) as f:
            eval_weights = json.load(f)
        trajs = load_trajectories(eval_df, h_cols, x_cols)
        rows = evaluate_trajectories(trajs, model, eval_weights, horizons,
                                      variant=f"pooled_holdout_{held_name}")
        all_rows += rows

    print_report(all_rows)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "rollout_report_leave_one_out_pooled.csv")
        pd.DataFrame(all_rows).to_csv(report_path, index=False)
        print(f"\nFull report written to {report_path}")
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="same-dataset mode: train/test split within this one CSV")
    ap.add_argument("--weights", help="same-dataset mode: weights matching --csv")
    ap.add_argument("--train_csv", help="cross-policy mode: CSV to train the predictor on "
                                         "(uses ALL its episodes, no split)")
    ap.add_argument("--train_weights", help="cross-policy mode: weights matching --train_csv "
                                             "(only used if the rollout mixes training-policy "
                                             "trajectories, which it currently doesn't -- kept "
                                             "for symmetry/future use)")
    ap.add_argument("--eval_csv", help="cross-policy mode: CSV to run rollouts on (a DIFFERENT "
                                        "policy's trajectories than --train_csv)")
    ap.add_argument("--eval_weights", help="cross-policy mode: weights matching --eval_csv -- "
                                            "REQUIRED for the exact F() chaining to be correct, "
                                            "since F depends on that policy's own w_gru/b_gru")
    ap.add_argument("--pool_csvs", help="leave-one-out pooling mode: comma-separated list of "
                                         "CSVs from different policies (e.g. v15,v10,indep2,"
                                         "sesgo_check1). For each one, trains on the OTHERS "
                                         "pooled and evaluates the rollout on it held out.")
    ap.add_argument("--pool_weights", help="leave-one-out pooling mode: comma-separated list of "
                                            "weights JSONs, SAME ORDER as --pool_csvs")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", default="1,5,10,20,50",
                     help="comma-separated list of rollout horizons (in decision steps)")
    ap.add_argument("--out_dir", default="results_rollout")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.pool_csvs or args.pool_weights:
        if not (args.pool_csvs and args.pool_weights):
            raise SystemExit("Pooling mode needs both --pool_csvs and --pool_weights.")
        csv_paths = [c.strip() for c in args.pool_csvs.split(",")]
        weight_paths = [w.strip() for w in args.pool_weights.split(",")]
        horizons = [int(h) for h in args.horizons.split(",")]
        run_leave_one_out(csv_paths, weight_paths, horizons, args.seed, args.out_dir)
        return

    cross_policy = bool(args.train_csv or args.eval_csv)
    if cross_policy:
        if not (args.train_csv and args.eval_csv and args.eval_weights):
            raise SystemExit("Cross-policy mode needs --train_csv, --eval_csv and --eval_weights.")
        train_df, h_cols, x_cols, train_has_actraw = wm.load_data(args.train_csv)
        eval_df, _, _, _ = wm.load_data(args.eval_csv)
        with open(args.eval_weights) as f:
            eval_weights = json.load(f)
        horizons = [int(h) for h in args.horizons.split(",")]

        print(f"[cross-policy] Training on ALL episodes of {args.train_csv} "
              f"({train_df['episode'].nunique()} episodes), evaluating rollouts on "
              f"{args.eval_csv} ({eval_df['episode'].nunique()} episodes, a DIFFERENT policy) "
              f"using that policy's own weights for F().")

        all_rows = []
        for variant, model_type in [("unweighted", "ridge"), ("weighted", "ridge_weighted"),
                                     ("mlp", "mlp")]:
            print(f"\n--- Training x(t+1) predictor on train_csv: {variant} ({model_type}) ---")
            model, _ = train_predictor(train_df, h_cols, x_cols, train_has_actraw,
                                        args.test_frac, args.seed, model_type=model_type,
                                        train_all=True)
            trajs = load_trajectories(eval_df, h_cols, x_cols)
            all_rows += evaluate_trajectories(trajs, model, eval_weights, horizons, variant=variant)

        print_report(all_rows)

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            train_tag = os.path.splitext(os.path.basename(args.train_csv))[0]
            eval_tag = os.path.splitext(os.path.basename(args.eval_csv))[0]
            report_name = f"rollout_report_crosspolicy_train-{train_tag}_eval-{eval_tag}.csv"
            report_path = os.path.join(args.out_dir, report_name)
            pd.DataFrame(all_rows).to_csv(report_path, index=False)
            print(f"\nFull report written to {report_path}")
        return

    if not args.csv or not args.weights:
        raise SystemExit("Need --csv and --weights (same-dataset mode), --train_csv/--eval_csv/"
                          "--eval_weights (cross-policy mode), or --selftest.")

    horizons = [int(h) for h in args.horizons.split(",")]

    df, h_cols, x_cols, has_actraw = wm.load_data(args.csv)
    with open(args.weights) as f:
        weights = json.load(f)

    all_rows = []
    for variant, model_type in [("unweighted", "ridge"), ("weighted", "ridge_weighted"),
                                 ("mlp", "mlp")]:
        print(f"\n--- Training x(t+1) predictor: {variant} ({model_type}) ---")
        model, test_episodes = train_predictor(df, h_cols, x_cols, has_actraw,
                                                 args.test_frac, args.seed, model_type=model_type)
        print(f"Trained. {len(test_episodes)} held-out test episodes used for rollouts.")
        df_test = df[df["episode"].isin(test_episodes)]
        trajs = load_trajectories(df_test, h_cols, x_cols)
        all_rows += evaluate_trajectories(trajs, model, weights, horizons, variant=variant)

    print_report(all_rows)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        n_ep = len(test_episodes)
        report_name = f"rollout_report_n{n_ep}ep_horizons{'-'.join(map(str, horizons))}_ridge_vs_weighted_vs_mlp.csv"
        report_path = os.path.join(args.out_dir, report_name)
        pd.DataFrame(all_rows).to_csv(report_path, index=False)
        print(f"\nFull report written to {report_path}")


if __name__ == "__main__":
    main()
