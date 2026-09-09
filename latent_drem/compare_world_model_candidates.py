"""
compare_world_model_candidates.py

Single-run harness to compare candidate predictors for the "world model"
project (see PROMPT_proximo_proyecto_modelo_del_mundo.md), built to avoid
re-running data collection or re-designing the comparison harness twice.

KEY REFRAMING (checked directly against gru_dynamics.py and controllers.py
before writing this, not assumed from memory):

    h_next = F(h, x, weights)   -- the GRU recurrence -- depends ONLY on
    (h, x). It does NOT depend on `action` at all: GRUNetwork.forward()
    fully updates self.h (gates z, r, n) BEFORE act_id_raw is computed,
    and the basal-ganglia instinct that can override act_id_raw acts only
    on the output logits, never on h. F(h,x) is already validated exactly
    (see gru_dynamics.py's own self-test, diff ~1e-10 vs the real forward).

    Consequence: h(t+1) is NOT something that needs to be learned. The
    only genuinely unknown function is the physics/environment response:

        x(t+1) ~= G(h(t), x(t), action(t))      <- must be learned
        h(t+1)  = F(h(t), x(t+1))                <- already known exactly

    So this harness's main target is G (predicting x(t+1)), with h(t+1)
    obtained for free via the exact F(). It also replicates the OLD
    approach (predicting h(t+1) directly from h(t)+action, ignoring x)
    as an explicit baseline, so the improvement (or lack of it) is
    measured quantitatively in the same run, not asserted.

Causal note on which action column to use:
    `action`      = final act_id AFTER the stuck/recruit instinct can
                    override it. This is what actually drives
                    action_table -> wheel velocities -> physics. This is
                    the physically-causal signal for predicting x(t+1).
    `act_id_raw`  = the network's own raw decision, before any override.
                    Available as an optional extra feature
                    (--add_actraw_feature) for ablation, never as a
                    replacement for `action`.

Expected CSV columns (as written by collect_h_x_actraw_vectors.py):
    episode, agent, decision, stalled, heard_signal, near_wall,
    own_success, action, act_id_raw, h_0..h_15, x_0..x_10

Expected weights JSON: the same model file used by the simulator, i.e.
    {"brain_a": {"w_gru": [...], "b_gru": [...], ...},
     "brain_b": {"w_gru": [...], "b_gru": [...], ...}}
(only w_gru/b_gru are actually needed here, since F() does not use
w_out/w_res/b_out -- those only affect the readout, not the recurrence.)

Repo layout note: this script expects to sit next to gru_dynamics.py (i.e.
inside latent_diag/, alongside jacobian.py, predict_h_next.py, etc.), and
weights live one level up in ../models/. The import below is made robust
to a couple of likely layouts so it works either way.

Usage (quick timing probe on a small subset, e.g. 30 of your 300 episodes
-- run this BEFORE committing to the full run, to get a real per-machine
time estimate instead of a guessed one):
    python3 compare_world_model_candidates.py \
        --csv h_x_actraw_300.csv \
        --weights ../models/social3d_v15_full_final.json \
        --subset_episodes 30 \
        --out_dir results_subset30/

Usage (full real data, once the subset timing looks acceptable):
    python3 compare_world_model_candidates.py \
        --csv h_x_actraw_300.csv \
        --weights ../models/social3d_v15_full_final.json \
        --out_dir results/

Usage (self-test, no real data needed -- generates synthetic weights and
a synthetic trajectory CSV in memory, runs the full pipeline, and checks
nothing crashes and the oracle sanity check is near-perfect):
    python3 compare_world_model_candidates.py --selftest
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Make the gru_dynamics.py import work whether this file lives inside
# latent_diag/ (recommended, next to gru_dynamics.py) or is invoked from
# the repo root with latent_diag/ as a subfolder. Checked against the
# actual tree the author shared, not assumed.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.join(_THIS_DIR, "latent_diag"),
                   os.getcwd(), os.path.join(os.getcwd(), "latent_diag")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

from gru_dynamics import F as gru_F

N_ACTIONS = 5
H_DIM = 16
X_DIM = 11


# ------------------------------------------------------------------
# Data loading / pair construction
# ------------------------------------------------------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    h_cols = [f"h_{k}" for k in range(H_DIM)]
    x_cols = [f"x_{k}" for k in range(X_DIM)]
    missing = [c for c in h_cols + x_cols + ["episode", "agent", "decision", "action"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    has_actraw = "act_id_raw" in df.columns
    return df, h_cols, x_cols, has_actraw


def one_hot(a, n=N_ACTIONS):
    v = np.zeros(n)
    a = int(a)
    if 0 <= a < n:
        v[a] = 1.0
    return v


def build_pairs(df, h_cols, x_cols, has_actraw):
    """
    Build one row per consecutive (t, t+1) pair within the same
    (episode, agent), never crossing that boundary.

    Returns a dict of parallel numpy arrays:
        h_t, x_t, action_t, actraw_t (or None), agent_t (str array),
        h_next (true), x_next (true), episode_id (for grouped split)
    """
    h_t_list, x_t_list, action_t_list, actraw_t_list = [], [], [], []
    agent_t_list, h_next_list, x_next_list, ep_list = [], [], [], []

    grouped = df.sort_values(["episode", "agent", "decision"]).groupby(["episode", "agent"])
    for (ep, agent), group in grouped:
        h_vals = group[h_cols].values
        x_vals = group[x_cols].values
        actions = group["action"].values
        actraw = group["act_id_raw"].values if has_actraw else None

        n = len(group)
        if n < 2:
            continue
        for t in range(n - 1):
            h_t_list.append(h_vals[t])
            x_t_list.append(x_vals[t])
            action_t_list.append(actions[t])
            actraw_t_list.append(actraw[t] if has_actraw else -1)
            agent_t_list.append(agent)
            h_next_list.append(h_vals[t + 1])
            x_next_list.append(x_vals[t + 1])
            ep_list.append(ep)

    return {
        "h_t": np.array(h_t_list),
        "x_t": np.array(x_t_list),
        "action_t": np.array(action_t_list),
        "actraw_t": np.array(actraw_t_list),
        "agent_t": np.array(agent_t_list),
        "h_next": np.array(h_next_list),
        "x_next": np.array(x_next_list),
        "episode": np.array(ep_list),
    }


def subset_by_episode(df, n_episodes, seed):
    """Restrict the dataframe to a random subset of unique episodes, for a
    quick real-hardware timing probe before committing to the full run.
    Keeps both agents for each selected episode."""
    uniq = df["episode"].unique()
    if n_episodes >= len(uniq):
        return df
    rng = np.random.RandomState(seed)
    chosen = rng.choice(uniq, size=n_episodes, replace=False)
    return df[df["episode"].isin(chosen)].copy()


def episode_split(episodes, test_frac, seed):
    """Split by UNIQUE episode id, never by row, to avoid leaking
    information between train and test from the same trajectory
    (same discipline as predict_h_next.py)."""
    uniq = np.unique(episodes)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    test_eps = set(uniq[:n_test].tolist())
    is_test = np.array([e in test_eps for e in episodes])
    return ~is_test, is_test


# ------------------------------------------------------------------
# Feature builders
# ------------------------------------------------------------------

def feats_old_baseline(pairs, idx):
    """Replicates the OLD approach: [h(t), one-hot action(t)] -> h(t+1).
    Kept only as an explicit comparison point, not as the recommended
    path (see module docstring: action isn't even an input to F)."""
    h_t = pairs["h_t"][idx]
    acts = np.array([one_hot(a) for a in pairs["action_t"][idx]])
    return np.concatenate([h_t, acts], axis=1)


def feats_xpred(pairs, idx, add_actraw=False):
    """New approach: [h(t), x(t), one-hot action(t)] -> x(t+1).
    This is the only genuinely unknown function (physics)."""
    h_t = pairs["h_t"][idx]
    x_t = pairs["x_t"][idx]
    acts = np.array([one_hot(a) for a in pairs["action_t"][idx]])
    blocks = [h_t, x_t, acts]
    if add_actraw:
        actraws = np.array([one_hot(a) for a in pairs["actraw_t"][idx]])
        blocks.append(actraws)
    return np.concatenate(blocks, axis=1)


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

def make_models(seed, skip_trees=False):
    """One dict of models per feature set. Tree ensembles wrapped in
    MultiOutputRegressor since sklearn's RF/GBM don't natively share
    structure across output dimensions the way a single MLP does.

    skip_trees=True drops random_forest/gradient_boosting entirely --
    use this once a small-subset probe already showed they buy nothing
    on the metric that matters (chained h(t+1) R^2), to avoid paying
    their cost (minutes-to-hours) on the full dataset for no benefit."""
    models = {
        "ridge": Ridge(alpha=1.0),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=2000,
            random_state=seed,
        ),
    }
    if not skip_trees:
        models["random_forest"] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
        )
        models["gradient_boosting"] = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
        )
    return models


def r2_per_dim_and_overall(y_true, y_pred):
    per_dim = [r2_score(y_true[:, k], y_pred[:, k]) for k in range(y_true.shape[1])]
    overall = r2_score(y_true, y_pred)  # uniform-average multioutput
    return overall, per_dim


# ------------------------------------------------------------------
# End-to-end validation through the exact F()
# ------------------------------------------------------------------

def _as_array_weights(raw):
    """gru_dynamics.F only needs w_gru/b_gru as numpy arrays. Works whether
    `raw` came from json.load (lists) or was already numpy (synthetic)."""
    return {"w_gru": np.array(raw["w_gru"]), "b_gru": np.array(raw["b_gru"])}


def chain_through_F(h_t, x_next_pred, agent_t, weights_by_agent):
    """h(t+1)_hat = F(h(t), x(t+1)_hat, weights[agent]) using the exact,
    already-validated recurrence. No learning involved in this step."""
    w_a = _as_array_weights(weights_by_agent["brain_a"])
    w_b = _as_array_weights(weights_by_agent["brain_b"])
    out = np.zeros_like(h_t)
    for i in range(h_t.shape[0]):
        w = w_a if agent_t[i] == "a" else w_b
        out[i] = gru_F(h_t[i], x_next_pred[i], w)
    return out


# ------------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------------

def run_comparison(pairs, weights_by_agent, test_frac, seed, add_actraw, out_dir, skip_trees=False):
    t_start = time.perf_counter()
    train_mask, test_mask = episode_split(pairs["episode"], test_frac, seed)
    n_train, n_test = train_mask.sum(), test_mask.sum()
    print(f"Rows: {len(pairs['episode'])} total | train={n_train} | test={n_test} "
          f"(split by unique episode, frac={test_frac})")
    if skip_trees:
        print("[skip_trees] Skipping random_forest/gradient_boosting -- subset probe already "
              "showed no gain over Ridge on chained h(t+1) R^2, at a much higher cost.")

    models = make_models(seed, skip_trees=skip_trees)
    rows_report = []

    # --- Oracle sanity check: feed the TRUE x(t+1) through F and compare
    #     to the true h(t+1). Should be ~1.0 R^2 (near-exact) -- this is
    #     purely a check that F() is wired correctly here, not a model
    #     result. If this isn't ~1.0, stop and debug before trusting
    #     anything else below.
    h_oracle = chain_through_F(pairs["h_t"][test_mask], pairs["x_next"][test_mask],
                                pairs["agent_t"][test_mask], weights_by_agent)
    oracle_r2, oracle_per_dim = r2_per_dim_and_overall(pairs["h_next"][test_mask], h_oracle)
    print(f"\n[Sanity check] True x(t+1) -> F() -> h(t+1) vs true h(t+1): "
          f"R^2 overall = {oracle_r2:.6f} (expected ~1.0; if not, F()/weights are miswired)")
    rows_report.append({"target": "h_next (oracle via true x, sanity check)",
                         "model": "F() exact, no model", "r2_overall": oracle_r2})

    # --- OLD baseline: h(t)+action -> h(t+1) directly, ignoring x.
    print("\n=== OLD baseline: [h(t), action(t)] -> h(t+1) directly (ignores x) ===")
    X_old_train = feats_old_baseline(pairs, train_mask)
    X_old_test = feats_old_baseline(pairs, test_mask)
    y_old_train = pairs["h_next"][train_mask]
    y_old_test = pairs["h_next"][test_mask]
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_old_train, y_old_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(X_old_test)
        pred_s = time.perf_counter() - t0
        overall, per_dim = r2_per_dim_and_overall(y_old_test, pred)
        print(f"  {name:18s}  R^2 overall = {overall:.4f}  "
              f"(min dim {min(per_dim):.3f}, max dim {max(per_dim):.3f})  "
              f"|  fit={fit_s:.1f}s predict={pred_s:.1f}s")
        rows_report.append({"target": "h_next (OLD: h+action, no x)", "model": name,
                             "r2_overall": overall, "fit_seconds": fit_s, "predict_seconds": pred_s})

    # --- NEW approach: h(t)+x(t)+action -> x(t+1), then chain through F.
    print("\n=== NEW approach: [h(t), x(t), action(t)] -> x(t+1), then h(t+1) via exact F() ===")
    X_new_train = feats_xpred(pairs, train_mask, add_actraw=add_actraw)
    X_new_test = feats_xpred(pairs, test_mask, add_actraw=add_actraw)
    y_x_train = pairs["x_next"][train_mask]
    y_x_test = pairs["x_next"][test_mask]
    y_h_test = pairs["h_next"][test_mask]
    h_t_test = pairs["h_t"][test_mask]
    agent_test = pairs["agent_t"][test_mask]

    fresh_models = make_models(seed, skip_trees=skip_trees)  # avoid reusing already-fit instances from above
    for name, model in fresh_models.items():
        t0 = time.perf_counter()
        model.fit(X_new_train, y_x_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        x_pred = model.predict(X_new_test)
        pred_s = time.perf_counter() - t0

        x_overall, x_per_dim = r2_per_dim_and_overall(y_x_test, x_pred)
        h_chained = chain_through_F(h_t_test, x_pred, agent_test, weights_by_agent)
        h_overall, h_per_dim = r2_per_dim_and_overall(y_h_test, h_chained)

        print(f"  {name:18s}  x(t+1) R^2 = {x_overall:.4f}  "
              f"(min dim {min(x_per_dim):.3f}, max dim {max(x_per_dim):.3f})  |  "
              f"chained h(t+1) R^2 = {h_overall:.4f} "
              f"(min dim {min(h_per_dim):.3f}, max dim {max(h_per_dim):.3f})  "
              f"|  fit={fit_s:.1f}s predict={pred_s:.1f}s")
        rows_report.append({"target": "x_next", "model": name, "r2_overall": x_overall,
                             "fit_seconds": fit_s, "predict_seconds": pred_s})
        rows_report.append({"target": "h_next (chained via predicted x, NEW)", "model": name,
                             "r2_overall": h_overall})

    total_s = time.perf_counter() - t_start
    n_episodes_used = len(np.unique(pairs["episode"]))
    print(f"\nTotal wall-clock time for this run: {total_s:.1f}s "
          f"({len(pairs['episode'])} rows across {n_episodes_used} episodes)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        models_tag = "ridge_mlp" if skip_trees else "all4models"
        report_name = f"comparison_report_n{n_episodes_used}ep_{models_tag}.csv"
        report_path = os.path.join(out_dir, report_name)
        pd.DataFrame(rows_report).to_csv(report_path, index=False)
        print(f"Full report written to {report_path}")

    return rows_report


# ------------------------------------------------------------------
# Self-test (synthetic, no real data / no PyBullet needed)
# ------------------------------------------------------------------

def make_synthetic_weights(seed=0):
    rng = np.random.default_rng(seed)
    def one_agent():
        return {
            "w_gru": (rng.normal(size=(27, 48)) * 0.5).tolist(),
            "b_gru": (rng.normal(size=(48,)) * 0.1).tolist(),
            "w_out": (rng.normal(size=(16, 5)) * 0.5).tolist(),
            "b_out": (rng.normal(size=(5,)) * 0.1).tolist(),
            "w_res": (rng.normal(size=(11, 5)) * 0.5).tolist(),
        }
    return {"brain_a": one_agent(), "brain_b": one_agent()}


def make_synthetic_csv(weights, n_episodes=6, n_decisions=25, seed=1):
    """Generates trajectories that actually respect h(t+1)=F(h(t),x(t+1))
    exactly, plus a made-up but deterministic x(t+1)=G_true(x(t),action(t))
    so the 'physics' being learned is a real, learnable (if nonlinear)
    function -- not pure noise. This is only for wiring/self-test, not a
    claim about real physics."""
    rng = np.random.default_rng(seed)
    rows = []
    for agent_key, agent_name in [("brain_a", "a"), ("brain_b", "b")]:
        w = {k: np.array(v) for k, v in weights[agent_key].items()}
        for ep in range(n_episodes):
            h = np.zeros(16)
            x = rng.normal(size=11) * 0.3
            for dec in range(n_decisions):
                action = rng.integers(0, N_ACTIONS)
                actraw = rng.integers(0, N_ACTIONS)
                h = gru_F(h, x, w)
                rows.append({
                    "episode": ep, "agent": agent_name, "decision": dec,
                    "stalled": False, "heard_signal": False, "near_wall": False,
                    "own_success": False, "action": int(action), "act_id_raw": int(actraw),
                    **{f"h_{k}": float(h[k]) for k in range(16)},
                    **{f"x_{k}": float(x[k]) for k in range(11)},
                })
                # toy deterministic "physics": next x depends smoothly and
                # nonlinearly on current x and the chosen action
                bump = np.zeros(11)
                bump[action % 11] = 0.4
                x = np.tanh(x * 0.9 + bump + rng.normal(size=11) * 0.01)
    return pd.DataFrame(rows)


def run_selftest():
    print("=== SELF-TEST: synthetic weights + synthetic trajectories ===")
    print("(No real CSV / PyBullet needed -- this only checks the harness")
    print(" is wired correctly: shapes, grouping, F() integration, etc.)\n")
    weights = make_synthetic_weights()
    df = make_synthetic_csv(weights)
    h_cols = [f"h_{k}" for k in range(H_DIM)]
    x_cols = [f"x_{k}" for k in range(X_DIM)]
    pairs = build_pairs(df, h_cols, x_cols, has_actraw=True)
    run_comparison(pairs, weights, test_frac=0.34, seed=42, add_actraw=False, out_dir=None)
    print("\nSelf-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to h_x_actraw CSV (from collect_h_x_actraw_vectors.py)")
    ap.add_argument("--weights", help="path to model weights JSON (brain_a/brain_b)")
    ap.add_argument("--test_frac", type=float, default=0.2,
                     help="fraction of EPISODES held out for testing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--add_actraw_feature", action="store_true",
                     help="ablation: also feed one-hot act_id_raw as an extra "
                          "input feature to the x(t+1) predictor. Off by default: "
                          "the physically-causal driver of x(t+1) is `action` "
                          "(final, post-instinct), not the raw intent.")
    ap.add_argument("--out_dir", default="results",
                     help="where to write comparison_report.csv")
    ap.add_argument("--subset_episodes", type=int, default=None,
                     help="quick timing probe: restrict to N random episodes "
                          "(both agents kept) before running the comparison, "
                          "e.g. --subset_episodes 30 to estimate per-machine "
                          "time before committing to the full 300-episode run.")
    ap.add_argument("--skip_trees", action="store_true",
                     help="skip random_forest/gradient_boosting entirely (Ridge+MLP only). "
                          "Use this for the full-dataset run once a subset probe already "
                          "showed the trees don't beat Ridge on chained h(t+1) R^2.")
    ap.add_argument("--selftest", action="store_true",
                     help="run on synthetic data instead of real CSV/weights")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.csv or not args.weights:
        raise SystemExit("Need --csv and --weights (or use --selftest).")

    df, h_cols, x_cols, has_actraw = load_data(args.csv)
    if args.add_actraw_feature and not has_actraw:
        raise SystemExit("--add_actraw_feature requested but CSV has no act_id_raw column.")

    n_episodes_total = df["episode"].nunique()
    if args.subset_episodes:
        df = subset_by_episode(df, args.subset_episodes, args.seed)
        print(f"[subset] Using {args.subset_episodes} of {n_episodes_total} episodes "
              f"for a timing probe. Scale times roughly by "
              f"{n_episodes_total / args.subset_episodes:.1f}x for the full run -- "
              f"treat that as a LOWER bound: tree-based models (RF/GBM) often scale "
              f"worse than linearly with rows, not better.")

    with open(args.weights) as f:
        weights = json.load(f)

    pairs = build_pairs(df, h_cols, x_cols, has_actraw)
    run_comparison(pairs, weights, args.test_frac, args.seed, args.add_actraw_feature,
                    args.out_dir, skip_trees=args.skip_trees)


if __name__ == "__main__":
    main()
