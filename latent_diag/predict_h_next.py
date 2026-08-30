"""
Step 5 of the latent-space plan: cheap world-model feasibility check.

This does NOT build a world model. It only asks a much smaller
question: is h(t+1) predictable from h(t) (and the action taken) well
enough that training a full world model would plausibly be worth
attempting later? This reuses the already-collected h_vectors_300.csv
-- no new simulation is run.

Method: train a simple regressor (Ridge regression, i.e. regularized
linear regression -- deliberately simple, this is a feasibility check,
not a serious predictive model) to predict h(t+1) from [h(t), one-hot
action(t)], using consecutive decisions within the same episode and
agent (never crossing an episode/agent boundary). Report R^2 on held-out
episodes (not held-out rows -- splitting by episode avoids leaking
information between train and test from the same trajectory).

A high R^2 (e.g. > 0.8-0.9) would suggest the internal dynamics are
regular enough that a world-model approach is a reasonable next
project. A low R^2 would suggest the dynamics are too noisy/chaotic for
a simple predictive model to work well, and a world-model project would
likely need a much more powerful predictor (or might not be worth
pursuing at all).

Usage:
    python3 predict_h_next.py --csv h_vectors_300.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def build_pairs(df, h_cols, n_actions=5):
    """Build (h(t), action(t)) -> h(t+1) pairs, never crossing episode/agent boundaries."""
    X_list, y_list, ep_list = [], [], []
    grouped = df.sort_values(["episode", "agent", "decision"]).groupby(["episode", "agent"])
    for (ep, agent), group in grouped:
        h_vals = group[h_cols].values
        actions = group["action"].values
        if len(h_vals) < 2:
            continue
        for t in range(len(h_vals) - 1):
            action_onehot = np.zeros(n_actions)
            a = int(actions[t])
            if 0 <= a < n_actions:
                action_onehot[a] = 1.0
            X_list.append(np.concatenate([h_vals[t], action_onehot]))
            y_list.append(h_vals[t + 1])
            ep_list.append(ep)
    return np.array(X_list), np.array(y_list), np.array(ep_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2, help="fraction of EPISODES held out for testing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading data...", flush=True)
    df = pd.read_csv(args.csv)
    h_cols = [c for c in df.columns if c.startswith("h_")]
    print(f"Loaded {len(df)} rows, {len(h_cols)} h-dimensions.")

    print("Building (h(t), action(t)) -> h(t+1) pairs...", flush=True)
    X, y, episodes = build_pairs(df, h_cols)
    print(f"Built {len(X)} prediction pairs.")

    rng = np.random.RandomState(args.seed)
    unique_eps = np.unique(episodes)
    rng.shuffle(unique_eps)
    n_test_eps = max(1, int(len(unique_eps) * args.test_frac))
    test_eps = set(unique_eps[:n_test_eps])
    train_mask = np.array([ep not in test_eps for ep in episodes])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"Train: {len(X_train)} pairs from {len(unique_eps) - n_test_eps} episodes. "
          f"Test: {len(X_test)} pairs from {n_test_eps} held-out episodes.")

    print("Fitting Ridge regression...", flush=True)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    print("Evaluating...", flush=True)
    pred_test = model.predict(X_test)
    r2_overall = r2_score(y_test, pred_test)
    r2_per_dim = r2_score(y_test, pred_test, multioutput="raw_values")

    # baseline: predicting h(t+1) = h(t) (no change) -- sanity check for how much
    # of the "predictability" is just h barely changing tick to tick
    naive_pred = X_test[:, :len(h_cols)]  # the h(t) part of the input
    r2_naive = r2_score(y_test, naive_pred)

    print(f"\n=== Results ===")
    print(f"Ridge regression R^2 (overall, held-out episodes): {r2_overall:.3f}")
    print(f"Naive baseline R^2 (predict h(t+1) = h(t)):        {r2_naive:.3f}")
    print(f"R^2 per h-dimension: {[round(v, 3) for v in r2_per_dim]}")
    print()
    if r2_overall > 0.9:
        print("Interpretation: very high predictability -- h(t+1) is almost fully "
              "determined by h(t) and the action. A world-model approach looks feasible.")
    elif r2_overall > 0.7:
        print("Interpretation: reasonably good predictability -- a world model may be "
              "feasible, but would likely need a more expressive predictor than this "
              "simple linear baseline to capture the remaining structure.")
    else:
        print("Interpretation: predictability is weak -- either the dynamics are genuinely "
              "noisy/complex relative to what a simple linear model can capture (a more "
              "powerful model, e.g. a small MLP, should be tried before concluding a world "
              "model is infeasible), or there is a large irreducibly stochastic component.")
    if r2_naive > r2_overall - 0.05:
        print("\nNote: the naive 'no change' baseline performs almost as well as the fitted "
              "model -- most of the apparent predictability may simply reflect that h changes "
              "slowly tick to tick, not that the model learned meaningful dynamics.")


if __name__ == "__main__":
    main()
