"""
Plots h(t) trajectory in PCA space, side by side with the physical
(x, y) position trajectory, for real stuck episodes found by search
(same search logic as diagnose_corner_trap.py).

This is step 3 of the latent-space plan: the clustering step (step 2)
told us HOW MANY dominant memory regions exist per agent, but not
whether the system actually settles into (and stays in) that region
during a stuck episode, or just passes through it. Plotting h(t) over
time, with a time gradient, next to the matching physical trajectory,
lets us see this directly.

The PCA projection used here is fit on a large reference dataset
(e.g. h_vectors_300.csv, already collected) so that trajectories from
different episodes/cases can be compared on the SAME axes -- fitting a
fresh PCA per case would make the 2D coordinates meaningless across
cases.

Usage:
    python3 plot_h_trajectory.py --model models/social3d_v15_full_final.json \
        --pca_reference_csv h_vectors_300.csv --episodes 100 --min_streak 40 --max_cases 3
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
import pybullet as p

from foraging_env_3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate_3d import world_for_episode


def fit_reference_pca(csv_path):
    df = pd.read_csv(csv_path)
    h_cols = [c for c in df.columns if c.startswith("h_")]
    pca = PCA(n_components=2)
    pca.fit(df[h_cols].values)
    print(f"Reference PCA fit on {len(df)} rows from {csv_path} "
          f"(explained variance: {pca.explained_variance_ratio_[0]:.1%} + "
          f"{pca.explained_variance_ratio_[1]:.1%})")
    return pca


def plot_case(case_num, agent_label, h_seq, pos_seq, pca, out_prefix):
    h_pca = pca.transform(np.array(h_seq))
    pos_arr = np.array(pos_seq)
    t = np.arange(len(h_seq))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    points = h_pca.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", array=t[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)
    sc = ax.scatter(h_pca[:, 0], h_pca[:, 1], c=t, cmap="viridis", s=15, zorder=3)
    ax.scatter(h_pca[0, 0], h_pca[0, 1], c="black", marker="o", s=80, label="start", zorder=4)
    ax.scatter(h_pca[-1, 0], h_pca[-1, 1], c="red", marker="X", s=100, label="end", zorder=4)
    ax.set_xlabel("PC1 (reference)")
    ax.set_ylabel("PC2 (reference)")
    ax.set_title(f"h(t) trajectory in PCA space\nagent {agent_label.upper()}, case {case_num}")
    ax.legend()
    ax.autoscale()
    plt.colorbar(sc, ax=ax, label="decision index")

    ax = axes[1]
    points_xy = pos_arr.reshape(-1, 1, 2)
    segments_xy = np.concatenate([points_xy[:-1], points_xy[1:]], axis=1)
    lc_xy = LineCollection(segments_xy, cmap="viridis", array=t[:-1])
    lc_xy.set_linewidth(2)
    ax.add_collection(lc_xy)
    sc2 = ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c=t, cmap="viridis", s=15, zorder=3)
    ax.scatter(pos_arr[0, 0], pos_arr[0, 1], c="black", marker="o", s=80, label="start", zorder=4)
    ax.scatter(pos_arr[-1, 0], pos_arr[-1, 1], c="red", marker="X", s=100, label="end", zorder=4)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_title(f"Physical (x, y) trajectory\nagent {agent_label.upper()}, case {case_num}")
    ax.legend()
    ax.set_aspect("equal")
    ax.autoscale()
    plt.colorbar(sc2, ax=ax, label="decision index")

    plt.tight_layout()
    fname = f"{out_prefix}_case{case_num}_agent{agent_label}.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pca_reference_csv", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--min_streak", type=int, default=40)
    ap.add_argument("--max_cases", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_prefix", default="h_trajectory")
    args = ap.parse_args()

    pca = fit_reference_pca(args.pca_reference_csv)

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)
    cases_found = 0

    for ep_idx in range(args.episodes):
        if cases_found >= args.max_cases:
            break
        world_name = world_for_episode(ep_idx)
        seed_a, seed_b = int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1))
        env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
        ctrl_a.reset(); ctrl_b.reset()

        signals = [0, 0]
        at = [False, False]
        num_decisions = args.max_steps // env.brain_ratio

        turn_streak = [0, 0]
        h_history = [[], []]
        pos_history = [[], []]

        for dec in range(num_decisions):
            pos_pre = [np.array(p.getBasePositionAndOrientation(env.robots[i])[0][:2]) for i in range(2)]

            out = None
            for _tick in range(env.brain_ratio):
                out = env.step(controllers, signals)

            for i in range(2):
                if out[i]["success"]:
                    at[i] = True

            for i in range(2):
                if at[i]:
                    continue
                cur_pos = np.array(out[i]["pos"][:2])
                net_move = float(np.linalg.norm(cur_pos - pos_pre[i]))
                act_id = out[i]["signal"]
                is_turn_stuck = (act_id in (1, 2)) and (net_move < 0.02)
                h_now = controllers[i].brain.h.copy()

                if is_turn_stuck:
                    turn_streak[i] += 1
                    h_history[i].append(h_now.copy())
                    pos_history[i].append(cur_pos.copy())
                else:
                    if turn_streak[i] >= args.min_streak:
                        cases_found += 1
                        agent_label = "a" if i == 0 else "b"
                        print(f"Case {cases_found}: episode {ep_idx}, agent {agent_label.upper()}, "
                              f"streak length {turn_streak[i]}")
                        plot_case(cases_found, agent_label, h_history[i], pos_history[i], pca, args.out_prefix)
                        if cases_found >= args.max_cases:
                            break
                    turn_streak[i] = 0
                    h_history[i] = []
                    pos_history[i] = []

            if cases_found >= args.max_cases or all(at):
                break

    if cases_found == 0:
        print(f"No stuck case found with streak >= {args.min_streak} in {args.episodes} episodes. "
              f"Try more episodes or a lower --min_streak.")

    env.close()


if __name__ == "__main__":
    main()
