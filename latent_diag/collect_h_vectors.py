"""
Collection of h vectors (GRU memory state) with context labels, for the
latent-space analysis (PCA / classifier). This script does not perform
any PCA yet -- it only gathers and saves the raw data, one h vector per
decision, with the labels needed later for coloring/classification.

Labels saved for each decision:
  agent (a/b), episode, decision, stalled (bool), heard_signal (bool),
  near_wall (bool), own_success (bool), chosen action,
  and the 16 components of h (h_0 .. h_15).

Usage (start SMALL before scaling up):
    python3 collect_h_vectors.py --model models/social3d_v15_full_final.json --episodes 30 --out h_vectors_pilot.csv
    # if the pilot looks good:
    python3 collect_h_vectors.py --model models/social3d_v15_full_final.json --episodes 300 --out h_vectors_300.csv
"""
import argparse
import csv
import json

import numpy as np
import pybullet as p

from foraging_env3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate_3d import world_for_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output CSV file")
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)

    fieldnames = ["episode", "agent", "decision", "stalled", "heard_signal",
                  "near_wall", "own_success", "action"] + [f"h_{k}" for k in range(16)]

    n_rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep_idx in range(args.episodes):
            if ep_idx % 10 == 0:
                print(f"  ep {ep_idx}/{args.episodes}... ({n_rows} rows so far)", flush=True)

            world_name = world_for_episode(ep_idx)
            seed_a, seed_b = int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1))
            env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
            ctrl_a.reset(); ctrl_b.reset()

            signals = [0, 0]
            at = [False, False]
            num_decisions = args.max_steps // env.brain_ratio

            for dec in range(num_decisions):
                sensors_pre = [env._get_sensors(env.robots[i]) for i in range(2)]
                other_signal_pre = [signals[1 - i] for i in range(2)]

                out = None
                for _tick in range(env.brain_ratio):
                    out = env.step(controllers, signals)

                for i in range(2):
                    if out[i]["success"]:
                        at[i] = True

                for i in range(2):
                    near_wall = bool(np.max(sensors_pre[i]) > 0.82)
                    heard = bool(other_signal_pre[i] == 4)
                    h_vec = controllers[i].brain.h.copy()
                    row = {
                        "episode": ep_idx,
                        "agent": "a" if i == 0 else "b",
                        "decision": dec,
                        "stalled": out[i]["stalled"],
                        "heard_signal": heard,
                        "near_wall": near_wall,
                        "own_success": at[i],
                        "action": out[i]["signal"],
                    }
                    for k in range(16):
                        row[f"h_{k}"] = float(h_vec[k])
                    writer.writerow(row)
                    n_rows += 1

                if all(at):
                    break

    print(f"\nDone. {n_rows} rows saved to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
