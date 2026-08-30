"""
Step 4c of the latent-space plan: revised convergence metric.

The earlier velocity-based metric (measure_h_convergence.py) compared
average step-to-step SPEED before vs after an initial window, and
turned out to measure the wrong thing: visualizing extreme cases showed
that even "unconverged" cases (high ratio) were often confined to a
small region of h-space but still vibrating within it -- the velocity
metric penalizes that residual vibration the same as never confining
at all.

This version instead measures SPATIAL CONFINEMENT: after the initial
jump window (first N_INITIAL decisions), how tightly clustered are the
remaining h vectors around their own centroid? A confined case has a
small spread (low mean distance to centroid) even if individual steps
still move around a bit. This is compared against the spread of the
initial window itself, giving a confinement ratio: spread_settled /
spread_initial. Low ratio = settles into a small region (confinement).
Ratio near 1 = no more confined later than at the start (no real
convergence to a region).

Aggregated per agent, across many real stuck cases, with progress
printed periodically (long-running search).

Usage:
    python3 measure_h_confinement.py --model models/social3d_v15_full_final.json --episodes 300 --min_streak 15 --out h_confinement.csv
"""
import argparse
import csv
import json

import numpy as np
import pybullet as p

from foraging_env_3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate_3d import world_for_episode

N_INITIAL = 5  # decisions counted as the "initial jump" window


def spread(h_arr):
    """Mean distance of each row to the centroid of h_arr."""
    centroid = h_arr.mean(axis=0)
    dists = np.linalg.norm(h_arr - centroid, axis=1)
    return float(np.mean(dists))


def summarize_streak(h_seq):
    h_arr = np.array(h_seq)
    if len(h_arr) <= N_INITIAL + 2:
        return None  # need enough points in both windows for a meaningful spread

    initial = h_arr[:N_INITIAL]
    settled = h_arr[N_INITIAL:]
    initial_spread = spread(initial)
    settled_spread = spread(settled)
    ratio = settled_spread / initial_spread if initial_spread > 1e-9 else float("nan")
    return {
        "streak_len": len(h_seq),
        "initial_spread": initial_spread,
        "settled_spread": settled_spread,
        "settled_to_initial_spread_ratio": ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--min_streak", type=int, default=15)
    ap.add_argument("--max_cases_per_agent", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)
    results = {"a": [], "b": []}
    counts = {"a": 0, "b": 0}

    for ep_idx in range(args.episodes):
        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx}/{args.episodes}... (found so far: A={counts['a']}, B={counts['b']})", flush=True)
        if counts["a"] >= args.max_cases_per_agent and counts["b"] >= args.max_cases_per_agent:
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
                label = "a" if i == 0 else "b"
                if counts[label] >= args.max_cases_per_agent:
                    continue
                cur_pos = np.array(out[i]["pos"][:2])
                net_move = float(np.linalg.norm(cur_pos - pos_pre[i]))
                act_id = out[i]["signal"]
                is_turn_stuck = (act_id in (1, 2)) and (net_move < 0.02)
                h_now = controllers[i].brain.h.copy()
                if is_turn_stuck:
                    turn_streak[i] += 1
                    h_history[i].append(h_now.copy())
                else:
                    if turn_streak[i] >= args.min_streak:
                        summary = summarize_streak(h_history[i])
                        if summary is not None:
                            results[label].append(summary)
                            counts[label] += 1
                    turn_streak[i] = 0
                    h_history[i] = []
            if all(at) or (counts["a"] >= args.max_cases_per_agent and counts["b"] >= args.max_cases_per_agent):
                break

    print(f"\n=== h confinement summary: {args.model} ===\n")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "streak_len", "initial_spread", "settled_spread", "settled_to_initial_spread_ratio"])
            for label in ["a", "b"]:
                for c in results[label]:
                    writer.writerow([label, c["streak_len"], c["initial_spread"],
                                      c["settled_spread"], c["settled_to_initial_spread_ratio"]])
        print(f"Per-case data saved to {args.out}\n")

    for label in ["a", "b"]:
        cases = results[label]
        print(f"--- Agent {label.upper()} -- {len(cases)} stuck cases found ---")
        if not cases:
            print("  No cases found.\n")
            continue
        ratios = [c["settled_to_initial_spread_ratio"] for c in cases if not np.isnan(c["settled_to_initial_spread_ratio"])]
        streak_lens = [c["streak_len"] for c in cases]
        print(f"  Median streak length: {np.median(streak_lens):.0f}")
        print(f"  Median settled/initial SPREAD ratio: {np.median(ratios):.3f} "
              f"(lower = more confined to a small region after the initial jump)")
        print(f"  Mean settled/initial SPREAD ratio:   {np.mean(ratios):.3f}")
        print(f"  Individual ratios: {[round(r,3) for r in ratios]}\n")

    env.close()


if __name__ == "__main__":
    main()
