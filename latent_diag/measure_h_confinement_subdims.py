"""
Follow-up to step 5: repeats the spatial-confinement analysis from
measure_h_confinement.py, but restricted to a SUBSET of h dimensions
instead of all 16 -- specifically the hard-to-predict dimensions
identified by predict_h_next.py (h_3, h_4, h_10, R^2 = 0.47-0.67),
versus the easy-to-predict ones (h_2, h_5, h_7, h_9, h_12, h_13, h_14,
R^2 = 0.95-0.998), as a comparison.

Rationale: steps 3/4 found messy, inconsistent confinement patterns
when using all 16 dimensions together. If that noise is concentrated in
the hard-to-predict dimensions, restricting the analysis to the easy
dimensions should show CLEANER confinement, and restricting to the hard
dimensions should show the noise concentrated there (or possibly reveal
a different, cleaner pattern of its own, isolated from the "well-behaved"
dimensions diluting it).

Prints progress every 20 episodes (long-running search).

Usage:
    python3 measure_h_confinement_subdims.py --model models/social3d_v15_full_final.json --episodes 300 --min_streak 15
"""
import argparse
import json

import numpy as np
import pybullet as p

from foraging_env_3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate_3d import world_for_episode

N_INITIAL = 5
HARD_DIMS = [3, 4, 10]
EASY_DIMS = [2, 5, 7, 9, 12, 13, 14]


def spread(h_arr):
    centroid = h_arr.mean(axis=0)
    dists = np.linalg.norm(h_arr - centroid, axis=1)
    return float(np.mean(dists))


def summarize_streak(h_seq, dims):
    h_arr = np.array(h_seq)[:, dims]
    if len(h_arr) <= N_INITIAL + 2:
        return None
    initial_spread = spread(h_arr[:N_INITIAL])
    settled_spread = spread(h_arr[N_INITIAL:])
    ratio = settled_spread / initial_spread if initial_spread > 1e-9 else float("nan")
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--min_streak", type=int, default=15)
    ap.add_argument("--max_cases_per_agent", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)
    results = {"a": [], "b": []}   # each item: full 16-dim h_seq
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
                    if turn_streak[i] >= args.min_streak and len(h_history[i]) > N_INITIAL + 2:
                        results[label].append(list(h_history[i]))
                        counts[label] += 1
                    turn_streak[i] = 0
                    h_history[i] = []
            if all(at) or (counts["a"] >= args.max_cases_per_agent and counts["b"] >= args.max_cases_per_agent):
                break

    print(f"\n=== Confinement by dimension subset: {args.model} ===\n")

    for label in ["a", "b"]:
        seqs = results[label]
        print(f"--- Agent {label.upper()} -- {len(seqs)} stuck cases ---")
        if not seqs:
            print("  No cases found.\n")
            continue
        for name, dims in [("ALL 16 dims", list(range(16))), ("HARD-to-predict dims (3,4,10)", HARD_DIMS),
                            ("EASY-to-predict dims (2,5,7,9,12,13,14)", EASY_DIMS)]:
            ratios = [summarize_streak(seq, dims) for seq in seqs]
            ratios = [r for r in ratios if r is not None and not np.isnan(r)]
            print(f"  {name}: median ratio = {np.median(ratios):.3f}, "
                  f"mean = {np.mean(ratios):.3f}, range = [{min(ratios):.3f}, {max(ratios):.3f}]")
        print()

    env.close()


if __name__ == "__main__":
    main()
