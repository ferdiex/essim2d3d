"""
episode_count_baseline.py

Baseline for the truncated_eval_vs_full.py result: instead of cutting
episode LENGTH (max_steps), cuts episode COUNT (fewer full-length
episodes). Matched to roughly the SAME speedup as the confirmed
1500-step point (~1.3x) so the comparison is apples-to-apples --
9 episodes -> 7 episodes gives 9/7 ~= 1.29x, close enough.

The reduced-episode score uses the FIRST `--reduced_episodes` seeds of
the same `--episodes` seed list (not a different random subset) --
mirrors how a real GA would just evaluate fewer episodes per candidate,
not a differently-sampled set.

Same population-generation scheme, same statistics reporting, same
discovery+confirmation discipline as truncated_eval_vs_full.py -- only
the thing being cut is different.

NOTE ON TIMING: do not assume this is fast. Each candidate now needs
TWO real evaluate_individual_3d calls (9 episodes + 7 episodes, both
full max_steps=2000), so per-candidate cost is similar to or a bit
higher than truncated_eval_vs_full.py's. A single pop=30 run is
probably several minutes, not 2-3 -- there's no reliable way to know
in advance without running it once and timing it.

Usage:
    python3 episode_count_baseline.py \
        --base_model ../models/social3d_v15_full_final.json \
        --pop 30 --episodes 9 --reduced_episodes 7 --max_steps 2000 \
        --sigma 0.03 --seed 1 \
        --out_dir results_episode_baseline/seed1/

Usage (self-test: statistics reporting only, no PyBullet):
    python3 episode_count_baseline.py --selftest
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.join(_THIS_DIR, "latent_diag"),
                   os.getcwd(), os.path.join(os.getcwd(), "latent_diag")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def spearman_report(a, b):
    rho, p = stats.spearmanr(a, b)
    n = len(a)
    if n < 4 or abs(rho) >= 1.0:
        return rho, p, (float("nan"), float("nan")), n
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return rho, p, (lo, hi), n


def run(base_model_path, pop_size, episodes, reduced_episodes, max_steps, sigma, seed, out_dir):
    from dream_screen_vs_real import make_population
    from evaluate3d import evaluate_individual_3d
    from foraging_env3d import Foraging3DEnv

    if reduced_episodes >= episodes:
        raise SystemExit(f"--reduced_episodes ({reduced_episodes}) must be < --episodes ({episodes}).")

    print(f"[1/2] Building population: {pop_size} candidates (base + N(0,{sigma}))")
    population = make_population(base_model_path, pop_size, sigma, seed)

    rng = np.random.RandomState(seed)
    episode_seeds_full = [(int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1)))
                           for _ in range(episodes)]
    episode_seeds_reduced = episode_seeds_full[:reduced_episodes]  # SAME first seeds, not
    # a different random subset -- mirrors what a real GA would do if it just ran fewer episodes

    env = Foraging3DEnv(world_name="random_obstacles")

    print(f"[2/2] Evaluating {pop_size} candidates: full={episodes} episodes vs "
          f"reduced={reduced_episodes} episodes (both max_steps={max_steps})...")
    rows = []
    for i, cand in enumerate(population):
        t0 = time.perf_counter()
        full_fitness, _, _, _ = evaluate_individual_3d(
            cand, h_prob=0.0, use_bg=True, max_steps=max_steps,
            episodes=episodes, env=env, episode_seeds=episode_seeds_full)
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        reduced_fitness, _, _, _ = evaluate_individual_3d(
            cand, h_prob=0.0, use_bg=True, max_steps=max_steps,
            episodes=reduced_episodes, env=env, episode_seeds=episode_seeds_reduced)
        reduced_time = time.perf_counter() - t0

        print(f"  candidate {i}: full={full_fitness:.1f} ({full_time:.1f}s)  "
              f"reduced={reduced_fitness:.1f} ({reduced_time:.1f}s)")
        rows.append({"candidate": i, "full_fitness": full_fitness, "full_time_s": full_time,
                     "reduced_fitness": reduced_fitness, "reduced_time_s": reduced_time})

    df = pd.DataFrame(rows)
    rho, p, (lo, hi), n = spearman_report(df["full_fitness"].values, df["reduced_fitness"].values)
    total_full = df["full_time_s"].sum()
    total_reduced = df["reduced_time_s"].sum()

    print(f"\n=== Spearman (full vs reduced-episode-count): rho={rho:.3f}  p={p:.4f}  "
          f"95% CI=({lo:.3f},{hi:.3f})  n={n} ===")
    print(f"Total time -- full: {total_full:.1f}s, reduced: {total_reduced:.1f}s "
          f"({total_full / max(total_reduced, 1e-9):.2f}x)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        model_tag = os.path.splitext(os.path.basename(base_model_path))[0]
        report_name = (f"episode_baseline_report_{model_tag}_ep{episodes}"
                        f"_reduced{reduced_episodes}_steps{max_steps}_sigma{sigma}_seed{seed}.csv")
        report_path = os.path.join(out_dir, report_name)
        df.to_csv(report_path, index=False)
        print(f"Full report written to {report_path}")
    return df, rho, p


def run_selftest():
    print("=== SELF-TEST (statistics reporting only, no PyBullet) ===\n")
    rng = np.random.RandomState(0)
    full = rng.uniform(100, 1000, 30)
    reduced = full * 0.7 + rng.normal(0, 30, 30)
    rho, p, (lo, hi), n = spearman_report(full, reduced)
    assert rho > 0.7 and p < 0.05, "FAILED: should detect strong correlation in fake data"
    print(f"Strongly-correlated fake data: rho={rho:.3f} p={p:.4f} -- detected correctly.")
    print("\nSelf-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--reduced_episodes", type=int, default=7,
                     help="9->7 gives ~1.29x speedup, matched to the confirmed "
                          "1500-step truncation's ~1.3x")
    ap.add_argument("--max_steps", type=int, default=2000,
                     help="FULL length for both full and reduced evaluation -- "
                          "only episode COUNT differs, not length")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results_episode_baseline")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.base_model:
        raise SystemExit("Need --base_model (or use --selftest).")

    run(args.base_model, args.pop, args.episodes, args.reduced_episodes, args.max_steps,
        args.sigma, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
