"""
truncated_eval_vs_full.py

A different acceleration mechanism from everything else this session:
no learned world model involved at all, so it doesn't inherit any of
A2's problems (the sign-flip non-replication, the discrete-event blind
spot). Real physics throughout -- just less of it per candidate.

Question: does fitness measured over a SHORT episode already rank a
population similarly to fitness over the FULL episode? If so, cheap
truncated evaluation could screen a generation before spending the full
budget on survivors -- standard technique in evolutionary computation,
unrelated to anything about world models.

IMPORTANT CORRECTION vs. earlier this session: dream_screen_vs_real.py
used max_steps=6000 (400 decisions) as "real fitness". That does NOT
match what train_social3d.py actually trains against
(--max_steps default = 2000, ~133 decisions). This script anchors
"full" to that real value, not the longer diagnostic-collection one.

  full_steps      = 2000  (train_social3d.py's actual default)
  truncated_steps = 1000  (half budget, ~67 decisions)

Same population-generation scheme as dream_screen_vs_real.py (base +
gaussian noise, same seed/sigma semantics), same episode_seeds shared
between the full and truncated evaluation of each candidate for a fair
comparison (matches evaluate_individual_3d's own fairness requirement).

Usage:
    python3 truncated_eval_vs_full.py \
        --base_model ../models/social3d_v15_full_final.json \
        --pop 30 --episodes 9 --full_steps 2000 --truncated_steps 1000 \
        --sigma 0.03 --seed 1 \
        --out_dir results_truncated_eval/seed1_sigma0.03/

Usage (self-test: only the statistics-reporting function, with fake
data -- the population/physics parts need PyBullet and a real base
model and aren't covered here, same limitation as dream_screen_vs_real.py):
    python3 truncated_eval_vs_full.py --selftest
"""

import argparse
import json
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


def spearman_report(full_fitness, truncated_fitness):
    rho, p = stats.spearmanr(full_fitness, truncated_fitness)
    n = len(full_fitness)
    if n < 4 or abs(rho) >= 1.0:
        return rho, p, (float("nan"), float("nan")), n
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return rho, p, (lo, hi), n


def run(base_model_path, pop_size, episodes, full_steps, truncated_steps, sigma, seed, out_dir):
    from dream_screen_vs_real import make_population  # reuse the exact same
    # population-generation logic (base + gaussian noise, train_social3d.py's
    # own init scheme) instead of duplicating it
    from evaluate3d import evaluate_individual_3d
    from foraging_env3d import Foraging3DEnv

    print(f"[1/2] Building population: {pop_size} candidates (base + N(0,{sigma}))")
    population = make_population(base_model_path, pop_size, sigma, seed)

    rng = np.random.RandomState(seed)
    episode_seeds = [(int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1)))
                      for _ in range(episodes)]
    env = Foraging3DEnv(world_name="random_obstacles")

    print(f"[2/2] Evaluating {pop_size} candidates at full_steps={full_steps} "
          f"and truncated_steps={truncated_steps}...")
    rows = []
    for i, cand in enumerate(population):
        t0 = time.perf_counter()
        full_fitness, _, _, _ = evaluate_individual_3d(
            cand, h_prob=0.0, use_bg=True, max_steps=full_steps,
            episodes=episodes, env=env, episode_seeds=episode_seeds)
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        trunc_fitness, _, _, _ = evaluate_individual_3d(
            cand, h_prob=0.0, use_bg=True, max_steps=truncated_steps,
            episodes=episodes, env=env, episode_seeds=episode_seeds)
        trunc_time = time.perf_counter() - t0

        print(f"  candidate {i}: full={full_fitness:.1f} ({full_time:.1f}s)  "
              f"truncated={trunc_fitness:.1f} ({trunc_time:.1f}s)")
        rows.append({"candidate": i, "full_fitness": full_fitness, "full_time_s": full_time,
                     "truncated_fitness": trunc_fitness, "truncated_time_s": trunc_time})

    df = pd.DataFrame(rows)
    rho, p, (lo, hi), n = spearman_report(df["full_fitness"].values, df["truncated_fitness"].values)
    total_full = df["full_time_s"].sum()
    total_trunc = df["truncated_time_s"].sum()

    print(f"\n=== Spearman (full vs truncated): rho={rho:.3f}  p={p:.4f}  "
          f"95% CI=({lo:.3f},{hi:.3f})  n={n} ===")
    print(f"Total time -- full: {total_full:.1f}s, truncated: {total_trunc:.1f}s "
          f"({total_full / max(total_trunc, 1e-9):.1f}x)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        model_tag = os.path.splitext(os.path.basename(base_model_path))[0]
        report_name = (f"truncated_eval_report_{model_tag}_full{full_steps}"
                        f"_trunc{truncated_steps}_sigma{sigma}_seed{seed}.csv")
        report_path = os.path.join(out_dir, report_name)
        df.to_csv(report_path, index=False)
        print(f"Full report written to {report_path}")
    return df, rho, p


def run_selftest():
    print("=== SELF-TEST (statistics reporting only, no PyBullet) ===\n")
    rng = np.random.RandomState(0)
    full = rng.uniform(100, 1000, 30)
    truncated = full * 0.6 + rng.normal(0, 20, 30)  # strongly correlated fake data
    rho, p, (lo, hi), n = spearman_report(full, truncated)
    assert rho > 0.8 and p < 0.05, "FAILED: should detect strong correlation in fake data"
    print(f"Strongly-correlated fake data: rho={rho:.3f} p={p:.4f} -- detected correctly.")

    uncorrelated = rng.uniform(100, 1000, 30)
    rho2, p2, _, _ = spearman_report(full, uncorrelated)
    print(f"Uncorrelated fake data: rho={rho2:.3f} p={p2:.4f} (not asserted, just illustrative).")
    print("\nSelf-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--full_steps", type=int, default=2000,
                     help="matches train_social3d.py's actual default max_steps")
    ap.add_argument("--truncated_steps", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results_truncated_eval")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.base_model:
        raise SystemExit("Need --base_model (or use --selftest).")

    run(args.base_model, args.pop, args.episodes, args.full_steps, args.truncated_steps,
        args.sigma, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
