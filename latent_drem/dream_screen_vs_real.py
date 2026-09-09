"""
dream_screen_vs_real.py

Step 3 of the A2 plan (see README_world_model_addition.md): before
wiring the world-model screening into a real GA run, check whether it
actually ranks candidates the way real physics does. Not "does the
model predict states accurately" (already measured, several times) --
specifically "does it preserve the ORDER of a population by fitness",
which is the only thing that matters for a screening filter.

For a small population of candidate weight-perturbations (same
init scheme train_social3d.py uses to seed a generation: base + gaussian
noise), computes two scores per candidate:

  1. REAL fitness: evaluate_individual_3d() from evaluate3d.py, used
     completely unmodified -- same function the real GA calls. Full
     physics, all fitness terms (shaping, frontal, stuck, social
     rescue/recruit/abandon, terminal bonus).

  2. DREAMED shaping score: cheap, no physics beyond ONE real decision
     step per episode (needed to get a real starting h/x -- see below).
     From there, each candidate's own GRUNetwork.forward() (the real
     class from controllers.py, not a reimplementation -- avoids the
     drift risk of duplicating that logic) makes its own decisions,
     and the Ridge world-model predicts x(t+1) from them, chained for
     `--horizon` steps (default 20, the range where earlier rollout
     fidelity checks showed low error). Score = sum of
     (dist_closed_this_step * DIST_MULT), the dominant CONTINUOUS
     shaping term in evaluate3d.py, while not yet "at" the goal --
     same formula, read directly from evaluate3d.py, not approximated.

     Deliberately EXCLUDED from the dreamed score (documented
     simplifications, not oversights):
       - Frontal/stuck/social-rescue/recruit/abandon terms: all much
         smaller than DIST_MULT in aggregate and would need either
         joint two-agent physics or the basal-ganglia instinct, neither
         of which the world model captures.
       - The terminal bonus: already established, across five separate
         attempts (weighted loss, MLP, cross-policy, pooled training,
         longer horizons), that this isn't reliably predictable. Not
         attempted here either -- this script deliberately only tests
         the piece of the fitness we already trust.

  Both scores computed with the SAME episode seeds per candidate (like
  evaluate_individual_3d's own docstring insists on for fair
  within-generation comparison).

Reports the Spearman rank correlation between the two across the
population, plus wall-clock time for each approach -- the actual
answer to "is this filter usable AND actually cheaper".

Needs to run on a machine with PyBullet (for evaluate_individual_3d and
the one real decision step used to seed the dream) -- same environment
as everything else in this session using train_social3d.py/evaluate3d.py.

Usage:
    python3 dream_screen_vs_real.py \
        --base_model ../models/social3d_v15_full_final.json \
        --train_csv ../h_x_actraw_300.csv \
        --pop 8 --episodes 9 --horizon 20 \
        --out_dir results_dream_screen/

Usage (self-test: only the parts that don't need PyBullet -- the
dreaming/scoring math and the correlation reporting -- using synthetic
weights and a synthetic Ridge predictor):
    python3 dream_screen_vs_real.py --selftest
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_THIS_DIR, os.path.join(_THIS_DIR, "latent_diag"),
                   os.getcwd(), os.path.join(os.getcwd(), "latent_diag")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from controllers import GRUNetwork  # real class, not reimplemented
import compare_world_model_candidates as wm
import validate_rollout_fidelity as vr

SUCCESS_THRESHOLD = vr.SUCCESS_THRESHOLD
HUNGER_DECAY = vr.HUNGER_DECAY
DIST_MULT = 150_000  # evaluate3d.py -- kept as a literal here (not imported)
                      # since importing evaluate3d.py pulls in the pybullet
                      # world-building side effects; value checked directly
                      # against evaluate3d.py this session, re-check there
                      # if this script starts giving surprising results.


def make_population(base_weights_path, pop_size, sigma, seed):
    """Same init scheme as train_social3d.py: base + gaussian noise,
    individual 0 is the unperturbed base (so the correlation check
    includes at least one 'known good' point)."""
    from train_social3d import get_weight_size, json_to_flat, weights_to_json
    with open(base_weights_path) as f:
        base_data = json.load(f)
    base_flat = json_to_flat(base_data)
    rng = np.random.RandomState(seed)
    w_size = get_weight_size()
    assert len(base_flat) == w_size, (
        f"Base model flat size {len(base_flat)} != get_weight_size() {w_size} "
        f"-- shape mismatch, check the model file matches this repo's current architecture.")
    population = [base_flat.copy()]
    for _ in range(pop_size - 1):
        population.append(base_flat + rng.randn(w_size) * sigma)
    return [weights_to_json(flat) for flat in population]


def get_real_starting_states(episode_seeds, world_name_fn, use_bg, seed_probe_model):
    """Runs ONE real decision (env.reset + one env.step, ~brain_ratio
    physics ticks, NOT a full episode) per episode seed, using an
    arbitrary controller (the base model) just to obtain a real starting
    (h, x) pair for both agents -- matches exactly how h_x_actraw CSVs
    define their own 't=0' (see collect_h_x_actraw_vectors.py: the first
    logged row is already post-first-decision, same convention used
    throughout this session's rollout fidelity checks). This is a small,
    fixed cost per episode (not per candidate) -- reused for every
    candidate's dream on that episode."""
    from foraging_env3d import Foraging3DEnv
    from collect_h_x_actraw_vectors import InstrumentedController  # reuses the exact
    # last_x-saving wrapper already validated in that script, instead of
    # duplicating it (Unified3DController itself doesn't store last_x --
    # confirmed by the AttributeError this fixed).

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = InstrumentedController(env.cfg, seed_probe_model, agent_type="brain_a", use_bg=use_bg)
    ctrl_b = InstrumentedController(env.cfg, seed_probe_model, agent_type="brain_b", use_bg=use_bg)

    states = []
    for ep_idx, (seed_a, seed_b) in enumerate(episode_seeds):
        world_name = world_name_fn(ep_idx)
        env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
        ctrl_a.reset(); ctrl_b.reset()
        signals = [0, 0]
        env.step([ctrl_a, ctrl_b], signals)
        states.append({
            "h_a": ctrl_a.brain.h.copy(), "x_a": ctrl_a.last_x.copy(),
            "h_b": ctrl_b.brain.h.copy(), "x_b": ctrl_b.last_x.copy(),
        })
    return states


def dream_episode_score(weights_json, ridge_model, start_state, horizon):
    """Dreams `horizon` steps for both agents independently (no joint
    physics -- see module docstring), each agent deciding its own
    action via the real GRUNetwork, scored by the same
    dist_closed * DIST_MULT formula evaluate3d.py uses, while not yet
    past SUCCESS_THRESHOLD."""
    total = 0.0
    for agent_key, agent_slot in [("brain_a", "a"), ("brain_b", "b")]:
        net = GRUNetwork(weights_json[agent_key])
        net.h = start_state[f"h_{agent_slot}"].copy()
        x_t = start_state[f"x_{agent_slot}"].copy()
        decay = HUNGER_DECAY[agent_slot]
        dist_t = vr.dist_from_hunger(x_t[9], decay)
        at_goal = dist_t < SUCCESS_THRESHOLD

        for _ in range(horizon):
            h_t = net.h.copy()
            action_t = net.forward(x_t)  # updates net.h in place
            feat = np.concatenate([h_t, x_t, wm.one_hot(action_t)])[None, :]
            x_next = ridge_model.predict(feat)[0]
            dist_next = vr.dist_from_hunger(x_next[9], decay)

            if not at_goal:
                total += (dist_t - dist_next) * DIST_MULT
                if dist_next < SUCCESS_THRESHOLD:
                    at_goal = True

            x_t, dist_t = x_next, dist_next
    return total


def run(base_model_path, train_csv, pop_size, episodes, horizon, sigma, seed, out_dir):
    print(f"[1/4] Building population: {pop_size} candidates (base + N(0,{sigma}))")
    population = make_population(base_model_path, pop_size, sigma, seed)

    print(f"[2/4] Training Ridge x(t+1) predictor on {train_csv} (all episodes)...")
    df, h_cols, x_cols, has_actraw = wm.load_data(train_csv)
    ridge_model, _ = vr.train_predictor(df, h_cols, x_cols, has_actraw,
                                         test_frac=0.0, seed=seed, model_type="ridge",
                                         train_all=True)

    rng = np.random.RandomState(seed)
    episode_seeds = [(int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1)))
                      for _ in range(episodes)]
    from evaluate3d import world_for_episode

    print(f"[3/4] Getting {episodes} real starting states (one real decision each, "
          f"not full episodes)...")
    with open(base_model_path) as f:
        base_model = json.load(f)
    start_states = get_real_starting_states(episode_seeds, world_for_episode,
                                             use_bg=True, seed_probe_model=base_model)

    print(f"[4/4] Scoring {pop_size} candidates: REAL fitness vs DREAMED shaping score...")
    from evaluate3d import evaluate_individual_3d
    from foraging_env3d import Foraging3DEnv
    env = Foraging3DEnv(world_name="random_obstacles")

    rows = []
    for i, cand in enumerate(population):
        t0 = time.perf_counter()
        real_fitness, _vocal, _stuck, _succ = evaluate_individual_3d(
            cand, h_prob=0.0, use_bg=True, max_steps=6000, episodes=episodes,
            env=env, episode_seeds=episode_seeds)
        real_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        dreamed_score = float(np.mean([
            dream_episode_score(cand, ridge_model, start_states[ep], horizon)
            for ep in range(episodes)
        ]))
        dream_time = time.perf_counter() - t0

        print(f"  candidate {i}: real_fitness={real_fitness:.1f} ({real_time:.1f}s)  "
              f"dreamed_score={dreamed_score:.1f} ({dream_time:.2f}s)")
        rows.append({"candidate": i, "real_fitness": real_fitness, "real_time_s": real_time,
                     "dreamed_score": dreamed_score, "dream_time_s": dream_time})

    df_out = pd.DataFrame(rows)
    spearman = df_out["real_fitness"].corr(df_out["dreamed_score"], method="spearman")
    total_real_time = df_out["real_time_s"].sum()
    total_dream_time = df_out["dream_time_s"].sum()

    print(f"\n=== Spearman rank correlation (real fitness vs dreamed score): {spearman:.3f} ===")
    print(f"Total time -- real: {total_real_time:.1f}s, dreamed: {total_dream_time:.2f}s "
          f"({total_real_time / max(total_dream_time, 1e-9):.0f}x)")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "dream_screen_vs_real_report.csv")
        df_out.to_csv(report_path, index=False)
        print(f"Full report written to {report_path}")
    return df_out, spearman


def run_selftest():
    """Only exercises the parts that don't need PyBullet: dreaming +
    scoring + correlation reporting, with synthetic weights/data. The
    real-fitness half (evaluate_individual_3d, get_real_starting_states)
    needs PyBullet and is NOT covered here -- that part only gets
    validated on the real run."""
    print("=== SELF-TEST (dreaming/scoring math only, no PyBullet) ===\n")
    weights = wm.make_synthetic_weights(seed=0)
    df = wm.make_synthetic_csv(weights, n_episodes=10, n_decisions=40, seed=1)
    h_cols = [f"h_{k}" for k in range(wm.H_DIM)]
    x_cols = [f"x_{k}" for k in range(wm.X_DIM)]
    ridge_model, _ = vr.train_predictor(df, h_cols, x_cols, has_actraw=True,
                                         test_frac=0.0, seed=42, model_type="ridge",
                                         train_all=True)

    start_state = {
        "h_a": np.zeros(16), "x_a": np.random.RandomState(0).normal(size=11) * 0.3,
        "h_b": np.zeros(16), "x_b": np.random.RandomState(1).normal(size=11) * 0.3,
    }
    score = dream_episode_score(weights, ridge_model, start_state, horizon=10)
    print(f"dream_episode_score() ran without error, returned {score:.2f} "
          f"(synthetic data -- number itself isn't meaningful).")

    # correlation reporting sanity check with fake data
    fake = pd.DataFrame({"real_fitness": [1, 5, 3, 8, 2], "dreamed_score": [1, 4, 3, 9, 2]})
    corr = fake["real_fitness"].corr(fake["dreamed_score"], method="spearman")
    assert corr > 0.8, "FAILED: Spearman correlation reporting isn't working as expected"
    print(f"Spearman correlation reporting check OK ({corr:.3f} on obviously-correlated fake data).")
    print("\nSelf-test finished without errors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", help="weights JSON to build the population around")
    ap.add_argument("--train_csv", help="CSV to train the Ridge x(t+1) predictor on")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=9,
                     help="episodes per candidate for BOTH real and dreamed scoring "
                          "(9 matches train_social3d.py's per-eval_repeat default)")
    ap.add_argument("--horizon", type=int, default=20,
                     help="dream horizon in decisions -- kept at 20, the range where "
                          "earlier rollout-fidelity checks showed low error")
    ap.add_argument("--sigma", type=float, default=0.03,
                     help="perturbation noise, matches train_social3d.py's population init")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results_dream_screen")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.base_model or not args.train_csv:
        raise SystemExit("Need --base_model and --train_csv (or use --selftest).")

    run(args.base_model, args.train_csv, args.pop, args.episodes, args.horizon,
        args.sigma, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
