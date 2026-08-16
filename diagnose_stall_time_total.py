"""
Diagnostico agregado: tiempo TOTAL atascado por episodio (suma de todos
los ticks con stalled=True en el episodio completo, no por racha
individual). Mas robusto que diagnose_histeresis_stall.py para comparar
el efecto del cambio de histeresis (v17) en agregado -- las rachas
individuales pueden divergir mucho entre corridas porque el momento
exacto de liberacion cambia la trayectoria siguiente (efecto mariposa),
pero el TOTAL de tiempo atascado por episodio es una metrica mas estable
para juzgar si ayuda en conjunto, con mas muestra.

Uso:
    python3 diagnose_stall_time_total.py --model models/social3d_v15_full_final.json --episodes 150
"""
import argparse
import json

import numpy as np
import pybullet as p

from foraging_env3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate3d import world_for_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)

    stalled_ticks_per_ep = {0: [], 1: []}
    total_ticks_per_ep = {0: [], 1: []}
    success_flags = {0: [], 1: []}

    for ep_idx in range(args.episodes):
        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx}/{args.episodes}...", flush=True)

        world_name = world_for_episode(ep_idx)
        seed_a, seed_b = int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1))
        env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
        ctrl_a.reset(); ctrl_b.reset()

        signals = [0, 0]
        at = [False, False]
        stalled_count = [0, 0]
        tick_count = [0, 0]

        for step_idx in range(args.max_steps):
            out = env.step(controllers, signals)
            for i in range(2):
                if at[i]:
                    continue
                if out[i]["success"]:
                    at[i] = True
                    continue
                tick_count[i] += 1
                if out[i]["stalled"]:
                    stalled_count[i] += 1

            if all(at):
                break

        for i in range(2):
            stalled_ticks_per_ep[i].append(stalled_count[i])
            total_ticks_per_ep[i].append(max(tick_count[i], 1))
            success_flags[i].append(at[i])

    print(f"\n=== Tiempo total atascado por episodio: {args.model} ({args.episodes} episodios) ===\n")
    for i, nombre in [(0, "A"), (1, "B")]:
        stalled = np.array(stalled_ticks_per_ep[i])
        total = np.array(total_ticks_per_ep[i])
        pct = stalled / total * 100
        success_rate = np.mean(success_flags[i]) * 100
        print(f"--- Agente {nombre} ---")
        print(f"% de ticks atascado por episodio (mientras navegaba, no post-exito):")
        print(f"  mediana={np.median(pct):.1f}%, promedio={np.mean(pct):.1f}%, "
              f"p75={np.percentile(pct,75):.1f}%, max={np.max(pct):.1f}%")
        print(f"Tasa de exito en esta muestra: {success_rate:.1f}% ({sum(success_flags[i])}/{args.episodes})")
        print()

    env.close()


if __name__ == "__main__":
    main()
