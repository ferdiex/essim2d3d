"""
Prueba de orden de procesamiento en 2D -- analogo a --reverse_order en
3D (test_physics_gru.py), pero para el motor de 2D (foraging_env.py +
GRUController via train_social2d.py).

ADVERTENCIA IMPORTANTE, distinta a 3D: en 2D, los sorteos de la capa de
instintos (recluta 70%, auxilio 20%, annealing) usan np.random.rand()
-- un generador aleatorio GLOBAL COMPARTIDO, no uno propio por agente
(a diferencia de 3D, donde Unified3DController tiene su propio rng).
Esto significa que invertir el orden de procesamiento tambien cambia
QUE numeros aleatorios le tocan a cada agente -- hay una fuente de
ruido extra mezclada con cualquier sesgo sistematico real. Se usan
muchos episodios para promediar ese ruido.

Uso:
    python3 test2d_reverse_order.py --model models/social2d_emergence.json --episodes 300
"""
import argparse

import numpy as np

from foraging_env2d import ForagingEnv, ForagingEnvConfig
from controllers import GRUController


def run_episode(env, c_a, c_b, max_steps, reverse_order):
    obs, info = env.reset()
    c_a.reset(); c_b.reset()
    at = info['individual_success']
    p_pre = [pp.copy() for pp in env.agent_pos]

    for step in range(max_steps):
        if reverse_order:
            raw_b = c_b.act(obs[1])
            raw_a = c_a.act(obs[0])
        else:
            raw_a = c_a.act(obs[0])
            raw_b = c_b.act(obs[1])
        actions_raw = [raw_a, raw_b]

        actions = [None, None]
        order = [1, 0] if reverse_order else [0, 1]
        for i in order:
            dist_moved = np.linalg.norm(env.agent_pos[i] - p_pre[i])
            if at[i] and np.random.rand() < 0.7:
                actions[i] = 4
            elif actions_raw[i] in [1, 2] and dist_moved < 0.5 and np.random.rand() < 0.2:
                actions[i] = 4
            else:
                actions[i] = actions_raw[i]

        for i in range(2):
            p_pre[i] = env.agent_pos[i].copy()

        obs, _, term, trun, info = env.step(actions)
        at = info['individual_success']
        if term or trun:
            break

    return at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for reverse in [False, True]:
        env = ForagingEnv(ForagingEnvConfig(num_agents=2))
        c_a = GRUController(model_path=args.model, num_agents=2, agent_idx=0)
        c_b = GRUController(model_path=args.model, num_agents=2, agent_idx=1)

        np.random.seed(args.seed)
        success_a, success_b = 0, 0

        for ep in range(args.episodes):
            at = run_episode(env, c_a, c_b, args.max_steps, reverse_order=reverse)
            if at[0]:
                success_a += 1
            if at[1]:
                success_b += 1

        label = "REVERSE_ORDER (B antes que A)" if reverse else "orden normal (A antes que B)"
        print(f"=== {label} ===")
        print(f"A: {success_a}/{args.episodes} ({success_a/args.episodes:.1%})")
        print(f"B: {success_b}/{args.episodes} ({success_b/args.episodes:.1%})\n")


if __name__ == "__main__":
    main()
