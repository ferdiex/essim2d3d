"""
Version 2D de diagnose_signal_effect_windowed3d.py -- misma pregunta,
mismo tipo de medicion, pero contra el motor 2D (foraging_env.py +
GRUController de controllers.py). Corre rapido (sin fisica real, sin
pybullet) -- deberia tomar segundos, no minutos.

Pregunta: en 2D, ¿escuchar la señal del compañero (env.agent_signals[other]
> 0.5) se asocia con (a) acercarse mas al compañero de lo que ya se
acercaba a la comida, y/o (b) resolver un atasco (dist_moved bajo en la
decision anterior -> dist_moved alto en esta), comparado con no
escucharla? Mismo tipo de comparacion "dentro del mismo rollout" que se
uso en 3D, para poder comparar directamente los dos resultados.

Nota: el "atasco" en 2D no tiene histeresis persistente como en 3D
(no hay stall_counts/unstall_counts) -- se aproxima por decision,
usando el mismo umbral (dist_moved<1.1) que ya usa la propia fitness de
2D (train_social2d.py linea 139) para penalizar y contar "total_stuck".

Uso:
    python3 diagnose_signal_effect_windowed2d.py --model models/social2d_XXXXX.json --episodes 150
"""
import argparse
import json

import numpy as np

from foraging_env2d import ForagingEnv, ForagingEnvConfig
from controllers import GRUController

STUCK_THRESHOLD_2D = 1.1  # mismo umbral que usa train_social2d.py linea 139


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    env = ForagingEnv(ForagingEnvConfig(num_agents=2))
    c_a = GRUController(model_path=args.model, num_agents=2, agent_idx=0)
    c_b = GRUController(model_path=args.model, num_agents=2, agent_idx=1)
    controllers = [c_a, c_b]

    rng = np.random.RandomState(args.seed)

    closing_heard = []
    closing_silent = []

    stuck_before_heard_total = 0
    stuck_before_heard_resolved = 0
    stuck_before_silent_total = 0
    stuck_before_silent_resolved = 0

    for ep_idx in range(args.episodes):
        if ep_idx % 20 == 0:
            print(f"  ep {ep_idx}/{args.episodes}... "
                  f"(heard={len(closing_heard)}, silent={len(closing_silent)}, "
                  f"stuck+heard={stuck_before_heard_total}, stuck+silent={stuck_before_silent_total})",
                  flush=True)

        obs, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        c_a.reset(); c_b.reset()
        at = info['individual_success']
        p_pre = [pp.copy() for pp in env.agent_pos]
        dist_moved_prev = [0.0, 0.0]  # movimiento de la decision ANTERIOR (para saber si "venia atascado")

        for step in range(args.max_steps):
            d_pre = [np.linalg.norm(env.agent_pos[i] - env.food_pos) for i in range(2)]
            actions_raw = [c_a.act(obs[0]), c_b.act(obs[1])]

            actions = []
            for i in range(2):
                dist_moved = np.linalg.norm(env.agent_pos[i] - p_pre[i])
                if at[i] and rng.rand() < 0.7:
                    actions.append(4)
                elif actions_raw[i] in [1, 2] and dist_moved < 0.5 and rng.rand() < 0.2:
                    actions.append(4)
                else:
                    actions.append(actions_raw[i])

            heard_before = [env.agent_signals[1 - i] > 0.5 for i in range(2)]  # lo que escucha cada uno ESTA decision
            stuck_before = [dist_moved_prev[i] < STUCK_THRESHOLD_2D for i in range(2)]

            for i in range(2):
                p_pre[i] = env.agent_pos[i].copy()
            for i in range(2):
                env.agent_signals[i] = 1.0 if actions[i] == 4 else 0.0

            obs, _, term, trun, info = env.step(actions)
            at_new = info['individual_success']

            for i in range(2):
                other = 1 - i
                dist_moved = np.linalg.norm(env.agent_pos[i] - p_pre[i])
                dist_moved_prev[i] = dist_moved

                # acercamiento (solo si el companero ya llego -- mismo condicional que la fitness real)
                if at[other]:
                    acercamiento_comida = d_pre[i] - np.linalg.norm(env.agent_pos[i] - env.food_pos)
                    d_soc_pre = np.linalg.norm(p_pre[i] - env.agent_pos[other])
                    d_soc_post = np.linalg.norm(env.agent_pos[i] - env.agent_pos[other])
                    acercamiento_companero = d_soc_pre - d_soc_post
                    closing_delta = acercamiento_companero - acercamiento_comida
                    if heard_before[i]:
                        closing_heard.append(closing_delta)
                    else:
                        closing_silent.append(closing_delta)

                # desatoro
                if stuck_before[i]:
                    resolved = dist_moved >= STUCK_THRESHOLD_2D
                    if heard_before[i]:
                        stuck_before_heard_total += 1
                        if resolved:
                            stuck_before_heard_resolved += 1
                    else:
                        stuck_before_silent_total += 1
                        if resolved:
                            stuck_before_silent_resolved += 1

            at = at_new
            if term or trun:
                break

    print(f"\n=== Efecto de escuchar la señal (2D), dentro de episodios reales: {args.model} ===")
    print(f"({args.episodes} episodios)\n")

    print("--- Acercamiento al companero (solo cuando el companero ya llego) ---")
    print(f"Decisiones con señal escuchada:  n={len(closing_heard)}, "
          f"promedio={np.mean(closing_heard) if closing_heard else float('nan'):+.4f}")
    print(f"Decisiones SIN señal (silencio): n={len(closing_silent)}, "
          f"promedio={np.mean(closing_silent) if closing_silent else float('nan'):+.4f}")
    if closing_heard and closing_silent:
        diff = np.mean(closing_heard) - np.mean(closing_silent)
        se = np.sqrt(np.var(closing_heard)/len(closing_heard) + np.var(closing_silent)/len(closing_silent))
        print(f"Diferencia (heard - silent): {diff:+.4f} (error estandar ~{se:.4f})")
        print(">>> Efecto detectable" if se > 0 and abs(diff) > 2*se else ">>> No distinguible de ruido")

    print("\n--- Desatoro (dist_moved<1.1 en la decision anterior, ¿se resolvio esta decision?) ---")
    if stuck_before_heard_total > 0:
        r1 = stuck_before_heard_resolved / stuck_before_heard_total
        print(f"Con señal escuchada:  {stuck_before_heard_resolved}/{stuck_before_heard_total} ({r1:.1%})")
    if stuck_before_silent_total > 0:
        r2 = stuck_before_silent_resolved / stuck_before_silent_total
        print(f"Sin señal (silencio): {stuck_before_silent_resolved}/{stuck_before_silent_total} ({r2:.1%})")

    if stuck_before_heard_total > 0 and stuck_before_silent_total > 0:
        p1, n1 = stuck_before_heard_resolved, stuck_before_heard_total
        p2, n2 = stuck_before_silent_resolved, stuck_before_silent_total
        prop1, prop2 = p1/n1, p2/n2
        p_pool = (p1+p2)/(n1+n2)
        se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2)) if p_pool not in (0, 1) else 0
        if se > 0:
            z = (prop1-prop2)/se
            from scipy.stats import norm
            pval = 2*(1-norm.cdf(abs(z)))
            print(f"\nDiferencia: {(prop1-prop2)*100:+.1f}pp, z={z:.2f}, p-valor={pval:.4f}")


if __name__ == "__main__":
    main()
