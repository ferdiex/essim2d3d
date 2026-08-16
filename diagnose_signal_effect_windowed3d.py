"""
Diagnostico dirigido: escuchar la señal, ¿ayuda a acercarse al companero
y a desatorarse, comparado con no escucharla?

No corre pares de episodios (evita el problema de que las trayectorias
divergen apenas el comportamiento cambia). En cambio, corre episodios
NORMALES (señal real, tal como es) y compara, DENTRO del mismo rollout,
las decisiones donde el agente escucho señal (x[10]!=0 en ese instante)
contra las decisiones donde no escucho nada -- usando exactamente la
misma metrica que ya usa el fitness para SOCIAL_RESCUE_BONUS
(acercamiento_companero - acercamiento_comida, evaluate3d.py linea
204-222), y ademas una metrica de desatoro (stalled antes -> no stalled
despues).

Responde directo lo que se observo visualmente: "el blanco parece ir
hacia la comida cuando el gris señala, y a veces lo desatora" -- ¿esto
se sostiene estadisticamente dentro de episodios reales, o es lo que ya
sabiamos (ruido, no se nota en el agregado de --mute_social)?

Uso:
    python3 diagnose_signal_effect_windowed3d.py --model models/social3d_v14_bonus_boost_pilot_final.json --episodes 40
"""
import argparse
import json
import random

import numpy as np
import pybullet as p

from foraging_env3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate3d import world_for_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # Buckets: decisiones con señal escuchada (heard) vs sin señal (silent)
    closing_heard = []   # acercamiento_companero - acercamiento_comida
    closing_silent = []

    stuck_before_heard_total = 0
    stuck_before_heard_resolved = 0     # stalled antes, NO stalled despues
    stuck_before_silent_total = 0
    stuck_before_silent_resolved = 0

    for ep_idx in range(args.episodes):
        if ep_idx % 10 == 0:
            print(f"  ep {ep_idx}/{args.episodes}... "
                  f"(heard={len(closing_heard)}, silent={len(closing_silent)}, "
                  f"stuck+heard={stuck_before_heard_total}, stuck+silent={stuck_before_silent_total})",
                  flush=True)

        world_name = world_for_episode(ep_idx)
        seed_a, seed_b = int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1))
        env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
        ctrl_a.reset(); ctrl_b.reset()

        signals = [0, 0]
        at = [False, False]
        num_decisions = args.max_steps // env.brain_ratio

        p_pre = [np.array(p.getBasePositionAndOrientation(env.robots[i])[0][:2]) for i in range(2)]
        d_food_pre = [float(np.linalg.norm(p_pre[i] - np.array(env.target_pos[:2]))) for i in range(2)]

        for dec in range(num_decisions):
            stalled_before = list(env.stalled)  # copia, antes de esta decision
            other_signal_before = [signals[1 - i] for i in range(2)]  # lo que va a "escuchar" cada uno esta decision

            out = None
            for _tick in range(env.brain_ratio):
                out = env.step(controllers, signals)

            for i in range(2):
                if out[i]["success"]:
                    at[i] = True

            cur_pos = [np.array(out[i]["pos"][:2]) for i in range(2)]
            d_food_cur = [float(np.linalg.norm(cur_pos[i] - np.array(env.target_pos[:2]))) for i in range(2)]

            for i in range(2):
                other = 1 - i
                heard = (other_signal_before[i] == 4)  # el otro estaba señalizando ANTES de esta decision

                # metrica de acercamiento (solo tiene sentido si el otro ya llego,
                # que es cuando puede estar señalizando de verdad con proposito
                # de "recluta" -- mismo condicional que usa el fitness)
                if at[other]:
                    acercamiento_comida = d_food_pre[i] - d_food_cur[i]
                    d_soc_pre = float(np.linalg.norm(p_pre[i] - cur_pos[other]))
                    d_soc_post = float(np.linalg.norm(cur_pos[i] - cur_pos[other]))
                    acercamiento_companero = d_soc_pre - d_soc_post
                    closing_delta = acercamiento_companero - acercamiento_comida
                    if heard:
                        closing_heard.append(closing_delta)
                    else:
                        closing_silent.append(closing_delta)

                # metrica de desatoro: estaba stalled antes de esta decision,
                # ¿sigue stalled despues?
                if stalled_before[i]:
                    resolved = not out[i]["stalled"]
                    if heard:
                        stuck_before_heard_total += 1
                        if resolved:
                            stuck_before_heard_resolved += 1
                    else:
                        stuck_before_silent_total += 1
                        if resolved:
                            stuck_before_silent_resolved += 1

            p_pre = cur_pos
            d_food_pre = d_food_cur

            if all(at):
                break

    print(f"=== Efecto de escuchar la señal, dentro de episodios reales: {args.model} ===")
    print(f"({args.episodes} episodios)\n")

    print("--- Acercamiento al companero (solo cuando el companero ya llego) ---")
    print(f"Decisiones con señal escuchada:  n={len(closing_heard)}, "
          f"promedio={np.mean(closing_heard) if closing_heard else float('nan'):+.4f}, "
          f"std={np.std(closing_heard) if closing_heard else float('nan'):.4f}")
    print(f"Decisiones SIN señal (silencio): n={len(closing_silent)}, "
          f"promedio={np.mean(closing_silent) if closing_silent else float('nan'):+.4f}, "
          f"std={np.std(closing_silent) if closing_silent else float('nan'):.4f}")
    if closing_heard and closing_silent:
        diff = np.mean(closing_heard) - np.mean(closing_silent)
        # error estandar de la diferencia de medias (aprox, muestras independientes)
        se = np.sqrt(np.var(closing_heard)/len(closing_heard) + np.var(closing_silent)/len(closing_silent))
        print(f"\nDiferencia (heard - silent): {diff:+.4f}  (error estandar ~{se:.4f})")
        if se > 0 and abs(diff) > 2 * se:
            print(">>> Diferencia mayor a 2x su propio error estandar -- consistente con efecto real, "
                  "no ruido de muestreo.")
        else:
            print(">>> Diferencia chica frente a su propio error estandar -- no se puede distinguir de ruido "
                  "con este tamaño de muestra.")

    print("\n--- Desatoro (estaba stalled, ¿se resolvio en la misma decision?) ---")
    if stuck_before_heard_total > 0:
        rate_heard = stuck_before_heard_resolved / stuck_before_heard_total
        print(f"Con señal escuchada:  {stuck_before_heard_resolved}/{stuck_before_heard_total} "
              f"se resolvieron ({rate_heard:.1%})")
    else:
        print("Con señal escuchada: sin datos (0 casos de stuck+heard en esta muestra)")
    if stuck_before_silent_total > 0:
        rate_silent = stuck_before_silent_resolved / stuck_before_silent_total
        print(f"Sin señal (silencio): {stuck_before_silent_resolved}/{stuck_before_silent_total} "
              f"se resolvieron ({rate_silent:.1%})")
    else:
        print("Sin señal (silencio): sin datos (0 casos de stuck+silent en esta muestra)")

    env.close()


if __name__ == "__main__":
    main()
