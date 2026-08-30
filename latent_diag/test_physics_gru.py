"""
Prueba comparativa grippy vs slippery, SOLO para el GRU (Unified3DController).

No modifica essim3d.py, foraging_env3d.py ni controllers.py -- los importa
y usa tal cual estan. Braitenberg (FusionController) no interviene en nada
de este script.

Mide, para el MISMO controlador GRU bajo cada preset de fisica:
  - % de decisiones con self.stalled[i] == True (atasco total, ya existente
    en foraging_env3d.py)
  - Eventos de inclinacion fuerte (roll o pitch > TILT_THRESHOLD_DEG),
    proxy directo del "casi parado / trepando la pared" -- esto NO se
    loguea en ningun lado hoy, se agrega aca sin tocar el codigo fuente.
  - Tilt maximo observado por episodio.
  - Tasa de exito (llegar a la comida).

Uso:
    python3 test_physics_gru.py --model models/social_emergence.json \
        --episodes 5 --steps 2000

Correr TAL CUAL en tu maquina (necesita config_loader.py, world_builder.py,
los mundos y los pesos entrenados -- ninguno de esos esta en este sandbox).
"""
import argparse
import json
import math

import numpy as np
import pybullet as p

from foraging_env3d import Foraging3DEnv
from controllers import Unified3DController

TILT_THRESHOLD_DEG = 30.0  # ajustable: que tan inclinado cuenta como "evento"


FREE_SPAWN_THRESH = 0.08  # mismo umbral que success = dist < 0.08 en foraging_env3d.py


def run_episode(env, ctrl_a, ctrl_b, max_steps, seed_a, seed_b, mute_social=None, reverse_order=False):
    env.reset(seed_a=seed_a, seed_b=seed_b)
    ctrl_a.reset()
    ctrl_b.reset()
    signals = [0, 0]
    tilt_events = [0, 0]
    max_tilt = [0.0, 0.0]
    stalled_steps = [0, 0]

    # Distancia de spawn ANTES de cualquier decision -- si ya nace dentro
    # del radio de exito (FREE_SPAWN_THRESH), ese "exito" es azar de
    # posicion, no navegacion. Se marca para poder filtrarlo despues, sin
    # descartar el episodio (los otros datos -stalled, tilt- siguen siendo
    # utiles igual).
    initial_dist = [0.0, 0.0]
    for i, rid in enumerate(env.robots):
        pos0, _ori0 = p.getBasePositionAndOrientation(rid)
        initial_dist[i] = float(np.linalg.norm(np.array(pos0[:2]) - np.array(env.target_pos[:2])))
    free_spawn = [d < FREE_SPAWN_THRESH for d in initial_dist]

    # exito SOSTENIDO (una vez True, queda True), mismo criterio que
    # evaluate_3d.py ("at = [False, False] # individual_success acumulado").
    # out[i]["success"] es MOMENTANEO (se recalcula cada paso segun
    # distancia actual) -- si un robot empuja a otro fuera del radio de
    # exito despues de haber llegado, out[i]["success"] vuelve a False,
    # pero el logro ya ocurrio y debe contar igual.
    ever_success = [False, False]
    success_step = [-1, -1]
    out = None

    for step in range(max_steps):
        out = env.step([ctrl_a, ctrl_b], signals, mute_social=mute_social, reverse_order=reverse_order)
        for i in range(2):
            if out[i]["stalled"]:
                stalled_steps[i] += 1
            if out[i]["success"] and not ever_success[i]:
                ever_success[i] = True
                success_step[i] = step
            elif out[i]["success"]:
                ever_success[i] = True
            _pos, ori = p.getBasePositionAndOrientation(env.robots[i])
            roll, pitch, _yaw = p.getEulerFromQuaternion(ori)
            tilt_deg = max(abs(math.degrees(roll)), abs(math.degrees(pitch)))
            max_tilt[i] = max(max_tilt[i], tilt_deg)
            if tilt_deg > TILT_THRESHOLD_DEG:
                tilt_events[i] += 1
        if ever_success[0] and ever_success[1]:
            break

    n = step + 1
    return {
        "steps_run": n,
        "stalled_pct": [100.0 * s / n for s in stalled_steps],
        "tilt_events_pct": [100.0 * t / n for t in tilt_events],
        "max_tilt_deg": max_tilt,
        "success": ever_success,
        "success_step": success_step,
        "initial_dist": initial_dist,
        "free_spawn": free_spawn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path al JSON del GRU (brain_a/brain_b)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--world", default="random_obstacles")
    ap.add_argument("--physics", choices=["both", "grippy", "slippery"], default="both",
                     help="que fisica(s) correr. 'both' (default) corre las dos, como hasta ahora.")
    ap.add_argument("--both_a", action="store_true",
                     help="PARCHE QUIRURGICO (diagnostico, ver handoff de la sesion): "
                          "corre los pesos de brain_a en LOS DOS slots fisicos (A y B), "
                          "cada uno con su propia instancia/estado. Aisla si una diferencia "
                          "observada es de los PESOS (brain_a vs brain_b) o del SLOT/rol "
                          "fisico (spawn, orden de decision). Si el 'slot B' sigue "
                          "comportandose distinto al 'slot A' incluso con los mismos pesos, "
                          "es el slot, no los pesos.")
    ap.add_argument("--both_b", action="store_true",
                     help="Mismo parche quirurgico pero al reves: corre los pesos de "
                          "brain_b en LOS DOS slots fisicos (A y B). Mutuamente excluyente "
                          "con --both_a.")
    ap.add_argument("--swap_creation_order", action="store_true",
                     help="DIAGNOSTICO (sesion 2026-08-11): invierte cual cuerpo se crea "
                          "PRIMERO en PyBullet (robot_id_b antes que robot_id_a), sin tocar "
                          "el mapeo self.robots=[robot_id_a, robot_id_b]. Aisla si la ventaja "
                          "de exito del slot B viene del orden de creacion de cuerpos "
                          "(posible sesgo del solver de colisiones por indice) o se queda "
                          "pegada al slot igual.")
    ap.add_argument("--reverse_order", action="store_true",
                     help="DIAGNOSTICO (sesion 2026-08-14): invierte el ORDEN DE PROCESAMIENTO "
                          "del loop de decision (agente B procesado antes que A en cada paso), "
                          "sin tocar el mapeo self.robots ni cual controlador corresponde a cual "
                          "slot. Complementa a --swap_creation_order (esa aisla el orden de "
                          "creacion de cuerpos en pybullet, esta aisla el orden de computo del "
                          "codigo). Pendiente del handoff original, nunca antes medible con este "
                          "harness -- solo existia en essim3d.py (visual, no cuantitativo).")
    ap.add_argument("--stall_activation_threshold", type=int, default=25,
                     help="Paper 3 (espacio latente): umbral de ticks consecutivos sin desplazamiento "
                          "efectivo para activar stalled=True. Default=25, el mismo valor heredado sin "
                          "recalibracion usado en el Paper 2 -- pasar otro valor aqui explora la segunda "
                          "linea de trabajo futuro anunciada en el Paper 2 (revisar este criterio), sin "
                          "afectar la reproducibilidad de los resultados ya publicados (que usan el default).")
    ap.add_argument("--mute_social", action="store_true",
                     help="Apaga el canal social (mute_social=[True,True] en env.step()) para "
                          "comparar contra la señal activa (default). Sirve para medir si la "
                          "comunicacion realmente aporta algo, o si el desempeño es igual con "
                          "el canal mudo.")
    args = ap.parse_args()

    if args.both_a and args.both_b:
        raise SystemExit("--both_a y --both_b son mutuamente excluyentes, usa uno solo")

    with open(args.model) as f:
        weights = json.load(f)

    physics_list = ["grippy", "slippery"] if args.physics == "both" else [args.physics]

    for physics in physics_list:
        label = ""
        if args.both_a:
            label = "  [both_a: brain_a en los dos slots]"
        elif args.both_b:
            label = "  [both_b: brain_b en los dos slots]"
        if args.swap_creation_order:
            label += "  [swap_creation_order: robot_id_b creado primero]"
        if args.mute_social:
            label += "  [MUTE_SOCIAL: canal social apagado]"
        if args.stall_activation_threshold != 25:
            label += f"  [stall_activation_threshold={args.stall_activation_threshold} (default Paper 2: 25)]"
        print(f"\n=== physics={physics}{label} ===")
        env = Foraging3DEnv(world_name=args.world, physics=physics,
                             swap_creation_order=args.swap_creation_order,
                             stall_activation_threshold=args.stall_activation_threshold)
        if args.both_a:
            agent_type_a, agent_type_b = "brain_a", "brain_a"
        elif args.both_b:
            agent_type_a, agent_type_b = "brain_b", "brain_b"
        else:
            agent_type_a, agent_type_b = "brain_a", "brain_b"
        ctrl_a = Unified3DController(env.cfg, weights, agent_type=agent_type_a)
        ctrl_b = Unified3DController(env.cfg, weights, agent_type=agent_type_b)

        agg_stalled = [[], []]
        agg_tilt_events = [[], []]
        agg_max_tilt = [[], []]
        successes = [0, 0]
        free_spawn_count = [0, 0]
        successes_excl_free = [0, 0]  # exitos SIN contar los free_spawn en el numerador
        non_free_count = [0, 0]       # denominador correcto para la tasa filtrada
        first_arrival = {"slot_0": 0, "slot_1": 0, "tie": 0}  # solo cuenta cuando AMBOS llegan

        for ep in range(args.episodes):
            mute_arg = [True, True] if args.mute_social else None
            res = run_episode(env, ctrl_a, ctrl_b, args.steps, seed_a=ep * 2, seed_b=ep * 2 + 1, mute_social=mute_arg, reverse_order=args.reverse_order)
            for i in range(2):
                agg_stalled[i].append(res["stalled_pct"][i])
                agg_tilt_events[i].append(res["tilt_events_pct"][i])
                agg_max_tilt[i].append(res["max_tilt_deg"][i])
                if res["success"][i]:
                    successes[i] += 1
                if res["free_spawn"][i]:
                    free_spawn_count[i] += 1
                else:
                    non_free_count[i] += 1
                    if res["success"][i]:
                        successes_excl_free[i] += 1
            if res["success"][0] and res["success"][1]:
                s0, s1 = res["success_step"]
                if s0 < s1:
                    first_arrival["slot_0"] += 1
                elif s1 < s0:
                    first_arrival["slot_1"] += 1
                else:
                    first_arrival["tie"] += 1
            free_tag = ""
            if any(res["free_spawn"]):
                free_tag = f" FREE_SPAWN={res['free_spawn']} initial_dist={[f'{d:.3f}' for d in res['initial_dist']]}"
            print(f"  ep {ep}: steps={res['steps_run']} "
                  f"stalled%={[f'{v:.1f}' for v in res['stalled_pct']]} "
                  f"tilt_events%={[f'{v:.1f}' for v in res['tilt_events_pct']]} "
                  f"max_tilt_deg={[f'{v:.1f}' for v in res['max_tilt_deg']]} "
                  f"success={res['success']} success_step={res['success_step']}{free_tag}")

        env.close()

        print(f"  --- resumen physics={physics} ---")
        for i in range(2):
            print(f"  robot_{i}: stalled_avg={np.mean(agg_stalled[i]):.1f}% "
                  f"tilt_events_avg={np.mean(agg_tilt_events[i]):.1f}% "
                  f"max_tilt_avg={np.mean(agg_max_tilt[i]):.1f}deg "
                  f"exitos_crudos={successes[i]}/{args.episodes}")
            filt_pct = (100.0 * successes_excl_free[i] / non_free_count[i]) if non_free_count[i] > 0 else float('nan')
            print(f"            spawns_regalados(<{FREE_SPAWN_THRESH}m)={free_spawn_count[i]}/{args.episodes} "
                  f"| exitos_FILTRADOS (excluye spawns regalados)={successes_excl_free[i]}/{non_free_count[i]} "
                  f"({filt_pct:.1f}%)")
        n_both = first_arrival["slot_0"] + first_arrival["slot_1"] + first_arrival["tie"]
        print(f"  quien llega primero (solo episodios donde AMBOS llegaron, n={n_both}): "
              f"slot_0={first_arrival['slot_0']} slot_1={first_arrival['slot_1']} "
              f"empate={first_arrival['tie']}")


if __name__ == "__main__":
    main()
