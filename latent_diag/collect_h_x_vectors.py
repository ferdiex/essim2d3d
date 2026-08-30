"""
collect_h_x_vectors.py

Version modificada de collect_h_vectors.py (Paper 3) que ADEMAS de h
(16 dims) guarda x (11 dims, la observacion real que ve la red en cada
decision). Necesario para calcular x_fijo (paso 4 de Sussillo-Barak):
las CSV anteriores (h_vectors_pilot.csv, etc.) no guardaban x, solo h
y algunas etiquetas de contexto -- no alcanza para reconstruir "que
estaba viendo el robot" en cada atasco.

No se toca collect_h_vectors.py NI foraging_env3d.py (el codigo
congelado del Paper 2 sigue intacto). En vez de eso, se agrega una
subclase chica de Unified3DController que SOLO intercepta y guarda el
`x` que ya se calculaba y se descartaba (foraging_env3d.py linea 300:
`v_l, v_r, signal, _x = controllers[i].get_action(...)` -- el `_x` se
tiraba). La subclase llama exactamente al metodo original (super()) y
no cambia ningun comportamiento, solo agrega una linea para recordar
el ultimo x usado.

Columnas nuevas respecto al original: x_0 .. x_10 (la misma
observacion que usamos en gru_dynamics.py / jacobian.py /
find_fixed_points.py).

Usage (empezar CHICO antes de escalar, igual que el original):
    python3 collect_h_x_vectors.py --model models/social3d_v15_full_final.json --episodes 30 --out h_x_vectors_pilot.csv
    # si el pilot se ve bien:
    python3 collect_h_x_vectors.py --model models/social3d_v15_full_final.json --episodes 300 --out h_x_vectors_300.csv
"""
import argparse
import csv
import json

import numpy as np
import pybullet as p

from foraging_env_3d import Foraging3DEnv
from controllers import Unified3DController
from evaluate_3d import world_for_episode


class InstrumentedController(Unified3DController):
    """
    Identico a Unified3DController -- unica diferencia: guarda el
    ultimo `x` usado en self.last_x, para poder leerlo desde afuera
    despues de cada paso de decision (foraging_env3d.py lo calcula
    pero lo descarta, ver comentario del modulo).
    """
    def get_action(self, *args, **kwargs):
        v_l, v_r, signal, x = super().get_action(*args, **kwargs)
        self.last_x = x.copy()
        return v_l, v_r, signal, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output CSV file")
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    ctrl_a = InstrumentedController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = InstrumentedController(env.cfg, weights, agent_type="brain_b")
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)

    fieldnames = (
        ["episode", "agent", "decision", "stalled", "heard_signal",
         "near_wall", "own_success", "action"]
        + [f"h_{k}" for k in range(16)]
        + [f"x_{k}" for k in range(11)]   # <-- NUEVO respecto al original
    )

    n_rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep_idx in range(args.episodes):
            if ep_idx % 10 == 0:
                print(f"  ep {ep_idx}/{args.episodes}... ({n_rows} rows so far)", flush=True)

            world_name = world_for_episode(ep_idx)
            seed_a, seed_b = int(rng.randint(0, 2**31 - 1)), int(rng.randint(0, 2**31 - 1))
            env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
            ctrl_a.reset(); ctrl_b.reset()
            ctrl_a.last_x = np.zeros(11)  # por si el primer tick de un
            ctrl_b.last_x = np.zeros(11)  # episodio no fuera decide_now

            signals = [0, 0]
            at = [False, False]
            num_decisions = args.max_steps // env.brain_ratio

            for dec in range(num_decisions):
                sensors_pre = [env._get_sensors(env.robots[i]) for i in range(2)]
                other_signal_pre = [signals[1 - i] for i in range(2)]

                out = None
                for _tick in range(env.brain_ratio):
                    out = env.step(controllers, signals)

                for i in range(2):
                    if out[i]["success"]:
                        at[i] = True

                for i in range(2):
                    near_wall = bool(np.max(sensors_pre[i]) > 0.82)
                    heard = bool(other_signal_pre[i] == 4)
                    h_vec = controllers[i].brain.h.copy()
                    x_vec = controllers[i].last_x

                    row = {
                        "episode": ep_idx,
                        "agent": "a" if i == 0 else "b",
                        "decision": dec,
                        "stalled": out[i]["stalled"],
                        "heard_signal": heard,
                        "near_wall": near_wall,
                        "own_success": at[i],
                        "action": out[i]["signal"],
                    }
                    for k in range(16):
                        row[f"h_{k}"] = float(h_vec[k])
                    for k in range(11):
                        row[f"x_{k}"] = float(x_vec[k])
                    writer.writerow(row)
                    n_rows += 1

                if all(at):
                    break

    print(f"\nDone. {n_rows} rows saved to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
