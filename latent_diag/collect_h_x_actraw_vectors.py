"""
collect_h_x_actraw_vectors.py

Version del Paper 3 que guarda, ademas de h (16) y x (11), la decision
CRUDA de la red (act_id_raw) -- la salida de GRUNetwork.forward(x)
ANTES de que el "instinto de auxilio" (Ganglio Basal, controllers.py
lineas 300-349) la pueda pisar cuando el agente lleva mucho tiempo
atascado.

Por que hace falta: la columna 'action' que ya se guardaba en
collect_h_x_vectors.py es la decision FINAL (despues del atropello del
instinto), que durante el atasco esta dominada por ese mecanismo
(hasta 95% de probabilidad de forzar señalizar/wander cuando la racha
se alarga) -- no sirve para probar si la red en si misma tiene un
sesgo hacia giro_der. act_id_raw si sirve, porque es la eleccion de la
red sin ninguna intervencion externa.

Como se captura: GRUNetwork.forward() ya calcula exactamente
act_id_raw como su valor de retorno (controllers.py linea 154:
`return np.argmax(logits)`), asi que alcanza con instrumentar esa
clase para que se acuerde de su propio ultimo resultado. No hace
falta tocar nada mas.

No se modifica controllers.py ni foraging_env3d.py (codigo congelado
del Paper 2).

Uso (empezar CHICO antes de escalar, igual que siempre):
    python3 collect_h_x_actraw_vectors.py --model models/social3d_v15_full_final.json --episodes 30 --out h_x_actraw_pilot.csv
    python3 collect_h_x_actraw_vectors.py --model models/social3d_v15_full_final.json --episodes 300 --out h_x_actraw_300.csv
"""
import argparse
import csv
import json

import numpy as np
import pybullet as p

from foraging_env_3d import Foraging3DEnv
from controllers import Unified3DController, GRUNetwork
from evaluate_3d import world_for_episode


class InstrumentedController(Unified3DController):
    """Igual que en collect_h_x_vectors.py: guarda el ultimo x usado."""
    def get_action(self, *args, **kwargs):
        v_l, v_r, signal, x = super().get_action(*args, **kwargs)
        self.last_x = x.copy()
        return v_l, v_r, signal, x


class GRUNetworkInstrumented(GRUNetwork):
    """
    Identica a GRUNetwork -- unica diferencia: se acuerda de su propio
    ultimo resultado (que ES act_id_raw, ver comentario del modulo).
    """
    def forward(self, x):
        accion_cruda = super().forward(x)
        self.last_raw_action = accion_cruda
        return accion_cruda


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
    # Truco: cambiar la clase de la instancia YA CREADA de self.brain,
    # sin reconstruirla -- conserva los pesos y el estado h intactos,
    # solo le agrega la memoria de su ultimo resultado.
    ctrl_a.brain.__class__ = GRUNetworkInstrumented
    ctrl_b.brain.__class__ = GRUNetworkInstrumented
    controllers = [ctrl_a, ctrl_b]

    rng = np.random.RandomState(args.seed)

    fieldnames = (
        ["episode", "agent", "decision", "stalled", "heard_signal",
         "near_wall", "own_success", "action", "act_id_raw"]
        + [f"h_{k}" for k in range(16)]
        + [f"x_{k}" for k in range(11)]
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
            ctrl_a.last_x = np.zeros(11)
            ctrl_b.last_x = np.zeros(11)
            ctrl_a.brain.last_raw_action = 0
            ctrl_b.brain.last_raw_action = 0

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
                    act_raw = int(controllers[i].brain.last_raw_action)

                    row = {
                        "episode": ep_idx,
                        "agent": "a" if i == 0 else "b",
                        "decision": dec,
                        "stalled": out[i]["stalled"],
                        "heard_signal": heard,
                        "near_wall": near_wall,
                        "own_success": at[i],
                        "action": out[i]["signal"],
                        "act_id_raw": act_raw,
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
