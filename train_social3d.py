"""
Driver de evolucion (Fase 1b) para 3D. Adaptado de train_social2d.py (2D).

Usa evaluate_individual_3d (evaluate3d.py) y Foraging3DEnv
(foraging_env3d.py). Arranca la poblacion desde nav_brain.json (Fase 1a)
como semilla, en vez de pesos aleatorios.

NOVEDAD (sesion 2026-08-07, tras ver oscilacion fuerte en BEST/AVG con
1 sola evaluacion por individuo por generacion): se agrega --eval_repeats,
que evalua cada individuo sobre N tandas de 9 episodios (semillas distintas
por tanda, mismas para TODOS los individuos de esa generacion) y promedia,
para que el ranking dependa de habilidad real y no de que le haya tocado
una tanda de escenarios facil o dificil.

Uso:
    python3 train_social3d.py --load nav_brain.json --gen 100 --pop 24 \
        --eval_repeats 3 --name social3d_v1
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *a, **k):
        return iterable

from foraging_env3d import Foraging3DEnv
from evaluate3d import evaluate_individual_3d

IN_S, OUT_S, H = 11, 5, 16


def get_weight_size():
    single = ((IN_S + H) * 3 * H) + (3 * H) + (H * OUT_S) + OUT_S + (IN_S * OUT_S)
    return single * 2


def weights_to_json(flat_weights):
    half = len(flat_weights) // 2

    def get_brain(w):
        idx, d, ws = 0, {}, (IN_S + H) * 3 * H
        d['w_gru'] = w[idx:idx + ws].reshape(IN_S + H, 3 * H).tolist(); idx += ws
        d['b_gru'] = w[idx:idx + 3 * H].tolist(); idx += 3 * H
        d['w_out'] = w[idx:idx + H * OUT_S].reshape(H, OUT_S).tolist(); idx += H * OUT_S
        d['b_out'] = w[idx:idx + OUT_S].tolist(); idx += OUT_S
        d['w_res'] = w[idx:idx + IN_S * OUT_S].reshape(IN_S, OUT_S).tolist()
        return d

    return {"brain_a": get_brain(flat_weights[:half]), "brain_b": get_brain(flat_weights[half:])}


def json_to_flat(data):
    def flatten(d):
        w = []
        for k in ['w_gru', 'b_gru', 'w_out', 'b_out', 'w_res']:
            w.extend(np.array(d[k]).flatten())
        return np.array(w)

    if 'brain_a' in data:
        return np.concatenate([flatten(data['brain_a']), flatten(data['brain_b'])])
    single = flatten(data)
    return np.concatenate([single, single])


def save_best_model(flat_weights, filename):
    data = weights_to_json(flat_weights)
    data['type'] = 'gru'
    data['agents'] = 2
    with open(filename, 'w') as f:
        json.dump(data, f)


_WORKER_ENV = None


def _init_worker(brain_ratio, num_threads):
    global _WORKER_ENV
    _WORKER_ENV = Foraging3DEnv(world_name="random_obstacles",
                                 num_threads=num_threads, brain_ratio=brain_ratio)


def _eval_worker(args):
    flat_weights, h_prob, use_bg, max_steps, episode_seeds_list = args
    global _WORKER_ENV
    weights_json = weights_to_json(flat_weights)

    fs, vocals, stucks = [], [], []
    # DIAGNOSTICO sesion 2026-08-11: alternar swap_slots entre tandas de
    # eval_repeats, para que cada individuo se evalue por igual en los dos
    # slots fisicos -- ver comentario largo en evaluate_individual_3d
    # (evaluate3d.py). Con eval_repeats impar (default 3) no queda un
    # 50/50 exacto (2 tandas normales, 1 swapped) -- usar eval_repeats par
    # si se quiere balance exacto.
    for r, episode_seeds in enumerate(episode_seeds_list):
        f, v, s, _succ = evaluate_individual_3d(
            weights_json, h_prob, use_bg, max_steps=max_steps, episodes=9,
            env=_WORKER_ENV, episode_seeds=episode_seeds, swap_slots=(r % 2 == 1)
        )
        fs.append(f); vocals.append(v); stucks.append(s)

    return (float(np.mean(fs)), float(np.mean(vocals)), float(np.mean(stucks)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--pop", type=int, default=24)
    parser.add_argument("--load", type=str, default="nav_brain.json")
    parser.add_argument("--name", type=str, default="social3d_v1")
    parser.add_argument("--no_bg", action="store_true")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--brain_ratio", type=int, default=15)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--eval_repeats", type=int, default=3,
                         help="Tandas de 9 episodios por individuo, promediadas, para bajar el ruido de seleccion")
    parser.add_argument("--save_last_n", type=int, default=15,
                         help="Guarda checkpoint cada generacion durante las ultimas N generaciones (ademas de cada 10)")
    args = parser.parse_args()

    use_bg = not args.no_bg
    w_size = get_weight_size()

    if args.load and os.path.exists(args.load):
        print(f"[LOAD] Semilla de Fase 1a: {args.load}")
        with open(args.load) as f:
            base_data = json.load(f)
        base = json_to_flat(base_data)
        population = [base + np.random.randn(w_size) * 0.03 for _ in range(args.pop)]
        population[0] = base
    else:
        print("[LOAD] Sin semilla, arrancando de pesos aleatorios")
        population = [np.random.randn(w_size) * 0.5 for _ in range(args.pop)]

    for d in ["models", "logs"]:
        os.makedirs(d, exist_ok=True)

    log_path = f"logs/{args.name}_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("gen,best,avg,screams,stuck,time\n")

    n_proc = args.processes or mp.cpu_count()
    print(f"[POOL] {n_proc} procesos, brain_ratio={args.brain_ratio}, "
          f"max_steps={args.max_steps}, pop={args.pop}, gen={args.gen}, "
          f"eval_repeats={args.eval_repeats}")

    with mp.Pool(processes=n_proc, initializer=_init_worker,
                 initargs=(args.brain_ratio, args.num_threads)) as pool:
        for gen in range(args.gen):
            t_s = time.time()
            h_prob = max(0.0, 0.3 * (1.0 - gen / 80.0))

            # R tandas de 9 semillas cada una, TODAS fijas para esta
            # generacion (mismas para los args.pop individuos), pero
            # distintas entre si (R examenes distintos) y distintas de
            # generacion en generacion.
            gen_rng = np.random.RandomState(seed=gen)
            episode_seeds_list = []
            for _r in range(args.eval_repeats):
                seeds = [(int(gen_rng.randint(0, 2**31 - 1)),
                          int(gen_rng.randint(0, 2**31 - 1))) for _ in range(9)]
                episode_seeds_list.append(seeds)

            tasks = [(ind, h_prob, use_bg, args.max_steps, episode_seeds_list) for ind in population]
            results = list(tqdm(
                pool.imap(_eval_worker, tasks),
                total=len(tasks), desc=f"gen {gen:03d}/{args.gen}", leave=False, unit="ind"
            ))

            fits = [r[0] for r in results]
            screams = [r[1] for r in results]
            stucks = [r[2] for r in results]

            idx = np.argsort(fits)[::-1]
            best_f, avg_f, avg_s, avg_stuck = fits[idx[0]], np.mean(fits), np.mean(screams), np.mean(stucks)
            population = [population[i] for i in idx]

            new_pop = population[:10] if args.pop >= 10 else population[:max(2, args.pop // 2)]
            elite_pool = min(25, len(population))
            while len(new_pop) < args.pop:
                p1 = population[np.random.randint(0, elite_pool)]
                p2 = population[np.random.randint(0, elite_pool)]
                child = np.where(np.random.rand(w_size) > 0.5, p1, p2)
                if np.random.rand() < 0.2:
                    child = child + np.random.randn(w_size) * 0.12
                new_pop.append(child)
            population = new_pop

            dt = time.time() - t_s
            print("GEN {} (h:{:.2f}) | BEST: {:12.0f} | AVG: {:12.0f} | SCREAMS: {:5.1f} | STUCK: {:5.1f} | {:.1f}s".format(
                gen, h_prob, best_f, avg_f, avg_s, avg_stuck, dt))

            with open(log_path, "a") as f:
                f.write(f"{gen},{best_f:.2f},{avg_f:.2f},{avg_s:.2f},{avg_stuck:.2f},{dt:.2f}\n")

            save_every_10 = (gen % 10 == 0 or gen == args.gen - 1)
            save_final_stretch = (gen >= args.gen - args.save_last_n)
            if save_every_10 or save_final_stretch:
                save_best_model(population[0], f"models/{args.name}_gen{gen}.json")

    save_best_model(population[0], f"models/{args.name}_final.json")
    print("[DONE] Evolucion 3D finalizada.")


if __name__ == "__main__":
    main()
