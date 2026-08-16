# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import time
import csv
import json
import numpy as np
import imageio
import random
import pygame
from foraging_env2d import ForagingEnv, ForagingEnvConfig
from controllers import make_controller
from pathlib import Path

class DummyController:
    def reset(self):
        pass

WF_ESCAPE_STEPS = 0
WF_ESCAPE_TURN = 2

def detect_model_type(model_path):
    """Lee el JSON del modelo y devuelve el tipo que el propio archivo declara
    (campo 'type' o 'encoding'), normalizado a los choices de --type.
    Devuelve None si no se pudo leer o no declara tipo."""
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    # Algunos modelos duales (brain_a/brain_b) guardan el tipo dentro,
    # otros lo guardan al nivel raiz (como el braitenberg que trae 'encoding').
    probe = data.get('brain_a', data) if isinstance(data, dict) else data
    declared = None
    if isinstance(probe, dict):
        declared = probe.get('type') or data.get('type') or data.get('encoding')

    if not declared:
        return None

    declared = str(declared).lower()
    if declared == "residual":
        declared = "res"
    return declared


def get_version():
    try:
        return Path(__file__).resolve().with_name("VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "0.0.0"

def wall_follow_policy(obs):
    global WF_ESCAPE_STEPS, WF_ESCAPE_TURN

    num_sensors = len(obs) - 2
    prox = np.array(obs[:num_sensors], dtype=np.float32)


    # Confirmed mapping (clockwise):
    # 0 = front
    # 1 = front-left
    # 2 = left
    # 3 = back-left
    # 4 = back
    # 5 = back-right
    # 6 = right
    # 7 = front-right

    front       = float(prox[0])
    front_left  = float(prox[1])
    left        = float(prox[2])
    back_left   = float(prox[3])
    right       = float(prox[6])
    front_right = float(prox[7])

    # --- escape mode ---
    if WF_ESCAPE_STEPS > 0:
        WF_ESCAPE_STEPS -= 1
        if WF_ESCAPE_STEPS >= 1:   # menos pasos de retroceso
            return 3  # backward
        return WF_ESCAPE_TURN

    # tuned thresholds 
    hard_block   = 0.70
    front_block  = 0.40
    wall_present = 0.18
    too_close    = 0.65
    corner_block = 0.65

    # 1) Severe frontal or front-left impact → exhaust system
    if front > hard_block or front_left > hard_block:
        WF_ESCAPE_STEPS = 2   # escape más corto
        WF_ESCAPE_TURN = 1
        return 3

    # 2) Hard impact on the left → escape
    if left > hard_block or back_left > corner_block:
        WF_ESCAPE_STEPS = 2
        WF_ESCAPE_TURN = 1
        return 3

    # 3) Moderate frontal blockage → turn right
    if front > front_block or front_left > front_block:
        return 1

    # 4) If left wall is present:
    if left > wall_present:
        if left > too_close:
            if left < 0.75:   # corrección más gradual
                return 0      # avanza un poco
            else:
                return 1      # corrige derecha
        return 0              # avanza

    # 5) If there is no left wall, but something is detected on the right → gentle zig-zag
    if right > wall_present or front_right > wall_present:
        return 2 if front_right > 0.5 else 0

    # 6) If nothing relevant is detected → forward
    return 0




def main():
    parser = argparse.ArgumentParser(description="EFSIM Master Runner - High Performance")
    parser.add_argument("--type", type=str, default="gru", choices=["mlp", "gru", "res", "random", "braitenberg", "wall_follow"])
    parser.add_argument("--model", type=str, required=True, help="Path to the model .json file")
    parser.add_argument("--map", type=str, default="default", help="Name of the map in /worlds (ex: u_shape, n_shape, default)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=2, help="1 for old models, 2 for social")
    parser.add_argument("--render", action="store_true", help="Show simulation window")
    parser.add_argument("--save_video", action="store_true", help="Save the first episode")
    parser.add_argument("--save_snapshot", action="store_true", help="Save PNG in high resolution at the end") 
    parser.add_argument("--out", type=str, default=None, help="Video file name")
    parser.add_argument("--sleep", type=float, default=0.02, help="Rendering speed")
    parser.add_argument("--no_bg", action="store_true", help="Disable Basal Ganglion (Silent Mode)")
    parser.add_argument("--no_social_filter", action="store_true", help="Disable follower's manual social filter")
    parser.add_argument("--random_spawns", action="store_true", help="Randomize agent starting positions")
    parser.add_argument("--random_food", action="store_true", help="Randomize food position")
    parser.add_argument("--fixed_orientations", action="store_true", help="Fix initial orientation to 0.0")
    parser.add_argument(
        "--social_mode",
        type=str,
        default="normal",
        choices=["normal", "off", "shuffled", "mute_sender"],
        help="Modo de canal social para ablaciones"
    )
    parser.add_argument(
        "--social_variant",
        type=str,
        default="angle_strength",
        choices=["angle_strength", "angle_only", "strength_only"],
        help="Codificación del canal social"
    )
    parser.add_argument("--save_csv", action="store_true", help="Save metrics per episode to CSV")
    parser.add_argument("--csv_out", type=str, default=None, help="CSV output file name")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")
    parser.add_argument("--version", action="version", version=f"EFSIM {get_version()}")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[RUN] seed={args.seed}")

    # --- AUTO-DETECCIÓN DE TIPO ---
    # El propio JSON del modelo suele declarar su tipo/encoding. Si no coincide
    # con --type, avisamos y usamos el declarado (evita crashes tipo KeyError
    # por cargar un modelo GRU como si fuera 'res', etc.)
    if args.type not in ("wall_follow", "random"):
        declared_type = detect_model_type(args.model)
        if declared_type and declared_type != args.type:
            print(
                f"[AVISO] El modelo '{args.model}' declara type/encoding='{declared_type}', "
                f"pero se pidió --type {args.type}. Usando '{declared_type}' (el que declara el archivo)."
            )
            args.type = declared_type

    # 1. Environment Setup
    env_config = ForagingEnvConfig(
        map_name=args.map,
        num_agents=args.agents,
        u_env=(args.map == "u_shape"), 
        max_steps=1000,
        randomize_spawns=args.random_spawns,
        randomize_food=args.random_food,
        randomize_orientations=not args.fixed_orientations,
        social_mode=args.social_mode,
        social_variant=args.social_variant,
    )

    render_mode = "rgb_array" if args.save_video else ("human" if args.render else None)
    env = ForagingEnv(config=env_config, render_mode=render_mode)

    # 2. Load Controllers

    if args.type == "wall_follow":
        controllers = [DummyController() for _ in range(args.agents)]
    else:
        controllers = [
            make_controller(
                args.type,
                num_agents=args.agents,
                agent_idx=i,
                model_path=args.model,
                use_bg=not args.no_bg,
                use_social_filter=not args.no_social_filter,
            ) for i in range(args.agents)
        ]

    print(f"\n--- SYSTEM EFSIM: Corriendo {args.type.upper()} with {args.agents} agents ---")

    results = []
    for ep in range(args.episodes):
        obs_list, info = env.reset()
        
        # --- INTELLIGENT FOOD ---
        if args.random_food:
            if args.map == "u_shape": 
                # If it's the U-maze, we force the food INSIDE the maze.
                env.food_pos = np.array([
                    np.random.uniform(280, 500), 
                    np.random.uniform(280, 430)
                ], dtype=np.float32)
            else:
                # If it's the default map, any spot is good.
                env.food_pos = env._sample_food().astype(np.float32)
            
        for c in controllers:
            c.reset()
        frames, done = [], False

        stuck_counters = [0] * args.agents
        last_positions = [p.copy() for p in env.agent_pos]
        ep_stuck_steps = 0

        while not done:
            actions = []
            for i in range(args.agents):
                if args.type == "wall_follow":
                    actions.append(wall_follow_policy(obs_list[i]))
                else:
                    act_id = controllers[i].act(obs_list[i])
                    actions.append(act_id)
                
            # GANGLIO BASAL (Kept exactly as original) ---
            for i in range(args.agents):
                dist_moved = np.linalg.norm(env.agent_pos[i] - last_positions[i])
                if info['individual_success'][i] and random.random() < 0.7:
                    actions[i] = 4
                elif actions[i] in [1, 2] and dist_moved < 0.5 and random.random() < 0.2:
                    actions[i] = 4
            # ------------------------------------------------------------

            obs_list, _, terminated, truncated, info = env.step(actions)          

            at = info['individual_success']
            for i in range(args.agents):
                if not at[i]:
                    dist_moved = np.linalg.norm(env.agent_pos[i] - last_positions[i])
                    if dist_moved < 1.0:
                        stuck_counters[i] += 1
                        if stuck_counters[i] > 30:
                            ep_stuck_steps += 1
                    else:
                        stuck_counters[i] = 0
                last_positions[i] = env.agent_pos[i].copy()

            if args.save_video or args.render:
                frame = env.render()
                if args.save_video and ep == 0:
                    frames.append(frame)

            done = terminated or truncated
            if args.render and not args.save_video:
                time.sleep(args.sleep)

        if args.save_video and ep == 0 and len(frames) > 0:
            if info['success']:
                for _ in range(20):
                    frames.append(frames[-1])

            video_name = args.out if args.out else f"video_{args.type}_{args.map}.mp4"
            imageio.mimsave(video_name, frames, fps=30, macro_block_size=1)
            print(f"[LOG] Saved video as: {video_name}")
            
        if args.save_snapshot:
            # 1. Rendering
            env.render() 
            # 2. We obtain the active Pygame surface
            final_surface = pygame.display.get_surface()
            
            if final_surface is not None:
                hi_res_size = (env.config.world_width * 3, env.config.world_height * 3)
                hi_res_surface = pygame.transform.smoothscale(final_surface, hi_res_size)

                snap_name = f"snapshot_ep{ep+1}_highres.png"
                pygame.image.save(hi_res_surface, snap_name)
                print(f"[LOG] High-Res Snapshot saved: {snap_name}")
            else:
                print("[ERROR] Could not capture surface for snapshot.")

        info['stuck_steps'] = ep_stuck_steps
        results.append(info)
        status = "SUCCESS" if info['success'] else "FAIL"
        print(
            f"Episodio {ep+1}: {status} in {info['step_count']} steps | "
            f"SuccessSteps={info['success_steps']} | "
            f"Gap={info['goal_gap']} | "
            f"Screams={info['signal_counts']} | "
            f"Stuck={ep_stuck_steps}"
        )

    success_rate = np.mean([1.0 if r["success"] else 0.0 for r in results])
    avg_steps = np.mean([r["step_count"] for r in results])
    avg_stuck = np.mean([r["stuck_steps"] for r in results])

    valid_gaps = [r["goal_gap"] for r in results if r["goal_gap"] >= 0]
    avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else -1.0

    total_signal_counts = [sum(r["signal_counts"]) for r in results]
    avg_total_signals = np.mean(total_signal_counts)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Group Success: {success_rate*100:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average Stuck Steps: {avg_stuck:.1f}")
    print(f"Average Goal Gap: {avg_gap:.1f}" if avg_gap >= 0 else "Average Goal Gap: N/A")
    print(f"Average total signals: {avg_total_signals:.1f}")

    if args.save_csv:
        csv_name = args.csv_out if args.csv_out else f"results_{args.type}_{args.map}_{args.social_mode}.csv"
        with open(csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "success",
                "step_count",
                "success_steps",
                "first_goal_step",
                "second_goal_step",
                "goal_gap",
                "signal_counts",
                "stuck_steps",
                "social_mode",
                "social_variant",
                "random_spawns",
                "random_food",
                "u_env",
            ])

            for ep_idx, r in enumerate(results, start=1):
                writer.writerow([
                    ep_idx,
                    int(r["success"]),
                    r["step_count"],
                    "|".join(map(str, r["success_steps"])),
                    r["first_goal_step"],
                    r["second_goal_step"],
                    r["goal_gap"],
                    "|".join(map(str, r["signal_counts"])),
                    r["stuck_steps"],
                    args.social_mode,
                    args.social_variant,
                    int(args.random_spawns),
                    int(args.random_food),
                    int(args.map == "u_shape"), # Cambiado aquí también
                ])

        print(f"[LOG] CSV saved as: {csv_name}")

    env.close()


if __name__ == "__main__":
    main()
