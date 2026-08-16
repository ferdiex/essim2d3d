import pybullet as p
import pybullet_data
import numpy as np
import argparse
import random
import time
import json
import sys
import csv
import glob
import re
import os

from config_loader import load_robot_config, load_world_file, load_braitenberg
from world_builder import build_world, _resolve_physics
from controllers import FusionController, Unified3DController

from collections import deque
import random as _random_module


class PinballSim:
    def __init__(self, cfg=None, world_name="random_obstacles", controller_name="social2d_emergence.json", record_video=False, headless=False, fast_mode=False, hunger_gate=False, swap_brains=False, reverse_order=False, reverse_spawn_order=False, force_brain_ratio=False, brain_ratio=15, physics=None, audit_physics=False, log_csv=False):
        """
        Initializes the simulation environment and selects the appropriate controller 
        based on the input configuration (GRU brain or Braitenberg).
        """
        self.log_data = [] # Buffer para el log
        self.record_video = record_video
        self.video_log_id = None
        self.video_path = None
        self.csv_path = None
        
        self.paused = False
        self.cfg = cfg if cfg is not None else load_robot_config()
        self.world_data = load_world_file(world_name)
        self.rng_a = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        self.rng_b = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        # RNGs separados para el escape de Braitenberg (no reusar rng_a/rng_b
        # de spawn, para no alterar esa secuencia; cada robot con el suyo,
        # no el modulo random global compartido entre los dos).
        self.escape_rng_a = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        self.escape_rng_b = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        # Idem para el stuck_act del GRU (Unified3DController): antes se
        # sorteaba con el modulo random global, compartido y orden-dependiente
        # entre agente A y B. Cada uno con el suyo.
        self.gru_rng_a = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        self.gru_rng_b = _random_module.Random(_random_module.randint(0, 2**31 - 1))
        self.swap_brains = swap_brains
        self.reverse_order = reverse_order
        self.reverse_spawn_order = reverse_spawn_order
        
        # --- Controller Injection Logic (DUAL) ---
        path = f"models/{controller_name}" if "/" not in controller_name else controller_name
        if not path.endswith(".json"): path += ".json"
        
        with open(path, "r") as f: weights = json.load(f)
        
        if "social" in controller_name:
            # swap_brains: si True, el slot A (spawn cerca/mirando) recibe los
            # pesos de brain_b, y el slot B (spawn lejos/de espaldas) recibe
            # los de brain_a. Sirve para saber si el fallo sigue a los PESOS
            # o al SLOT/rol fisico (spawn, decay via agent_type).
            type_for_slot_a = 'brain_b' if self.swap_brains else 'brain_a'
            type_for_slot_b = 'brain_a' if self.swap_brains else 'brain_b'
            self.ctrl_a = Unified3DController(self.cfg, weights, agent_type=type_for_slot_a, rng=self.gru_rng_a, hunger_gate=hunger_gate)
            self.ctrl_b = Unified3DController(self.cfg, weights, agent_type=type_for_slot_b, rng=self.gru_rng_b, hunger_gate=hunger_gate)
            self.is_gru = True
        else:
            self.b_cfg = load_braitenberg(controller_name)
            self.ctrl_a = FusionController(self.cfg, self.b_cfg, rng=self.escape_rng_a)
            self.ctrl_b = FusionController(self.cfg, self.b_cfg, rng=self.escape_rng_b)
            self.is_gru = False

        # Si no se paso --physics explicito, el default depende del
        # controlador: Braitenberg fue diseñado/calibrado contra los
        # valores "slippery" (ver PHYSICS_PRESETS en world_builder.py);
        # forzarlo a "grippy" fue lo que rompio su escape. GRU si usa
        # "grippy" como default.
        if physics is None:
            physics = "grippy" if self.is_gru else "slippery"
        self.physics = physics
        self.phys = _resolve_physics(self.physics)  # dict {lateralFriction, restitution}
        self.audit_physics_flag = audit_physics
        self.log_csv_flag = log_csv

        self.controllers = [self.ctrl_a, self.ctrl_b]
        
        # Timing sync: 15 physics steps = 1 brain decision (approx 62.5ms)
        self.brain_ratio = brain_ratio 
        self.force_brain_ratio = force_brain_ratio
        # ----------------------------

        self.headless = headless
        p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.cfg['world']['gravity'])

        if not headless:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            v = self.cfg['visualization']
            p.resetDebugVisualizerCamera(v['camera_distance'], v['camera_yaw'], v['camera_pitch'], v['camera_target'])
    
    
    
        self._world_ids = build_world(self.world_data, self.cfg['world']['size'], physics=self.physics)
               
        # --- SPAWN DUAL (uniforme; orden de sorteo opcionalmente invertido) ---
        if self.reverse_spawn_order:
            self.start_pos_b, self.start_yaw_b = self.sample_valid_spawn(rng=self.rng_b)
            self.robot_id_b = p.loadURDF(self.cfg['robot']['urdf_path'], self.start_pos_b, p.getQuaternionFromEuler([0,0,self.start_yaw_b]))
            self.start_pos_a, self.start_yaw_a = self.sample_valid_spawn(rng=self.rng_a)
            self.robot_id_a = p.loadURDF(self.cfg['robot']['urdf_path'], self.start_pos_a, p.getQuaternionFromEuler([0,0,self.start_yaw_a]))
        else:
            self.start_pos_a, self.start_yaw_a = self.sample_valid_spawn(rng=self.rng_a)
            self.robot_id_a = p.loadURDF(self.cfg['robot']['urdf_path'], self.start_pos_a, p.getQuaternionFromEuler([0,0,self.start_yaw_a]))
            self.start_pos_b, self.start_yaw_b = self.sample_valid_spawn(rng=self.rng_b)
            self.robot_id_b = p.loadURDF(self.cfg['robot']['urdf_path'], self.start_pos_b, p.getQuaternionFromEuler([0,0,self.start_yaw_b]))

        self.robots = [self.robot_id_a, self.robot_id_b]
        
        # Setup para ambos
        for rid in self.robots: 
            self.setup_physics(rid)

        if self.audit_physics_flag:
            self.audit_physics()
        
        self.find_joints_dual() # <--- Nueva versión para dos robots
        
        # Estados en listas (Índice 0 = Robot A, Índice 1 = Robot B)
        self.fast_mode = fast_mode
        self.paused_agents = [False, False]
        self.stalled = [False, False]
        self.stall_counts = [0, 0]
        self.unstall_counts = [0, 0]
        self.prev_positions = [None, None]

        # --- Detector de "olisqueo" (separado del de atasco total) ---
        # atasco total: 25 pasos con desplazamiento CASI CERO (disp<0.0001).
        # olisqueo: se sigue moviendo (no dispara stalled), pero el
        # desplazamiento NETO en una ventana de pasos es chico Y hay un
        # sensor cerca de un obstaculo -- sospecha de "raspar" pared sin
        # despegarse, en vez de quedar completamente quieto.
        self.hug_window = 30
        self.hug_disp_thresh = 0.02   # desplazamiento neto minimo esperado en la ventana
        # Antes 0.3 (~1.5 cuerpos de distancia). Alineado con REACT_START de
        # FusionController en controllers.py (~1 cuerpo, lectura=0.625) --
        # el detector de olisqueo y el umbral de reaccion de Braitenberg
        # ahora usan el mismo criterio de "cerca de verdad".
        self.hug_sensor_thresh = 0.625  # sensor "cerca de algo" (rango normalizado 0-1)
        self.hug_pos_history = [deque(maxlen=self.hug_window), deque(maxlen=self.hug_window)]
        self.hug_step_counts = [0, 0]   # pasos totales clasificados como "olisqueando"
        self.hug_episode_active = [False, False]
        self.hug_episode_counts = [0, 0]  # cuantos EPISODIOS de olisqueo (no pasos)
        self.last_sensors = [None, None]  # cache del ultimo sensor leido por robot
        self.hug_debug_buffer = []  # [step, agent, pos_x, pos_y, yaw, alpha, seek_l, seek_r, avoid_l, avoid_r, escape_timer]
        self.hug_now = [False, False]  # estado actual (no acumulado), leido por Braitenberg como atasco temprano
        self.escape_log = []  # [step, agent, pos_x, pos_y, dist_target, escape_direction, intento_num]

    def setup_physics(self, robot_id): # <--- Ahora recibe ID
        # Chasis y links no-rueda usan friccion Y restitucion del preset de
        # mundo (grippy/slippery). Antes solo se seteaba lateralFriction;
        # restitution del robot quedaba en el default de PyBullet (0.0)
        # sin importar el preset -- confirmado con audit_physics().
        chassis_fric = self.phys["lateralFriction"]
        chassis_rest = self.phys["restitution"]
        p.changeDynamics(robot_id, -1, linearDamping=0.9, angularDamping=0.9,
                          lateralFriction=chassis_fric, restitution=chassis_rest)
        for j in range(p.getNumJoints(robot_id)):
            name = p.getJointInfo(robot_id, j)[1].decode('utf-8')
            fric = self.cfg['robot']['friction']['wheel'] if "wheel" in name else chassis_fric
            p.changeDynamics(robot_id, j, lateralFriction=fric, restitution=chassis_rest)

    def audit_physics(self):
        """Imprime, una sola vez, los valores REALES que PyBullet tiene
        aplicados (via getDynamicsInfo, no lo que creemos haber seteado)
        para el chasis del robot A, una rueda de A, y el primer body del
        mundo (pared). getDynamicsInfo(): indice 1 = lateralFriction,
        indice 5 = restitution."""
        print(f"\n[AUDIT] physics preset activo: {self.physics!r} -> {self.phys}")

        info_chassis = p.getDynamicsInfo(self.robot_id_a, -1)
        print(f"[AUDIT] robot_a chasis (link -1): lateralFriction={info_chassis[1]}, restitution={info_chassis[5]}")

        for j in range(p.getNumJoints(self.robot_id_a)):
            name = p.getJointInfo(self.robot_id_a, j)[1].decode('utf-8')
            if "wheel" in name:
                info_wheel = p.getDynamicsInfo(self.robot_id_a, j)
                print(f"[AUDIT] robot_a link '{name}' (joint {j}): lateralFriction={info_wheel[1]}, restitution={info_wheel[5]}")

        if len(self._world_ids) > 1:
            wall_id = self._world_ids[1]  # ids[0] es plane.urdf (piso), no pared;
                                           # la primera pared real es ids[1]
            info_wall = p.getDynamicsInfo(wall_id, -1)
            print(f"[AUDIT] mundo body_id={wall_id} (pared real): lateralFriction={info_wall[1]}, restitution={info_wall[5]}\n")
        else:
            print("[AUDIT] no hay suficientes world_ids capturados, no se pudo auditar pared\n")

    def find_joints_dual(self):
        # Guardamos los índices de las ruedas en diccionarios usando el ID del robot como llave
        self.left_wheels = {}
        self.right_wheels = {}
        
        for rid in self.robots:
            for j in range(p.getNumJoints(rid)):
                name = p.getJointInfo(rid, j)[1].decode('utf-8')
                if "left_wheel" in name: self.left_wheels[rid] = j
                if "right_wheel" in name: self.right_wheels[rid] = j

    def get_sensors(self, rid): # <--- Ahora recibe el ID del robot
        if not p.isConnected(): return np.zeros(8)
        pos, ori = p.getBasePositionAndOrientation(rid)
        yaw = p.getEulerFromQuaternion(ori)[2]
        
        angles = yaw + np.linspace(0, 2*np.pi, 8, endpoint=False)
        s_range = self.cfg['sensors']['range']
        sensor_z = pos[2] + 0.03  # entre ruedas/indicador (~0.051) y techo de muros/obstaculos (0.08)
        starts = [[pos[0], pos[1], sensor_z]] * 8
        ends = [[pos[0] + np.cos(a)*s_range, pos[1] + np.sin(a)*s_range, sensor_z] for a in angles]
        
        results = p.rayTestBatch(starts, ends)
               
        if results is None: return np.zeros(8) 
        # No detectarse a sí mismo (res[0] != rid)
        readings = [(1.0 - res[2]) if (res[0] != -1 and res[0] != rid) else 0.0 for res in results]
        return np.array(readings)
 
    def sample_valid_spawn(self, max_tries=2000, rng=None):
        rng = rng if rng is not None else __import__("random")
        half, r, wall_margin = self.cfg['world']['size'] / 2.0, 0.06, 0.075
        rects = self.world_data.get("rectangles", [])
        for _ in range(max_tries):
            x, y, yaw = rng.uniform(-half+wall_margin, half-wall_margin), rng.uniform(-half+wall_margin, half-wall_margin), rng.uniform(-np.pi, np.pi)
            if not any((abs(x-rect["pos"][0])<=(rect["size"][0]/2+r)) and (abs(y-rect["pos"][1])<=(rect["size"][1]/2+r)) for rect in rects):
                return [x, y, 0.02], yaw
        return [0.0, 0.0, 0.02], 0.0

    def _reset_agents(self):
        """Mismo reset que antes disparaba la tecla R -- extraido a metodo
        propio (sesion 2026-08-11) para poder reusarlo desde el auto-reset
        headless sin duplicar logica."""
        step_count = 0
        for i, rid in enumerate(self.robots):
            s_pos, s_yaw = self.sample_valid_spawn()
            p.resetBasePositionAndOrientation(rid, s_pos, p.getQuaternionFromEuler([0, 0, s_yaw]))
            self.paused_agents[i] = False
            self.stalled[i] = False
            self.stall_counts[i] = 0
            self.unstall_counts[i] = 0
            if self.is_gru: self.controllers[i].reset()
        self.paused = False
        return step_count

    def run(self, steps=None, auto_reset_every=None):
        target_pos = self.world_data.get("food_pos", [0.0, 0.4])
        step_count = 0
        dt = float(self.cfg["world"]["timestep"])
        next_tick = time.perf_counter()
        w_r = self.cfg['robot']['wheel_radius']
        
        # Inicialización de velocidades y caché de color para rendimiento
        vs = [[0.0, 0.0], [0.0, 0.0]] 
        signals = [0, 0]
        self.last_colors = [[-1,-1,-1,-1], [-1,-1,-1,-1]] # <--- PARCHE RENDIMIENTO
        
        if self.record_video and not self.headless and p.isConnected():
            self.video_path = next_indexed_filename("gru_run_", ".mp4")
            self.video_log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.video_path)
            print(f"[VIDEO] Recording started: {self.video_path}")

        try:
            while p.isConnected():
                # DIAGNOSTICO sesion 2026-08-11: auto-reset por cantidad de
                # pasos, para juntar varios episodios independientes en una
                # sola corrida headless (--log_csv), sin depender de la
                # tecla R (que nunca dispara en headless -- getKeyboardEvents
                # siempre devuelve {} ahi). Mismo _reset_agents() que la
                # tecla R, sin duplicar logica.
                if auto_reset_every and step_count > 0 and step_count % auto_reset_every == 0:
                    self._reset_agents()
                    print(f">>> AUTO-RESET at step {step_count} (auto_reset_every={auto_reset_every})")

                # PARCHE: Límite de pasos real
                if steps is not None and step_count >= steps: 
                    print(f"[MISSION] Target steps reached: {step_count}")
                    for i in (0, 1):
                        pct = 100.0 * self.hug_step_counts[i] / max(step_count, 1)
                        print(f"[HUG] robot_{i}: {self.hug_step_counts[i]}/{step_count} pasos "
                              f"olisqueando ({pct:.1f}%), {self.hug_episode_counts[i]} episodios")
                    break # <--- Esto matará el proceso al llegar al límite

                # --- Brain Sync (Para ambos) ---
                if (not self.is_gru and not self.force_brain_ratio) or (step_count % self.brain_ratio == 0):
                    # Congelamos una copia de signals ANTES de que cualquiera de
                    # los dos agentes decida. Sin esto, quien procesa primero en
                    # el loop (ver 'order' abajo) lee la senal VIEJA del otro
                    # (del ciclo anterior), mientras que quien procesa segundo
                    # lee la senal NUEVA recien calculada por el primero -- una
                    # asimetria de un ciclo de retraso entre A y B, no percibida
                    # como tal por ninguno. Con la foto congelada, los dos leen
                    # exactamente el mismo instante, sin importar el orden.
                    signals_snapshot = list(signals)

                    # Orden invertido: 1 primero, 0 despues. Sirve para saber si
                    # quien "gana" es quien procesa primero en el loop (orden),
                    # no quien es A/B ni que pesos carga.
                    order = [1, 0] if self.reverse_order else [0, 1]
                    for i in order:
                        if not self.paused_agents[i]:
                            sensors = self.get_sensors(self.robots[i])
                            self.last_sensors[i] = sensors
                            r_pos, r_ori = p.getBasePositionAndOrientation(self.robots[i])
                            yaw = p.getEulerFromQuaternion(r_ori)[2]
                            
                            # --- Intermitencia Social (DESACTIVADA para diagnostico) ---
                            # Antes: el receptor solo "oia" la senal 15 pasos si y 15 no.
                            # Ahora: senal social siempre disponible, sin el latido artificial
                            # que no existe en 2D, para aislar su efecto del resto.
                            social_pulse = signals_snapshot[1-i]
                            
                            if self.is_gru:
                                other_pos, _ = p.getBasePositionAndOrientation(self.robots[1 - i])
                                res = self.controllers[i].get_action(
                                    sensors, r_pos, yaw, target_pos, self.stalled[i], social_pulse,
                                    other_pos=other_pos
                                )
                                # 4 valores (v_l, v_r, signal, x)
                                vs[i][0], vs[i][1], signals[i], current_x = res

                                # Guardamos en el buffer: [Paso, Agente(0 o 1), ...11 valores de X]
                                if self.log_csv_flag:
                                    if not hasattr(self, 'log_buffer'): self.log_buffer = []
                                    max_sensor = float(np.max(current_x[:8]))
                                    self.log_buffer.append([step_count, i] + list(current_x) + [r_pos[0], r_pos[1], yaw, max_sensor])
                            else:
                                # FusionController (Braitenberg 3D): solo (v_l, v_r), sin canal social.
                                # is_stuck temprano: stalled (atasco total) O hug_now (rozando
                                # sin avanzar, ver detector de olisqueo) -- asi la alternancia
                                # dispara mientras todavia esta frenando, no solo cuando ya
                                # quedo 100% detenido. Solo afecta a Braitenberg, no al GRU.
                                early_stuck = self.stalled[i] or self.hug_now[i]
                                res = self.controllers[i].get_action(
                                    sensors, r_pos, yaw, target_pos, early_stuck
                                )
                                vs[i][0], vs[i][1] = res
                                signals[i] = 0

                                # --- Log de eventos de alternancia (escape) ---
                                # escape_timer llega a exactamente 100 solo en el
                                # tick donde se DISPARA un nuevo evento (despues se
                                # decrementa). Loguea posicion + distancia al
                                # objetivo, para poder confirmar/descartar la
                                # sospecha de "pared magnetica" (el compas lo jala
                                # de vuelta al mismo punto tras cada escape).
                                dbg = self.controllers[i].debug
                                if dbg.get("triggered"):
                                    dx_t = target_pos[0] - r_pos[0]
                                    dy_t = target_pos[1] - r_pos[1]
                                    dist_t = float(np.sqrt(dx_t**2 + dy_t**2))
                                    self.escape_log.append([
                                        step_count, i, r_pos[0], r_pos[1], dist_t,
                                        self.controllers[i].escape_direction,
                                        self.controllers[i].consecutive_stalls,
                                    ])
                                    print(f"[ESCAPE] step={step_count} agent={i} pos=({r_pos[0]:.3f},{r_pos[1]:.3f}) "
                                          f"dist_target={dist_t:.3f} dir={self.controllers[i].escape_direction} "
                                          f"intento_num={self.controllers[i].consecutive_stalls}")

                                # --- Etiqueta discreta para behavioral cloning ---
                                # Braitenberg no elige de una tabla discreta como el GRU
                                # (Unified3DController.action_table); calcula (v_l, v_r)
                                # continuo. Para poder imitarlo con la misma arquitectura
                                # de salida (argmax sobre 5 acciones), se mapea su
                                # (v_l, v_r) a la accion discreta mas cercana de esa misma
                                # tabla, normalizando por max_speed.
                                _action_table = {
                                    0: (1.0, 1.0), 1: (-0.3, 0.6), 2: (0.6, -0.3),
                                    3: (-0.5, -0.5), 4: (0.8, 0.8),
                                }
                                _max_speed = self.cfg['robot']['max_speed']
                                _vl_n, _vr_n = vs[i][0] / _max_speed, vs[i][1] / _max_speed
                                imitation_action_id = min(
                                    _action_table,
                                    key=lambda k: (_action_table[k][0]-_vl_n)**2 + (_action_table[k][1]-_vr_n)**2
                                )

                                # Logging con sensores REALES (no ceros) + accion discreta
                                # equivalente, para poder usarse como dataset de cloning.
                                dx, dy = target_pos[0] - r_pos[0], target_pos[1] - r_pos[1]
                                dist = np.sqrt(dx**2 + dy**2)
                                bearing = (np.arctan2(dy, dx) - yaw + np.pi) % (2*np.pi) - np.pi
                                decay = 1.25 if i == 1 else 1.0
                                magnet = float(np.exp(-dist / decay))
                                if self.log_csv_flag:
                                    if not hasattr(self, 'log_buffer'): self.log_buffer = []
                                    self.log_buffer.append(
                                        [step_count, i] + list(sensors) + [bearing/np.pi, magnet, float(self.stalled[i]), r_pos[0], r_pos[1], yaw, imitation_action_id]
                                    )

                # --- Success Gate Individual (Alta Frecuencia) ---
                for i, rid in enumerate(self.robots):
                    r_pos_now, _ = p.getBasePositionAndOrientation(rid)
                    dist_actual = np.linalg.norm(np.array(r_pos_now[:2]) - np.array(target_pos[:2]))
                    
                    # Identidad Base: A=Blanco, B=Gris
                    base_color = [1, 1, 1, 1] if i == 0 else [0.7, 0.7, 0.7, 1]
                   
                    # 1. ÉXITO DINÁMICO (Solo si está cerca)
                    if dist_actual < 0.08:
                        self.paused_agents[i] = True
                        v_l_final, v_r_final, f_motor = 0.0, 0.0, 0.0
                        signals[i] = 4 # Emite señal de "estoy aquí"
                        # Parpadea entre Verde y su Identidad (Blanco/Gris)
                        color = [0, 1, 0, 1] if (step_count // 30) % 2 == 0 else base_color
                    
                    else:
                        # REVERSIBILIDAD: Si sale del área, recupera el control
                        self.paused_agents[i] = False
                        
                        # 2. ATASCADO (Rojo) -- antes pisaba vs[i] con un comando
                        # fijo (-0.4,-0.2), lo que impedia que la logica de escape
                        # de cada controlador (alternancia de FusionController,
                        # stuck_act de Unified3DController) llegara al motor. Ahora
                        # se deja pasar vs[i] real; el rojo queda solo como
                        # indicador visual de que el agente esta atascado.
                        if self.stalled[i]:
                            v_l_final, v_r_final, f_motor, color = vs[i][0], vs[i][1], 1.5, [1, 0, 0, 1]
                        
                        # 3. SOCIAL: ESCUCHANDO (Azul)
                        elif signals[1-i] == 4:
                            v_l_final, v_r_final, f_motor, color = vs[i][0], vs[i][1], 1.5, [0, 0, 1, 1]
                        
                        # 4. SOCIAL: EMITIENDO (Magenta)
                        elif signals[i] == 4:
                            v_l_final, v_r_final, f_motor, color = vs[i][0], vs[i][1], 1.5, [1, 0, 1, 1]
                        
                        # 5. NEUTRAL
                        else:
                            v_l_final, v_r_final, f_motor = vs[i][0], vs[i][1], 1.5
                            color = [1, 1, 0, 1] if dist_actual < 0.4 else base_color

                    # Actualización visual optimizada
                    if color != self.last_colors[i]:
                        p.changeVisualShape(rid, -1, rgbaColor=color)
                        self.last_colors[i] = color

                    p.setJointMotorControl2(rid, self.left_wheels[rid], p.VELOCITY_CONTROL, 
                                            targetVelocity=v_l_final / w_r, force=f_motor)
                    p.setJointMotorControl2(rid, self.right_wheels[rid], p.VELOCITY_CONTROL, 
                                            targetVelocity=v_r_final / w_r, force=f_motor)
                                       

                # --- UI / Reset Logic (solo con ventana activa) ---
                keys = p.getKeyboardEvents() if not self.headless else {}

                if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED: # SPACE
                    self.fast_mode = not self.fast_mode
                    print(f"FAST MODE: {self.fast_mode}")
                    next_tick = time.perf_counter()  
                    
                if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED: self.paused = not self.paused

                if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                    # Toggle en vivo grippy <-> slippery, para comparar visualmente
                    # sin reiniciar el proceso. Reusa setup_physics() (chasis+ruedas
                    # de cada robot) y aplica el mismo preset a paredes/obstaculos
                    # ya creados (self._world_ids[0] es plane.urdf, sin fisica de
                    # contacto seteada por build_world, se deja afuera).
                    self.physics = "slippery" if self.physics == "grippy" else "grippy"
                    self.phys = _resolve_physics(self.physics)
                    for rid in self.robots:
                        self.setup_physics(rid)
                    for body_id in self._world_ids[1:]:
                        p.changeDynamics(body_id, -1, lateralFriction=self.phys["lateralFriction"],
                                          restitution=self.phys["restitution"])
                    print(f">>> PHYSICS TOGGLE: ahora '{self.physics}' -> {self.phys}")

                if ord('h') in keys and keys[ord('h')] & p.KEY_WAS_TRIGGERED:
                    # Toggle en vivo del PARCHE DE DIAGNOSTICO hunger_gate (ver
                    # comentario en Unified3DController.__init__). Solo aplica
                    # si el controlador activo es GRU -- FusionController no
                    # tiene este atributo.
                    # NOTA: se uso 'h' y no 'g' porque PyBullet reserva 'g'
                    # para mostrar/ocultar sus propios paneles de debug --
                    # se pisaban y confundia cual toggle estaba pasando.
                    if self.is_gru:
                        new_val = not self.ctrl_a.hunger_gate
                        self.ctrl_a.hunger_gate = new_val
                        self.ctrl_b.hunger_gate = new_val
                        print(f">>> HUNGER_GATE TOGGLE: ahora {new_val}")
                    else:
                        print(">>> HUNGER_GATE: no aplica, el controlador activo no es GRU")
                
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    step_count = 0
                    for i, rid in enumerate(self.robots):
                        s_pos, s_yaw = self.sample_valid_spawn()
                        p.resetBasePositionAndOrientation(rid, s_pos, p.getQuaternionFromEuler([0,0,s_yaw]))
                        self.paused_agents[i] = False
                        self.stalled[i] = False
                        self.stall_counts[i] = 0
                        self.unstall_counts[i] = 0
                        if self.is_gru: self.controllers[i].reset()
                    self.paused = False
                    print(">>> SIMULATION RESET: Dual agents redeployed.")
                
                if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
                    for i, rid in enumerate(self.robots):
                        pos, _ = p.getBasePositionAndOrientation(rid)
                        p.resetBasePositionAndOrientation(rid, [pos[0], pos[1], 0.05], [0, 0, 0, 1])
                        self.stalled[i] = False
                        self.stall_counts[i] = 0
                        self.unstall_counts[i] = 0
                        self.prev_positions[i] = None

                if self.paused:
                    time.sleep(0.01)
                    continue

                p.stepSimulation()
                step_count += 1

                # --- Stall Detection DUAL ---
                for i, rid in enumerate(self.robots):
                    curr_pos, _ = p.getBasePositionAndOrientation(rid)
                    if self.prev_positions[i] is not None:
                        disp = np.linalg.norm(np.array(curr_pos) - np.array(self.prev_positions[i]))
                        cmd_moving = (abs(vs[i][0]) + abs(vs[i][1])) > 0.08
                        stuck_now = cmd_moving and disp < 0.0001
                        
                        if stuck_now: self.stall_counts[i] += 1; self.unstall_counts[i] = 0
                        else: self.unstall_counts[i] += 1; self.stall_counts[i] = 0
                        
                        if self.stall_counts[i] >= 25: self.stalled[i] = True   
                        elif self.unstall_counts[i] >= 30: self.stalled[i] = False 
                    self.prev_positions[i] = curr_pos

                    # --- Deteccion de "olisqueo" ---
                    # Ventana de posiciones -> desplazamiento NETO (no paso a
                    # paso) en los ultimos hug_window pasos. Si es chico Y hay
                    # un sensor cerca de un obstaculo Y el agente NO esta ya
                    # clasificado como atascado total (self.stalled), cuenta
                    # como "olisqueando": se mueve, pero no avanza neto, cerca
                    # de algo. Independiente del detector de atasco total.
                    self.hug_pos_history[i].append(np.array(curr_pos[:2]))
                    if len(self.hug_pos_history[i]) == self.hug_window and self.last_sensors[i] is not None:
                        net_disp = np.linalg.norm(self.hug_pos_history[i][-1] - self.hug_pos_history[i][0])
                        near_obstacle = float(np.max(self.last_sensors[i])) > self.hug_sensor_thresh
                        is_hugging = (net_disp < self.hug_disp_thresh) and near_obstacle and not self.stalled[i]
                        self.hug_now[i] = is_hugging
                        if is_hugging:
                            self.hug_step_counts[i] += 1
                            if not self.hug_episode_active[i]:
                                self.hug_episode_active[i] = True
                                self.hug_episode_counts[i] += 1
                            # Diagnostico "donde" + "por que" -- solo tiene sentido
                            # para FusionController (Braitenberg), que expone
                            # self.debug con alpha/seek/avoid del ultimo get_action.
                            # El GRU no tiene un desglose equivalente.
                            dbg = getattr(self.controllers[i], "debug", None)
                            if dbg is not None:
                                self.hug_debug_buffer.append([
                                    step_count, i, curr_pos[0], curr_pos[1],
                                    dbg.get("alpha"), dbg.get("seek_l"), dbg.get("seek_r"),
                                    dbg.get("avoid_l"), dbg.get("avoid_r"), dbg.get("escape_timer"),
                                ])
                        else:
                            self.hug_episode_active[i] = False

                # --- Reloj de Sincronización ---
                if not self.fast_mode:
                    next_tick += dt
                    delay = next_tick - time.perf_counter()
                    if delay > 0: time.sleep(delay)
                    else: next_tick = time.perf_counter()

        except p.error:
            print("\n[INFO] Simulator closed.")

        finally:
            print(f"[INFO] Run ended at iteration {step_count}.")

            # Guardar CSV diagnóstico
            if hasattr(self, 'log_buffer') and self.log_buffer:
                self.csv_path = self.csv_out_name if getattr(self, 'csv_out_name', None) else next_indexed_filename("gru_diagnostic_", ".csv")
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "agent", "s0","s1","s2","s3","s4","s5","s6","s7", "angle", "magnet", "stuck", "pos_x", "pos_y", "yaw", "max_sensor/imitation_action_id"])
                    writer.writerows(self.log_buffer)
                print(f"[DIAGNOSTIC] Log saved: {self.csv_path} ({len(self.log_buffer)} samples).")

            # CSV de diagnostico de olisqueo -- se llena solo si hubo
            # episodios reales, asi que no hace falta flag: si esta vacio,
            # simplemente no se escribe nada.
            if self.hug_debug_buffer:
                hug_csv_path = next_indexed_filename("hug_debug_", ".csv")
                with open(hug_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "agent", "pos_x", "pos_y", "alpha", "seek_l", "seek_r", "avoid_l", "avoid_r", "escape_timer"])
                    writer.writerows(self.hug_debug_buffer)
                print(f"[DIAGNOSTIC] Hug debug saved: {hug_csv_path} ({len(self.hug_debug_buffer)} samples).")

            # CSV de eventos de alternancia (escape) -- igual que hug_debug,
            # se llena solo si hubo eventos reales, no hace falta flag.
            if self.escape_log:
                escape_csv_path = next_indexed_filename("escape_events_", ".csv")
                with open(escape_csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "agent", "pos_x", "pos_y", "dist_target", "escape_direction", "intento_num"])
                    writer.writerows(self.escape_log)
                print(f"[DIAGNOSTIC] Escape events saved: {escape_csv_path} ({len(self.escape_log)} eventos).")
                
            if self.video_log_id is not None and p.isConnected():
                p.stopStateLogging(self.video_log_id)
                print(f"[VIDEO] Recording saved: {self.video_path}")

            if p.isConnected():
                p.disconnect()

def next_indexed_filename(prefix: str, ext: str, logs_dir: str = "logs") -> str:
    """Devuelve la ruta dentro de logs_dir (creandolo si no existe), siguiendo
    la convencion del resto del proyecto (train_social3d.py, train_bc_nav.py
    ya escriben/leen de logs/). Antes escribia en el directorio actual."""
    os.makedirs(logs_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    max_n = 0
    for path in glob.glob(os.path.join(logs_dir, f"{prefix}*{ext}")):
        base = os.path.basename(path)
        m = pattern.match(base)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(logs_dir, f"{prefix}{max_n+1:03d}{ext}")

def parse_args():
    parser = argparse.ArgumentParser(description="ePuck 3D reactive simulation")
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps")
    parser.add_argument("--auto_reset_every", type=int, default=None,
                         help="DIAGNOSTICO (sesion 2026-08-11): resetea automaticamente cada N "
                              "pasos de fisica, sin depender de la tecla R (que no dispara en "
                              "--headless). Util para juntar varios episodios independientes en "
                              "una sola corrida headless con --log_csv. Default: sin auto-reset "
                              "(comportamiento identico a siempre).")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad de spawns")
    parser.add_argument("--log_csv", action="store_true",
                         help="Genera el CSV de diagnostico/cloning (sensores + accion por paso). "
                              "Apagado por default -- antes se generaba siempre, sin flag.")
    parser.add_argument("--csv_out", type=str, default=None, help="Nombre exacto del CSV de diagnóstico (si no se da, usa el autoincrement)")
    parser.add_argument(
        "--controller",
        type=str,
        default="social2d_emergence.json",
        help="Controller JSON file name (inside models/) or path",
    )
    parser.add_argument(
        "--world",
        type=str,
        default="random_obstacles.json",
        help="World JSON file name (inside worlds/) or path",
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record MP4 video of the run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Correr sin interfaz gráfica (p.DIRECT en vez de p.GUI)"
    )
    parser.add_argument(
        "--fast_mode",
        action="store_true",
        help="Iniciar en fast mode desde el arranque (sin espera de tiempo real)"
    )
    parser.add_argument(
        "--hunger_gate",
        action="store_true",
        help="PARCHE DE DIAGNOSTICO (default apagado): atenua x[9] (hambre) a 0 "
             "cuando el sensor frontal (x[0]) detecta algo cerca. Solo afecta GRU. "
             "Toggleable en vivo con la tecla H."
    )
    parser.add_argument("--swap_brains", action="store_true", help="Intercambia que pesos (brain_a/brain_b) recibe cada slot fisico (A=cerca, B=lejos)")
    parser.add_argument("--reverse_order", action="store_true", help="Invierte el orden de procesamiento del loop (1 antes que 0)")
    parser.add_argument("--reverse_spawn_order", action="store_true", help="Invierte cual robot sortea su posicion primero")
    parser.add_argument("--force_brain_ratio", action="store_true", help="Fuerza que el Braitenberg tambien decida cada brain_ratio pasos, no cada paso")
    parser.add_argument("--brain_ratio", type=int, default=15, help="Cada cuantos pasos de fisica decide el GRU (default 15 = ~62.5ms)")
    parser.add_argument("--physics", type=str, default=None, choices=["grippy", "slippery"],
                         help="Preset de friccion/rebote de obstaculos y paredes. Si no se pasa, "
                              "se elige automaticamente segun el controlador: 'slippery' para "
                              "Braitenberg (los valores originales, necesita desprenderse libre), "
                              "'grippy' para GRU (necesita friccion para pivotar/girar). "
                              "Pasar el flag fuerza el valor sin importar el controlador.")
    parser.add_argument("--audit_physics", action="store_true",
                         help="Imprime una sola vez, al arrancar, los valores REALES de "
                              "friccion/restitucion aplicados al robot y a una pared "
                              "(via getDynamicsInfo), para confirmar que el preset se "
                              "aplico de verdad y no asumirlo.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            print(f"[RUN] seed={args.seed}")
    cfg = load_robot_config()
    steps = args.steps if args.steps is not None else cfg.get("sim", {}).get("steps", 10000)

    print(f"[RUN] steps={steps}")
    print(f"[RUN] controller={args.controller}")
    print(f"[RUN] world={args.world}")
    print(f"[RUN] physics={args.physics if args.physics is not None else 'auto (segun controlador)'}")

    sim = PinballSim(
        cfg=cfg,
        world_name=args.world,
        controller_name=args.controller,
        headless=args.headless,
        fast_mode=args.fast_mode,
        hunger_gate=args.hunger_gate,
        swap_brains=args.swap_brains,
        reverse_order=args.reverse_order,
        reverse_spawn_order=args.reverse_spawn_order,
        force_brain_ratio=args.force_brain_ratio,
        brain_ratio=args.brain_ratio,
        physics=args.physics,
        audit_physics=args.audit_physics,
        log_csv=args.log_csv
    )
    sim.csv_out_name = args.csv_out
    
    
    sim.run(steps=steps, auto_reset_every=args.auto_reset_every)


if __name__ == "__main__":
    main()
