"""
Wrapper de entorno 3D para Fase 1b (evolucion del canal social).

Expone una interfaz reset()/step() reutilizable por el genetico, separada
de PinballSim (essim3d.py) para no tocar el archivo de diagnostico/demo ya
probado hoy. Reutiliza las mismas piezas (build_world, config_loader,
get_sensors, sample_valid_spawn) para que el comportamiento fisico sea
identico al de essim3d.py.

Optimizaciones para evaluacion masiva (miles de episodios headless):
  - Conexion DIRECT (sin GUI) siempre, sin overhead de visualizacion,
    camara, colores, teclado ni video.
  - Robots cargados UNA sola vez por instancia de env; reset() solo
    reposiciona (resetBasePositionAndOrientation), no vuelve a hacer
    loadURDF ni a reconstruir el mundo -> mucho mas rapido que crear un
    PinballSim nuevo por episodio.
  - numThreads=1 por proceso: con multiprocessing.Pool ya se usan todos
    los nucleos a nivel de poblacion: si ademas PyBullet intenta paralelizar
    internamente, los procesos se pisan entre si y todo se hace mas lento.
  - numSolverIterations configurable (default = el mismo que PyBullet trae
    por default en essim3d.py, no se baja la fidelidad sin que se pida
    explicitamente).
  - Decision del cerebro cada `brain_ratio` pasos de fisica (igual que
    essim3d.py), pero SIN el sleep de sincronizacion en tiempo real.
"""
import random as _random_module

import numpy as np
import pybullet as p
import pybullet_data

from config_loader import load_robot_config, load_world_file
from world_builder import build_world, clear_world, _resolve_physics


class Foraging3DEnv:
    def __init__(self, cfg=None, world_name="random_obstacles", num_threads=1,
                 solver_iterations=None, brain_ratio=15, physics="slippery",
                 swap_creation_order=False):
        self.cfg = cfg if cfg is not None else load_robot_config()
        self.brain_ratio = brain_ratio
        self.physics = physics
        self._world_cache = {}  # world_name -> world_data (evita releer el JSON cada vez)

        # v17 (sesion 2026-08-14): parametros de la ventana tolerante del
        # detector de atasco (ver _reset_state y step() para el detalle).
        # WINDOW_SIZE=30 mismo largo que la racha perfecta anterior, para
        # que la comparacion sea directa. MIN_GOOD=24 (80% de la ventana)
        # -- moderado: no exige perfeccion (30/30) pero tampoco libera con
        # cualquier cosa (ej. 15/30 seria demasiado laxo).
        self.STALL_WINDOW_SIZE = 30
        self.STALL_WINDOW_MIN_GOOD = 24

        # success_threshold (sesion 2026-08-14): se evaluo subirlo a 0.1219
        # (misma proporcion real que 2D) pero el re-calibrado del decay dio
        # una diferencia minima (0.554 vs 0.563) -- no ameritaba el costo de
        # otra reevolucion completa solo para esto. Revertido a 0.08 (el
        # valor validado, usado en v15/v17). Si se retoma en el futuro, ver
        # handoff 2026-08-13/14 para el valor calculado (0.1219) y el plan
        # completo (incluye re-medir SOCIAL_SIGNAL_DECAY con
        # check_recruit_distance.py).
        self.success_threshold = 0.08

        p.connect(p.DIRECT)  # sin GUI, siempre, sin excepcion
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        try:
            p.setPhysicsEngineParameter(numThreads=num_threads)
        except TypeError:
            # Esta version de pybullet no expone numThreads aca; no es
            # critico (el paralelismo real ya viene de multiprocessing.Pool
            # a nivel de poblacion, esto era una optimizacion extra).
            pass
        if solver_iterations is not None:
            p.setPhysicsEngineParameter(numSolverIterations=solver_iterations)
        p.setGravity(0, 0, self.cfg['world']['gravity'])

        self.world_name = world_name
        self.world_data = self._load_world_cached(world_name)
        self._world_ids = build_world(self.world_data, self.cfg['world']['size'], physics=self.physics)
        self.phys = _resolve_physics(self.physics)  # dict {lateralFriction, restitution}

        self.target_pos = self.world_data.get("food_pos", [0.0, 0.4])
        self.rng_a = _random_module.Random()
        self.rng_b = _random_module.Random()

        urdf = self.cfg['robot']['urdf_path']
        # DIAGNOSTICO (sesion 2026-08-11): swap_creation_order invierte cual
        # cuerpo se crea PRIMERO en PyBullet (indice interno del solver),
        # sin tocar el mapeo externo self.robots=[robot_id_a, robot_id_b]
        # ni ningun otro punto del codigo -- sirve para aislar si la
        # ventaja de exito del slot B viene del orden de creacion de
        # cuerpos (posible sesgo del solver de colisiones por indice) o si
        # se queda pegada al slot igual. Default False = comportamiento
        # identico al de siempre (robot_id_a se crea primero).
        if swap_creation_order:
            self.robot_id_b = p.loadURDF(urdf, [0, 0, 0.02], [0, 0, 0, 1])
            self.robot_id_a = p.loadURDF(urdf, [0, 0, 0.02], [0, 0, 0, 1])
        else:
            self.robot_id_a = p.loadURDF(urdf, [0, 0, 0.02], [0, 0, 0, 1])
            self.robot_id_b = p.loadURDF(urdf, [0, 0, 0.02], [0, 0, 0, 1])
        self.robots = [self.robot_id_a, self.robot_id_b]

        for rid in self.robots:
            self._setup_physics(rid)
        self._find_joints()

        self.w_r = self.cfg['robot']['wheel_radius']
        self.max_speed = self.cfg['robot']['max_speed']

        self._reset_state()

    # ------------------------------------------------------------
    def _load_world_cached(self, world_name):
        if world_name not in self._world_cache:
            self._world_cache[world_name] = load_world_file(world_name)
        return self._world_cache[world_name]

    def switch_world(self, world_name):
        """Cambia de mapa (default/u_shape/n_shape) sin reconectar PyBullet
        ni recargar los robots: borra los bodies del mundo actual y
        construye el nuevo. Si ya es el mismo mundo, no hace nada."""
        if world_name == self.world_name:
            return
        clear_world(self._world_ids)
        self.world_name = world_name
        self.world_data = self._load_world_cached(world_name)
        self._world_ids = build_world(self.world_data, self.cfg['world']['size'], physics=self.physics)
        self.target_pos = self.world_data.get("food_pos", [0.0, 0.4])

    # ------------------------------------------------------------
    def _setup_physics(self, robot_id):
        # Antes: lateralFriction=0.01 fijo (ignoraba el preset grippy/
        # slippery) y restitution nunca seteada (quedaba en el default de
        # PyBullet, 0.0). Confirmado con auditoria en essim3d.py que esto
        # hacia que el preset de fisica no tuviera efecto real en el
        # contacto robot-pared/obstaculo -- las mediciones de evolucion
        # hechas hasta hoy (grippy vs slippery, 6.7%=6.7% en LOG_fase1b)
        # corrieron con esta inconsistencia. Mismo fix que essim3d.py.
        chassis_fric = self.phys["lateralFriction"]
        chassis_rest = self.phys["restitution"]
        p.changeDynamics(robot_id, -1, linearDamping=0.9, angularDamping=0.9,
                          lateralFriction=chassis_fric, restitution=chassis_rest)
        for j in range(p.getNumJoints(robot_id)):
            name = p.getJointInfo(robot_id, j)[1].decode('utf-8')
            fric = self.cfg['robot']['friction']['wheel'] if "wheel" in name else chassis_fric
            p.changeDynamics(robot_id, j, lateralFriction=fric, restitution=chassis_rest)

    def _find_joints(self):
        self.left_wheels, self.right_wheels = {}, {}
        for rid in self.robots:
            for j in range(p.getNumJoints(rid)):
                name = p.getJointInfo(rid, j)[1].decode('utf-8')
                if "left_wheel" in name: self.left_wheels[rid] = j
                if "right_wheel" in name: self.right_wheels[rid] = j

    def _sample_valid_spawn(self, rng, max_tries=2000, min_food_dist=0.15):
        half = self.cfg['world']['size'] / 2.0
        r, wall_margin = 0.06, 0.075
        rects = self.world_data.get("rectangles", [])
        target = np.array(self.target_pos[:2])
        for _ in range(max_tries):
            x = rng.uniform(-half + wall_margin, half - wall_margin)
            y = rng.uniform(-half + wall_margin, half - wall_margin)
            yaw = rng.uniform(-np.pi, np.pi)
            # DIAGNOSTICO sesion 2026-08-11: antes, el spawn podia caer
            # dentro (o casi) del radio de exito (0.08m) por puro azar --
            # "exito" sin haber navegado nada. Confirmado con datos:
            # ~2.3% de los spawns en random_obstacles caian dentro de
            # 0.08m. min_food_dist=0.15 (casi el doble del umbral de
            # exito, margen de seguridad) para que ningun spawn cuente
            # como "regalo".
            if np.linalg.norm(np.array([x, y]) - target) < min_food_dist:
                continue
            if not any((abs(x - rect["pos"][0]) <= (rect["size"][0] / 2 + r)) and
                       (abs(y - rect["pos"][1]) <= (rect["size"][1] / 2 + r)) for rect in rects):
                return [x, y, 0.02], yaw
        return [0.0, 0.0, 0.02], 0.0

    def _get_sensors(self, rid):
        pos, ori = p.getBasePositionAndOrientation(rid)
        yaw = p.getEulerFromQuaternion(ori)[2]
        angles = yaw + np.linspace(0, 2 * np.pi, 8, endpoint=False)
        s_range = self.cfg['sensors']['range']
        sensor_z = pos[2] + 0.03
        starts = [[pos[0], pos[1], sensor_z]] * 8
        ends = [[pos[0] + np.cos(a) * s_range, pos[1] + np.sin(a) * s_range, sensor_z] for a in angles]
        results = p.rayTestBatch(starts, ends)
        readings = [(1.0 - res[2]) if (res[0] != -1 and res[0] != rid) else 0.0 for res in results]
        return np.array(readings)

    def _reset_state(self):
        self.stalled = [False, False]
        self.stall_counts = [0, 0]
        self.unstall_counts = [0, 0]
        self.prev_positions = [None, None]
        self.paused_agents = [False, False]
        self.step_count = 0
        # v17 (sesion 2026-08-14): ventana tolerante para la liberacion del
        # detector de atasco. ANTES: unstall_counts exigia una racha
        # PERFECTA de 30 ticks consecutivos sin un solo tick malo -- un
        # solo traspie reseteaba todo a 0, aunque el resto de la ventana
        # hubiera sido buena. diagnose_histeresis_stall.py (checkpoint v15)
        # mostro que el robot llega a unstall_counts~29 de 30 repetidamente,
        # sin nunca completar la racha perfecta, mientras SI se desplaza de
        # verdad (camino recorrido real ~0.017-0.024 durante las rachas
        # "atascadas"). recent_ticks_ok guarda los ultimos WINDOW_SIZE
        # resultados (tick bueno/malo) por agente, para contar "buenos en
        # la ventana" en vez de exigir racha perfecta.
        from collections import deque
        self.recent_ticks_ok = [deque(maxlen=self.STALL_WINDOW_SIZE) for _ in range(2)]
        self._last_action = [(0.0, 0.0, 0), (0.0, 0.0, 0)]  # (v_l, v_r, signal) cacheado entre decisiones
        self._last_sensors = [np.zeros(8), np.zeros(8)]

    # ------------------------------------------------------------
    def reset(self, seed_a=None, seed_b=None, target_pos=None, world_name=None,
              spawn_a=None, spawn_b=None):
        """Reposiciona los robots (sin recargar URDF). Si world_name difiere
        del mundo actual, cambia de mapa primero (clear_world + build_world).

        spawn_a/spawn_b: opcional, tupla ([x,y,z], yaw) para fijar la
        posicion exacta de arranque (para escenarios controlados, ej. un
        robot forzado a nacer detras de un obstaculo). Si se pasan, tienen
        prioridad sobre seed_a/seed_b para ese robot."""
        if world_name is not None:
            self.switch_world(world_name)

        if seed_a is not None: self.rng_a = _random_module.Random(seed_a)
        if seed_b is not None: self.rng_b = _random_module.Random(seed_b)
        if target_pos is not None: self.target_pos = target_pos

        pos_a, yaw_a = spawn_a if spawn_a is not None else self._sample_valid_spawn(self.rng_a)
        pos_b, yaw_b = spawn_b if spawn_b is not None else self._sample_valid_spawn(self.rng_b)

        p.resetBasePositionAndOrientation(self.robot_id_a, pos_a, p.getQuaternionFromEuler([0, 0, yaw_a]))
        p.resetBaseVelocity(self.robot_id_a, [0, 0, 0], [0, 0, 0])
        p.resetBasePositionAndOrientation(self.robot_id_b, pos_b, p.getQuaternionFromEuler([0, 0, yaw_b]))
        p.resetBaseVelocity(self.robot_id_b, [0, 0, 0], [0, 0, 0])

        self._reset_state()

    def step(self, controllers, signals, mute_social=None, h_prob=0.0, reverse_order=False):
        """
        Un paso de FISICA (no de decision). Decide accion nueva solo cada
        `brain_ratio` pasos (igual que essim3d.py); en los pasos intermedios
        reusa la ultima accion calculada.

        controllers: [ctrl_a, ctrl_b]  (Unified3DController)
        signals:     [signal_a, signal_b]  señal social del paso anterior
                     (se actualiza in-place con la nueva señal si hubo
                     decision nueva en este paso)
        mute_social: opcional, [bool, bool]. Si mute_social[i]=True, ese
                     agente SIEMPRE percibe social_pulse=0 (nunca "escucha"
                     al otro), sin importar lo que el otro emita realmente.
                     Para ablacion: aislar si el receptor usa la señal o no,
                     sin tocar la emision del otro agente.
        h_prob:      prob. de annealing (Cortex y Annealing, ver
                     controllers.py). Default 0.0 = sin cambio de
                     comportamiento para callers que no lo pasan (p.ej.
                     essim3d.py en modo visualizacion interactiva).
        reverse_order: sesion 2026-08-14, diagnostico del sesgo de slot.
                     Invierte el ORDEN DE PROCESAMIENTO del loop de
                     decision (agente B antes que A), sin tocar el
                     mapeo self.robots=[robot_id_a, robot_id_b] ni el
                     indice que usa el resto del codigo -- analogo a
                     swap_creation_order pero para el orden de computo,
                     no de creacion de cuerpos en pybullet. Default
                     False = identico al comportamiento de siempre.
                     Ya existia como flag en essim3d.py (interactivo)
                     pero nunca se habia podido medir con el harness
                     real (test_physics_gru.py no lo tenia) -- pendiente
                     del handoff original, seccion del sesgo de slot.

        Devuelve una lista de dicts, uno por agente:
          {pos, yaw, dist, signal, act_id, stalled, success}
        """
        decide_now = (self.step_count % self.brain_ratio == 0)
        mute = mute_social if mute_social is not None else [False, False]

        if decide_now:
            order = [1, 0] if reverse_order else [0, 1]
            new_actions = {}
            new_sensors = {}
            for i in order:
                sensors = self._get_sensors(self.robots[i])
                new_sensors[i] = sensors
                pos, ori = p.getBasePositionAndOrientation(self.robots[i])
                yaw = p.getEulerFromQuaternion(ori)[2]
                other_pos, _ = p.getBasePositionAndOrientation(self.robots[1 - i])
                social_pulse = 0 if mute[i] else signals[1 - i]
                v_l, v_r, signal, _x = controllers[i].get_action(
                    sensors, pos, yaw, self.target_pos, self.stalled[i],
                    social_pulse, other_pos=other_pos, h_prob=h_prob
                )
                new_actions[i] = (v_l, v_r, signal)
            self._last_action = [new_actions[0], new_actions[1]]
            self._last_sensors = [new_sensors[0], new_sensors[1]]
            for i in order:
                signals[i] = new_actions[i][2]

        out = []
        for i, rid in enumerate(self.robots):
            pos, ori = p.getBasePositionAndOrientation(rid)
            yaw = p.getEulerFromQuaternion(ori)[2]
            dist = float(np.linalg.norm(np.array(pos[:2]) - np.array(self.target_pos[:2])))
            success = dist < self.success_threshold

            v_l, v_r, signal = self._last_action[i]
            if success:
                self.paused_agents[i] = True
                v_l_final, v_r_final, f_motor = 0.0, 0.0, 0.0
                signal = 4
            else:
                self.paused_agents[i] = False
                v_l_final, v_r_final, f_motor = v_l, v_r, 1.5

            p.setJointMotorControl2(rid, self.left_wheels[rid], p.VELOCITY_CONTROL,
                                     targetVelocity=v_l_final / self.w_r, force=f_motor)
            p.setJointMotorControl2(rid, self.right_wheels[rid], p.VELOCITY_CONTROL,
                                     targetVelocity=v_r_final / self.w_r, force=f_motor)

            out.append({"pos": pos, "yaw": yaw, "dist": dist, "signal": signal,
                        "success": success, "sensors": self._last_sensors[i]})

        p.stepSimulation()
        self.step_count += 1

        for i, rid in enumerate(self.robots):
            curr_pos, _ = p.getBasePositionAndOrientation(rid)
            if self.prev_positions[i] is not None:
                disp = np.linalg.norm(np.array(curr_pos) - np.array(self.prev_positions[i]))
                v_l, v_r, _ = self._last_action[i]
                cmd_moving = (abs(v_l) + abs(v_r)) > 0.08
                stuck_now = cmd_moving and disp < 0.0001
                if stuck_now:
                    self.stall_counts[i] += 1; self.unstall_counts[i] = 0
                else:
                    self.unstall_counts[i] += 1; self.stall_counts[i] = 0

                # v17: ventana tolerante para LIBERAR (des-marcar stalled).
                # ANTES: unstall_counts>=30 exigia racha perfecta (un tick
                # malo la reseteaba a 0). AHORA: se cuenta cuantos de los
                # ultimos STALL_WINDOW_SIZE ticks fueron "buenos" (no
                # stuck_now) -- si >= STALL_WINDOW_MIN_GOOD, se libera,
                # tolerando algunos ticks malos sueltos dentro de la
                # ventana sin resetear todo el progreso acumulado.
                self.recent_ticks_ok[i].append(not stuck_now)
                good_in_window = sum(self.recent_ticks_ok[i])

                if self.stall_counts[i] >= 25:
                    self.stalled[i] = True
                elif good_in_window >= self.STALL_WINDOW_MIN_GOOD and len(self.recent_ticks_ok[i]) >= self.STALL_WINDOW_MIN_GOOD:
                    self.stalled[i] = False
            self.prev_positions[i] = curr_pos
            out[i]["stalled"] = self.stalled[i]

        return out

    def close(self):
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    # Smoke test: reset/step con Unified3DController (pesos random), y
    # tambien un cambio de mundo (switch_world) para confirmar que no
    # deja basura fisica ni revienta.
    import time
    from controllers import Unified3DController

    env = Foraging3DEnv(world_name="random_obstacles", num_threads=1, brain_ratio=15)

    in_s, out_s, h = 11, 5, 16
    dummy_weights = {
        "w_gru": np.random.randn(in_s + h, 3 * h).tolist(),
        "b_gru": np.zeros(3 * h).tolist(),
        "w_out": np.random.randn(h, out_s).tolist(),
        "b_out": np.zeros(out_s).tolist(),
        "w_res": np.random.randn(in_s, out_s).tolist(),
    }
    weights = {"brain_a": dummy_weights, "brain_b": dummy_weights}
    ctrl_a = Unified3DController(env.cfg, weights, agent_type="brain_a")
    ctrl_b = Unified3DController(env.cfg, weights, agent_type="brain_b")

    env.reset(seed_a=0, seed_b=1)
    signals = [0, 0]
    N = 1000
    t0 = time.time()
    for _ in range(N):
        env.step([ctrl_a, ctrl_b], signals)
    dt = time.time() - t0
    print(f"[SMOKE] {N} pasos de fisica en {dt:.2f}s ({N/dt:.0f} pasos/seg)")

    # Probar cambio de mundo: default -> u_shape -> n_shape -> default
    t0 = time.time()
    env.reset(seed_a=2, seed_b=3, world_name="u_shape")
    env.reset(seed_a=4, seed_b=5, world_name="n_shape")
    env.reset(seed_a=6, seed_b=7, world_name="random_obstacles")
    dt_switch = time.time() - t0
    print(f"[SMOKE] 3 cambios de mundo en {dt_switch:.3f}s "
          f"({dt_switch/3*1000:.1f} ms c/u)")
    print(f"[SMOKE] target_pos actual: {env.target_pos} (deberia ser el de random_obstacles)")

    # Correr unos pasos mas post-switch para confirmar que la fisica sigue sana
    for _ in range(200):
        env.step([ctrl_a, ctrl_b], signals)
    print("[SMOKE] 200 pasos post-switch OK, sin excepciones.")

    env.close()
