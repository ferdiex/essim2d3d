"""
Fitness de Fase 1b (evolucion del canal social) para 3D.

Porta evaluate_individual() de train_social2d.py (2D) a Foraging3DEnv.

Decisiones de diseño acordadas en la sesion del 2026-08-07:
  - Curriculum de mapas: IGUAL al comportamiento REAL de 2D (no la intencion
    documentada en mensajes viejos) -> los 3 mapas (random_obstacles,
    u_shape, n_shape) ciclan siempre, desde la generacion 0. Episodios
    0-2 -> random_obstacles, 3-5 -> u_shape, 6-8 -> n_shape. El parametro
    "u_mode"/gen>=30 de 2D era dead code, no se replica.
  - Deteccion de atasco: se usa env.stalled (calibrado a la escala real de
    3D, 25 pasos consecutivos con desplazamiento < 0.0001) EN VEZ DE los
    umbrales de 2D (dist_moved < 0.5 / < 1.1), que estaban calibrados a
    una escala de mundo ~100x mas grande y no aplican aca.
  - Constantes de fitness recalibradas EMPIRICAMENTE con evaluate3d.py
    --debug (no analiticamente, las escalas 2D/3D no son comparables):
      * DIST_MULT subido de 2000 a 150,000 (el original era ~75x chico).
      * FAIL_PENALTY bajado de -300,000 a -60,000 (el original ahogaba
        cualquier shaping de distancia, sin importar DIST_MULT).
      * SACADO el floor `max(0.1, ...)` del retorno: con fitness promedio
        negativo (la mayoria de la poblacion en generaciones tempranas),
        ese floor volvia CASI TODOS los individuos identicos (0.1),
        borrando toda diferencia y dejando la seleccion por ranking
        (argsort) practicamente aleatoria entre ellos. Este era el bug
        real detras del estancamiento observado (BEST/AVG en 0 la
        mayoria de las generaciones), mas alla de la calibracion de
        constantes.
"""
import numpy as np
import pybullet as p

from foraging_env3d import Foraging3DEnv
from controllers import Unified3DController

EPISODES = 9
MAX_STEPS = 2000  # pasos de FISICA por episodio (no decisiones; ver brain_ratio)

# --- Constantes de fitness, recalibradas EMPIRICAMENTE a la escala de 3D ---
DIST_MULT = 150_000.0       # CALIBRADO empiricamente (gen 0-10 test): con 2000 el shaping
                             # quedaba ~75x mas chico que FAIL_PENALTY, sin gradiente util
FRONTAL_UNBLOCK_BONUS = 100.0   # antes 50.0 -- 2D usa 100.0/-50.0
# (train_social2d.py linea 133-135). 3D estaba a la mitad sin razon
# documentada (ver handoff 2026-08-13). Recalibrado a la magnitud real
# de 2D, no solo la proporcion -- ambos escalados x2.
FRONTAL_FORWARD_PENALTY = -50.0  # antes -25.0
STUCK_PENALTY = -100.0
SOCIAL_RESCUE_BONUS = 20000.0   # antes 500.0 (x40, sesion 2026-08-13):
# 500 era ~6000x mas chico que TERMINAL_BONUS_BASE, y menos que una sola
# decision de navegacion normal via DIST_MULT -- incentivo invisible en
# la practica. Subido a un orden comparable a lo que DIST_MULT ya
# acumula en varias decisiones, sin acercarse al terminal.
SOCIAL_RESCUE_DIST_DELTA = 0.05  # CALIBRAR (sesion 2026-08-11, v8): antes umbral sobre
# acercamiento crudo al companero; ahora umbral sobre la DIFERENCIA entre acercamiento
# al companero y acercamiento a la comida (ver seccion del bono). Mismo valor 0.05
# heredado sin verificar contra la nueva formula -- revisar/ajustar con datos de v8
# (si el bono nunca se paga, bajar; si se paga casi siempre, subir).
RECRUIT_BONUS = 750.0        # x5 (sesion 2026-08-07): refuerzo del Ganglio Basal
                              # antes de Fase 2, sin tocar las probabilidades 70%/20%
ABANDON_PENALTY = -2500.0    # x5, mismo criterio
TERMINAL_BONUS_BASE = 3_000_000.0  # CALIBRAR: mismo orden que 2D a proposito (domina el fitness)
TERMINAL_GAP_BONUS_PER_STEP = 10_000.0
TERMINAL_GAP_THRESHOLD = 11  # PP (sesion 2026-08-13): antes 40, copiado sin
# recalibrar del gap<40 de train_social2d.py (2D). En 2D, 40 sobre 500 pasos
# (1 paso=1 decision) es ~8% de la duracion del episodio. En 3D, con
# max_steps=2000 y brain_ratio=15 (~133 decisiones/episodio), el mismo 40
# representaba ~30% -- casi 4x mas permisivo, incentivo mas debil a llegar
# realmente sincronizados. 11 ~ 8% de 133, iguala la exigencia proporcional
# real de 2D.
FAIL_PENALTY = -60_000.0    # CALIBRADO empiricamente (ver debug de evaluate3d.py): con -300,000
                             # ahogaba por completo el shaping de distancia, sin importar DIST_MULT
FRONTAL_SENSOR_THRESHOLD = 0.85


def world_for_episode(ep_idx):
    if ep_idx < 3:
        return "random_obstacles"
    elif ep_idx < 6:
        return "u_shape"
    else:
        return "n_shape"


def evaluate_individual_3d(weights_json, h_prob, use_bg, rand_spawns=True,
                            max_steps=MAX_STEPS, episodes=EPISODES, env=None,
                            episode_seeds=None, debug=False, swap_slots=False):
    """
    weights_json: dict con 'brain_a'/'brain_b' (formato de save_best_model/
                  load_weights de train_social2d.py).
    h_prob: prob. de exploracion aleatoria (annealing), igual que 2D.
    use_bg: activa/desactiva el Ganglio Basal (instintos).
    env:    Foraging3DEnv ya creado (para reusar entre individuos dentro del
            mismo worker de multiprocessing). Si None, crea uno nuevo (mas
            lento, solo para pruebas sueltas).
    episode_seeds: lista de (seed_a, seed_b), una por episodio, para fijar
            los spawns. IMPORTANTE: pasar las MISMAS semillas para todos los
            individuos de una misma generacion, si no las comparaciones de
            fitness dentro de la generacion no son justas.
    debug: si True, imprime desglose por episodio (mundo, exito, distancia
            cerrada, fitness aportado).
    swap_slots: DIAGNOSTICO sesion 2026-08-11 -- si True, brain_b ocupa el
            slot fisico 0 y brain_a el slot fisico 1 (al reves del default).
            Se confirmo con datos que el GRU desarrolla un comportamiento
            que resuelve mejor desde el slot fisico 1 (no es de los pesos,
            no es del spawn, no es del motor de fisica -- ver handoff de
            la sesion) -- sin alternar, la evolucion tiene un incentivo
            gratis para explotar esa ventaja de posicion en vez de aprender
            coordinacion real por señal. Alternar swap_slots entre
            repeticiones de evaluacion (ver train_social3d.py) reparte
            el incentivo entre los dos slots por igual.

    NOTA (misma sesion, después de mute_social, ACTUALIZADO tras sección
    AA/BB del handoff): bearing_fix se probó activo (hardcodeado True)
    primero -- resultó insuficiente por sí solo (confirmado con
    --mute_social, sin diferencia funcional). La causa real resultó ser
    que x[10] (el índice que 2D reserva para la señal social) recibía
    is_stuck en 3D en vez de la señal real. Arreglado en controllers.py:
    x[10] ahora es la señal social real (fiel a 2D), x[8] volvió a ser
    siempre bearing a la comida. El parámetro bearing_fix ya no existe
    en Unified3DController -- se sacó de punta a punta (controllers.py,
    essim3d.py, y estas 4 construcciones) para no dejar un flag fantasma
    que no hace nada. Ver handoff, secciones Y-BB, para el detalle
    completo.

    Devuelve (fitness_promedio, screams_promedio, stuck_promedio). SIN floor:
    puede ser negativo, y debe serlo cuando corresponda para que la seleccion
    por ranking tenga informacion real entre individuos que fallan.
    """
    own_env = env is None
    if own_env:
        env = Foraging3DEnv(world_name="random_obstacles")

    if swap_slots:
        ctrl_a = Unified3DController(env.cfg, weights_json, agent_type="brain_b", use_bg=use_bg)
        ctrl_b = Unified3DController(env.cfg, weights_json, agent_type="brain_a", use_bg=use_bg)
    else:
        ctrl_a = Unified3DController(env.cfg, weights_json, agent_type="brain_a", use_bg=use_bg)
        ctrl_b = Unified3DController(env.cfg, weights_json, agent_type="brain_b", use_bg=use_bg)
    controllers = [ctrl_a, ctrl_b]

    total_f, total_vocal, total_stuck, success_episodes = 0.0, 0, 0, 0

    try:
        for ep_idx in range(episodes):
            f_before_episode = total_f
            world_name = world_for_episode(ep_idx)
            if episode_seeds is not None:
                seed_a, seed_b = episode_seeds[ep_idx]
                env.reset(world_name=world_name, seed_a=seed_a, seed_b=seed_b)
            else:
                env.reset(world_name=world_name)
            ctrl_a.reset(); ctrl_b.reset()

            signals = [0, 0]
            success_step = [-1, -1]

            pa, _ = p.getBasePositionAndOrientation(env.robot_id_a)
            pb, _ = p.getBasePositionAndOrientation(env.robot_id_b)
            p_pre = [np.array(pa[:2]), np.array(pb[:2])]
            d_start = [float(np.linalg.norm(p_pre[i] - np.array(env.target_pos[:2]))) for i in range(2)]

            at = [False, False]  # individual_success acumulado (una vez True, queda True)
            num_decisions = max_steps // env.brain_ratio

            for dec in range(num_decisions):
                d_pre = [float(np.linalg.norm(p_pre[i] - np.array(env.target_pos[:2]))) for i in range(2)]

                out = None
                for _tick in range(env.brain_ratio):
                    out = env.step(controllers, signals, h_prob=h_prob)

                cur_pos = [np.array(out[i]["pos"][:2]) for i in range(2)]

                for i in range(2):
                    if out[i]["success"]:
                        at[i] = True
                        if success_step[i] == -1:
                            success_step[i] = dec

                # NN + Cortex/Annealing (sesion 2026-08-13): antes vivian
                # aca, a medias (NN solo fingia signals[i] para el receptor,
                # nunca tocaba la decision propia; el annealing era un
                # `pass`, no hacia nada). Ahora los dos viven DENTRO de
                # Unified3DController.get_action() (mismo lugar que el
                # 20% de atasco), portados fieles a train_social2d.py --
                # signals[i] ya sale bien seteado desde
                # Foraging3DEnv.step() (via new_actions[i][2] = act_id real
                # del controlador, que ya incluye los tres instintos).
                # Nada que hacer aca -- ver controllers.py.
                for i in range(2):
                    total_vocal += 1 if signals[i] == 4 else 0

                for i in range(2):
                    other = 1 - i
                    dist_now = out[i]["dist"]
                    sensors_i = out[i]["sensors"]
                    p_frente = max(sensors_i[0], sensors_i[1], sensors_i[7])

                    if p_frente > FRONTAL_SENSOR_THRESHOLD:
                        if out[i]["signal"] in (1, 2):
                            total_f += FRONTAL_UNBLOCK_BONUS
                        elif out[i]["signal"] == 0:
                            total_f += FRONTAL_FORWARD_PENALTY

                    if not at[i]:
                        acercamiento_comida = d_pre[i] - dist_now
                        total_f += acercamiento_comida * DIST_MULT
                        if out[i]["stalled"]:
                            total_f += STUCK_PENALTY
                            total_stuck += 1
                        if at[other] and signals[other] == 4:
                            d_soc_pre = float(np.linalg.norm(p_pre[i] - cur_pos[other]))
                            d_soc_post = float(np.linalg.norm(cur_pos[i] - cur_pos[other]))
                            acercamiento_companero = d_soc_pre - d_soc_post
                            # ARREGLO (sesion 2026-08-11, v8): antes se pagaba
                            # este bono con solo acercarse al companero --
                            # pero el companero que ya llego ESTA en la
                            # comida, asi que acercarse a la comida (que ya
                            # se hace siempre, x[8] es bearing a la comida)
                            # y acercarse al companero eran casi la misma
                            # accion. Se cobraba gratis, sin usar x[10] para
                            # nada. Ahora exige que el acercamiento al
                            # companero sea MAYOR que el acercamiento a la
                            # comida en la misma ventana -- evidencia real
                            # de una correccion hacia el companero
                            # especificamente, no un efecto colateral de ir
                            # a donde ya se iba.
                            if (acercamiento_companero - acercamiento_comida) > SOCIAL_RESCUE_DIST_DELTA:
                                total_f += SOCIAL_RESCUE_BONUS
                    else:
                        if not at[other]:
                            if signals[i] == 4:
                                total_f += RECRUIT_BONUS
                            else:
                                total_f += ABANDON_PENALTY

                p_pre = cur_pos

                if all(at):
                    break

            if all(at):
                success_episodes += 1
                gap = abs(success_step[0] - success_step[1])
                mult = 1.0 if gap < TERMINAL_GAP_THRESHOLD else 0.3
                total_f += (TERMINAL_BONUS_BASE + (500 - gap) * TERMINAL_GAP_BONUS_PER_STEP) * mult
            else:
                total_f += FAIL_PENALTY

            if debug:
                d_end = [float(np.linalg.norm(p_pre[i] - np.array(env.target_pos[:2]))) for i in range(2)]
                closed = [d_start[i] - d_end[i] for i in range(2)]
                print(f"  [ep {ep_idx}] world={world_name:16s} at={at} "
                      f"d_start={[round(x,3) for x in d_start]} d_end={[round(x,3) for x in d_end]} "
                      f"closed={[round(x,3) for x in closed]} "
                      f"fitness_este_episodio={total_f - f_before_episode:,.0f}")

    finally:
        if own_env:
            env.close()

    return (total_f / episodes, total_vocal / episodes, total_stuck / episodes, success_episodes)


if __name__ == "__main__":
    import json
    import time

    with open("nav_brain.json") as f:
        weights = json.load(f)

    env = Foraging3DEnv(world_name="random_obstacles")
    t0 = time.time()
    fit, vocal, stuck, succ = evaluate_individual_3d(
        weights, h_prob=0.1, use_bg=True, max_steps=2000, episodes=9, env=env, debug=True
    )
    dt = time.time() - t0
    print(f"[SMOKE] fitness={fit:.1f} vocal={vocal:.1f} stuck={stuck:.1f} success_episodes={succ}/9 ({dt:.2f}s / 9 episodios)")
    env.close()
