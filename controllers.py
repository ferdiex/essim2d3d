import numpy as np
import random
import json

# ============================================================
#  SECCION 3D  (usada por essim3d.py)
# ============================================================

# DIAGNOSTICO REVERSIBLE (sesion 2026-08-11): v9 (FF+GG+HH juntos) mostro
# regresion fuerte (exitos cayeron a menos de la mitad de v8, giros raros
# en el propio eje reportados visualmente). Flags para aislar si es FF
# sola o la combinacion con GG/HH -- NO se borro ni comento la logica de
# GG/HH, solo se las envuelve en un condicional facil de revertir
# (poner en True de nuevo). Default de esta prueba: GG y HH apagados,
# FF sigue activa (no tiene flag, no es opcional -- es la formula
# correcta del GRU, no un mecanismo nuevo que se pueda "apagar").
ENABLE_SOCIAL_FILTER = False  # GG -- apagada, GG sola (v11) ya se probo y
# no mostro diferencia en --mute_social (3 checkpoints, ver handoff
# 2026-08-13, seccion v11). Se apaga para probar HH sola, aislada.
ENABLE_BEACON = False          # HH -- apagada de nuevo (colapso confirmado
# en v12, ver handoff). Se prueba SOCIAL_RESIDUAL_BOOST aislado, sin GG
# ni HH, para no confundir el efecto del boost con el de otro mecanismo.

# SOCIAL_RESIDUAL_BOOST (sesion 2026-08-13, diagnostico x[10]): amplifica
# SOLO la contribucion de x[10] en el camino residual (w_res, entrada
# directa x->logits sin pasar por la memoria del GRU). No toca x[10] en
# si (no afecta el chequeo de GG `abs(x[10])>0.05`, ni HH que no usa
# x[10]) -- solo el peso de su aporte a los logits. Motivado por
# diagnose_x10_incontext.py: en contexto real, el efecto de x[10] sobre
# el gap giro_der-giro_izq es +3.48 (moviendo x10 de -0.8 a +0.8),
# comparado con std=6.14 de variabilidad natural del gap -- la señal
# compite en desventaja (~mitad del ruido de fondo). 1.0 = sin cambio
# (comportamiento identico a antes). Boost>1.0 SOLO tiene efecto util si
# se re-mide con diagnose_x10_incontext.py antes de reevolucionar --
# aplicarlo sobre un checkpoint YA entrenado no lo hace usar la señal
# mejor de golpe (los pesos de w_out/memoria no cambiaron), solo hace
# que el camino residual directo pese mas -- el efecto real requiere
# reevolucionar con el boost activo para que los pesos se adapten.
SOCIAL_RESIDUAL_BOOST = 3.0  # activo para la corrida completa v15 (paquete
# final: boost + bono + recruit_prob + FRONTAL_* recalibrado, todo junto)

# JALON_ATTENUATION (sesion 2026-08-14): parche directo para el "jalon"
# confirmado con datos (diagnose_jalones.py: +22.7pp de cambio de accion,
# progreso neto NEGATIVO en conflicto) y confirmado CAUSALMENTE en vivo
# por el usuario (mover al agente que señaliza lejos de la comida
# desatasca al otro de inmediato). No es un rediseño del mecanismo de
# arbitraje -- es una regla de prioridad simple puesta encima, mismo
# estilo que GG/HH: si x[8] (comida) y x[10] (señal) apuntan en sentidos
# opuestos con magnitud significativa, se atenua el aporte de x[10] al
# camino residual (no se apaga del todo -- la señal sigue "sonando" mas
# debil, la comida gana el desempate). Mismo umbral de magnitud que ya
# valido diagnose_jalones.py como el que mostro efecto real (0.05).
JALON_CONFLICT_MIN_MAG = 0.05
JALON_ATTENUATION_FACTOR = 1.0  # sesion 2026-08-14: APAGADO. Se probaron
# 0.3 (bajo el costo de eficiencia pero no elimino la indecision) y 0.0
# (elimino la indecision pero el progreso empeoro mas que sin parche --
# probable desfase entre la memoria h, que ya acumulo la influencia de
# la señal, y el corte abrupto del input actual). Ninguno de los dos
# supero claramente al baseline en progreso neto. Se revierte, se vuelve
# a v15 como version de referencia mientras se piensa una solucion
# distinta (no de recalibracion de este mismo numero).

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class GRUNetwork:
    """Raw GRU implementation: Input(11) + Hidden(16) = 27."""
    def __init__(self, weights):
        self.w_gru = np.array(weights['w_gru'])
        self.b_gru = np.array(weights['b_gru'])
        self.w_out = np.array(weights['w_out'])
        self.b_out = np.array(weights['b_out'])
        self.w_res = np.array(weights['w_res'])
        self.hidden_dim = 16
        self.h = np.zeros(self.hidden_dim)

    def reset(self):
        self.h = np.zeros(self.hidden_dim)

    def forward(self, x):
        combined = np.concatenate([x, self.h])
        gates = np.dot(combined, self.w_gru) + self.b_gru
        z = sigmoid(gates[:16])
        r = sigmoid(gates[16:32])
        # FF (sesion 2026-08-11): formula GRU corregida. Antes:
        # n = tanh(gates[32:] * r) -- tomaba un resultado YA calculado
        # (con h sin resetear, de la misma matmul de z/r) y lo
        # multiplicaba por r despues. No es la formula estandar de GRU, y
        # no coincide con GRUController.act() (2D, real, linea 384), que
        # aplica el reset gate sobre h ANTES de la matmul: recalcula con
        # [x, r*h] como entrada nueva, usando solo el ultimo bloque de
        # columnas de w_gru. Afecta CADA decision -- se replica exacto.
        # SIN FLAG: no es un mecanismo opcional nuevo, es la formula del
        # GRU en si -- no tiene sentido "aislarla apagandola", volver a
        # la version vieja seria reintroducir el bug a proposito.
        combined_reset = np.concatenate([x, r * self.h])
        n = np.tanh(np.dot(combined_reset, self.w_gru[:, 32:]) + self.b_gru[32:])
        self.h = (1 - z) * self.h + z * n

        near_wall = np.max(x[:8]) > 0.82
        multi = 0.3 if near_wall else 5.0
        res = np.dot(x, self.w_res)

        # JALON_ATTENUATION (sesion 2026-08-14): si x[8] (comida) y x[10]
        # (señal) tironean en sentidos opuestos con magnitud real, se
        # atenua el aporte de x[10] -- la comida gana el desempate. Se
        # aplica ANTES del boost, sobre el mismo termino, para que en
        # conflicto el boost tambien quede atenuado (no se anulan entre
        # si, se componen: boost normal x atenuacion = boost mas chico).
        in_conflict = (abs(x[8]) > JALON_CONFLICT_MIN_MAG and abs(x[10]) > JALON_CONFLICT_MIN_MAG
                       and np.sign(x[8]) != np.sign(x[10]))
        effective_boost = SOCIAL_RESIDUAL_BOOST * JALON_ATTENUATION_FACTOR if in_conflict else SOCIAL_RESIDUAL_BOOST

        # SOCIAL_RESIDUAL_BOOST (sesion 2026-08-13): amplifica SOLO el
        # aporte de x[10] a `res`, dejando todo lo demas (incluido x[10]
        # mismo, y las demas columnas de w_res) intacto. Con boost=1.0
        # esta linea no hace nada (0 * lo que sea = 0, se suma cero).
        res = res + x[10] * self.w_res[10, :] * (effective_boost - 1.0)
        logits = np.dot(self.h, self.w_out) + self.b_out + res * multi

        # GG (sesion 2026-08-11): filtro de atencion social, portado fiel
        # de GRUController.act() (2D, real, linea 393-403, rama activa --
        # no la version comentada que restringia a agent_idx==1). Si hay
        # pared cerca Y señal social activa (x[10]), penaliza fuerte
        # "seguir de frente" (accion 0). No fuerza el giro -- deja que la
        # red decida usando x[10] por si sola. Nunca existio en 3D.
        if ENABLE_SOCIAL_FILTER:
            social_active = abs(x[10]) > 0.05
            if near_wall and social_active:
                logits[0] -= 30.0

        # HH (sesion 2026-08-11): "El Faro", portado fiel de
        # GRUController.act() (2D, real, linea 405-408). Empuja la accion
        # de señalizar/wander cuando el hambre (x[9]) supera un umbral
        # (cerca de la comida) -- capa DENTRO del controlador, distinta y
        # adicional al instinto de "reclutar" que ya vive en la fitness
        # (evaluate3d.py). Nunca existio en 3D.
        #
        # UMBRAL RECALIBRADO (no 0.32 copiado de 2D): en 2D, odor=exp(-dist/40),
        # 0.32 dispara a dist~45.6 -- MAS CERCA que el propio umbral de exito
        # de 2D (65), o sea "El Faro" en 2D casi coincide con "ya llegue".
        # Copiar 0.32 a 3D (hunger_decay=1.0, escala de mundo mucho mas chica
        # y con proporcion decay/mundo distinta a la de 2D) dispara a
        # dist~1.14 -- ~76% del ancho del mundo, casi siempre activo (asi
        # se detecto: SCREAMS se disparo de ~20 a ~200+ en v9 con 0.32).
        # Recalibrado para disparar a ~el doble del radio de exito (0.16m,
        # con margen antes de llegar): x[9]=exp(-0.16/1.0)~0.85. No derivado
        # de hunger_decay en si (cambiarlo invalidaria todos los checkpoints
        # ya evolucionados con 1.0) -- solo el umbral de ESTE mecanismo.
        FARO_HUNGER_THRESHOLD = 0.85
        if ENABLE_BEACON and x[9] > FARO_HUNGER_THRESHOLD:
            logits[4] += 20.0

        return np.argmax(logits)

class Unified3DController:
    """Adapter: 3D Physics <-> 2D Logic."""
    def __init__(self, config, brain_weights, agent_type='brain_a', rng=None, hunger_gate=False, use_bg=True):
        self.max_speed = config['robot']['max_speed']
        self.brain = GRUNetwork(brain_weights[agent_type])
        # RNG propio por robot para el sorteo de stuck_act (antes: modulo
        # random global, compartido y orden-dependiente entre agente A y B).
        # Mismo patron que rng= en FusionController.
        self.rng = rng if rng is not None else random.Random()
        # PARCHE DE DIAGNOSTICO (apagado por default, no cambia el comportamiento
        # actual salvo que se active explicitamente): atenua el tiron de hambre
        # (x[9]) cuando el sensor frontal detecta algo cerca, mismo criterio
        # (x[0] < 0.2) que ya se usa para gatear la señal social. Sirve para
        # confirmar/descartar si el forcejeo hambre-vs-miedo es de fuerza
        # relativa de señales (si esto lo resuelve) o de como la red las
        # combina internamente (si no cambia nada). NO es una decision de
        # diseño final, solo un experimento reversible.
        self.hunger_gate = hunger_gate

        # SINTONÍA FINA: brain_b pondera el miedo ~1.78x más fuerte que
        # brain_a (medido en w_res). Compensamos amplificando su señal de
        # hambre de entrada, sin tocar los pesos evolucionados.
        self.hunger_decay = 1.25 if agent_type == 'brain_b' else 1.0

        self.action_table = {
            0: [1.0, 1.0],    # Forward
            1: [-0.3, 0.6],   # Left  -> pivote, casi sin traslación neta
            2: [0.6, -0.3],   # Right -> pivote, casi sin traslación neta
            3: [-0.5, -0.5],  # Reverse (sin cambios)
            4: [0.8, 0.8]     # Signal/Wander (sin cambios)
        }

        self.stuck_override_prob = 0.20  # Opcion 4: mismo 20% que essim2d.py, sin sostener nada
        self.stuck_streak = 0  # v16 (sesion 2026-08-14): cuenta decisiones SEGUIDAS
        # con is_stuck=True. Se resetea a 0 apenas is_stuck=False (no se acumula
        # a traves de momentos de navegacion normal). Usado para escalar la
        # probabilidad de escape -- ver ESCALATE_STUCK_THRESHOLD abajo.

        # NN (sesion 2026-08-13): "Instinto de Reclutamiento", portado de
        # train_social2d.py (2D) linea 104-106 -- antes solo vivia a medias en
        # evaluate3d.py (fingia el signals[i] que ve el receptor, nunca
        # tocaba la decision real del propio agente). Ahora vive aca, junto
        # al 20% de atasco, mismo patron. use_bg=True por default: en 2D
        # este instinto SIEMPRE esta activo (no depende de use_bg alli --
        # use_bg en 2D solo gatea el Faro dentro de GRUController.act()).
        # Se agrega el flag aca para preservar la comparacion "biological
        # impulse" vs "modo mudo" que ya usaba evaluate3d.py.
        self.use_bg = use_bg
        self.recruit_prob = 0.90  # antes 0.70 (sesion 2026-08-14): subido para
        # aumentar la disponibilidad real de la señal (~5% de las decisiones
        # traian señal, medido con diagnose_x10_incontext.py) -- no cambia el
        # mecanismo, solo la frecuencia con la que un agente que ya llego
        # efectivamente señaliza en vez de quedar en silencio ese 30% libre.

    def reset(self):
        self.brain.reset()
        self.stuck_streak = 0

    def get_action(self, sensors_3d, robot_pos, robot_yaw, target_pos, is_stuck, other_signal=0, other_pos=None, h_prob=0.0):
        # 1. Inputs
        x = np.zeros(11)
        x[:8] = np.clip(sensors_3d, 0.0, 1.0)

        # x[8] = bearing a la comida, SIEMPRE -- fiel a 2D (foraging_env.py,
        # rel_angle nunca se sobreescribe ahi). El bearing_fix viejo que
        # pisaba x[8] con la direccion al companero quedo sacado (sesion
        # 2026-08-11, seccion AA/BB del handoff): esa sustitucion no era
        # fiel a 2D -- 2D nunca pisaba rel_angle, mandaba la señal social
        # en su PROPIO indice (10), separado. Ver mas abajo.
        dx, dy = target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]
        rel_angle = (np.arctan2(dy, dx) - robot_yaw + np.pi) % (2 * np.pi) - np.pi
        x[8] = rel_angle / np.pi
        dist = np.sqrt(dx**2 + dy**2)

        x[9] = np.exp(-dist / self.hunger_decay)

        # PARCHE DE DIAGNOSTICO: mismo umbral (x[0] < 0.2) que el gating social
        # de abajo. Si hay algo cerca al frente, el hambre se apaga del todo
        # para esta decision -- ver comentario en __init__.
        if self.hunger_gate and x[0] >= 0.2:
            x[9] = 0.0

        # x[10] = señal social real (sesion 2026-08-11, hallazgo mayor,
        # seccion AA del handoff): este indice es el que 2D reservaba
        # exclusivamente para la señal social (foraging_env.py,
        # observation.append(social_signal), indice 10 con num_sensors=8).
        # Hasta hoy, este mismo indice en 3D recibia is_stuck -- un
        # booleano sin relacion, chocando con lo que los pesos heredados
        # de 2D aprendieron a interpretar ahi. is_stuck como INPUT de la
        # red no hace falta: el Ganglio Basal (mas abajo) lee el
        # parametro is_stuck crudo directamente, nunca paso por x.
        #
        # Formula identica a 2D (social_variant="angle_strength"):
        # angulo_al_companero * exp(-distancia/decay). decay=0.563
        # calibrado con datos reales de 3D (check_recruit_distance.py,
        # mediana de 70936 muestras de distancia robot-robot en ventanas
        # de reclutamiento real), no convertido a ciegas del 180.0 de 2D
        # (escala de mundo distinta, 800x600 vs 1.5x1.5).
        SOCIAL_SIGNAL_DECAY = 0.563  # sesion 2026-08-11, calibrado con
        # check_recruit_distance.py bajo success_threshold=0.08. Se probo
        # recalibrar a 0.554 tras subir el threshold a 0.1219 (sesion
        # 2026-08-14), pero esa exploracion se abandono (diferencia minima,
        # no ameritaba el costo de reevolucionar) y success_threshold volvio
        # a 0.08 -- este valor (0.563) es el que corresponde a la
        # configuracion real en uso.
        if other_signal == 4 and other_pos is not None:
            dx_s = other_pos[0] - robot_pos[0]
            dy_s = other_pos[1] - robot_pos[1]
            angle_to_sender = (np.arctan2(dy_s, dx_s) - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            dist_social = np.sqrt(dx_s**2 + dy_s**2)
            signal_strength = np.exp(-dist_social / SOCIAL_SIGNAL_DECAY)
            x[10] = (angle_to_sender / np.pi) * signal_strength
        else:
            x[10] = 0.0

        # 2. Ganglio Basal -- portado fiel de train_social2d.py (2D), lineas
        # 97-114 (evaluate_individual). 2D SIEMPRE calcula primero
        # actions_raw = c.act(obs), y DESPUES le superpone tres instintos en
        # orden de prioridad fijo. Antes, 3D solo tenia el segundo instinto
        # (Opcion 4 / auxilio) y ademas lo evaluaba ANTES de llamar a la red
        # (short-circuit), sin calcular nunca la decision propia cuando
        # is_stuck ganaba el sorteo. Ahora se calcula siempre primero, igual
        # que 2D, para que el tercer instinto (annealing) tenga sobre que
        # actuar.
        act_id_raw = self.brain.forward(x)

        # 2.a NN (sesion 2026-08-13): Instinto de Reclutamiento, 70% de
        # probabilidad, SOLO si el agente ya llego a la comida (at_goal).
        # Portado fiel de train_social2d.py linea 104-106
        # (`if at[i] and np.random.rand()<0.7: actions.append(4)`). ANTES en
        # 3D esto vivia a medias en evaluate3d.py: fingia el signals[i] que
        # ve el receptor (para fitness/x[10] del otro agente) pero nunca
        # tocaba la decision real de ESTE agente -- la red nunca tuvo que
        # aprender a decidir cuando gritar, porque incluso cuando "decidia"
        # no importaba, el 70% igual se imponia por fuera sin pasar por
        # act_id. Ahora si pasa por ac. Nota: at_goal usa el mismo umbral
        # de exito que Foraging3DEnv.step() (0.08m) -- una vez ahi, la
        # velocidad queda congelada en el motor de fisica igual en 2D y 3D
        # (confirmado, seccion "OO" del handoff se cae por esto), asi que
        # este instinto solo decide QUE SEÑAL emite el agente inmovil, no
        # si se mueve.
        SUCCESS_THRESHOLD = 0.08
        at_goal = dist < SUCCESS_THRESHOLD

        # 2.b Auxilio/atasco (Opcion 4, base sin cambios: 20% por decision,
        # ruido puntual, no secuestro, se re-evalua siempre).
        #
        # v16 (sesion 2026-08-14): ESCALADA CON MEMORIA. Diagnostico
        # (diagnose_trompo.py, checkpoint v15) mostro que el 20% fijo sin
        # memoria deja una cola larga sin resolver -- en brain_b, 35% de las
        # rachas de atasco duran >=30 decisiones (max observado: 133, casi
        # el episodio completo). Con 20% fijo, fallar 30 sorteos seguidos es
        # ~0.12% de probabilidad por puro azar -- una cola asi de larga y
        # frecuente sugiere que el empujon a veces no resuelve la situacion
        # fisica real (el robot vuelve a trabarse), no solo mala suerte del
        # sorteo. Se agrega una rampa MODERADA (ni tan conservadora que no
        # cambie nada, ni tan agresiva que se vuelva determinista):
        #   racha 1-8:  20% (sin cambio -- ahi resuelve el 73-88% de los casos)
        #   racha 9-13: sube linealmente 15pp por decision (35%, 50%, 65%, 80%, 95%)
        #   racha 14+:  95% (nunca 100%, para no eliminar toda variabilidad)
        # stuck_streak se resetea a 0 apenas is_stuck=False (no persiste
        # entre momentos de navegacion normal).
        if is_stuck:
            self.stuck_streak += 1
        else:
            self.stuck_streak = 0

        ESCALATE_STUCK_THRESHOLD = 8
        if self.stuck_streak <= ESCALATE_STUCK_THRESHOLD:
            effective_stuck_prob = self.stuck_override_prob
        else:
            extra = min(0.75, (self.stuck_streak - ESCALATE_STUCK_THRESHOLD) * 0.15)
            effective_stuck_prob = min(0.95, self.stuck_override_prob + extra)

        # 2.c Cortex y Annealing (sesion 2026-08-13): portado fiel de
        # train_social2d.py linea 111-112 (`elif actions_raw[i]==4 and
        # np.random.rand()<h_prob: actions.append(np.random.randint(0,4))`).
        # ANTES en 3D existia la variable h_prob (calculada igual, rampa
        # 0.3->0.0) pero la rama equivalente en evaluate3d.py era un `pass`
        # -- nunca reemplazaba nada. Mecanismo anti-fijacion: si la red
        # ELIGE señalizar por su cuenta, con prob. h_prob (alta al inicio,
        # decae con las generaciones) se la reemplaza por una accion
        # aleatoria de navegacion -- evita que "señalizar" se vuelva un
        # reflejo barato desde el arranque de la evolucion. h_prob=0.0 por
        # default (sin cambio de comportamiento si el caller no lo pasa,
        # p.ej. essim3d.py en modo visualizacion interactiva).
        if self.use_bg and at_goal and self.rng.random() < self.recruit_prob:
            act_id = 4
        elif is_stuck and self.rng.random() < effective_stuck_prob:
            act_id = 4
        elif act_id_raw == 4 and self.rng.random() < h_prob:
            act_id = self.rng.randint(0, 4)
        else:
            act_id = act_id_raw

        # 3. Salida (v_l, v_r, y el ID de acción para el canal social)
        v_l, v_r = self.action_table.get(act_id, [1.0, 1.0])
        return v_l * self.max_speed, v_r * self.max_speed, act_id, x

class FusionController:
    def __init__(self, config, controller_json, rng=None):
        self.max_speed = config['robot']['max_speed']
        # Cargar pesos o ceros si no existen
        self.weights_l = np.array(controller_json.get('left_weights', np.zeros(8)))
        self.weights_r = np.array(controller_json.get('right_weights', np.zeros(8)))
        self.bias_l = controller_json.get('left_bias', 0.3)
        self.bias_r = controller_json.get('right_bias', 0.3)
        self.target_threshold = 0.10 # 10cm: Se detiene al llegar
        self.escape_timer = 0
        self.escape_total_ticks = 0  # duracion total del evento de escape actual (ya no fija en 100)
        self.escape_direction = 1
        self.consecutive_stalls = 0
        # RNG propio del robot -- antes se usaba el modulo random global,
        # compartido entre los dos robots (consumia del mismo stream,
        # orden-dependiente entre A y B). Cada instancia ahora tiene el
        # suyo, mismo patron que rng_a/rng_b para el spawn en essim3d.py.
        self.rng = rng if rng is not None else random.Random()
        # Base de cuantos pasos son reversa pura antes de pivotar (CASO B).
        # El valor real por evento se aleatoriza y escala con reintentos
        # consecutivos, ver _plan_escape().
        self.reverse_phase_ticks = 30
        self.normal_mode_ticks = 0  # pasos seguidos en modo normal (no atascado, no escapando)
        self.debug = {"alpha": None, "seek_l": None, "seek_r": None,
                      "avoid_l": None, "avoid_r": None, "escape_timer": 0, "triggered": False}

    def _plan_escape(self):
        """Arma los parametros de un nuevo evento de escape: direccion,
        duracion de reversa y duracion total. La DIRECCION es alternante
        (pinballesco: si falla, invierte, como el juguete real) -- la
        DURACION es la que se aleatoriza y escala con consecutive_stalls
        (rompe la periodicidad exacta de 100 pasos que se veia en
        escape_events log, sin volver indecisa la direccion)."""
        first_attempt = (self.consecutive_stalls == 0)
        self.consecutive_stalls += 1
        n = min(self.consecutive_stalls, 4)  # tope de escalado, para no crecer sin limite

        if first_attempt:
            self.escape_direction = self.rng.choice([-1, 1])
        else:
            self.escape_direction *= -1  # alternancia real, como antes

        self.reverse_phase_ticks = self.rng.randint(20, 40) + (n - 1) * 15
        self.escape_total_ticks = self.reverse_phase_ticks + self.rng.randint(50, 90) + (n - 1) * 10
        self.escape_timer = self.escape_total_ticks

    def get_action(self, sensors, robot_pos, robot_yaw, target_pos, is_stuck=False):
        # 1. Initial control variables (Shielding
        seek_l = seek_r = 0.0
        reverse_kick = 0.0

        # 2. Basal Ganglia Activation (With Forced Alternation)
        triggered_this_call = False
        if is_stuck and self.escape_timer <= 0:
            self._plan_escape()
            triggered_this_call = True

        # 3. Basal Ganglia Logic (Action Selection)
        if self.escape_timer > 0:
            self.escape_timer -= 1
            self.normal_mode_ticks = 0  # sigue en escape, no cuenta como "normal sostenido"
            front_free = np.max(sensors) < 0.2

            # CASE B, fase 1: choque de frente, todavia dentro de la
            # ventana de reversa pura -> retrocede derecho, SIN girar y
            # SIN blend de avoid/seek (bypass total), para ganar distancia
            # real antes de intentar pivotar.
            if (not front_free) and self.escape_timer >= (self.escape_total_ticks - self.reverse_phase_ticks):
                self.debug = {"alpha": None, "seek_l": None, "seek_r": None,
                              "avoid_l": None, "avoid_r": None, "escape_timer": self.escape_timer,
                              "triggered": triggered_this_call, "phase": "reverse"}
                return -self.max_speed, -self.max_speed

            # CASE A: REAR HOOK (Free front) -> Pivot Turn
            if front_free:
                seek_l = self.max_speed * self.escape_direction
                seek_r = -self.max_speed * self.escape_direction
                seek_k = 0.0
                reverse_kick = -0.1

            # CASE B, fase 2: ya reverso lo suficiente -> ahora si pivotea
            else:
                seek_k = 0.0
                seek_l = self.max_speed * self.escape_direction
                seek_r = -self.max_speed * self.escape_direction
                reverse_kick = -0.05

            if not is_stuck:
                pass

        else:
            # NORMAL MODE: Compass active
            seek_k = 0.7
            reverse_kick = 0.0
            # Reset del contador de reintentos, pero solo tras modo normal
            # SOSTENIDO (30 pasos seguidos), no un solo instante -- un
            # respiro de un tick antes de volver a chocar contra lo mismo
            # no cuenta como "atasco resuelto", y resetear ahi anulaba el
            # escalado casi siempre (confirmado: intento_num rara vez
            # pasaba de 2 en el log real).
            self.normal_mode_ticks += 1
            if self.normal_mode_ticks >= 30:
                self.consecutive_stalls = 0

        # 4. Navigation Calculation (Only if the compass is not inhibited)
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist < self.target_threshold:
            self.debug = {"alpha": None, "seek_l": None, "seek_r": None,
                          "avoid_l": None, "avoid_r": None, "escape_timer": self.escape_timer,
                          "triggered": triggered_this_call}
            return 0.0, 0.0

        if seek_k > 0: # In normal mode
            angle_to_target = np.arctan2(dy, dx)
            angle_diff = (angle_to_target - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            seek_l = -angle_diff * seek_k
            seek_r = angle_diff * seek_k

        # 5. Braitenberg and Fusion
        avoid_l = float(np.dot(self.weights_l, sensors)) + self.bias_l
        avoid_r = float(np.dot(self.weights_r, sensors)) + self.bias_r

        # Zona muerta de reaccion: antes alpha_base = clip(sensor*2, 0, 1)
        # empezaba a reaccionar desde lectura=0.0 (cualquier cosa detectada,
        # ~2.7 cuerpos del robot en el borde del rango) y llegaba a evasion
        # completa en lectura=0.5 (~1.3 cuerpos). Confirmado con
        # escape_events log que disparaba a 1.5-2.7 cuerpos de la pared --
        # demasiado lejos para un carrito reactivo tipo juguete. Ahora: sin
        # reaccion hasta REACT_START (~1 cuerpo), evasion completa recien en
        # REACT_FULL (~0.5 cuerpo). Formula real del sensor:
        # lectura = 1 - distancia/rango (rango=0.2, cuerpo robot=0.075).
        REACT_START = 0.625  # lectura en ~1.0 cuerpo de distancia
        REACT_FULL = 0.812   # lectura en ~0.5 cuerpo de distancia
        s_max = np.max(sensors)
        if s_max <= REACT_START:
            alpha_base = 0.0
        else:
            alpha_base = np.clip((s_max - REACT_START) / (REACT_FULL - REACT_START), 0, 1)
        alpha = 0.0 if self.escape_timer > 10 else (0.35 * alpha_base if self.escape_timer > 0 else alpha_base)

        v_l = (1 - alpha) * (self.max_speed + seek_l) + (alpha * avoid_l) + reverse_kick
        v_r = (1 - alpha) * (self.max_speed + seek_r) + (alpha * avoid_r) + reverse_kick

        # Desempate de simetria: cuando el obstaculo queda justo en linea
        # recta con la comida, seek Y avoid quedan practicamente iguales
        # para las dos ruedas (empate matematico real, ver discusion) -- sin
        # ninguna asimetria, el robot no tiene forma de "elegir" un lado.
        # Sesgo minimo y constante (1% de max_speed), siempre presente pero
        # despreciable frente a cualquier diferencial real de seek/avoid;
        # solo decide cuando todo lo demas esta empatado. No es una regla
        # nueva de decision, es reactivo igual -- solo inclina la balanza en
        # el caso patologico en vez de dejarlo al azar numerico.
        TIE_BREAK_BIAS = 0.01 * self.max_speed
        v_l -= TIE_BREAK_BIAS
        v_r += TIE_BREAK_BIAS

        # Escalado proporcional en vez de recorte independiente: si CUALQUIERA
        # de las dos ruedas se pasa de max_speed, achicar las DOS por el
        # mismo factor, para conservar la proporcion entre ellas. Antes cada
        # rueda se recortaba por separado (np.clip individual) -- eso rompe
        # la proporcion y, con angulos grandes hacia el objetivo (>~24.5 con
        # seek_k=0.7), deja las dos ruedas en extremos opuestos: giro puro en
        # el lugar, sin avance neto, en vez de avanzar en curva. Confirmado
        # analiticamente (24.5 grados) y es la explicacion del "baile"
        # nervioso sin ningun obstaculo cerca.
        peak = max(abs(v_l), abs(v_r), self.max_speed)
        scale = self.max_speed / peak
        v_l *= scale
        v_r *= scale

        # Diagnostico: valores internos del ultimo get_action(), para poder
        # loguear "por que" desde afuera (essim3d.py) sin duplicar la formula.
        self.debug = {"alpha": alpha, "seek_l": seek_l, "seek_r": seek_r,
                      "avoid_l": avoid_l, "avoid_r": avoid_r, "escape_timer": self.escape_timer,
                      "triggered": triggered_this_call}

        return v_l, v_r


# ============================================================
#  SECCION 2D  (usada por essim2d.py)
# ============================================================

class MLPController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.w1 = np.zeros((self.in_size, self.h_size))
        self.b1 = np.zeros(self.h_size)
        self.w2 = np.zeros((self.h_size, self.out_size))
        self.b2 = np.zeros(self.out_size)
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        h = np.maximum(0, x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return int(np.argmax(out))

    def reset(self): pass

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']
            self.w1 = np.array(data['w1'])
            self.b1 = np.array(data['b1'])
            self.w2 = np.array(data['w2'])
            self.b2 = np.array(data['b2'])

class GRUController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0, use_social_filter=True):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.use_bg = True
        self.use_social_filter = use_social_filter
        self.w_gru = np.zeros((self.in_size + self.h_size, 3 * self.h_size))
        self.b_gru = np.zeros(3 * self.h_size)
        self.w_out = np.zeros((self.h_size, self.out_size))
        self.b_out = np.zeros(self.out_size)
        # MATRIZ RESIDUAL: Conexión directa (Res-GRU)
        self.w_res = np.zeros((self.in_size, self.out_size))
        self.h = np.zeros((1, self.h_size))
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        concat = np.column_stack([x, self.h])
        gates = concat @ self.w_gru + self.b_gru

        z = 1.0 / (1.0 + np.exp(-gates[:, :self.h_size]))
        r = 1.0 / (1.0 + np.exp(-gates[:, self.h_size:2*self.h_size]))
        h_tilde = np.tanh(np.column_stack([x, r * self.h]) @ self.w_gru[:, 2*self.h_size:] + self.b_gru[2*self.h_size:])
        self.h = (1 - z) * self.h + z * h_tilde

        # 1. Cálculo de salida limpia (GRU + Residual)
        out_gru = self.h @ self.w_out + self.b_out
        near_wall = np.max(x[0, :8]) > 0.82
        multi = 0.3 if near_wall else 5.0
        out_total = out_gru + (x @ self.w_res) * multi

        # --- LÓGICA V44: EL FILTRO DE ATENCIÓN SOCIAL ---
        # Solo aplicamos una regla: Si eres el Seguidor (1), hay un muro enfrente
        # y escuchas al Faro (x[0,10]), entonces DEJA DE ACELERAR contra el muro.
        #if self.use_social_filter and self.agent_idx == 1: COMM
        if self.use_social_filter:
            social_active = abs(x[0, 10]) > 0.05
            if near_wall and social_active:
                # Quitamos el avance (Acción 0) para que el agente "se detenga a escuchar"
                out_total[0, 0] -= 30.0
                # NO FORZAMOS GIRO. Dejamos que la red GRU use el sensor social obs[10]
                # para decidir el giro por sí sola, sin vibraciones externas.

        # --- GANGLIO BASAL: EL FARO (Acción 4) ---
        # Umbral 0.32 para que nunca se apague en la meta
        if self.use_bg and self.out_size > 4 and x[0, 9] > 0.32:
            out_total[0, 4] += 20.0

        return int(np.argmax(out_total))

    def reset(self):
        self.h = np.zeros((1, self.h_size))

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']

            # Carga de parámetros con verificación de existencia
            if 'w_gru' in data: self.w_gru = np.array(data['w_gru'])
            if 'b_gru' in data: self.b_gru = np.array(data['b_gru'])
            if 'w_out' in data: self.w_out = np.array(data['w_out'])
            if 'b_out' in data: self.b_out = np.array(data['b_out'])
            if 'w_res' in data: self.w_res = np.array(data['w_res'])

class ResController:
    def __init__(self, model_path=None, num_agents=1, agent_idx=0):
        self.in_size = 11 if num_agents > 1 else 10
        self.out_size = 5 if num_agents > 1 else 4
        self.h_size = 16
        self.agent_idx = agent_idx
        self.w1 = np.zeros((self.in_size, self.h_size))
        self.b1 = np.zeros(self.h_size)
        self.w2 = np.zeros((self.h_size, self.out_size))
        self.b2 = np.zeros(self.out_size)
        self.w_skip = np.eye(self.in_size, self.out_size)
        if model_path: self.load(model_path)

    def act(self, obs, info=None):
        x = np.atleast_2d(obs)
        h = np.maximum(0, x @ self.w1 + self.b1)
        res = h @ self.w2 + self.b2
        skip = x @ self.w_skip
        out = res + skip
        return int(np.argmax(out))

    def reset(self): pass

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            if 'brain_a' in data:
                data = data['brain_a'] if self.agent_idx == 0 else data['brain_b']
            self.w1 = np.array(data['w1'])
            self.b1 = np.array(data['b1'])
            self.w2 = np.array(data['w2'])
            self.b2 = np.array(data['b2'])
            if 'w_skip' in data:
                self.w_skip = np.array(data['w_skip'])

class RandomController:
    def __init__(self, num_agents=1): self.out_size = 5 if num_agents > 1 else 4
    def act(self, obs, info=None): return np.random.randint(0, self.out_size)
    def reset(self): pass

class BraitenbergController:
    """
    Si el JSON del modelo trae left_weights/right_weights (el mismo formato
    que usa FusionController en 3D), calcula v_l/v_r con producto punto contra
    los sensores y traduce esa diferencia de velocidades a una accion discreta
    (0-4), ya que en 2D no hay control continuo de ruedas como en 3D.
    Si el JSON no trae pesos (o no se pasa model_path), cae al heuristico
    fijo original (evita por umbral 0.5).

    Incluye deteccion de atasco: al ser 100% reactivo, un Braitenberg puro
    puede quedar en equilibrio contra una esquina cóncava. Si un sensor se
    mantiene "pegado" varios pasos seguidos, dispara una maniobra de escape
    (retroceder + girar), alternando el lado de giro en cada atasco para no
    volver a la misma esquina.
    """
    def __init__(self, model_path=None, num_agents=1):
        self.num_agents = num_agents
        self.weights_l = None
        self.weights_r = None
        self.bias_l = 0.3
        self.bias_r = 0.3
        # umbral bajo el cual consideramos "ir derecho" en vez de girar
        self.turn_deadzone = 0.05

        # --- Deteccion de atasco en esquinas + escape ---
        self.stuck_sensor_threshold = 0.35  # sensor "pegado" a un obstaculo
        self.stuck_patience = 15            # pasos seguidos para considerar atasco
        self.stuck_counter = 0
        self.escape_backoff_steps = 6
        self.escape_turn_steps = 10
        self.escape_timer = 0
        self.escape_turn = 1                # 1=izquierda, 2=derecha (alterna)

        if model_path:
            self.load(model_path)

    def load(self, path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception:
            return
        if 'left_weights' in data and 'right_weights' in data:
            self.weights_l = np.array(data['left_weights'])
            self.weights_r = np.array(data['right_weights'])
            self.bias_l = data.get('left_bias', 0.3)
            self.bias_r = data.get('right_bias', 0.3)

    def act(self, obs, info=None):
        sensors = np.asarray(obs[:8], dtype=np.float64)

        # --- Modo escape activo: retrocede y luego gira ---
        if self.escape_timer > 0:
            self.escape_timer -= 1
            if self.escape_timer >= self.escape_turn_steps:
                return 3  # retrocede
            return self.escape_turn  # gira

        # --- Deteccion de atasco (sensor pegado N pasos seguidos) ---
        if float(np.max(sensors)) > self.stuck_sensor_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter > self.stuck_patience:
            self.stuck_counter = 0
            self.escape_turn = 2 if self.escape_turn == 1 else 1  # alterna lado
            self.escape_timer = self.escape_backoff_steps + self.escape_turn_steps
            return 3

        # --- Comportamiento normal: pesos del Braitenberg ---
        if self.weights_l is not None:
            v_l = float(np.dot(self.weights_l, sensors)) + self.bias_l
            v_r = float(np.dot(self.weights_r, sensors)) + self.bias_r
            diff = v_r - v_l
            if abs(diff) < self.turn_deadzone:
                return 0 if (v_l + v_r) >= 0 else 3  # avanzar / retroceder
            return 1 if diff > 0 else 2  # 1=izquierda, 2=derecha

        # --- Fallback: heuristico original sin pesos ---
        left_sensors, right_sensors = sensors[0:3], sensors[5:8]
        if np.max(left_sensors) > 0.5: return 2
        if np.max(right_sensors) > 0.5: return 1
        return 0

    def reset(self):
        self.stuck_counter = 0
        self.escape_timer = 0

class HeuristicCalibrationController:
    """
    Heurístico para la U con reacople activo.

    B:
    - sigue un lado preferido (izq/der)
    - si choca: backoff + turn + advance
    - si pierde la pared: reacquire = turn + advance
    """

    def __init__(self, agent_idx):
        self.agent_idx = agent_idx
        self.phase = "normal"
        self.timer = 0
        self.turn_dir = 0
        self.free_counter = 0
        self.subphase = None

    def reset(self):
        self.phase = "normal"
        self.timer = 0
        self.turn_dir = 0
        self.free_counter = 0
        self.subphase = None

    def act(self, obs):
        prox = obs[0:8]
        odor = obs[9]
        soc = obs[10] if len(obs) > 10 else 0.0

        front = max(prox[0], prox[1], prox[7])
        left = max(prox[1], prox[2], prox[3])
        right = max(prox[5], prox[6], prox[7])

        if self.agent_idx == 1:
            print(
                f"B -> phase={self.phase:>12} sub={str(self.subphase):>7} "
                f"timer={self.timer} free={self.free_counter} "
                f"front={front:.2f} L={left:.2f} R={right:.2f} "
                f"soc={soc:.2f} odor={odor:.2f}"
            )

        # Líder
        if self.agent_idx == 0 and odor > 0.30:
            return 4

        # lado preferido
        preferred = +1 if soc > 0 else -1

        # --------------------------------------------------
        # BACKOFF
        # --------------------------------------------------
        if self.phase == "backoff":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "turn"
                self.timer = 3
            return 3

        # --------------------------------------------------
        # TURN
        # --------------------------------------------------
        if self.phase == "turn":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "advance"
                self.timer = 8
            return 2 if self.turn_dir > 0 else 1

        # --------------------------------------------------
        # ADVANCE
        # --------------------------------------------------
        if self.phase == "advance":
            self.timer -= 1

            if front > 0.50:
                self.phase = "backoff"
                self.timer = 4
                return 3

            if self.timer <= 0:
                self.phase = "normal"
                self.free_counter = 0

            if left > 0.82:
                return 1
            if right > 0.82:
                return 2

            return 0

        # --------------------------------------------------
        # REACQUIRE_WALL = turn a bit + advance a bit
        # --------------------------------------------------
        if self.phase == "reacquire_wall":
            if self.subphase == "turn":
                self.timer -= 1
                if self.timer <= 0:
                    self.subphase = "advance"
                    self.timer = 5

                return 2 if preferred > 0 else 1

            if self.subphase == "advance":
                self.timer -= 1

                # si reacopló la pared, salir
                if preferred > 0 and left > 0.15:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                if preferred < 0 and right > 0.15:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                # si aparece pared frontal, maniobra completa
                if front > 0.50:
                    self.phase = "backoff"
                    self.subphase = None
                    self.timer = 4
                    self.turn_dir = preferred
                    return 3

                if self.timer <= 0:
                    self.phase = "normal"
                    self.subphase = None
                    self.free_counter = 0
                    return 0

                return 0

        # --------------------------------------------------
        # NORMAL
        # --------------------------------------------------
        if front > 0.50:
            self.phase = "backoff"
            self.timer = 4
            self.turn_dir = preferred
            return 3

        if left > 0.85:
            return 1
        if right > 0.85:
            return 2

        # Si perdió la pared del lado preferido, intentar reacoplarla
        if preferred > 0:
            if left < 0.04 and front < 0.25:
                self.phase = "reacquire_wall"
                self.subphase = "turn"
                self.timer = 2
                return 2
        else:
            if right < 0.04 and front < 0.25:
                self.phase = "reacquire_wall"
                self.subphase = "turn"
                self.timer = 2
                return 1

        # En libre: avanzar con sesgo suave, no órbitas grandes
        self.free_counter += 1

        if preferred > 0:
            if self.free_counter % 8 == 0:
                return 2
            return 0
        else:
            if self.free_counter % 8 == 0:
                return 1
            return 0

def make_controller(type, num_agents=1, agent_idx=0, model_path=None, use_bg=True, use_social_filter=True):
    t = type.lower()
    if t == "mlp": return MLPController(model_path, num_agents, agent_idx)
    if t == "gru":
        ctrl = GRUController(
            model_path=model_path,
            num_agents=num_agents,
            agent_idx=agent_idx,
            use_social_filter=use_social_filter,
        )
        ctrl.use_bg = use_bg
        return ctrl
    if t == "res" or t == "residual": return ResController(model_path, num_agents, agent_idx)
    if t == "random": return RandomController(num_agents)
    if t == "braitenberg": return BraitenbergController(model_path=model_path, num_agents=num_agents)
    if t == "heuristic": return HeuristicCalibrationController(agent_idx)
    raise ValueError(f"Tipo de controlador desconocido: {type}")
