import sys, os, time, json, argparse, numpy as np, multiprocessing as mp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from foraging_env2d import ForagingEnv, ForagingEnvConfig
from controllers import make_controller

def get_weight_size(agent_type, num_agents):
    in_s, out_s, h = (11 if num_agents > 1 else 10), (5 if num_agents > 1 else 4), 16
    if agent_type == "mlp": single = (in_s * h) + h + (h * out_s) + out_s
    else: 
        # Tamaño GRU + Matriz Residual (in_s * out_s) = 1484 para in=11, out=5
        single = ((in_s + h) * 3 * h) + (3 * h) + (h * out_s) + out_s + (in_s * out_s)
    return single * 2 if num_agents > 1 else single

def set_weights_team(c_a, c_b, weights, agent_type, num_agents):
    half = len(weights) // 2
    w_list, controllers = [weights[:half], weights[half:]], [c_a, c_b]
    for i in range(num_agents):
        c, w, idx = controllers[i], w_list[i], 0
        in_s, out_s, h = (11 if num_agents > 1 else 10), (5 if num_agents > 1 else 4), 16
        if agent_type == "gru":
            ws = (in_s + h) * 3 * h
            c.w_gru = w[idx:idx+ws].reshape(in_s+h, 3*h); idx += ws
            c.b_gru = w[idx:idx+3*h]; idx += 3*h
            c.w_out = w[idx:idx+h*out_s].reshape(h, out_s); idx += h*out_s
            c.b_out = w[idx:idx+out_s]; idx += out_s
            c.w_res = w[idx:idx+in_s*out_s].reshape(in_s, out_s)
        else:
            ws1, ws2 = in_s * h, h * out_s
            c.w1 = w[idx:idx+ws1].reshape(in_s, h); idx += ws1
            c.b1 = w[idx:idx+h]; idx += h
            c.w2 = w[idx:idx+ws2].reshape(h, out_s); idx += ws2
            c.b2 = w[idx:idx+out_s]

def save_best_model(weights, agent_type, num_agents, filename):
    half = len(weights) // 2
    in_s, out_s, h = (11 if num_agents > 1 else 10), (5 if num_agents > 1 else 4), 16
    def get_brain(w):
        idx, d, ws = 0, {}, (in_s + h) * 3 * h
        if agent_type == "gru":
            d['w_gru'] = w[idx:idx+ws].reshape(in_s+h, 3*h).tolist(); idx += ws
            d['b_gru'] = w[idx:idx+3*h].tolist(); idx += 3*h
            d['w_out'] = w[idx:idx+h*out_s].reshape(h, out_s).tolist(); idx += h*out_s
            d['b_out'] = w[idx:idx+out_s].tolist(); idx += out_s
            d['w_res'] = w[idx:idx+in_s*out_s].reshape(in_s, out_s).tolist()
        return d
    data = {"brain_a": get_brain(weights[:half]), "brain_b": get_brain(weights[half:]), "type": agent_type, "agents": num_agents}
    with open(filename, 'w') as f: json.dump(data, f)

def load_weights(path, agent_type, num_agents):
    with open(path, 'r') as f: data = json.load(f)
    target_single = get_weight_size(agent_type, 2) // 2
    def flatten(d):
        w = []
        # Mantener orden estricto de recuperación
        keys = ['w_gru', 'b_gru', 'w_out', 'b_out', 'w_res'] if agent_type == "gru" else ['w1', 'b1', 'w2', 'b2']
        for k in keys:
            if k in d: w.extend(np.array(d[k]).flatten())
        # FIX V29: Relleno de ceros para compatibilidad con versiones v20 (1429 params)
        if len(w) < target_single:
            w.extend(np.zeros(target_single - len(w)))
        return np.array(w[:target_single])
    
    if 'brain_a' in data: 
        return np.concatenate([flatten(data['brain_a']), flatten(data['brain_b'])])
    single = flatten(data)
    return np.concatenate([single, single])

def evaluate_individual(weights, agent_type, num_agents, h_prob, use_bg, u_mode, rand_spawns):
    # --- EFSIM V51: ARQUITECTURA DE GANGLIO BASAL (INSTINTOS SIMÉTRICOS) ---
    total_f, total_vocal, total_stuck = 0, 0, 0
    episodes = 9 
    
    for ep_idx in range(episodes):
        current_u_flag = (ep_idx >= 3 and ep_idx < 6)
        current_map_name = "n_shape" if ep_idx >= 6 else "default"

        config = ForagingEnvConfig(num_agents=num_agents, max_steps=500, u_env=current_u_flag,
                                   map_name=current_map_name, randomize_spawns=rand_spawns, 
                                   randomize_food=True, odor_radius=5.0)
        env = ForagingEnv(config=config)
        
        if current_u_flag: env.food_pos = np.array([np.random.uniform(280, 500), np.random.uniform(280, 430)])
        elif current_map_name == "n_shape": env.food_pos = np.array([np.random.uniform(280, 500), np.random.uniform(150, 250)])
        else: env.food_pos = env._sample_food().astype(np.float32)

        c_a = make_controller(agent_type, 2, 0, use_bg=use_bg); c_b = make_controller(agent_type, 2, 1, use_bg=use_bg)    
        set_weights_team(c_a, c_b, weights, agent_type, 2)
        
        obs, info = env.reset(); c_a.reset(); c_b.reset()
        done, steps_ep = False, 0
        at = info['individual_success']
        p_pre = [p.copy() for p in env.agent_pos] # Inicializar p_pre
        
        while not done:
            steps_ep += 1
            d_pre = [np.linalg.norm(env.agent_pos[i] - env.food_pos) for i in range(2)]
            actions_raw = [c_a.act(obs[0]), c_b.act(obs[1])]
            
            # --- GANGLIO BASAL: CAPA DE INSTINTOS ---
            actions = []
            for i in range(2):
                dist_moved = np.linalg.norm(env.agent_pos[i] - p_pre[i])
                
                # 1. Instinto de Reclutamiento (En la meta) - 70% prob.
                if at[i] and np.random.rand() < 0.7:
                    actions.append(4)
                # 2. Instinto de Auxilio (Atascado girando) - 20% prob.
                elif actions_raw[i] in [1, 2] and dist_moved < 0.5 and np.random.rand() < 0.2:
                    actions.append(4)
                # 3. Cortex y Annealing (El resto del tiempo)
                elif actions_raw[i] == 4 and np.random.rand() < h_prob:
                    actions.append(np.random.randint(0, 4))
                else:
                    actions.append(actions_raw[i])
            
            # Guardar p_pre ANTES del step
            for i in range(num_agents): p_pre[i] = env.agent_pos[i].copy()
            
            # Señalización y Ejecución
            for i in range(num_agents): 
                env.agent_signals[i] = 1.0 if actions[i] == 4 else 0.0
                if actions[i] == 4: total_vocal += 1
            
            obs, _, term, trun, info = env.step(actions)
            at = info['individual_success']
            
            for i in range(2):
                other = 1 - i
                dist_moved = np.linalg.norm(env.agent_pos[i] - p_pre[i])
                p_frente = max(obs[i][0], obs[i][1], obs[i][7])
                
                # Fitness: Desbloqueo frontal
                if p_frente > 0.85:
                    if actions[i] == 0: total_f -= 50.0
                    if actions[i] in [1, 2]: total_f += 100.0

                if not at[i]:
                    total_f += (d_pre[i] - np.linalg.norm(env.agent_pos[i] - env.food_pos)) * 200.0 
                    if dist_moved < 1.1: total_f -= 100.0; total_stuck += 1
                    if at[other] and env.agent_signals[other] > 0.5:
                        d_soc_pre = np.linalg.norm(p_pre[i] - env.agent_pos[other])
                        d_soc_post = np.linalg.norm(env.agent_pos[i] - env.agent_pos[other])
                        if (d_soc_pre - d_soc_post) > 1.2: total_f += 1000.0
                else:
                    if not at[other]:
                        if actions[i] == 4: total_f += 300.0
                        else: total_f -= 1000.0
                
            done = term or trun

        if all(at):
            gap = abs(info['success_steps'][0] - info['success_steps'][1])
            mult = 1.0 if gap < 40 else 0.3
            total_f += (3000000 + (500 - gap) * 10000.0) * mult
        else:
            total_f -= 300000 
            
    return (max(0.1, total_f / episodes), total_vocal / episodes, total_stuck / episodes)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="gru")
    parser.add_argument("--gen", type=int, default=100)
    parser.add_argument("--pop", type=int, default=128)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--name", type=str, default="forage_v-1")
    parser.add_argument("--u_env", action="store_true", help="Activa el laberinto en U para benchmark")
    parser.add_argument("--no_bg", action="store_true", help="Desactiva el Ganglio Basal (Modo Mudo)")
    parser.add_argument("--random_spawns", action="store_true", help="Activa spawns aleatorios")
    args = parser.parse_args()
    
    # Esta variable se la pasaremos al entorno
    u_mode = args.u_env
    
    use_bg = not args.no_bg # Si pones --no_bg, use_bg será False    
    status = "Biological Impulse" if use_bg else "MODO MUDO (Baseline)"
    print(f"\n=== INICIANDO ENTRENAMIENTO: {status} ===")
    
    w_size = get_weight_size(args.type, 2)
    
    if args.load and os.path.exists(args.load):
        print(f"Cargando base biogenética: {args.load}")
        base = load_weights(args.load, args.type, 2)
        population = [base + np.random.randn(w_size)*0.03 for _ in range(args.pop)]
        population[0] = base
    else: 
        population = [np.random.randn(w_size)*0.5 for _ in range(args.pop)]
        
    for d in ["models", "logs"]: 
        if not os.path.exists(d): os.makedirs(d)
        
    log_path = "logs/{}_log.csv".format(args.name)
    if not os.path.exists(log_path): # Escribir cabecera si el archivo es nuevo
        with open(log_path, "w") as f: f.write("gen,best,avg,screams,stuck,time\n")
        
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for gen in range(args.gen):
            t_s = time.time()
            
            # --- ESTRATEGIA EFSIM: CURRICULUM LEARNING V29 ---
            # Si el usuario pidió --u_env, activamos el laberinto solo después de la Gen 30
            if args.u_env:
                current_u_mode = True if gen >= 30 else False
            else:
                current_u_mode = False
            
            # Visualización en consola para saber qué estamos entrenando
            env_status = "U-LABYRINTH" if current_u_mode else "OBSTÁCULOS ALEATORIOS"
            
            # --- ANNEALING DE HEURÍSTICA ---
            h_prob = max(0.0, 0.3 * (1.0 - gen / 80.0))
            
            # Pasamos 'current_u_mode' en lugar de 'u_mode'
            # Actualiza esta línea para incluir args.random_spawns
            results = pool.starmap(evaluate_individual, [
                (ind, args.type, 2, h_prob, use_bg, current_u_mode, args.random_spawns) for ind in population
            ])
            
            fits = [r[0] for r in results]
            screams = [r[1] for r in results]
            stucks = [r[2] for r in results] 
            
            idx = np.argsort(fits)[::-1]
            best_f, avg_f, avg_s = fits[idx[0]], np.mean(fits), np.mean(screams)
            population = [population[i] for i in idx]
            
            # Selección y Crossover
            new_pop = population[:10]
            while len(new_pop) < args.pop:
                p1, p2 = population[np.random.randint(0, 25)], population[np.random.randint(0, 25)]
                child = np.where(np.random.rand(w_size) > 0.5, p1, p2)
                if np.random.rand() < 0.2: # Mutación
                    child += np.random.randn(w_size) * 0.12
                new_pop.append(child)
            
            population = new_pop
            dt = time.time() - t_s
            
            avg_stuck = np.mean(stucks)
            
            print("GEN {} (h:{:.2f}) | {:10s} | BEST: {:8.0f} | AVG: {:8.0f} | SCREAMS: {:4.1f} | STUCK: {:4.1f} | {:.1f}s".format(
                gen, h_prob, env_status, best_f, avg_f, avg_s, avg_stuck, dt))
                
            with open(log_path, "a") as f: 
                f.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    gen, best_f, avg_f, avg_s, avg_stuck, dt))
                
            if gen % 10 == 0 or gen == args.gen - 1:
                save_best_model(population[0], args.type, 2, "models/{}_gen{}.json".format(args.name, gen))

    save_best_model(population[0], args.type, 2, "models/{}_final.json".format(args.name))
    print("Entrenamiento Finalizado.")

if __name__ == "__main__": 
    main()
