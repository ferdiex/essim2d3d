"""
find_fixed_points.py

Paso 3 del intento de Sussillo & Barak (2013) para el Paper 3.

Implementa la busqueda de puntos fijos (minimizar q(h) = 0.5*||h-F(h,x)||^2
con muchos reinicios) y la valida contra un caso SINTETICO donde
conocemos de antemano cual es la respuesta correcta -- antes de
correrla sobre los datos reales del checkpoint v15.

------------------------------------------------------------------
COMO SE CONSTRUYE EL CASO SINTETICO (con respuesta conocida)
------------------------------------------------------------------
INTENTO 1 (fallido, dejado documentado abajo porque enseña algo real):
se probo forzar solo W_n_x=0 y b_n=0, esperando que h*=0 fuera el
UNICO punto fijo. La busqueda encontro 18 puntos fijos validos
distintos (q~1e-9, todos genuinos), no 1. Esto NO fue un bug del
codigo: con esa construccion, 'n' todavia depende de h (via r*h @
W_n_h), asi que la ecuacion de punto fijo z*(n-h)=0 tiene multiples
soluciones reales ademas de h=0 (el sistema sigue siendo no lineal en
h). O sea: el GRU de juguete SI podia tener varios puntos fijos
genuinos -- la prueba no estaba mal ejecutada, estaba mal DISEÑADA
para dar una unica respuesta conocida.

INTENTO 2 (el que se usa aca): se fuerzan a cero DOS partes:
  - W_z_h = 0  (la puerta z deja de depender de h, solo depende de x)
  - W_n_h = 0  (la propuesta n deja de depender de h, solo depende de x)

Con esto, dado un x fijo, z y n quedan CONSTANTES (no dependen de h
para nada), y la recurrencia se vuelve lineal y elemento-a-elemento:

    h_next_i = (1 - z_i) * h_i + z_i * n_i        (z_i, n_i fijos)

Esto es una contraccion simple hacia un unico punto fijo, con formula
cerrada conocida de antemano:

    h*_i = n_i     (para cualquier h inicial, converge ahi)

porque 0 < z_i < 1 siempre (sigmoid nunca toca 0 ni 1), asi que
0 < 1-z_i < 1 y la recurrencia SIEMPRE contrae hacia n_i, sin importar
donde arranque h. Ahora si hay una sola respuesta correcta, calculable
sin optimizar nada, para comparar contra lo que encuentre la busqueda.
------------------------------------------------------------------
"""

import numpy as np
from gru_dynamics import F, q
from jacobian import analytical_jacobian


def search_fixed_points(x, weights, h_inits, n_iters=3000, lr=0.05, tol=1e-9, verbose=False):
    """
    Version SIN correa (la original, paso 3). Se deja intacta para no
    romper la validacion sintetica que ya se corrio con ella.
    """
    return _search_core(x, weights, h_inits, n_iters, lr, tol, lambda_leash=0.0, verbose=verbose)


def search_fixed_points_con_correa(x, weights, h_inits, n_iters=3000, lr=0.05, tol=1e-9,
                                    lambda_leash=0.01, verbose=False):
    """
    Version CON correa (freno de distancia al h inicial real).

    En vez de minimizar solo q(h) = 0.5*||h-F(h,x)||^2, se minimiza:

        q_reg(h) = q(h) + 0.5 * lambda_leash * ||h - h_init||^2

    El segundo termino es un resorte que tira a h de vuelta hacia el
    punto de partida (que es un h REAL, sacado de datos de atasco de
    verdad). Si el optimizador quiere alejarse mucho para bajar q un
    poquito mas, el resorte se lo cobra caro -- evita el "paseo" que
    encontramos en el diagnostico barato.

    Devuelve el q SIN regularizar (el que importa para saber si es un
    punto fijo de verdad), no el q_reg (que es solo una herramienta
    interna de la busqueda).
    """
    return _search_core(x, weights, h_inits, n_iters, lr, tol, lambda_leash=lambda_leash, verbose=verbose)


def _search_core(x, weights, h_inits, n_iters, lr, tol, lambda_leash, verbose):
    n_dim = h_inits.shape[1]
    results = []

    for idx, h0 in enumerate(h_inits):
        h = h0.copy()
        h_init_fijo = h0.copy()  # el ancla de la correa, no cambia durante la busqueda
        m = np.zeros(n_dim)
        v = np.zeros(n_dim)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        for t in range(1, n_iters + 1):
            e = h - F(h, x, weights)
            J = analytical_jacobian(h, x, weights)
            grad_q = e - J.T @ e
            grad_leash = lambda_leash * (h - h_init_fijo)
            grad = grad_q + grad_leash

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            h = h - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            q_now = q(h, x, weights)  # el q SIN regularizar, el que importa
            if q_now < tol and lambda_leash == 0.0:
                break  # con correa no cortamos temprano por q solo, queremos ver donde se asienta

        q_final = q(h, x, weights)
        distancia_recorrida = float(np.linalg.norm(h - h0))
        results.append({'h': h, 'q': q_final, 'h_init': h0,
                         'distancia_recorrida': distancia_recorrida})

        if verbose and (idx + 1) % 10 == 0:
            print(f"  reinicio {idx+1}/{len(h_inits)} -- q_final={q_final:.2e} "
                  f"dist_recorrida={distancia_recorrida:.2f}")

    return results


def dedupe(results, q_tol=1e-6, dist_tol=1e-3):
    """
    Se queda solo con los candidatos que realmente convergieron
    (q < q_tol), y funde los que caen muy cerca entre si (misma
    'canoa', encontrada desde distintos puntos de partida).
    """
    good = [r for r in results if r['q'] < q_tol]
    good.sort(key=lambda r: r['q'])

    kept = []
    for cand in good:
        is_dup = False
        for k in kept:
            if np.linalg.norm(cand['h'] - k['h']) < dist_tol:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


if __name__ == "__main__":
    rng = np.random.default_rng(2)

    # --- Construccion del caso sintetico de respuesta conocida (intento 2) ---
    weights = {
        'w_gru': rng.normal(size=(27, 48)) * 0.6,
        'b_gru': rng.normal(size=(48,)) * 0.2,
    }
    weights['w_gru'][11:, 0:16] = 0.0    # W_z_h = 0  (z ya no depende de h)
    weights['w_gru'][11:, 32:48] = 0.0   # W_n_h = 0  (n ya no depende de h)
    # (con esto, para un x fijo, z y n quedan constantes -- la recurrencia
    #  es lineal elemento-a-elemento y converge SIEMPRE a h*=n, ver arriba)

    x_fijo = rng.normal(size=11) * 0.5   # un x cualquiera, no importa cual

    # --- Respuesta correcta, calculada en forma cerrada (sin optimizar) ---
    combined_x0 = np.concatenate([x_fijo, np.zeros(16)])  # h no importa: z,n no dependen de h
    gates_x0 = combined_x0 @ weights['w_gru'] + weights['b_gru']
    n_pre_correcto = gates_x0[32:48]
    h_estrella_correcto = np.tanh(n_pre_correcto)  # esta es LA respuesta correcta

    # --- Reinicios: mezcla de puntos cerca de 0 y lejos, para exigir ---
    n_restarts = 50
    h_inits = np.concatenate([
        rng.normal(size=(25, 16)) * 0.1,   # cerca del punto fijo esperado
        rng.normal(size=(25, 16)) * 1.5,   # lejos, para probar de verdad
    ])

    print(f"Corriendo busqueda de puntos fijos sobre el caso sintetico "
          f"({n_restarts} reinicios)...")
    results = search_fixed_points(x_fijo, weights, h_inits, verbose=True)

    found = dedupe(results)

    print(f"\nCandidatos que convergieron (q < 1e-6): "
          f"{sum(1 for r in results if r['q'] < 1e-6)}/{n_restarts}")
    print(f"Puntos fijos unicos despues de deduplicar: {len(found)}")

    for i, fp in enumerate(found):
        dist_a_correcto = np.linalg.norm(fp['h'] - h_estrella_correcto)
        print(f"  punto fijo #{i}: q={fp['q']:.2e}, "
              f"distancia a la respuesta correcta conocida={dist_a_correcto:.2e}")

    # --- Veredicto ---
    todos_cerca_del_correcto = all(
        np.linalg.norm(fp['h'] - h_estrella_correcto) < 1e-3 for fp in found
    )
    tasa_convergencia = sum(1 for r in results if r['q'] < 1e-6) / n_restarts

    print()
    if len(found) == 1 and todos_cerca_del_correcto and tasa_convergencia > 0.95:
        print("OK -- la busqueda encontro EL UNICO punto fijo conocido, de forma "
              "confiable, desde puntos de partida cercanos y lejanos, y no "
              "inventó ningun punto fijo falso. El codigo esta listo para "
              "probarse sobre datos reales.")
    else:
        print("ATENCION -- la busqueda no fue confiable en el caso sintetico "
              "(deberiamos haber encontrado un unico punto fijo, siempre el "
              "mismo). Revisar antes de usar esto sobre datos reales.")
