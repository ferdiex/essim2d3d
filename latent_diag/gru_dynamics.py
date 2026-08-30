"""
gru_dynamics.py

Paso 1 del intento de Sussillo & Barak (2013) para el Paper 3.

Extrae la recurrencia del GRU (GRUNetwork.forward en controllers.py,
lineas 80-98) como una funcion PURA:

    h_siguiente = F(h, x, weights)

Sin mutar self.h, sin calcular logits, sin argmax. Esto es exactamente
lo que necesita el metodo de puntos fijos: encontrar h* tal que
F(h*, x) ≈ h*, para un x FIJO.

IMPORTANTE: la logica de decision (near_wall, GG/filtro social,
HH/El Faro, JALON_ATTENUATION, SOCIAL_RESIDUAL_BOOST) en controllers.py
actua SOLO sobre los logits de salida (que accion se elige), nunca
sobre la actualizacion de h. Por eso F(h,x) de aca abajo es identica
en su efecto sobre h a la del controlador real -- no es una
simplificacion, es la misma matematica, solo que aislada del resto.

Formula (identica a controllers.py lineas 81-98):
    combined       = [x, h]                                  (27,)
    gates          = combined @ w_gru + b_gru                (48,)
    z              = sigmoid(gates[0:16])
    r              = sigmoid(gates[16:32])
    combined_reset = [x, r * h]                               (27,)
    n              = tanh(combined_reset @ w_gru[:, 32:] + b_gru[32:])
    h_siguiente    = (1 - z) * h + z * n
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def F(h, x, weights):
    """
    Un paso de la dinamica recurrente del GRU. Pura: no muta nada,
    no calcula logits, no elige accion. Solo devuelve h_siguiente.

    Parametros
    ----------
    h : np.ndarray, shape (16,)
        Estado oculto actual.
    x : np.ndarray, shape (11,)
        Input fijo (observacion).
    weights : dict con 'w_gru' (27, 48) y 'b_gru' (48,)
        Mismos pesos que carga GRUNetwork (ver controllers.py).

    Retorna
    -------
    h_next : np.ndarray, shape (16,)
    """
    w_gru = weights['w_gru']
    b_gru = weights['b_gru']

    combined = np.concatenate([x, h])                     # (27,)
    gates = combined @ w_gru + b_gru                       # (48,)
    z = sigmoid(gates[:16])
    r = sigmoid(gates[16:32])

    combined_reset = np.concatenate([x, r * h])            # (27,)
    n = np.tanh(combined_reset @ w_gru[:, 32:] + b_gru[32:])

    h_next = (1 - z) * h + z * n
    return h_next


def q(h, x, weights):
    """
    Funcion de 'velocidad' de Sussillo-Barak:
        q(h) = 0.5 * ||h - F(h, x)||^2

    q(h) = 0  <=>  h es un punto fijo exacto de la dinamica para ese x.
    Se usa como funcion objetivo a minimizar en el paso de busqueda
    (todavia no implementado en este script -- este es solo el paso 1).
    """
    diff = h - F(h, x, weights)
    return 0.5 * float(np.dot(diff, diff))


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Verificacion cruzada contra GRUNetwork.forward real, con
    # pesos aleatorios (misma disciplina de "validar antes de
    # confiar" que se uso en measure_h_confinement.py, etc.)
    # ------------------------------------------------------------
    import sys
    sys.path.insert(0, "/home/claude")
    from controllers_ref import GRUNetwork

    rng = np.random.default_rng(0)
    weights = {
        'w_gru': rng.normal(size=(27, 48)) * 0.5,
        'b_gru': rng.normal(size=(48,)) * 0.1,
        'w_out': rng.normal(size=(16, 5)) * 0.5,   # no usado por F, pero
        'b_out': rng.normal(size=(5,)) * 0.1,      # GRUNetwork los pide
        'w_res': rng.normal(size=(11, 5)) * 0.5,   # al construirse
    }

    net = GRUNetwork(weights)

    n_checks = 200
    max_abs_diff = 0.0
    for i in range(n_checks):
        h0 = rng.normal(size=16) * 0.5
        x0 = rng.normal(size=11) * 0.5

        # Referencia real: fijamos net.h, llamamos forward(), leemos el h
        # que quedo guardado adentro (antes de cualquier logica de logits).
        net.h = h0.copy()
        net.forward(x0)
        h_next_real = net.h.copy()

        h_next_pure = F(h0, x0, weights)

        max_abs_diff = max(max_abs_diff, np.max(np.abs(h_next_real - h_next_pure)))

    print(f"Verificacion sobre {n_checks} pares (h, x) aleatorios:")
    print(f"  Diferencia maxima absoluta entre F(h,x) y GRUNetwork.forward: {max_abs_diff:.2e}")
    if max_abs_diff < 1e-10:
        print("  OK -- F(h,x) reproduce exactamente la dinamica real del GRU.")
    else:
        print("  ATENCION -- hay una discrepancia real, revisar antes de seguir.")
