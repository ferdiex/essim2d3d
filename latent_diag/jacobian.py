"""
jacobian.py

Paso 2 del intento de Sussillo & Barak (2013) para el Paper 3.

Deriva a mano el Jacobiano de F(h,x) respecto a h (16x16) -- necesario
para, una vez encontrados los puntos fijos, saber si son
estables/inestables y si hay direcciones (autovectores) que se alineen
con comportamientos como "giro_der".

No se usa ninguna libreria de autodiff (JAX/PyTorch) -- el GRU tiene
solo 16 dimensiones y funciones con derivada conocida (sigmoid, tanh),
asi que se deriva analiticamente. Se valida el resultado contra
diferencias finitas antes de confiar en el (misma disciplina de
"validar antes de usar" del resto del proyecto).

Derivacion (ver comentario largo abajo para el paso a paso):

    J = diag(1 - z) + diag(n - h) @ dz_dh + diag(z) @ dn_dh

donde dz_dh y dn_dh salen de propagar la regla de la cadena por las
puertas del GRU.
"""

import numpy as np
from gru_dynamics import sigmoid, F


def _split_weights(w_gru):
    """
    w_gru tiene forma (27, 48): 27 = 11 (x) + 16 (h), 48 = 16+16+16 (z,r,n).
    Nos interesa solo la parte que multiplica a h (filas 11:27) para
    cada puerta, porque es la unica via por la que h de este paso
    afecta a h del paso siguiente.
    """
    W_z_h = w_gru[11:, 0:16]    # (16,16)
    W_r_h = w_gru[11:, 16:32]   # (16,16)
    W_n_h = w_gru[11:, 32:48]   # (16,16)
    return W_z_h, W_r_h, W_n_h


def analytical_jacobian(h, x, weights):
    """
    Jacobiano d h_next / d h, evaluado en (h, x). Shape (16, 16).

    Paso a paso (todas las 'diag(v) @ M' se implementan como
    v[:, None] * M, que es lo mismo pero sin construir la matriz
    diagonal explicita):

    1. z = sigmoid(gates_z),  gates_z = x@W_z_x + h@W_z_h + b_z
       dz/dh = diag(z*(1-z)) @ W_z_h.T

    2. r = sigmoid(gates_r),  gates_r = x@W_r_x + h@W_r_h + b_r
       dr/dh = diag(r*(1-r)) @ W_r_h.T

    3. u = r * h  (entrada resetada que va al calculo de n)
       du/dh = diag(r) + diag(h) @ dr/dh
             = diag(r) + diag(h*r*(1-r)) @ W_r_h.T

    4. n_pre = x@W_n_x + u@W_n_h + b_n
       dn_pre/dh = W_n_h.T @ (du/dh)

    5. n = tanh(n_pre)
       dn/dh = diag(1 - n**2) @ dn_pre/dh

    6. h_next = (1-z)*h + z*n
       dh_next/dh = diag(1-z) + diag(n-h) @ dz/dh + diag(z) @ dn/dh
    """
    w_gru = weights['w_gru']
    b_gru = weights['b_gru']
    W_z_h, W_r_h, W_n_h = _split_weights(w_gru)

    # --- forward pass (para tener z, r, n en el punto de evaluacion) ---
    combined = np.concatenate([x, h])
    gates = combined @ w_gru + b_gru
    z = sigmoid(gates[:16])
    r = sigmoid(gates[16:32])
    u = r * h
    combined_reset = np.concatenate([x, u])
    n_pre = combined_reset @ w_gru[:, 32:] + b_gru[32:]
    n = np.tanh(n_pre)

    # --- paso 1: dz/dh ---
    dz_dh = (z * (1 - z))[:, None] * W_z_h.T          # (16,16)

    # --- paso 2: dr/dh ---
    dr_dh = (r * (1 - r))[:, None] * W_r_h.T          # (16,16)

    # --- paso 3: du/dh ---
    du_dh = np.diag(r) + h[:, None] * dr_dh           # (16,16)

    # --- paso 4: dn_pre/dh ---
    dnpre_dh = W_n_h.T @ du_dh                        # (16,16)

    # --- paso 5: dn/dh ---
    dn_dh = (1 - n ** 2)[:, None] * dnpre_dh          # (16,16)

    # --- paso 6: dh_next/dh ---
    J = np.diag(1 - z) + (n - h)[:, None] * dz_dh + z[:, None] * dn_dh

    return J


def finite_difference_jacobian(h, x, weights, eps=1e-6):
    """
    Jacobiano numerico por diferencias finitas centradas, columna por
    columna. Sirve solo para VALIDAR el analitico -- no se usa en la
    busqueda de puntos fijos (seria mas lento y menos preciso).
    """
    n_dim = h.shape[0]
    J_num = np.zeros((n_dim, n_dim))
    for j in range(n_dim):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[j] += eps
        h_minus[j] -= eps
        J_num[:, j] = (F(h_plus, x, weights) - F(h_minus, x, weights)) / (2 * eps)
    return J_num


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    n_checks = 30
    worst = 0.0
    worst_case = None

    for i in range(n_checks):
        weights = {
            'w_gru': rng.normal(size=(27, 48)) * 0.5,
            'b_gru': rng.normal(size=(48,)) * 0.1,
        }
        h0 = rng.normal(size=16) * 0.5
        x0 = rng.normal(size=11) * 0.5

        J_analytic = analytical_jacobian(h0, x0, weights)
        J_numeric = finite_difference_jacobian(h0, x0, weights)

        diff = np.max(np.abs(J_analytic - J_numeric))
        if diff > worst:
            worst = diff
            worst_case = i

    print(f"Verificacion sobre {n_checks} combinaciones aleatorias de (pesos, h, x):")
    print(f"  Diferencia maxima absoluta (analitico vs. diferencias finitas): {worst:.2e}")
    print(f"  (peor caso: combinacion #{worst_case})")
    if worst < 1e-5:
        print("  OK -- el Jacobiano analitico coincide con diferencias finitas.")
    else:
        print("  ATENCION -- discrepancia mayor a la esperada, revisar la derivacion.")
