"""
Comparacion directa de pesos entre brain_a y brain_b -- sin correr
episodios, sin fisica, solo numpy sobre el JSON de pesos ya entrenado.

Compara, columna por columna (accion: 0=adelante,1=giro_izq,2=giro_der,
3=atras,4=señal), la magnitud (norma L2) de los pesos que alimentan cada
logit, en w_res (camino residual x->logits) y w_out (memoria h->logits),
mas el bias b_out. Foco especial en giro_izq (1) vs giro_der (2) -- si
brain_b tiene una asimetria de magnitud o de bias entre esas dos
columnas que brain_a no tiene, explicaria directamente el sesgo
sistematico hacia "girar a la derecha" que se observo en el atractor.

Tambien compara la fila 10 de w_res (x[10], señal social) contra el
resto de las filas, para ver si hay diferencia de peso relativo de la
señal entre A y B.

Uso:
    python3 compare_weights_ab.py --model models/social3d_v15_full_final.json
"""
import argparse
import json

import numpy as np

ACTION_NAMES = ["adelante", "giro_izq", "giro_der", "atras", "señal"]
INPUT_NAMES = ["sens0", "sens1", "sens2", "sens3", "sens4", "sens5", "sens6", "sens7",
               "bearing_comida", "hambre", "señal_social"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    with open(args.model) as f:
        weights = json.load(f)

    wa = weights["brain_a"]
    wb = weights["brain_b"]

    w_res_a = np.array(wa["w_res"])   # (11, 5)
    w_res_b = np.array(wb["w_res"])
    w_out_a = np.array(wa["w_out"])   # (16, 5)
    w_out_b = np.array(wb["w_out"])
    b_out_a = np.array(wa["b_out"])   # (5,)
    b_out_b = np.array(wb["b_out"])

    print("=== Comparacion de pesos: brain_a vs brain_b ===\n")

    print("--- Bias por accion (b_out) ---")
    print(f"{'accion':>12} | {'A':>10} | {'B':>10} | {'diff (B-A)':>12}")
    for j, name in enumerate(ACTION_NAMES):
        diff = b_out_b[j] - b_out_a[j]
        marker = "  <<<" if name in ("giro_izq", "giro_der") else ""
        print(f"{name:>12} | {b_out_a[j]:+10.3f} | {b_out_b[j]:+10.3f} | {diff:+12.3f}{marker}")

    print(f"\nBrecha giro_der - giro_izq en el BIAS: A={b_out_a[2]-b_out_a[1]:+.3f}, "
          f"B={b_out_b[2]-b_out_b[1]:+.3f}")
    print("(un bias fuerte hacia giro_der en B, ausente en A, explicaria directamente "
          "el sesgo sistematico del atractor -- sin depender de ningun input.)\n")

    print("--- Magnitud (norma L2) de w_out por columna de accion (memoria h -> logit) ---")
    print(f"{'accion':>12} | {'A':>10} | {'B':>10} | {'ratio B/A':>10}")
    for j, name in enumerate(ACTION_NAMES):
        na = np.linalg.norm(w_out_a[:, j])
        nb = np.linalg.norm(w_out_b[:, j])
        ratio = nb / na if na > 1e-9 else float('inf')
        marker = "  <<<" if name in ("giro_izq", "giro_der") else ""
        print(f"{name:>12} | {na:10.3f} | {nb:10.3f} | {ratio:10.2f}{marker}")

    print("\n--- Magnitud (norma L2) de w_res por columna de accion (residual x -> logit) ---")
    print(f"{'accion':>12} | {'A':>10} | {'B':>10} | {'ratio B/A':>10}")
    for j, name in enumerate(ACTION_NAMES):
        na = np.linalg.norm(w_res_a[:, j])
        nb = np.linalg.norm(w_res_b[:, j])
        ratio = nb / na if na > 1e-9 else float('inf')
        marker = "  <<<" if name in ("giro_izq", "giro_der") else ""
        print(f"{name:>12} | {na:10.3f} | {nb:10.3f} | {ratio:10.2f}{marker}")

    print("\n--- Peso de la fila x[10] (señal social) en w_res, relativo a las demas filas ---")
    for label, w_res in [("A", w_res_a), ("B", w_res_b)]:
        row_norms = np.linalg.norm(w_res, axis=1)
        social_norm = row_norms[10]
        other_mean = np.mean(np.delete(row_norms, 10))
        print(f"{label}: ||fila x[10]||={social_norm:.3f}, promedio del resto de filas={other_mean:.3f}, "
              f"ratio={social_norm/other_mean if other_mean > 1e-9 else float('inf'):.2f}")

    print("\n--- Resumen por fila de input, w_res (que tan fuerte pesa cada input) ---")
    print(f"{'input':>16} | {'||fila|| A':>11} | {'||fila|| B':>11} | {'ratio B/A':>10}")
    for i, name in enumerate(INPUT_NAMES):
        na = np.linalg.norm(w_res_a[i, :])
        nb = np.linalg.norm(w_res_b[i, :])
        ratio = nb / na if na > 1e-9 else float('inf')
        print(f"{name:>16} | {na:11.3f} | {nb:11.3f} | {ratio:10.2f}")


if __name__ == "__main__":
    main()
