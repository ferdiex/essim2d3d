"""
run_real_search.py

Paso 5 (al fin, los datos reales) del intento de Sussillo-Barak para
el Paper 3. Usa:

  - social3d_v15_full_final.json  -> pesos reales (brain_a, brain_b)
  - x_fijo.json                   -> x fijo por agente (opcion 1 y 2,
                                      calculados por compute_x_fijo.py)
  - h_x_vectors_pilot.csv         -> h reales, para samplear los puntos
                                      de partida de la busqueda (en vez
                                      de ruido al azar) -- siguiendo el
                                      espiritu del paper original de
                                      Sussillo-Barak, que recomienda
                                      arrancar de estados de la propia
                                      trayectoria real, no de cualquier
                                      lado del espacio.

Para cada agente:
  1. Sampleamos ~150 h iniciales de filas reales stalled=True de ESE
     agente (mas 30 al azar, para no depender 100% de lo que ya vieron).
  2. Corremos la busqueda con x_fijo = opcion 1 (promedio general).
  3. Deduplicamos, reportamos cuantos puntos fijos unicos aparecen.
  4. Para cada uno: autovalores del Jacobiano (estabilidad) y la
     direccion (autovector) del autovalor mas grande.
  5. Repetimos todo con x_fijo = opcion 2 (primer instante), como
     chequeo de robustez.
"""
import json

import numpy as np
import pandas as pd

from gru_dynamics import q
from find_fixed_points import search_fixed_points_con_correa, dedupe
from jacobian import analytical_jacobian


ACTION_NAMES = {
    0: "adelante",
    1: "giro_izq",
    2: "giro_der",
    3: "reversa",
    4: "señalizar/wander",
}


def cargar_pesos(path, agente):
    with open(path) as f:
        raw = json.load(f)
    brain = raw[f"brain_{agente}"]
    return {
        "w_gru": np.array(brain["w_gru"]),
        "b_gru": np.array(brain["b_gru"]),
        "w_out": np.array(brain["w_out"]),  # no lo usa F, pero lo usamos
        "b_out": np.array(brain["b_out"]),  # despues para interpretar
    }


def sampleos_h_iniciales(df, agente, n_de_datos=80, n_azar=10, seed=0):
    rng = np.random.default_rng(seed)
    h_cols = [f"h_{k}" for k in range(16)]

    df_ag = df[(df["agent"] == agente) & (df["stalled"] == True)]
    n_disponibles = len(df_ag)
    n_pedir = min(n_de_datos, n_disponibles)

    idx = rng.choice(n_disponibles, size=n_pedir, replace=False)
    h_de_datos = df_ag[h_cols].values[idx]

    h_azar = rng.normal(size=(n_azar, 16)) * 0.5

    return np.concatenate([h_de_datos, h_azar]), n_disponibles


def analizar_puntos_fijos(agente, weights, x_fijo, h_inits, w_out, etiqueta):
    print(f"\n=== agente {agente} -- x_fijo: {etiqueta} ({len(h_inits)} reinicios) ===")
    results = search_fixed_points_con_correa(
        x_fijo, weights, h_inits, n_iters=5000, lr=0.05, lambda_leash=0.001
    )

    q_finales = np.array([r["q"] for r in results])
    dist_finales = np.array([r["distancia_recorrida"] for r in results])
    print(f"  distribucion de q final: min={q_finales.min():.1e}  "
          f"mediana={np.median(q_finales):.1e}  max={q_finales.max():.1e}")
    print(f"  distribucion de distancia recorrida: min={dist_finales.min():.2f}  "
          f"mediana={np.median(dist_finales):.2f}  max={dist_finales.max():.2f}")

    tasa = np.mean(q_finales < 1e-4)
    print(f"  tasa de 'suficientemente lento' (q<1e-4): {tasa*100:.0f}%")

    encontrados = dedupe(results, q_tol=1e-4, dist_tol=1e-2)
    print(f"  puntos fijos unicos encontrados: {len(encontrados)}")

    resumen = []
    for i, fp in enumerate(encontrados):
        h_star = fp["h"]
        J = analytical_jacobian(h_star, x_fijo, weights)
        eigvals, eigvecs = np.linalg.eig(J)
        orden = np.argsort(-np.abs(eigvals))
        eigvals = eigvals[orden]
        eigvecs = eigvecs[:, orden]

        top_eigval = eigvals[0]
        top_eigvec = np.real(eigvecs[:, 0])

        # que tan estable es (radio espectral: si el mayor |autovalor| < 1,
        # el punto fijo es localmente estable/atractor)
        radio_espectral = float(np.abs(top_eigval))
        estable = radio_espectral < 1.0

        # proyeccion de la direccion "resbaladiza" (top_eigvec) sobre las
        # 5 acciones, via w_out -- para ver a que accion apunta mas
        proyeccion_acciones = top_eigvec @ w_out
        accion_dominante = int(np.argmax(np.abs(proyeccion_acciones)))

        print(f"  -- punto fijo #{i} -- q={fp['q']:.1e}")
        print(f"     radio espectral (mayor |autovalor|): {radio_espectral:.3f} "
              f"({'ESTABLE' if estable else 'inestable'})")
        print(f"     accion hacia la que mas apunta la direccion dominante: "
              f"{ACTION_NAMES[accion_dominante]} "
              f"(proyecciones: {np.round(proyeccion_acciones, 2)})")

        resumen.append({
            "h_star": h_star.tolist(),
            "q": float(fp["q"]),
            "radio_espectral": radio_espectral,
            "estable": bool(estable),
            "accion_dominante": ACTION_NAMES[accion_dominante],
            "proyeccion_acciones": proyeccion_acciones.tolist(),
        })

    return resumen


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="h_x_vectors_pilot.csv")
    ap.add_argument("--x_fijo", default="x_fijo.json")
    ap.add_argument("--pesos", default="social3d_v15_full_final.json")
    ap.add_argument("--out", default="resultado_puntos_fijos.json")
    ap.add_argument("--n_de_datos", type=int, default=80)
    ap.add_argument("--n_azar", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Cargando datos ({args.csv})...")
    df = pd.read_csv(args.csv)

    with open(args.x_fijo) as f:
        x_fijo_data = json.load(f)

    resultado_final = {}

    for agente in ["a", "b"]:
        weights = cargar_pesos(args.pesos, agente)
        h_inits, n_disp = sampleos_h_iniciales(df, agente, n_de_datos=args.n_de_datos, n_azar=args.n_azar)
        print(f"\nAgente {agente}: {n_disp} filas stalled disponibles para samplear h iniciales")

        resultado_final[agente] = {}
        for etiqueta, clave_json in [
            ("opcion 1 (promedio general)", "opcion_1_promedio_general"),
            ("opcion 2 (primer instante)", "opcion_2_promedio_primer_instante"),
        ]:
            x_fijo = np.array(x_fijo_data[agente][clave_json])
            resumen = analizar_puntos_fijos(
                agente, weights, x_fijo, h_inits, weights["w_out"], etiqueta
            )
            resultado_final[agente][etiqueta] = resumen

    with open(args.out, "w") as f:
        json.dump(resultado_final, f, indent=2)
    print(f"\nGuardado en {args.out}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
