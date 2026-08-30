"""
compute_x_fijo.py

Paso 4 (calculo de x_fijo) del intento de Sussillo-Barak para el
Paper 3. Lee el CSV generado por collect_h_x_vectors.py (que SI trae
x_0..x_10) y calcula, por agente:

  - opcion 1 (recomendada, la principal): promedio de x sobre TODAS
    las filas con stalled=True. Como stalled=True ya significa "lleva
    >=25 pasos sin desplazamiento efectivo" (criterio de
    foraging_env3d.py), promediar sobre esas filas es una forma
    simple y razonable de capturar "que esta viendo el robot, en
    general, cuando ya esta atascado" -- no hace falta identificar a
    mano el instante exacto en que h "se asento" (lo que se vio en
    plot_h_trajectory_umap.py).

  - opcion 2 (chequeo de robustez, si da el tiempo): promedio de x
    SOLO en el primer instante de cada racha de atasco (la primera
    fila de cada tramo consecutivo de stalled=True, agrupando por
    episodio+agente). Sirve para confirmar si el resultado del punto
    fijo depende mucho de "que tan atascado ya esta" el robot al
    tomar la foto.

  - opcion 3 (evitada, por pedido explicito): un x por cada racha
    individual, sin promediar. No se calcula aca.

Uso:
    python3 compute_x_fijo.py --csv h_x_vectors_300.csv --out x_fijo.json
"""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV generado por collect_h_x_vectors.py")
    ap.add_argument("--out", default="x_fijo.json", help="donde guardar el resultado")
    args = ap.parse_args()

    print(f"Leyendo {args.csv}...")
    df = pd.read_csv(args.csv)
    x_cols = [f"x_{k}" for k in range(11)]

    for col in x_cols:
        if col not in df.columns:
            raise ValueError(
                f"Falta la columna {col} en {args.csv} -- este script necesita "
                f"un CSV generado con collect_h_x_vectors.py (no el "
                f"collect_h_vectors.py original, que no guarda x)."
            )

    resultado = {}

    for agente in sorted(df["agent"].unique()):
        print(f"\n--- agente {agente} ---")
        df_ag = df[df["agent"] == agente].reset_index(drop=True)
        df_stalled = df_ag[df_ag["stalled"] == True]

        n_filas_stalled = len(df_stalled)
        print(f"  filas con stalled=True: {n_filas_stalled} de {len(df_ag)}")

        # ---------- opcion 1: promedio sobre todas las filas stalled ----------
        x_opcion1 = df_stalled[x_cols].mean().values
        print(f"  opcion 1 (promedio general del atasco): {np.round(x_opcion1, 3)}")

        # ---------- opcion 2: promedio solo del primer instante de cada racha ----------
        # una racha nueva empieza cuando stalled pasa de False (o no existe fila
        # anterior) a True, dentro del mismo episodio.
        es_inicio_racha = (
            (df_ag["stalled"] == True)
            & (
                (df_ag["stalled"].shift(1) != True)
                | (df_ag["episode"] != df_ag["episode"].shift(1))
            )
        )
        df_inicios = df_ag[es_inicio_racha]
        n_rachas = len(df_inicios)
        print(f"  rachas de atasco detectadas: {n_rachas}")

        if n_rachas > 0:
            x_opcion2 = df_inicios[x_cols].mean().values
            print(f"  opcion 2 (promedio del primer instante): {np.round(x_opcion2, 3)}")
        else:
            x_opcion2 = None
            print("  opcion 2: no se encontraron rachas, no se puede calcular")

        resultado[agente] = {
            "opcion_1_promedio_general": x_opcion1.tolist(),
            "opcion_2_promedio_primer_instante": (
                x_opcion2.tolist() if x_opcion2 is not None else None
            ),
            "n_filas_stalled": int(n_filas_stalled),
            "n_rachas": int(n_rachas),
        }

    with open(args.out, "w") as f:
        json.dump(resultado, f, indent=2)

    print(f"\nGuardado en {args.out}")
    print("\nSiguiente paso: usar 'opcion_1_promedio_general' de cada agente como "
          "x_fijo en find_fixed_points.py, y comparar contra 'opcion_2' como "
          "chequeo de robustez si da el tiempo.")


if __name__ == "__main__":
    main()
