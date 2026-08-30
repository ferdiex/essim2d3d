"""
diagnostico_streak_length.py

Diagnostico adicional (barato, sin simular nada nuevo) para tratar de
aterrizar mejor el hallazgo de Sussillo-Barak: en vez de comparar A vs
B como dos grupos nomas, se prueba algo mas fino y mas convincente
para el paper:

    Los puntos con radio espectral mas ESTABLE (<1), sacados de un h
    real de una racha de atasco real, ¿vienen de rachas que en la
    realidad duraron MAS tiempo? Y los mas INESTABLES, ¿de rachas mas
    cortas?

Si esto se cumple (correlacion negativa entre radio_espectral y
duracion de la racha: mas estable = racha mas larga), es evidencia
CONTINUA (no solo categorica A-vs-B) de que la geometria local del
espacio latente predice el comportamiento real -- mucho mas solido
para el paper que "A es una cosa, B es otra cosa", porque no depende
de que A y B sean grupos distintos, es una relacion que deberia
sostenerse DENTRO de cada agente tambien.

Ademas, chequeo extra (gratis, un value_counts): la frecuencia real de
cada accion durante el atasco, por agente, para ver si "giro_der" es
mas frecuente en B en los datos crudos, no solo en la proyeccion del
autovector.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from find_fixed_points import search_fixed_points_con_correa, dedupe
from jacobian import analytical_jacobian
from gru_dynamics import q as q_fn


ACTION_NAMES = {0: "adelante", 1: "giro_izq", 2: "giro_der", 3: "reversa", 4: "señalizar/wander"}


def agregar_streak_length(df):
    """
    Para cada fila, calcula a que racha de atasco pertenece (streak_id,
    unico por episodio+agente+racha) y cuantas decisiones duro esa
    racha en total (streak_length). Mismo criterio de "inicio de
    racha" que ya se uso en compute_x_fijo.py.
    """
    df = df.sort_values(["agent", "episode", "decision"]).reset_index(drop=True)

    es_inicio = (
        (df["stalled"] == True)
        & (
            (df["stalled"].shift(1) != True)
            | (df["episode"] != df["episode"].shift(1))
            | (df["agent"] != df["agent"].shift(1))
        )
    )
    streak_id = es_inicio.cumsum()
    # solo nos importan las filas que SI estan en una racha (stalled=True)
    streak_id = streak_id.where(df["stalled"] == True, other=-1)
    df["streak_id"] = streak_id

    largo_por_streak = df[df["streak_id"] != -1].groupby("streak_id").size()
    df["streak_length"] = df["streak_id"].map(largo_por_streak)

    return df


def sampleos_con_metadata(df, agente, n_de_datos=80, seed=0):
    """
    Igual que sampleos_h_iniciales en run_real_search.py, pero ADEMAS
    devuelve el streak_length real de cada fila sampleada, para poder
    correlacionarlo despues con el radio espectral encontrado.
    """
    rng = np.random.default_rng(seed)
    h_cols = [f"h_{k}" for k in range(16)]

    df_ag = df[(df["agent"] == agente) & (df["stalled"] == True)].reset_index(drop=True)
    n_pedir = min(n_de_datos, len(df_ag))
    idx = rng.choice(len(df_ag), size=n_pedir, replace=False)

    h_inits = df_ag.loc[idx, h_cols].values
    streak_lengths = df_ag.loc[idx, "streak_length"].values

    return h_inits, streak_lengths


def main():
    print("Cargando datos (n=300)...")
    df = pd.read_csv("h_x_vectors_300.csv")
    df = agregar_streak_length(df)

    with open("social3d_v15_full_final.json") as f:
        pesos_raw = json.load(f)
    with open("x_fijo_300.json") as f:
        x_fijo_data = json.load(f)

    # ------------------------------------------------------------
    # Chequeo extra gratis: frecuencia real de cada accion durante
    # el atasco, por agente (accion cruda del CSV, no la proyeccion
    # del autovector).
    # ------------------------------------------------------------
    print("\n=== Frecuencia real de acciones durante stalled=True (dato crudo) ===")
    for agente in ["a", "b"]:
        df_ag = df[(df["agent"] == agente) & (df["stalled"] == True)]
        conteo = df_ag["action"].value_counts(normalize=True).sort_index()
        etiquetas = {k: f"{ACTION_NAMES.get(k, k)}: {v*100:.1f}%" for k, v in conteo.items()}
        print(f"  agente {agente}: {etiquetas}")

    # ------------------------------------------------------------
    # Diagnostico principal: correlacion radio_espectral vs streak_length
    # ------------------------------------------------------------
    print("\n=== Correlacion: radio espectral (estabilidad local) vs duracion "
          "real de la racha de atasco ===")

    filas_pooled = []

    for agente in ["a", "b"]:
        brain = pesos_raw[f"brain_{agente}"]
        weights = {"w_gru": np.array(brain["w_gru"]), "b_gru": np.array(brain["b_gru"])}
        x_fijo = np.array(x_fijo_data[agente]["opcion_1_promedio_general"])

        h_inits, streak_lengths = sampleos_con_metadata(df, agente, n_de_datos=80)

        print(f"\n  agente {agente}: corriendo busqueda con correa sobre "
              f"{len(h_inits)} h reales...")
        results = search_fixed_points_con_correa(
            x_fijo, weights, h_inits, n_iters=5000, lr=0.05, lambda_leash=0.001
        )

        radios = []
        for r in results:
            J = analytical_jacobian(r["h"], x_fijo, weights)
            radios.append(np.max(np.abs(np.linalg.eigvals(J))))
        radios = np.array(radios)
        qs = np.array([r["q"] for r in results])

        # solo nos quedamos con busquedas que realmente asentaron en algo
        # razonable (q chico) -- mismo criterio de "confiar en el punto"
        # que usamos antes
        mask = qs < 1e-3
        radios_ok = radios[mask]
        streaks_ok = streak_lengths[mask]

        if len(radios_ok) >= 5:
            rho, p = stats.spearmanr(radios_ok, streaks_ok)
            print(f"  agente {agente} (n={len(radios_ok)}): "
                  f"Spearman rho={rho:.3f}, p={p:.3f}")
            print(f"    (rho negativo = mas estable -> racha real mas larga, "
                  f"lo esperado si la teoria es correcta)")
        else:
            print(f"  agente {agente}: muy pocos puntos confiables (n={len(radios_ok)}), "
                  f"no se calcula correlacion")

        for rad, sl in zip(radios_ok, streaks_ok):
            filas_pooled.append({"agente": agente, "radio_espectral": rad, "streak_length": sl})

    df_pooled = pd.DataFrame(filas_pooled)
    if len(df_pooled) >= 10:
        rho, p = stats.spearmanr(df_pooled["radio_espectral"], df_pooled["streak_length"])
        print(f"\n  POOLED (A+B juntos, n={len(df_pooled)}): Spearman rho={rho:.3f}, p={p:.3f}")
        print("  (esta es la version mas exigente del chequeo: si sigue dando "
              "negativo y significativo con A y B mezclados, la relacion no es "
              "solo 'es cuestion de que agente es', es una propiedad continua "
              "de la geometria local)")

    df_pooled.to_csv("diagnostico_streak_vs_estabilidad.csv", index=False)
    print("\nGuardado en diagnostico_streak_vs_estabilidad.csv")


if __name__ == "__main__":
    main()
